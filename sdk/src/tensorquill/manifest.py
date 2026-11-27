"""The run manifest: a run's project, name and setup, in one small file.

A run is still identified by its run ID, and every file and folder is still keyed by it. The
manifest adds the labels a person lists and compares runs by:

- beside local files it is `<run_id>.run.json`, next to `<run_id>.rank<N>.jsonl`, written by
  rank 0 at `tensorquill.init`;
- in a server's store it is `runs/<run_id>/run.json`, written by the server from the copy that
  rides on rank 0's first batch of each stream (`flight.py`).

A run recorded before the manifest existed has none. It reads as project `uncategorized` with
its run ID as its name (`run_block`).

This module imports nothing else of the package, so `capture`, `evidence`, `runs` and
`compare` can all use it.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import unicodedata
import warnings
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1
DEFAULT_PROJECT = "uncategorized"  # W&B's default project
PROJECT_ENV = "TENSORQUILL_PROJECT"
NAME_ENV = "TENSORQUILL_RUN_NAME"
# W&B's rule for a project name (`wandb/sdk/lib/...`: at most 128 characters, none of these),
# applied because the name becomes part of a path in listings.
PROJECT_MAX_LENGTH = 128
PROJECT_FORBIDDEN = "/\\#?%:"
# A run ID names the files of a run beside its records (`<run_id>.rank<N>.jsonl`, `<run_id>.run.json`),
# so it may hold no path separator, no drive mark (`C:x` is a path of its own on Windows) and no
# control character, and may not be `.` or `..` (BUG-06: `../../escaped` wrote outside the output
# folder, `sweep1/trial2` into a subfolder whose manifest landed in a doubled one). Narrower than the
# server's rule for the folder it makes of the ID (`plain_run_id` in `server/src/ingest.rs`: letters,
# digits, `-`, `_` and `.` only): an ID such as `exp 1` stays in the folder and is still taken here.
RUN_ID_FORBIDDEN = "/\\:"
MANIFEST_SUFFIX = ".run.json"  # local: <run_id>.run.json
STORE_MANIFEST = "run.json"  # a server's store: runs/<run_id>/run.json
METADATA_KEY = "run_manifest"  # the key in a Flight batch's app_metadata
# The largest manifest sent on a stream, as JSON text. The server stores none over 64 KiB
# (`MANIFEST_MAX_BYTES` in `server/src/ingest.rs`); this is below it, since the two write
# numbers a little differently.
MANIFEST_SEND_BYTES = 60_000
_RANK_FILE = re.compile(r"^(?P<run_id>.+)\.rank\d+\.jsonl$")


def _say(message: str, stacklevel: int = 2) -> None:
    """`warnings.warn(message, RuntimeWarning)`, which never raises (lane Y6, XC3). Under
    `python -W error` (or a filter that makes a RuntimeWarning an error) the message is logged
    on the `tensorquill` logger instead (stderr, through logging's last-resort handler, when no
    handler is set): a project name that had to be made to fit, a manifest that could not be
    written or one too large for a stream is said, and the run goes on. A filter that ignores
    the warning still silences it. `stacklevel` counts from the caller, as `warnings.warn`'s
    does. The same as `tracker._say` and `capture._warn`, kept here since this module imports
    nothing else of the package; `flight.py` uses it too."""
    try:
        warnings.warn(message, RuntimeWarning, stacklevel=stacklevel + 1)
    except Exception:  # noqa: BLE001  nothing here may stop a run
        try:
            import logging

            logging.getLogger("tensorquill").warning("%s", message)
        except Exception:  # noqa: BLE001
            pass


def project_problem(project: str) -> str | None:
    """Why `project` is not a project name by W&B's rule, or None when it is one."""
    if not project:
        return "it is empty"
    if len(project) > PROJECT_MAX_LENGTH:
        return f"it is {len(project)} characters long and at most {PROJECT_MAX_LENGTH} are allowed"
    bad = sorted({c for c in project if c in PROJECT_FORBIDDEN})
    if bad:
        return "it holds " + ", ".join(repr(c) for c in bad) + f" (none of {' '.join(PROJECT_FORBIDDEN)} is allowed)"
    return None


def conform_project(project: str) -> str:
    """`project` with each forbidden character replaced by `-` and cut to the allowed length."""
    cleaned = "".join("-" if c in PROJECT_FORBIDDEN else c for c in project)[:PROJECT_MAX_LENGTH]
    return cleaned or DEFAULT_PROJECT


def _run_id_forbidden(c: str) -> bool:
    return c in RUN_ID_FORBIDDEN or unicodedata.category(c) == "Cc"


def run_id_problem(run_id: str) -> str | None:
    """Why `run_id` cannot name a run's files in the output folder (`RUN_ID_FORBIDDEN`), or None."""
    if not run_id:
        return "it is empty"
    if run_id in (".", ".."):
        return f"it is {run_id!r}, which names a folder, not a run"
    bad = sorted({c for c in run_id if _run_id_forbidden(c)})
    if bad:
        return ("it holds " + ", ".join(repr(c) for c in bad) + " (it names files in the output folder, so it may "
                "hold no '/', '\\', ':' or control character)")
    return None


def conform_run_id(run_id: str) -> str:
    """`run_id` with each character `run_id_problem` refuses replaced by `-`, and `-` for `.` or `..`.
    It depends on `run_id` alone, so every rank makes the same ID of the one rank 0 shared."""
    cleaned = "".join("-" if _run_id_forbidden(c) else c for c in run_id)
    return "-" if cleaned in ("", ".", "..") else cleaned


# Whether `init` takes an `os.PathLike` run ID (a `Path`) by its text. False refuses it like an `int`.
RUN_ID_TAKES_PATHS = True


def check_run_id(run_id: Any) -> str | None:
    """`init`'s explicit `run_id`, normalised and checked before anything is opened, and the only form
    of it `init` goes on with: None as None, a string as itself (a subclass as a plain `str`), and,
    while `RUN_ID_TAKES_PATHS`, an `os.PathLike` (a `Path`) as its text, `os.fspath(run_id)`, judged
    as that string. Anything else raises `ValueError`, as a `group` or `name` that is not a string
    does. A string that could name a file outside the output folder (`run_id_problem`) raises
    `ValueError` too. The empty string is returned as it is and, as None, means "not given"."""
    if run_id is None:
        return None
    wanted = "a string or a path" if RUN_ID_TAKES_PATHS else "a string"
    if RUN_ID_TAKES_PATHS and not isinstance(run_id, str) and isinstance(run_id, os.PathLike):
        try:
            text = os.fspath(run_id)
        except Exception as error:  # noqa: BLE001  a path whose __fspath__ raises or gives neither str nor bytes
            raise ValueError(f"run_id must be {wanted}, and {type(run_id).__name__}'s path cannot be read: "
                             f"{type(error).__name__}: {error}") from error
        if not isinstance(text, str):
            raise ValueError(f"run_id must be {wanted}, not {type(run_id).__name__} of {type(text).__name__}")
        run_id = text
    if not isinstance(run_id, str):
        raise ValueError(f"run_id must be {wanted}, not {type(run_id).__name__}")
    run_id = str.__str__(run_id)
    if run_id:
        problem = run_id_problem(run_id)
        if problem:
            raise ValueError(f"run_id {run_id!r} cannot be used: {problem}")
    return run_id


_RUN_ID_SOURCES = {
    "explicit": "rank 0's run_id", "env": "MLFLOW_RUN_ID", "mlflow": "the MLflow run", "wandb": "the W&B run",
    "launcher": "the launcher's job ID", "generated": "the generated run ID",
}


def fit_run_id(run_id: Any, source: str) -> Any:
    """A run ID TensorQuill was not given (the environment's, a tracker's, a launcher's, or one rank 0
    shared from an older TensorQuill) made to fit (`conform_run_id`) when `run_id_problem` refuses it,
    with a warning, as `resolve_project` fits a project name from there: an ID TensorQuill was never
    given must not stop a run from starting, nor name a file outside the output folder.

    One that is not a string (a rank 0 on an older TensorQuill given `run_id=123` shares the integer,
    and a stand-in tracker may give any object) is taken by its text, with a warning, and then fitted
    the same way: what comes back is a string, unless `run_id` is None or empty."""
    if run_id is not None and not isinstance(run_id, str):
        # Strings by construction from every source this package reads, except a rank 0 on an older
        # TensorQuill given `run_id=123` (shared through the store as JSON) or a tracker object that is
        # not the real one: taken by its text, which is what such a rank named its files with.
        try:
            text = os.fspath(run_id) if isinstance(run_id, os.PathLike) else str(run_id)
        except Exception:  # noqa: BLE001  a value whose __str__ or __fspath__ raises
            text = type(run_id).__name__
        if not isinstance(text, str):
            text = type(run_id).__name__
        _say(f"{_RUN_ID_SOURCES.get(source, source)} gives a run ID that is not a string "
             f"({type(run_id).__name__}); the run is recorded under its text, {text!r}. Pass run_id= to choose "
             "the ID.", stacklevel=4)
        run_id = text or "-"
    if not isinstance(run_id, str) or not run_id:
        return run_id
    problem = run_id_problem(run_id)
    if problem is None:
        return run_id
    fitted = conform_run_id(run_id)
    _say(
        f"{_RUN_ID_SOURCES.get(source, source)} gives the run ID {run_id!r}, which cannot name TensorQuill's files "
        f"({problem}); the run is recorded under {fitted!r}. Pass run_id= to choose the ID.",
        stacklevel=4,
    )
    return fitted


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def resolve_project(argument: str | None, tracker: str | None = None) -> str:
    """The argument, then `TENSORQUILL_PROJECT`, then the tracker's, then `uncategorized`.

    An argument that breaks the rule raises `ValueError`, at `init`, like any other bad
    argument. A name from the environment or the tracker that breaks it (an MLflow experiment
    is often a path) is made to fit, with a warning: a name TensorQuill was never given must
    not stop a run from starting.
    """
    if argument is not None:
        if not isinstance(argument, str):
            raise ValueError(f"project must be a string, not {type(argument).__name__}")
        problem = project_problem(argument)
        if problem:
            raise ValueError(f"project {argument!r} cannot be used: {problem}")
        return argument
    for found, where in ((os.environ.get(PROJECT_ENV), PROJECT_ENV), (tracker, "the tracker's project")):
        if not _text(found):
            continue
        problem = project_problem(found)
        if problem is None:
            return found
        fitted = conform_project(found)
        _say(
            f"{where} is {found!r}, which cannot be a TensorQuill project name ({problem}); "
            f"the run is filed under {fitted!r}. Pass project= to choose the name.",
            stacklevel=4,
        )
        return fitted
    return DEFAULT_PROJECT


def resolve_name(argument: str | None, run_id: str | None, tracker: str | None = None) -> str | None:
    """The argument, then `TENSORQUILL_RUN_NAME`, then the tracker's run name, then the run ID.
    A label: it may repeat within a project."""
    if argument is not None and not isinstance(argument, str):
        raise ValueError(f"name must be a string, not {type(argument).__name__}")
    return _text(argument) or _text(os.environ.get(NAME_ENV)) or _text(tracker) or run_id


def _plain(value: Any) -> Any:
    """A config value as JSON can hold it: numbers, strings, booleans and None as they are (a
    number that is not finite as its text), lists and dictionaries of those, anything else
    as its text."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    try:
        return str(value)
    except Exception:  # noqa: BLE001  a config value's own __str__
        return f"<{type(value).__name__}>"


def clean_config(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise ValueError(f"config must be a dictionary of hyperparameters, not {type(config).__name__}")
    return {str(k): _plain(v) for k, v in config.items()}


def clean_tags(tags: Any) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        return [tags]
    try:
        return [str(tag) for tag in tags]
    except TypeError:
        raise ValueError(f"tags must be a list of strings, not {type(tags).__name__}") from None


def _versions() -> tuple[str | None, str | None]:
    package = sys.modules.get(__package__ or "tensorquill")
    torch = sys.modules.get("torch")
    return getattr(package, "__version__", None), getattr(torch, "__version__", None) and str(torch.__version__)


# --- The run's numerical settings (item OG1) -------------------------------------------------
#
# Whether healthy data-parallel replicas come out bit-identical depends on these settings, so a
# replica result is only read against them. They are read as `init` finds them, on the rank that
# writes the manifest (rank 0), and never set. OpGuard's `runtime_policy.py` records the first six
# before and after it sets them; `CUDA_DEVICE_MAX_CONNECTIONS` is the seventh, since its value hid
# two DeepSpeed races in their reproductions. Warn-only is its own read of the first setting, and
# two more TF32 controls are read beside `allow_tf32`: `cuda.matmul.fp32_precision` (newer
# PyTorch) and `cudnn.allow_tf32`.
#
# Each entry is named by what was read, and is `{"value": v}` (an environment variable that is not
# set is `{"value": None}`) or `{"not_read": why}`: a setting this PyTorch lacks, or whose read
# raised, is said, never left out or guessed. A warning given while reading is kept beside the
# value (`"warning"`) and never shown or raised, so `python -W error` does not stop `init`. None of
# these reads initialises CUDA: they are flags of the process and of the C++ context.

NUMERICAL_SETTINGS_KEY = "numerical_settings"
NOT_RECORDED = "the run manifest was written before TensorQuill recorded the run's numerical settings"
SETTING_TEXT_MAX = 256  # an environment variable's value, or a value's text, kept to this many characters
_ERROR_TEXT_MAX = 200
_TORCH_SETTINGS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    # (name, attribute path below `torch`, whether it is called)
    ("torch.are_deterministic_algorithms_enabled()", ("are_deterministic_algorithms_enabled",), True),
    ("torch.is_deterministic_algorithms_warn_only_enabled()", ("is_deterministic_algorithms_warn_only_enabled",), True),
    ("torch.backends.cudnn.deterministic", ("backends", "cudnn", "deterministic"), False),
    ("torch.backends.cudnn.benchmark", ("backends", "cudnn", "benchmark"), False),
    ("torch.backends.cuda.matmul.allow_tf32", ("backends", "cuda", "matmul", "allow_tf32"), False),
    ("torch.utils.deterministic.fill_uninitialized_memory", ("utils", "deterministic", "fill_uninitialized_memory"), False),
    # The newer control of the same choice as `allow_tf32` (PyTorch 2.9 and later; "none", "ieee" or
    # "tf32"), and cuDNN's own TF32 switch for convolutions.
    ("torch.backends.cuda.matmul.fp32_precision", ("backends", "cuda", "matmul", "fp32_precision"), False),
    ("torch.backends.cudnn.allow_tf32", ("backends", "cudnn", "allow_tf32"), False),
)
_ENV_SETTINGS = ("CUBLAS_WORKSPACE_CONFIG", "CUDA_DEVICE_MAX_CONNECTIONS")


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]


def _setting_value(value: Any) -> dict[str, Any]:
    """A setting's value as the manifest keeps it: plain JSON as it is, anything else as its
    text, and a text longer than `SETTING_TEXT_MAX` cut, with its length."""
    if value is None or isinstance(value, (bool, int)):
        return {"value": value}
    if isinstance(value, float):
        return {"value": value if math.isfinite(value) else str(value)}
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception:  # noqa: BLE001  a value's own __str__
            value = f"<{type(value).__name__}>"
    if len(value) > SETTING_TEXT_MAX:
        return {"value": value[:SETTING_TEXT_MAX], "cut_from_length": len(value)}
    return {"value": value}


class _Missing(Exception):
    """A setting this PyTorch does not have."""


def _read_one(read: Any) -> dict[str, Any]:
    """`read()` as an entry: its value, or why it could not be read. Never raises."""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            entry = _setting_value(read())
        if caught:
            first = caught[0]
            entry["warning"] = _cut(f"{first.category.__name__}: {first.message}", _ERROR_TEXT_MAX)
        return entry
    except _Missing as missing:
        return {"not_read": str(missing)}
    except Exception as error:  # noqa: BLE001  a read that raises is recorded, never raised
        return {"not_read": _cut(f"reading it raised {type(error).__name__}: {error}", _ERROR_TEXT_MAX)}


def _torch_reader(torch: Any, path: tuple[str, ...], call: bool) -> Any:
    def read() -> Any:
        found = torch
        for depth, part in enumerate(path):
            try:
                found = getattr(found, part)
            except AttributeError:
                version = getattr(torch, "__version__", None)
                raise _Missing(f"this PyTorch{f' ({version})' if version else ''} has no "
                               f"torch.{'.'.join(path[: depth + 1])}") from None
        return found() if call else found

    return read


def _read_numerical_settings() -> dict[str, Any]:
    torch = sys.modules.get("torch")  # never imported here: `init` has imported it already
    settings: dict[str, Any] = {}
    for name, path, call in _TORCH_SETTINGS:
        settings[name] = ({"not_read": "torch is not imported"} if torch is None
                          else _read_one(_torch_reader(torch, path, call)))
    for variable in _ENV_SETTINGS:
        settings[f"os.environ.get('{variable}')"] = _read_one(lambda variable=variable: os.environ.get(variable))
    return {"read_at": "init", "settings": settings}


def numerical_settings_block() -> dict[str, Any]:
    """The manifest's `numerical_settings`: each setting as this process has it now. Never raises."""
    try:
        return _read_numerical_settings()
    except Exception as error:  # noqa: BLE001  nothing here may stop a run
        return {"read_at": "init", "not_read": _cut(f"reading them raised {type(error).__name__}: {error}",
                                                     _ERROR_TEXT_MAX)}


def numerical_settings(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    """What a run manifest says of the run's numerical settings, for every reader that shows them:
    the block with `recorded: True`, or `{"recorded": False, "why": ...}` for a run without a
    manifest or one written before they were recorded. A run whose settings were not recorded is
    never read as having the defaults."""
    if not isinstance(manifest, Mapping):
        return {"recorded": False, "why": "the run has no manifest"}
    block = manifest.get(NUMERICAL_SETTINGS_KEY)
    if not isinstance(block, Mapping):
        return {"recorded": False, "why": NOT_RECORDED}
    return {"recorded": True, **block}


def build_manifest(
    *,
    run_id: str | None,
    run_id_source: str,
    project: str,
    name: str | None,
    group: str | None = None,
    tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
    world_size: int | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    sdk_version, torch_version = _versions()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "project": project,
        "name": name,
        "run_id": run_id,
        "group": group,
        "tags": list(tags or ()),
        "created_at": created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "world_size": world_size,
        "sdk_version": sdk_version,
        "torch_version": torch_version,
        # Where the run ID came from: explicit, env (MLFLOW_RUN_ID), mlflow, wandb, launcher or generated.
        "run_id_source": run_id_source,
        "config": dict(config or {}),
        # As `init` found them on this rank, read and never set (item OG1; `numerical_settings_block`).
        NUMERICAL_SETTINGS_KEY: numerical_settings_block(),
    }


def manifest_to_send(manifest: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    """The copy of the manifest that rides on a stream, and what was left out of it, if anything.

    The manifest travels inside a batch's metadata, so it must stay small: a `config` holding a
    whole vocabulary would otherwise go with the first batch of every stream, and the server
    would drop it on arrival. Over `MANIFEST_SEND_BYTES` the copy goes without its `config`
    (marked `config_left_out`), so the store still has the project and the name; if it is
    still too large it is not sent. The local file always has everything.
    """
    if not manifest:
        return None, None

    def size(document: Mapping[str, Any]) -> int:
        return len(json.dumps(document, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8"))

    whole = dict(manifest)
    full = size(whole)
    if full <= MANIFEST_SEND_BYTES:
        return whole, None
    without = {**whole, "config": {}, "config_left_out": True}
    if size(without) <= MANIFEST_SEND_BYTES:
        return without, (f"the run manifest is {full:,} bytes with its config, more than the {MANIFEST_SEND_BYTES:,} a "
                         "stream carries, so the server's copy has the project and the name and no config")
    return None, (f"the run manifest is {size(without):,} bytes even without its config, more than the "
                  f"{MANIFEST_SEND_BYTES:,} a stream carries, so it is not sent and the server's store has no manifest "
                  "for this run")


def local_manifest_path(records_file: str | Path, run_id: str) -> Path:
    """`<run_id>.run.json` in the folder of a rank's `.jsonl` file, whatever that file is named."""
    return Path(records_file).parent / f"{run_id}{MANIFEST_SUFFIX}"


def read_manifest(path: str | Path) -> dict[str, Any] | None:
    """The manifest at `path`, or None when there is no file or it is not a JSON object."""
    try:
        found = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> bool:
    """Write the manifest by a temporary file and a rename. A run that starts again under the
    same run ID keeps the first start's `created_at`. Never raises: a manifest that cannot be
    written is warned about and the run goes on without one. A scratch file that the failed
    write or rename left (`.<name>.<pid>.tmp`) is removed (lane Z6, L23), and a removal that
    fails is let be; the manifest already there is left as it was."""
    path = Path(path)
    scratch: Path | None = None
    try:
        earlier = read_manifest(path)
        document = dict(manifest)
        if earlier and _text(earlier.get("created_at")) and earlier.get("run_id") == document.get("run_id"):
            document["created_at"] = earlier["created_at"]
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        scratch.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        os.replace(scratch, path)
        return True
    except Exception as error:  # noqa: BLE001  nothing here may stop a run
        if scratch is not None:
            try:
                scratch.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001  a scratch file that stays is untidy, never a failure
                pass
        # Four frames up is the caller of `tensorquill.init` (this function, `init`, the
        # `torch.compiler.disable` wrapper around it, the caller); three named the wrapper.
        _say(
            f"tensorquill could not write the run manifest {path} ({error}); the run goes on, and is listed "
            f"under {DEFAULT_PROJECT!r} with its run ID as its name.",
            stacklevel=4,
        )
        return False


def run_id_of_file(path: str | Path) -> str | None:
    """The run ID a file named `<run_id>.rank<N>.jsonl` carries. The match is anchored at the
    end, since a run ID may hold dots (`run-slurm1234.0`)."""
    found = _RANK_FILE.match(Path(path).name)
    return found.group("run_id") if found else None


def manifest_candidates(source: str | Path, run_id: str | None = None) -> list[Path]:
    """Where the manifest of the run at `source` would be: `source` is a manifest itself, a
    run folder of a server's store or a `rank=<n>` folder in it, or a rank's `.jsonl` file."""
    source = Path(source)
    if source.name == STORE_MANIFEST or source.name.endswith(MANIFEST_SUFFIX):
        return [source]
    if source.is_dir():
        # Imported here: `evidence` imports this module. The one reader of `rank=<n>` (ASCII
        # digits, as the server writes it), so every module reads a store's folders alike
        # (lane X5, DET-17: `isdecimal()` here took `rank=٣` for rank 3).
        from .evidence import rank_number

        folder = source.resolve()
        if rank_number(folder.name) is not None:
            folder = folder.parent
        return [folder / STORE_MANIFEST]
    ids = [i for i in (run_id, run_id_of_file(source)) if i]
    return [local_manifest_path(source, i) for i in dict.fromkeys(str(i) for i in ids)]


def find_manifest(source: Any, run_id: Any = None) -> dict[str, Any] | None:
    """The manifest of the run at `source` (see `manifest_candidates`), a manifest given as a
    dictionary, or None. A manifest that names another run ID than `run_id` is not this run's."""
    if isinstance(source, Mapping):
        found: dict[str, Any] | None = dict(source)
    elif isinstance(source, (str, Path)):
        found = next((m for m in map(read_manifest, manifest_candidates(source, _text(run_id))) if m is not None), None)
    else:
        return None
    if found is not None and run_id is not None and found.get("run_id") not in (None, run_id):
        return None
    return found


def run_block(run_id: Any, manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    """The `run` block of an evidence file: the project, the name and the run ID. Without a
    manifest (a run recorded before there was one) the project is `uncategorized` and the name
    is the run ID."""
    manifest = manifest if isinstance(manifest, Mapping) else {}
    project = _text(manifest.get("project"))
    return {
        "project": project if project and project_problem(project) is None else DEFAULT_PROJECT,
        "name": _text(manifest.get("name")) or (str(run_id) if run_id is not None else None),
        "run_id": run_id,
    }
