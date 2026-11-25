"""The Rust detectors on records in hand, through `tensorquill-server detect`.

The server binary judges one rank's rows read from an Arrow stream and prints the report it
would write as that rank's `incidents.json`. `server_report` is the Python side: records in,
that report out. `build_evidence(..., detectors="rust")` takes its incidents and its per-step
MoE samples from it (`detect --samples`), and passes its own threshold options on as the
server's one `--thresholds` JSON object.

Needs `pyarrow` (the `flight` extra) and the binary. `find_binary` says where it is looked
for: `TENSORQUILL_SERVER_BIN`, then beside the interpreter (where `pip install
tensorquill-server` puts it), then `PATH`. A binary is run only when `tensorquill-server
protocol` prints this module's `PROTOCOL`.

`resolve_detectors` turns `detectors="auto"` into `"rust"` or `"python"`: Rust when all of that
is there, and Python otherwise, with the reason.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any, Iterable, Mapping

from .identity import is_scalar_record

BINARY_ENV = "TENSORQUILL_SERVER_BIN"
BINARY_NAME = "tensorquill-server"
# The number `tensorquill-server protocol` must print for this module to run the binary. It is
# `PROTOCOL` in `server/src/main.rs`, and the two go up together whenever what `detect` reads,
# takes or prints changes in a way the other side would get wrong (a kind added or removed, a
# threshold or option added, a report key renamed or dropped). Only the same number is taken:
# a larger one is a newer binary whose reports this module was not written for.
# 3 (lanes W2 and W3, Nov 19, 2026): attempts ordered by first appearance (contract 1), the
# capture-stopped state (contract 3), and the detector rules of lane W3.
# 4 (wave AF-W, Nov 21, 2026): `detect --follow`, for `--watch` with the Rust detectors.
# 5 (wave AL, Nov 24, 2026): protocol 5's row layout (`WAVE_AL_CONTRACT.md`), which `detect` reads:
# `layers` and `moe_stats` as typed map columns in the model's order, the probe split into
# `probe_json`, `probe_maps` and `probe_histograms`, and row version 2 with the record's own
# version in `record_version`.
PROTOCOL = 5
INSTALL_HINT = "pip install tensorquill-server"
DETECTOR_CHOICES = ("auto", "python", "rust")
_BATCH_ROWS = 500
_PROTOCOL_TIMEOUT = 10.0
# What each binary answered, by path, modification time and size: one probe per binary and process.
_protocols: dict[tuple[str, int, int], int | str] = {}


def _beside_interpreter() -> str | None:
    """The binary an installed `tensorquill-server` wheel put with this interpreter's scripts,
    whether or not that folder is on `PATH` (a virtual environment that was never activated,
    `uvx`, `pip install --user`)."""
    name = BINARY_NAME + (".exe" if os.name == "nt" else "")
    folders: list[str] = []
    try:
        folders.append(sysconfig.get_path("scripts"))
    except (KeyError, OSError):  # an interpreter with no such scheme
        pass
    if sys.executable:
        # Not resolved: a virtual environment's interpreter is a link to the one it was made from.
        folders.append(os.path.dirname(os.path.abspath(sys.executable)))
    try:
        folders.append(sysconfig.get_path("scripts", sysconfig.get_preferred_scheme("user")))
    except (KeyError, OSError, AttributeError, ValueError):
        pass
    for folder in folders:
        if folder and os.path.isfile(os.path.join(folder, name)):
            return os.path.join(folder, name)
    return None


def find_binary(binary: str | os.PathLike[str] | None = None) -> str | None:
    """Where `tensorquill-server` is, or None: `binary`, else `TENSORQUILL_SERVER_BIN`, else
    the one beside this interpreter (installed with `pip install tensorquill-server`), else
    the one on `PATH`. The first of these that is named is the answer, whether or not a file
    is there: a wrong `TENSORQUILL_SERVER_BIN` is said, not passed over."""
    chosen = binary or os.environ.get(BINARY_ENV) or _beside_interpreter() or shutil.which(BINARY_NAME)
    return str(chosen) if chosen else None


def binary_protocol(binary: str | os.PathLike[str]) -> int:
    """The protocol number `binary` reports (`tensorquill-server protocol`). Raises
    `RuntimeError` when it reports none: a binary from before the number, or another program."""
    path = str(binary)
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
    except OSError as error:
        raise RuntimeError(f"{path} cannot be read: {error}") from error
    if key not in _protocols:
        try:
            done = subprocess.run([path, "protocol"], stdin=subprocess.DEVNULL, capture_output=True,
                                  timeout=_PROTOCOL_TIMEOUT)
            printed = done.stdout.decode("ascii", "replace").strip()
            if done.returncode == 0 and printed.isdigit():
                _protocols[key] = int(printed)
            else:
                _protocols[key] = f"exit status {done.returncode}" if done.returncode else f"it printed {printed[:40]!r}"
        except subprocess.TimeoutExpired:
            _protocols[key] = f"no answer in {_PROTOCOL_TIMEOUT:g} s"
        except OSError as error:
            _protocols[key] = f"it cannot be run: {error}"
    answer = _protocols[key]
    if isinstance(answer, str):
        raise RuntimeError(
            f"{path} does not report a protocol number (`tensorquill-server protocol`: {answer}), so it is older "
            f"than this tensorquill, which needs protocol {PROTOCOL}: {INSTALL_HINT} --upgrade, or rebuild it"
        )
    return answer


def _only_the_fallback_wheel() -> bool:
    """Whether `tensorquill-server` is installed as the wheel for platforms without a binary
    (tagged `py3-none-any`, holding no program). False when the package is not installed."""
    from importlib import metadata

    try:
        wheel = metadata.distribution(BINARY_NAME).read_text("WHEEL") or ""
    except Exception:  # not installed, or metadata that cannot be read: the plain install hint
        return False
    return "Tag: py3-none-any" in wheel


def server_binary(binary: str | os.PathLike[str] | None = None) -> str:
    """The `tensorquill-server` to run (`find_binary`), once it has reported this module's
    `PROTOCOL`. Raises `RuntimeError` when there is none, or it reports another number or none."""
    chosen = find_binary(binary)
    if not chosen:
        if _only_the_fallback_wheel():
            # Installing again would bring the same wheel: say what does bring a binary here.
            raise RuntimeError(
                "the Rust detectors need the tensorquill-server binary, and the tensorquill-server package installed "
                "here has none for this platform (its wheels with the binary are for Linux x86_64, Linux arm64 and "
                f"macOS arm64): build it from source (cargo build --release in server/) and set {BINARY_ENV} to its path"
            )
        raise RuntimeError(
            f"the Rust detectors need the tensorquill-server binary and none was found: {INSTALL_HINT} (wheels for "
            f"Linux x86_64, Linux arm64 and macOS arm64), or build it (cargo build --release in server/) and set "
            f"{BINARY_ENV} to its path"
        )
    if not Path(chosen).is_file():
        raise RuntimeError(f"{chosen} is not a file: {BINARY_ENV} or binary= must name the tensorquill-server binary")
    spoken = binary_protocol(chosen)
    if spoken != PROTOCOL:
        raise RuntimeError(
            f"{chosen} speaks protocol {spoken} and this tensorquill needs protocol {PROTOCOL}: install a tensorquill "
            "and a tensorquill-server that match (pip install --upgrade tensorquill tensorquill-server), or rebuild "
            "the binary from this source"
        )
    return chosen


def rust_ready(binary: str | os.PathLike[str] | None = None) -> str:
    """What `server_report` needs, checked without running it: the binary (returned) and
    `pyarrow`. Raises `RuntimeError` naming what is missing."""
    chosen = server_binary(binary)
    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise RuntimeError("the Rust detectors read Arrow: pip install 'tensorquill[flight]'") from error
    return chosen


def resolve_detectors(
    detectors: str, binary: str | os.PathLike[str] | None = None,
) -> tuple[str, str | None]:
    """Which detectors `detectors` means here, and why when that is a fallback.

    `"python"` and `"rust"` are themselves, unchecked. `"auto"` is `"rust"` when `rust_ready`
    passes (a binary that reports this module's `PROTOCOL`, and `pyarrow`), and otherwise
    `"python"` with the one sentence to show the user: what is missing and what installs it.
    The choice is made on what is installed, once; a Rust run that then fails is an error, as
    it is when `"rust"` was asked for.
    """
    if detectors not in DETECTOR_CHOICES:
        raise ValueError(f"detectors must be 'auto', 'python' or 'rust', not {detectors!r}")
    if detectors != "auto":
        return str(detectors), None
    try:
        rust_ready(binary)
    except RuntimeError as error:
        return "python", f"using the Python detectors, not the Rust ones: {error}"
    return "rust", None


def records_stream(records: Iterable[Mapping[str, Any]]) -> bytes:
    """One rank's records as the Arrow IPC stream `tensorquill-server detect` reads, in
    `stream_order`: grouped by attempt in the order the attempts first appear, each attempt's
    rows in step order, rows of one step in the order given. A record whose identity fields are
    not valid raises `identity.MalformedRecordError` (see `_stream`)."""
    return _stream(records)


def _stream(records: Iterable[Mapping[str, Any]]) -> bytes:
    """`records` as an Arrow IPC stream of `flight_schema()`, in `stream_order`. Each record's
    identity fields are checked first, in the order given (`identity.check`, `WAVE_AF_CONTRACT.md`
    section 1): one that is not valid raises `identity.MalformedRecordError` naming its position
    in `records`, and nothing is judged. Then the scalar-only records are set apart (`identity.is_scalar_record`,
    `WAVE_AL_CONTRACT.md` section 4.6; Q4 a): the Rust detectors are never sent one, and judge the
    training records alone."""
    from .identity import check, in_list

    listed = list(records)
    for index, record in enumerate(listed):
        check(record, in_list(index))
    listed = _training(listed)
    try:
        import pyarrow as pa
    except ImportError as error:
        raise RuntimeError("the Rust detectors read Arrow: pip install 'tensorquill[flight]'") from error
    from .flight import _row, flight_schema, rows_batch

    # Records written without a run ID still have to travel in the column.
    ordered = [record if isinstance(record.get("run_id"), str) else {**record, "run_id": ""}
               for record in stream_order(listed)]
    rows = [_row(record) for record in ordered]
    schema = flight_schema()
    out = io.BytesIO()
    with pa.ipc.new_stream(out, schema) as writer:
        for start in range(0, len(rows), _BATCH_ROWS):
            # A value only Arrow refuses travels in `extra_json`, as on the sender's path (wave AJ).
            end = start + _BATCH_ROWS
            writer.write_batch(rows_batch(ordered[start:end], rows[start:end], schema))
    return out.getvalue()


def server_report(
    records: Iterable[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, Any] | None = None,
    binary: str | os.PathLike[str] | None = None,
    timeout: float | None = 300.0,
    samples: bool = False,
) -> dict[str, Any]:
    """The server's detectors over one rank's records: the report `tensorquill-server` writes
    as `rank=<rank>/incidents.json`, made here from records in hand.

    Pass every attempt of the rank (`load_records(..., all_attempts=True)`): the detectors
    compare a resume with what it replaced. Records of a second rank or run are refused.

    `thresholds` are the detector options, sent as the one JSON object `--thresholds` takes:
    `overload_threshold`, `drift_tolerance`, `starve_window`, `starve_ratio` and `top_k` (None,
    a number, or a number per router name). A key left out keeps the server's default, and
    the report's own `thresholds` lists what was applied. An unknown key or a value the
    server refuses raises `ValueError`.

    `samples=True` asks for one more key, `samples` (`detect --samples`): the per-step MoE
    samples as `build_evidence` lists them, one for every record and router, judged by the
    same rules as the incidents. The server's own `incidents.json` never has them. For a long
    MoE run they are most of the output: 97 MB for 10,000 steps of 24 routers of 32 experts.

    Raises `RuntimeError` when the binary or `pyarrow` is missing, or the binary fails.
    """
    records = list(records)
    _something_to_judge(records)
    chosen = server_binary(binary)
    command = [chosen, "detect", *(["--samples"] if samples else [])]
    if thresholds is not None:
        try:
            command += ["--thresholds", json.dumps(dict(thresholds), allow_nan=False)]
        except (TypeError, ValueError) as error:
            raise ValueError(f"the thresholds cannot be sent as JSON: {error}") from error
    try:
        done = subprocess.run(command, input=records_stream(records), capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{chosen} detect did not finish in {timeout:g} s") from error
    if done.returncode != 0:
        raise _detect_failure(chosen, done.stderr.decode("utf-8", "replace").strip() or f"exit status {done.returncode}")
    return _report_of(chosen, done.stdout, samples)


def _training(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """`records` without the scalar-only ones (`identity.is_scalar_record`), in their order: what the Rust
    detectors read (`WAVE_AL_CONTRACT.md` section 4.6). No detector reads a scalar-only record (Q4 a)."""
    return [record for record in records if not is_scalar_record(record)]


def _something_to_judge(records: list[Mapping[str, Any]]) -> None:
    """Raise `ValueError` when no record of `records` is one the detectors read."""
    if not records:
        raise ValueError("no records to judge")
    if not _training(records):
        raise ValueError("no records to judge: every record given is a scalar-only record, which no detector reads")


def _detect_failure(chosen: str, message: str) -> Exception:
    """The error for a `detect` that failed with `message` on its standard error: `ValueError` for
    thresholds it refuses (an option the caller chose, not a failure of the binary), else
    `RuntimeError`."""
    if "invalid value" in message and "--thresholds" in message:
        # The server names the key and why.
        refused = message.split("for '--thresholds <JSON>':", 1)[-1].split("\n", 1)[0].strip()
        return ValueError(f"the Rust detectors refuse these thresholds: {refused}")
    if "unrecognized subcommand" in message or "unexpected argument" in message:
        message += " (this tensorquill-server is too old to have `detect`, `--thresholds` or `--samples`: rebuild it)"
    return RuntimeError(f"{chosen} detect failed: {message}")


def _report_of(chosen: str, text: bytes, samples: bool) -> dict[str, Any]:
    """The report `detect` printed as `text`, checked to be one."""
    try:
        report = json.loads(text)
    except ValueError as error:
        raise RuntimeError(f"{chosen} detect did not print a report: {text[:200]!r}") from error
    if not isinstance(report, dict) or not isinstance(report.get("incidents"), list):
        raise RuntimeError(f"{chosen} detect printed something that is not a report")
    if samples and not isinstance(report.get("samples"), list):
        raise RuntimeError(f"{chosen} detect --samples printed a report without samples")
    return report


# --- one rank's detectors kept between a watcher's looks: `detect --samples --follow` ------------
#
# Wave AF-W. Before it, `--watch` with the Rust detectors made every record of the run into a row
# again at every look, sent them all to a new `detect --samples`, which judged them all, and read
# all of the run's samples back: about 7.2 s an update at 10,000 steps, as much as a fresh read.

# What each look of a `FollowedDetect` did, appended when this is a list (the tests read it): the
# counter that shows an append was not judged again.
LOOKS: list[dict[str, Any]] | None = None
# The same, one JSON line a look, in the file this names: how a test reads what a `--watch` command
# in another process did. Private; for tests.
LOOKS_ENV = "TENSORQUILL_FOLLOW_LOG"


class FollowedDetect:
    """One watcher's `tensorquill-server detect --samples --follow`: the rank's Rust detectors kept
    between looks, as the live server keeps them between flushes.

    `report(records, thresholds)` gives what `server_report(records, thresholds=..., samples=True)`
    gives. The first look sends every record. A later look sends only the records appended since,
    when the look is a plain append:
    - the records given before are the first records now, each the same record in its place;
    - the thresholds are the same;
    - each new record is of the newest attempt, with a step above every step before, in order.

    Scalar-only records are set apart first (`identity.is_scalar_record`, `WAVE_AL_CONTRACT.md` section 4.6):
    they are never sent, never kept and never decide what a look is, so a scalar-only record logged at
    any step leaves an append an append.

    Anything else (a first look, late rows, a new attempt, a truncated or replaced file, which give
    other records, a threshold changed) ends the child and starts another that is sent every
    record: a full recomputation. So does a refusal by the binary, which checks each append again,
    and refuses rows that name a router not seen before or give one more experts. The samples of
    an append are added to those kept; `samples_text` gives their JSON text, each sample's made once.

    `last` says what the last look did: `mode` (`full`, `append`, or `same` when no record was new),
    `records_sent`, `rows_read` (what the binary says it read), and `why` for a full look. `close`
    ends the child; the watcher calls it on every way out.
    """

    def __init__(self, binary: str | os.PathLike[str] | None = None, *, timeout: float | None = 300.0) -> None:
        self.binary = binary
        self.timeout = timeout
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr: Any = None
        self._chosen = ""
        self._thresholds: str | None = None
        self._records: list[Mapping[str, Any]] = []
        self._newest: int | None = None  # the newest attempt read, as the binary orders attempts
        self._top: int | None = None  # the newest attempt's highest step read
        self._report: dict[str, Any] | None = None
        self.samples: list[dict[str, Any]] = []
        self._texts: list[str] = []
        self.last: dict[str, Any] = {}

    # -- the child ------------------------------------------------------------------------------

    def close(self) -> None:
        """End the child, if there is one, and wait for it."""
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait()
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if self._stderr is not None:
                self._stderr.close()
                self._stderr = None

    def _start(self, thresholds: str | None) -> None:
        import tempfile

        self.close()
        self._chosen = server_binary(self.binary)
        command = [self._chosen, "detect", "--samples", "--follow"]
        if thresholds is not None:
            command += ["--thresholds", thresholds]
        # Standard error to a file, not a pipe: a child writing much of it could never block on it.
        self._stderr = tempfile.TemporaryFile()
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr)
        self._thresholds = thresholds

    def _failure(self) -> Exception:
        """What the child said when it ended, as `server_report` raises it."""
        process = self._process
        code = process.wait() if process is not None else None
        said = b""
        if self._stderr is not None:
            self._stderr.seek(0)
            said = self._stderr.read()
        self.close()
        return _detect_failure(self._chosen, said.decode("utf-8", "replace").strip() or f"exit status {code}")

    def _ask(self, records: list[Mapping[str, Any]]) -> dict[str, Any]:
        """Send `records` as one frame and read the child's line for it."""
        import select
        import struct
        import time

        process = self._process
        assert process is not None and process.stdin is not None and process.stdout is not None
        frame = records_stream(records)
        try:
            process.stdin.write(struct.pack("<Q", len(frame)))
            process.stdin.write(frame)
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            raise self._failure() from None
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        chunks: list[bytes] = []
        fd = process.stdout.fileno()
        while True:
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                self.close()
                raise RuntimeError(f"{self._chosen} detect --follow did not answer in {self.timeout:g} s")
            ready, _, _ = select.select([fd], [], [], left)
            if not ready:
                continue
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                raise self._failure()
            chunks.append(chunk)
            if chunk.endswith(b"\n"):
                break
        text = b"".join(chunks)
        try:
            answer = json.loads(text)
        except ValueError as error:
            raise RuntimeError(f"{self._chosen} detect --follow did not print a report: {text[:200]!r}") from error
        if isinstance(answer, dict) and "refused" in (answer.get("follow") or {}):
            return answer  # an append the binary would not take: the caller starts again
        return _report_of(self._chosen, text, True) if not isinstance(answer, dict) or "incidents" not in answer else answer

    # -- one look -------------------------------------------------------------------------------

    def report(self, records: Iterable[Mapping[str, Any]], *, thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """What `server_report(records, thresholds=thresholds, samples=True)` gives, judging only
        what was appended since the last look when that is all that changed (see the class)."""
        from .evidence import _same_record

        records = list(records)
        _something_to_judge(records)
        records = _training(records)
        try:
            sent = None if thresholds is None else json.dumps(dict(thresholds), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"the thresholds cannot be sent as JSON: {error}") from error
        known = len(self._records)
        why: str | None = None
        if self._process is None or self._report is None:
            why = "first look"
        elif sent != self._thresholds:
            why = "the thresholds changed"
        elif len(records) < known or not all(_same_record(self._records[i], records[i]) for i in range(known)):
            why = "the records read before are not the first records now"
        else:
            why = self._not_appended(records[known:])
        if why is None and len(records) == known:
            self._note("same", 0, 0, None)
            return self._handed()
        if why is None:
            answer = self._ask(records[known:])
            follow = answer.pop("follow", None) or {}
            if "refused" in follow:
                why = f"the binary refused the append: {follow['refused']}"
            else:
                if follow.get("samples_from") != len(self.samples):
                    self.close()
                    raise RuntimeError(f"{self._chosen} detect --follow gave samples from {follow.get('samples_from')}, "
                                       f"where {len(self.samples)} are kept")
                self.samples.extend(answer.pop("samples"))
                self._keep(records, answer, _whole(records[-1].get("step")))  # each appended step is above the last
                self._note("append", len(records) - known, follow.get("rows"), None)
                return self._handed()
        self._start(sent)
        answer = self._ask(records)
        follow = answer.pop("follow", None) or {}
        self.samples = answer.pop("samples")
        self._texts = []
        last = stream_order(records)[-1]  # the binary's last row read: the newest attempt's highest step
        self._keep(records, answer, _whole(last.get("step")))
        self._newest = _whole(last.get("attempt", 0))
        self._note("full", len(records), follow.get("rows"), why)
        return self._handed()

    def _not_appended(self, new: list[Mapping[str, Any]]) -> str | None:
        """Why `new` is not a plain append to what was read, or None when it is."""
        top = self._top
        for record in new:
            attempt, step = record.get("attempt", 0), record.get("step")
            if _whole(attempt) != self._newest:
                return f"a record of attempt {attempt}, where the newest attempt read is {self._newest}"
            if top is not None and _whole(step) <= top:
                return f"a record of step {step}, not above the last step read ({top})"
            top = _whole(step)
        return None

    def _keep(self, records: list[Mapping[str, Any]], answer: dict[str, Any], top: int) -> None:
        """Keep what was read. `top` is the step of the last row the binary read, in stream order:
        the newest attempt's highest step, as `detect --follow` takes it (`last_step`). Before wave AJ
        (A1) it was the highest step of every attempt, so after a resume below an earlier attempt's
        last step each look restarted the child in full until the new attempt passed it."""
        self._records = records
        self._top = top
        self._report = answer

    def _handed(self) -> dict[str, Any]:
        """The kept report, as `server_report` gives one: a copy of it, with the kept samples."""
        assert self._report is not None
        return {**json.loads(json.dumps(self._report)), "samples": self.samples}

    def _note(self, mode: str, sent: int, read: Any, why: str | None) -> None:
        self.last = {"mode": mode, "records_sent": sent, "rows_read": read, **({"why": why} if why else {})}
        if LOOKS is not None:
            LOOKS.append(dict(self.last))
        if logged := os.environ.get(LOOKS_ENV):
            with open(logged, "a", encoding="utf-8") as file:
                file.write(json.dumps(self.last) + "\n")

    def samples_text(self, samples: Any) -> str | None:
        """The JSON text of `samples` as `write_evidence` writes it, when that is the list kept here;
        each sample's text is made once. None for any other list."""
        if samples is not self.samples or len(self._texts) > len(samples):
            return None
        from .evidence import _json_text

        for sample in samples[len(self._texts):]:
            self._texts.append(_json_text(sample))
        return "[" + ",".join(self._texts) + "]"


# --- every rank of a run: `tensorquill-server detect --run` -----------------------------------
#
# `detect` reads one rank. With `--run` it reads the rows of every rank of one run, grouped by
# their `rank` column, judges each rank as before, and compares the ranks' batch hashes as the
# server does for a run's status. That comparison is `duplicate_batches`, whose rule reads the
# rows of several ranks.

# What the cross-rank comparison reads of a record: the keys, and its `data`.
_DATA_KEYS = ("schema_version", "run_id", "rank", "attempt", "step", "ts", "data")


def _whole(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def stream_order(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """`records` in the order the Rust detectors read them: by rank, then each rank's attempts in
    the order they first appear (contract 1: an attempt number is an identity, not an order),
    then by step within an attempt, rows of one step in the order given (lane W3's step-order
    rule). A stable sort, so nothing else moves. A key that is not a whole number sorts as 0;
    the binary refuses such a row anyway. The scalar-only records are not among them: the Rust
    detectors never read one (`WAVE_AL_CONTRACT.md` section 4.6)."""
    listed = _training(records)
    places: dict[tuple[int, int], int] = {}
    for record in listed:
        places.setdefault((_whole(record.get("rank", 0)), _whole(record.get("attempt", 0))), len(places))

    def key(record: Mapping[str, Any]) -> tuple[int, int, int]:
        rank, attempt = _whole(record.get("rank", 0)), _whole(record.get("attempt", 0))
        return rank, places[(rank, attempt)], _whole(record.get("step", 0))

    return sorted(listed, key=key)


def run_stream(records: Iterable[Mapping[str, Any]]) -> bytes:
    """The records of every rank of one run as the Arrow IPC stream `tensorquill-server detect
    --run` reads, in `stream_order` (as `records_stream` orders one rank), with the same check."""
    return _stream(records)


def server_run_report(
    records: Iterable[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, Any] | None = None,
    binary: str | os.PathLike[str] | None = None,
    timeout: float | None = 300.0,
) -> dict[str, Any]:
    """The server's detectors over every rank of one run, from records in hand: what
    `tensorquill-server detect --run` prints.

    Pass every attempt of every rank (`load_records(..., all_attempts=True)`); records of a
    second run are refused. Returns `{"run_id", "ranks", "reports", "ranks_compared",
    "duplicate_batches", "incidents", ...}`: `reports` holds each rank's report as
    `server_report` gives it for that rank alone, in rank order; `duplicate_batches` is the
    cross-rank finding or None; `incidents` lists it as an incident; and `ranks_compared` are
    the ranks that had a hashed batch to compare, whether or not anything was found.

    `thresholds`, `binary` and the errors are `server_report`'s.
    """
    records = list(records)
    _something_to_judge(records)
    chosen = server_binary(binary)
    command = [chosen, "detect", "--run"]
    if thresholds is not None:
        try:
            command += ["--thresholds", json.dumps(dict(thresholds), allow_nan=False)]
        except (TypeError, ValueError) as error:
            raise ValueError(f"the thresholds cannot be sent as JSON: {error}") from error
    try:
        done = subprocess.run(command, input=run_stream(records), capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{chosen} detect --run did not finish in {timeout:g} s") from error
    if done.returncode != 0:
        message = done.stderr.decode("utf-8", "replace").strip() or f"exit status {done.returncode}"
        if "invalid value" in message and "--thresholds" in message:
            refused = message.split("for '--thresholds <JSON>':", 1)[-1].split("\n", 1)[0].strip()
            raise ValueError(f"the Rust detectors refuse these thresholds: {refused}")
        if "unrecognized subcommand" in message or "unexpected argument" in message:
            message += " (this tensorquill-server is too old to have `detect --run`: rebuild it)"
        raise RuntimeError(f"{chosen} detect --run failed: {message}")
    try:
        document = json.loads(done.stdout)
    except ValueError as error:
        raise RuntimeError(f"{chosen} detect --run did not print a document: {done.stdout[:200]!r}") from error
    if not isinstance(document, dict) or not isinstance(document.get("reports"), list) \
            or not isinstance(document.get("incidents"), list):
        raise RuntimeError(f"{chosen} detect --run printed something that is not a run's document")
    return document


def server_duplicate_batches(
    records: Iterable[Mapping[str, Any]],
    *,
    binary: str | os.PathLike[str] | None = None,
    timeout: float | None = 300.0,
) -> dict[str, Any] | None:
    """`duplicate_batches` across the ranks of a run, from the Rust rule: the incident
    `data_checks.detect_duplicate_batches(resolve_attempts(records))` gives (here with its
    `class`), or None.

    Pass every attempt of every rank: the Rust side leaves out what a resume rolled back, as
    `resolve_attempts` does. Only each record's keys and `data` are sent, so this costs far
    less than `server_run_report`, and nothing else of a rank is judged. None is also the
    answer when fewer than two ranks drew a hashed batch; `server_run_report`'s
    `ranks_compared` tells the two apart.
    """
    # A scalar-only record is set apart before its keys are slimmed away, which would make it a step's.
    slim = [{key: record[key] for key in _DATA_KEYS if key in record} for record in _training(records)]
    if not slim:
        return None
    document = server_run_report(slim, binary=binary, timeout=timeout)
    found = [i for i in document["incidents"] if isinstance(i, dict) and i.get("kind") == "duplicate_batches"]
    return found[0] if found else None
