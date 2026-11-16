"""The Python detectors' outputs, frozen as files, and both detector paths held to them.

`golden/` has one file per record set. It holds what `tensorquill.build_evidence(records)`,
the Python detectors, gave for `incidents`, `not_checked`, `run_state` and the per-step MoE
`samples`. The Python detectors are to be deleted once the Rust ones do the same work, and
from then on these files are the reference. Until then each file is checked twice:

- the Python path still gives it, which keeps the files honest while the Python code exists;
- the Rust path (`detectors="rust"`) gives the same, compared as the parity tests compare
  (`differences`: numbers to 1e-12, a NaN in Python as `null` in Rust). Its `samples` are
  still made in Python today; they are compared so that the test holds when they move.

The record sets are `CASES` of test_server_detect.py and the seven Gate 1 fixtures, made here
by `experiments/moe-bakeoff/generate.py` and passed through the SDK's writer and reader as
that script does it (its `artifacts/` folder is not in the repository). The two places where
the paths differ on purpose, pinned at the end of test_server_detect.py, are not frozen.

A file is strict JSON with sorted keys, one incident, check or sample to a line:

- `case` is the record set's name, and `records` its length and a SHA-256 of its rows: in
  order, as compact JSON with sorted keys and numbers rounded to nine significant digits. A
  file is only compared with output made from the records it was frozen from.
- A NaN is written `{"$float": "nan"}` and an infinity `{"$float": "inf"}` or `"-inf"`.

The Python path is held to a file exactly in its keys, order, integers, text and kinds of
value, and to 1e-12 in its finite numbers. The files were written on macOS, and a platform's
`log` and `exp` may differ in the last bit, both in the fixtures and in the routing
diversity. The record hash rounds its numbers for the same reason.

To rewrite the files on purpose, from `sdk/`:

    pytest tests/test_golden.py --write-golden

It rewrites the files that no longer hold what the Python detectors give, adds the missing
ones, removes those of cases that are gone, and then runs the tests on the result. Read the
diff before committing it.

Needs `pyarrow` (the `flight` extra), because the record sets are built in modules that send
to the server. The Rust half also needs the server binary
(`cargo build --manifest-path server/Cargo.toml`, or `TENSORQUILL_SERVER_BIN`).
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow.flight")

import tensorquill  # noqa: E402

from test_flight_e2e import BINARY  # noqa: E402
from test_server_detect import CASES  # noqa: E402
from test_server_detectors import differences  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden"
GENERATE = Path(__file__).resolve().parents[2] / "experiments" / "moe-bakeoff" / "generate.py"
FROZEN = ("incidents", "not_checked", "run_state", "samples")
TAG = "$float"
TOLERANCE = 1e-12  # relative, as `differences` has it
HASH_DIGITS = 9
REWRITE = "if the change is meant, rewrite the files with `pytest tests/test_golden.py --write-golden` and read the diff"


def _gate_1() -> dict[str, Any]:
    """The Gate 1 fixtures, as the script that makes them defines them."""
    spec = importlib.util.spec_from_file_location("moe_bakeoff_generate", GENERATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # `dataclass` looks a class's module up while the script runs
    spec.loader.exec_module(module)
    return {f"gate 1 {fixture.name}": fixture for fixture in module.FIXTURES}


GATE_1 = _gate_1()
NAMES = [*CASES, *GATE_1]


@functools.lru_cache(maxsize=None)
def records(name: str) -> list[dict]:
    if name in CASES:
        return CASES[name]
    fixture = GATE_1[name]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"{fixture.name}.jsonl"
        sink = tensorquill.JsonlSink(path)
        sink.write(fixture.records())
        sink.close()
        return tensorquill.load_records(path, all_attempts=True)


def file_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") + ".json"


# --- how a file is written and read ---------------------------------------------------------


def tagged(value: Any, digits: int | None = None) -> Any:
    """A value as strict JSON holds it: a NaN or an infinity as a `$float` object, a tuple as
    a list. With `digits`, finite floats are rounded to that many significant digits."""
    if isinstance(value, float):
        if math.isnan(value):
            return {TAG: "nan"}
        if math.isinf(value):
            return {TAG: "inf" if value > 0 else "-inf"}
        return value if digits is None else float(f"{value:.{digits}g}")
    if isinstance(value, dict):
        assert TAG not in value, f"a key named {TAG} cannot be frozen"
        return {key: tagged(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tagged(item, digits) for item in value]
    return value


def untagged(value: Any) -> Any:
    """A parsed file with its `$float` objects as the floats they stand for."""
    if isinstance(value, dict):
        return float(value[TAG]) if set(value) == {TAG} else {key: untagged(item) for key, item in value.items()}
    return [untagged(item) for item in value] if isinstance(value, list) else value


def _line(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def text(document: dict) -> str:
    """A tagged document as its file's text, with each item of a list on a line of its own,
    so that a diff shows which incident or sample changed."""
    parts = []
    for key, value in sorted(document.items()):
        body = "[\n" + ",\n".join(map(_line, value)) + "\n]" if isinstance(value, list) and value else _line(value)
        parts.append(f"{_line(key)}:{body}")
    return "{\n" + ",\n".join(parts) + "\n}\n"


def parsed(written: str) -> Any:
    def refuse(token: str) -> None:
        raise ValueError(f"{token} is not strict JSON")

    return json.loads(written, parse_constant=refuse)


def records_hash(rows: list[dict]) -> str:
    return hashlib.sha256(_line(tagged(rows, HASH_DIGITS)).encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=None)
def stamp(name: str) -> dict:
    return {"count": len(records(name)), "sha256": records_hash(records(name))}


@functools.lru_cache(maxsize=None)
def python_document(name: str) -> dict:
    """What the Python detectors give for a case now, as its file would be read back."""
    evidence = tensorquill.build_evidence(records(name))
    document = {"case": name, "records": stamp(name), **{key: evidence.get(key, []) for key in FROZEN}}
    return parsed(text(tagged(document)))


def golden(name: str, folder: Path = GOLDEN) -> dict:
    """A case's golden file, which must have been frozen from the records the case has now."""
    path = folder / file_name(name)
    assert path.is_file(), f"{path.name} is missing: {REWRITE}"
    held = parsed(path.read_text(encoding="utf-8"))
    assert set(held) == {"case", "records", *FROZEN}, f"{path.name} holds {sorted(held)}"
    assert (held["case"], held["records"]) == (name, stamp(name)), (
        f"{path.name} was frozen from other records than {name!r} has now: {REWRITE}"
    )
    return held


def changes(held: Any, now: Any, path: str) -> list[str]:
    """Where output differs from a frozen value, both as read back from a file: kinds of
    value, keys, lengths, order, integers and text exactly, a finite number to `TOLERANCE`."""
    if type(held) is not type(now):
        return [f"{path}: {held!r} != {now!r}"]
    if isinstance(held, dict):
        out = [f"{path}: keys differ by {sorted(set(held) ^ set(now))}"] if set(held) != set(now) else []
        return out + [d for key in sorted(set(held) & set(now)) for d in changes(held[key], now[key], f"{path}.{key}")]
    if isinstance(held, list):
        if len(held) != len(now):
            return [f"{path}: {len(held)} items != {len(now)}"]
        return [d for i, (a, b) in enumerate(zip(held, now)) for d in changes(a, b, f"{path}[{i}]")]
    same = math.isclose(held, now, rel_tol=TOLERANCE, abs_tol=0.0) if isinstance(held, float) else held == now
    return [] if same else [f"{path}: {held!r} != {now!r}"]


def brief(found: list[str], limit: int = 20) -> list[str]:
    return found[:limit] + ([f"and {len(found) - limit} more"] if len(found) > limit else [])


def write_golden(folder: Path = GOLDEN) -> list[str]:
    """Make `folder` hold what the Python detectors give now, and name the files written or removed.

    A file that still holds its case's output, as the test below judges it, is left as it is:
    rewriting on another machine must not change last digits in every file. The file of a
    case that is gone is removed.
    """
    folder.mkdir(parents=True, exist_ok=True)
    touched = []
    for name in NAMES:
        path, now = folder / file_name(name), python_document(name)
        try:
            held = parsed(path.read_text(encoding="utf-8"))
            current = set(held) == set(now) and not any(changes(held[key], now[key], key) for key in now)
        except (OSError, ValueError):
            current = False
        if not current:
            path.write_text(text(now), encoding="utf-8")
            touched.append(path.name)
    for path in sorted(folder.glob("*.json")):
        if path.name not in map(file_name, NAMES):
            path.unlink()
            touched.append(path.name)
    return touched


@pytest.fixture(scope="module", autouse=True)
def _rewritten(request):
    """With `--write-golden` (conftest.py), the files are rewritten before a test reads them."""
    if request.config.getoption("--write-golden", default=False):
        touched = write_golden()
        reporter = request.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line("")
            named = f": {', '.join(touched)}" if 0 < len(touched) <= 10 else ""
            reporter.write_line(f"--write-golden wrote or removed {len(touched)} of {len(NAMES)} files in {GOLDEN}{named}")


# --- the files ------------------------------------------------------------------------------


def test_each_case_has_its_own_file_and_no_file_is_left_over():
    files = {file_name(name) for name in NAMES}
    assert len(files) == len(NAMES), "two cases would share a file"
    found = {path.name for path in GOLDEN.glob("*.json")}
    assert found == files, f"missing {sorted(files - found)}, left over {sorted(found - files)}: {REWRITE}"


@pytest.mark.parametrize("name", NAMES)
def test_the_python_detectors_still_give_the_golden_file(name):
    held, now = golden(name), python_document(name)
    assert brief([d for key in FROZEN for d in changes(held[key], now[key], key)]) == [], REWRITE


@pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
@pytest.mark.parametrize("name", NAMES)
def test_the_rust_detectors_give_the_golden_file(name):
    held = untagged(golden(name))
    rust = tensorquill.build_evidence(records(name), detectors="rust", server_binary=BINARY)
    assert rust["detectors"] == "rust"
    assert brief([d for key in FROZEN for d in differences(held[key], rust.get(key, []), key)]) == []


# --- the rules the files are written and compared by ----------------------------------------


def test_a_nan_and_an_infinity_are_tagged_in_a_file_and_floats_again_for_the_rust_comparison():
    incident = {"value": float("nan"), "limits": (float("inf"), float("-inf")), "text": "nan", "none": None}
    written = text(tagged({"case": "x", "incidents": [incident], "not_checked": []}))
    assert written == (
        '{\n"case":"x",\n"incidents":[\n'
        '{"limits":[{"$float":"inf"},{"$float":"-inf"}],"none":null,"text":"nan","value":{"$float":"nan"}}\n'
        '],\n"not_checked":[]\n}\n'
    )
    back = untagged(parsed(written))["incidents"][0]
    assert math.isnan(back["value"]) and back["limits"] == [math.inf, -math.inf] and back["text"] == "nan"
    # The server writes all three as `null`.
    assert differences(back, {"value": None, "limits": [None, None], "text": None, "none": None}) == []
    with pytest.raises(ValueError, match="strict JSON"):
        parsed('{"value": NaN}')


def test_the_python_comparison_is_exact_but_for_the_last_digits_of_a_number():
    held = parsed(text(tagged({"a": [1.5, 2, float("nan")], "b": "nan"})))
    assert changes(held, {"a": [1.5 * (1 + 1e-13), 2, {TAG: "nan"}], "b": "nan"}, "x") == []
    for other in (
        {"a": [1.5 * (1 + 1e-11), 2, {TAG: "nan"}], "b": "nan"},  # a number past the tolerance
        {"a": [1.5, 2.0, {TAG: "nan"}], "b": "nan"},  # a float where an integer was
        {"a": [1.5, True, {TAG: "nan"}], "b": "nan"},
        {"a": [2, 1.5, {TAG: "nan"}], "b": "nan"},  # the order
        {"a": [1.5, 2, {TAG: "inf"}], "b": "nan"},  # another kind of non-finite number
        {"a": [1.5, 2, None], "b": "nan"},
        {"a": [1.5, 2, {TAG: "nan"}], "b": {TAG: "nan"}},  # the text "nan" is not a NaN
        {"a": [1.5, 2, {TAG: "nan"}, 3], "b": "nan"},
        {"a": [1.5, 2, {TAG: "nan"}], "b": "nan", "c": None},
    ):
        assert changes(held, other, "x") != [], other


def test_the_record_hash_changes_with_a_value_a_kind_or_the_order_but_not_with_a_last_bit():
    rows = [{"step": 0, "loss": 0.1 + 0.2, "bad": float("nan")}, {"step": 1, "loss": 2.5}]
    assert 0.1 + 0.2 != 0.3
    assert records_hash(rows) == records_hash([{"bad": float("nan"), "loss": 0.3, "step": 0}, rows[1]])
    for other in (
        rows[::-1], rows[:1], [{**rows[0], "loss": 0.3000001}, rows[1]], [{**rows[0], "step": 0.0}, rows[1]],
        [{**rows[0], "bad": None}, rows[1]], [{**rows[0], "bad": float("inf")}, rows[1]],
    ):
        assert records_hash(other) != records_hash(rows), other


def test_rewriting_touches_only_the_files_that_are_wrong(tmp_path):
    folder = tmp_path / "golden"
    assert len(write_golden(folder)) == len(NAMES) and write_golden(folder) == []
    kept, wrong, gone = (folder / file_name(name) for name in ("moe edges", "everything", "attention"))
    before = kept.read_bytes()
    wrong.write_text(wrong.read_text(encoding="utf-8").replace('"kind":"grad_spike"', '"kind":"grad_spikes"'), encoding="utf-8")
    gone.unlink()
    (folder / "a-case-that-is-gone.json").write_text("{}\n", encoding="utf-8")
    assert sorted(write_golden(folder)) == sorted(["a-case-that-is-gone.json", gone.name, wrong.name])
    assert kept.read_bytes() == before and not (folder / "a-case-that-is-gone.json").exists()
    assert golden("everything", folder) == python_document("everything") and write_golden(folder) == []
