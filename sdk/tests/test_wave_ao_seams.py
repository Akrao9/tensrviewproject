"""Wave AO's seam (lane AO-S): what no lane of wave AO could test alone, on the joined tree.

1. BUG-20's callers: the MCP series, `first_divergence` and `compare_runs` take the attempt order of
   the whole record stream, as `scalar_series` does, also where a rollback leaves out every training
   record of the older attempt (so the records left show the newer attempt first).
2. The two overflow guards no lane made (lane AO-9's list): an integer too large for a float is
   unreadable to the Python detectors, as the Rust detectors read it (`detect/extra_json.rs`), and
   not finite to `first_divergence`; nothing raises.

Records are written here by hand, in the shapes `JsonlSink` writes. Tests that need pyarrow or a
`tensorquill-server` binary (`TENSORQUILL_SERVER_BIN`) skip one by one without them.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import pytest

from tensorquill import evidence, server_detect

BINARY = server_detect.find_binary()
needs_pyarrow = pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow is not installed")
needs_binary = pytest.mark.skipif(BINARY is None or not os.path.isfile(BINARY),
                                  reason="tensorquill-server is not built (TENSORQUILL_SERVER_BIN)")


def _write(path: Path, records: list[dict], tail: str = "") -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + tail)
    return path


def _step(attempt: int, step: int, run_id: str = "r", **more) -> dict:
    return {"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": attempt, "step": step, **more}


def _point(attempt: int, step: int, seq: int, run_id: str = "r", **scalars) -> dict:
    return {"schema_version": 2, "record": "scalars", "run_id": run_id, "rank": 0, "attempt": attempt,
            "step": step, "seq": seq, "scalars": scalars}


# --- 1. BUG-20's callers -------------------------------------------------------------------------------
#
# Lane AO-5's shapes. LATE: attempt 100 trains steps 0 to 6, attempt 200 resumes at step 5 and logs
# `val` at step 2, then attempt 100 flushes its own late `val` at step 2; both points survive the
# rollback (2 < 5), and 200, which first appears later in the stream, stands. ROLLED_BACK_WHOLE:
# attempt 100 trained only steps 10 to 12, all rolled back by attempt 200's resume at step 5, so once
# resolved the records left begin with attempt 200's: only the stream read before the rollback says
# 100 is the older attempt.

LATE = [*[_step(100, s, scalars={"loss": 10.0}) for s in range(7)],
        _point(200, 2, 0, val=2.0),
        _step(200, 5, scalars={"loss": 1.0}),
        _point(100, 2, 0, val=999.0)]

ROLLED_BACK_WHOLE = [*[_step(100, s) for s in (10, 11, 12)], _point(200, 2, 0, val=2.0), _step(200, 5),
                     _point(100, 2, 0, val=999.0)]


@pytest.mark.parametrize("records", [LATE, ROLLED_BACK_WHOLE], ids=["late", "rolled_back_whole"])
def test_bug20_the_mcp_series_gives_the_newer_attempts_point(tmp_path, records):
    from tensorquill.mcp_server import Source, get_series

    path = _write(tmp_path / "r.rank0.jsonl", records)
    assert evidence.scalar_series(path)["val"] == [[2, 2.0]]
    assert get_series(Source(path), {"kind": "scalar", "name": "val"})["values"] == {"value": [2.0]}


def _compared_run(run_id: str, late: bool, whole: bool) -> list[dict]:
    """A run long enough to compare. Attempt 20 logs `val` at step 31 and trains steps 35 to 99;
    attempt 10 trained steps 0 to 29 (`whole=False`, kept below 35) or 40 to 45 (`whole=True`,
    rolled back whole), and with `late` its own late `val` at 31 follows, far off. The newer
    attempt's point stands, so a run with the late point trains as one without it."""
    trained = range(40, 46) if whole else range(30)
    rows = [_step(10, s, run_id, scalars={"loss": 1.0, "val": 2.0}) for s in trained]
    rows.append(_point(20, 31, 0, run_id, val=2.0))
    rows += [_step(20, s, run_id, scalars={"loss": 1.0, "val": 2.0}) for s in range(35, 100)]
    if late:
        rows.append(_point(10, 31, 0, run_id, val=999.0))
    return rows


@pytest.mark.parametrize("whole", [False, True], ids=["late", "rolled_back_whole"])
@pytest.mark.parametrize("resolved", [False, True], ids=["raw", "loaded"])
def test_bug20_first_divergence_reads_the_newer_attempts_point(whole, resolved):
    """Raw rows (`all_attempts=True`), and for the shape whose older attempt keeps its first records,
    rows `load_records` resolved too. Rows resolved by `load_records` from the shape rolled back
    whole no longer show which attempt is older: the README says to give such rows raw."""
    from tensorquill.divergence import first_divergence

    a, b = _compared_run("a", True, whole), _compared_run("b", False, whole)
    if resolved:
        if whole:
            pytest.skip("rows resolved by load_records lose the older attempt's place (README, first_divergence)")
        a, b = evidence.load_records(a, scalars=True), evidence.load_records(b, scalars=True)
    found = first_divergence(a, b)
    assert found["first_exact_difference"] is None, found["first_exact_difference"]


@pytest.mark.parametrize("whole", [False, True], ids=["late", "rolled_back_whole"])
@pytest.mark.parametrize("torn", [False, True], ids=["whole_file", "torn_last_line"])
def test_bug20_compare_runs_on_one_file_a_rank_reads_the_newer_attempts_point(tmp_path, whole, torn):
    from tensorquill.compare import compare_runs

    tail = '{"schema_version": 1, "run_id": "a", "rank": 0, "attempt": 20, "step": 10' if torn else ""
    a = _write(tmp_path / "a.rank0.jsonl", _compared_run("a", True, whole), tail)
    b = _write(tmp_path / "b.rank0.jsonl", _compared_run("b", False, whole))
    part = compare_runs(a, b)["divergence"]
    assert part["first_exact_difference"] is None, part["lines"]
    assert part["state"] == "no_divergence", part["lines"]


# --- 2. An integer too large for a float ----------------------------------------------------------------

BIG = 10**400


def _plain_run(at: int = 20, steps: int = 40, **big) -> list[dict]:
    """A healthy run whose record at `at` holds BIG in the places `big` names."""
    rows = []
    for step in range(steps):
        row = {"schema_version": 1, "run_id": "r", "rank": 0, "step": step, "grad_norm": 1.0,
               "layers": {"blk": {"mean": 0.0, "std": 1.0, "abs_max": 1.0, "zero_frac": 0.0}},
               "grad_norms": {"w": 1.0}}
        if step == at:
            if "grad_norm" in big:
                row["grad_norm"] = BIG
            if "clip_grad_norm" in big:
                row["clip_grad_norm"] = BIG
            if "layer" in big:
                row["layers"]["blk"]["abs_max"] = BIG
            if "parameter" in big:
                row["grad_norms"]["w"] = BIG
        rows.append(row)
    return rows


PLACES = ["grad_norm", "clip_grad_norm", "layer", "parameter"]


def _judged(evidence_: dict) -> tuple:
    return (evidence_["run_state"], sorted(i["kind"] for i in evidence_["incidents"]))


@pytest.mark.parametrize("place", PLACES)
def test_the_python_detectors_read_an_integer_too_large_for_a_float_as_unreadable(place):
    """`evidence._finite` gives no value and `_is_non_finite` says no: a finite number that no float
    holds is neither a measurement nor a NaN or an infinity, so no `non_finite` fires."""
    found = evidence.build_evidence(_plain_run(**{place: True}), detectors="python")
    assert _judged(found) == _judged(evidence.build_evidence(_plain_run(), detectors="python"))
    assert evidence._finite(BIG) is None and evidence._finite(-BIG) is None
    assert evidence._is_non_finite(BIG) is False and evidence._is_non_finite(-BIG) is False
    assert evidence._is_non_finite(math.inf) is True and evidence._finite(2**1023) == float(2**1023)


@needs_pyarrow
@needs_binary
@pytest.mark.parametrize("place", PLACES)
def test_both_engines_judge_a_record_holding_an_integer_too_large_for_a_float_alike(place):
    """The Rust detectors read such a number as unreadable where a float is wanted
    (`detect/extra_json.rs`, wave AJ); the Python detectors now read it so too."""
    records = _plain_run(**{place: True})
    python, rust = (evidence.build_evidence(records, detectors=engine) for engine in ("python", "rust"))
    assert _judged(python) == _judged(rust)
    assert "non_finite" not in _judged(rust)[1]


def _loss_run(run_id: str, third) -> list[dict]:
    return [{"schema_version": 1, "run_id": run_id, "rank": 0, "step": step,
             "scalars": {"loss": third if step == 2 else 1.0 + 0.01 * step}} for step in range(60)]


@pytest.mark.parametrize("sign", [1, -1])
def test_first_divergence_reads_an_integer_too_large_for_a_float_as_an_infinity(sign):
    """`divergence._number` keeps the point, as the infinity of its sign: the runs then compare as
    they do with the infinity recorded, and the difference at step 2 is not lost."""
    from tensorquill.divergence import _number, first_divergence

    assert _number(sign * BIG) == sign * math.inf
    big = first_divergence(_loss_run("a", sign * BIG), _loss_run("b", 3.0))
    infinite = first_divergence(_loss_run("a", sign * math.inf), _loss_run("b", 3.0))
    assert big == infinite
    assert big["first_exact_difference"] == {"step": 2, "series": 1}
