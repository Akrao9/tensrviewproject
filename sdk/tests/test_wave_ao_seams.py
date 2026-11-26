"""Wave AO's seam (lane AO-S): what no lane of wave AO could test alone, on the joined tree.

1. BUG-20's callers: the MCP series, `first_divergence` and `compare_runs` take the attempt order of
   the whole record stream, as `scalar_series` does, also where a rollback leaves out every training
   record of the older attempt (so the records left show the newer attempt first).

Records are written here by hand, in the shapes `JsonlSink` writes. Tests that need pyarrow or a
`tensorquill-server` binary (`TENSORQUILL_SERVER_BIN`) skip one by one without them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorquill import evidence


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
