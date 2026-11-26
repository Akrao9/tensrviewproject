"""Wave AO, lane AO-5: findings on `evidence.py` at the wave AL candidate.

Each section is one finding: the test that showed it first, then the pins beside the fix. The
records are written here, by hand, as `JsonlSink` writes them; none comes from a golden fixture.

Needs nothing beyond the SDK and pytest, except where a test says so: the Rerun export tests
replace Rerun with a recorder and need NumPy, and the tests that compare the two engines need
pyarrow and a `tensorquill-server` binary (`TENSORQUILL_SERVER_BIN`); each skips on its own.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from tensorquill import evidence


def _write(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def _step(attempt: int, step: int, **more) -> dict:
    return {"schema_version": 1, "run_id": "r", "rank": 0, "attempt": attempt, "step": step, **more}


def _point(attempt: int, step: int, seq: int, **scalars) -> dict:
    return {"schema_version": 2, "record": "scalars", "run_id": "r", "rank": 0, "attempt": attempt, "step": step,
            "seq": seq, "scalars": scalars}


def _rerun_recorder(monkeypatch) -> list[tuple[int | None, str, object]]:
    """Rerun replaced by a recorder of `(step, entity, value)` for each `rr.log`; the entity path
    escaping is Rerun's own rule for the characters these tests use (a `/` in a part is `\\/`)."""
    pytest.importorskip("numpy")
    logged: list[tuple[int | None, str, object]] = []
    now: list[int | None] = [None]
    rr = mock.MagicMock(name="rerun")
    rr.escape_entity_path_part.side_effect = lambda part: part.replace("\\", "\\\\").replace("/", "\\/")
    rr.set_time.side_effect = lambda _timeline, sequence: now.__setitem__(0, sequence)
    rr.Scalars.side_effect = lambda value: ("scalars", value)
    rr.log.side_effect = lambda entity, value, **_: logged.append((now[0], entity, value))
    blueprint = mock.MagicMock(name="rerun.blueprint")
    rr.blueprint = blueprint
    monkeypatch.setitem(sys.modules, "rerun", rr)
    monkeypatch.setitem(sys.modules, "rerun.blueprint", blueprint)
    return logged


# ---- BUG-20: a superseded attempt's late scalar ------------------------------------------------
#
# Attempt 100 trains steps 0 to 6. Attempt 200 resumes at step 5 and logs `val` at step 2 first;
# attempt 100, still running, then flushes its own late `val` at step 2. Both points survive the
# rollback (2 < 5). The attempt that first appears later in the whole stream is 200 (contract,
# AL-4′ decision 2), so its point stands; in the scalar-only records alone 200 appears first.

LATE = [*[_step(100, s, scalars={"loss": 10.0}) for s in range(7)],
        _point(200, 2, 0, val=2.0),
        _step(200, 5, scalars={"loss": 1.0}),
        _point(100, 2, 0, val=999.0)]


def test_bug20_a_superseded_attempts_late_scalar_does_not_replace_the_newer_one(tmp_path):
    path = _write(tmp_path / "r.rank0.jsonl", LATE)
    assert evidence.scalar_series(path)["val"] == [[2, 2.0]]
    assert evidence.scalar_series(LATE)["val"] == [[2, 2.0]]


def test_bug20_the_second_audits_order_gives_the_newer_attempts_value(tmp_path):
    """Attempt 10 trains step 0; attempt 20 logs `loss` at step 1; attempt 10 logs its late `loss` at
    step 1; attempt 20 trains step 2. The later attempt's 20.0 stands."""
    records = [_step(10, 0), _point(20, 1, 0, loss=20.0), _point(10, 1, 0, loss=10.0), _step(20, 2)]
    assert evidence.scalar_series(_write(tmp_path / "r.rank0.jsonl", records))["loss"] == [[1, 20.0]]


def test_bug20_the_order_is_the_whole_streams_before_any_record_is_resolved_away():
    """Attempt 100 trains steps 10 to 12 and is rolled back whole by attempt 200, which resumes at
    step 5: once resolved, 200's records come first. 100's late point at step 2 still survives
    (2 < 5) beside 200's, and 200 is still the later attempt."""
    records = [*[_step(100, s) for s in (10, 11, 12)], _point(200, 2, 0, val=2.0), _step(200, 5),
               _point(100, 2, 0, val=999.0)]
    assert evidence.scalar_series(records)["val"] == [[2, 2.0]]


def test_bug20_merged_scalar_points_takes_the_streams_order():
    training, scalar_only = evidence.set_apart_scalars(LATE)
    order = evidence.attempts_first_seen(LATE)
    assert order == {("r", 0): [100, 200]}
    assert evidence.merged_scalar_points(training, scalar_only, order=order)["val"] == {2: 2.0}
    assert [r["attempt"] for r in evidence.scalar_points_in_order(scalar_only, order=order)] == [100, 200]


def test_bug20_the_rerun_export_logs_the_newer_attempts_point(tmp_path, monkeypatch):
    logged = _rerun_recorder(monkeypatch)
    built = evidence.build_evidence(LATE, detectors="python")
    evidence.export_rerun(built, tmp_path / "r.rrd", records=evidence.load_records(LATE, scalars=True))
    assert [(step, value[1]) for step, entity, value in logged if entity == "/logged/val"] == [(2, 2.0)]


def test_bug20_pin_the_old_calls_still_work_and_order_by_first_appearance_among_them():
    """A caller that gives no order (`mcp_server`, `divergence` until the join) is read as before."""
    training, scalar_only = evidence.set_apart_scalars(LATE)
    assert [r["attempt"] for r in evidence.scalar_points_in_order(scalar_only)] == [200, 100]
    assert evidence.merged_scalar_points(training, scalar_only)["val"] == {2: 999.0}


def test_bug20_pin_within_one_attempt_the_highest_seq_stands_and_a_point_beats_training():
    records = [_step(1, 0, scalars={"loss": 5.0}), _point(1, 0, 1, loss=3.0), _point(1, 0, 0, loss=4.0),
               _step(1, 1, scalars={"loss": 2.0})]
    assert evidence.scalar_series(records)["loss"] == [[0, 3.0], [1, 2.0]]


def test_bug20_pin_all_attempts_keeps_the_rolled_back_point_and_the_later_attempt_still_stands():
    records = [*[_step(1, s) for s in range(4)], _point(1, 3, 0, val=1.0), _step(2, 2), _point(2, 3, 0, val=2.0)]
    assert evidence.scalar_series(records)["val"] == [[3, 2.0]]
    assert evidence.scalar_series(records, all_attempts=True)["val"] == [[3, 2.0]]
    # Without attempt 2's point, attempt 1's at 3 is rolled back by attempt 2's first training step, 2.
    assert "val" not in evidence.scalar_series(records[:6])
    assert evidence.scalar_series(records[:6], all_attempts=True)["val"] == [[3, 1.0]]
