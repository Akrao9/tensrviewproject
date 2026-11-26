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


FLIPPED = [*[_step(100, s) for s in (10, 11, 12)], _point(200, 2, 0, val=2.0), _step(200, 5),
           _point(100, 2, 0, val=999.0)]


def test_bug20_the_command_lines_rerun_export_takes_the_order_from_before_the_rollback(tmp_path, monkeypatch):
    """`python -m tensorquill <file> --rerun`: the records it exports are resolved, which here puts
    attempt 200 first; the order is taken from every record it read."""
    logged = _rerun_recorder(monkeypatch)
    path = _write(tmp_path / "r.rank0.jsonl", FLIPPED)
    monkeypatch.setattr(sys, "argv", ["tensorquill", str(path), "--detectors", "python", "--rerun",
                                      str(tmp_path / "r.rrd"), "--output", str(tmp_path / "e.json")])
    evidence.main()
    assert [(step, value[1]) for step, entity, value in logged if entity == "/logged/val"] == [(2, 2.0)]


def test_bug20_the_export_takes_an_order_given_for_records_already_resolved(tmp_path, monkeypatch):
    logged = _rerun_recorder(monkeypatch)
    built = evidence.build_evidence(LATE, detectors="python")
    evidence.export_rerun(built, tmp_path / "r.rrd", records=evidence.resolve_attempts(FLIPPED),
                          attempt_order=evidence.attempts_first_seen(FLIPPED))
    assert [(step, value[1]) for step, entity, value in logged if entity == "/logged/val"] == [(2, 2.0)]


# The two callers of the merge in other lanes' files. Each set its scalar-only records apart before
# calling it and gives no order yet: they turn on at the join, once the coordinator passes the
# stream's order there (`attempts_first_seen`; the report names the lines).

@pytest.mark.xfail(strict=True, reason="mcp_server.py (lane AO-9's file) gives merged_scalar_points no order yet")
def test_bug20_at_the_join_the_mcp_series_gives_the_newer_attempts_value(tmp_path):
    from tensorquill.mcp_server import Source, get_series

    path = _write(tmp_path / "r.rank0.jsonl", [_step(10, 0), _point(20, 1, 0, loss=20.0), _point(10, 1, 0, loss=10.0),
                                                _step(20, 2)])
    assert get_series(Source(path), {"kind": "scalar", "name": "loss"})["values"] == {"value": [20.0]}


@pytest.mark.xfail(strict=True, reason="divergence.py (lane AO-6's file) gives scalar_points_in_order no order yet")
def test_bug20_at_the_join_first_divergence_reads_the_newer_attempts_point():
    from tensorquill.divergence import first_divergence

    def run(name: str, late: bool) -> list[dict]:
        rows = [{**_step(10, s, scalars={"loss": 1.0, "val": 2.0}), "run_id": name} for s in range(30)]
        rows += [{**_point(20, 31, 0, val=2.0), "run_id": name}]
        rows += [{**_step(20, s, scalars={"loss": 1.0, "val": 2.0}), "run_id": name} for s in range(35, 60)]
        if late:  # attempt 10's late point at 31, below attempt 20's first training step (35)
            rows.append({**_point(10, 31, 0, val=999.0), "run_id": name})
        return rows

    assert first_divergence(run("a", late=True), run("b", late=False))["first_exact_difference"] is None
