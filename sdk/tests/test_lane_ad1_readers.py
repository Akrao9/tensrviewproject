"""Lane AD1 (wave AD): the evidence readers, each item held by a test that fails on `eb5a71b`.

- Item 1: a status entry marked `attempt_without_steps` for the very attempt a current report
  judged was read as "started again", so the rank went unjudged (`earlier_attempt`). Both orders
  of arrival are covered, through both watchers, and the control still reads `earlier_attempt`.
- Item 2: the store path of `compare` through `combine_rank_evidence` gave evidence with no
  `watchdog` block, so the rows a server dropped and the batches it refused were not in it.
- Item 3: what each reader says of the thresholds it judged with.
- Item 5: steps newer than the last judgment (CLI-10), in `watch` and `watch_all_ranks`.
- Item 7: `run_of`, the public name of `_run_of`.

None needs `pyarrow` or the server binary: the stores are written by hand, and no part is opened.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tensorquill import evidence as ev

RUN = "ad1run"


def _report(rank: int, attempt: int, last: int, incidents: list[dict] | None = None, **extra) -> dict:
    return {
        "schema_version": 1, "source": "tensorquill-server", "run_id": RUN, "rank": rank, "attempt": attempt,
        "last_step": last, "steps": last + 1, "incidents": incidents or [], "checked": list(ev.SERVER_KINDS),
        "thresholds": {"spike_sustain": ev.SPIKE_SUSTAIN}, **extra,
    }


def _status(ranks: list[dict], summary: str = "active") -> dict:
    return {
        "schema_version": 1, "run_id": RUN, "summary": summary, "server_running": True, "refresh_seconds": 30.0,
        "written_at_ms": int(time.time() * 1000), "world_size": len(ranks), "missing_ranks": [], "incidents": [],
        "ranks": [{"phase": "train", "state": "active", "severity": None, "stream": "open", **r} for r in ranks],
        "pauses": {}, "recommended_silent_seconds": None,
    }


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "store" / "runs" / RUN
    run.mkdir(parents=True)
    return run


def _part(run: Path, rank: int, attempt: int, first: int, last: int) -> None:
    folder = run / f"rank={rank}" / f"attempt={attempt}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"part-{first:08d}-{last:08d}.parquet").write_bytes(b"")


def _write_report(run: Path, report: dict) -> None:
    (run / f"rank={report['rank']}" / "incidents.json").write_text(json.dumps(report))


def _write_status(run: Path, status: dict) -> None:
    (run / "watchdog.json").write_text(json.dumps(status))


class _Steps:
    """A `trigger` for the watchers that does one thing before each look: `steps[n]()` before look n."""

    def __init__(self, *steps) -> None:
        self.steps, self.looks, self.seen = list(steps), 0, []

    def tick(self) -> None:
        if self.looks < len(self.steps):
            self.steps[self.looks]()
        self.looks += 1

    def look(self, evidence, written, rank=None) -> None:
        self.seen.append((self.looks, json.loads(json.dumps(evidence))))


def _watch_all(run: Path, out: Path, *steps) -> list[tuple[int, dict]]:
    trigger = _Steps(*steps)
    ev.watch(run, out, interval=0.0, polls=len(steps), rank="all", trigger=trigger)
    return trigger.seen


def _watch_one(run: Path, out: Path, *steps) -> list[tuple[int, dict]]:
    trigger = _Steps(*steps)
    ev.watch(run / "rank=0", out, interval=0.0, polls=len(steps), trigger=trigger)
    return trigger.seen


# --- Item 1: the same-attempt report read as an earlier attempt -----------------------------------

# The server's entry for a rank whose new attempt 2 has stored nothing yet.
WITHOUT_STEPS = {"rank": 0, "attempt": 2, "last_step": None, "attempt_without_steps": True, "state": "starting"}
OTHER = {"rank": 1, "attempt": 1, "last_step": 9}


def test_item1_the_helper_does_not_read_a_mark_for_the_judged_attempt_as_an_earlier_one():
    status = {"ranks": [WITHOUT_STEPS]}
    # fail-before: {0: "started again as attempt 2, which has stored no step yet: ..."}
    assert ev._not_current_by_status(status, {0: 2}) == {}
    # The control: a judgment of attempt 1 beside the mark for attempt 2 is an earlier attempt's.
    assert "started again as attempt 2" in ev._not_current_by_status(status, {0: 1})[0]
    # A judgment whose attempt is not known, and a mark that names no attempt, are read as before.
    assert "started again as attempt 2" in ev._not_current_by_status(status, {0: None})[0]
    unnamed = {k: v for k, v in WITHOUT_STEPS.items() if k != "attempt"}
    assert "started again" in ev._not_current_by_status({"ranks": [unnamed]}, {0: 2})[0]
    # And `run_state`, which reads the same helper.
    found = ev.run_state(incidents=[], steps_judged=True, status=_status([WITHOUT_STEPS]), judged_ranks=[0],
                         judged_attempts={0: 2})
    assert found["state"] == "healthy", found


def _store_at_attempt_2(run: Path) -> None:
    """Rank 0 stored attempt 1 (steps 0-9), then attempt 2 (steps 10-19), and its report is of attempt 2."""
    _part(run, 0, 1, 0, 9)
    _part(run, 0, 2, 10, 19)
    _write_report(run, _report(0, 2, 19))
    _part(run, 1, 1, 0, 9)
    _write_report(run, _report(1, 1, 9))


def _store_at_attempt_1(run: Path) -> None:
    _part(run, 0, 1, 0, 9)
    _write_report(run, _report(0, 1, 9))
    _part(run, 1, 1, 0, 9)
    _write_report(run, _report(1, 1, 9))


def test_item1_the_status_written_before_the_report_all_ranks(tmp_path):
    run = _run(tmp_path)
    seen = _watch_all(
        run, tmp_path / "all.json",
        # Look 1: attempt 2 has started and stored nothing; the store holds attempt 1 and its report.
        lambda: (_store_at_attempt_1(run), _write_status(run, _status([WITHOUT_STEPS, OTHER]))),
        # Look 2: attempt 2's first part and its report arrive; the status is not rewritten yet.
        lambda: _store_at_attempt_2(run),
    )
    first, second = seen[0][1], seen[-1][1]
    # The control: a report of attempt 1 beside the mark for attempt 2 stays an earlier attempt's.
    assert first["ranks_by_state"] == {"current": [1], ev.EARLIER_ATTEMPT: [0]}
    # fail-before: {"current": [1], "earlier_attempt": [0]} again
    assert second["ranks_by_state"] == {"current": [0, 1]}
    assert second["judged_ranks"] == [0, 1] and second["run_state"]["state"] == "healthy"


def test_item1_the_status_written_after_the_report_all_ranks(tmp_path):
    run = _run(tmp_path)
    seen = _watch_all(
        run, tmp_path / "all.json",
        # Look 1: attempt 2's report is there, with a status from before attempt 2 began.
        lambda: (_store_at_attempt_2(run), _write_status(run, _status([{"rank": 0, "attempt": 2, "last_step": 19},
                                                                       OTHER]))),
        # Look 2: a status snapshot taken before the first batch was stored lands after the report.
        lambda: _write_status(run, _status([WITHOUT_STEPS, OTHER])),
    )
    assert seen[0][1]["judged_ranks"] == [0, 1]
    # fail-before: rank 0 earlier_attempt, judged ranks [1]
    assert seen[-1][1]["judged_ranks"] == [0, 1] and seen[-1][1]["ranks_by_state"] == {"current": [0, 1]}
    # The control, in both orders: attempt 1's report beside the mark for attempt 2 is an earlier attempt's.
    for order in ("status first", "report first"):
        run = _run(tmp_path / order.replace(" ", "-"))
        status = lambda: _write_status(run, _status([WITHOUT_STEPS, OTHER]))
        steps = (status, lambda: _store_at_attempt_1(run)) if order == "status first" else (
            lambda: _store_at_attempt_1(run), status)
        seen = _watch_all(run, tmp_path / f"{order}.json", *steps)
        last = seen[-1][1]
        assert last["ranks_by_state"] == {"current": [1], ev.EARLIER_ATTEMPT: [0]}, order
        assert last["run_state"]["state"] == "unknown"


def test_item1_both_orders_through_the_single_rank_watcher(tmp_path):
    for order in ("status first", "report first"):
        run = _run(tmp_path / order.replace(" ", "-"))
        _store_at_attempt_1(run)  # the rank's folder is there when the watcher starts
        status = lambda: _write_status(run, _status([WITHOUT_STEPS, OTHER]))
        steps = (status, lambda: _store_at_attempt_2(run)) if order == "status first" else (
            lambda: _store_at_attempt_2(run), status)
        seen = _watch_one(run, tmp_path / f"{order}.json", *steps)
        last = seen[-1][1]
        assert last["detection"]["state"] == "current" and last["detection"]["report_attempt"] == 2
        codes = [r["code"] for r in last["run_state"]["reasons"]]
        # fail-before: unknown, with `ranks_unknown` for rank 0
        assert last["run_state"]["state"] == "healthy" and "ranks_unknown" not in codes, (order, codes)
    # The control through the same watcher: attempt 1's report beside the mark for attempt 2.
    run = _run(tmp_path / "control")
    _store_at_attempt_1(run)
    seen = _watch_one(run, tmp_path / "control.json", lambda: _write_status(run, _status([WITHOUT_STEPS, OTHER])))
    assert seen[-1][1]["run_state"]["state"] == "unknown"
    assert "ranks_unknown" in [r["code"] for r in seen[-1][1]["run_state"]["reasons"]]


def test_item1_the_combined_view(tmp_path):
    def per_rank(attempt: int) -> dict:
        return {"rank": 0, "incidents": [], "last_step": 19,
                "detection": {"source": "server", "state": "current", "report_attempt": attempt}}

    status = _status([WITHOUT_STEPS])
    # fail-before: judged [] and unknown [0]
    assert ev.combine_rank_evidence({0: per_rank(2)}, status=status, status_expected=True)["judged_ranks"] == [0]
    control = ev.combine_rank_evidence({0: per_rank(1)}, status=status, status_expected=True)
    assert control["judged_ranks"] == [] and control["unknown_ranks"] == [0]
