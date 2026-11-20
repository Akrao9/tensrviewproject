"""Lane X3 (wave X): the evidence core's open parts, each held by a test that failed before its fix.

- L6, open part: in the all-ranks evidence a rank the status marks `attempt_without_steps` was
  still counted among the judged ranks; only `run_state` called it unknown.
- DET-14, open part: at a run's first step the expert-load table compared the step with itself.
- Thresholds: `all_ranks_evidence` took any threshold a caller gave without checking it.
- C03, the Python side: a rank whose capture stopped and that then reads `unresponsive` or `lost`
  raised `job_stalled` or `rank_silent`.
- C22, the evidence half: the probe's count of head positions whose sums overflow was not in the
  evidence.

None needs `pyarrow` or the server binary.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

import tensorquill
from tensorquill import evidence as ev

# --- L6: a rank started again with nothing stored is not among the judged ranks ------------------

RUN = "x3run"


def _report(rank: int, attempt: int, last: int, incidents: list[dict] | None = None) -> dict:
    return {
        "schema_version": 1, "source": "tensorquill-server", "run_id": RUN, "rank": rank, "attempt": attempt,
        "last_step": last, "steps": last + 1, "incidents": incidents or [], "checked": list(ev.SERVER_KINDS),
        "thresholds": {"spike_sustain": ev.SPIKE_SUSTAIN},
    }


def _status(ranks: list[dict], summary: str = "active") -> dict:
    return {
        "schema_version": 1, "run_id": RUN, "summary": summary, "server_running": True, "refresh_seconds": 30.0,
        "written_at_ms": int(time.time() * 1000), "world_size": len(ranks), "missing_ranks": [],
        "ranks": [{"phase": "train", "state": "active", "severity": None, "stream": "open", **r} for r in ranks],
        "pauses": {}, "recommended_silent_seconds": None,
    }


def _store(tmp_path: Path, entries: list[dict], incidents: dict[int, list[dict]] | None = None) -> Path:
    """A run folder with one attempt-1 part and a current report per rank, and `entries` as the status."""
    run = tmp_path / "store" / "runs" / RUN
    for entry in entries:
        folder = run / f"rank={entry['rank']}" / "attempt=1"
        folder.mkdir(parents=True)
        (folder / "part-00000000-00000009.parquet").write_bytes(b"")
        report = _report(entry["rank"], 1, 9, (incidents or {}).get(entry["rank"]))
        (folder.parent / "incidents.json").write_text(json.dumps(report))
    status = _status(entries)
    status["incidents"] = []
    (run / "watchdog.json").write_text(json.dumps(status))
    return run


STARTED_AGAIN = {"attempt": 2, "last_step": None, "attempt_without_steps": True}
NAN = {"kind": "non_finite", "step": 5, "layers": ["a"], "grads": []}


def test_l6_a_rank_started_again_is_not_judged_in_the_all_ranks_evidence(tmp_path):
    run = _store(tmp_path, [{"rank": 0, "attempt": 1, "last_step": 9}, {"rank": 1, **STARTED_AGAIN}],
                 incidents={1: [NAN]})
    evidence = ev.all_ranks_evidence(run)
    assert evidence["judged_ranks"] == [0]
    assert evidence["unknown_ranks"] == [1]
    assert evidence["clean_ranks"] == [0] and evidence["ranks_with_incidents"] == []
    assert evidence["incident_groups"] == []  # rank 1's earlier attempt is not counted for the run
    assert evidence["ranks_by_state"] == {"current": [0], ev.EARLIER_ATTEMPT: [1]}
    entry = next(e for e in evidence["ranks"] if e["rank"] == 1)
    assert entry["detection"]["judged"] is False and entry["detection"]["state"] == ev.EARLIER_ATTEMPT
    assert "started again as attempt 2" in entry["detection"]["reason"]
    # Its report is shown as what it was, never as the rank's incidents.
    assert "incidents" not in entry and [i["kind"] for i in entry["last_report"]["incidents"]] == ["non_finite"]
    # The run state names it once, with the status's reason, not "current: ...".
    unknown = [r for r in evidence["run_state"]["reasons"] if r["code"] == "ranks_unknown"]
    assert evidence["run_state"]["state"] == "unknown" and unknown[0]["ranks"] == [1]
    # The summary lists it among the unknown ranks, with why.
    summary = ev._all_ranks_summary(evidence)
    assert "1 of 2 ranks are judged" in summary and f"rank 1 ({ev.EARLIER_ATTEMPT}): started again" in summary


def test_l6_the_watcher_writes_the_same(tmp_path):
    run = _store(tmp_path, [{"rank": 0, "attempt": 1, "last_step": 9}, {"rank": 1, **STARTED_AGAIN}])
    out = tmp_path / "all.json"
    ev.watch(run, out, interval=0.0, polls=1, rank="all")
    written = json.loads(out.read_text())
    assert written["judged_ranks"] == [0] and written["unknown_ranks"] == [1]
    # The fixture's reports do not say what was checked: that is said of rank 0 alone.
    assert written["kind_checks_not_stated"]["ranks"] == [0]


def test_l6_a_rank_at_the_attempt_the_status_shows_stays_judged(tmp_path):
    run = _store(tmp_path, [{"rank": 0, "attempt": 1, "last_step": 9}, {"rank": 1, "attempt": 1, "last_step": 9}])
    evidence = ev.all_ranks_evidence(run)
    assert evidence["judged_ranks"] == [0, 1] and evidence["unknown_ranks"] == []
    assert evidence["run_state"]["state"] == "healthy"


def test_l6_a_status_at_another_attempt_with_steps_is_not_judged_either(tmp_path):
    # The same rule as `run_state`: a report of another attempt than the status shows.
    run = _store(tmp_path, [{"rank": 0, "attempt": 1, "last_step": 9}, {"rank": 1, "attempt": 3, "last_step": 4}])
    evidence = ev.all_ranks_evidence(run)
    assert evidence["judged_ranks"] == [0] and evidence["unknown_ranks"] == [1]


def _per_rank(rank: int, attempt: int, incidents: list[dict] | None = None) -> dict:
    return {"rank": rank, "incidents": incidents or [], "last_step": 9,
            "detection": {"source": "server", "state": "current", "report_attempt": attempt}}


def test_l6_the_combined_view_moves_such_a_rank_to_the_unknown_ranks():
    status = _status([{"rank": 0, "attempt": 1, "last_step": 9}, {"rank": 1, **STARTED_AGAIN}])
    status["incidents"] = []
    combined = ev.combine_rank_evidence({0: _per_rank(0, 1), 1: _per_rank(1, 1, [NAN])}, status=status,
                                        status_expected=True)
    assert combined["judged_ranks"] == [0] and combined["unknown_ranks"] == [1]
    assert combined["incident_groups"] == [] and combined["rank_count"] == 2
    unknown = [r for r in combined["run_state"]["reasons"] if r["code"] == "ranks_unknown"]
    assert unknown and unknown[0]["ranks"] == [1]
    # Without a status nothing is compared.
    combined = ev.combine_rank_evidence({0: _per_rank(0, 1), 1: _per_rank(1, 1)})
    assert combined["judged_ranks"] == [0, 1]


# --- DET-14: the expert-load table at a run's first step ---------------------------------------


def _moe(steps: list[list[int]]) -> dict:
    return tensorquill.build_evidence([{"run_id": "t", "step": s, "moe": {"router": counts}}
                                       for s, counts in enumerate(steps)], detectors="python")


def test_det14_the_first_step_is_not_compared_with_itself():
    evidence = _moe([[10, 30, 60, 0], [25, 25, 25, 25]])
    lines = ev.expert_load_table(evidence, 0, 1).splitlines()
    assert lines[0] == "**Layer 01 expert load** · step 0 (no earlier step to compare with)"
    rows = lines[4:]
    assert len(rows) == 4 and all("| n/a |" in row for row in rows)  # no delta, never +0.00
    assert "+0.00" not in "\n".join(lines)
    # Sorted by load, largest first.
    assert [row.split(" | ")[0] for row in rows] == ["| #003", "| #002", "| #001", "| #004"]
    # The next step is compared with the first, as before.
    second = ev.expert_load_table(evidence, 1, 1).splitlines()
    assert second[0] == "**Layer 01 expert load** · step 1 vs. 0"
    assert "| #001 | 25.00% | +15.00 |" in "\n".join(second)


def test_det14_a_one_step_run_has_its_table_without_a_comparison():
    lines = ev.expert_load_table(_moe([[5, 5]]), 0, 1).splitlines()
    assert lines[0].endswith("(no earlier step to compare with)")
    assert all("| n/a |" in row for row in lines[4:])
