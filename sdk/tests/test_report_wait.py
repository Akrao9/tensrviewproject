"""The watchers' wait for a server's report (`report_grace`), on stores written by hand.

A part is stored first and reported on a moment later, so a report that was current and is then
behind, missing or unreadable is waited for, `report_grace` seconds from its last current look.
What these tests hold the watchers to (the outside review of Nov 18, 2026: DET-04, DET-05):

- one rank (`watch(rank_folder)`): a write made inside the wait, for a change of the status, says
  the report is not current; once the same report is current again, the file says so;
- every rank (`watch(run_folder, rank="all")`): a rank inside its wait reads as its last current
  report when another rank or the status causes a write, as the single-rank watcher writes from
  its last current report: judged, not unknown for that moment.

Nothing here needs the server binary or `pyarrow`: a watcher reads only the names of the parts
when a report is current, and the reports and the status are written here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorquill import evidence as ev
from test_mcp_server import report

RUN = "r"


@pytest.fixture(autouse=True)
def parts_are_never_read(monkeypatch):
    def refuse(path):
        raise AssertionError(f"a watcher with a current report read {path}")

    monkeypatch.setattr(ev, "_read_part", refuse)


def write_report(folder: Path, rank: int, last_step: int) -> None:
    (folder / f"rank={rank}" / "incidents.json").write_text(json.dumps({**report(rank, [], last_step), "run_id": RUN}))


def write_status(folder: Path, states: list[str], step: int = 59) -> None:
    summary = "stalled" if "active" not in states else "degraded" if set(states) != {"active"} else "active"
    (folder / "watchdog.json").write_text(json.dumps({
        "schema_version": 1, "run_id": RUN, "written_at_ms": 1_000, "silent_floor_seconds": 300.0, "silent_factor": 5.0,
        "world_size": len(states), "missing_ranks": [], "summary": summary,
        "ranks": [{"rank": rank, "attempt": 1, "last_step": step, "seconds_since_progress": 0.5 if state == "active" else 400.0,
                   "stream": "open", "state": state, "phase": "train", "heartbeat_age_seconds": 3.0,
                   "severity": None if state == "active" else "warning"} for rank, state in enumerate(states)],
    }))


def hand_run(root: Path, ranks: int) -> Path:
    """A run folder as the server leaves it, written by hand: each rank's current report through step
    59, one part whose name says it holds steps 0 to 59, and the status of ranks that are stepping."""
    folder = root / "runs" / RUN
    for rank in range(ranks):
        parts = folder / f"rank={rank}" / "attempt=1"
        parts.mkdir(parents=True)
        (parts / "part-0-59.parquet").write_bytes(b"")
        write_report(folder, rank, 59)
        assert ev.read_server_report(folder / f"rank={rank}")["state"] == "current"
    write_status(folder, ["active"] * ranks)
    return folder


def reasons(evidence: dict) -> list[str]:
    return [item["code"] for item in evidence["run_state"]["reasons"]]


def test_one_ranks_file_written_inside_the_wait_is_rewritten_when_the_same_report_is_current_again(tmp_path, monkeypatch):
    folder = hand_run(tmp_path / "store", 1)
    rank = folder / "rank=0"
    good = (rank / "incidents.json").read_text()
    looks = 0
    real = ev.read_watchdog

    def between_looks(path):
        nonlocal looks
        looks += 1
        if looks == 2:  # the report is being replaced, and the status says the job went quiet
            (rank / "incidents.json").write_text(good[:50])
            write_status(folder, ["silent"])
        elif looks == 3:  # the same report as before is back; nothing else changed
            (rank / "incidents.json").write_text(good)
        return real(path)

    monkeypatch.setattr(ev, "read_watchdog", between_looks)
    written: list[dict] = []
    out = ev.watch(rank, tmp_path / "rank0.json", interval=0.0, polls=4, report_grace=3600.0,
                   on_update=lambda evidence, path: written.append(evidence))
    # Inside the wait the stall is written from the last current report, and the file says the report
    # is not current: the run's state is unknown for that.
    assert [e["detection"]["state"] for e in written[:2]] == ["current", "unreadable"]
    assert [i["kind"] for i in written[1]["incidents"]] == ["job_stalled"] and "report_not_current" in reasons(written[1])
    # Once the same report is current again, the file is rewritten to say so, once.
    assert [e["detection"]["state"] for e in written] == ["current", "unreadable", "current"]
    final = json.loads(out.read_text())
    assert final["detection"]["state"] == "current" and "report_not_current" not in reasons(final)
    assert [i["kind"] for i in final["incidents"]] == ["job_stalled"]


@pytest.mark.parametrize("grace", [3600.0, 0.0])
def test_a_rank_inside_its_wait_reads_as_its_last_current_report_when_another_rank_causes_a_write(tmp_path, monkeypatch, grace):
    folder = hand_run(tmp_path / "store", 3)
    real = ev.read_server_report
    looked: dict[str, int] = {}

    def flicker(path, **options):
        # Rank 1 is behind on its second look only: a part stored, its report not written yet.
        found = real(path, **options)
        name = Path(path).name
        looked[name] = looked.get(name, 0) + 1
        if name == "rank=1" and looked[name] == 2 and found["state"] == "current":
            return {**found, "state": "behind", "reason": "a part has been stored and its report has not followed yet"}
        return found

    monkeypatch.setattr(ev, "read_server_report", flicker)
    written: list[dict] = []

    def on_update(evidence, path):
        written.append(evidence)
        if len(written) == 1:  # rank 2 moves on, which the second look finds
            (folder / "rank=2" / "attempt=1" / "part-60-109.parquet").write_bytes(b"")
            write_report(folder, 2, 109)

    out = ev.watch(folder, tmp_path / "all.json", interval=0.0, polls=4, rank="all", report_grace=grace, on_update=on_update)

    def detection(evidence: dict, rank: int) -> dict:
        return next(e for e in evidence["ranks"] if e["rank"] == rank)["detection"]

    assert detection(written[1], 2)["report_last_step"] == 109
    if grace:
        # Inside its wait, rank 1 reads as its last current report at the write rank 2 caused: judged,
        # with that report, and the run's state is what it was. Back with the same report, it is no news.
        assert len(written) == 2
        assert written[1]["judged_ranks"] == [0, 1, 2] and written[1]["unknown_ranks"] == []
        assert (detection(written[1], 1)["state"], detection(written[1], 1)["report_last_step"]) == ("current", 59)
        assert written[1]["run_state"] == written[0]["run_state"]
    else:
        # With no wait, behind after being current is stale at once: unknown, then current again.
        assert len(written) == 3
        assert written[1]["judged_ranks"] == [0, 2] and detection(written[1], 1)["state"] == "stale"
        assert written[2]["judged_ranks"] == [0, 1, 2]
    final = json.loads(out.read_text())
    assert final["judged_ranks"] == [0, 1, 2] and detection(final, 1)["state"] == "current"
