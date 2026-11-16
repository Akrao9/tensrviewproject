"""`watch(run_folder, rank="all")`: one evidence file for every rank of a run, through the real server.

A watcher of one rank reads that rank's `incidents.json` and the name of its newest part. Here
the same is done for every `rank=<n>` folder of a run, and the run's `watchdog.json`, into one
evidence file of its own shape (`"view": "all_ranks"`). What these tests hold it to:

- a rank whose report is not current (behind, stale, missing, unreadable, unusable, nothing
  stored, never started) is unknown, and is never counted among the ranks that have no incidents;
- identical incidents on several ranks are one group naming the ranks, and an incident on one
  rank is named with that rank alone;
- no part is read and no detector run here, in any state (every test patches `_read_part` to fail);
- the file is written only when something changed, compared with what was last written;
- ranks may appear and disappear between looks, and a server may be killed and restarted;
- one look of 64 ranks is cheap.

Needs `pyarrow` (the `flight` extra) and the server binary
(`cargo build --manifest-path server/Cargo.toml`, or `TENSORQUILL_SERVER_BIN`).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")

from tensorquill import evidence as ev  # noqa: E402
from tensorquill.evidence import incident_summary, read_server_report, watch  # noqa: E402
from tensorquill.flight import FlightSender, send_file  # noqa: E402

from test_flight_e2e import BINARY, Server  # noqa: E402
from test_server_detectors import FAST, RUN, everything, record, wobble  # noqa: E402
from test_server_reports import QUICK_STATUS, STATUS_OLD, no_part_is_read  # noqa: E402
from test_traincheck import _summary, _violation  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")


STARTED: list[Server] = []


@pytest.fixture(autouse=True)
def parts_are_never_read(monkeypatch):
    no_part_is_read(monkeypatch)


@pytest.fixture(autouse=True)
def servers_are_stopped():
    """Every server a test started is gone when it ends, whether it passed, failed or never stopped it:
    a `Popen` child outlives pytest, and a failing run would otherwise leave one behind."""
    yield
    for server in STARTED:
        if server.process.poll() is None:
            server.process.kill()
            server.process.wait(20)
        if not server.process.stdout.closed:
            server.process.stdout.close()
    STARTED.clear()


def steady(steps: int = 60, spike: tuple[int, float] | None = None, nan_at: tuple[int, str] | None = None,
           salt: int = 0, attempt: int = 1) -> list[dict]:
    """A healthy rank's steps. `spike=(step, factor)` multiplies the gradient norm there;
    `nan_at=(step, layer)` makes that layer's mean not a number there. Both are incidents.
    `salt` changes the logged loss, as a rank's own data does."""
    out = []
    for step in range(steps):
        norm = wobble(step) * (spike[1] if spike and step == spike[0] else 1.0)
        values = {"grad_norm": norm, "lr": [1e-3], "scalars": {"train/loss": wobble(step, 3 + salt)}}
        if nan_at and step == nan_at[0]:
            values["layers"] = {nan_at[1]: {"abs_max": 1.0, "mean": "nan", "std": 1.0, "zero_frac": 0.0}}
        out.append(record(step, attempt, **values))
    return out


class Run:
    """One server and several ranks of one run, each sending its own `.jsonl` file."""

    def __init__(self, tmp_path: Path, *server_args: str) -> None:
        self.tmp = tmp_path
        self.store = tmp_path / "store"
        self.server_args = server_args
        self.server = self.start(*server_args)

    def start(self, *server_args: str) -> Server:
        server = Server(self.store, 0, *server_args)
        STARTED.append(server)
        return server

    @property
    def folder(self) -> Path:
        return self.store / "runs" / RUN

    def rank_folder(self, rank: int) -> Path:
        return self.folder / f"rank={rank}"

    def report_file(self, rank: int) -> Path:
        return self.rank_folder(rank) / "incidents.json"

    def file(self, rank: int) -> Path:
        return self.tmp / f"{RUN}.rank{rank}.jsonl"

    def send(self, rank: int, records: list[dict], *, judged: bool = True) -> None:
        """Send records for `rank` and return once they are stored and, if the server is
        judging, reported on."""
        with self.file(rank).open("a", encoding="utf-8") as out:
            for item in records:
                out.write(json.dumps({**item, "rank": rank}) + "\n")
        assert send_file(self.file(rank), self.server.endpoint, RUN, rank, timeout=30.0)
        self.wait_for(rank, records[-1], judged)

    def wait_for(self, rank: int, last: dict, judged: bool = True) -> None:
        deadline = time.monotonic() + 10
        wanted = (last.get("attempt", 0), last["step"])
        while True:
            look = read_server_report(self.rank_folder(rank))
            if look["stored"] == wanted and (look["state"] == "current" or not judged):
                return
            assert time.monotonic() < deadline, f"the server did not store and report on rank {rank}'s rows"
            time.sleep(0.05)

    def keep_open(self, rank: int, records: list[dict], world_size: int | None = None) -> FlightSender:
        """Send as a job that is still running does: the stream stays open."""
        self.file(rank).write_text("".join(json.dumps({**item, "rank": rank}) + "\n" for item in records))
        sender = FlightSender(self.file(rank), self.server.endpoint, RUN, rank, world_size=world_size,
                              batch_seconds=0.05, poll_seconds=0.02)
        self.wait_for(rank, records[-1])
        return sender

    def restart(self, *server_args: str, kill: bool = False) -> None:
        if kill:
            self.server.kill()
        else:
            assert self.server.terminate() == 0
        self.server_args = server_args
        self.server = self.start(*server_args)


def healthy_run(tmp_path: Path, ranks: int = 4, *server_args: str, steps: int = 60) -> Run:
    run = Run(tmp_path, *(server_args or FAST))
    for rank in range(ranks):
        run.send(rank, steady(steps))
    return run


def look(run: Run, tmp_path: Path, **options) -> tuple[dict, str]:
    """One look by a watcher that has seen nothing before: its evidence and its summary."""
    tmp_path.mkdir(exist_ok=True)
    out = watch(run.folder, tmp_path / "all.json", interval=0.0, polls=1, rank="all", **options)
    assert out is not None, "the watcher wrote nothing"
    evidence = json.loads(out.read_text())
    return evidence, incident_summary(evidence)


def entry(evidence: dict, rank: int) -> dict:
    [found] = [e for e in evidence["ranks"] if e["rank"] == rank]
    return found


def assert_unknown(evidence: dict, summary: str, rank: int, state: str) -> None:
    """A rank that is not current is unknown: stated, and in no list that says "no incidents"."""
    detection = entry(evidence, rank)["detection"]
    assert detection["state"] == state and detection["judged"] is False and detection["reason"]
    assert rank in evidence["unknown_ranks"] and rank not in evidence["judged_ranks"]
    assert rank not in evidence["clean_ranks"] and rank not in evidence["ranks_with_incidents"]
    assert rank in evidence["ranks_by_state"][state]
    assert all(rank not in group["ranks"] for group in evidence["incident_groups"])
    assert "incidents" not in entry(evidence, rank)
    assert "No incidents detected on any of" not in summary and f"({state})" in summary
    assert "unknown" in summary.splitlines()[0]


def test_ranks_that_are_all_current_with_no_incidents_are_said_to_be_clean(tmp_path):
    run = healthy_run(tmp_path, 4)
    evidence, summary = look(run, tmp_path)
    assert evidence["view"] == "all_ranks" and evidence["view_version"] == 1
    assert "schema_version" not in evidence and "incidents" not in evidence and "samples" not in evidence
    assert evidence["rank_count"] == 4 and evidence["run_id"] == RUN
    assert evidence["judged_ranks"] == evidence["clean_ranks"] == [0, 1, 2, 3]
    assert evidence["unknown_ranks"] == evidence["ranks_with_incidents"] == evidence["never_reporting_ranks"] == []
    assert evidence["incident_groups"] == [] and evidence["run_incidents"] == []
    assert evidence["ranks_by_state"] == {"current": [0, 1, 2, 3]} and evidence["last_step"] == {"min": 59, "max": 59}
    for rank in range(4):
        detection = entry(evidence, rank)["detection"]
        assert (detection["state"], detection["judged"], detection["report_last_step"], detection["stored_last_step"]) == (
            "current", True, 59, 59)
        assert entry(evidence, rank)["incidents"] == []
    assert summary.splitlines() == [
        "No incidents detected on any of the 4 ranks: every rank's report from tensorquill-server is current through step 59."
    ]
    assert run.server.terminate() == 0


def test_identical_incidents_on_every_rank_are_one_group_that_names_the_ranks(tmp_path):
    run = Run(tmp_path, *FAST)
    for rank in range(4):
        run.send(rank, steady(60, nan_at=(30, "blk2"), salt=rank))
    evidence, summary = look(run, tmp_path)
    [group] = evidence["incident_groups"]
    assert (group["kind"], group["step"], group["ranks"], group["rank_ranges"]) == ("non_finite", 30, [0, 1, 2, 3], "0-3")
    assert group["on_all_judged_ranks"] is True and group["numbers_differ"] is False and group["incident"]["layers"] == ["blk2"]
    # The loss each rank logged at the step differs, and is not what makes incidents different.
    assert len({e["incident"]["scalars"]["train/loss"]["value"] for e in [group]}) == 1
    losses = {json.loads(run.report_file(r).read_text())["incidents"][0]["scalars"]["train/loss"]["value"] for r in range(4)}
    assert len(losses) > 1
    assert evidence["clean_ranks"] == [] and evidence["ranks_with_incidents"] == [0, 1, 2, 3]
    assert all(entry(evidence, r)["incidents"] == [{"kind": "non_finite", "class": "invariant", "step": 30, "group": 0}]
               for r in range(4))
    assert (group["class"], group["incident"]["class"]) == ("invariant", "invariant")
    assert summary.splitlines() == [
        "1 distinct incident on 4 of 4 ranks; every rank's report from tensorquill-server is current through step 59.",
        "Common to all 4 judged ranks (0-3):",
        "  - NaN or Inf at step 30 in blk2",
        "By class: invariant non_finite",
    ]
    assert run.server.terminate() == 0


def test_the_same_fault_with_different_numbers_is_still_one_group_and_says_whose_numbers(tmp_path):
    run = Run(tmp_path, *FAST)
    for rank, factor in enumerate((8.0, 9.0, 10.0, 10.0)):
        run.send(rank, steady(60, spike=(40, factor)))
    evidence, summary = look(run, tmp_path)
    [group] = evidence["incident_groups"]
    assert (group["kind"], group["step"], group["ranks"]) == ("grad_spike", 40, [0, 1, 2, 3])
    assert group["numbers_differ"] is True and group["incident_rank"] == 0
    assert round(group["incident"]["ratio"]) == 8
    lines = summary.splitlines()
    assert lines[2].startswith("  - Gradient spike at step 40: norm 8") and lines[2].endswith("(rank 0's numbers)")
    assert lines[3:] == ["By class: anomaly grad_spike"]
    assert run.server.terminate() == 0


def test_an_incident_on_one_rank_only_is_named_and_the_others_are_listed_as_having_none(tmp_path):
    run = Run(tmp_path, *FAST)
    for rank in range(4):
        run.send(rank, steady(60, spike=(40, 8.0) if rank == 2 else None))
    evidence, summary = look(run, tmp_path)
    [group] = evidence["incident_groups"]
    assert (group["ranks"], group["on_all_judged_ranks"]) == ([2], False)
    assert evidence["clean_ranks"] == [0, 1, 3] and evidence["ranks_with_incidents"] == [2]
    assert entry(evidence, 2)["incidents"] == [{"kind": "grad_spike", "class": "anomaly", "step": 40, "group": 0}]
    assert entry(evidence, 1)["incidents"] == []
    assert summary.splitlines() == [
        "1 distinct incident on 1 of 4 ranks; every rank's report from tensorquill-server is current through step 59.",
        "Only on some ranks:",
        "  - rank 2 only: Gradient spike at step 40: norm 8, 8× the median of recent steps",
        "No incidents on ranks 0-1, 3 (3 of 4 judged).",
        "By class: anomaly grad_spike",
    ]
    assert run.server.terminate() == 0


def test_incidents_that_differ_by_step_are_listed_apart_with_their_ranks(tmp_path):
    run = Run(tmp_path, *FAST)
    # Every rank has the fault at step 30; ranks 0-1 also spike at 40, rank 2 spikes at 45.
    for rank in range(4):
        spike = (40, 8.0) if rank < 2 else (45, 8.0) if rank == 2 else None
        records = steady(60, spike=spike, nan_at=(30, "blk2"))
        run.send(rank, records)
    evidence, summary = look(run, tmp_path)
    shown = [(g["kind"], g["step"], g["ranks"]) for g in evidence["incident_groups"]]
    assert shown == [("grad_spike", 40, [0, 1]), ("grad_spike", 45, [2]), ("non_finite", 30, [0, 1, 2, 3])]
    assert [g["id"] for g in evidence["incident_groups"]] == [0, 1, 2]
    assert evidence["clean_ranks"] == []  # rank 3 has the common incident
    assert "  - ranks 0-1 (2 of 4): Gradient spike at step 40" in summary and "  - rank 2 only: Gradient spike at step 45" in summary
    assert summary.index("Common to all 4") < summary.index("Only on some ranks")
    assert run.server.terminate() == 0


def test_the_same_kind_at_the_same_step_naming_different_things_is_two_groups(tmp_path):
    run = Run(tmp_path, *FAST)
    for rank in range(4):
        run.send(rank, steady(60, nan_at=(30, "blk2" if rank < 3 else "blk3")))
    evidence, summary = look(run, tmp_path)
    shown = [(g["kind"], g["step"], g["ranks"], g["incident"]["layers"]) for g in evidence["incident_groups"]]
    assert shown == [("non_finite", 30, [0, 1, 2], ["blk2"]), ("non_finite", 30, [3], ["blk3"])]
    assert summary.splitlines()[1:] == [
        "Only on some ranks:",
        "  - ranks 0-2 (3 of 4): NaN or Inf at step 30 in blk2",
        "  - rank 3 only: NaN or Inf at step 30 in blk3",
        "By class: invariant non_finite",
    ]
    assert run.server.terminate() == 0


# --- ranks that are not current are unknown ---------------------------------------------------


def test_a_report_that_is_missing_unreadable_or_not_this_rank_s_makes_the_rank_unknown(tmp_path):
    run = healthy_run(tmp_path, 4)
    assert run.server.terminate() == 0
    good = json.loads(run.report_file(3).read_text())

    run.report_file(1).unlink()
    run.report_file(2).write_text(json.dumps(good)[:200])
    run.report_file(3).write_text(json.dumps({**good, "rank": 0}))
    (run.folder / "rank=7").mkdir()  # a rank folder with nothing in it
    evidence, summary = look(run, tmp_path)
    for rank, state in ((1, "missing"), (2, "unreadable"), (3, "unusable"), (7, "nothing_stored")):
        assert_unknown(evidence, summary, rank, state)
    assert evidence["judged_ranks"] == evidence["clean_ranks"] == [0]
    assert evidence["never_reporting_ranks"] == [7] and evidence["rank_count"] == 5
    first = summary.splitlines()
    # Rank 0 is judged and has none; nothing is said about the others.
    assert first[0] == ("1 of 5 ranks are judged from tensorquill-server's current reports through step 59;"
                        " 4 are unknown, and nothing is said about incidents on them.")
    assert first[1] == "No incidents on the 1 judged rank (ranks 0)."
    assert "  - rank 1 (missing): the server wrote no incidents.json for this rank" in summary
    assert "  - rank 2 (unreadable): incidents.json is not complete JSON" in summary
    assert "  - rank 3 (unusable): incidents.json is for run 'detect' rank 0, not this folder" in summary
    assert "  - rank 7 (nothing_stored): nothing is stored for this rank yet" in summary
    # The hint to judge one by hand names the command.
    assert "python -m tensorquill <run folder> --rank <n>" in summary


def test_other_thresholds_make_every_rank_unknown_with_the_reason(tmp_path):
    run = healthy_run(tmp_path, 3)
    assert run.server.terminate() == 0
    evidence, summary = look(run, tmp_path, overload_threshold=0.5)
    assert evidence["judged_ranks"] == [] and evidence["unknown_ranks"] == [0, 1, 2]
    for rank in range(3):
        assert_unknown(evidence, summary, rank, "other_thresholds")
    assert summary.splitlines()[0] == "0 of 3 ranks are judged from tensorquill-server's current reports; 3 are unknown, and nothing is said about incidents on them."
    assert "  - ranks 0-2 (other_thresholds): the server applied overload_threshold 0.35 where 0.5 was asked for" in summary


def test_a_rank_behind_its_store_is_unknown_with_its_last_report_shown_apart(tmp_path):
    run = Run(tmp_path, *FAST)
    for rank in range(3):
        run.send(rank, steady(60))
    run.send(3, steady(50, spike=(40, 8.0)))
    before = run.report_file(3).read_bytes()
    run.send(3, steady(100, spike=(40, 8.0))[50:])  # rank 3's parts go on
    assert run.server.terminate() == 0
    run.report_file(3).write_bytes(before)  # what a kill right after the last part would leave
    evidence, summary = look(run, tmp_path)
    assert_unknown(evidence, summary, 3, "behind")
    detection = entry(evidence, 3)["detection"]
    assert (detection["report_last_step"], detection["stored_last_step"]) == (49, 99)
    # What it reported is shown, as a report that stops at step 49: not as the rank's incidents.
    last = entry(evidence, 3)["last_report"]
    assert (last["last_step"], [i["kind"] for i in last["incidents"]]) == (49, ["grad_spike"])
    assert evidence["incident_groups"] == [] and evidence["clean_ranks"] == [0, 1, 2]
    assert "Rank 3's last report, through step 49, held grad_spike" in summary
    assert "No incidents on the 3 judged ranks (ranks 0-2)." in summary


@pytest.mark.parametrize("grace, state, writes", [(3600.0, "current", 1), (0.0, "stale", 2)])
def test_a_rank_that_was_current_and_is_behind_later_waits_its_grace_and_then_is_stale(tmp_path, grace, state, writes):
    run = Run(tmp_path, *FAST)
    for rank in range(3):
        run.send(rank, steady(50))
    before = run.report_file(2).read_bytes()
    run.send(2, steady(100)[50:])
    assert run.server.terminate() == 0
    seen = []

    def on_update(evidence, path):
        seen.append(evidence)
        if len(seen) == 1:
            # The watcher has seen rank 2 current. Now its report is the one from before the last part.
            run.report_file(2).write_bytes(before)

    out = watch(run.folder, tmp_path / "all.json", interval=0.0, polls=4, rank="all", on_update=on_update,
                report_grace=grace)
    assert entry(seen[0], 2)["detection"]["state"] == "current"
    # Within the grace a rank between a part and its report is no news: nothing is written.
    # Past it, the report is stale, and the rank is unknown from then on, once.
    assert len(seen) == writes
    final = json.loads(out.read_text())
    assert entry(final, 2)["detection"]["state"] == state
    if state == "stale":
        assert_unknown(final, incident_summary(final), 2, "stale")
        assert entry(final, 2)["detection"]["reason"].startswith("the server's report stops at step 49 of attempt 1")
        assert final["clean_ranks"] == [0, 1]


def test_a_rank_that_goes_from_one_unknown_state_to_another_is_rewritten(tmp_path):
    run = healthy_run(tmp_path, 3)
    assert run.server.terminate() == 0
    good = run.report_file(2).read_text()
    seen = []

    def on_update(evidence, path):
        seen.append(entry(evidence, 2)["detection"]["state"])
        if len(seen) == 1:
            run.report_file(2).unlink()
        elif len(seen) == 2:
            run.report_file(2).write_text(good[:100])

    watch(run.folder, tmp_path / "all.json", interval=0.0, polls=5, rank="all", on_update=on_update, report_grace=0.0)
    assert seen == ["current", "missing", "unreadable"]


def test_ranks_that_become_expected_between_looks_are_written_though_nothing_else_changed(tmp_path):
    run = healthy_run(tmp_path, 2)
    assert run.server.terminate() == 0
    status_file = run.folder / "watchdog.json"
    wait_for_status(run, lambda s: [r["rank"] for r in s["ranks"]] == [0, 1])
    seen = []

    def on_update(evidence, path):
        seen.append(evidence)
        if len(seen) == 1:
            # Only the world size changes: the same stamp, the same reports, two more ranks expected.
            status = json.loads(status_file.read_text())
            status["world_size"] = 4
            status_file.write_text(json.dumps(status))

    watch(run.folder, tmp_path / "all.json", interval=0.0, polls=4, rank="all", on_update=on_update)
    assert [e["rank_count"] for e in seen] == [2, 4] and seen[1]["never_reporting_ranks"] == [2, 3]


def test_a_new_attempt_that_ends_where_the_last_one_did_is_a_change(tmp_path):
    run = healthy_run(tmp_path, 2)
    seen = []

    def on_update(evidence, path):
        seen.append(evidence)
        if len(seen) == 1:
            # Rank 1 starts over: attempt 2 runs the same 60 steps, so its last step and its step count are
            # what they were, and only the attempt (and the server's rebuild) say that the report moved on.
            run.send(1, steady(60, attempt=2))

    watch(run.folder, tmp_path / "all.json", interval=0.0, polls=4, rank="all", on_update=on_update)
    assert len(seen) == 2
    before, after = entry(seen[0], 1)["detection"], entry(seen[1], 1)["detection"]
    assert (before["report_attempt"], after["report_attempt"]) == (1, 2)
    assert before["report_last_step"] == after["report_last_step"] == 59
    assert run.server.terminate() == 0


def test_a_rank_never_seen_current_does_not_wait_and_is_not_stale(tmp_path):
    run = Run(tmp_path, *FAST)
    run.send(0, steady(50))
    before = run.report_file(0).read_bytes()
    run.send(0, steady(100)[50:])
    assert run.server.terminate() == 0
    run.report_file(0).write_bytes(before)
    # `behind` from the first look, whatever the grace: it was never current here, so nothing to wait for;
    # and `behind` it stays, since stale means behind for longer than the grace after being current.
    evidence, summary = look(run, tmp_path, report_grace=3600.0)
    assert_unknown(evidence, summary, 0, "behind")


# --- a server killed and restarted -------------------------------------------------------------


def wait_for_status(run: Run, wanted) -> dict:
    deadline = time.monotonic() + 10
    while True:
        try:
            status = json.loads((run.folder / "watchdog.json").read_text())
            if wanted(status):
                return status
        except (FileNotFoundError, ValueError):
            pass
        assert time.monotonic() < deadline, "the server's status did not reach the wanted state"
        time.sleep(0.05)


def test_after_a_restart_without_detection_the_rank_that_sent_again_is_stale_and_the_others_stay_current(tmp_path):
    run = Run(tmp_path, *FAST, *QUICK_STATUS)
    senders = [run.keep_open(rank, steady(60), world_size=4) for rank in range(4)]
    seen = []

    def on_update(evidence, path):
        seen.append(evidence)
        if len(seen) == 1:
            # The server is killed and comes back with detection off. Only rank 1 reconnects, and what
            # it sends is stored without being judged.
            run.restart(*FAST, "--no-detect", *QUICK_STATUS, kill=True)
            for sender in senders:
                sender.close(0.0)
            run.send(1, steady(120)[60:], judged=False)

    out = watch(run.folder, tmp_path / "all.json", interval=0.0, polls=4, rank="all", on_update=on_update,
                report_grace=0.0)
    assert entry(seen[0], 1)["detection"]["state"] == "current" and seen[0]["clean_ranks"] == [0, 1, 2, 3]
    final = json.loads(out.read_text())
    summary = incident_summary(final)
    # The new server took the run up from the old one's status, so it knows the other ranks and the
    # world size although only rank 1 reconnected (the watchdog's state survives a restart).
    status = wait_for_status(run, lambda s: [r["rank"] for r in s["ranks"]] == [0, 1, 2, 3])
    assert status["world_size"] == 4 and status["missing_ranks"] == []
    # Ranks 0, 2 and 3 have stored parts and a current report: they are current, whatever the status
    # does not know about them, and they are not "never reporting".
    assert final["ranks_by_state"] == {"current": [0, 2, 3], "stale": [1]}
    assert final["never_reporting_ranks"] == [] and final["clean_ranks"] == [0, 2, 3]
    assert_unknown(final, summary, 1, "stale")
    assert entry(final, 1)["detection"]["reason"] == (
        "the server's report stops at step 59 of attempt 1 and the store holds step 119 of attempt 1")
    assert "No incidents on the 3 judged ranks (ranks 0, 2-3)." in summary
    assert "  - rank 1 (stale): the server's report stops at step 59" in summary

    # The next server judges: the rank sends again, the server rebuilds it from the store, and
    # every rank is current once more.
    run.restart(*FAST, *QUICK_STATUS)
    run.send(1, steady(130)[120:])
    evidence, summary = look(run, tmp_path / "again")
    assert evidence["judged_ranks"] == evidence["clean_ranks"] == [0, 1, 2, 3] and evidence["unknown_ranks"] == []
    assert entry(evidence, 1)["detection"]["report_last_step"] == 129 and entry(evidence, 0)["detection"]["report_last_step"] == 59
    assert evidence["last_step"] == {"min": 59, "max": 129}
    assert summary.splitlines()[0] == (
        "No incidents detected on any of the 4 ranks: every rank's report from tensorquill-server is current"
        " through steps 59-129.")
    assert run.server.terminate() == 0


def test_after_a_kill_every_report_stands_and_the_status_is_called_old(tmp_path):
    run = Run(tmp_path, *FAST, *QUICK_STATUS)
    senders = [run.keep_open(rank, steady(60), world_size=3) for rank in range(3)]
    try:
        time.sleep(0.5)  # a status is written
        run.server.kill()
        time.sleep(STATUS_OLD)
        evidence, summary = look(run, tmp_path)
    finally:
        for sender in senders:
            sender.close(0.0)
    # Every part the server stored was judged before the kill: the reports are current. But the dead
    # server's status still says "active", and "no incidents" is not the whole story.
    assert evidence["judged_ranks"] == evidence["clean_ranks"] == [0, 1, 2]
    assert evidence["watchdog"]["not_written_for_seconds"] > 0
    lines = summary.splitlines()
    assert lines[0].startswith("No incidents detected on any of the 3 ranks")
    assert lines[1].startswith("The server's status was last written") and "Nothing is watching this run for stalls" in lines[1]


def test_a_stopped_server_qualifies_the_headline(tmp_path):
    run = Run(tmp_path, *FAST, *QUICK_STATUS)
    senders = [run.keep_open(rank, steady(60)) for rank in range(2)]
    try:
        assert run.server.terminate() == 0
        evidence, summary = look(run, tmp_path)
    finally:
        for sender in senders:
            sender.close(0.0)
    assert evidence["watchdog"]["summary"] == "server_stopped"
    assert summary.splitlines() == [
        "No incidents detected on any of the 2 ranks: every rank's report from tensorquill-server is current through step 59.",
        "The server has stopped: nothing is watching this run for stalls, and no new steps are stored.",
    ]


# --- ranks that appear and disappear -----------------------------------------------------------


def test_a_rank_that_appears_between_looks_joins_the_view_and_one_that_goes_is_unknown_or_gone(tmp_path):
    run = Run(tmp_path, *FAST, *QUICK_STATUS)
    for rank in (0, 1):
        run.send(rank, steady(60))
    wait_for_status(run, lambda s: [r["rank"] for r in s["ranks"]] == [0, 1])
    seen = []

    def on_update(evidence, path):
        seen.append(evidence)
        if len(seen) == 1:
            run.send(2, steady(60, spike=(40, 8.0)))  # a new rank starts
        elif len(seen) == 2:
            shutil.rmtree(run.rank_folder(1))  # a folder is gone

    watch(run.folder, tmp_path / "all.json", interval=0.0, polls=6, rank="all", on_update=on_update)
    assert [e["rank_count"] for e in seen] == [2, 3, 3]
    assert [e["judged_ranks"] for e in seen] == [[0, 1], [0, 1, 2], [0, 2]]
    assert [g["ranks"] for g in seen[1]["incident_groups"]] == [[2]]
    # The server's status still lists rank 1, so it stays in the view, unknown, and says why.
    gone = entry(seen[2], 1)["detection"]
    assert gone["state"] == "nothing_stored" and "no rank folder" in gone["reason"]
    assert seen[2]["never_reporting_ranks"] == [1] and 1 not in seen[2]["clean_ranks"]


def test_a_rank_the_status_does_not_know_leaves_the_view_when_its_folder_goes(tmp_path):
    run = Run(tmp_path, *FAST, "--watchdog-seconds", "0")  # no status file at all
    for rank in (0, 1, 2):
        run.send(rank, steady(60))
    assert not (run.folder / "watchdog.json").exists()
    seen = []

    def on_update(evidence, path):
        seen.append(evidence)
        if len(seen) == 1:
            shutil.rmtree(run.rank_folder(2))

    watch(run.folder, tmp_path / "all.json", interval=0.0, polls=3, rank="all", on_update=on_update)
    assert [e["rank_count"] for e in seen] == [3, 2]
    assert seen[1]["judged_ranks"] == [0, 1] and "watchdog" not in seen[1]


def test_ranks_the_job_expects_but_the_store_does_not_have_are_never_reporting_not_healthy(tmp_path):
    run = Run(tmp_path, *FAST, *QUICK_STATUS)
    senders = [run.keep_open(rank, steady(60), world_size=6) for rank in range(4)]
    try:
        wait_for_status(run, lambda s: s["world_size"] == 6)
        evidence, summary = look(run, tmp_path)
    finally:
        for sender in senders:
            sender.close(0.0)
    assert evidence["judged_ranks"] == evidence["clean_ranks"] == [0, 1, 2, 3]
    assert evidence["never_reporting_ranks"] == [4, 5] and evidence["rank_count"] == 6 and evidence["world_size"] == 6
    for rank in (4, 5):
        assert_unknown(evidence, summary, rank, "nothing_stored")
        assert entry(evidence, rank)["detection"]["reason"] == "the job's world_size is 6 and nothing is stored for this rank"
    assert "  - ranks 4-5 (nothing_stored): the job's world_size is 6" in summary
    assert summary.splitlines()[0].startswith("4 of 6 ranks are judged")


# --- writes only on a change -----------------------------------------------------------------


def count_writes(run: Run, tmp_path: Path, polls: int, between=None, **options) -> list[dict]:
    """The evidence of each write over `polls` looks. `between(n_writes)` runs after a write."""
    seen: list[dict] = []

    def on_update(evidence, path):
        seen.append(evidence)
        if between is not None:
            between(len(seen))

    watch(run.folder, tmp_path / "counted.json", interval=0.0, polls=polls, rank="all", on_update=on_update, **options)
    return seen


def test_nothing_is_written_again_until_something_changes_and_then_once(tmp_path):
    run = healthy_run(tmp_path, 3)

    def more(n):
        if n == 1:
            run.send(1, steady(110)[60:])

    seen = count_writes(run, tmp_path, 6, between=more)
    # Six looks, one change (rank 1 sent 50 more steps): the first look and that change.
    assert len(seen) == 2
    assert entry(seen[0], 1)["detection"]["report_last_step"] == 59 and entry(seen[1], 1)["detection"]["report_last_step"] == 109
    assert entry(seen[1], 0)["detection"]["report_last_step"] == 59
    assert run.server.terminate() == 0


def test_a_rank_between_a_part_and_its_report_is_no_news_but_is_not_left_stuck_once_written(tmp_path, monkeypatch):
    run = healthy_run(tmp_path, 3)
    real = ev.read_server_report
    calls: dict[str, int] = {}
    flicker = {"rank": 1, "calls": {2}}

    def wrapped(folder, **options):
        result = real(folder, **options)
        name = Path(folder).name
        calls[name] = calls.get(name, 0) + 1
        if name == f"rank={flicker['rank']}" and calls[name] in flicker["calls"] and result["state"] == "current":
            return {**result, "state": "behind", "reason": "a part has been stored and its report has not followed yet"}
        return result

    monkeypatch.setattr(ev, "read_server_report", wrapped)
    # Rank 1 is behind on its 2nd and 3rd looks, within its grace: nothing is written for that, and
    # nothing when it is current again with the same report.
    flicker["calls"] = {2, 3}
    assert len(count_writes(run, tmp_path / "a", 6)) == 1

    # But if something else is written while it is behind, the file says so, and it must be rewritten
    # when the rank comes back with the report it had: the comparison is with what was written.
    calls.clear()
    flicker["calls"] = {2}

    def more(n):
        if n == 1:
            run.send(2, steady(110)[60:])  # a change on another rank, found on the next look

    seen = count_writes(run, tmp_path / "b", 5, between=more)
    assert len(seen) == 3
    assert entry(seen[1], 1)["detection"]["state"] == "behind" and seen[1]["judged_ranks"] == [0, 2]
    assert entry(seen[2], 1)["detection"]["state"] == "current" and seen[2]["judged_ranks"] == [0, 1, 2]
    assert run.server.terminate() == 0


def test_a_change_of_status_alone_is_written(tmp_path):
    run = Run(tmp_path, *FAST, *QUICK_STATUS, "--silent-seconds", "1", "--silent-factor", "0")
    senders = [run.keep_open(rank, steady(60)) for rank in range(2)]
    try:
        seen = []
        watch(run.folder, tmp_path / "all.json", interval=0.25, polls=14, rank="all",
              on_update=lambda evidence, path: seen.append([i["kind"] for i in evidence["run_incidents"]]))
    finally:
        for sender in senders:
            sender.close(0.0)
    # Nothing arrives, no report changes; the status says the job stalled, and that is written once.
    assert seen[0] == [] and seen[-1] == ["job_stalled"] and seen.count(["job_stalled"]) == 1


def test_a_stall_is_one_run_level_incident_and_the_ranks_are_still_judged(tmp_path):
    run = Run(tmp_path, *FAST, *QUICK_STATUS, "--silent-seconds", "1", "--silent-factor", "0")
    senders = [run.keep_open(rank, steady(60)) for rank in range(3)]
    try:
        wait_for_status(run, lambda s: s["summary"] == "stalled")
        evidence, summary = look(run, tmp_path)
    finally:
        for sender in senders:
            sender.close(0.0)
    [stall] = evidence["run_incidents"]
    assert stall["kind"] == "job_stalled" and stall["step"] == 59 and stall["assessment"] == "suspected_hang"
    assert evidence["incident_groups"] == [] and evidence["judged_ranks"] == evidence["clean_ranks"] == [0, 1, 2]
    assert entry(evidence, 1)["stream"]["state"] == "silent"
    lines = summary.splitlines()
    assert lines[0] == ("No incidents in the ranks' step records on any of the 3 ranks: every rank's report from"
                        " tensorquill-server is current through step 59.")
    assert lines[1].startswith("Job stalled after step 59: none of 3 ranks has sent a new step") and len(lines) == 3
    assert lines[2] == "By class: observation job_stalled (observations claim no failure and do not make the run unhealthy)"
    assert (stall["class"], evidence["run_state"]["state"]) == ("observation", "unknown")


def test_traincheck_is_read_once_for_the_run_and_a_bad_path_is_listed_not_checked(tmp_path):
    run = healthy_run(tmp_path, 3)
    path = tmp_path / "violations_summary.json"
    path.write_text(json.dumps(_summary([_violation("zero_grad() resets .grad", 3, 5, "APIContainRelation")], first=3)))
    evidence, summary = look(run, tmp_path, traincheck=path)
    [hit] = evidence["run_incidents"]
    assert hit["kind"] == "traincheck_violation" and evidence["traincheck"]["distinct_invariants_violated"] == 1
    assert evidence["incident_groups"] == [] and "TrainCheck reports 1 violated invariant" in summary
    evidence, summary = look(run, tmp_path / "bad", traincheck=tmp_path / "missing.json")
    assert evidence["run_incidents"] == [] and "Not checked (traincheck)" in summary
    assert run.server.terminate() == 0


# --- what it will not do, and the command line -------------------------------------------------


def tensorquill(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"}
    return subprocess.run([sys.executable, "-m", "tensorquill", *args], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=120)


def test_what_all_ranks_cannot_do_is_refused_with_a_message_that_says_what_to_do(tmp_path):
    run = healthy_run(tmp_path, 2)
    assert run.server.terminate() == 0
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    folder = str(run.folder)
    rerun = tensorquill(folder, "--rank", "all", "--rerun", str(cwd / "x.rrd"), cwd=cwd)
    assert rerun.returncode == 2 and "--rank all cannot be combined with --rerun" in rerun.stderr and "use --rank N" in rerun.stderr
    assert not (cwd / "x.rrd").exists()
    assert tensorquill(folder, "--rank", "all", "--watch", "--rerun", "x.rrd", cwd=cwd).returncode == 2
    again = tensorquill(folder, "--rank", "all", "--recompute", cwd=cwd)
    assert again.returncode == 2 and "--rank all cannot be combined with --recompute" in again.stderr
    assert "--rank N --watch --recompute" in again.stderr
    top = tensorquill(folder, "--rank", "all", "--top-k", "2", cwd=cwd)
    assert top.returncode == 2 and "--top-k" in top.stderr
    one = tensorquill(str(run.rank_folder(0)), "--rank", "all", cwd=cwd)
    assert one.returncode == 2 and "is one rank's folder" in one.stderr
    jsonl = tensorquill(str(run.file(0)), "--rank", "all", cwd=cwd)
    assert jsonl.returncode == 2 and "needs a run folder" in jsonl.stderr
    bad = tensorquill(folder, "--rank", "everything", cwd=cwd)
    assert bad.returncode == 2 and "neither a rank number nor 'all'" in bad.stderr
    assert list(cwd.iterdir()) == []  # nothing was written by any of them

    with pytest.raises(ValueError, match="cannot recompute"):
        watch(run.folder, tmp_path / "x.json", rank="all", polls=1, server_reports=False)
    with pytest.raises(ValueError, match="top_k"):
        watch(run.folder, tmp_path / "x.json", rank="all", polls=1, top_k=2)
    with pytest.raises(ValueError, match="rank must be"):
        watch(run.folder, tmp_path / "x.json", rank="al", polls=1)
    assert not (tmp_path / "x.json").exists()


def test_the_command_line_writes_one_look_and_keeps_watching_with_watch(tmp_path):
    run = Run(tmp_path, *FAST)
    for rank in range(3):
        run.send(rank, steady(60, spike=(40, 8.0) if rank == 1 else None))
    assert run.server.terminate() == 0
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    done = tensorquill(str(run.folder), "--rank", "all", cwd=cwd)
    assert done.returncode == 0, done.stderr
    # The default name keeps it apart from a rank's <run_id>.evidence.json.
    written = cwd / f"{RUN}.all-ranks.evidence.json"
    assert written.exists() and [p.name for p in cwd.iterdir()] == [written.name]
    evidence = json.loads(written.read_text())
    assert evidence["view"] == "all_ranks" and [g["ranks"] for g in evidence["incident_groups"]] == [[1]]
    assert "rank 1 only: Gradient spike at step 40" in done.stdout and f"wrote evidence -> {written.name}" in done.stdout

    out = cwd / "w.json"
    # A process started from a background shell job inherits SIGINT as ignored, and Python then never raises
    # KeyboardInterrupt in it: the child gets the default back.
    proc = subprocess.Popen([sys.executable, "-m", "tensorquill", str(run.folder), "--rank", "all", "--watch",
                             "--interval", "0.2", "-o", str(out)], cwd=cwd, stdout=subprocess.PIPE, text=True,
                            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    try:
        deadline = time.monotonic() + 30
        while not out.exists():
            assert time.monotonic() < deadline and proc.poll() is None
            time.sleep(0.1)
        time.sleep(0.6)
        proc.send_signal(signal.SIGINT)
        stdout, _ = proc.communicate(timeout=30)
    finally:
        proc.kill()
    assert proc.returncode == 0 and "watching every rank of" in stdout and f"updated {out}" in stdout
    assert "rank 1 only: Gradient spike at step 40" in stdout

    empty = tmp_path / "empty"
    empty.mkdir()
    nothing = tensorquill(str(empty), "--rank", "all", cwd=cwd)
    assert nothing.returncode == 1 and "nothing to show" in nothing.stderr


def test_a_rank_number_and_the_default_still_give_the_per_rank_evidence_unchanged(tmp_path):
    run = healthy_run(tmp_path, 3)
    assert run.server.terminate() == 0
    for options in ({"rank": 1}, {"rank": None, "default_rank": 0}):
        out = watch(run.folder, tmp_path / "one.json", interval=0.0, polls=1, **options)
        evidence = json.loads(out.read_text())
        assert evidence["schema_version"] == 1 and "view" not in evidence and evidence["incidents"] == []
        assert evidence["detection"]["source"] == "server" and evidence["detection"]["rank"] == (options["rank"] or 0)
        assert incident_summary(evidence).splitlines()[0] == "No incidents detected."
    shown = tensorquill(str(run.folder), cwd=tmp_path)
    assert shown.returncode == 0 and (
        "several ranks here; reading rank 0 (choose another with --rank N, or every rank with --rank all)" in shown.stdout)


# --- 64 ranks: what a look costs, and what the summary does when every rank differs -------------


def old_newest_part(folder: Path):
    """`_newest_part` as it was written with pathlib, before a look of 64 ranks made its cost matter."""
    newest, newest_path, unknown = None, None, 0
    for part in folder.rglob("part-*.parquet"):
        match = ev._PART_NAME.search(part.name)
        key, _, attempt = part.parent.name.partition("=")
        if match is None or key != "attempt" or not attempt.lstrip("-").isdigit() or part.parent.parent != folder:
            unknown += 1
            continue
        at = (int(attempt), int(match.group(2)))
        if newest is None or at > newest:
            newest, newest_path = at, part
    return newest, newest_path, unknown


def test_the_scandir_walk_for_the_newest_part_finds_what_the_pathlib_walk_found(tmp_path):
    layouts = {
        "empty": [],
        "two attempts": ["attempt=1/part-00000000-00000049.parquet", "attempt=1/part-00000050-00000099.parquet",
                         "attempt=2/part-00000000-00000009.parquet"],
        "older attempt reaches further": ["attempt=1/part-00000000-00000199.parquet", "attempt=2/part-00000100-00000129.parquet"],
        "negative": ["attempt=-1/part--5--1.parquet", "attempt=0/part-00000000-00000003.parquet"],
        "odd names": ["attempt=1/part-00000000-00000049.parquet", "attempt=x/part-00000000-00000070.parquet",
                      "attempt=1/part-.parquet", "attempt=1/part-1-2.parquet.tmp", "attempt=1/other.parquet",
                      "part-00000000-00000999.parquet", "attempt=1/nested/part-00000000-00000500.parquet",
                      "attempt=3/part-abc-def.parquet", "attempt=/part-0-1.parquet", "attemptx=1/part-0-1.parquet"],
        "a directory named like a part": ["attempt=1/part-00000000-00000049.parquet/part-0-1.parquet",
                                          "attempt=2/part-00000000-00000009.parquet"],
        "only unknown": ["part-1-2.parquet", "x/part-1-2.parquet"],
        "nested attempt directories": ["attempt=1/part-00000000-00000049.parquet",
                                       "attempt=1/attempt=9/part-00000000-00000900.parquet"],
    }
    for name, paths in layouts.items():
        folder = tmp_path / name.replace(" ", "_") / "rank=0"
        folder.mkdir(parents=True)
        for path in paths:
            target = folder / path
            if target.name == "part-00000000-00000049.parquet" and name == "a directory named like a part":
                target.mkdir(parents=True)
                (target / "part-0-1.parquet").write_bytes(b"")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"")
        assert ev._newest_part(folder) == old_newest_part(folder), name
    assert ev._newest_part(tmp_path / "missing") == (None, None, 0) == old_newest_part(tmp_path / "missing")
    # A rank folder that is itself called attempt=5, with a part directly in it: still not a part of an attempt.
    odd = tmp_path / "odd" / "attempt=5"
    odd.mkdir(parents=True)
    (odd / "part-00000000-00000009.parquet").write_bytes(b"")
    assert ev._newest_part(odd) == old_newest_part(odd) == (None, None, 1)


def stored_template(tmp_path: Path) -> dict:
    """A report as the server writes it for a rank that has everything the detectors find (10 incidents)."""
    run = Run(tmp_path / "template", *FAST)
    run.tmp.mkdir(parents=True, exist_ok=True)
    run.send(0, everything())
    template = json.loads(run.report_file(0).read_text())
    assert run.server.terminate() == 0
    assert len(template["incidents"]) == 10
    return template


def fabricated_run(root: Path, ranks: int, steps: int, template: dict, incidents_for) -> Path:
    """A run folder of `ranks` ranks of `steps` steps in the server's layout: part names (50 steps each,
    empty files: no part is ever opened), a report for each rank that covers them, and a status."""
    run = root / "store" / "runs" / RUN
    last = steps - 1
    for rank in range(ranks):
        attempt = run / f"rank={rank}" / "attempt=1"
        attempt.mkdir(parents=True)
        for first in range(0, steps, 50):
            (attempt / f"part-{first:08}-{min(first + 49, last):08}.parquet").write_bytes(b"")
        report = {**template, "rank": rank, "run_id": RUN, "attempt": 1, "last_step": last, "steps": steps,
                  "incidents": incidents_for(rank)}
        (run / f"rank={rank}" / "incidents.json").write_text(json.dumps(report))
    status = {
        "schema_version": 1, "run_id": RUN, "summary": "active", "server_running": True, "refresh_seconds": 30.0,
        "written_at_ms": int(time.time() * 1000), "world_size": ranks, "missing_ranks": [], "incidents": [],
        "ranks": [{"rank": r, "attempt": 1, "last_step": last, "phase": "train", "phase_seconds": 0.2,
                   "heartbeat_age_seconds": None, "seconds_since_progress": 0.2, "severity": None, "state": "active",
                   "stream": "open"} for r in range(ranks)],
        "pauses": {}, "recommended_silent_seconds": None, "silent_factor": 5.0, "silent_floor_seconds": 300.0,
        "critical_seconds": 900.0,
    }
    (run / "watchdog.json").write_text(json.dumps(status))
    return run


def millis(function) -> float:
    start = time.perf_counter()
    function()
    return (time.perf_counter() - start) * 1000


def test_one_look_at_64_ranks_of_10000_steps_is_cheap_and_groups_what_is_common(tmp_path):
    template = stored_template(tmp_path)
    run = fabricated_run(tmp_path / "f", 64, 10_000, template, lambda rank: template["incidents"])
    out = tmp_path / "all.json"
    # Each is a look by a watcher that has seen nothing: 64 rank folders of 200 part names, 64 reports, a
    # status, the grouping, and the write of the evidence by rename. 250 ms is the budget; the bound is
    # wider so that a busy machine does not fail it, and still fails a look that reads parts, which cost
    # seconds, or one that lists parts with pathlib (the measured number is in the plan).
    times = [millis(lambda: watch(run, out, interval=0.0, polls=1, rank="all")) for _ in range(5)]
    assert statistics.median(times) < 1000, times
    evidence = json.loads(out.read_text())
    assert evidence["rank_count"] == 64 and evidence["judged_ranks"] == list(range(64))
    assert len(evidence["incident_groups"]) == 10 and all(g["on_all_judged_ranks"] for g in evidence["incident_groups"])
    assert all(g["rank_ranges"] == "0-63" for g in evidence["incident_groups"])
    summary = incident_summary(evidence)
    assert summary.splitlines()[:2] == [
        "10 distinct incidents on 64 of 64 ranks; every rank's report from tensorquill-server is current through step 9999.",
        "Common to all 64 judged ranks (0-63):",
    ]
    # Unchanged, a look writes nothing and costs the reading alone.
    seen = []
    several = millis(lambda: watch(run, out, interval=0.0, polls=6, rank="all", on_update=lambda e, p: seen.append(1)))
    assert len(seen) == 1 and several < 6000


def test_when_every_rank_has_a_different_incident_the_summary_is_still_a_few_lines(tmp_path):
    template = stored_template(tmp_path)
    run = fabricated_run(
        tmp_path / "f", 64, 2000, template,
        lambda rank: [{"kind": "non_finite", "step": 100 + rank, "layers": [f"blk{rank}"], "gradients": []}],
    )
    out = tmp_path / "all.json"
    watch(run, out, interval=0.0, polls=1, rank="all")
    evidence = json.loads(out.read_text())
    assert len(evidence["incident_groups"]) == 64 and [g["ranks"] for g in evidence["incident_groups"]][:2] == [[0], [1]]
    lines = incident_summary(evidence).splitlines()
    assert len(lines) == 1 + 1 + ev.SUMMARY_PARTIAL_LINES + 1 + 1 == 12
    assert lines[0] == "64 distinct incidents on 64 of 64 ranks; every rank's report from tensorquill-server is current through step 1999."
    assert lines[2] == "  - rank 0 only: NaN or Inf at step 100 in blk0" and lines[9] == "  - rank 7 only: NaN or Inf at step 107 in blk7"
    assert lines[10] == "  - and 56 more distinct incidents on other ranks (non_finite x56); see incident_groups in the evidence file"
    assert lines[11] == "By class: invariant non_finite"
    # Common and partial together: ten incidents on every rank and one extra on rank 5 only.
    run = fabricated_run(tmp_path / "g", 64, 2000, template, lambda rank: template["incidents"] + (
        [{"kind": "non_finite", "step": 9, "layers": ["x"], "gradients": []}] if rank == 5 else []))
    watch(run, out, interval=0.0, polls=1, rank="all")
    lines = incident_summary(json.loads(out.read_text())).splitlines()
    assert lines[0].startswith("11 distinct incidents on 64 of 64 ranks") and lines.count("Only on some ranks:") == 1
    assert "  - rank 5 only: NaN or Inf at step 9 in x" in lines


def test_64_ranks_from_a_real_server_are_judged_grouped_and_cheap_to_look_at(tmp_path):
    run = Run(tmp_path, *FAST, *QUICK_STATUS)
    runs = [steady(60, spike=(40, 8.0) if rank % 2 else None, nan_at=(30, "blk2")) for rank in range(64)]
    started = time.monotonic()
    senders = []
    try:
        for rank in range(64):
            senders.append(run.keep_open(rank, runs[rank], world_size=64))
        sent = time.monotonic() - started
        wait_for_status(run, lambda s: len(s["ranks"]) == 64)
        out = tmp_path / "all.json"
        times = [millis(lambda: watch(run.folder, out, interval=0.0, polls=1, rank="all")) for _ in range(5)]
        evidence, summary = look(run, tmp_path)
    finally:
        for sender in senders:
            sender.close(0.0)
    assert sent < 120 and statistics.median(times) < 1000, (sent, times)
    assert evidence["judged_ranks"] == list(range(64)) and evidence["never_reporting_ranks"] == []
    shown = {(g["kind"], g["step"]): g["ranks"] for g in evidence["incident_groups"]}
    assert shown == {("grad_spike", 40): list(range(1, 64, 2)), ("non_finite", 30): list(range(64))}
    assert evidence["clean_ranks"] == []
    assert "  - ranks 1, 3, 5, 7, 9, 11, 13, 15, and 24 more (32 of 64): Gradient spike at step 40" in summary
    assert evidence["world_size"] == 64 and len(evidence["ranks"]) == 64 and entry(evidence, 63)["stream"]["state"] in ("active", "finished")
    assert run.server.terminate() == 0
