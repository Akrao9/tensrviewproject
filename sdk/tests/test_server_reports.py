"""`watch` reading the report `tensorquill-server` keeps for a rank, through the real binary.

A watcher takes its incidents from `runs/<run_id>/rank=<rank>/incidents.json` and from the
status file, and reads no part, while that report is current: it covers exactly the newest
stored part. An empty list in a report that is missing, behind the store, from another
server or about other thresholds is not "no incidents", and then the watcher reads the
parts and runs the detectors itself. Each case here is produced by a real server: a healthy
run, `--no-detect`, a restart with detection turned off, a kill, a clean stop, and a resume.

Needs `pyarrow` (the `flight` extra) and the server binary
(`cargo build --manifest-path server/Cargo.toml`, or `TENSORQUILL_SERVER_BIN`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")

import tensorquill  # noqa: E402
from tensorquill import evidence as ev  # noqa: E402
from tensorquill.evidence import incident_summary, read_server_report, watch  # noqa: E402
from tensorquill.flight import FlightSender, send_file  # noqa: E402

from test_flight_e2e import BINARY, Server  # noqa: E402
from test_server_detectors import FAST, PORTED, RUN, Stream, differences, everything, record, reference, wobble  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")

QUICK_STATUS = ("--watchdog-seconds", "0.2")
# With that, an unfinished run's status is rewritten at least every 0.6 s, and a reader
# calls it old after twice that.
STATUS_OLD = 1.5


def healthy(steps, attempt: int = 1) -> list[dict]:
    return [record(step, attempt, grad_norm_total=wobble(step), lr=[1e-3], scalars={"train/loss": wobble(step, 3)})
            for step in steps]


def push(stream: Stream, records: list[dict]) -> None:
    """Send records and return once they are stored, whether or not the server reports on them."""
    with stream.file.open("a", encoding="utf-8") as out:
        for item in records:
            out.write(json.dumps(item) + "\n")
    stream.sent.extend(records)
    assert send_file(stream.file, stream.server.endpoint, stream.run, timeout=30.0)
    wanted = (records[-1].get("attempt", 0), records[-1]["step"])
    deadline = time.monotonic() + 10
    while ev._newest_part(stream.incidents_file.parent)[0] != wanted:
        assert time.monotonic() < deadline, "the server did not store the rows"
        time.sleep(0.05)


def look(stream: Stream, tmp_path: Path, **options) -> tuple[dict, str]:
    """One look by a watcher that has seen nothing before: its evidence and its summary."""
    tmp_path.mkdir(exist_ok=True)
    out = watch(stream.incidents_file.parent, tmp_path / "live.json", interval=0.0, polls=1, **options)
    assert out is not None, "the watcher wrote nothing"
    evidence = json.loads(out.read_text())
    return evidence, incident_summary(evidence)


def no_part_is_read(monkeypatch) -> None:
    def refuse(path):
        raise AssertionError(f"a watcher with a current report read {path}")

    monkeypatch.setattr(ev, "_read_part", refuse)


def test_a_current_report_with_no_incidents_is_said_to_be_current(tmp_path, monkeypatch):
    stream = Stream(tmp_path, *FAST, *QUICK_STATUS)
    stream.send(healthy(range(120)))
    no_part_is_read(monkeypatch)
    evidence, summary = look(stream, tmp_path)
    assert evidence["incidents"] == [] and evidence["samples"] == []
    detection = evidence["detection"]
    assert (detection["source"], detection["state"]) == ("server", "current")
    assert (detection["report_attempt"], detection["report_last_step"], detection["report_steps"]) == (1, 119, 120)
    assert (detection["stored_attempt"], detection["stored_last_step"]) == (1, 119)
    assert summary.splitlines() == [
        "No incidents detected.",
        "Incidents are from tensorquill-server's report for rank 0: the server's report is current through"
        " step 119 of attempt 1 (120 steps judged).",
    ]
    # The two sources the server never has are not listed as unchecked on every look.
    assert "not_checked" not in evidence
    # Nothing changes, so nothing is written again: one evidence file in four looks.
    writes = []
    watch(stream.incidents_file.parent, tmp_path / "again.json", interval=0.0, polls=4, on_update=lambda *a: writes.append(a))
    assert len(writes) == 1
    # The rank closed its stream, so the run is finished: a status nobody rewrites is not a warning.
    time.sleep(STATUS_OLD)
    evidence, summary = look(stream, tmp_path)
    assert evidence["watchdog"]["summary"] == "finished" and "last written" not in summary
    assert stream.server.terminate() == 0


def test_a_stall_shows_from_the_status_though_no_part_arrives(tmp_path, monkeypatch):
    stream = Stream(tmp_path, *FAST, *QUICK_STATUS, "--silent-seconds", "1", "--silent-factor", "0")
    sender = keep_open(stream, healthy(range(60)))
    no_part_is_read(monkeypatch)
    seen = []
    try:
        # The job sends nothing more. The report does not change; the status does.
        out = watch(stream.incidents_file.parent, tmp_path / "live.json", interval=0.25, polls=12,
                    on_update=lambda evidence, path: seen.append([i["kind"] for i in evidence["incidents"]]))
        status = json.loads((stream.store / "runs" / RUN / "watchdog.json").read_text())
        # The stream stays open and silent for longer than a status may go unwritten.
        time.sleep(STATUS_OLD)
        later = json.loads((stream.store / "runs" / RUN / "watchdog.json").read_text())
        still_stalled, still_summary = look(stream, tmp_path / "later")
    finally:
        sender.close(0.0)
    assert seen[0] == [] and seen[-1] == ["job_stalled"]
    evidence = json.loads(out.read_text())
    assert evidence["detection"]["source"] == "server" and evidence["detection"]["report_last_step"] == 59
    # The incident is the one the server put in its status, and reads as Python's does.
    [incident] = evidence["incidents"]
    assert status["summary"] == "stalled" and incident["step"] == 59 and incident["assessment"] == "suspected_hang"
    assert {k: v for k, v in incident.items() if k not in ("ranks", "written_at_ms", "severity")} == {
        k: v for k, v in ev.detect_watchdog_incidents(status)[0].items()
        if k not in ("ranks", "written_at_ms", "severity")}
    assert incident_summary(evidence).startswith("Job stalled after step 59")
    # The verdict has not changed for longer than a status may go unwritten, and the server
    # is alive: it kept rewriting the status, so the stall is not mistaken for a dead server.
    assert later["summary"] == "stalled" and later["written_at_ms"] > status["written_at_ms"]
    assert later["refresh_seconds"] == pytest.approx(0.6)
    assert [i["kind"] for i in still_stalled["incidents"]] == ["job_stalled"]
    assert "not_written_for_seconds" not in still_stalled["watchdog"] and "last written" not in still_summary
    assert stream.server.terminate() == 0


def test_the_incidents_of_a_current_report_are_python_s_and_the_summary_shows_them(tmp_path, monkeypatch):
    records = everything()
    stream = Stream(tmp_path, *FAST, *QUICK_STATUS)
    stream.send(records)
    monkeypatch.setattr(ev, "_read_part", lambda path: pytest.fail("a part was read"))
    evidence, summary = look(stream, tmp_path)
    assert evidence["detection"]["source"] == "server"
    assert differences(reference(records), evidence["incidents"]) == []
    assert {i["kind"] for i in evidence["incidents"]} == PORTED - {"repeated_steps", "resume_lr_change"}
    # The MoE incident has no per-step samples behind it here, and is still described.
    assert "First sampled bad step 60  ·  Layer 01" in summary and "Overloaded: Expert 9, Expert 10" in summary
    assert "No incidents detected" not in summary
    python_lines = incident_summary(tensorquill.build_evidence(records)).splitlines()
    ours = summary.splitlines()
    # Every other incident reads as it does when Python worked it out, but for a NaN, which
    # is null in the server's strict JSON and cannot be told from an Inf.
    expected = [line.replace("sometimes_nan nan", "sometimes_nan not finite") for line in python_lines[2:]]
    assert [line for line in expected if not line.startswith("Heatmap")] == ours[2:-1]
    assert any("not finite" in line for line in ours)
    assert stream.server.terminate() == 0


def test_a_store_written_with_no_detect_is_worked_out_here(tmp_path):
    records = everything(150)
    stream = Stream(tmp_path, *FAST, "--no-detect")
    push(stream, records)
    assert read_server_report(stream.incidents_file.parent)["state"] == "missing"
    evidence, summary = look(stream, tmp_path)
    assert (evidence["detection"]["source"], evidence["detection"]["state"]) == ("python", "missing")
    assert differences(reference(records), [i for i in evidence["incidents"] if i["kind"] in PORTED]) == []
    assert evidence["samples"]  # the parts were read
    assert "worked out here from the stored parts, because the server wrote no incidents.json" in summary
    assert stream.server.terminate() == 0


def from_store(stream: Stream) -> str:
    """What Python reports from the parts the server stored, as text (a NaN equals itself there)."""
    return as_text(tensorquill.build_evidence(stream.incidents_file.parent)["incidents"])


def as_text(incidents: list[dict]) -> str:
    return json.dumps(incidents, sort_keys=True)


def keep_open(stream: Stream, records: list[dict]) -> FlightSender:
    """Send records as a job that is still running does: the stream stays open."""
    stream.file.write_text("".join(json.dumps(item) + "\n" for item in records))
    stream.sent.extend(records)
    sender = FlightSender(stream.file, stream.server.endpoint, stream.run, batch_seconds=0.05, poll_seconds=0.02)
    deadline = time.monotonic() + 10
    while read_server_report(stream.incidents_file.parent)["stored"] != (records[-1]["attempt"], records[-1]["step"]) or (
        read_server_report(stream.incidents_file.parent)["state"] != "current"
    ):
        assert time.monotonic() < deadline, "the server did not report on the rows"
        time.sleep(0.05)
    return sender


@pytest.mark.parametrize("grace, later", [(3600.0, []), (0.0, [("python", "stale")])])
def test_a_report_left_behind_by_a_restart_without_detection_goes_stale(tmp_path, grace, later):
    records = everything()
    stream = Stream(tmp_path, *FAST)
    stream.send(records[:100])
    folder = stream.incidents_file.parent
    seen = []

    def on_update(evidence, path):
        seen.append((evidence["detection"]["source"], evidence["detection"]["state"]))
        if len(seen) == 1:
            # The server is restarted with its detectors off; the job goes on, and the
            # steps in which most of the run's incidents happen are stored unjudged.
            assert evidence["incidents"] and differences(reference(records[:100]), evidence["incidents"]) == []
            assert stream.server.terminate() == 0
            stream.server = Server(stream.store, 0, *FAST, "--no-detect")
            push(stream, records[100:])

    # Within the grace the watcher waits for the report to catch up and writes nothing new.
    # Once it has passed, the old report is stale and its incidents are not shown as the run's.
    out = watch(folder, tmp_path / "live.json", interval=0.0, polls=4, on_update=on_update, report_grace=grace)
    assert seen == [("server", "current"), *later]
    state = read_server_report(folder)
    assert state["state"] == "behind" and state["report"]["last_step"] == 99 and state["stored"] == (1, 299)
    if later:
        evidence = json.loads(out.read_text())
        assert as_text(evidence["incidents"]) == from_store(stream) and len(evidence["incidents"]) > len(reference(records[:100]))
        assert "the server's report stops at step 99 of attempt 1 and the store holds step 299 of attempt 1" in (
            incident_summary(evidence))
    assert stream.server.terminate() == 0


def test_after_a_kill_the_report_stands_and_the_status_is_called_old(tmp_path, monkeypatch):
    records = everything()
    stream = Stream(tmp_path, *FAST, *QUICK_STATUS)
    sender = keep_open(stream, records[:200])
    try:
        time.sleep(0.5)  # a status is written
        stream.server.kill()
        # Every stored step was judged before the kill, so the report is still the store's.
        evidence, summary = look(stream, tmp_path)
        assert (evidence["detection"]["source"], evidence["detection"]["state"]) == ("server", "current")
        assert differences(reference(records[:200]), evidence["incidents"]) == []
        assert evidence["watchdog"]["summary"] == "active" and "status was last written" not in summary

        # Later the dead server's last status still says "active". It is not taken as news.
        time.sleep(STATUS_OLD)
        evidence, summary = look(stream, tmp_path)
        assert evidence["watchdog"]["not_written_for_seconds"] > 0
        assert "The server's status was last written" in summary and "Nothing is watching this run for stalls" in summary

        # A new server, and the rank has not reconnected: nothing changes on disk, and the
        # old report is still current for what is stored.
        stream.server = Server(stream.store, 0, *FAST, *QUICK_STATUS)
        time.sleep(0.5)
        evidence, summary = look(stream, tmp_path)
        assert evidence["detection"]["state"] == "current" and "The server's status was last written" in summary
    finally:
        sender.close(0.0)

    # The rank reconnects: the server rebuilds its state from the store and reports on.
    document = stream.send(records[200:])
    assert document["rebuilds"] == 1
    status_file = stream.store / "runs" / RUN / "watchdog.json"
    deadline = time.monotonic() + 10
    while json.loads(status_file.read_text())["summary"] != "finished":  # the new server's first status
        assert time.monotonic() < deadline
        time.sleep(0.05)
    evidence, summary = look(stream, tmp_path)
    assert evidence["detection"]["report_last_step"] == 299 and "status was last written" not in summary
    assert differences(reference(records), evidence["incidents"]) == []
    assert stream.server.terminate() == 0


def test_a_kill_between_a_part_and_its_report_leaves_a_report_that_is_behind(tmp_path):
    records = everything()
    stream = Stream(tmp_path, *FAST)
    stream.send(records[:100])
    before = stream.incidents_file.read_bytes()
    stream.send(records[100:])
    stream.server.kill()
    # What a kill right after the last part was stored would leave: the part, and the
    # report from before it.
    stream.incidents_file.write_bytes(before)
    evidence, summary = look(stream, tmp_path)
    assert (evidence["detection"]["source"], evidence["detection"]["state"]) == ("python", "behind")
    assert as_text(evidence["incidents"]) == from_store(stream) and len(evidence["incidents"]) > len(reference(records[:100]))

    # The next server judges everything stored once the rank sends again.
    stream.server = Server(stream.store, 0, *FAST)
    more = [record(step, grad_norm_total=1.0, lr=[1e-3]) for step in range(300, 320)]
    stream.send(more)
    evidence, _ = look(stream, tmp_path)
    assert (evidence["detection"]["source"], evidence["detection"]["report_last_step"]) == ("server", 319)
    assert differences(reference(records + more), evidence["incidents"]) == []
    assert stream.server.terminate() == 0


def test_a_stopped_server_is_not_reported_as_a_quiet_run(tmp_path):
    stream = Stream(tmp_path, *FAST, *QUICK_STATUS)
    sender = keep_open(stream, healthy(range(60)))
    try:
        assert stream.server.terminate() == 0
        evidence, summary = look(stream, tmp_path)
    finally:
        sender.close(0.0)
    assert evidence["watchdog"]["summary"] == "server_stopped" and evidence["incidents"] == []
    assert summary.splitlines()[:2] == [
        "No incidents detected.",
        "The server has stopped: nothing is watching this run for stalls, and no new steps are stored.",
    ]
    assert evidence["detection"]["state"] == "current"


def test_a_resume_from_an_earlier_step_is_current_though_an_older_attempt_went_further(tmp_path, monkeypatch):
    stream = Stream(tmp_path, *FAST)
    first = healthy(range(200))
    stream.send(first)
    resumed = healthy(range(100, 130), attempt=2)
    document = stream.send(resumed)
    assert (document["attempt"], document["last_step"]) == (2, 129)
    folder = stream.incidents_file.parent
    assert sorted(p.name for p in folder.glob("attempt=1/*"))[-1] == "part-00000150-00000199.parquet"
    state = read_server_report(folder)
    assert state["state"] == "current" and state["stored"] == (2, 129)
    no_part_is_read(monkeypatch)
    evidence, _ = look(stream, tmp_path)
    assert evidence["detection"]["source"] == "server"
    assert differences(reference(first + resumed), evidence["incidents"]) == []
    assert stream.server.terminate() == 0


def test_reports_that_cannot_be_used_and_why(tmp_path):
    records = everything(120)
    stream = Stream(tmp_path, *FAST)
    document = stream.send(records)
    assert stream.server.terminate() == 0
    folder = stream.incidents_file.parent
    assert read_server_report(folder)["state"] == "current"

    # Other thresholds than the server's.
    for options in ({"overload_threshold": 0.5}, {"top_k": 2}, {"drift_tolerance": 1e-3}):
        assert read_server_report(folder, **options)["state"] == "other_thresholds"
    evidence, summary = look(stream, tmp_path, overload_threshold=0.5)
    assert (evidence["detection"]["source"], evidence["detection"]["state"]) == ("python", "other_thresholds")
    assert evidence["incidents"] == tensorquill.build_evidence(records, overload_threshold=0.5)["incidents"]
    assert "overload_threshold 0.35 where 0.5 was asked for" in summary

    def with_report(change) -> dict:
        changed = json.loads(json.dumps(document))
        change(changed)
        stream.incidents_file.write_text(json.dumps(changed))
        return read_server_report(folder)

    # A report from the server as it was with six kinds, whose `not_checked` held kind names.
    def older(report):
        report["checked"] = ["moe_imbalance", "starved_experts", "grad_spike", "activation_spike",
                             "activation_drift", "dead_activations"]
        report["not_checked"] = ["stalled_weights", "non_finite"]
        report["incidents"] = []

    state = with_report(older)
    assert state["state"] == "unusable" and "does not cover stalled_weights" in state["reason"]
    evidence, summary = look(stream, tmp_path)
    assert evidence["detection"]["source"] == "python" and "an older server wrote it" in summary
    assert differences(reference(records), [i for i in evidence["incidents"] if i["kind"] in PORTED]) == []

    assert with_report(lambda r: r.update(rank=3))["state"] == "unusable"
    assert with_report(lambda r: r.update(run_id="another"))["state"] == "unusable"
    assert with_report(lambda r: r.update(schema_version=2))["state"] == "unusable"
    assert with_report(lambda r: r.update(source="someone"))["state"] == "unusable"
    assert with_report(lambda r: r.update(last_step=500))["state"] == "unusable"  # past the store
    assert with_report(lambda r: r.update(attempt=0))["state"] == "behind"
    assert with_report(lambda r: r.update(last_step=None))["state"] == "behind"
    stream.incidents_file.write_text(json.dumps(document)[:200])
    assert read_server_report(folder)["state"] == "unreadable"
    stream.incidents_file.unlink()
    assert read_server_report(folder)["state"] == "missing"
    assert read_server_report(tmp_path / "store" / "runs" / RUN / "rank=7")["state"] == "nothing_stored"


def test_recompute_ignores_a_current_report(tmp_path):
    records = everything(120)
    stream = Stream(tmp_path, *FAST)
    stream.send(records)
    evidence, summary = look(stream, tmp_path, server_reports=False)
    assert "detection" not in evidence and evidence["samples"]
    assert evidence["incidents"] == tensorquill.build_evidence(records)["incidents"]
    assert "tensorquill-server's report" not in summary
    assert stream.server.terminate() == 0
