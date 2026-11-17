"""One run state (healthy, degraded, unknown, unhealthy) in `evidence.json`, the all-ranks view,
the CLI's output and the server's `watchdog.json`.

The rule it is held to above all: the state never reads `healthy` while required evidence is
missing or stale. The server half needs `pyarrow` and the server binary.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

import tensorquill
from tensorquill import evidence as ev

ACTIVE = {"schema_version": 1, "summary": "active", "ranks": [{"rank": 0, "state": "active"}], "missing_ranks": [], "incidents": []}


def state(**given) -> tuple[str, list[str]]:
    options = {"incidents": [], "steps_judged": True, "status": ACTIVE, **given}
    found = ev.run_state(**options)
    return found["state"], [r["code"] for r in found["reasons"]]


# --- the rule, one input at a time ----------------------------------------------------------------


def test_healthy_only_with_steps_a_status_and_nothing_against_it():
    assert state() == ("healthy", [])
    assert state(status={**ACTIVE, "summary": "finished"}) == ("healthy", [])


def test_an_incident_is_unhealthy_and_wins_over_everything_else():
    assert state(incidents=[{"kind": "grad_spike", "step": 3}]) == ("unhealthy", ["incidents"])
    found = ev.run_state(incidents=[{"kind": "grad_spike"}, {"kind": "grad_spike"}, {"kind": "non_finite"}],
                         steps_judged=False, status=None, not_checked=[{"check": "traincheck"}])
    assert found["state"] == "unhealthy"
    assert [r["code"] for r in found["reasons"]] == ["incidents", "no_steps", "no_status", "not_checked"]
    assert found["reasons"][0]["kinds"] == ["grad_spike", "non_finite"]


@pytest.mark.parametrize("given, code", [
    ({"steps_judged": False}, "no_steps"),
    ({"status": {**ACTIVE, "summary": "server_stopped"}}, "server_stopped"),
    ({"status_age": 12.0}, "status_old"),
    ({"status": {**ACTIVE, "missing_ranks": [3]}}, "ranks_unknown"),
    ({"unknown_ranks": {2: "missing: no incidents.json"}}, "ranks_unknown"),
    ({"status": {**ACTIVE, "summary": "stalled"}}, "status_summary"),  # stalled, yet no stall incident
    ({"status": {**ACTIVE, "summary": "degraded"}}, "status_summary"),
    ({"status": {**ACTIVE, "summary": "something new"}}, "status_summary"),
    ({"status": {**ACTIVE, "summary": None}}, "status_summary"),
    ({"detection": {"source": "server", "state": "behind", "reason": "the report stops at step 9"}}, "report_not_current"),
])
def test_missing_or_stale_evidence_is_unknown_never_healthy(given, code):
    assert state(**given) == ("unknown", [code])


@pytest.mark.parametrize("report_state", [
    "nothing_stored", "missing", "behind", "stale", "unreadable", "unusable", "other_thresholds", "anything else",
])
def test_every_report_state_but_current_is_unknown(report_state):
    assert state(detection={"source": "server", "state": report_state, "reason": "x"})[0] == "unknown"
    assert state(detection={"source": "server", "state": "current", "reason": "x"}) == ("healthy", [])
    # Worked out here from every stored part: complete, whatever the report's state was.
    assert state(detection={"source": "python", "state": report_state, "reason": "x"}) == ("healthy", [])


def test_ranks_the_status_expects_are_unknown_until_judged():
    # One rank connected of three, before the status's silence floor calls the others missing.
    found = ev.run_state(incidents=[], steps_judged=True, status={**ACTIVE, "world_size": 3})
    assert (found["state"], found["reasons"][0]["code"], found["reasons"][0]["ranks"]) == ("unknown", "ranks_unknown", [1, 2])
    # A rank the all-ranks view judged from its own report is not unknown, listed or not.
    found = ev.run_state(incidents=[], steps_judged=True, status={**ACTIVE, "world_size": 3}, judged_ranks=[0, 1, 2])
    assert found["state"] == "healthy"
    found = ev.run_state(incidents=[], steps_judged=True, status={**ACTIVE, "missing_ranks": [5]})
    assert found["reasons"][0]["ranks"] == [5]


def test_a_stall_is_unknown_and_its_incident_an_observation():
    """The stream cannot tell a hang from a long pause or a dead sender, so `job_stalled` claims
    no failure (an observation) and the stalled summary makes the state unknown, never healthy."""
    stalled = {**ACTIVE, "summary": "stalled"}
    assert state(status=stalled, incidents=[{"kind": "job_stalled", "step": 4}]) == (
        "unknown", ["status_summary", "observations"])


def test_an_optional_monitor_missing_is_degraded():
    assert state(status=None) == ("degraded", ["no_status"])
    found = ev.run_state(incidents=[], steps_judged=True, status=ACTIVE,
                         not_checked=[{"check": "resume_lr"}, {"check": "attention"}, {"check": "resume_lr"}])
    assert (found["state"], found["reasons"][0]["checks"]) == ("degraded", ["resume_lr", "attention"])
    # Unknown outranks degraded.
    assert state(status=None, steps_judged=False) == ("unknown", ["no_steps", "no_status"])


def test_the_line_names_what_decides_it():
    evidence = {"run_state": ev.run_state(incidents=[], steps_judged=True, status=None)}
    assert ev.run_state_line(evidence).startswith("Run state: degraded (no progress status")
    assert ev.run_state_line({}) is None


# --- in the evidence ---------------------------------------------------------------------------------


def records(steps=30, spike_at=None):
    return [{"schema_version": 1, "run_id": "r", "rank": 0, "step": s, "ts": float(s),
             "grad_norm": (9.0 if s == spike_at else 1.0) * (0.9 + (s * 37 % 21) / 100)} for s in range(steps)]


def test_build_evidence_carries_it_and_schema_version_stays_1(tmp_path):
    evidence = tensorquill.build_evidence(records())
    assert evidence["schema_version"] == 1
    # Records alone, which no server was watching: the stall kinds are not configured, and nothing is missing.
    assert evidence["run_state"] == {"state": "healthy", "reasons": []}
    # A status that was named and is not there: nothing watches for stalls where something should.
    assert tensorquill.build_evidence(records(), watchdog=tmp_path / "watchdog.json")["run_state"]["state"] == "degraded"
    assert tensorquill.build_evidence(records(spike_at=20))["run_state"]["state"] == "unhealthy"
    assert tensorquill.build_evidence([])["run_state"]["state"] == "unknown"


def test_a_status_beside_the_records_counts(tmp_path):
    status = tmp_path / "watchdog.json"
    status.write_text(json.dumps({**ACTIVE, "summary": "finished", "written_at_ms": 1}))
    assert tensorquill.build_evidence(records(), watchdog=status)["run_state"] == {"state": "healthy", "reasons": []}
    status.write_text(json.dumps({**ACTIVE, "summary": "active", "server_running": True, "refresh_seconds": 1.0,
                                  "written_at_ms": int((time.time() - 60) * 1000)}))
    found = tensorquill.build_evidence(records(), watchdog=status)["run_state"]
    assert (found["state"], [r["code"] for r in found["reasons"]]) == ("unknown", ["status_old"])


def test_the_watcher_calls_a_status_old_by_its_own_clock(tmp_path):
    """A status stamped in the future (the server's clock ahead of this one) that is never
    rewritten: the wall clock cannot call it old, the watcher's own clock does."""
    source = tmp_path / "run.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in records()))
    status = tmp_path / "watchdog.json"
    status.write_text(json.dumps({**ACTIVE, "server_running": True, "refresh_seconds": 0.1,
                                  "written_at_ms": int((time.time() + 3600) * 1000)}))
    seen: list[dict] = []
    ev.watch(source, tmp_path / "e.json", interval=0.1, polls=6, watchdog=status,
             on_update=lambda evidence, path: seen.append(evidence["run_state"]))
    assert seen[0] == {"state": "healthy", "reasons": []}
    assert seen[-1]["state"] == "unknown" and [r["code"] for r in seen[-1]["reasons"]] == ["status_old"]


def test_the_cli_prints_it(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records(spike_at=20)))
    out = subprocess.run([sys.executable, "-m", "tensorquill", str(path)], capture_output=True, text=True, check=True,
                         env={**__import__("os").environ, "PYTHONPATH": str(Path(ev.__file__).parents[1])}).stdout
    assert "Run state: unhealthy (1 incident listed: grad_spike)" in out


# --- through the server -----------------------------------------------------------------------------


def _server():
    pytest.importorskip("pyarrow.flight")
    from test_flight_e2e import BINARY

    if BINARY is None:
        pytest.skip("tensorquill-server is not built")
    import test_watch_all_ranks as twa

    return twa


def comparable(found: dict) -> tuple:
    """What both languages must agree on: the state, the codes, the kinds, the ranks and the checks."""
    return (found["state"], sorted(
        (r["code"], tuple(sorted(r.get("kinds", ()))), tuple(r.get("ranks", ())), tuple(sorted(r.get("checks", ()))))
        for r in found["reasons"]
    ))


def server_state(run) -> dict:
    return json.loads((run.folder / "watchdog.json").read_text())["run_state"]


def wait_for_state(run, wanted: str) -> dict:
    deadline = time.monotonic() + 10
    while True:
        try:
            found = server_state(run)
            if found["state"] == wanted:
                return found
        except (FileNotFoundError, KeyError, ValueError):
            pass
        assert time.monotonic() < deadline, f"watchdog.json never said {wanted}"
        time.sleep(0.05)


def test_the_server_and_the_all_ranks_view_agree(tmp_path):
    twa = _server()
    from test_server_reports import QUICK_STATUS

    run = twa.Run(tmp_path, *twa.FAST, *QUICK_STATUS)
    # Two ranks, still running: healthy.
    senders = [run.keep_open(rank, twa.steady(60), world_size=2) for rank in (0, 1)]
    found = wait_for_state(run, "healthy")
    evidence, _ = twa.look(run, tmp_path / "a")
    assert comparable(evidence["run_state"]) == comparable(found) == ("healthy", [])
    # An incident on rank 1.
    run.file(1).open("a").write("".join(json.dumps({**r, "rank": 1}) + "\n" for r in twa.steady(120, spike=(100, 9.0))[60:]))
    run.wait_for(1, twa.steady(120)[-1])
    found = wait_for_state(run, "unhealthy")
    evidence, _ = twa.look(run, tmp_path / "b")
    assert comparable(evidence["run_state"]) == comparable(found)
    assert found["reasons"][0]["kinds"] == ["grad_spike"]
    for sender in senders:
        sender.close()
    # Stopped with nothing left open: the run finished and keeps its verdict.
    found = wait_for_state(run, "unhealthy")
    assert run.server.terminate() == 0
    evidence, _ = twa.look(run, tmp_path / "c")
    assert comparable(evidence["run_state"]) == comparable(server_state(run))


def test_a_stopped_server_and_a_rank_never_judged_are_unknown_in_both(tmp_path):
    twa = _server()
    from test_server_reports import QUICK_STATUS

    run = twa.Run(tmp_path, *twa.FAST, *QUICK_STATUS)
    sender = run.keep_open(0, twa.steady(60), world_size=3)  # ranks 1 and 2 never come
    found = wait_for_state(run, "unknown")
    evidence, _ = twa.look(run, tmp_path / "a")
    assert comparable(evidence["run_state"]) == comparable(found)
    assert next(r for r in found["reasons"] if r["code"] == "ranks_unknown")["ranks"] == [1, 2]
    assert run.server.terminate() == 0  # the stream is still open: the server stopped under it
    sender.close()
    found = server_state(run)
    assert "server_stopped" in [r["code"] for r in found["reasons"]] and found["state"] == "unknown"
    evidence, _ = twa.look(run, tmp_path / "b")
    assert comparable(evidence["run_state"]) == comparable(found)


def test_what_a_rank_could_not_check_makes_the_run_degraded_in_both(tmp_path):
    twa = _server()
    from test_attention import attention
    from test_server_reports import QUICK_STATUS

    run = twa.Run(tmp_path, *twa.FAST, *QUICK_STATUS)
    # Rank 1 ran the probe twice and the second one failed, which its report lists.
    probes = {19: {"attention": {"blk": attention(0.4, 9.0)}}, 39: {"error": "RuntimeError: CUDA out of memory"}}
    probed = [{**r, "probe": probes[r["step"]]} if r["step"] in probes else r for r in twa.steady(60)]
    senders = [run.keep_open(0, twa.steady(60)), run.keep_open(1, probed)]
    found = wait_for_state(run, "degraded")
    assert found["reasons"] == [{**found["reasons"][0], "code": "not_checked", "checks": ["probe"]}]
    evidence, _ = twa.look(run, tmp_path / "a")
    assert comparable(evidence["run_state"]) == comparable(found)
    for sender in senders:
        sender.close()
    assert run.server.terminate() == 0


def test_a_rule_that_is_still_warming_up_leaves_the_run_healthy_in_both(tmp_path):
    twa = _server()
    from test_attention import attention
    from test_server_reports import QUICK_STATUS

    run = twa.Run(tmp_path, *twa.FAST, *QUICK_STATUS)
    # Rank 1 ran the probe twice: too few probes for the attention rules, which need four. That
    # was a `not_checked` entry and a degraded run; it is a kind checked without enough history.
    probed = [{**r, "probe": {"attention": {"blk": attention(0.4, 9.0)}}} if r["step"] in (19, 39) else r
              for r in twa.steady(60)]
    senders = [run.keep_open(0, twa.steady(60)), run.keep_open(1, probed)]
    found = wait_for_state(run, "healthy")
    assert found == {"state": "healthy", "reasons": []}
    report = json.loads(run.report_file(1).read_text())
    assert [n["check"] for n in report["not_checked"]] == ["flight_recorder", "traincheck"]
    for kind in ev.ATTN_KINDS:
        assert report["kind_checks"][kind] == {
            "value": "checked", "records": 2, "records_needed": 4, "enough_history": False,
            "needs": "probe with attention (probe=)",
        }
    assert report["kind_checks"]["grad_spike"]["enough_history"] is True  # 60 of 13
    evidence, summary = twa.look(run, tmp_path / "a")
    assert comparable(evidence["run_state"]) == comparable(found)
    # One rank's evidence from the same store says which kinds "nothing found" does not cover.
    one = tensorquill.build_evidence(run.folder, rank=1)
    assert one["run_state"]["state"] == "healthy" and "not_checked" not in one
    assert list(ev.kinds_by_value(one)["not_enough_history"]) == list(ev.ATTN_KINDS)
    assert "3 of them without enough history yet (attention_collapse 2 of 4 records" in ev.incident_summary(one)
    for sender in senders:
        sender.close()
    assert run.server.terminate() == 0


def test_no_detect_is_unknown_in_both(tmp_path):
    twa = _server()
    from test_server_reports import QUICK_STATUS

    run = twa.Run(tmp_path, *twa.FAST, *QUICK_STATUS, "--no-detect")
    # Sent directly: `keep_open` waits for a report, which this server never writes.
    run.file(0).write_text("".join(json.dumps({**r, "rank": 0}) + "\n" for r in twa.steady(60)))
    from tensorquill.flight import send_file

    assert send_file(run.file(0), run.server.endpoint, twa.RUN, 0, timeout=30.0)
    found = wait_for_state(run, "unknown")
    evidence, _ = twa.look(run, tmp_path / "a")
    assert comparable(evidence["run_state"]) == comparable(found)
    assert {r["code"] for r in found["reasons"]} == {"no_steps", "ranks_unknown"}
    assert "--no-detect" in next(r for r in found["reasons"] if r["code"] == "ranks_unknown")["detail"]
    assert run.server.terminate() == 0


def test_the_per_rank_watcher_is_unknown_while_the_status_is_old(tmp_path):
    twa = _server()
    from test_server_reports import QUICK_STATUS, STATUS_OLD

    run = twa.Run(tmp_path, *twa.FAST, *QUICK_STATUS)
    sender = run.keep_open(0, twa.steady(60))
    seen: list[dict] = []
    ev.watch(run.rank_folder(0), tmp_path / "e.json", interval=0.0, polls=1, on_update=lambda e, p: seen.append(e))
    assert seen[-1]["run_state"] == {"state": "healthy", "reasons": []}
    run.server.kill()  # no clean stop: the status claims a running server and stops being rewritten
    time.sleep(STATUS_OLD * 1.5)
    ev.watch(run.rank_folder(0), tmp_path / "f.json", interval=0.0, polls=1, on_update=lambda e, p: seen.append(e))
    assert comparable(seen[-1]["run_state"])[0] == "unknown"
    assert "status_old" in [r["code"] for r in seen[-1]["run_state"]["reasons"]]
    sender.close()
