"""Wave AH, P1-1: a rank that was already stopped when its stream dropped keeps its severity and
its incident while it reads `reconnecting`, and the trigger fires once for the whole episode.

Before it, a rank that was `critical` with a suspected hang and was then killed read
`reconnecting` with no severity for the whole silence floor (300 s by default): the run read
`active` with no incident (proof 1 of `wave-af-proof-cpu.md`, phase C).

Three parts: the shared cases (`fixtures/wave_ah/cases.json`, which the server's
`watchdog/wave_ah_tests.rs` reads too), the trigger on made statuses, and the real server driven
with raw Flight calls through a critical stall, a drop, the grace period's end, a stream opened
again with no progress, and a new step, with the real evidence fed to a real `Trigger`.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tensorquill import evidence as ev
from tensorquill.evidence import detect_watchdog_incidents, incident_summary
from tensorquill.mcp_server import all_ranks_look
from tensorquill.trigger import Trigger, state_path_for

import test_trigger as tt

CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_ah" / "cases.json").read_text())["cases"]


@pytest.fixture
def hook(tmp_path):
    script, log = tmp_path / "hook.py", tmp_path / "hook.log"
    script.write_text(tt.HOOK)

    def command() -> str:
        return " ".join(shlex.quote(str(part)) for part in (sys.executable, script, log))

    def calls() -> list[dict]:
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    return command, calls


def brief(incident: dict) -> dict:
    out = {"kind": incident["kind"], "severity": incident["severity"], "assessment": incident["assessment"],
           "step": incident["step"], "ranks": [r["rank"] for r in incident["ranks"]]}
    if "other_ranks" in incident:
        out["other_ranks"] = [r["rank"] for r in incident["other_ranks"]]
    if "lost_ranks" in incident:
        out["lost_ranks"] = incident["lost_ranks"]
    return out


@pytest.mark.parametrize("name", sorted(CASES))
def test_python_reads_the_shared_cases_as_the_server_does(name):
    case = CASES[name]
    assert [brief(i) for i in detect_watchdog_incidents(case["status"])] == case["expected"]


def test_one_rule_says_which_ranks_are_stopped():
    stopped = ev.rank_is_stopped
    assert all(stopped({"state": state}) for state in ("silent", "lost", "unresponsive"))
    assert stopped({"state": "reconnecting", "severity": "warning"})
    assert stopped({"state": "reconnecting", "severity": "critical"})
    assert not stopped({"state": "reconnecting", "severity": None})
    assert not stopped({"state": "reconnecting"})
    assert not any(stopped({"state": state, "severity": "critical"}) for state in ("active", "starting", "paused", "finished"))


def test_the_summary_says_the_connection_dropped_after_the_rank_had_stopped():
    case = CASES["a_critical_rank_that_disconnects_keeps_its_incident"]
    evidence = {"run_id": "ah", "incidents": detect_watchdog_incidents(case["status"])}
    text = incident_summary(evidence)
    assert "its connection has dropped since it stopped and has not come back yet" in text
    assert "its heartbeats continue" not in text
    # The stack groups are of the stopped ranks: a held one among them.
    status = json.loads(json.dumps(case["status"]))
    status["ranks"][1]["stack"] = {"still_seconds": 900.0, "unchanged_seconds": 800.0, "threads": [
        {"name": "MainThread", "main": True, "frames": [{"function": "step", "file": "/job/train.py", "line": 27}]}]}
    [incident] = detect_watchdog_incidents(status)
    assert [place["rank"] for place in incident["stuck_at"]] == [1]


# --- the trigger, on made statuses ----------------------------------------------------------


def held(rank, last_step, attempt=7, severity="critical"):
    return {**tt._rank(rank, "reconnecting", severity, last_step), "stream": "dropped", "stopped_as": "silent",
            "attempt": attempt}


def test_one_episode_across_the_drop_the_loss_and_the_reconnect(tmp_path, hook):
    command, calls = hook
    run = tt._Looks(tmp_path, command, calls)
    stepping = lambda step: tt._ranked(0, "active", step, 7)  # noqa: E731
    assert run.look("degraded", stepping(100), tt._ranked(1, "silent", 49, 7)) == 1
    assert run.look("degraded", stepping(110), held(1, 49)) == 0  # killed: reconnecting, still critical
    assert run.look("degraded", stepping(120), held(1, 49), restart=True) == 0
    assert run.look("degraded", stepping(130), tt._ranked(1, "lost", 49, 7)) == 0  # the grace ended
    assert run.look("degraded", stepping(140), tt._ranked(1, "silent", 49, 7)) == 0  # its stream opened again
    assert run.fired() == [("rank_silent", 49)]
    # A new step ends the episode; the next stall of the rank is a new one.
    assert run.look("active", stepping(150), tt._ranked(1, "active", 60, 7)) == 0
    assert run.look("degraded", stepping(400), held(1, 60)) == 1
    assert run.fired() == [("rank_silent", 49), ("rank_silent", 60)]


def test_a_stall_first_seen_while_the_rank_reconnects_opens_an_episode_that_can_end(tmp_path, hook):
    """The watcher starts after the kill: its first critical stall is of a `reconnecting` rank. The
    episode must hold that rank, or no later step of it could end the episode."""
    command, calls = hook
    run = tt._Looks(tmp_path, command, calls)
    assert run.look("degraded", tt._ranked(0, "active", 100, 7), held(1, 49)) == 1
    state = json.loads(state_path_for(run.path).read_text())
    assert state["episode"]["ranks"] == {"1": {"last_step": 49, "attempt": 7}}, state["episode"]
    assert run.look("degraded", tt._ranked(0, "active", 110, 7), held(1, 49)) == 0
    # A replacement attempt takes the rank and stalls before it stores a step: a new episode.
    assert run.look("active", tt._ranked(0, "active", 120, 7), tt._ranked(1, "starting", None, 8)) == 0
    assert run.look("degraded", tt._ranked(0, "active", 500, 7), tt._ranked(1, "silent", None, 8)) == 1
    assert [kind for kind, _ in run.fired()] == ["rank_silent", "rank_silent"]


# --- the real server ------------------------------------------------------------------------

w2 = pytest.importorskip("test_lane_w2_attempt_order")
RUN = "ah-held"
FLAGS = ("--silent-seconds", "2", "--silent-factor", "0", "--critical-seconds", "5", "--watchdog-seconds", "0.1",
         "--flush-seconds", "0.1")
ATTEMPT = 5


STREAM = r"""
import json, sys
import pyarrow as pa
import pyarrow.flight as flight
from tensorquill.flight import flight_schema, records_to_batch

endpoint, run, rank, attempt = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
client = flight.FlightClient(endpoint)
descriptor = flight.FlightDescriptor.for_path("runs", run, f"rank={rank}")
writer, reader = client.do_put(descriptor, flight_schema(), flight.FlightCallOptions(timeout=300.0))
offset = 0
print("open", flush=True)
for line in sys.stdin:
    word, *numbers = line.split()
    if word == "close":
        writer.done_writing()
        writer.close()
        break
    first, end = map(int, numbers)
    records = [{"schema_version": 1, "run_id": run, "rank": rank, "attempt": attempt, "step": step,
                "ts": float(step), "grad_norm": 1.0, "lr": [0.1]} for step in range(first, end)]
    offset += 100 * len(records)
    metadata = {"schema_version": 1, "run_id": run, "rank": rank, "world_size": 2, "sender_attempt": attempt,
                "min_step": first, "max_step": end - 1, "end_byte_offset": offset}
    writer.write_with_metadata(records_to_batch(records), pa.py_buffer(json.dumps(metadata).encode()))
    reader.read()
    print("stored", flush=True)
client.close()
"""


class RankStream:
    """One `DoPut` stream of a rank, held by a process of its own, as a training process holds
    its sender's: `drop` kills it, which is what the server sees when a job is killed."""

    def __init__(self, endpoint: str, rank: int) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-c", STREAM, endpoint, RUN, str(rank), str(ATTEMPT)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, env={**os.environ, "PYTHONPATH": str(Path(ev.__file__).resolve().parents[1])},
        )
        assert self.process.stdout.readline().strip() == "open"

    def send(self, steps: range) -> None:
        """Send the steps and wait for the server's acknowledgement, which comes once they are stored."""
        self.process.stdin.write(f"send {steps[0]} {steps[-1] + 1}\n")
        self.process.stdin.flush()
        assert self.process.stdout.readline().strip() == "stored"

    def drop(self) -> None:
        """SIGKILL: the connection goes with the process, and the stream never ends."""
        self.process.kill()
        self._reap()

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.stdin.write("close\n")
                self.process.stdin.flush()
                self.process.wait(20)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
        self._reap()

    def _reap(self) -> None:
        self.process.wait(20)
        for pipe in (self.process.stdin, self.process.stdout):
            try:
                pipe.close()
            except OSError:
                pass


def rank_of(status: dict, rank: int) -> dict:
    return next(r for r in status["ranks"] if r["rank"] == rank)


@w2.needs_server
def test_the_real_server_holds_a_critical_stall_through_the_drop_and_the_trigger_fires_once(tmp_path, hook):
    command, calls = hook
    store = tmp_path / "store"
    server = w2.Server(store, *FLAGS)
    folder = store / "runs" / RUN
    trigger = Trigger(command())
    out = tmp_path / "all.evidence.json"
    stop = threading.Event()
    seen: list[tuple[str, str, str | None]] = []

    def look(wanted, seconds=30.0) -> dict:
        """The first status in which rank 1 reads `wanted`, and what a watcher makes of it then."""
        status = w2.wait_status(store, lambda s: wanted(rank_of(s, 1)), seconds, run_id=RUN)
        one = rank_of(status, 1)
        assert wanted(one), one
        evidence = all_ranks_look(folder)
        trigger.look(evidence, out)
        stalls = [i for i in evidence["run_incidents"] if i["kind"] in ("rank_silent", "job_stalled")]
        seen.append((one["state"], one["severity"], stalls[0]["assessment"] if stalls else None))
        return {"status": status, "rank": one, "stalls": stalls, "fired": len(trigger.deliveries)}

    def keep_stepping(stream: RankStream) -> None:
        step = 0
        while not stop.is_set():
            stream.send(range(step, step + 5))
            step += 5
            stop.wait(0.2)

    zero, one = RankStream(server.endpoint, 0), RankStream(server.endpoint, 1)
    stepping = threading.Thread(target=keep_stepping, args=(zero,), daemon=True)
    try:
        one.send(range(0, 10))
        stepping.start()
        # 1. Silent with its stream open, then critical: the stall fires, once.
        at = look(lambda r: r["state"] == "silent" and r["severity"] == "critical")
        assert at["status"]["summary"] == "degraded" and at["fired"] == 1
        assert [(i["kind"], i["severity"], i["assessment"]) for i in at["stalls"]] == [("rank_silent", "critical", "suspected_hang")]
        # 2. Its process is killed: `reconnecting`, and the stall is still there, still critical.
        one.drop()
        at = look(lambda r: r["state"] == "reconnecting")
        assert (at["rank"]["severity"], at["rank"]["stopped_as"], at["rank"]["stream"]) == ("critical", "silent", "dropped")
        assert at["status"]["summary"] == "degraded" and at["fired"] == 1
        assert [(i["kind"], i["severity"], i["assessment"]) for i in at["stalls"]] == [("rank_silent", "critical", "suspected_hang")]
        assert [r["rank"] for r in at["stalls"][0]["ranks"]] == [1]
        # 3. The grace period ends: lost, a lost connection, and not back at warning.
        at = look(lambda r: r["state"] == "lost")
        assert at["rank"]["severity"] == "critical" and at["fired"] == 1
        assert [(i["kind"], i["severity"], i["assessment"]) for i in at["stalls"]] == [("rank_silent", "critical", "lost_connection")]
        # 4. Its stream opens again and stores nothing new: stopped as it was.
        one = RankStream(server.endpoint, 1)
        at = look(lambda r: r["stream"] == "open")
        assert (at["rank"]["state"], at["rank"]["severity"]) == ("silent", "critical") and at["fired"] == 1
        assert [(i["kind"], i["severity"]) for i in at["stalls"]] == [("rank_silent", "critical")]
        # 5. A new step: the rank recovers, and nothing fired a second time.
        one.send(range(10, 20))
        at = look(lambda r: r["state"] == "active")
        assert at["rank"]["severity"] is None and at["stalls"] == [] and at["status"]["summary"] == "active"
        assert at["fired"] == 1
        # 6. A healthy rank that disconnects: the ordinary grace, no severity, no incident.
        one.drop()
        at = look(lambda r: r["state"] == "reconnecting")
        assert at["rank"]["severity"] is None and "stopped_as" not in at["rank"]
        assert at["status"]["summary"] == "active" and at["stalls"] == [] and at["fired"] == 1
    finally:
        stop.set()
        stepping.join(10)
        for stream in (zero, one):
            stream.close()
        trigger.close(wait=10)
        server.stop()
    assert [c["payload"]["kind"] for c in calls()] == ["rank_silent"], seen
