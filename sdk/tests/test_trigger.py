"""The trigger of `--watch` (`tensorquill.trigger`): a command and a webhook, once for each new incident.

The plan's done test is the first one here: a seeded failure starts the command with no human step.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import tensorquill
from tensorquill.evidence import detect_watchdog_incidents, main, watch
from tensorquill.trigger import STALL_KINDS, Trigger, state_path_for

SRC = Path(__file__).resolve().parents[1] / "src"

# What `--on-incident` runs in these tests: it appends one line with what it was given to the
# file named first, and exits with the status named second (a path: 1 the first time, 0 after).
HOOK = """
import json, os, sys
payload = json.load(sys.stdin)
env = {k: v for k, v in os.environ.items() if k.startswith("TENSORQUILL_")}
with open(sys.argv[1], "a") as log:
    log.write(json.dumps({"payload": payload, "env": env}) + "\\n")
status = sys.argv[2] if len(sys.argv) > 2 else "0"
if not status.isdigit():
    seen = os.path.exists(status)
    open(status, "a").close()
    status = "0" if seen else "1"
sys.exit(int(status))
"""


@pytest.fixture
def hook(tmp_path):
    """`(command, calls)`: a command line for `--on-incident` and a reader of what it was given."""
    script, log = tmp_path / "hook.py", tmp_path / "hook.log"
    script.write_text(HOOK)

    def command(status: object = "") -> str:
        return " ".join(shlex.quote(str(part)) for part in (sys.executable, script, log, status) if str(part))

    def calls() -> list[dict]:
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    return command, calls


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    """A proxy from the machine's environment must not stand between a test and its local server."""
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("no_proxy", "*")


def _records(steps, rank=0, **more):
    return [{"run_id": "r", "rank": rank, "attempt": 1, "step": s, "grad_norm": 1.0, **more} for s in steps]


def _with(records, step, **change):
    """`records` with one step changed: the planted failure."""
    return [{**r, **change} if r["step"] == step else r for r in records]


NAN = {"grad_norm": "nan"}  # non_finite: an invariant
SPIKE = {"grad_norm": 100.0}  # grad_spike: an anomaly
THROTTLED = {"system": {"nvml": {"throttle": ["hw_thermal"], "temperature_c": 95}}}  # gpu_throttled: an observation


def _write(path, records, mode="w"):
    with path.open(mode) as file:
        file.writelines(json.dumps(r) + "\n" for r in records)


def _settle(trigger, done, seconds=20.0):
    """Tick as a watcher's later looks do until `done()` or the time is up."""
    deadline = time.monotonic() + seconds
    while not done() and time.monotonic() < deadline:
        trigger.close(wait=0.05)
        trigger.tick()
    return done()


# --- The done test, and once per incident ------------------------------------------------------


def test_a_planted_failure_in_a_growing_file_starts_the_command_once(tmp_path, hook, capsys):
    command, calls = hook
    records = _with(_records(range(60)), 30, **NAN)
    source = tmp_path / "w.jsonl"
    _write(source, records[:20])

    def grow(evidence, out):
        if len(evidence["incidents"]) == 0 and source.read_text().count("\n") == 20:
            _write(source, records[20:40], "a")  # the failure arrives with no one at the keyboard

    trigger = Trigger(command())
    out = watch(source, interval=0.02, polls=6, on_update=grow, trigger=trigger, compact_window=5)
    trigger.close(wait=20)
    assert out == tmp_path / "w.evidence.compact.json"
    (call,) = calls()
    payload, env = call["payload"], call["env"]
    incident = json.loads(out.read_text())["incidents"][0]
    assert incident["kind"] == "non_finite" and incident["step"] == 30
    assert payload == {
        "event": "incident", "version": 1, "id": "r/rank0/non_finite/30", "attempt": 1,
        "sent_at_ms": payload["sent_at_ms"], "run_id": "r", "evidence": str(out.resolve()),
        "run_state": "unhealthy", "kind": "non_finite", "class": "invariant", "step": 30, "rank": 0, "ranks": [0],
        "new_ranks": [0], "assessment": None, "summary": "NaN or Inf at step 30 in <total>", "incident": incident,
    }
    # The command inherits the watcher's environment, so a `TENSORQUILL_*` variable the watcher
    # itself was started with (CI's server job sets `TENSORQUILL_SERVER_BIN`) is there unchanged.
    inherited = {k: v for k, v in os.environ.items() if k.startswith("TENSORQUILL_")}
    assert env == {
        **inherited,
        "TENSORQUILL_EVIDENCE": str(out.resolve()), "TENSORQUILL_RUN_ID": "r",
        "TENSORQUILL_INCIDENT_ID": "r/rank0/non_finite/30", "TENSORQUILL_INCIDENT_KIND": "non_finite",
        "TENSORQUILL_INCIDENT_CLASS": "invariant", "TENSORQUILL_INCIDENT_STEP": "30",
        "TENSORQUILL_INCIDENT_RANKS": "0", "TENSORQUILL_INCIDENT_SUMMARY": "NaN or Inf at step 30 in <total>",
        "TENSORQUILL_INCIDENT_ASSESSMENT": "",
    }
    printed = capsys.readouterr().out
    assert "trigger: non_finite at step 30 (rank 0): started the command (pid " in printed

    # What fired is beside the evidence, under one name for the full and the compact file.
    state = tmp_path / "w.evidence.fired.json"
    assert state == state_path_for(out) == state_path_for(tmp_path / "w.evidence.json")
    (delivery,) = json.loads(state.read_text())["deliveries"]
    assert (delivery["kind"], delivery["step"], delivery["ranks"]) == ("non_finite", 30, [0])
    assert delivery["command"]["status"] == "done" and delivery["command"]["attempts"] == 1

    # Later looks, as the file grows and the evidence is written again: the incident is still
    # listed, and does not fire again.
    def more(evidence, out):
        if source.read_text().count("\n") == 40:
            _write(source, records[40:], "a")

    watch(source, interval=0.02, polls=5, on_update=more, trigger=trigger, compact_window=5)
    trigger.close(wait=5)
    assert len(json.loads(out.read_text())["incidents"]) == 1 and len(calls()) == 1

    # A restarted watcher, and one that writes the full file instead: not again either.
    for options in ({"compact_window": 5}, {}):
        again = Trigger(command())
        watch(source, interval=0.02, polls=3, trigger=again, **options)
        again.close(wait=5)
        assert again.deliveries == json.loads(state.read_text())["deliveries"]
    assert len(calls()) == 1 and (tmp_path / "w.evidence.json").exists()

    # A second failure of another kind and step is a new incident, for the restarted watcher too.
    _write(source, _records(range(60, 70)) + _records(range(65, 67)), "a")  # steps written twice
    last = Trigger(command())
    watch(source, interval=0.02, polls=3, trigger=last)
    last.close(wait=20)
    assert [(c["payload"]["kind"], c["payload"]["class"]) for c in calls()] == [
        ("non_finite", "invariant"), ("repeated_steps", "invariant"),
    ]


def test_the_command_line_wakes_the_command_with_no_human_step(tmp_path, hook):
    """`python -m tensorquill run.jsonl --watch --on-incident COMMAND`, as a user starts it."""
    command, calls = hook
    source = tmp_path / "run.jsonl"
    _write(source, _records(range(20)))
    watcher = subprocess.Popen(
        [sys.executable, "-m", "tensorquill", str(source), "--watch", "--interval", "0.1", "--compact",
         "--on-incident", command(),
         # Since Nov 17, 2026 `--watch` needs the Rust detectors unless this is typed. The test is of the
         # trigger and runs without the binary; tests/test_watch_rust.py fires it through the Rust detectors.
         "--detectors", "python"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(SRC)}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while not (tmp_path / "run.evidence.compact.json").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert calls() == []  # a healthy run so far: nothing started
        _write(source, _with(_records(range(20, 40)), 30, **NAN), "a")
        while not calls() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)  # a few more looks
    finally:
        watcher.send_signal(signal.SIGINT)
        try:
            printed, _ = watcher.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            watcher.kill()
            printed, _ = watcher.communicate()
    (call,) = calls()
    assert call["payload"]["kind"] == "non_finite" and call["payload"]["step"] == 30
    assert call["env"]["TENSORQUILL_EVIDENCE"] == str((tmp_path / "run.evidence.compact.json").resolve())
    # `--compact` given: the watcher writes the compact file alone, as before the trigger turned it on.
    assert not (tmp_path / "run.evidence.json").exists() and "hands over the compact evidence" not in printed
    assert "trigger: each new incident fires the command once; firing: invariants, a stall at critical\n" in printed
    assert "trigger: non_finite at step 30 (rank 0): started the command" in printed
    assert "Traceback" not in printed


def test_a_trigger_turns_on_the_compact_evidence_and_the_full_file_is_still_written(tmp_path, hook, capsys):
    """Nov 17, 2026: `--on-incident` or `--webhook` with no `--compact` writes the full file as
    before and hands over the compact one beside it; with `--compact` the watcher writes the
    compact file alone, as it always has, and that one is handed over."""
    command, calls = hook
    source = tmp_path / "run.jsonl"
    _write(source, _with(_records(range(40)), 30, **NAN))

    # Through the Python API with the option the command line passes.
    trigger = Trigger(command(), compact_window=5)
    out = watch(source, interval=0.0, polls=3, trigger=trigger)
    trigger.close(wait=20)
    assert out == tmp_path / "run.evidence.json" and "compact" not in json.loads(out.read_text())
    compact = tmp_path / "run.evidence.compact.json"
    assert json.loads(compact.read_text())["compact"]["window"] == 5
    (call,) = calls()
    assert call["env"]["TENSORQUILL_EVIDENCE"] == call["payload"]["evidence"] == str(compact.resolve())
    assert state_path_for(out).exists()  # the same state file as before, for the full name and the compact one
    # A name without `.json` (`-o out.txt`): the state file is still `out.txt.fired.json`, as before.
    named = Trigger(command(), compact_window=5)
    named.look({"run_id": "r", "incidents": []}, tmp_path / "out.txt", rank=0)
    assert (tmp_path / "out.txt.fired.json").exists() and (tmp_path / "out.compact.txt").exists()
    assert not list(tmp_path.glob("out.compact*.fired.json"))

    # A compact file that cannot be written: said once, and the full file is handed over.
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "e.evidence.compact.json").mkdir()  # a directory where the file would go
    failing = Trigger(command(), compact_window=5, state_path=tmp_path / "b.fired.json")
    evidence = {"run_id": "r", "incidents": [{"kind": "non_finite", "step": 3, "layers": [], "gradients": []}]}
    for _ in range(2):
        failing.look(evidence, blocked / "e.evidence.json", rank=0)
    failing.close(wait=20)
    assert calls()[-1]["payload"]["evidence"] == str((blocked / "e.evidence.json").resolve())
    assert capsys.readouterr().out.count("could not be written") == 1

    # The command line: the option is set by the trigger alone, and said at the start.
    from tensorquill.trigger import from_args, add_arguments
    import argparse
    for given, expected in (([], 5), (["--compact"], None)):
        parser = argparse.ArgumentParser()
        parser.add_argument("--watch", action="store_true")
        parser.add_argument("--compact", action="store_true")
        parser.add_argument("--compact-window", type=int)
        add_arguments(parser)
        args = parser.parse_args(["--watch", "--webhook", "http://127.0.0.1:9/x", *given])
        args.compact_window = 5 if args.compact else args.compact_window
        assert from_args(args, parser).compact_window == expected
    printed = capsys.readouterr().out
    assert printed.count("hands over the compact evidence") == 1


# --- What fires -------------------------------------------------------------------------------


def test_an_anomaly_fires_only_with_the_option_and_an_observation_only_by_name(tmp_path, hook):
    command, calls = hook
    source = tmp_path / "w.jsonl"
    _write(source, _with(_with(_records(range(60)), 30, **SPIKE), 40, **THROTTLED))
    built = tensorquill.build_evidence(source)
    assert [(i["kind"], i["class"]) for i in built["incidents"]] == [
        ("grad_spike", "anomaly"), ("gpu_throttled", "observation"),
    ]

    def fire(**options):
        trigger = Trigger(command(), **options)
        watch(source, interval=0.0, polls=3, trigger=trigger)
        trigger.close(wait=20)
        _settle(trigger, lambda: all(d["command"]["status"] == "done" for d in trigger.deliveries))
        return [c["payload"]["kind"] for c in calls()]

    # By default neither fires, and nothing is written down as fired.
    assert fire() == []
    assert json.loads(state_path_for(tmp_path / "w.evidence.json").read_text())["deliveries"] == []
    # The anomaly with the option; the observation still not.
    assert fire(anomalies=True) == ["grad_spike"]
    assert fire(anomalies=True) == ["grad_spike"]  # and once
    # An observation fires only when asked for by name.
    assert fire(kinds=["gpu_throttled"]) == ["grad_spike", "gpu_throttled"]
    assert calls()[-1]["payload"]["class"] == "observation"


def test_a_change_of_run_state_never_fires_and_an_unknown_kind_only_by_name(tmp_path, hook, capsys):
    command, calls = hook
    path = tmp_path / "e.evidence.json"
    trigger = Trigger(command())
    evidence = {"run_id": "r", "incidents": [], "run_state": {"state": "healthy", "reasons": []}}
    for state in ("healthy", "degraded", "unknown", "unhealthy", "healthy"):
        trigger.look({**evidence, "run_state": {"state": state, "reasons": []}}, path, rank=0)
    assert trigger.deliveries == []

    # A kind from a newer server has no class here: said once, and not fired.
    newer = {**evidence, "incidents": [{"kind": "brand_new", "class": "invariant", "step": 7}]}
    trigger.look(newer, path, rank=0)
    trigger.look(newer, path, rank=0)
    assert trigger.deliveries == []
    printed = capsys.readouterr().out
    assert printed.count("incident kind 'brand_new' is not known to this version") == 1
    assert "--fire-kind brand_new" in printed
    named = Trigger(command(), kinds=["brand_new"])
    named.look(newer, path, rank=0)
    named.close(wait=20)
    (call,) = calls()
    assert (call["payload"]["kind"], call["payload"]["class"], call["payload"]["summary"]) == (
        "brand_new", None, "Incident at step 7: brand_new",
    )
    assert call["env"]["TENSORQUILL_INCIDENT_CLASS"] == ""


def test_in_the_all_ranks_view_a_shared_incident_fires_once_for_its_ranks(tmp_path, hook):
    command, calls = hook
    path = tmp_path / "r.all-ranks.evidence.json"

    def view(ranks, run_incidents=()):
        incident = {"kind": "non_finite", "class": "invariant", "step": 5, "layers": ["block.0"], "gradients": []}
        return {
            "view": "all_ranks", "run_id": "r", "run_state": {"state": "unhealthy", "reasons": []},
            "incident_groups": [{
                "id": 0, "kind": "non_finite", "class": "invariant", "step": 5, "ranks": ranks,
                "rank_ranges": "", "on_all_judged_ranks": True, "numbers_differ": False, "incident_rank": ranks[0],
                "incident": incident,
            }],
            "run_incidents": list(run_incidents),
        }

    trigger = Trigger(command())
    trigger.look(view([0, 1]), path)
    assert _settle(trigger, lambda: len(calls()) == 1)
    first = calls()[0]
    assert (first["payload"]["rank"], first["payload"]["ranks"], first["payload"]["new_ranks"]) == (None, [0, 1], [0, 1])
    assert first["payload"]["id"] == "r/ranks0-1/non_finite/5" and first["env"]["TENSORQUILL_INCIDENT_RANKS"] == "0-1"
    assert first["payload"]["summary"] == "NaN or Inf at step 5 in block.0"

    trigger.look(view([0, 1]), path)  # the same ranks on a later look
    # Two more ranks show it: once more, for those two, and not once a rank.
    trigger.look(view([0, 1, 2, 3]), path)
    assert _settle(trigger, lambda: len(calls()) == 2)
    second = calls()[1]["payload"]
    assert (second["ranks"], second["new_ranks"], second["id"]) == ([0, 1, 2, 3], [2, 3], "r/ranks2-3/non_finite/5")

    # Incidents of the whole run: an invariant fires with no rank. An observation does not, and
    # a stall at `warning` is one of those. (Before Nov 16, 2026 this block held that a stall
    # never fires unless named; the rule now is in the tests under "A stall" below.)
    hang = {"kind": "collective_hang", "class": "invariant", "step": 9}
    stall = {"kind": "job_stalled", "class": "observation", "step": 9, "ranks": [], "severity": "warning"}
    throttled = {"kind": "gpu_throttled", "class": "observation", "step": 9}
    trigger.look(view([0, 1, 2, 3], [stall, throttled, hang]), path)
    assert _settle(trigger, lambda: len(calls()) == 3)
    third = calls()[2]
    assert (third["payload"]["kind"], third["payload"]["ranks"], third["payload"]["id"]) == (
        "collective_hang", [], "r/run/collective_hang/9",
    )
    assert third["env"]["TENSORQUILL_INCIDENT_RANKS"] == ""
    restarted = Trigger(command())
    restarted.look(view([0, 1, 2, 3], [stall, throttled, hang]), path)
    restarted.close(wait=2)
    assert len(calls()) == 3 and len(restarted.deliveries) == 3

    # The same stall, critical now: it fires once for the run, with no rank, and the shared
    # incident and the invariant beside it do not fire again.
    critical = {**stall, "severity": "critical"}
    for _ in range(3):
        restarted.look(view([0, 1, 2, 3], [critical, throttled, hang]), path)
    assert _settle(restarted, lambda: len(calls()) == 4)
    fourth = calls()[3]
    assert (fourth["payload"]["kind"], fourth["payload"]["class"], fourth["payload"]["rank"], fourth["payload"]["ranks"]) == (
        "job_stalled", "observation", None, [],
    )
    assert fourth["payload"]["id"] == "r/run/job_stalled/9" and fourth["env"]["TENSORQUILL_INCIDENT_RANKS"] == ""
    again = Trigger(command())
    again.look(view([0, 1, 2, 3], [critical, throttled, hang]), path)
    again.close(wait=2)
    assert len(calls()) == 4 and len(again.deliveries) == 4


# --- A stall: fired with no option once the server calls it critical ---------------------------
#
# `job_stalled` and `rank_silent` are observations, read from the server's status. They fire
# by name from their warning, as any observation does, and with no option once their severity
# is `critical` (the plan's "Four open choices settled", item 4).


def _rank(rank, state, severity=None, last_step=40, phase="train"):
    return {"rank": rank, "state": state, "severity": severity, "last_step": last_step, "phase": phase,
            "stream": "dropped" if state == "lost" else "open", "seconds_since_progress": 400.0}


def _status(summary, *ranks, missing=()):
    """A `watchdog.json` as the server writes it, less what no reader here uses."""
    return {"schema_version": 1, "run_id": "r", "written_at_ms": 1, "server_running": True, "summary": summary,
            "silent_floor_seconds": 300.0, "critical_seconds": 900.0, "world_size": len(ranks) + len(missing),
            "missing_ranks": list(missing), "pauses": {}, "ranks": list(ranks)}


STALLED = {  # kind: the status at `warning`, and the same stall once a stopped rank is `critical`
    "job_stalled": [_status("stalled", _rank(0, "silent", severity), _rank(1, "silent", "warning", 38))
                    for severity in ("warning", "critical")],
    "rank_silent": [_status("degraded", _rank(0, "active", None, 90), _rank(1, "silent", severity))
                    for severity in ("warning", "critical")],
}


def _stall_calls(calls):
    return [(c["payload"]["kind"], c["payload"]["step"], c["payload"]["incident"]["severity"]) for c in calls()]


@pytest.mark.parametrize("kind", STALL_KINDS)
def test_a_stall_fires_with_no_option_once_it_is_critical_and_not_before(tmp_path, hook, kind, capsys):
    """Through the watcher, on a file that has stopped growing: only the status changes."""
    command, calls = hook
    warning, critical = STALLED[kind]
    source, status_file = tmp_path / "w.jsonl", tmp_path / "watchdog.json"
    _write(source, _records(range(41)))
    seen = []

    def watching(status, polls=4, **options):
        status_file.write_text(json.dumps(status))
        trigger = Trigger(command(), **options)
        watch(source, interval=0.02, polls=polls, watchdog=status_file, trigger=trigger,
              on_update=lambda evidence, path: seen.append([(i["kind"], i["severity"]) for i in evidence["incidents"]]))
        _settle(trigger, lambda: all(d["command"]["status"] == "done" for d in trigger.deliveries))
        return trigger

    # At `warning` the stall is listed and nothing fires, however often it is looked at.
    quiet = watching(warning)
    assert seen == [[(kind, "warning")]] and quiet.deliveries == [] and calls() == []
    state = state_path_for(tmp_path / "w.evidence.json")
    assert json.loads(state.read_text())["deliveries"] == []

    # The server calls it critical: the watcher writes the evidence again for that alone, and
    # the command starts once. The kind is an observation still, and the run state is not worse for it.
    fired = watching(critical)
    assert seen[1:] == [[(kind, "critical")]]
    (call,) = calls()
    payload = call["payload"]
    assert (payload["kind"], payload["class"], payload["step"], payload["rank"]) == (kind, "observation", 40, 0)
    assert payload["incident"]["severity"] == "critical" and payload["incident"]["assessment"] == "suspected_hang"
    assert payload["run_state"] == "unknown" and payload["id"] == f"r/rank0/{kind}/40"
    assert call["env"]["TENSORQUILL_INCIDENT_KIND"] == kind and call["env"]["TENSORQUILL_INCIDENT_CLASS"] == "observation"
    assert f"trigger: {kind} at step 40 (rank 0): started the command" in capsys.readouterr().out
    (delivery,) = json.loads(state.read_text())["deliveries"]
    assert (delivery["kind"], delivery["step"], delivery["ranks"], delivery["command"]["status"]) == (kind, 40, [0], "done")
    assert fired.deliveries == [delivery]

    # The stall goes on: the status is written again with a later clock, a restarted watcher
    # reads it, and one that names the kind does too. Nothing more starts.
    later = {**critical, "written_at_ms": 2, "ranks": [{**r, "seconds_since_progress": 2000.0} for r in critical["ranks"]]}
    for options in ({}, {}, {"kinds": [kind]}):
        assert watching(later, polls=3, **options).deliveries == [delivery]
    assert len(calls()) == 1


@pytest.mark.parametrize("kind", STALL_KINDS)
def test_a_stall_named_with_fire_kind_fires_at_warning_and_not_again_at_critical(tmp_path, hook, kind):
    command, calls = hook
    warning, critical = STALLED[kind]
    path = tmp_path / "e.evidence.json"

    def look(trigger, status, wanted):
        evidence = tensorquill.build_evidence(_records(range(41)), watchdog=status)
        assert [(i["kind"], i["class"]) for i in evidence["incidents"]] == [(kind, "observation")]
        trigger.look(evidence, path, rank=0)
        assert _settle(trigger, lambda: len(calls()) == wanted)
        trigger.close(wait=5)
        trigger.tick()

    named = Trigger(command(), kinds=[kind])
    look(named, warning, 1)
    assert _stall_calls(calls) == [(kind, 40, "warning")]
    for status in (critical, critical, warning):  # it becomes critical, stays so, and a restart at warning again
        look(named, status, 1)
    # A watcher started again without the name, when the stall is already critical: fired.json
    # holds the firing, so the default rule does not fire it a second time either.
    look(Trigger(command()), critical, 1)
    assert _stall_calls(calls) == [(kind, 40, "warning")] and len(json.loads(state_path_for(path).read_text())["deliveries"]) == 1


def test_only_a_stopped_rank_makes_a_stall_critical_so_a_declared_pause_cannot_fire(tmp_path, hook):
    """The server gives a severity only to a silent, unresponsive or lost rank (`run_status` in
    `watchdog.rs`), and an incident's severity is taken from its stopped ranks alone."""
    command, calls = hook
    trigger = Trigger(command())

    def fired(status, **incident):
        found = detect_watchdog_incidents(status)
        found = [{**i, **incident} for i in found]
        trigger.look({"run_id": "r", "incidents": found}, tmp_path / "e.evidence.json", rank=0)
        trigger.close(wait=5)
        return [i.get("severity") for i in found], len(trigger.deliveries)

    paused = _rank(0, "paused", None, 90, phase="eval")
    # Every rank in a declared pause: the run is active and there is no incident at all.
    assert fired(_status("active", paused, _rank(1, "paused", None, 90, phase="eval"))) == ([], 0)
    # A rank in a long evaluation beside one that went silent a moment ago: a warning.
    assert fired(_status("degraded", paused, _rank(1, "silent", "warning"))) == (["warning"], 0)
    # Even a status that called the paused rank critical, which the server never writes, would not.
    assert fired(_status("degraded", {**paused, "severity": "critical"}, _rank(1, "silent", "warning"))) == (["warning"], 0)
    # A rank that never connected has no severity: the incident stays a warning.
    assert fired(_status("degraded", _rank(0, "active", None, 90), missing=[1])) == (["warning"], 0)
    # An incident with no severity, or another word for it, does not fire.
    silent = _status("degraded", paused, _rank(1, "silent", "warning"))
    assert fired(silent, severity=None) == ([None], 0)
    assert fired(silent, severity="CRITICAL") == (["CRITICAL"], 0)
    assert calls() == []
    # And the rank beside the paused one that stays silent until it is critical does: the stall
    # is that rank's, which is in training and not in a declared pause.
    assert fired(_status("degraded", paused, _rank(1, "silent", "critical"))) == (["critical"], 1)
    # That was the run's stall: a later one, of another kind and step, does not fire again
    # (one firing for one stall), whatever its assessment.
    lost = _status("stalled", _rank(0, "lost", "critical", 55))
    assert [i["assessment"] for i in detect_watchdog_incidents(lost)] == ["lost_connection"]
    assert fired(lost) == (["critical"], 1)
    assert _settle(trigger, lambda: len(calls()) == 1)


def test_one_stall_fires_once_when_its_kind_or_step_changes(tmp_path, hook):
    """One firing for one stall (Nov 17, 2026): a stall is keyed by its run. Under the key of
    rank, kind and step this hang fired three times."""
    command, calls = hook
    path = tmp_path / "e.evidence.json"
    trigger = Trigger(command())
    looks = [
        # Rank 1 is silent at step 40 and critical while rank 0 goes on: `rank_silent`.
        _status("degraded", _rank(0, "active", None, 90), _rank(1, "silent", "critical", 40), _rank(2, "active", None, 90)),
        # Rank 2 falls silent further back, still at warning: the incident's step moves to it.
        _status("degraded", _rank(0, "active", None, 95), _rank(1, "silent", "critical", 40), _rank(2, "silent", "warning", 30)),
        # Then rank 0 stops too: the job is stalled, already critical on its first look.
        _status("stalled", _rank(0, "silent", "warning", 99), _rank(1, "silent", "critical", 40), _rank(2, "silent", "warning", 30)),
    ]
    for status in looks:
        trigger.look({"run_id": "r", "incidents": detect_watchdog_incidents(status)}, path, rank=0)
        assert _settle(trigger, lambda: len(calls()) == 1)
        trigger.close(wait=5)
        trigger.tick()
    time.sleep(0.3)
    assert _stall_calls(calls) == [("rank_silent", 40, "critical")]
    assert [d["kind"] for d in json.loads(state_path_for(path).read_text())["deliveries"]] == ["rank_silent"]
    # A restarted watcher on the stalled job: nothing either.
    again = Trigger(command())
    again.look({"run_id": "r", "incidents": detect_watchdog_incidents(looks[-1])}, path, rank=0)
    again.close(wait=1)
    assert len(calls()) == 1 and len(again.deliveries) == 1

    # The converse: `rank_silent` named fires at warning, and the `job_stalled` it becomes is
    # not fired at critical. In the all-ranks view as well (the stall is the run's).
    for name, evidence_of in (("one", lambda found: {"run_id": "q", "incidents": found}),
                              ("all", lambda found: {"run_id": "q", "view": "all_ranks", "incident_groups": [],
                                                     "run_incidents": found})):
        before = len(calls())
        named = Trigger(command(), kinds=["rank_silent"], state_path=tmp_path / f"{name}.fired.json")
        warning = _status("degraded", _rank(0, "active", None, 90), _rank(1, "silent", "warning", 40))
        stalled = _status("stalled", _rank(0, "silent", "critical", 90), _rank(1, "silent", "critical", 40))
        for status in (warning, stalled, stalled):
            named.look(evidence_of(detect_watchdog_incidents(status)), tmp_path / f"{name}.json", rank=0)
            assert _settle(named, lambda: len(calls()) == before + 1)
            named.close(wait=5)
        time.sleep(0.3)
        assert [(c["payload"]["kind"], c["payload"]["incident"]["severity"]) for c in calls()[before:]] == [
            ("rank_silent", "warning")]


def _ranked(rank, state, last_step, attempt=None, severity="critical"):
    entry = _rank(rank, state, severity if state in ("silent", "lost", "unresponsive") else None, last_step)
    return {**entry, **({"attempt": attempt} if attempt is not None else {})}


class _Looks:
    """A trigger fed made statuses, one look each, and the stall firings it made."""

    def __init__(self, tmp_path, command, calls, name="e", **options):
        self.path, self.command, self.calls, self.options = tmp_path / f"{name}.evidence.json", command, calls, options
        self.trigger = Trigger(command(), **options)

    def look(self, summary, *ranks, restart=False):
        if restart:  # the watcher started again: what it knows is the state file
            self.trigger = Trigger(self.command(), **self.options)
        found = detect_watchdog_incidents(_status(summary, *ranks))
        state = state_path_for(self.path)
        before = len(json.loads(state.read_text())["deliveries"]) if state.exists() else 0
        self.trigger.look({"run_id": "r", "incidents": found}, self.path, rank=0)
        self.trigger.close(wait=10)
        return len(self.trigger.deliveries) - before

    def fired(self):
        assert _settle(self.trigger, lambda: all(d["command"]["status"] == "done" for d in self.trigger.deliveries))
        return [(c["payload"]["kind"], c["payload"]["step"]) for c in self.calls()]


def test_a_stall_episode_ends_when_the_run_steps_past_it_and_the_next_stall_fires(tmp_path, hook):
    """Nov 17, 2026, the author's rule: one firing for one stall episode; progress re-arms it."""
    command, calls = hook
    run = _Looks(tmp_path, command, calls)
    assert run.look("stalled", _ranked(0, "silent", 99, 7), _ranked(1, "silent", 97, 7)) == 1
    for _ in range(3):  # the same stall, looked at again and by a restarted watcher
        assert run.look("stalled", _ranked(0, "silent", 99, 7), _ranked(1, "silent", 97, 7)) == 0
    assert run.look("stalled", _ranked(0, "silent", 99, 7), _ranked(1, "silent", 97, 7), restart=True) == 0
    # A lagging rank's late rows raise its own step, not the run's: the same episode.
    assert run.look("stalled", _ranked(0, "silent", 99, 7), _ranked(1, "silent", 98, 7)) == 0
    # The run steps past the step it stalled at, and stalls again at critical: a new episode.
    assert run.look("active", _ranked(0, "active", 150, 7), _ranked(1, "active", 150, 7)) == 0
    assert run.look("stalled", _ranked(0, "silent", 200, 7), _ranked(1, "silent", 200, 7)) == 1
    assert run.look("stalled", _ranked(0, "silent", 200, 7), _ranked(1, "silent", 200, 7), restart=True) == 0
    assert run.fired() == [("job_stalled", 99), ("job_stalled", 200)]
    ids = [d["id"] for d in run.trigger.deliveries]
    assert ids == ["r/rank0/job_stalled/99", "r/rank0/job_stalled/200"]
    # A stall at warning in between does not fire, and does not end the episode either.
    assert run.look("stalled", _ranked(0, "silent", 200, 7, "warning"), _ranked(1, "silent", 200, 7, "warning")) == 0


def test_a_new_attempt_re_arms_the_episode_and_a_missing_attempt_is_no_change(tmp_path, hook):
    """A job started again hangs before it passes the step it stalled at: the status's `attempt`
    has changed (the server keeps it from the first batch of the new attempt), so it fires again,
    at the same step, under an id of its own."""
    command, calls = hook
    run = _Looks(tmp_path, command, calls)
    assert run.look("stalled", _ranked(0, "lost", 99, 1700000000)) == 1
    assert run.look("stalled", _ranked(0, "lost", 99)) == 0  # a status that does not say: no change
    assert run.look("stalled", _ranked(0, "silent", 99, 1700000900)) == 1
    assert run.fired() == [("job_stalled", 99), ("job_stalled", 99)]
    assert [d["id"] for d in run.trigger.deliveries] == ["r/rank0/job_stalled/99", "r/rank0/job_stalled/99/2"]


def test_a_staggered_restart_does_not_fire_the_old_stall_of_a_rank_not_yet_started_again(tmp_path, hook):
    """A job of two ranks stalls and fires; it is started again, and rank 0's new attempt is seen a
    look before rank 1's. Rank 1, still stopped exactly as it was, does not fire again: a new
    attempt ends an episode only when no rank of it is still where it stopped. Once both are
    started again, a hang of the new attempts fires."""
    command, calls = hook
    run = _Looks(tmp_path, command, calls)
    assert run.look("stalled", _ranked(0, "silent", 99, 1), _ranked(1, "lost", 99, 1)) == 1
    starting = {**_ranked(0, "starting", 99, 2)}
    assert run.look("degraded", starting, _ranked(1, "lost", 99, 1)) == 0
    assert run.look("degraded", starting, _ranked(1, "lost", 99, 1)) == 0
    # Both started again, below the old step; then the new attempts hang.
    assert run.look("active", _ranked(0, "active", 99, 2), _ranked(1, "active", 99, 2)) == 0
    assert run.look("stalled", _ranked(0, "silent", 99, 2), _ranked(1, "silent", 99, 2)) == 1
    assert run.fired() == [("job_stalled", 99), ("job_stalled", 99)]


def test_a_silent_rank_beside_stepping_ranks_is_one_episode_and_a_rank_that_joins_it_too(tmp_path, hook):
    """`rank_silent` is judged by the stopped rank's own step: the ranks that go on stepping do
    not end it, however many looks. A rank that stops during the episode joins it."""
    command, calls = hook
    run = _Looks(tmp_path, command, calls)
    for newest in range(90, 300, 10):
        fired = run.look("degraded", _ranked(0, "active", newest), _ranked(1, "silent", 40), _ranked(2, "active", newest))
        assert fired == (1 if newest == 90 else 0)
    # Rank 2 falls silent too: it joins. Rank 1 then recovers while rank 2 is still critical:
    # rank 1 stored a step beyond the one it stalled at, so the episode is over, and rank 2's
    # stall fires, once.
    assert run.look("degraded", _ranked(0, "active", 300), _ranked(1, "silent", 40), _ranked(2, "silent", 290)) == 0
    assert run.look("degraded", _ranked(0, "active", 310), _ranked(1, "active", 310), _ranked(2, "silent", 290)) == 1
    # The job then stops with rank 2 still at 290: that episode, no second firing.
    assert run.look("stalled", _ranked(0, "silent", 320), _ranked(1, "silent", 320), _ranked(2, "silent", 290)) == 0
    # Rank 2 steps past 290 and the job stalls again: a new episode.
    assert run.look("stalled", _ranked(0, "silent", 400), _ranked(1, "silent", 400), _ranked(2, "silent", 400)) == 1
    assert run.fired() == [("rank_silent", 40), ("rank_silent", 290), ("job_stalled", 400)]

    # The case the author named: a silent rank that becomes a stalled job, with the silent rank
    # where it stopped, is one episode, however far the others had stepped meanwhile.
    other = _Looks(tmp_path, command, calls, name="other")
    before = len(calls())
    assert other.look("degraded", _ranked(0, "active", 90), _ranked(1, "silent", 40)) == 1
    assert other.look("degraded", _ranked(0, "active", 500), _ranked(1, "silent", 40)) == 0
    assert other.look("stalled", _ranked(0, "silent", 520), _ranked(1, "silent", 40)) == 0
    assert _settle(other.trigger, lambda: len(calls()) == before + 1)


def test_a_rank_that_recovered_ends_its_episode_and_another_ranks_later_stall_fires(tmp_path, hook):
    """Rank 1 stalls and fires, recovers and steps on (it is no longer in the stall); later rank 2
    stalls. Rank 1's progress, read from the stall's `other_ranks`, ended its episode: rank 2's
    stall fires. A rank of the episode that is only `reconnecting` (a server restart) has not
    stepped, and ends nothing. A stall that fired before any step was stored ends on the first."""
    command, calls = hook
    run = _Looks(tmp_path, command, calls)
    assert run.look("degraded", _ranked(0, "active", 90), _ranked(1, "silent", 40), _ranked(2, "active", 90)) == 1
    # Rank 2 stops while rank 1 is only reconnecting: it joins, nothing ends, nothing fires.
    assert run.look("degraded", _ranked(0, "active", 120), _ranked(1, "reconnecting", 40), _ranked(2, "silent", 110)) == 0
    # Rank 1 steps on: its episode is over, and rank 2's stall, critical, fires.
    assert run.look("degraded", _ranked(0, "active", 150), _ranked(1, "active", 150), _ranked(2, "silent", 110)) == 1
    assert run.look("degraded", _ranked(0, "active", 160), _ranked(1, "active", 160), _ranked(2, "silent", 110)) == 0
    assert run.fired() == [("rank_silent", 40), ("rank_silent", 110)]
    again = _Looks(tmp_path, command, calls, name="again")
    before = len(calls())
    assert again.look("degraded", _ranked(0, "active", 90), _ranked(1, "silent", 40), _ranked(2, "active", 90)) == 1
    assert again.look("degraded", _ranked(0, "active", 150), _ranked(1, "active", 150), _ranked(2, "active", 150)) == 0
    assert again.look("degraded", _ranked(0, "active", 300), _ranked(1, "active", 300), _ranked(2, "silent", 200)) == 1
    assert _settle(again.trigger, lambda: len(calls()) == before + 2)
    # Nothing stored when it fired (`last_step` null): the first stored step is progress.
    empty = _Looks(tmp_path, command, calls, name="empty")
    assert empty.look("stalled", _ranked(0, "silent", None)) == 1
    assert empty.look("stalled", _ranked(0, "silent", None)) == 0
    assert empty.look("stalled", _ranked(0, "silent", 9)) == 1


def test_a_stall_whose_firing_is_removed_from_the_state_file_fires_again(tmp_path, hook, capsys):
    """What the "given up" line tells the user to do: remove the incident from the state file."""
    command, calls = hook
    run = _Looks(tmp_path, command, calls)
    assert run.look("stalled", _ranked(0, "silent", 99)) == 1
    state = state_path_for(run.path)
    document = json.loads(state.read_text())
    assert document["episode"]["delivery"] == "r/rank0/job_stalled/99"
    state.write_text(json.dumps({**document, "deliveries": []}))
    assert run.look("stalled", _ranked(0, "silent", 99), restart=True) == 1
    # An episode of a shape this version does not read: said, and a stall may fire; other kinds still fire.
    broken = json.loads(state.read_text())
    state.write_text(json.dumps({**broken, "deliveries": [], "episode": {"delivery": "x", "ranks": [1, 2]}}))
    assert run.look("stalled", _ranked(0, "silent", 99), restart=True) == 1
    assert "cannot be read; a stall may fire again" in capsys.readouterr().out


def test_a_stall_named_at_warning_does_not_fire_again_at_critical_in_its_episode(tmp_path, hook):
    command, calls = hook
    run = _Looks(tmp_path, command, calls, kinds=["rank_silent", "job_stalled"])
    assert run.look("degraded", _ranked(0, "active", 90), _ranked(1, "silent", 40, severity="warning")) == 1
    assert run.look("degraded", _ranked(0, "active", 95), _ranked(1, "silent", 40)) == 0
    assert run.look("stalled", _ranked(0, "silent", 95), _ranked(1, "silent", 40)) == 0
    assert run.fired() == [("rank_silent", 40)]


def test_state_files_of_both_earlier_keys_load_and_their_stall_is_the_open_episode(tmp_path, hook):
    """A file keyed by rank, kind and step (before Nov 17, 2026) and one keyed by the run (Nov 17)
    hold no episode: the first stall seen is taken as theirs, without firing; progress re-arms it.
    Of two firings with tries left, only the episode's is sent."""
    command, calls = hook
    for name, ids in (("per-rank", ["r/rank0/rank_silent/40", "r/rank0/rank_silent/30"]),
                      ("per-run", ["r/rank0/rank_silent/40"])):
        run = _Looks(tmp_path, command, calls, name=name)
        before = len(calls())
        deliveries = [{"id": i, "kind": "rank_silent", "step": int(i.rsplit("/", 1)[1]), "ranks": [0],
                       "first_seen_ms": 1, "command": {"status": "failed", "attempts": 1}} for i in ids]
        state_path_for(run.path).write_text(json.dumps({"version": 1, "evidence": str(run.path), "deliveries": deliveries}))
        run.trigger = Trigger(command(), retry_seconds=0.0)
        assert run.look("stalled", _ranked(0, "silent", 99), _ranked(1, "silent", 30)) == 0
        _settle(run.trigger, lambda: all(d["command"]["status"] in ("done", "gave_up") for d in run.trigger.deliveries[-1:]), 5)
        # The retry left is made once, for the stall as it reads now, under the last firing's id;
        # an earlier firing's is not.
        sent = [c["payload"]["id"] for c in calls()[before:]]
        assert sent == [ids[-1]], sent
        assert run.look("stalled", _ranked(0, "silent", 99), _ranked(1, "silent", 30)) == 0
        assert run.look("stalled", _ranked(0, "silent", 150), _ranked(1, "silent", 150)) == 1
        assert json.loads(state_path_for(run.path).read_text())["episode"]["run_step"] == 150


def test_a_state_file_of_the_earlier_key_still_loads_and_its_stall_does_not_fire_again(tmp_path, hook):
    """A `fired.json` written before Nov 17, 2026 holds a stall under (rank, kind, step); read
    now, it is the run's stall, and the stall it becomes fires nothing."""
    command, calls = hook
    path = tmp_path / "e.evidence.json"
    written = {"version": 1, "evidence": str(path), "deliveries": [
        {"id": "r/rank0/rank_silent/40", "kind": "rank_silent", "step": 40, "ranks": [0], "first_seen_ms": 1,
         "command": {"status": "done", "attempts": 1, "exit_status": 0, "at_ms": 1}},
        {"id": "r/rank0/non_finite/12", "kind": "non_finite", "step": 12, "ranks": [0], "first_seen_ms": 1,
         "command": {"status": "done", "attempts": 1, "exit_status": 0, "at_ms": 1}},
    ]}
    state_path_for(path).write_text(json.dumps(written))
    trigger = Trigger(command())
    stalled = _status("stalled", _rank(0, "silent", "critical", 90), _rank(1, "silent", "critical", 40))
    evidence = {"run_id": "r", "incidents": [
        *detect_watchdog_incidents(stalled), {"kind": "non_finite", "step": 12, "layers": [], "gradients": []},
        {"kind": "non_finite", "step": 14, "layers": [], "gradients": []},
    ]}
    trigger.look(evidence, path, rank=0)
    assert _settle(trigger, lambda: len(calls()) == 1)
    trigger.close(wait=5)
    # Only the NaN at a step that had not fired: the old key holds for every other kind.
    assert [(c["payload"]["kind"], c["payload"]["step"]) for c in calls()] == [("non_finite", 14)]
    assert [d["id"] for d in trigger.deliveries] == ["r/rank0/rank_silent/40", "r/rank0/non_finite/12", "r/rank0/non_finite/14"]


def test_a_state_file_given_by_name_is_read_before_the_first_tick_can_write_it(tmp_path, hook):
    """DET-01 (Nov 18, 2026): both watchers call `tick()` before their first `look()`, and a
    trigger given `state_path=` used to save its empty state at that tick, before reading the
    file, so a restarted watcher fired again for every incident and stall episode it listed.
    The file is read when the trigger is made. The command line passes no `state_path` (it is
    taken from the evidence at the first look), which the last lines hold unchanged."""
    command, calls = hook
    source, status_file = tmp_path / "w.jsonl", tmp_path / "watchdog.json"
    _write(source, _with(_records(range(41)), 30, **NAN))
    status_file.write_text(json.dumps(STALLED["job_stalled"][1]))
    state = tmp_path / "kept" / "mine.fired.json"
    first = Trigger(command(), state_path=state)
    watch(source, interval=0.02, polls=3, watchdog=status_file, trigger=first)
    assert _settle(first, lambda: len(calls()) == 2 and all(
        d["command"]["status"] == "done" for d in first.deliveries))
    first.close(wait=20)
    assert sorted(c["payload"]["kind"] for c in calls()) == ["job_stalled", "non_finite"]
    saved = state.read_text()
    assert json.loads(saved)["episode"]["kind"] == "job_stalled" and len(json.loads(saved)["deliveries"]) == 2

    # A restarted watcher given the same file: nothing fires again, and the file is as it was.
    again = Trigger(command(), state_path=state)
    watch(source, interval=0.02, polls=3, watchdog=status_file, trigger=again)
    again.close(wait=5)
    assert len(calls()) == 2 and state.read_text() == saved

    # Read when the trigger is made: a tick or a close before any look writes nothing over it.
    early = Trigger(command(), state_path=state)
    assert early.deliveries == json.loads(saved)["deliveries"]
    early.tick()
    early.close()
    assert state.read_text() == saved

    # With no `state_path`, as the command line makes it: nothing is read or written before a look.
    plain = Trigger(command())
    plain.tick()
    plain.close()
    assert plain.deliveries == [] and not state_path_for(tmp_path / "w.evidence.json").exists()


def test_a_critical_stall_in_a_status_that_is_no_longer_written_still_fires(tmp_path, hook):
    """Pinned as what is, not as what must be: a server killed while a run was stalled leaves its
    last `watchdog.json`, and a watcher started later fires once for the stall it lists. The
    evidence says the status is old; the incident is listed all the same."""
    command, calls = hook
    old = {**STALLED["job_stalled"][1], "written_at_ms": int((time.time() - 3600) * 1000), "refresh_seconds": 30.0}
    evidence = tensorquill.build_evidence(_records(range(41)), watchdog=old)
    assert evidence["run_state"]["state"] == "unknown"
    assert any(reason["code"] == "status_old" for reason in evidence["run_state"]["reasons"]), evidence["run_state"]
    trigger = Trigger(command())
    trigger.look(evidence, tmp_path / "e.evidence.json", rank=0)
    assert _settle(trigger, lambda: len(calls()) == 1)
    assert _stall_calls(calls) == [("job_stalled", 40, "critical")]


def test_a_failing_command_for_a_critical_stall_is_retried_as_any_other(tmp_path, hook):
    command, calls = hook
    source, status_file = tmp_path / "w.jsonl", tmp_path / "watchdog.json"
    _write(source, _records(range(41)))
    status_file.write_text(json.dumps(STALLED["job_stalled"][1]))
    flaky = Trigger(command(tmp_path / "failed-once"), retry_seconds=0.0)
    watch(source, interval=0.05, polls=10, watchdog=status_file, trigger=flaky)  # one look writes, the rest retry
    assert _settle(flaky, lambda: flaky.deliveries[0]["command"]["status"] == "done")
    assert [c["payload"]["attempt"] for c in calls()] == [1, 2] and {c["payload"]["id"] for c in calls()} == {"r/rank0/job_stalled/40"}
    assert json.loads(state_path_for(tmp_path / "w.evidence.json").read_text())["deliveries"] == flaky.deliveries
    # A webhook is counted apart for a stall as for any incident: given up after its tries.
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        port = taken.getsockname()[1]
    hooked = Trigger(webhook=f"http://127.0.0.1:{port}/hook", attempts=2, retry_seconds=0.0, webhook_timeout=2.0,
                     state_path=tmp_path / "hooked.fired.json")
    watch(source, tmp_path / "hooked.json", interval=0.05, polls=6, watchdog=status_file, trigger=hooked)
    (delivery,) = hooked.deliveries
    assert (delivery["kind"], delivery["webhook"]["status"], delivery["webhook"]["attempts"]) == ("job_stalled", "gave_up", 2)


# --- Failing and slow targets -----------------------------------------------------------------


def test_a_failing_command_is_retried_a_bounded_number_of_times_and_the_watcher_goes_on(tmp_path, hook, capsys):
    command, calls = hook
    source = tmp_path / "w.jsonl"
    _write(source, _with(_records(range(40)), 30, **NAN))
    seen = []
    trigger = Trigger(command(3), attempts=3, retry_seconds=0.0)
    # The file has stopped growing, as a run that died at its bad step leaves it: the retries
    # happen on looks that write nothing.
    out = watch(source, interval=0.05, polls=30, on_update=lambda e, p: seen.append(p), trigger=trigger)
    assert out == tmp_path / "w.evidence.json" and len(seen) == 1  # every look was made, one wrote
    assert _settle(trigger, lambda: trigger.deliveries[0]["command"]["status"] == "gave_up")
    assert [c["payload"]["attempt"] for c in calls()] == [1, 2, 3]
    assert {c["payload"]["id"] for c in calls()} == {"r/rank0/non_finite/30"}
    entry = trigger.deliveries[0]["command"]
    assert entry == {"status": "gave_up", "attempts": 3, "at_ms": entry["at_ms"], "exit_status": 3,
                     "error": "the command exited with status 3"}
    printed = capsys.readouterr().out
    assert "the command exited with status 3 (try 1 of 3); tried again on a later look" in printed
    assert "the command exited with status 3 (try 3 of 3); given up" in printed
    for _ in range(3):
        trigger.tick()
    # Given up for good, for a restarted watcher too.
    again = Trigger(command(3), attempts=3, retry_seconds=0.0)
    watch(source, interval=0.0, polls=3, trigger=again)
    again.close(wait=2)
    assert len(calls()) == 3

    # A command that fails once and then works: two tries, and done.
    other = tmp_path / "other.jsonl"
    _write(other, _with(_records(range(40)), 12, **NAN))
    flaky = Trigger(command(tmp_path / "failed-once"), retry_seconds=0.0)
    watch(other, interval=0.05, polls=10, trigger=flaky)
    assert _settle(flaky, lambda: flaky.deliveries[0]["command"]["status"] == "done")
    assert [c["payload"]["attempt"] for c in calls()[3:]] == [1, 2] and flaky.deliveries[0]["command"]["attempts"] == 2

    # A command the shell cannot find fails the same way, and nothing is raised.
    missing = Trigger("tensorquill-no-such-command-anywhere", attempts=1)
    watch(other, tmp_path / "missing.json", interval=0.0, polls=2, trigger=missing)
    assert _settle(missing, lambda: missing.deliveries[0]["command"]["status"] == "gave_up")
    assert missing.deliveries[0]["command"]["exit_status"] == 127


def test_a_slow_command_does_not_hold_the_watcher_and_one_runs_at_a_time(tmp_path):
    path = tmp_path / "e.evidence.json"
    evidence = {"run_id": "r", "incidents": [
        {"kind": "non_finite", "step": 3, "layers": ["a"], "gradients": [], "names": ["p" * 100] * 4000},
        {"kind": "repeated_steps", "step": 8},
    ]}
    # It never reads its input, which is larger than a pipe holds, and runs for a minute.
    trigger = Trigger("sleep 60", command_wait=3600.0)
    started = time.monotonic()
    try:
        for _ in range(5):
            trigger.look(evidence, path, rank=0)
            trigger.tick()
        assert time.monotonic() - started < 10  # five looks, none waited for the command
        first, second = trigger.deliveries
        assert first["command"]["status"] == "started" and second["command"] == {"status": "pending", "attempts": 0}
        # Started is fired: a watcher restarted while it runs does not start it again.
        restarted = Trigger("sleep 60", command_wait=0.0)
        try:
            restarted.look(evidence, path, rank=0)
            assert [d["command"]["status"] for d in restarted.deliveries] == ["started", "started"]
            assert restarted.deliveries[0]["command"]["pid"] == first["command"]["pid"]  # the first one's, untouched
            assert restarted.deliveries[1]["command"]["attempts"] == 1  # the one that was waiting
        finally:
            restarted.close(kill=True)
    finally:
        pid = trigger.deliveries[0]["command"]["pid"]
        trigger.close(kill=True)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # `close(kill=True)` is the test's own cleanup: the watcher never stops a command

    # After `command_wait` the next incident's command is no longer held back.
    state = tmp_path / "other.fired.json"
    impatient = Trigger("sleep 60", command_wait=0.0, state_path=state)
    try:
        impatient.look(evidence, path, rank=0)
        time.sleep(0.05)
        impatient.tick()
        assert [d["command"]["status"] for d in impatient.deliveries] == ["started", "started"]
        assert json.loads(state.read_text())["deliveries"] == impatient.deliveries
    finally:
        impatient.close(kill=True)


class _Receiver:
    """A local HTTP server that keeps what is posted to it and answers as told."""

    def __init__(self, answers=()):
        self.requests: list[dict] = []
        self.answers = list(answers)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                status, location = (outer.answers.pop(0) if outer.answers else (200, None))
                self.send_response(status)
                if location:
                    self.send_header("Location", location)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_GET = do_POST

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/hooks/tq?token=secret"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def receiver():
    made = []

    def make(answers=()):
        made.append(_Receiver(answers))
        return made[-1]

    yield make
    for one in made:
        one.stop()


def test_the_webhook_posts_the_incident_as_json_once(tmp_path, receiver, capsys):
    hooked = receiver()
    source = tmp_path / "w.jsonl"
    # A NaN in the incident's logged context must not make the body something no parser reads.
    _write(source, _with(_records(range(40), scalars={"loss": 2.0}), 30, grad_norm="nan", scalars={"loss": float("nan")}))
    trigger = Trigger(webhook=hooked.url)
    out = watch(source, interval=0.0, polls=4, trigger=trigger, compact_window=5)
    (request,) = hooked.requests
    assert request["path"] == "/hooks/tq?token=secret"
    headers = {k.lower(): v for k, v in request["headers"].items()}
    assert headers["content-type"] == "application/json" and headers["x-tensorquill-event"] == "incident"
    assert headers["x-tensorquill-delivery"] == "r/rank0/non_finite/30"

    def strict(text):  # refuses NaN and Infinity, as other languages' parsers do
        return json.loads(text, parse_constant=lambda name: pytest.fail(f"{name} in the body"))

    body = strict(request["body"])
    assert sorted(body) == sorted([
        "event", "version", "id", "attempt", "sent_at_ms", "run_id", "evidence", "run_state", "kind", "class", "step",
        "rank", "ranks", "new_ranks", "assessment", "summary", "incident",
    ])
    assert body["assessment"] is None  # a stall's label; an invariant has none
    assert (body["event"], body["version"], body["id"], body["attempt"]) == ("incident", 1, "r/rank0/non_finite/30", 1)
    assert (body["run_id"], body["kind"], body["class"], body["step"]) == ("r", "non_finite", "invariant", 30)
    assert (body["rank"], body["ranks"], body["new_ranks"], body["run_state"]) == (0, [0], [0], "unhealthy")
    assert body["evidence"] == str(out.resolve()) and body["summary"] == "NaN or Inf at step 30 in <total>"
    assert abs(body["sent_at_ms"] / 1000 - time.time()) < 60
    written = json.loads(out.read_text())["incidents"][0]
    assert body["incident"]["gradients"] == written["gradients"] == ["<total>"]
    # NaN in the evidence, and in the body the string the capture and the evidence file write (lane U7,
    # Nov 19, 2026; it was null). Run through the Rust detectors (test_watch_rust.py), the report
    # already has null there, as the server writes a NaN.
    assert body["incident"]["scalars"]["loss"]["value"] in ("nan", None)
    delivery = trigger.deliveries[0]["webhook"]
    assert (delivery["status"], delivery["attempts"], delivery["http_status"]) == ("delivered", 1, 200)
    printed = capsys.readouterr().out
    assert f"posted to the webhook at 127.0.0.1:{hooked.server.server_port} (HTTP 200)" in printed
    assert "secret" not in printed and "secret" not in state_path_for(out).read_text()  # the URL is not repeated

    # Later looks and a restart post nothing more.
    watch(source, interval=0.0, polls=3, trigger=Trigger(webhook=hooked.url))
    assert len(hooked.requests) == 1


def test_a_webhook_that_answers_badly_is_retried_and_a_redirect_is_not_followed(tmp_path, receiver, capsys):
    elsewhere = receiver()
    hooked = receiver([(500, None), (302, elsewhere.url), (200, None)])
    path = tmp_path / "e.evidence.json"
    evidence = {"run_id": "r", "incidents": [{"kind": "non_finite", "step": 3, "layers": ["a"], "gradients": []}]}
    trigger = Trigger(webhook=hooked.url, retry_seconds=0.0)
    trigger.look(evidence, path, rank=2)
    assert trigger.deliveries[0]["webhook"]["status"] == "failed"
    trigger.tick()
    trigger.tick()
    trigger.tick()
    assert [json.loads(r["body"])["attempt"] for r in hooked.requests] == [1, 2, 3]
    ids = {v for r in hooked.requests for k, v in r["headers"].items() if k.lower() == "x-tensorquill-delivery"}
    assert ids == {"r/rank2/non_finite/3"}  # the same id on every try
    assert elsewhere.requests == []  # the redirect led to a host the user did not name
    entry = trigger.deliveries[0]["webhook"]
    assert (entry["status"], entry["attempts"]) == ("delivered", 3) and "error" not in entry
    printed = capsys.readouterr().out
    assert "answered HTTP 500 (try 1 of 4)" in printed
    assert "answered HTTP 302, a redirect, which is not followed (try 2 of 4)" in printed


def test_an_unreachable_webhook_leaves_the_watcher_running_and_the_command_fired_once(tmp_path, hook, capsys):
    command, calls = hook
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        port = taken.getsockname()[1]  # closed again on leaving: nothing listens there
    source = tmp_path / "w.jsonl"
    _write(source, _with(_records(range(40)), 30, **NAN) + _records([34]))  # two invariants: a step written twice
    assert [i["kind"] for i in tensorquill.build_evidence(source)["incidents"]] == ["repeated_steps", "non_finite"]
    seen = []
    trigger = Trigger(command(), f"http://127.0.0.1:{port}/hook", attempts=2, retry_seconds=0.0, webhook_timeout=2.0)
    out = watch(source, interval=0.05, polls=30, on_update=lambda e, p: seen.append(p), trigger=trigger)
    assert out is not None and len(seen) == 1 and json.loads(out.read_text())["run_state"]["state"] == "unhealthy"
    assert _settle(trigger, lambda: all(
        d["webhook"]["status"] == "gave_up" and d["command"]["status"] == "done" for d in trigger.deliveries
    ))
    # The webhook was tried twice for each incident and given up; the command ran once for each:
    # a failing webhook does not start the command again.
    assert [d["webhook"]["attempts"] for d in trigger.deliveries] == [2, 2]
    assert sorted(c["payload"]["kind"] for c in calls()) == ["non_finite", "repeated_steps"]
    printed = capsys.readouterr().out
    assert f"the webhook at 127.0.0.1:{port} could not be reached" in printed and "given up" in printed

    # A host that accepts and never answers costs one timeout a look, however many incidents wait.
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(8)
        slow = Trigger(webhook=f"http://127.0.0.1:{silent.getsockname()[1]}/", webhook_timeout=0.3,
                       state_path=tmp_path / "slow.fired.json")
        started = time.monotonic()
        slow.look(json.loads(out.read_text()), out, rank=0)
        assert 0.25 < time.monotonic() - started < 3.0
        assert [d["webhook"]["attempts"] for d in slow.deliveries] == [1, 0]
        assert "timed out" in capsys.readouterr().out


# --- The state file, and what is refused ------------------------------------------------------


def test_the_state_file_and_an_incident_that_a_resume_rolled_back(tmp_path, hook, receiver, capsys):
    command, calls = hook
    path = tmp_path / "e.evidence.json"
    assert state_path_for("a/run.evidence.compact.json") == Path("a/run.evidence.fired.json")
    assert state_path_for("named.json") == Path("named.fired.json") and state_path_for("plain") == Path("plain.fired.json")
    evidence = {"run_id": "r", "incidents": [{"kind": "non_finite", "step": 3, "layers": ["a"], "gradients": []}]}

    # A state file this version cannot read: said, and the incidents listed fire again.
    state_path_for(path).write_text("{not json")
    trigger = Trigger(command())
    trigger.look(evidence, path, rank=0)
    assert _settle(trigger, lambda: len(calls()) == 1)
    assert "is not a state file this version reads" in capsys.readouterr().out
    assert json.loads(state_path_for(path).read_text())["version"] == 1

    # A webhook added on a restart is not sent the incidents that fired before it was there.
    later = Trigger(command(), "http://127.0.0.1:9/never")
    later.look(evidence, path, rank=0)
    assert "webhook" not in later.deliveries[0] and len(calls()) == 1

    # A firing still waiting when a resumed run rolls its incident back sends nothing.
    refusing = receiver([(500, None)])
    waiting = Trigger(webhook=refusing.url, retry_seconds=0.0, state_path=tmp_path / "w.fired.json")
    waiting.look(evidence, path, rank=1)
    assert waiting.deliveries[0]["webhook"] == {"status": "failed", "attempts": 1, "error": waiting.deliveries[0]["webhook"]["error"]}
    waiting.look({"run_id": "r", "incidents": []}, path, rank=1)
    for _ in range(3):
        waiting.tick()
    assert len(refusing.requests) == 1 and waiting.deliveries[0]["webhook"]["attempts"] == 1
    waiting.look(evidence, path, rank=1)  # listed again: the try that was waiting is made
    assert len(refusing.requests) == 2 and waiting.deliveries[0]["webhook"]["status"] == "delivered"

    # A trigger that breaks inside says so and does not raise into the watcher.
    broken = Trigger(command(), state_path=tmp_path / "b.fired.json")
    broken._collect = lambda *args: 1 / 0
    broken.look(evidence, path, rank=0)
    broken.tick()
    assert "trigger: ZeroDivisionError: division by zero; the watcher goes on" in capsys.readouterr().out


def test_a_trigger_needs_a_target_and_the_command_line_refuses_what_it_would_ignore(tmp_path, monkeypatch, capsys):
    with pytest.raises(ValueError, match="needs a command"):
        Trigger()
    for url in ("ftp://example.com/x", "example.com/hook", "file:///etc/passwd", "http://"):
        with pytest.raises(ValueError, match="http:// or https://") as error:
            Trigger(webhook=url)
        assert url == "http://" or url not in str(error.value)  # a webhook's URL is often its secret
    with pytest.raises(ValueError, match="one string"):
        Trigger(command="   ")

    source = tmp_path / "run.jsonl"
    _write(source, _records(range(5)))
    for options, message in (
        (["--on-incident", "true"], "they need --watch"),
        (["--webhook", "http://127.0.0.1:9/x"], "they need --watch"),
        (["--watch", "--fire-anomalies"], "give one of those"),
        (["--watch", "--fire-kind", "job_stalled"], "give one of those"),
        (["--watch", "--webhook", "ftp://example.com"], "http:// or https://"),
        (["--rank", "all", "--on-incident", "true"], "they need --watch"),
    ):
        monkeypatch.setattr(sys, "argv", ["tensorquill", str(source), *options])
        with pytest.raises(SystemExit):
            main()
        assert message in capsys.readouterr().err
    assert not list(tmp_path.glob("*.evidence*"))  # refused before anything was written or sent


# --- A server's store: one rank's watcher and the all-ranks view, through the real server -------


def test_watchers_of_a_servers_store_fire_once_for_one_rank_and_once_for_all_ranks(tmp_path, hook):
    """Needs `pyarrow` and the server binary, as `test_watch_all_ranks.py` does."""
    pytest.importorskip("pyarrow.flight")
    import test_watch_all_ranks as store

    if store.BINARY is None:
        pytest.skip("tensorquill-server is not built")
    command, calls = hook
    run = store.Run(tmp_path, *store.FAST)
    try:
        for rank in range(3):  # the same NaN on every rank: one incident group
            run.send(rank, store.steady(60, nan_at=(40, "block.0")))

        # Every rank at once: one firing for the three ranks, not three.
        trigger = Trigger(command())
        out = watch(run.folder, tmp_path / "all.json", interval=0.05, polls=3, rank="all", trigger=trigger,
                    compact_window=5)
        assert _settle(trigger, lambda: len(calls()) == 1)
        written = json.loads(out.read_text())
        assert written["view"] == "all_ranks" and written["compact"]["samples"]["total"] == 0  # marked, nothing to leave out
        (group,) = written["incident_groups"]
        payload = calls()[0]["payload"]
        assert (payload["kind"], payload["step"], payload["rank"], payload["ranks"]) == ("non_finite", 40, None, [0, 1, 2])
        assert payload["incident"] == group["incident"] and payload["evidence"] == str(out.resolve())
        assert calls()[0]["env"]["TENSORQUILL_INCIDENT_RANKS"] == "0-2"
        assert (tmp_path / "all.fired.json").exists()
        # A restart of that watcher, with more steps stored meanwhile: nothing more.
        run.send(1, [store.record(step, 1, grad_norm=store.wobble(step), lr=[1e-3]) for step in range(60, 110)])
        again = Trigger(command())
        watch(run.folder, tmp_path / "all.json", interval=0.05, polls=3, rank="all", trigger=again)
        again.close(wait=2)
        assert len(calls()) == 1

        # One rank's watcher, from the server's report for that rank: its own evidence file, its
        # own state, and the rank in what the command gets.
        one = Trigger(command())
        single = watch(run.folder, tmp_path / "one.json", interval=0.05, polls=3, rank=1, trigger=one)
        assert _settle(one, lambda: len(calls()) == 2)
        assert json.loads(single.read_text())["detection"]["source"] == "server"
        payload = calls()[1]["payload"]
        assert (payload["id"], payload["rank"], payload["ranks"]) == (f"{store.RUN}/rank1/non_finite/40", 1, [1])
        watch(run.folder, tmp_path / "one.json", interval=0.05, polls=2, rank=1, trigger=Trigger(command()))
        assert len(calls()) == 2
    finally:
        for server in store.STARTED:
            if server.process.poll() is None:
                server.process.kill()
                server.process.wait(20)
            if not server.process.stdout.closed:
                server.process.stdout.close()
        store.STARTED.clear()


# --- A stream that goes quiet altogether, and a real server -------------------------------------


def test_ranks_that_send_nothing_at_all_still_reach_critical_and_fire_once(tmp_path, hook):
    """Two streams stay open and nothing arrives on them, not a step and not a heartbeat: the
    server's own timer takes the stall from `warning` to `critical`, and the all-ranks watcher
    fires once, then. Needs `pyarrow` and the server binary, as `test_watch_all_ranks.py` does."""
    pytest.importorskip("pyarrow.flight")
    import test_watch_all_ranks as store
    from test_server_reports import QUICK_STATUS

    if store.BINARY is None:
        pytest.skip("tensorquill-server is not built")
    command, calls = hook
    # Silent after 1 s, critical after 4 s; nothing but the server's timer moves the status.
    run = store.Run(tmp_path, *store.FAST, *QUICK_STATUS, "--silent-seconds", "1", "--silent-factor", "0",
                    "--critical-seconds", "4")
    senders = []
    try:
        senders = [run.keep_open(rank, store.steady(60)) for rank in range(2)]
        trigger = Trigger(command())
        seen = []

        def on_update(evidence, path):
            stalls = [i["severity"] for i in evidence["run_incidents"] if i["kind"] == "job_stalled"]
            seen.append((stalls, len(trigger.deliveries), [r["stream"].get("heartbeat_age_seconds") for r in evidence["ranks"]]))

        watch(run.folder, tmp_path / "all.json", interval=0.2, polls=45, rank="all", trigger=trigger, on_update=on_update)
        assert _settle(trigger, lambda: len(calls()) == 1)
    finally:
        for sender in senders:
            sender.close(0.0)
        for server in store.STARTED:
            if server.process.poll() is None:
                server.process.kill()
                server.process.wait(20)
            if not server.process.stdout.closed:
                server.process.stdout.close()
        store.STARTED.clear()
    # What was written, in order: healthy, the stall at warning, the stall at critical (each as
    # often as something else in the status changed, such as the second rank's own severity).
    # Nothing had fired when a look without the stall or with it at warning was written, nor when
    # it was first written at critical; and no heartbeat was ever sent.
    order = [stalls for n, (stalls, _, _) in enumerate(seen) if n == 0 or stalls != seen[n - 1][0]]
    assert order == [[], ["warning"], ["critical"]], seen
    assert all(fired == 0 for stalls, fired, _ in seen if stalls != ["critical"]), seen
    assert next(fired for stalls, fired, _ in seen if stalls == ["critical"]) == 0
    assert all(age == [None, None] for _, _, age in seen)
    assert _stall_calls(calls) == [("job_stalled", 59, "critical")]
    (delivery,) = trigger.deliveries
    assert (delivery["ranks"], delivery["command"]["attempts"]) == ([None], 1)


# --- A real hang: a rank that stops stepping while its process lives, and a real server ---------

# One rank as a launcher starts it. It steps, then stops stepping without closing: its process,
# its stream and its heartbeats go on, as those of a job hung in training do.
_STALLING_RANK = """
import os, time
import torch, tensorquill
from torch import nn

model = nn.Linear(8, 1)
# `every=10`: steps leave the device ten at a time; the flush after the loop puts every step in
# the file before the stop (a full ring's newest step otherwise waits for the next one).
session = tensorquill.init(model, run_id=os.environ["TQ_RUN_ID"], every=10, heartbeat_seconds=0.5, stack_seconds=1.0)
session._sink.sender._batch_seconds, session._sink.sender._backoff_cap = 0.25, 0.5
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for _ in range(int(os.environ["TQ_STEPS"])):
    optimizer.zero_grad()
    loss = model(torch.randn(16, 8)).pow(2).mean()
    loss.backward()
    session.log({"train/loss": loss})
    optimizer.step()
    time.sleep(0.01)
session.flush()
print("stopped stepping", flush=True)

def stuck_here():
    while True:
        time.sleep(0.2)

stuck_here()
"""
# Silent after 2 s without a step, critical after 10 s; the status is looked at every 0.2 s and
# an unchanged one is written again every 0.6 s.
_STALL_SERVER = ("--silent-seconds", "2", "--silent-factor", "0", "--critical-seconds", "10",
                 "--watchdog-seconds", "0.2", "--flush-seconds", "0.3")


def test_a_real_rank_that_stops_stepping_starts_the_command_once_when_the_server_calls_it_critical(tmp_path, hook):
    """`python -m tensorquill <store>/runs/<run> --watch --on-incident COMMAND`, with no
    `--fire-kind`, beside a real `tensorquill-server` and a real training process: nothing starts
    while the stall is a warning, and the command starts once when it is critical. The same for
    `--rank all`. Needs `pyarrow` and the server binary, as `test_two_ranks_e2e.py` does."""
    pytest.importorskip("pyarrow.flight")
    import test_two_ranks_e2e as e2e

    if e2e.BINARY is None:
        pytest.skip("tensorquill-server is not built")
    command, calls = hook
    run_id = "stall"
    work, store = tmp_path / "work", tmp_path / "store"
    work.mkdir()
    status_file = store / "runs" / run_id / "watchdog.json"
    outputs = {"one": tmp_path / "one.json", "all": tmp_path / "all.json"}
    clean = {k: v for k, v in os.environ.items()
             if k not in ("TENSORQUILL_TOKEN", "TENSORQUILL_TLS_CA", "MLFLOW_RUN_ID", "TORCHELASTIC_RUN_ID")}
    clean["PYTHONPATH"] = str(SRC)
    # Since Nov 17, 2026 `--watch` needs the Rust detectors' binary before its first look: it is told where
    # the binary this test's server runs from is, as an installed tensorquill-server would be found.
    clean.setdefault("TENSORQUILL_SERVER_BIN", str(e2e.BINARY))
    processes: list[subprocess.Popen] = []
    logs = []

    def watcher(name: str) -> subprocess.Popen:
        log = (tmp_path / f"watcher-{name}-{len(logs)}.log").open("w")
        logs.append(log)
        started = subprocess.Popen(
            [sys.executable, "-m", "tensorquill", str(status_file.parent), "--watch", "--interval", "0.2",
             *(("--rank", "all") if name == "all" else ()), "-o", str(outputs[name]), "--on-incident", command()],
            cwd=tmp_path, env=clean, stdout=log, stderr=subprocess.STDOUT,
            # A process started from a background shell job inherits SIGINT as ignored.
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
        )
        processes.append(started)
        return started

    def stop(process: subprocess.Popen) -> None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(20)

    def until(ready, what: str, seconds: float = 60.0):
        deadline = time.monotonic() + seconds
        while True:
            found = ready()
            if found:
                return found
            assert time.monotonic() < deadline, f"timed out waiting for {what}"
            time.sleep(0.05)

    def status() -> dict:
        return tensorquill.evidence.read_watchdog(status_file) or {}

    def stall(name: str) -> dict | None:
        """The stall in the evidence a watcher last wrote, or None."""
        try:
            evidence = json.loads(outputs[name].read_text())
        except (OSError, ValueError):
            return None
        listed = evidence.get("run_incidents" if name == "all" else "incidents") or ()
        return next((i for i in listed if i["kind"] == "job_stalled"), None)

    def fired(name: str) -> list[dict]:
        # The command line's trigger hands over the compact file, written beside the full one.
        handed = outputs[name].with_name(outputs[name].stem + ".compact.json")
        return [c["payload"] for c in calls() if c["payload"]["evidence"] == str(handed.resolve())]

    def deliveries(name: str) -> list[dict]:
        path = state_path_for(outputs[name])
        return json.loads(path.read_text())["deliveries"] if path.exists() else []

    server = e2e.Server(store, 0, *_STALL_SERVER)
    try:
        rank_log = (tmp_path / "rank0.log").open("w")
        logs.append(rank_log)
        rank = subprocess.Popen(
            [sys.executable, "-c", _STALLING_RANK], cwd=work, stdout=rank_log, stderr=subprocess.STDOUT,
            env={**clean, "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "TENSORQUILL_ENDPOINT": server.endpoint,
                 "TQ_RUN_ID": run_id, "TQ_STEPS": "300", "PYTHONUNBUFFERED": "1"},
        )
        processes.append(rank)
        until(lambda: status().get("summary") == "active" and status()["ranks"][0]["last_step"] is not None,
              "the rank's first stored steps")
        # Started while the rank is still stepping (300 steps take 3 s or more), so each has its
        # first look behind it before the stall begins.
        watchers = {name: watcher(name) for name in outputs}

        # The rank stops stepping. The server calls it silent, a warning, and both watchers write
        # that down; neither starts anything, and a second later that is still so.
        for name in outputs:
            until(lambda: (stall(name) or {}).get("severity") == "warning", f"the stall at warning in {name}.json")
        time.sleep(1.0)
        assert [stall(name)["severity"] for name in outputs] == ["warning", "warning"], "the warning did not last the check"
        assert calls() == [] and [deliveries(name) for name in outputs] == [[], []]
        assert rank.poll() is None  # its process lives
        last_step = status()["ranks"][0]["last_step"]

        # No step arrives (the rank's heartbeats do, every 0.5 s), and the server goes on writing
        # its status: three more writes are seen here. That is how the severity reaches a watcher
        # of a job that takes no more steps.
        written = {status().get("written_at_ms")}
        until(lambda: written.add(status().get("written_at_ms")) or len(written) >= 4, "the status written again")

        # The stall becomes critical, and each watcher starts the command.
        for name in outputs:
            until(lambda: fired(name), f"the command of the {name} watcher", 90.0)
        assert status()["ranks"][0]["severity"] == "critical" and status()["summary"] == "stalled"
        assert rank.poll() is None
        # More looks, and more rewrites of the status by the server: nothing more starts.
        seen = {status().get("written_at_ms")}
        until(lambda: seen.add(status().get("written_at_ms")) or len(seen) >= 5, "more rewrites of the status")
        # A restarted watcher of each kind, on the stall that is still critical: nothing more.
        for name in outputs:
            stop(watchers[name])
            again = watcher(name)
            until(lambda: f"watching {'every rank of ' if name == 'all' else ''}{status_file.parent}" in
                  (tmp_path / f"watcher-{name}-{len(logs) - 1}.log").read_text(), f"the restarted {name} watcher")
            time.sleep(1.5)
            stop(again)

        for name, who, ranks in (("one", "rank 0", [0]), ("all", "the run", [])):
            (payload,) = fired(name)  # once, and not before it was critical
            incident = payload["incident"]
            assert (payload["kind"], payload["class"], payload["step"]) == ("job_stalled", "observation", last_step)
            assert (incident["severity"], incident["assessment"]) == ("critical", "suspected_hang")
            assert payload["ranks"] == ranks and payload["run_state"] == "unknown"
            # Where the rank is stuck travels with it: the frame its main thread has not left.
            assert [place["function"] for place in incident["stuck_at"]] == ["stuck_here"]
            (delivery,) = deliveries(name)
            assert (delivery["kind"], delivery["step"], delivery["command"]["attempts"]) == ("job_stalled", last_step, 1)
            printed = "".join(path.read_text() for path in sorted(tmp_path.glob(f"watcher-{name}-*.log")))
            assert "firing: invariants, a stall at critical\n" in printed
            assert printed.count(f"trigger: job_stalled at step {last_step} ({who}): started the command") == 1
            assert "Traceback" not in printed
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(20)
        for log in logs:
            log.close()
        server.kill()


# --- Nov 17, 2026: one firing for one stall, a lost connection, and a fresh window at a restart --
#
# Real training processes beside a real `tensorquill-server`, watched by `python -m tensorquill`
# as a user starts it. Needs `pyarrow` and the server binary, as `test_two_ranks_e2e.py` does.

# One rank. It steps `TQ_STEPS` times, or until the file `TQ_STOP` exists, and then stops stepping
# without closing: its process, its stream and its heartbeats go on, as those of a hung job do.
# With `TQ_RESUME`, once that file exists it takes `TQ_MORE` more steps and stops again.
_RANK_UNTIL = """
import os, time
import torch, tensorquill
from torch import nn

if os.environ.get("TQ_OLD_SENDER"):  # a sender from before Nov 17, 2026: no attempt in its heartbeats
    import tensorquill.flight as _flight
    _flight.Heartbeat._take_attempt = staticmethod(lambda: None)

model = nn.Linear(8, 1)
session = tensorquill.init(model, run_id=os.environ["TQ_RUN_ID"], every=10, heartbeat_seconds=0.5, stack_seconds=1.0)
session._sink.sender._batch_seconds, session._sink.sender._backoff_cap = 0.25, 0.5
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
stop, steps, step = os.environ.get("TQ_STOP"), int(os.environ.get("TQ_STEPS", "0")), 0
while (os.path.exists(stop) is False) if stop else step < steps:
    optimizer.zero_grad()
    loss = model(torch.randn(16, 8)).pow(2).mean()
    loss.backward()
    session.log({"train/loss": loss})
    optimizer.step()
    step += 1
    time.sleep(0.01)
# Steps leave the device ten at a time: the last ones are flushed before the stop (a full ring's
# newest step otherwise waits until the next one ends).
while step % 10:
    session.log({"train/loss": torch.tensor(0.0)})
    step += 1
session.flush()
print("stopped stepping", flush=True)
resume = os.environ.get("TQ_RESUME")
if resume:
    while not os.path.exists(resume):
        time.sleep(0.05)
    for _ in range(int(os.environ["TQ_MORE"])):
        optimizer.zero_grad()
        loss = model(torch.randn(16, 8)).pow(2).mean()
        loss.backward()
        session.log({"train/loss": loss})
        optimizer.step()
        time.sleep(0.01)
    session.flush()
    print("stopped stepping again", flush=True)

def stuck_here():
    while True:
        time.sleep(0.2)

stuck_here()
"""


class _RealJob:
    """A real server and real ranks of one run, and the command-line watchers of it."""

    def __init__(self, tmp_path: Path, run_id: str, *flags: str) -> None:
        pytest.importorskip("pyarrow.flight")
        import test_two_ranks_e2e as e2e

        if e2e.BINARY is None:
            pytest.skip("tensorquill-server is not built")
        self.e2e, self.tmp, self.run_id, self.flags = e2e, tmp_path, run_id, flags
        self.store, self.work = tmp_path / "store", tmp_path / "work"
        self.work.mkdir()
        self.status_file = self.store / "runs" / run_id / "watchdog.json"
        self.env = {k: v for k, v in os.environ.items()
                    if k not in ("TENSORQUILL_TOKEN", "TENSORQUILL_TLS_CA", "MLFLOW_RUN_ID", "TORCHELASTIC_RUN_ID")}
        self.env["PYTHONPATH"] = str(SRC)
        # Since Nov 17, 2026 `--watch` needs the Rust detectors' binary before its first look: it is told where
        # the binary this test's server runs from is, as an installed tensorquill-server would be found.
        self.env.setdefault("TENSORQUILL_SERVER_BIN", str(e2e.BINARY))
        self.processes: list[subprocess.Popen] = []
        self.logs: list = []
        self.servers = [e2e.Server(self.store, 0, *flags)]

    @property
    def server(self):
        return self.servers[-1]

    def restart_server(self) -> float:
        """SIGKILL the server and start another on the same port and store; the time it started."""
        port = self.server.port
        self.server.kill()
        started = time.time()
        self.servers.append(self.e2e.Server(self.store, port, *self.flags))
        return started

    def rank(self, rank: int, world: int, **env: str) -> subprocess.Popen:
        log = (self.tmp / f"rank{rank}.log").open("w")
        self.logs.append(log)
        process = subprocess.Popen(
            [sys.executable, "-c", _RANK_UNTIL], cwd=self.work, stdout=log, stderr=subprocess.STDOUT,
            env={**self.env, "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world),
                 "TENSORQUILL_ENDPOINT": self.server.endpoint, "TQ_RUN_ID": self.run_id, "PYTHONUNBUFFERED": "1", **env},
        )
        self.processes.append(process)
        return process

    def watcher(self, output: Path, *options: str) -> Path:
        log_path = self.tmp / f"watcher-{len(self.logs)}.log"
        log = log_path.open("w")
        self.logs.append(log)
        self.processes.append(subprocess.Popen(
            [sys.executable, "-m", "tensorquill", str(self.status_file.parent), "--watch", "--interval", "0.2",
             "-o", str(output), *options],
            cwd=self.tmp, env=self.env, stdout=log, stderr=subprocess.STDOUT,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
        ))
        return log_path

    def status(self) -> dict:
        return tensorquill.evidence.read_watchdog(self.status_file) or {}

    @staticmethod
    def until(ready, what: str, seconds: float = 60.0):
        deadline = time.monotonic() + seconds
        while True:
            found = ready()
            if found:
                return found
            assert time.monotonic() < deadline, f"timed out waiting for {what}"
            time.sleep(0.05)

    def close(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                if process.args[:3] == [sys.executable, "-m", "tensorquill"]:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(20)
                        continue
                    except subprocess.TimeoutExpired:
                        pass
                process.kill()
                process.wait(20)
        for server in self.servers:
            if server.process.poll() is None:
                server.kill()
        for log in self.logs:
            log.close()


# Silent after 2 s without a step, critical after 6 s; the status is looked at every 0.2 s and an
# unchanged one is written again every 0.6 s.
_QUICK_STALL = ("--silent-seconds", "2", "--silent-factor", "0", "--critical-seconds", "6",
                "--watchdog-seconds", "0.2", "--flush-seconds", "0.3")


def test_a_silent_rank_that_becomes_a_stalled_job_fires_once_through_the_webhook(tmp_path, receiver):
    """One firing for one stall, through `--rank all --watch --webhook`, which turns on the
    compact evidence: the webhook names the compact file, and the full file is written as before."""
    hooked = receiver()
    job = _RealJob(tmp_path, "silent-then-stalled", *_QUICK_STALL)
    output = tmp_path / "all.json"
    compact = tmp_path / "all.compact.json"
    stop = tmp_path / "stop-rank1"
    try:
        job.rank(0, 2, TQ_STEPS="100")  # stops after 100 steps
        job.rank(1, 2, TQ_STOP=str(stop))  # steps until told to stop
        job.until(lambda: len(job.status().get("ranks", ())) == 2 and all(
            r["last_step"] is not None for r in job.status()["ranks"]), "both ranks' first stored steps")
        log = job.watcher(output, "--rank", "all", "--webhook", hooked.url)
        # Rank 0 is silent and then critical while rank 1 steps: `rank_silent`, posted once.
        job.until(lambda: hooked.requests, "the post for the silent rank", 90.0)
        assert job.status()["summary"] == "degraded"
        stop.touch()
        # Rank 1 stops too: the job is stalled, and already critical (rank 0 is).
        def stalled():
            try:
                listed = json.loads(output.read_text()).get("run_incidents") or ()
            except (OSError, ValueError):
                return None
            return next((i for i in listed if i["kind"] == "job_stalled" and i["severity"] == "critical"), None)

        job.until(stalled, "the stalled job at critical in the full evidence", 90.0)
        seen = {job.status().get("written_at_ms")}
        job.until(lambda: seen.add(job.status().get("written_at_ms")) or len(seen) >= 5, "more writes of the status")
        time.sleep(1.0)
        printed = log.read_text()
    finally:
        job.close()
    (request,) = hooked.requests  # once for the stall, not again when it became a stalled job
    body = json.loads(request["body"])
    assert (body["kind"], body["incident"]["severity"], body["assessment"]) == ("rank_silent", "critical", "suspected_hang")
    assert body["evidence"] == str(compact.resolve())
    full, small = json.loads(output.read_text()), json.loads(compact.read_text())
    assert "compact" not in full and full["view"] == "all_ranks" and small["compact"]["samples"]["total"] == 0
    assert [i["kind"] for i in full["run_incidents"]] == ["job_stalled"]
    (delivery,) = json.loads(state_path_for(output).read_text())["deliveries"]
    assert delivery["kind"] == "rank_silent"
    assert "hands over the compact evidence" in printed and printed.count("posted to the webhook") == 1
    assert "Traceback" not in printed


def test_a_rank_killed_mid_run_reads_as_a_lost_connection_and_fires_once(tmp_path, hook):
    """An out-of-memory kill, as the stream sees it: rank 1 is killed with SIGKILL while rank 0
    goes on living and waits (as it would in a collective). The stall is a lost connection, in
    Rust's incidents and Python's alike, and the command is started once with that label."""
    command, calls = hook
    job = _RealJob(tmp_path, "killed", *_QUICK_STALL)
    output = tmp_path / "all.json"
    try:
        job.rank(0, 2, TQ_STEPS="100")
        victim = job.rank(1, 2, TQ_STEPS="100")
        job.until(lambda: len(job.status().get("ranks", ())) == 2 and all(
            r["last_step"] == 99 for r in job.status()["ranks"]), "both ranks' last steps stored")
        log = job.watcher(output, "--rank", "all", "--on-incident", command())
        victim.kill()  # SIGKILL: no clean end of the stream
        victim.wait(20)
        job.until(calls, "the command", 90.0)
        status = job.until(lambda: job.status() if job.status().get("incidents") and
                           job.status()["incidents"][0]["severity"] == "critical" else None, "a critical status")
        seen = {job.status().get("written_at_ms")}
        job.until(lambda: seen.add(job.status().get("written_at_ms")) or len(seen) >= 4, "more writes of the status")
        time.sleep(1.0)
        printed = log.read_text()
    finally:
        job.close()
    # Rust wrote the status's incidents; Python reads the same status to the same incidents.
    assert detect_watchdog_incidents(status) == json.loads(json.dumps(status["incidents"]))
    (stall,) = status["incidents"]
    assert (stall["kind"], stall["assessment"], stall["lost_ranks"]) == ("job_stalled", "lost_connection", [1])
    assert {r["rank"]: r["state"] for r in stall["ranks"]} == {0: "silent", 1: "lost"}
    (call,) = calls()
    payload, env = call["payload"], call["env"]
    assert (payload["kind"], payload["assessment"], payload["step"]) == ("job_stalled", "lost_connection", 99)
    assert env["TENSORQUILL_INCIDENT_ASSESSMENT"] == "lost_connection"
    assert "read as a lost connection, not a hang" in payload["summary"] and "out-of-memory kill" in payload["summary"]
    assert env["TENSORQUILL_EVIDENCE"] == str((tmp_path / "all.compact.json").resolve())
    assert printed.count("trigger: job_stalled (lost connection) at step 99 (the run): started the command") == 1
    assert "Traceback" not in printed


def test_a_restart_gives_a_restored_rank_a_fresh_window_before_it_can_fire(tmp_path, hook):
    """A rank hung for 8 s (a warning; critical at 12 s) when the server is killed and started
    again. Under the rule before Nov 17, 2026 it was critical about 4 s after the restart and
    fired; now its window starts at the restart: silent 3 s after it, critical 12 s after it."""
    command, calls = hook
    job = _RealJob(tmp_path, "restarted", "--silent-seconds", "3", "--silent-factor", "0", "--critical-seconds", "12",
                   "--watchdog-seconds", "0.2", "--flush-seconds", "0.3")
    output = tmp_path / "one.json"
    try:
        rank = job.rank(0, 1, TQ_STEPS="100")
        job.until(lambda: (job.status().get("ranks") or [{}])[0].get("last_step") == 99, "the rank's last step stored")
        log = job.watcher(output, "--on-incident", command())
        job.until(lambda: (job.status().get("ranks") or [{}])[0].get("severity") == "warning", "the warning", 30.0)
        job.until(lambda: job.status()["ranks"][0]["seconds_since_progress"] >= 8.0, "8 s of quiet", 30.0)
        assert job.status()["ranks"][0]["severity"] == "warning" and calls() == []
        restarted = job.restart_server()
        restarted_ms = restarted * 1000
        # The first statuses of the new server: no stopped rank, no incident, the quiet time kept.
        first = job.until(lambda: job.status() if job.status().get("written_at_ms", 0) > restarted_ms else None,
                          "the new server's first status")
        assert first["summary"] == "active" and first["incidents"] == [], first
        assert first["ranks"][0]["state"] in ("reconnecting", "active")
        assert first["ranks"][0]["seconds_since_progress"] >= 8.0
        # Nothing fires inside the window and the warning that follows it: 9 s after the restart
        # (the old rule fired at about 4 s), nothing has started.
        while time.time() < restarted + 9.0:
            assert calls() == [], "fired inside the fresh window"
            time.sleep(0.1)
        silent = job.status()["ranks"][0]
        assert (silent["state"], silent["severity"]) == ("silent", "warning"), silent
        # After it, the command starts, once.
        job.until(calls, "the command after the window", 60.0)
        time.sleep(1.5)
        assert rank.poll() is None  # the training process was never touched
        # A server restart alone is not progress and not a new attempt: the episode stays open.
        # Restarted again, past the window and the critical time, nothing more fires.
        again = job.restart_server()
        job.until(lambda: job.status().get("written_at_ms", 0) > again * 1000 and
                  job.status()["ranks"][0]["severity"] == "critical", "critical again after the second restart", 60.0)
        time.sleep(2.0)
        assert len(calls()) == 1
        printed = log.read_text()
    finally:
        job.close()
    (call,) = calls()
    incident = call["payload"]["incident"]
    assert (call["payload"]["kind"], incident["severity"], incident["assessment"]) == ("job_stalled", "critical", "suspected_hang")
    assert incident["written_at_ms"] - restarted_ms >= 11_000  # critical 12 s after the restart
    assert "Traceback" not in printed



def test_a_real_job_that_steps_past_its_stall_and_stalls_again_fires_again(tmp_path, hook):
    """One firing for one stall episode, with a real server: the rank stalls (fires once over
    many looks), steps past the step it stalled at, stalls again, and fires again."""
    command, calls = hook
    job = _RealJob(tmp_path, "stall-twice", *_QUICK_STALL)
    output, resume = tmp_path / "one.json", tmp_path / "resume"
    try:
        job.rank(0, 1, TQ_STEPS="100", TQ_RESUME=str(resume), TQ_MORE="100")
        job.until(lambda: (job.status().get("ranks") or [{}])[0].get("last_step") == 99, "the first 100 steps")
        log = job.watcher(output, "--on-incident", command())
        job.until(calls, "the first firing", 90.0)
        seen = {job.status().get("written_at_ms")}
        job.until(lambda: seen.add(job.status().get("written_at_ms")) or len(seen) >= 6, "more writes of the status")
        assert len(calls()) == 1
        resume.touch()
        job.until(lambda: job.status()["ranks"][0]["last_step"] == 199, "100 more steps")
        job.until(lambda: len(calls()) == 2, "the second firing", 90.0)
        time.sleep(1.5)
        printed = log.read_text()
    finally:
        job.close()
    assert [(c["payload"]["kind"], c["payload"]["step"], c["payload"]["incident"]["severity"]) for c in calls()] == [
        ("job_stalled", 99, "critical"), ("job_stalled", 199, "critical")]
    state = json.loads(state_path_for(output).read_text())
    assert [d["step"] for d in state["deliveries"]] == [99, 199] and state["episode"]["run_step"] == 199
    assert "Traceback" not in printed


def test_a_new_attempt_that_hangs_before_it_passes_the_old_step_fires_again(tmp_path, hook):
    """The job is killed and started again (a new attempt), resumes below the step it stalled
    at and hangs there. The status's `attempt` changes with the new attempt's first stored batch:
    nothing fires before that (the rank reads stalled from the old attempt's last batch while its
    new process has stored nothing), and the new stall fires once."""
    command, every_call = hook
    # The new attempt numbers its steps from 0 again, which `repeated_steps` (an invariant) reports
    # on its own: only the stalls are counted here.
    calls = lambda: [c for c in every_call() if c["payload"]["kind"] in STALL_KINDS]  # noqa: E731
    job = _RealJob(tmp_path, "attempts", *_QUICK_STALL)
    output = tmp_path / "all.json"
    try:
        first = job.rank(0, 1, TQ_STEPS="100")
        job.until(lambda: (job.status().get("ranks") or [{}])[0].get("last_step") == 99, "the first attempt's steps")
        old_attempt = job.status()["ranks"][0]["attempt"]
        log = job.watcher(output, "--rank", "all", "--on-incident", command())
        job.until(calls, "the first firing", 90.0)
        first.kill()
        first.wait(20)
        time.sleep(1.0)
        job.rank(0, 1, TQ_STEPS="30")  # a new session: a new attempt, which stops at step 29
        new_attempt = job.until(lambda: (lambda a: a if a != old_attempt else None)(job.status()["ranks"][0]["attempt"]),
                                "the new attempt's first stored batch")
        assert len(calls()) == 1, "fired before the new attempt stored a step"
        job.until(lambda: len(calls()) == 2, "the new attempt's stall", 90.0)
        seen = {job.status().get("written_at_ms")}
        job.until(lambda: seen.add(job.status().get("written_at_ms")) or len(seen) >= 5, "more writes of the status")
        time.sleep(1.0)
        status, printed = job.status(), log.read_text()
    finally:
        job.close()
    assert len(calls()) == 2
    (rank,) = status["ranks"]
    # The new attempt's own step, below the old attempt's 99 (until Nov 18, 2026 the status kept the
    # highest step of any attempt, 99, until the new one passed it).
    assert rank["attempt"] == new_attempt and rank["last_step"] == 29
    payload = calls()[1]["payload"]
    assert (payload["kind"], payload["incident"]["severity"]) == ("job_stalled", "critical")
    assert payload["incident"]["ranks"][0]["attempt"] == new_attempt
    assert payload["id"] != calls()[0]["payload"]["id"]
    assert [c["payload"]["kind"] for c in every_call() if c["payload"]["kind"] not in STALL_KINDS] in ([], ["repeated_steps"])
    assert "Traceback" not in printed



def _restarted_job(tmp_path, command, calls, *, old_sender=False, view="all"):
    """A job that stalls and fires, is killed, and is started again (a new attempt) that hangs
    before it stores a step. Silent after 2 s, critical 6 s later; a new attempt starts for 4 s.
    Returns what the status and the stall firings said, and when the new attempt started."""
    job = _RealJob(tmp_path, "restarted-job", "--silent-seconds", "2", "--silent-factor", "0", "--critical-seconds", "6",
                   "--start-seconds", "4", "--watchdog-seconds", "0.2", "--flush-seconds", "0.3")
    output = tmp_path / "all.json"
    old = {"TQ_OLD_SENDER": "1"} if old_sender else {}
    stalls = lambda: [c for c in calls() if c["payload"]["kind"] in STALL_KINDS]  # noqa: E731
    try:
        first = job.rank(0, 1, TQ_STEPS="100", **old)
        job.until(lambda: (job.status().get("ranks") or [{}])[0].get("last_step") == 99, "the first attempt's steps")
        old_attempt = job.status()["ranks"][0]["attempt"]
        log = job.watcher(output, *(("--rank", "all") if view == "all" else ()), "--on-incident", command())
        job.until(stalls, "the first firing", 90.0)
        first.kill()
        first.wait(20)
        started = time.time()
        second = job.rank(0, 1, TQ_STEPS="0", **old)  # a new session that never steps
        seen = []
        deadline = started + (25.0 if old_sender else 60.0)
        while time.time() < deadline and len(stalls()) < 2:
            status = job.status()
            seen.append((time.time() - started, (status.get("ranks") or [{}])[0], len(stalls())))
            time.sleep(0.1)
        time.sleep(1.0)
        assert second.poll() is None  # never touched
        return {"old_attempt": old_attempt, "seen": seen, "stalls": stalls(), "started": started,
                "status": job.status(), "printed": log.read_text()}
    finally:
        job.close()


@pytest.mark.parametrize("view", ["all", "one"])
def test_a_job_started_again_that_hangs_before_its_first_stored_step_fires_after_its_start_window(tmp_path, hook, view):
    """The heartbeats carry the attempt (Nov 17, 2026): the server sees the new attempt before it
    stores anything and gives it the start window of a new rank; the status shows the attempt,
    so the trigger's episode re-arms, and the hang fires once it is critical (4 + 6 s), not before.
    The new process's stream opens only after the beat that names the attempt, so the rank never
    reads as an open stream of the old attempt. With `--rank all` and with one rank's watcher."""
    command, calls = hook
    found = _restarted_job(tmp_path, command, calls, view=view)
    # From the server's first status that shows the killed process's stream dropped on.
    dropped = next(n for n, (_, r, _) in enumerate(found["seen"]) if r.get("stream") == "dropped")
    gap = [(t, r) for t, r, _ in found["seen"][dropped:]
           if r.get("stream") == "open" and r.get("attempt") == found["old_attempt"]]
    assert not gap, gap[:3]
    new = [(t, r) for t, r, _ in found["seen"] if r.get("attempt") not in (None, found["old_attempt"])]
    assert new, "the new attempt never showed in the status"
    first_seen, rank = new[0]
    assert rank["attempt_without_steps"] is True and rank["last_step"] == 99
    assert first_seen < 4.0, first_seen  # from its first beat, sent as the session starts
    # Not called stopped inside its start window: starting (or reconnecting) until it has passed.
    assert all(r["state"] in ("starting", "reconnecting", "active") for t, r in new if t < first_seen + 3.5), new
    assert not [n for t, r, n in found["seen"] if n > 1 and t < first_seen + 9.0], "fired inside the window"
    first, again = found["stalls"]
    assert again["payload"]["incident"]["ranks"][0]["attempt"] == rank["attempt"]
    assert again["payload"]["incident"]["severity"] == "critical" and again["payload"]["id"] != first["payload"]["id"]
    assert "started again (a new attempt) and stored no step of it yet" in again["payload"]["summary"]
    assert "Traceback" not in found["printed"]


def test_a_sender_without_attempts_in_its_heartbeats_still_streams_and_is_watched_as_before(tmp_path, hook):
    """An older sender against this server: its beats name no attempt. It streams and fires as
    before, and a job of it started again that hangs before storing a step is not seen as a new
    attempt (as before Nov 17, 2026): no second firing."""
    command, calls = hook
    found = _restarted_job(tmp_path, command, calls, old_sender=True)
    assert len(found["stalls"]) == 1
    assert all(r.get("attempt") == found["old_attempt"] for _, r, _ in found["seen"])
    assert not any(r.get("attempt_without_steps") for _, r, _ in found["seen"])
    assert "Traceback" not in found["printed"]


def test_the_heartbeat_names_the_attempt_the_sessions_records_carry(tmp_path):
    """The beat's attempt must be the rows' attempt, or the server would see a new attempt on every
    beat. Read where the session sets it, so a change to how `capture.py` hands over its counter
    shows here."""
    pytest.importorskip("pyarrow.flight")
    import torch
    from torch import nn

    from tensorquill.flight import Heartbeat

    beat = Heartbeat.__new__(Heartbeat)  # no thread, no connection: only the counter's setter
    import threading
    beat._changed, beat._counter, beat.attempt = threading.Event(), None, None
    model = nn.Linear(4, 1)
    sink = type("Sink", (), {"heartbeat": beat, "write": lambda self, r: rows.append(r), "close": lambda self: None,
                             "flush": lambda self: None})()
    rows: list = []
    session = tensorquill.init(model, run_id="beat", sink=sink, every=1)
    assert beat.attempt == session.attempt and beat._changed.is_set()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.randn(2, 4)).sum().backward()
    session.log({})
    optimizer.step()
    session.close()
    records = [r for batch in rows for r in (batch if isinstance(batch, list) else [batch])]
    assert records and {r["attempt"] for r in records} == {beat.attempt}
