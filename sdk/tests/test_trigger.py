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
from tensorquill.evidence import main, watch
from tensorquill.trigger import Trigger, state_path_for

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
        "new_ranks": [0], "summary": "NaN or Inf at step 30 in <total>", "incident": incident,
    }
    assert env == {
        "TENSORQUILL_EVIDENCE": str(out.resolve()), "TENSORQUILL_RUN_ID": "r",
        "TENSORQUILL_INCIDENT_ID": "r/rank0/non_finite/30", "TENSORQUILL_INCIDENT_KIND": "non_finite",
        "TENSORQUILL_INCIDENT_CLASS": "invariant", "TENSORQUILL_INCIDENT_STEP": "30",
        "TENSORQUILL_INCIDENT_RANKS": "0", "TENSORQUILL_INCIDENT_SUMMARY": "NaN or Inf at step 30 in <total>",
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
         "--on-incident", command()],
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
    assert "trigger: each new incident fires the command once; firing: invariants" in printed
    assert "trigger: non_finite at step 30 (rank 0): started the command" in printed
    assert "Traceback" not in printed


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

    # Incidents of the whole run: an invariant fires with no rank, an observation does not.
    hang = {"kind": "collective_hang", "class": "invariant", "step": 9}
    stall = {"kind": "job_stalled", "class": "observation", "step": 9, "ranks": []}
    trigger.look(view([0, 1, 2, 3], [stall, hang]), path)
    assert _settle(trigger, lambda: len(calls()) == 3)
    third = calls()[2]
    assert (third["payload"]["kind"], third["payload"]["ranks"], third["payload"]["id"]) == (
        "collective_hang", [], "r/run/collective_hang/9",
    )
    assert third["env"]["TENSORQUILL_INCIDENT_RANKS"] == ""
    restarted = Trigger(command())
    restarted.look(view([0, 1, 2, 3], [stall, hang]), path)
    restarted.close(wait=2)
    assert len(calls()) == 3 and len(restarted.deliveries) == 3


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
        "rank", "ranks", "new_ranks", "summary", "incident",
    ])
    assert (body["event"], body["version"], body["id"], body["attempt"]) == ("incident", 1, "r/rank0/non_finite/30", 1)
    assert (body["run_id"], body["kind"], body["class"], body["step"]) == ("r", "non_finite", "invariant", 30)
    assert (body["rank"], body["ranks"], body["new_ranks"], body["run_state"]) == (0, [0], [0], "unhealthy")
    assert body["evidence"] == str(out.resolve()) and body["summary"] == "NaN or Inf at step 30 in <total>"
    assert abs(body["sent_at_ms"] / 1000 - time.time()) < 60
    written = json.loads(out.read_text())["incidents"][0]
    assert body["incident"]["gradients"] == written["gradients"] == ["<total>"]
    assert body["incident"]["scalars"]["loss"]["value"] is None  # NaN in the evidence file, null in the body
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
