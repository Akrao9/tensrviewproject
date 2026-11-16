"""`--watch` with `--detectors rust`: the watcher judges through `tensorquill-server detect`.

Whenever the watcher runs the detectors itself (every rebuild of a `.jsonl` file, and a
server's store whose report cannot be used), `detectors="rust"` sends the records read so far
to the server binary. What is held here:

- a growing file gives the same evidence at every look as the Python detectors give:
  incidents, `not_checked` and the run state compared as the parity tests compare them
  (numbers to 1e-12, a logged NaN as `null`), the rest exactly. The files are the 57 parity
  record sets and the seven Gate 1 fixtures, each written a stretch at a time;
- `--compact` and the trigger (`--on-incident`, the `fired.json` state) do the same through
  this path: the trigger's own tests on a local file run again with the option given;
- a missing binary is said before the first look, not at the first rebuild.

Needs `pyarrow` (the `flight` extra) and the server binary
(`cargo build --manifest-path server/Cargo.toml`, or `TENSORQUILL_SERVER_BIN`).
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import tensorquill  # noqa: E402
from tensorquill.compact import compact_evidence  # noqa: E402
from tensorquill.evidence import watch  # noqa: E402
from tensorquill.server_detect import BINARY_ENV  # noqa: E402
from tensorquill.trigger import Trigger, state_path_for  # noqa: E402

import test_trigger  # noqa: E402
from test_flight_e2e import BINARY  # noqa: E402
from test_server_detect import CASES, GATE_FIXTURES, JUDGED  # noqa: E402
from test_server_detectors import differences, everything  # noqa: E402
from test_trigger import _no_proxy, hook, receiver  # noqa: E402,F401  (fixtures of the tests taken from there)

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
RECORD_SETS = {**CASES, **{f"gate 1 {name}": records for name, records in GATE_FIXTURES.items()}}
LOOKS = 4  # stretches a file is written in


@pytest.fixture(autouse=True)
def binary(monkeypatch):
    monkeypatch.setenv(BINARY_ENV, str(BINARY))


def lines(records) -> str:
    return "".join(json.dumps(r) + "\n" for r in records)


def watched(path: Path, records: list[dict], cuts: list[int], **options) -> list[dict]:
    """The evidence a watcher writes at each look while `records` are appended to `path`,
    `cuts[0]` of them at the start and up to each later cut after each write."""
    path.write_text(lines(records[:cuts[0]]), encoding="utf-8")
    seen: list[dict] = []

    def grow(evidence, out):
        assert json.loads(out.read_text(encoding="utf-8")).keys() >= {"incidents", "run_state"}
        seen.append(json.loads(json.dumps(evidence)))
        if len(seen) < len(cuts):
            with path.open("a", encoding="utf-8") as file:
                file.write(lines(records[cuts[len(seen) - 1]:cuts[len(seen)]]))

    watch(path, interval=0.0, polls=len(cuts) + 1, on_update=grow, **options)
    return seen


def same(python: dict, rust: dict) -> list[str]:
    """Where the Rust watcher's evidence differs from the Python watcher's."""
    rust = dict(rust)
    if rust.pop("detectors", None) != "rust":
        return ["the evidence does not say the Rust detectors made it"]
    if "detectors" in python:
        return ["the Python watcher's evidence names an engine"]
    found = [d for key in JUDGED for d in differences(python.get(key, []), rust.get(key, []), key)]

    def rest(evidence):
        return json.dumps({k: v for k, v in evidence.items() if k not in JUDGED}, sort_keys=True)

    return found + (["the keys made in Python differ"] if rest(python) != rest(rust) else [])


def cuts_for(records: list[dict]) -> list[int]:
    return sorted({max(1, len(records) * part // LOOKS) for part in range(1, LOOKS + 1)})


@pytest.mark.parametrize("name", RECORD_SETS)
def test_a_growing_file_gives_the_python_watchers_evidence_at_every_look(name, tmp_path):
    records = RECORD_SETS[name]
    cuts = cuts_for(records)
    python = watched(tmp_path / "python.jsonl", records, cuts)
    rust = watched(tmp_path / "rust.jsonl", records, cuts, detectors="rust")
    assert len(python) == len(rust) == len(cuts)
    for at, (ours, theirs) in zip(cuts, zip(python, rust)):
        assert same(ours, theirs) == [], f"after {at} of {len(records)} records"
    # The last look is the whole file, as one read of it gives.
    assert same(tensorquill.build_evidence(records), rust[-1]) == []


def test_the_record_sets_put_incidents_and_unchecked_things_in_front_of_both_watchers(tmp_path):
    kinds, checks = set(), set()
    for records in RECORD_SETS.values():
        evidence = tensorquill.build_evidence(records)
        kinds |= {i["kind"] for i in evidence["incidents"]}
        checks |= {item["check"] for item in evidence.get("not_checked", ())}
    assert len(kinds) >= 25 and len(checks) >= 3 and len(RECORD_SETS) >= 57


def test_the_watcher_sends_what_it_has_read_so_far_once_for_each_rebuild(tmp_path, monkeypatch):
    import tensorquill.server_detect as module

    sent: list[int] = []
    real = module.server_report

    def counted(records, **kwargs):
        records = list(records)
        sent.append(len(records))
        return real(records, **kwargs)

    monkeypatch.setattr(module, "server_report", counted)
    records = everything()
    seen = watched(tmp_path / "w.jsonl", records, [100, 200, 300], detectors="rust")
    # One call a rebuild, each over every record read so far; no call on a look that found nothing new.
    assert sent == [100, 200, 300] and len(seen) == 3
    # A line still being written is not sent until it is whole.
    path = tmp_path / "torn.jsonl"
    path.write_text(lines(records[:50]) + json.dumps(records[50])[:20], encoding="utf-8")
    sent.clear()
    watch(path, interval=0.0, polls=2, detectors="rust")
    assert sent == [50]


def test_thresholds_given_to_the_watcher_go_to_the_rust_rules(tmp_path):
    options = {"overload_threshold": 0.5, "drift_tolerance": 1e-3, "starve_window": 20, "starve_ratio": 0.5, "top_k": 2}
    records = everything()
    python = watched(tmp_path / "python.jsonl", records, [150, 300], **options)
    rust = watched(tmp_path / "rust.jsonl", records, [150, 300], detectors="rust", **options)
    assert [same(a, b) for a, b in zip(python, rust)] == [[], []]
    usual = tensorquill.build_evidence(records)
    assert [(i["kind"], i["step"]) for i in rust[-1]["incidents"]] != [(i["kind"], i["step"]) for i in usual["incidents"]]
    # A value the Rust rules refuse stops the watcher with the server's reason, and nothing is written.
    path = tmp_path / "refused.jsonl"
    path.write_text(lines(records[:20]), encoding="utf-8")
    with pytest.raises(ValueError, match="starve_window"):
        watch(path, interval=0.0, polls=1, detectors="rust", starve_window=0)
    assert not path.with_suffix(".evidence.json").exists()


def test_the_compact_file_is_the_same_through_rust(tmp_path):
    from test_compact import _moe_records

    records = _moe_records(200, collapse_from=100)
    python = tmp_path / "python.jsonl"
    rust = tmp_path / "rust.jsonl"
    watched(python, records, [90, 200], compact_window=2)
    full = watched(rust, records, [90, 200], compact_window=2, detectors="rust")
    assert [len(e["samples"]) for e in full] == [90, 200]  # `on_update` still gets the full evidence
    out = tmp_path / "rust.evidence.compact.json"
    assert out.exists() and not (tmp_path / "rust.evidence.json").exists()
    small = json.loads(out.read_text(encoding="utf-8"))
    assert small.pop("detectors") == "rust" and small["compact"]["kept_steps"] == [[98, 102]]
    assert small == json.loads((tmp_path / "python.evidence.compact.json").read_text(encoding="utf-8"))
    assert small == json.loads(json.dumps(compact_evidence(tensorquill.build_evidence(records), window=2)))


def test_a_planted_failure_fires_the_same_through_rust_as_through_python(tmp_path, hook):
    command, calls = hook
    records = test_trigger._with(test_trigger._with(test_trigger._records(range(80)), 30, **test_trigger.NAN), 60,
                                 **test_trigger.SPIKE)
    fired: dict[str, list] = {}
    for engine in ("python", "rust"):
        path = tmp_path / f"{engine}.jsonl"
        trigger = Trigger(command(), anomalies=True)
        watched(path, records, [20, 40, 80], trigger=trigger, compact_window=5, detectors=engine)
        # One command runs at a time: the second starts on a later look, once the first has ended.
        wanted = 2 * (1 + ("python", "rust").index(engine))
        assert test_trigger._settle(trigger, lambda: len(calls()) == wanted), calls()
        trigger.close(wait=20)
        state = json.loads(state_path_for(path.with_suffix(".evidence.compact.json")).read_text(encoding="utf-8"))
        fired[engine] = [(d["kind"], d["step"], d["ranks"], d["command"]["status"], d["command"]["attempts"])
                         for d in state["deliveries"]]
    assert fired["rust"] == fired["python"] == [("non_finite", 30, [0], "done", 1), ("grad_spike", 60, [0], "done", 1)]
    # Each watcher started the command once for each incident, with the same payload but for
    # the file it names and the time.
    by_engine: dict[str, list] = {"python": [], "rust": []}
    for call in calls():
        payload = call["payload"]
        engine = "rust" if "rust.evidence" in payload.pop("evidence") else "python"
        payload.pop("sent_at_ms")
        assert call["env"]["TENSORQUILL_INCIDENT_KIND"] == payload["kind"]
        by_engine[engine].append(payload)
    key = lambda payload: payload["step"]  # noqa: E731
    assert len(by_engine["rust"]) == 2
    assert differences(sorted(by_engine["python"], key=key), sorted(by_engine["rust"], key=key), "payloads") == []


# The trigger's own tests that watch a local file, run again with the option given: every one
# in `test_trigger.py` that calls `watch` on a `.jsonl` file. Left out: those that hand the
# trigger evidence they built themselves (no watcher runs), the two that start
# `python -m tensorquill` or call `main` (the command line has its own test below), and the
# one on a server's store, whose incidents are the server's.
TAKEN = [
    "test_a_planted_failure_in_a_growing_file_starts_the_command_once",
    "test_an_anomaly_fires_only_with_the_option_and_an_observation_only_by_name",
    "test_a_failing_command_is_retried_a_bounded_number_of_times_and_the_watcher_goes_on",
    "test_the_webhook_posts_the_incident_as_json_once",
    "test_an_unreachable_webhook_leaves_the_watcher_running_and_the_command_fired_once",
]
GIVEN: list[str] = []


@pytest.fixture(autouse=True)
def the_rust_option_is_given(monkeypatch, request):
    """For the tests taken from `test_trigger.py` (collected here under `...__taken_from_test_trigger`):
    every `watch` they call gets the option. The functions themselves are left as they are,
    so `test_trigger.py` still runs them through Python."""
    if not request.node.name.endswith("__taken_from_test_trigger"):
        yield
        return
    real = test_trigger.watch
    GIVEN.clear()

    def with_option(source, *args, **options):
        if options.get("rank") != "all":
            options.setdefault("detectors", "rust")
            GIVEN.append(str(source))
        return real(source, *args, **options)

    monkeypatch.setattr(test_trigger, "watch", with_option)
    yield
    assert GIVEN, "the test taken from test_trigger.py did not watch anything"


for _name in TAKEN:
    globals()[f"{_name}__taken_from_test_trigger"] = getattr(test_trigger, _name)


# --- the command line, a store, and what is refused --------------------------------------------


def test_the_command_line_watches_through_rust_with_compact_and_the_trigger(tmp_path, hook):
    command, calls = hook
    records = test_trigger._with(test_trigger._records(range(60)), 30, **test_trigger.NAN)
    source = tmp_path / "w.jsonl"
    source.write_text(lines(records[:20]), encoding="utf-8")
    out = tmp_path / "w.evidence.compact.json"
    # A process started from a background shell job inherits SIGINT as ignored: the child gets the default back.
    process = subprocess.Popen(
        [sys.executable, "-m", "tensorquill", str(source), "--watch", "--interval", "0.1", "--detectors", "rust",
         "--compact", "--on-incident", command()],
        cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
    )
    try:
        deadline = time.monotonic() + 60
        while not out.exists():
            assert time.monotonic() < deadline and process.poll() is None, process.stderr.read()
            time.sleep(0.05)
        assert json.loads(out.read_text(encoding="utf-8"))["incidents"] == []
        with source.open("a", encoding="utf-8") as file:
            file.write(lines(records[20:]))  # the failure arrives with no one at the keyboard
        while not calls():
            assert time.monotonic() < deadline and process.poll() is None, process.stderr.read()
            time.sleep(0.05)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=30)
    finally:
        process.kill()
    assert process.returncode == 0, stderr
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["detectors"] == "rust" and "compact" in written
    assert [(i["kind"], i["step"]) for i in written["incidents"]] == [("non_finite", 30)]
    (call,) = calls()
    assert call["payload"]["id"] == "r/rank0/non_finite/30" and call["payload"]["incident"] == written["incidents"][0]
    assert "NaN or Inf at step 30" in stdout and "trigger: non_finite at step 30 (rank 0): started the command" in stdout


def test_a_store_whose_report_cannot_be_used_is_judged_here_through_rust(tmp_path):
    from test_evidence import _steps, _write_parts

    records = _steps(1, range(60), moe={"r": [10, 10]}) + _steps(2, range(30, 50), moe={"r": [10, 10]})
    folder = tmp_path / "store" / "runs" / "r"
    _write_parts(folder, records)
    evidence = {}
    for engine in ("python", "rust"):
        out = watch(folder, tmp_path / f"{engine}.json", interval=0.0, polls=1, detectors=engine)
        evidence[engine] = json.loads(out.read_text(encoding="utf-8"))
    # Parts and no incidents.json: the detectors ran here, with the engine asked for.
    assert evidence["rust"]["detection"]["state"] == "missing" and evidence["rust"]["detection"] == evidence["python"]["detection"]
    assert same(evidence["python"], evidence["rust"]) == []
    # (The watcher names the store's status file, which is not there: a status was expected.)
    assert same(
        tensorquill.build_evidence(records, watchdog=folder / "watchdog.json"),
        {k: v for k, v in evidence["rust"].items() if k != "detection"},
    ) == []
    assert evidence["rust"]["kind_checks"]["job_stalled"]["value"] == "unavailable"


def test_a_missing_binary_is_said_before_the_first_look(tmp_path, monkeypatch):
    path = tmp_path / "w.jsonl"
    path.write_text(lines(everything()[:20]), encoding="utf-8")
    monkeypatch.setenv(BINARY_ENV, str(tmp_path / "missing"))
    looked = []
    with pytest.raises(RuntimeError, match="is not a file"):
        watch(path, interval=0.0, polls=3, detectors="rust", on_update=lambda *a: looked.append(a))
    assert looked == [] and not path.with_suffix(".evidence.json").exists()
    # A file that does not exist yet is the same: the check does not wait for the first record.
    with pytest.raises(RuntimeError, match="is not a file"):
        watch(tmp_path / "later.jsonl", interval=0.0, polls=1, detectors="rust")
    with pytest.raises(ValueError, match="'python' or 'rust'"):
        watch(path, interval=0.0, polls=1, detectors="server")
    done = subprocess.run([sys.executable, "-m", "tensorquill", str(path), "--watch", "--detectors", "rust"],
                          capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert done.returncode == 2 and "is not a file" in done.stderr and "Traceback" not in done.stderr
    # The Python detectors need neither.
    assert watch(path, interval=0.0, polls=1) == path.with_suffix(".evidence.json")


def test_a_binary_too_old_for_the_samples_is_said_and_no_evidence_is_written(tmp_path, monkeypatch):
    """The binary is there, so the check before the first look passes; it refuses `--samples`
    at the first rebuild. The watcher stops with the message that says to rebuild it."""
    path = tmp_path / "w.jsonl"
    path.write_text(lines(everything()[:20]), encoding="utf-8")
    old = tmp_path / "old-server"
    old.write_text("#!/bin/sh\necho \"error: unexpected argument '--samples' found\" >&2\nexit 2\n", encoding="utf-8")
    old.chmod(0o755)
    monkeypatch.setenv(BINARY_ENV, str(old))
    with pytest.raises(RuntimeError, match="too old to have `detect`, `--thresholds` or `--samples`: rebuild it"):
        watch(path, interval=0.0, polls=1, detectors="rust")
    assert not path.with_suffix(".evidence.json").exists()
    done = subprocess.run([sys.executable, "-m", "tensorquill", str(path), "--watch", "--detectors", "rust"],
                          capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert done.returncode == 2 and "rebuild it" in done.stderr and "Traceback" not in done.stderr
    assert not path.with_suffix(".evidence.json").exists()
