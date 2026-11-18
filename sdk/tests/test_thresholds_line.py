"""One printed line whenever the thresholds that were applied are not the defaults.

A reader of a server's report takes the thresholds the server applied (`test_reader_thresholds.py`).
So that this is never silent, the summary of the evidence (`incident_summary`, which is what the
command line prints for every look) names each applied threshold that is not the default, with
its value and the default, in one line (`evidence.thresholds_line`). At the defaults there is no
line and the summary is what it was.

The first tests need nothing but the package. Those that take the `store` fixture need `pyarrow`
(the `flight` extra) and the server binary, as `test_reader_thresholds.py` does, and are skipped
without them.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time

import pytest

import tensorquill
from tensorquill.evidence import THRESHOLD_DEFAULTS, THRESHOLD_NAMES, incident_summary, thresholds_line

# Every one differs from its default (the values of `test_reader_thresholds.APPLIED`).
APPLIED = {"overload_threshold": 0.5, "drift_tolerance": 1e-3, "starve_window": 20, "starve_ratio": 0.5, "top_k": 2}
NAMED = {
    "overload_threshold": "overload_threshold 0.5 (default: 0.35)",
    "drift_tolerance": "drift_tolerance 0.001 (default: 1e-06)",
    "starve_window": "starve_window 20 (default: 50)",
    "starve_ratio": "starve_ratio 0.5 (default: 0.1)",
    "top_k": "top_k 2 (default: what the run recorded)",
}
PREFIX = "Thresholds that are not the defaults: "
SERVER_LINE = PREFIX + "the server applied " + ", ".join(NAMED[name] for name in THRESHOLD_NAMES) + "."
HERE_LINE = PREFIX + "the detectors ran here with " + ", ".join(NAMED[name] for name in THRESHOLD_NAMES) + "."
# A small local capture: one rank, twenty steps, enough for a summary and a run state.
RECORDS = [{"run_id": "r", "rank": 0, "attempt": 1, "step": s, "grad_norm": 1.0} for s in range(20)]


def one_rank(thresholds, source="server"):
    """One rank's evidence from a server's store, as far as the line reads it."""
    return {"schema_version": 1, "run_id": "r", "incidents": [], "samples": [],
            "detection": {"source": source, "state": "current", "reason": "the server's report is current",
                          "rank": 0, "report_steps": 60, "thresholds": thresholds}}


def all_ranks(thresholds, by_rank=None):
    return {"view": "all_ranks", "run_id": "r", "rank_count": 2, "judged_ranks": [0, 1], "unknown_ranks": [],
            "clean_ranks": [0, 1], "incident_groups": [], "run_incidents": [], "ranks": [],
            "thresholds": thresholds, **({"thresholds_by_rank": by_rank} if by_rank else {})}


# --- at the defaults: no line, and the summary is what it was ----------------------------------


def test_at_the_defaults_there_is_no_line_in_any_evidence():
    defaults = dict(THRESHOLD_DEFAULTS)
    # As a file read back gives them, and one unit in the last place apart, which a float that
    # went to the server as JSON text may come back as: the same thresholds.
    read_back = json.loads(json.dumps(defaults))
    nudged = {**defaults, "drift_tolerance": math.nextafter(1e-6, 1.0), "overload_threshold": math.nextafter(0.35, 0.0)}
    for values in (defaults, read_back, nudged):
        for evidence in (one_rank(values), one_rank(values, "python"), all_ranks(values)):
            assert thresholds_line(evidence) is None
            assert "Thresholds" not in incident_summary(evidence) and "default" not in incident_summary(evidence)
    # A local file at the defaults lists none of them, as before the key, also when one was given
    # at its default value (see "a local file" below).
    for given in ({}, {"starve_window": 50, "drift_tolerance": 1e-6}):
        local = tensorquill.build_evidence(RECORDS, **given)
        assert "detection" not in local and "thresholds" not in local and thresholds_line(local) is None
    # Evidence that does not list what was applied says nothing: a view with no judged rank
    # (`thresholds` is None there), and a report from before the key.
    for evidence in (all_ranks(None), {"incidents": []}, {"incidents": [], "detection": {"source": "server"}}, one_rank("0.5")):
        assert thresholds_line(evidence) is None


def test_the_summary_at_the_defaults_is_unchanged_and_the_line_is_added_last():
    quiet = one_rank(dict(THRESHOLD_DEFAULTS))
    loud = one_rank({**THRESHOLD_DEFAULTS, "starve_window": 20})
    assert incident_summary(loud).splitlines() == [
        *incident_summary(quiet).splitlines(), PREFIX + "the server applied starve_window 20 (default: 50).",
    ]
    quiet, loud = all_ranks(dict(THRESHOLD_DEFAULTS)), all_ranks({**THRESHOLD_DEFAULTS, "top_k": {"router.b": 3, "router.a": 2}})
    assert incident_summary(loud).splitlines() == [
        *incident_summary(quiet).splitlines(),
        PREFIX + 'the server applied top_k {"router.a": 2, "router.b": 3} (default: what the run recorded).',
    ]


# --- off the defaults: each one that differs, with its value and the default ---------------------


@pytest.mark.parametrize("name", THRESHOLD_NAMES)
def test_only_the_threshold_that_differs_is_named_with_its_value_and_the_default(name):
    values = {**THRESHOLD_DEFAULTS, name: APPLIED[name]}
    for evidence in (one_rank(values), all_ranks(values)):
        assert thresholds_line(evidence) == f"{PREFIX}the server applied {NAMED[name]}."
        assert incident_summary(evidence).splitlines()[-1] == thresholds_line(evidence)
    # When the detectors ran here, with what the reader gave and the server's values for the rest.
    assert thresholds_line(one_rank(values, "python")) == f"{PREFIX}the detectors ran here with {NAMED[name]}."


def test_all_five_are_named_in_one_line_in_their_usual_order():
    for evidence in (one_rank(APPLIED), all_ranks(APPLIED)):
        assert thresholds_line(evidence) == SERVER_LINE and "\n" not in SERVER_LINE
        assert incident_summary(evidence).count(PREFIX) == 1


def test_ranks_judged_by_different_thresholds_are_each_named_with_their_ranks():
    by_rank = [
        {"ranks": [0, 1], "rank_ranges": "0-1", "thresholds": {**THRESHOLD_DEFAULTS, "starve_window": 20}},
        {"ranks": [2], "rank_ranges": "2", "thresholds": dict(THRESHOLD_DEFAULTS)},
        {"ranks": [3], "rank_ranges": "3", "thresholds": {**THRESHOLD_DEFAULTS, "overload_threshold": 0.5, "top_k": 2}},
    ]
    evidence = all_ranks(by_rank[0]["thresholds"], by_rank)
    # The ranks judged by the defaults are not named: nothing differs for them.
    assert thresholds_line(evidence) == (
        PREFIX + "the server applied on ranks 0-1: starve_window 20 (default: 50); on rank 3: overload_threshold 0.5"
        " (default: 0.35), top_k 2 (default: what the run recorded)."
    )
    # The first set at the defaults and another not: still said, though `thresholds` alone is the defaults.
    swapped = all_ranks(dict(THRESHOLD_DEFAULTS), [by_rank[1], by_rank[0]])
    assert thresholds_line(swapped) == PREFIX + "the server applied on ranks 0-1: starve_window 20 (default: 50)."


# --- a real server started with `--thresholds`, and one at the defaults ---------------------------


@pytest.fixture
def store():
    """`test_reader_thresholds`, whose helpers start the servers; every server is stopped after."""
    pytest.importorskip("pyarrow.flight")
    import test_reader_thresholds as helpers
    import test_watch_all_ranks

    if helpers.BINARY is None:
        pytest.skip("tensorquill-server is not built")
    yield helpers
    for server in [*helpers.STARTED, *test_watch_all_ranks.STARTED]:
        if server.process.poll() is None:
            server.process.kill()
            server.process.wait(20)
        if not server.process.stdout.closed:
            server.process.stdout.close()
    helpers.STARTED.clear()
    test_watch_all_ranks.STARTED.clear()


def test_one_ranks_evidence_from_a_server_with_other_thresholds_says_so_and_at_the_defaults_does_not(tmp_path, store):
    folder = store.judged_by(tmp_path / "other").incidents_file.parent
    evidence = store.one_look(folder, tmp_path / "other.json")
    assert evidence["detection"]["source"] == "server" and evidence["detection"]["thresholds"] == APPLIED
    assert thresholds_line(evidence) == SERVER_LINE and incident_summary(evidence).splitlines()[-1] == SERVER_LINE

    # A reader that gives one of its own, which differs: the detectors run here with it and with
    # the server's values for the rest, and the line says that.
    here = store.one_look(folder, tmp_path / "here.json", starve_window=50)
    assert here["detection"]["source"] == "python" and here["detection"]["state"] == "other_thresholds"
    assert thresholds_line(here) == PREFIX + "the detectors ran here with " + ", ".join(
        NAMED[name] for name in THRESHOLD_NAMES if name != "starve_window") + "."

    # At the defaults, given as nothing and as an empty object: the five are listed and no line is printed.
    for name, thresholds in (("none", None), ("empty", {})):
        folder = store.judged_by(tmp_path / name, thresholds).incidents_file.parent
        evidence = store.one_look(folder, tmp_path / f"{name}.json")
        assert evidence["detection"]["source"] == "server" and evidence["detection"]["thresholds"] == THRESHOLD_DEFAULTS
        assert thresholds_line(evidence) is None and "Thresholds that are not the defaults" not in incident_summary(evidence)


def test_the_all_ranks_view_says_so_and_at_the_defaults_does_not(tmp_path, store):
    run = store.Run(tmp_path / "other", *store.FAST, "--thresholds", json.dumps(APPLIED))
    records = store.everything()
    for rank in range(3):
        run.send(rank, records[:200])
    evidence, summary = store.all_ranks(run, tmp_path / "other.json")
    assert evidence["judged_ranks"] == [0, 1, 2] and evidence["thresholds"] == APPLIED
    assert summary.splitlines()[-1] == SERVER_LINE == thresholds_line(evidence)

    # The server comes back at its defaults and only rank 2 sends again: ranks 0 and 1 are still
    # judged by the other values, and the line names them and not rank 2.
    run.restart(*store.FAST)
    run.send(2, records[200:])
    assert run.server.terminate() == 0
    evidence, summary = store.all_ranks(run, tmp_path / "mixed.json")
    assert len(evidence["thresholds_by_rank"]) == 2
    assert summary.splitlines()[-1] == SERVER_LINE.replace("the server applied ", "the server applied on ranks 0-1: ")

    quiet = store.Run(tmp_path / "defaults", *store.FAST)
    for rank in range(2):
        quiet.send(rank, records)
    assert quiet.server.terminate() == 0
    evidence, summary = store.all_ranks(quiet, tmp_path / "defaults.json")
    assert evidence["judged_ranks"] == [0, 1] and evidence["thresholds"] == THRESHOLD_DEFAULTS
    assert thresholds_line(evidence) is None and "Thresholds that are not the defaults" not in summary


def test_the_command_line_prints_the_line_for_every_look_it_reports(tmp_path, store):
    def watched(folder, out):
        """What `--watch` on one rank prints for its first look."""
        process = subprocess.Popen(
            [sys.executable, "-m", "tensorquill", str(folder), "--watch", "--interval", "0.2", "-o", str(out)],
            cwd=tmp_path, stdout=subprocess.PIPE, text=True,
            # Since Nov 17, 2026 `--watch` needs the Rust detectors' binary before its first look: it is told where
            # the binary this test's server runs from is, as an installed tensorquill-server would be found.
            env={**os.environ, "TENSORQUILL_SERVER_BIN": str(store.BINARY)},
            # A process started from a background shell job inherits SIGINT as ignored.
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
        )
        try:
            deadline = time.monotonic() + 60
            while not out.exists():
                assert time.monotonic() < deadline and process.poll() is None
                time.sleep(0.1)
            time.sleep(0.5)  # the lines of that look are printed after the file is written
            process.send_signal(signal.SIGINT)
            printed, _ = process.communicate(timeout=30)
        finally:
            process.kill()
        return printed

    run = store.Run(tmp_path / "other", *store.FAST, "--thresholds", json.dumps(APPLIED))
    quiet = store.Run(tmp_path / "defaults", *store.FAST)
    for started in (run, quiet):
        for rank in range(2):
            started.send(rank, store.everything())
        assert started.server.terminate() == 0
    cwd = tmp_path / "cwd"
    cwd.mkdir()

    # One look at every rank, and a watcher of one rank: each prints the line once, with its summary.
    looked = store.cli(str(run.folder), "--rank", "all", cwd=cwd)
    assert looked.returncode == 0, looked.stderr
    assert looked.stdout.count(SERVER_LINE + "\n") == 1
    printed = watched(run.rank_folder(1), tmp_path / "one.json")
    assert printed.count(SERVER_LINE + "\n") == 1
    assert printed.index(SERVER_LINE) < printed.index("Run state:")  # part of the summary, before the state

    # At the defaults neither prints it.
    looked = store.cli(str(quiet.folder), "--rank", "all", cwd=cwd)
    assert looked.returncode == 0, looked.stderr
    for text in (looked.stdout, watched(quiet.rank_folder(1), tmp_path / "quiet.json")):
        assert "Run state:" in text and "Thresholds that are not the defaults" not in text and "(default" not in text


# --- a local file: the detectors ran here with the options typed -----------------------------------
#
# `build_evidence` lists the five it ran with as `thresholds` (the name a server's report gives
# them) only when one is not the default, so the line is printed for a local `.jsonl` file and
# for a store's parts read without `--watch`, and at the defaults the evidence and what is
# printed are what they were.

FLAGS = {"overload_threshold": "--overload-threshold", "drift_tolerance": "--drift-tolerance",
         "starve_window": "--starve-window", "starve_ratio": "--starve-ratio", "top_k": "--top-k"}
# Each typed at its default value (`top_k`'s default, what the run recorded, has no value to type).
TYPED_DEFAULTS = {name: value for name, value in THRESHOLD_DEFAULTS.items() if value is not None}
ONE_READ_MODES = {"plain": [], "rank": ["--rank", "0"], "compact": ["--compact"]}


def flags(values) -> list[str]:
    return [text for name, value in values.items() for text in (FLAGS[name], str(value))]


def capture(folder):
    path = folder / "run.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in RECORDS), encoding="utf-8")
    return path


def command(*args, cwd, env=None) -> subprocess.CompletedProcess:
    cwd.mkdir(parents=True, exist_ok=True)
    return subprocess.run([sys.executable, "-m", "tensorquill", *map(str, args)], cwd=cwd, capture_output=True,
                          text=True, timeout=120, env=env)


def watched(*args, out, cwd, env=None) -> str:
    """What `--watch` prints for its first look, written to `out`."""
    cwd.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [sys.executable, "-m", "tensorquill", *map(str, args), "--watch", "--interval", "0.2", "-o", str(out)],
        cwd=cwd, stdout=subprocess.PIPE, text=True, env=env,
        # A process started from a background shell job inherits SIGINT as ignored.
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
    )
    try:
        deadline = time.monotonic() + 60
        while not out.exists():
            assert time.monotonic() < deadline and process.poll() is None
            time.sleep(0.1)
        time.sleep(0.5)  # the lines of that look are printed after the file is written
        process.send_signal(signal.SIGINT)
        printed, _ = process.communicate(timeout=30)
    finally:
        process.kill()
    return printed


def test_a_local_files_evidence_lists_what_the_detectors_ran_with_when_one_is_not_the_default():
    for name in THRESHOLD_NAMES:
        evidence = tensorquill.build_evidence(RECORDS, **{name: APPLIED[name]})
        assert evidence["thresholds"] == {**THRESHOLD_DEFAULTS, name: APPLIED[name]}
        assert thresholds_line(evidence) == f"{PREFIX}the detectors ran here with {NAMED[name]}."
        assert incident_summary(evidence).splitlines()[-1] == thresholds_line(evidence)
    evidence = tensorquill.build_evidence(RECORDS, **APPLIED)
    assert list(evidence["thresholds"]) == list(THRESHOLD_NAMES) and evidence["thresholds"] == APPLIED
    assert thresholds_line(evidence) == HERE_LINE == thresholds_line(json.loads(json.dumps(evidence)))
    named = tensorquill.build_evidence(RECORDS, top_k={"router.b": 3, "router.a": 2})
    assert named["thresholds"]["top_k"] == {"router.b": 3, "router.a": 2}
    assert thresholds_line(named) == (
        PREFIX + 'the detectors ran here with top_k {"router.a": 2, "router.b": 3} (default: what the run recorded).')
    # The key is all that is added: nothing else moves.
    plain, loud = tensorquill.build_evidence(RECORDS), tensorquill.build_evidence(RECORDS, drift_tolerance=1e-3)
    assert loud.pop("thresholds") and loud == plain


@pytest.mark.parametrize("mode", ONE_READ_MODES.values(), ids=ONE_READ_MODES.keys())
def test_one_read_of_a_local_file_prints_the_line_and_at_the_defaults_prints_what_it_did(tmp_path, mode):
    path = capture(tmp_path)
    loud = command(path, "-o", tmp_path / "loud.json", *mode, *flags(APPLIED), cwd=tmp_path / "loud")
    assert loud.returncode == 0, loud.stderr
    assert loud.stdout.count(HERE_LINE + "\n") == 1
    assert loud.stdout.index(HERE_LINE) < loud.stdout.index("Run state:")  # part of the summary, before the state
    assert json.loads((tmp_path / "loud.json").read_text(encoding="utf-8"))["thresholds"] == APPLIED

    # At the defaults, none typed and each typed at its default value: the same output, byte for
    # byte, and the same file, with no line and no key.
    printed, written = [], []
    for name, typed in (("none", []), ("typed", flags(TYPED_DEFAULTS))):
        done = command(path, "-o", "evidence.json", *mode, *typed, cwd=tmp_path / name)
        assert done.returncode == 0, done.stderr
        printed.append(done.stdout)
        written.append((tmp_path / name / "evidence.json").read_bytes())
    assert printed[0] == printed[1] and written[0] == written[1]
    assert "Run state:" in printed[0] and PREFIX not in printed[0] and "(default" not in printed[0]
    assert "thresholds" not in json.loads(written[0])


@pytest.mark.parametrize("mode", [[], ["--compact"]], ids=["plain", "compact"])
def test_watching_a_local_file_prints_the_line_and_at_the_defaults_does_not(tmp_path, mode):
    path = capture(tmp_path)
    # Since Nov 17, 2026 `--watch` needs the Rust detectors unless `--detectors python` is typed. This
    # test needs nothing but the package, so it types it; the next test watches through Rust.
    python = ("--detectors", "python")
    loud = watched(path, *mode, *python, *flags(APPLIED), out=tmp_path / "loud.json", cwd=tmp_path)
    assert loud.count(HERE_LINE + "\n") == 1 and loud.index(HERE_LINE) < loud.index("Run state:")
    quiet = watched(path, *mode, *python, *flags(TYPED_DEFAULTS), out=tmp_path / "quiet.json", cwd=tmp_path)
    assert "Run state:" in quiet and PREFIX not in quiet and "(default" not in quiet
    assert "thresholds" not in json.loads((tmp_path / "quiet.json").read_text(encoding="utf-8"))


def test_the_rust_detectors_on_a_local_file_print_the_same_line_as_the_python_ones(tmp_path, store):
    path = capture(tmp_path)
    env = {**os.environ, "TENSORQUILL_SERVER_BIN": str(store.BINARY)}
    for name, mode in ONE_READ_MODES.items():
        for detectors in ("python", "rust"):
            out = tmp_path / f"{name}-{detectors}.json"
            done = command(path, "-o", out, *mode, "--detectors", detectors, *flags(APPLIED), cwd=tmp_path, env=env)
            assert done.returncode == 0, done.stderr
            assert done.stdout.count(HERE_LINE + "\n") == 1, (name, detectors)
            assert json.loads(out.read_text(encoding="utf-8"))["thresholds"] == APPLIED
        quiet = command(path, "-o", tmp_path / "quiet.json", *mode, "--detectors", "rust", cwd=tmp_path, env=env)
        assert quiet.returncode == 0, quiet.stderr
        assert "Run state:" in quiet.stdout and PREFIX not in quiet.stdout
        assert "thresholds" not in json.loads((tmp_path / "quiet.json").read_text(encoding="utf-8"))
    loud = watched(path, "--detectors", "rust", *flags(APPLIED), out=tmp_path / "watched.json", cwd=tmp_path, env=env)
    assert loud.count(HERE_LINE + "\n") == 1
    quiet = watched(path, "--detectors", "rust", out=tmp_path / "watched-quiet.json", cwd=tmp_path, env=env)
    assert "Run state:" in quiet and PREFIX not in quiet


def test_a_read_of_a_stores_parts_without_watch_prints_the_line_for_the_options_typed(tmp_path, store):
    folder = store.judged_by(tmp_path / "defaults", None).incidents_file.parent
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    loud = store.cli(str(folder), "-o", str(tmp_path / "loud.json"), *flags(APPLIED), cwd=cwd)
    assert loud.returncode == 0, loud.stderr
    assert loud.stdout.count(HERE_LINE + "\n") == 1
    quiet = store.cli(str(folder), "-o", str(tmp_path / "quiet.json"), cwd=cwd)
    assert quiet.returncode == 0, quiet.stderr
    assert "Run state:" in quiet.stdout and PREFIX not in quiet.stdout
