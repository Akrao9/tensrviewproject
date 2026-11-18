"""A reader of a server's report and the server's thresholds.

The rule (Nov 16, 2026, open to change): a reader takes the thresholds the server applied,
and a report is refused (`other_thresholds`) only for a threshold the reader gave itself that
differs from the applied one. Three cases, each through `read_server_report`, the one-rank
watcher, the all-ranks view and the command line:

- adopt: no threshold given, the report is current and the evidence lists what was applied;
- match: the thresholds given are the applied ones;
- refuse: one given differs. The one-rank watcher then runs the detectors here with it and
  keeps the server's values for the rest; the all-ranks view lists the rank as unknown.

Also here: the command line has a flag for each of the six, and `--rank all` takes `--top-k`.

Needs `pyarrow` (the `flight` extra) and the server binary
(`cargo build --manifest-path server/Cargo.toml`, or `TENSORQUILL_SERVER_BIN`).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")

import tensorquill  # noqa: E402
from tensorquill.evidence import (  # noqa: E402
    SERVER_KINDS, THRESHOLD_DEFAULTS, THRESHOLD_NAMES, applied_thresholds, incident_summary, read_server_report, watch,
)

import test_watch_all_ranks  # noqa: E402
from test_flight_e2e import BINARY, Server  # noqa: E402
from test_server_detectors import FAST, RUN, Stream, differences, everything  # noqa: E402
from test_watch_all_ranks import Run  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")

# Every one differs from its default, and together they change what `everything()` reports.
APPLIED = {"overload_threshold": 0.5, "drift_tolerance": 1e-3, "starve_window": 20, "starve_ratio": 0.5, "top_k": 2,
           "update_warmup": 30}
# One value of each that the server did not apply.
OTHER = {"overload_threshold": 0.35, "drift_tolerance": 1e-6, "starve_window": 50, "starve_ratio": 0.1, "top_k": 3,
         "update_warmup": 1000}
FLAGS = {"overload_threshold": "--overload-threshold", "drift_tolerance": "--drift-tolerance",
         "starve_window": "--starve-window", "starve_ratio": "--starve-ratio", "top_k": "--top-k",
         "update_warmup": "--update-warmup"}
STARTED: list[Server] = []


@pytest.fixture(autouse=True)
def servers_are_stopped():
    yield
    for server in [*STARTED, *test_watch_all_ranks.STARTED]:
        if server.process.poll() is None:
            server.process.kill()
            server.process.wait(20)
        if not server.process.stdout.closed:
            server.process.stdout.close()
    STARTED.clear()
    test_watch_all_ranks.STARTED.clear()


def judged_by(tmp_path: Path, thresholds: dict | None = APPLIED) -> Stream:
    """One rank of `everything()`, stored and reported on by a server started with `thresholds`,
    which has stopped."""
    arguments = [*FAST, *(("--thresholds", json.dumps(thresholds)) if thresholds is not None else ())]
    server = Server(tmp_path / "store", 0, *arguments)
    STARTED.append(server)
    stream = Stream(tmp_path, *arguments, server=server)
    stream.send(everything())
    assert server.terminate() == 0
    return stream


def kinds_and_steps(incidents) -> list[tuple]:
    # A logged NaN is in these incidents, so they are compared by kind and step.
    return [(i["kind"], i["step"]) for i in incidents if i["kind"] in SERVER_KINDS]


def here(**thresholds) -> list[tuple]:
    return kinds_and_steps(tensorquill.build_evidence(everything(), **thresholds)["incidents"])


def one_look(folder: Path, out: Path, **options) -> dict:
    written = watch(folder, out, interval=0.0, polls=1, **options)
    assert written is not None, "the watcher wrote nothing"
    return json.loads(written.read_text(encoding="utf-8"))


def cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"}
    # `--detectors rust` looks for the binary in the environment or on PATH; a local run that names
    # neither still has the one this file was collected with.
    env.setdefault("TENSORQUILL_SERVER_BIN", str(BINARY))
    return subprocess.run([sys.executable, "-m", "tensorquill", *args], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=120)


def flags(thresholds: dict) -> list[str]:
    return [text for name, value in thresholds.items() for text in (FLAGS[name], str(value))]


def test_the_thresholds_change_what_this_run_reports():
    # Otherwise "the server's incidents" and "the defaults' incidents" could not be told apart below.
    assert here(**APPLIED) != here()
    assert here(**{**APPLIED, "starve_window": 50}) != here(**APPLIED)


# --- the report ----------------------------------------------------------------------------


def test_a_reader_that_gives_no_threshold_takes_the_servers(tmp_path):
    folder = judged_by(tmp_path).incidents_file.parent
    look = read_server_report(folder)
    assert look["state"] == "current"
    assert applied_thresholds(look["report"]) == APPLIED
    # `top_k=None` is "not given", as for the other four: it is not "what the run recorded".
    assert read_server_report(folder, top_k=None, overload_threshold=None)["state"] == "current"


@pytest.mark.parametrize("given", [APPLIED, *({name: APPLIED[name]} for name in THRESHOLD_NAMES)],
                         ids=["all six", *THRESHOLD_NAMES])
def test_a_reader_that_gives_the_applied_thresholds_is_current(tmp_path, given):
    folder = judged_by(tmp_path).incidents_file.parent
    assert read_server_report(folder, **given)["state"] == "current"


@pytest.mark.parametrize("name", THRESHOLD_NAMES)
def test_a_reader_is_refused_only_for_a_threshold_it_gave_that_differs(tmp_path, name):
    folder = judged_by(tmp_path).incidents_file.parent
    look = read_server_report(folder, **{name: OTHER[name]})
    assert look["state"] == "other_thresholds"
    # The reason names the one that differs, and none of the four that were taken.
    assert look["reason"] == f"the server applied {name} {APPLIED[name]!r} where {OTHER[name]!r} was asked for"
    # Beside four that match, it is still refused.
    assert read_server_report(folder, **{**APPLIED, name: OTHER[name]})["state"] == "other_thresholds"


def test_a_server_at_the_defaults_is_refused_for_a_threshold_given_and_taken_without(tmp_path):
    folder = judged_by(tmp_path, None).incidents_file.parent
    look = read_server_report(folder)
    assert look["state"] == "current" and applied_thresholds(look["report"]) == THRESHOLD_DEFAULTS
    assert read_server_report(folder, **{k: v for k, v in THRESHOLD_DEFAULTS.items() if v is not None})["state"] == "current"
    assert read_server_report(folder, top_k=2)["state"] == "other_thresholds"


def test_applied_thresholds_reads_what_a_report_lists_and_the_default_for_what_it_does_not():
    assert applied_thresholds(None) == applied_thresholds({}) == THRESHOLD_DEFAULTS
    # A server from before `--thresholds` listed no `top_k`.
    older = {"thresholds": {"overload_threshold": 0.35, "drift_tolerance": 1e-6, "starve_window": 50, "starve_ratio": 0.1,
                            "spike_sustain": 3}}
    assert applied_thresholds(older) == THRESHOLD_DEFAULTS
    listed = {"thresholds": {**APPLIED, "top_k": {"r1": 2, "r2": 8}, "spike_factor": 5.0}}
    assert applied_thresholds(listed) == {**APPLIED, "top_k": {"r1": 2, "r2": 8}}
    # Values no rule could have applied are not passed on to the detectors here.
    odd = {"thresholds": {"overload_threshold": "0.5", "drift_tolerance": -1, "starve_window": 0, "starve_ratio": True,
                          "top_k": {"r1": 0}}}
    assert applied_thresholds(odd) == THRESHOLD_DEFAULTS


# --- the one-rank watcher ------------------------------------------------------------------


def test_the_watcher_adopts_matches_and_recomputes_with_its_own(tmp_path):
    stream = judged_by(tmp_path)
    folder = stream.incidents_file.parent

    adopted = one_look(folder, tmp_path / "adopt.json")
    assert (adopted["detection"]["source"], adopted["detection"]["state"]) == ("server", "current")
    assert adopted["detection"]["thresholds"] == APPLIED
    assert kinds_and_steps(adopted["incidents"]) == here(**APPLIED)
    assert (adopted["overload_threshold"], adopted["starve_window"], adopted["starve_ratio"]) == (0.5, 20, 0.5)

    matched = one_look(folder, tmp_path / "match.json", **APPLIED)
    assert (matched["detection"]["source"], matched["detection"]["state"]) == ("server", "current")
    assert matched["incidents"] == adopted["incidents"] and matched["detection"]["thresholds"] == APPLIED

    # One of its own that differs: the detectors run here with it, and with the server's four for the rest.
    refused = one_look(folder, tmp_path / "refuse.json", starve_window=50)
    assert (refused["detection"]["source"], refused["detection"]["state"]) == ("python", "other_thresholds")
    assert refused["detection"]["reason"] == "the server applied starve_window 20 where 50 was asked for"
    assert refused["detection"]["thresholds"] == {**APPLIED, "starve_window": 50}
    assert kinds_and_steps(refused["incidents"]) == here(**{**APPLIED, "starve_window": 50})
    assert (refused["overload_threshold"], refused["starve_window"], refused["starve_ratio"]) == (0.5, 50, 0.5)
    assert set(refused["top_k"].values()) == {2}


def test_a_watcher_that_adopted_keeps_the_servers_thresholds_when_it_has_to_judge_here(tmp_path):
    stream = judged_by(tmp_path)
    folder = stream.incidents_file.parent
    seen: list[dict] = []

    def lose_the_report(evidence, path):
        seen.append(evidence)
        if len(seen) == 1:
            stream.incidents_file.unlink()  # as after a restart with --no-detect and a cleaned store

    watch(folder, tmp_path / "live.json", interval=0.0, polls=3, report_grace=0.0, on_update=lose_the_report)
    assert [(e["detection"]["source"], e["detection"]["state"]) for e in seen] == [("server", "current"), ("python", "missing")]
    # The incidents did not move with who judged them: a trigger would not fire again.
    assert seen[1]["detection"]["thresholds"] == APPLIED
    assert kinds_and_steps(seen[1]["incidents"]) == kinds_and_steps(seen[0]["incidents"]) == here(**APPLIED)

    # A watcher that never saw the server's report has nothing to take: the defaults.
    fresh = one_look(folder, tmp_path / "fresh.json")
    assert (fresh["detection"]["source"], fresh["detection"]["state"]) == ("python", "missing")
    assert fresh["detection"]["thresholds"] == THRESHOLD_DEFAULTS
    assert kinds_and_steps(fresh["incidents"]) == here()


# --- the all-ranks view --------------------------------------------------------------------


def all_ranks(run: Run, out: Path, **options) -> tuple[dict, str]:
    evidence = one_look(run.folder, out, rank="all", **options)
    return evidence, incident_summary(evidence)


def test_the_all_ranks_view_adopts_matches_and_lists_a_refused_rank_as_unknown(tmp_path):
    run = Run(tmp_path, *FAST, "--thresholds", json.dumps(APPLIED))
    for rank in range(2):
        run.send(rank, everything())
    assert run.server.terminate() == 0

    evidence, _ = all_ranks(run, tmp_path / "adopt.json")
    assert evidence["judged_ranks"] == [0, 1] and evidence["thresholds"] == APPLIED
    assert "thresholds_by_rank" not in evidence
    assert sorted((g["kind"], g["step"]) for g in evidence["incident_groups"]) == sorted(here(**APPLIED))

    matched, _ = all_ranks(run, tmp_path / "match.json", **APPLIED)
    assert matched["judged_ranks"] == [0, 1] and matched["incident_groups"] == evidence["incident_groups"]
    # `top_k` was refused with rank="all" before this rule: it is compared like the others.
    assert all_ranks(run, tmp_path / "top.json", top_k=2)[0]["judged_ranks"] == [0, 1]

    for name in THRESHOLD_NAMES:
        refused, summary = all_ranks(run, tmp_path / f"refuse-{name}.json", **{name: OTHER[name]})
        assert refused["judged_ranks"] == [] and refused["ranks_by_state"] == {"other_thresholds": [0, 1]}
        assert refused["thresholds"] is None and refused["incident_groups"] == []
        assert f"(other_thresholds): the server applied {name} {APPLIED[name]!r} where {OTHER[name]!r} was asked for" in summary


def test_ranks_judged_by_different_thresholds_are_both_taken_and_the_evidence_says_which(tmp_path):
    run = Run(tmp_path, *FAST, "--thresholds", json.dumps(APPLIED))
    records = everything()
    for rank in range(3):
        run.send(rank, records[:200])
    # The server comes back with its defaults, and only rank 2 sends again: its report is rebuilt.
    run.restart(*FAST)
    run.send(2, records[200:])
    assert run.server.terminate() == 0

    evidence, summary = all_ranks(run, tmp_path / "all.json")
    assert evidence["judged_ranks"] == [0, 1, 2]
    assert evidence["thresholds"] == APPLIED  # the first judged rank's
    assert evidence["thresholds_by_rank"] == [
        {"ranks": [0, 1], "rank_ranges": "0-1", "thresholds": APPLIED},
        {"ranks": [2], "rank_ranges": "2", "thresholds": THRESHOLD_DEFAULTS},
    ]
    lines = summary.splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("The judged ranks were not all judged by the same thresholds"))
    assert lines[at + 1] == ("  - ranks 0-1: overload_threshold 0.5, drift_tolerance 0.001, starve_window 20,"
                             " starve_ratio 0.5, top_k 2")
    assert lines[at + 2] == ("  - rank 2: overload_threshold 0.35, drift_tolerance 1e-06, starve_window 50,"
                             " starve_ratio 0.1, top_k None")
    # A reader that gives the first server's values is refused for the rank the second one judged.
    given, _ = all_ranks(run, tmp_path / "given.json", **APPLIED)
    assert given["ranks_by_state"] == {"current": [0, 1], "other_thresholds": [2]} and "thresholds_by_rank" not in given


# --- the command line ----------------------------------------------------------------------


def test_the_command_line_gives_only_the_thresholds_typed(tmp_path):
    run = Run(tmp_path, *FAST, "--thresholds", json.dumps(APPLIED))
    for rank in range(2):
        run.send(rank, everything())
    assert run.server.terminate() == 0
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    written = cwd / f"{RUN}.all-ranks.evidence.json"

    # Adopt: nothing typed, so nothing is compared (the command line once always gave 0.35).
    adopt = cli(str(run.folder), "--rank", "all", cwd=cwd)
    assert adopt.returncode == 0, adopt.stderr
    evidence = json.loads(written.read_text(encoding="utf-8"))
    assert evidence["judged_ranks"] == [0, 1] and evidence["thresholds"] == APPLIED
    assert "other_thresholds" not in adopt.stdout

    # Match: all six have a flag, `--top-k` among them with `--rank all`.
    match = cli(str(run.folder), "--rank", "all", *flags(APPLIED), cwd=cwd)
    assert match.returncode == 0, match.stderr
    assert json.loads(written.read_text(encoding="utf-8"))["judged_ranks"] == [0, 1]

    # Refuse: each flag reaches the reader.
    for name in THRESHOLD_NAMES:
        refuse = cli(str(run.folder), "--rank", "all", FLAGS[name], str(OTHER[name]), cwd=cwd)
        assert refuse.returncode == 0, refuse.stderr
        assert json.loads(written.read_text(encoding="utf-8"))["ranks_by_state"] == {"other_thresholds": [0, 1]}
        assert f"the server applied {name} {APPLIED[name]!r} where {OTHER[name]!r} was asked for" in refuse.stdout


@pytest.mark.parametrize("typed, source, state", [
    ((), "server", "current"),
    (tuple(flags(APPLIED)), "server", "current"),
    (("--starve-window", "50"), "python", "other_thresholds"),
], ids=["adopt", "match", "refuse"])
def test_the_command_lines_watcher_of_one_rank(tmp_path, typed, source, state):
    folder = judged_by(tmp_path).incidents_file.parent
    out = tmp_path / "w.json"
    # A process started from a background shell job inherits SIGINT as ignored: the child gets the default back.
    process = subprocess.Popen(
        [sys.executable, "-m", "tensorquill", str(folder), "--watch", "--interval", "0.2", "-o", str(out), *typed],
        cwd=tmp_path, stdout=subprocess.PIPE, text=True, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
    )
    try:
        deadline = time.monotonic() + 60
        while not out.exists():
            assert time.monotonic() < deadline and process.poll() is None
            time.sleep(0.1)
        process.send_signal(signal.SIGINT)
        process.communicate(timeout=30)
    finally:
        process.kill()
    evidence = json.loads(out.read_text(encoding="utf-8"))
    assert (evidence["detection"]["source"], evidence["detection"]["state"]) == (source, state)
    expected = {**APPLIED, "starve_window": 50} if state == "other_thresholds" else APPLIED
    assert evidence["detection"]["thresholds"] == expected and kinds_and_steps(evidence["incidents"]) == here(**expected)


def test_each_of_the_six_flags_reaches_the_detectors_on_a_local_file(tmp_path):
    def text(evidence):  # a logged NaN is in these incidents: as text it equals itself
        return json.dumps(evidence, sort_keys=True)

    capture = tmp_path / "run.jsonl"
    capture.write_text("".join(json.dumps(r) + "\n" for r in everything()), encoding="utf-8")
    done = cli(str(capture), "-o", str(tmp_path / "five.json"), *flags(APPLIED), cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    written = json.loads((tmp_path / "five.json").read_text(encoding="utf-8"))
    assert text(written) == text(tensorquill.build_evidence(everything(), **APPLIED))
    assert text(written) != text(tensorquill.build_evidence(everything()))
    # One at a time, each flag gives what its option gives, and with none the defaults stand.
    for name in THRESHOLD_NAMES:
        done = cli(str(capture), "-o", str(tmp_path / "one.json"), FLAGS[name], str(APPLIED[name]), cwd=tmp_path)
        assert done.returncode == 0, done.stderr
        written = json.loads((tmp_path / "one.json").read_text(encoding="utf-8"))
        assert text(written) == text(tensorquill.build_evidence(everything(), **{name: APPLIED[name]})), name
    done = cli(str(capture), "-o", str(tmp_path / "none.json"), cwd=tmp_path)
    written = json.loads((tmp_path / "none.json").read_text(encoding="utf-8"))
    assert text(written) == text(tensorquill.build_evidence(everything()))
    # The same six through the Rust detectors.
    rust = cli(str(capture), "-o", str(tmp_path / "rust.json"), "--detectors", "rust", *flags(APPLIED), cwd=tmp_path)
    assert rust.returncode == 0, rust.stderr
    written = json.loads((tmp_path / "rust.json").read_text(encoding="utf-8"))
    assert written.pop("detectors") == "rust"
    assert differences(tensorquill.build_evidence(everything(), **APPLIED), written, "evidence") == []
