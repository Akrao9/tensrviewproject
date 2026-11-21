"""Checks of the TrainCheck bug runs that need no long training: the catalog has the 20 rows,
the committed results cover every configuration it names, and the harness works end to end on
one short run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from catalog import BUGS, SHARED_RUN  # noqa: E402

HOW = {"shipped", "swapped", "recreated", "simulated", "not run"}


def test_the_catalog_has_one_row_for_each_of_the_20_bugs() -> None:
    assert [b["id"] for b in BUGS] == [f"B{i}" for i in range(1, 21)]
    assert len({b["paper"] for b in BUGS}) == 20
    for bug in BUGS:
        assert bug["how"] in HOW, bug["id"]
        assert bug["synopsis"] and bug["in_clone"] and bug["cpu"], bug["id"]
        if bug["how"] == "not run":
            assert bug["script"] is None and bug.get("needs") and bug.get("reach"), bug["id"]
        else:
            assert (HERE / bug["script"]).is_file(), bug["id"]
            assert bug.get("runs") or bug["id"] in SHARED_RUN, bug["id"]


def test_the_committed_results_cover_every_configuration() -> None:
    results = json.loads((HERE / "results.json").read_text())
    for bug in BUGS:
        for tag, _ in bug.get("runs", []):
            for variant in ("buggy", "healthy"):
                assert f"{bug['id']}|{tag}|{variant}" in results, (bug["id"], tag, variant)
    table = (HERE / "RESULTS.md").read_text()
    for bug in BUGS:
        assert f"| {bug['id']} {bug['paper']} |" in table


def test_the_harness_runs_a_script_and_builds_its_evidence() -> None:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(HERE.parent.parent / "sdk" / "src"), str(HERE)]))
    proc = subprocess.run(
        [sys.executable, "b19_pt51800.py", "--variant", "buggy", "--tag", "pytest", "--steps", "12"],
        cwd=HERE, env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(line for line in proc.stdout.splitlines() if line.startswith("RESULT "))
    result = json.loads((HERE / "out" / "B19-PT-51800-buggy-pytest" / "result.json").read_text())
    assert json.loads(line[len("RESULT "):])["run_id"] == result["run_id"]
    assert result["symptom"]["steps"] == 12
    assert result["symptom"]["bug_present"] is True
    rank0 = result["ranks"]["rank0"]
    assert rank0["run_state"] in {"healthy", "unhealthy", "degraded", "unknown"}
    assert "activation_spike" in rank0["relevant_kinds"]


# --- Lane U8 (audit X03): a catch is credited only when both runs completed with valid
# observations; a failed, nonzero or timed-out run is recorded for its own case. Every test
# points OUT, RESULTS and TABLE at its own temporary folder, so the committed files are never
# written; no training runs.

import run_all  # noqa: E402

CHILD = r"""
import json, os, subprocess, sys, time
from pathlib import Path
variant = sys.argv[sys.argv.index("--variant") + 1]
mode = os.environ["FAKE_MODE_" + variant.upper()]
if mode == "hang":
    # A grandchild that holds the output open, as a two-process run's ranks do.
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    time.sleep(120)
rid = "fake-" + variant
folder = Path(os.environ["FAKE_OUT"]) / rid
folder.mkdir(parents=True, exist_ok=True)
result = {"run_id": rid, "world_size": 1, "settings": {}, "symptom": {"bug_present": variant == "buggy"},
          "ranks": {"rank0": {"incidents": [{"kind": "replica_drift", "step": 3}] if variant == "buggy" else []}}}
if mode == "no_symptom":
    result["symptom"] = {}
if mode != "no_result_json":
    (folder / "result.json").write_text(json.dumps(result))
print("RESULT " + (json.dumps({"run_id": rid}) if mode != "bad_line" else "{not json"), flush=True)
if mode == "crash":
    raise RuntimeError("the run failed")
sys.exit(2 if mode == "exit2" else 0)
"""


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    child = tmp_path / "child.py"
    child.write_text(CHILD)
    monkeypatch.setattr(run_all, "OUT", tmp_path / "out")
    monkeypatch.setattr(run_all, "RESULTS", tmp_path / "results.json")
    monkeypatch.setattr(run_all, "TABLE", tmp_path / "RESULTS.md")
    monkeypatch.setenv("FAKE_OUT", str(tmp_path / "out"))

    def run(buggy="ok", healthy="ok"):
        monkeypatch.setenv("FAKE_MODE_BUGGY", buggy)
        monkeypatch.setenv("FAKE_MODE_HEALTHY", healthy)
        return run_all.run_one(str(child), "buggy", "", []), run_all.run_one(str(child), "healthy", "", [])
    return run, child


def test_a_completed_pair_credits_the_buggy_runs_own_incident(sweep):
    run, _ = sweep
    buggy, healthy = run()
    assert buggy["ok"] and healthy["ok"] and run_all.problem(buggy) is None
    pair = run_all._pair({"B9||buggy": buggy, "B9||healthy": healthy}, "B9", "")
    assert pair["caught"] == [("replica_drift", 3, "")] and pair["false"] == [] and "why" not in pair


@pytest.mark.parametrize("mode,reason", [
    ("exit2", "exited with status 2 after printing its result"),
    ("crash", "exited with status 1 after printing its result"),
    ("bad_line", "its RESULT line or result.json cannot be read"),
    ("no_result_json", "its RESULT line or result.json cannot be read"),
    ("no_symptom", "the symptom was not measured"),
])
def test_a_healthy_control_that_did_not_complete_credits_no_catch(sweep, mode, reason):
    run, _ = sweep
    buggy, healthy = run(healthy=mode)
    assert buggy["ok"] and not healthy["ok"] and healthy["error"].startswith(reason), healthy
    pair = run_all._pair({"B9||buggy": buggy, "B9||healthy": healthy}, "B9", "")
    assert "caught" not in pair and pair["why"].startswith("the healthy control failed")
    # The report says so, rather than listing the buggy run's incident as caught.
    results = {"B9||buggy": buggy, "B9||healthy": healthy}
    row = report_row(results, "B9")
    assert "default: not judged: the healthy control failed" in row and "`replica_drift` @ 3" not in row


def test_a_buggy_run_that_exits_nonzero_is_a_failed_run(sweep):
    run, _ = sweep
    buggy, healthy = run(buggy="exit2")
    assert not buggy["ok"] and buggy["returncode"] == 2 and buggy["run_id"] == "fake-buggy"
    assert "default: buggy run failed (exited with status 2" in report_row({"B9||buggy": buggy, "B9||healthy": healthy}, "B9")


def test_stored_entries_are_checked_again_when_only_the_report_is_written(sweep):
    """`--report` reads results.json: an entry stored as ok with a nonzero exit, or a healthy
    control on which the bug showed, is not judged."""
    run, _ = sweep
    buggy, healthy = run()
    stored = {**healthy, "returncode": 2}  # as the sweep before Nov 19, 2026 recorded such a run
    assert stored["ok"] and run_all.problem(stored) == "exited with status 2 after printing its result"
    assert "not judged: the healthy control failed" in report_row({"B9||buggy": buggy, "B9||healthy": stored}, "B9")
    showed = {**healthy, "result": {**healthy["result"], "symptom": {"bug_present": True}}}
    assert "not judged: the bug showed on the healthy control" in report_row({"B9||buggy": buggy, "B9||healthy": showed}, "B9")
    two_ranks = {**healthy, "result": {**healthy["result"], "world_size": 2}}
    assert run_all.problem(two_ranks) == "result has 1 rank summaries for world size 2"


def test_a_timeout_is_recorded_for_its_case_and_the_sweep_goes_on(sweep, monkeypatch):
    run, child = sweep
    monkeypatch.setattr(run_all, "TIMEOUT", 2)
    monkeypatch.setattr(run_all, "BUGS", [
        {"id": "B9", "script": str(child), "runs": [("", [])]},
        {"id": "B10", "script": str(child), "runs": [("", [])]},
    ])
    monkeypatch.setenv("FAKE_MODE_BUGGY", "hang")
    monkeypatch.setenv("FAKE_MODE_HEALTHY", "ok")
    import time
    started = time.monotonic()
    results = run_all.run(None)
    assert time.monotonic() - started < 30  # two hangs of 2 s each; a grandchild left alive costs 30 s more each
    assert set(results) == {"B9||buggy", "B9||healthy", "B10||buggy", "B10||healthy"}
    for bug in ("B9", "B10"):
        assert results[f"{bug}||buggy"]["ok"] is False and results[f"{bug}||buggy"]["returncode"] is None
        assert results[f"{bug}||buggy"]["error"] == "timed out after 2 s"
        assert results[f"{bug}||healthy"]["ok"] is True
    assert json.loads(run_all.RESULTS.read_text()).keys() == results.keys()
    log = (run_all.OUT / "child.py-buggy-default.log").read_text()
    assert "killed after 2 s" in log


# --- After wave AD: a run of several ranks is read through the all-ranks view. Since wave Z
# (Nov 20, 2026) one rank's evidence holds no `replica_drift`: a rank records an observation and
# the ranks are compared across them. The harness went on reading each rank alone, so a table
# written from a fresh sweep would have said B3, B9 and B10 were not caught.


@pytest.fixture
def table(tmp_path, monkeypatch):
    """`report_row` writes `run_all.TABLE`: point it away from the committed file."""
    monkeypatch.setattr(run_all, "TABLE", tmp_path / "RESULTS.md")


def _two_ranks(variant: str) -> dict:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(HERE.parent.parent / "sdk" / "src"), str(HERE)]))
    proc = subprocess.run(
        [sys.executable, "b10_x_ddp.py", "--variant", variant, "--tag", "pytest", "--steps", "30", "--replica-every", "5"],
        cwd=HERE, env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(line for line in proc.stdout.splitlines() if line.startswith("RESULT "))
    run_id = json.loads(line[len("RESULT "):])["run_id"]
    return {"ok": True, "returncode": 0, "run_id": run_id, "args": [],
            "result": json.loads((HERE / "out" / run_id / "result.json").read_text())}


def test_a_run_of_two_ranks_is_read_through_the_all_ranks_view(table):
    """B10 on two CPU processes, short, both ways. Fail before the harness read the all-ranks view:
    the buggy run's result held no `replica_drift` anywhere, and the pair read as not caught."""
    buggy, healthy = _two_ranks("buggy"), _two_ranks("healthy")
    for entry in (buggy, healthy):
        result = entry["result"]
        assert run_all.problem(entry) is None and run_all.replicas_not_compared(entry) is None
        assert all(i["kind"] != "replica_drift" for rank in result["ranks"].values() for i in rank["incidents"])  # in no rank's evidence
        view = result["all_ranks"]
        check = view["relevant_kinds"]["replica_drift"]
        assert check["value"] == "checked" and check["coverage"]["compared"] == check["coverage"]["steps"] >= 1, check
        assert view["judged_ranks"] == [0, 1] and not view["unknown_ranks"], view
        assert (HERE / "out" / entry["run_id"] / f"{entry['run_id']}.all-ranks.evidence.json").is_file()
    assert buggy["result"]["symptom"]["bug_present"] is True and healthy["result"]["symptom"]["bug_present"] is False
    assert [i["kind"] for i in buggy["result"]["cross_rank"]] == ["replica_drift"]
    assert healthy["result"]["cross_rank"] == [] and healthy["result"]["all_ranks"]["run_state"] == "healthy"
    pair = run_all._pair({"B10||buggy": buggy, "B10||healthy": healthy}, "B10", "")
    assert [(kind, where) for kind, _, where in pair["caught"]] == [("replica_drift", "across")] and "why" not in pair
    row = report_row({"B10||buggy": buggy, "B10||healthy": healthy}, "B10")
    assert "default: `replica_drift` @ " in row and "(across ranks)" in row and "not caught" not in row


def _stored(variant: str, view: dict | None) -> dict:
    """A stored entry of a two-rank run with no incident in either rank's evidence."""
    result = {"run_id": f"fake-{variant}", "world_size": 2, "settings": {}, "symptom": {"bug_present": variant == "buggy"},
              "ranks": {"rank0": {"incidents": []}, "rank1": {"incidents": []}}, "cross_rank": [], "relevant": ["replica_drift"]}
    if view is not None:
        result["all_ranks"] = view
    return {"ok": True, "returncode": 0, "run_id": result["run_id"], "args": [], "result": result}


def _view(value: str, steps: int, compared: int) -> dict:
    return {"run_state": "healthy", "incidents": [],
            "relevant_kinds": {"replica_drift": {"value": value, "coverage": {"steps": steps, "compared": compared}}}}


def test_a_run_of_several_ranks_without_the_all_ranks_view_is_not_a_valid_observation(table):
    """An entry as the harness wrote it before: each rank read alone. It is not judged, so a
    table written again from old results does not say "not caught" of a drift nothing looked for."""
    old = _stored("healthy", None)
    assert run_all.problem(old) == "a run of several ranks without the all-ranks view, where its replicas are compared"
    row = report_row({"B9||buggy": _stored("buggy", None), "B9||healthy": old}, "B9")
    assert "buggy run failed (a run of several ranks without the all-ranks view" in row and "not caught" not in row


@pytest.mark.parametrize("uncompared", ["healthy", "buggy"])
def test_replicas_nobody_compared_do_not_read_as_replicas_that_agree(table, uncompared):
    """A pair whose bug is replica drift is judged only when both runs compared their replicas."""
    compared, not_compared = _view("checked", 4, 4), _view("not_configured", 4, 0)
    entries = {v: _stored(v, not_compared if v == uncompared else compared) for v in ("buggy", "healthy")}
    assert run_all.problem(entries[uncompared]) is None  # the run completed; what is missing is the comparison
    pair = run_all._pair({"B9||buggy": entries["buggy"], "B9||healthy": entries["healthy"]}, "B9", "")
    assert "caught" not in pair and "replicas were not compared" in pair["why"] and uncompared in pair["why"]
    row = report_row({"B9||buggy": entries["buggy"], "B9||healthy": entries["healthy"]}, "B9")
    assert "default: not judged: the ranks' replicas were not compared" in row and "not caught" not in row
    both = {v: _stored(v, compared) for v in ("buggy", "healthy")}
    assert "default: not caught" in report_row({"B9||buggy": both["buggy"], "B9||healthy": both["healthy"]}, "B9")


def test_an_incident_of_the_run_is_told_apart_from_a_ranks_own():
    assert run_all._fmt([("replica_drift", 49, "across"), ("grad_spike", 7, "0,1"), ("x", 1, "1,across"), ("y", 2, "")]) == \
        "`replica_drift` @ 49 (across ranks), `grad_spike` @ 7 (ranks 0,1), `x` @ 1 (ranks 1 and across ranks), `y` @ 2"


def report_row(results, bug_id):
    """The report's row for `bug_id`, written from `results` with a one-bug catalog."""
    bug = {"id": bug_id, "paper": "PT-0", "synopsis": "s", "location": "l", "in_clone": "c", "how": "recreated",
           "cpu": "yes", "review": "r", "script": "x.py", "runs": [("", [])]}
    assert run_all.TABLE != HERE / "RESULTS.md", "a test that writes the report must point TABLE at a temporary file"
    real = run_all.BUGS
    run_all.BUGS = [bug]
    try:
        run_all.write_table(results)
    finally:
        run_all.BUGS = real
    return next(line for line in run_all.TABLE.read_text().splitlines() if line.startswith(f"| {bug_id} "))


def test_the_committed_table_is_what_the_committed_results_give(tmp_path, monkeypatch):
    """Scored again under the stricter rule, every published row stays as it is."""
    monkeypatch.setattr(run_all, "TABLE", tmp_path / "RESULTS.md")
    run_all.write_table(json.loads((HERE / "results.json").read_text()))
    assert run_all.TABLE.read_bytes() == (HERE / "RESULTS.md").read_bytes()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
