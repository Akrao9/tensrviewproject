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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
