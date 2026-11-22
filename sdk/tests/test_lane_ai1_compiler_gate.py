"""Lane AI-1: the Granite compiler gate's driver, arms and pass line (`examples/granite_compiler_gate.py`).

The pass line is shown on made-up summaries: it passes only with every criterion met on an established
run, fails on any one missed, and never passes an arm that did not run. Then the toy driver runs on CPU:
capture off and TensorQuill in fresh processes over two rounds (plus NeuralDBG when its clone is beside
the project), every arm's result written, exit 0 as a dry run whose timing is not established.
A CPU run checks the harness only; the claim is the GPU run's.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(EXAMPLES))
import granite_compiler_gate as gate  # noqa: E402

# A NeuralDBG clone beside the project (read only), for the third arm when neuraldbg is not installed.
for candidate in (Path(__file__).resolve().parents[n] / "NeuralDBG" for n in range(2, 7)):
    if (candidate / "neuraldbg" / "__init__.py").exists():
        NEURALDBG = candidate
        break
else:
    NEURALDBG = None


def measured(arm, rnd, ms, *, breaks=0, recompiles=0, skips=0, ok=True):
    return {"arm": arm, "round": rnd, "state": "measured", "median_ms": ms, "mean_ms": ms, "graph_breaks": breaks,
            "recompiles_after_warmup": recompiles, "cudagraph_skips": skips, "frames_ok": 1,
            "observes": "x", "samples": "y", "checks": [{"name": "c", "ok": ok, "detail": ""}]}


def judged(results, established=True, toy=False):
    args = SimpleNamespace(toy=toy, eager=False)
    return gate.verdict(args, gate.summarize(args, results), established)


def three_rounds(tq_ms, **kw):
    return {"off": [measured("off", r, 100.0) for r in range(3)],
            "tensorquill": [measured("tensorquill", r, ms, **kw) for r, ms in enumerate(tq_ms)]}


def test_the_pass_line_passes_only_with_every_criterion_met():
    status, criteria = judged(three_rounds([100.5, 100.9, 101.5]))
    assert status == "pass", criteria
    assert all(c["ok"] for c in criteria)


def test_the_overhead_is_the_median_of_the_paired_round_ratios():
    results = three_rounds([101.0, 103.0, 100.2])
    results["off"][1]["mean_ms"] = 102.0  # round 2's baseline was slower: paired, that round is +0.98%
    summary = gate.summarize(SimpleNamespace(toy=False, eager=False), results)
    ratios = summary["tensorquill"]["overhead_by_round"]
    assert ratios == pytest.approx([0.01, 103 / 102 - 1, 0.002])
    assert summary["tensorquill"]["overhead_median"] == pytest.approx(103 / 102 - 1)


@pytest.mark.parametrize("change, failing", [
    ({"tq_ms": [102.0, 101.5, 103.0]}, "median overhead"),
    ({"breaks": 1}, "zero graph breaks"),
    ({"recompiles": 2}, "zero recompiles"),
    ({"skips": 3}, "CUDA-graph skips"),
    ({"ok": False}, "capture checks"),
])
def test_any_one_criterion_missed_fails_the_gate(change, failing):
    tq_ms = change.pop("tq_ms", [100.1, 100.2, 100.3])
    status, criteria = judged(three_rounds(tq_ms, **change))
    assert status == "fail"
    assert [c["name"] for c in criteria if c["ok"] is False] == [next(c["name"] for c in criteria if failing in c["name"])]


def test_an_arm_that_did_not_run_or_crashed_is_never_passed():
    results = three_rounds([100.1, 100.2, 100.3])
    results["tensorquill"][1] = {"arm": "tensorquill", "round": 1, "state": "error", "error": "RuntimeError: x"}
    status, criteria = judged(results)
    assert status == "fail" and criteria[0]["ok"] is False
    results["tensorquill"][1] = {"arm": "tensorquill", "round": 1, "state": "not_run", "reason": "absent"}
    assert judged(results)[0] == "fail"
    summary = gate.summarize(SimpleNamespace(), {"wandb": [{"arm": "wandb", "round": 0, "state": "not_run",
                                                           "reason": "wandb is not installed"}]})
    assert summary["wandb"]["states"] == ["not_run"] and "median_ms" not in summary["wandb"]


def test_the_overhead_is_judged_on_the_rounds_mean_step_not_the_per_step_median():
    # A periodic cost (the flushes) moves the mean and leaves the per-step median where it was.
    results = three_rounds([100.0, 100.0, 100.0])
    for r in results["tensorquill"]:
        r["mean_ms"] = 102.0
    status, criteria = judged(results)
    assert status == "fail" and [c["name"] for c in criteria if c["ok"] is False] == ["median overhead 1% or less"]


def test_a_gpu_run_without_cuda_graphs_is_not_proven_not_passed():
    status, criteria = judged(three_rounds([100.1, 100.2, 100.3], skips=None))
    assert status == "not_proven"
    assert {c["name"]: c["ok"] for c in criteria}["zero CUDA-graph skips under reduce-overhead"] is None


def test_a_check_that_could_not_be_made_is_unknown_not_failed_nor_passed():
    results = three_rounds([100.1, 100.2, 100.3])
    results["tensorquill"][0]["checks"].append({"name": "samples", "ok": None, "detail": "not counted"})
    summary = gate.summarize(SimpleNamespace(), results)["tensorquill"]
    assert summary["checks_ok"] and summary["unknown_checks"] and not summary["failed_checks"]


def test_off_cpu_the_timing_and_cuda_graph_lines_are_not_established_not_passed():
    results = three_rounds([150.0, 150.0, 150.0], skips=None)  # 50% over: not judged on a dry run
    status, criteria = judged(results, established=False, toy=True)
    assert status == "dry_run"
    by_name = {c["name"]: c["ok"] for c in criteria}
    assert by_name["median overhead 1% or less"] is None
    assert by_name["zero CUDA-graph skips under reduce-overhead"] is None


def run_driver(tmp_path, *extra):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]))
    return subprocess.run([sys.executable, str(EXAMPLES / "granite_compiler_gate.py"), "--toy", "--rounds", "2",
                           "--steps", "50", "--out", str(tmp_path / "gate"), *extra],
                          env=env, capture_output=True, text=True, timeout=600)


def test_the_toy_driver_runs_each_arm_in_a_fresh_process_and_writes_every_result(tmp_path):
    arms = "off,tensorquill,wandb" + (",neuraldbg" if NEURALDBG else "")
    extra = ["--arms", arms] + (["--neuraldbg-path", str(NEURALDBG)] if NEURALDBG else [])
    proc = run_driver(tmp_path, *extra)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = tmp_path / "gate"
    result = json.loads((out / "gate.json").read_text())
    assert result["status"] == "dry_run" and result["claim"].startswith("not established")
    pids, measured_runs = set(), 0
    for arm in arms.split(","):
        for rnd in range(2):
            arm_result = json.loads((out / f"arm-{arm}-r{rnd}.json").read_text())
            if arm == "wandb":
                assert arm_result["state"] in ("measured", "not_run", "error")  # never run here: any state is written
                continue
            assert arm_result["state"] == "measured"
            measured_runs += 1
            pids.add(arm_result["pid"])
            # The arm measured the tensorquill under test, not the one the environment installed.
            assert Path(arm_result["tensorquill_file"]).resolve().is_relative_to(SRC)
    assert len(pids) == measured_runs  # a new process for every arm and round
    tq = result["arms"]["tensorquill"]
    assert tq["states"] == ["measured"] and tq["checks_ok"] and tq["graph_breaks"] == 0
    assert tq["recompiles_after_warmup"] == 0 and tq["frames_ok"] >= 1
    try:
        import wandb  # noqa: F401
    except ImportError:
        assert result["arms"]["wandb"]["states"] == ["not_run"]
    if NEURALDBG:
        nd = result["arms"]["neuraldbg"]
        assert nd["states"] == ["measured"] and nd["checks_ok"]
        assert nd["graph_breaks"] >= 1  # its hooks are dynamo.disable'd: counted, not raised


def test_the_driver_refuses_without_the_two_arms_the_pass_line_needs(tmp_path):
    proc = run_driver(tmp_path, "--arms", "off,wandb")
    assert proc.returncode != 0 and "must include off and tensorquill" in proc.stdout + proc.stderr
