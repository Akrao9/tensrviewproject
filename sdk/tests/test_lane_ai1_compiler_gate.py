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
# The folders from the project's own up to four above it; a slice, as a short path has fewer (BUG-30).
for candidate in (parent / "NeuralDBG" for parent in Path(__file__).resolve().parents[2:7]):
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
    # Since the author's change of Nov 26, 2026 the overhead line is reported here and scored by
    # overhead_harness.py: every other criterion is met, and that one is not judged.
    assert all(c["ok"] for c in criteria if c.get("scored", True))
    assert [c["name"] for c in criteria if not c.get("scored", True)] == ["median overhead 1% or less"]


def test_the_overhead_is_the_median_of_the_paired_round_ratios():
    results = three_rounds([101.0, 103.0, 100.2])
    results["off"][1]["mean_ms"] = 102.0  # round 2's baseline was slower: paired, that round is +0.98%
    summary = gate.summarize(SimpleNamespace(toy=False, eager=False), results)
    ratios = summary["tensorquill"]["overhead_by_round"]
    assert ratios == pytest.approx([0.01, 103 / 102 - 1, 0.002])
    assert summary["tensorquill"]["overhead_median"] == pytest.approx(103 / 102 - 1)


@pytest.mark.parametrize("change, failing", [
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


def test_the_overhead_is_reported_on_the_rounds_mean_step_not_the_per_step_median():
    # A periodic cost (the flushes) moves the mean and leaves the per-step median where it was.
    results = three_rounds([100.0, 100.0, 100.0])
    for r in results["tensorquill"]:
        r["mean_ms"] = 102.0
    status, criteria = judged(results)
    line = next(c for c in criteria if c["name"] == "median overhead 1% or less")
    assert "+2.00%" in line["detail"]  # the mean step's figure, not the per-step median's +0.00%
    assert line["ok"] is None and status == "pass"  # reported, not scored (the author, Nov 26, 2026)


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
    # The author, Nov 22, 2026: the paired median is the primary 1% criterion, the ratio of arm medians beside it.
    assert result["overhead_criterion"]["primary"].startswith("overhead_median: the median of the paired rounds")
    assert "never scored" in result["overhead_criterion"]["alongside"]
    assert result["arms"]["tensorquill"]["overhead_of_median_rounds"] is not None
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


def test_the_slowest_step_of_every_round_is_reported_and_never_scored():
    results = three_rounds([100.2, 100.2, 100.2])
    for r in results["tensorquill"]:
        r["step_ms"] = [100.0] * 49 + [160.0] + [100.0] * 150  # one slow step: the median does not move
    args = SimpleNamespace(toy=False, eager=False, warmup=50)
    summary = gate.summarize(args, results)
    worst = summary["tensorquill"]["worst_step_by_round"]
    assert [w["step"] for w in worst] == [99, 99, 99] and all(w["ms"] == 160.0 for w in worst)
    assert worst[0]["over_median_ms"] == pytest.approx(59.8) and worst[1]["file"] == "arm-tensorquill-r1.json"
    assert summary["off"]["worst_step_by_round"][0] == {"round": 0, "step": None, "ms": None}  # no per-step list given
    status, _ = gate.verdict(args, summary, True)
    assert status == "pass"


def test_the_figure_moe_dogfood_printed_is_reported_beside_the_scored_one():
    plain = SimpleNamespace(toy=False, eager=False)
    results = three_rounds([100.6, 103.0, 100.2])
    results["off"][1]["mean_ms"] = 102.0
    tq = gate.summarize(plain, results)["tensorquill"]
    assert tq["overhead_of_median_rounds"] == pytest.approx(100.6 / 100.0 - 1)  # each arm's median round, then the ratio
    assert tq["overhead_median"] == pytest.approx(100.6 / 100.0 - 1)  # the scored one: the median of the paired rounds
    _, criteria = gate.verdict(plain, gate.summarize(plain, results), True)
    line = next(c for c in criteria if c["name"].startswith("median overhead"))
    assert line["ok"] is None and "moe_dogfood.py" in line["detail"] and "OTHER SIDE" not in line["detail"]
    # Baselines that moved between rounds: paired, +0.50%, +1.82%, +1.67%; by median rounds, +0.50%.
    results = three_rounds([100.5, 112.0, 91.5])
    results["off"][1]["mean_ms"], results["off"][2]["mean_ms"] = 110.0, 90.0
    summary = gate.summarize(plain, results)
    assert summary["tensorquill"]["overhead_of_median_rounds"] == pytest.approx(0.005)
    _, criteria = gate.verdict(plain, summary, True)
    line = next(c for c in criteria if c["name"].startswith("median overhead"))
    assert line["ok"] is None and "OTHER SIDE OF THE LINE" in line["detail"]  # reported, the other said aloud
