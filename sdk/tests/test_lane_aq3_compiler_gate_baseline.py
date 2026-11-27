"""Lane AQ-3: the four-arm compiler gate when capture off itself cannot run (`examples/granite_compiler_gate.py`).

On the A100 every arm of `granite-gate-4arm`, capture off too, failed in CUDA-graph recording:
Granite's experts call `torch._grouped_mm`, whose path on that GPU copies the group offsets to the
host, which a CUDA graph cannot record. The gate then ran all twelve arm processes into the same
failure and read `fail`, as if TensorQuill had failed. Capture off failing says the model cannot be
measured in that mode on that machine: the gate stops after it and reads not proven (exit 3), with
capture off's error named. The arm processes are stood in for here; nothing runs on a GPU.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
import granite_compiler_gate as gate  # noqa: E402

GRAPH_ERROR = ("RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU "
               "tensor is pinned. Please use tensor.pin_memory() or allocate the tensor with pin_memory=True.")


def fake_arms(ran: list, failing: set[str]):
    def run(cmd, env=None, stdout=None, stderr=None, timeout=None):
        arm, rnd, path = (cmd[cmd.index(flag) + 1] for flag in ("--arm", "--round", "--result"))
        ran.append((arm, int(rnd)))
        if arm in failing:
            result = {"arm": arm, "round": int(rnd), "state": "error", "error": GRAPH_ERROR}
        else:
            result = {"arm": arm, "round": int(rnd), "state": "measured", "median_ms": 100.0, "mean_ms": 100.0,
                      "step_ms": [100.0], "graph_breaks": 0, "recompiles_after_warmup": 0, "cudagraph_skips": 0,
                      "frames_ok": 1, "observes": "x", "samples": "y", "checks": []}
        Path(path).write_text(json.dumps(result))
        return SimpleNamespace(returncode=1 if arm in failing else 0)
    return run


def run_gate(tmp_path, monkeypatch, failing, arms="off,tensorquill"):
    ran: list = []
    monkeypatch.setattr(gate.subprocess, "run", fake_arms(ran, failing))
    args = gate.build_parser().parse_args(["--out", str(tmp_path / "gate"), "--toy", "--rounds", "3", "--arms", arms])
    code = gate.driver_main(args)
    return code, json.loads((tmp_path / "gate" / "gate.json").read_text()), ran


def test_capture_off_failing_in_the_first_round_stops_the_gate_as_not_proven(tmp_path, monkeypatch):
    code, result, ran = run_gate(tmp_path, monkeypatch, failing={"off", "tensorquill"})
    assert ran == [("off", 0)]  # nothing ran after capture off failed
    assert code == gate.NOT_PROVEN
    assert result["status"] == "not_proven"
    first = result["criteria"][0]
    assert first["ok"] is None and "capture off" in first["name"]
    assert "CUDA graph capture" in first["detail"]
    assert "capture off" in result["claim"]


def test_capture_off_failing_after_another_arm_still_stops_there(tmp_path, monkeypatch):
    code, result, ran = run_gate(tmp_path, monkeypatch, failing={"off"}, arms="tensorquill,off")
    assert ran == [("tensorquill", 0), ("off", 0)]
    assert code == gate.NOT_PROVEN and result["status"] == "not_proven"


def test_tensorquill_failing_with_capture_off_measured_is_still_a_failure(tmp_path, monkeypatch):
    code, result, ran = run_gate(tmp_path, monkeypatch, failing={"tensorquill"})
    assert len(ran) == 6  # every round ran
    assert code == 1 and result["status"] == "fail"


# ---- Gate 2's overhead line is scored by overhead_harness.py, not here (the author, Nov 26, 2026) ----


def measured(arm, rnd, ms):
    return {"arm": arm, "round": rnd, "state": "measured", "median_ms": ms, "mean_ms": ms, "graph_breaks": 0,
            "recompiles_after_warmup": 0, "cudagraph_skips": 0, "frames_ok": 1, "observes": "x", "samples": "y",
            "checks": [{"name": "c", "ok": True, "detail": ""}]}


def judged(tq_ms, skips=0):
    results = {"off": [measured("off", r, 100.0) for r in range(3)],
               "tensorquill": [measured("tensorquill", r, ms) for r, ms in enumerate(tq_ms)]}
    for r in results["tensorquill"]:
        r["cudagraph_skips"] = skips
    args = SimpleNamespace(toy=False, eager=False)
    return gate.verdict(args, gate.summarize(args, results), True)


@pytest.mark.parametrize("tq_ms", [[100.2, 100.3, 100.4], [103.0, 102.0, 104.0]])
def test_the_three_round_median_is_measured_and_reported_and_never_decides_the_gate(tq_ms):
    status, criteria = judged(tq_ms)
    line = next(c for c in criteria if c["name"] == "median overhead 1% or less")
    assert line["ok"] is None and line["scored"] is False
    assert "overhead_harness.py" in line["detail"] and "not judged here" in line["detail"]
    assert "%" in line["detail"]  # the measured figure is still written
    assert status == "pass"  # over 1% or under, the other criteria decide


def test_the_criterion_record_keeps_the_nov_22_instruction_and_names_the_nov_26_change():
    criterion = gate.OVERHEAD_CRITERION
    assert criterion["primary"].startswith("overhead_median: the median of the paired rounds")
    assert "overhead_harness.py" in criterion["scored_by"] and "Nov 26, 2026" in criterion["scored_by"]
    assert "scored figure" not in criterion["primary"] or "until Nov 26, 2026" in criterion["primary"]


def test_a_gpu_run_without_cuda_graphs_is_still_not_proven():
    status, _ = judged([100.2, 100.3, 100.4], skips=None)
    assert status == "not_proven"
