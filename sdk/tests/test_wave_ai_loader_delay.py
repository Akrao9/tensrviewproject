"""Wave AI: the loader-delay measurement keeps every step's timing for capture off and on
(`examples/loader_delay_check.py`).

The author's terms, Nov 22, 2026: "Preserve per-step timings for capture off and on, including step
100, and report both the worst delay and the preregistered median overhead. A passing median alone
doesn't explain the spike." The summary is shown on made-up timings: one slow `next()` is named with
its step and round and leaves the median where it was. Then the script runs on CPU: each arm in a fresh
process, every step of both arms written, TensorQuill's own readings beside the script's in the `on`
arm only. Nothing is judged: a CPU run shows the files are produced, not what a GPU shows.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(EXAMPLES))
import loader_delay_check as ld  # noqa: E402


def arm(name, rnd, excess, step_ms=30.0, gc=()):
    steps = len(excess)
    return {"arm": name, "round": rnd, "state": "measured", "injected_ms": [20.0] * steps,
            "next_ms": [20.0 + e for e in excess], "excess_ms": list(excess), "step_ms": [step_ms + e for e in excess],
            "rest_ms": [step_ms - 20.0] * steps, "gc_collections": list(gc)}


def test_one_slow_step_is_named_with_its_step_and_round_and_the_median_stays_put():
    quiet = [1.0] * 300
    spiked = [1.0] * 300
    spiked[100] = 45.0
    pause = {"step": 100, "in_next": True, "generation": 2, "collected": 15, "thread": "tensorquill-writer", "ms": 44.0}
    results = {"off": [arm("off", r, quiet) for r in range(3)],
               "on": [arm("on", 0, quiet), arm("on", 1, spiked, gc=[pause, {**pause, "step": 50, "ms": 0.2}]), arm("on", 2, quiet)]}
    summary = ld.summarize(results, [50, 100, 150], 15.0)
    on, off = summary["arms"]["on"], summary["arms"]["off"]
    assert on["worst_excess"] == {"ms": 45.0, "step": 100, "round": 1} and off["worst_excess"]["ms"] == 1.0
    assert on["excess_at_named_steps"]["100"] == [1.0, 45.0, 1.0] and on["excess_at_named_steps"]["50"] == [1.0] * 3
    assert on["over_slack"] == [{"round": 1, "step": 100, "excess_ms": 45.0, "step_ms": 75.0}] and off["over_slack"] == []
    assert on["gc_over_1ms"] == [{"round": 1, **pause}] and on["gc_collections"] == 2 and off["gc_collections"] == 0
    both = summary["on_minus_off"]
    assert both["excess_ms_at_named_steps"]["100"] == {"on_median": 1.0, "off_median": 1.0, "on_max": 45.0, "off_max": 1.0}
    assert both["toy_step_time_overhead_median"] == pytest.approx(0.0)  # the median does not show the spike
    assert "granite_compiler_gate.py" in both["note"]


def test_an_arm_that_failed_is_listed_and_the_other_is_still_summarized():
    results = {"off": [arm("off", 0, [1.0] * 10)], "on": [{"arm": "on", "round": 0, "state": "error", "error": "boom"}]}
    summary = ld.summarize(results, [5], 15.0)
    assert summary["arms"]["on"] == {"rounds_measured": 0, "errors": ["boom"]}
    assert summary["arms"]["off"]["worst_excess"]["ms"] == 1.0 and "on_minus_off" not in summary


def test_the_named_steps_default_to_every_flush_boundary():
    args = ld.argparse.Namespace(name_steps=None, every=50, steps=300)
    assert ld.named_steps(args) == [50, 100, 150, 200, 250]
    args.name_steps = [100, 100, 7, 999]
    assert ld.named_steps(args) == [7, 100]


def test_on_cpu_every_step_of_both_arms_is_written_from_fresh_processes(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    out = tmp_path / "delay"
    done = subprocess.run([sys.executable, str(EXAMPLES / "loader_delay_check.py"), "--out", str(out), "--rounds", "2",
                           "--steps", "60", "--every", "20", "--wait-ms", "2"],
                          env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads((out / "loader_delay.json").read_text())
    assert result["judged"] is False and result["named_steps"] == [20, 40]
    for name in ("off", "on"):
        rounds = result["per_step"][name]
        assert [r["round"] for r in rounds] == [0, 1] and all(r["state"] == "measured" for r in rounds)
        for r in rounds:
            assert len(r["next_ms"]) == len(r["step_ms"]) == len(r["excess_ms"]) == len(r["injected_ms"]) == 60
            assert all(n >= i for n, i in zip(r["next_ms"], r["injected_ms"]))  # next() never beats its sleep
            assert all(s >= n for s, n in zip(r["step_ms"], r["next_ms"]))
            assert json.loads((out / f"arm-{name}-r{r['round']}.json").read_text())["step_ms"] == r["step_ms"]
            assert isinstance(r["gc_collections"], list)
    on, off = result["per_step"]["on"][0], result["per_step"]["off"][0]
    assert "tq_data_wait_ms" not in off and "tensorquill_file" not in off  # no session in the off arm
    assert Path(on["tensorquill_file"]).resolve().is_relative_to(SRC)
    assert len(on["tq_data_wait_ms"]) == 60 and all(isinstance(v, float) for v in on["tq_data_wait_ms"])
    assert on["tq_step_time_ms"][0] is None  # TensorQuill has no step time for the first step: kept as absent
    assert all(isinstance(v, float) for v in on["tq_step_time_ms"][1:])
    assert set(result["on_minus_off"]["excess_ms_at_named_steps"]) == {"20", "40"}
    assert result["arms"]["on"]["worst_excess"]["step"] in range(60)
