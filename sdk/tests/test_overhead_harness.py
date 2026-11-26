"""The overhead harness (`examples/overhead_harness.py`): its statistics, its round layout, its arms,
its check that the GPU is not shared, and a CPU dry run of the whole script.

The statistics are shown on synthetic timings shaped like the recorded A100 runs: a step of about
143 ms; the capture-off arm moving about 4 ms from one round to the next; and noise of each fresh
process of its own, at the paired standard deviation the six recorded three-round gate sets show
(0.4% to 1.6%). Rates are asserted over many seeded runs, never on one draw. The dry run shows only
that the script runs and writes its files: CPU timings say nothing about a GPU.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(EXAMPLES))
import overhead_harness as oh  # noqa: E402

BASE_MS = 143.0
DRIFT_MS = 4.0  # the capture-off arm alone: 146.3, 142.2, 143.1 ms


def arms_with(**factors: float) -> list[dict]:
    """The two capture-off arms the harness always has, then one arm per planted factor."""
    arms = [dict(spec) for spec in oh.BASELINE_ARMS]
    for name, _ in factors.items():
        arms.append(oh.normalize_arm({"name": name, "init": {"capture": "routers"}}, default_limit=0.01))
    return arms


def simulate(rounds: int, factors: dict[str, float], *, paired_sd: float, seed: int,
             drift_ms: float = DRIFT_MS, trend_ms_per_process: float = 0.0, order=None) -> dict:
    """Block times by round and arm. Each round moves every arm by a common amount (up to
    `drift_ms` apart from round to round); each process adds noise of its own with SD
    paired_sd / sqrt(2), so the paired difference of two arms has SD `paired_sd`; and a trend
    over the run's process sequence (a machine warming up) moves each process by its position."""
    rng = random.Random(seed)
    names = ["off", "off-again", *factors]
    orders = order or oh.round_orders(names, rounds, seed)
    process_sd = paired_sd / math.sqrt(2)
    times: dict[int, dict[str, float]] = {}
    position = 0
    for r in range(rounds):
        common = rng.uniform(-drift_ms / 2, drift_ms / 2)
        times[r] = {}
        for name in orders[r]:
            factor = factors.get(name, 1.0)
            times[r][name] = ((BASE_MS + common) * factor * (1 + rng.gauss(0, process_sd))
                              + trend_ms_per_process * position)
            position += 1
    return times


def analyzed(times, factors, **kw):
    return oh.analyze(times, arms_with(**factors), **kw)


# ---- the interval ----------------------------------------------------------------------------


def test_the_interval_for_the_median_comes_from_order_statistics_with_its_stated_coverage():
    values = [0.01 * i for i in range(10)]
    lo, hi, coverage = oh.median_interval(values, 0.95)
    # n = 10: [x_(2), x_(9)] covers 1 - 2 P(B <= 1) = 1 - 22/1024
    assert (lo, hi) == (values[1], values[8])
    assert coverage == pytest.approx(1 - 2 * 11 / 1024)
    assert coverage >= 0.95


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_below_six_rounds_there_is_no_95_percent_interval(n):
    assert oh.median_interval([0.1 * i for i in range(n)], 0.95) is None


def test_the_interval_holds_its_coverage_on_skewed_noise():
    """Distribution-free: on a skewed distribution its true median is covered at least 95% of the time."""
    rng = random.Random(3)
    covered = 0
    trials = 2000
    for _ in range(trials):
        values = [rng.expovariate(1.0) for _ in range(15)]
        lo, hi, _ = oh.median_interval(values, 0.95)
        covered += lo <= math.log(2) <= hi
    assert covered / trials >= 0.95


# ---- the verdicts ----------------------------------------------------------------------------


@pytest.mark.parametrize("interval, verdict", [
    ((0.001, 0.008), "met"),
    ((-0.004, 0.0099), "met"),
    ((0.011, 0.02), "not met"),
    ((0.004, 0.012), "cannot tell"),
    (None, "cannot tell"),
])
def test_met_only_below_the_interval_not_met_only_above_it(interval, verdict):
    assert oh.judge(interval, 0.01) == verdict


def test_an_arm_reported_only_has_no_verdict_against_a_limit():
    assert oh.judge((0.03, 0.05), None) == "reported"


# ---- the statistics on synthetic runs shaped like the record ---------------------------------


@pytest.mark.parametrize("paired_sd, rounds", [(0.004, 15), (0.011, 40), (0.016, 60)])
def test_a_planted_1_percent_is_found_under_the_recorded_noise_with_enough_rounds(paired_sd, rounds):
    found = 0
    trials = 200
    for seed in range(trials):
        result = analyzed(simulate(rounds, {"planted": 1.01}, paired_sd=paired_sd, seed=seed), {"planted": 1.01})
        arm = result["arms"]["planted"]
        found += arm["interval"] is not None and arm["interval"][0] > 0
        assert abs(arm["median"] - 0.01) < 4 * paired_sd  # the estimate is the planted 1%, not the drift
    assert found / trials >= 0.9, found


def test_the_drift_between_rounds_alone_would_hide_nothing_the_pairing_removes_it():
    """With no noise of each process, a 4 ms common drift leaves the paired 1% exact in every round."""
    times = simulate(8, {"planted": 1.01}, paired_sd=0.0, seed=1)
    result = analyzed(times, {"planted": 1.01})
    assert result["arms"]["planted"]["median"] == pytest.approx(0.01)
    assert result["arms"]["planted"]["interval"][1] - result["arms"]["planted"]["interval"][0] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("paired_sd", [0.004, 0.016])
def test_two_identical_arms_are_not_called_different(paired_sd):
    called = 0
    trials = 400
    for seed in range(trials):
        result = analyzed(simulate(30, {}, paired_sd=paired_sd, seed=seed), {})
        called += not result["noise"]["identical_arms_agree"]
    assert called / trials <= 0.06, called  # the interval's coverage is at least 95%


def test_a_baseline_drifting_through_the_run_does_not_create_a_difference():
    """A machine that slows by 0.1 ms with every process launched (over 4 ms a round of 5 arms, about
    18 ms over the run): with the harness's balanced orders an arm with no effect reads no
    difference, and the identical pair is not called different more often than its coverage allows."""
    called_zero = called_pair = 0
    trials = 300
    for seed in range(trials):
        factors = {"no-effect": 1.0, "other": 1.0, "third": 1.0}
        result = analyzed(simulate(36, factors, paired_sd=0.004, seed=seed, trend_ms_per_process=0.1), factors)
        called_zero += not (result["arms"]["no-effect"]["interval"][0] <= 0 <= result["arms"]["no-effect"]["interval"][1])
        called_pair += not result["noise"]["identical_arms_agree"]
    assert called_zero / trials <= 0.06, called_zero
    assert called_pair / trials <= 0.06, called_pair


def test_the_same_drift_with_the_baseline_always_first_would_create_one_so_the_order_matters():
    """The control for the test above: a fixed order lets the trend through as a false difference."""
    factors = {"no-effect": 1.0, "other": 1.0, "third": 1.0}
    names = ["off", "off-again", "other", "third", "no-effect"]  # the baseline first, this arm last
    fixed = [names for _ in range(36)]
    called = 0
    for seed in range(100):
        result = analyzed(simulate(36, factors, paired_sd=0.004, seed=seed, trend_ms_per_process=0.1, order=fixed),
                          factors)
        called += result["arms"]["no-effect"]["interval"][0] > 0
    assert called >= 90


@pytest.mark.parametrize("rounds", [2, 3, 5])
def test_too_few_rounds_give_cannot_tell_even_for_a_large_difference(rounds):
    result = analyzed(simulate(rounds, {"planted": 1.10}, paired_sd=0.004, seed=0), {"planted": 1.10})
    assert result["arms"]["planted"]["interval"] is None
    assert result["arms"]["planted"]["verdict"] == "cannot tell"
    assert "rounds" in result["arms"]["planted"]["why"]


def test_an_effect_at_the_limit_is_cannot_tell_not_a_verdict():
    result = analyzed(simulate(20, {"planted": 1.01}, paired_sd=0.011, seed=5), {"planted": 1.01})
    assert result["arms"]["planted"]["verdict"] == "cannot tell"


def test_clear_cases_get_their_verdicts():
    factors = {"cheap": 1.002, "dear": 1.03}
    result = analyzed(simulate(30, factors, paired_sd=0.004, seed=2), factors)
    assert result["arms"]["cheap"]["verdict"] == "met"
    assert result["arms"]["dear"]["verdict"] == "not met"


def test_identical_arms_called_different_make_every_verdict_cannot_tell():
    factors = {"cheap": 1.002}
    times = simulate(30, factors, paired_sd=0.004, seed=2)
    for r in times:
        times[r]["off-again"] *= 1.01  # the "identical" arm is 1% slower: the noise is not under control
    result = analyzed(times, factors)
    assert result["noise"]["identical_arms_agree"] is False
    assert result["arms"]["cheap"]["verdict"] == "cannot tell"
    assert "identical" in result["arms"]["cheap"]["why"]


def test_a_run_with_a_problem_gives_cannot_tell_with_the_problem_named():
    factors = {"cheap": 1.002}
    result = analyzed(simulate(30, factors, paired_sd=0.004, seed=2), factors,
                      problems=["the GPU was shared: 1 other process"])
    assert result["arms"]["cheap"]["verdict"] == "cannot tell"
    assert "shared" in result["arms"]["cheap"]["why"]


def test_an_arm_is_paired_with_its_own_baseline():
    arms = arms_with(routers=1.0)
    arms.append(oh.normalize_arm({"name": "k3", "init": {"capture": "routers"}, "baseline": "routers"},
                                 default_limit=0.005))
    times = {r: {"off": 100.0, "off-again": 100.0, "routers": 101.0, "k3": 101.0 * 1.002} for r in range(10)}
    result = oh.analyze(times, arms)
    assert result["arms"]["routers"]["median"] == pytest.approx(0.01)
    assert result["arms"]["k3"]["median"] == pytest.approx(0.002)
    assert result["arms"]["k3"]["baseline"] == "routers"


def test_the_smallest_difference_told_is_the_identical_pairs_half_width_and_rounds_needed_follow():
    result = analyzed(simulate(30, {}, paired_sd=0.011, seed=4), {})
    noise = result["noise"]
    lo, hi = result["arms"]["off-again"]["interval"]
    assert noise["smallest_difference_told"] == pytest.approx((hi - lo) / 2)
    assert noise["paired_sd"] == pytest.approx(0.011, rel=0.35)
    needed = oh.rounds_needed(noise["paired_sd"], 0.005)
    assert needed == noise["rounds_needed"]["0.01"]


def test_rounds_needed_by_formula_agree_with_a_simulation_of_the_harness_interval():
    """The planning formula and the harness's own interval, simulated, agree within about 25%."""
    sd, true, limit = 0.011, 0.005, 0.01
    n = oh.rounds_needed(sd, limit - true)
    rate = oh.simulate_power(n, sd, true, limit, trials=600, seed=0)
    assert 0.8 <= rate <= 0.97, (n, rate)


# ---- the round layout ------------------------------------------------------------------------


def test_every_round_holds_every_arm_once_and_each_block_puts_each_arm_in_each_position_once():
    names = ["off", "off-again", "a", "b"]
    orders = oh.round_orders(names, 12, seed=7)
    assert all(sorted(o) == sorted(names) for o in orders)
    for block in range(3):
        rows = orders[4 * block: 4 * block + 4]
        for position in range(4):
            assert sorted(row[position] for row in rows) == sorted(names)


def test_the_orders_follow_the_seed():
    names = ["off", "off-again", "a"]
    assert oh.round_orders(names, 9, seed=1) == oh.round_orders(names, 9, seed=1)
    assert oh.round_orders(names, 9, seed=1) != oh.round_orders(names, 9, seed=2)
    assert oh.round_orders(names, 9, seed=1)[:4] == oh.round_orders(names, 4, seed=1)  # a resume gets the same


def test_the_baseline_is_not_always_first():
    orders = oh.round_orders(["off", "off-again", "a"], 30, seed=0)
    assert len({o.index("off") for o in orders}) == 3


# ---- the arms --------------------------------------------------------------------------------


def test_every_run_has_two_capture_off_arms_and_the_ready_made_sets():
    arms = oh.resolve_arms("routers", [], None, default_limit=0.01)
    assert [a["name"] for a in arms] == ["off", "off-again", "routers"]
    assert arms[0]["init"] is None and arms[1]["init"] is None
    assert arms[2]["init"] == {"capture": "routers"} and arms[2]["limit"] == 0.01
    full = {a["name"]: a for a in oh.resolve_arms("capture", [], None, default_limit=0.005)}
    assert full["full"]["init"] == {"capture": "full"} and full["full"]["limit"] is None
    grads = [a["name"] for a in oh.resolve_arms("grads", [], None, default_limit=0.01)]
    assert grads[:3] == ["off", "off-again", "routers"] and all(g.startswith("grads-") for g in grads[3:])


def test_an_arm_from_the_command_line_takes_any_init_settings_and_environment():
    arm = oh.parse_arm_option('pinned={"init": {"capture": "routers", "some_future_switch": 3}, '
                              '"env": {"TENSORQUILL_PINNED": "1"}, "limit": 0.005, "baseline": "routers"}')
    arms = oh.resolve_arms("routers", [arm], None, default_limit=0.01)
    pinned = arms[-1]
    assert pinned["name"] == "pinned"
    assert pinned["init"] == {"capture": "routers", "some_future_switch": 3}
    assert pinned["env"] == {"TENSORQUILL_PINNED": "1"}
    assert pinned["limit"] == 0.005 and pinned["baseline"] == "routers"


def test_arms_from_a_json_file(tmp_path):
    path = tmp_path / "arms.json"
    path.write_text(json.dumps([{"name": "g7", "init": {"capture": "routers", "grads": 7}}]))
    arms = oh.resolve_arms("none", [], path, default_limit=0.01)
    assert [a["name"] for a in arms] == ["off", "off-again", "g7"]


@pytest.mark.parametrize("arm, message", [
    ({"name": "off", "init": {}}, "reserved"),
    ({"name": "x", "init": {"sink": "y"}}, "sink"),
    ({"name": "x", "init": {"run_id": "y"}}, "run_id"),
    ({"name": "x", "init": {}, "baseline": "nowhere"}, "baseline"),
    ({"name": "x", "init": {}, "env": {"A": 1}}, "env"),
    ({"name": "x y", "init": {}}, "name"),
    ({"name": "x", "init": {}, "limit": "1%"}, "limit"),
    ({"name": "x", "init": {}, "colour": "red"}, "colour"),
])
def test_an_arm_the_harness_cannot_run_is_refused_before_anything_runs(arm, message):
    with pytest.raises(oh.ArmError, match=message):
        oh.resolve_arms("none", [arm], None, default_limit=0.01)


def test_an_arm_whose_flush_interval_does_not_divide_the_window_is_refused():
    arms = oh.resolve_arms("none", [{"name": "x", "init": {"capture": "routers", "every": 30}}], None,
                           default_limit=0.01)
    with pytest.raises(oh.ArmError, match="every"):
        oh.check_alignment(arms, warmup=50, steps=200, window=50)
    oh.check_alignment(arms[:2], warmup=50, steps=200, window=50)


def test_router_arms_on_a_model_without_a_router_spec_are_refused():
    arms = oh.resolve_arms("routers", [], None, default_limit=0.01)
    with pytest.raises(oh.ArmError, match="router"):
        oh.check_routers(arms, model_has_routers=False)
    oh.check_routers(arms, model_has_routers=True)
    opted_out = oh.resolve_arms("none", [{"name": "dense", "init": {"capture": "full", "routers": None}}], None,
                                default_limit=0.01)
    oh.check_routers(opted_out, model_has_routers=False)


def test_the_init_settings_an_arm_process_passes():
    routers = {"r": object()}
    assert oh.init_kwargs({"capture": "routers"}, routers, sink="S", run_id="R") == {
        "capture": "routers", "routers": routers, "sink": "S", "run_id": "R"}
    assert oh.init_kwargs({"capture": "full", "routers": None}, routers, sink="S", run_id="R") == {
        "capture": "full", "sink": "S", "run_id": "R"}


# ---- the GPU is not shared -------------------------------------------------------------------


def test_nvidia_smi_output_is_read():
    apps = oh.parse_smi_apps("1234, 1500\n5678, 300\n")
    assert apps == [{"pid": 1234, "used_mib": 1500.0}, {"pid": 5678, "used_mib": 300.0}]
    assert oh.parse_smi_apps("") == [] and oh.parse_smi_apps("No running processes found\n") == []
    gpu = oh.parse_smi_gpu("3, 512, 41, 1410, 65.3, Default\n")
    assert gpu == {"utilization_gpu": 3.0, "memory_used_mib": 512.0, "temperature_c": 41.0, "clock_sm_mhz": 1410.0,
                   "power_w": 65.3, "compute_mode": "Default"}


def idle(processes=0, util=0.0):
    return {"where": "between arms", "processes": [{"pid": 9}] * processes, "utilization_gpu": util}


def during(processes):
    return {"where": "arm before timing", "processes": [{"pid": 1}] * processes, "utilization_gpu": 90.0}


def test_a_gpu_with_only_this_runs_processes_is_exclusive():
    seen = oh.exclusivity([idle(), idle(), during(1), during(1), idle()], method="nvml", device_type="cuda")
    assert seen["status"] == "exclusive" and seen["usable"]


def test_a_process_on_the_gpu_between_arms_marks_the_run_not_usable():
    seen = oh.exclusivity([idle(), idle(processes=1), during(1)], method="nvml", device_type="cuda")
    assert seen["status"] == "shared" and not seen["usable"] and seen["reasons"]


def test_a_second_process_while_an_arm_runs_marks_the_run_not_usable():
    seen = oh.exclusivity([idle(), during(2), idle()], method="nvidia-smi", device_type="cuda")
    assert seen["status"] == "shared" and not seen["usable"]


def test_the_gpu_busy_while_this_run_is_idle_marks_it_not_usable():
    seen = oh.exclusivity([idle(util=40.0), during(1)], method="nvml", device_type="cuda")
    assert seen["status"] == "shared" and not seen["usable"]


def test_a_reader_that_cannot_see_processes_is_said_so_not_taken_as_proof():
    """In a container the process list may be empty even for this run's own arm: utilization is all
    that was checked, and the result says so."""
    seen = oh.exclusivity([idle(), during(0), during(0), idle()], method="nvidia-smi", device_type="cuda")
    assert seen["status"] == "utilization only" and seen["usable"] and seen["reasons"]


def test_no_reader_on_cuda_is_not_checked_and_not_usable():
    seen = oh.exclusivity([], method=None, device_type="cuda")
    assert seen["status"] == "not checked" and not seen["usable"]


def test_off_cuda_the_check_does_not_apply():
    seen = oh.exclusivity([], method=None, device_type="cpu")
    assert seen["status"] == "not applicable"


# ---- the status and exit code ----------------------------------------------------------------


def test_status_and_exit_follow_the_run_sheet():
    factors = {"cheap": 1.002}
    good = analyzed(simulate(30, factors, paired_sd=0.004, seed=3), factors)  # a seed whose identical pair agrees
    assert oh.overall(good, errors={}, dry_run=False) == ("pass", 0)
    assert oh.overall(good, errors={}, dry_run=True) == ("dry_run", 0)
    assert oh.overall(good, errors={"cheap": ["boom"]}, dry_run=False) == ("fail", 1)
    dear = analyzed(simulate(30, {"dear": 1.03}, paired_sd=0.004, seed=3), {"dear": 1.03})
    assert oh.overall(dear, errors={}, dry_run=False) == ("fail", 1)
    few = analyzed(simulate(4, factors, paired_sd=0.004, seed=2), factors)
    assert oh.overall(few, errors={}, dry_run=False) == ("not_proven", 3)


# ---- the whole script on CPU: it runs, nothing more ------------------------------------------


def run_harness(*args, timeout=900):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC), str(EXAMPLES)]))
    return subprocess.run([sys.executable, str(EXAMPLES / "overhead_harness.py"), *args], env=env,
                          capture_output=True, text=True, timeout=timeout)


def test_the_cpu_dry_run_runs_every_arm_in_a_fresh_process_and_writes_its_files(tmp_path):
    out = tmp_path / "run"
    proc = run_harness("--out", str(out), "--model", "toy", "--device", "cpu", "--rounds", "2",
                       "--warmup", "50", "--steps", "50", "--window", "50", "--compile", "default",
                       "--arm", 'grads-1={"init": {"capture": "routers", "grads": 1}, "env": {"AQ3_MARK": "on"}}',
                       "--seed", "11")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads((out / "result.json").read_text())
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["status"] == "dry_run" and verdict["code"] == 0
    assert [a["name"] for a in result["arms"]] == ["off", "off-again", "routers", "grads-1"]
    assert result["seed"] == 11 and result["orders"] == oh.round_orders(["off", "off-again", "routers", "grads-1"], 2, 11)
    pids = set()
    for name in ("off", "off-again", "routers", "grads-1"):
        for r in range(2):
            arm = json.loads((out / "arms" / f"{name}-r{r}.json").read_text())
            assert arm["state"] == "measured", arm
            assert len(arm["window_ms"]) == 1 and arm["block_ms"] > 0
            pids.add(arm["pid"])
            if name == "grads-1":
                assert arm["env_applied"] == {"AQ3_MARK": "on"} and arm["records"] == 100
            if name.startswith("off"):
                assert arm["records"] is None
    assert len(pids) == 8  # a fresh process for every arm and round
    assert all(v["verdict"] == "cannot tell" for k, v in verdict["arms"].items() if k != "off")
    assert result["gpu_check"]["status"] == "not applicable"
    table = (out / "table.txt").read_text()
    assert "routers" in table and "cannot tell" in table and "dry run" in table


def test_a_resumed_run_completes_the_planned_rounds_and_refuses_more_without_saying_so(tmp_path):
    out = tmp_path / "run"
    common = ("--out", str(out), "--model", "toy", "--device", "cpu", "--warmup", "50", "--steps", "50",
              "--window", "50", "--compile", "none", "--arm-set", "none", "--seed", "3")
    first = run_harness(*common, "--rounds", "1")
    assert first.returncode == 0, first.stdout + first.stderr
    (out / "arms" / "off-again-r0.json").unlink()  # as if the session ended inside round 1
    again = run_harness(*common, "--rounds", "1", "--resume")
    assert again.returncode == 0, again.stdout + again.stderr
    assert "off-again round 1" in again.stdout and "off round 1" not in again.stdout.replace("off-again round 1", "")
    more = run_harness(*common, "--rounds", "2", "--resume")
    assert more.returncode == 2 and "--extend" in more.stdout + more.stderr
    extended = run_harness(*common, "--rounds", "2", "--resume", "--extend")
    assert extended.returncode == 0, extended.stdout + extended.stderr
    result = json.loads((out / "result.json").read_text())
    assert result["extended_after_a_look"] == [{"from": 1, "to": 2}]
    assert "added after a look" in (out / "table.txt").read_text()


def test_an_arm_the_harness_refuses_exits_2_before_any_process_starts(tmp_path):
    out = tmp_path / "run"
    proc = run_harness("--out", str(out), "--model", "toy", "--device", "cpu", "--rounds", "2",
                       "--arm", 'x={"init": {"capture": "routers", "every": 30}}')
    assert proc.returncode == 2
    assert not (out / "arms").exists()


def test_a_baseline_that_cannot_run_stops_the_run_as_not_proven(tmp_path):
    """Capture off failing in the first round says this model cannot be measured in this mode; the
    harness stops there instead of running every other arm into the same failure."""
    out = tmp_path / "run"
    proc = run_harness("--out", str(out), "--model", "toy", "--device", "cpu", "--rounds", "3", "--warmup", "50",
                       "--steps", "50", "--window", "50", "--compile", "none", "--arm-set", "none",
                       "--fail-baseline-for-test")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["status"] == "not_proven"
    assert "capture off" in verdict["reasons"][0]
    assert json.loads((out / "arms" / "off-r0.json").read_text())["state"] == "error"
    assert not list((out / "arms").glob("*-r1.json"))  # stopped in round 1: no arm ran again
