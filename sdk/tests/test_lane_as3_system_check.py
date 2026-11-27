"""Lane AS-3: the two `system-check` tolerances of `examples/gpu_system_check.py`, as the owner set
them on Nov 27, 2026, after the A100 run of Nov 23 (`a2e652d`) failed both for reasons that were not
TensorQuill's.

- **Data wait.** That run's loader returned about 24 ms after its sleep on every step with capture off
  as well as on, so every step was over "the sleep plus 15 ms". The rule now allows 15 ms over the
  loader's own delay, measured in the same run with capture off, per sleep length, as a median. Held
  here end to end on CPU by a loader that does a fixed 60 ms of work after its sleep (on the old rule
  the script exits 1 on the 98% line; on the new one it exits 3, not proven, as any CPU run does), and
  by stand-in records: a constant loader delay passes, the capture adding more than the slack on top of
  it fails, one slow capture-off step does not move the measured delay, a loader quicker than its
  sleep allows no less than the old rule.
- **NVML power.** Two power readings were apart from the nearest `nvidia-smi` sample while the load
  changed. The rule now compares TensorQuill's median `power_w` with `nvidia-smi`'s median
  `power.draw` over the window from the first NVML reading to the last; 25% or 25 W stays. Held by
  stand-in readings: one reading far from its nearest sample, with the medians close, passes; medians
  more than 25% apart fail; `nvidia-smi` samples outside the window are not counted; too few samples
  in the window is not proven, not failed. The other NVML pairs are still judged reading by reading.

No GPU is used: the judges are given made-up numbers, as in `test_lane_ai1_system_check.py`.
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import gpu_system_check as gs  # noqa: E402

CUDA = torch.device("cuda")


class FakeSmi:
    def __init__(self, samples, errors=()):
        self.samples, self.errors = samples, list(errors)

    def nearest(self, t):
        return gs.Smi.nearest(self, t)


def judged(fn, *a, **k):
    checks = gs.Checks()
    out = fn(checks, *a, **k)
    return checks, [c["name"] for c in checks.items if not c["ok"]], out


# ---- data wait: end to end on CPU ---------------------------------------------------------------

def _slow_loader(monkeypatch, extra_s):
    """Every batch takes its sleep plus `extra_s` of the loader's own, capture on or off."""
    real = gs.SleepyBatches.delay

    def item(self, i):
        gs.time.sleep(real(self, i) + extra_s)
        return torch.randn(self.batch, self.dim)

    monkeypatch.setattr(gs.SleepyBatches, "__getitem__", item)


def test_a_loader_late_by_its_own_work_passes_when_capture_adds_nothing(tmp_path, monkeypatch):
    # The A100 case on CPU: the loader is late by 60 ms on every step whether or not TensorQuill is on.
    # Under the old rule (the sleep plus the slack) every step is over the 30 ms slack and the script
    # exits 1; under the new one the slack is on top of the delay measured with capture off.
    # The upper bounds that are not under test are widened for a loaded machine, as in lane AI-1's test.
    if torch.cuda.is_available():
        return  # this is the CPU path; a GPU machine runs the script itself
    monkeypatch.setattr(gs, "OUTLIER_MS", 5000.0)
    monkeypatch.setattr(gs, "OWN_TIMER_S", 1.0)
    monkeypatch.setattr(gs, "OFF_STEPS", 20, raising=False)  # the default of --off-steps, kept short here
    _slow_loader(monkeypatch, 0.060)
    out = tmp_path / "sys"
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        code = gs.main(["--out", str(out), "--steps", "50", "--every", "10", "--wait-ms", "5", "--wait-slack-ms", "30"])
    result = json.loads((out / "system.json").read_text())
    failed = [c["name"] for c in result["checks"] if not c["ok"]]
    assert failed == [] and code == gs.NOT_PROVEN, (failed, printed.getvalue())
    own = result["data_wait"]["loader_own_delay"]
    assert own["steps"] == 20 and {row["sleep_ms"] for row in own["by_sleep"]} == {5.0, 20.0}
    assert all(row["delay_ms"] >= 60.0 for row in own["by_sleep"])  # the 60 ms are measured, not assumed


def test_the_capture_off_steps_must_number_at_least_twenty(tmp_path):
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            gs.main(["--out", str(tmp_path / "sys"), "--off-steps", "19"])
        except SystemExit as stop:
            assert stop.code == 2
        else:
            raise AssertionError("--off-steps 19 was accepted")
    assert not (tmp_path / "sys").exists()


# ---- data wait: stand-in records ----------------------------------------------------------------

def data(steps=50, wait=0.01):
    return gs.SleepyBatches(steps, 1, 1, wait)


def rows(d, extra):
    return [{"step": s, "data_wait": d.delay(s) + extra, "step_time": d.delay(s) + extra + 0.003} for s in range(d.steps)]


def off_timer(d, extra, overrides=None):
    own = {s: d.delay(s) + extra for s in range(d.steps)}
    own.update(overrides or {})
    return own


SLACK_NAME = "data_wait is at most the sleep plus the loader's own delay plus 15 ms on 98% of steps"


def test_the_loaders_own_delay_is_the_median_per_sleep_length_and_never_below_zero():
    off = data(40)
    delay = gs.loader_own_delay(off_timer(off, 0.024, {0: 0.5, 5: 0.9}), off)  # a cold first step, a slow long one
    assert set(delay) == {0.01, 0.04}
    assert abs(delay[0.01] - 0.024) < 1e-9 and abs(delay[0.04] - 0.024) < 1e-9
    quick = gs.loader_own_delay({s: off.delay(s) - 0.002 for s in range(off.steps)}, off)
    assert quick == {0.01: 0.0, 0.04: 0.0}  # never less than the old rule allows


def test_a_constant_loader_delay_passes_and_the_capture_adding_more_than_the_slack_fails():
    on, off = data(), data(20)
    delay = gs.loader_own_delay(off_timer(off, 0.024), off)
    checks, failed, out = judged(gs.check_data_wait, rows(on, 0.030), on, 0.015, own_delay=delay)
    assert failed == [] and out["over_slack"] == []
    checks, failed, out = judged(gs.check_data_wait, rows(on, 0.024 + 0.016), on, 0.015, own_delay=delay)
    assert failed == [SLACK_NAME] and len(out["over_slack"]) == on.steps
    assert abs(out["over_slack"][0]["allowed_ms"] - (on.delay(0) + 0.024 + 0.015) * 1e3) < 1e-6


def test_the_long_sleeps_are_judged_against_their_own_measured_delay():
    # A loader whose own delay grows with the sleep: the long steps (one in ten) are judged against
    # the long steps' capture-off median, so they are not all over the slack.
    on, off = data(), data(20)
    late = lambda d: {s: d.delay(s) + (0.050 if s % 10 == 5 else 0.010) for s in range(d.steps)}  # noqa: E731
    delay = gs.loader_own_delay(late(off), off)
    recorded = [{"step": s, "data_wait": w, "step_time": w + 0.003} for s, w in late(on).items()]
    assert judged(gs.check_data_wait, recorded, on, 0.015, own_delay=delay)[1] == []
    pooled = {k: 0.010 for k in delay}  # one number for both lengths would fail every long step
    assert judged(gs.check_data_wait, recorded, on, 0.015, own_delay=pooled)[1] == [SLACK_NAME]


def test_without_a_measured_delay_the_rule_is_the_sleep_plus_the_slack():
    on = data()
    assert judged(gs.check_data_wait, rows(on, 0.020), on, 0.015)[1] == [SLACK_NAME]
    assert judged(gs.check_data_wait, rows(on, 0.010), on, 0.015)[1] == []


# ---- NVML power: stand-in readings ----------------------------------------------------------------

POWER_NAME = "power_w median against nvidia-smi power.draw median over the steady window"


def nvml_case(powers=(250.0, 330.0, 250.0, 250.0), smi_power=240.0, before=None, smi_every=0.25):
    """NVML readings one second apart from t=100; `nvidia-smi` every `smi_every` s over them, and
    optionally samples at `before` W in the 3 s before the first reading."""
    systems = {}
    for i, watts in enumerate(powers):
        systems[99 + 50 * i] = {"nvml": {"read_at": 100.0 + i, "temperature_c": 60, "sm_clock_mhz": 1400,
                                         "power_w": watts, "power_limit_w": 400.0}}
    end = 100.0 + len(powers) - 1
    times = [97.0 + k * smi_every for k in range(int((end + 1 - 97.0) / smi_every))]
    samples = [{"t": t, "temperature.gpu": 61, "clocks.sm": 1410,
                "power.draw": before if before is not None and t < 100.0 else smi_power,
                "enforced.power.limit": 400.0} for t in times]
    return systems, CUDA, FakeSmi(samples)


def test_one_power_reading_far_from_its_nearest_sample_passes_when_the_medians_agree():
    checks, failed, out = judged(gs.check_nvml, *nvml_case())
    assert failed == [] and not checks.not_proven
    power = [c for c in checks.items if "power" in c["name"] and "limit" not in c["name"]]
    assert [c["name"] for c in power] == [POWER_NAME]
    window = out["power_window"]
    assert window["start"] == 100.0 and window["end"] == 103.0
    assert window["tensorquill_median_w"] == 250.0 and window["nvidia_smi_median_w"] == 240.0
    assert window["readings"] == 4 and len(window["nvidia_smi_w"]) == 13  # 100.0 to 103.0 every 0.25 s


def test_medians_more_than_25_percent_apart_fail():
    _, failed, _ = judged(gs.check_nvml, *nvml_case(powers=(330.0, 330.0, 330.0), smi_power=240.0))
    assert failed == [POWER_NAME]
    _, failed, _ = judged(gs.check_nvml, *nvml_case(powers=(60.0, 60.0, 60.0), smi_power=84.0))
    assert failed == []  # 24 W apart: within the 25 W floor, which stays


def test_samples_outside_the_window_are_not_counted():
    # Idle samples before the first reading would put nvidia-smi's median at 60 W if they counted.
    checks, failed, out = judged(gs.check_nvml, *nvml_case(before=60.0))
    assert failed == [] and out["power_window"]["nvidia_smi_median_w"] == 240.0


def test_too_few_samples_in_the_window_is_not_proven_not_failed():
    checks, failed, out = judged(gs.check_nvml, *nvml_case(powers=(250.0, 250.0, 250.0), smi_every=1.5))
    assert failed == [] and any("power" in why and "window" in why for why in checks.not_proven)
    checks, failed, out = judged(gs.check_nvml, *nvml_case(powers=(250.0, 250.0)))
    assert failed == ["NVML readings on at least three flushes"]
    assert any("power" in why for why in checks.not_proven)


def test_the_other_nvml_pairs_are_still_judged_reading_by_reading():
    systems, dev, smi = nvml_case()
    systems[149]["nvml"]["temperature_c"] = 70
    _, failed, _ = judged(gs.check_nvml, systems, dev, smi)
    assert len(failed) == 1 and "temperature_c" in failed[0] and "reading at" in failed[0]
