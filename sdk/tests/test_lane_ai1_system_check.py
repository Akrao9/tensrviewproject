"""Lane AI-1: GPU memory, NVML and data-wait readings against readings taken alongside
(`examples/gpu_system_check.py`).

The three judges are shown passing and failing on made-up readings, the CUDA and NVML ones too (no GPU
is needed to judge numbers): an allocator reading off by more than the tolerance, `nvidia-smi` using
less than TensorQuill says is reserved, an NVML temperature off, no `nvidia-smi` sample near a reading,
and `nvidia-ml-py` absent, which is not proven rather than failed. Then the script runs on CPU: NVML
and CUDA memory absent and said to be by TensorQuill, the data wait checked, exit 3 (not proven); a
data-wait failure there exits 1, not 3.
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import gpu_system_check as gs  # noqa: E402

MIB = gs.MIB
CUDA = torch.device("cuda")


class FakeSmi:
    def __init__(self, samples):
        self.samples = samples

    def nearest(self, t):
        return gs.Smi.nearest(self, t)


def data(steps=20, wait=0.01):
    return gs.SleepyBatches(steps, 1, 1, wait)


def waits(d, extra=0.002, overrides=None):
    rows = []
    for s in range(d.steps):
        w = d.delay(s) + extra
        if overrides and s in overrides:
            w = overrides[s]
        rows.append({"step": s, "data_wait": w, "step_time": w + 0.003})
    return rows


def judged(fn, *a, **k):
    checks = gs.Checks()
    fn(checks, *a, **k)
    return checks, [c["name"] for c in checks.items if not c["ok"]]


def test_data_wait_within_the_injected_sleep_and_slack_passes():
    d = data()
    checks, failed = judged(gs.check_data_wait, waits(d), d, 0.015)
    assert not failed and len(checks.items) == 5


def test_data_wait_below_the_sleep_or_far_over_it_fails():
    d = data()
    assert judged(gs.check_data_wait, waits(d, overrides={3: 0.005}), d, 0.015)[1] == ["data_wait is at least the injected sleep"]
    late = judged(gs.check_data_wait, waits(d, overrides={4: 0.2}), d, 0.015)[1]
    assert "data_wait is at most the sleep plus 100 ms on every step" in late
    many = judged(gs.check_data_wait, waits(d, extra=0.03), d, 0.015)[1]
    assert many == ["data_wait is at most the sleep plus 15 ms on 98% of steps"]
    gone = [r for r in waits(d) if r["step"] != 7]
    assert judged(gs.check_data_wait, gone, d, 0.015)[1] == ["every step has a data_wait"]


def memory_case(tq_alloc=100 * MIB, used=900, reserved=200 * MIB):
    systems = {49: {"cuda": {"allocated": tq_alloc, "reserved": reserved, "peak_allocated": 150 * MIB, "total": 80000 * MIB}}}
    alongside = {49: {"t": 10.0, "allocated": 100 * MIB, "reserved": 200 * MIB, "peak": 150 * MIB}}
    smi = FakeSmi([{"t": 10.1, "memory.used": used, "memory.total": 81000}])
    return systems, alongside, CUDA, smi


def test_cuda_memory_pairs_at_each_flush_and_fails_when_off():
    systems, alongside, dev, smi = memory_case()
    checks, failed = judged(gs.check_memory, systems, alongside, dev, smi, total=80000 * MIB)
    assert not failed and not checks.not_proven
    systems, alongside, dev, smi = memory_case(tq_alloc=200 * MIB)
    assert judged(gs.check_memory, systems, alongside, dev, smi, total=80000 * MIB)[1] == ["step 49: allocated against torch.cuda"]
    systems, alongside, dev, smi = memory_case(used=100)
    assert judged(gs.check_memory, systems, alongside, dev, smi, total=80000 * MIB)[1] == [
        "step 49: nvidia-smi memory.used covers TensorQuill's reserved"]
    systems, alongside, dev, smi = memory_case()
    assert judged(gs.check_memory, systems, {**alongside, 99: alongside[49]}, dev, smi, total=80000 * MIB)[1] == [
        "a CUDA memory reading at every flush"]


def nvml_case(temp=60, dt=0.2, note=None):
    reading = {"read_at": 100.0, "temperature_c": temp, "sm_clock_mhz": 1400, "power_w": 250.0, "power_limit_w": 400.0,
               "ecc_uncorrected": 0, "ecc_corrected": 0}
    systems = {s: {"nvml": dict(reading, read_at=100.0 + i)} for i, s in enumerate((99, 149, 199))}
    if note:
        systems = {49: {"nvml": {"unavailable": note}}}
    smi = FakeSmi([{"t": 100.0 + i + dt, "temperature.gpu": 61, "clocks.sm": 1410, "power.draw": 240.0,
                    "enforced.power.limit": 400.0, "ecc.errors.uncorrected.volatile.total": None,
                    "ecc.errors.corrected.volatile.total": None} for i in range(3)])
    return systems, CUDA, smi


def test_nvml_against_nvidia_smi_by_read_at():
    checks, failed = judged(gs.check_nvml, *nvml_case())
    assert not failed and not checks.not_proven and len(checks.items) == 1 + 3 * 4  # ECC [N/A]: left out
    hot = judged(gs.check_nvml, *nvml_case(temp=70))[1]
    assert len(hot) == 3 and all("temperature_c" in name for name in hot)
    far = judged(gs.check_nvml, *nvml_case(dt=5.0))[1]
    assert far == ["an nvidia-smi sample within 1 s of the reading"] * 3


def test_nvidia_smi_without_usable_samples_is_not_proven_not_failed():
    systems, dev, _ = nvml_case()
    smi = FakeSmi([]); smi.errors = ["CalledProcessError: Field \"enforced.power.limit\" is not a valid field"]
    checks, failed = judged(gs.check_nvml, systems, dev, smi)
    assert failed == [] and "no usable sample" in checks.not_proven[0]
    systems, alongside, dev, smi = memory_case()
    smi.samples[0]["memory.used"] = None  # [N/A]
    smi.errors = []
    checks, failed = judged(gs.check_memory, systems, alongside, dev, smi, total=80000 * MIB)
    assert failed == [] and "no usable nvidia-smi memory sample" in checks.not_proven[0]


def test_the_scripts_own_timer_must_agree_with_the_recorded_data_wait():
    d = data()
    rows = waits(d)
    own = {r["step"]: r["data_wait"] for r in rows}
    assert not judged(gs.check_data_wait, rows, d, 0.015, own)[1]
    own[6] += 0.02
    assert judged(gs.check_data_wait, rows, d, 0.015, own)[1] == [
        "data_wait is within 5 ms of the script's own timer around next()"]


def test_nvml_absent_on_a_gpu_is_not_proven_not_failed():
    checks, failed = judged(gs.check_nvml, *nvml_case(note="pynvml is not installed (pip install nvidia-ml-py): x"))
    assert not failed and checks.not_proven and "pynvml" in checks.not_proven[0]


def test_on_cpu_the_script_checks_the_data_wait_and_exits_not_proven(tmp_path, monkeypatch):
    # Real sleeps under a loaded test run: the upper bounds are widened here, the lower bound is not.
    monkeypatch.setattr(gs, "OUTLIER_MS", 5000.0)
    monkeypatch.setattr(gs, "OWN_TIMER_S", 1.0)
    out = tmp_path / "sys"
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        code = gs.main(["--out", str(out), "--steps", "200", "--wait-ms", "5", "--wait-slack-ms", "5000"])
    if torch.cuda.is_available():
        return  # a GPU machine runs the real thing; this test is the CPU path
    assert code == gs.NOT_PROVEN, printed.getvalue()
    result = json.loads((out / "system.json").read_text())
    assert result["status"] == "not_proven" and all(c["ok"] for c in result["checks"])
    assert any("NVML is absent" in why for why in result["not_proven"])
    assert {c["part"] for c in result["checks"]} == {"data_wait", "cuda_memory", "nvml"}


def test_a_data_wait_failure_on_cpu_exits_1_not_3(tmp_path, monkeypatch):
    if torch.cuda.is_available():
        return
    real = gs.SleepyBatches.delay
    monkeypatch.setattr(gs.SleepyBatches, "__getitem__",
                        lambda self, i: (gs.time.sleep(real(self, i) / 2), torch.randn(self.batch, self.dim))[1])
    with contextlib.redirect_stdout(io.StringIO()):
        assert gs.main(["--out", str(tmp_path / "sys"), "--steps", "200", "--wait-ms", "5"]) == 1
