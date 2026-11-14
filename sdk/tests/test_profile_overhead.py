import sys
from pathlib import Path
from types import SimpleNamespace

from torch.autograd import DeviceType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import profile_overhead as po  # noqa: E402


def _event(name, start, end, device=DeviceType.CUDA):
    return SimpleNamespace(
        name=name, device_type=device,
        time_range=SimpleNamespace(start=start, end=end, elapsed_us=lambda: end - start),
    )


def _cpu(name, start, end):
    return _event(name, start, end, DeviceType.CPU)


def test_summarize_and_verdict_separate_host_time_from_gpu_work():
    # Two steps. Baseline: one matmul and one sync per step.
    base_events = [
        _event("gemm", 0, 400), _event("gemm", 1000, 1400),
        _cpu("cudaLaunchKernel", 0, 5), _cpu("cudaLaunchKernel", 1000, 1005),
        _cpu("cudaStreamSynchronize", 100, 400), _cpu("cudaStreamSynchronize", 1100, 1400),
        _cpu("aten::mm", 0, 10),
    ]
    base = po.summarize(base_events, steps=2, wall_ms=2.0)
    assert base["busy_ms"] == 0.4 and base["kernels"] == 1 and base["launches"] == 1
    assert base["syncs"] == 1 and base["sync_wait_ms"] == 0.3
    assert abs(base["idle_ms"] - 0.6) < 1e-9

    # Capture adds three tiny kernels per step, overlapping kernels count once in busy time.
    hook = [_event("triton_red_router", t, t + 10) for t in (500, 520, 540, 1500, 1520, 1540)]
    launches = [_cpu("cudaLaunchKernel", t, t + 5) for t in (500, 520, 540, 1500, 1520, 1540)]
    overlap = [_event("gemm_tail", 300, 450)]
    capture = po.summarize(base_events + hook + launches + overlap, steps=2, wall_ms=2.4)
    assert capture["kernels"] == 4.5 and capture["launches"] == 4
    assert abs(capture["busy_ms"] - (0.4 + 0.03 + 0.025)) < 1e-9

    text = po.report(("capture off", "capture=routers"), (1.0, 1.2), (base, capture))
    assert "triton_red_router" in text
    assert "syncs the CPU with the GPU 1 times per step" in text
    assert "host time" in text

    gpu_bound = po.verdict((1.0, 1.02), base, capture)
    assert "GPU work" in gpu_bound[-1]
    empty = po.summarize([_cpu("aten::mm", 0, 10)], steps=1, wall_ms=1.0)
    assert po.verdict((1.0, 1.1), empty, empty) == ["No GPU activity was recorded, so only the wall times apply."]
