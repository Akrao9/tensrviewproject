"""Find where TensorQuill capture's added time goes on a GPU: GPU work, kernel launches or syncs.

Runs the Granite benchmark's training step twice, with capture off and on, first timed
without the profiler, then under `torch.profiler` for one flush window (50 steps). For
each arm it reports, per step: wall time, GPU busy time, kernels, kernel launches, and
CPU syncs with the time the CPU spent waiting in them. The difference between the arms
says whether the overhead is GPU work (the hooks' kernels) or host time (launching them,
or waiting at syncs), and lists the kernels that capture added.

    python examples/profile_overhead.py --out runs/profile-a100
    python examples/profile_overhead.py --out runs/profile-a100 --capture full --trace

`--toy` runs a small MoE without downloads, for checking the script itself.
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import torch
from torch.autograd import DeviceType

import tensorquill

FLUSH_EVERY = 50
LAUNCH_CALLS = frozenset(
    {"cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx", "cudaGraphLaunch"}
)
# A blocking copy to the CPU (`.item()`, `.cpu()`, `.tolist()`) is a cudaMemcpyAsync
# followed by a cudaStreamSynchronize, so the synchronize calls count the syncs.
SYNC_CALLS = frozenset({"cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize"})
COPY_CALLS = frozenset({"cudaMemcpyAsync", "cudaMemcpy"})


def _union_us(intervals: Iterable[tuple[float, float]]) -> float:
    total, end = 0.0, -math.inf
    for start, stop in sorted(intervals):
        if stop <= end:
            continue
        total += stop - max(start, end)
        end = stop
    return total


def summarize(events: Iterable[Any], steps: int, wall_ms: float) -> dict[str, Any]:
    """Per-step totals from profiler events (`prof.events()`) over `steps` steps."""
    device, cpu = [], []
    for event in events:
        (cpu if event.device_type == DeviceType.CPU else device).append(event)
    calls = Counter(e.name for e in cpu if e.name in LAUNCH_CALLS | SYNC_CALLS | COPY_CALLS)
    kernel_time: Counter[str] = Counter()
    for e in device:
        kernel_time[e.name] += e.time_range.elapsed_us()
    busy_ms = _union_us((e.time_range.start, e.time_range.end) for e in device) / 1000
    sync_wait_ms = sum(e.time_range.elapsed_us() for e in cpu if e.name in SYNC_CALLS) / 1000
    return {
        "wall_ms": wall_ms / steps,
        "busy_ms": busy_ms / steps,
        "idle_ms": max(wall_ms - busy_ms, 0.0) / steps,
        "kernels": len(device) / steps,
        "launches": sum(n for name, n in calls.items() if name in LAUNCH_CALLS) / steps,
        "syncs": sum(n for name, n in calls.items() if name in SYNC_CALLS) / steps,
        "sync_wait_ms": sync_wait_ms / steps,
        "kernel_counts": Counter({k: v / steps for k, v in Counter(e.name for e in device).items()}),
        "kernel_ms": Counter({k: v / 1000 / steps for k, v in kernel_time.items()}),
    }


ROWS = (
    ("wall_ms", "Wall time under the profiler (ms)"),
    ("busy_ms", "GPU busy (ms)"),
    ("idle_ms", "GPU idle (ms)"),
    ("kernels", "GPU kernels and copies"),
    ("launches", "Kernel launches"),
    ("syncs", "CPU syncs"),
    ("sync_wait_ms", "CPU waiting in syncs (ms)"),
)


def report(labels: tuple[str, str], timed: tuple[float, float], stats: tuple[dict, dict], top: int = 12) -> str:
    base, capture = stats
    lines = [
        f"{'Per step':<36}{labels[0]:>18}{labels[1]:>18}{'added':>12}",
        f"{'Wall time, no profiler (ms)':<36}{timed[0]:>18.3f}{timed[1]:>18.3f}{timed[1] - timed[0]:>+12.3f}",
    ]
    for key, label in ROWS:
        a, b = base[key], capture[key]
        lines.append(f"{label:<36}{a:>18.3f}{b:>18.3f}{b - a:>+12.3f}")

    added = []
    for name in set(base["kernel_counts"]) | set(capture["kernel_counts"]):
        dn = capture["kernel_counts"].get(name, 0) - base["kernel_counts"].get(name, 0)
        dt = capture["kernel_ms"].get(name, 0) - base["kernel_ms"].get(name, 0)
        if dn > 0.5:
            added.append((dt, dn, name))
    if added:
        lines.append(f"\nKernels added per step by capture (top {top} by time):")
        for dt, dn, name in sorted(added, reverse=True)[:top]:
            lines.append(f"  {dn:>7.1f} × {dt * 1000:>8.1f} µs total  {name[:110]}")

    lines.append("")
    lines += verdict(timed, base, capture)
    return "\n".join(lines)


def verdict(timed: tuple[float, float], base: dict, capture: dict) -> list[str]:
    added_wall = timed[1] - timed[0]
    added_busy = capture["busy_ms"] - base["busy_ms"]
    added_launches = capture["launches"] - base["launches"]
    if not (base["kernels"] or capture["kernels"]):
        return ["No GPU activity was recorded, so only the wall times apply."]
    lines = []
    if base["syncs"] >= 1:
        lines.append(
            f"The run syncs the CPU with the GPU {base['syncs']:.0f} times per step without capture, "
            f"and the GPU is idle {base['idle_ms']:.1f} ms per step: the CPU cannot run ahead, so any "
            "added launch time shows up in the step time."
        )
    if added_wall <= 0:
        lines.append("Capture added no measurable wall time in this run.")
    elif added_busy >= 0.7 * added_wall:
        lines.append(
            f"Most of the added {added_wall:.2f} ms is GPU work (+{added_busy:.2f} ms busy): "
            "make the kernels above cheaper or fewer."
        )
    else:
        lines.append(
            f"Most of the added {added_wall:.2f} ms is host time: GPU busy rose only {added_busy:+.2f} ms, "
            f"with {added_launches:+.0f} launches and {capture['syncs'] - base['syncs']:+.2f} syncs per step. "
            "Fewer kernels per step (fusing the router hooks) or CUDA graphs would remove it."
        )
    return lines


# ---- models -----------------------------------------------------------------------


class _Out(NamedTuple):
    loss: torch.Tensor


class _ToyMoE(torch.nn.Module):
    def __init__(self, layers: int = 4, experts: int = 8, width: int = 64, top_k: int = 2):
        super().__init__()
        self.embed = torch.nn.Embedding(256, width)
        self.routers = torch.nn.ModuleList(torch.nn.Linear(width, experts) for _ in range(layers))
        self.mlps = torch.nn.ModuleList(torch.nn.Linear(width, width) for _ in range(layers))
        self.head = torch.nn.Linear(width, 256)
        self.top_k = top_k

    def forward(self, input_ids, labels=None, use_cache=False):
        x = self.embed(input_ids)
        for router, mlp in zip(self.routers, self.mlps):
            weights = router(x).softmax(-1).topk(self.top_k, dim=-1).values.sum(-1, keepdim=True)
            x = x + mlp(x) * weights
        logits = self.head(x)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
        return _Out(loss)


def build_toy(device: torch.device):
    torch.manual_seed(0)
    models = [_ToyMoE().to(device) for _ in range(2)]
    models[1].load_state_dict(models[0].state_dict())
    routers = {f"routers.{i}": tensorquill.Router(8, top_k=2) for i in range(4)}
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3) for m in models]
    batches = torch.randint(0, 256, (4 * FLUSH_EVERY, 2, 32), device=device)
    return models, optimizers, routers, batches


def build_granite(args):
    from moe_dogfood import build_pair, load_batches

    models, optimizers, routers = build_pair()
    batches = load_batches(args, args.warmup + max(args.steps, args.profile_steps))
    return models, optimizers, routers, batches


def train_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=batch, labels=batch, use_cache=False).loss
    loss.backward()
    optimizer.step()
    return loss.detach()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="new directory for captures and traces")
    parser.add_argument("--capture", choices=("routers", "full"), default="routers")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--steps", type=int, default=100, help="timed steps per round, without the profiler")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=FLUSH_EVERY, help="profiled steps per arm")
    parser.add_argument("--trace", action="store_true", help="also write a Chrome trace per arm (large)")
    parser.add_argument("--eager", action="store_true", help="do not compile")
    parser.add_argument("--toy", action="store_true", help="small MoE, no downloads, CPU allowed")
    args = parser.parse_args()
    if any(n <= 0 or n % FLUSH_EVERY for n in (args.warmup, args.steps, args.profile_steps)):
        parser.error("warmup, steps and profile steps must be positive multiples of 50")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.toy and device.type != "cuda":
        parser.error("a CUDA GPU is required; use --toy to check the script on CPU")

    args.out.mkdir(parents=True)
    torch._dynamo.reset()
    models, optimizers, routers, batches = build_toy(device) if args.toy else build_granite(args)
    labels = ("capture off", f"capture={args.capture}")
    name = torch.cuda.get_device_name() if device.type == "cuda" else "CPU"
    mode = "eager" if args.eager else "Inductor, CUDA graphs off"
    print(f"{torch.__version__} on {name} | {'toy MoE' if args.toy else 'Granite'} | {mode}", flush=True)

    with ExitStack() as stack:
        stack.enter_context(torch._dynamo.config.patch(skip_nnmodule_hook_guards=False))
        stack.enter_context(
            tensorquill.init(
                models[1], routers=routers, every=FLUSH_EVERY, capture=args.capture,
                run_id="profile", sink=tensorquill.JsonlSink(args.out / "profile.jsonl"),
            )
        )
        runners = models if args.eager else [
            torch.compile(m, backend="inductor", fullgraph=True, options={"triton.cudagraphs": False})
            for m in models
        ]
        for runner, optimizer in zip(runners, optimizers):
            for batch in batches[: args.warmup]:
                loss = train_step(runner, optimizer, batch)
            if not math.isfinite(loss.item()):
                raise RuntimeError("non-finite warmup loss")

        timed: list[list[float]] = [[], []]
        timed_batches = batches[args.warmup : args.warmup + args.steps]
        for round_index in range(args.rounds):
            for index in ((0, 1) if round_index % 2 == 0 else (1, 0)):
                _sync(device)
                start = time.perf_counter()
                for batch in timed_batches:
                    train_step(runners[index], optimizers[index], batch)
                _sync(device)
                timed[index].append((time.perf_counter() - start) * 1000 / args.steps)
                print(f"Round {round_index + 1} {labels[index]}: {timed[index][-1]:.3f} ms/step", flush=True)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        stats = []
        profiled = batches[args.warmup : args.warmup + args.profile_steps]
        for index in range(2):
            _sync(device)
            with torch.profiler.profile(activities=activities) as prof:
                start = time.perf_counter()
                for batch in profiled:
                    train_step(runners[index], optimizers[index], batch)
                _sync(device)
                wall_ms = (time.perf_counter() - start) * 1000
            stats.append(summarize(prof.events(), args.profile_steps, wall_ms))
            if args.trace:
                trace = args.out / f"trace-{'off' if index == 0 else args.capture}.json"
                prof.export_chrome_trace(str(trace))
                print(f"Trace: {trace}")
            del prof
            gc.collect()

    medians = (statistics.median(timed[0]), statistics.median(timed[1]))
    print()
    print(report(labels, medians, (stats[0], stats[1])))
    print(f"\nCapture written to {args.out / 'profile.jsonl'}")


if __name__ == "__main__":
    main()
