"""Router counting, variants side by side, in a compiled graph on this machine's GPU. A measurement: nothing is judged.

Why: on an A100 (a2e652d, Nov 23, 2026) router-only capture cost about 5% of a Granite step, against under 1% at
d987509. The profile's new kernels (indexing_backward_kernel_stride_1<long>, a radix sort, an async assert, a
min/max reduce, each 24 times a step) are what `index_add_` on an int64 tensor runs on CUDA; lane AD3 (8676f95)
changed the count from a compare matrix to that. This script counts the same assignments each way and times them.

  compare     (flat == expert_ids).sum(0): the count before lane AD3 (a bool matrix of assignments by experts)
  index_i64   zeros(int64).index_add_(0, slot, ones): the count now
  index_i32   the same into int32, widened to int64 after: Inductor can lower it itself (Triton has no int64 atomic add)
  scatter_i32 zeros(int32).scatter_add_(0, slot, ones), widened after

Usage: python router_count_ab.py [--out DIR]. Needs only PyTorch; uses CUDA if there is one (CPU is a dry run).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

SHAPES = {
    # name: (tokens, top_k, experts, routers)
    "granite": (2 * 512, 8, 32, 24),  # the benchmark's batch of 2 x 512 tokens, top-8 of 32, 24 routers
    "large": (16384, 8, 256, 4),  # lane AD3's case: the compare matrix is 33.5 million entries a router
}


def compare(flat, experts, width):
    ids = torch.arange(experts, device=flat.device)
    hits = (flat.reshape(-1, 1) == ids).sum(0)
    return torch.cat([hits, torch.zeros(width - experts, dtype=hits.dtype, device=flat.device)])


def index_i64(flat, experts, width):
    hits = torch.zeros(width, dtype=torch.int64, device=flat.device)
    hits.index_add_(0, flat, torch.ones_like(flat))
    return hits


def index_i32(flat, experts, width):
    hits = torch.zeros(width, dtype=torch.int32, device=flat.device)
    hits.index_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    return hits.to(torch.int64)


def scatter_i32(flat, experts, width):
    hits = torch.zeros(width, dtype=torch.int32, device=flat.device)
    hits.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    return hits.to(torch.int64)


VARIANTS = {"compare": compare, "index_i64": index_i64, "index_i32": index_i32, "scatter_i32": scatter_i32}


def make_step(count, top_k, experts, routers, width):
    """One 'step': each router's logits from a small matmul, its top-k, and the count added to a running total,
    as the hook does inside the model's compiled graph. `count=None` is the control: the same model work, no count."""

    def step(hidden, weights, counts):
        out = hidden
        for r in range(routers):
            logits = out @ weights[r]
            idx = logits.topk(top_k, dim=-1).indices
            out = out + logits.softmax(-1) @ weights[r].t() * 1e-3
            if count is not None:
                counts[r].add_(count(idx.detach().reshape(-1).to(torch.int64), experts, width))
        return out.sum()

    return step


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure(name, count, shape, device, backend, warmup, steps):
    tokens, top_k, experts, routers = shape
    width = 2 * experts + 7
    torch.manual_seed(0)
    hidden = torch.randn(tokens, 256, device=device)
    weights = torch.randn(routers, 256, experts, device=device) * 0.05
    counts = torch.zeros(routers, width, dtype=torch.int64, device=device)
    step = make_step(count, top_k, experts, routers, width)
    if backend != "eager":
        torch._dynamo.reset()
        step = torch.compile(step, backend=backend)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmup):
            step(hidden, weights, counts)
        sync(device)
        counts.zero_()
        times = []
        for _ in range(steps):
            start = time.perf_counter()
            step(hidden, weights, counts)
            sync(device)
            times.append((time.perf_counter() - start) * 1e3)
    kernels = []
    if device.type == "cuda":
        from torch.profiler import ProfilerActivity, profile

        with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            for _ in range(10):
                step(hidden, weights, counts)
            sync(device)
        rows = sorted(prof.key_averages(), key=lambda e: -getattr(e, "self_device_time_total", 0))
        kernels = [{"name": e.key[:110], "calls_per_step": e.count / 10,
                    "device_us_per_step": getattr(e, "self_device_time_total", 0) / 10}
                   for e in rows[:8] if getattr(e, "self_device_time_total", 0) > 0]
    return {"variant": name, "median_ms": statistics.median(times), "mean_ms": statistics.fmean(times),
            "max_ms": max(times), "steps": steps,
            "peak_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
            "counts_first_router": counts[0, :experts].tolist()[:8], "counts_total": int(counts.sum()),
            "counts": counts.cpu(), "kernels": kernels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="router-count-ab")
    parser.add_argument("--backend", default=None, help="inductor on CUDA by default; eager on CPU")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--shapes", default="granite,large")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = args.backend or ("inductor" if device.type == "cuda" else "eager")
    where = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU (a dry run: says nothing about a GPU)"
    print(f"torch {torch.__version__} on {where} | backend {backend} | {args.rounds} rounds x {args.steps} steps")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"torch": torch.__version__, "device": where, "backend": backend, "shapes": {}}
    for shape_name in args.shapes.split(","):
        shape = SHAPES[shape_name]
        tokens, top_k, experts, routers = shape
        print(f"\n== {shape_name}: {tokens} tokens, top-{top_k} of {experts} experts, {routers} routers")
        rows, reference = [], None
        for name, count in [("no count", None), *VARIANTS.items()]:
            rounds = [measure(name, count, shape, device, backend, args.warmup, args.steps) for _ in range(args.rounds)]
            best = sorted(rounds, key=lambda r: r["median_ms"])[len(rounds) // 2]  # the median round
            counts = best.pop("counts")
            for r in rounds:
                r.pop("counts", None)
            if count is not None:
                if reference is None:
                    reference = counts
                best["equals_compare"] = bool(torch.equal(counts, reference))
            best["round_medians_ms"] = [r["median_ms"] for r in rounds]
            rows.append(best)
        base = rows[0]["median_ms"]
        print(f"{'variant':<12} {'median ms':>10} {'added ms':>9} {'max ms':>8} {'peak MiB':>9}  counts equal")
        for row in rows:
            row["added_ms"] = row["median_ms"] - base
            peak = f"{row['peak_mib']:.0f}" if row["peak_mib"] is not None else "n/a"
            print(f"{row['variant']:<12} {row['median_ms']:>10.3f} {row['added_ms']:>+9.3f} {row['max_ms']:>8.2f} {peak:>9}  "
                  f"{row.get('equals_compare', '')}")
        for row in rows[1:]:
            print(f"  {row['variant']}: top device kernels per step")
            for k in row["kernels"][:5]:
                print(f"    {k['calls_per_step']:6.1f} x {k['device_us_per_step']:9.1f} us  {k['name']}")
        result["shapes"][shape_name] = {"tokens": tokens, "top_k": top_k, "experts": experts, "routers": routers, "rows": rows}
    (out / "router_count_ab.json").write_text(json.dumps(result, indent=1))
    print(f"\nMEASURED: {out / 'router_count_ab.json'} (nothing is judged)")


if __name__ == "__main__":
    main()
