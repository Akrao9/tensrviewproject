"""Time the Python detectors on a synthetic 64-rank store, against the plan's budget.

    cd sdk
    pip install -e '.[flight]'
    python examples/benchmark_detectors.py            # 64 ranks, 10,000 steps: about 6 minutes, 3 GB of disk
    python examples/benchmark_detectors.py --quick    # 4 ranks, 600 steps: about half a minute

The plan (Step 14) ports the detectors to Rust only if the Python worker needs more than
250 ms per 50-step flush across 64 ranks, or one CPU core in steady state. This script
measures that. It needs no GPU and no server.

1. **A real capture gives the record shapes.** A small MoE shaped like the Granite
   benchmark (24 blocks, 32 experts, top-8, 219 parameters) trains for 105 steps under
   `tensorquill.init(..., capture="full")`, with the weight check, clipped gradient norm,
   logged loss and replica spread that a distributed run records. The records from the
   second weight check on, which carry `weights_unchanged`, are the templates.
2. **A synthetic store in the server's layout.** `runs/<run_id>/rank=<r>/attempt=<a>/
   part-<min:08>-<max:08>.parquet`, written with `tensorquill.flight.records_to_batch`,
   Zstd level 1 like the server, one 50-step part per rank and flush. Values are the
   template's with +-10% jitter, so the run is healthy and no detector fires. A few
   distinct streams are written and shared by the ranks (the `rank` column is swapped), which
   costs the detectors nothing but makes the store quick to build.
3. **Timings, median and worst of several runs**, warm file cache, one process:
   - (a) one rank over the whole run: `load_records` and `build_evidence`; and the 50-step
     increment a flush adds: reading one part, and the detectors over 50 steps.
   - (b) one `--watch` update as the run grows, in steady state: a `_Tail` that has read
     everything but the newest flush, then the 64 new parts land and it polls, rebuilds the
     evidence and replaces the file. Poll, build and write are timed apart. A straight line
     through the sizes gives each stage's constant cost and cost per step.
   - (c) the same update for every one of the 64 ranks in turn, as 64 `--rank` workers would
     do it, summed.
   - (d) peak memory of each, in a fresh process each, and the profile of the slowest stage.
"""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import gc
import io
import json
import math
import multiprocessing
import os
import platform
import pstats
import random
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BUDGET_MS = 250.0  # the plan's budget per 50-step flush
FLUSH_STEPS = 50
RUN_ID = "bench"
# Granite 3.0 1B-A400M, which the repository's GPU benchmark uses: 24 blocks, 32 experts, top-8.
# README: 64.886 ms per step with capture off on the RTX PRO 6000.
STEP_SECONDS = 0.065
VARIANTS = 8  # distinct streams, shared by the ranks


# --- 1. A real capture ----------------------------------------------------------------


def _build_model(blocks: int, experts: int, top_k: int):
    """A small MoE with Granite's layout: blocks of attention, a router, two fused expert
    weights and two norms, named like the Hugging Face model."""
    import torch
    from torch import nn

    dim, ff, vocab = 16, 16, 64

    class Norm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight

    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(dim, dim, bias=False) for _ in range(4))

        def forward(self, x):
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            return self.o_proj(torch.softmax(q @ k.transpose(-1, -2) / dim**0.5, -1) @ v)

    class RouterModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.Linear(dim, experts, bias=False)

        def forward(self, x):
            return self.layer(x)

    class Fused(nn.Module):
        def __init__(self, inner: int, outer: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(experts, inner, outer) * 0.05)

    class MoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = RouterModule()
            self.input_linear, self.output_linear = Fused(dim, ff), Fused(ff, dim)

        def forward(self, x):
            shape = x.shape
            x = x.reshape(-1, dim)
            logits = self.router(x)
            top = logits.topk(top_k, -1)
            gates = torch.zeros_like(logits).scatter(-1, top.indices, torch.softmax(top.values, -1))
            hidden = torch.relu(torch.einsum("td,edf->tef", x, self.input_linear.weight))
            out = torch.einsum("tef,efd->ted", hidden, self.output_linear.weight)
            return (gates[..., None] * out).sum(1).reshape(shape)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn, self.block_sparse_moe = Attention(), MoE()
            self.input_layernorm, self.post_attention_layernorm = Norm(), Norm()

        def forward(self, x):
            x = x + self.self_attn(self.input_layernorm(x))
            return x + self.block_sparse_moe(self.post_attention_layernorm(x))

    class Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, dim)
            self.layers = nn.ModuleList(Block() for _ in range(blocks))
            self.norm = Norm()

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(dim, vocab, bias=False)

        def forward(self, ids):
            x = self.model.embed_tokens(ids)
            for block in self.model.layers:
                x = block(x)
            return self.lm_head(self.model.norm(x))

    return Model(), vocab


def capture_template(blocks: int, experts: int, top_k: int) -> dict[str, Any]:
    """Train the model for two weight-check intervals under a real `tensorquill.init` and
    return a plain and a weight-check record, with the cadence the capture used."""
    import torch
    from torch import nn

    import tensorquill
    from tensorquill.capture import Router
    from tensorquill.sink import MemorySink

    torch.manual_seed(0)
    model, vocab = _build_model(blocks, experts, top_k)
    sink = MemorySink()
    routers = {
        f"model.layers.{i}.block_sparse_moe.router": Router(experts, top_k=top_k) for i in range(blocks)
    }
    session = tensorquill.init(
        model, run_id=RUN_ID, every=FLUSH_STEPS, sink=sink, capture="full", routers=routers,
        heartbeat_seconds=None,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    steps = 2 * FLUSH_STEPS + 5
    for step in range(steps):
        ids = torch.randint(0, vocab, (8, 32))
        loss = nn.functional.cross_entropy(model(ids).reshape(-1, vocab), ids.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        session.log(loss=loss.detach())
        if (step + 1) % FLUSH_STEPS == 0:  # what a DDP run records on its replica-check steps
            session.record_replica_spread(torch.zeros(len(session.replica_names), 3, dtype=torch.float64))
        optimizer.step()
    session.close()
    records = sink.records
    checks = [i for i, r in enumerate(records) if "weights_unchanged" in r]
    if not checks:
        raise RuntimeError("the capture recorded no second weight check")
    first, every = checks[0], FLUSH_STEPS
    return {
        "plain": records[first - 1],
        "check": records[first],
        "check_every": every,
        "check_phase": records[first]["step"] % every,
        "attempt": records[first]["attempt"],
        "shape": {"blocks": blocks, "experts": experts, "top_k": top_k},
        "parameters": len(records[first]["grad_norms"]),
        "layers": len(records[first]["layers"]),
        "json_bytes": {
            "plain": len(json.dumps(records[first - 1], separators=(",", ":"))),
            "check": len(json.dumps(records[first], separators=(",", ":"))),
        },
    }


# --- 2. A synthetic store -------------------------------------------------------------


def synth_record(template: dict[str, Any], variant: int, rank: int, step: int) -> dict[str, Any]:
    """The record for `step` of stream `variant`: the template's values, +-10% apart, so the
    run is healthy. Deterministic in (variant, step)."""
    from tensorquill.capture import _router_stats

    rnd = random.Random(variant * 1_000_003 + step)
    u = rnd.random
    check = step % template["check_every"] == template["check_phase"]
    base = template["check"] if check else template["plain"]

    def jit(x: float, spread: float = 0.1) -> float:
        return x * (1.0 + spread * (2.0 * u() - 1.0))

    grad_norms = {name: jit(v) for name, v in base["grad_norms"].items()}
    total = math.sqrt(sum(v * v for v in grad_norms.values()))
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "rank": rank,
        "attempt": template["attempt"],
        "step": step,
        "ts": base["ts"] + (step - base["step"]) * STEP_SECONDS,
        "layers": {
            name: {stat: (jit(v) if v else v) for stat, v in stats.items()}
            for name, stats in base["layers"].items()
        },
        "grad_norm": total,
        "clip_grad_norm": total,
        "grad_norms": grad_norms,
        "lr": [base["lr"][0] * (0.55 + 0.45 * math.cos(math.pi * min(step, 10_000) / 10_000))],
        "scalars": {"loss": 2.8 * math.exp(-step / 5000.0) + 0.05 * u()},
    }
    moe: dict[str, list[int]] = {}
    for name, counts in base["moe"].items():
        total_tokens = sum(counts)
        raw = [c * (0.8 + 0.4 * u()) for c in counts]
        scale = total_tokens / sum(raw)
        new = [max(1, int(r * scale)) for r in raw]
        new[new.index(max(new))] += total_tokens - sum(new)  # the same assignments in all
        moe[name] = new
    record["moe"] = moe
    record["moe_top_k"] = dict(base["moe_top_k"])
    record["moe_stats"] = {name: _router_stats(counts) for name, counts in moe.items()}
    if check:
        grown = 1.0 + 1e-4 * step / template["check_every"]
        record["weight_norms"] = {name: v * grown * jit(1.0, 1e-5) for name, v in base["weight_norms"].items()}
        for key in ("weights_unchanged", "optimizer_ownership", "replica_spread"):
            if key in base:
                record[key] = json.loads(json.dumps(base[key]))
    return record


def part_path(store: Path, rank: int, attempt: int, low: int, high: int) -> Path:
    return (
        store / "runs" / RUN_ID / f"rank={rank}" / f"attempt={attempt}" / f"part-{low:08}-{high:08}.parquet"
    )


_worker_template: dict[str, Any] | None = None


def _write_task(args: tuple[str, int, int, int, list[int]]) -> int:
    """Generate one stream's 50-step part and write it for every rank that shares the stream."""
    global _worker_template
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tensorquill.flight import records_to_batch

    store_name, variant, low, high, ranks = args
    store = Path(store_name)
    if _worker_template is None:
        _worker_template = json.loads((store / "template.json").read_text())
    template = _worker_template
    records = [synth_record(template, variant, ranks[0], step) for step in range(low, high + 1)]
    table = pa.Table.from_batches([records_to_batch(records)])
    column = table.schema.get_field_index("rank")
    written = 0
    for rank in ranks:
        shared = table.set_column(
            column, table.schema.field("rank"), pa.array([rank] * table.num_rows, pa.int32())
        )
        path = part_path(store, rank, template["attempt"], low, high)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(shared, path, compression="zstd", compression_level=1)
        written += path.stat().st_size
    return written


def write_store(
    store: Path, template: dict[str, Any], ranks: int, steps: int, workers: int
) -> dict[str, Any]:
    """Write `ranks` x `steps` records as 50-step Parquet parts and return the manifest."""
    store.mkdir(parents=True, exist_ok=True)
    (store / "template.json").write_text(json.dumps(template))
    variants = min(VARIANTS, ranks)
    tasks = [
        (str(store), variant, low, low + FLUSH_STEPS - 1, [r for r in range(ranks) if r % variants == variant])
        for low in range(0, steps, FLUSH_STEPS)
        for variant in range(variants)
    ]
    started = time.perf_counter()
    if workers <= 1:
        sizes = [_write_task(task) for task in tasks]
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            sizes = pool.map(_write_task, tasks, chunksize=4)
    manifest = {
        "ranks": ranks,
        "steps": steps,
        "shape": template["shape"],
        "parts": ranks * (steps // FLUSH_STEPS),
        "bytes": sum(sizes),
        "write_seconds": round(time.perf_counter() - started, 1),
    }
    (store / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def prepare_store(
    store: Path, shape: dict[str, int], ranks: int, steps: int, workers: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse `store` if it holds this shape and at least these ranks and steps, and build it if it
    is missing or empty. Anything else is refused, never deleted: `--store` is the caller's
    directory, and may hold more than an earlier store."""
    manifest_path = store / "manifest.json"
    if manifest_path.exists() and (store / "template.json").exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            fits = manifest["shape"] == shape and manifest["ranks"] >= ranks and manifest["steps"] >= steps
        except (ValueError, KeyError, TypeError):
            manifest, fits = None, False
        if fits:
            return json.loads((store / "template.json").read_text()), manifest
        held = (f"a store of shape {manifest.get('shape')}, {manifest.get('ranks')} ranks x {manifest.get('steps')} steps"
                if isinstance(manifest, dict) else "an unreadable manifest.json")
        raise SystemExit(f"--store {store} holds {held}; this run needs shape {shape}, {ranks} ranks x {steps} steps. "
                         "Nothing was deleted: give a new or empty --store, or remove that one yourself.")
    if store.exists() and (not store.is_dir() or any(store.iterdir())):
        raise SystemExit(f"--store {store} is not empty and holds no benchmark store. Nothing was deleted: "
                         "give a new or empty --store.")
    template = capture_template(**shape)
    return template, write_store(store, template, ranks, steps, workers)


def make_view(store: Path, ranks: Sequence[int], steps: int, dest: Path, attempt: int) -> Path:
    """`runs/<run_id>` under `dest` holding hard links to the parts of `ranks` below `steps`,
    which is how the store looked when the run had that many steps."""
    run_dir = dest / "runs" / RUN_ID
    for rank in ranks:
        directory = run_dir / f"rank={rank}" / f"attempt={attempt}"
        directory.mkdir(parents=True, exist_ok=True)
        for low in range(0, steps, FLUSH_STEPS):
            source = part_path(store, rank, attempt, low, low + FLUSH_STEPS - 1)
            target = directory / source.name
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
    return run_dir


# --- 3. Timing ------------------------------------------------------------------------


def summarize(values: Sequence[float]) -> dict[str, float]:
    return {"median": statistics.median(values), "worst": max(values), "best": min(values), "runs": len(values)}


def timed(fn: Callable[[], Any], repeats: int) -> tuple[dict[str, dict[str, float]], Any]:
    """Wall and CPU seconds of `repeats` calls; the last result."""
    wall: list[float] = []
    cpu: list[float] = []
    result = None
    for _ in range(repeats):
        result = None
        gc.collect()
        started, cpu_started = time.perf_counter(), time.process_time()
        result = fn()
        wall.append(time.perf_counter() - started)
        cpu.append(time.process_time() - cpu_started)
    return {"wall": summarize(wall), "cpu": summarize(cpu)}, result


def ms(seconds: float) -> str:
    return f"{seconds * 1000:,.1f}"


def steady_tail(run_dir: Path, rank: int, flush_low: int):
    """A `_Tail` as a long-running `--watch` has it just before the flush that starts at step
    `flush_low`: every part read, the rank's records in memory, the flush's parts new.

    The state is set directly, so that the update timed from here is the steady-state one
    and not a cold start from step 0 (which also reads the whole run)."""
    from tensorquill import evidence

    tail = evidence._Tail(run_dir, rank)
    parts = evidence._parquet_parts(run_dir)
    new = {p for p in parts if p.name.startswith(f"part-{flush_low:08}-")}
    tail._seen = set(parts) - new
    mine = evidence._parquet_parts(run_dir / f"rank={rank}")
    tail.records = [
        r for p in mine if int(evidence._PART_NAME.search(p.name).group(1)) < flush_low for r in evidence._read_part(p)
    ]
    return tail, len(new)


def watch_update(run_dir: Path, rank: int, flush_low: int, output: Path) -> dict[str, Any]:
    """One `--watch` update, as `evidence.watch` runs it, timed in three parts."""
    from tensorquill import evidence

    tail, new_parts = steady_tail(run_dir, rank, flush_low)
    gc.collect()
    t0, c0 = time.perf_counter(), time.process_time()
    grown = tail.poll()
    t1, c1 = time.perf_counter(), time.process_time()
    # The watcher names the store's status file, there or not: a server's store is expected to have one.
    built = evidence.build_evidence(tail.records, watchdog=evidence._watchdog_beside(run_dir))
    t2, c2 = time.perf_counter(), time.process_time()
    written = evidence.write_evidence(built, output)
    t3, c3 = time.perf_counter(), time.process_time()
    assert grown and len(tail.records) == flush_low + FLUSH_STEPS, (grown, len(tail.records))
    return {
        "poll": t1 - t0, "build": t2 - t1, "write": t3 - t2, "total": t3 - t0,
        "cpu": c3 - c0, "bytes": written.stat().st_size, "steps": len(tail.records),
        "incidents": [i["kind"] for i in built["incidents"]], "new_parts": new_parts,
    }


def child_memory(mode: str, store: Path, args: argparse.Namespace, size: int, ranks: int, attempt: int) -> dict[str, Any]:
    """Run one scenario in a fresh process and return its peak resident memory above the
    interpreter's after imports."""
    command = [
        sys.executable, str(Path(__file__).resolve()), "--child", mode, "--store", str(store),
        "--child-size", str(size), "--child-ranks", str(ranks), "--child-attempt", str(attempt),
    ]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        return {"error": done.stderr.strip().splitlines()[-1] if done.stderr.strip() else "failed"}
    return json.loads(done.stdout.strip().splitlines()[-1])


def _rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1 << 20) if sys.platform == "darwin" else peak / 1024  # bytes on macOS, kB on Linux


def run_child(args: argparse.Namespace) -> None:
    """The measured part of one memory scenario. Prints one JSON line."""
    from tensorquill import evidence
    from tensorquill.flight import row_to_record  # noqa: F401  (imported before the baseline)

    import pyarrow.parquet  # noqa: F401

    store, size, ranks = Path(args.store), args.child_size, args.child_ranks
    baseline = _rss_mb()
    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)
        if args.child == "rank_build":  # one rank's whole run, from its own directory
            run_dir = make_view(store, [0], size, scratch_path, args.child_attempt)
            records = evidence.load_records(run_dir / "rank=0", all_attempts=True)
            evidence.build_evidence(records)
            count = len(records)
        elif args.child == "watch_update":  # a steady-state update with the run at `size` steps
            run_dir = make_view(store, range(ranks), size, scratch_path, args.child_attempt)
            result = watch_update(run_dir, 0, size - FLUSH_STEPS, scratch_path / "evidence.json")
            count = result["steps"]
        elif args.child == "run_dir_load":  # the CLI's path: one rank, from the directory of all
            run_dir = make_view(store, range(ranks), size, scratch_path, args.child_attempt)
            rows_read = [0]
            read_part = evidence._read_part

            def counting(path: Path) -> list[dict[str, Any]]:
                rows = read_part(path)
                rows_read[0] += len(rows)
                return rows

            evidence._read_part = counting
            started = time.perf_counter()
            kept = len(evidence.load_records(run_dir, rank=0, all_attempts=True))
            seconds = time.perf_counter() - started
            count = rows_read[0]  # rows read, of which one rank's are kept
            assert kept == size, kept
        else:
            raise SystemExit(f"unknown scenario {args.child}")
    out = {"peak_mb": round(_rss_mb() - baseline, 1), "baseline_mb": round(baseline, 1), "records": count}
    if args.child == "run_dir_load":
        out["seconds"] = round(seconds, 3)
    print(json.dumps(out))


# --- 4. Measurements ------------------------------------------------------------------


def measure_one_rank(store: Path, sizes: list[int], repeats: int, attempt: int, scratch: Path) -> dict[str, Any]:
    """(a): `load_records` and `build_evidence` for rank 0 over the whole run, and the
    increment one flush adds."""
    from tensorquill import evidence

    out: dict[str, Any] = {"sizes": {}}
    for size in sizes:
        view = make_view(store, [0], size, scratch / f"a{size}", attempt)
        rank_dir = view / "rank=0"
        load_stats, records = timed(lambda: evidence.load_records(rank_dir, all_attempts=True), repeats)
        build_stats, built = timed(lambda: evidence.build_evidence(records), repeats)
        out["sizes"][size] = {
            "load": load_stats, "build": build_stats, "records": len(records),
            "incidents": [i["kind"] for i in built["incidents"]],
        }
    big = records  # the largest run
    parts = sorted((scratch / f"a{sizes[-1]}" / "runs" / RUN_ID / "rank=0").rglob("part-*.parquet"))
    pick = parts[len(parts) // 2]
    out["read_part"] = timed(lambda: evidence._read_part(pick), repeats)[0]
    slice_ = big[len(big) // 2 : len(big) // 2 + FLUSH_STEPS]
    out["build_50"] = timed(lambda: evidence.build_evidence(slice_), repeats)[0]
    out["per_record_build_us"] = out["sizes"][sizes[-1]]["build"]["wall"]["median"] / len(big) * 1e6
    return out


def measure_updates(
    store: Path, ranks: int, sizes: list[int], runs: int, attempt: int, scratch: Path
) -> dict[str, Any]:
    """(b): the steady-state `--watch` update that takes the run to each size, `runs` times,
    from a fresh tail each time."""
    from tensorquill import evidence

    out: dict[str, Any] = {}
    for size in sizes:
        view = make_view(store, range(ranks), size, scratch / f"b{size}", attempt)
        started = time.perf_counter()
        listed = len(evidence._parquet_parts(view, 0))
        listing = time.perf_counter() - started
        rows = [
            watch_update(view, 0, size - FLUSH_STEPS, scratch / f"b{size}.evidence.json") for _ in range(runs)
        ]
        out[size] = {
            "rows": rows,
            "listing": listing,
            "listed": listed,
            "files": ranks * (size // FLUSH_STEPS),
            **{key: summarize([row[key] for row in rows]) for key in ("poll", "build", "write", "total", "cpu")},
            "bytes": rows[-1]["bytes"],
            "incidents": rows[-1]["incidents"],
        }
    return out


def measure_all_ranks(
    store: Path, ranks: int, size: int, attempt: int, scratch: Path
) -> dict[str, Any]:
    """(c): the update at `size` steps for every rank in turn, as `ranks` `--rank` workers."""
    view = make_view(store, range(ranks), size, scratch / "c", attempt)
    rows = [
        watch_update(view, rank, size - FLUSH_STEPS, scratch / "c.evidence.json") for rank in range(ranks)
    ]
    return {
        "size": size,
        "ranks": ranks,
        "total": sum(row["total"] for row in rows),
        "cpu": sum(row["cpu"] for row in rows),
        "per_rank": summarize([row["total"] for row in rows]),
        "poll": sum(row["poll"] for row in rows),
        "build": sum(row["build"] for row in rows),
        "write": sum(row["write"] for row in rows),
    }


def profile_stage(
    store: Path, ranks: int, size: int, update: dict[str, Any], attempt: int, scratch: Path, top: int
) -> dict[str, Any]:
    """cProfile of whichever stage of the update at `size` steps is slowest, in a pass of
    its own so the profiler does not touch the timings above."""
    from tensorquill import evidence

    stage = max(("poll", "build", "write"), key=lambda s: update["rows"][-1][s])
    view = make_view(store, range(ranks), size, scratch / "p", attempt)
    tail, _ = steady_tail(view, 0, size - FLUSH_STEPS)
    profile = cProfile.Profile()
    if stage == "poll":
        profile.runcall(tail.poll)
    else:
        tail.poll()
        if stage == "build":
            profile.runcall(evidence.build_evidence, tail.records)
        else:
            built = evidence.build_evidence(tail.records)
            profile.runcall(evidence.write_evidence, built, scratch / "p.evidence.json")
    stats = pstats.Stats(profile)
    total = stats.total_tt
    root = {"poll": "poll", "build": "build_evidence", "write": "write_evidence"}[stage]
    rows = []
    for (path, line, name), (cc, nc, tt, ct, callers) in stats.stats.items():
        rows.append({"function": f"{Path(path).name}:{line}({name})", "calls": nc, "own": tt, "cumulative": ct})
    rows.sort(key=lambda r: -r["own"])
    # The direct callees of the stage's entry point, so detectors that call each other do not count twice.
    callees = []
    for (path, line, name), (cc, nc, tt, ct, callers) in stats.stats.items():
        for (cpath, cline, cname), value in callers.items():
            if cname == root:
                callees.append({"function": f"{Path(path).name}:{line}({name})", "cumulative": value[3], "calls": value[1]})
    callees.sort(key=lambda r: -r["cumulative"])
    return {"stage": stage, "steps": size, "profiled_seconds": total, "own": rows[:top], "callees": callees[:top]}


# --- 5. Report ------------------------------------------------------------------------


def machine() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version(), "cpus": os.cpu_count()}
    try:
        if sys.platform == "darwin":
            brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            memory = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
        else:
            brand = platform.processor() or platform.machine()
            memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        info["cpu"], info["memory_gb"] = brand, round(memory / (1 << 30))
    except (OSError, ValueError):
        pass
    import pyarrow
    import torch

    info.update(torch=torch.__version__, pyarrow=pyarrow.__version__, gpu="none used (the detectors run on the CPU)")
    with contextlib.suppress(OSError, AttributeError):
        info["load_average_at_start"] = [round(x, 2) for x in os.getloadavg()]
    return info


def budget_summary(report: dict[str, Any], ranks: int) -> dict[str, Any]:
    """Each stage's cost as `constant + per_step * steps` (a least-squares line through the
    medians of (b)), and what a worker that kept the detectors' state between flushes would
    still cost: reading one part and running the detectors over its 50 steps, for every rank.
    The second is a projection from the measured slope, not a measurement."""
    updates = report["updates"]
    sizes = sorted(updates)
    fit: dict[str, dict[str, float]] = {}
    if len(sizes) >= 2:
        for stage in ("poll", "build", "write", "total"):
            slope, constant = statistics.linear_regression(sizes, [updates[n][stage]["median"] for n in sizes])
            fit[stage] = {"constant": constant, "per_step": slope}
    out: dict[str, Any] = {"fit": fit}
    if "build" in fit:
        read = report["one_rank"]["read_part"]
        per_rank = read["wall"]["median"] + fit["build"]["per_step"] * FLUSH_STEPS
        per_rank_cpu = read["cpu"]["median"] + fit["build"]["per_step"] * FLUSH_STEPS
        flush_seconds = report["step_seconds"] * FLUSH_STEPS
        out["incremental_floor"] = {
            "per_rank": per_rank, "ranks": ranks, "total": per_rank * ranks,
            "core_fraction": per_rank_cpu * ranks / flush_seconds,
            "detectors_per_rank": fit["build"]["per_step"] * FLUSH_STEPS, "read_per_rank": read["wall"]["median"],
        }
    return out


def verdict(milliseconds: float) -> str:
    ratio = milliseconds / BUDGET_MS
    return f"{'within' if ratio <= 1 else 'OVER'} budget ({ratio:.1f}x)"


def print_report(report: dict[str, Any]) -> None:
    out = print
    machine_info = report["machine"]
    out("\n== Machine ==")
    for key, value in machine_info.items():
        out(f"  {key}: {value}")
    busy = max((machine_info.get("load_average_at_start") or [0])[:1] + (machine_info.get("load_average_at_end") or [0])[:1])
    if busy > (machine_info.get("cpus") or 1):
        out(f"  WARNING: the 1-minute load average ({busy}) is above the {machine_info['cpus']} cores: "
            "something else was running, and the timings include its interference")
    store = report["store"]
    out("\n== Store ==")
    out(
        f"  {store['ranks']} ranks x {store['steps']} steps, shape {store['shape']}, {store['parts']} parts of "
        f"{FLUSH_STEPS} steps, {store['bytes'] / 1e6:,.0f} MB"
        f" ({store['bytes'] / (store['ranks'] * store['steps']) / 1e3:.1f} KB per record on disk, "
        f"{report['template']['json_bytes']['plain'] / 1e3:.1f} KB as JSON)"
    )
    out(f"  template: {report['template']['parameters']} parameters, {report['template']['layers']} block layers")

    one = report["one_rank"]
    out("\n== (a) One rank, whole run (median / worst of several runs, ms) ==")
    out(f"  {'steps':>7} {'load_records':>22} {'build_evidence':>24}   incidents")
    for size, row in one["sizes"].items():
        load, build = row["load"]["wall"], row["build"]["wall"]
        out(
            f"  {size:>7} {ms(load['median']):>10} / {ms(load['worst']):>9} {ms(build['median']):>11} / {ms(build['worst']):>10}"
            f"   {row['incidents'] or 'none'}"
        )
    part, slice_ = one["read_part"]["wall"], one["build_50"]["wall"]
    out(f"  one 50-step increment: read its part {ms(part['median'])} / {ms(part['worst'])},"
        f" build_evidence over 50 steps {ms(slice_['median'])} / {ms(slice_['worst'])}"
        f" (whole-run build: {one['per_record_build_us']:.0f} us per record)")

    out(f"\n== (b) The --watch update that takes the run to N steps (a flush of {FLUSH_STEPS} steps from {store['ranks']} ranks; ms, median / worst) ==")
    out(f"  {'steps':>7} {'poll':>16} {'build':>18} {'write':>16} {'total':>18} {'cpu':>8}  json   vs {BUDGET_MS:g} ms")
    for size, row in report["updates"].items():
        cells = " ".join(
            f"{ms(row[k]['median']):>8} / {ms(row[k]['worst']):>8}" for k in ("poll", "build", "write", "total")
        )
        out(
            f"  {size:>7} {cells} {ms(row['cpu']['median']):>8} {row['bytes'] / 1e6:>5.1f}MB  {verdict(row['total']['median'] * 1000)}"
        )
        if row["incidents"]:
            out(f"          incidents in the run: {row['incidents']}")
    last = report["updates"][max(report["updates"])]
    out(f"  (a poll lists the {last['listed']} part files of rank 0, of {last['files']} in the directory: "
        f"{ms(last['listing'])} ms of it)")

    budget = report.get("budget", {})
    if budget.get("fit"):
        out("  straight line through the medians, ms: " + "; ".join(
            f"{stage} = {line['constant'] * 1000:,.0f} + {line['per_step'] * 1e6:,.0f} us x steps"
            for stage, line in budget["fit"].items()
        ))
    floor = budget.get("incremental_floor")
    if floor:
        flush_seconds = report["step_seconds"] * FLUSH_STEPS
        out(
            f"  floor if the detectors kept their state between flushes: per rank {ms(floor['per_rank'])} ms"
            f" (read the part {ms(floor['read_per_rank'])} + detectors {ms(floor['detectors_per_rank'])}),"
            f" x {floor['ranks']} ranks = {ms(floor['total'])} ms => {verdict(floor['total'] * 1000)};"
            f" {floor['core_fraction'] * 100:,.0f}% of one core at {report['step_seconds'] * 1000:.0f} ms/step"
            f" ({flush_seconds:.2f} s between flushes)  [projected from the slope above]"
        )

    allr = report.get("all_ranks")
    if allr:
        out(f"\n== (c) The same update for each of {allr['ranks']} ranks in turn, at {allr['size']} steps ==")
        out(
            f"  total {ms(allr['total'])} ms (poll {ms(allr['poll'])}, build {ms(allr['build'])}, write {ms(allr['write'])};"
            f" cpu {ms(allr['cpu'])}); per rank median {ms(allr['per_rank']['median'])} / worst {ms(allr['per_rank']['worst'])}"
            f"  => {verdict(allr['total'] * 1000)}"
        )
        flush_seconds = report["step_seconds"] * FLUSH_STEPS
        out(f"  one core: {allr['cpu']:.1f} cpu-s per flush every {flush_seconds:.2f} s of training"
            f" ({report['step_seconds'] * 1000:.0f} ms/step) = {allr['cpu'] / flush_seconds * 100:,.0f}% of a core")

    out("\n== (d) Peak memory above the interpreter, fresh process each (MB) ==")
    for name, row in report["memory"].items():
        if "error" in row:
            out(f"  {name}: {row['error']}")
        else:
            took = f", {ms(row['seconds'])} ms" if "seconds" in row else ""
            each = f" ({row['peak_mb'] * 1024 / row['records']:.0f} KB each)" if row["records"] >= 500 else ""
            out(f"  {name}: {row['peak_mb']:,.0f} MB for {row['records']:,} records read{each}{took}")
    for line in report.get("memory_notes", []):
        out(f"  {line}")

    prof = report["profile"]
    out(f"\n== Profile of the slowest stage: {prof['stage']} at {prof['steps']} steps ({prof['profiled_seconds']:.2f} s under cProfile) ==")
    out("  direct callees, cumulative:")
    for row in prof["callees"]:
        out(f"    {row['cumulative']:8.3f} s  {row['calls']:>9,} calls  {row['function']}")
    out("  own time:")
    for row in prof["own"]:
        out(f"    {row['own']:8.3f} s  {row['calls']:>9,} calls  {row['function']}")


# --- main -----------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ranks", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10_000, help="steps per rank in the store")
    parser.add_argument("--sizes", default="500,2000,10000", help="run lengths for (a) and (b)")
    parser.add_argument("--repeats", type=int, default=5, help="runs per timing in (a)")
    parser.add_argument("--flushes", type=int, default=5, help="runs of the update timed per size in (b)")
    parser.add_argument("--all-ranks-steps", type=int, default=2000, help="run length for (c); 0 skips it")
    parser.add_argument("--profile-steps", type=int, default=2000)
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--store", type=Path, help="keep the store here and reuse it (default: a temporary directory)")
    parser.add_argument("--json", type=Path, help="write the results here")
    parser.add_argument("--quick", action="store_true", help="4 ranks, 600 steps, shorter runs: a smoke test")
    # The measured part of a memory scenario, run in a fresh process by this script itself.
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--child-size", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--child-ranks", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--child-attempt", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.quick:
        args.ranks, args.steps, args.sizes = 4, 600, "100,300,600"
        args.repeats, args.flushes, args.all_ranks_steps, args.profile_steps = 3, 3, 300, 300
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    sizes = sorted({int(s) for s in args.sizes.split(",")})
    steps = max(args.steps, sizes[-1], args.all_ranks_steps, args.profile_steps)
    shape = {"blocks": args.blocks, "experts": args.experts, "top_k": args.top_k}
    cleanup = args.store is None
    store = args.store or Path(tempfile.mkdtemp(prefix="tensorquill-bench-"))
    store.parent.mkdir(parents=True, exist_ok=True)
    # Next to the store, so that the views of it can be hard links.
    scratch = Path(tempfile.mkdtemp(prefix="tensorquill-bench-view-", dir=store.parent))
    try:
        print(f"building the store in {store} ...", flush=True)
        template, manifest = prepare_store(store, shape, args.ranks, steps, args.workers)
        print(f"  {manifest['parts']} parts, {manifest['bytes'] / 1e6:,.0f} MB, {manifest['write_seconds']} s", flush=True)
        attempt = template["attempt"]
        report: dict[str, Any] = {
            "machine": machine(), "store": manifest, "template": template, "step_seconds": STEP_SECONDS,
        }

        print("(a) one rank, whole run ...", flush=True)
        report["one_rank"] = measure_one_rank(store, sizes, args.repeats, attempt, scratch)
        print("(b) --watch updates ...", flush=True)
        report["updates"] = measure_updates(store, args.ranks, sizes, args.flushes, attempt, scratch)
        report["budget"] = budget_summary(report, args.ranks)
        if args.all_ranks_steps:
            print(f"(c) all {args.ranks} ranks at {args.all_ranks_steps} steps ...", flush=True)
            report["all_ranks"] = measure_all_ranks(store, args.ranks, args.all_ranks_steps, attempt, scratch)
        print("memory (a fresh process each) ...", flush=True)
        memory: dict[str, Any] = {}
        for size in sizes:
            memory[f"one rank, build over {size} steps"] = child_memory("rank_build", store, args, size, 1, attempt)
        top = sizes[-1]
        memory[f"--watch update at {top} steps"] = child_memory("watch_update", store, args, top, args.ranks, attempt)
        probe_steps = min(100, sizes[0])
        loaded = child_memory("run_dir_load", store, args, probe_steps, args.ranks, attempt)
        memory[f"--rank 0 on the run directory of {args.ranks} ranks x {probe_steps} steps"] = loaded
        notes = []
        if "peak_mb" in loaded and loaded["records"] > probe_steps:
            per_record_kb = loaded["peak_mb"] * 1024 / loaded["records"]
            notes.append(
                f"reading one rank from the directory of all {args.ranks} read {loaded['records']:,} rows to keep {probe_steps}"
                f" and held them all at once: about {per_record_kb * args.ranks * top / 1e6:,.1f} GB at {top} steps"
                f" ({per_record_kb * args.ranks * sizes[len(sizes) // 2] / 1e6:,.1f} GB at {sizes[len(sizes) // 2]})"
            )
        elif "peak_mb" in loaded:
            notes.append(
                f"reading one rank from the directory of all {args.ranks} read only that rank's parts"
                f" ({loaded['records']:,} rows of the {args.ranks * probe_steps:,} in the directory)"
            )
        report["memory"], report["memory_notes"] = memory, notes
        print("profile ...", flush=True)
        size = min(args.profile_steps, max(report["updates"]))
        base = report["updates"][size] if size in report["updates"] else report["updates"][max(report["updates"])]
        report["profile"] = profile_stage(store, args.ranks, size, base, attempt, scratch, top=14)
        with contextlib.suppress(OSError, AttributeError):
            report["machine"]["load_average_at_end"] = [round(x, 2) for x in os.getloadavg()]
        return report
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if cleanup:
            shutil.rmtree(store, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.child:
        run_child(args)
        return
    report = run(args)
    print_report(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=str) + "\n")
        print(f"\nresults -> {args.json}")


if __name__ == "__main__":
    main()
