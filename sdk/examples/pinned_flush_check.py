"""The pinned-memory flush (`init(pinned_flush=True)`) against the blocking one, on a GPU.

    python examples/pinned_flush_check.py --out runs/pinned-flush
    python examples/pinned_flush_check.py --out runs/pinned-flush --dim 2048 --blocks 12 --batch 8192
    python examples/pinned_flush_check.py --out runs/pinned-dry --dry-run      # no GPU: the script only

What runs: `--rounds` (3) rounds, each of two arms in fresh processes, their order alternating by round:
`blocking` (`pinned_flush=False`, today's `.cpu()` at a flush) and `pinned` (`pinned_flush=True`). Both
train the same seeded model, a small MoE like `gpu_check.py`'s (a one-hot mix in place of `gather`, so the
backward has no atomic adds), on the same seeded batches, with SGD, under
`tensorquill.init(model, routers=..., capture="full", every=--every)`, deterministic algorithms on and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Nothing in the loop reads a value from the GPU.

What is measured, after `--warmup` steps:

- **the records**: each arm's records file, without what differs between any two runs (`ts`, `attempt`,
  `run_id`, `step_time`, `data_wait`, `system`), compared between the two arms of a round, and between
  rounds of the blocking arm (if those differ the GPU run is not deterministic and the comparison of paths
  says nothing: `inconclusive`);
- **the step time on the host**: `perf_counter` from before the forward to after `optimizer.step()`,
  with no device synchronisation in the step, so a step's time includes any wait of the training thread
  for the GPU. Split into flush steps (the step whose end fills the ring: `(step + 1) % every == 0`) and
  the others: median, minimum and maximum per arm and round, and over all rounds;
- **synchronising calls**: `torch.cuda.set_sync_debug_mode("warn")` over the measured steps and
  `close()`, every warning kept with the thread that raised it and the step it came in (a flush step or
  not). The pinned arm must have none on the training thread.

Pass, exit 0: in every round the two arms' records are the same, the pinned arm used the pinned path to its
end (`session.pinned_flush` still true at close) and made no synchronising call on the training thread at a
flush step, the step after it or `close()`. Calls at other steps are counted for both arms and reported
apart (they are not the flush's).
The step times are reported, not judged. Exit 1: a check failed, an arm failed, or `inconclusive`. Exit 2:
bad arguments, or no CUDA without `--dry-run`.

`--dry-run` runs the same on the CPU, small, one round: the pinned arm then falls back to the blocking
copy (TensorQuill says so once) and no synchronising call can be counted. It shows that the script works,
nothing about the pinned path; the verdict's `state` is `dry-run`.

Written in lane AQ-1 and not run on a GPU there.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tensorquill  # noqa: E402

N_EXPERTS = 8
ARMS = ("blocking", "pinned")
VOLATILE = {"ts", "attempt", "run_id", "step_time", "data_wait", "system"}
SYNC_TEXT = "called a synchronizing CUDA operation"


class Router(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, N_EXPERTS)

    def forward(self, h):
        logits = self.proj(h)
        return logits, logits.argmax(dim=-1)


class Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.router = Router(dim)
        self.experts = nn.ModuleList(nn.Linear(dim, dim) for _ in range(N_EXPERTS))

    def forward(self, h):
        _, choice = self.router(h)
        stacked = torch.stack([expert(h) for expert in self.experts], dim=1)  # [B, E, D]
        # Not `gather` (its backward adds atomically) and not `one_hot` (it reads its input's range): no
        # synchronising call and the same values on every run.
        onehot = (choice[:, None] == torch.arange(N_EXPERTS, device=h.device)).to(h.dtype)
        return h + torch.relu((stacked * onehot[:, :, None]).sum(1))


class MoE(nn.Module):
    def __init__(self, dim: int, blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim) for _ in range(blocks))
        self.out = nn.Linear(dim, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.out(x)


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items() if key not in VOLATILE}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, float) and value != value:
        return "nan"
    return value


def _spread(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median_ms": None, "min_ms": None, "max_ms": None}
    return {"n": len(values), "median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values)}


def run_arm(args: argparse.Namespace) -> dict:
    """One arm in this process: train, time every step, count synchronising calls, write the records."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.dry_run else "cpu")
    torch.manual_seed(args.seed)
    model = MoE(args.dim, args.blocks).to(device)
    batches = torch.Generator(device=device).manual_seed(args.seed + 1)  # made on the device: no copy to it
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    work = Path(args.result).parent
    records = work / f"records-{args.arm}-r{args.round}.jsonl"
    sink = tensorquill.JsonlSink(records)
    step_now = {"step": -1, "measuring": False}
    syncs: list[dict] = []
    said: list[str] = []
    training = threading.get_ident()

    def show(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        if SYNC_TEXT in text:
            if step_now["measuring"]:
                syncs.append({"step": step_now["step"], "thread": "training" if threading.get_ident() == training
                              else threading.current_thread().name})
        elif "TensorQuill" in text:
            said.append(text)

    times: list[float] = []
    debug = device.type == "cuda"
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = show
        session = tensorquill.init(
            model, sink=sink, run_id=f"pinned-check-{args.arm}-r{args.round}", every=args.every, capture="full",
            routers={f"blocks.{i}.router": N_EXPERTS for i in range(args.blocks)},
            pinned_flush=args.arm == "pinned", heartbeat_seconds=None,
        )
        for step in range(args.steps):
            x = torch.randn(args.batch, args.dim, generator=batches, device=device)
            if step == args.warmup:
                if debug:
                    torch.cuda.synchronize()  # the script's, outside every measured step
                    torch.cuda.set_sync_debug_mode("warn")
                step_now["measuring"] = True
            step_now["step"] = step
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            model(x).pow(2).mean().backward()
            opt.step()
            times.append((time.perf_counter() - start) * 1000.0)
        step_now["step"] = "close"
        session.close()
        step_now["measuring"] = False
        if debug:
            torch.cuda.set_sync_debug_mode(0)
    measured = range(args.warmup, args.steps)
    flush_steps = [s for s in measured if (s + 1) % args.every == 0]
    other_steps = [s for s in measured if (s + 1) % args.every != 0]
    by_thread: dict[str, dict[str, int]] = {}
    for sync in syncs:
        counts = by_thread.setdefault(sync["thread"], {"flush": 0, "after_flush": 0, "other": 0, "close": 0})
        counts[_label(sync["step"], args.every)] += 1
    return {
        "arm": args.arm, "round": args.round, "state": "measured", "device": str(device),
        "torch": torch.__version__, "records_file": records.name,
        "pinned_used_to_the_end": bool(session.pinned_flush), "writer_error": repr(session.writer_error),
        "tensorquill_warnings": said,
        "flush_steps": _spread([times[s] for s in flush_steps]), "other_steps": _spread([times[s] for s in other_steps]),
        "step_ms": times, "sync_calls": by_thread,
        "sync_debug_mode": debug,
    }


def _label(step: int | str, every: int) -> str:
    """A flush step (its end fills the ring), the step after it (which ends the held step), the others."""
    if step == "close":
        return "close"
    return {0: "flush", 1: "after_flush"}.get((int(step) + 1) % every, "other") if every > 1 else "flush"


def _records(work: Path, name: str) -> list:
    return [_clean(json.loads(line)) for line in (work / name).read_text().splitlines() if line.strip()]


def summarize(results: dict[str, list[dict]], work: Path, dry_run: bool, warmup: int, every: int) -> dict:
    rounds = sorted({r["round"] for arm in results.values() for r in arm})
    by = {(r["arm"], r["round"]): r for arm in results.values() for r in arm}
    failed = [f"{arm} round {rnd}" for arm in ARMS for rnd in rounds if by.get((arm, rnd), {}).get("state") != "measured"]
    checks: list[dict] = []
    deterministic = True
    if not failed:
        base = _records(work, by[("blocking", rounds[0])]["records_file"])
        for rnd in rounds[1:]:
            deterministic &= _records(work, by[("blocking", rnd)]["records_file"]) == base
        for rnd in rounds:
            blocking = _records(work, by[("blocking", rnd)]["records_file"])
            pinned_arm = by[("pinned", rnd)]
            pinned = _records(work, pinned_arm["records_file"])
            on_training = pinned_arm["sync_calls"].get("training", {})
            training_syncs = sum(on_training.get(key, 0) for key in ("flush", "after_flush", "close"))
            checks.append({
                "round": rnd,
                "records_equal": blocking == pinned and bool(blocking),
                "records": len(blocking),
                "pinned_used_to_the_end": pinned_arm["pinned_used_to_the_end"],
                # At the flush steps, the steps after them and close: the waits this path removes.
                "pinned_training_thread_sync_calls": training_syncs,
                "pinned_training_thread_sync_calls_at_other_steps": on_training.get("other", 0),
            })
    arms = {}
    for arm in ARMS:
        runs = [by[(arm, rnd)] for rnd in rounds if (arm, rnd) in by and by[(arm, rnd)].get("state") == "measured"]
        flush = [t for r in runs for s, t in enumerate(r["step_ms"]) if s >= warmup and (s + 1) % every == 0]
        other = [t for r in runs for s, t in enumerate(r["step_ms"]) if s >= warmup and (s + 1) % every != 0]
        arms[arm] = {
            "flush_steps": _spread(flush), "other_steps": _spread(other),
            "per_round": [{"round": r["round"], "flush_steps": r["flush_steps"], "other_steps": r["other_steps"],
                           "sync_calls": r["sync_calls"]} for r in runs],
        }
    if failed:
        state = "failed"
    elif dry_run:
        state = "dry-run"
    elif not deterministic:
        state = "inconclusive"
    else:
        good = all(c["records_equal"] and c["pinned_used_to_the_end"] and c["pinned_training_thread_sync_calls"] == 0
                   for c in checks)
        state = "pass" if good else "fail"
    return {"state": state, "failed_arms": failed, "deterministic_blocking_rounds": deterministic,
            "checks": checks, "arms": arms, "not_run_on_a_gpu": dry_run}


def main(argv: list[str] | None = None) -> int:
    cuda = torch.cuda.is_available()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="new directory for the arms' results, records and verdict.json")
    parser.add_argument("--dry-run", action="store_true", help="on the CPU, small, one round: the script only")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--every", type=int, help="steps between TensorQuill's flushes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm-timeout", type=float, default=1800)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--round", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_:
        return 2 if exit_.code else 0
    small = args.dry_run
    defaults = {"rounds": 1 if small else 3, "dim": 32 if small else 1024, "blocks": 2 if small else 8,
                "batch": 64 if small else 4096, "steps": 24 if small else 520, "warmup": 4 if small else 20,
                "every": 5 if small else 50}
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.arm is not None:
        try:
            result = run_arm(args)
        except Exception as error:  # noqa: BLE001  the driver reports an arm that failed
            result = {"arm": args.arm, "round": args.round, "state": "failed", "error": repr(error)}
        Path(args.result).write_text(json.dumps(result, indent=1))
        return 0 if result["state"] == "measured" else 1
    if args.out is None:
        print("--out is required", file=sys.stderr)
        return 2
    if not cuda and not args.dry_run:
        print("no CUDA device: this check needs a GPU (or --dry-run to check the script on the CPU)", file=sys.stderr)
        return 2
    if min(args.rounds, args.steps, args.every) < 1 or args.warmup < 0 or args.warmup >= args.steps:
        print("--rounds, --steps and --every must be at least 1, and --warmup below --steps", file=sys.stderr)
        return 2
    if args.out.exists() and any(args.out.iterdir()):
        print(f"{args.out} is not empty", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    passthrough = ["--dim", str(args.dim), "--blocks", str(args.blocks), "--batch", str(args.batch), "--steps",
                   str(args.steps), "--warmup", str(args.warmup), "--every", str(args.every), "--seed", str(args.seed)]
    if args.dry_run:
        passthrough.append("--dry-run")
    for rnd in range(args.rounds):
        for arm in (ARMS if rnd % 2 == 0 else ARMS[::-1]):
            result_file = args.out / f"arm-{arm}-r{rnd}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--arm", arm, "--round", str(rnd),
                       "--result", str(result_file), *passthrough]
            try:
                subprocess.run(command, timeout=args.arm_timeout, check=False)
            except subprocess.TimeoutExpired:
                pass
            try:
                results[arm].append(json.loads(result_file.read_text()))
            except (OSError, ValueError):
                results[arm].append({"arm": arm, "round": rnd, "state": "failed", "error": "no result written"})
    verdict = summarize(results, args.out, args.dry_run, args.warmup, args.every)
    verdict["arguments"] = {key: getattr(args, key) for key in ("dim", "blocks", "batch", "steps", "warmup", "every",
                                                                "rounds", "seed", "dry_run")}
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=1))
    for arm in ARMS:
        a = verdict["arms"][arm]
        print(f"{arm:8s} flush steps {a['flush_steps']}  other steps {a['other_steps']}")
    for check in verdict["checks"]:
        print(f"round {check['round']}: {check}")
    print(f"state: {verdict['state']}")
    return 0 if verdict["state"] in ("pass", "dry-run") else 1


if __name__ == "__main__":
    sys.exit(main())
