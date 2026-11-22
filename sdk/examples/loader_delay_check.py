"""Every step's time in the data loader's `next()` and in the whole step, with capture off and on.

    python examples/loader_delay_check.py --out runs/loader-delay

A measurement, not a check. On the Mac `gpu_system_check.py` was written on, with TensorQuill on and
a 50-step flush, step 100 spent 30 to 60 ms longer in the loader's `next()` than the loader's own
sleep, in every run; with TensorQuill off it did not. The cause is not established (bug list,
section 14). The author's terms, Nov 22, 2026: measure it on the GPU before any fix; keep the
per-step timings for capture off and on, step 100 included; report the worst delay beside the
preregistered median overhead, because "a passing median alone doesn't explain the spike".

What runs: `--rounds` (3) rounds, each of two arms in fresh processes, their order alternating by
round. Both arms run the same loop as `gpu_system_check.py`: the small MoE of `gpu_check.py`, fed by
a `torch.utils.data.DataLoader` that sleeps a known time before each batch (`--wait-ms`, four times
that on steps 5, 15, 25, ...). The `on` arm runs it inside
`tensorquill.init(..., every=--every, system=True, data_wait=True)`; the `off` arm runs it with no
TensorQuill session. The script's own `perf_counter` brackets `next()` and the whole step in both
arms; on CUDA the device is synchronized at the end of every step in both arms, so a step's time is
its own.

What is kept, in `loader_delay.json` and in `arm-<arm>-r<round>.json`, for every step of every arm
and round: the injected sleep, the time in `next()`, the excess of one over the other, and the step's
time; for the `on` arm also TensorQuill's own `data_wait` and `step_time` for that step. The summary
gives, per arm: the worst whole-step delay (below) with its step and round; the worst excess in
`next()` with its step and round; the slowest rest of a step
(its time outside `next()`) against that round's median; every garbage collection of the interpreter
that took 1 ms or more, with its step, the thread it ran on and whether the training thread was in
`next()` at the time (a collection on any thread holds the interpreter lock, so the training thread
waits for it wherever it is); the excess at the named steps
(`--name-steps`, by default every flush boundary, so step 100 at `--every 50`) in each round; the
steps over `--slack-ms`; and the median step time. `on_minus_off` sets the two arms side by side at
each named step.

A step's **whole-step delay** is its time less the loader's injected sleep for that step, less the
round's median of the same quantity: how much longer that step took than a usual one, the known
sleeps taken out (every tenth step sleeps four times as long). It is reported twice, over every step
and over every step but the first, because the first step of a process is slow in both arms (on the
CPU dry runs it was the slowest part outside `next()` in nearly every round of both). It is reported
apart from the median overhead: one does not stand for the other.

The median step-time overhead printed here is this toy loop's, with a sleeping loader. It is not the
preregistered figure: that one is the Granite compiler gate's (`granite_compiler_gate.py`), whose
loop has no loader and so cannot show a delay in `next()`. The gate reports its own slowest step per
arm and round.

Exit status: 0 when both arms were measured in every round, 1 when an arm process failed or wrote no
result (what was measured is still written), 2 bad arguments. There is no pass line.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_check import MoE, routers  # noqa: E402
from gpu_system_check import SleepyBatches  # noqa: E402

ARMS = ("off", "on")


def run_arm(args) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    torch.manual_seed(0)
    model = MoE(args.dim, args.blocks).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    data = SleepyBatches(args.steps, args.batch, args.dim, args.wait_ms / 1000)
    loader = torch.utils.data.DataLoader(data, batch_size=None, sampler=range(args.steps), num_workers=0)
    next_ms: list[float] = []
    step_ms: list[float] = []
    collections: list[dict[str, Any]] = []  # the interpreter's garbage collections during training, both arms
    began_gc = [0.0]

    def on_gc(phase: str, info: dict) -> None:
        if phase == "start":
            began_gc[0] = time.perf_counter()
        else:
            collections.append({"step": len(step_ms), "in_next": len(next_ms) == len(step_ms) and in_next[0],
                                "generation": info.get("generation"), "collected": info.get("collected"),
                                "thread": threading.current_thread().name, "ms": (time.perf_counter() - began_gc[0]) * 1e3})

    in_next = [False]

    def train() -> None:
        batches = iter(loader)
        for _ in range(args.steps):
            began = time.perf_counter()
            in_next[0] = True
            x = next(batches)  # with capture on, TensorQuill times this call too
            in_next[0] = False
            got = time.perf_counter()
            x = x.to(device, non_blocking=True)
            loss = (model(x) - x.sum(dim=1, keepdim=True)).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if cuda:
                torch.cuda.synchronize()
            ended = time.perf_counter()
            next_ms.append((got - began) * 1e3)
            step_ms.append((ended - began) * 1e3)

    def timed_train() -> None:
        gc.callbacks.append(on_gc)
        try:
            train()
        finally:
            gc.callbacks.remove(on_gc)

    result: dict[str, Any] = {"arm": args.arm, "round": args.round, "torch": torch.__version__,
                              "python": platform.python_version(), "cpu_count": os.cpu_count(),
                              "gc_threshold": list(gc.get_threshold()),
                              "device": torch.cuda.get_device_name() if cuda else "CPU",
                              "steps": args.steps, "every": args.every, "wait_ms": args.wait_ms}
    if args.arm == "on":
        import tensorquill

        jsonl = Path(args.result).with_suffix(".jsonl")
        with tensorquill.init(model, run_id="loader-delay", every=args.every, sink=tensorquill.JsonlSink(jsonl),
                              routers=routers(model), system=True, data_wait=True):
            timed_train()
        by_step = {r["step"]: r for r in tensorquill.load_records(jsonl) if isinstance(r.get("step"), int)}

        def recorded(key: str) -> list[float | None]:
            values = [(by_step.get(s) or {}).get(key) for s in range(args.steps)]
            return [v * 1e3 if isinstance(v, (int, float)) and not isinstance(v, bool) else None for v in values]

        result.update(tensorquill_file=tensorquill.__file__, capture=str(jsonl),
                      tq_data_wait_ms=recorded("data_wait"), tq_step_time_ms=recorded("step_time"))
    else:
        timed_train()
    injected = [data.delay(s) * 1e3 for s in range(args.steps)]
    result.update(state="measured", injected_ms=injected, next_ms=next_ms,
                  excess_ms=[n - i for n, i in zip(next_ms, injected)], step_ms=step_ms,
                  rest_ms=[t - n for t, n in zip(step_ms, next_ms)], gc_collections=collections)
    return result


def arm_main(args) -> int:
    try:
        result = run_arm(args)
    except Exception as error:  # noqa: BLE001  a failed arm's result is written too
        result = {"arm": args.arm, "round": args.round, "state": "error", "error": f"{type(error).__name__}: {error}"}
    Path(args.result).write_text(json.dumps(result))
    return 0 if result["state"] == "measured" else 1


def named_steps(args) -> list[int]:
    if args.name_steps:
        return sorted({s for s in args.name_steps if 0 <= s < args.steps})
    return list(range(args.every, args.steps, args.every))


def whole_step_delay(result: dict) -> list[float]:
    """Each step's time less its injected sleep, less the round's median of that: the step's delay."""
    net = [t - i for t, i in zip(result["step_ms"], result["injected_ms"])]
    usual = statistics.median(net)
    return [n - usual for n in net]


def _worst(rows: list[tuple[float, int, int]]) -> dict[str, Any] | None:
    if not rows:
        return None
    ms, step, rnd = max(rows, key=lambda item: item[0])
    return {"ms": ms, "step": step, "round": rnd}


def summarize(results: dict[str, list[dict]], names: list[int], slack_ms: float) -> dict[str, Any]:
    """Per arm: the worst excess in `next()`, the excess at each named step by round, the steps over
    the slack, the median step time. Reported, never judged."""
    arms: dict[str, Any] = {}
    for arm, rounds in results.items():
        measured = [r for r in rounds if r.get("state") == "measured"]
        entry: dict[str, Any] = {"rounds_measured": len(measured),
                                 "errors": [r.get("error", "no result") for r in rounds if r.get("state") != "measured"]}
        if measured:
            worst = max(((r["excess_ms"][s], s, r["round"]) for r in measured for s in range(len(r["excess_ms"]))),
                        key=lambda item: item[0])
            delays = {r["round"]: whole_step_delay(r) for r in measured}
            rows = [(d, s, rnd) for rnd, ds in delays.items() for s, d in enumerate(ds)]
            entry.update(
                worst_whole_step=_worst(rows),
                worst_whole_step_after_first=_worst([row for row in rows if row[1] > 0]),
                worst_whole_step_by_round=[{"round": rnd, **{k: v for k, v in (_worst([(d, s, rnd) for s, d in enumerate(ds) if s > 0]) or {}).items() if k != "round"}}
                                           for rnd, ds in delays.items()],
                whole_step_delay_at_named_steps={str(s): [ds[s] for ds in delays.values()] for s in names},
                worst_excess={"ms": worst[0], "step": worst[1], "round": worst[2]},
                worst_excess_by_round=[{"round": r["round"], "ms": max(r["excess_ms"]),
                                        "step": r["excess_ms"].index(max(r["excess_ms"]))} for r in measured],
                excess_at_named_steps={str(s): [r["excess_ms"][s] for r in measured] for s in names},
                step_ms_at_named_steps={str(s): [r["step_ms"][s] for r in measured] for s in names},
                over_slack=[{"round": r["round"], "step": s, "excess_ms": e, "step_ms": r["step_ms"][s]}
                            for r in measured for s, e in enumerate(r["excess_ms"]) if e > slack_ms],
                excess_ms_median=statistics.median(e for r in measured for e in r["excess_ms"]),
                # The step's time outside next(): the forward, backward, optimizer step and TensorQuill's own work.
                worst_rest_by_round=[{"round": r["round"], "ms": max(r["rest_ms"]), "step": r["rest_ms"].index(max(r["rest_ms"])),
                                      "over_median_ms": max(r["rest_ms"]) - statistics.median(r["rest_ms"])} for r in measured],
                gc_over_1ms=[{"round": r["round"], **c} for r in measured for c in r.get("gc_collections", []) if c["ms"] >= 1.0],
                gc_collections=sum(len(r.get("gc_collections", [])) for r in measured),
                step_ms_median_by_round=[statistics.median(r["step_ms"]) for r in measured],
                step_ms_mean_by_round=[statistics.fmean(r["step_ms"]) for r in measured],
            )
        arms[arm] = entry
    summary: dict[str, Any] = {"arms": arms, "named_steps": names, "slack_ms": slack_ms}
    off, on = arms.get("off", {}), arms.get("on", {})
    if off.get("rounds_measured") and on.get("rounds_measured"):
        summary["on_minus_off"] = {
            "excess_ms_at_named_steps": {
                s: {"on_median": statistics.median(on["excess_at_named_steps"][s]),
                    "off_median": statistics.median(off["excess_at_named_steps"][s]),
                    "on_max": max(on["excess_at_named_steps"][s]), "off_max": max(off["excess_at_named_steps"][s])}
                for s in map(str, names)},
            "worst_excess_ms": {"on": on["worst_excess"], "off": off["worst_excess"]},
            "worst_whole_step_ms": {"on": on["worst_whole_step"], "off": off["worst_whole_step"]},
            "worst_whole_step_after_first_ms": {"on": on["worst_whole_step_after_first"],
                                                "off": off["worst_whole_step_after_first"]},
            "toy_step_time_overhead_median": (statistics.median(on["step_ms_median_by_round"])
                                              / statistics.median(off["step_ms_median_by_round"]) - 1),
            "toy_step_time_overhead_mean": (statistics.median(on["step_ms_mean_by_round"])
                                            / statistics.median(off["step_ms_mean_by_round"]) - 1),
            "note": "the toy overheads are this sleeping-loader loop's; the preregistered median overhead is "
                    "granite_compiler_gate.py's",
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cuda = torch.cuda.is_available()
    parser.add_argument("--out", type=Path, help="new directory for the arms' results and loader_delay.json")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--dim", type=int, default=1024 if cuda else 64)
    parser.add_argument("--blocks", type=int, default=4 if cuda else 2)
    parser.add_argument("--batch", type=int, default=4096 if cuda else 256)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--every", type=int, default=50, help="steps between TensorQuill's flushes in the on arm")
    parser.add_argument("--wait-ms", type=float, default=20.0, help="the loader's sleep before each batch")
    parser.add_argument("--slack-ms", type=float, default=15.0, help="steps whose next() is this far over the sleep are listed")
    parser.add_argument("--name-steps", type=int, nargs="*", help="steps to report by name (default: every flush boundary)")
    parser.add_argument("--arm-timeout", type=float, default=1800)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--round", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.arm:
        return arm_main(args)
    if args.out is None:
        parser.error("--out is required")
    if args.rounds < 1 or args.steps <= args.every:
        parser.error("--rounds must be at least 1 and --steps more than --every")
    args.out.mkdir(parents=True)
    names = named_steps(args)
    results: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for round_index in range(args.rounds):
        for arm in (ARMS if round_index % 2 == 0 else ARMS[::-1]):
            path = args.out / f"arm-{arm}-r{round_index}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--arm", arm, "--round", str(round_index),
                       "--result", str(path), "--dim", str(args.dim), "--blocks", str(args.blocks),
                       "--batch", str(args.batch), "--steps", str(args.steps), "--every", str(args.every),
                       "--wait-ms", str(args.wait_ms)]
            with (args.out / f"arm-{arm}-r{round_index}.log").open("w") as log:
                try:
                    code: Any = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.arm_timeout).returncode
                except subprocess.TimeoutExpired:
                    code = "timeout"
            try:
                result = json.loads(path.read_text())
            except (OSError, ValueError):
                result = {"arm": arm, "round": round_index, "state": "error",
                          "error": f"the arm process wrote no result (exit {code})"}
            results[arm].append(result)
            if result.get("state") == "measured":
                worst = max(result["excess_ms"])
                print(f"Round {round_index + 1} {arm:<3} worst next() {worst:+.2f} ms over its sleep at step "
                      f"{result['excess_ms'].index(worst)}; median step {statistics.median(result['step_ms']):.2f} ms", flush=True)
            else:
                print(f"Round {round_index + 1} {arm:<3} {result.get('error')}", flush=True)
    summary = summarize(results, names, args.slack_ms)
    first = next((r for rs in results.values() for r in rs if r.get("state") == "measured"), {})
    document = {"measurement": "loader_delay", "judged": False, "torch": first.get("torch"), "device": first.get("device"),
                "python": first.get("python"), "cpu_count": first.get("cpu_count"), "gc_threshold": first.get("gc_threshold"),
                "rounds": args.rounds, "steps": args.steps, "every": args.every, "wait_ms": args.wait_ms,
                **summary, "per_step": results}
    (args.out / "loader_delay.json").write_text(json.dumps(document, indent=1))
    for step in map(str, names):
        row = summary.get("on_minus_off", {}).get("excess_ms_at_named_steps", {}).get(step)
        if row:
            print(f"step {step}: next() over its sleep, on {row['on_median']:+.2f} ms median ({row['on_max']:+.2f} max), "
                  f"off {row['off_median']:+.2f} ms median ({row['off_max']:+.2f} max)")
    for arm in ARMS:
        entry = summary["arms"][arm]
        if entry.get("worst_excess"):
            w = entry["worst_excess"]
            rest = max(entry["worst_rest_by_round"], key=lambda item: item["over_median_ms"])
            slow_gc = max(entry["gc_over_1ms"], key=lambda item: item["ms"], default=None)
            whole, later = entry["worst_whole_step"], entry["worst_whole_step_after_first"]
            print(f"{arm}: worst whole-step delay {whole['ms']:+.2f} ms at step {whole['step']} (round {whole['round'] + 1})"
                  + (f"; after the first step {later['ms']:+.2f} ms at step {later['step']} (round {later['round'] + 1})" if later else ""))
            print(f"{arm}: worst delay in next() {w['ms']:+.2f} ms at step {w['step']} (round {w['round'] + 1}); "
                  f"{len(entry['over_slack'])} steps over {args.slack_ms:g} ms; worst rest of a step "
                  f"{rest['over_median_ms']:+.2f} ms over its median at step {rest['step']} (round {rest['round'] + 1}); "
                  + (f"longest garbage collection {slow_gc['ms']:.1f} ms at step {slow_gc['step']} "
                     f"(generation {slow_gc['generation']}, on thread {slow_gc['thread']}"
                     f"{', while the training thread was in next()' if slow_gc['in_next'] else ''})"
                     if slow_gc else "no garbage collection of 1 ms or more"))
    both = summary.get("on_minus_off")
    if both:
        print(f"median step-time overhead of this toy loop, reported apart from the delays: "
              f"{100 * both['toy_step_time_overhead_median']:+.2f}% (not the preregistered figure, which is the compiler gate's)")
    failed = [e for arm in ARMS for e in summary["arms"][arm]["errors"]]
    print(f"{'MEASURED' if not failed else 'INCOMPLETE'}: result {args.out / 'loader_delay.json'} (a measurement: nothing is judged)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
