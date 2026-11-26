"""The Granite compiler gate: four arms, each in a fresh process, over paired rounds.

    python examples/granite_compiler_gate.py --out runs/granite-gate                # GPU, Granite, reduce-overhead
    python examples/granite_compiler_gate.py --out runs/gate-toy --toy              # CPU dry run of the harness

The arms train the same model on the same batches with the same warm-up (`--warmup`, 50) and
timed steps (`--steps`, 200):

- `off`: no tool attached. The baseline.
- `tensorquill`: `tensorquill.init(model, capture="routers", routers=...)`, the loss logged with
  `session.log` on every step (a copy on the device, no sync), flushed every 50 steps.
- `neuraldbg`: `with NeuralDbg(model)` wrapped before `torch.compile`, as its README says, and its
  README's loop: `dbg.record_loss(loss.item())` and `dbg.step_iteration()` on every step. Its
  hooks sit on every leaf module, read on every forward and backward, and are
  `torch._dynamo.disable`d. `pip install neuraldbg`, or `--neuraldbg-path` to a clone (read
  only: bytecode writing is turned off for that arm's process).
- `wandb`: `wandb.init(mode="offline")` (no account, no network) and
  `wandb.watch(model, log="all", log_freq=--wandb-log-freq)`, with `wandb.log({"train/loss":
  loss.item()})` on every step, which commits what `watch` collected. `pip install wandb`.

Every arm compiles with `fullgraph=False`, so a graph break is counted instead of raised, and,
on CUDA with Inductor, `mode="reduce-overhead"` (CUDA graphs). Each round runs every arm once,
each in a new Python process, in an order that rotates by round. Each arm process writes
`arm-<arm>-r<round>.json` and its log; the driver writes `gate.json` with every arm's result,
the failing and the absent ones too. An arm whose package is not installed is `not_run`, with
the reason, and is never counted as passed; an arm that crashed is `error`, with its message.

Per arm and round it records: the mean and median step time over the timed steps, graph breaks,
graph compiles after warm-up (the recompiles), CUDA-graph skips, frames compiled, what the tool
observes and how often it samples, and the tool's own result checked against an expectation
the script computes itself (TensorQuill: every step recorded and each router's assignments
summing to tokens x top_k; NeuralDBG: one hook pair on each leaf module of the model; wandb: the
parameters it watches against the model's trainable ones, and the samples it took).

The overhead of an arm is, per round, its mean step time over the timed steps (the block's wall
time over its steps, as `moe_dogfood.py` times it, so the flush and sample steps count) over the
`off` arm's in the same round, minus one; the arm's overhead is the median of those. The same
ratio of per-step medians is written beside it. The median of the paired rounds' overheads is the
primary 1% criterion (the author, Nov 22, 2026); the ratio of the arms' median rounds, the figure
`moe_dogfood.py` printed, is reported alongside for comparison with earlier results and never scored
(said aloud when it falls on the other side of the line). `gate.json` says which is which
(`overhead_criterion`).
The slowest timed step of every arm and round is reported with how far it is over that round's
median step (`worst_step_by_round`), and every step's time is kept in the arm's file (`step_ms`):
a passing median does not show one slow step. None of that is scored. The pass line, for the TensorQuill
arm only: median overhead 1% or less, zero graph breaks, zero recompiles after warm-up, zero
CUDA-graph skips under `reduce-overhead`, and its capture checks. Exit 0 when it passes, 1 when
it does not. With `--toy`, `--eager` or without CUDA, the timing and CUDA-graph lines are not
established and are written as such (`dry_run`): exit 0 then rests on the capture checks and the
compile counters only. On a GPU without CUDA graphs (`--mode default`) every line but the
CUDA-graph one is judged, and a run that meets them is `not_proven`, exit 3.

If capture off fails in the first round, nothing can be measured against it: the gate stops there,
runs no other arm, and reads `not_proven`, exit 3, with capture off's error. On the A100 every arm,
capture off too, failed in CUDA-graph recording under `reduce-overhead` (Granite's experts call
`torch._grouped_mm`, whose path on that GPU copies to the host), and the gate used to run all twelve
arm processes into that failure and read `fail`, as if TensorQuill had failed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

import tensorquill

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ARMS = ("off", "tensorquill", "neuraldbg", "wandb")
FLUSH_EVERY = 50
OVERHEAD_LIMIT = 0.01
NOT_PROVEN = 3  # the run sheet's exit status for "not proven"
TOY_TOKENS = (2, 32)  # batch size and sequence length of the toy batches


# ---- the model, batches and the step -------------------------------------------------------


def build(args, device: torch.device):
    """The model, its routers for TensorQuill, the batches and the tokens per batch."""
    torch.manual_seed(0)
    total = args.warmup + args.steps
    if args.toy:
        from profile_overhead import _ToyMoE

        model = _ToyMoE().to(device).train()
        routers = {f"routers.{i}": tensorquill.Router(8, top_k=model.top_k) for i in range(len(model.routers))}
        generator = torch.Generator().manual_seed(0)
        batches = torch.randint(0, 256, (total, *TOY_TOKENS), generator=generator).to(device)
        return model, routers, batches, model.top_k
    from moe_dogfood import build_model, granite_routers, load_batches

    model = build_model()
    routers = granite_routers(model)
    batches = load_batches(args, total)
    return model, routers, batches, model.config.num_experts_per_tok


def train_step(runner, optimizer, batch, cuda_graphs: bool):
    optimizer.zero_grad(set_to_none=True)
    if cuda_graphs:
        torch.compiler.cudagraph_mark_step_begin()
    loss = runner(input_ids=batch, labels=batch, use_cache=False).loss
    loss.backward()
    return loss


def counter_snapshot() -> dict[str, int]:
    counters = torch._dynamo.utils.counters
    return {
        "graph_breaks": int(sum(counters["graph_break"].values())),
        "unique_graphs": int(counters["stats"]["unique_graphs"]),
        "frames_total": int(counters["frames"]["total"]),
        "frames_ok": int(counters["frames"]["ok"]),
        "cudagraph_skips": int(counters["inductor"]["cudagraph_skips"]),
    }


# ---- the tools -----------------------------------------------------------------------------


class Tool:
    """One arm's tool: attached before compiling, told about each step, closed, then checked."""

    name = "off"
    observes = "nothing: no tool attached"
    samples = "never"

    def after_backward(self, loss: torch.Tensor) -> None:
        pass

    def after_step(self, loss: torch.Tensor) -> None:
        pass

    def close(self) -> None:
        pass

    def check(self, args, model, routers, tokens: int, top_k: int) -> tuple[list[dict], dict]:
        return [], {}


class TensorQuillTool(Tool):
    name = "tensorquill"

    def __init__(self, args, model, routers, out: Path) -> None:
        self.path = out
        self.session = tensorquill.init(
            model, routers=routers, capture="routers", every=FLUSH_EVERY,
            run_id=f"granite-gate-{args.round}", sink=tensorquill.JsonlSink(out),
        )
        self.observes = (f"routing counts of {len(routers)} routers on every step, and the logged loss; "
                         f"copied to the host once per flush")
        self.samples = f"every step; one transfer to the CPU every {FLUSH_EVERY} steps"

    def after_backward(self, loss):
        self.session.log({"train/loss": loss})  # before optimizer.step(), where the step ends

    def close(self):
        self.session.close()

    def check(self, args, model, routers, tokens, top_k):
        records = tensorquill.load_records(self.path)
        expected_steps = args.warmup + args.steps
        steps = [r["step"] for r in records]
        checks = [{
            "name": "every optimizer step recorded",
            "ok": steps == list(range(expected_steps)),
            "detail": f"{len(steps)} records, steps {steps[:1]}..{steps[-1:]}, expected 0..{expected_steps - 1}",
        }]
        want = tokens * top_k
        wrong = [(r["step"], name, sum(r["moe"].get(name, []))) for r in records for name in routers
                 if sum(r["moe"].get(name, [])) != want]
        checks.append({
            "name": "each router's assignments sum to tokens x top_k on every step",
            "ok": not wrong and bool(records),
            "detail": f"expected {want} ({tokens} tokens x top_k {top_k}) for {len(routers)} routers"
                      + (f"; {len(wrong)} wrong, first {wrong[0]}" if wrong else ""),
        })
        return checks, {"records": len(records), "routers": len(routers)}


class NeuralDbgTool(Tool):
    name = "neuraldbg"

    def __init__(self, args, model) -> None:
        from neuraldbg import NeuralDbg  # ImportError: the arm is not_run

        self.dbg = NeuralDbg(model).__enter__()
        self.observes = ("on every leaf module: input and output statistics on each forward (data anomaly, "
                         "activation regime; several .item() reads each) and the gradient norm on each "
                         "backward; the loss through record_loss(loss.item())")
        self.samples = "every forward and backward call of every leaf module, every step"

    def after_step(self, loss):
        self.dbg.record_loss(loss.item())
        self.dbg.step_iteration()

    def close(self):
        self.dbg.__exit__(None, None, None)

    def check(self, args, model, routers, tokens, top_k):
        # The leaves NeuralDbg's own walk takes: leaf modules, and the root.
        expected = sum(1 for name, m in model.named_modules() if name == "" or not list(m.children()))
        hooked = int(getattr(self.dbg, "_hooked_leaf_count", -1))
        events = self.dbg.get_events()
        checks = [{
            "name": "a hook pair on each leaf module of the model",
            "ok": hooked == expected,
            "detail": f"NeuralDbg hooked {hooked}; the model has {expected} (leaf modules and the root)",
        }, {
            "name": "the step counter reached the steps trained",
            "ok": self.dbg.step == args.warmup + args.steps,
            "detail": f"dbg.step {self.dbg.step}, steps trained {args.warmup + args.steps}",
        }]
        kinds: dict[str, int] = {}
        for event in events:
            kind = getattr(getattr(event, "event_type", None), "value", str(getattr(event, "event_type", "?")))
            kinds[kind] = kinds.get(kind, 0) + 1
        return checks, {"hooked_modules": hooked, "events": len(events), "event_kinds": kinds,
                        "losses_recorded": len(self.dbg.loss_history)}


def _wandb_torch_history():
    """wandb's TorchHistory class, wherever this version keeps it, or None."""
    for module in ("wandb.integration.torch.wandb_torch", "wandb.wandb_torch"):
        try:
            mod = __import__(module, fromlist=["TorchHistory"])
            return getattr(mod, "TorchHistory", None)
        except Exception:  # noqa: BLE001
            continue
    return None


class WandbTool(Tool):
    name = "wandb"

    def __init__(self, args, model, out_dir: Path) -> None:
        import wandb  # ImportError: the arm is not_run

        self.wandb = wandb
        self.samples_taken = None
        history = _wandb_torch_history()
        if history is not None and hasattr(history, "log_tensor_stats"):
            original = history.log_tensor_stats
            self.samples_taken = 0

            def counted(this, *a, **k):
                self.samples_taken += 1
                return original(this, *a, **k)

            history.log_tensor_stats = counted
        out_dir.mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(mode="offline", project="tensorquill-compiler-gate", dir=str(out_dir),
                              name=f"wandb-r{args.round}")
        self.log_freq = args.wandb_log_freq
        wandb.watch(model, log="all", log_freq=self.log_freq)
        self.observes = ("histograms of every trainable parameter and of its gradient (log='all'), "
                         "and the loss through wandb.log(loss.item())")
        self.samples = f"parameters and gradients every {self.log_freq} calls of its hooks; the loss every step"

    def after_step(self, loss):
        self.wandb.log({"train/loss": loss.item()})

    def close(self):
        self.wandb.finish()

    def check(self, args, model, routers, tokens, top_k):
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        watched = None
        try:
            watched = len(self.run._torch_history._hook_handles)  # noqa: SLF001  # what wandb hooked
        except Exception:  # noqa: BLE001
            pass
        steps = args.warmup + args.steps
        checks = [{
            "name": "samples were taken",
            "ok": None if self.samples_taken is None else self.samples_taken > 0,  # None: not known
            "detail": (f"{self.samples_taken} log_tensor_stats calls over {steps} steps, log_freq {self.log_freq}"
                       if self.samples_taken is not None else "wandb's TorchHistory was not found: samples not counted"),
        }]
        return checks, {"trainable_parameters": trainable, "hook_handles": watched,
                        "samples_taken": self.samples_taken, "log_freq": self.log_freq}


# ---- one arm, in its own process -----------------------------------------------------------


def run_arm(args) -> dict[str, Any]:
    out = Path(args.result)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result: dict[str, Any] = {
        "arm": args.arm, "round": args.round, "toy": args.toy, "device": str(device),
        "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "torch": torch.__version__, "tensorquill_file": tensorquill.__file__,
        "warmup": args.warmup, "steps": args.steps, "pid": os.getpid(),
    }
    if not args.toy and device.type != "cuda":
        return {**result, "state": "not_run", "reason": "a CUDA GPU is required without --toy"}
    if args.arm == "neuraldbg":
        try:
            import neuraldbg  # noqa: F401
        except ImportError as error:
            return {**result, "state": "not_run",
                    "reason": f"neuraldbg is not importable ({error}): pip install neuraldbg, or --neuraldbg-path"}
        result["package"] = {"neuraldbg": getattr(neuraldbg, "__version__", "?"), "file": neuraldbg.__file__}
    if args.arm == "wandb":
        try:
            import wandb
        except ImportError as error:
            return {**result, "state": "not_run", "reason": f"wandb is not installed ({error}): pip install wandb"}
        result["package"] = {"wandb": getattr(wandb, "__version__", "?")}

    backend = args.backend or ("inductor" if device.type == "cuda" else "aot_eager")
    mode = None if args.eager or backend != "inductor" else args.mode
    cuda_graphs = device.type == "cuda" and mode == "reduce-overhead"
    result.update(backend=None if args.eager else backend, mode=mode, eager=args.eager, cuda_graphs=cuda_graphs,
                  fullgraph=False)

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    model, routers, batches, top_k = build(args, device)
    tokens = int(batches.shape[1] * batches.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=device.type == "cuda")
    tool: Tool
    if args.arm == "tensorquill":
        tool = TensorQuillTool(args, model, routers, out.with_suffix(".jsonl"))
    elif args.arm == "neuraldbg":
        tool = NeuralDbgTool(args, model)
    elif args.arm == "wandb":
        tool = WandbTool(args, model, out.parent / "wandb")
    else:
        tool = Tool()
    result.update(observes=tool.observes, samples=tool.samples)

    with torch._dynamo.config.patch(skip_nnmodule_hook_guards=False):
        runner = model if args.eager else torch.compile(model, backend=backend, mode=mode, fullgraph=False)
        last_loss = None
        try:
            for batch in batches[: args.warmup]:
                loss = train_step(runner, optimizer, batch, cuda_graphs)
                tool.after_backward(loss)
                optimizer.step()
                tool.after_step(loss)
            sync(device)
            warm = counter_snapshot()
            times: list[float] = []
            events = []
            start = time.perf_counter()
            for batch in batches[args.warmup:]:
                if device.type == "cuda":
                    begin = torch.cuda.Event(enable_timing=True)
                    begin.record()
                else:
                    begin = time.perf_counter()
                loss = train_step(runner, optimizer, batch, cuda_graphs)
                tool.after_backward(loss)
                optimizer.step()
                tool.after_step(loss)
                if device.type == "cuda":
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    events.append((begin, end))
                else:
                    times.append((time.perf_counter() - begin) * 1000)
                last_loss = loss.detach()  # only the last: no reference kept to earlier CUDA-graph outputs
            sync(device)
            block_ms = (time.perf_counter() - start) * 1000 / args.steps
            times += [b.elapsed_time(e) for b, e in events]
            final = counter_snapshot()
        finally:
            tool.close()
    last = float(last_loss) if last_loss is not None else math.nan
    checks, seen = tool.check(args, model, routers, tokens, top_k)
    checks.insert(0, {"name": "the final loss is finite", "ok": math.isfinite(last), "detail": f"{last:.4f}"})
    return {
        **result, "state": "measured",
        "median_ms": statistics.median(times), "mean_ms": block_ms, "step_ms": times,
        "counters_after_warmup": warm, "counters_final": final,
        "graph_breaks": final["graph_breaks"],
        "recompiles_after_warmup": final["unique_graphs"] - warm["unique_graphs"],
        "frames_after_warmup": final["frames_total"] - warm["frames_total"],
        "cudagraph_skips": final["cudagraph_skips"] if cuda_graphs else None,
        "frames_ok": final["frames_ok"],
        "final_loss": last, "checks": checks, "seen": seen,
    }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def arm_main(args) -> int:
    out = Path(args.result)
    try:
        result = run_arm(args)
    except Exception as error:  # noqa: BLE001  # every arm's result is written, a crash too
        result = {"arm": args.arm, "round": args.round, "state": "error",
                  "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    out.write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: v for k, v in result.items() if k not in ("step_ms", "traceback")}, default=str, indent=1))
    return 0 if result["state"] in ("measured", "not_run") else 1


# ---- the driver ----------------------------------------------------------------------------


def arm_command(args, arm: str, round_index: int, result: Path) -> list[str]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--arm", arm, "--round", str(round_index),
           "--result", str(result), "--out", str(args.out), "--warmup", str(args.warmup), "--steps", str(args.steps),
           "--batch-size", str(args.batch_size), "--seq-len", str(args.seq_len), "--mode", args.mode,
           "--wandb-log-freq", str(args.wandb_log_freq)]
    if args.backend:
        cmd += ["--backend", args.backend]
    for flag in ("toy", "eager"):
        if getattr(args, flag):
            cmd.append(f"--{flag}")
    return cmd


def arm_env(args, arm: str) -> dict[str, str]:
    env = dict(os.environ)
    # The arms measure the tensorquill this driver imported, not whatever the environment installed.
    paths = [str(Path(tensorquill.__file__).resolve().parents[1]), str(HERE)]
    if arm == "neuraldbg" and args.neuraldbg_path:
        paths.append(str(Path(args.neuraldbg_path).resolve()))
        env["PYTHONDONTWRITEBYTECODE"] = "1"  # a clone that is read, never written
    env["PYTHONPATH"] = os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    if arm == "wandb":
        env.update(WANDB_MODE="offline", WANDB_SILENT="true", WANDB_DIR=str(Path(args.out) / "wandb"))
    return env


def summarize(args, results: dict[str, list[dict]]) -> dict[str, Any]:
    off = {r["round"]: r for r in results.get("off", []) if r.get("state") == "measured"}
    arms: dict[str, Any] = {}
    for arm, rounds in results.items():
        states = sorted({r.get("state", "missing") for r in rounds})
        entry: dict[str, Any] = {"states": states, "rounds": len(rounds)}
        measured = [r for r in rounds if r.get("state") == "measured"]
        if "not_run" in states:
            entry["reason"] = next(r["reason"] for r in rounds if r.get("state") == "not_run")
        if "error" in states or "missing" in states:
            entry["errors"] = [r.get("error", "no result") for r in rounds if r.get("state") in ("error", "missing")]
        if measured:
            first = measured[0]
            # The round's mean step: what the flushes (TensorQuill), samples (wandb) and per-step work all cost.
            # The per-step median would leave out the few steps a periodic cost falls on.
            ratios = [r["mean_ms"] / off[r["round"]]["mean_ms"] - 1 for r in measured if r["round"] in off]
            median_ratios = [r["median_ms"] / off[r["round"]]["median_ms"] - 1 for r in measured if r["round"] in off]
            skips = [r["cudagraph_skips"] for r in measured if r["cudagraph_skips"] is not None]
            off_rounds_ms = statistics.median(r["mean_ms"] for r in off.values()) if off else None
            entry.update(
                observes=first["observes"], samples=first["samples"],
                median_ms=statistics.median(r["median_ms"] for r in measured),
                mean_ms=statistics.median(r["mean_ms"] for r in measured),
                median_ms_by_round=[r["median_ms"] for r in measured],
                mean_ms_by_round=[r["mean_ms"] for r in measured],
                overhead_by_round=ratios if arm != "off" else None,
                overhead_median=statistics.median(ratios) if ratios and arm != "off" else None,
                step_median_overhead_median=statistics.median(median_ratios) if median_ratios and arm != "off" else None,
                # moe_dogfood.py's figure, reported beside the scored one: each arm's median round, then the ratio.
                overhead_of_median_rounds=(statistics.median(r["mean_ms"] for r in measured) / off_rounds_ms - 1
                                           if off_rounds_ms and arm != "off" else None),
                worst_step_by_round=[worst_step(r, getattr(args, "warmup", 0)) for r in measured],
                graph_breaks=max(r["graph_breaks"] for r in measured),
                recompiles_after_warmup=max(r["recompiles_after_warmup"] for r in measured),
                frames_ok=min(r["frames_ok"] for r in measured),
                cudagraph_skips=max(skips) if skips else None,
                checks_ok=all(c["ok"] is not False for r in measured for c in r["checks"]),
                failed_checks=[c for r in measured for c in r["checks"] if c["ok"] is False],
                unknown_checks=[c for r in measured for c in r["checks"] if c["ok"] is None],
                seen=first.get("seen"), config={k: first.get(k) for k in ("backend", "mode", "cuda_graphs", "eager", "fullgraph")},
            )
        arms[arm] = entry
    return arms


# The author, Nov 22, 2026: "record the median of the three paired overheads as the primary 1% criterion.
# Report the ratio of arm medians alongside it for comparison with earlier results."
OVERHEAD_CRITERION = {
    "primary": "overhead_median: the median of the paired rounds' overheads (each round's mean step time over "
               "capture off's in the same round, minus one), 1% or less; this is the scored figure",
    "alongside": "overhead_of_median_rounds: the ratio of the arms' median rounds, as moe_dogfood.py printed it on "
                 "earlier runs; reported for comparison, never scored",
}


def worst_step(result: dict, warmup: int) -> dict[str, Any]:
    """The slowest timed step of one arm's round: reported, never scored. `step` counts from the first
    warm-up step (the TensorQuill arm's recorded step); every step's time is `step_ms` in the arm's file."""
    times = result.get("step_ms") or []
    if not times:
        return {"round": result.get("round"), "step": None, "ms": None}
    index = max(range(len(times)), key=times.__getitem__)
    return {"round": result.get("round"), "step": warmup + index, "ms": times[index],
            "over_median_ms": times[index] - result["median_ms"], "file": f"arm-{result.get('arm')}-r{result.get('round')}.json"}


def established_here(args) -> bool:
    """Whether this run can establish the timing claim: Granite on CUDA, compiled."""
    return not args.toy and torch.cuda.is_available() and not args.eager


def verdict(args, arms: dict[str, Any], established: bool) -> tuple[str, list[dict]]:
    """The TensorQuill arm's pass line, as (status, criteria). status: pass, fail, not_proven (a GPU run
    whose CUDA-graph line was not checked: CUDA graphs off) or dry_run (toy, CPU or eager)."""
    tq, off = arms.get("tensorquill", {}), arms.get("off", {})
    criteria: list[dict] = []

    def criterion(name: str, ok: bool | None, detail: str) -> None:
        criteria.append({"name": name, "ok": ok, "detail": detail})

    measured = tq.get("states") == ["measured"] and off.get("states") == ["measured"]
    criterion("capture off and TensorQuill measured in every round", measured,
              f"off {off.get('states')}, tensorquill {tq.get('states')}")
    if not measured:
        return "fail", criteria
    criterion("TensorQuill's capture checks", tq["checks_ok"], f"{len(tq['failed_checks'])} failed")
    criterion("the model was compiled", tq["frames_ok"] >= 1 or args.eager, f"frames ok {tq['frames_ok']}")
    criterion("zero graph breaks", tq["graph_breaks"] == 0, str(tq["graph_breaks"]))
    criterion("zero recompiles after warm-up", tq["recompiles_after_warmup"] == 0, str(tq["recompiles_after_warmup"]))
    overhead = tq["overhead_median"]
    if established:
        criterion(f"median overhead {100 * OVERHEAD_LIMIT:g}% or less", overhead <= OVERHEAD_LIMIT,
                  f"{100 * overhead:+.2f}% (rounds {[f'{100 * r:+.2f}%' for r in tq['overhead_by_round']]}; "
                  f"by each arm's median round, as moe_dogfood.py printed it: {100 * tq['overhead_of_median_rounds']:+.2f}%"
                  + ("" if (tq["overhead_of_median_rounds"] <= OVERHEAD_LIMIT) == (overhead <= OVERHEAD_LIMIT)
                     else ", WHICH FALLS ON THE OTHER SIDE OF THE LINE") + ")")
    else:
        criterion(f"median overhead {100 * OVERHEAD_LIMIT:g}% or less", None,
                  f"not established ({'toy' if args.toy else 'eager' if args.eager else 'no CUDA'}): "
                  f"{100 * overhead:+.2f}% measured")
    if tq["cudagraph_skips"] is None:
        criterion("zero CUDA-graph skips under reduce-overhead", None, "not checked: CUDA graphs were off")
    else:
        criterion("zero CUDA-graph skips under reduce-overhead", tq["cudagraph_skips"] == 0, str(tq["cudagraph_skips"]))
    if any(c["ok"] is False for c in criteria):
        return "fail", criteria
    if not established:
        return "dry_run", criteria
    return ("pass" if all(c["ok"] for c in criteria) else "not_proven"), criteria


def baseline_stopped(args, out: Path, device: str, results: dict[str, list[dict]], off: dict) -> int:
    """Capture off could not run in the first round: the mode cannot be judged here, so the gate stops."""
    error = off.get("error") or off.get("reason") or "no result"
    detail = f"round 1: {error}; nothing can be measured against it here, so no other arm was run"
    if "CUDA graph capture" in error:
        detail += ("; CUDA graphs could not record the model with no tool attached, so this mode cannot be judged "
                   "on this machine")
    criteria = [{"name": "capture off ran in this mode on this machine", "ok": None, "detail": detail}]
    gate = {"status": "not_proven", "criteria": criteria, "toy": args.toy, "rounds": args.rounds,
            "warmup": args.warmup, "steps": args.steps, "torch": torch.__version__, "device": device,
            "tensorquill_file": tensorquill.__file__, "arms": {arm: {"results": rounds} for arm, rounds in results.items()},
            "overhead_criterion": OVERHEAD_CRITERION,
            "claim": "not established: capture off itself could not run in this mode, so TensorQuill was not measured"}
    (out / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
    print(f"\nNOT ESTABLISHED: {criteria[0]['name']}: {detail}")
    print(f"\nGate: not_proven ({gate['claim']}). Result: {out / 'gate.json'}")
    return NOT_PROVEN


def driver_main(args) -> int:
    out = Path(args.out)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if "off" not in arms or "tensorquill" not in arms or not set(arms) <= set(ARMS):
        raise SystemExit(f"--arms must include off and tensorquill, from {', '.join(ARMS)}: the pass line compares them")
    out.mkdir(parents=True)
    device = "CUDA " + torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    print(f"torch {torch.__version__} on {device} | {'toy MoE' if args.toy else 'Granite'} | "
          f"{args.rounds} rounds x {len(arms)} arms, warm-up {args.warmup}, timed {args.steps}", flush=True)
    results: dict[str, list[dict]] = {arm: [] for arm in arms}
    for round_index in range(args.rounds):
        order = arms[round_index % len(arms):] + arms[: round_index % len(arms)]
        for arm in order:
            path = out / f"arm-{arm}-r{round_index}.json"
            log = out / f"arm-{arm}-r{round_index}.log"
            started = time.time()
            with log.open("w") as handle:
                try:
                    proc = subprocess.run(arm_command(args, arm, round_index, path), env=arm_env(args, arm),
                                          stdout=handle, stderr=subprocess.STDOUT, timeout=args.arm_timeout)
                    code = proc.returncode
                except subprocess.TimeoutExpired:
                    code = "timeout"
            try:
                result = json.loads(path.read_text())
            except (OSError, ValueError):
                result = {"arm": arm, "round": round_index, "state": "error",
                          "error": f"the arm process wrote no result (exit {code}); see {log.name}"}
                path.write_text(json.dumps(result, indent=1))
            result["exit"] = code
            results[arm].append(result)
            shown = (f"{result['median_ms']:.3f} ms/step median" if result.get("state") == "measured"
                     else result.get("reason") or result.get("error"))
            print(f"Round {round_index + 1} {arm:<12} {result.get('state'):<9} {shown} "
                  f"({time.time() - started:.0f} s, exit {code})", flush=True)
            if arm == "off" and round_index == 0 and result.get("state") != "measured":
                return baseline_stopped(args, out, device, results, result)
    summary = summarize(args, results)
    status, criteria = verdict(args, summary, established_here(args))
    gate = {"status": status, "criteria": criteria, "toy": args.toy, "rounds": args.rounds, "warmup": args.warmup,
            "steps": args.steps, "torch": torch.__version__, "device": device,
            "tensorquill_file": tensorquill.__file__, "arms": summary,
            "overhead_criterion": OVERHEAD_CRITERION,
            "claim": {"pass": "established", "fail": "failed",
                      "not_proven": "not established: the CUDA-graph line was not checked (CUDA graphs off)",
                      "dry_run": "not established: a CPU, toy or eager dry run checks the harness, not the GPU claim"}[status]}
    (out / "gate.json").write_text(json.dumps(gate, indent=1, default=str))

    print(f"\n{'arm':<12}{'state':<12}{'mean ms':>10}{'median ms':>10}{'overhead':>10}{'breaks':>8}{'recomp':>8}{'cg skips':>9}  checks")
    for arm, entry in summary.items():
        if "median_ms" in entry:
            over = "" if entry["overhead_median"] is None else f"{100 * entry['overhead_median']:+.2f}%"
            skips = "n/a" if entry["cudagraph_skips"] is None else entry["cudagraph_skips"]
            print(f"{arm:<12}{','.join(entry['states']):<12}{entry['mean_ms']:>10.3f}{entry['median_ms']:>10.3f}{over:>10}"
                  f"{entry['graph_breaks']:>8}{entry['recompiles_after_warmup']:>8}{skips!s:>9}  "
                  f"{'ok' if entry['checks_ok'] else 'FAILED'}")
            print(f"{'':<12}observes: {entry['observes']}\n{'':<12}samples: {entry['samples']}")
        else:
            print(f"{arm:<12}{','.join(entry['states']):<12}  {entry.get('reason') or entry.get('errors')}")
    print()
    for c in criteria:
        mark = {True: "PASS", False: "FAIL", None: "NOT ESTABLISHED"}[c["ok"]]
        print(f"{mark}: {c['name']}: {c['detail']}")
    print("\nSlowest timed step of each round (reported, not scored; every step's time is in the arm's file):")
    for arm, entry in summary.items():
        for w in entry.get("worst_step_by_round", []):
            if w["ms"] is not None:
                print(f"  {arm:<12} round {w['round'] + 1}: step {w['step']} took {w['ms']:.3f} ms, "
                      f"{w['over_median_ms']:+.3f} ms over that round's median step")
    print(f"\nGate: {status} ({gate['claim']}). Result: {out / 'gate.json'}")
    return {"pass": 0, "dry_run": 0, "not_proven": NOT_PROVEN}.get(status, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="new directory for the arms' results")
    parser.add_argument("--arms", default=",".join(ARMS), help=f"comma-separated, from {', '.join(ARMS)}")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200, help="timed steps per arm and round")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--backend", help="torch.compile backend (default: inductor on CUDA, aot_eager on CPU)")
    parser.add_argument("--mode", default="reduce-overhead", help="torch.compile mode with Inductor")
    parser.add_argument("--eager", action="store_true", help="do not compile (no compile counters are checked)")
    parser.add_argument("--toy", action="store_true", help="the small MoE of profile_overhead.py, no downloads, CPU allowed")
    parser.add_argument("--neuraldbg-path", help="a NeuralDBG clone to import from, if it is not installed")
    parser.add_argument("--wandb-log-freq", type=int, default=FLUSH_EVERY,
                        help=f"wandb.watch log_freq (wandb's default is 1000; default here {FLUSH_EVERY}, the flush interval)")
    parser.add_argument("--arm-timeout", type=float, default=3600, help="seconds before an arm process is stopped")
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--round", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if any(n <= 0 or n % FLUSH_EVERY for n in (args.warmup, args.steps)):
        parser.error("warmup and timed steps must be positive multiples of 50")
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if args.arm:
        return arm_main(args)
    if not args.toy and not torch.cuda.is_available():
        parser.error("a CUDA GPU is required; use --toy to check the harness on CPU")
    return driver_main(args)


if __name__ == "__main__":
    sys.exit(main())
