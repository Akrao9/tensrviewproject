"""What TensorQuill records under FSDP in a plain training loop: FSDP2 (`fully_shard` per block) and
FSDP1 (`FullyShardedDataParallel(use_orig_params=True)`, one unit per block). The multi-GPU test
plan's T5.

Every rank trains the same small residual model (`--blocks` blocks of a LayerNorm and a linear
layer, and a linear head) on its own random batches with AdamW, under a TensorQuill session
(`capture="full"`, `updates=True`, a weight check every `--every` steps, `probe=` set). Two plants:

- a gradient spike: at `--grad-step` the gradient of one parameter of block `--grad-block` is
  multiplied by `--grad-factor`, in place after the backward, on every rank's local shard (a hook
  on a sharded parameter would see the gathered tensor, not the gradient FSDP keeps). The
  parameter is chosen from the shard sizes of every rank: the largest one of the block that every
  rank holds part of, or the one named by `--grad-param`;
- an activation spike: from `--act-step` on, block `--act-block`'s output is multiplied by
  `--act-factor` (a forward hook registered before sharding and before TensorQuill's).

On every step each rank also computes, by collectives of its own, the whole parameters' numbers
the capture should combine to: each gradient's norm over the whole parameter (FSDP2:
`grad.full_tensor()`; FSDP1: `summon_full_params(with_grads=True)`), each weight's norm after the
step, and each update's RMS (the whole weight before and after the step: with `weight_decay=0`,
AdamW's update is their difference). These are the script's own and never read from the capture.

It fails unless, on every rank:

- one capture, every record naming its rank, steps 0 to `--steps` - 1, as many as optimizer steps;
- the activation detector finds the spike at `--act-step`, first in the planted block;
- the ranks' `grad_norm`s combine by the root of the sum of their squares to the script's whole
  norm on every step, within `--tolerance` (a rank's norm is over its own shard: this is the shard
  semantics the README states for the update size), and the same for every parameter's
  `grad_norms`, an empty shard counting 0;
- `update_rms` is on every record and no record says `update_not_read`; the ranks' RMSs, weighted by
  their shards' element counts, combine to the script's own within `--update-tolerance`;
- the ranks' `weight_norms` combine to the script's whole weight norms where a weight check ran;
- no record says `grads_not_read`;
- the probe: the author's decision of Nov 24, 2026 (after this row was written, which expected the
  probe to run) turns the probe off on a sharded model, so every rank's first record must say
  `probe: {"not_run": ...}` naming the sharding, and the probe callback must never be entered;
- where a rank's own total gradient norm at `--grad-step` reaches the detector's factor over its
  own median before, that rank's `grad_spike` incident is dated `--grad-step` and lists the
  planted parameter. A rank below the factor is reported, not judged: its norm is over its shard.

NOT PROVEN (exit 3) when a rank's shard of the planted parameter is empty, or when the script's own
whole gradient norm at `--grad-step` is not at least the detector's factor over its median before
(the plant did not make a spike the detector could be asked to find).

The driver (run without `--worker`) starts the ranks under `torch.distributed.run`, waits at most
`--timeout` seconds, ends any rank still running, and judges from the files the ranks wrote:

    python examples/fsdp_check.py --nproc 2 --arm fsdp2 --out runs/fsdp2-2
    python examples/fsdp_check.py --nproc 2 --arm fsdp1 --out runs/fsdp1-2
    python examples/fsdp_check.py --cpu --nproc 2 --arm fsdp2 --out runs/fsdp2-cpu   # Gloo on the CPU

Under `torchrun` directly it works too, rank 0 judging (torchrun turns every non-zero status into
its own; `verdict.json` holds the real one). Writes, under `--out`: `fsdp-check.rank<r>.jsonl`,
`facts.rank<r>.json` (what each rank computed itself), `launcher.log` and `verdict.json`.
Exit status: 0 PASS, 1 FAIL, 2 bad arguments or a missing requirement, 3 NOT PROVEN.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

EXIT_PASS, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_PROVEN = 0, 1, 2, 3
RUN_ID = "fsdp-check"
ARMS = ("fsdp2", "fsdp1")
PORT_TAKEN = ("failed to listen on any local network address", "Address already in use", "EADDRINUSE")


# ---- the arithmetic --------------------------------------------------------------------------------


def combine(values: list[float | None]) -> float | None:
    """The whole norm from each rank's norm over its own shard: the root of the sum of squares. None
    when there is nothing to combine or a rank has no norm."""
    if not values or any(v is None for v in values):
        return None
    return math.sqrt(sum(float(v) * float(v) for v in values))


def combine_rms(pairs: list[tuple[float | None, int]]) -> float | None:
    """The whole parameter's RMS from each rank's `(rms over its shard, the shard's element count)`:
    each square weighted by its count. A rank with an empty shard has nothing to add; one with values
    and no reading makes the whole unknown (None)."""
    total = 0
    acc = 0.0
    for rms, n in pairs:
        if n == 0:
            continue
        if rms is None:
            return None
        acc += float(rms) * float(rms) * n
        total += n
    return math.sqrt(acc / total) if total else None


def choose_param(sizes: list[dict[str, int]], block: str, wanted: str | None = None) -> tuple[str | None, str | None]:
    """The parameter of `block` to plant the gradient spike on, from every rank's shard sizes, and why
    the plant cannot be seen everywhere (None when every rank holds part of it).

    `wanted` (a name inside the block) is taken as named. Otherwise the largest parameter of the block
    that every rank holds part of; under FSDP1 a whole parameter can be one rank's alone, since the
    flattened unit is what is sharded."""
    prefix = block + "."
    names = [n for n in sizes[0] if n.startswith(prefix)]
    if wanted is not None:
        name = prefix + wanted
        if name not in sizes[0]:
            return None, f"no parameter {name} (the block has {', '.join(names) or 'none'})"
        empty = [str(r) for r, s in enumerate(sizes) if s.get(name, 0) == 0]
        return name, (f"its shard is empty on rank {', '.join(empty)}" if empty else None)
    held = [n for n in names if all(s.get(n, 0) > 0 for s in sizes)]
    if not held:
        return None, f"no parameter of {block} has values on every rank"
    return max(held, key=lambda n: sum(s[n] for s in sizes)), None


FSDP_PREFIXES = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.")


def clean(name: str) -> str:
    """A parameter's name without FSDP1's wrapper prefixes. Outside `summon_full_params`, FSDP1's
    `named_parameters` keeps them (`_fsdp_wrapped_module.blocks.0._fsdp_wrapped_module.fc.weight`),
    and those are the names a session started on the wrapped model records."""
    for prefix in FSDP_PREFIXES:
        name = name.replace(prefix, "")
    return name


def cleaned(values: dict | None) -> dict:
    return {clean(k): v for k, v in (values or {}).items()}


def relative(ours: float, theirs: float) -> float:
    return abs(ours - theirs) / max(abs(theirs), 1e-12)


# ---- the model --------------------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return x + nn.functional.gelu(self.fc(self.norm(x)))


class Net(nn.Module):
    def __init__(self, dim: int, blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim) for _ in range(blocks))
        self.head = nn.Linear(dim, dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# ---- one rank -------------------------------------------------------------------------------------


def _shard(model: Net, arm: str, device: torch.device, world: int) -> nn.Module:
    if arm == "fsdp2":
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh(device.type, (world,))  # named, so a Mac's default does not pick MPS
        for block in model.blocks:
            fully_shard(block, mesh=mesh)
        fully_shard(model, mesh=mesh)
        return model
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    return FSDP(model, device_id=device, use_orig_params=True, auto_wrap_policy=ModuleWrapPolicy({Block}))


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


class Whole:
    """The whole parameters, gathered by the script itself: FSDP2 through each DTensor's
    `full_tensor()`, FSDP1 through `summon_full_params`. Collectives: every rank calls them alike."""

    def __init__(self, model: nn.Module, arm: str) -> None:
        self.model, self.arm = model, arm

    @torch.no_grad()
    def grad_norms(self) -> dict[str, float]:
        if self.arm == "fsdp2":
            out = {}
            for name, p in self.model.named_parameters():
                out[name] = float(p.grad.full_tensor().double().norm()) if p.grad is not None else 0.0
            return out
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.summon_full_params(self.model, with_grads=True):
            return {clean(name): float(p.grad.double().norm()) if p.grad is not None else 0.0
                    for name, p in self.model.named_parameters()}

    @torch.no_grad()
    def weights(self) -> dict[str, torch.Tensor]:
        if self.arm == "fsdp2":
            return {name: p.full_tensor().detach().double().cpu() for name, p in self.model.named_parameters()}
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.summon_full_params(self.model):
            return {clean(name): p.detach().double().cpu().clone() for name, p in self.model.named_parameters()}


def _layer_name(model: nn.Module, block: nn.Module) -> str | None:
    """The name TensorQuill gives the planted block: the selected layer that is it, or the FSDP1
    wrapper around it."""
    import tensorquill

    for name, module in tensorquill.capture._select_layers(model, "blocks"):
        if module is block or getattr(module, "_fsdp_wrapped_module", None) is block:
            return name
    return None


def worker(args: argparse.Namespace) -> int:
    import torch.distributed as dist

    import tensorquill

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    out: Path = args.out
    (out / f"pid.rank{rank}").write_text(str(os.getpid()))
    if args.cpu:
        device = torch.device("cpu")
        dist.init_process_group("gloo")
    else:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
    try:
        torch.manual_seed(0)  # the same weights on every rank
        model = Net(args.dim, args.blocks).to(device)
        act_block = model.blocks[args.act_block]
        flag = {"on": False}

        def plant(_module, _inputs, output):
            return output * args.act_factor if flag["on"] else output

        act_block.register_forward_hook(plant)  # before sharding and before TensorQuill's hooks
        model = _shard(model, args.arm, device, world)
        recorded_names = [name for name, _ in model.named_parameters()]  # what the session will record
        names = [clean(name) for name in recorded_names]
        sizes = {clean(name): _local(p).numel() for name, p in model.named_parameters()}
        every_sizes: list[Any] = [None] * world
        dist.all_gather_object(every_sizes, sizes)
        grad_block = f"blocks.{args.grad_block}"
        planted, plant_why = choose_param(every_sizes, grad_block, args.grad_param)
        layer = _layer_name(model, act_block)

        entered = {"n": 0}
        fixed = torch.randn(args.batch, args.seq, args.dim, generator=torch.Generator().manual_seed(7)).to(device)

        def probe(m):
            entered["n"] += 1
            return m(fixed)

        jsonl = out / f"{RUN_ID}.rank{rank}.jsonl"
        session = tensorquill.init(
            model, run_id=RUN_ID, sink=tensorquill.JsonlSink(jsonl), capture="full", every=args.every,
            weights=args.every, updates=True, probe=probe, probe_every=args.probe_every,
            heartbeat_seconds=None, stack_seconds=None,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
        whole = Whole(model, args.arm)
        data = torch.Generator().manual_seed(100 + rank)
        ref_grad, ref_update, ref_weight = [], [], []
        steps_taken = 0
        params = {clean(name): p for name, p in model.named_parameters()}
        for step in range(args.steps):
            flag["on"] = step >= args.act_step
            x = torch.randn(args.batch, args.seq, args.dim, generator=data).to(device)
            y = torch.randn(args.batch, args.seq, args.dim, generator=data).to(device)
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            if step == args.grad_step and planted is not None and params[planted].grad is not None:
                params[planted].grad.mul_(args.grad_factor)  # this rank's shard of it
            ref_grad.append(whole.grad_norms())
            before = whole.weights()
            optimizer.step()
            steps_taken += 1
            after = whole.weights()
            ref_update.append({n: float((after[n] - before[n]).pow(2).mean().sqrt()) for n in names})
            ref_weight.append({n: float(after[n].norm()) for n in names})
            optimizer.zero_grad()
        session.close()
        facts = {
            "rank": rank, "world_size": world, "arm": args.arm, "device": str(device), "torch": torch.__version__,
            "steps_taken": steps_taken, "names": names, "recorded_names": recorded_names, "shard_sizes": sizes, "grad_block": grad_block,
            "planted": planted, "plant_why": plant_why, "act_block": layer, "probe_entered": entered["n"],
            "ref_grad": ref_grad, "ref_update_rms": ref_update, "ref_weight": ref_weight, "closed": True,
        }
        scratch = out / f".facts.rank{rank}.json.tmp"
        scratch.write_text(json.dumps(facts), encoding="utf-8")
        os.replace(scratch, out / f"facts.rank{rank}.json")
        dist.barrier()
    finally:
        dist.destroy_process_group()
    if rank == 0 and not args.worker:  # started by torchrun directly: rank 0 judges
        return judge(args, world)
    return EXIT_PASS


# ---- the judgement ------------------------------------------------------------------------------


def judge(args: argparse.Namespace, world: int) -> int:
    import tensorquill
    from tensorquill.evidence import SPIKE_FACTOR, SPIKE_WINDOW

    out: Path = args.out
    lines: list[str] = []
    failures: list[str] = []
    not_proven: list[str] = []

    def check(ok: bool, text: str) -> bool:
        lines.append(f"{'PASS' if ok else 'FAIL'}: {text}")
        print(lines[-1], flush=True)
        if not ok:
            failures.append(text)
        return ok

    def unproven(text: str) -> None:
        lines.append(f"NOT PROVEN: {text}")
        print(lines[-1], flush=True)
        not_proven.append(text)

    facts: dict[int, dict] = {}
    records: dict[int, list[dict]] = {}
    for r in range(world):
        path = out / f"facts.rank{r}.json"
        capture = out / f"{RUN_ID}.rank{r}.jsonl"
        if path.exists():
            facts[r] = json.loads(path.read_text())
        if capture.exists():
            records[r] = [x for x in tensorquill.load_records(capture) if x.get("record") != "scalars"]
    document: dict[str, Any] = {
        "check": "fsdp", "arm": args.arm, "world_size": world, "steps": args.steps, "act_step": args.act_step,
        "grad_step": args.grad_step, "torch": torch.__version__, "cpu": bool(args.cpu),
        "captures": sorted(p.name for p in out.glob(f"{RUN_ID}.rank*.jsonl")),
        "ranks": {}, "combined": {},
    }
    if not check(sorted(facts) == list(range(world)) and sorted(records) == list(range(world)),
                 f"every rank wrote its facts and its capture (facts {sorted(facts)}, captures {sorted(records)})"):
        return _finish(out, document, lines, failures, not_proven)
    first = facts[0]
    sizes = [facts[r]["shard_sizes"] for r in range(world)]
    names = first["names"]
    planted, block = first["planted"], first["grad_block"]
    document["act_block"] = first["act_block"]
    document["grad_plant"] = {"parameter": planted, "block": block, "why": first["plant_why"],
                              "shard_sizes": [s.get(planted, 0) for s in sizes] if planted else None,
                              "factor": args.grad_factor}
    print(f"{args.arm} on {world} ranks ({first['device']}, torch {first['torch']}); gradient plant: "
          f"{planted} x{args.grad_factor:g} at step {args.grad_step}; activation plant: {first['act_block']} "
          f"x{args.act_factor:g} from step {args.act_step}", flush=True)

    # Steps and identity.
    for r in range(world):
        steps = [x["step"] for x in records[r]]
        check(steps == list(range(args.steps)) and facts[r]["steps_taken"] == args.steps,
              f"rank {r}: {len(steps)} records, steps 0 to {args.steps - 1}, equal to its {facts[r]['steps_taken']} optimizer steps")
        check(all(x.get("rank") == r for x in records[r]) and facts[r]["closed"], f"rank {r}: every record names rank {r}, session closed")

    # The plant itself, in the script's own whole norm.
    totals = [math.sqrt(sum(v * v for v in step.values())) for step in first["ref_grad"]]
    if planted is None:
        unproven(f"no parameter to plant the gradient spike on: {first['plant_why']}")
    elif first["plant_why"]:
        unproven(f"{planted}: {first['plant_why']}, so that rank cannot see the planted gradient")
    history = totals[max(0, args.grad_step - SPIKE_WINDOW):args.grad_step]
    whole_ratio = totals[args.grad_step] / statistics.median(history) if history and statistics.median(history) > 0 else None
    document["grad_plant"]["whole_ratio"] = whole_ratio
    if planted is not None and (whole_ratio is None or whole_ratio < SPIKE_FACTOR):
        unproven(f"the script's own whole gradient norm at step {args.grad_step} is x{whole_ratio} its median before, "
                 f"under the detector's x{SPIKE_FACTOR:g}: no spike for the detector to find")
    elif planted is not None:
        lines.append(f"PLANT: the script's own whole gradient norm at step {args.grad_step} is x{whole_ratio:.1f} its median "
                     f"before (the detector's factor is x{SPIKE_FACTOR:g})")
        print(lines[-1], flush=True)
    # What the session calls the parameters (FSDP1 keeps its wrapper prefixes outside summon_full_params).
    recorded = facts[0]["recorded_names"]
    document["recorded_names"] = {"prefixed": recorded != names, "example": recorded[0] if recorded else None}
    if recorded != names:
        lines.append(f"REPORTED: the capture names parameters as {args.arm}'s named_parameters does outside "
                     f"summon_full_params, wrapper prefixes kept (e.g. {recorded[0]!r}); compared here by the name without them")
        print(lines[-1], flush=True)

    # Per rank: detection, probe, nothing unread.
    for r in range(world):
        evidence = tensorquill.build_evidence(records[r])
        incidents = evidence["incidents"]
        act = next((i for i in incidents if i["kind"] == "activation_spike"), None)
        act_found = None if act is None else {"step": act["step"], "first": act["modules"][0]["module"]}
        check(act_found == {"step": args.act_step, "first": first["act_block"]},
              f"rank {r}: activation spike at step {args.act_step}, first in {first['act_block']} (found {act_found})")
        probe = (records[r][0].get("probe") or {}) if records[r] else {}
        not_run = probe.get("not_run")
        probes_run = [x["probe"] for x in records[r] if "probe" in x and "not_run" not in x["probe"]]
        check(isinstance(not_run, str) and "sharded" in not_run and facts[r]["probe_entered"] == 0 and not probes_run,
              f"rank {r}: the probe is off on a sharded model, said once ({not_run!r}), its callback entered "
              f"{facts[r]['probe_entered']} times, {len(probes_run)} probe passes")
        unread = [x["grads_not_read"] for x in records[r] if "grads_not_read" in x]
        check(not unread, f"rank {r}: no grads_not_read ({unread[:1]})")
        no_update = [x["update_not_read"] for x in records[r] if "update_not_read" in x]
        missing = [x["step"] for x in records[r] if not x.get("update_rms")]
        check(not no_update and not missing,
              f"rank {r}: update_rms on every record, no update_not_read ({no_update[:1]}; missing on steps {missing[:5]})")
        own = [x.get("grad_norm") for x in records[r]]
        before = [v for v in own[max(0, args.grad_step - SPIKE_WINDOW):args.grad_step] if isinstance(v, (int, float))]
        own_ratio = (own[args.grad_step] / statistics.median(before)
                     if before and statistics.median(before) > 0 and isinstance(own[args.grad_step], (int, float)) else None)
        spike = next((i for i in incidents if i["kind"] == "grad_spike"), None)
        listed = [] if spike is None else [clean(p["name"]) for p in spike.get("parameters", [])]
        grad_found = None if spike is None else {"step": spike["step"], "lists_planted": planted in listed}
        if planted is not None and own_ratio is not None and own_ratio >= SPIKE_FACTOR:
            check(grad_found == {"step": args.grad_step, "lists_planted": True},
                  f"rank {r}: its own total at step {args.grad_step} is x{own_ratio:.1f} its median, and its grad_spike "
                  f"incident is there and lists {planted} (found {grad_found})")
        else:
            line = (f"REPORTED: rank {r}: its own total gradient norm at step {args.grad_step} is "
                    f"x{own_ratio if own_ratio is None else round(own_ratio, 2)} its median, under the detector's "
                    f"x{SPIKE_FACTOR:g} on its shard; grad_spike incident: {grad_found}")
            lines.append(line)
            print(line, flush=True)
        document["ranks"][str(r)] = {
            "activation_spike": act_found, "grad_spike": grad_found, "own_ratio_at_grad_step": own_ratio,
            "probe": {"not_run": not_run, "entered": facts[r]["probe_entered"]},
            "shard_sizes": facts[r]["shard_sizes"],
        }

    # The ranks' numbers combined against the script's whole ones.
    by_step = [{x["step"]: x for x in records[r]} for r in range(world)]
    grad_worst, grad_problems, grad_steps = 0.0, [], 0
    update_worst, update_problems = 0.0, []
    weight_worst, weight_problems, weight_checks = 0.0, [], 0
    for step in range(args.steps):
        rows = [by_step[r].get(step) for r in range(world)]
        if any(row is None for row in rows):
            grad_problems.append(f"step {step}: not recorded on every rank")
            continue
        whole_grad = first["ref_grad"][step]
        total = combine([row.get("grad_norm") for row in rows])
        want = math.sqrt(sum(v * v for v in whole_grad.values()))
        if total is None:
            grad_problems.append(f"step {step}: a rank has no grad_norm")
        else:
            grad_steps += 1
            grad_worst = max(grad_worst, relative(total, want))
        for name in names:
            parts = []
            for r, row in enumerate(rows):
                value = cleaned(row.get("grad_norms")).get(name)
                if value is None and sizes[r].get(name, 0) == 0:
                    value = 0.0  # an empty shard has no gradient to read
                parts.append(value)
            got = combine(parts)
            if got is None:
                grad_problems.append(f"step {step}: {name} has no norm on a rank that holds part of it")
            else:
                grad_worst = max(grad_worst, relative(got, whole_grad[name]) if whole_grad[name] > 0 else abs(got))
            rms = combine_rms([(cleaned(row.get("update_rms")).get(name), sizes[r].get(name, 0)) for r, row in enumerate(rows)])
            ref = first["ref_update_rms"][step][name]
            if rms is None:
                update_problems.append(f"step {step}: {name} has no update_rms on a rank that holds part of it")
            elif ref > 0:
                update_worst = max(update_worst, relative(rms, ref))
        if all("weight_norms" in row for row in rows):
            weight_checks += 1
            for name in names:
                parts = []
                for r, row in enumerate(rows):
                    value = cleaned(row["weight_norms"]).get(name)
                    parts.append(0.0 if value is None and sizes[r].get(name, 0) == 0 else value)
                got = combine([None if isinstance(v, str) else v for v in parts])
                if got is None:
                    weight_problems.append(f"step {step}: {name} has no weight norm on a rank that holds part of it")
                else:
                    weight_worst = max(weight_worst, relative(got, first["ref_weight"][step][name]))
    document["combined"] = {
        "grad_norm": {"steps": grad_steps, "worst": grad_worst, "problems": grad_problems[:20]},
        "update_rms": {"worst": update_worst, "problems": update_problems[:20]},
        "weight_norms": {"checks": weight_checks, "worst": weight_worst, "problems": weight_problems[:20]},
    }
    check(grad_steps == args.steps and not grad_problems and grad_worst < args.tolerance,
          f"the ranks' grad_norm and grad_norms combine (root of the sum of squares) to the script's whole norms on "
          f"{grad_steps} of {args.steps} steps, largest relative difference {grad_worst:.2e} (tolerance {args.tolerance:g})")
    for problem in grad_problems[:5]:
        print(f"  {problem}")
    check(not update_problems and update_worst < args.update_tolerance,
          f"the ranks' update_rms, weighted by shard size, combine to the script's own update RMS (weights before and "
          f"after the step), largest relative difference {update_worst:.2e} (tolerance {args.update_tolerance:g})")
    for problem in update_problems[:5]:
        print(f"  {problem}")
    check(weight_checks > 0 and not weight_problems and weight_worst < args.tolerance,
          f"the ranks' weight_norms combine to the script's whole weight norms on {weight_checks} weight checks, "
          f"largest relative difference {weight_worst:.2e}")
    return _finish(out, document, lines, failures, not_proven)


def _finish(out: Path, document: dict, lines: list[str], failures: list[str], not_proven: list[str]) -> int:
    status = "FAIL" if failures else "NOT PROVEN" if not_proven else "PASS"
    code = {"FAIL": EXIT_FAIL, "NOT PROVEN": EXIT_NOT_PROVEN, "PASS": EXIT_PASS}[status]
    document.update(status=status, exit=code, passed=status == "PASS", failures=failures, not_proven=not_proven, lines=lines)
    scratch = out / ".verdict.json.tmp"
    scratch.write_text(json.dumps(document, indent=1), encoding="utf-8")
    os.replace(scratch, out / "verdict.json")
    print(f"\n{status}: {out / 'verdict.json'}", flush=True)
    return code


# ---- the driver ---------------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _end(pids: list[int]) -> list[int]:
    """SIGTERM, then SIGKILL, to each process group and process; the ones still alive."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            for send in (os.killpg, os.kill):
                try:
                    send(pid, sig)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        time.sleep(1.0)
    alive = []
    for pid in pids:
        try:
            os.kill(pid, 0)
            alive.append(pid)
        except OSError:
            pass
    return alive


def launch(script: Path, args: argparse.Namespace, forwarded: list[str]) -> tuple[int | None, list[int], str]:
    """Start `script --worker <forwarded>` on `args.nproc` ranks under `torch.distributed.run`, wait at most
    `args.timeout` seconds, and end whatever is left (each rank writes `pid.rank<r>` under `args.out`).
    Returns torchrun's exit status (None when it timed out), the pids still alive after SIGKILL, and the log."""
    out: Path = args.out
    # torchrun's own parser takes `--nproc` (an abbreviation of its `--nproc-per-node`) even after the
    # script's name, so the driver's own options stay with the driver.
    kept, skip = [], False
    for item in forwarded:
        if skip:
            skip = False
        elif item == "--nproc":
            skip = True
        elif not item.startswith("--nproc="):
            kept.append(item)
    forwarded = kept
    for attempt in range(3):
        out.mkdir(parents=True, exist_ok=attempt > 0)
        command = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(args.nproc),
                   "--master-addr", "127.0.0.1", "--master-port", str(_free_port()),
                   str(Path(script).resolve()), "--worker", *forwarded]
        env = dict(os.environ)
        if args.cpu:
            env.setdefault("OMP_NUM_THREADS", "1")
        log_path = out / "launcher.log"
        with log_path.open("w") as log:
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            launcher = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = launcher.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                status = None
        pids = [launcher.pid] + [int(p.read_text()) for p in out.glob("pid.rank*") if p.read_text().strip().isdigit()]
        left = _end(pids) if status is None else []
        text = log_path.read_text(errors="replace")
        if status not in (None, 0) and any(m in text for m in PORT_TAKEN) and not list(out.glob("facts.rank*.json")):
            continue  # another process took the port between choosing and listening: this harness's race
        break
    print(f"ranks ended: {'exit ' + str(status) if status is not None else f'timed out after {args.timeout:g} s'}"
          + (f"; still alive after SIGKILL: {left}" if left else ""), flush=True)
    if status not in (0, None):
        print(text[-3000:], flush=True)
    return status, left, text


def drive(args: argparse.Namespace, forwarded: list[str], script: Path | None = None, judge_fn=None) -> int:
    """The ranks started and ended by `launch`, then judged from their files (`judge_fn`, this file's
    `judge` by default); a job that had to be ended is a FAIL whatever the files say."""
    status, left, _ = launch(script or Path(__file__), args, forwarded)
    out: Path = args.out
    code = (judge_fn or judge)(args, args.nproc)
    if status is None or left:
        verdict = json.loads((out / "verdict.json").read_text())
        verdict["failures"].append(f"the ranks did not end within {args.timeout:g} s" + (f"; left running: {left}" if left else ""))
        verdict.update(status="FAIL", exit=EXIT_FAIL, passed=False)
        (out / "verdict.json").write_text(json.dumps(verdict, indent=1), encoding="utf-8")
        return EXIT_FAIL
    return code


# ---- arguments ------------------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="new directory for the run")
    p.add_argument("--arm", choices=ARMS, default="fsdp2")
    p.add_argument("--nproc", type=int, default=2, help="ranks the driver starts (one per GPU)")
    p.add_argument("--cpu", action="store_true", help="Gloo ranks on the CPU, for checking the script")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--every", type=int, default=10, help="steps between transfers, and between weight checks")
    p.add_argument("--probe-every", type=int, default=10)
    p.add_argument("--grad-step", type=int, default=25)
    p.add_argument("--grad-block", type=int, default=1)
    p.add_argument("--grad-param", help="the parameter inside the block to plant on (default: chosen from the shard sizes)")
    p.add_argument("--grad-factor", type=float, default=100.0)
    p.add_argument("--act-step", type=int, default=40)
    p.add_argument("--act-block", type=int, default=2)
    p.add_argument("--act-factor", type=float, default=50.0)
    p.add_argument("--tolerance", type=float, default=1e-4, help="relative, for gradient and weight norms")
    p.add_argument("--update-tolerance", type=float, default=1e-3,
                   help="relative, for the update RMS (the script's is a difference of two whole weights)")
    p.add_argument("--timeout", type=float, default=900.0, help="seconds the driver waits for the ranks")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def _usage(p: argparse.ArgumentParser, text: str) -> int:
    p.print_usage(sys.stderr)
    print(f"{Path(__file__).name}: error: {text}", file=sys.stderr)
    return EXIT_USAGE


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(argv)
    from tensorquill.evidence import ACT_MIN_HISTORY, SPIKE_MIN_HISTORY

    if args.grad_step < SPIKE_MIN_HISTORY or args.act_step < ACT_MIN_HISTORY:
        return _usage(p, f"--grad-step and --act-step need {max(SPIKE_MIN_HISTORY, ACT_MIN_HISTORY)} steps of history before them")
    if not args.grad_step < args.act_step:
        return _usage(p, "--grad-step must be before --act-step (the activation spike raises the gradients too)")
    if not args.act_step < args.steps:
        return _usage(p, "--act-step must be before --steps")
    if not (0 <= args.grad_block < args.blocks and 0 <= args.act_block < args.blocks):
        return _usage(p, "--grad-block and --act-block must name blocks below --blocks")
    if args.worker or "WORLD_SIZE" in os.environ:
        return worker(args)
    if args.nproc < 2:
        return _usage(p, "--nproc must be 2 or more: one rank holds the whole of every parameter")
    if not args.cpu and torch.cuda.device_count() < args.nproc:
        return _usage(p, f"{args.nproc} CUDA GPUs are needed ({torch.cuda.device_count()} found); use --cpu for Gloo ranks on the CPU")
    if args.cpu and args.arm == "fsdp2":
        stream = getattr(getattr(torch, "cpu", None), "Stream", None)
        if stream is None or not hasattr(stream, "record_event"):
            return _usage(p, f"FSDP2 on the CPU needs PyTorch 2.7 or later (torch.cpu.Stream.record_event); this is {torch.__version__}")
    if args.out.exists() and any(args.out.iterdir()):
        return _usage(p, f"{args.out} is not empty; choose a new --out")
    return drive(args, argv)


if __name__ == "__main__":
    sys.exit(main())
