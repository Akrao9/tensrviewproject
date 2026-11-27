"""Megatron-core under tensor parallelism with sequence parallelism: TensorQuill's router counts on
each rank against Megatron's own `routing_map`, and the `replica_params` filter. The multi-GPU test
plan's T9, its tensor-parallel part (pipeline and expert parallelism wait for the author's Q5).

A new script beside `megatron_routing_check.py` (Megatron's `TopKRouter` alone, in a one-process
group, with capacity dropping), which is left byte for byte as it was: that check's arms are about
dropping and need one process; this one needs `torchrun`, a model-parallel group of `--tp` ranks
and a layer whose input is already split along the sequence. It reuses that script's Megatron
import and its stand-in router.

Every rank builds a model of a LayerNorm and Megatron's `MoELayer` (local experts,
`moe_token_dispatcher_type="allgather"`, `sequence_parallel=True`, `tensor_model_parallel_size=--tp`,
no capacity factor), as Megatron's own `tests/unit_tests/transformer/moe/test_moe_layer.py` builds
one. Each step the ranks of a tensor-parallel group make the same global batch `[--seq, --batch,
hidden]` and each takes its slice of the sequence (`--seq / --tp` positions), as sequence
parallelism hands a layer its input; the router therefore routes only that rank's tokens. After the
backward, the gradients of the parameters Megatron marks `sequence_parallel` (the router's weight,
and the norm, marked here as Megatron marks its own) are summed over the group, as
`finalize_model_grads` does under sequence parallelism, so those weights stay the same on every
rank of the group, while the experts' weights (marked `tensor_model_parallel`) are each rank's shard.

Two sessions in turn (`--arms`), each with `routers={"moe.router": tensorquill.Router(E, top_k)}`
and `session.record_replica_spread(tp_group)` on every step (the explicit gather, a collective):

- `filtered`: `replica_params=lambda name, p: not getattr(p, "tensor_model_parallel", False)`, the
  filter the README gives for Megatron. The session must compare exactly the parameters without
  that mark, and no rank's evidence may hold `replica_drift`;
- `unfiltered`: no filter. The experts' shards differ across the group, so `replica_drift` must name
  them and only parameters with the mark (the README's documented false drift, shown once).

In both, on every rank and step, TensorQuill's `moe["moe.router"]` must equal the column sums of the
`routing_map` the router returned (read by a forward hook on it), the router must have been given
`--seq / --tp * --batch` tokens (the sequence-parallel slice), and the counts of the group's ranks
must add up to `--seq * --batch * top_k`. A map in another form than `[tokens, experts]` (Megatron
can return dense top-k indices for its Flex backends) is NOT PROVEN.

`--cpu` runs a stand-in on Gloo ranks: `megatron_routing_check.py`'s `StandInTopKRouter` (Megatron's
`TopKRouter` contract) and two linear layers sharded by hand and marked `tensor_model_parallel`. It
checks this script, not Megatron.

    python examples/megatron_tp_check.py --nproc 2 --tp 2 --out runs/megatron-tp2
    python examples/megatron_tp_check.py --nproc 8 --tp 2 --out runs/megatron-tp2-dp4
    python examples/megatron_tp_check.py --nproc 8 --tp 8 --out runs/megatron-tp8
    python examples/megatron_tp_check.py --cpu --nproc 2 --tp 2 --out runs/megatron-tp-cpu

Writes `<arm>.rank<r>.jsonl`, `facts.rank<r>.json`, `launcher.log` and `verdict.json` under `--out`.
Exit status: 0 PASS, 1 FAIL, 2 bad arguments, no CUDA or no importable Megatron (one line says
what), 3 NOT PROVEN.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fsdp_check  # noqa: E402  the driver: launch, bound, end
from megatron_routing_check import DEFAULT_CLONE, MissingDependency, StandInTopKRouter, import_megatron  # noqa: E402

EXIT_PASS, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_PROVEN = 0, 1, 2, 3
ARMS = ("filtered", "unfiltered")
ROUTER = "moe.router"


def REPLICA_FILTER(_name: str, parameter: Any) -> bool:  # noqa: N802  the README's filter, as written there
    return not getattr(parameter, "tensor_model_parallel", False)


# ---- the arithmetic --------------------------------------------------------------------------------


def sequence_slice(seq: int, tp: int, tp_rank: int) -> tuple[int, int]:
    """The positions of the sequence a tensor-parallel rank holds under sequence parallelism."""
    if seq % tp:
        raise ValueError(f"a sequence of {seq} does not split over {tp} ranks")
    size = seq // tp
    return tp_rank * size, (tp_rank + 1) * size


def expected_replicas(named: list[tuple[str, Any]]) -> list[str]:
    """The trainable parameters every rank of a tensor-parallel group holds whole: those without
    Megatron's `tensor_model_parallel` mark, in the model's order."""
    return [name for name, p in named if getattr(p, "requires_grad", True) and REPLICA_FILTER(name, p)]


def count_problems(records: list[dict], spy: list[list[int]], router: str) -> list[str]:
    """Each step where the recorded counts are not the routing map's column sums."""
    by_step = {r["step"]: r for r in records}
    problems = []
    for step, want in enumerate(spy):
        record = by_step.get(step)
        if record is None:
            problems.append(f"step {step}: not recorded")
            continue
        got = (record.get("moe") or {}).get(router)
        if got != want:
            problems.append(f"step {step}: recorded {got}, routing_map sums {want}")
    return problems


def map_form_problem(maps: list[Any], tokens: int, experts: int) -> str | None:
    """Why the router's maps are not the `[tokens, experts]` bool map this check compares, or None."""
    for m in maps:
        if m.dtype != torch.bool or tuple(m.shape) != (tokens, experts):
            if len(m.shape) == 2 and m.shape[1] != experts:
                return (f"the router returned {m.dtype} [{', '.join(map(str, m.shape))}], dense top-k indices, not a "
                        f"[{tokens}, {experts}] routing map")
            return f"the router returned {m.dtype} [{', '.join(map(str, m.shape))}], not a bool [{tokens}, {experts}] map"
    return None


# ---- the models ---------------------------------------------------------------------------------


class MegatronNet(nn.Module):
    def __init__(self, config, submodules) -> None:
        from megatron.core.transformer.moe.moe_layer import MoELayer

        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size)
        for p in self.norm.parameters():
            p.sequence_parallel = True  # as Megatron marks its own norms under sequence parallelism
        self.moe = MoELayer(config, submodules)

    def forward(self, x):
        out, _ = self.moe(self.norm(x))
        return out


class StandInNet(nn.Module):
    """On the CPU: a router with `TopKRouter`'s contract and two linear layers sharded by hand, each
    rank's shard its own values, marked as Megatron marks its tensor-parallel weights."""

    def __init__(self, hidden: int, experts: int, topk: int, ffn: int, tp: int, tp_rank: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.moe = nn.Module()
        self.moe.router = StandInTopKRouter(hidden, experts, topk, None, "probs", False, {})
        for p in list(self.norm.parameters()) + [self.moe.router.weight]:
            p.sequence_parallel = True
        shard = torch.Generator().manual_seed(1000 + tp_rank)
        self.moe.fc1 = nn.Parameter(torch.randn(ffn // tp, hidden, generator=shard) * 0.1)
        self.moe.fc2 = nn.Parameter(torch.randn(hidden, ffn // tp, generator=shard) * 0.1)
        for p in (self.moe.fc1, self.moe.fc2):
            p.tensor_model_parallel = True

    def forward(self, x):
        h = self.norm(x)
        probs, _ = self.moe.router(h)
        y = nn.functional.gelu(h @ self.moe.fc1.t()) @ self.moe.fc2.t()
        return y + probs.sum(-1).view(*h.shape[:2], 1) * h  # the router's probabilities reach the loss


# ---- one rank -------------------------------------------------------------------------------------


def worker(args: argparse.Namespace) -> int:
    import torch.distributed as dist

    import tensorquill

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    out: Path = args.out
    (out / f"pid.rank{rank}").write_text(str(os.getpid()))
    tp = args.tp
    if args.cpu:
        dist.init_process_group("gloo")
        device = torch.device("cpu")
        groups = [dist.new_group(list(range(start, start + tp))) for start in range(0, world, tp)]
        tp_group, tp_rank, dp_rank = groups[rank // tp], rank % tp, rank // tp
        torch.manual_seed(0)
        model: nn.Module = StandInNet(args.hidden, args.experts, args.topk, args.ffn, tp, tp_rank)
        kind = "stand-in (CPU, Gloo)"
    else:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
        import_megatron(args.megatron if args.megatron and args.megatron.is_dir() else None)
        from megatron.core import parallel_state
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.spec_utils import get_submodules
        from megatron.core.transformer.transformer_config import TransformerConfig

        parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp)
        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(0)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_rank, dp_rank = parallel_state.get_tensor_model_parallel_rank(), parallel_state.get_data_parallel_rank()
        config = TransformerConfig(
            num_layers=1, hidden_size=args.hidden, num_attention_heads=4, num_moe_experts=args.experts,
            moe_router_topk=args.topk, moe_router_load_balancing_type="none", moe_aux_loss_coeff=0.0,
            moe_token_dispatcher_type="allgather", moe_grouped_gemm=False, moe_ffn_hidden_size=args.ffn,
            add_bias_linear=False, tensor_model_parallel_size=tp, sequence_parallel=True,
            use_cpu_initialization=True,
        )
        submodules = get_submodules(get_gpt_layer_local_submodules(num_experts=args.experts, moe_grouped_gemm=False).mlp)
        model = MegatronNet(config, submodules).cuda()
        import megatron.core

        kind = f"Megatron-core MoELayer from {Path(megatron.core.__file__).resolve().parent}"
    model.train()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    tensor_parallel = [n for n, p in named if getattr(p, "tensor_model_parallel", False)]
    summed = [p for _, p in named if getattr(p, "sequence_parallel", False)]
    start, end = sequence_slice(args.seq, tp, tp_rank)
    seen: dict[str, Any] = {"counts": [], "tokens": [], "maps": []}

    def spy(_module, inputs, output):
        hidden = inputs[0]
        seen["tokens"].append(int(hidden.numel() // hidden.shape[-1]))
        routing_map = output[1]
        seen["maps"].append([str(routing_map.dtype).removeprefix("torch."), list(routing_map.shape)])
        if routing_map.dtype == torch.bool and routing_map.dim() == 2 and routing_map.shape[1] == args.experts:
            seen["counts"].append(routing_map.detach().sum(0).tolist())
        else:
            seen["counts"].append(None)

    model.get_submodule(ROUTER).register_forward_hook(spy)
    facts: dict[str, Any] = {"rank": rank, "world_size": world, "tp": tp, "tp_rank": tp_rank, "dp_rank": dp_rank,
                             "device": str(device), "torch": torch.__version__, "kind": kind,
                             "tensor_parallel": tensor_parallel, "expected_replicas": expected_replicas(named),
                             "arms": {}}
    step_base = 0
    for arm in args.arms:
        for key in seen:
            seen[key] = []
        jsonl = out / f"{arm}.rank{rank}.jsonl"
        session = tensorquill.init(
            model, run_id=f"megatron-tp-{arm}", sink=tensorquill.JsonlSink(jsonl), capture="routers",
            routers={ROUTER: tensorquill.Router(args.experts, top_k=args.topk)}, every=args.every,
            replica_params=REPLICA_FILTER if arm == "filtered" else None,
            heartbeat_seconds=None, stack_seconds=None,
        )
        replica_names = list(session.replica_names)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        try:
            for step in range(args.steps):
                data = torch.Generator().manual_seed(10_000 * dp_rank + step_base + step)  # one batch per group
                hidden = torch.randn(args.seq, args.batch, args.hidden, generator=data)
                local = hidden[start:end].to(device).requires_grad_(True)
                loss = model(local).float().pow(2).mean()
                loss.backward()
                for p in summed:  # finalize_model_grads under sequence parallelism
                    if p.grad is not None:
                        dist.all_reduce(p.grad, group=tp_group)
                session.record_replica_spread(tp_group)
                optimizer.step()
                optimizer.zero_grad()
        finally:
            session.close()
        step_base += args.steps
        facts["arms"][arm] = {"replica_names": replica_names, "counts": seen["counts"], "tokens": seen["tokens"],
                              "map_forms": [list(f) for f in {(m[0], tuple(m[1])) for m in seen["maps"]}]}
    scratch = out / f".facts.rank{rank}.json.tmp"
    scratch.write_text(json.dumps(facts), encoding="utf-8")
    os.replace(scratch, out / f"facts.rank{rank}.json")
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0 and not args.worker:  # under torchrun directly
        return judge(args, world)
    return EXIT_PASS


# ---- the judgement ------------------------------------------------------------------------------


def judge(args: argparse.Namespace, world: int) -> int:
    import tensorquill

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

    facts = {r: json.loads((out / f"facts.rank{r}.json").read_text()) for r in range(world)
             if (out / f"facts.rank{r}.json").exists()}
    document: dict[str, Any] = {"check": "megatron_tp", "megatron": not args.cpu, "tp": args.tp, "world_size": world,
                                "seq": args.seq, "batch": args.batch, "experts": args.experts, "topk": args.topk,
                                "steps": args.steps, "arms": {}}
    if not check(sorted(facts) == list(range(world)), f"every rank wrote its facts ({sorted(facts)})"):
        return finish(out, document, lines, failures, not_proven)
    document["kind"] = facts[0]["kind"]
    print(f"{facts[0]['kind']} | TP {args.tp} of {world} ranks | {args.experts} experts, top-{args.topk} | "
          f"sequence {args.seq} x batch {args.batch}", flush=True)
    local_tokens = args.seq // args.tp * args.batch
    for arm in args.arms:
        arm_facts = [facts[r]["arms"][arm] for r in range(world)]
        records = {r: [x for x in tensorquill.load_records(out / f"{arm}.rank{r}.jsonl") if x.get("record") != "scalars"]
                   for r in range(world)}
        mismatches = 0
        tokens = [sorted(set(a["tokens"])) for a in arm_facts]
        check(all(t == [local_tokens] for t in tokens),
              f"[{arm}] every rank's router was given {local_tokens} tokens a step, its slice of the sequence ({tokens})")
        form = arm_facts[0]["map_forms"]
        why = map_form_problem([SimpleNamespace(dtype=getattr(torch, f[0]), shape=tuple(f[1]))
                                for a in arm_facts for f in a["map_forms"]], local_tokens, args.experts)
        if why is not None:
            lines.append(f"NOT PROVEN: [{arm}] {why}")
            print(lines[-1], flush=True)
            not_proven.append(why)
        for r in range(world):
            check([x["step"] for x in records[r]] == list(range(args.steps)), f"[{arm}] rank {r}: steps 0 to {args.steps - 1}")
            if why is None:
                problems = count_problems(records[r], arm_facts[r]["counts"], ROUTER)
                mismatches += len(problems)
                check(not problems, f"[{arm}] rank {r}: moe[{ROUTER!r}] equals its routing_map's column sums on "
                                    f"{args.steps - len(problems)} of {args.steps} steps {problems[:2]}")
        sums_off = []
        if why is None:
            for group_start in range(0, world, args.tp):
                for step in range(args.steps):
                    total = sum(sum(arm_facts[r]["counts"][step]) for r in range(group_start, group_start + args.tp))
                    if total != args.seq * args.batch * args.topk:
                        sums_off.append(f"group from rank {group_start}, step {step}: {total}")
            check(not sums_off, f"[{arm}] each tensor-parallel group's counts add up to seq x batch x top_k = "
                                f"{args.seq * args.batch * args.topk} on every step ({sums_off[:2]})")
        names = [a["replica_names"] for a in arm_facts]
        expected = facts[0]["expected_replicas"]
        tensor_parallel = facts[0]["tensor_parallel"]
        drifts = []
        for r in range(world):
            evidence = tensorquill.build_evidence(records[r])
            drift = next((i for i in evidence["incidents"] if i["kind"] == "replica_drift"), None)
            drifts.append(None if drift is None else sorted(drift["parameters"]))
        spreads = [sum(1 for x in records[r] if "replica_spread" in x) for r in range(world)]
        check(all(n == args.steps for n in spreads), f"[{arm}] replica_spread on every step of every rank ({spreads})")
        if arm == "filtered":
            check(all(n == expected for n in names) and not set(expected) & set(tensor_parallel),
                  f"[{arm}] the session compares exactly the {len(expected)} parameters without tensor_model_parallel "
                  f"({expected})")
            check(all(d is None for d in drifts), f"[{arm}] no replica_drift on any rank ({drifts})")
        else:
            check(all(set(tensor_parallel) <= set(n) for n in names),
                  f"[{arm}] without the filter the session compares the sharded weights too")
            check(all(d and set(d) <= set(tensor_parallel) for d in drifts),
                  f"[{arm}] replica_drift on every rank names sharded weights only, the documented false drift ({drifts})")
        document["arms"][arm] = {
            "tokens_per_rank": [t[0] if len(t) == 1 else t for t in tokens], "count_mismatches": mismatches,
            "group_sums_off": sums_off[:10], "replica_names": names[0], "expected_replicas": expected,
            "tensor_parallel": tensor_parallel, "drift": drifts[0], "drift_by_rank": drifts, "map_form": form,
        }
    return finish(out, document, lines, failures, not_proven)


def finish(out: Path, document: dict, lines: list[str], failures: list[str], not_proven: list[str]) -> int:
    status = "FAIL" if failures else "NOT PROVEN" if not_proven else "PASS"
    code = {"FAIL": EXIT_FAIL, "NOT PROVEN": EXIT_NOT_PROVEN, "PASS": EXIT_PASS}[status]
    document.update(status=status, exit=code, passed=status == "PASS", failures=failures, not_proven=not_proven, lines=lines)
    (out / "verdict.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(f"\n{status}: {out / 'verdict.json'}", flush=True)
    return code


# ---- arguments ------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="new directory for the run")
    p.add_argument("--nproc", type=int, default=2, help="ranks the driver starts (one per GPU)")
    p.add_argument("--tp", type=int, default=2, help="tensor-parallel size")
    p.add_argument("--cpu", action="store_true", help="the stand-in on Gloo ranks, for checking the script")
    p.add_argument("--megatron", type=Path, default=DEFAULT_CLONE, help="a Megatron-LM clone, if megatron.core is not installed")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--ffn", type=int, default=128)
    p.add_argument("--seq", type=int, default=32)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    def usage(text: str) -> int:
        p.print_usage(sys.stderr)
        print(f"megatron_tp_check.py: error: {text}", file=sys.stderr)
        return EXIT_USAGE

    if args.tp < 2:
        return usage("--tp must be 2 or more: tensor parallelism over one rank is megatron_routing_check.py")
    if args.worker or "WORLD_SIZE" in os.environ:
        return worker(args)
    if args.nproc % args.tp:
        return usage(f"--nproc ({args.nproc}) must be a multiple of --tp ({args.tp})")
    if args.seq % args.tp:
        return usage(f"--seq ({args.seq}) must split over --tp ({args.tp}) ranks under sequence parallelism")
    if args.ffn % args.tp:
        return usage(f"--ffn ({args.ffn}) must split over --tp ({args.tp}) ranks")
    if not args.cpu:
        if torch.cuda.device_count() < args.nproc:
            return usage(f"Megatron's layers need CUDA: {args.nproc} CUDA GPUs are needed ({torch.cuda.device_count()} "
                         "found); use --cpu for the stand-in")
        try:
            import_megatron(args.megatron if args.megatron and args.megatron.is_dir() else None)
        except MissingDependency as error:
            return usage(str(error))
    if args.out.exists() and any(args.out.iterdir()):
        return usage(f"{args.out} is not empty; choose a new --out")
    return fsdp_check.drive(args, argv, script=Path(__file__), judge_fn=judge)


if __name__ == "__main__":
    sys.exit(main())
