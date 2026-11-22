"""Check TensorQuill's counts of Megatron's routing map against the router's own output, with
capacity dropping off and on.

Megatron-LM's `TopKRouter.forward` returns `(probs, routing_map)`, both `[tokens, experts]`. With
`moe_expert_capacity_factor` set, `TopKRouter.routing` passes its top-k choice through
`moe_utils.apply_router_token_dropping`, which drops the assignments above each expert's capacity
and, with `moe_pad_expert_input_to_capacity`, pads every expert's column up to it. TensorQuill,
given `routers={"router": tensorquill.Router(E, top_k)}`, counts the returned map per expert
(`moe`), leaving out entries whose probability is exactly zero, which it records per expert as
`moe_left_out` (the padding). It records no per-token map and no drop count of its own.

For each step the script keeps what Megatron's own code path computed: the top-k map before
dropping (the `routing_map` argument `apply_router_token_dropping` is called with, read by wrapping
that function in the router module's namespace) and the map and probabilities the router returned.
From those alone it derives, per expert,

    kept    = before & after          (real assignments the router kept)
    padding = after & ~before         (entries pad_to_capacity added)
    dropped = before & ~after         (assignments dropped at capacity)
    tokens_per_expert = after.sum(0)  (what Megatron's token dispatcher sends, token_dispatcher.py:558)

and fails unless TensorQuill's record for the same step has `moe == kept`, `moe_left_out ==
padding` (zeros without padding), `moe + moe_left_out == tokens_per_expert`, and the drop it
implies, `before.sum(0) - moe`, equal to `dropped`. Three arms (`--arms`): `off` (no capacity
factor: nothing may be dropped, and every token's `top_k` assignments must be counted), `on`
(a capacity factor: at least one assignment must be dropped, so the comparison is not vacuous; the
inputs are skewed towards two experts with `--skew` for that) and `on-pad` (the same with padding:
at least one padded entry).

Two routers:

- default: Megatron's own `TopKRouter`, imported from an installed `megatron.core` or from the
  clone named by `--megatron` (default `../../Megatron-LM` beside the project), on CUDA, in a
  one-process model-parallel group. Megatron needs CUDA (`Router.gating` moves the weight to
  `torch.cuda.current_device()`) and imports `triton` from `megatron/core/__init__.py`.
- `--cpu`: a stand-in `TopKRouter` that follows `TopKRouter.routing`'s softmax path (top-k of the
  logits, softmax over the k, scattered into `[tokens, experts]`), and passes its choice through
  Megatron's own `apply_router_token_dropping` and `get_capacity`: their source is read from the
  clone's `moe_utils.py` and executed alone, since the package cannot be imported without triton.
  Without the clone, an inline copy of those two functions is used, and the result says which.

Writes `<arm>.jsonl` (the capture) and `result.json` under `--out`; exits 0 when every comparison
holds, 1 when one does not, 2 when Megatron cannot be imported or CUDA is missing (one line says
what to install).

    python examples/megatron_routing_check.py --out runs/megatron-routing
    python examples/megatron_routing_check.py --cpu --out runs/megatron-routing-cpu
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import nn

import tensorquill

EXIT_PASS, EXIT_FAIL, EXIT_MISSING = 0, 1, 2
# A clone beside the project: the first `Megatron-LM` folder found next to a parent of this file.
DEFAULT_CLONE = next((p / "Megatron-LM" for p in Path(__file__).resolve().parents if (p / "Megatron-LM").is_dir()), None)
ARMS = ("off", "on", "on-pad")
DROPPING_FUNCTIONS = ("get_capacity", "apply_router_token_dropping")


# ---- Megatron's dropping, for the CPU stand-in -------------------------------------------------


def _inline_dropping() -> dict[str, Callable]:
    """A copy of `moe_utils.get_capacity` and `apply_router_token_dropping` at Megatron-LM 4b11b8168,
    for when no clone is at hand."""

    def get_capacity(num_tokens, num_experts, capacity_factor, min_capacity=None):
        capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
        if min_capacity is not None and capacity < min_capacity:
            capacity = min_capacity
        return capacity

    def apply_router_token_dropping(routing_probs, routing_map, router_topk, capacity_factor,
                                    drop_policy="probs", pad_to_capacity=False):
        num_tokens, num_experts = routing_probs.shape
        expert_capacity = get_capacity(num_tokens=num_tokens * router_topk, num_experts=num_experts,
                                       capacity_factor=capacity_factor)
        if expert_capacity > num_tokens:
            capacity_mask = torch.ones_like(routing_probs).bool()
        elif drop_policy == "probs":
            _, keep = torch.topk(routing_probs, k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(routing_probs).scatter(0, keep, 1).bool()
        elif drop_policy == "position":
            _, keep = torch.topk(routing_map.int(), k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(routing_probs).scatter(0, keep, 1).bool()
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")
        final_map = capacity_mask if pad_to_capacity else torch.logical_and(routing_map, capacity_mask)
        return routing_probs * final_map, final_map

    return {"get_capacity": get_capacity, "apply_router_token_dropping": apply_router_token_dropping}


def megatron_dropping(clone: Path | None) -> tuple[dict[str, Callable], str]:
    """Megatron's `get_capacity` and `apply_router_token_dropping`, executed from the clone's
    `megatron/core/transformer/moe/moe_utils.py` with only `torch`, `math` and `typing` in scope
    (their bodies use nothing else), and where they came from. The inline copy when the clone or
    either function is missing, or when they do not execute alone."""
    source = None if clone is None else clone / "megatron" / "core" / "transformer" / "moe" / "moe_utils.py"
    if source is None or not source.is_file():
        return _inline_dropping(), "inline copy of Megatron-LM 4b11b8168 (no clone found)"
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        wanted = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in DROPPING_FUNCTIONS]
        if sorted(node.name for node in wanted) != sorted(DROPPING_FUNCTIONS):
            raise LookupError("not both functions are defined at the top level")
        for node in wanted:
            node.decorator_list = []
        scope: dict[str, Any] = {"torch": torch, "math": math, "Optional": Optional, "Tuple": Tuple,
                                 "Union": Union, "List": List}
        exec(compile(ast.Module(body=wanted, type_ignores=[]), str(source), "exec"), scope)  # noqa: S102
        functions = {name: scope[name] for name in DROPPING_FUNCTIONS}
    except Exception as error:  # noqa: BLE001  the inline copy is the fallback, and the result says so
        return _inline_dropping(), f"inline copy of Megatron-LM 4b11b8168 ({source} unusable: {error})"
    return functions, f"executed from {source}"


class StandInTopKRouter(nn.Module):
    """`TopKRouter`'s contract on CPU: `forward(hidden [s, b, h])` returns `(probs, routing_map)`,
    both `[s * b, experts]`, through the softmax path of `topk_routing_with_score_function`
    (`use_pre_softmax=False`: top-k of the logits, softmax over those k) and, with a capacity factor,
    `apply_router_token_dropping` looked up in `namespace` at each call, as `router.py` does."""

    def __init__(self, hidden: int, experts: int, topk: int, capacity_factor: float | None, drop_policy: str,
                 pad_to_capacity: bool, namespace: dict[str, Callable]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(experts, hidden))
        nn.init.normal_(self.weight, std=0.02)
        self.topk, self.capacity_factor, self.drop_policy, self.pad = topk, capacity_factor, drop_policy, pad_to_capacity
        self.namespace = namespace

    def forward(self, hidden: torch.Tensor):
        logits = (hidden @ self.weight.t()).view(-1, self.weight.shape[0])
        top_logits, top_indices = torch.topk(logits, k=self.topk, dim=1)
        top_probs = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)
        probs = torch.zeros_like(logits).scatter(1, top_indices, top_probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
        if self.capacity_factor is not None:
            probs, routing_map = self.namespace["apply_router_token_dropping"](
                probs, routing_map, router_topk=self.topk, capacity_factor=self.capacity_factor,
                drop_policy=self.drop_policy, pad_to_capacity=self.pad,
            )
        return probs, routing_map


# ---- Megatron's own router, for the GPU machine ------------------------------------------------


class MissingDependency(Exception):
    """What to install, in one line."""


def import_megatron(clone: Path | None):
    """`megatron.core`'s router module, installed or from the clone (put on `sys.path`)."""
    try:
        import megatron.core.transformer.moe.router as router_module  # noqa: F401
    except ImportError:
        if clone is not None and (clone / "megatron" / "core").is_dir() and str(clone) not in sys.path:
            sys.path.insert(0, str(clone))
            for name in [m for m in sys.modules if m == "megatron" or m.startswith("megatron.")]:
                del sys.modules[name]
        try:
            import megatron.core.transformer.moe.router as router_module  # noqa: F811
        except ImportError as error:
            raise MissingDependency(
                f"Megatron-LM's router cannot be imported ({type(error).__name__}: {error}); install megatron-core "
                f"(pip install megatron-core, with triton) or pass --megatron <clone> on a CUDA machine where it imports"
            ) from error
    import megatron.core.transformer.moe.router as router_module
    return router_module


def megatron_router(router_module, hidden: int, experts: int, topk: int, capacity_factor: float | None,
                    drop_policy: str, pad_to_capacity: bool):
    """A `TopKRouter` as Megatron's own `test_routers.py` builds one, in a one-process model-parallel
    group: float32 weights, no bias, no auxiliary loss, plain top-k softmax routing."""
    from megatron.core import parallel_state
    from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
    from megatron.core.transformer.transformer_config import TransformerConfig

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
        torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(1, 1)
    config = TransformerConfig(
        num_layers=1, hidden_size=hidden, num_attention_heads=4, num_moe_experts=experts,
        moe_router_topk=topk, moe_router_load_balancing_type="none", moe_aux_loss_coeff=0.0,
        moe_expert_capacity_factor=capacity_factor, moe_token_drop_policy=drop_policy,
        moe_pad_expert_input_to_capacity=pad_to_capacity, add_bias_linear=False, use_cpu_initialization=True,
    )
    return router_module.TopKRouter(config, pg_collection=get_default_pg_collection()).cuda()


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---- the run ------------------------------------------------------------------------------------


class Layer(nn.Module):
    """The router as an MoE layer holds it, so its name is `router`."""

    def __init__(self, router: nn.Module) -> None:
        super().__init__()
        self.router = router

    def forward(self, hidden):
        return self.router(hidden)


class DroppingSpy:
    """Wraps `apply_router_token_dropping` in a namespace (Megatron's `router` module, or the stand-in's
    dict) and keeps the routing map each call was given: the router's top-k choice before dropping."""

    def __init__(self, namespace: Any) -> None:
        self.namespace, self.before = namespace, []
        get = namespace.get if isinstance(namespace, dict) else lambda n: getattr(namespace, n)
        self.original = get("apply_router_token_dropping")

        def spy(routing_probs, routing_map, *args, **kwargs):
            self.before.append(routing_map.detach().clone())
            return self.original(routing_probs, routing_map, *args, **kwargs)

        self._set(spy)

    def _set(self, value) -> None:
        if isinstance(self.namespace, dict):
            self.namespace["apply_router_token_dropping"] = value
        else:
            setattr(self.namespace, "apply_router_token_dropping", value)

    def restore(self) -> None:
        self._set(self.original)


def skewed_weight(experts: int, hidden: int, generator: torch.Generator) -> torch.Tensor:
    """Router weights whose first two rows share a direction `skewed_inputs` adds to every token, so
    experts 0 and 1 are chosen far above their share and a capacity factor of 1 drops assignments."""
    weight = torch.randn(experts, hidden, generator=generator) * 0.1
    direction = torch.zeros(hidden)
    direction[0] = 1.0
    weight[0] += direction
    weight[1] += 0.8 * direction
    return weight


def run_arm(arm: str, args, out: Path, make_router: Callable, spy_namespace: Any, device: torch.device) -> dict:
    capacity = None if arm == "off" else args.capacity_factor
    router = make_router(capacity, arm == "on-pad")
    generator = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        router.weight.copy_(skewed_weight(args.experts, args.hidden, generator).to(router.weight))
    model = Layer(router).to(device).train()
    spy = DroppingSpy(spy_namespace)
    jsonl = out / f"{arm}.jsonl"
    returned: list[tuple[torch.Tensor, torch.Tensor]] = []
    try:
        with tensorquill.init(
            model, run_id=f"megatron-routing-{arm}", sink=tensorquill.JsonlSink(jsonl), capture="routers",
            routers={"router": tensorquill.Router(args.experts, top_k=args.topk)}, every=1, auto_step=False,
        ) as session:
            for step in range(args.steps):
                hidden = torch.randn(args.seq, args.batch, args.hidden, generator=generator)
                hidden[..., 0] += args.skew  # towards experts 0 and 1 (`skewed_weight`)
                hidden = hidden.to(device).requires_grad_(True)
                probs, routing_map = model(hidden)
                returned.append((probs.detach().clone(), routing_map.detach().clone()))
                probs.sum().backward()  # a training forward and backward, as in a real step
                session.step(step)
    finally:
        spy.restore()
    records = tensorquill.load_records(jsonl)
    return judge(arm, args, records, returned, spy.before, capacity)


def expected_counts(before: torch.Tensor, after: torch.Tensor) -> dict[str, list[int]]:
    """Per expert, from the router's own maps alone (see the module docstring)."""
    before, after = before.bool().cpu(), after.bool().cpu()
    return {
        "chosen": before.sum(0).tolist(),
        "kept": (before & after).sum(0).tolist(),
        "padding": (after & ~before).sum(0).tolist(),
        "dropped": (before & ~after).sum(0).tolist(),
        "tokens_per_expert": after.sum(0).tolist(),
    }


def judge(arm: str, args, records: list[dict], returned: list, before_maps: list, capacity) -> dict:
    failures: list[str] = []
    lines: list[str] = []

    def check(ok: bool, text: str) -> None:
        lines.append(f"{'PASS' if ok else 'FAIL'}: [{arm}] {text}")
        print(lines[-1])
        if not ok:
            failures.append(text)

    tokens = args.seq * args.batch
    by_step = {r["step"]: r for r in records}
    check(sorted(by_step) == list(range(args.steps)), f"{len(records)} records, steps 0 to {args.steps - 1}")
    if capacity is None:
        check(not before_maps, f"the router never called apply_router_token_dropping ({len(before_maps)} calls)")
        before_maps = [m for _, m in returned]  # nothing dropped: the choice is what the router returned
    else:
        check(len(before_maps) == args.steps, f"apply_router_token_dropping called once a step ({len(before_maps)} calls)")
    form = [type(m).__name__ + ":" + str(m.dtype) for _, m in returned[:1]]
    check(all(m.dtype == torch.bool and m.shape == (tokens, args.experts) for _, m in returned),
          f"the router returned a bool routing map [{tokens}, {args.experts}] every step ({form})")
    source = (records[0].get("moe_source") or {}).get("router") if records else None
    print(f"  moe_source: {source}")
    totals = {"chosen": 0, "kept": 0, "dropped": 0, "padding": 0}
    per_step = []
    for step, ((probs, after), before) in enumerate(zip(returned, before_maps)):
        want = expected_counts(before, after)
        # A kept assignment of probability exactly 0 would be left out by TensorQuill's rule: say so.
        zero_kept = int(((before & after).cpu() & (probs.cpu() == 0)).sum())
        record = by_step.get(step, {})
        moe = (record.get("moe") or {}).get("router")
        left_out = (record.get("moe_left_out") or {}).get("router")
        implied_drop = None if moe is None else [c - m for c, m in zip(want["chosen"], moe)]
        problems = []
        if zero_kept:
            problems.append(f"{zero_kept} kept assignments have probability 0 (the expectation would not hold)")
        if moe != want["kept"]:
            problems.append(f"moe {moe} != kept {want['kept']}")
        if left_out != want["padding"]:
            problems.append(f"moe_left_out {left_out} != padding {want['padding']}")
        if moe is not None and left_out is not None and [a + b for a, b in zip(moe, left_out)] != want["tokens_per_expert"]:
            problems.append(f"moe + moe_left_out != tokens_per_expert {want['tokens_per_expert']}")
        if implied_drop != want["dropped"]:
            problems.append(f"chosen - moe {implied_drop} != dropped {want['dropped']}")
        if problems:
            failures.append(f"step {step}: " + "; ".join(problems))
            print(f"FAIL: [{arm}] step {step}: " + "; ".join(problems))
        for key in totals:
            totals[key] += sum(want[key])
        per_step.append({"step": step, "recorded": {"moe": moe, "moe_left_out": left_out}, "router": want})
    matched = args.steps - sum(1 for f in failures if f.startswith("step "))
    check(matched == args.steps, f"TensorQuill's moe and moe_left_out equal the router's kept and padded entries, and "
                                 f"chosen - moe equals its dropped assignments, per expert, on {matched} of {args.steps} steps")
    print(f"  router's own totals over {args.steps} steps: chosen {totals['chosen']}, kept {totals['kept']}, "
          f"dropped {totals['dropped']}, padding {totals['padding']}")
    check(totals["chosen"] == args.steps * tokens * args.topk,
          f"the router chose top_k={args.topk} experts for each of {tokens} tokens every step")
    if arm == "off":
        check(totals["dropped"] == 0 and totals["padding"] == 0, "dropping off: nothing dropped, nothing padded")
    else:
        check(totals["dropped"] > 0, f"dropping on: {totals['dropped']} assignments dropped (the comparison is not vacuous)")
        if arm == "on-pad":
            check(totals["padding"] > 0, f"pad_to_capacity: {totals['padding']} padded entries")
        else:
            check(totals["padding"] == 0, "no padding without pad_to_capacity")
    return {"arm": arm, "capacity_factor": capacity, "pad_to_capacity": arm == "on-pad", "passed": not failures,
            "failures": failures, "totals": totals, "moe_source": source, "steps": per_step, "lines": lines}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="new directory for the run")
    parser.add_argument("--cpu", action="store_true", help="the stand-in router with Megatron's dropping, on CPU")
    parser.add_argument("--megatron", type=Path, default=DEFAULT_CLONE, help="a Megatron-LM clone (default: the first Megatron-LM folder beside a parent of this script, %(default)s)")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--capacity-factor", type=float, default=1.0)
    parser.add_argument("--drop-policy", choices=["probs", "position"], default="probs")
    parser.add_argument("--skew", type=float, default=1.0, help="added to every token's first channel")
    args = parser.parse_args(argv)

    clone = args.megatron if args.megatron and args.megatron.is_dir() else None
    if args.cpu:
        device = torch.device("cpu")
        namespace, origin = megatron_dropping(clone)
        router_kind = f"stand-in TopKRouter (CPU) with Megatron's apply_router_token_dropping {origin}"

        def make_router(capacity, pad):
            return StandInTopKRouter(args.hidden, args.experts, args.topk, capacity, args.drop_policy, pad, namespace)

        spy_namespace: Any = namespace
    else:
        if not torch.cuda.is_available():
            print("megatron_routing_check.py: Megatron's TopKRouter needs CUDA; use --cpu for the stand-in")
            return EXIT_MISSING
        try:
            router_module = import_megatron(clone)
        except MissingDependency as error:
            print(f"megatron_routing_check.py: {error}")
            return EXIT_MISSING
        device = torch.device("cuda")
        router_kind = f"Megatron-LM TopKRouter from {Path(router_module.__file__).resolve()}"

        def make_router(capacity, pad):
            return megatron_router(router_module, args.hidden, args.experts, args.topk, capacity, args.drop_policy, pad)

        spy_namespace = router_module

    args.out.mkdir(parents=True)
    print(f"{torch.__version__} | {router_kind} | {args.experts} experts, top-{args.topk}, {args.seq}x{args.batch} "
          f"tokens, capacity factor {args.capacity_factor:g} ({args.drop_policy}) | arms {args.arms}", flush=True)
    results = []
    for arm in args.arms:
        print(f"\n--- {arm} ---")
        results.append(run_arm(arm, args, args.out, make_router, spy_namespace, device))
    passed = all(r["passed"] for r in results)
    document = {"check": "megatron_routing", "router": router_kind, "megatron_router": not args.cpu,
                "passed": passed, "arms": results, "torch": torch.__version__}
    (args.out / "result.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(f"\nResult: {args.out / 'result.json'}")
    print("All checks passed" if passed else f"{sum(len(r['failures']) for r in results)} check(s) failed")
    return EXIT_PASS if passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
