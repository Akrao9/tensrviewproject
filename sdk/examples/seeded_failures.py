"""Gate 2's two seeded failures that no GPU run had planted yet: a gradient spike and stalled weights.

Trains a dense Hugging Face model on WikiText-2 with full capture, plants one failure in ONE named
parameter (a middle block's `self_attn.k_proj.weight` unless `--param` or `--layer` say otherwise),
and fails unless the real capture and `build_evidence` report it in time, on that parameter:

- `grad_spike`: from `--seed-step` the gradient of that parameter is multiplied by `--factor` with
  a tensor hook, so no other gradient changes. PASS when the `grad_spike` incident starts at the
  seeded step or within one flush interval (50 steps) after it and lists the seeded parameter.
  The incident is raised only when the TOTAL gradient norm reaches 5 times its median
  (`SPIKE_FACTOR`), and the parameters are listed after that. One parameter times 100 gets there
  only if it holds about 5% of the total norm; a k_proj holds a fraction of a percent (0.2% in
  the toy), so x100 is invisible to the detector. Without `--factor` the script measures the
  parameter's share of the total on the first batches and uses x100 or the factor that takes
  the total to twice the threshold, whichever is larger. `--factor 100` shows the blind spot.
  The detector is not changed.
- `stalled_weights`: from `--freeze-step` that parameter's learning rate is 0 (its own param group,
  still in the optimizer, gradient finite), so its weights stop changing. The detector needs the
  float64 norm to be exactly unchanged on 3 weight checks in a row (checks fall at steps 49, 99,
  149, ...). PASS when the incident starts at the first step those 3 checks could show it or within
  one check interval (50 steps) after, and lists the seeded parameter.

Each case prints a PASS or FAIL line, with the numbers behind it, and the exit status is 1 if either
fails.

    python examples/seeded_failures.py --case both --out runs/seeded

`--toy` uses a tiny random Llama on random tokens, for checking the script on CPU. Weights are
bfloat16 on a GPU, as in `dense_outlier_check.py`. With a learning rate of 1e-5 some bfloat16
weights (RMSNorm's, near 1.0) never change, so `stalled_weights` reports those from the third weight
check on, before a freeze at step 100 can show. The detector builds ONE incident, at the first step
any parameter is stalled, and lists at most 20 in model order, so the seeded parameter is not named:
the capture has it (`weights_unchanged`) and the incident does not. `--dtype float32` trains
without that background and is the control; `--toy --dtype bfloat16` reproduces it on CPU.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

import tensorquill
from tensorquill.capture import WEIGHT_EVERY
from tensorquill.evidence import SPIKE_FACTOR, SPIKE_MIN_HISTORY, SPIKE_WINDOW, STALL_STEPS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_outlier_check import FLUSH_EVERY, MODEL, build_models, load_batches, train_step  # noqa: E402

CASES = ("grad_spike", "stalled_weights")
DEFAULT_PARAMETER = "self_attn.k_proj.weight"  # suffix; the block is the middle one unless --layer says
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


# ---- the two faults ---------------------------------------------------------------------


class GradientScale:
    """Multiplies one parameter's gradient by `factor` once `on` is set, with a tensor hook, so
    the gradients of every other parameter are untouched."""

    def __init__(self, parameter: torch.nn.Parameter, factor: float) -> None:
        self.factor = factor
        self.on = False
        self.handle = parameter.register_hook(self)

    def __call__(self, grad: torch.Tensor) -> torch.Tensor | None:
        return grad * self.factor if self.on else None

    def remove(self) -> None:
        self.handle.remove()


def make_optimizer(model: torch.nn.Module, seeded: torch.nn.Parameter, lr: float, device) -> tuple[torch.optim.Optimizer, dict]:
    """AdamW with the seeded parameter in a param group of its own, so its learning rate can be
    set to 0 without touching the others. Returns the optimizer and that group."""
    others = [p for p in model.parameters() if p is not seeded]
    optimizer = torch.optim.AdamW(
        [{"params": others}, {"params": [seeded]}], lr=lr, fused=device.type == "cuda"
    )
    return optimizer, optimizer.param_groups[1]


def seeded_parameter_name(model: torch.nn.Module, args) -> str:
    """The seeded parameter's full name: `--param`, or the middle block's (or `--layer`'s) k_proj."""
    names = {name for name, _ in model.named_parameters()}
    if args.param:
        if args.param not in names:
            raise SystemExit(f"--param {args.param!r} is not a parameter of the model")
        return args.param
    blocks = [name for name, _ in tensorquill.capture._select_layers(model, "blocks")]
    layer = args.layer if args.layer is not None else len(blocks) // 2
    if not 0 <= layer < len(blocks):
        raise SystemExit(f"--layer must be 0 to {len(blocks) - 1}")
    name = f"{blocks[layer]}.{DEFAULT_PARAMETER}"
    if name not in names:
        raise SystemExit(f"{name} is not a parameter of this model; name one with --param")
    return name


def gradient_share(model: torch.nn.Module, name: str, batches: torch.Tensor, count: int = 4) -> float:
    """The smallest share of the total gradient norm `name` holds over the first `count` batches,
    from plain forward and backward passes before any training (the weights hardly move in the
    first few hundred steps at this learning rate). The smallest, so that a factor chosen from it
    is not too weak on any of them."""
    shares = []
    for batch in batches[:count]:
        model.zero_grad(set_to_none=True)
        model(input_ids=batch, labels=batch, use_cache=False).loss.backward()
        norms = {n: float(p.grad.detach().float().norm()) for n, p in model.named_parameters() if p.grad is not None}
        total = math.sqrt(sum(v * v for v in norms.values()))
        shares.append(norms[name] / total if total > 0 else 0.0)
    model.zero_grad(set_to_none=True)
    return min(shares)


def total_ratio(factor: float, share: float) -> float:
    """What multiplying one gradient, holding `share` of the total norm, by `factor` does to the total."""
    return math.sqrt(1 + (factor * factor - 1) * share * share)


def factor_for(share: float, target: float = 2 * SPIKE_FACTOR) -> float:
    """The factor that takes the total to `target` times its value (twice the detector's threshold
    by default, so a bad batch or two does not decide the result), rounded up to two digits."""
    if share <= 0:
        return math.inf
    exact = math.sqrt(1 + (target * target - 1) / (share * share))
    magnitude = 10 ** (math.floor(math.log10(exact)) - 1)
    return math.ceil(exact / magnitude) * magnitude


# ---- when each case may first be seen ---------------------------------------------------


def first_visible_stall(freeze_step: int, check_every: int = WEIGHT_EVERY, consecutive: int = STALL_STEPS) -> int:
    """The step of the first weight check at which `consecutive` checks in a row could show a
    parameter frozen from `freeze_step`, the first step whose update it does not take.

    Weight checks fall on steps `check_every - 1`, `2 * check_every - 1`, ... A check reads the
    norm after the step's update, and it is unchanged only if the previous check is already past
    the last update that landed (step `freeze_step - 1`).
    """
    first = (-(-freeze_step // check_every) + 1) * check_every - 1  # the first check with c - check_every >= freeze_step - 1
    return first + (consecutive - 1) * check_every


# ---- the verdicts -----------------------------------------------------------------------


@dataclass
class Verdict:
    case: str
    passed: bool
    summary: str
    notes: list[str] = field(default_factory=list)


def gradient_facts(records: list[dict[str, Any]], seeded: str, seed_step: int) -> dict[str, Any]:
    """What the capture saw of the total gradient norm and the seeded parameter's share of it.

    The share is the median over the `SPIKE_WINDOW` steps before the seed; `predicted_ratio` is
    what multiplying that parameter's gradient by `factor` does to the total (given the share).
    """
    key = "clip_grad_norm" if any("clip_grad_norm" in r for r in records) else "grad_norm"
    by_step = {r["step"]: r for r in records if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])}
    before = [by_step[s] for s in range(max(0, seed_step - SPIKE_WINDOW), seed_step) if s in by_step]
    facts: dict[str, Any] = {"key": key}
    if not before or seed_step not in by_step:
        return facts
    shares = [r["grad_norms"][seeded] / r[key] for r in before if r[key] > 0 and seeded in r.get("grad_norms", {})]
    facts["baseline"] = statistics.median(r[key] for r in before)
    facts["share"] = statistics.median(shares) if shares else None
    facts["at_seed"] = by_step[seed_step][key]
    facts["ratio_at_seed"] = facts["at_seed"] / facts["baseline"] if facts["baseline"] > 0 else None
    return facts


def judge_grad_spike(
    evidence: dict[str, Any], records: list[dict[str, Any]], seeded: str, seed_step: int, factor: float,
    window: int = FLUSH_EVERY,
) -> Verdict:
    """PASS when a `grad_spike` incident starts at `seed_step` or up to `window` steps after it,
    and its listed parameters include `seeded` (its rank among them is printed)."""
    facts = gradient_facts(records, seeded, seed_step)
    notes = []
    if "share" in facts and facts["share"] is not None:
        # Multiplying one gradient by `factor` takes the total from G to sqrt(G^2 + (factor^2 - 1) g^2).
        predicted = math.sqrt(1 + (factor * factor - 1) * facts["share"] ** 2)
        notes.append(
            f"{seeded} is {100 * facts['share']:.2f}% of the total gradient norm (median over the "
            f"{SPIKE_WINDOW} steps before the seed): x{factor:g} on it predicts a total of "
            f"x{predicted:.2f}; the total at step {seed_step} was x{facts['ratio_at_seed']:.2f} its median "
            f"({facts['at_seed']:.4g} against {facts['baseline']:.4g}); the detector fires at x{SPIKE_FACTOR:g}"
        )
    incident = next((i for i in evidence["incidents"] if i["kind"] == "grad_spike"), None)
    if incident is None:
        return Verdict("grad_spike", False, f"no grad_spike incident (gradient of {seeded} x{factor:g} from step {seed_step})", notes)
    names = [p["name"] for p in incident["parameters"]]
    where = f"step {incident['step']} (ratio {incident['ratio']:.1f}), seeded from step {seed_step}"
    notes.append(f"incident lists {len(names)} parameter(s): {names}")
    if len(incident.get("spike_steps", [])) > 1:
        notes.append(f"{incident['spike_count']} spike steps in all, first {incident['spike_steps'][:6]}")
    if not seed_step <= incident["step"] <= seed_step + window:
        return Verdict("grad_spike", False, f"grad_spike starts at {where}, outside {seed_step} to {seed_step + window}", notes)
    if seeded not in names:
        return Verdict("grad_spike", False, f"grad_spike at {where} does not list {seeded}", notes)
    rank = names.index(seeded) + 1
    return Verdict(
        "grad_spike", True,
        f"grad_spike at {where}, lists {seeded} (rank {rank} of {len(names)} listed, ratio "
        f"{incident['parameters'][rank - 1]['ratio']:.1f} on its own median; gradient x{factor:g})", notes,
    )


def unchanged_streak_start(records: list[dict[str, Any]], seeded: str, consecutive: int = STALL_STEPS) -> int | None:
    """The first step at which `seeded` had been exactly unchanged on `consecutive` weight checks
    in a row with a finite, non-zero gradient: what the capture saw, whatever the incident says."""
    streak = 0
    for record in sorted(records, key=lambda r: r["step"]):
        unchanged = record.get("weights_unchanged")
        if unchanged is None:
            continue
        grad = record.get("grad_norms", {}).get(seeded)
        if isinstance(grad, (int, float)) and math.isfinite(grad) and grad > 0 and seeded in unchanged:
            streak += 1
            if streak >= consecutive:
                return int(record["step"])
        else:
            streak = 0
    return None


def judge_stalled_weights(
    evidence: dict[str, Any], records: list[dict[str, Any]], seeded: str, freeze_step: int,
    window: int = WEIGHT_EVERY,
) -> Verdict:
    """PASS when a `stalled_weights` incident starts at the first step 3 weight checks could show
    the freeze, or up to `window` steps after, and its listed parameters include `seeded`.

    Other parameters may be listed too (real bfloat16 runs always stall some RMSNorm weights) and
    are printed. The detector builds ONE incident, at the first step any parameter is stalled,
    and lists at most 20 of them in model order, so a freeze that comes after, or sits past the
    twentieth, is not named even though the capture saw it. That is reported, not worked around.
    """
    expected = first_visible_stall(freeze_step)
    notes = [f"weights frozen from step {freeze_step}; 3 checks in a row could first show it at step {expected}"]
    seen = unchanged_streak_start(records, seeded)
    notes.append(
        f"the capture's own records show {seeded} unchanged on 3 checks in a row at step {seen}"
        if seen is not None else f"the capture's records never show {seeded} unchanged on 3 checks in a row"
    )
    incident = next((i for i in evidence["incidents"] if i["kind"] == "stalled_weights"), None)
    if incident is None:
        return Verdict("stalled_weights", False, f"no stalled_weights incident ({seeded} frozen from step {freeze_step})", notes)
    names = list(incident["parameters"])
    extras = [n for n in names if n != seeded]
    notes.append(
        f"incident at step {incident['step']} lists {len(names)} of {incident['stalled']} stalled parameter(s)"
        f" ({incident['with_gradients']} with a gradient)"
    )
    if extras:
        notes.append(f"besides the seeded one ({len(extras)}): {extras}")
    optimizer = incident.get("optimizer", {})
    notes.append(
        f"optimizer evidence: {optimizer.get('state')}, {optimizer.get('not_held_count', 0)} stalled parameter(s) not held"
        if optimizer.get("state") == "known" else f"optimizer evidence: {optimizer}"
    )
    where = f"step {incident['step']}"
    if seeded not in names:
        why = ""
        if incident["step"] < expected:
            why = f": it began at step {incident['step']}, before the freeze could show, with other parameters"
        elif incident["stalled"] > len(names):
            why = f": {incident['stalled'] - len(names)} stalled parameters are not listed (the incident keeps 20)"
        return Verdict("stalled_weights", False, f"stalled_weights at {where} does not list {seeded}{why}", notes)
    if not expected <= incident["step"] <= expected + window:
        return Verdict(
            "stalled_weights", False,
            f"stalled_weights at {where} lists {seeded}, outside {expected} to {expected + window}", notes,
        )
    return Verdict(
        "stalled_weights", True,
        f"stalled_weights at {where} (first possible {expected}) lists {seeded}"
        + (f", with {len(extras)} other parameter(s)" if extras else ""), notes,
    )


# ---- running a case ---------------------------------------------------------------------


def steps_for(case: str, args) -> int:
    """Enough steps for the failure to be seen and one more interval to spare, in whole intervals."""
    needed = args.seed_step if case == "grad_spike" else first_visible_stall(args.freeze_step)
    return (needed // FLUSH_EVERY + 2) * FLUSH_EVERY


def make_batches(args, device, total: int) -> torch.Tensor:
    if args.toy:
        # Its own generator, so the tokens do not depend on the order of other random draws.
        return torch.randint(0, 128, (total, 2, 32), generator=torch.Generator().manual_seed(0)).to(device)
    return load_batches(args.model, args.batch_size, args.seq_len, total, device)


def run_case(case: str, args, device, batches: torch.Tensor, out: Path) -> Verdict:
    """Train with full capture, plant the failure, and judge what `build_evidence` reports."""
    steps = args.steps or steps_for(case, args)
    models, optimizers = build_models(args, device)  # two identical models and their optimizers; one is enough
    model = models[0]
    del models, optimizers
    dtype = DTYPES[args.dtype] if args.dtype else next(model.parameters()).dtype
    model.to(dtype)
    name = seeded_parameter_name(model, args)
    seeded = dict(model.named_parameters())[name]
    optimizer, group = make_optimizer(model, seeded, args.lr, device)
    factor = args.factor
    if case == "grad_spike" and factor is None:
        share = gradient_share(model, name, batches)
        factor = max(100.0, factor_for(share))
        print(f"{name} holds {100 * share:.2f}% of the total gradient norm: x100 on it moves the total x{total_ratio(100, share):.2f}"
              f" (the detector fires at x{SPIKE_FACTOR:g}); using x{factor:g}, which moves it x{total_ratio(factor, share):.1f}")
    scale = GradientScale(seeded, factor) if case == "grad_spike" else None
    jsonl = out / f"{case}.jsonl"
    out.mkdir(parents=True)
    if case == "grad_spike":
        print(f"Seeding {name} ({dtype}): gradient x{factor:g} from step {args.seed_step}, {steps} steps", flush=True)
    else:
        print(f"Seeding {name} ({dtype}): learning rate 0 from step {args.freeze_step}, {steps} steps", flush=True)
    try:
        with tensorquill.init(
            model, run_id=f"seeded-{case}", capture="full", every=FLUSH_EVERY, weights=WEIGHT_EVERY,
            sink=tensorquill.JsonlSink(jsonl),
        ) as session:
            assert name in session.param_names, f"{name} is not captured"
            for step in range(steps):
                if scale is not None and step == args.seed_step:
                    scale.on = True
                if case == "stalled_weights" and step == args.freeze_step:
                    group["lr"] = 0.0
                train_step(model, optimizer, batches[step])
    finally:
        if scale is not None:
            scale.remove()

    records = tensorquill.load_records(jsonl)
    evidence = tensorquill.build_evidence(records)
    print(f"\n--- Detection ({case}) ---")
    print(tensorquill.incident_summary(evidence))
    if case == "grad_spike":
        verdict = judge_grad_spike(evidence, records, name, args.seed_step, factor)
    else:
        verdict = judge_stalled_weights(evidence, records, name, args.freeze_step)
    print()
    for note in verdict.notes:
        print(f"  {note}")
    print(f"{'PASS' if verdict.passed else 'FAIL'}: {verdict.summary}")
    if not verdict.passed and case == "stalled_weights" and dtype == torch.bfloat16:
        print("  bfloat16 weights near 1.0 do not move at this learning rate, so other parameters stall first; "
              "--dtype float32 leaves them out")
    print(f"Capture: {jsonl}\n")
    return verdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=[*CASES, "both"], default="both")
    parser.add_argument("--out", type=Path, required=True, help="new directory; each case writes a folder in it")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dtype", choices=list(DTYPES), help="weight dtype (default: bfloat16 on a GPU, float32 for --toy)")
    parser.add_argument("--layer", type=int, help="block to seed (default: the middle one)")
    parser.add_argument("--param", help=f"full name of the parameter to seed (default: the block's {DEFAULT_PARAMETER})")
    parser.add_argument("--seed-step", type=int, default=250, help="grad_spike: first step with the scaled gradient")
    parser.add_argument("--factor", type=float,
                        help="grad_spike: gradient multiplier (default: 100, or more if 100 on this parameter cannot "
                             "take the total gradient norm to twice the detector's threshold)")
    parser.add_argument("--freeze-step", type=int, default=100, help="stalled_weights: first step without an update")
    parser.add_argument("--steps", type=int, help="steps per case (default: enough for the failure and one interval more)")
    parser.add_argument("--toy", action="store_true", help="tiny random Llama on random tokens, CPU allowed")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed_step < SPIKE_MIN_HISTORY or args.freeze_step < 1:
        parser.error(f"--seed-step must be at least {SPIKE_MIN_HISTORY} (the detector needs history) and --freeze-step at least 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.toy and device.type != "cuda":
        parser.error("a CUDA GPU is required; use --toy to check the script on CPU")
    cases = CASES if args.case == "both" else (args.case,)
    for case in cases:
        need = args.seed_step + 1 if case == "grad_spike" else first_visible_stall(args.freeze_step) + 1
        if args.steps and args.steps < need:
            parser.error(f"--steps {args.steps} ends before {case} can be seen; it needs at least {need}")

    args.out.mkdir(parents=True)
    total = max(args.steps or steps_for(case, args) for case in cases)
    batches = make_batches(args, device, total)
    name = torch.cuda.get_device_name() if device.type == "cuda" else "CPU"
    print(f"{torch.__version__} on {name} | {'toy Llama' if args.toy else args.model} | eager\n", flush=True)

    verdicts = [run_case(case, args, device, batches, args.out / case) for case in cases]
    failed = [v.case for v in verdicts if not v.passed]
    if failed:
        print(f"{len(failed)} case(s) failed: {', '.join(failed)}")
        sys.exit(1)
    print("All cases passed")


if __name__ == "__main__":
    main()
