"""Gate 2's two seeded failures that no GPU run had planted yet: a gradient spike and stalled weights.

Trains a dense Hugging Face model on WikiText-2 with full capture, plants one failure in ONE named
parameter (a middle block's `self_attn.k_proj.weight` unless `--param` or `--layer` say otherwise),
and fails unless the real capture and `build_evidence` report it in time, on that parameter:

- `grad_spike`: from `--seed-step` the gradient of that parameter is multiplied by `--factor` with
  a tensor hook, so no other gradient changes. PASS when the `grad_spike` incident starts at the
  seeded step or within one flush interval (50 steps) after it and lists the seeded parameter.
  The detector has two rules. The TOTAL gradient norm at 5 times its median (`SPIKE_FACTOR`):
  one parameter times 100 gets there only if it holds about 5% of the total norm, and a k_proj
  holds a fraction of a percent (0.2% in the toy). And ONE parameter at 5 times its own median
  for 3 recorded steps in a row (`SPIKE_SUSTAIN`), whatever the total does. Without `--factor`
  the script measures the parameter's share of the total on the first batches and uses x100 or
  the factor that takes the total to twice the threshold, whichever is larger, which tests the
  first rule. `--factor 100` leaves the total where it was and tests the second.
- `stalled_weights`: from `--freeze-step` that parameter's learning rate is 0 (its own param group,
  still in the optimizer, gradient finite), so its weights stop changing. The detector needs the
  float64 norm to be exactly unchanged on 3 weight checks in a row (checks fall at steps 49, 99,
  149, ...). PASS when the incident lists the seeded parameter with an onset at the first step
  those 3 checks could show it or within one check interval (50 steps) after.

- `update_spike` (`--case update_spike`, not part of `both` or `all`): the same seeded gradient,
  times `--factor` (10,000 unless given) from `--seed-step`, with `init(..., updates=True)`, which
  records the size of AdamW's update to each parameter from the optimizer's own state. PASS when
  the `update_spike` incident is at the seeded step and the parameter whose update moved most is
  the seeded one. The rule's baseline is the mean of the last `--update-window` step-to-step
  changes that were not flagged (the detector's default, 100), so the seed step must come after them.
  Adam divides by the gradient's own size, so one bad gradient moves a parameter only a few
  learning rates: x100 on one parameter is often under the rule's limit, and x10,000 is not.

Each case prints a PASS or FAIL line, with the numbers behind it, and the exit status is 1 if either
fails.

    python examples/seeded_failures.py --case both --out runs/seeded

`--case healthy` trains the same model on the same batches with nothing planted, and passes when
`build_evidence` lists no incident; `--case all` runs it and then the two seeded cases. It is the
other half of a pair for `python -m tensorquill compare`: each case is a run of the project
`seeded-failures`, named after its case, with what was planted in its `config`:

    python -m tensorquill compare runs/seeded/healthy/healthy.jsonl runs/seeded/grad_spike/grad_spike.jsonl

With bfloat16 weights the healthy run is not clean (see below), so use `--dtype float32` for the pair.

`--toy` uses a tiny random Llama on random tokens, for checking the script on CPU. Weights are
bfloat16 on a GPU, as in `dense_outlier_check.py`. With a learning rate of 1e-5 some bfloat16
weights (RMSNorm's, near 1.0) never change, so `stalled_weights` starts with those, from the third
weight check on, before a freeze at step 100 can show. Each parameter is judged on its own history,
so the incident's `onsets` name the seeded parameter at the step its own freeze shows, as one that
had changed before, beside the earlier onset of weights that never changed. `--dtype float32`
trains without that background; `--toy --dtype bfloat16` reproduces it on CPU.
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
from tensorquill.evidence import SPIKE_FACTOR, SPIKE_MIN_HISTORY, SPIKE_SUSTAIN, SPIKE_WINDOW, STALL_STEPS, UPDATE_WINDOW

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_outlier_check import FLUSH_EVERY, MODEL, build_models, load_batches, train_step  # noqa: E402

CASES = ("grad_spike", "stalled_weights")
UPDATE = "update_spike"  # its own case: `both` and `all` are the two above, as on the GPU runs so far
UPDATE_FACTOR = 1e4
HEALTHY = "healthy"  # nothing planted: the run the seeded ones are compared with
PROJECT = "seeded-failures"
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
    total = f"{incident['ratio']:.1f}" if isinstance(incident.get("ratio"), (int, float)) else "unknown"
    if incident.get("trigger") == "parameter":
        rule = f"the parameter's own norm, {incident['sustained']} steps in a row; total ratio {total}"
    else:
        rule = f"ratio {total}"
    where = f"step {incident['step']} ({rule}), seeded from step {seed_step}"
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


def judge_update_spike(evidence: dict[str, Any], seeded: str, seed_step: int, factor: float) -> Verdict:
    """PASS when the `update_spike` incident is at `seed_step` and the parameter whose own update
    moved most is `seeded`."""
    incident = next((i for i in evidence["incidents"] if i["kind"] == UPDATE), None)
    check = evidence["kind_checks"][UPDATE]
    notes = [f"update_spike: {check['value']}, {check['records']} of {check['records_needed']} records"
             + (f" ({check['reason']})" if check.get("reason") else "")]
    if incident is None:
        return Verdict(UPDATE, False, f"no update_spike incident (gradient of {seeded} x{factor:g} from step {seed_step})", notes)
    names = [p["name"] for p in incident["parameters"]]
    notes.append(
        f"the largest RMS update went {incident['previous']:.3g} to {incident['update_rms']:.3g}: a change of "
        f"{incident['change']:.3g} against a limit of {incident['threshold']:.3g} (usual change {incident['baseline']:.3g}, "
        f"gradient norm up {incident['grad_rise']:.3g}); {incident['spike_count']} step(s) flagged, first {incident['spike_steps'][:6]}"
    )
    where = f"step {incident['step']}, seeded from step {seed_step}"
    if incident["step"] != seed_step:
        return Verdict(UPDATE, False, f"update_spike is at {where}", notes)
    if not names or names[0] != seeded:
        return Verdict(UPDATE, False, f"update_spike at {where} names {names[:1] or 'no parameter'} first, not {seeded}", notes)
    top = incident["parameters"][0]
    return Verdict(
        UPDATE, True,
        f"update_spike at {where}, names {seeded} first (its update went {top['previous']:.3g} to {top['update_rms']:.3g}; gradient x{factor:g})",
        notes,
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
    """PASS when a `stalled_weights` incident lists `seeded` with an onset at the first step 3
    weight checks could show the freeze, or up to `window` steps after.

    The incident is dated at the first step any parameter stalled, and real bfloat16 runs stall
    some RMSNorm weights before a seeded freeze can show. Each parameter is judged on its own
    history, so the seeded one is looked for among the incident's `onsets`, and the step that
    counts is its own onset, not the incident's.
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
    # An incident from before onsets were recorded has one onset: its own step and parameters.
    onsets = incident.get("onsets") or [
        {"step": incident["step"], "parameters": incident["parameters"], "stalled": incident["stalled"]}
    ]
    notes.append(
        f"incident from step {incident['step']}: {incident.get('stalled_total', incident['stalled'])} stalled"
        f" parameter(s) in all ({incident['with_gradients']} with a gradient), onsets at steps "
        f"{[onset['step'] for onset in onsets]}"
    )
    for onset in onsets:
        others = [n for n in onset["parameters"] if n != seeded]
        if others:
            trained = onset.get("trained_before")
            notes.append(
                f"step {onset['step']}: {onset['stalled']} parameter(s)"
                + (f", {trained} of them changed at an earlier check" if trained is not None else "")
                + f"; besides the seeded one ({len(others)} listed): {others}"
            )
    optimizer = incident.get("optimizer", {})
    notes.append(
        f"optimizer evidence: {optimizer.get('state')}, {optimizer.get('not_held_count', 0)} stalled parameter(s) not held"
        if optimizer.get("state") == "known" else f"optimizer evidence: {optimizer}"
    )
    own = next((onset for onset in onsets if seeded in onset["parameters"]), None)
    if own is None:
        why = ""
        unlisted = sum(onset["stalled"] - len(onset["parameters"]) for onset in onsets)
        omitted = incident.get("onsets_omitted", 0)
        if unlisted or omitted:
            why = f": {unlisted} stalled parameter(s) and {omitted} onset(s) are not listed (an onset keeps 20 names)"
        return Verdict("stalled_weights", False, f"stalled_weights from step {incident['step']} does not list {seeded}{why}", notes)
    where = f"step {own['step']}"
    if not expected <= own["step"] <= expected + window:
        return Verdict(
            "stalled_weights", False,
            f"stalled_weights lists {seeded} at {where}, outside {expected} to {expected + window}", notes,
        )
    others = sum(onset["stalled"] for onset in onsets) - 1
    return Verdict(
        "stalled_weights", True,
        f"stalled_weights lists {seeded} at {where} (first possible {expected})"
        + (", after it changed at earlier checks" if own.get("trained_before") else "")
        + (f", with {others} other parameter(s)" if others else ""), notes,
    )


# ---- running a case ---------------------------------------------------------------------


def steps_for(case: str, args) -> int:
    """Enough steps for the failure to be seen and one more interval to spare, in whole intervals.
    The healthy run takes as many as the longer seeded case, so it reaches every step they do."""
    if case == HEALTHY:
        return max(steps_for(seeded, args) for seeded in CASES)
    needed = args.seed_step if case in ("grad_spike", UPDATE) else first_visible_stall(args.freeze_step)
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
              f" (the total rule fires at x{SPIKE_FACTOR:g}); using x{factor:g}, which moves it x{total_ratio(factor, share):.1f}."
              f" --factor 100 tests the one-parameter rule instead")
    if case == UPDATE and factor is None:
        factor = UPDATE_FACTOR
    scale = GradientScale(seeded, factor) if case in ("grad_spike", UPDATE) else None
    jsonl = out / f"{case}.jsonl"
    out.mkdir(parents=True)
    planted: dict[str, Any] = {"case": case, "parameter": name, "dtype": str(dtype), "lr": args.lr, "steps": steps}
    if case in ("grad_spike", UPDATE):
        planted.update(seed_step=args.seed_step, factor=factor)
        print(f"Seeding {name} ({dtype}): gradient x{factor:g} from step {args.seed_step}, {steps} steps", flush=True)
    elif case == HEALTHY:
        print(f"Seeding nothing ({dtype}): {steps} steps", flush=True)
    else:
        planted.update(freeze_step=args.freeze_step)
        print(f"Seeding {name} ({dtype}): learning rate 0 from step {args.freeze_step}, {steps} steps", flush=True)
    try:
        with tensorquill.init(
            model, run_id=f"seeded-{case}", capture="full", every=FLUSH_EVERY, weights=WEIGHT_EVERY,
            sink=tensorquill.JsonlSink(jsonl), project=PROJECT, name=case, config=planted, updates=case == UPDATE,
        ) as session:
            assert name in session.param_names, f"{name} is not captured"
            for step in range(steps):
                if scale is not None and step == args.seed_step:
                    scale.on = True
                if case == "stalled_weights" and step == args.freeze_step:
                    group["lr"] = 0.0
                train_step(model, optimizer, batches[step], session)
    finally:
        if scale is not None:
            scale.remove()

    records = tensorquill.load_records(jsonl)
    evidence = tensorquill.build_evidence(records, **({"update_window": args.update_window} if case == UPDATE else {}))
    print(f"\n--- Detection ({case}) ---")
    print(tensorquill.incident_summary(evidence))
    if case == "grad_spike":
        verdict = judge_grad_spike(evidence, records, name, args.seed_step, factor)
    elif case == UPDATE:
        verdict = judge_update_spike(evidence, name, args.seed_step, factor)
    elif case == HEALTHY:
        kinds = sorted({incident["kind"] for incident in evidence["incidents"]})
        verdict = Verdict(case, not kinds, "no incident, with nothing planted" if not kinds else
                          f"nothing was planted and the evidence lists {', '.join(kinds)}")
    else:
        verdict = judge_stalled_weights(evidence, records, name, args.freeze_step)
    print()
    for note in verdict.notes:
        print(f"  {note}")
    print(f"{'PASS' if verdict.passed else 'FAIL'}: {verdict.summary}")
    print(f"Capture: {jsonl}\n")
    return verdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=[*CASES, UPDATE, "both", HEALTHY, "all"], default="both",
                        help="both: grad_spike and stalled_weights; healthy: nothing planted; all: healthy, then both; "
                             "update_spike: the seeded gradient read as the size of the optimizer's update")
    parser.add_argument("--update-window", type=int, default=UPDATE_WINDOW,
                        help="update_spike: how many of the last step-to-step changes make the rule's baseline; the seed "
                             f"step must come after the first of them (default: the detector's, {UPDATE_WINDOW})")
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
    cases = {"both": CASES, "all": (HEALTHY, *CASES)}.get(args.case, (args.case,))
    for case in cases:
        if case == HEALTHY:
            continue
        # A spike in one parameter is known once it has lasted SPIKE_SUSTAIN recorded steps.
        if case == UPDATE:
            if args.update_window < 1 or args.seed_step < args.update_window + 1:
                parser.error(f"--seed-step {args.seed_step} is inside the baseline: update_spike is first judged at step "
                             f"{args.update_window + 1} with --update-window {args.update_window}")
            need = args.seed_step + 1
        else:
            need = args.seed_step + SPIKE_SUSTAIN if case == "grad_spike" else first_visible_stall(args.freeze_step) + 1
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
