"""Create the bake-off's synthetic fixtures with the real TensorQuill pipeline.

Each fixture is a list of step records in the shape `tensorquill.init` writes. They go
through the SDK's own writer (`JsonlSink`), reader (`load_records`), detectors
(`build_evidence`) and exporter (`export_rerun`), so a viewer shows what a captured run
would show. Nothing here duplicates a detector rule, a color or a table.

`seeded` is the visible smoke-test fixture, with the answer key in the README. `unseen-1`
to `unseen-6` are the trial's fixtures: MoE and dense, with other incident steps, layers
and experts. Their names say nothing about the failure.

Run with an environment that has the SDK and rerun-sdk: `pip install -e 'sdk[rerun]'`.
"""

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import tensorquill
from tensorquill.capture import SCHEMA_VERSION, STATS, _router_stats
from tensorquill.evidence import resolve_attempts, write_evidence

# `attempt` is the time `tensorquill.init` ran, in milliseconds; `ts` is each record's time.
START_MS = 1_700_000_000_000
SECONDS_PER_STEP = 0.5


@dataclass(frozen=True)
class Event:
    """An MoE failure that starts at `step` in `layer` and lasts until `until` (or the end).

    `hot` maps an expert to its share of the router's assignments, `dead` lists experts that
    receive none. With `flagged=False` it is a decoy that the detector must leave alone.
    """

    step: int
    layer: int
    hot: dict[int, float] = field(default_factory=dict)
    dead: tuple[int, ...] = ()
    until: int | None = None
    flagged: bool = True

    def active(self, step: int, layer: int) -> bool:
        return layer == self.layer and step >= self.step and (self.until is None or step <= self.until)


@dataclass(frozen=True)
class Fixture:
    """One synthetic run: its step records, and the failure planted in it.

    `answer` is what the detectors must report, written out here and never read back from
    their output: the incident `kind` and `step`, plus the layer and expert IDs, or the
    parameter or module. `events` lists the MoE samples that carry a planted failure, so a
    check can compare every flagged sample, not only the first.
    """

    name: str
    description: str
    answer: dict
    records: Callable[[], list[dict]]
    steps: tuple[int, ...]
    events: tuple[Event, ...] = ()
    top_k: int = 1
    flagged_exactly: bool = True

    def flagged_samples(self) -> set[tuple[int, int]]:
        """The (step, layer) pairs a correct detector flags, from the planted events alone."""
        return {
            (step, event.layer)
            for event in self.events if event.flagged
            for step in self.steps if event.active(step, event.layer)
        }


def base_record(run_id: str, step: int) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "rank": 0,
        "attempt": START_MS,
        "step": step,
        "ts": START_MS / 1000 + step * SECONDS_PER_STEP,
        "layers": {},
        "grad_norm": None,
        "grad_norms": {},
    }


def distribute(total: int, weights: list[float]) -> list[int]:
    """Whole numbers in proportion to `weights` that add up to `total` (largest remainder)."""
    scale = sum(weights)
    exact = [total * weight / scale for weight in weights]
    counts = [int(value) for value in exact]
    by_remainder = sorted(range(len(counts)), key=lambda i: (counts[i] - exact[i], i))
    for i in by_remainder[: total - sum(counts)]:
        counts[i] += 1
    return counts


# --- MoE fixtures ----------------------------------------------------------------------


def router_name(layer: int) -> str:
    # The evidence numbers routers from 1 in the order they were captured; the name carries
    # the same number so that "layer 31" means one thing everywhere in this experiment.
    return f"layers.{layer:02}.router"


def routing_counts(
    fixture: str, step: int, layer: int, *, experts: int, tokens: int, top_k: int,
    events: tuple[Event, ...], pinned: dict[int, float], skew: float, noise: float,
) -> list[int]:
    """One router's tokens-per-expert counts at one step, summing to `tokens * top_k`.

    Healthy experts share the assignments in proportion to a per-layer weight (spread by
    `skew`), plus sampling noise of `noise` times a multinomial's. `pinned` experts hold a
    fixed share all the time, as the always-picked experts of a trained model do.
    """
    total = tokens * top_k
    shares = dict(pinned)
    dead: set[int] = set()
    for event in events:
        if event.active(step, layer):
            shares.update(event.hot)
            dead.update(event.dead)
    fixed = {expert: round(share * total) for expert, share in shares.items()}
    free = [expert for expert in range(1, experts + 1) if expert not in fixed and expert not in dead]
    remaining = total - sum(fixed.values())
    layer_rng = random.Random(f"{fixture}/layer/{layer}")
    base = [math.exp(skew * layer_rng.gauss(0, 1)) for _ in range(experts)]
    scale = sum(base[expert - 1] for expert in free)
    step_rng = random.Random(f"{fixture}/step/{step}/layer/{layer}")
    weights = [0.0] * experts
    for expert in free:
        expected = remaining * base[expert - 1] / scale
        jitter = noise * step_rng.gauss(0, 1) / math.sqrt(expected)
        weights[expert - 1] = base[expert - 1] * max(0.0, 1.0 + jitter)
    counts = distribute(remaining, weights)
    for expert, count in fixed.items():
        counts[expert - 1] = count
    assert sum(counts) == total and max(counts) <= tokens
    return counts


def moe_records(
    name: str, *, steps: tuple[int, ...], layers: int, experts: int, tokens: int,
    top_k: int = 1, events: tuple[Event, ...] = (), pinned: dict[int, float] | None = None,
    skew: float = 0.15, noise: float = 1.0,
) -> Callable[[], list[dict]]:
    def build() -> list[dict]:
        records = []
        for step in steps:
            counts = {
                router_name(layer): routing_counts(
                    name, step, layer, experts=experts, tokens=tokens, top_k=top_k,
                    events=events, pinned=pinned or {}, skew=skew, noise=noise,
                )
                for layer in range(1, layers + 1)
            }
            records.append({
                **base_record(name, step),
                "moe": counts,
                "moe_top_k": {router: top_k for router in counts},
                "moe_stats": {router: _router_stats(row) for router, row in counts.items()},
            })
        return records

    return build


# --- Dense fixtures --------------------------------------------------------------------

BLOCKS = 12
BASE_GRAD_NORM = {
    "embed.weight": 0.5, "attn.qkv.weight": 0.30, "attn.proj.weight": 0.15,
    "mlp.fc1.weight": 0.25, "mlp.fc2.weight": 0.20, "head.weight": 0.4,
}
ACTIVATION_BLOCKS = 20
OUTLIER_RATIO = 11.0  # a block's largest activation over its standard deviation, when healthy
MASSIVE_RATIO = 800.0  # what a trained LLM's massive-activation block carries all the time
MASSIVE_BLOCK = 1


def dense_param_names() -> list[str]:
    kinds = ("attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight")
    return ["embed.weight", *(f"blocks.{b}.{kind}" for b in range(BLOCKS) for kind in kinds), "head.weight"]


def dense_records(
    name: str, *, steps: int, grad_factor: Callable[[int, str], float] = lambda step, param: 1.0,
    act_factor: Callable[[int, int], float] = lambda step, block: 1.0,
    loss_bump: Callable[[int], float] = lambda step: 0.0,
) -> Callable[[], list[dict]]:
    """A dense transformer with `capture="full"`: activations, gradient norms, lr and loss."""

    def build() -> list[dict]:
        params = dense_param_names()
        records = []
        for step in range(steps):
            rng = random.Random(f"{name}/step/{step}")
            decay = 1.0 - 0.3 * step / steps
            grad_norms = {
                param: BASE_GRAD_NORM[param.split(".", 2)[-1] if param.startswith("blocks.") else param]
                * decay * math.exp(0.06 * rng.gauss(0, 1)) * grad_factor(step, param)
                for param in params
            }
            layers = {}
            for block in range(ACTIVATION_BLOCKS):
                std = (0.4 + 0.04 * block) * math.exp(0.04 * rng.gauss(0, 1))
                ratio = MASSIVE_RATIO if block == MASSIVE_BLOCK else OUTLIER_RATIO
                abs_max = std * ratio * math.exp(0.04 * rng.gauss(0, 1)) * act_factor(step, block)
                zero_frac = max(0.0, 0.03 + 0.002 * block + 0.003 * rng.gauss(0, 1))
                layers[f"blocks.{block}"] = dict(zip(STATS, (0.01 * rng.gauss(0, 1), std, abs_max, zero_frac)))
            lr = 3e-4 * min(1.0, (step + 1) / 20) * 0.5 * (1.0 + math.cos(math.pi * step / steps))
            loss = 1.8 + 1.5 * math.exp(-step / 80) + 0.02 * rng.gauss(0, 1) + loss_bump(step)
            records.append({
                **base_record(name, step),
                "layers": layers,
                "grad_norm": math.sqrt(sum(norm * norm for norm in grad_norms.values())),
                "grad_norms": grad_norms,
                "lr": [lr],
                "scalars": {"loss": loss},
            })
        return records

    return build


# --- The fixtures ----------------------------------------------------------------------

DEAD_BLOCK = (3, 9, 14, 21, 22, 30, 37, 41, 44, 50, 52, 58, 61, 63, 67, 70, 73, 78, 80, 84,
              88, 91, 95, 96, 99, 102, 105, 108, 110, 113, 115, 117, 119, 121, 123, 124,
              125, 126, 127, 128)
SPIKE_PARAM = "blocks.7.mlp.fc2.weight"
SPIKE_FACTORS = {308: 2.8, 309: 3.4, 310: 80.0, 311: 4.0}  # a short build-up, then the spike
OUTLIER_BLOCK = 13
OUTLIER_STEPS = range(260, 264)  # four steps: the outlier fades before it looks like drift
LOSS_BUMPS = {310: 1.4, 311: 0.8, 312: 0.4, 313: 0.2}

SEEDED_STEPS = (1750, 1800, 1850, 1900)
SEEDED_EVENTS = (Event(step=1850, layer=31, hot={42: 0.36}, dead=tuple(range(1, 15))),)
HOT_STEPS = tuple(range(1000, 1800, 100))
HOT_EVENTS = (
    Event(step=1300, layer=17, hot={53: 0.44}, dead=tuple(range(20, 28))),
    Event(step=1500, layer=9, hot={5: 0.40}),  # later, in an earlier layer: not the first
)
DEAD_STEPS = tuple(range(1600, 2500, 100))
DEAD_EVENTS = (Event(step=2000, layer=8, dead=DEAD_BLOCK),)
TOPK_STEPS = tuple(range(0, 620, 20))
TOPK_EVENTS = (Event(step=400, layer=11, hot={expert: 0.36 / 4 for expert in range(9, 15)}),)
TWO_STEPS = tuple(range(300, 1000, 100))
TWO_EVENTS = (
    Event(step=500, layer=9, hot={13: 0.34}, until=600, flagged=False),  # just under 35%
    Event(step=700, layer=3, hot={7: 0.39, 22: 0.36}),
)

FIXTURES = (
    Fixture(
        "seeded",
        "MoE, 60 layers x 128 experts. From step 1850 layer 31 sends 36% of tokens to expert 42"
        " and none to experts 1-14.",
        {"kind": "moe_imbalance", "step": 1850, "layer": 31, "overloaded": [42], "dead": list(range(1, 15))},
        moe_records("seeded", steps=SEEDED_STEPS, layers=60, experts=128, tokens=128_000,
                    events=SEEDED_EVENTS, skew=0.0, noise=0.0),
        SEEDED_STEPS, SEEDED_EVENTS,
    ),
    Fixture(
        "unseen-1",
        "MoE, 48 layers x 64 experts. From step 1300 layer 17 sends 44% of tokens to expert 53 and"
        " none to experts 20-27. From step 1500 layer 9 sends 40% to expert 5 (later, so not first).",
        {"kind": "moe_imbalance", "step": 1300, "layer": 17, "overloaded": [53], "dead": list(range(20, 28))},
        moe_records("unseen-1", steps=HOT_STEPS, layers=48, experts=64, tokens=65_536, events=HOT_EVENTS),
        HOT_STEPS, HOT_EVENTS,
    ),
    Fixture(
        "unseen-2",
        "Dense, 12 blocks. At step 310 the gradient of blocks.7.mlp.fc2.weight jumps 80x after two"
        " steps of build-up (rising since 308); the loss jumps with it.",
        {"kind": "grad_spike", "step": 310, "parameter": SPIKE_PARAM, "rising_since": 308},
        dense_records("unseen-2", steps=400, loss_bump=lambda step: LOSS_BUMPS.get(step, 0.0),
                      grad_factor=lambda step, param: SPIKE_FACTORS.get(step, 1.0) if param == SPIKE_PARAM else 1.0),
        tuple(range(400)),
    ),
    Fixture(
        "unseen-3",
        "MoE, top-4 routing, 24 layers x 64 experts. At step 400 layer 11 concentrates on experts 9-14,"
        " each picked by 36% of tokens. Expert 3 is picked by 70% of tokens in every layer, always.",
        {"kind": "moe_imbalance", "step": 400, "layer": 11, "overloaded": list(range(9, 15)), "dead": []},
        moe_records("unseen-3", steps=TOPK_STEPS, layers=24, experts=64, tokens=32_768, top_k=4,
                    events=TOPK_EVENTS, pinned={3: 0.7 / 4}),
        TOPK_STEPS, TOPK_EVENTS, top_k=4, flagged_exactly=False,
    ),
    Fixture(
        "unseen-4",
        "MoE, 32 layers x 128 experts. From step 2000 layer 8 sends nothing to 40 experts, and no"
        " expert is overloaded.",
        {"kind": "moe_imbalance", "step": 2000, "layer": 8, "overloaded": [], "dead": list(DEAD_BLOCK)},
        moe_records("unseen-4", steps=DEAD_STEPS, layers=32, experts=128, tokens=98_304, events=DEAD_EVENTS),
        DEAD_STEPS, DEAD_EVENTS,
    ),
    Fixture(
        "unseen-5",
        "Dense, 20 monitored blocks. For four steps from 260 the largest activation of blocks.13 is"
        " 100x normal. blocks.1 carries massive activations all along, which is healthy.",
        {"kind": "activation_spike", "step": 260, "module": f"blocks.{OUTLIER_BLOCK}", "stat": "abs_max"},
        dense_records("unseen-5", steps=400,
                      act_factor=lambda step, block: 100.0 if block == OUTLIER_BLOCK and step in OUTLIER_STEPS else 1.0),
        tuple(range(400)),
    ),
    Fixture(
        "unseen-6",
        "MoE, 24 layers x 32 experts. At step 700 layer 3 sends 39% of tokens to expert 7 and 36% to"
        " expert 22. At steps 500-600 layer 9 sends 34% to expert 13, just under the 35% rule.",
        {"kind": "moe_imbalance", "step": 700, "layer": 3, "overloaded": [7, 22], "dead": []},
        moe_records("unseen-6", steps=TWO_STEPS, layers=24, experts=32, tokens=32_768, events=TWO_EVENTS),
        TWO_STEPS, TWO_EVENTS,
    ),
)
BY_NAME = {fixture.name: fixture for fixture in FIXTURES}


# --- Export through the pipeline -------------------------------------------------------


def export_fixture(fixture: Fixture, directory: Path, *, rerun: bool = True) -> dict:
    """Write `<name>.jsonl`, `<name>.evidence.json` and `<name>.rrd` into `directory`.

    The steps are the ones `python -m tensorquill <name>.jsonl -o ... --rerun ...` runs.
    """
    directory.mkdir(parents=True, exist_ok=True)
    jsonl = directory / f"{fixture.name}.jsonl"
    jsonl.unlink(missing_ok=True)
    sink = tensorquill.JsonlSink(jsonl)
    sink.write(fixture.records())
    sink.close()
    records = tensorquill.load_records(jsonl, all_attempts=True)
    evidence = tensorquill.build_evidence(records)
    write_evidence(evidence, directory / f"{fixture.name}.evidence.json")
    if rerun:
        tensorquill.export_rerun(evidence, directory / f"{fixture.name}.rrd", records=resolve_attempts(records))
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, nargs="?", default=Path(__file__).parent / "artifacts")
    parser.add_argument("--fixture", action="append", choices=sorted(BY_NAME), help="only this fixture (repeatable)")
    parser.add_argument("--no-rerun", action="store_true", help="skip the .rrd recording")
    args = parser.parse_args()
    chosen = [BY_NAME[name] for name in args.fixture] if args.fixture else list(FIXTURES)
    for fixture in chosen:
        evidence = export_fixture(fixture, args.output / fixture.name, rerun=not args.no_rerun)
        if fixture.name == "seeded":
            # The Rune inspector bakes this path in at build time (src/lib.rs).
            write_evidence(evidence, args.output / "stats.json")
        first = tensorquill.incident_summary(evidence).splitlines()[0]
        print(f"{fixture.name}: {first}")
    # The scorer's key stays outside the fixture folders that participants and agents get.
    keys = {fixture.name: {"description": fixture.description, "answer": fixture.answer} for fixture in FIXTURES}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "answer-key.json").write_text(json.dumps(keys, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
