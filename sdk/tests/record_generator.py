"""Generated runs of step records, for tests that give the same input to both detector engines.

Standard library only, seeded and deterministic: `generate(seed)` gives the same `Case` on every
machine and interpreter (it draws from `random.Random(seed)` alone and never iterates a set). A case is
one run as `tensorquill.init` writes it: one `.jsonl` file per rank (`<run_id>.rank<rank>.jsonl`),
whose lines are the case's text. Python reads the records back from that text, as a reader does.

A case has a base and at most two malformations:

- the base is a healthy run, or a run with one planted fault of a kind the detectors look for
  (`FAULTS`); a fault that needs history is planted after enough steps for its rule
  (`evidence.records_needed`);
- a malformation (`MALFORMATIONS`) is a value TensorQuill's own capture does not write, put on a few
  records: a non-finite number in the sink's spelling or in another writer's (`NaN`, `Infinity`,
  `"NaN"`), an integer too large for a float or for 64 bits, a key left out, a value of the wrong
  type, an empty or a very long name, steps out of order or written twice, a negative router count.
  Most of them land on what the detectors read (maps, routers, layers, probes, the system block); a
  small share on the identity fields, which both engines refuse alike before judging.

`case.files()` is the text of each rank's file; `write(case, folder)` writes them;
`write_corpus(folder, seeds)` writes many cases' files and a `cases.json` index, for a reader in
another language (the Rust converter of step 2 of the plan is to be checked against such a corpus).

`shrink(records, still)` makes a failing input smaller while `still(records)` holds: it drops
records, then keys of the records that remain, then entries of their maps.

To reproduce a case: `generate(seed)`; to see it: `python tests/record_generator.py <seed>`, which
prints its description and its lines.
"""

from __future__ import annotations

import copy
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

# A change to how cases are drawn changes what every seed means: the hashes pinned in the tests then
# fail, and the tables of seeds that disagree have to be made again.
GENERATOR_VERSION = 1

RUN_PREFIX = "gen"
FAULTS = (
    "grad_spike", "non_finite", "update_spike", "stalled_weights", "loss_scale_trouble", "repeated_steps",
    "moe_imbalance", "starved_experts", "activation_spike", "dead_activations", "attention_collapse",
    "attention_logit_growth", "resume_lr_change", "data_stalled", "replica_drift", "memory_growth",
    "memory_pressure", "allocator_retries", "hardware_errors", "gpu_throttled", "input_starvation",
    "duplicate_batches",
)
MALFORMATIONS = (
    "non_finite_sink", "non_finite_foreign", "nan_text", "huge_int", "past_int64", "past_float53",
    "wrong_type", "missing_key", "empty_name", "long_name", "odd_name", "steps_out_of_order",
    "step_repeated", "negative_count", "null_value", "unknown_key", "identity",
)
FEATURES = ("grad_norms", "update_rms", "weight_norms", "layers", "moe", "attention", "system", "timing",
            "data", "scaler", "clip", "replica", "lr", "unchanged")


class Token:
    """A number spelled as another writer spells it, written into the line as it is (`NaN`,
    `Infinity`, `-Infinity`): Python's `json` reads these; the sink never writes them."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return f"Token({self.text!r})"

    def value(self) -> float:
        return {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}[self.text]


@dataclass
class Case:
    seed: int
    base: str
    malformations: list[str]
    ranks: list[int]
    features: list[str]
    records: list[dict[str, Any]] = field(default_factory=list)  # in the order written, every rank

    @property
    def run_id(self) -> str:
        return f"{RUN_PREFIX}-{self.seed}"

    def describe(self) -> str:
        bad = ", ".join(self.malformations) or "none"
        return (f"seed {self.seed}: base {self.base}, malformed {bad}, ranks {self.ranks}, "
                f"{len(self.records)} records, features {', '.join(self.features)}")

    def files(self) -> dict[int, str]:
        """Each rank's file as text, in the rank order of `ranks`. A record whose `rank` is not one
        of the case's ranks (a malformed identity) is in the file of the rank it was made for."""
        texts: dict[int, list[str]] = {rank: [] for rank in self.ranks}
        for record in self.records:
            texts[record.get("_file", record.get("rank", 0))].append(line(record))
        return {rank: "".join(lines) for rank, lines in texts.items()}


# --- how a record is written ---------------------------------------------------------------------


def _sink(value: Any) -> Any:
    """What `JsonlSink` writes for a value: a non-finite float as "nan", "inf" or "-inf"."""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {key: _sink(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sink(item) for item in value]
    return value


def _tokens(value: Any, found: list[str]) -> Any:
    if isinstance(value, Token):
        found.append(value.text)
        return f"\x00TOKEN{len(found) - 1}\x00"
    if isinstance(value, dict):
        return {key: _tokens(item, found) for key, item in value.items()}
    if isinstance(value, list):
        return [_tokens(item, found) for item in value]
    return value


def line(record: dict[str, Any]) -> str:
    """One record as its line of the file: the sink's spelling, with each `Token` as its own text
    (another writer's). A key that starts with `_` is the generator's and is not written."""
    found: list[str] = []
    kept = {key: value for key, value in record.items() if not key.startswith("_")}
    text = json.dumps(_tokens(_sink(kept), found), separators=(",", ":"), allow_nan=False)
    for number, token in enumerate(found):
        text = text.replace(json.dumps(f"\x00TOKEN{number}\x00"), token)
    return text + "\n"


def read(text: str) -> list[dict[str, Any]]:
    """The records of a file's text, as a reader parses its lines."""
    return [json.loads(row) for row in text.splitlines() if row.strip()]


# --- the healthy run -----------------------------------------------------------------------------


def _wobble(rng: random.Random, centre: float, spread: float = 0.05) -> float:
    return round(centre * (1 + spread * (2 * rng.random() - 1)), 9)


def _router_counts(rng: random.Random, experts: int, top_k: int, tokens: int) -> list[int]:
    weights = [1 + 0.1 * rng.random() for _ in range(experts)]
    total = tokens * top_k
    counts = [int(total * w / sum(weights)) for w in weights]
    counts[0] += total - sum(counts)
    return counts


def _healthy(rng: random.Random, run_id: str, rank: int, attempt: int, steps: list[int],
             features: list[str], shape: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    params, blocks, routers = shape["params"], shape["blocks"], shape["routers"]
    for step in steps:
        record: dict[str, Any] = {
            "schema_version": 1, "run_id": run_id, "rank": rank, "attempt": attempt, "step": step,
            "ts": 1_700_000_000.0 + step, "grad_norm": _wobble(rng, 1.0),
            "scalars": {"train/loss": round(2.5 - step / 400 + 0.01 * rng.random(), 9)},
        }
        if "lr" in features:
            record["lr"] = [round(1e-3 * min(1.0, (step + 1) / 50), 12)]
        if "grad_norms" in features:
            record["grad_norms"] = {name: _wobble(rng, 0.5) for name in params}
        if "update_rms" in features:
            record["update_rms"] = {name: _wobble(rng, 1e-3) for name in params}
        if "weight_norms" in features and step % 10 == 0:
            record["weight_norms"] = {name: _wobble(rng, 10.0, 0.001) for name in params}
        if "layers" in features:
            record["layers"] = {
                name: {"mean": round(0.01 * rng.random(), 9), "std": _wobble(rng, 1.0),
                       "abs_max": _wobble(rng, 3.0), "zero_frac": round(0.01 * rng.random(), 9)}
                for name in blocks
            }
        if "moe" in features:
            record["moe"] = {name: _router_counts(rng, experts, top_k, 512) for name, (experts, top_k) in routers}
            record["moe_top_k"] = {name: top_k for name, (_, top_k) in routers}
        if "attention" in features and step % 5 == 0:
            record["probe"] = {"attention": {
                name: {"max_logit": [_wobble(rng, 4.0) for _ in range(4)],
                       "entropy": [_wobble(rng, 2.0) for _ in range(4)],
                       "entropy_share": [_wobble(rng, 0.7) for _ in range(4)],
                       "masked_rows": 0, "invalid_rows": 0}
                for name in blocks
            }}
        if "system" in features and step % 10 == 9:
            record["system"] = {
                "cuda": {"device": 0, "allocated": 2 * 2**30, "reserved": 4 * 2**30, "peak_allocated": 3 * 2**30,
                         "total": 16 * 2**30, "alloc_retries": 0, "ooms": 0},
                "nvml": {"temperature_c": 60, "sm_clock_mhz": 1410, "power_w": 250.0, "power_limit_w": 400.0,
                         "ecc_uncorrected": 0, "ecc_corrected": 0, "throttle": [], "xid": []},
            }
        if "timing" in features:
            record["data_wait"] = round(0.002 * (1 + rng.random()), 9)
            record["step_time"] = round(0.1 * (1 + 0.05 * rng.random()), 9)
        if "data" in features:
            record["data"] = {"batches": 1, "batch_hash": f"{rank}-{step:06d}", "epoch": step // 50,
                              "sampler_epoch": step // 50, "shuffle": True}
        if "scaler" in features:
            record["grad_scale"] = 65536.0
            record["step_skipped"] = False
        if "clip" in features:
            record["clip_grad_norm"] = round(record["grad_norm"] * 1.2, 9)
        if "replica" in features and step % 10 == 0:
            record["replica_spread"] = {name: [0.0, 0.0, 0.0] for name in params}
        if "unchanged" in features:
            record["weights_unchanged"] = []
        out.append(record)
    return out


# --- planted faults ------------------------------------------------------------------------------


def _plant(rng: random.Random, fault: str, records: list[dict[str, Any]], features: list[str],
           shape: dict[str, Any]) -> None:
    """Change `records` (one rank's, one attempt's) so that `fault` happens in them."""
    steps = [r["step"] for r in records]
    late = steps[min(len(steps) - 1, max(30, len(steps) * 2 // 3))]
    params, blocks, routers = shape["params"], shape["blocks"], shape["routers"]
    for record in records:
        step = record["step"]
        if fault == "grad_spike" and step == late:
            record["grad_norm"] *= 40
            if "grad_norms" in record:
                record["grad_norms"][params[0]] *= 60
        elif fault == "non_finite" and step == late:
            target = rng.choice([k for k in ("grad_norm", "grad_norms", "scalars") if k in record])
            if target == "grad_norm":
                record["grad_norm"] = math.nan
            else:
                key = rng.choice(list(record[target]))
                record[target][key] = rng.choice([math.nan, math.inf, -math.inf])
        elif fault == "update_spike" and step >= late:
            record.setdefault("update_rms", {name: 1e-3 for name in params})
            record["update_rms"] = {k: v * (30 if step == late else 1) for k, v in record["update_rms"].items()}
        elif fault == "stalled_weights" and late <= step < late + 8:
            record["weights_unchanged"] = [params[0]]
        elif fault == "loss_scale_trouble" and step >= late - 10:
            record["grad_scale"] = 65536.0 / 2 ** min(16, step - late + 11)
            record["step_skipped"] = True
        elif fault == "moe_imbalance" and step >= late and "moe" in record:
            name, (experts, top_k) = routers[0]
            counts = [1] * experts
            counts[0] = 512 * top_k - (experts - 1)
            record["moe"][name] = counts
        elif fault == "starved_experts" and "moe" in record:
            name, (experts, top_k) = routers[0]
            counts = record["moe"][name]
            moved = counts[-1] - 1
            counts[-1] = 1
            counts[0] += moved
        elif fault == "activation_spike" and step == late and "layers" in record:
            record["layers"][blocks[0]]["abs_max"] *= 50
            record["layers"][blocks[0]]["std"] *= 20
        elif fault == "dead_activations" and step >= late - 60 and "layers" in record:
            record["layers"][blocks[-1]].update({"zero_frac": 1.0, "std": 0.0, "abs_max": 0.0, "mean": 0.0})
        elif fault == "attention_collapse" and step >= late and "probe" in record:
            record["probe"]["attention"][blocks[0]]["entropy_share"] = [0.01] * 4
            record["probe"]["attention"][blocks[0]]["entropy"] = [0.02] * 4
        elif fault == "attention_logit_growth" and "probe" in record:
            growth = 1 + max(0, step - steps[0]) / 5
            record["probe"]["attention"][blocks[0]]["max_logit"] = [4.0 * growth] * 4
        elif fault == "data_stalled" and step >= late and "data" in record:
            record["data"]["batch_hash"] = f"stuck-{record['rank']}"
        elif fault == "replica_drift" and step >= late and "replica_spread" in record:
            record["replica_spread"][params[0]] = [0.5, 0.25, 0.75]
        elif fault == "memory_growth" and "system" in record:
            record["system"]["cuda"]["allocated"] = 2 * 2**30 + (step + 1) * 2**26
            record["system"]["cuda"]["peak_allocated"] = record["system"]["cuda"]["allocated"]
        elif fault == "memory_pressure" and step >= late and "system" in record:
            record["system"]["cuda"]["reserved"] = 16 * 2**30 - 2**20
            record["system"]["cuda"]["ooms"] = 1
        elif fault == "allocator_retries" and step >= late and "system" in record:
            record["system"]["cuda"]["alloc_retries"] = step - late + 1
        elif fault == "hardware_errors" and step >= late and "system" in record:
            record["system"]["nvml"]["ecc_uncorrected"] = 1 + (step - late)
            record["system"]["nvml"]["xid"] = [79]
        elif fault == "gpu_throttled" and step >= late and "system" in record:
            record["system"]["nvml"]["throttle"] = ["hw_thermal_slowdown"]
            record["system"]["nvml"]["temperature_c"] = 92
        elif fault == "input_starvation" and step >= late - 25 and "timing" in features:
            record["data_wait"] = round(record["step_time"] * 0.8, 9)
    if fault == "repeated_steps":
        again = [copy.deepcopy(r) for r in records if late - 5 <= r["step"] < late]
        index = next(i for i, r in enumerate(records) if r["step"] == late)
        records[index:index] = again


def _needs(fault: str) -> tuple[str, ...]:
    return {
        "update_spike": ("update_rms", "grad_norms"), "stalled_weights": ("unchanged", "grad_norms"),
        "loss_scale_trouble": ("scaler",), "moe_imbalance": ("moe",), "starved_experts": ("moe",),
        "activation_spike": ("layers",), "dead_activations": ("layers",), "attention_collapse": ("attention",),
        "attention_logit_growth": ("attention",), "data_stalled": ("data",), "replica_drift": ("replica",),
        "memory_growth": ("system",), "memory_pressure": ("system",), "allocator_retries": ("system",),
        "hardware_errors": ("system",), "gpu_throttled": ("system",), "input_starvation": ("timing",),
        "duplicate_batches": ("data",), "resume_lr_change": ("lr",),
    }.get(fault, ())


# --- malformations -------------------------------------------------------------------------------

_IDENTITY = ("schema_version", "run_id", "rank", "attempt", "step")


def _numeric_places(record: dict[str, Any]) -> list[tuple[Any, ...]]:
    """Where a number is in `record`, as paths of keys and list places, identity fields left out."""
    found: list[tuple[Any, ...]] = []

    def walk(value: Any, path: tuple[Any, ...]) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            found.append(path)
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(item, (*path, key))
        elif isinstance(value, list):
            for at, item in enumerate(value):
                walk(item, (*path, at))

    for key, value in record.items():
        if key not in _IDENTITY and not key.startswith("_"):
            walk(value, (key,))
    return found


def _map_places(record: dict[str, Any]) -> list[tuple[Any, ...]]:
    """The dicts of `record` with at least one entry, as paths (identity fields left out)."""
    found: list[tuple[Any, ...]] = []

    def walk(value: Any, path: tuple[Any, ...]) -> None:
        if isinstance(value, dict):
            if value and path:
                found.append(path)
            for key, item in value.items():
                walk(item, (*path, key))

    walk({k: v for k, v in record.items() if k not in _IDENTITY and not k.startswith("_")}, ())
    return found


def _get(record: Any, path: tuple[Any, ...]) -> Any:
    for key in path:
        record = record[key]
    return record


def _set(record: Any, path: tuple[Any, ...], value: Any) -> None:
    _get(record, path[:-1])[path[-1]] = value


def _rename(record: dict[str, Any], path: tuple[Any, ...], name: str) -> None:
    """Rename the first entry of the map at `path` to `name`, keeping the entries' order."""
    table = _get(record, path)
    first = next(iter(table))
    renamed = {(name if key == first else key): value for key, value in table.items()}
    _set(record, path, renamed)


def _placed(step: Any) -> bool:
    """Whether a record's step is one a malformation may pick: a number of 0 or more (a step an earlier
    malformation made `None` or text is not)."""
    try:
        return step >= 0
    except TypeError:
        return False


def _malform(rng: random.Random, kind: str, records: list[dict[str, Any]]) -> None:
    """Put the malformation `kind` on one to three records of `records` (every rank's, in order)."""
    training = [i for i, r in enumerate(records) if _placed(r.get("step", -1))]
    picked = sorted(rng.sample(training, min(len(training), rng.choice([1, 1, 2, 3]))))
    for index in picked:
        record = records[index]
        numbers, maps = _numeric_places(record), _map_places(record)
        if kind in ("non_finite_sink", "non_finite_foreign", "nan_text", "huge_int", "past_int64", "past_float53",
                    "wrong_type", "null_value") and numbers:
            path = rng.choice(numbers)
            value: Any = {
                "non_finite_sink": lambda: rng.choice([math.nan, math.inf, -math.inf]),
                "non_finite_foreign": lambda: Token(rng.choice(["NaN", "Infinity", "-Infinity"])),
                "nan_text": lambda: rng.choice(["NaN", "Infinity", "nan ", "NAN", "-nan"]),
                "huge_int": lambda: rng.choice([10**400, -(10**400), 10**309]),
                "past_int64": lambda: rng.choice([2**63, 2**64, -(2**63) - 1]),
                "past_float53": lambda: 2**53 + 1,
                "wrong_type": lambda: rng.choice([True, False, "1.0", "", [1.0], {"x": 1.0}, []]),
                "null_value": lambda: None,
            }[kind]()
            _set(record, path, value)
        elif kind == "missing_key":
            keys = [k for k in record if k not in _IDENTITY and not k.startswith("_")]
            if keys:
                del record[rng.choice(keys)]
        elif kind in ("empty_name", "long_name", "odd_name") and maps:
            name = {"empty_name": "", "long_name": "w" * rng.choice([300, 5000]),
                    "odd_name": rng.choice(["a.b/c d", "é́", " ", "0", "nan", "key\"quote"])}[kind]
            _rename(record, rng.choice(maps), name)
        elif kind == "negative_count" and isinstance(record.get("moe"), dict) and record["moe"]:
            name = rng.choice(list(record["moe"]))
            counts = record["moe"][name]
            at = rng.randrange(len(counts))
            counts[at] = -abs(counts[at]) - rng.choice([0, 1, 10**6])
        elif kind == "unknown_key":
            record[rng.choice(["moe_source", "replica_spread_scope", "optimizer_ownership", "zz_new"])] = rng.choice(
                [{"r": {"form": "scores", "x": 1}}, "text", 7, [1, "a"], {"state": "known", "unheld": 5}])
        elif kind == "identity":
            key = rng.choice(_IDENTITY)
            record[key] = rng.choice([None, "1", 1.5, True, -1, 2**63])
    if kind == "steps_out_of_order" and len(training) > 3:
        a, b = sorted(rng.sample(training, 2))
        records[a], records[b] = records[b], records[a]
    elif kind == "step_repeated" and training:
        index = rng.choice(training)
        records.insert(index + 1, copy.deepcopy(records[index]))


# --- a case --------------------------------------------------------------------------------------


def generate(seed: int, *, base: str | None = None, malformations: Iterable[str] | None = None) -> Case:
    """The case of `seed`. `base` and `malformations` override what the seed draws for them (the rest is
    still drawn from the seed)."""
    rng = random.Random(seed)
    drawn_base = "healthy" if rng.random() < 0.25 else rng.choice(FAULTS)
    base = drawn_base if base is None else base
    roll = rng.random()
    drawn_bad = [] if roll < 0.3 else [rng.choice(MALFORMATIONS)] if roll < 0.9 else rng.sample(MALFORMATIONS, 2)
    bad = list(drawn_bad if malformations is None else malformations)
    features = [f for f in FEATURES if rng.random() < 0.55]
    for needed in _needs(base):
        if needed not in features:
            features.append(needed)
    features = [f for f in FEATURES if f in features]  # the one order
    ranks = [0, 1] if base in ("duplicate_batches", "replica_drift") or rng.random() < 0.2 else [0]
    if rng.random() < 0.05:
        ranks = [0, 1, 2]
    shape = {
        "params": [f"p{i}.weight" for i in range(rng.choice([1, 2, 3]))],
        "blocks": [f"blk{i}" for i in range(rng.choice([1, 2, 3]))],
        "routers": [(f"r{i}", (rng.choice([4, 8]), rng.choice([1, 2]))) for i in range(rng.choice([1, 2]))],
    }
    length = rng.choice([40, 80, 130, 180])
    resumes = base == "resume_lr_change" or rng.random() < 0.15
    case = Case(seed, base, bad, ranks, features)
    run_id = case.run_id
    for rank in ranks:
        first = _healthy(rng, run_id, rank, 1, list(range(length)), features, shape)
        if base not in ("healthy", "resume_lr_change", "duplicate_batches") and (rank == 0 or base == "replica_drift"):
            _plant(rng, base, first, features, shape)
        records = first
        if resumes:
            back = rng.randrange(5, length // 2)
            second = _healthy(rng, run_id, rank, 2, list(range(length - back, length - back + 40)), features, shape)
            if base == "resume_lr_change" and "lr" in features:
                for record in second:
                    record["lr"] = [record["lr"][0] * 10]
            records = first + second
        if base == "duplicate_batches" and rank == 1:
            for record in records:
                if 10 <= record["step"] < 25 and "data" in record:
                    record["data"]["batch_hash"] = f"0-{record['step']:06d}"
        for record in records:
            record["_file"] = rank
        case.records.extend(records)
    for kind in bad:
        _malform(rng, kind, case.records)
    return case


def write(case: Case, folder: Path) -> dict[int, Path]:
    """The case's files in `folder`, by rank, named as `tensorquill.init` names them."""
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for rank, text in case.files().items():
        path = folder / f"{case.run_id}.rank{rank}.jsonl"
        path.write_text(text, encoding="utf-8")
        out[rank] = path
    return out


def write_corpus(folder: Path, seeds: Iterable[int]) -> Path:
    """Many cases' files in `folder`, one subfolder a seed, and `cases.json` naming each: what another
    language's reader of `.jsonl` lines (step 2's converter) is held to."""
    index = []
    for seed in seeds:
        case = generate(seed)
        files = write(case, folder / str(seed))
        index.append({"seed": seed, "base": case.base, "malformations": case.malformations,
                      "files": {str(rank): str(path.relative_to(folder)) for rank, path in files.items()}})
    path = folder / "cases.json"
    path.write_text(json.dumps({"generator_version": GENERATOR_VERSION, "cases": index}, indent=1) + "\n")
    return path


# --- shrinking -----------------------------------------------------------------------------------


def _without(records: list[dict[str, Any]], start: int, size: int) -> list[dict[str, Any]]:
    return records[:start] + records[start + size:]


def shrink(records: list[dict[str, Any]], still: Callable[[list[dict[str, Any]]], bool], *,
           budget: int = 400) -> list[dict[str, Any]]:
    """A smaller list for which `still` holds, from `records`, for which it must hold. Drops runs of
    records (halving the run each round, as delta debugging does), then each record's keys that are
    not identity fields, then entries of their maps and items of their lists. `budget` bounds the
    calls of `still`."""
    calls = 0

    def holds(candidate: list[dict[str, Any]]) -> bool:
        nonlocal calls
        if calls >= budget or not candidate:
            return False
        calls += 1
        try:
            return still(candidate)
        except Exception:  # noqa: BLE001  a candidate that breaks the check is not one that still fails
            return False

    current = [copy.deepcopy(r) for r in records]
    size = max(1, len(current) // 2)
    while size >= 1 and calls < budget:
        start, moved = 0, False
        while start < len(current) and calls < budget:
            candidate = _without(current, start, size)
            if holds(candidate):
                current, moved = candidate, True
            else:
                start += size
        if not moved:
            size //= 2
    for index in range(len(current)):
        for key in list(current[index]):
            if key in _IDENTITY or key not in current[index]:
                continue
            candidate = copy.deepcopy(current)
            del candidate[index][key]
            if holds(candidate):
                current = candidate
    for index in range(len(current)):
        for path in sorted(_map_places(current[index]), key=len, reverse=True):
            try:
                keys = list(_get(current[index], path))
            except (KeyError, IndexError, TypeError):
                continue
            for key in keys:
                candidate = copy.deepcopy(current)
                try:
                    del _get(candidate[index], path)[key]
                except (KeyError, IndexError, TypeError):
                    continue
                if holds(candidate):
                    current = candidate
    return current


if __name__ == "__main__":
    for given in sys.argv[1:] or ["0"]:
        shown = generate(int(given))
        print(shown.describe())
        for rank, text in shown.files().items():
            print(f"--- rank {rank}")
            sys.stdout.write(text)
