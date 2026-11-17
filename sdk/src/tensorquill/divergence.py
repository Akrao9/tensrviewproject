"""Where two paired runs part: the first step, and the layer or parameter, where a series leaves
the two runs' own early agreement and stays out.

This is for PAIRED runs: the same job run twice, or a run before and after one change. Each
series one run recorded is set against the same series of the other, on the steps both recorded:

    d[step] = b[step] - a[step]              only steps present in both; nothing is interpolated
    movement = how much the series itself changes from one record to the next, early in the runs
    early    = the first EARLY_STEPS common steps: how far apart the two runs normally are

A series has LEFT at the first step from which |d| is above its line on SUSTAIN common records
in a row AND on at least STAY_SHARE of all its common records from there to the end: it left and
stayed out. A few records out that come back are counted and not called a departure. The line is
OUT_FACTOR times the largest early |d|, and never less than MOVE_FLOOR of the series' own movement
(the floor is the whole line for two runs that were identical early, as two CPU runs are).

Runs that never agree in their first steps (another seed, another model, other data) get NO
ANSWER, with the reason: there, the early difference is as large as the series' own movement, so
no later step can be called the one where they parted. "No answer" is a result, never a step.

The answer names a step and a series. It is where to look first, not a cause: it does not say
which run is the wrong one, or that what changed there caused what came after.

The numbers below were chosen on recorded pairs (the v3 plan's Decisions entry "Where two runs
part" has the measurements). They are read when `first_divergence` is called.
"""

from __future__ import annotations

import math
import re
import statistics
from typing import Any, Iterable, Mapping

__all__ = ["first_divergence", "divergence_summary"]

SCHEMA_VERSION = 1

#: Common steps, from the first one, that say how far apart the two runs normally are.
EARLY_STEPS = 30
#: A difference is out of agreement above this many times the largest early difference.
OUT_FACTOR = 12.0
#: Common records in a row a series must stay out before it has left.
SUSTAIN = 3
#: The line is never below this share of the series' own record-to-record movement. It is the
#: whole line when the two runs were identical early.
MOVE_FLOOR = 0.01
#: Two runs agree early on a series when their typical early difference is below this share of
#: the series' own movement; the runs are paired when half of the series that move agree.
AGREE_RATIO = 0.2
#: A series that left must also be out on this share of its common records from that step to
#: the end of the runs. Run-to-run noise on a GPU puts a series out for a few records and back.
STAY_SHARE = 0.5
#: Early samples a series needs when every series of the two runs is identical in the early
#: steps. One recorded less often than every step (weight norms, every 50) has fewer inside them;
#: its first samples then stand in. When the runs differ at all in the early steps, a series
#: needs samples on half of them (`EARLY_STEPS // 2`), or how far apart it normally is cannot be
#: told and it is not compared: weight norms judged on two noisy samples named a wrong step.
MIN_EARLY_SAMPLES = 2

#: Record keys that are wall-clock readings: two identical runs differ in them from step 0.
WALL_CLOCK_KEYS = ("ts", "data_wait", "step_time", "system")
#: A logged scalar is taken as a wall-clock reading when one of these is a whole word of its name
#: (`perf/step_time_ms`, `tokens_per_sec`). Whole words only: `val_mse` and `train/beta` are series.
WALL_CLOCK_WORDS = frozenset((
    "time", "times", "timing", "duration", "elapsed", "latency", "throughput", "ms", "sec", "secs", "second",
    "seconds", "speed", "runtime", "walltime", "wallclock", "memory", "mem", "util", "utilization", "fps",
))
_WORD_BREAKS = re.compile(r"[^a-z0-9]+")
#: Record keys that identify a record and are not series.
_IDENTITY_KEYS = ("schema_version", "run_id", "rank", "attempt", "step")

# Ties at the first step go to the series nearest the start of the step's own work: a block's
# output changes before the loss computed from it and the gradients computed from that, and the
# weight check comes after the optimizer's step.
_KIND_ORDER = {"layers": 0, "moe": 0, "scalars": 1, "grad_norms": 2, "grad_norm": 3, "weight_norms": 4}
_NON_FINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


# ---- reading the records -----------------------------------------------------------------


def _number(value: Any) -> float | None:
    """A recorded number as a float (non-finite ones included), or None for anything else."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return _NON_FINITE.get(value)
    return None


def _json_number(value: float | None) -> float | str | None:
    """Strict JSON has no NaN or Infinity: a non-finite value is written as the records write it."""
    if value is None or math.isfinite(value):
        return value
    return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")


def _finite_or_none(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _is_wall_clock(name: str) -> bool:
    return not WALL_CLOCK_WORDS.isdisjoint(_WORD_BREAKS.split(name.lower()))


class _Run:
    """One rank's series of one run: `series[id][step] = value`."""

    def __init__(self, records: Iterable[Mapping[str, Any]], rank: int) -> None:
        self.series: dict[str, dict[int, float]] = {}
        self.about: dict[str, dict[str, Any]] = {}
        self.layer_order: dict[str, int] = {}
        self.steps: set[int] = set()
        self.run_ids: list[Any] = []
        self.ranks_seen: set[Any] = set()
        self.wall_clock: set[str] = set()
        self.not_compared: set[str] = set()
        self.records = 0
        for record in records:
            record_rank = record.get("rank", 0)
            self.ranks_seen.add(record_rank)
            if record_rank != rank or record.get("step") is None:
                continue
            self.records += 1
            if record.get("run_id") not in self.run_ids:
                self.run_ids.append(record.get("run_id"))
            self._read(record, int(record["step"]))

    def _put(self, kind: str, name: str | None, stat: str | None, step: int, value: Any) -> None:
        number = _number(value)
        if number is None:
            return
        series_id = "/".join(part for part in (kind, name, stat) if part is not None)
        if series_id not in self.series:
            self.series[series_id] = {}
            self.about[series_id] = {"kind": kind, "name": name, "stat": stat}
        self.series[series_id][step] = number  # a step written twice: the later record stands

    def _read(self, record: Mapping[str, Any], step: int) -> None:
        self.steps.add(step)
        for key, value in record.items():
            if key in _IDENTITY_KEYS or value is None:
                continue
            if key in WALL_CLOCK_KEYS:
                self.wall_clock.add(key)
            elif key in ("grad_norm", "clip_grad_norm"):
                self._put("grad_norm", None if key == "grad_norm" else "clipped", None, step, value)
            elif key == "scalars" and isinstance(value, Mapping):
                for name, scalar in value.items():
                    if _is_wall_clock(str(name)):
                        self.wall_clock.add(f"scalars/{name}")
                    else:
                        self._put("scalars", str(name), None, step, scalar)
            elif key in ("grad_norms", "weight_norms") and isinstance(value, Mapping):
                for name, norm in value.items():
                    self._put(key, str(name), None, step, norm)
            elif key == "layers" and isinstance(value, Mapping):
                for name, stats in value.items():
                    self.layer_order.setdefault(str(name), len(self.layer_order))
                    if isinstance(stats, Mapping):
                        for stat, number in stats.items():
                            self._put("layers", str(name), str(stat), step, number)
            elif key == "moe" and isinstance(value, Mapping):
                for name, counts in value.items():
                    self._router(str(name), counts, step)
            else:
                self.not_compared.add(key)

    def _router(self, name: str, counts: Any, step: int) -> None:
        """A router's largest expert share and the share of its experts with no tokens."""
        if not isinstance(counts, (list, tuple)) or not counts:
            return
        numbers = [_number(c) for c in counts]
        if any(n is None or not math.isfinite(n) for n in numbers):
            return
        total = math.fsum(numbers)  # type: ignore[arg-type]
        if total <= 0:
            return  # no token was routed at this step: no share to compare
        self._put("moe", name, "max_share", step, max(numbers) / total)  # type: ignore[type-var]
        self._put("moe", name, "dead_frac", step, sum(1 for n in numbers if n == 0) / len(numbers))

    def describe(self) -> dict[str, Any]:
        return {
            "run_id": self.run_ids[0] if len(self.run_ids) == 1 else (self.run_ids or None),
            "records": self.records,
            "first_step": min(self.steps) if self.steps else None,
            "last_step": max(self.steps) if self.steps else None,
        }


# ---- one series --------------------------------------------------------------------------


def _same(a: float, b: float) -> bool:
    """Exactly the same recorded value; two NaNs, or two infinities of one sign, are the same."""
    return a == b or (math.isnan(a) and math.isnan(b))


def _gap(a: float, b: float) -> float:
    """|b - a|; infinite when one side is not finite and the other differs from it."""
    if _same(a, b):
        return 0.0
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return abs(b - a)


def _movement(values: list[float]) -> list[float]:
    return [abs(y - x) for x, y in zip(values, values[1:]) if math.isfinite(x) and math.isfinite(y)]


def _identical_early(a: Mapping[int, float], b: Mapping[int, float], early_last: int) -> bool:
    return all(_same(a[s], b[s]) for s in a if s <= early_last and s in b)


def _judge(a: Mapping[int, float], b: Mapping[int, float], early_last: int, identical: bool) -> dict[str, Any]:
    """One series of both runs: its early agreement, its line, and where it left if it did.

    `identical`: every series of the two runs has the same values in the early steps.
    """
    steps = sorted(set(a) & set(b))
    out: dict[str, Any] = {"common_steps": len(steps)}
    if not steps:
        return {**out, "status": "not_comparable", "why": "no step has this series in both runs"}
    n_early = sum(1 for s in steps if s <= early_last)
    if not identical and n_early < EARLY_STEPS // 2:
        return {
            **out, "status": "not_comparable",
            "why": f"{n_early} of its samples fall in the early steps (up to step {early_last}; {EARLY_STEPS // 2} are "
                   "needed) and the runs are not identical there, so how far apart it normally is cannot be told",
        }
    n_early = max(n_early, MIN_EARLY_SAMPLES)
    if len(steps) < n_early + SUSTAIN:
        return {
            **out, "status": "not_comparable",
            "why": f"{len(steps)} common steps: {n_early} early ones and {SUSTAIN} more are needed",
        }
    early, later = steps[:n_early], steps[n_early:]
    gaps = [_gap(a[s], b[s]) for s in early]
    moves = _movement([a[s] for s in early]) + _movement([b[s] for s in early])
    movement = statistics.median(moves) if moves else 0.0
    largest = max(gaps)
    typical = statistics.median(gaps)
    if not math.isfinite(largest):
        ratio = math.inf
    elif movement > 0:
        ratio = typical / movement
    else:
        ratio = 0.0 if largest == 0 else math.inf
    out["early"] = {
        "samples": n_early, "last_step": early[-1],
        "largest_difference": _finite_or_none(largest),
        "typical_difference": _finite_or_none(typical),
        "own_movement": movement,
        "ratio": _finite_or_none(ratio),
    }
    out["_ratio"], out["_moves"] = ratio, movement > 0
    out["_first_exact"] = next((s for s in steps if not _same(a[s], b[s])), None)
    if not math.isfinite(largest):
        return {**out, "status": "not_comparable", "why": "a value was not finite in one run only in the early steps"}
    if not ratio < AGREE_RATIO:
        why = (
            f"the runs do not agree on it early: the typical early difference is {ratio:.2g} times the series' own "
            f"step-to-step movement (agreement is below {AGREE_RATIO:g})"
            if math.isfinite(ratio) else
            "the runs do not agree on it early: the series does not move in the early steps and its two values differ"
        )
        return {**out, "status": "not_comparable", "why": why}

    line = max(OUT_FACTOR * largest, MOVE_FLOOR * movement)
    out["threshold"] = line
    out["status"] = "agrees"
    is_out = [_gap(a[step], b[step]) > line for step in later]
    out_after = [0] * (len(later) + 1)  # out_after[i]: records out from index i to the end
    for index in range(len(later) - 1, -1, -1):
        out_after[index] = out_after[index + 1] + is_out[index]
    run_start: int | None = None
    length = 0
    came_back: list[tuple[int, int]] = []
    index = 0
    while index < len(later):
        if not is_out[index]:
            index += 1
            continue
        end = index
        while end < len(later) and is_out[end]:
            end += 1
        stretch = end - index
        if stretch >= SUSTAIN and out_after[index] >= STAY_SHARE * (len(later) - index):
            run_start, length = index, stretch
            break
        if stretch >= SUSTAIN:
            came_back.append((later[index], stretch))
        elif end == len(later):
            # Out at the end of the runs, for fewer records than a departure needs.
            out["left_at_end"] = {"step": later[index], "records": stretch}
        index = end
    if came_back:
        out["came_back"] = {
            "stretches": len(came_back), "first_step": came_back[0][0], "longest": max(n for _, n in came_back),
        }
    if run_start is None:
        return out
    onset = later[run_start]
    before = steps[n_early + run_start - 1]
    first = [_gap(a[s], b[s]) for s in later[run_start:run_start + SUSTAIN]]
    size = statistics.median(first)
    non_finite = not math.isfinite(size)
    out.update(
        status="diverged", step=onset,
        before={"step": before, "a": _json_number(a[before]), "b": _json_number(b[before])},
        at={"step": onset, "a": _json_number(a[onset]), "b": _json_number(b[onset])},
        difference=_finite_or_none(_gap(a[onset], b[onset])),
        size=None if non_finite else size,
        times_threshold=None if non_finite or line == 0 else size / line,
        in_own_movement=None if non_finite or movement == 0 else size / movement,
        non_finite=non_finite or not (math.isfinite(a[onset]) and math.isfinite(b[onset])),
        sustained=length,
        out_share=out_after[run_start] / (len(later) - run_start),
    )
    out["_rank_size"] = math.inf if non_finite or movement == 0 else size / movement
    return out


# ---- the two runs ------------------------------------------------------------------------


def _thresholds() -> dict[str, Any]:
    return {
        "early_steps": EARLY_STEPS, "out_factor": OUT_FACTOR, "sustain": SUSTAIN,
        "stay_share": STAY_SHARE,
        "move_floor": MOVE_FLOOR, "agree_ratio": AGREE_RATIO, "min_early_samples": MIN_EARLY_SAMPLES,
        "min_early_samples_when_not_identical": EARLY_STEPS // 2,
        "min_common_steps": EARLY_STEPS + SUSTAIN,
    }


def _public(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if not key.startswith("_")}


def first_divergence(
    records_a: Iterable[Mapping[str, Any]],
    records_b: Iterable[Mapping[str, Any]],
    *,
    rank: int = 0,
) -> dict[str, Any]:
    """The first step, and the series, where run B leaves its early agreement with run A.

    `records_a` and `records_b` are two runs' step records as `load_records` returns them. One
    rank's records are read (`rank`, named in the result). Compared: every logged scalar that is
    not a wall-clock reading, the total gradient norm, each parameter's gradient norm and weight
    norm, each captured module's activation statistics, and each MoE router's largest expert
    share and share of experts with no tokens.

    The result is plain data, ready for `json.dumps(..., allow_nan=False)`:

    - `answer`: `"diverged"`, `"no_divergence"` or `"no_answer"`, and `reason`, a sentence.
    - `divergence`: for `"diverged"`, the step, the series (`kind`, `name`, `stat`), both runs'
      values at the last common step before it and at it, the line it crossed and by how much.
    - `diverged_series`: every series that left, first step first.
    - `series`: for each series, what was compared, on how many common steps, and how it ended.
    - `not_comparable`: series in one run only, too short, or not agreeing early, each with why.
    - `thresholds`, `rank`, `runs`, `common_steps`, `early`, `agreement`, `left_out`.
    """
    a, b = _Run(records_a, rank), _Run(records_b, rank)
    common = sorted(a.steps & b.steps)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "answer": "no_answer",
        "reason": "",
        "rank": rank,
        "runs": {"a": a.describe(), "b": b.describe()},
        "common_steps": {
            "count": len(common), "first": common[0] if common else None, "last": common[-1] if common else None,
        },
        "thresholds": _thresholds(),
        "early": None,
        "agreement": None,
        "first_exact_difference": None,
        "divergence": None,
        "diverged_series": [],
        "series": {},
        "not_comparable": [],
        "left_out": {
            "wall_clock": sorted(a.wall_clock | b.wall_clock),
            "not_compared": sorted(a.not_compared | b.not_compared),
        },
    }

    def no_answer(reason: str) -> dict[str, Any]:
        result["reason"] = reason
        return result

    for label, run in (("A", a), ("B", b)):
        if not run.records:
            seen = sorted(run.ranks_seen, key=str)
            return no_answer(
                f"No answer: run {label} has no records of rank {rank}"
                + (f" (it has ranks {seen})." if seen else " (it has no records).")
            )
    if not common:
        return no_answer(
            f"No answer: the two runs have no step in common (A recorded steps {min(a.steps)} to {max(a.steps)}, "
            f"B {min(b.steps)} to {max(b.steps)}). Only steps both runs recorded are compared."
        )
    needed = EARLY_STEPS + SUSTAIN
    if len(common) < needed:
        return no_answer(
            f"No answer: the runs have {len(common)} steps in common and {needed} are needed: {EARLY_STEPS} to learn "
            f"how far apart the runs normally are and {SUSTAIN} more for a series to leave that and stay out."
        )
    early_last = common[EARLY_STEPS - 1]
    result["early"] = {"steps": EARLY_STEPS, "first": common[0], "last": early_last}

    for series_id in sorted(set(a.series) ^ set(b.series)):
        side = "A" if series_id in a.series else "B"
        result["not_comparable"].append({"series": series_id, "why": f"recorded in run {side} only"})
    shared = [series_id for series_id in a.series if series_id in b.series]
    if not shared:
        return no_answer("No answer: no series was recorded in both runs, so there is nothing to compare.")

    identical = all(_identical_early(a.series[series_id], b.series[series_id], early_last) for series_id in shared)
    result["early"]["identical"] = identical
    judged: dict[str, dict[str, Any]] = {}
    for series_id in shared:
        entry = {**a.about[series_id], **_judge(a.series[series_id], b.series[series_id], early_last, identical)}
        judged[series_id] = entry
        result["series"][series_id] = _public(entry)
        if entry["status"] == "not_comparable":
            result["not_comparable"].append({"series": series_id, "why": entry["why"]})

    exact = [e["_first_exact"] for e in judged.values() if e.get("_first_exact") is not None]
    if exact:
        result["first_exact_difference"] = {"step": min(exact), "series": sum(1 for s in exact if s == min(exact))}

    with_early = [e for e in judged.values() if "_ratio" in e]
    if not with_early:
        return no_answer(
            f"No answer: no series has enough common steps ({len(shared)} recorded in both runs; "
            f"each needs its early samples and {SUSTAIN} more)."
        )
    # Series that do not move early (a zero fraction that stays 0) are the same in any two runs,
    # so they cannot say the runs are a pair. They vote only when nothing moves.
    voters = [e for e in with_early if e["_moves"]] or with_early
    agreeing = sum(1 for e in voters if e["_ratio"] < AGREE_RATIO)
    middle = statistics.median(e["_ratio"] for e in voters)
    result["agreement"] = {
        "series": len(voters), "agreeing": agreeing,
        "median_ratio": _finite_or_none(middle), "paired": middle < AGREE_RATIO,
    }
    if not middle < AGREE_RATIO:
        size = f"{middle:.2g} times" if math.isfinite(middle) else "not comparable with"
        return no_answer(
            f"No answer: the two runs do not agree in their first {EARLY_STEPS} common steps. On {len(voters)} series, "
            f"the typical early difference between the runs is {size} the series' own step-to-step movement "
            f"(median over the series; {agreeing} of them agree, below {AGREE_RATIO:g}). This is for paired runs, "
            "the same job run twice or a run before and after one change; with another seed, model or data "
            "there is no early agreement to leave, so no step can be named."
        )

    left = [(series_id, e) for series_id, e in judged.items() if e["status"] == "diverged"]

    def order(item: tuple[str, dict[str, Any]]) -> tuple[Any, ...]:
        series_id, entry = item
        capture = a.layer_order.get(entry["name"], 0) if entry["kind"] == "layers" else 0
        return (entry["step"], _KIND_ORDER.get(entry["kind"], 9), capture, -entry["_rank_size"], series_id)

    left.sort(key=order)
    result["diverged_series"] = [
        {"series": series_id, **{k: e[k] for k in ("kind", "name", "stat", "step", "size", "times_threshold", "in_own_movement")}}
        for series_id, e in left
    ]
    compared = sum(1 for e in judged.values() if e["status"] != "not_comparable")
    if not left:
        result["answer"] = "no_divergence"
        tail = [(series_id, e["left_at_end"]) for series_id, e in judged.items() if "left_at_end" in e]
        reason = (
            f"No divergence found: on {compared} series over {len(common)} common steps, no series left the agreement "
            f"of the first {EARLY_STEPS} steps and stayed out ({SUSTAIN} records in a row and {STAY_SHARE:.0%} of the "
            "records after)."
        )
        back = [e["came_back"]["stretches"] for e in judged.values() if "came_back" in e]
        if back:
            reason += f" Out for {SUSTAIN} records or more and back again: {len(back)} series."
        if result["first_exact_difference"] is None:
            reason += " The two runs recorded exactly the same value on every compared series at every common step."
        if tail:
            tail.sort(key=lambda item: item[1]["step"])
            reason += (
                f" Out of agreement when the runs ended, for fewer than {SUSTAIN} records: {len(tail)} series "
                f"(first: {tail[0][0]} from step {tail[0][1]['step']}); that is too short to call."
            )
        result["reason"] = reason
        return result

    series_id, entry = left[0]
    together = [other for other, e in left[1:] if e["step"] == entry["step"]]
    result["answer"] = "diverged"
    result["divergence"] = {
        "series": series_id,
        **{k: entry[k] for k in (
            "kind", "name", "stat", "step", "before", "at", "difference", "threshold", "size", "times_threshold",
            "in_own_movement", "non_finite", "sustained", "out_share",
        )},
        "same_step": len(together),
        "same_step_series": together[:20],
    }
    where = entry["name"] if entry["name"] is not None else series_id
    what = {
        "layers": f"activation {entry['stat']} of {where}", "moe": f"router {entry['stat']} of {where}",
        "scalars": f"logged scalar {where}", "grad_norms": f"gradient norm of {where}",
        "weight_norms": f"weight norm of {where}", "grad_norm": "total gradient norm" + (" after clipping" if entry["name"] else ""),
    }.get(entry["kind"], series_id)
    at, before = entry["at"], entry["before"]
    if entry["non_finite"]:
        by = "a value that is not finite in one run only"
    elif entry["times_threshold"] is None:
        by = "the first difference after early steps in which the runs were identical and the series did not move"
    else:
        by = f"{entry['times_threshold']:.3g} times its line"
        if entry["in_own_movement"] is not None:
            by += f", {entry['in_own_movement']:.3g} times the series' own step-to-step movement"
    result["reason"] = (
        f"The runs part at step {entry['step']}, first on the {what}: A {at['a']}, B {at['b']} "
        f"(at step {before['step']}: A {before['a']}, B {before['b']}), {by}, and it stayed out for "
        f"{entry['sustained']} records in a row ({entry['out_share']:.0%} of its records from there on). "
        + (f"{len(together)} more series left at the same step. " if together else "")
        + f"{len(left)} of {compared} compared series left in all. This is where to look first, not a cause."
    )
    return result


def divergence_summary(result: Mapping[str, Any]) -> str:
    """A few lines for a person: the answer, what was compared, and what was not."""
    lines = [str(result.get("reason", ""))]
    series = result.get("series") or {}
    compared = sum(1 for e in series.values() if e.get("status") != "not_comparable")
    steps = result.get("common_steps") or {}
    if series:
        lines.append(
            f"Compared: rank {result.get('rank')}, {compared} series on {steps.get('count')} common steps "
            f"({steps.get('first')} to {steps.get('last')}); {len(result.get('not_comparable') or [])} series not comparable."
        )
    exact = result.get("first_exact_difference")
    if exact and result.get("answer") != "no_answer":
        lines.append(
            f"The first recorded value that differs at all is at step {exact['step']} ({exact['series']} series)."
        )
    left_out = (result.get("left_out") or {}).get("wall_clock") or []
    if left_out:
        lines.append(f"Left out as wall-clock readings: {', '.join(left_out)}.")
    return "\n".join(line for line in lines if line)
