"""Turn captured JSONL records into detector evidence and a Rerun recording.

The evidence dictionary uses the same schema as `experiments/moe-bakeoff`, so the stock
Rerun blueprint, `verify_rerun.py` and the Rune inspector can all read it directly, while
adding `incidents` for MoE routing imbalance, chronically starved experts, gradient
spikes, weights left unchanged by the optimizer, repeated step numbers, activation
spikes, drift and dead outputs, non-finite activations/gradients and cross-GPU
parameter drift.
"""

from __future__ import annotations

import argparse
import bisect
import contextvars
import copy
import json
import math
import os
import re
import secrets
import statistics
import sys
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import manifest as _manifest
# A record's identity fields (WAVE_AF_CONTRACT.md, section 1): checked where records are loaded,
# before either engine is chosen, by lane F's one check (`identity.check`).
from .identity import (
    MAX_EXPECTED_RANKS, MalformedRecordError, check as check_identity, in_file, in_list, in_part, is_scalar_record,
)
from .sink import read_end_marker, writers_block
from .data_checks import (
    CHANGES_LISTED,
    DATA_STALL_STEPS,
    detect_data_stalled,
    detect_epoch_order_repeated,
    detect_resume_state,
)
from .evidence_system import (
    INPUT_WINDOW,
    MEMORY_GROWTH_READINGS,
    SYSTEM_KINDS,
    describe_system,
    detect_memory_imbalance,
    detect_system_incidents,
    latest_memory,
)

OVERLOAD_THRESHOLD = 0.35
DRIFT_TOLERANCE = 1e-6
# An expert is starved when its share of a router's assignments, summed over the last
# STARVE_WINDOW samples, stays below STARVE_RATIO of an even share. A single step with no
# tokens is ordinary noise; hundreds of near-empty steps in a row are not.
STARVE_WINDOW = 50
STARVE_RATIO = 0.1
# Skip the check until an even share would have received this many assignments in the
# window, so that small routers and short runs are not judged on noise.
STARVE_MIN_EXPECTED = 100
# Top-k routers (k >= 3): no expert can take more than 1/k of the assignments, and in
# trained models some experts are picked by nearly every token, while with small batches
# single experts often jump to most tokens for one step. So a fixed share threshold either
# never fires or fires on healthy layers. Instead the router's diversity must fall below
# DIVERSITY_DROP times its median over the previous OVERLOAD_WINDOW unflagged samples.
# The overloaded experts are then those whose fraction of tokens rose by at least
# OVERLOAD_JUMP over their average in that window. On a healthy Granite run the lowest
# diversity ratio seen was 0.82, and the seeded collapse fell to 0.29.
DIVERSITY_DROP = 0.6
OVERLOAD_JUMP = 0.25
OVERLOAD_MIN_HISTORY = 10
OVERLOAD_WINDOW = 50
# A gradient-norm spike is a step whose total norm is at least SPIKE_FACTOR times the
# median of the previous SPIKE_WINDOW recorded steps (once SPIKE_MIN_HISTORY are known).
SPIKE_FACTOR = 5.0
SPIKE_WINDOW = 50
SPIKE_MIN_HISTORY = 10
# A single parameter's gradient norm at SPIKE_FACTOR times its own median is ordinary: on healthy
# Granite 3.0 1B captures recorded on GPUs on Nov 15, 2026 (six of them, the rule chosen on two) it
# happened 277 to 473 times in 650 steps of 218 parameters, twice in a row 2 to 15 times, and three
# times in a row never, while every seeded fault held it from its seeded step on. So a
# parameter is reported on its own only when it stays there this many recorded steps in a row.
SPIKE_SUSTAIN = 3
# A jump in the size of the optimizer's update (`update_spike`), by the rule of Altenbernd et
# al. (arXiv 2604.00726, eqs. 8 to 13). R is the largest RMS update over the parameters on a
# step (`update_rms`, with `init(..., updates=True)`), taken on both steps over the parameters
# both name (L13). A step is flagged when R changed from
# the step before by more than `baseline / (UPDATE_ALPHA * sqrt(rise))`: `rise` is how far the
# total gradient norm before clipping rose from the step before (a norm that did not rise
# flags nothing), and `baseline` is the mean of that change over the last UPDATE_WINDOW steps
# that could be compared and were not flagged (the `update_window` option). The paper takes
# the mean once, over the run's learning-rate warm-up, which a reader of the records does not
# know. A mean frozen after a fixed number of steps flagged a clean toy run whose warm-up was
# longer than it; the mean of the last steps did not (Decisions: "update_spike").
UPDATE_ALPHA = 0.05
UPDATE_WINDOW = 100
# Activation detectors compare each watched module with its own history, because trained
# LLMs carry large outliers by design: on healthy Granite the largest value in middle
# blocks was 740-840 times the standard deviation. Healthy Granite (650 steps, 24 blocks)
# peaked at 2.2x for one-step jumps and 1.27x for drift, so these leave a wide margin.
ACT_SPIKE_FACTOR = 5.0  # abs_max or std vs the median of the previous ACT_WINDOW records
ACT_WINDOW = 50
ACT_MIN_HISTORY = 10
ACT_DRIFT_FACTOR = 3.0  # recent median of std or abs_max/std vs the ACT_WINDOW records before them
ACT_RECENT = 10
ACT_DEAD_RISE = 0.25  # zero fraction above its baseline, as a recent median
# Records a module needs before drift is judged: the ACT_WINDOW of the baseline and the ACT_RECENT
# after it, with the module's first ACT_WINDOW records before both (rules contract, part 2).
ACT_DRIFT_NEEDS = 2 * ACT_WINDOW + ACT_RECENT
# Attention rules over the probe's per-head statistics (`probe=`, every `probe_every` steps,
# 1,000 by default). Probes are sparse, so each attention module (a block's attention call) is
# judged against the median of its own last ATTN_WINDOW earlier probes that showed nothing,
# from its ATTN_MIN_HISTORY-th probe on, and a change counts only when it holds on ATTN_SUSTAIN
# probes in a row (dated at the first). With the defaults nothing can be reported before a
# module's fourth probe: its first two fill the baseline, the third is the first one judged,
# and the fourth confirms it. The capture probes on every `probe_every`-th step it commits,
# counting from 1, so with 1,000 that is the 4,000th step of a run's first attempt (the
# incident is then dated at the third probe, the 3,000th). Until then the attention kinds are
# `checked` without enough history (`kind_checks`), which does not degrade the run.
# Calibrated on SmolLM2-360M fine-tunes (seeded and healthy probes)
# and on tiny Llamas trained from scratch on CPU; see the plan's decision of Nov 16, 2026.
ATTN_MIN_HISTORY = 2
ATTN_WINDOW = 5
ATTN_SUSTAIN = 2
ATTN_COLLAPSE_RATIO = 0.5  # the layer's mean entropy share at or below this times its baseline
ATTN_UNIFORM = 0.98  # the layer's mean entropy share at or above this (1.0 is uniform) ...
ATTN_UNIFORM_RISE = 0.2  # ... having risen at least this much above its baseline
ATTN_LOGIT_FACTOR = 2.0  # the layer's largest pre-softmax logit at or above this times its baseline
ATTN_KINDS = ("attention_collapse", "attention_uniform", "attention_logit_growth")
ATTN_MODULES = 20  # modules listed in one incident; `affected` counts them all
# A parameter is reported as not updating once its weights are exactly unchanged on this
# many weight checks in a row while it has a gradient. Checks run every 50 steps by default.
STALL_STEPS = 3
STALL_ONSETS = 20  # onsets listed in one incident; `onsets_omitted` counts the rest
# Loss scaling (first guesses): GradScaler skips a few steps in a row at the start while it
# halves its initial 65,536 scale, so only a longer run of skips, or a scale below 1 (the
# gradients overflow even unscaled), is reported.
SCALE_SKIP_RUN = 10
SCALE_FLOOR = 1.0
MISSING_COLOR = (48, 54, 64)
DEAD_COLOR = (236, 84, 99)
OVERLOADED_COLOR = (255, 190, 94)
STARVED_COLOR = (170, 120, 240)
# An expert with no tokens in this sample that no detector flagged, for example a
# transient gap in a healthy layer.
IDLE_COLOR = (150, 158, 170)


def _is_non_finite(value: Any) -> bool:
    """A recorded NaN or infinity (a float, or the sink's string). An integer too large for a float
    is not one: it is finite, and unreadable where a float is wanted, as the Rust detectors read it
    (`detect/extra_json.rs`), so it raises nothing and is no `non_finite` value (wave AO, BUG-25)."""
    if isinstance(value, str):
        return value in ("nan", "inf", "-inf")
    if isinstance(value, float):
        return not math.isfinite(value)
    return False


# What the typed columns of `flight.py` carry (wave AJ, lane AJ-2; OPEN_BUGS.md section 15, A4, A5,
# A9). A map entry the column cannot hold is not read on its record by either engine: in Rust it
# never reaches the column (lane AJ-1), and here it is skipped where it is read. The rest of the
# map is judged as before. The records of `load_records` are JSON, so a tuple is never among them;
# a tuple given to these readers directly is not a list, as `flight.py` refuses it, so the SDK in
# either mode reads the same records alike. No reader here ever promised a tuple.
_INT64 = (-(2**63), 2**63 - 1)
_WIRE_NON_FINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}  # as JsonlSink writes them
_UNCARRIED = object()


def layer_entries(record: Mapping[str, Any]) -> list[tuple[Any, Mapping[str, Any]]]:
    """The layers a record holds, `(name, stats)` in stored order: the entries of its `layers`
    that are objects (WAVE_AL_CONTRACT.md section 1.5). An entry that is not an object (`5`, a
    null struct in protocol 5's column) and a `layers` value that is not an object (`[1, 2]`,
    `null`) are absent to every detector, as Rust reads them, and are never raised on."""
    layers = record.get("layers")
    if not isinstance(layers, Mapping):
        return []
    return [(name, stats) for name, stats in layers.items() if isinstance(stats, Mapping)]


def _recorded_diversity(stats: Any) -> float | None:
    """A router's recorded `diversity` when it is a finite number, else None, so the rule takes it
    from the counts (WAVE_AL_CONTRACT.md section 1.5): as Rust's `Number` reads it, a bool, a
    string (the sink's `"nan"` among them), `null`, a list, an object and a value that is not
    finite are absent, and so is the whole entry when it is not an object."""
    value = stats.get("diversity") if isinstance(stats, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:  # an integer past a float's range reads as infinite in Rust: not finite
        return None
    return number if math.isfinite(number) else None


def _router_counts(value: Any) -> list[int] | None:
    """A router's counts as the `moe_counts` column carries them, a list of signed 64-bit integers
    (a bool is not one), else None: that router is then absent from the record, as Rust sees it."""
    if not isinstance(value, list):
        return None
    low, high = _INT64
    if all(isinstance(c, int) and not isinstance(c, bool) and low <= c <= high for c in value):
        return value
    return None


def _router_total(value: Any) -> int | None:
    """The sum of a router's counts when the column carries them (`_router_counts`), else None, in one
    pass over the counts (lane AF-W: the read rule reads a read router's counts once)."""
    if not isinstance(value, list):
        return None
    low, high = _INT64
    total = 0
    for c in value:
        if not isinstance(c, int) or isinstance(c, bool) or not low <= c <= high:
            return None
        total += c
    return total


def _wire_float(value: Any) -> Any:
    """One value as a `float64` column carries it (`flight.py`'s `_float`): a number, None for a
    null, a non-finite float for the sink's "nan", "inf" and "-inf"; `_UNCARRIED` for anything else,
    a bool, other text and an integer too large for a float among them."""
    if value is None:
        return None
    if isinstance(value, bool):
        return _UNCARRIED
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except OverflowError:
            return _UNCARRIED
    if isinstance(value, str) and value in _WIRE_NON_FINITE:
        return _WIRE_NON_FINITE[value]
    return _UNCARRIED


def _carried(value: Any) -> bool:
    """Whether a `float64` column carries `value` (`_wire_float`): a number, or the sink's "nan",
    "inf" or "-inf". A null is no value, and so is anything the column cannot carry (a bool, other
    text), which Rust never sees (its column is null): such a field is absent to every rule."""
    found = _wire_float(value)
    return found is not None and found is not _UNCARRIED


def _gradient_key(records: Iterable[Mapping[str, Any]]) -> str:
    """The run's total gradient norm series: the pre-clipping `clip_grad_norm` once any record holds
    one the column carries (`_carried`), else `grad_norm`, as Rust switches (`Gradients::row`,
    `measure.rs`). A record with `clip_grad_norm` that holds no value does not say the run clips: an
    explicit null switched the whole run off `grad_norm` here and not in Rust (BUG-09)."""
    return "clip_grad_norm" if any(_carried(r.get("clip_grad_norm")) for r in records) else "grad_norm"


def _spread_values(value: Any) -> list[float] | None:
    """One parameter's gathered spread as the `replica_spread` column carries it (WAVE_AJ_CONTRACT.md,
    section 1), a list of floats, a null read as NaN as Rust reads it; else None, and the entry is
    absent. A value that is not finite stays one (`non_finite`); a value that is not a number leaves
    the entry out. `[]` is held and says nothing: it is judged as no drift and covers no parameter."""
    if not isinstance(value, list):
        return None
    values = [_wire_float(v) for v in value]
    if any(v is _UNCARRIED for v in values):
        return None
    return [math.nan if v is None else v for v in values]


def _held_spread_table(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """A record's `replica_spread` when the column holds it: a map with at least one held entry, or
    an empty map. A map whose every entry is refused reads as no `replica_spread` (WAVE_AJ_CONTRACT.md,
    section 2: the column is null, and Rust reads no spread)."""
    table = _spread_table(record)
    if table is None or (table and all(_spread_values(v) is None for v in table.values())):
        return None
    return table


def _object(record: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    """`record[field]` when it is an object, else an empty one. A field outside the identity fields
    that is not what its column holds is absent to the rules that read it, and never raised on
    (WAVE_AF_CONTRACT.md section 1; BUG-12): Rust's column is null for it."""
    value = record.get(field)
    return value if isinstance(value, Mapping) else {}


def _unchanged_names(record: Mapping[str, Any]) -> list[str] | None:
    """A record's `weights_unchanged` as its column holds it, a list of names, or None: a value that
    is not a list of strings (`true`, `[1]`) is no weight check, as Rust reads it (BUG-12)."""
    names = record.get("weights_unchanged")
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return names
    return None


def _weight_check(record: Mapping[str, Any]) -> bool:
    """Whether a record is a weight check as both engines count one: `weight_norms` with at least
    one norm the column carries (a null counts, as Rust counts the entry)."""
    norms = record.get("weight_norms")
    return isinstance(norms, Mapping) and any(_wire_float(v) is not _UNCARRIED for v in norms.values())


def _held_update_sizes(record: Mapping[str, Any]) -> bool:
    """Whether a record carries `update_rms` as both engines count it: a map with at least one entry
    its column holds (`_wire_float`; a null counts, as Rust's map column keeps the entry). A map whose
    every entry is refused (a text that is not the sink's spelling, a bool, an integer too large for
    a float) is an absent key (WAVE_AJ_CONTRACT.md section 2; AQ5-D1): the column is null in Rust."""
    sizes = record.get("update_rms")
    return isinstance(sizes, Mapping) and any(_wire_float(v) is not _UNCARRIED for v in sizes.values())


def effective_overload_threshold(
    n_experts: int,
    routed_tokens: int,
    base_threshold: float = OVERLOAD_THRESHOLD,
) -> float:
    """Calibrate the overload share threshold for router size and sample count.

    Keeps `base_threshold` (0.35) for large routers and token counts (such as the 128-expert
    bake-off and 8-expert toy run), while raising the bar when `1 / n_experts` is already
    near or above `base_threshold` (e.g. 2- or 3-expert routers) or when `routed_tokens` is
    small enough that multinomial noise is large.
    """
    if n_experts <= 1 or routed_tokens <= 0:
        return 1.0
    mean_share = 1.0 / n_experts
    se_share = math.sqrt(mean_share * (1.0 - mean_share) / routed_tokens)
    return max(base_threshold, 1.5 * mean_share, mean_share + 3.0 * se_share)


def _dead_experts_significant(
    n_experts: int,
    routed_tokens: int,
    overloaded: list[int],
    dead_count: int = 0,
    diversity: float = 1.0,
) -> bool:
    if n_experts <= 1 or routed_tokens <= 0 or dead_count <= 0:
        return False
    if overloaded:
        return True
    expected_empty = n_experts * ((1.0 - 1.0 / n_experts) ** routed_tokens)
    if expected_empty >= 0.01:
        return False
    return (dead_count / n_experts) >= 0.25 or diversity < 0.5


def _mark_starved(samples: list[dict[str, Any]], window: int, ratio: float) -> None:
    """Set `sample["starved"]` from each router's trailing window of `window` samples."""
    _StarvedMarks(window, ratio).feed(samples)


class _StarvedMarks:
    """`_mark_starved`'s state: each router's trailing window, so that the samples of later records
    can be marked as they come (`feed`, in step order), as `_GradSpikes` is fed (lane AD11). A sample
    is marked when it is fed and never again: the window ends at it."""

    __slots__ = ("window", "ratio", "layers")

    def __init__(self, window: int, ratio: float) -> None:
        self.window, self.ratio = window, ratio
        # Each router's samples in the window, the sums of their counts, and their tokens.
        self.layers: dict[Any, tuple[list[dict[str, Any]], list[int], int]] = {}

    def feed(self, samples: Iterable[dict[str, Any]]) -> None:
        window, ratio, layers = self.window, self.ratio, self.layers
        for sample in samples:  # in step order
            sample["starved"] = []
            if sample.get("missing") or sample["routed_tokens"] <= 0:
                continue
            recent, sums, tokens = layers.get(sample["layer"], ([], [], 0))
            counts = sample["counts"]
            if len(sums) != len(counts):
                recent, sums, tokens = [], [0] * len(counts), 0
            recent.append(sample)
            sums = [a + b for a, b in zip(sums, counts)]
            tokens += sample["routed_tokens"]
            if len(recent) > window:
                old = recent.pop(0)
                sums = [a - b for a, b in zip(sums, old["counts"])]
                tokens -= old["routed_tokens"]
            layers[sample["layer"]] = (recent, sums, tokens)
            even = 1.0 / len(counts)
            if len(recent) < window or tokens * even < STARVE_MIN_EXPECTED:
                continue
            dead = set(sample["dead"])
            sample["starved"] = [
                expert
                for expert, total in enumerate(sums, 1)
                if total / tokens < ratio * even and expert not in dead
            ]
            sample["starved_shares"] = {
                str(expert): sums[expert - 1] / tokens for expert in sample["starved"]
            }


def detect_starved_experts(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return every starved expert with the first step it was flagged, or None."""
    state = _StarvedFirsts()
    state.feed(samples)
    return state.result()


class _StarvedFirsts:
    """`detect_starved_experts`'s state: the first step each expert was flagged, fed samples in
    step order (lane AD11)."""

    __slots__ = ("first",)

    def __init__(self) -> None:
        self.first: dict[tuple[int, int], dict[str, Any]] = {}

    def feed(self, samples: Iterable[Mapping[str, Any]]) -> None:
        first = self.first
        for sample in samples:
            for expert in sample.get("starved", ()):
                key = (sample["layer"], expert)
                if key not in first:
                    first[key] = {
                        "layer": sample["layer"],
                        "router": sample.get("router"),
                        "expert": expert,
                        "step": sample["step"],
                        "window_share": sample["starved_shares"][str(expert)],
                    }

    def result(self) -> dict[str, Any] | None:
        if not self.first:
            return None
        experts = sorted(copy.deepcopy(list(self.first.values())), key=lambda e: (e["window_share"], e["layer"], e["expert"]))
        return {"kind": "starved_experts", "step": min(e["step"] for e in experts), "experts": experts}


def _top_k_entries(record: Mapping[str, Any]) -> Iterable[tuple[Any, int]]:
    """A record's `moe_top_k` entries the column carries: an integer that is not a bool and fits int32
    (WAVE_AJ_CONTRACT.md, section 1). Any other entry, and a value that is not an object, is absent."""
    given = record.get("moe_top_k")
    if not isinstance(given, Mapping):
        return ()
    return [(name, k) for name, k in given.items() if isinstance(k, int) and not isinstance(k, bool) and -(2**31) <= k < 2**31]


def _fixed_top_k(routers: Iterable[str], override: int | Mapping[str, int] | None) -> dict[str, int]:
    """The routers whose experts per token `override` fixes, with the value: every router for one
    value, the named ones for a value per router. A fixed router's recorded values are not read."""
    if override is None:
        return {}
    if isinstance(override, int):
        return {name: override for name in routers}
    return {str(name): int(k) for name, k in override.items()}


def _router_top_k(
    records: list[dict[str, Any]], routers: list[str], override: int | Mapping[str, int] | None
) -> dict[str, int]:
    """Each router's experts per token at the end of the run: the last `moe_top_k` recorded for it,
    or `override`, else 1 (the evidence's `top_k`). Each step is judged with the value in force at
    that step (`_MoeSamples`), as Rust judges it (BUG-13).
    An entry the `moe_top_k` column cannot carry (anything but an integer that is not a bool and fits
    int32, WAVE_AJ_CONTRACT.md, section 1) is absent on its record, as Rust never sees it (wave AJ:
    before, `int(k)` raised on text and read 2.0 as 2)."""
    found: dict[str, int] = {}
    for record in records:
        for name, k in _top_k_entries(record):
            found[name] = k
    if isinstance(override, int):
        found = {name: override for name in routers}
    elif override is not None:
        found.update({name: int(k) for name, k in override.items()})
    return {name: found.get(name, 1) for name in routers}


def _jump_overloaded(
    counts: list[int], k: int, diversity: float, window: list[tuple[list[int], int, float]]
) -> list[int]:
    """In a router whose diversity just dropped, the experts that took the extra tokens."""
    if len(window) < OVERLOAD_MIN_HISTORY:
        return []
    if diversity >= DIVERSITY_DROP * _median([d for _, _, d in window]):
        return []
    tokens = sum(counts) / k
    base_tokens = sum(total for _, total, _ in window) / k
    if tokens <= 0 or base_tokens <= 0:
        return []
    overloaded = []
    for expert, count in enumerate(counts, 1):
        base = sum(c[expert - 1] for c, _, _ in window if expert - 1 < len(c)) / base_tokens
        if count / tokens - base >= OVERLOAD_JUMP:
            overloaded.append(expert)
    return overloaded


def _newest_stamp(records: Iterable[Mapping[str, Any]]) -> float | None:
    """The latest `ts` of `records` that is a finite number (`_finite`), or None: the time of the
    last recorded step that a Flight Recorder dump is compared with. A bool, text and an integer too
    large for a float are no time (wave AO)."""
    stamps = [stamp for stamp in (_finite(r.get("ts")) for r in records) if stamp is not None]
    return max(stamps) if stamps else None


def _finite(value: Any) -> float | None:
    """A finite number, else None. A bool is not a number (DET-16): an imported `true` is not a
    measurement of 1. Flags such as `step_skipped` are read as flags, not through here. An integer
    too large for a float is None too, as `_recorded_diversity` reads it: unreadable where a float
    is wanted, as the Rust detectors read it (wave AO, BUG-25)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


SKIPPED_BY = ("caller", "scaler", "both")


def skipped_by(record: Mapping[str, Any]) -> str | None:
    """Who reported that a record's update was skipped (STEP_AND_EVIDENCE_CONTRACT.md, section 1):
    `caller` (`session.step(skipped=True)`), `scaler` (the loss scaler given to `init`) or
    `both`; None when the update was not skipped. A skipped record without `step_skipped_by`
    was written before the key, when only a scaler could write it, and is read as `scaler`; so
    is a value that is not one of the three."""
    if not record.get("step_skipped"):
        return None
    by = record.get("step_skipped_by")
    return by if by in SKIPPED_BY else "scaler"


def partial_step(record: Mapping[str, Any]) -> bool:
    """Whether a record is a step on which the scaler skipped some optimizers and applied others
    (WAVE_AE_CONTRACT.md, section 4): a skip the scaler confirmed (`skipped_by` `scaler` or
    `both`) whose `step_skipped_part` is `{"skipped": n, "of": m}`, two whole numbers with
    0 < n < m. The key is the scaler's verdict, so on a skip only the caller reported, and on an
    applied step, it says nothing; a key that states no mixed outcome (every optimizer skipped,
    none, or not two counts) leaves the record a full skip."""
    if skipped_by(record) not in ("scaler", "both"):
        return False
    part = record.get("step_skipped_part")
    if not isinstance(part, Mapping):
        return False
    skipped, of = part.get("skipped"), part.get("of")
    return type(skipped) is int and type(of) is int and 0 < skipped < of


PARTIAL_STEP_WORDS = "{n} step(s) on which the scaler skipped some optimizers and applied others"


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


# The score beside a ratio (DETECTOR_RULES_CONTRACT.md, part 1): how far a value is from the
# median of the earlier values its ratio was taken against, in units of their spread. A
# description only: no rule reads it.
SCORE_MIN_HISTORY = 10  # earlier finite values a score needs
SCORE_SCALE = 1.4826  # the median absolute deviation to a standard deviation, for normal data
SCORE_NOTES = ("short_history", "no_dispersion")
_SCORE_WORDS = {
    "short_history": f"no score: fewer than {SCORE_MIN_HISTORY} earlier values",
    # Not "the earlier values did not vary": six equal values and four others have a zero MAD.
    "no_dispersion": "no score: the earlier values had zero median absolute deviation",
}


def robust_score(value: float, window: Iterable[Any]) -> dict[str, Any]:
    """`{"score": number}`, or `{"score": None, "score_note": why}`: `(value - median) / (1.4826 *
    MAD)` over the finite values of `window`, which never holds `value` itself. With fewer than
    SCORE_MIN_HISTORY finite values the note is `short_history`; with a MAD of 0 (more than half the
    window one value) nothing is divided and the note is `no_dispersion`."""
    earlier = [v for v in (_finite(x) for x in window) if v is not None]
    if len(earlier) < SCORE_MIN_HISTORY:
        return {"score": None, "score_note": "short_history"}
    centre = _median(earlier)
    spread = _median([abs(v - centre) for v in earlier])
    if spread == 0:
        return {"score": None, "score_note": "no_dispersion"}
    return {"score": (value - centre) / (SCORE_SCALE * spread)}


def score_words(entry: Mapping[str, Any]) -> str | None:
    """What a reader shows for an entry's score: "score 31.2", or why there is none. None only
    when the entry has no `score` key at all (evidence written before it): a null score is
    never shown as 0, and its words are never left out."""
    if not isinstance(entry, Mapping) or "score" not in entry:
        return None
    score = _finite(entry.get("score"))
    if score is not None:
        return f"score {score:.1f}"
    return _SCORE_WORDS.get(entry.get("score_note"), "no score")  # type: ignore[arg-type]


def _with_score(entry: Mapping[str, Any]) -> str:
    """" (score 31.2)" after a printed ratio, or "" for an entry without the key."""
    words = score_words(entry)
    return f" ({words})" if words else ""


def scalar_context(
    records: list[dict[str, Any]], step: int, *, window: int = SPIKE_WINDOW
) -> dict[str, dict[str, Any]]:
    """Values logged with `session.log` at `step`, each with the median of its finite
    values over the previous `window` records, so an incident shows what the loss did.
    A bool logged as a scalar is left out (DET-16): it is not a measurement, and the row sent
    to the Rust detectors, whose scalars are numbers, cannot carry it either. So is every other
    entry the `scalars` column cannot hold (`_wire_float`: a text other than the sink's "nan",
    "inf" and "-inf", an integer too large for a float, a list, an object) and a name Arrow cannot
    encode, and its siblings are kept, as Rust reads them (AQ5-D7, the author, Nov 26, 2026;
    `WAVE_AJ_CONTRACT.md` section 3). Before, such an entry was shown with its raw value. A null is
    no value, as in the median before it, and is left out too, as Rust's context skips the column's
    null (seed 3569, filed under AQ5-D7; before, it was shown as a value of None)."""
    index = next((i for i, record in enumerate(records) if record.get("step") == step), None)
    if index is None:
        return {}
    context = {}
    for name, value in _object(records[index], "scalars").items():
        if value is None or _wire_float(value) is _UNCARRIED or not _encodable(name):
            continue
        before = [
            v for v in (_finite(_object(r, "scalars").get(name)) for r in records[max(0, index - window) : index])
            if v is not None
        ]
        context[name] = {"value": value, "median_before": _median(before) if before else None}
    return context


def _encodable(name: Any) -> bool:
    """Whether a map's name is text Arrow can hold as a key: a string without a lone surrogate."""
    if not isinstance(name, str):
        return False
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


class _Baseline:
    """A parameter's last `window` gradient norms that were not part of an excursion, with
    their lowest value kept beside them: a norm below `factor` times the lowest cannot reach
    `factor` times the median, so most steps need no median at all."""

    __slots__ = ("values", "low", "count", "start")

    def __init__(self) -> None:
        self.values: deque[float] = deque()
        self.low = math.inf
        self.count = 0  # recorded steps in a row at or above factor times the median
        self.start: dict[str, Any] | None = None  # the first of them

    def keep(self, norm: float, window: int) -> None:
        self.values.append(norm)
        if norm < self.low:
            self.low = norm
        if len(self.values) > window and self.values.popleft() <= self.low:
            self.low = min(self.values)


def detect_grad_spikes(
    records: list[dict[str, Any]],
    *,
    factor: float = SPIKE_FACTOR,
    window: int = SPIKE_WINDOW,
    min_history: int = SPIKE_MIN_HISTORY,
    sustain: int = SPIKE_SUSTAIN,
) -> dict[str, Any] | None:
    """Return the first gradient-norm spike and the parameters that spiked most, or None.

    Two rules, and the incident is the first one either of them establishes (the total's,
    when both do on the same record). Once reported it is not replaced or dated again.

    The total: a step whose total gradient norm is at least `factor` times its median over
    the previous `window` recorded steps. Each parameter's norm is then compared with its own
    median over the same window, so the report names where the spike is, not only that the
    total jumped. `rising_since` is the first step in the window at which that parameter was
    already above half the spike factor, which points at where a spike started when it builds
    over several steps.

    One parameter (`"trigger": "parameter"`): a parameter whose own norm stays at least
    `factor` times its own median for `sustain` recorded steps in a row, whatever the total
    does. A small parameter cannot move the total (one holding 0.2% of it, times 100, moves
    it 1.02 times), and a single step that high is ordinary noise, so the rule asks for it to
    last; the incident is dated at the first of those steps and is established `sustain - 1`
    recorded steps later, unless the total has spiked by then, in which case that incident
    already names the parameter with `rising_since`. Norms that are part of such an
    excursion stay out of that parameter's median, so an excursion that lasts is still
    measured against what came before it. `grad_norm`, `baseline` and `ratio` are the
    total's at the step the incident is dated at.

    Without per-parameter norms (router-only capture) no parameter is named and only the
    total is judged. The total is the pre-clipping `clip_grad_norm` whenever the run clips,
    since per-parameter norms are read after clipping, and `grad_norm` otherwise.
    """
    # One series for the whole run. When the run clips, the pre-clipping norm is used:
    # the per-parameter norms are read after clipping, so a clipped spike stays flat there.
    key = _gradient_key(records)
    state = _GradSpikes(key, factor=factor, window=window, min_history=min_history, sustain=sustain)
    state.feed(records)
    return state.result()


def _window_store(window: int) -> Any:
    """Where a rule keeps the values it reads `[-window:]` of: the last `window` of them (lane AD11:
    the whole run was kept). A `window` below 1 reads a slice that is not a trailing window
    (`values[-0:]` is all of them), so then everything is kept, as before."""
    return deque(maxlen=window) if window > 0 else []


class _GradSpikes:
    """`detect_grad_spikes`'s state, fed records in step order (`feed`, any number of times, each
    batch at or after the steps fed before) and read by `result`. The function feeds every record
    at once; the `--watch` path (`_Incremental`) feeds each look's new records to the state it kept.

    Only what the rules read is kept (lane AD11): the last `window + 1` totals (`baseline` and the
    score read `[-window:]`, and the parameter path's `total_window` the `window` before the
    newest), each parameter's last `window` norms (`_spiking_parameters`), its `_Baseline`, the
    count of totals for `min_history`, and the first 20 spike steps with their count."""

    def __init__(self, key: str, *, factor: float = SPIKE_FACTOR, window: int = SPIKE_WINDOW,
                 min_history: int = SPIKE_MIN_HISTORY, sustain: int = SPIKE_SUSTAIN) -> None:
        self.key, self.factor, self.window, self.min_history, self.sustain = key, factor, window, min_history, sustain
        self.totals: Any = deque(maxlen=window + 1) if window > 0 else []
        self.total_count = 0  # every total kept, for `min_history`
        self.per_param: dict[str, Any] = {}
        self.baselines: dict[str, _Baseline] = {}
        self.first: dict[str, Any] | None = None
        self.spike_count = 0
        self.spike_steps: list[int] = []  # the first 20

    def feed(self, records: Iterable[Mapping[str, Any]]) -> None:
        key, factor, window, min_history, sustain = self.key, self.factor, self.window, self.min_history, self.sustain
        per_param, baselines = self.per_param, self.baselines
        for record in sorted(records, key=lambda r: r["step"]):
            if record.get("step_skipped"):
                continue  # a loss-scaler overflow step: its gradients are Inf or NaN by design
            total = _finite(record.get(key))
            if total is None:
                continue
            step = int(record["step"])
            totals = list(self.totals)
            baseline = _median(totals[-window:]) if self.total_count >= min_history else None
            if baseline is not None and baseline > 0 and total >= factor * baseline:
                self.spike_count += 1
                if len(self.spike_steps) < 20:
                    self.spike_steps.append(step)
                if self.first is None:
                    self.first = {
                        "step": step,
                        "grad_norm": total,
                        "baseline": baseline,
                        "ratio": total / baseline,
                        # Against the same earlier totals as `baseline` (part 1 of the rules contract).
                        **robust_score(total, totals[-window:]),
                        "parameters": _spiking_parameters(record, per_param, factor, window),
                    }
            self.totals.append(total)
            self.total_count += 1
            totals.append(total)
            lasted: list[dict[str, Any]] = []
            for name, value in _object(record, "grad_norms").items():
                norm = _finite(value)
                if norm is None:
                    continue
                past = per_param.get(name)
                if past is None:
                    past = per_param[name] = _window_store(window)
                past.append((step, norm))
                own = baselines.get(name)
                if own is None:
                    own = baselines[name] = _Baseline()
                if len(own.values) >= min_history and norm >= factor * own.low:
                    usual = _median(list(own.values))
                    if usual > 0 and norm >= factor * usual:
                        if own.count == 0:
                            own.start = {
                                "step": step, "total": total, "total_baseline": baseline,
                                # The totals `baseline` was taken over: those before this one.
                                "total_window": totals[-(window + 1):-1],
                                # Scored against the parameter's own window, which its `baseline` is.
                                "parameter": {"name": name, "grad_norm": norm, "baseline": usual,
                                              "ratio": norm / usual, **robust_score(norm, own.values),
                                              "rising_since": step},
                            }
                        own.count += 1
                        if own.count == sustain and own.start is not None:
                            lasted.append(own.start)
                        continue
                own.count = 0
                own.keep(norm, window)
            if lasted and self.first is None:
                # Usually every excursion that lasts to this record began on the same step; if
                # not, the incident is the earliest, with the parameters that began there.
                began = min(start["step"] for start in lasted)
                starts = [start for start in lasted if start["step"] == began]
                parameters = sorted((start["parameter"] for start in starts), key=lambda p: (-p["ratio"], p["name"]))
                usual = starts[0]["total_baseline"]
                self.first = {
                    "step": began,
                    "grad_norm": starts[0]["total"],
                    "baseline": usual,
                    "ratio": starts[0]["total"] / usual if usual else None,
                    # A score only beside a ratio, and over the totals that ratio's baseline was taken over.
                    **(robust_score(starts[0]["total"], starts[0]["total_window"]) if usual else {}),
                    "parameters": parameters[:10],
                    "trigger": "parameter",
                    "sustained": sustain,
                }

    def result(self) -> dict[str, Any] | None:
        """The incident `detect_grad_spikes` returns for the records fed so far, a new copy each time."""
        if self.first is None:
            return None
        return copy.deepcopy({"kind": "grad_spike", **self.first, "spike_count": self.spike_count,
                              "spike_steps": list(self.spike_steps)})


def detect_update_spikes(
    records: list[dict[str, Any]], *, window: int = UPDATE_WINDOW, alpha: float = UPDATE_ALPHA
) -> dict[str, Any] | None:
    """Return the first jump in the size of the optimizer's update and the parameters whose
    own update moved most, or None.

    Each record's `update_rms` gives the RMS of Adam's update to each parameter. A record is
    compared with the last one before it that could be judged, in the same attempt, and `R` is
    the largest update over the parameters both of them name, on each side (a parameter recorded
    from a later step on is not a jump, L13): `change` is how far `R` moved (either way) and
    `rise` how far the total gradient norm rose (0 when it fell), the totals taken over the
    parameters both records name where their `grad_norms` differ (`_shared_totals`). Two
    records that share no parameter are not compared. The first `window` changes are not judged. After that
    `baseline` is the mean of the last `window` changes that were not flagged, and a record is
    flagged when `rise > 0` and `change > baseline / (alpha * sqrt(rise))` (Altenbernd et al.,
    arXiv 2604.00726, eq. 13). A flagged change stays out of the baseline, so a fault that lasts
    is still measured against what came before it; every other change joins it, whether or not
    the gradient norm rose. The paper's mean is taken once, over the learning-rate warm-up.

    The total is the pre-clipping `clip_grad_norm` whenever the run clips and `grad_norm`
    otherwise, as for `grad_spike`. Left out, and never used as the record before: a step the
    loss scaler skipped (the optimizer's state did not move), a record without `update_rms`
    or without a finite total, and values that are not finite. A total that is not finite is
    `non_finite`'s to report, not this rule's.

    `parameters` names up to ten parameters by how far their own RMS update moved from the
    record before, largest first, so the incident says where the jump is. `lr` and `lr_before`
    are the learning rates of the flagged record and of the record it was compared with, each
    only when that record carries `lr`.

    Separately clipped groups (WAVE_AE_CONTRACT.md, section 1): a pair whose totals would be
    split by parameter (`_split_totals`) is not compared when either record says it was clipped
    by more than one `clip_grad_norm_` call (`clip_calls`, `_clip_calls`), since the split
    assumes one call scaled every gradient by one factor. Nothing is flagged on it, its change
    joins no baseline, and its later record is still the record the next is compared with. A
    pair that needs no split is compared as before, whatever its `clip_calls`. The pairs not
    compared are said by `update_spike_not_checked`.
    """
    return _update_spikes_judged(records, window=window, alpha=alpha)[0]


def _clip_calls(record: Mapping[str, Any]) -> int | None:
    """A record's `clip_calls` (WAVE_AE_CONTRACT.md, section 1): the number of `clip_grad_norm_`
    calls of its step when that is a whole number above 1, else None. An absent key means one
    call, or a count that is not known (under `torch.compile`, or a capture written before the
    key); a value that is not a whole number above 1 (1, text, a fraction, a boolean, and a float
    such as 2.0, which the capture never writes) is read as no key."""
    calls = record.get("clip_calls")
    return calls if type(calls) is int and calls > 1 else None


CLIP_GROUPS_REASON = (
    "clipped by more than one clip_grad_norm_ call: the pre-clipping total cannot be split by parameter,"
    " so {n} comparison(s) of update_spike were not made"
)
# With no pair compared (WAVE_AE_CONTRACT.md, section 1, the author's second word): "the reader must
# make clear that zero were judged".
CLIP_GROUPS_NONE_REASON = (
    "clipped by more than one clip_grad_norm_ call: the pre-clipping total cannot be split by parameter,"
    " so {n} comparison(s) of update_spike were not made; none of the comparisons of update_spike was made, so nothing was judged"
)


def update_spike_not_checked(not_compared: Mapping[str, int] | None) -> dict[str, Any] | None:
    """The `clip_groups` entry of `not_checked` (WAVE_AE_CONTRACT.md, section 1), from what
    `_update_spikes_judged` gives as its second value: `comparisons` the pairs of `update_spike`
    that were not made because either record was clipped by more than one call and the pair
    needed a per-parameter split, `compared` the pairs that were made (those in the first
    `window`, which are not judged, among them), and `step` the later record of the first pair
    not made. None when every pair was made. While `compared` is above zero `update_spike` stays
    `checked` (`_PARTIAL_CHECKS`); at zero it reads `unavailable` with this entry's reason, which
    then says that none was made."""
    if not not_compared:
        return None
    count, compared = int(not_compared["comparisons"]), int(not_compared["compared"])
    reason = (CLIP_GROUPS_REASON if compared else CLIP_GROUPS_NONE_REASON).format(n=count)
    return {"check": "clip_groups", "kinds": ["update_spike"], "comparisons": count, "compared": compared,
            "step": int(not_compared["step"]), "reason": reason}


def _update_spikes_judged(
    records: list[dict[str, Any]], *, window: int = UPDATE_WINDOW, alpha: float = UPDATE_ALPHA
) -> tuple[dict[str, Any] | None, dict[str, int] | None]:
    """`detect_update_spikes`'s incident, and the pairs it did not compare because a split total
    met more than one clip call: `{"comparisons", "step", "compared"}`, with `compared` the pairs
    it did compare, or None when every pair was compared."""
    key = _gradient_key(records)
    # attempt, total, each parameter's update size, each parameter's gradient norm (or None), the
    # learning rates the record carries (or None), its clip calls above 1 (or None)
    before: tuple[Any, float, dict[str, float], Mapping[str, Any] | None, list[float] | None, int | None] | None = None
    not_compared: dict[str, int] | None = None
    compared = 0  # the pairs compared, judged or among the first `window`
    recent: deque[float] = deque(maxlen=window)  # the last `window` changes that were not flagged
    first: dict[str, Any] | None = None
    spike_count, spikes = 0, []  # every flagged record counted; the first 20 kept (lane AD11)
    for record in sorted(records, key=lambda r: r["step"]):
        if record.get("step_skipped"):
            continue
        sizes = record.get("update_rms")
        total = _finite(record.get(key))
        if not isinstance(sizes, Mapping) or total is None:
            continue
        own = {str(name): size for name, size in ((n, _finite(v)) for n, v in sizes.items()) if size is not None}
        if not own:
            continue
        attempt = record.get("attempt", 0)
        grads = record.get("grad_norms") if isinstance(record.get("grad_norms"), Mapping) else None
        # R on both sides is the largest update over the parameters both records name (L13, Nov 20,
        # 2026): a parameter recorded from a later step on (unfrozen after `init`), or one that is
        # left out of this record, is not a jump. Records that share no parameter are not compared,
        # and their change joins no baseline. The gradient totals are taken over the parameters
        # both name too (`_shared_totals`).
        shared = [name for name in own if name in before[2]] if before is not None and before[0] == attempt else []
        calls = _clip_calls(record)
        split = _split_totals(total, grads, before[1], before[3]) if shared else None
        if shared and split is not None and (calls is not None or before[5] is not None):
            # The split assumes one clip call scaled every gradient by one factor (WAVE_AE_CONTRACT.md,
            # section 1): with more than one the pair is not compared, and its change joins no baseline.
            if not_compared is None:
                not_compared = {"comparisons": 0, "step": int(record["step"])}
            not_compared["comparisons"] += 1
        elif shared:
            compared += 1
            largest, was = max(own[name] for name in shared), max(before[2][name] for name in shared)
            now, then = split if split is not None else (total, before[1])
            change, rise = abs(largest - was), max(0.0, now - then)
            if len(recent) < window:
                recent.append(change)
            else:
                baseline = math.fsum(recent) / window
                threshold = baseline / (alpha * math.sqrt(rise)) if rise > 0 and baseline > 0 else None
                if threshold is None or change <= threshold:
                    recent.append(change)
                else:
                    step = int(record["step"])
                    spike_count += 1
                    if len(spikes) < 20:
                        spikes.append(step)
                    if first is None:
                        moved = sorted(
                            ({"name": name, "update_rms": own[name], "previous": before[2][name],
                              "change": abs(own[name] - before[2][name])}
                             for name in shared),
                            key=lambda p: (-p["change"], p["name"]),
                        )
                        lr, lr_before = _lr_list(record), before[4]
                        first = {
                            "step": step, "update_rms": largest, "previous": was, "change": change,
                            "baseline": baseline, "threshold": threshold, "grad_norm": now,
                            "grad_norm_before": then, "grad_rise": rise, "parameters": moved[:10],
                            # The learning rates at the flagged record and at the one it was compared
                            # with, each only where that record carries them (rules contract, part 4).
                            # The rule does not read them.
                            **({"lr": lr} if lr is not None else {}),
                            **({"lr_before": lr_before} if lr_before is not None else {}),
                        }
        before = (attempt, total, own, grads, _lr_list(record), calls)
    if not_compared is not None:
        not_compared["compared"] = compared
    if first is None:
        return None, not_compared
    return {"kind": "update_spike", **first, "spike_count": spike_count, "spike_steps": spikes}, not_compared


def _lr_list(record: Mapping[str, Any]) -> list[float] | None:
    """A record's `lr` list (one rate a parameter group), or None when it carries none, or one
    that is not a list of finite numbers."""
    lr = record.get("lr")
    if not isinstance(lr, list) or not lr:
        return None
    values = [_finite(v) for v in lr]
    return None if any(v is None for v in values) else values  # type: ignore[return-value]


def _shared_totals(
    total: float, grads: Mapping[str, Any] | None, before: float, grads_before: Mapping[str, Any] | None
) -> tuple[float, float]:
    """The gradient totals of two records compared by `update_spike`, over the parameters both
    name (L13): each record's total times the share of its gradient those parameters carry,
    `total * sqrt(shared² / all²)` over its `grad_norms`. For a total that is `grad_norm`, the
    root of the squares of `grad_norms`, that is the shared parameters' own norm; for a
    pre-clipping `clip_grad_norm` it is theirs before clipping, since clipping scales every
    gradient by one factor and the per-parameter norms are read after it.

    The whole totals when both records name the same parameters (so no record set whose
    parameters never change is judged otherwise), and when they cannot be split: a record
    without `grad_norms`, a norm there that is not a finite number, no shared parameter, or a
    record whose norms are all 0. The server's rule is the same (`shared_totals`, detect.rs)."""
    split = _split_totals(total, grads, before, grads_before)
    return (total, before) if split is None else split


def _split_totals(
    total: float, grads: Mapping[str, Any] | None, before: float, grads_before: Mapping[str, Any] | None
) -> tuple[float, float] | None:
    """`_shared_totals` when it splits the totals by parameter, and None when it uses them whole.
    A split is a split whatever its factor, 1 included (a parameter whose norm is 0): whether the
    pair needs one is read from the records' `grad_norms`, not from the numbers it gives."""
    if grads is None or grads_before is None or grads.keys() == grads_before.keys():
        return None
    shared = [name for name in grads if name in grads_before]
    sides = []
    for side, norms in ((total, grads), (before, grads_before)):
        values = [_finite(v) for v in norms.values()]
        if not shared or any(v is None for v in values):
            return None
        whole = math.fsum(v * v for v in values)
        if whole <= 0:
            return None
        part = math.fsum(norm * norm for norm in (_finite(norms[name]) for name in shared))
        sides.append(side * math.sqrt(part / whole))
    return sides[0], sides[1]


def update_not_read_not_checked(records: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The `update_size` entry of `not_checked`: optimizers whose update size was asked for
    (`init(..., updates=True)`) and could not be read.

    The update size is read from Adam's and AdamW's own state. Any other optimizer, one that
    steps copies of the weights, or a loop whose optimizer hooks are off and that never calls
    `session.record_updates`, leaves no `update_rms`, and the capture says why once, under
    `update_not_read` on the first record it wrote after it knew. Without this entry such a
    run would read "no update spike", which is not "update size checked". The server writes
    the same entry (detect.rs).
    """
    first: dict[str, tuple[int, str]] = {}
    for record in records:
        said = record.get("update_not_read")
        if not isinstance(said, Mapping):
            continue
        for name, why in said.items():
            first.setdefault(str(name), (int(record["step"]), str(why)))
    if not first:
        return None
    names = sorted(first)
    return {
        "check": "update_size",
        "kinds": ["update_spike"],
        "optimizers": names,
        "reason": "; ".join(f"{name}: {first[name][1]}" for name in names),
        "step": min(step for step, _ in first.values()),
    }


# What a gradient that was not read does to each kind is decided by what the kind is judged from
# (STEP_AND_EVIDENCE_CONTRACT.md, section 2, which replaces lane Z2's call that made all three
# `unavailable`): `grad_spike` is judged from the total gradient norm, which leaves the unread
# parameter out, so it is `unavailable`. `non_finite` (layer statistics and each norm that was
# read) stays `checked`, and `stalled_weights` (each parameter's own history) is `checked` for
# the parameters read and names the unread ones under `parameters_not_judged`.
GRADS_NOT_READ_KINDS = ("grad_spike",)
GRADS_NOT_READ_NAMED = "stalled_weights"


def grads_not_read_not_checked(records: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The `gradients` entry of `not_checked` (L23): parameters whose gradient could not be read.

    The capture leaves such a gradient out of `grad_norms` on its own (one on `meta`, a nested
    one, or one whose norm raised) and says which parameter, and why, once, under
    `grads_not_read` on the first record it wrote after it knew (`record_grads`). Without this
    entry a kind would read "nothing found" for a parameter that was never looked at. The kinds
    the entry lists (`GRADS_NOT_READ_KINDS`) are `unavailable`; `stalled_weights` stays checked
    and names the parameters it did not judge (`name_gradients_not_judged`); a finding on what
    was read always stands; and the run is `degraded`, as for any entry. This is the Python
    detectors' rule; under the Rust ones the server's report gives the same entry, and
    `build_evidence` takes it as the server wrote it (wave AF: no second pass adds this one).
    """
    first: dict[str, tuple[int, str]] = {}
    for record in records:
        said = record.get("grads_not_read")
        if not isinstance(said, Mapping):
            continue
        for name, why in said.items():
            first.setdefault(str(name), (int(record["step"]), str(why)))
    if not first:
        return None
    names = sorted(first)
    return {
        "check": "gradients",
        "kinds": list(GRADS_NOT_READ_KINDS),
        "parameters": names,
        "reason": "gradient not read for " + "; ".join(f"{name}: {first[name][1]}" for name in names),
        "step": min(step for step, _ in first.values()),
    }


def name_gradients_not_judged(checks: dict[str, dict[str, Any]], not_checked: Iterable[Mapping[str, Any]]) -> None:
    """Name the parameters whose gradient was not read on the `stalled_weights` entry of
    `checks`, as `parameters_not_judged`, when a `gradients` entry of `not_checked` lists them
    and the kind is not itself `unavailable` (STEP_AND_EVIDENCE_CONTRACT.md, section 2)."""
    names = sorted({str(name) for item in not_checked if isinstance(item, Mapping) and item.get("check") == "gradients"
                    for name in item.get("parameters") or ()})
    check = checks.get(GRADS_NOT_READ_NAMED)
    if names and isinstance(check, dict) and check.get("value") != "unavailable":
        check["parameters_not_judged"] = names


# `layers_not_read` (DETECTOR_RULES_CONTRACT.md, part 7): modules whose output the capture cannot
# read at all, said once for the run. With some modules read (`coverage` "partial") the activation
# kinds stay checked for those and name the others under `modules_not_judged`; with none read on any
# record ("none") they are `unavailable`. A module read on some records and not on others is not
# named (v1 does not represent an intermittent gap): it is judged on the records that hold it.
LAYERS_NOT_READ_KINDS = ("activation_spike", "activation_drift", "dead_activations")


def layers_not_read_not_checked(records: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The `layers` entry of `not_checked`: the modules the capture said it could not read, under
    `layers_not_read` (module to why), with `coverage` "partial" when some record carried layer
    statistics and "none" when none did, and then the activation kinds under `kinds`."""
    first: dict[str, tuple[int, str]] = {}
    read = False
    for record in records:
        # A record holds a layer only through an entry that is an object (row S5, the owner's
        # approved change, Nov 24): one whose only entries are not objects read no layer, as Rust reads it.
        read = read or bool(layer_entries(record))
        said = record.get("layers_not_read")
        if not isinstance(said, Mapping):
            continue
        for name, why in said.items():
            first.setdefault(str(name), (int(record["step"]), str(why)))
    if not first:
        return None
    names = sorted(first)
    return {
        "check": "layers",
        "coverage": "partial" if read else "none",
        **({} if read else {"kinds": list(LAYERS_NOT_READ_KINDS)}),
        "modules": names,
        "reason": "layer not read: " + "; ".join(f"{name}: {first[name][1]}" for name in names),
        "step": min(step for step, _ in first.values()),
    }


def name_modules_not_judged(checks: dict[str, dict[str, Any]], not_checked: Iterable[Mapping[str, Any]]) -> None:
    """Name the modules the capture could not read on each activation kind's entry of `checks`,
    as `modules_not_judged`, when a `layers` entry of `not_checked` lists them with coverage
    "partial" and the kind is not itself `unavailable`, as `stalled_weights` names parameters."""
    names = sorted({str(name) for item in not_checked if isinstance(item, Mapping) and item.get("check") == "layers"
                    and item.get("coverage") == "partial" for name in item.get("modules") or ()})
    for kind in LAYERS_NOT_READ_KINDS:
        check = checks.get(kind)
        if names and isinstance(check, dict) and check.get("value") != "unavailable":
            check["modules_not_judged"] = names


# `moe_source` (part 7): how each router's counts reached the capture (`form`) and where in the model
# they were taken (`point`). A reader says "dispatched" only when `point` says so.
MOE_SOURCE_WORDS = {
    "requested": "the assignments the router asked for",
    "dispatched": "the dispatched assignments",
    "unknown": "the assignments the model gave; whether a capacity limit was applied before this point is not known",
}


def moe_source(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]] | None:
    """Each router's `moe_source` as the capture said it, once (the first record that names the
    router), or None when no record carries the key (a capture from before it)."""
    found: dict[str, dict[str, Any]] = {}
    for record in records:
        said = record.get("moe_source")
        if not isinstance(said, Mapping):
            continue
        for router, source in said.items():
            if isinstance(source, Mapping) and str(router) not in found:
                found[str(router)] = {key: source[key] for key in ("form", "point") if key in source}
    return found or None


def moe_source_words(evidence: Mapping[str, Any]) -> dict[str, str]:
    """What a reader says each router's counts are, from the evidence's `moe_source`: by `point`,
    and "the assignments the model gave" with "not known" for any point but `requested` and
    `dispatched`. Empty when the evidence has no `moe_source`: a reader makes no claim."""
    found = evidence.get("moe_source")
    if not isinstance(found, Mapping):
        return {}
    return {str(router): MOE_SOURCE_WORDS.get(source.get("point"), MOE_SOURCE_WORDS["unknown"])  # type: ignore[arg-type]
            for router, source in found.items() if isinstance(source, Mapping)}


def _spiking_parameters(
    record: dict[str, Any],
    per_param: Mapping[str, Iterable[tuple[int, float]]],
    factor: float,
    window: int,
) -> list[dict[str, Any]]:
    found = []
    for name, value in _object(record, "grad_norms").items():
        norm = _finite(value)
        past = list(per_param.get(name, ()))[-window:]
        if norm is None or len(past) < 2:
            continue
        baseline = _median([v for _, v in past])
        if baseline <= 0 or norm < factor * baseline:
            continue
        rising = int(record["step"])
        for past_step, past_norm in reversed(past):
            if past_norm >= 0.5 * factor * baseline:
                rising = past_step
            else:
                break
        found.append(
            {"name": name, "grad_norm": norm, "baseline": baseline,
             "ratio": norm / baseline, **robust_score(norm, [v for _, v in past]), "rising_since": rising}
        )
    found.sort(key=lambda p: (p["rising_since"], -p["ratio"]))
    return found[:10]


def _weight_kind(name: str) -> str:
    lowered = name.lower()
    if "embed" in lowered or lowered.endswith(("wte.weight", "wpe.weight")):
        return "embedding"
    if "norm" in lowered or ".ln" in lowered or lowered.startswith("ln"):
        return "norm"
    if lowered.endswith(".bias") or lowered == "bias":
        return "bias"
    return "matrix"


def weight_norm_trends(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Each kind of parameter's float64 norm now, relative to its first recorded norm.

    Numbers only, with no threshold and no incident: steady growth, and shrinkage under
    weight decay, are normal in a healthy run, and a cutoff for "too much" needs real
    failing runs to set (OLMo 2's embedding shrinkage is the case to look for). Parameters
    are grouped by name into embeddings, norms, biases and matrices. Returns None with
    fewer than two weight checks.
    """
    # A weight check as `weight_jumps` counts one (`_weight_check`): a `weight_norms` that is not an
    # object, or holds no norm the column carries, is none, so the last check is the last that holds one.
    checked = [r for r in sorted(records, key=lambda r: r["step"]) if _weight_check(r)]
    if len(checked) < 2:
        return None
    first, last = checked[0], checked[-1]
    kinds: dict[str, list[tuple[str, float]]] = {}
    for name, norm in last["weight_norms"].items():
        base, now = _finite(first["weight_norms"].get(name)), _finite(norm)
        if base is None or now is None or base <= 0:
            continue
        kinds.setdefault(_weight_kind(name), []).append((name, now / base))
    if not kinds:
        return None
    summary: dict[str, Any] = {}
    for kind, ratios in sorted(kinds.items()):
        values = sorted(v for _, v in ratios)
        lowest = min(ratios, key=lambda item: item[1])
        highest = max(ratios, key=lambda item: item[1])
        summary[kind] = {
            "parameters": len(ratios),
            # The middle value, or the mean of the two middle ones (lane U7, Nov 19, 2026: the upper one).
            "median_ratio": statistics.median(values),
            "lowest": {"name": lowest[0], "ratio": lowest[1]},
            "highest": {"name": highest[0], "ratio": highest[1]},
        }
    return {"baseline_step": int(first["step"]), "latest_step": int(last["step"]), "kinds": summary}


# Four measurements with no incident and no threshold (DETECTOR_RULES_CONTRACT.md, part 6). Each is
# a block of numbers in one rank's evidence, as `weight_trends` is: present when the records hold
# what it reads, read by no rule, changing no run state. Nothing is judged from them in this wave.
TREND_MIN_POINTS = 6  # points a trend needs
MEASUREMENTS_LISTED = 10  # entries a measurement lists
MEASUREMENT_BLOCKS = ("gradient_trend", "attention_trends", "update_trends", "weight_jumps")


def robust_trend(steps: list[int], values: list[float]) -> dict[str, Any] | None:
    """The robust local trend of a series (written again from the description of
    LLM-Training-Debugger's `robust_local_trend`): a line through the median of the first third of
    the values and the median of the last third, each placed at the median of its own step
    numbers, which are the series' actual steps. `growth` is the last third's median over the
    first's, `per_100_steps` the slope over 100 steps as a share of the first. None under
    TREND_MIN_POINTS points, or when the two thirds sit at one step."""
    n = len(values)
    if n < TREND_MIN_POINTS:
        return None
    k = max(1, n // 3)
    try:  # a step is any integer of 0 or more; one past a float's range has no place on a line
        left_step, right_step = _median([float(v) for v in steps[:k]]), _median([float(v) for v in steps[-k:]])
    except OverflowError:
        return None
    left, right = _median(values[:k]), _median(values[-k:])
    if right_step - left_step <= 0:
        return None
    slope = (right - left) / (right_step - left_step)
    return {"points": n, "first_step": steps[0], "last_step": steps[-1], "first": left, "last": right,
            "growth": right / left if left > 0 else None,
            "per_100_steps": 100 * slope / left if left > 0 else None}


def gradient_trend(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The trend of the total gradient norm (gap B4): the pre-clipping `clip_grad_norm` when the
    run clips, `grad_norm` otherwise, as `grad_spike` reads it; a skipped step and a value that
    is not finite are left out. `series` names the norm read. None under TREND_MIN_POINTS."""
    rows = sorted(records, key=lambda r: r["step"])
    key = _gradient_key(rows)
    pairs = [(int(r["step"]), value) for r in rows if not r.get("step_skipped")
             and (value := _finite(r.get(key))) is not None]
    found = robust_trend([step for step, _ in pairs], [value for _, value in pairs])
    return {"series": key, **found} if found is not None else None


def attention_trends(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The trend of each attention module's largest logit, probe by probe (gap Q1): `modules`
    with a trend, and `listed`, the MEASUREMENTS_LISTED with the largest `growth`, each with its
    `module`. None when no module has TREND_MIN_POINTS probes."""
    series: dict[str, tuple[list[int], list[float]]] = {}
    for record in sorted(records, key=lambda r: r["step"]):
        probe = record.get("probe")
        attention = probe.get("attention") if isinstance(probe, Mapping) else None
        if not isinstance(attention, Mapping):
            continue
        for module, stats in attention.items():
            _, logit, _ = _attention_values(stats)
            if logit is not None:
                steps, values = series.setdefault(str(module), ([], []))
                steps.append(int(record["step"]))
                values.append(logit)
    found = [{"module": module, **trend} for module, (steps, values) in series.items()
             if (trend := robust_trend(steps, values)) is not None]
    if not found:
        return None
    found.sort(key=lambda t: (-(t["growth"] if t["growth"] is not None else -math.inf), t["module"]))
    return {"statistic": "max_logit", "modules": len(found), "listed": found[:MEASUREMENTS_LISTED]}


def update_trends(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Each parameter's lowest update against its own earlier ones (gap Q2): `update_rms` over the
    median of the parameter's ACT_WINDOW records before, from SCORE_MIN_HISTORY earlier finite
    values on. `parameters` with an update size, `judged`, and `lowest`: the MEASUREMENTS_LISTED
    whose update fell furthest, each with `name`, `ratio`, `step`, `baseline` and `update_rms`.
    None when no record carries `update_rms` on an update that was applied.

    A skipped update (`step_skipped` true, whoever reported it) is left out: it enters no baseline and
    gives no ratio (wave AJ, A7, the author's decision of Nov 22). A partial skip (`step_skipped_part`)
    is left out whole: the record counts the optimizers the scaler skipped and names none, so no
    parameter's update can be told applied there, and an ambiguous parameter is not judged on that
    step, as `update_spike` reads it (WAVE_AE_CONTRACT.md, section 4)."""
    # Each parameter's last ACT_WINDOW values, which is all the rule reads (lane AD11: the whole run
    # was kept), and never fewer than SCORE_MIN_HISTORY, so that its length still counts them.
    past: dict[str, deque[float]] = {}
    lowest: dict[str, dict[str, Any]] = {}
    for record in sorted(records, key=lambda r: r["step"]):
        sizes = record.get("update_rms")
        if not isinstance(sizes, Mapping) or record.get("step_skipped"):
            continue
        for name, value in sizes.items():
            now = _finite(value)
            if now is None:
                continue
            name = str(name)
            earlier = past.get(name)
            if earlier is None:
                earlier = past[name] = deque(maxlen=max(ACT_WINDOW, SCORE_MIN_HISTORY))
            if len(earlier) >= SCORE_MIN_HISTORY:
                base = _median(list(earlier)[-ACT_WINDOW:])
                if base > 0 and (name not in lowest or now / base < lowest[name]["ratio"]):
                    lowest[name] = {"name": name, "ratio": now / base, "step": int(record["step"]),
                                    "baseline": base, "update_rms": now}
            earlier.append(now)
    if not past:
        return None
    return {"parameters": len(past), "judged": len(lowest),
            "lowest": sorted(lowest.values(), key=lambda p: (p["ratio"], p["name"]))[:MEASUREMENTS_LISTED]}


def weight_jumps(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The largest change of each tensor's norm between two consecutive weight checks (gap Q3),
    either way: `checks`, `parameters` compared, and `largest`, the MEASUREMENTS_LISTED largest,
    each with `name`, `ratio`, `step`, `previous_step`, `before` and `after`. None with fewer than
    two weight checks, or no tensor with a positive finite norm at two of them.

    A weight check is a record whose `weight_norms` holds at least one norm the column carries
    (`_weight_check`), as Rust counts the rows (wave AJ, A9: a map of text alone was a check here
    and not in Rust).

    Two checks are compared only within one attempt: the first check after a reload is the new
    baseline, and the change across the reload is no jump (wave AJ, A10, the author's decision of
    Nov 22). `checks` still counts every check. `weight_norm_trends` reads across reloads as before."""
    checks = [r for r in sorted(records, key=lambda r: r["step"]) if _weight_check(r)]
    if len(checks) < 2:
        return None
    best: dict[str, dict[str, Any]] = {}
    for before, after in zip(checks, checks[1:]):
        if before.get("attempt") != after.get("attempt"):
            continue
        for name, value in after["weight_norms"].items():
            a, b = _finite(before["weight_norms"].get(name)), _finite(value)
            if a is None or b is None or a <= 0 or b <= 0:
                continue
            ratio = max(b / a, a / b)
            name = str(name)
            if name not in best or ratio > best[name]["ratio"]:
                best[name] = {"name": name, "ratio": ratio, "step": int(after["step"]),
                              "previous_step": int(before["step"]), "before": a, "after": b}
    if not best:
        return None
    return {"checks": len(checks), "parameters": len(best),
            "largest": sorted(best.values(), key=lambda p: (-p["ratio"], p["name"]))[:MEASUREMENTS_LISTED]}


def measurement_blocks(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The four measurements of one rank's records, each only where the records hold what it reads."""
    found = {"gradient_trend": gradient_trend(records), "attention_trends": attention_trends(records),
             "update_trends": update_trends(records), "weight_jumps": weight_jumps(records)}
    return {name: block for name, block in found.items() if block is not None}


HEAD_NUMBERS = ("norm", "mean", "log_z", "gap", "gap_relative")


def head_logit_trends(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The probe's language-model head numbers over the run (`probe` records' `head`): for
    each number its first and latest value and the lowest and highest in between.

    Numbers only, with no threshold and no incident. The logit norm grows in healthy AdamW
    runs too (Chen et al., arXiv 2610.38277, p. 2), the mean has drifted both ways in
    published runs, so it is signed and given as values, not ratios, and a cutoff for the
    BF16-to-FP32 `gap` needs real failing runs to set. Probes that did not read the head
    are counted under `not_read` with the first one's step and reason. Returns None when
    no probe carries a `head` entry (no probe, a failed one, or `probe_head=False`).

    `overflow_positions` is there when a probe counted positions whose FP32 logits are finite
    but whose sums overflow (the probe's `overflow_positions`, which leaves them out of the four
    numbers, apart from positions with a NaN or an Inf): how many probes did, how many positions
    in all, the most in one probe, and the first and latest step of one, over every probe that
    has a `head` entry, read or not (C22, Nov 19, 2026; until then the count was in the records
    only, so a number taken over fewer positions than the probe saw did not say so). A count is an
    integer above 0 that is an `i64`, as `data_checks._int` reads one (the author, Nov 27, 2026).
    """
    seen: list[tuple[int, dict[str, Any]]] = []
    for record in sorted(records, key=lambda r: r["step"]):
        probe = record.get("probe")
        if isinstance(probe, dict) and "error" not in probe and isinstance(probe.get("head"), dict):
            seen.append((int(record["step"]), probe["head"]))
    if not seen:
        return None
    read = [(step, head) for step, head in seen if "not_read" not in head]
    trends: dict[str, Any] = {"probes": len(seen), "read": len(read)}
    if read:
        trends.update(module=read[-1][1].get("module"), first_step=read[0][0], latest_step=read[-1][0])
        numbers: dict[str, Any] = {}
        for name in HEAD_NUMBERS:
            series = [(step, value) for step, head in read if (value := _finite(head.get(name))) is not None]
            if series:
                values = [value for _, value in series]
                numbers[name] = {"first": series[0][1], "latest": series[-1][1], "lowest": min(values),
                                 "highest": max(values), "probes": len(series)}
        trends["numbers"] = numbers
    skipped = [(step, head) for step, head in seen if "not_read" in head]
    if skipped:
        trends["not_read"] = {"probes": len(skipped), "first_step": skipped[0][0],
                              "reason": str(skipped[0][1]["not_read"])}
    overflowed = [(step, count) for step, head in seen
                  if isinstance(count := head.get("overflow_positions"), int) and not isinstance(count, bool)
                  and 0 < count < 2**63]
    if overflowed:
        trends["overflow_positions"] = {
            "probes": len(overflowed), "positions": sum(count for _, count in overflowed),
            "most": max(count for _, count in overflowed),
            "first_step": overflowed[0][0], "latest_step": overflowed[-1][0],
        }
    return trends


def _watchdog_beside(source: Any) -> Path | None:
    if not isinstance(source, (str, Path)) or not Path(source).is_dir():
        return None
    directory = Path(source).resolve()
    status = directory / "watchdog.json"
    if rank_number(directory.name) is not None and not status.is_file():
        return directory.parent / status.name
    return status


def read_watchdog(source: str | Path | Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The server's progress status (`watchdog.json`), or None when there is none to read.

    A missing, half-written or unrecognised file is None, never an error: the status is
    rewritten every few seconds, and a reader may meet it mid-replace.
    """
    if source is None:
        return None
    if isinstance(source, Mapping):
        document: Any = dict(source)
    else:
        try:
            document = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    if not isinstance(document, dict) or document.get("schema_version") != 1 or "ranks" not in document:
        return None
    return document


def _library_file(name: str) -> bool:
    return "site-packages" in name or "dist-packages" in name or "/lib/python" in name or name.startswith("<")


def stuck_at(rank: Mapping[str, Any]) -> dict[str, Any] | None:
    """Where a stopped rank's main thread was at its last heartbeat, from the stack its process
    sent (see `Heartbeat`): the innermost frame, the nearest frame that is not a library's
    (`caller`), for how long the stack had not changed, and whether the call was reached
    through one of TensorQuill's own functions (`in_tensorquill`). None when the rank sent no stack. It is where Python was; a call into
    C or CUDA shows as the Python line that made it."""
    stack = rank.get("stack")
    threads = stack.get("threads") if isinstance(stack, Mapping) else None
    if not isinstance(threads, list):
        return None
    frames = next((t.get("frames") for t in threads if isinstance(t, Mapping) and t.get("main") is True), None)
    if not isinstance(frames, list):
        return None
    frames = [f for f in frames if isinstance(f, Mapping) and isinstance(f.get("file"), str)]
    if not frames:
        return None

    def brief(frame: Mapping[str, Any]) -> dict[str, Any]:
        name = frame.get("function")
        return {"function": name if isinstance(name, str) else "?", "file": frame["file"], "line": frame.get("line")}

    place: dict[str, Any] = {"rank": rank.get("rank"), **brief(frames[0])}
    own = next((f for f in frames if f.get("own") is True), None)
    caller = next((f for f in frames[1:] if not _library_file(f["file"]) and f.get("own") is not True), None)
    if caller is not None:
        place["caller"] = brief(caller)
    if own is not None:
        place["in_tensorquill"] = brief(own)
    for key in ("unchanged_seconds", "still_seconds"):
        if isinstance(stack.get(key), (int, float)) and not isinstance(stack[key], bool):
            place[key] = stack[key]
    return place


# Grouping the stopped ranks' stacks (the all-ranks view's `stopped_stacks`).
#
# What "the same stack" means: the main thread's frames, every frame the heartbeat kept (at most
# `flight.STACK_FRAMES`, innermost first), compared frame by frame on
#   - the function name;
#   - the file, as the path after the last `site-packages/` or `dist-packages/`, else its base
#     name: ranks on other hosts may have other install prefixes and checkout folders, and the
#     heartbeat keeps only a path's last 200 characters;
#   - the line, for frames of the job's own code and TensorQuill's, and not for a library's or
#     Python's: in the job's code two lines are two different calls (a `loss.item()` and a
#     `barrier()`), while a library's line numbers differ between versions, and the hosts of one
#     job may not all have the same patch release. The cost: two ranks blocked at two lines of
#     the same library function, reached the same way, are one group. (Measured Nov 18, 2026 on
#     `thread_stacks` itself, 12 samples each: a main thread in a `queue.get(timeout=...)` loop,
#     `Event.wait()`, an `Event.wait(timeout)` loop and a `time.sleep` loop each kept one line in
#     every sample, so the line is not dropped because waits move; they did not.)
# The whole kept depth, not the innermost few frames: two ranks in the same collective reached
# from different places in the job (one from the forward, one from the optimizer step) are a
# mismatch worth seeing, and the outer frames all ranks share never split a group. A stack
# deeper than the kept depth is compared on what was kept. Other threads are left out: their
# frames are cut to four and they differ by timing (the NCCL watchdog, pin-memory threads).
STACK_ODD_SHARE = 0.25  # an odd group: one rank, or at most this share of the ranks with a stack
STACK_GROUPS_SHOWN = 8
_STOPPED_STATES = ("silent", "lost", "unresponsive")


def rank_is_stopped(rank: Any) -> bool:
    """Whether a rank's entry in a server status is a stopped rank: the one rule of every reader
    here, and `is_stopped` in the server's `watchdog.rs` (shared cases:
    `tests/fixtures/wave_ah/cases.json`).

    `silent`, `lost` and `unresponsive` are stopped. So is a `reconnecting` rank with a severity:
    the server gives one only to a rank that was already stopped when its stream dropped, and
    says what it was in `stopped_as` (wave AH, P1-1, Nov 22, 2026: until then a hung rank that was
    killed read as a healthy rank reconnecting for the whole silence floor). A `reconnecting` rank
    with no severity was healthy when its stream dropped and is not stopped."""
    if not isinstance(rank, Mapping):
        return False
    state = rank.get("state")
    return state in _STOPPED_STATES or (state == "reconnecting" and rank.get("severity") in ("warning", "critical"))
_STACK_CAVEATS = (
    "an observation of where each stopped rank's main thread was, never the cause: the papers disagree on whether"
    " the odd rank is the culprit (Mycroft takes it as the stuck one; ByteRobust treats ranks outside the largest"
    " group as suspects to check)",
    "only each rank's main process: a rank stuck in a data-loader worker or another child process shows the main"
    " process waiting for it",
    "each rank's last heartbeat stack, not one snapshot taken at once; and one snapshot cannot name a slow rank,"
    " whose stack looks like everyone's",
)


def _frame_key(frame: Mapping[str, Any]) -> tuple[Any, ...]:
    name = str(frame.get("file", "")).replace("\\", "/")
    for marker in ("site-packages/", "dist-packages/"):
        if marker in name:
            name = name.rsplit(marker, 1)[1]
            break
    else:
        name = name.rsplit("/", 1)[-1]
    library = _library_file(str(frame.get("file", ""))) and frame.get("own") is not True
    return (frame.get("function"), name, None if library else frame.get("line"))


def _main_frames(rank: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    stack = rank.get("stack")
    threads = stack.get("threads") if isinstance(stack, Mapping) else None
    if not isinstance(threads, list):
        return None
    frames = next((t.get("frames") for t in threads if isinstance(t, Mapping) and t.get("main") is True), None)
    if not isinstance(frames, list):
        return None
    frames = [f for f in frames if isinstance(f, Mapping) and isinstance(f.get("file"), str)]
    return frames or None


def stack_signature(rank: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """What two stopped ranks must share to be in one group (see above), or None with no stack."""
    frames = _main_frames(rank)
    return tuple(_frame_key(f) for f in frames) if frames else None


def group_stopped_stacks(status: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The stopped ranks of the server's status (silent, lost, unresponsive) grouped by where
    their main thread was, and the odd group named when one is clearly apart. None when no rank
    is stopped.

    `verdict` is one of `no_stacks` (no stopped rank sent one: an unresponsive rank never does),
    `one_stack`, `two_stacks_differ` (two ranks that differ: nothing says which is odd),
    `all_alike`, `odd_group` (exactly two groups and the smaller is one rank, or at most a quarter
    of the ranks with a stack: `odd_group` indexes `groups`) or `no_clear_odd_group` (an even
    split, or more than two places). The ranks that sent no stack are always listed: one of them
    may be the rank that matters. `caveats` say what this is not."""
    if not status:
        return None
    stopped = [r for r in status.get("ranks", ()) if rank_is_stopped(r)]
    if not stopped:
        return None
    by_place: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    without: list[Any] = []
    for rank in sorted(stopped, key=lambda r: (not isinstance(r.get("rank"), int), r.get("rank"))):
        signature = stack_signature(rank)
        if signature is None:
            without.append(rank.get("rank"))
        else:
            by_place.setdefault(signature, []).append(rank)
    with_stack = [r.get("rank") for members in by_place.values() for r in members]
    ordered = sorted(by_place.values(), key=lambda members: (-len(members), members[0].get("rank")))
    groups = []
    for members in ordered:
        place = stuck_at(members[0]) or {}
        ranks = [m.get("rank") for m in members]
        groups.append({
            "ranks": ranks, "rank_ranges": _rank_ranges(ranks),
            "where": {k: place[k] for k in ("function", "file", "line") if k in place},
            **({"caller": place["caller"]} if "caller" in place else {}),
            **({"in_tensorquill": place["in_tensorquill"]} if "in_tensorquill" in place else {}),
            "frames": len(_main_frames(members[0]) or ()),
        })
    count = len(with_stack)
    odd = None
    if count == 0:
        verdict = "no_stacks"
    elif count == 1:
        verdict = "one_stack"
    elif len(groups) == 1:
        verdict = "all_alike"
    elif count == 2:
        verdict = "two_stacks_differ"
    elif len(groups) == 2 and len(groups[1]["ranks"]) < len(groups[0]["ranks"]) and (
            len(groups[1]["ranks"]) == 1 or len(groups[1]["ranks"]) <= STACK_ODD_SHARE * count):
        verdict, odd = "odd_group", 1
    else:
        verdict = "no_clear_odd_group"
    return {
        "stopped_ranks": sorted((r.get("rank") for r in stopped), key=lambda n: (not isinstance(n, int), n)),
        "ranks_with_stack": sorted(with_stack),
        "ranks_without_stack": without,
        "verdict": verdict,
        "odd_group": odd,
        "groups": groups,
        "same_stack": "the main thread's kept frames (innermost first), each compared on function, file (after"
                      " site-packages, else the base name) and line (the job's and TensorQuill's own code only)",
        "caveats": list(_STACK_CAVEATS),
    }


def stopped_stacks_line(found: Mapping[str, Any] | None) -> str | None:
    """The stack grouping in a sentence or two, for the summary text, or None."""
    if not isinstance(found, Mapping):
        return None
    groups = found.get("groups") or []
    without = found.get("ranks_without_stack") or []
    tail = ""
    if without:
        tail = (f" Rank{'s' if len(without) != 1 else ''} {_rank_ranges(without, 8)} sent no stack"
                " (no heartbeat, stopped for less than `stack_seconds`, or unresponsive), so"
                f" {'they are' if len(without) != 1 else 'it is'} not compared.")

    def place(group: Mapping[str, Any]) -> str:
        text = _where(group.get("where") or {})
        if group.get("caller"):
            text += f", called from {_where(group['caller'])}"
        return text

    verdict = found.get("verdict")
    if verdict == "no_stacks":
        return "Stacks: no stopped rank sent its stack, so nothing is said about where they are." + tail
    if verdict == "one_stack":
        return (f"Stacks: only rank {groups[0]['rank_ranges']} sent a stack ({place(groups[0])}), so there is nothing to"
                " compare it with and no rank is named." + tail)
    if verdict == "all_alike":
        return (f"Stacks: all {len(groups[0]['ranks'])} stopped ranks that sent one are at the same place"
                f" ({place(groups[0])}); no rank stands apart." + tail)
    if verdict == "two_stacks_differ":
        return ("Stacks: the two stopped ranks that sent one are at different places"
                f" (rank {groups[0]['rank_ranges']}: {place(groups[0])}; rank {groups[1]['rank_ranges']}: {place(groups[1])});"
                " with two there is no telling which is the odd one, so none is named." + tail)
    if verdict == "odd_group":
        main, odd = groups[0], groups[found.get("odd_group", 1)]
        many = len(odd["ranks"]) != 1
        return (f"Stacks: rank{'s' if many else ''} {odd['rank_ranges']} {'are' if many else 'is'} somewhere else than the"
                f" other {len(main['ranks'])} stopped rank{'s' if len(main['ranks']) != 1 else ''}: {place(odd)}, where"
                f" ranks {main['rank_ranges']} are in {place(main)}. An observation, not the cause: the papers disagree"
                " on whether the odd rank is the culprit, a rank stuck in a data-loader worker shows only its main"
                " process's stack, and one snapshot cannot name a slow rank." + tail)
    shown = "; ".join(f"rank{'s' if len(g['ranks']) != 1 else ''} {g['rank_ranges']}: {place(g)}"
                      for g in groups[:3])
    more = len(groups) - 3
    return (f"Stacks: the stopped ranks are at {len(groups)} places ({shown}"
            + (f"; and {more} more" if more > 0 else "") + "); no one group stands clearly apart, so none is named." + tail)


def detect_watchdog_incidents(status: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Turn the server's progress status into incidents.

    `job_stalled`: no rank is making progress. `rank_silent`: some ranks are silent, lost
    (their stream dropped and did not return) or never connected while others progress.
    `duplicate_batches` (see `data_checks.detect_duplicate_batches`) is the server's comparison of
    the ranks' batch hashes, whatever the summary.
    The others are observations of the stream the server receives, and the cause is not known:
    a hang, a long evaluation or checkpoint pause, a crashed launcher and a dead telemetry
    sender all look the same. The server's `--silent-seconds` is the pause it tolerates.
    `stuck_at` lists, for each stopped rank whose process sent its stack, where its main
    thread was (see `stuck_at`); it narrows the cause and does not change the assessment.

    `assessment` labels a stall: `lost_connection` when a rank's stream dropped and did not
    come back (a killed or crashed process, an out-of-memory kill among them, or a network
    failure) and that explains the stall: every stopped rank is lost, or the whole job has
    stopped (`job_stalled`) and at least one rank is lost, since ranks of a job that stops
    together wait for the one that is gone (a collective cannot finish without it).
    `rank_silent` also lists `other_ranks`: the ranks that are not stopped, each with its state,
    last step and attempt, so a reader can tell that a rank which stopped has stepped on since.
    `suspected_hang` when a stopped rank is still connected and no lost rank explains it,
    a lost rank beside a silent one while other ranks step among them. `lost_ranks` names the
    lost ranks whenever there are any. `ranks_never_connected` when no rank stopped. The server
    builds the same incidents (`progress_incidents` in `watchdog.rs`).
    The stacks themselves stay in the status file: an incident's `ranks` are without them.
    Each carries its `class`, as the server's `watchdog.json` writes it.
    """
    duplicates = status.get("duplicate_batches") if status else None
    found = [{"kind": "duplicate_batches", **duplicates}] if isinstance(duplicates, Mapping) else []
    found = _progress_incidents(status) + found
    with_classes(found)
    return found


def _missing_omitted(status: Mapping[str, Any] | None) -> int:
    """How many missing ranks the status counts and does not list (`missing_ranks_omitted`, written by
    a server since wave AR, lane AR-4, only when some are left out), or 0."""
    found = status.get("missing_ranks_omitted") if status else None
    return found if isinstance(found, int) and not isinstance(found, bool) and found > 0 else 0


def _missing_counted(status: Mapping[str, Any] | None) -> list[int]:
    """The ranks a status counts as missing and does not list, as the server's `run_state` rebuilds
    them: every rank not seen below the larger of its world size and its highest connected rank plus
    one, at most `MAX_EXPECTED_RANKS`, less the listed ones. Empty for a status with no count, so such
    a status reads as before."""
    if not _missing_omitted(status):
        return []
    connected = {r.get("rank") for r in status.get("ranks") or () if isinstance(r, Mapping)}
    numbers = [r for r in connected if isinstance(r, int) and not isinstance(r, bool) and 0 <= r < MAX_EXPECTED_RANKS]
    world = status.get("world_size")
    below = world if isinstance(world, int) and not isinstance(world, bool) and 0 < world <= MAX_EXPECTED_RANKS else 0
    below = max(below, max(numbers, default=-1) + 1)
    listed = set(status.get("missing_ranks") or ())
    return [rank for rank in range(below) if rank not in connected and rank not in listed]


def _progress_incidents(status: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not status or status.get("summary") not in ("stalled", "degraded"):
        return []
    full = list(status.get("ranks", ()))
    if not status.get("missing_ranks") and not any(rank_is_stopped(r) for r in full):
        # No rank stopped and none is missing: the summary's cause is not progress, as when a rank
        # whose capture stopped is beside an active one (contract 3). As `progress_incidents` in
        # watchdog.rs (lane W2).
        return []
    places = [place for place in map(stuck_at, (r for r in full if rank_is_stopped(r)))
              if place is not None]
    ranks = [{k: v for k, v in r.items() if k != "stack"} for r in full]
    stopped = [r for r in ranks if rank_is_stopped(r)]
    lost = [r.get("rank") for r in stopped if r.get("state") == "lost"]
    # Never "confirmed" from the stream alone: that takes a second source (see build_evidence).
    if stopped and (len(lost) == len(stopped) or (lost and status["summary"] == "stalled")):
        assessment = "lost_connection"
    elif stopped:
        assessment = "suspected_hang"
    else:
        assessment = "ranks_never_connected"
    severity = "critical" if any(r.get("severity") == "critical" for r in stopped) else "warning"
    steps = [r["last_step"] for r in ranks if r.get("last_step") is not None]
    common = {
        "severity": severity,
        "assessment": assessment,
        "written_at_ms": status.get("written_at_ms"),
        "silent_floor_seconds": status.get("silent_floor_seconds"),
        "critical_seconds": status.get("critical_seconds"),
        "world_size": status.get("world_size"),
    }
    if lost:
        common["lost_ranks"] = lost
    if places:
        common["stuck_at"] = places
    if status["summary"] == "stalled":
        held = [r for r in ranks if r.get("state") not in ("finished",)]
        return [{
            "kind": "job_stalled",
            "step": max(steps, default=0),
            "ranks": held,
            "missing_ranks": list(status.get("missing_ranks", ())),
            **({"missing_ranks_omitted": _missing_omitted(status)} if _missing_omitted(status) else {}),
            **common,
            **_capture_stopped_keys(held),
        }]
    affected = [r for r in stopped]
    behind = [r["last_step"] for r in affected if r.get("last_step") is not None]
    # The ranks that are not stopped, briefly: what a reader needs to tell that a rank which was
    # silent has stepped on since (the trigger's stall episodes).
    others = [{"rank": r.get("rank"), "state": r.get("state"), "last_step": r.get("last_step"), "attempt": r.get("attempt")}
              for r in ranks if not rank_is_stopped(r)]
    return [{
        "kind": "rank_silent",
        "step": min(behind) if behind else max(steps, default=0),
        "ranks": affected,
        "other_ranks": others,
        "missing_ranks": list(status.get("missing_ranks", ())),
        **({"missing_ranks_omitted": _missing_omitted(status)} if _missing_omitted(status) else {}),
        "newest_step": max(steps, default=0),
        **common,
        **_capture_stopped_keys(affected),
    }]


def _capture_stopped_keys(ranks: list[dict[str, Any]]) -> dict[str, Any]:
    """A silence incident's link to a rank whose capture had stopped while training went on
    (STEP_AND_EVIDENCE_CONTRACT.md, section 3). Such a rank is one that stopped (its state is
    `silent`, `lost` or `unresponsive`) and whose status entry has a `capture_stopped` object, as
    the server reads it (`progress_incidents` in watchdog.rs): a rank whose capture stopped and that
    still reads another state in a stalled job gives neither key. Each such rank's entry gains
    `capture_last_step`, the last step its current attempt stored, or null when that attempt stored
    none (a rank the status marks `attempt_without_steps`: a step an earlier attempt stored is not
    this one's); and the incident gains `capture_stopped` (the status entry's value, unchanged) and
    `capture_last_step` of the first such rank, by rank. Nothing for a rank whose capture never
    stopped."""
    stopped = sorted((r for r in ranks if rank_is_stopped(r)
                      and isinstance(r.get("capture_stopped"), Mapping)), key=lambda r: r.get("rank", 0))
    for entry in stopped:
        entry["capture_last_step"] = None if entry.get("attempt_without_steps") else entry.get("last_step")
    if not stopped:
        return {}
    return {"capture_stopped": stopped[0]["capture_stopped"], "capture_last_step": stopped[0]["capture_last_step"]}


def capture_stopped_step_words(said: Any) -> str | None:
    """The step a rank's capture stopped at, as the server's status gives it inside the rank's
    `capture_stopped` (`step`, from the heartbeat's `capture_stopped_step`, rules contract, part 7):
    "after committing step N", "before committing any step" for null, and None for a status
    without the key (a sender older than it), which says nothing of the step."""
    if not isinstance(said, Mapping) or "step" not in said:
        return None
    step = said["step"]
    if step is None:
        return "before committing any step"
    if isinstance(step, int) and not isinstance(step, bool):
        return f"after committing step {step}"
    return None


def _capture_stopped_text(incident: Mapping[str, Any]) -> str:
    """The sentence for a silence incident whose rank's capture had stopped (`_capture_stopped_keys`)."""
    if incident.get("capture_stopped") is None:
        return ""
    # The rank the keys came from: one that stopped after its capture had (`capture_last_step` on its entry).
    named = next((r for r in sorted((r for r in incident.get("ranks") or () if isinstance(r, Mapping)),
                                    key=lambda r: r.get("rank", 0)) if "capture_last_step" in r), {})
    said = incident["capture_stopped"]
    reason = said.get("reason") if isinstance(said, Mapping) else said
    ago = _finite(said.get("seconds")) if isinstance(said, Mapping) else None
    why = f"{reason}" + (f", {ago:.0f} s ago" if ago is not None else "")
    last = incident.get("capture_last_step")
    # The step the capture itself said it stopped at, where the status carries it; else the last
    # step this attempt stored.
    when = capture_stopped_step_words(said) or (
        f"after step {last}" if last is not None else "before this attempt stored a step")
    who = f"Rank {named['rank']}'s capture" if "rank" in named else "Capture"
    return (f" {who} had stopped {when} ({why}) while training went on: TensorQuill cannot tell from this rank"
            " whether training itself stalled.")


def _optimizer_evidence(ownership: Mapping[str, Any] | None, stalled: list[str]) -> dict[str, Any]:
    if not ownership:
        return {"state": "unknown", "reason": "not recorded by this capture"}
    if ownership.get("state") != "known":
        return {"state": "unknown", "reason": ownership.get("reason", "not established")}
    unheld = set(ownership.get("unheld", ()))
    not_held = [name for name in stalled if name in unheld]
    return {"state": "known", "not_held": not_held[:20], "not_held_count": len(not_held),
            "held_count": len(stalled) - len(not_held)}


def detect_stalled_weights(
    records: list[dict[str, Any]], *, steps: int = STALL_STEPS
) -> dict[str, Any] | None:
    """Return the parameters with a gradient that stopped changing, and when each did, or None.

    Uses `weights_unchanged`, which the capture sets when a parameter's float64 norm is
    exactly what it was at the previous weight check. Equal norms are strong evidence
    but not proof that the weights are equal, so this is a warning: a parameter is reported
    once that holds on `steps` checks in a row while its gradient is finite and non-zero
    (contract 2, signed off Nov 19, 2026). At a weight check (a row with the list):

    - listed, with a finite gradient above zero: the streak grows;
    - listed, with no gradient entry: the streak holds (it was not asked to move then);
    - listed, with a zero or non-finite gradient: the streak starts again;
    - not listed, a parameter seen before with a gradient: the streak starts again, with or
      without a gradient there. The list leaves out a parameter the check found changed and
      one it could not compare, and either breaks the streak; neither ends the parameter's
      history, so a later streak can still fire.

    A row whose total gradient norm is not finite adds nothing to a streak (mixed-precision
    loss scaling skips those optimizer steps on purpose) but its resets apply. A `grad_norm` the
    float64 column refuses (a bool, other text, a list, an integer too large for a float) is read
    as left out, as Rust reads it (AQ5-D3): it does not stop the streak. A skipped
    update (`step_skipped`), whoever reported it, is read the same way: the weights were not
    meant to move (STEP_AND_EVIDENCE_CONTRACT.md, section 1): a streak carries across it. So is
    a step on which the scaler skipped some optimizers and applied others (`step_skipped_part`,
    WAVE_AE_CONTRACT.md, section 4): the record does not say whose update was applied. Within
    one attempt a row without the list is no weight check, and every streak holds across it.

    A resume starts every streak again (WAVE_AE_CONTRACT.md, section 2, amending contract 2 on
    the author's word): over the records in step order, when a record's attempt differs from the
    attempt of the record before it (a record without `attempt` is of attempt 0), every
    parameter's streak is set to zero before that record is read, whether or not it carries the
    list. A process's first weight check compares nothing and carries no list: it is the new
    attempt's baseline, in the author's words, and the checks after it count. Only the streaks start
    again: which parameters were seen changing and which were already reported are kept, so a
    stall raised before the resume stands with its step. A parameter stalled right through a
    resume is reported up to three checks later.

    There is one incident, dated at the first step any parameter stalled; `step`,
    `parameters`, `stalled` and `optimizer` describe that first onset. Every parameter is
    judged on its own history, so one that stalls later is named too: `onsets` lists each
    step at which parameters first stalled, with the parameters (those not shown unchanged at
    an earlier check first), how many were (`trained_before`), and the optimizer evidence for
    them; `stalled_total` counts every parameter that stalled at any step. `trained_before`
    counts a parameter that was missing from the list at an earlier check: it changed there,
    or the check could not compare it, and the record does not say which. A parameter never
    seen changing and one seen changing or not compared are different findings: with bf16 or
    fp16 weights and no float32 master copy, an update smaller than the weight's spacing
    rounds away from the first step (norm weights near 1.0 at a small learning rate), while a
    parameter that trained and then stopped was frozen during the run. `weight_dtypes` (how
    many trainable parameters have each dtype, when the capture recorded it) says which
    precision that is.

    `optimizer` says whether an optimizer that stepped holds the parameters: `known` with
    the stalled parameters no optimizer holds, or `unknown` with the reason, for DeepSpeed,
    Megatron, FSDP1, `auto_step=False` or captures that did not record it. A stalled
    parameter that no optimizer holds is the likely cause. Captures without
    `weights_unchanged` (older files, or Apple MPS) are not checked: a float32 norm stays the
    same while bf16 weights change by a few elements.
    """
    streak: dict[str, int] = {}
    changed: set[str] = set()  # not shown unchanged at some weight check: changed, or not compared
    reported: set[str] = set()
    onsets: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    stalled_steps = 0
    onset_steps = 0
    attempt_before: Any = None
    for index, record in enumerate(sorted(records, key=lambda r: r["step"])):
        attempt = record.get("attempt", 0)
        if index and attempt != attempt_before:
            # A new attempt: every streak starts again; what was seen changing and reported is kept.
            for name in streak:
                streak[name] = 0
        attempt_before = attempt
        unchanged = _unchanged_names(record)
        if unchanged is None:
            continue
        # A step whose total gradient norm is not finite is no evidence of a stall (`non_finite` is
        # its finding). A value the column refuses (a bool, other text, a list, an integer too large
        # for a float) is absent, as Rust reads it, and so is evidence (AQ5-D3, the author: "a top-level
        # grad_norm that is not a number is absent in Python too").
        total = _wire_float(record.get("grad_norm"))
        evidence = (total is None or total is _UNCARRIED or math.isfinite(total)) and skipped_by(record) is None
        grads = _object(record, "grad_norms")
        unchanged_set = set(unchanged)
        for name in streak:
            if name not in unchanged_set:
                streak[name] = 0
        for name, value in grads.items():
            grad = _finite(value)
            if grad is None or grad <= 0 or name not in unchanged_set:
                streak[name] = 0
            elif evidence:
                streak[name] = streak.get(name, 0) + 1
            else:
                streak.setdefault(name, 0)
        changed.update(name for name in streak if name not in unchanged_set)
        if not evidence:
            continue
        stalled =[name for name, count in streak.items() if count >= steps]
        if stalled:
            stalled_steps += 1
            new = [name for name in stalled if name not in reported]
            if new:
                reported.update(new)
                onset_steps += 1
                trained = [name for name in new if name in changed]
                listed = trained + [name for name in new if name not in changed]
                if len(onsets) < STALL_ONSETS:
                    onsets.append({
                        "step": int(record["step"]),
                        "parameters": listed[:20],
                        "stalled": len(new),
                        "trained_before": len(trained),
                        "optimizer": _optimizer_evidence(record.get("optimizer_ownership"), new),
                    })
            if first is None:
                first = {
                    "kind": "stalled_weights",
                    "step": int(record["step"]),
                    "parameters": stalled[:20],
                    "stalled": len(stalled),
                    "with_gradients": sum(1 for v in grads.values() if (_finite(v) or 0) > 0),
                    "consecutive": steps,
                    "optimizer": _optimizer_evidence(record.get("optimizer_ownership"), stalled),
                }
                dtypes = record.get("weight_dtypes")
                if isinstance(dtypes, dict) and dtypes:
                    first["weight_dtypes"] = dtypes
    if first is not None:
        first["stalled_steps"] = stalled_steps
        first["stalled_total"] = len(reported)
        first["onsets"] = onsets
        first["onsets_omitted"] = onset_steps - len(onsets)
    return first


def _first_in_forward_pass_leads(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The hits ordered largest ratio first, except that the first one in the capture's order leads.

    A change in one block's output travels down the residual stream, so every later block shows it
    too and the largest ratio sits in the last of them (a fault seeded in block 16 of 32 listed
    block 31 first). The first hit, in the order the capture lists the modules, which is the
    model's, is where it showed first. The rest stay by ratio, so the largest is second and a
    list cut to 20 keeps both. Ties keep the model's order.
    """
    if not hits:
        return hits
    first = hits[0]
    return [first] + [h for h in sorted(hits, key=lambda h: -h["ratio"]) if h is not first]


def detect_activation_anomalies(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Activation spikes, drift and dead outputs, each judged against the module's history.

    - `activation_spike`: `abs_max` or `std` at least ACT_SPIKE_FACTOR times its median
      over the previous ACT_WINDOW records. The first module in the model leads the list
      (where it showed first), then the rest by ratio.
    - `activation_drift` (DETECTOR_RULES_CONTRACT.md, part 2): the median of the last
      ACT_RECENT records of `std`, or of the outlier ratio `abs_max / std`, in that order,
      moved ACT_DRIFT_FACTOR times above or below its median over the ACT_WINDOW records
      before those ACT_RECENT. A module is judged from its ACT_DRIFT_NEEDS-th record, so that
      the baseline holds none of its first ACT_WINDOW records: a model trained from scratch
      grows its activations for as long as it trains, and against its first records that
      ratio has no ceiling. A slow drift is not seen, and after a shift the new level becomes
      the baseline within ACT_WINDOW records.
    - `dead_activations`: the recent median `zero_frac` rose ACT_DEAD_RISE above its
      median over the module's first ACT_WINDOW records, as when units stop firing.

    Each ratio of a spike or a drift has its `score` beside it, over the same values as its
    baseline (`robust_score`).

    Returns at most one incident per kind: the first step it happened, the modules
    affected at that step, and how many steps were affected in all.
    """
    state = _ActivationAnomalies()
    state.feed(records)
    return state.result()


class _ActivationAnomalies:
    """`detect_activation_anomalies`'s state, fed records in step order (`feed`, any number of times,
    each batch at or after the steps fed before) and read by `result`, as `_GradSpikes` is.

    Only what the rules read is kept (lane AD11: every module's whole history was): each module's
    last ACT_WINDOW + ACT_RECENT rows (the spike reads the last ACT_WINDOW, the drift the last
    ACT_RECENT and the ACT_WINDOW before them), how many rows it has had, and its dead-output
    baseline, the median `zero_frac` of its first ACT_WINDOW rows, taken when it has
    ACT_WINDOW + ACT_RECENT rows, when those are all still kept."""

    def __init__(self) -> None:
        self.history: dict[str, deque[tuple[float, float, float]]] = {}
        self.rows: dict[str, int] = {}  # rows each module has had
        # Each module's baseline for dead outputs, from its first ACT_WINDOW records. A module's
        # history only grows, so those records never change once the check starts; they are
        # summarized once, not again for every later record.
        self.baselines: dict[str, float] = {}
        self.found: dict[str, dict[str, Any]] = {}
        self.counts: dict[str, int] = {}

    def _report(self, kind: str, step: int, hits: list[dict[str, Any]]) -> None:
        if not hits:
            return
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if kind not in self.found:
            self.found[kind] = {"kind": kind, "step": step, "modules": hits[:20], "affected": len(hits)}

    def feed(self, records: Iterable[Mapping[str, Any]]) -> None:
        history, rows, baselines = self.history, self.rows, self.baselines
        outlier = lambda kept: _median([a / s for s, a, _ in kept if s > 0] or [0.0])
        for record in sorted(records, key=lambda r: r["step"]):
            step = int(record["step"])
            spikes, drifts, dead = [], [], []
            for name, stats in layer_entries(record):  # an entry that is not an object is absent (section 1.5)
                std, amax, zfrac = (_finite(stats.get(k)) for k in ("std", "abs_max", "zero_frac"))
                if std is None or amax is None or zfrac is None:
                    continue
                kept = history.get(name)
                if kept is None:
                    kept = history[name] = deque(maxlen=ACT_WINDOW + ACT_RECENT)
                if rows.get(name, 0) >= ACT_MIN_HISTORY:
                    recent = list(kept)[-ACT_WINDOW:]
                    for key, value, earlier in (
                        ("abs_max", amax, [p[1] for p in recent]),
                        ("std", std, [p[0] for p in recent]),
                    ):
                        base = _median(earlier)
                        if base > 0 and value >= ACT_SPIKE_FACTOR * base:
                            spikes.append({"module": name, "stat": key, "value": value,
                                           "baseline": base, "ratio": value / base, **robust_score(value, earlier)})
                            break
                kept.append((std, amax, zfrac))
                count = rows[name] = rows.get(name, 0) + 1
                past = list(kept) if count >= ACT_WINDOW + ACT_RECENT or count >= ACT_DRIFT_NEEDS else None
                if count >= ACT_DRIFT_NEEDS:
                    # The trailing baseline (rules contract, part 2): the last ACT_RECENT records against
                    # the ACT_WINDOW before them, from the record whose baseline holds none of the first
                    # ACT_WINDOW. The score is over the same ACT_WINDOW values as the baseline.
                    last, trail = past[-ACT_RECENT:], past[-(ACT_WINDOW + ACT_RECENT):-ACT_RECENT]
                    for key, base, now, earlier in (
                        ("std", _median([p[0] for p in trail]), _median([p[0] for p in last]), [p[0] for p in trail]),
                        ("abs_max/std", outlier(trail), outlier(last), [a / s for s, a, _ in trail if s > 0]),
                    ):
                        if base > 0 and now > 0 and max(now / base, base / now) >= ACT_DRIFT_FACTOR:
                            drifts.append({"module": name, "stat": key, "baseline": base,
                                           "recent": now, "ratio": now / base, **robust_score(now, earlier)})
                            break
                if count >= ACT_WINDOW + ACT_RECENT:
                    # `dead_activations` keeps the module's first ACT_WINDOW records as its baseline,
                    # taken at its ACT_WINDOW + ACT_RECENT-th row, while all of them are still kept.
                    last = past[-ACT_RECENT:]
                    if name not in baselines:
                        baselines[name] = _median([p[2] for p in past[:ACT_WINDOW]])
                    base_zero = baselines[name]
                    now_zero = _median([p[2] for p in last])
                    if now_zero - base_zero >= ACT_DEAD_RISE:
                        dead.append({"module": name, "baseline": base_zero, "recent": now_zero})
            self._report("activation_spike", step, _first_in_forward_pass_leads(spikes))
            self._report("activation_drift", step, drifts)
            self._report("dead_activations", step, dead)

    def result(self) -> list[dict[str, Any]]:
        """The incidents `detect_activation_anomalies` returns for the records fed so far, new copies."""
        return [{**copy.deepcopy(incident), "affected_steps": self.counts[kind]} for kind, incident in self.found.items()]


def _attention_values(stats: Any) -> tuple[float | None, float | None, int]:
    """One attention module's probe statistics as the rules read them: the mean of its heads'
    finite `entropy_share` (math.fsum over the heads, in order), the largest finite `max_logit`,
    and how many heads had a finite share. NaN shares (rows all masked or invalid) and the
    `-inf` logit of a head with no visible score are left out; with none left, None."""
    if not isinstance(stats, dict):
        return None, None, 0
    shares = [v for v in (_finite(x) for x in stats.get("entropy_share") or ()) if v is not None] \
        if isinstance(stats.get("entropy_share"), list) else []
    logits = [v for v in (_finite(x) for x in stats.get("max_logit") or ()) if v is not None] \
        if isinstance(stats.get("max_logit"), list) else []
    share = math.fsum(shares) / len(shares) if shares else None
    return share, (max(logits) if logits else None), len(shares)


class _AttentionSeries:
    """One attention module's share or logit: its baseline and, per kind, the run of probes
    in a row that met the kind's rule (the global index of its last probe and its first one)."""

    __slots__ = ("baseline", "runs")

    def __init__(self) -> None:
        self.baseline: deque[float] = deque()
        self.runs: dict[str, tuple[int, int, dict[str, Any]]] = {}


def _attention_hit(kind: str, value: float, base: float) -> bool:
    if kind == "attention_collapse":
        return base > 0 and value <= ATTN_COLLAPSE_RATIO * base
    if kind == "attention_uniform":
        return value >= ATTN_UNIFORM and value - base >= ATTN_UNIFORM_RISE
    return base > 0 and value >= ATTN_LOGIT_FACTOR * base


def detect_attention_anomalies(
    records: list[dict[str, Any]], *, not_checked: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Attention entropy collapse, uniform attention and logit growth, from the probe's records.

    Each attention module (a block's attention call, named as the probe names it) is judged
    against its own earlier probes: the median of its last ATTN_WINDOW probes that met no rule,
    once it has ATTN_MIN_HISTORY of them. A rule must hold on ATTN_SUSTAIN probes in a row,
    counted over every probe that recorded attention (a probe without the module breaks the
    run), and the incident is dated at the first of them.

    - `attention_collapse`: the layer's mean entropy share at most ATTN_COLLAPSE_RATIO times its
      baseline (attention concentrating on few keys, as in Zhai et al. 2023).
    - `attention_uniform`: the mean entropy share at least ATTN_UNIFORM and at least
      ATTN_UNIFORM_RISE above its baseline: attention that had structure spreads evenly over
      every key, as after a seeded outlier upstream. A layer that was near uniform from the
      start is never reported.
    - `attention_logit_growth`: the largest pre-softmax logit at least ATTN_LOGIT_FACTOR times
      its baseline.

    One incident per kind: the step of the first probe, the probe that confirmed it, the
    modules whose run started there (in the probe's order, at most ATTN_MODULES), how many,
    and how many probes were part of a confirmed run in all. `not_checked` gets a `probe`
    entry when any probe failed (`{"error": ...}` in place of its results), or, when none did,
    when the capture said the probe was given and not run (`{"not_run": why}`, once: a sharded
    model). Too few probes for
    a module to be judged and confirmed is not an entry: the kinds are then `checked` without
    enough history yet (`kind_checks`, `RECORDS_NEEDED`), and the run is not degraded for it.
    """
    shares: dict[str, _AttentionSeries] = {}
    logits: dict[str, _AttentionSeries] = {}
    found: dict[str, dict[str, Any]] = {}
    affected: dict[str, set[int]] = {}
    probe_steps: list[int] = []
    asked, failed, first_failure = 0, 0, None
    off: tuple[int, Any] | None = None  # the capture's note that `probe=` was given and is not run
    for record in sorted(records, key=lambda r: r["step"]):
        probe = record.get("probe")
        if isinstance(probe, dict) and "not_run" in probe:
            # Not a probe that was asked for and failed: the capture turned the probe off and says
            # why, once (a sharded model, where a probe's forward would be a collective).
            if off is None:
                off = (int(record["step"]), probe["not_run"])
            continue
        if isinstance(probe, dict):
            asked += 1
            if "error" in probe:
                failed += 1
                if first_failure is None:
                    first_failure = (int(record["step"]), probe["error"])
        attention = probe.get("attention") if isinstance(probe, dict) and "error" not in probe else None
        if not isinstance(attention, dict) or not attention:
            continue
        index = len(probe_steps)
        step = int(record["step"])
        probe_steps.append(step)
        confirmed: dict[str, list[dict[str, Any]]] = {}
        for module, stats in attention.items():
            share, logit, heads = _attention_values(stats)
            for series_of, value, kinds in (
                (shares, share, ("attention_collapse", "attention_uniform")),
                (logits, logit, ("attention_logit_growth",)),
            ):
                series = series_of.setdefault(module, _AttentionSeries())
                if value is None:
                    series.runs.clear()
                    continue
                if len(series.baseline) < ATTN_MIN_HISTORY:
                    series.baseline.append(value)
                    continue
                base = _median(list(series.baseline))
                flagged = False
                for kind in kinds:
                    if not _attention_hit(kind, value, base):
                        series.runs.pop(kind, None)
                        continue
                    flagged = True
                    last, length, onset = series.runs.get(kind, (-2, 0, {}))
                    if last != index - 1:
                        length, onset = 0, {
                            "step": step, "value": value, "baseline": base, "baseline_probes": len(series.baseline),
                            "heads": heads, "earlier": list(series.baseline),
                        }
                    length += 1
                    series.runs[kind] = (index, length, onset)
                    if length >= ATTN_SUSTAIN:
                        steps = affected.setdefault(kind, set())
                        steps.update(probe_steps[index - length + 1 : index + 1])
                    if length == ATTN_SUSTAIN:
                        hit = {"module": module, "value": onset["value"], "baseline": onset["baseline"]}
                        if kind == "attention_uniform":
                            hit["rise"] = onset["value"] - onset["baseline"]
                        else:
                            hit["ratio"] = onset["value"] / onset["baseline"]
                            # Over the earlier probes of the baseline: at most ATTN_WINDOW, so always
                            # `short_history` (rules contract, part 1). Said so that no reader leaves it out.
                            hit.update(robust_score(onset["value"], onset["earlier"]))
                        hit.update(baseline_probes=onset["baseline_probes"], heads=onset["heads"])
                        confirmed.setdefault(kind, []).append(hit)
                if not flagged:
                    series.baseline.append(value)
                    if len(series.baseline) > ATTN_WINDOW:
                        series.baseline.popleft()
        for kind, hits in confirmed.items():
            if kind not in found:
                found[kind] = {
                    "kind": kind,
                    "step": probe_steps[index - ATTN_SUSTAIN + 1],
                    "confirmed_step": step,
                    "modules": hits[:ATTN_MODULES],
                    "affected": len(hits),
                }
    for kind, incident in found.items():
        incident["affected_probes"] = len(affected[kind])
        incident["probes"] = len(probe_steps)
    if not_checked is not None and first_failure is not None:
        # A probe that failed leaves its step unjudged, and a rule that must hold on probes
        # in a row may have missed a change there.
        step, error = first_failure
        not_checked.append({
            "check": "probe",
            "kinds": list(ATTN_KINDS),
            "reason": f"{failed} of {asked} probe{'s' if asked != 1 else ''} failed, the first at step {step}:"
                      f" {error if isinstance(error, str) else json.dumps(error, separators=(',', ':'))}",
            "step": step,
        })
    elif not_checked is not None and off is not None:
        step, why = off
        not_checked.append({
            "check": "probe",
            "kinds": list(ATTN_KINDS),
            "reason": f"the probe was not run: {why if isinstance(why, str) else json.dumps(why, separators=(',', ':'))}",
            "step": step,
        })
    return [found[kind] for kind in ATTN_KINDS if kind in found]


def detect_loss_scale_trouble(
    records: list[dict[str, Any]], *, run: int = SCALE_SKIP_RUN, floor: float = SCALE_FLOOR
) -> dict[str, Any] | None:
    """Return the first step where mixed-precision loss scaling is in trouble, or None:
    `run` or more consecutive scaler-confirmed skips among observations judged by this rule, or
    the loss scale below `floor`. Needs `step_skipped` and `grad_scale`, recorded with
    `init(..., scaler=)`.

    Only a skip the scaler confirmed counts (`step_skipped_by` `scaler` or `both`, or no such
    key: STEP_AND_EVIDENCE_CONTRACT.md, section 1, as the author amended it). A skip only the
    caller reported never adds a skip. With the scaler's outcome unknown (no `scaler_applied`)
    it is passed over: it neither adds to the streak nor ends it, and is in neither
    `skipped_steps` nor `recorded_steps`, so a streak can span more training steps than it
    counts. Where the SDK observed the scaler apply that update (`scaler_applied: true`) it ends
    the streak, as a recorded applied update does, and counts in `recorded_steps`. A loss scale recorded on a caller's skip is still read against `floor`,
    and raises with reason `scale` and the skip counts unchanged.

    A step on which the scaler skipped some optimizers and applied others (`partial_step`,
    WAVE_AE_CONTRACT.md, section 4, amending section 1 of STEP_AND_EVIDENCE_CONTRACT.md on the
    author's word) is counted once in `recorded_steps`, not in `skipped_steps`, and apart as
    `partial_steps`; an update occurred, so it ends the streak of fully skipped updates. The
    incident carries `partial_steps` only when it is above zero."""
    return _loss_scale_judged(records, run=run, floor=floor)[0]


def _loss_scale_judged(
    records: list[dict[str, Any]], *, run: int = SCALE_SKIP_RUN, floor: float = SCALE_FLOOR
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """`detect_loss_scale_trouble`'s incident, and the rule's own counters after the last record:
    `{"recorded_steps", "skipped_steps", "streak"}`, as the behaviour cases' `counters_at_end`
    states them whether or not anything was raised, and `partial_steps` after them only when it
    is above zero (WAVE_AE_CONTRACT.md, section 4)."""
    streak, skipped, total, partial, first = 0, 0, 0, 0, None
    for record in sorted(records, key=lambda r: r["step"]):
        if "step_skipped" not in record:
            continue
        by = skipped_by(record)
        if by == "caller" and record.get("scaler_applied") is True:
            total += 1  # the scaler applied this update: observed, and the streak ends
            streak = 0
        elif partial_step(record):
            total += 1  # some optimizers' updates were applied: counted apart, and the streak ends
            partial += 1
            streak = 0
        elif by != "caller":
            total += 1
            skipped += by is not None
            streak = streak + 1 if by is not None else 0
        scale = _finite(record.get("grad_scale"))
        if first is None and (streak >= run or (scale is not None and scale < floor)):
            first = {
                "kind": "loss_scale_trouble",
                "step": int(record["step"]),
                "consecutive_skips": streak,
                "grad_scale": scale,
                "reason": "skips" if streak >= run else "scale",
            }
    partial_count = {"partial_steps": partial} if partial else {}
    if first is not None:
        first["skipped_steps"] = skipped
        first["recorded_steps"] = total
        first.update(partial_count)
    return first, {"recorded_steps": total, "skipped_steps": skipped, "streak": streak, **partial_count}


def detect_repeated_steps(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return step numbers recorded more than once, as after a resume without `start_step`.

    Pass every attempt (`load_records(..., all_attempts=True)`). A later attempt that
    starts after the run's first step resumed from a checkpoint, and the steps it writes
    again are expected. One that starts at the run's first step again restarted the
    numbering, and the steps it shares with earlier attempts are reported. So is a step
    written twice within one attempt.
    """
    seen: dict[tuple[Any, int, int], int] = {}
    for record in records:
        key = (record.get("rank", 0), record.get("attempt", 0), int(record["step"]))
        seen[key] = seen.get(key, 0) + 1
    repeated_steps = {step for (_, _, step), n in seen.items() if n > 1}
    for rank in {rank for rank, _, _ in seen}:
        steps: dict[int, set[int]] = {}
        for r, attempt, step in seen:
            if r == rank:
                steps.setdefault(attempt, set()).add(step)
        attempts = list(steps)  # in the order they first appear (contract 1)
        start = min(steps[attempts[0]])
        earlier = set(steps[attempts[0]])
        for attempt in attempts[1:]:
            if min(steps[attempt]) <= start:
                repeated_steps |= steps[attempt] & earlier
            earlier |= steps[attempt]
    repeated = sorted(repeated_steps)
    if not repeated:
        return None
    return {"kind": "repeated_steps", "step": repeated[0], "repeated": len(repeated),
            "steps": repeated[:20]}


RESUME_LR_TOLERANCE = 1e-6  # relative; the same scheduler restored gives the same value


def _lr_values(record: Mapping[str, Any]) -> list[float] | None:
    """A record's learning rates, `_lr_list`'s: an `lr` that is not a list (`0.001`) is none (BUG-12)."""
    return _lr_list(record)


def detect_resume_lr_changes(
    records: list[dict[str, Any]], *, not_checked: list[dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    """Report where a resumed attempt's learning rate differs from the run's history.

    Pass every attempt (`load_records(..., all_attempts=True)`). At the first step S of
    each later attempt that resumed from a checkpoint (it starts after the run's first
    step, so a restart of the numbering is left to `repeated_steps`), the learning rate is
    compared with the run's history as it stood before that attempt, which is every earlier
    attempt with each rollback applied:

    - `same_step`: the history has step S, because an earlier attempt ran past the
      checkpoint. The two values come from the same step, so any difference is reported.
    - `extrapolated`: it does not, as after a hard kill that lost the last steps. The
      history's last values are continued along their recent slope. A scheduler that
      legitimately changes at S looks the same as a reset, so this basis reports only the
      distinctive case, a resumed value equal to the run's first learning rate while the
      history is elsewhere (warmup starting over). Any other difference goes into
      `not_checked` with its numbers.

    A difference is an observation, not a verdict: some restarts change the schedule on
    purpose, and a record cannot show the intent. A resume with no learning rate recorded
    on either side is added to `not_checked`.
    """
    by_rank: dict[Any, dict[int, dict[int, list[float] | None]]] = {}
    for record in records:
        attempts = by_rank.setdefault(record.get("rank", 0), {})
        attempts.setdefault(record.get("attempt", 0), {})[int(record["step"])] = _lr_values(record)
    changes: list[dict[str, Any]] = []

    def skip(rank: Any, step: int, reason: str, **numbers: Any) -> None:
        if not_checked is not None:
            not_checked.append({"check": "resume_lr", "step": step, "rank": rank, "reason": reason, **numbers})

    for rank, attempts in sorted(by_rank.items(), key=lambda kv: str(kv[0])):
        order = list(attempts)  # in the order they first appear (contract 1)
        run_start = min(attempts[order[0]])
        first_lr = attempts[order[0]][run_start]
        history: dict[int, tuple[list[float], int]] = {}  # step -> (learning rate, attempt)
        for attempt in order:
            steps = attempts[attempt]
            resumed_at = min(steps)
            if attempt != order[0] and resumed_at > run_start:
                result = _compare_resume(history, steps[resumed_at], resumed_at, first_lr)
                if isinstance(result, str):
                    skip(rank, resumed_at, result)
                elif result is not None and result["basis"] == "extrapolated" and not result["equals_first_lr"]:
                    skip(
                        rank, resumed_at,
                        "the earlier attempts ended before the resume step, so the learning rate cannot be"
                        " compared at the same step; a schedule that changes there looks the same",
                        expected_lr=result["expected_lr"], resumed_lr=result["resumed_lr"],
                    )
                elif result is not None:
                    changes.append({"step": resumed_at, "rank": rank, "attempt": attempt, **result})
            # This attempt rolls back whatever earlier attempts recorded from its first step on.
            history = {k: v for k, v in history.items() if k < resumed_at}
            history.update({k: (v, attempt) for k, v in steps.items() if v is not None})
    if not changes:
        return None
    changes.sort(key=lambda c: c["step"])
    return {
        "kind": "resume_lr_change",
        "step": changes[0]["step"],
        "resumes": len(changes),
        "changes": changes[:5],
        "note": "an observation: some restarts change the schedule on purpose",
    }


def _compare_resume(
    history: dict[int, tuple[list[float], int]],
    new: list[float] | None,
    resumed_at: int,
    first_lr: list[float] | None,
) -> dict[str, Any] | str | None:
    """A change to report, a reason the resume cannot be checked, or None when it continues."""
    if new is None:
        return "no learning rate recorded in the resumed attempt"
    if not history:
        return "no learning rate recorded in the earlier attempts"
    if resumed_at in history:
        basis, (expected, source) = "same_step", history[resumed_at]
        slope_tolerance = 0.0
    else:
        recent = sorted(k for k in history if k < resumed_at)[-10:]
        if not recent:
            return "the earlier attempts recorded no learning rate before the resume step"
        last = recent[-1]
        expected, source = history[last]
        slope = [0.0] * len(expected)
        if len(recent) > 1 and len(history[recent[0]][0]) == len(expected):
            span = last - recent[0]
            slope = [(a - b) / span for a, b in zip(expected, history[recent[0]][0])]
            expected = [a + m * (resumed_at - last) for a, m in zip(expected, slope)]
        basis = "extrapolated"
        slope_tolerance = 0.5 * max((abs(m) for m in slope), default=0.0) * (resumed_at - last)
    differs = len(new) != len(expected) or any(
        abs(n - e) > max(RESUME_LR_TOLERANCE * abs(e), slope_tolerance) for n, e in zip(new, expected)
    )
    if not differs:
        return None
    resets = (
        first_lr is not None
        and len(first_lr) == len(new)
        and all(abs(n - f) <= RESUME_LR_TOLERANCE * abs(f) for n, f in zip(new, first_lr))
        and any(abs(e - f) > RESUME_LR_TOLERANCE * abs(f) for e, f in zip(expected, first_lr))
    )
    return {
        "previous_attempt": source,
        "basis": basis,
        "expected_lr": expected,
        "resumed_lr": new,
        "first_lr": first_lr,
        "equals_first_lr": resets,
    }


def load_records(
    source: str | Path | Iterable[Mapping[str, Any]],
    *,
    rank: int | None = None,
    all_attempts: bool = False,
    scalars: bool = False,
) -> list[dict[str, Any]]:
    """Read captured step records.

    `source` is a `.jsonl` file from `tensorquill.init`, a directory of the Parquet parts
    `tensorquill-server` wrote (a run's `runs/<run_id>` directory, or a `rank=<rank>`
    directory inside it), one `.parquet` part, or an iterable of dicts. Reading Parquet
    needs `pyarrow` (`pip install pyarrow`) and gives the same dicts as the
    `.jsonl` file the parts were streamed from.

    `rank` keeps one rank's records. When a job resumed from an earlier checkpoint under
    the same run ID, the records the later attempt replaced are left out (see
    `resolve_attempts`); `all_attempts=True` returns everything that was written.

    Every record read is checked first, whatever its rank (`identity.check`): a record whose
    `schema_version`, `rank`, `attempt` or `step` is not valid raises `MalformedRecordError`,
    which names the field, the value, and the file and line, the part and row, or the place in
    the list (WAVE_AF_CONTRACT.md, section 1). Nothing is judged from such an input.

    Scalar-only records (`identity.is_scalar_record`, written by `session.log(..., step=...)`: WAVE_AL_CONTRACT.md
    section 4) are set apart here, after that check: by default they are left out, so the records
    returned are the training records alone, as before them. `scalars=True` keeps them, in their
    place among the training records, for a reader of series (`scalar_series`); a store's
    `scalars-*` objects are then read beside its parts. Either way the attempts are resolved with
    the training records' cutoffs (`resolve_attempts`).
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            # One rank at a time: a run of many ranks never has to fit in memory at once.
            records = [
                r for part in _parquet_parts(path, rank, scalars=scalars) for r in _checked_part(part)
                if rank is None or r.get("rank", 0) == rank
            ]
        elif path.suffix == ".parquet":
            records = _checked_part(path)
        else:
            records = []
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip():
                    record = json.loads(line)
                    check_identity(record, in_file(path, number))
                    records.append(record)
    else:
        records = []
        for index, item in enumerate(source):
            check_identity(item, in_list(index))
            records.append(dict(item))
    if rank is not None:
        records = [r for r in records if r.get("rank", 0) == rank]
    if not scalars:
        records = [r for r in records if not is_scalar_record(r)]
    return records if all_attempts else resolve_attempts(records)


def attempts_first_seen(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, Any], list[Any]]:
    """Each rank's attempts in the order they first appear in `records`, oldest first, by
    `(run_id, rank)`: the order `resolve_attempts` reads from the records themselves, in the shape
    its `order` takes. Give the whole stream, training and scalar-only records together and before
    any is resolved away: a rollback can leave out an older attempt's first records, and the
    scalar-only records alone can show a newer attempt first (BUG-20)."""
    order: dict[tuple[str, Any], dict[Any, None]] = {}
    for record in records:
        order.setdefault((_run_of(record), record.get("rank", 0)), {}).setdefault(record.get("attempt", 0))
    return {key: list(attempts) for key, attempts in order.items()}


def scalar_points_in_order(
    scalar_only: Iterable[Mapping[str, Any]], *, order: Mapping[tuple[str, Any], Iterable[Any]] | None = None
) -> list[Mapping[str, Any]]:
    """Scalar-only records in the order their points stand at one step, the last standing: by
    attempt, the later attempt last, then by `seq` (Q6 a: the highest `seq` wins). `seq` counts per
    session, so it orders the records of one attempt only; a later attempt's point stands over an
    earlier one's at the same step, as its training records do.

    "Later" is the order the attempts first appear in the whole record stream (WAVE_AL_CONTRACT.md,
    lane AL-4′'s decision 2), which `order` gives as `resolve_attempts` takes it: each rank's
    attempts oldest first by `(run_id, rank)` (`attempts_first_seen` of every record read, or
    `attempt_order_across` for a rank's several files). An attempt `order` does not list keeps its
    place of first appearance among these records, after those it lists. Without `order` the
    records' own first appearance is taken, as before: it is the stream's only when no training
    record came between them, so a caller that has the stream should give its order (BUG-20)."""
    listed = list(scalar_only)
    places: dict[tuple[str, Any], dict[Any, int]] = {}
    for key, attempts in (order or {}).items():
        known = places.setdefault(key, {})
        for attempt in attempts:
            known.setdefault(attempt, len(known))
    for record in listed:
        known = places.setdefault((run_of(record), record.get("rank", 0)), {})
        known.setdefault(record.get("attempt", 0), len(known))
    return sorted(listed, key=lambda r: (places[(run_of(r), r.get("rank", 0))][r.get("attempt", 0)], int(r["seq"])))


def merged_scalar_points(
    training: Iterable[Mapping[str, Any]], scalar_only: Iterable[Mapping[str, Any]],
    *, order: Mapping[tuple[str, Any], Iterable[Any]] | None = None,
) -> dict[str, dict[int, Any]]:
    """Each logged scalar's points, `{name: {step: value}}`, the steps ascending: the training
    records' `scalars` values and the scalar-only records' points, by step (WAVE_AL_CONTRACT.md
    section 4.6). At one step a scalar-only record's value stands over the training records' and,
    among scalar-only records, the later attempt's, then the highest `seq` (`scalar_points_in_order`,
    which `order` is given to; Q6 a). Of training records of one step, the later stands. A step no
    training record has is a point of its own; no record, step or sample is made. Values are as
    recorded (a non-finite one as the sink's string); `None` is no value. Names come in the order
    they first appear, the training records' first."""
    points: dict[str, dict[int, Any]] = {}
    for record in sorted(training, key=lambda r: int(r["step"])):
        logged = record.get("scalars")
        if isinstance(logged, Mapping):
            for name, value in logged.items():
                if value is not None:
                    points.setdefault(str(name), {})[int(record["step"])] = value
    for record in scalar_points_in_order(scalar_only, order=order):
        logged = record.get("scalars")
        if isinstance(logged, Mapping):
            for name, value in logged.items():
                if value is not None:
                    points.setdefault(str(name), {})[int(record["step"])] = value
    return {name: dict(sorted(at.items())) for name, at in points.items()}


def scalar_series(
    source: str | Path | Iterable[Mapping[str, Any]],
    *,
    rank: int | None = None,
    all_attempts: bool = False,
) -> dict[str, list[list[Any]]]:
    """Each logged scalar's series of one rank, `{name: [[step, value], ...]}` in step order: the
    training records' values and the scalar-only records' points, merged by step
    (`merged_scalar_points`). `source` is what `load_records` reads, and the attempts are resolved
    as it resolves them (with the training records' cutoffs), or not with `all_attempts`.

    Records of several ranks with no `rank` given raise `ValueError`, as `build_evidence` refuses
    them, and so do records of several run IDs (`check_one_run`).

    At one step, of two attempts' scalar-only points the later attempt's stands, "later" as the
    attempts first appear in every record read, before any is resolved away (`attempts_first_seen`)."""
    every = load_records(source, rank=rank, all_attempts=True, scalars=True)
    order = attempts_first_seen(every)
    records = every if all_attempts else resolve_attempts(every)
    ranks = sorted({int(r.get("rank", 0)) for r in records})
    if len(ranks) > 1:
        raise ValueError(f"these records come from ranks {ranks}; pass rank= to read one of them")
    check_one_run(records)
    training, scalar_only = set_apart_scalars(records)
    return {name: [[step, value] for step, value in at.items()]
            for name, at in merged_scalar_points(training, scalar_only, order=order).items()}


def set_apart_scalars(records: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The training records and the scalar-only records of `records` (`identity.is_scalar_record`),
    each in the order given (WAVE_AL_CONTRACT.md section 4.6). No engine is given a scalar-only
    record: none of the kinds judges one (D12), and none is a step."""
    training: list[dict[str, Any]] = []
    scalar_only: list[dict[str, Any]] = []
    for record in records:
        (scalar_only if is_scalar_record(record) else training).append(record)  # type: ignore[arg-type]
    return training, scalar_only


def read_jsonl(path: str | Path) -> tuple[list[dict[str, Any]], int]:
    """Every record of a `.jsonl` capture, and how many lines were left out because they are not
    a JSON object: a line torn by a killed process, which a restarted writer then appends after
    (the writer keeps the fragment on a line of its own, lane U4), or a last line still being
    written. The records after such a line are read (lane U7, Nov 19, 2026: a plain read raised
    on it and gave no evidence at all). A whole last line with no newline is read, as before."""
    data = Path(path).read_bytes()
    records: list[dict[str, Any]] = []
    left_out = 0
    for number, line in enumerate(data.splitlines(), 1):  # `number`: the line in the file, from 1
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            record = None
        if isinstance(record, dict):
            # A whole record whose identity is not valid refuses the file (section 1): it is not a
            # torn line, which is left out and counted.
            check_identity(record, in_file(path, number))
            records.append(record)
        else:
            left_out += 1
    return records, left_out


def _load_counted(source: Any, rank: int | None, *, checked: bool = False,
                  scalars: bool = False) -> tuple[list[dict[str, Any]], int]:
    """`load_records(source, rank=rank, all_attempts=True, scalars=scalars)`, with a `.jsonl` file
    read by `read_jsonl`: its records and the lines left out (0 for any other source).

    Records whose identity was checked already are copied and not checked again: `read_jsonl`'s,
    checked with their file and line, and with `checked` the records handed in, which a watcher's
    `_Tail` checked as it read them (each once; at 10,000 records a second check at every look was
    about 40 ms of a 0.5 s update, wave AF-W)."""
    if isinstance(source, (str, Path)) and not Path(source).is_dir() and Path(source).suffix not in (".parquet",):
        records, left_out = read_jsonl(source)
        return _copies(records, rank, scalars=scalars), left_out
    if checked and not isinstance(source, (str, Path)):
        return _copies(source, rank, scalars=scalars), 0
    return load_records(source, rank=rank, all_attempts=True, scalars=scalars), 0


def _copies(records: Iterable[Mapping[str, Any]], rank: int | None, *, scalars: bool = False) -> list[dict[str, Any]]:
    """What `load_records(records, rank=rank, all_attempts=True, scalars=scalars)` gives for records
    already checked."""
    return [dict(item) for item in records
            if (rank is None or item.get("rank", 0) == rank) and (scalars or not is_scalar_record(item))]


def lines_left_out_line(evidence: Mapping[str, Any]) -> str | None:
    """One line on the capture's lines that were not whole records, or None."""
    left_out = evidence.get("lines_left_out")
    if not isinstance(left_out, int) or isinstance(left_out, bool) or left_out <= 0:
        return None
    return (f"Lines left out: {left_out} line{'s' if left_out != 1 else ''} of the capture"
            f" {'are' if left_out != 1 else 'is'} not a whole JSON record (torn by a killed process, or still being"
            " written) and were skipped; the steps after them are read.")


def resolve_attempts(
    records: list[dict[str, Any]], *, order: Mapping[tuple[str, Any], list[Any]] | None = None
) -> list[dict[str, Any]]:
    """Leave out the records a later attempt replaced.

    Each `tensorquill.init` stamps its records with `attempt`, its start time. A job that
    resumes from the checkpoint at step S writes step S and onwards again as a new
    attempt; from then on, what earlier attempts of that rank recorded at step S or later
    is history that training rolled back, including steps the new attempt has not reached
    yet. Records without `attempt` (written before it existed) count as attempt 0. The
    order of the remaining records is unchanged.

    "Later" is the order in which a rank's attempts first appear in `records`, not their
    numbers (contract 1, signed off Nov 19, 2026): an attempt number is an identity, and a
    replacement on a host whose clock is behind has a lower one. A local file is appended
    across attempts, so its order is arrival order; a store's parts are read in the rank's
    stored attempt order (`attempts.json`, else by number: `_parquet_parts`). `order` gives a
    rank's attempts oldest first by `(run_id, rank)` where the records alone cannot, as for a
    rank's several files (`resolve_attempts_across`); an attempt it does not list keeps its
    place of first appearance after those it does.

    An attempt is of one run: a later attempt of another run ID never rolls back this one's
    records (lane U7, Nov 19, 2026). A record with no `run_id` is of no named run (`_run_of`).

    Scalar-only records (`identity.is_scalar_record`) are resolved with the training records'
    cutoffs (WAVE_AL_CONTRACT.md section 4.6): one is left out where a later attempt's first
    training step is at or below its step, as a training record of its attempt there would be.
    Their own steps cut nothing, so an attempt whose first line is a baseline at step 0 rolls back
    only from its first training step, and an attempt holding scalar-only records alone rolls
    back nothing. Such an attempt still takes its place in the order of first appearance.
    """
    first_step: dict[tuple[str, Any, int], int] = {}
    appeared_keys: dict[tuple[str, Any, int], None] = {}  # every attempt, in the order it first appears
    for record in records:
        key = (_run_of(record), record.get("rank", 0), record.get("attempt", 0))
        appeared_keys.setdefault(key)
        if is_scalar_record(record):
            continue
        step = int(record["step"])
        if key not in first_step or step < first_step[key]:
            first_step[key] = step
    if len({(run, rank) for run, rank, _ in appeared_keys}) == len(appeared_keys):
        return records  # one attempt per rank of each run: nothing was replaced
    cutoff: dict[tuple[str, Any, int], float] = {}
    for run, rank in {(run, rank) for run, rank, _ in appeared_keys}:
        cut = math.inf
        # `appeared_keys` holds a rank's attempts in the order they first appear.
        appeared = [a for n, r, a in appeared_keys if (n, r) == (run, rank)]
        given = [a for a in (order or {}).get((run, rank), ()) if a in appeared]
        for attempt in reversed(given + [a for a in appeared if a not in given]):
            cutoff[(run, rank, attempt)] = cut
            cut = min(cut, first_step.get((run, rank, attempt), math.inf))
    return [
        record for record in records
        if int(record["step"]) < cutoff[(_run_of(record), record.get("rank", 0), record.get("attempt", 0))]
    ]


def superseded_rows(records: Iterable[Mapping[str, Any]]) -> int:
    """How many records belong to an attempt that is not the newest seen so far, in the order
    given: a late row of a superseded attempt (contract 1, a reader with only the local file).
    They are kept and judged by the rollback as before, and do not move the run's last step."""
    newest: dict[tuple[str, Any], int] = {}
    places: dict[tuple[str, Any], dict[Any, int]] = {}
    late = 0
    for record in records:
        key = (_run_of(record), record.get("rank", 0))
        known = places.setdefault(key, {})
        place = known.setdefault(record.get("attempt", 0), len(known))
        if place < newest.get(key, -1):
            late += 1
        newest[key] = max(newest.get(key, -1), place)
    return late


def attempt_order_report(
    files: Iterable[Iterable[Mapping[str, Any]]], names: Iterable[str] | None = None
) -> dict[tuple[str, Any], dict[str, Any]]:
    """How the attempts of each rank are ordered when its records are in several local files
    (contract 1, "One run and rank in more than one local file", as the author amended it,
    Nov 19, 2026), by `(run_id, rank)`: `order` (oldest first), `attempt_order` and, when it
    applies, `reason` and `by_number`.

    1. In each file an attempt's place is its first appearance there, and any two attempts in
       one file keep that file's order.
    2. An attempt is placed once every attempt any file puts before it is placed.
    3. Among attempts ready at once, the earliest first-record `ts` goes first (an attempt held
       by several files counts its earliest finite first `ts`); the lower attempt number decides
       when a candidate has no finite `ts`, or two share the earliest. Those pairs are named in
       `by_number`, each with why (`"no ts"` or `"same ts"`).
    4. Files that disagree (one puts A before B, another B before A, directly or through other
       attempts) give numeric order, `attempt_order: "numeric"`, and a `reason` naming the
       attempts and the files (`names`, else "file 1", "file 2", ...).
    5. Otherwise `attempt_order: "first_seen"`. The order the files are given in never decides.
    """
    listed = [list(records) for records in files]
    labels = list(names) if names is not None else [f"file {n}" for n in range(1, len(listed) + 1)]
    firsts: dict[tuple[str, Any], dict[Any, float | None]] = {}
    edges: dict[tuple[str, Any], dict[Any, dict[Any, set[str]]]] = {}  # earlier -> later -> files
    for label, records in zip(labels, listed):
        places: dict[tuple[str, Any], list[Any]] = {}
        for record in records:
            key, attempt = (_run_of(record), record.get("rank", 0)), record.get("attempt", 0)
            here = places.setdefault(key, [])
            if attempt in here:
                continue
            here.append(attempt)  # its first record in this file
            ts, stamps = _finite(record.get("ts")), firsts.setdefault(key, {})
            if attempt not in stamps:
                stamps[attempt] = ts
            elif ts is not None and (stamps[attempt] is None or ts < stamps[attempt]):  # type: ignore[operator]
                stamps[attempt] = ts  # the earliest finite first `ts` among the files
        for key, here in places.items():
            later_of = edges.setdefault(key, {})
            for at, earlier in enumerate(here):
                for later in here[at + 1:]:
                    later_of.setdefault(earlier, {}).setdefault(later, set()).add(label)

    def number(attempt: Any) -> int:
        return _whole_attempt(attempt) or 0

    out: dict[tuple[str, Any], dict[str, Any]] = {}
    for key, stamps in firsts.items():
        later_of = edges.get(key, {})
        clash = sorted(
            ((a, b) for a, laters in later_of.items() for b in laters if a in later_of.get(b, {})), key=str
        )
        waiting = {a: sum(a in laters for laters in later_of.values()) for a in stamps}
        order: list[Any] = []
        by_number: list[dict[str, Any]] = []
        while waiting and not clash:
            ready = [a for a, n in waiting.items() if n == 0]
            if not ready:  # a disagreement through other attempts
                clash = sorted(((a, b) for a in waiting for b in later_of.get(a, {}) if b in waiting), key=str)
                break
            if len(ready) == 1:
                pick = ready[0]
            elif any(stamps[a] is None for a in ready):
                pick = min(ready, key=number)
                by_number += [{"attempts": [pick, other], "why": "no ts"} for other in ready if other != pick]
            else:
                earliest = min(stamps[a] for a in ready)  # type: ignore[type-var]
                tied = [a for a in ready if stamps[a] == earliest]
                pick = min(tied, key=number)
                by_number += [{"attempts": [pick, other], "why": "same ts"} for other in tied if other != pick]
            order.append(pick)
            del waiting[pick]
            for later in later_of.get(pick, {}):
                if later in waiting:
                    waiting[later] -= 1
        if clash:
            attempts = sorted({a for pair in clash for a in pair}, key=number)
            named = sorted({f for a, b in clash for f in later_of[a][b]} | {f for a, b in clash for f in later_of.get(b, {}).get(a, ())})
            out[key] = {
                "order": sorted(stamps, key=number), "attempt_order": "numeric",
                "reason": f"the files disagree on the order of attempts {', '.join(map(str, attempts))}"
                          f" ({', '.join(named)}): ordered by number",
            }
        else:
            out[key] = {"order": order, "attempt_order": "first_seen", **({"by_number": by_number} if by_number else {})}
    return out


def attempt_order_across(
    files: Iterable[Iterable[Mapping[str, Any]]], names: Iterable[str] | None = None
) -> dict[tuple[str, Any], list[Any]]:
    """Each rank's attempts oldest first when its records are in several local files: the
    `order` of `attempt_order_report`."""
    return {key: info["order"] for key, info in attempt_order_report(files, names).items()}


def in_attempt_order(
    records: Iterable[Mapping[str, Any]], order: Mapping[tuple[str, Any], list[Any]]
) -> list[dict[str, Any]]:
    """`records` with each rank's attempts in `order` (oldest first, by `(run_id, rank)`): a
    stable sort, so the records of one attempt keep their order. An attempt `order` does not
    list keeps its place of first appearance after those it lists. Every reader that takes
    first appearance as the order (`resolve_attempts`, the detectors, the stream to the Rust
    detectors) then reads `order`."""
    listed = list(records)
    places: dict[tuple[str, Any], dict[Any, int]] = {}
    for key, attempts in order.items():
        places[key] = {attempt: at for at, attempt in enumerate(attempts)}
    for record in listed:
        known = places.setdefault((_run_of(record), record.get("rank", 0)), {})
        known.setdefault(record.get("attempt", 0), len(known))
    return sorted(listed, key=lambda r: places[(_run_of(r), r.get("rank", 0))][r.get("attempt", 0)])  # type: ignore[return-value]


def resolve_attempts_across(files: Iterable[Iterable[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """`resolve_attempts` over a rank's several files, with the attempts in `attempt_order_across`
    order: the records of every file with those a later attempt replaced left out, each rank's
    attempts in that order (`in_attempt_order`) and one attempt's records in the order the files
    give them. The order the files are given in never decides (contract 1): a reader that takes
    the later record at a step, or first appearance as the attempts' order (`scalar_points_in_order`),
    reads the later attempt last (BUG-21)."""
    listed = [list(records) for records in files]
    order = attempt_order_across(listed)
    return resolve_attempts(in_attempt_order([r for records in listed for r in records], order), order=order)


def check_one_run(records: Iterable[Mapping[str, Any]]) -> None:
    """Raise `ValueError` when `records` are of more than one run. Checked before attempts are
    resolved or any rule runs (lane U7, Nov 19, 2026): two jobs appending to one file gave evidence
    for the first run showing the second's incidents. The Rust detectors refuse the same rows
    (`detect_rows` in detect.rs)."""
    runs = sorted({_run_of(r) for r in records})
    if len(runs) > 1:
        raise ValueError(
            f"these records come from more than one run ({', '.join(repr(run) if run else 'no run_id' for run in runs)}):"
            " evidence is of one rank of one run, so read each run's records on their own. A record with no run_id"
            " (written before it existed) is of no named run, and joins only records that have none either"
        )


def run_of(record: Mapping[str, Any]) -> str:
    """A record's run ID, with "" for a record that names none (written before `run_id` existed,
    or not a string): as the Rust detectors read it (`server_detect.records_stream`), so rows
    with no run ID make one stream with each other and none with a named run."""
    run = record.get("run_id")
    return run if isinstance(run, str) else ""


# The private name `run_of` had until lane AD1, kept for the modules that import it (`divergence.py`,
# switched to `run_of` when wave AD is merged).
_run_of = run_of


_PART_NAME = re.compile(r"part-(-?\d+)-(-?\d+)\.parquet$")
# A scalars object's name, its first and last `seq` (WAVE_AL_CONTRACT.md section 4.5).
_SCALARS_NAME = re.compile(r"scalars-(-?\d+)-(-?\d+)\.parquet$")
_WHOLE = re.compile(r"-?[0-9]+")  # ASCII digits only: `int("²")` raises (lane V3's note)
ATTEMPT_ORDER_FILE = "attempts.json"


def _attempt_number(value: str) -> int | None:
    return int(value) if _WHOLE.fullmatch(value) else None


def stored_attempt_order(rank_folder: Path) -> tuple[list[int] | None, str]:
    """The rank's attempts oldest first, as the server wrote them to `attempts.json`
    (contract 1), and how the order was set: `"first_seen"`, `"numeric_then_first_seen"` (the
    file lists attempts that were in the store before it, `by: "store"`) or `"numeric"` (no
    readable file: the attempts go by number, as stores written before the file existed do)."""
    try:
        document = json.loads((rank_folder / ATTEMPT_ORDER_FILE).read_text(encoding="utf-8"))
        entries = document["attempts"]
        order = [entry["attempt"] for entry in entries]
    except (OSError, ValueError, KeyError, TypeError):
        return None, "numeric"
    if not all(isinstance(a, int) and not isinstance(a, bool) for a in order) or len(set(order)) != len(order):
        return None, "numeric"
    by_store = any(isinstance(entry, Mapping) and entry.get("by") == "store" for entry in entries)
    return order, "numeric_then_first_seen" if by_store else "first_seen"


def _attempt_place(order: list[int] | None, attempt: int) -> tuple[int, int]:
    """A sort key for `attempt` in a rank's stored order: by number without one; an attempt the
    file does not list goes just before the newest (contract 1, rule 3), by number among such."""
    if order is None:
        return (attempt, 0)
    if attempt in order:
        return (2 * order.index(attempt), 0)
    return (2 * len(order) - 3, attempt)


_RANK_FOLDER = re.compile(r"rank=([0-9]+)")


def rank_number(name: str) -> int | None:
    """The rank a folder named `rank=<n>` holds, or None for any other name. `<n>` is what the
    server writes (`server/src/storage.rs`): ASCII digits, a rank of at least 0. Every reader of
    the name asks this one (DET-17, Nov 19, 2026: some took `isdecimal()`, which lets in other
    scripts' digits, and some `lstrip("-").isdigit()`, which lets in `--1` and `²` and then
    raised in `int()`)."""
    found = _RANK_FOLDER.fullmatch(name)
    return int(found.group(1)) if found else None


def _other_rank_folder(entry: os.DirEntry[str], rank: int) -> bool:
    number = rank_number(entry.name)
    return number is not None and number != rank


def _parquet_parts(directory: Path, rank: int | None = None, *, scalars: bool = False) -> list[Path]:
    """The server's part files under `directory`, ordered by rank, attempt and first step.

    With `scalars`, also the attempts' `scalars-<first_seq>-<last_seq>.parquet` objects
    (WAVE_AL_CONTRACT.md section 4.5), each after its attempt's parts, by first `seq`, and one whose name
    does not parse after the named ones, by name. Without it they are never listed, as before them: only a
    reader of series reads them. This is the one listing of them in the package.

    Attempts go in the rank's stored order (`attempts.json`, contract 1), or by number in a
    store written before that file. With `rank`, the `rank=<n>` folders of other ranks directly under `directory` are not
    entered: the server writes a rank's records only into its own folder, so reading and
    discarding them would cost every other rank's share of the run on each look. Parts
    anywhere else are listed as before and their records filtered by the caller.
    """

    orders: dict[Path, list[int] | None] = {}

    def order(path: Path) -> tuple[int, tuple[int, int], int, int, str]:
        numbers = {"rank": 0, "attempt": 0}
        rank_folder: Path | None = None
        relative = path.relative_to(directory.parent).parts[:-1]
        for depth, parent in enumerate(relative):
            name, _, value = parent.partition("=")
            if name == "rank" and rank_number(parent) is not None:
                numbers[name] = int(value)
                rank_folder = directory.parent.joinpath(*relative[: depth + 1])
            elif name == "attempt" and (number := _attempt_number(value)) is not None:
                numbers[name] = number
        if rank_folder is None:
            rank_folder = path.parent.parent if path.parent.name.startswith("attempt=") else path.parent
        if rank_folder not in orders:
            orders[rank_folder] = stored_attempt_order(rank_folder)[0]
        family = 1 if path.name.startswith("scalars-") else 0
        match = (_SCALARS_NAME if family else _PART_NAME).search(path.name)
        place = _attempt_place(orders[rank_folder], numbers["attempt"])
        # A scalars object whose name does not parse (a server reads its footer instead) comes after the named
        # ones of its attempt; a part's keeps its place as before.
        first = int(match.group(1)) if match else (2**63 if family else 0)
        return numbers["rank"], place, family, first, path.name

    patterns = ("part-*.parquet", "scalars-*.parquet") if scalars else ("part-*.parquet",)
    found: list[Path] = []
    for pattern in patterns:
        if rank is None:
            found.extend(directory.rglob(pattern))
        else:
            found.extend(directory.glob(pattern))
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False) and not _other_rank_folder(entry, rank):
                        found.extend((directory / entry.name).rglob(pattern))
    return sorted(found, key=order)


def _checked_part(path: Path) -> list[dict[str, Any]]:
    """`_read_part`, each row's identity checked (`identity.check`) with the part and its row."""
    rows = _read_part(path)
    for row, record in enumerate(rows):
        check_identity(record, in_part(path, row))
    return rows


def _read_part(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError('reading Parquet parts needs pyarrow: pip install pyarrow') from error
    from .flight import row_to_record

    # `partitioning=None`: the part's own columns only. Left to itself, pyarrow before 22 also
    # reads `rank=<n>` and `attempt=<n>` out of the store's folder names, as dictionary
    # columns that clash with the file's own `rank` and `attempt` (seen with pyarrow 21.0.0).
    return [row_to_record(row) for row in pq.read_table(path, partitioning=None).to_pylist()]


def json_safe(value: Any) -> Any:
    """`value` with every NaN and infinity as the string the capture writes for it (`"nan"`,
    `"inf"`, `"-inf"`, as `sink._finite` does), so it is strict JSON. The evidence in memory keeps
    its floats; only what is written changes (lane U7, Nov 19, 2026: a bare NaN made the file
    unreadable to a strict JSON parser on exactly the job that was failing)."""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, Mapping):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def replace_atomically(path: Path, text: str) -> None:
    """Write `text` to `path` through a temporary file of this writer's own in the same folder
    and a rename, so a reader never sees half of it and two writers to one path never take each
    other's temporary file (lane U7, Nov 19, 2026: both used `<path>.tmp`, and the second rename
    raised `FileNotFoundError`). The file gets the usual permissions, as `write_text` gives them."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _json_text(value: Any) -> str:
    """`json.dumps(json_safe(value), separators=(",", ":"), allow_nan=False)`. A value that is plain
    JSON already (dicts, lists and tuples of finite numbers, strings, bools and None) is dumped as it
    is, which gives the same text, without the walk `json_safe` makes over every value (lane AD11:
    4.5 s of a 10,000-step watcher's update); anything else, a NaN among it, goes through `json_safe`."""
    try:
        return json.dumps(value, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError):
        return json.dumps(json_safe(value), separators=(",", ":"), allow_nan=False)


# Stands for the samples in the text of the rest of an evidence file (`_evidence_text`).
_SAMPLES_HERE = f"\x00tensorquill samples {secrets.token_hex(16)}\x00"


def _evidence_text(evidence: Mapping[str, Any], samples_text: str | None = None) -> str:
    """The text `write_evidence` writes. With `samples_text`, the text of `evidence["samples"]`
    made beforehand (`_Incremental.samples_text`), which is put in the place the samples have in the
    rest of the text, as `json.dumps` puts a list: "[" + its items joined by "," + "]"."""
    if samples_text is not None and "samples" in evidence:
        rest = _json_text({key: (_SAMPLES_HERE if key == "samples" else value) for key, value in evidence.items()})
        here = json.dumps(_SAMPLES_HERE)
        if rest.count(here) == 1:
            return rest.replace(here, samples_text) + "\n"
    return _json_text(evidence) + "\n"


def write_evidence(evidence: Mapping[str, Any], path: str | Path) -> Path:
    """Write `evidence` as strict JSON (`json_safe`) through a temporary file and a rename, so
    a reader polling the file never sees half of it."""
    return _write_evidence_text(_evidence_text(evidence), path)


def _write_evidence_text(text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_atomically(path, text)
    return path


def _write_output(evidence: Mapping[str, Any], path: Path, named: bool, compact_window: int | None,
                  samples_text: str | None = None) -> Path:
    """`write_evidence`, or with `compact_window` set the compact file instead (`compact.write_compact`).
    `samples_text` is the text of the samples made beforehand (`_Incremental.samples_text`)."""
    if compact_window is None:
        return _write_evidence_text(_evidence_text(evidence, samples_text), path)
    from .compact import write_compact

    return write_compact(evidence, path, window=compact_window, named=named)


class _Tail:
    """Reads what is new in a `.jsonl` file or a directory of Parquet parts.

    A file cut short, replaced or rewritten is read again from the start (lane AD11): a `.jsonl`
    file that is shorter than what was read, that is another file at the path (another device or
    inode), or whose first CHECKED_BYTES or last CHECKED_BYTES read have changed; and a part that was
    read and has changed (its size, time or file) or is gone. Before, such a file was read on from
    where it was, or not at all, and the evidence went on showing rows the source no longer holds.

    A `.jsonl` file whose modification time (`st_mtime_ns`) changed while it did not grow since the
    last read is read again from the start too (WAVE_AF_CONTRACT.md, section 3, EV-07): a rewrite in
    place of the same length and the same ends. An append, which also moves the time, grows the file
    and is still read as an append, from the offset. A `touch` that changes no byte is read again
    and gives the same records.

    The limit: a rewrite in place that keeps the same file, at least the same length, the same
    first and last CHECKED_BYTES (4 KiB) of what was read, and the same modification time (or one
    that also grows the file) is not detected, and the watcher goes on reading from its offset. A
    file cut to empty is read again but holds no record, so the watcher still writes nothing for it
    and the last evidence file stays as it was.

    `records` holds the training records read; the scalar-only records are set apart as they are
    read, into `scalars` (WAVE_AL_CONTRACT.md section 4.6), so the watcher judges the training
    records alone. A store's `scalars-*` objects are not listed here."""

    CHECKED_BYTES = 4096  # at the start of a `.jsonl` file, and before the end of what was read

    def __init__(self, source: Path, rank: int | None) -> None:
        self.source = source
        self.rank = rank
        self.records: list[dict[str, Any]] = []
        self.scalars: list[dict[str, Any]] = []  # the scalar-only records read, in their order
        self._offset = 0
        self._seen: set[Path] = set()
        self._stats: dict[Path, tuple[int, int, int]] = {}  # each part read: its size, time and file
        self._file: tuple[int, int] | None = None  # the `.jsonl` file read: its device and file
        # Its size and modification time at the last read, from the handle it was read through (EV-07).
        self._size_and_time: tuple[int, int] | None = None
        self._head = b""  # the first bytes read, and the last ones
        self._last = b""
        self.left_out = 0  # whole lines that were not a JSON object (lane U7, Nov 19, 2026)
        self.restarts = 0  # how often what was read was read again from the start
        self._lines = 0  # the lines of the `.jsonl` file read so far: a refusal names the next ones by number

    def _restart(self) -> None:
        self.records = []
        self.scalars = []
        self._offset = 0
        self._seen = set()
        self._stats = {}
        self._file = None
        self._size_and_time = None
        self._head = self._last = b""
        self.left_out = 0
        self._lines = 0
        self.restarts += 1

    @staticmethod
    def _stat(path: Path) -> tuple[int, int, int] | None:
        try:
            found = path.stat()
        except OSError:
            return None
        return found.st_size, found.st_mtime_ns, found.st_ino

    def _jsonl_changed(self) -> bool:
        """Whether the `.jsonl` file no longer holds what was read from it."""
        if not self._offset:
            return False
        try:
            found = self.source.stat()
            if (found.st_dev, found.st_ino) != self._file or found.st_size < self._offset:
                return True
            if self._size_and_time is not None:
                size, mtime = self._size_and_time
                if found.st_mtime_ns != mtime and found.st_size <= size:
                    # Written to since the last read without growing: rewritten in place (EV-07). The
                    # size compared is the file's at that read, not what was read of it, so a last
                    # line finished since is an append.
                    return True
            with self.source.open("rb") as file:
                head = file.read(len(self._head))
                file.seek(self._offset - len(self._last))
                last = file.read(len(self._last))
        except OSError:
            return True
        return head != self._head or last != self._last

    def poll(self) -> bool:
        """Add newly written records; return whether there were any, or whether what was read was
        read again from the start."""
        new: list[dict[str, Any]] = []
        restarted = False
        if self.source.is_dir():
            parts = _parquet_parts(self.source, self.rank)
            if self._seen:
                listed = set(parts)
                # A part read without its stat kept (one a caller marked as read) is checked for being there only.
                if any(part not in listed or (part in self._stats and self._stat(part) != self._stats[part])
                       for part in self._seen):
                    self._restart()
                    restarted = True
            for part in parts:
                if part not in self._seen:  # a part is written once and never changed
                    stat = self._stat(part)
                    new.extend(_checked_part(part))  # each row's identity (section 1)
                    self._seen.add(part)
                    self._stats[part] = stat if stat is not None else (-1, -1, -1)
        elif self.source.exists():
            if self._jsonl_changed():
                self._restart()
                restarted = True
            with self.source.open("rb") as file:
                file.seek(self._offset)
                data = file.read()
                found = os.fstat(file.fileno())
            # What the file was when read, whole lines or none: what a later look compares with (EV-07).
            self._size_and_time = (found.st_size, found.st_mtime_ns)
            end = data.rfind(b"\n") + 1  # a last line with no newline is still being written
            whole = data[:end].splitlines()
            for number, line in enumerate(whole, self._lines + 1):
                if line.strip():
                    try:
                        record = json.loads(line)
                    except ValueError:
                        record = None
                    if isinstance(record, dict):
                        # Checked as it is read, once (section 1): a record not valid refuses the file,
                        # and the watcher writes and replaces nothing for it.
                        check_identity(record, in_file(self.source, number))
                        new.append(record)
                    else:
                        self.left_out += 1  # a line torn by a killed process: counted, and said
            self._lines += len(whole)
            if end:
                if not self._offset:
                    self._file = (found.st_dev, found.st_ino)
                    self._head = data[:min(end, self.CHECKED_BYTES)]
                self._offset += end
                self._last = (self._last + data[max(0, end - self.CHECKED_BYTES):end])[-self.CHECKED_BYTES:]
        if self.rank is not None:
            new = [r for r in new if r.get("rank", 0) == self.rank]
        new, scalar_only = set_apart_scalars(new)
        self.records.extend(new)
        self.scalars.extend(scalar_only)
        # A scalar-only line changes no evidence (the watcher judges the training records), so it
        # alone is not news to the watcher.
        return bool(new) or restarted


# The kinds `tensorquill-server` works out itself and writes to a rank's `incidents.json`.
SERVER_KINDS = (
    "moe_imbalance", "starved_experts", "grad_spike", "update_spike", "stalled_weights", "repeated_steps",
    "loss_scale_trouble", "resume_lr_change", "resume_data_mismatch", "resume_incomplete", "data_stalled", "epoch_order_repeated",
    "activation_spike", "activation_drift", "dead_activations", *ATTN_KINDS, "non_finite", "replica_drift",
    *SYSTEM_KINDS,
)
# What each incident kind claims, on every incident as `class` (in `evidence.json`, and in the
# server's `incidents.json` and `watchdog.json`, whose table in `server/src/detect.rs` must be
# this one):
#   invariant    something that should never happen in a healthy run, by a fixed rule
#   anomaly      a statistical judgement against a baseline (the run's own history)
#   observation  context that does not claim a failure, such as an intended change at a resume
# Only invariants and anomalies make a run `unhealthy` (see `run_state`).
INCIDENT_CLASS_NAMES = ("invariant", "anomaly", "observation")
INCIDENT_CLASSES: dict[str, str] = {
    "moe_imbalance": "anomaly",
    "starved_experts": "anomaly",
    "grad_spike": "anomaly",
    "update_spike": "anomaly",
    "stalled_weights": "invariant",
    "repeated_steps": "invariant",
    "loss_scale_trouble": "invariant",
    "resume_lr_change": "observation",
    "resume_data_mismatch": "observation",
    "resume_incomplete": "observation",
    "data_stalled": "invariant",
    "epoch_order_repeated": "invariant",
    "duplicate_batches": "invariant",
    "activation_spike": "anomaly",
    "activation_drift": "anomaly",
    "dead_activations": "anomaly",
    "attention_collapse": "anomaly",
    "attention_uniform": "anomaly",
    "attention_logit_growth": "anomaly",
    "non_finite": "invariant",
    "replica_drift": "invariant",
    "memory_growth": "anomaly",
    "memory_pressure": "observation",
    "allocator_retries": "observation",
    "hardware_errors": "invariant",
    "gpu_throttled": "observation",
    "input_starvation": "anomaly",
    "memory_imbalance": "observation",
    "job_stalled": "observation",
    "rank_silent": "observation",
    "collective_hang": "invariant",
    "collective_desync": "invariant",
    "traincheck_violation": "invariant",
}


def incident_class(kind: Any) -> str | None:
    """The class of an incident kind (`INCIDENT_CLASSES`), or None for a kind this version
    does not know. Taken from the kind, never from an incident's own `class`."""
    return INCIDENT_CLASSES.get(kind) if isinstance(kind, str) else None


def _class_key(kind: Any) -> dict[str, str]:
    found = incident_class(kind)
    return {"class": found} if found is not None else {}


def with_classes(incidents: Iterable[dict[str, Any]]) -> None:
    """Put `class` on each incident of a known kind, right after `kind`, in place. A kind
    this version does not know keeps whatever it has."""
    for incident in incidents:
        found = incident_class(incident.get("kind"))
        if found is None:
            continue
        items = [(k, v) for k, v in incident.items() if k != "class"]
        at = next((i + 1 for i, (k, _) in enumerate(items) if k == "kind"), 0)
        items.insert(at, ("class", found))
        incident.clear()
        incident.update(items)


# --- What was checked ----------------------------------------------------------------------
#
# `evidence["kind_checks"]` gives every incident kind one of three values, so that "no
# incidents" can be read against what could have been found:
#   checked         records carried the kind's inputs and its rule ran over them (or the file
#                   it is read from was given and read)
#   not_configured  the capture was never set up to provide them: no record carries the
#                   input, or the file was not given. Nothing is said about the kind, and
#                   nothing is wrong with the monitoring
#   unavailable     the inputs were asked for and could not be judged: a `not_checked` entry
#                   names the kind (a probe that failed, a reading that could not be taken,
#                   a resume that could not be compared, a file that could not be read), or
#                   a server stores the run and its progress status is missing
# Only `unavailable` makes a run `degraded` (see `run_state`).
#
# A kind is `checked` from its first record, and a rule that is still warming up does not
# degrade a run. So each kind also says whether its rule has had enough history
# (`enough_history`), and the server's kinds give the two numbers it is worked out from:
# `records`, the records that carried the kind's input, against `records_needed`
# (`RECORDS_NEEDED`). "Nothing found" for a kind that is `checked` without enough history
# is not a verdict: its rule could not have fired yet.
KIND_CHECK_VALUES = ("checked", "not_configured", "unavailable")
# What each count of `input_counts` counts, as an evidence file names it under `needs`.
INPUT_NEEDS: dict[str, str] = {
    "moe": "moe: a router's expert counts",
    "grad_total": "grad_norm or clip_grad_norm",
    "update": "update_rms with grad_norm or clip_grad_norm (updates=True)",
    "weight_check": "weights_unchanged with grad_norms (full capture)",
    "steps": "step records",
    "loss_scale": "step_skipped (scaler=)",
    "lr": "lr",
    "data": "data (session.watch_loader)",
    "checkpoint": "checkpoint, or grad_scale (scaler=)",
    "layers": "layers (full capture)",
    "attention": "probe with attention (probe=)",
    "numbers": "layers, grad_norm, clip_grad_norm or grad_norms",
    "replica_spread": "replica_spread (replica_group=)",
    "cuda": "system.cuda (system=True)",
    "nvml": "system.nvml (system=True)",
    "data_wait": "data_wait with step_time (data_wait=True)",
}
# The input each of the server's kinds is judged from. `KIND_INPUTS` in server/src/detect.rs
# must be this table, and test_kind_checks.py holds the two identical.
KIND_INPUTS: dict[str, str] = {
    "moe_imbalance": "moe", "starved_experts": "moe", "grad_spike": "grad_total",
    "update_spike": "update", "stalled_weights": "weight_check", "repeated_steps": "steps", "loss_scale_trouble": "loss_scale", "resume_lr_change": "lr",
    "resume_data_mismatch": "data", "resume_incomplete": "checkpoint", "data_stalled": "data",
    "epoch_order_repeated": "data", "activation_spike": "layers", "activation_drift": "layers",
    "dead_activations": "layers", "attention_collapse": "attention", "attention_uniform": "attention",
    "attention_logit_growth": "attention", "non_finite": "numbers", "replica_drift": "replica_spread",
    "memory_growth": "cuda", "memory_pressure": "cuda", "allocator_retries": "cuda", "hardware_errors": "nvml",
    "gpu_throttled": "nvml", "input_starvation": "data_wait",
}
# How many records carrying a kind's input (`records`, as `input_counts` counts them) its
# rule needs: `records_needed` in `kind_checks`, and `enough_history` is `records >=
# records_needed`. One definition for every kind:
#
#   For each rule of the kind, take the fewest records it can report an incident on. The
#   kind needs the largest of those: from then on every one of its rules could have fired.
#
# A kind with one rule and no warm-up needs 1. The rules, and the fewest records each can
# report on (the largest is the entry below):
#   moe_imbalance         share (top-1, top-2) 1; collapse (top-k >= 3, during warm-up) 1;
#                         diversity drop (top-k >= 3) OVERLOAD_MIN_HISTORY earlier samples and
#                         the one judged
#   starved_experts       a full window: the `starve_window` option (STARVE_WINDOW)
#   grad_spike            the total: SPIKE_MIN_HISTORY earlier norms and the one judged, 11;
#                         one parameter: SPIKE_MIN_HISTORY earlier norms, then SPIKE_SUSTAIN in
#                         a row
#   update_spike          one record to compare the first with, the `update_window` option
#                         (UPDATE_WINDOW) changes that make the first baseline, and the one judged
#   stalled_weights       STALL_STEPS weight checks in a row
#   loss_scale_trouble    a scale below the floor 1; SCALE_SKIP_RUN consecutive scaler-confirmed skips
#   data_stalled          no batch: DATA_STALL_STEPS steps in a row, 3; the same batch: the
#                         batch and DATA_STALL_STEPS steps that repeat it
#   epoch_order_repeated  the shortest pass that can be repeated has two different batches,
#                         and the next pass must match both
#   activation_spike      ACT_MIN_HISTORY earlier records and the one judged
#   activation_drift      ACT_DRIFT_NEEDS: the module's first ACT_WINDOW records, which are never
#                         its baseline, then ACT_WINDOW records as the baseline and ACT_RECENT
#                         more (rules contract, part 2)
#   dead_activations      the first ACT_WINDOW records as the baseline and ACT_RECENT more
#   the attention kinds   ATTN_MIN_HISTORY probes as the baseline, then ATTN_SUSTAIN in a row
#   memory_growth         MEMORY_GROWTH_READINGS readings, each above the one before
#   hardware_errors       an Xid 1; an ECC count above the attempt's first reading 2
#   input_starvation      the share: one window of INPUT_WINDOW steps, 20; the rise: the first
#                         full window as the baseline and one more step
#   every other kind      1: the rule reads one record (`non_finite`, `replica_drift`,
#                         `memory_pressure`, `allocator_retries`, `gpu_throttled`), or compares a
#                         resume or a repeated step with what earlier attempts wrote, which the
#                         count of records leaves out once a later attempt has replaced them
#                         (`repeated_steps` and the three resume kinds)
#
# This is a floor, not a promise. `records` counts every record that carried the input, and
# a rule may count fewer: a parameter, module or router seen on fewer records has less
# history of its own; a total that is not finite or belongs to a skipped step is left out;
# the memory, hardware and data-wait rules start over at each attempt; an epoch must end
# before the next can be compared with it; and a starved expert also needs enough tokens in
# the window. So `enough_history` false means the kind's slowest rule cannot have fired, and
# true means the count has reached what that rule needs. A kind can report before it has
# enough history, through one of its rules that needs less. `server/src/detect.rs` has the
# same table (`RECORDS_NEEDED`) and the parity tests hold the two equal.
RECORDS_NEEDED: dict[str, int] = {
    "moe_imbalance": OVERLOAD_MIN_HISTORY + 1,
    "starved_experts": STARVE_WINDOW,
    "grad_spike": SPIKE_MIN_HISTORY + SPIKE_SUSTAIN,
    "update_spike": UPDATE_WINDOW + 2,
    "stalled_weights": STALL_STEPS,
    "repeated_steps": 1,
    "loss_scale_trouble": SCALE_SKIP_RUN,
    "resume_lr_change": 1,
    "resume_data_mismatch": 1,
    "resume_incomplete": 1,
    "data_stalled": DATA_STALL_STEPS + 1,
    "epoch_order_repeated": 4,
    "activation_spike": ACT_MIN_HISTORY + 1,
    "activation_drift": ACT_DRIFT_NEEDS,
    "dead_activations": ACT_WINDOW + ACT_RECENT,
    "attention_collapse": ATTN_MIN_HISTORY + ATTN_SUSTAIN,
    "attention_uniform": ATTN_MIN_HISTORY + ATTN_SUSTAIN,
    "attention_logit_growth": ATTN_MIN_HISTORY + ATTN_SUSTAIN,
    "non_finite": 1,
    "replica_drift": 1,
    "memory_growth": MEMORY_GROWTH_READINGS,
    "memory_pressure": 1,
    "allocator_retries": 1,
    "hardware_errors": 2,
    "gpu_throttled": 1,
    "input_starvation": INPUT_WINDOW + 1,
}
# The kinds a `not_checked` entry stands for when it lists none itself (`kinds`). The
# `attention` entry ("too few probes") is no longer written: it is read from evidence and
# server reports made before `enough_history`.
_CHECK_KINDS: dict[str, tuple[str, ...]] = {
    "resume_lr": ("resume_lr_change",),
    "resume_data": ("resume_data_mismatch",),
    "resume_state": ("resume_incomplete",),
    "attention": ATTN_KINDS,
    "probe": ATTN_KINDS,
    "update_size": ("update_spike",),
    "flight_recorder": ("collective_hang", "collective_desync"),
    "traincheck": ("traincheck_violation",),
}
# The `not_checked` entries that leave the kinds they name `checked`: some of a kind's judgments
# were not made and every other was. `clip_groups` (WAVE_AE_CONTRACT.md, section 1) names
# `update_spike`, whose pairs that needed no split were judged.
_PARTIAL_CHECKS = frozenset({"clip_groups"})


def _partial_check(item: Mapping[str, Any]) -> bool:
    """Whether a `not_checked` entry leaves the kinds it names `checked` (`_PARTIAL_CHECKS`): some
    of their judgments were made. A `clip_groups` entry whose `compared` is 0 says none was, and
    makes `update_spike` `unavailable` (WAVE_AE_CONTRACT.md, section 1, the author's second word);
    one without the key, written before it, reads as before it."""
    if item.get("check") not in _PARTIAL_CHECKS:
        return False
    compared = item.get("compared")
    return not (type(compared) is int and compared == 0)
_STATUS_NEEDS = "the server's progress status (watchdog.json)"
_NO_STATUS = "no progress status (the server's watchdog.json): whether the job is still making progress is not checked"


def input_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """How many of these records carry each input the server's kinds are judged from
    (`INPUT_NEEDS`). Give the records that count (`load_records` without `all_attempts`).

    "Carries" means the field holds something a rule reads: an empty `layers` or
    `grad_norms` does not count, an empty `weights_unchanged` does (a weight check that
    found every parameter changed), and a probe that failed carries no attention. A `layers`
    whose only entries are not objects carries no layer (`layer_entries`, row S5).
    """
    counts = dict.fromkeys(INPUT_NEEDS, 0)
    for record in records:
        counts["steps"] += 1
        moe = record.get("moe")
        if isinstance(moe, Mapping) and any(
                (total := _router_total(value)) is not None and total > 0 for value in moe.values()):
            counts["moe"] += 1
        # A total the column carries (`_carried`), as Rust counts it: a `grad_norm` of true is none (BUG-09).
        total = _carried(record.get("grad_norm")) or _carried(record.get("clip_grad_norm"))
        # A record holds a layer only through an entry that is an object (section 1.5; row S5, the
        # owner's approved change, Nov 24): a value that is not an object, or one whose only entries
        # are not objects, holds none, as Rust reads it.
        norms, layers = bool(_object(record, "grad_norms")), bool(layer_entries(record))
        counts["grad_total"] += total
        # A step the loss scaler skipped is no update history (RUST-12): the update rule skips it.
        # A map with no entry its column holds is an absent key (WAVE_AJ_CONTRACT.md section 2;
        # AQ5-D1, the author: "Python brought to Rust"), as `_weight_check` reads `weight_norms`.
        counts["update"] += total and _held_update_sizes(record) and not record.get("step_skipped")
        counts["weight_check"] += _unchanged_names(record) is not None and norms
        # A skip only the caller reported is not the loss scale's to judge unless the scaler was seen
        # to apply that update (`detect_loss_scale_trouble`).
        counts["loss_scale"] += record.get("step_skipped") is not None and (
            skipped_by(record) != "caller" or record.get("scaler_applied") is True)
        counts["lr"] += _lr_values(record) is not None
        counts["data"] += isinstance(record.get("data"), Mapping)
        # `resume_incomplete` judges a resume from a checkpoint manifest or from the loss scale
        # (E3): either is its input.
        counts["checkpoint"] += isinstance(record.get("checkpoint"), Mapping) or _finite(record.get("grad_scale")) is not None
        counts["layers"] += layers
        probe = record.get("probe")
        counts["attention"] += (
            isinstance(probe, dict) and "error" not in probe
            and isinstance(probe.get("attention"), dict) and bool(probe["attention"])
        )
        counts["numbers"] += layers or total or norms
        counts["replica_spread"] += bool(_held_spread_table(record))
        system = record.get("system")
        for part in ("cuda", "nvml"):
            counts[part] += isinstance(system, dict) and isinstance(system.get(part), dict)
        wait, took = _finite_number(record.get("data_wait")), _finite_number(record.get("step_time"))
        counts["data_wait"] += wait is not None and took is not None and took > 0 and wait >= 0
    return counts


def _finite_number(value: Any) -> float | None:
    return _finite(value)


def _kinds_not_checked(not_checked: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Each kind a `not_checked` entry names, with the first entry that names it."""
    named: dict[str, Mapping[str, Any]] = {}
    for item in not_checked:
        if not isinstance(item, Mapping) or _partial_check(item):
            continue
        kinds = item.get("kinds")
        if not isinstance(kinds, (list, tuple)):
            kinds = _CHECK_KINDS.get(str(item.get("check")), ())
        for kind in kinds:
            named.setdefault(str(kind), item)
    return named


def _unavailable(entry: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    return {"value": "unavailable", **extra, "check": entry.get("check"), "reason": entry.get("reason")}


def records_needed(starve_window: int = STARVE_WINDOW, update_window: int = UPDATE_WINDOW) -> dict[str, int]:
    """`RECORDS_NEEDED` as a run applies it: `starved_experts` needs the window the run was
    judged with (the `starve_window` option) and `update_spike` the baseline it was judged with
    (the `update_window` option, and two records more); every other kind its constant."""
    return {**RECORDS_NEEDED, "starved_experts": int(starve_window), "update_spike": int(update_window) + 2}


# What a kind's clean check does not show, said in its `kind_checks` entry as `note` when the
# kind is `checked`. `KIND_NOTES` in server/src/detect.rs must be this table.
KIND_NOTES: dict[str, str] = {
    "replica_drift": (
        "equal replicas do not rule out silent data corruption: a corrupted gradient is reduced with the others"
        " and the same wrong result goes back to every rank, so the replicas stay equal (SCOUT, arXiv 2608.11034,"
        " pp. 1 and 3)"
    ),
}


def _noted(kind: str, value: Any) -> dict[str, str]:
    return {"note": KIND_NOTES[kind]} if kind in KIND_NOTES and value == "checked" else {}


def kind_checks(
    counts: Mapping[str, int], not_checked: Iterable[Mapping[str, Any]], *, starve_window: int = STARVE_WINDOW,
    update_window: int = UPDATE_WINDOW,
) -> dict[str, dict[str, Any]]:
    """The value of each of the server's kinds (`KIND_INPUTS`), from how many records carried
    its input (`input_counts`) and the checks that could not be made.

    A kind a `not_checked` entry names is `unavailable`, with that entry's `check` and
    `reason`, however many records carried its input: a probe that failed once leaves a
    step unjudged, as one failed memory reading does. Otherwise it is `checked` when any
    record carried the input and `not_configured` when none did.

    Every entry also has `records`, `records_needed` (`RECORDS_NEEDED`, with the
    `starve_window` and the `update_window` the run was judged with) and `enough_history`,
    which is `records >= records_needed` whatever the value. A kind is `checked` from its first
    record, so it is `enough_history` that says whether "nothing found" is a verdict yet. A
    checked kind whose clean check leaves something open says so under `note` (`KIND_NOTES`).
    `tensorquill-server` gives the same object in its report (`kind_checks`).

    A `moe_routing` entry (routers recorded with no tokens, WAVE_AF_CONTRACT.md, section 8) makes
    the MoE kinds `unavailable` only when no router of the records was read (`counts["moe"]` is 0).
    While some router was, the kinds are judged on it as before, and the entry names the others.
    """
    if counts.get("moe", 0) > 0:
        not_checked = [item for item in not_checked
                       if not (isinstance(item, Mapping) and item.get("check") == MOE_ROUTING_CHECK)]
    named = _kinds_not_checked(not_checked)
    needed = records_needed(starve_window, update_window)
    checks: dict[str, dict[str, Any]] = {}
    for kind, counter in KIND_INPUTS.items():
        records = int(counts.get(counter, 0))
        history = {"records": records, "records_needed": needed[kind], "enough_history": records >= needed[kind],
                   "needs": INPUT_NEEDS[counter]}
        if kind in named:
            checks[kind] = _unavailable(named[kind], **history)
        else:
            value = "checked" if records > 0 else "not_configured"
            checks[kind] = {"value": value, **history, **_noted(kind, value)}
    return checks


# Said as `note` on a kind's entry that would read `not_configured` beside an incident of that
# kind (`checked_where_reported`): where the incident came from.
REPORTED_NOTES: dict[str, str] = {
    "resume_incomplete": (
        "reported from the loss scale (grad_scale), which this kind's records count leaves out: it counts"
        " checkpoint records only"
    ),
}
REPORTED_NOTE = "reported from input this entry's records count leaves out"


def checked_where_reported(
    checks: Mapping[str, Any], incidents: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """`checks` (a `kind_checks` object) with every one of the server's kinds (`SERVER_KINDS`)
    among `incidents` that it calls `not_configured` made `checked`, with a `note` on where the
    incident came from (`REPORTED_NOTES`): a kind judged from a rank's records that reported an
    incident was checked. `resume_incomplete` is judged from a checkpoint record and also from
    the loss scale alone, and its count is of checkpoint records only, so a resume that reset
    the loss scale was reported beside "never configured" (E3, Nov 19, 2026; both detectors).
    `records` and `enough_history` stay as counted (`enough_history` is `records >=
    records_needed`), the same in both detectors; counting the loss scale for the kind needs
    `KIND_INPUTS` changed in both, which is left open. The kinds that are not read from a rank's
    records (`other_kind_checks`) are left as they are: a rank's evidence calls the cross-rank
    `duplicate_batches` `not_configured` even when the server's status lists it."""
    reported = {i.get("kind") for i in incidents if isinstance(i, Mapping)}
    out = dict(checks)
    for kind in reported:
        check = out.get(kind)
        if kind in SERVER_KINDS and isinstance(check, Mapping) and check.get("value") == "not_configured":
            out[kind] = {**check, "value": "checked", "note": REPORTED_NOTES.get(str(kind), REPORTED_NOTE)}
    return out


def other_kind_checks(
    *,
    status: Mapping[str, Any] | None,
    status_expected: bool,
    not_checked: Iterable[Mapping[str, Any]],
    flight_recorder: bool,
    traincheck: bool,
) -> dict[str, dict[str, Any]]:
    """The value of each kind that is not read from one rank's step records.

    - `job_stalled` and `rank_silent` come from the server's progress status: `checked`
      when one was read; `unavailable` when one was expected (`status_expected`: the run is
      in a server's store, or a status was named) and there is none; `not_configured` for a
      local file no server was watching.
    - `collective_hang`, `collective_desync` and `traincheck_violation` come from files:
      `checked` when the file was given (`flight_recorder`, `traincheck`) and read,
      `unavailable` when it could not be read, `not_configured` when it was not given.
    - `duplicate_batches` and `memory_imbalance` compare ranks, which one rank's evidence
      cannot: `not_configured` here. The all-ranks view (`--rank all`) judges them.

    None of these is counted in records, so they have no `records` or `records_needed`.
    Their source is whole once it is read (the status is the server's own verdict, which
    waits out its silence thresholds there; a file holds a finished analysis), so
    `enough_history` is true exactly when the kind is `checked`.
    """
    named = _kinds_not_checked(not_checked)
    checks: dict[str, dict[str, Any]] = {}
    for kind in ("job_stalled", "rank_silent"):
        if status is not None:
            checks[kind] = {"value": "checked", "needs": _STATUS_NEEDS}
        elif status_expected:
            checks[kind] = _unavailable({"check": "status", "reason": _NO_STATUS}, needs=_STATUS_NEEDS)
        else:
            checks[kind] = {"value": "not_configured", "needs": _STATUS_NEEDS}
    for given, needs, kinds in (
        (flight_recorder, "PyTorch Flight Recorder dumps (flight_recorder=)", ("collective_hang", "collective_desync")),
        (traincheck, "TrainCheck results (traincheck=)", ("traincheck_violation",)),
    ):
        for kind in kinds:
            if kind in named:
                checks[kind] = _unavailable(named[kind], needs=needs)
            else:
                checks[kind] = {"value": "checked" if given else "not_configured", "needs": needs}
    for kind in ("duplicate_batches", "memory_imbalance"):
        checks[kind] = {"value": "not_configured", "needs": "every rank's report (--rank all)"}
    for kind, check in checks.items():
        # After `value`, before `needs`, where the server's kinds have it.
        checks[kind] = {"value": check["value"], "enough_history": check["value"] == "checked",
                        **{key: value for key, value in check.items() if key != "value"}}
    return checks


def kind_history(check: Any) -> str | None:
    """What a `kind_checks` entry says about a checked kind's history: `enough`,
    `not_enough` (its rule is still warming up: "nothing found" is not a verdict yet), or
    `not_stated` for an entry without `enough_history`, which a version before the field
    wrote and which is never read as enough. None for a kind that is not `checked`."""
    if not isinstance(check, Mapping) or check.get("value") != "checked":
        return None
    enough = check.get("enough_history")
    return "not_stated" if not isinstance(enough, bool) else "enough" if enough else "not_enough"


def checked_counts(evidence: Mapping[str, Any]) -> dict[str, Any] | None:
    """How many kinds have each value in an evidence file's `kind_checks`, with the names of
    the unavailable ones: `{"kinds", "checked", "not_configured", "unavailable", "unavailable_kinds"}`.
    Of the checked kinds, `not_enough_history` counts those whose rule is still warming up,
    named under `not_enough_history_kinds`, and `history_not_stated` those whose entry does
    not say (a file from before `enough_history`). None for evidence without the key.
    `kinds` is every kind this version knows (33)."""
    checks = evidence.get("kind_checks")
    if not isinstance(checks, Mapping):
        return None
    of = {value: [kind for kind in INCIDENT_CLASSES if isinstance(checks.get(kind), Mapping)
                  and checks[kind].get("value") == value] for value in KIND_CHECK_VALUES}
    history = {kind: kind_history(checks[kind]) for kind in of["checked"]}
    warming = [kind for kind, found in history.items() if found == "not_enough"]
    return {"kinds": len(INCIDENT_CLASSES), **{value: len(kinds) for value, kinds in of.items()},
            "unavailable_kinds": of["unavailable"], "not_enough_history": len(warming),
            "not_enough_history_kinds": warming,
            "history_not_stated": sum(found == "not_stated" for found in history.values())}


def kinds_by_value(evidence: Mapping[str, Any]) -> dict[str, Any] | None:
    """An evidence file's `kind_checks` by value, for a reader who wants names and not every
    entry. None for evidence without the key.

    - `checked`: the kinds whose rule ran with enough history. "Nothing found" is a verdict
      for these only.
    - `not_enough_history`: the kinds that are checked and still warming up, each with its
      `records` and `records_needed`. Their slowest rule could not have fired yet. One that
      is among the incidents all the same, found by a rule of the kind that needs fewer
      records, has `"reported": true`: the incident stands.
    - `not_configured`, and `unavailable` with each kind's reason.
    - `history_not_stated`, only when there are any: checked kinds of a file written before
      `enough_history`. They are not listed as checked, because the file does not say.
    """
    checks = evidence.get("kind_checks")
    if not isinstance(checks, Mapping):
        return None
    listed = [(kind, check) for kind, check in checks.items() if isinstance(check, Mapping)]
    history = {kind: kind_history(check) for kind, check in listed}
    not_stated = [kind for kind, _ in listed if history[kind] == "not_stated"]
    reported = _reported_kinds(evidence)
    return {
        "checked": [kind for kind, _ in listed if history[kind] == "enough"],
        "not_enough_history": {
            kind: {**{key: check[key] for key in ("records", "records_needed") if key in check},
                   **({"reported": True} if kind in reported else {})}
            for kind, check in listed if history[kind] == "not_enough"
        },
        "not_configured": [kind for kind, check in listed if check.get("value") == "not_configured"],
        "unavailable": {kind: check.get("reason") for kind, check in listed if check.get("value") == "unavailable"},
        **({"history_not_stated": not_stated} if not_stated else {}),
    }


def _reported_kinds(evidence: Mapping[str, Any]) -> set[Any]:
    incidents = evidence.get("incidents")
    return {i.get("kind") for i in incidents if isinstance(i, Mapping)} if isinstance(incidents, (list, tuple)) else set()


def checked_line(evidence: Mapping[str, Any]) -> str | None:
    """`9 of 33 kinds checked, 2 of them without enough history yet (grad_spike 5 of 13
    records, ...), 21 not configured, 2 unavailable (...)`, for the summary; None for
    evidence that does not say what was checked. The part about history is there only when a
    checked kind is still warming up, or when the file does not say. A warming-up kind that
    is among the incidents is marked "but reported by a faster rule", so the line is not read
    against the incident."""
    counts = checked_counts(evidence)
    if counts is None:
        return None
    line = f"{counts['checked']} of {counts['kinds']} kinds checked"
    if counts["not_enough_history"]:
        checks, reported = evidence["kind_checks"], _reported_kinds(evidence)
        each: list[str] = []
        unit = False  # " records" after the first count only
        for kind in counts["not_enough_history_kinds"]:
            seen, needed = checks[kind].get("records"), checks[kind].get("records_needed")
            starting = checks[kind].get("starting_state")
            if isinstance(starting, list) and starting and all(isinstance(r, Mapping) for r in starting):
                # A router still in the state it started in (rules contract, part 3): the kind's
                # records may be past its need, so their count would not say why it is short.
                said = f"{kind}: " + ", ".join(
                    f"router {r.get('router')} flagged since step 0 on {r.get('samples')} of {MOE_START_GRACE} samples"
                    for r in starting)
            else:
                said = f"{kind} {seen} of {needed}" if seen is not None and needed is not None else kind
                if not unit and " of " in said:
                    said, unit = said + " records", True
            each.append(said + (" but reported by a faster rule" if kind in reported else ""))
        line += f", {counts['not_enough_history']} of them without enough history yet ({', '.join(each)})"
    if counts["history_not_stated"]:
        line += f", {counts['history_not_stated']} of them with no word on their history (an older version wrote this)"
    line += f", {counts['not_configured']} not configured"
    if counts["unavailable"]:
        line += f", {counts['unavailable']} unavailable ({', '.join(counts['unavailable_kinds'])})"
    return line


# How long a watcher waits for the server's report to cover the newest stored part before
# it calls the report stale and works the incidents out itself. Measured with 64 ranks on an
# 8-core laptop: the report followed a part by 0.1 ms at the median and 77 ms at worst, and
# by 6 to 7 s while all 64 ranks were rebuilt from 10,000 stored steps after a restart or a
# new attempt. That rebuild grows with the run, so raise this for runs of 50,000 steps or more.
REPORT_GRACE_SECONDS = 30.0
# While a run is unfinished the server rewrites `watchdog.json` at least every
# `refresh_seconds`, which the status carries (three looks: 30 s by default), whether or not
# anything changed. Older than this many times that while it claims a running server, and
# nothing is watching the run for stalls.
STATUS_STALE_FACTOR = 2.0


def _own_rank(source: Path) -> int | None:
    if not source.is_dir():
        return None
    return rank_number(source.resolve().name)


def _rank_dirs(source: Path) -> list[Path]:
    """All `rank=*` directories directly under `source`, unless `source` is itself one."""
    if not source.is_dir() or _own_rank(source) is not None:
        return []
    return [path for path in source.glob("rank=*") if path.is_dir()]


def _rank_folder(source: Path, rank: int | None, dirs: list[Path] | None = None) -> Path | None:
    """The `rank=<n>` folder that `source` and `rank` name in a server's store, if any."""
    if not source.is_dir():
        return None
    own = _own_rank(source)
    if own is not None:
        return source if rank is None or own == rank else None
    if rank is not None:
        folder = source / f"rank={rank}"
        return folder if folder.is_dir() else None
    folders = _rank_dirs(source) if dirs is None else dirs
    return folders[0] if len(folders) == 1 else None


def _newest_part(folder: Path) -> tuple[tuple[int, int] | None, Path | None, int]:
    """The newest attempt stored in a rank's folder and its last step, read from the part
    names alone, with that part's path and how many parts have a name the server did not give.
    The newest attempt is the last in the rank's stored order (`attempts.json`, contract 1), or
    the highest number in a store written before that file."""
    # A walk of names with `os.scandir`, which builds no `Path` per part: with 200 parts a rank
    # (10,000 steps) that was 94% of a look at 64 ranks, and it grows with the run. The parts
    # counted, and the ones left out, are those `folder.rglob("part-*.parquet")` gives.
    newest: tuple[int, int] | None = None
    newest_key: tuple[tuple[int, int], int] | None = None
    newest_path: str | None = None
    unknown = 0
    order = stored_attempt_order(folder)[0]  # the rank's attempts in their stored order (contract 1)
    pending = [(str(folder), 0, "")]
    while pending:
        directory, depth, directory_name = pending.pop()
        try:
            entries = os.scandir(directory)
        except OSError:
            continue
        with entries:
            for entry in entries:
                name = entry.name
                if name.startswith("part-") and name.endswith(".parquet"):
                    match = _PART_NAME.search(name)
                    key, _, attempt = directory_name.partition("=")
                    number = _attempt_number(attempt)
                    if match is None or depth != 1 or key != "attempt" or number is None:
                        unknown += 1
                    else:
                        at = (number, int(match.group(2)))
                        ranked = (_attempt_place(order, number), at[1])
                        if newest_key is None or ranked > newest_key:
                            newest, newest_key, newest_path = at, ranked, entry.path
                if entry.is_dir(follow_symlinks=False):
                    pending.append((entry.path, depth + 1, name))
    return newest, Path(newest_path) if newest_path is not None else None, unknown


def _top_k_option(top_k: int | Mapping[str, int] | None) -> int | dict[str, int] | None:
    """`top_k` as it travels to the Rust detectors and comes back in a report's thresholds."""
    if top_k is None or isinstance(top_k, int):
        return top_k
    return {str(name): int(k) for name, k in top_k.items()}


# The six detector options: `build_evidence`'s, and the keys of the server's `--thresholds`.
THRESHOLD_NAMES = ("overload_threshold", "drift_tolerance", "starve_window", "starve_ratio", "top_k", "update_window")
THRESHOLD_DEFAULTS: dict[str, Any] = {
    "overload_threshold": OVERLOAD_THRESHOLD, "drift_tolerance": DRIFT_TOLERANCE,
    "starve_window": STARVE_WINDOW, "starve_ratio": STARVE_RATIO, "top_k": None,
    "update_window": UPDATE_WINDOW,
}


def _threshold_problem(name: str, value: Any) -> str | None:
    """Why the Rust detectors would refuse `value` for the option `name`, or None. The contract
    is `Thresholds::parse` in server/src/detect.rs, and so are the words: the three shares and
    tolerances finite and at least 0 (a whole number is one), the two windows whole and at
    least 1, `top_k` None, a whole number of at least 1, or one per router name. A bool is
    none of these (JSON's `true` is not a number)."""
    if name in ("overload_threshold", "drift_tolerance", "starve_ratio"):
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not number or not math.isfinite(value) or value < 0:
            return f"{name} must be a finite number of at least 0, got {value!r}"
    elif name in ("starve_window", "update_window"):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return f"{name} must be at least 1 and a whole number, got {value!r}"
    elif name == "top_k" and value is not None:
        each = list(value.values()) if isinstance(value, Mapping) else [value]
        for k in each:
            if isinstance(k, bool) or not isinstance(k, int):
                return f"top_k must be a whole number or one per router name, got {k!r}"
            if k < 1:
                return f"top_k must be at least 1, got {k}"
    return None


def check_thresholds(*, none_given: bool = False, **given: Any) -> None:
    """Raise `ValueError` for a detector option the Rust detectors refuse (`_threshold_problem`),
    so the Python detectors refuse the same ones (E6 / DET-15, Nov 19, 2026: they took a negative
    tolerance, `top_k=0` or `starve_window=0` and judged with it). `given` is any of the six
    `THRESHOLD_NAMES`; one left out is not checked, nor one that is None (`top_k` aside), which
    `watch` and the command line read as "not given". `none_given=True` checks None too, for
    `build_evidence`, whose options have no "not given"."""
    for name in THRESHOLD_NAMES:
        if name not in given or (given[name] is None and name != "top_k" and not none_given):
            continue
        problem = _threshold_problem(name, given[name])
        if problem is not None:
            raise ValueError(f"the detectors refuse these thresholds: {problem}")


def _given_thresholds(options: Mapping[str, Any]) -> dict[str, Any]:
    """The thresholds a caller gave: those of the six that are in `options` and not None.
    None is "not given" for every one of them, `top_k` included."""
    return {
        name: _top_k_option(options[name]) if name == "top_k" else options[name]
        for name in THRESHOLD_NAMES if options.get(name) is not None
    }


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _read_thresholds(report: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """`applied_thresholds`, and the names of the six that the report does not list in a form
    read here, each of which is given as this version's default (`thresholds_assumed`)."""
    listed = (report or {}).get("thresholds")
    listed = listed if isinstance(listed, Mapping) else {}
    out = dict(THRESHOLD_DEFAULTS)
    read: set[str] = set()
    for name in ("overload_threshold", "drift_tolerance", "starve_ratio"):
        value = listed.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            out[name] = value
            read.add(name)
    for name in ("starve_window", "update_window"):
        if _count(listed.get(name)):
            out[name] = listed[name]
            read.add(name)
    top_k = listed.get("top_k")
    if _count(top_k):
        out["top_k"] = top_k
    elif isinstance(top_k, Mapping) and top_k and all(_count(k) for k in top_k.values()):
        out["top_k"] = {str(name): k for name, k in top_k.items()}
    if "top_k" in listed and (top_k is None or out["top_k"] is not None):
        read.add("top_k")  # null is listed: "what the run recorded"
    return out, [name for name in THRESHOLD_NAMES if name not in read]


def applied_thresholds(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """The six thresholds a server's report says it applied (its `thresholds`), in the shape
    `build_evidence` takes them. One the report does not list reads as the default: a server
    from before `--thresholds` applied the defaults and listed no `top_k`. Which ones were read
    so is `thresholds_assumed`."""
    return _read_thresholds(report)[0]


def thresholds_assumed(report: Mapping[str, Any] | None) -> list[str]:
    """The names of the six thresholds that a server's report does not list in a form this
    version reads (not there, as from a server older than the threshold, or of another type),
    and that `applied_thresholds` therefore gives as this version's defaults, which that server
    may not have applied. Empty for a report of this version's server, which lists all six."""
    return _read_thresholds(report)[1]


def _judged_thresholds(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """The six thresholds one rank's evidence was judged with: those it lists as applied
    (`detection.thresholds`, a server's or this process's), else `build_evidence`'s `thresholds`,
    which it lists only when one is not the default, else the six defaults."""
    detection = evidence.get("detection")
    if isinstance(detection, Mapping) and isinstance(detection.get("thresholds"), Mapping):
        return {**THRESHOLD_DEFAULTS, **detection["thresholds"]}
    if isinstance(evidence.get("thresholds"), Mapping):
        return {**THRESHOLD_DEFAULTS, **evidence["thresholds"]}
    return dict(THRESHOLD_DEFAULTS)


def _thresholds_by_rank(sets: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each set of thresholds the ranks were judged with, with its ranks, in the order of the
    first rank judged with it: the all-ranks view's `thresholds_by_rank` entries."""
    judged_by: dict[str, dict[str, Any]] = {}
    for number, values in sets.items():
        judged_by.setdefault(json.dumps(values, sort_keys=True), {"ranks": [], "thresholds": dict(values)})["ranks"].append(number)
    return [{"ranks": found["ranks"], "rank_ranges": _rank_ranges(found["ranks"]), "thresholds": found["thresholds"]}
            for found in judged_by.values()]


def _same_threshold(asked: Any, applied: Any) -> bool:
    """Whether a report applied the value asked for. A float that went to the server as JSON
    text may come back one unit in the last place apart, which is the same threshold."""
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (asked, applied))
    if numbers and (isinstance(asked, float) or isinstance(applied, float)):
        return math.isclose(asked, applied, rel_tol=1e-12, abs_tol=0.0)
    return bool(asked == applied)


def _threshold_text(value: Any) -> str:
    if value is None:
        return "what the run recorded"  # `top_k` left to the records
    return json.dumps(dict(value), sort_keys=True) if isinstance(value, Mapping) else repr(value)


def _off_default(values: Any) -> list[str]:
    """`name value (default: D)` for each of the six thresholds in `values` that is not the default."""
    values = values if isinstance(values, Mapping) else {}
    return [
        f"{name} {_threshold_text(values[name])} (default: {_threshold_text(THRESHOLD_DEFAULTS[name])})"
        for name in THRESHOLD_NAMES
        if name in values and not _same_threshold(values[name], THRESHOLD_DEFAULTS[name])
    ]


def thresholds_line(evidence: Mapping[str, Any]) -> str | None:
    """One line that names every applied threshold that is not the default, with its value and
    the default; None when all six are the defaults.

    A reader of a server's report takes the thresholds the server applied (`read_server_report`),
    so a run judged by other values than the defaults must say so wherever it is summarised.
    It reads what the evidence lists as applied: `detection.thresholds` for one rank of a
    server's store, whether the server judged or this process did, and `thresholds` in the
    all-ranks view, or every set of `thresholds_by_rank` with its ranks when the judged ranks
    differ. Evidence with no `detection` (a local `.jsonl` file, or a store's parts read without
    `--watch`) is `build_evidence`'s: it lists the six as `thresholds` only when one is not the
    default, so at the defaults it lists none and gives None, as evidence from before the key does.
    The ranks put together by `combine_rank_evidence` (`compare`'s, which has `judged_ranks` and no
    `view`) are read as the all-ranks view is, set by set: some may be judged by their server's
    report and some here (lane AD1).
    """
    if evidence.get("view") == ALL_RANKS_VIEW or ("judged_ranks" in evidence and "detection" not in evidence):
        who = "the server applied" if evidence.get("view") == ALL_RANKS_VIEW else "the ranks were judged with"
        sets = evidence.get("thresholds_by_rank")
        if isinstance(sets, list) and len(sets) > 1:
            named = []
            for found in sets:
                ranks = [r for r in found.get("ranks") or () if isinstance(r, int)]
                off = _off_default(found.get("thresholds"))
                if off:
                    named.append(f"on rank{'s' if len(ranks) != 1 else ''} {_rank_ranges(ranks, 8)}: {', '.join(off)}")
            off = ["; ".join(named)] if named else []
        else:
            off = _off_default(evidence.get("thresholds"))
    else:
        detection = evidence.get("detection")
        if isinstance(detection, Mapping):
            who = "the server applied" if detection.get("source") == "server" else "the detectors ran here with"
            off = _off_default(detection.get("thresholds"))
        else:  # a local file, or a store's parts read without --watch: judged here
            who = "the detectors ran here with"
            off = _off_default(evidence.get("thresholds"))
    if not off:
        return None
    return f"Thresholds that are not the defaults: {who} {', '.join(off)}."


def read_server_report(
    folder: str | Path,
    *,
    overload_threshold: float | None = None,
    drift_tolerance: float | None = None,
    starve_window: int | None = None,
    starve_ratio: float | None = None,
    top_k: int | Mapping[str, int] | None = None,
    update_window: int | None = None,
) -> dict[str, Any]:
    """Read a rank's `incidents.json` from a server's store and say whether it can be used.

    The thresholds are the server's (`tensorquill-server --thresholds`), and the report lists
    the ones it applied (`applied_thresholds`). A reader takes those: a threshold left at None
    here is not compared. Only one given here that differs from what the server applied makes
    the report `other_thresholds`. `top_k=None` is "not given" too, so a reader cannot ask for
    "what the run recorded" against a server that was started with a `top_k`.

    `folder` is `runs/<run_id>/rank=<rank>`. The report has no clock in it; what makes it
    current is that it covers exactly what is stored: its `attempt` and `last_step` are
    those of the newest stored part, which are in the part's path, so no part is opened.
    An empty `incidents` list means "no incidents" only in a report that is current.

    Returns `{"state", "reason", "stored", "report"}`: `stored` is `(attempt, last step)`
    of the newest part or None, `report` the parsed file or None, and `state` one of

    - `current`: the report covers every stored step, and no threshold given here differs
      from the one the server applied.
    - `nothing_stored`: no parts and no report yet.
    - `missing`: parts, but no report (`--no-detect`, a server older than its detectors,
      or the moment between the first part and its report).
    - `behind`: the report stops before the newest stored part. Normal for a moment after
      each part; lasting, it means detection is off or failed, or the server died between
      storing a part and reporting on it.
    - `unreadable`: the file could not be read or parsed this time.
    - `unusable`: not a report for this rank, from a server with fewer kinds, ahead of the
      store, or beside parts whose names say nothing about their steps.
    - `other_thresholds`: a threshold given here differs from the one the server applied.

    A threshold the detectors refuse (`check_thresholds`: a negative or non-finite number, a
    window or `top_k` below 1 or not whole, a bool) raises `ValueError` before anything is read,
    as `build_evidence` does (L15, Nov 20, 2026: it read as `other_thresholds`, or as `current`
    for a bool equal to the server's value).
    """
    check_thresholds(overload_threshold=overload_threshold, drift_tolerance=drift_tolerance,
                     starve_window=starve_window, starve_ratio=starve_ratio, top_k=top_k,
                     update_window=update_window)
    folder = Path(folder)
    stored, _, unknown = _newest_part(folder)
    order, attempt_order = stored_attempt_order(folder)

    def result(state: str, reason: str, report: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"state": state, "reason": reason, "stored": stored, "report": report, "attempt_order": attempt_order}

    try:
        text = (folder / "incidents.json").read_text(encoding="utf-8")
    except FileNotFoundError:
        if stored is None and not unknown:
            return result("nothing_stored", "nothing is stored for this rank yet")
        return result("missing", "the server wrote no incidents.json for this rank (it ran with --no-detect, or is"
                      " older than its detectors)")
    except OSError as error:
        return result("unreadable", f"incidents.json cannot be read ({error})")
    try:
        report = json.loads(text)
    except ValueError:
        return result("unreadable", "incidents.json is not complete JSON")
    if (
        not isinstance(report, dict) or report.get("schema_version") != 1 or report.get("source") != "tensorquill-server"
        or not isinstance(report.get("incidents"), list) or not isinstance(report.get("attempt"), int)
    ):
        return result("unusable", "incidents.json is not a report this version can read")
    if f"rank={report.get('rank')}" != folder.resolve().name or str(report.get("run_id")) != folder.resolve().parent.name:
        return result("unusable", f"incidents.json is for run {report.get('run_id')!r} rank {report.get('rank')!r}, not this folder")
    absent = [kind for kind in SERVER_KINDS if kind not in (report.get("checked") or ())]
    if absent:
        return result("unusable", f"the server's report does not cover {', '.join(absent)} (an older server wrote it)", report)
    if unknown or stored is None:
        return result("unusable", "the stored parts are not named by attempt and step, so the report cannot be dated", report)
    applied = report.get("thresholds") or {}
    if applied.get("spike_sustain") != SPIKE_SUSTAIN:
        # A server from before Nov 16, 2026 judged gradients by their total alone and named
        # stalled weights at the first onset only: its report is not what this version gives.
        return result("unusable", "the server's report was made without the per-parameter gradient and stalled-weight"
                      " rules (an older server wrote it)", report)
    asked = _given_thresholds({
        "overload_threshold": overload_threshold, "drift_tolerance": drift_tolerance,
        "starve_window": starve_window, "starve_ratio": starve_ratio, "top_k": top_k,
        "update_window": update_window,
    })
    # A server started with `--thresholds` lists what it applied, `top_k` among them (null: what
    # the run recorded). A server from before that option has no `top_k` there, which reads the same.
    # Only what the reader gave is compared: the rest is taken as the server applied it.
    differing = [name for name, value in asked.items() if not _same_threshold(value, applied.get(name))]
    if differing:
        what = ", ".join(f"{name} {applied.get(name)!r} where {asked[name]!r} was asked for" for name in differing)
        return result("other_thresholds", f"the server applied {what}", report)
    last_step = report.get("last_step")
    reported = (report["attempt"], last_step if isinstance(last_step, int) else -(2**63))
    where = f"step {last_step} of attempt {report['attempt']}"
    if reported == stored:
        return result("current", f"the server's report is current through {where}", report)
    # Attempts compared by their place in the rank's stored order (contract 1), not by number.
    if (_attempt_place(order, reported[0]), reported[1]) < (_attempt_place(order, stored[0]), stored[1]):
        return result("behind", f"the server's report stops at {where} and the store holds step {stored[1]} of"
                      f" attempt {stored[0]}", report)
    return result("unusable", f"the server's report reaches {where}, past the stored step {stored[1]} of"
                  f" attempt {stored[0]}", report)


def _detection(look: Mapping[str, Any], source: str, rank: int | None) -> dict[str, Any]:
    """Where an evidence file's incidents came from, for its `detection` key."""
    report, stored = look.get("report") or {}, look.get("stored")
    return {
        "source": source,
        "state": look["state"],
        "reason": look["reason"],
        "rank": report.get("rank", rank),
        **({"stored_attempt": stored[0], "stored_last_step": stored[1]} if stored else {}),
        **({f"report_{key}": report[key] for key in ("attempt", "last_step", "steps", "rebuilds") if key in report}),
        # How the store's attempts were ordered (contract 1): `first_seen`, `numeric_then_first_seen`
        # or `numeric` for a store written before the order was kept.
        **({"attempt_order": look["attempt_order"]} if look.get("attempt_order") else {}),
    }


def _report_evidence(
    folder: Path,
    look: Mapping[str, Any],
    status: Mapping[str, Any] | None,
    *,
    traincheck: str | Path | None = None,
    flight_recorder: str | Path | None = None,
    flight_recorder_prefix: str | None = None,
) -> dict[str, Any]:
    """Evidence built from a current server report and the server's status, with no part read
    (one is, the newest, when a stuck collective has to be dated against the last step).

    The incidents are in the order `build_evidence` gives them. It has no `samples`,
    `weight_trends` or router sizes: those are per-step data, which stays in the parts. The
    probe's `head_trends` are the report's, which a server made them for from every stored
    probe (L15, Nov 20, 2026); a report from an older server has none, and neither does this.
    """
    report = look["report"]
    record_incidents = [dict(incident) for incident in report["incidents"]]
    incidents = [i for i in record_incidents if i.get("kind") != "replica_drift"]
    if status is not None and isinstance(status.get("incidents"), list):
        stream_incidents = [dict(incident) for incident in status["incidents"]]
    else:  # a status from a server that did not write the list yet
        stream_incidents = detect_watchdog_incidents(status)
    incidents.extend(stream_incidents)
    # The server lists these two as not checked in every report, since it never has their
    # files. Here they are checked when asked for, as in `build_evidence`.
    not_checked = [
        dict(item) for item in report.get("not_checked") or ()
        if isinstance(item, dict) and item.get("check") not in ("flight_recorder", "traincheck")
    ]

    def newest_ts() -> float | None:
        _, part, _ = _newest_part(folder)
        return _newest_stamp(_read_part(part) if part else ())

    last_step = report["last_step"] if isinstance(report.get("last_step"), int) else 0
    traincheck_summary, recorder_summary = _check_other_sources(
        incidents, stream_incidents, not_checked, last_step=last_step, newest_ts=newest_ts,
        traincheck=traincheck, flight_recorder=flight_recorder, flight_recorder_prefix=flight_recorder_prefix,
    )
    incidents.extend(i for i in record_incidents if i.get("kind") == "replica_drift")
    with_classes(incidents)
    # The same values at the top as in `detection.thresholds` (lane AD1: the top took the report's
    # values as they were, and `detection` only those it can read, so one that is not a threshold
    # the detectors take was on top and the default below).
    applied, assumed = _read_thresholds(report)
    own_rank = report.get("rank") if isinstance(report.get("rank"), int) else _own_rank(folder)
    evidence = {
        "schema_version": 1,
        "run_id": report.get("run_id"),
        "rank": own_rank,
        "run": _manifest.run_block(report.get("run_id"), _manifest.find_manifest(folder, report.get("run_id"))),
        "overload_threshold": applied["overload_threshold"],
        "starve_window": applied["starve_window"],
        "starve_ratio": applied["starve_ratio"],
        "incidents": incidents,
        "samples": [],
        # `thresholds`: the six the server judged by, which a reader takes unless it gave its own;
        # `thresholds_assumed`: those of them the report does not list, given as the defaults.
        "detection": {**_detection(look, "server", None), "moe_flagged": report.get("moe_flagged"),
                      "thresholds": applied, **({"thresholds_assumed": assumed} if assumed else {})},
        **({"not_checked": not_checked} if not_checked else {}),
        # The server's count of skipped updates by who reported them, passed on unchanged.
        **({"skipped_updates": dict(found)} if isinstance(found := report.get("skipped_updates"), Mapping) else {}),
        # The server's `checkpoint_tensors` block (CHECKPOINT_TENSOR_CONTRACT.md, section 4), unchanged.
        **({"checkpoint_tensors": dict(served)} if isinstance(served := report.get("checkpoint_tensors"), Mapping) else {}),
        **({"head_trends": dict(heads)} if isinstance(heads := report.get("head_trends"), Mapping) and heads else {}),
        # `moe_source` as the server carried it from the records (rules contract, part 7; wave AD's seam):
        # without it a reader of a server's report could not say whose assignments the counts are.
        **({"moe_source": dict(source)} if isinstance(source := report.get("moe_source"), Mapping) and source else {}),
        # The four measurements (rules contract, part 6) where the server's report has them, unchanged.
        **{name: dict(block) for name in MEASUREMENT_BLOCKS if isinstance(block := report.get(name), Mapping) and block},
        **({"flight_recorder": recorder_summary} if recorder_summary else {}),
        **({"traincheck": traincheck_summary} if traincheck_summary else {}),
        **({"watchdog": {**{k: status[k] for k in (
            "summary", "written_at_ms", "world_size", "pauses", "recommended_silent_seconds",
            "silent_floor_seconds", "critical_seconds",
        ) if k in status}, **_rank_dropped_rows_said(status, own_rank)}} if status else {}),
    }
    # What was checked: the server's own values for the kinds it judges, and this reader's for
    # the rest. A store is expected to hold the server's status. A report from a server older
    # than the key has none, and then this evidence does not say what was checked.
    rank_checks = report.get("kind_checks")
    if isinstance(rank_checks, Mapping) and all(
        isinstance(rank_checks.get(kind), Mapping) and rank_checks[kind].get("value") in KIND_CHECK_VALUES
        for kind in SERVER_KINDS
    ):
        others = other_kind_checks(
            status=status, status_expected=True, not_checked=not_checked,
            flight_recorder=flight_recorder is not None, traincheck=traincheck is not None,
        )
        evidence["kind_checks"] = checked_where_reported({
            kind: dict({**rank_checks, **others}[kind]) for kind in INCIDENT_CLASSES
        }, incidents)
    evidence["run_state"] = _per_rank_run_state(evidence, status, _status_age(status), steps_judged=bool(report.get("steps")),
                                                attempt=_whole_attempt(report.get("attempt")))
    return evidence


# --- One run state -----------------------------------------------------------------------
#
# `evidence["run_state"]`: one word for the whole picture, with every reason behind it.
#   healthy    every expected source is current and no invariant or anomaly is listed
#   degraded   a kind is `unavailable` (see "What was checked"): a check listed in
#              `not_checked`, or no progress status where one is expected, so stalls are not
#              watched. A kind that is `not_configured` does not degrade a run: a local
#              file that no server was watching has no status to miss
#   unknown    required evidence is stale or missing: no steps judged, a server report that is
#              not current, ranks without a current report, an old status, a stopped server,
#              or a status summary other than `active` or `finished` (a stall among them: the
#              stream cannot tell a hang from a pause or a dead sender)
#   unhealthy  an invariant or anomaly is listed, or a kind this version does not know
#              (incidents are never closed, so it stays so)
# Observations (`INCIDENT_CLASSES`) are listed as a reason of their own whose state is
# `healthy`: they never make the state worse than the other reasons do.
# The worst reason wins. `healthy` is only what is left when nothing at all was found
# against it, and every status summary or report state not named here reads `unknown`.
RUN_STATES = ("healthy", "degraded", "unknown", "unhealthy")
_STATUS_HEALTHY = ("active", "finished")
# The rank states a server's `watchdog.json` writes (`RankState` in watchdog.rs, and contract 3's
# `capture_stopped`). A reader meeting another one says so and never calls the run healthy.
RANK_STATES = ("active", "starting", "paused", "silent", "unresponsive", "reconnecting", "lost", "finished",
               "capture_stopped")


def _reason(state: str, code: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"state": state, "code": code, **extra, "detail": detail}


def _whole_attempt(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def newest_attempt(records: Iterable[Mapping[str, Any]]) -> int | None:
    """The newest attempt among `records` (a record that names none is attempt 0), or None for
    none: the one that first appears last (contract 1), not the highest number."""
    places = dict.fromkeys(_whole_attempt(r.get("attempt", 0)) or 0 for r in records)
    return next(reversed(places), None)


def _not_current_by_status(status: Mapping[str, Any], judged: Mapping[int, int | None]) -> dict[int, str]:
    """The judged ranks that the server's status says were judged by an earlier attempt, with why:
    the status marks the rank `attempt_without_steps` (the job was started again and the new
    attempt has stored nothing), or names another attempt than the one judged. The rule of
    `run_state` in server/src/watchdog.rs (lane U6), applied here (L6, Nov 19, 2026). A status
    entry that names no attempt (a server from before attempts) and an attempt not known here
    (None) are not compared.

    Nor is an entry written before the rank's first batch: attempt 0 with `last_step` null and no
    `attempt_without_steps`. The server gives a rank whose stream is open and that has stored
    nothing attempt 0, its word for "none yet" as well as the attempt of rows that name none, and
    the file is rewritten only every `--watchdog-seconds` (10 by default), so that entry can stand
    for seconds beside a report on steps the rank has stored since. Read as "attempt 0", it left
    the first rank to send out of the all-ranks view (CI, Nov 20, 2026). The server's own rule
    compares in memory, where the two are never out of step.

    Nor, for the same reason, is an entry marked `attempt_without_steps` for the very attempt that
    was judged (lane AD1, Nov 21, 2026). The server marks a new attempt so from its first heartbeat
    until its first batch is stored; a report of that attempt, current with the store, was made
    from batches it stored since, so the entry is a file written before them, whichever of the two
    files was written last. Read as "started again", it left the rank unjudged for up to
    `--watchdog-seconds`. A mark for another attempt than the one judged, or beside a judgment
    whose attempt is not known, is read as before: an earlier attempt's."""
    found: dict[int, str] = {}
    for entry in status.get("ranks") or ():
        if not isinstance(entry, Mapping):
            continue
        rank = entry.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank not in judged:
            continue
        shown, was = _whole_attempt(entry.get("attempt")), judged[rank]
        if entry.get("attempt_without_steps") is True and shown is not None and shown == was:
            continue  # written before this attempt's first batch, which the judgment covers
        if entry.get("attempt_without_steps") is True:
            found[rank] = (f"started again{f' as attempt {shown}' if shown is not None else ''}, which has stored no"
                           " step yet: what was judged is an earlier attempt's")
        elif shown == 0 and "last_step" in entry and entry["last_step"] is None:
            continue  # written before the rank's first batch: it names no attempt yet
        elif shown is not None and was is not None and shown != was:
            found[rank] = f"attempt {was} was judged, and the server's status shows attempt {shown}"
    return found


def run_state(
    *,
    incidents: Iterable[Mapping[str, Any]],
    steps_judged: bool,
    status: Mapping[str, Any] | None,
    status_age: float | None = None,
    not_checked: Iterable[Mapping[str, Any]] = (),
    detection: Mapping[str, Any] | None = None,
    unknown_ranks: Mapping[int, str] | None = None,
    judged_ranks: Iterable[int] = (),
    status_expected: bool = True,
    judged_attempts: Mapping[int, int | None] | None = None,
    report_awaited: bool = False,
) -> dict[str, Any]:
    """The run's state from what the evidence holds: `{"state", "reasons"}`.

    `report_awaited`: the watcher is inside its wait for the server's next report
    (`report_grace`), so a `detection` that is not current lists `report_not_current` with the
    state `healthy`, which decides nothing, and the last current result stands (lane AD1). Outside
    the wait it is `unknown`, as before.

    `status` is the server's whole `watchdog.json` (None when there is none) and `status_age`
    how long ago it was last written when that is too long (`_status_age`). A missing status
    is the reason `no_status` (degraded) only when one is expected (`status_expected`): for a
    local file that no server was watching, the stall kinds are not configured and the
    missing status is no reason at all. `detection` is the
    evidence's `detection` (where the incidents came from), `unknown_ranks` the ranks whose
    reports are not current with why and `judged_ranks` those that are (the all-ranks view).
    A rank the status expects (`missing_ranks`, or below `world_size`) that is neither listed in
    it nor judged is unknown too: the status calls a rank missing only after its silence floor,
    and until then a run of one connected rank out of four would otherwise read healthy.
    `judged_attempts` is the attempt each judged rank's incidents come from (None when not
    known): a rank the status marks `attempt_without_steps`, or shows at another attempt, is
    unknown too, as the server judges it (`_not_current_by_status`). Its incidents stay listed:
    they were found in what it stored.
    An incident's class is looked up from its kind (`incident_class`): observations go to the
    `observations` reason, whose state is `healthy`, and every other incident, a kind this
    version does not know among them, to `incidents`. Codes are stable: `incidents`;
    `no_steps`, `report_not_current`, `ranks_unknown`, `status_old`, `server_stopped`,
    `status_summary`, `capture_stopped`, `rank_state_unknown`; `no_status`, `not_checked`;
    `observations`.

    `capture_stopped` (contract 3, Nov 19, 2026): a rank whose attempt stopped capture while
    training went on is not judged, whatever state it reads (`capture_stopped`, or
    `unresponsive`, `reconnecting` or `lost` beside the kept stop): unknown, with its ranks and
    each one's reason. `rank_state_unknown`: a rank state this version does not know (a newer
    server's) is unknown too, never healthy.
    """
    reasons: list[dict[str, Any]] = []
    listed = list(incidents)
    observed = [i for i in listed if incident_class(i.get("kind")) == "observation"]
    incidents = [i for i in listed if incident_class(i.get("kind")) != "observation"]
    if incidents:
        kinds = list(dict.fromkeys(str(i.get("kind")) for i in incidents))
        reasons.append(_reason("unhealthy", "incidents", f"{len(incidents)} incident{'s' if len(incidents) != 1 else ''}"
                               f" listed: {', '.join(kinds)}", kinds=kinds))
    if not steps_judged:
        reasons.append(_reason("unknown", "no_steps", "no step has been judged yet"))
    if detection is not None and detection.get("source") == "server" and detection.get("state") != "current":
        if report_awaited:
            # Inside the watcher's wait for a report (lane AD1, CLI-10): the last current result is
            # shown as it is, as the all-ranks watcher shows it, beside a `freshness` block when the
            # store holds steps it does not cover. Listed, and not deciding.
            reasons.append(_reason("healthy", "report_not_current",
                                   f"the incidents are from the server's last current report, while its next one is"
                                   f" awaited: {detection.get('reason')}"))
        else:
            reasons.append(_reason("unknown", "report_not_current",
                                   f"the incidents are from the server's last current report: {detection.get('reason')}"))
    unknown = dict(unknown_ranks or {})
    if status is not None:
        judged = set(judged_ranks)
        listed = {r.get("rank") for r in status.get("ranks") or () if isinstance(r, Mapping)}
        world_size = status.get("world_size")
        expected = {r for r in status.get("missing_ranks") or () if isinstance(r, int) and not isinstance(r, bool)}
        if isinstance(world_size, int) and not isinstance(world_size, bool) and 0 < world_size <= MAX_EXPECTED_RANKS:
            expected |= set(range(world_size)) - listed
        expected |= set(_missing_counted(status))  # those the status counts and does not list (lane AR-4)
        for rank in sorted(expected - judged):
            unknown.setdefault(rank, "the server's status expects it and it has not connected")
        for rank, why in _not_current_by_status(status, judged_attempts or {}).items():
            unknown.setdefault(rank, why)
    if unknown:
        ranks = sorted(unknown)
        reasons.append(_reason("unknown", "ranks_unknown", f"{len(ranks)} rank{'s' if len(ranks) != 1 else ''} without"
                               f" a current report ({_rank_ranges(ranks, 8)}): nothing is said about incidents on"
                               " them", ranks=ranks))
    if status is not None:
        entries = [r for r in status.get("ranks") or () if isinstance(r, Mapping)]
        stopped = {r.get("rank"): r["capture_stopped"] for r in entries if r.get("capture_stopped") is not None}
        if stopped:
            ranks = sorted(stopped, key=str)
            why = {rank: (stop.get("reason") if isinstance(stop, Mapping) else None) or "unknown" for rank, stop in stopped.items()}
            # The step each capture stopped at, where the status says (rules contract, part 7): step 0
            # and null ("before any step") stay apart, and a status without the key gives none.
            at = {rank: stop["step"] for rank, stop in stopped.items() if isinstance(stop, Mapping) and "step" in stop}
            said = {rank: f"{why[rank]}, {words}" if (words := capture_stopped_step_words(stopped[rank])) else why[rank]
                    for rank in ranks}
            reasons.append(_reason(
                "unknown", "capture_stopped",
                f"capture stopped on {len(ranks)} rank{'s' if len(ranks) != 1 else ''} while training went on"
                f" ({', '.join(f'rank {rank}: {said[rank]}' for rank in ranks)}): nothing is judged there since",
                ranks=ranks, reasons={str(rank): why[rank] for rank in ranks},
                **({"steps": {str(rank): at[rank] for rank in ranks if rank in at}} if at else {})))
        strange = {r.get("rank"): r.get("state") for r in entries if r.get("state") not in RANK_STATES}
        if strange:
            ranks = sorted(strange, key=str)
            words = list(dict.fromkeys(str(strange[rank]) for rank in ranks))
            reasons.append(_reason(
                "unknown", "rank_state_unknown",
                f"the server's status gives a rank state this reader does not know ({', '.join(map(repr, words))},"
                f" {len(ranks)} rank{'s' if len(ranks) != 1 else ''}): a newer server's, so nothing is said about them",
                ranks=ranks, states=words))
    if status is None:
        if status_expected:
            reasons.append(_reason("degraded", "no_status", _NO_STATUS))
    else:
        summary = status.get("summary")
        if summary == "server_stopped":
            reasons.append(_reason("unknown", "server_stopped", "the server has stopped: no new steps are stored and"
                                   " nothing watches the run for stalls"))
        elif status_age is not None:
            reasons.append(_reason("unknown", "status_old", f"the server's status was last written {status_age:.0f} s"
                                   " ago: nothing is watching the run for stalls"))
        if summary not in (*_STATUS_HEALTHY, "server_stopped"):
            reasons.append(_reason("unknown", "status_summary", f"the server's status says {summary!r}",
                                   summary=summary))
    checks = [item for item in not_checked if isinstance(item, Mapping)]
    if checks:
        names = list(dict.fromkeys(str(item.get("check")) for item in checks))
        reasons.append(_reason("degraded", "not_checked", f"not checked: {', '.join(names)}", checks=names))
    if observed:
        kinds = list(dict.fromkeys(str(i.get("kind")) for i in observed))
        reasons.append(_reason("healthy", "observations", f"{len(observed)} observation{'s' if len(observed) != 1 else ''}"
                               f" listed, not counted as failures: {', '.join(kinds)}", kinds=kinds))
    state = max((r["state"] for r in reasons), key=RUN_STATES.index, default="healthy")
    return {"state": state, "reasons": reasons}


def run_state_line(evidence: Mapping[str, Any]) -> str | None:
    """`Run state: <state>` with the reasons that decide it, for the CLI; None without one."""
    found = evidence.get("run_state")
    if not isinstance(found, Mapping):
        return None
    state = found.get("state")
    deciding = [r["detail"] for r in found.get("reasons") or () if r.get("state") == state]
    fresh = freshness_text(evidence.get("freshness"))
    return f"Run state: {state}" + (f" ({'; '.join(deciding)})" if deciding else "") + (f"; {fresh}" if fresh else "")


# --- Steps newer than the last judgment (CLI-10, lane AD1) ---------------------------------------
#
# `evidence["freshness"]`, present only when the judgment does not cover the newest stored step
# under the thresholds in force (the same names as `runs.py`'s rows):
#   {"current": false, "judged_to_step": <int or null>, "records_to_step": <int or null>, "why": ...}
#   records_newer       the store holds steps the last judgment does not cover. The last result is
#                       shown as it is, and its state is not made unknown for it
#   thresholds_changed  the last judgment was made under other thresholds than those in force: no
#                       earlier result is carried over, and the state is unknown
#   nothing_judged      steps are stored and none is judged: the state is unknown
FRESHNESS_WHY = ("records_newer", "thresholds_changed", "nothing_judged")


def freshness_block(judged_to_step: Any, records_to_step: Any, why: str) -> dict[str, Any]:
    """The `freshness` block (see above): a whole step or None for each of the two steps."""
    whole = lambda step: step if isinstance(step, int) and not isinstance(step, bool) else None  # noqa: E731
    return {"current": False, "judged_to_step": whole(judged_to_step), "records_to_step": whole(records_to_step),
            "why": why}


def freshness_text(block: Any) -> str | None:
    """`as of step {judged_to_step}; records reach step {records_to_step}` for a `freshness` block,
    with "nothing judged yet" or "records reach no step" for a step that is null; None without one."""
    if not isinstance(block, Mapping) or block.get("current") is not False:
        return None
    judged, stored = block.get("judged_to_step"), block.get("records_to_step")
    first = f"as of step {judged}" if isinstance(judged, int) else "nothing judged yet"
    second = f"records reach step {stored}" if isinstance(stored, int) else "records reach no step"
    text = f"{first}; {second}"
    return text + (" (judged under other thresholds than those in force)" if block.get("why") == "thresholds_changed" else "")


def _per_rank_run_state(evidence: Mapping[str, Any], status: Mapping[str, Any] | None,
                        status_age: float | None, steps_judged: bool, attempt: int | None = None,
                        report_awaited: bool = False) -> dict[str, Any]:
    """`run_state` of one rank's evidence. `attempt` is the attempt its incidents were judged
    from: the report's, or the newest of the records the detectors read here."""
    # Evidence that says what was checked says whether a status was expected at all; without
    # the key (the server's report read by a watcher) one is, as before.
    checks = evidence.get("kind_checks")
    stall = checks.get("job_stalled") if isinstance(checks, Mapping) else None
    return run_state(
        incidents=evidence.get("incidents", ()), steps_judged=steps_judged, status=status, status_age=status_age,
        not_checked=evidence.get("not_checked", ()), detection=evidence.get("detection"),
        status_expected=not (isinstance(stall, Mapping) and stall.get("value") == "not_configured"),
        judged_attempts={rank: attempt} if isinstance(rank := evidence.get("rank"), int) and steps_judged else None,
        report_awaited=report_awaited,
    )


def _status_age(status: Mapping[str, Any] | None) -> float | None:
    """Seconds since a status that claims a running server was written, when that is too
    long ago for the server to be still writing it; None otherwise. A status with no
    `refresh_seconds` is from a server that wrote only on a change, and its age says nothing."""
    written, refresh = (status or {}).get("written_at_ms"), (status or {}).get("refresh_seconds")
    if not status or status.get("server_running") is not True or not isinstance(written, (int, float)):
        return None
    if status.get("summary") == "finished" or not isinstance(refresh, (int, float)) or refresh <= 0:
        return None  # every rank closed its stream: there is nothing left to watch
    age = time.time() - written / 1000.0
    return age if age > STATUS_STALE_FACTOR * refresh else None


class _StatusAgeTracker:
    """Track `watchdog.json` staleness across polls without trusting cross-host wall clocks alone.

    On poll 1, when clocks agree, this matches `_status_age`. Once `written_at_ms` advances
    during watching, or fails to advance for `STATUS_STALE_FACTOR * refresh_seconds` of local
    monotonic time, staleness comes from `time.monotonic()` directly.
    """

    def __init__(self) -> None:
        self._written: int | float | None = None
        self._seen_at: float | None = None
        self._observed_live = False

    def check(self, status: Mapping[str, Any] | None) -> float | None:
        if not status:
            return None
        written, refresh = status.get("written_at_ms"), status.get("refresh_seconds")
        if (
            status.get("server_running") is not True
            or status.get("summary") == "finished"
            or not isinstance(written, (int, float))
            or not isinstance(refresh, (int, float))
            or refresh <= 0
        ):
            self._written = None
            self._seen_at = None
            self._observed_live = False
            return None
        now = time.monotonic()
        if self._written is None:
            self._written = written
            self._seen_at = now
            self._observed_live = False
        elif written != self._written:
            self._written = written
            self._seen_at = now
            self._observed_live = True
        assert self._seen_at is not None
        mono_age = now - self._seen_at
        stale_after = STATUS_STALE_FACTOR * refresh
        if self._observed_live:
            return mono_age if mono_age > stale_after else None
        if mono_age > stale_after:
            return mono_age
        return _status_age(status)


def _status_stamp(status: Mapping[str, Any] | None, status_age: float | None) -> Any:
    return status and (
        status["summary"],
        tuple(status.get("missing_ranks", ())) + _omitted_stamp(status),
        json.dumps(status.get("pauses", {}), sort_keys=True),
        status.get("recommended_silent_seconds"),
        tuple(
            (r.get("rank"), r.get("state"), r.get("last_step"), r.get("severity"), r.get("phase"),
             r.get("attempt"), r.get("attempt_without_steps"))  # a job started again is news (L6)
            for r in status["ranks"]
        ),
        status_age is not None,
        # Where each stopped rank is, not for how long: a stack that arrives, or moves, is news.
        tuple((r.get("rank"), stack_signature(r)) for r in status["ranks"] if isinstance(r, Mapping) and r.get("stack")),
    )


def _omitted_stamp(status: Mapping[str, Any] | None) -> tuple[Any, ...]:
    """The count of missing ranks a status does not list, for the watcher's stamps: nothing for a
    status with no count, so its stamp is as before."""
    omitted = _missing_omitted(status)
    return (("omitted", omitted),) if omitted else ()


def _grace_alert_key(status: Mapping[str, Any] | None, status_age: float | None, other_stamp: Any) -> Any:
    """Status/external-source changes that should surface immediately during `--report-grace`:
    stalls, silent/missing ranks, stopped/stale server status, and TrainCheck/FR updates, while
    ignoring routine `last_step` bumps on a healthy status."""
    incidents = (
        status["incidents"]
        if status and isinstance(status.get("incidents"), list)
        else (detect_watchdog_incidents(status) if status else [])
    )
    summary = status.get("summary") if status and status.get("summary") in ("stalled", "degraded", "server_stopped") else None
    return (
        summary,
        tuple(status.get("missing_ranks", ())) + _omitted_stamp(status) if status else (),
        json.dumps(incidents, sort_keys=True),
        status_age is not None,
        other_stamp,
    )


def _detectors_option(detectors: Any, server_binary: Any = None, *, stacklevel: int = 3) -> str:
    """`detectors` as "python" or "rust". "auto" is decided here, once: Rust when the server
    binary and `pyarrow` are there, and otherwise Python with one `RuntimeWarning` that says
    what is missing and what installs it (`server_detect.resolve_detectors`)."""
    if detectors not in ("auto", "python", "rust"):
        raise ValueError(f"detectors must be 'auto', 'python' or 'rust', not {detectors!r}")
    if detectors != "auto":
        return str(detectors)
    from .server_detect import resolve_detectors

    chosen, why = resolve_detectors("auto", server_binary)
    if why is not None:
        warnings.warn(why, RuntimeWarning, stacklevel=stacklevel)  # at the caller of the public function
    return chosen


def _stat_key(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _other_sources_stamp(
    *,
    traincheck: str | Path | None,
    flight_recorder: str | Path | None,
    flight_recorder_prefix: str | None,
) -> tuple[Any, Any]:
    """Lightweight stat signature for TrainCheck and Flight Recorder inputs."""
    tc_key: Any = None
    if traincheck is not None:
        path = Path(traincheck)
        if path.is_dir():
            files = sorted([*path.glob("violations_summary.json"), *path.glob("*/violations_summary.json")])
            tc_key = ("dir", tuple((str(f), _stat_key(f)) for f in files))
        elif path.is_file():
            tc_key = ("file", _stat_key(path))
        else:
            tc_key = ("missing",)
    fr_key: Any = None
    if flight_recorder is not None:
        directory = Path(flight_recorder)
        if directory.is_dir():
            try:
                files = sorted(
                    p for p in directory.rglob("*")
                    if p.is_file() and (flight_recorder_prefix is None or flight_recorder_prefix in p.name)
                )
                fr_key = ("dir", tuple((str(p.relative_to(directory)), _stat_key(p)) for p in files))
            except OSError:
                fr_key = ("error",)
        else:
            fr_key = ("missing",)
    return (tc_key, fr_key)


def _thresholds_moved(look: Mapping[str, Any], judged: Mapping[str, Any] | None) -> bool:
    """Whether the rank's newest report, `behind` its store, lists other thresholds than the report
    `judged` that was last current: its server was started again with others, so a result judged
    under the old ones is not the run's under those in force (`thresholds_changed`, lane AD1)."""
    newest = look.get("report")
    if look.get("state") != "behind" or not isinstance(newest, Mapping) or not isinstance(judged, Mapping):
        return False
    now, then = applied_thresholds(newest), applied_thresholds(judged)
    return any(not _same_threshold(now[name], then[name]) for name in THRESHOLD_NAMES)


def _report_freshness(look: Mapping[str, Any], judged: bool) -> dict[str, Any] | None:
    """The `freshness` block of evidence built from `look`'s report (CLI-10, lane AD1), or None when
    the report covers the newest stored step: `records_newer` beside a report kept while the next
    one is awaited, `nothing_judged` when it judged no step and steps are stored. Read from the
    report and the newest part's name (`look["stored"]`): no part is opened."""
    report, stored = look.get("report") or {}, look.get("stored")
    if not judged:
        return freshness_block(None, stored[1], "nothing_judged") if stored else None
    if look.get("state") == "current" or not stored:
        return None
    if (report.get("attempt"), report.get("last_step")) == tuple(stored):
        return None  # e.g. a report torn for a moment, with nothing stored since the last current one
    return freshness_block(report.get("last_step"), stored[1], "records_newer")


# `watch` keeps the costly detectors' states between looks (`_Incremental`, lane AD11). False judges
# every record again at every look, as before; the tests compare the two.
_INCREMENTAL_WATCH = True


def _follow_for(build_options: Mapping[str, Any], old: Any = None) -> Any:
    """A new `server_detect.FollowedDetect` for a watcher judging with the Rust detectors, `old`
    closed first; None with the Python detectors, or when the watcher is told to judge every
    record at every look (`_INCREMENTAL_WATCH` False, which the tests compare with)."""
    if old is not None:
        old.close()
    if build_options.get("detectors") != "rust" or not _INCREMENTAL_WATCH:
        return None
    from .server_detect import FollowedDetect

    return FollowedDetect(build_options.get("server_binary"))


def watch(
    source: str | Path,
    output: str | Path | None = None,
    *,
    interval: float = 2.0,
    rank: int | str | None = None,
    default_rank: int | None = None,
    polls: int | None = None,
    on_update: Any = None,
    server_reports: bool = True,
    report_grace: float = REPORT_GRACE_SECONDS,
    compact_window: int | None = None,
    trigger: Any = None,
    **build_options: Any,
) -> Path | None:
    """Keep the evidence file up to date while a run is going, until interrupted.

    `rank="all"` is the all-ranks view of a run folder: one evidence file for every
    `rank=<n>` folder in it, in a shape of its own (see `watch_all_ranks`).

    `source` is a `.jsonl` file a job is writing, or a directory `tensorquill-server` is
    writing Parquet parts into. Every `interval` seconds it is looked at again and, when
    something changed, the evidence file is replaced by rename. `polls` stops after that
    many looks, for scripts and tests; `on_update(evidence, path)` is called after each
    write. Returns the evidence path, or None if nothing was ever written.

    For a rank's folder in a server's store, the incidents are taken from the report the
    server keeps there (`incidents.json`) and from its status, and no part is read, as long
    as `read_server_report` calls that report current. The evidence then has no per-step
    `samples`. When the report is missing or unusable, or has not caught up with the stored
    parts `report_grace` seconds after it was last current, the parts are read and the
    detectors run here, as they do for a `.jsonl` file, at every change (only the new records
    for the costliest detectors, as said below). While waiting for a report inside `report_grace`, status updates and
    external sources (`flight_recorder`, `traincheck`) still trigger fresh writes from the
    last current report. `default_rank`, when set alongside `rank=None`, locks onto that
    rank if a watched run directory ends up holding more than one `rank=*` folder.
    `evidence["detection"]` says which happened and why. `server_reports=False` always
    works the incidents out here.

    Inside the wait, the last current report is shown as it is and its state is not made unknown
    for the wait, as the all-ranks watcher shows it; the steps stored since are said beside it,
    `freshness` with `why: "records_newer"` (CLI-10, lane AD1; see `freshness_block`). A report
    that lists other thresholds than the last current one ends the wait at once
    (`thresholds_changed`): that result is not carried over, and the parts are judged here.
    When the report covers the newest stored step there is no `freshness` key.

    When the detectors run here, the costly ones (the MoE samples and their rules, `grad_spike`, the
    activation kinds, `non_finite`) keep their state between looks and are fed only the records that
    are new, as long as every record judged before is still judged, in its place, with the same
    options; otherwise they judge every record again (`_Incremental`, lane AD11). The evidence is the
    same either way. The `samples` handed to `on_update` are the ones kept for the next look: read
    them, do not change them in place.

    A `.jsonl` file cut short, replaced or rewritten under the watcher is read again from the start,
    and so is a store whose parts read have changed or are gone (`_Tail`); a rewrite in place of the
    same length is seen by the file's modification time. The limit: a rewrite in place that keeps
    the same file, at least the same length, the same first and last 4 KiB of what was read and the
    same modification time is not detected, and the watcher goes on from its offset. A file cut to empty still
    writes nothing: the last evidence file is left as it was until the file holds a record again.

    The five thresholds (`overload_threshold`, `drift_tolerance`, `starve_window`,
    `starve_ratio`, `top_k`) are the caller's only when given. For a server's store, one left
    out is the server's: its report is taken with what it applied, and when the detectors have
    to run here they run with those values too, so the incidents do not move with who judged
    them. One given here that differs from the server's makes its report `other_thresholds`,
    and the detectors run here with it. `evidence["detection"]["thresholds"]` lists the five
    that were applied. For a `.jsonl` file one left out is `build_evidence`'s default.

    `detectors="rust"` (with `server_binary`, as in `build_evidence`) judges through
    `tensorquill-server detect` whenever the detectors run here: on every rebuild for a
    `.jsonl` file, and for a server's store when its report cannot be used. Each rebuild sends
    every record read so far. A missing binary or
    `pyarrow` raises `RuntimeError` before the first look, not at the first rebuild.
    `detectors="auto"` is decided once, before the first look, as `build_evidence` decides it.
    The default here is "python"; the command line's `--watch` asks for "rust" (`main`).

    `compact_window` writes the compact file instead of the full one (`compact.compact_evidence`,
    that many recorded steps each side of a MoE incident), as `<name>.compact.json` unless
    `output` names the file; `on_update` still gets the full evidence.
    `trigger` (a `trigger.Trigger`) fires once for each new incident: it is told of every look
    and handed each evidence written, and never raises into the watcher.
    """
    # Said before the first look, not at the first rebuild hours into a run.
    check_thresholds(**{name: build_options[name] for name in THRESHOLD_NAMES if name in build_options})
    if isinstance(rank, str):
        if rank != ALL_RANKS:
            raise ValueError(f"rank must be a rank number or {ALL_RANKS!r}, not {rank!r}")
        return watch_all_ranks(
            source, output, interval=interval, polls=polls, on_update=on_update,
            server_reports=server_reports, report_grace=report_grace, compact_window=compact_window,
            trigger=trigger, **build_options,
        )
    source = Path(source)
    # "auto" is decided once, here: every rebuild then uses the same detectors, and a fallback
    # to Python is warned of once and not at every rebuild.
    build_options["detectors"] = _detectors_option(
        build_options.get("detectors", "python"), build_options.get("server_binary"))
    if build_options["detectors"] == "rust":
        # Said now, while someone is at the keyboard, not at the first rebuild hours into a run.
        from .server_detect import rust_ready

        rust_ready(build_options.get("server_binary"))
    picked_rank = rank
    tail = _Tail(source, picked_rank)
    # The costly detectors' states, kept between looks and fed only the new records (lane AD11).
    memo = _Incremental() if _INCREMENTAL_WATCH else None
    # With the Rust detectors, the binary's detectors kept between looks (wave AF-W), as `memo` keeps
    # the Python ones: one `detect --follow` child, judging only what was appended.
    follow = _follow_for(build_options)
    written: Path | None = None
    done = 0
    status_file = build_options.pop("watchdog", None)
    status_file = Path(status_file) if isinstance(status_file, (str, Path)) else _watchdog_beside(source)
    # The thresholds the caller gave. A server's report is taken with the ones it applied, and
    # is refused only for one given here that differs (`read_server_report`).
    thresholds = _given_thresholds(build_options)
    for name in THRESHOLD_NAMES:
        build_options.pop(name, None)
    # What the rank's server applied, of the thresholds not given here: the detectors run with
    # these when they have to run here, so the incidents do not change with who judged them.
    adopted: dict[str, Any] = {}
    other_sources = {
        name: build_options.get(name) for name in ("traincheck", "flight_recorder", "flight_recorder_prefix")
    }
    folder: Path | None = None
    waiting_since: float | None = None
    was_current = False
    last_current_report: dict[str, Any] | None = None
    emitted: Any = None
    emitted_grace: Any = None
    status_clock = _StatusAgeTracker()
    others_seen: set[int] = set()  # the ranks seen when the watcher locked onto `default_rank`
    told_absent: list[int] | None = None
    try:
        while polls is None or done < polls:
            if done:
                time.sleep(interval)
            done += 1
            if trigger is not None:
                trigger.tick()
            # A hung job writes no new records, so the server's status file and any local
            # external sources are read on every poll, even while waiting on --report-grace.
            status = read_watchdog(status_file) if status_file is not None else None
            status_age = status_clock.check(status)
            stamp = _status_stamp(status, status_age)
            other_stamp = _other_sources_stamp(**other_sources)

            dirs = _rank_dirs(source) if picked_rank is None else None
            if picked_rank is None and default_rank is not None and dirs is not None:
                world_size = status.get("world_size") if status else None
                if len(dirs) > 1 or (dirs and isinstance(world_size, int) and world_size > 1):
                    picked_rank = default_rank
                    others_seen = {n for d in dirs if (n := rank_number(d.name)) is not None}
                    others_seen |= {int(r.get("rank", 0)) for r in tail.records}
                    if any(r.get("rank", 0) != picked_rank for r in tail.records):
                        emitted = None
                        emitted_grace = None
                        tail = _Tail(source, picked_rank)
                        memo = _Incremental() if _INCREMENTAL_WATCH else None
                        follow = _follow_for(build_options, follow)
                    else:
                        tail.rank = picked_rank
                    dirs = None
            new_folder = _rank_folder(source, picked_rank, dirs) if server_reports else None
            if new_folder != folder:
                if folder is not None and picked_rank is not None:
                    emitted = None
                    emitted_grace = None
                waiting_since, was_current, last_current_report = None, False, None
                adopted = {}
            folder = new_folder

            look: dict[str, Any] | None = None
            report_look: dict[str, Any] | None = None
            in_grace = False
            if folder is not None:
                look = read_server_report(folder, **thresholds)
                if look["state"] in ("current", "behind", "other_thresholds"):  # this rank's report, readable here
                    adopted = {n: v for n, v in applied_thresholds(look["report"]).items() if n not in thresholds}
                if look["state"] == "current":
                    waiting_since, was_current = None, True
                    last_current_report = look["report"]
                    report_look = look
                elif look["state"] in ("missing", "behind", "unreadable") and was_current and last_current_report is not None:
                    # A part is stored first and reported on a moment later, so these are
                    # normal for that moment. The wait is counted from the last look that
                    # found the report current, not from the newest part: a report that
                    # never catches up must not hide behind the parts that keep arriving.
                    # A watcher that has not seen a current report yet does not wait at all.
                    waiting_since = time.monotonic() if waiting_since is None else waiting_since
                    if time.monotonic() - waiting_since < report_grace:
                        if _thresholds_moved(look, last_current_report):
                            # The server judges by other thresholds since the last current report
                            # (lane AD1): that result is not carried over, and the parts are judged
                            # here with the thresholds in force (`adopted`), as after the wait.
                            pass
                        else:
                            in_grace = True
                            report_look = {**look, "report": last_current_report}
                    elif look["state"] == "behind":
                        look["state"] = "stale"
            grace_key = _grace_alert_key(status, status_age, other_stamp)
            if report_look is not None:
                report = report_look["report"]
                # The look's state is part of it: a file written inside the wait says the report
                # is not current, and is rewritten once the same report is current again.
                seen = (
                    "server",
                    report_look["state"],
                    report["attempt"],
                    report.get("last_step"),
                    report.get("steps"),
                    report.get("rebuilds"),
                    stamp,
                    other_stamp,
                )
                if seen == emitted or (in_grace and grace_key == emitted_grace):
                    continue
                evidence = _report_evidence(folder, report_look, status, **other_sources)
                run_id = str(report.get("run_id") or source.name)
                path = Path(output) if output else Path(f"{run_id}.evidence.json")
            else:
                grown = tail.poll()
                if picked_rank is None and default_rank is not None:
                    ranks_seen = {int(r.get("rank", 0)) for r in tail.records}
                    if len(ranks_seen) > 1:
                        picked_rank = default_rank
                        others_seen |= ranks_seen
                        tail = _Tail(source, picked_rank)
                        memo = _Incremental() if _INCREMENTAL_WATCH else None
                        follow = _follow_for(build_options, follow)
                        grown = tail.poll()
                        folder = _rank_folder(source, picked_rank) if server_reports else None
                        emitted = None
                        emitted_grace = None
                        waiting_since, was_current, last_current_report = None, False, None
                        adopted = {}
                # The end marker appears after the last record: its arrival alone is news.
                ended = read_end_marker(source) if source.suffix == ".jsonl" else None
                # The thresholds taken from the rank's report are part of it (wave AF-W, the author's
                # follow-up): a server that now applies others (its report says so) is news with no new
                # record, and the parts are judged again under them. Before, nothing was written until
                # the next record, and the file showed the run under the thresholds no longer in force.
                seen = ("python", look and look["state"], stamp, other_stamp, ended and ended["state"],
                        json.dumps(adopted, sort_keys=True, default=str))
                if not tail.records:
                    # The rank this watcher locked onto has no record yet while others do: said
                    # once for each set of ranks seen, since nothing is written for it until it
                    # has one (DET-12, Nov 19, 2026: it wrote nothing and said nothing).
                    present = sorted(others_seen - {picked_rank})
                    if rank is None and picked_rank is not None and present and present != told_absent:
                        told_absent = present
                        warnings.warn(
                            f"rank {picked_rank} has no records here yet, and ranks {_rank_ranges(present, 8)} do: no"
                            f" evidence is written until rank {picked_rank} has a step. Choose another with rank=N"
                            f" (--rank N), or every rank with rank='all' (--rank all)",
                            RuntimeWarning, stacklevel=2,
                        )
                    continue
                if not grown and seen == emitted:
                    continue
                here = {**adopted, **thresholds}
                evidence = _build_evidence(
                    tail.records, memo, follow=follow, identity_checked=True,  # the tail checked each as it read it
                    **{"run_manifest": source, "records_file": source, **build_options, **here,
                       **({"watchdog": status_file} if status_file else {})}
                )
                if look is not None:
                    evidence["detection"] = {**_detection(look, "python", picked_rank),
                                             "thresholds": {**THRESHOLD_DEFAULTS, **here}}
                if tail.left_out:
                    evidence["lines_left_out"] = tail.left_out
                path = Path(output) if output else default_evidence_path(source, tail.records)
            if status_age is not None and "watchdog" in evidence:
                evidence["watchdog"]["not_written_for_seconds"] = round(status_age, 1)
            judged = bool(report_look["report"].get("steps")) if report_look is not None else bool(tail.records)
            attempt = (_whole_attempt(report_look["report"].get("attempt")) if report_look is not None
                       else newest_attempt(tail.records))
            if report_look is not None and (block := _report_freshness(report_look, judged)) is not None:
                evidence["freshness"] = block  # absent when the report covers the newest stored step
            evidence["run_state"] = _per_rank_run_state(evidence, status, status_age, steps_judged=judged, attempt=attempt,
                                                        report_awaited=in_grace)
            emitted = seen
            emitted_grace = grace_key
            samples_text = None
            if report_look is None and compact_window is None:
                samples_text = ((memo.samples_text(evidence) if memo is not None else None)
                                or (follow.samples_text(evidence.get("samples")) if follow is not None else None))
            written = _write_output(evidence, path, output is not None, compact_window, samples_text)
            if on_update is not None:
                on_update(evidence, written)
            if trigger is not None:
                own = _own_rank(folder) if report_look is not None else tail.records[0].get("rank", 0)
                trigger.look(evidence, written, rank=own)
    except KeyboardInterrupt:
        pass
    finally:
        if follow is not None:
            follow.close()  # on every way out: Ctrl-C, a record refused, an error, the last look
    return written


# --- The all-ranks view ------------------------------------------------------------------
#
# `watch(run_folder, rank="all")`: one evidence file for every `rank=<n>` folder of a run in a
# server's store. It reads what a single-rank watcher reads, for every rank: the rank's
# `incidents.json` and the name of its newest part (`read_server_report`), and the run's
# `watchdog.json`. It reads no part and runs no detector in any state: a rank whose report is
# not current is listed as unknown, with its state and the reason, and is never counted among
# the ranks that have no incidents. (The one exception is shared with the single-rank watcher:
# when `flight_recorder` finds a stuck collective in complete dumps, the newest stored part is
# opened once to date the dump against the last recorded step.)
ALL_RANKS = "all"
ALL_RANKS_VIEW = "all_ranks"
ALL_RANKS_VIEW_VERSION = 1
# `world_size` names the ranks a job should have; a bogus huge value is not taken as a list to expand.
# The number is `identity.MAX_EXPECTED_RANKS`, the server's `watchdog::MAX_RANKS` (imported above).
# Lines of the text summary given to incidents on only some ranks before the rest are counted.
SUMMARY_PARTIAL_LINES = 8


def _discover_rank_folders(source: Path) -> dict[int, Path]:
    """The `rank=<n>` folders directly under a run folder, by rank number."""
    found: dict[int, Path] = {}
    try:
        with os.scandir(source) as entries:
            for entry in entries:
                number = rank_number(entry.name)
                if number is not None and entry.is_dir():
                    found[number] = Path(entry.path)
    except OSError:
        return {}
    return found


def _ranks_without_folder(folders: Mapping[int, Path], status: Mapping[str, Any] | None) -> dict[int, str]:
    """Ranks the server's status knows of or expects that have no folder in the store, with why.

    The status only covers ranks connected since the server started (and `world_size`, when a
    rank sent it), so after a restart it knows less than the store does: the folders are the
    ranks that have stored anything, and the status adds only those that have not.
    """
    reasons: dict[int, str] = {}
    if not status:
        return reasons
    world_size = status.get("world_size")
    if isinstance(world_size, int) and not isinstance(world_size, bool) and 0 < world_size <= MAX_EXPECTED_RANKS:
        for number in range(world_size):
            if number not in folders:
                reasons[number] = f"the job's world_size is {world_size} and nothing is stored for this rank"
    for item in status.get("ranks") or ():
        number = item.get("rank") if isinstance(item, dict) else None
        if isinstance(number, int) and number not in folders:
            reasons[number] = "the server saw this rank connect, and nothing is stored for it (no rank folder)"
    for number in status.get("missing_ranks") or ():
        if isinstance(number, int) and number not in folders:
            reasons[number] = "the server's status lists this rank as never connected, and nothing is stored for it"
    for number in _missing_counted(status):
        if number not in folders:
            reasons.setdefault(number, "the server's status counts this rank as never connected (beyond the ranks it "
                                       "lists), and nothing is stored for it")
    return reasons


class _RankMemory:
    """What the all-ranks watcher keeps about one rank between looks, as the single-rank
    watcher keeps it for its one rank: whether the rank's report was ever seen current, since
    when it has been behind, and its look the last time it was current."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.was_current = False
        self.waiting_since: float | None = None
        self.last_current_look: dict[str, Any] | None = None


def _rank_look(
    folder: Path | None,
    absent: str | None,
    memory: _RankMemory,
    thresholds: Mapping[str, Any],
    report_grace: float,
    now: float,
) -> tuple[dict[str, Any], bool]:
    """One rank's `read_server_report`, with the single-rank watcher's wait applied to it.

    A report that is `behind`, `missing` or `unreadable` after having been current gets
    `report_grace` seconds from that moment, since a part is stored first and reported on a
    moment later; `behind` for longer is `stale`. A rank never seen current does not wait.
    Returns the look and whether the rank is inside its wait.
    """
    if folder is None:
        memory.reset()
        return {"state": "nothing_stored", "reason": absent, "stored": None, "report": None}, False
    look = read_server_report(folder, **thresholds)
    state = look["state"]
    in_grace = False
    if state == "current":
        memory.was_current, memory.waiting_since = True, None
    elif state == "nothing_stored":
        memory.reset()
    elif state in ("missing", "behind", "unreadable") and memory.was_current:
        if memory.waiting_since is None:
            memory.waiting_since = now
        if now - memory.waiting_since < report_grace:
            in_grace = True
        elif state == "behind":
            look["state"] = "stale"
    return look, in_grace


def _rank_key(look: Mapping[str, Any]) -> tuple[Any, ...]:
    """What a look of one rank says that a reader of the file would want to know changed:
    the rank's state and which report it is looking at, not how far the store has grown
    beside a report that is not current."""
    report = look.get("report") or {}
    return (look["state"], report.get("attempt"), report.get("last_step"), report.get("steps"), report.get("rebuilds"))


def _rank_ranges(ranks: Iterable[int], limit: int | None = None) -> str:
    """Ranks as ranges: `0-63`, `0-3, 5, 7-9`; with `limit`, the first that many ranges and a count."""
    spans: list[list[int]] = []
    for number in sorted(set(ranks)):
        if spans and number == spans[-1][1] + 1:
            spans[-1][1] = number
        else:
            spans.append([number, number])
    shown = spans if limit is None else spans[:limit]
    text = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in shown)
    hidden = sum(b - a + 1 for a, b in spans[len(shown):])
    return text + (f", and {hidden} more" if hidden else "")


def _incident_identity(incident: Mapping[str, Any]) -> str:
    """What makes two ranks' incidents "the same": the kind, the step and what it names (the
    layer, router, experts, parameters or modules), never the measured numbers or the logged
    scalars, which differ from rank to rank for the same fault. A kind this function does not
    know is the same only when it is equal, but for its `scalars`."""
    kind, step = incident.get("kind"), incident.get("step")
    if kind == "moe_imbalance":
        parts: Any = [incident.get("layer"), incident.get("router"), sorted(incident.get("overloaded") or ()),
                      sorted(incident.get("dead") or ())]
        if incident.get("since_start"):  # when it became reportable is a step, not a measurement
            parts.append(incident.get("reportable_step"))
    elif kind == "starved_experts":
        parts = sorted(([e.get("layer"), e.get("router"), e.get("expert")] for e in incident.get("experts") or ()), key=repr)
    elif kind in ("grad_spike", "update_spike"):
        parts = sorted(str(p.get("name")) for p in incident.get("parameters") or ())
    elif kind == "stalled_weights":
        parts = sorted(str(name) for name in incident.get("parameters") or ())
    elif kind in ("activation_spike", "activation_drift", "dead_activations"):
        # In the model's order: the first block named is where it started.
        parts = [[m.get("module"), m.get("stat")] for m in incident.get("modules") or ()]
    elif kind in ATTN_KINDS:
        parts = [incident.get("confirmed_step"), [m.get("module") for m in incident.get("modules") or ()]]
    elif kind == "non_finite":
        parts = [list(incident.get("layers") or ()), list(incident.get("gradients") or ())]
    elif kind == "loss_scale_trouble":
        parts = [incident.get("reason")]
    elif kind == "replica_drift":
        parts = sorted(incident.get("parameters") or ())
        if incident.get("source") is not None:  # from the ranks' observations: what was compared, too
            parts = [parts, incident.get("source"), incident.get("launch"), incident.get("group")]
    elif kind in (
        "repeated_steps", "resume_lr_change", "resume_data_mismatch", "resume_incomplete",
        "memory_growth", "memory_pressure", "allocator_retries",
    ):
        parts = []
    elif kind == "data_stalled":
        parts = [incident.get("reason")]
    elif kind == "epoch_order_repeated":
        parts = [incident.get("epoch")]
    elif kind == "hardware_errors":
        parts = sorted(incident.get("xid") or ())
    elif kind == "gpu_throttled":
        parts = sorted(str(r) for r in incident.get("reasons") or ())
    elif kind == "input_starvation":
        parts = [incident.get("rule")]
    else:
        parts = {k: v for k, v in incident.items() if k != "scalars"}
    return json.dumps([kind, step, parts], sort_keys=True, default=str)


def _without_scalars(incident: Mapping[str, Any]) -> dict[str, Any]:
    """The logged values around an incident are each rank's own data and differ by construction."""
    return {k: v for k, v in incident.items() if k != "scalars"}


def _group_incidents(looks: Mapping[int, Mapping[str, Any]], judged: list[int]) -> list[dict[str, Any]]:
    """The incidents of the ranks whose reports are current, grouped across ranks.

    Incidents with the same identity are one group with the ranks that show it; those that
    differ in kind, step or what they name are separate groups, so an incident on one rank is
    listed with that rank alone. A group holds one incident, the first rank's, and says whether
    the other ranks' incidents differ from it in their numbers (not counting `scalars`: the
    incident's logged context is the first rank's).
    """
    groups: dict[str, dict[str, Any]] = {}
    for number in judged:
        counted: set[str] = set()
        # Stamped from this version's table first, so a report without `class` groups the same.
        with_classes(i for i in looks[number]["report"]["incidents"] if isinstance(i, dict) and "kind" in i)
        for incident in looks[number]["report"]["incidents"]:
            if not isinstance(incident, dict) or "kind" not in incident:
                continue
            identity = _incident_identity(incident)
            if identity in counted:
                continue
            counted.add(identity)
            group = groups.get(identity)
            if group is None:
                groups[identity] = {"kind": incident["kind"], "step": incident.get("step"), "ranks": [number],
                                    "incident_rank": number, "numbers_differ": False, "incident": incident}
            else:
                group["ranks"].append(number)
                if not group["numbers_differ"] and _without_scalars(incident) != _without_scalars(group["incident"]):
                    group["numbers_differ"] = True
    order = {kind: i for i, kind in enumerate(SERVER_KINDS)}
    found = sorted(
        groups.values(),
        key=lambda g: (order.get(g["kind"], len(order)), g["step"] if isinstance(g["step"], int) else -1, g["ranks"][0]),
    )
    out = []
    for index, group in enumerate(found):
        out.append({
            "id": index, "kind": group["kind"], **_class_key(group["kind"]), "step": group["step"],
            "ranks": group["ranks"], "rank_ranges": _rank_ranges(group["ranks"]),
            "on_all_judged_ranks": len(group["ranks"]) == len(judged),
            "numbers_differ": group["numbers_differ"], "incident_rank": group["incident_rank"],
            "incident": group["incident"],
        })
    return out


# --- What was checked, for the whole run --------------------------------------------------
#
# The all-ranks view carries `kind_checks` too: one entry for each of the 33 kinds, with the
# same three values and the same `enough_history` as one rank's evidence, so `checked_counts`,
# `kinds_by_value` and `kind_history` read it as they read a rank's. The rule (`run_kind_checks`):
#
#   1. Only judged ranks count (those whose report is current). A rank that is not judged is
#      named beside the count (`unknown_ranks`) and is in no value: it is never "checked".
#   2. For a kind read from one rank's step records (the server's 25), the worst judged rank
#      gives the run's value: `unavailable` if any judged rank is, else `not_configured` if
#      any is, else `checked`. So `checked` means checked on every judged rank, and "nothing
#      found" for it is said of every judged rank. When the judged ranks differ, the entry
#      lists them by value under `ranks`; a kind that is `not_configured` or `unavailable`
#      for the run can so be checked, and even reported, on the ranks named there.
#   3. History the same way: `enough_history` is false when any judged rank is short of it
#      (`not_enough_history_ranks`), is left out when none is short and some rank's report
#      does not say (`history_not_stated_ranks`; `kind_history` reads that as `not_stated`,
#      never as enough), and is true only when every judged rank has enough. `records` is
#      the fewest any judged rank has and `records_needed` the most any needs.
#   4. `job_stalled`, `rank_silent` and the three file kinds belong to the run, not to a
#      rank: they are `other_kind_checks` of the run's status (expected, since the source is
#      a store) and of the files given.
#   5. The two cross-rank kinds are judged for the run, by rule 2 over what each judged rank
#      brings to the comparison, and need two ranks: `memory_imbalance` is `checked` when
#      every judged rank's report has a peak memory reading and there are at least two;
#      `duplicate_batches`, which the server compares and puts in its status, when the
#      status's `ranks_compared` lists every judged rank and at least two ranks
#      (`unavailable` without a status). A judged rank it does not list is `not_configured`
#      for the comparison, unless its own report says its data could not be read, and is
#      named under `not_compared_ranks` when it recorded `data`: the server compares the
#      ranks whose stores it has read since it started, and only hashed batches. A status
#      from a server before `ranks_compared` (Nov 18, 2026) does not say which ranks it
#      compared, so then the ranks that recorded `data` count, with a `note` that says so.
#   6. When no rank is judged, or a judged rank's report has no usable `kind_checks` (a
#      server older than the key wrote it), there is no run-level object and no count:
#      `kind_checks_not_stated` says why and names the ranks. A count over the ranks that do
#      say would read as the run's.
#   7. `degraded` is as before and agrees with it: a judged rank's `not_checked` entry makes
#      a kind `unavailable` on that rank, so for the run, and already degrades the run state.
# What is said of a cross-rank kind when one rank was observed (STEP_AND_EVIDENCE_CONTRACT.md,
# section 2, and REPLICA_OBSERVATION_CONTRACT.md, section 4a): it was compared with nothing, and is
# never said to agree.
NO_PEER_WORDS = "one rank observed, zero peers compared"
_CROSS_RANK_NEEDS = {
    "duplicate_batches": "data (session.watch_loader) on every judged rank, at least two ranks, and the server's"
                         " progress status (watchdog.json)",
    "memory_imbalance": "system.cuda (system=True) on every judged rank, and at least two ranks",
}
_DATA_KIND = "data_stalled"  # a server kind judged from `data`: its `records` count the steps that carried it
_MEMORY_KIND = "memory_pressure"  # a server kind judged from `system.cuda`
# Kinds named in one part of the count line before the rest are counted.
CHECKED_LINE_KINDS = 6


def stated_kind_checks(report: Any) -> Mapping[str, Any] | None:
    """A server report's `kind_checks` when it gives every one of the server's kinds one of
    the three values, else None (a server older than the key wrote the report). The test
    `_report_evidence` applies before it takes them. A kind among the report's incidents is
    never `not_configured` (`checked_where_reported`)."""
    checks = report.get("kind_checks") if isinstance(report, Mapping) else None
    if isinstance(checks, Mapping) and all(
        isinstance(checks.get(kind), Mapping) and checks[kind].get("value") in KIND_CHECK_VALUES for kind in SERVER_KINDS
    ):
        incidents = report.get("incidents")
        return checked_where_reported(checks, incidents if isinstance(incidents, (list, tuple)) else ())
    return None


def _worst_value(by_rank: Mapping[int, Mapping[str, Any]]) -> str:
    """The worst judged rank's value: `KIND_CHECK_VALUES` is in that order, best first."""
    return max((check["value"] for check in by_rank.values()), key=KIND_CHECK_VALUES.index)


def _ranks_by_value(by_rank: Mapping[int, Mapping[str, Any]]) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for value in KIND_CHECK_VALUES:
        ranks = sorted(rank for rank, check in by_rank.items() if check["value"] == value)
        if ranks:
            found[value] = ranks
    return found


def _run_check(by_rank: Mapping[int, Mapping[str, Any]], needs: str) -> dict[str, Any]:
    """One kind for the run from its entry on each judged rank (rules 2 and 3 above)."""
    value = _worst_value(by_rank)
    ranks = _ranks_by_value(by_rank)
    stated = {rank: check.get("enough_history") for rank, check in by_rank.items()}
    short = sorted(rank for rank, enough in stated.items() if enough is False)
    silent = sorted(rank for rank, enough in stated.items() if not isinstance(enough, bool))
    # Only what a checked rank says of its history is listed: the others have no rule running.
    warming = [rank for rank in short if by_rank[rank]["value"] == "checked"]
    not_stated = [rank for rank in silent if by_rank[rank]["value"] == "checked"]
    seen = [check["records"] for check in by_rank.values() if isinstance(check.get("records"), int)]
    needed = [check["records_needed"] for check in by_rank.values() if isinstance(check.get("records_needed"), int)]
    first_unavailable = next((by_rank[rank] for rank in ranks.get("unavailable", ())), None)
    return {
        "value": value,
        **({"enough_history": not short} if short or not silent else {}),
        **({"records": min(seen)} if len(seen) == len(by_rank) else {}),
        **({"records_needed": max(needed)} if len(needed) == len(by_rank) else {}),
        "needs": needs,
        **({"note": noted} if (noted := next((c["note"] for c in by_rank.values() if isinstance(c.get("note"), str)), None))
           and value == "checked" else {}),
        **({key: first_unavailable.get(key) for key in ("check", "reason")} if first_unavailable is not None else {}),
        **({"ranks": ranks} if len(ranks) > 1 else {}),
        **({"not_enough_history_ranks": warming} if warming else {}),
        **({"history_not_stated_ranks": not_stated} if not_stated else {}),
    }


def _run_parameters_not_judged(check: dict[str, Any], by_rank: Mapping[int, Mapping[str, Any]]) -> None:
    """The run's `stalled_weights` entry names the parameters some rank did not judge (rules
    contract, part 5): the sorted union of the ranks' `parameters_not_judged`, and
    `parameters_not_judged_by_rank` (rank, as a string, to its list) when the judged ranks' lists
    differ. A judged rank without the key judged every parameter it read: its list is empty, and
    it is in the map only through that difference, by its absence."""
    _run_names_not_judged(check, by_rank, "parameters_not_judged")


def _run_names_not_judged(check: dict[str, Any], by_rank: Mapping[int, Mapping[str, Any]], key: str) -> None:
    """The run's entry names what some rank's entry names under `key` (`parameters_not_judged`, or
    an activation kind's `modules_not_judged`, EV-03): the sorted union, and `<key>_by_rank` (rank,
    as a string, to its list) when the judged ranks' lists differ, a rank without the key counting
    as an empty list."""
    lists = {rank: sorted({str(name) for name in found.get(key) or ()})
             for rank, found in by_rank.items() if isinstance(found, Mapping)}
    union = sorted({name for names in lists.values() for name in names})
    if not union:
        return
    check[key] = union
    if len({tuple(names) for names in lists.values()}) > 1:
        check[f"{key}_by_rank"] = {str(rank): names for rank, names in sorted(lists.items()) if names}


def _run_starting_state(check: dict[str, Any], by_rank: Mapping[int, Mapping[str, Any]]) -> None:
    """The run's `moe_imbalance` entry names the routers some rank's entry lists as still in their
    starting state (EV-03; rules contract, part 3): `starting_state`, one `{"router", "samples"}`
    for each router named, with the fewest samples any rank gives it, in the order the ranks name
    them, and `starting_state_by_rank` (rank, as a string, to its list) when the judged ranks'
    lists differ. `enough_history` is then false for the run, as it is for the rank. Not on an
    entry that is `unavailable`, as a rank's is not."""
    lists = {rank: [dict(item) for item in found.get("starting_state") or () if isinstance(item, Mapping)]
             for rank, found in by_rank.items() if isinstance(found, Mapping)}
    if not any(lists.values()) or check.get("value") == "unavailable":
        return
    merged: dict[str, dict[str, Any]] = {}
    for rank in sorted(lists):
        for item in lists[rank]:
            name = str(item.get("router"))
            held = merged.get(name)
            fewer = held is not None and _count(item.get("samples")) and (
                not _count(held.get("samples")) or item["samples"] < held["samples"])
            if held is None or fewer:
                merged[name] = {"router": item.get("router"), "samples": item.get("samples")}
    check["starting_state"] = list(merged.values())
    check["enough_history"] = False
    if len({json.dumps(items, sort_keys=True, default=str) for items in lists.values()}) > 1:
        check["starting_state_by_rank"] = {str(rank): items for rank, items in sorted(lists.items()) if items}


def _cross_rank_check(kind: str, brings: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """A kind that compares ranks, from what each judged rank brings to the comparison
    (rule 5): the worst rank gives the value, and one rank alone compares with nothing."""
    check = {key: value for key, value in _run_check(brings, _CROSS_RANK_NEEDS[kind]).items()
             if key not in ("records", "records_needed", "not_enough_history_ranks", "history_not_stated_ranks")}
    if check["value"] == "checked" and len(brings) < 2:
        check["value"] = "not_configured"
    check["enough_history"] = check["value"] == "checked"
    return {"value": check["value"], "enough_history": check["enough_history"],
            **{key: value for key, value in check.items() if key not in ("value", "enough_history")}}


def run_kind_checks(
    rank_checks: Mapping[int, Mapping[str, Any]],
    others: Mapping[str, Mapping[str, Any]],
    *,
    memory: Mapping[int, Any],
    status: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """`kind_checks` for a whole run (the rule is in the comment above).

    `rank_checks` is each judged rank's `kind_checks` from its server's report
    (`stated_kind_checks`; at least one rank), `others` the run's `other_kind_checks`,
    `memory` each judged rank's latest memory reading (`memory` in its report) and `status`
    the server's status or None.
    """
    checks: dict[str, dict[str, Any]] = {
        kind: _run_check({rank: found[kind] for rank, found in rank_checks.items()}, INPUT_NEEDS[KIND_INPUTS[kind]])
        for kind in SERVER_KINDS
    }
    _run_parameters_not_judged(checks[GRADS_NOT_READ_NAMED], {rank: found[GRADS_NOT_READ_NAMED] for rank, found in rank_checks.items()})
    # What `_run_check` leaves out of a rank's entry and the run's must say, with its ranks (EV-03).
    for kind in LAYERS_NOT_READ_KINDS:
        _run_names_not_judged(checks[kind], {rank: found[kind] for rank, found in rank_checks.items()}, "modules_not_judged")
    _run_starting_state(checks["moe_imbalance"], {rank: found["moe_imbalance"] for rank, found in rank_checks.items()})
    checks.update({kind: dict(check) for kind, check in others.items()})

    def brings(has: Any, kind: str) -> dict[int, dict[str, Any]]:
        # A rank without the input is not configured for the comparison, unless its own
        # report says the input could not be read.
        found: dict[int, dict[str, Any]] = {}
        for rank, own in rank_checks.items():
            if has(rank):
                found[rank] = {"value": "checked", "enough_history": True}
            elif own[kind].get("value") == "unavailable":
                found[rank] = {"value": "unavailable", "enough_history": False,
                               "check": own[kind].get("check"), "reason": own[kind].get("reason")}
            else:
                found[rank] = {"value": "not_configured", "enough_history": False}
        return found

    def peak(rank: int) -> bool:
        reading = memory.get(rank)
        return isinstance(reading, Mapping) and _finite_number(reading.get("peak_allocated")) is not None

    checks["memory_imbalance"] = _cross_rank_check("memory_imbalance", brings(peak, _MEMORY_KIND))
    recorded = [rank for rank in rank_checks if rank_checks[rank][_DATA_KIND].get("records")]
    compared = _ranks_compared(status)
    if compared is None:
        duplicates = _cross_rank_check("duplicate_batches", brings(lambda rank: rank in recorded, _DATA_KIND))
        if duplicates["value"] == "checked":
            if status is None:
                duplicates = {**_unavailable({"check": "status", "reason": _NO_STATUS}, enough_history=False,
                                             needs=duplicates["needs"])}
            else:
                # A server before `ranks_compared` names the ranks it compared only in a finding, so this
                # is what the ranks recorded, not proof that the server held every rank's batches.
                duplicates["note"] = ("the server compares the ranks whose stores it has read since it started, and"
                                      " its status names them only when it finds a duplicate")
    else:
        # What the server compared: one rank alone compared with nothing.
        listed = set(compared) if len(set(compared)) >= 2 else set()
        duplicates = _cross_rank_check("duplicate_batches", brings(lambda rank: rank in listed, _DATA_KIND))
        missed = sorted(rank for rank in recorded if rank not in compared)
        if len(recorded) == 1 and not listed:
            # One rank observed: it has no rank to be compared with (contract, section 2).
            duplicates["value"], duplicates["enough_history"] = "not_configured", False
            duplicates["note"] = NO_PEER_WORDS
        elif missed:
            duplicates["not_compared_ranks"] = missed
            many = len(missed) != 1
            duplicates["note"] = (
                f"rank{'s' if many else ''} {_rank_ranges(missed, 8)} recorded data and {'are' if many else 'is'} not"
                " in the server's `ranks_compared`: it compares the ranks whose stores it has read since it started"
                " (none right after a restart), and only batches with a batch hash")
    checks["duplicate_batches"] = duplicates
    return {kind: checks[kind] for kind in INCIDENT_CLASSES}


def _ranks_compared(status: Mapping[str, Any] | None) -> list[int] | None:
    """The ranks whose batch hashes the server compared, from its status (`ranks_compared`,
    written whether or not a duplicate is found); None when the status does not say: none was
    read, or a server before the key wrote it."""
    found = status.get("ranks_compared") if isinstance(status, Mapping) else None
    if isinstance(found, list) and all(isinstance(rank, int) and not isinstance(rank, bool) for rank in found):
        return found
    return None


def _all_ranks_kind_checks(
    evidence: dict[str, Any],
    looks: Mapping[int, Mapping[str, Any]],
    status: Mapping[str, Any] | None,
    *,
    flight_recorder: bool,
    traincheck: bool,
) -> None:
    """Put what was checked into the all-ranks evidence: `kind_checks` for the run, or
    `kind_checks_not_stated` with why there is none, and on each judged rank's entry
    `kind_counts`, the count of its own report's kinds (the server's 25)."""
    judged = evidence["judged_ranks"]
    stated = {n: found for n in judged if (found := stated_kind_checks(looks[n]["report"])) is not None}
    for entry in evidence["ranks"]:
        if entry["rank"] in stated:
            entry["kind_counts"] = {**(checked_counts({"kind_checks": stated[entry["rank"]]}) or {}), "kinds": len(SERVER_KINDS)}
    silent = [n for n in judged if n not in stated]
    if not judged:
        evidence["kind_checks_not_stated"] = {
            "code": "no_judged_rank", "ranks": [],
            "detail": "no rank is judged, so nothing is said about what was checked",
        }
        return
    if silent:
        evidence["kind_checks_not_stated"] = {
            "code": "report_without_kind_checks", "ranks": silent,
            "detail": f"the report of rank{'s' if len(silent) != 1 else ''} {_rank_ranges(silent, 8)} does not say what was"
                      " checked (`kind_checks`: an older tensorquill-server wrote it), so the kinds are not counted for"
                      " the run",
        }
        return
    others = other_kind_checks(
        status=status, status_expected=True, not_checked=evidence.get("not_checked", ()),
        flight_recorder=flight_recorder, traincheck=traincheck,
    )
    evidence["kind_checks"] = run_kind_checks(
        stated, others, memory={n: looks[n]["report"].get("memory") for n in judged}, status=status)


_STATUS_INCIDENT_KINDS = ("job_stalled", "rank_silent", "duplicate_batches")  # from the server's status


def combine_rank_evidence(
    per_rank: Mapping[int, Mapping[str, Any]],
    *,
    status: Mapping[str, Any] | None = None,
    status_expected: bool = False,
    unknown_ranks: Mapping[int, str] | None = None,
    memory: Mapping[int, Any] | None = None,
    replica_records: Iterable[Mapping[str, Any]] | None = None,
    drift_tolerance: float = DRIFT_TOLERANCE,
    attempt_order: Mapping[tuple[str, Any], list[Any]] | None = None,
) -> dict[str, Any]:
    """The run-level part of the all-ranks view, built from one evidence per judged rank.

    The ranks' replica observations (SDK-22) are compared across ranks here. For local files,
    `replica_records` are every rank's records as written (all of them, not one rank's resolved
    ones), compared by `compare_replicas` with `drift_tolerance` and `attempt_order` (each rank's
    attempts oldest first, for a rank in several files). For a store, the server's result is
    taken from `status` (`replica` in `watchdog.json`), and `replica_not_stated` says when the
    status has none. Either way the result is under `replica`, its first incident from
    observations is a run incident, and `kind_checks`' `replica_drift` entry says what was
    compared (`replica_kind_check`). Records without an observation give no `replica` key.

    For the ranks `all_ranks_evidence` cannot read as a whole: a run of local `.jsonl` files
    (each rank judged by `build_evidence`, or read from its evidence file), and a store whose
    ranks are not all current (some judged from their reports, some from their parts). It
    applies the view's rules and no other: incidents grouped across ranks by
    `_incident_identity` (`incident_groups`), the run's own incidents apart (`run_incidents`:
    the server's status, read once, and `memory_imbalance` across ranks), what was checked by
    `run_kind_checks` (`kind_checks`, or `kind_checks_not_stated`), and `run_state` over all of
    it. `unknown_ranks` are the ranks with no evidence, with why: they are in no value.

    `status` is the server's status for a store (`status_expected=True`); a local run has none,
    and then `duplicate_batches`, which only the server compares, is `not_configured`.
    `memory` is each rank's latest memory reading where its evidence does not carry it (a
    server report's `memory`). An evidence may carry `last_step`, which the caller knows from
    the records or the report; the run's is `{"min", "max"}` over the judged ranks.

    A rank that the status shows to be judged by an earlier attempt (`_not_current_by_status`:
    marked `attempt_without_steps`, or shown at another attempt than its evidence's report) is
    moved to the unknown ranks, as `earlier_attempt` with why, and is in no value (L6).
    """
    unknown = dict(unknown_ranks or {})
    if status is not None:
        earlier = _not_current_by_status(status, {
            n: _whole_attempt((per_rank[n].get("detection") or {}).get("report_attempt")) for n in per_rank})
        for number, why in earlier.items():
            unknown[number] = f"{EARLIER_ATTEMPT}: {why}"
        per_rank = {n: found for n, found in per_rank.items() if n not in earlier}
    judged = sorted(per_rank)
    stream = ((list(status["incidents"]) if isinstance(status.get("incidents"), list) else detect_watchdog_incidents(status))
              if status is not None else [])
    looks = {n: {"report": {"incidents": [dict(i) for i in per_rank[n].get("incidents") or ()
                                          if isinstance(i, Mapping) and not (status is not None and i.get("kind") in _STATUS_INCIDENT_KINDS)]}}
             for n in judged}
    groups = _group_incidents(looks, judged)
    readings = {n: (memory or {}).get(n, per_rank[n].get("memory")) for n in judged}
    run_incidents = [dict(i) for i in stream]
    imbalance = detect_memory_imbalance(readings)
    if imbalance is not None:
        run_incidents.append(imbalance)
    with_classes(run_incidents)
    not_checked = [dict(item) for n in judged for item in per_rank[n].get("not_checked") or () if isinstance(item, Mapping)]
    steps = [s for n in judged if isinstance(s := per_rank[n].get("last_step"), int) and not isinstance(s, bool)]
    # What the judged ranks were judged with (lane AD1): some by their server's report, some by the
    # detectors here, each with its own six. As the all-ranks view lists them, and only when one of
    # them is not the default or the ranks differ, as `build_evidence` lists its own: a run judged
    # at the defaults throughout is what it was.
    by_thresholds = _thresholds_by_rank({n: _judged_thresholds(per_rank[n]) for n in judged})
    differing = len(by_thresholds) > 1
    combined: dict[str, Any] = {
        "rank_count": len(set(judged) | set(unknown)),
        "judged_ranks": judged,
        "unknown_ranks": sorted(unknown),
        "ranks_with_incidents": sorted({n for group in groups for n in group["ranks"]}),
        "last_step": {"min": min(steps), "max": max(steps)} if steps else None,
        **({"thresholds": by_thresholds[0]["thresholds"]}
           if by_thresholds and (differing or _off_default(by_thresholds[0]["thresholds"])) else {}),
        **({"thresholds_by_rank": by_thresholds} if differing else {}),
        "incident_groups": groups,
        "run_incidents": run_incidents,
        **({"stopped_stacks": stacks} if (stacks := group_stopped_stacks(status)) else {}),
        # Each judged rank's probe head numbers, from its evidence (L20).
        **({"head_trends_by_rank": heads} if (heads := _head_trends_by_rank(
            {n: per_rank[n].get("head_trends") for n in judged})) else {}),
        **({"skipped_updates_by_rank": skips} if (skips := _skipped_updates_by_rank(
            {n: per_rank[n].get("skipped_updates") for n in judged})) else {}),
        **({"checkpoint_tensors_by_rank": blocks} if (blocks := _checkpoint_tensors_by_rank(
            {n: per_rank[n].get("checkpoint_tensors") for n in judged})) else {}),
        # How each rank's observations were made and what it measured, with its rank (EV-04), and
        # what each rank did not check (EV-02): only where a rank has them.
        **_rank_blocks({n: per_rank[n] for n in judged}),
        **({"not_checked_by_rank": unchecked} if (unchecked := _not_checked_by_rank(
            {n: per_rank[n].get("not_checked") for n in judged})) else {}),
        # The rows the server dropped and the batches it refused, as the all-ranks view's `watchdog`
        # block gives them (L22, lane AD1): only when the status holds such counts, so the
        # evidence of a run that dropped and refused none is what it was.
        **({"watchdog": {**{k: status[k] for k in (
            "summary", "written_at_ms", "world_size", "pauses", "recommended_silent_seconds",
            "silent_floor_seconds", "critical_seconds",
        ) if k in status}, **said}} if (said := _dropped_rows_said(status)) else {}),
    }
    stated = {n: found for n in judged if (found := stated_kind_checks(per_rank[n])) is not None}
    silent = [n for n in judged if n not in stated]
    if not judged:
        combined["kind_checks_not_stated"] = {"code": "no_judged_rank", "ranks": [],
                                              "detail": "no rank is judged, so nothing is said about what was checked"}
    elif silent:
        combined["kind_checks_not_stated"] = {
            "code": "report_without_kind_checks", "ranks": silent,
            "detail": f"the evidence of rank{'s' if len(silent) != 1 else ''} {_rank_ranges(silent, 8)} does not say what"
                      " was checked (`kind_checks`: an older version wrote it), so the kinds are not counted for the run",
        }
    else:
        others = other_kind_checks(status=status, status_expected=status_expected, not_checked=not_checked,
                                   flight_recorder=False, traincheck=False)
        checks = run_kind_checks(stated, others, memory=readings, status=status)
        if status is None and not status_expected:
            # No server compared the ranks' batches: not looked for, which is not "unavailable".
            checks["duplicate_batches"] = {"value": "not_configured", "enough_history": False,
                                           "needs": _CROSS_RANK_NEEDS["duplicate_batches"]}
        combined["kind_checks"] = checks
    if replica_records is not None:
        listed = [dict(r) for r in replica_records]
        if any(_observation(r) is not None for r in listed):
            result, details = _replica_judged(listed, drift_tolerance, attempt_order)
            _fold_replica(combined, _replica_object(result, drift_tolerance), None, run_incidents, not_checked, details)
    else:
        _fold_replica(combined, *_replica_from_status(status), run_incidents, not_checked)
    with_classes(run_incidents)
    incidents = [group["incident"] for group in groups] + run_incidents
    combined["run_state"] = run_state(
        incidents=incidents, steps_judged=bool(steps),
        status=status, status_age=_status_age(status), not_checked=not_checked, unknown_ranks=unknown,
        judged_ranks=judged, status_expected=status_expected,
        # The attempt a rank's evidence was judged from, when it says (a server report's).
        judged_attempts={n: _whole_attempt((per_rank[n].get("detection") or {}).get("report_attempt")) for n in judged},
    )
    return combined


def _head_trends_by_rank(found: Mapping[int, Any]) -> list[dict[str, Any]]:
    """The probe's head numbers of each judged rank (`head_logit_trends`, with
    `overflow_positions`), for the all-ranks view's `head_trends_by_rank` (L20): one entry per
    rank whose evidence or report has them, `{"rank", "head_trends"}`, by rank. Each rank's are
    its own: a run's ranks probe their own batches, so they are not added up or compared. Not
    under `head_trends`, which one rank's evidence has and a reader of it takes as one rank's."""
    return [{"rank": number, "head_trends": dict(trends)} for number, trends in sorted(found.items())
            if isinstance(trends, Mapping) and trends]


def _skipped_updates_by_rank(found: Mapping[int, Any]) -> list[dict[str, Any]]:
    """Each judged rank's `skipped_updates` (STEP_AND_EVIDENCE_CONTRACT.md, section 1), for the
    all-ranks view's `skipped_updates_by_rank`: `{"rank", "skipped_updates"}` for each rank whose
    evidence or report has the key, by rank, unchanged. Not added up: each rank's skips are its
    own, and a rank without the key has a capture written before it."""
    return [{"rank": number, "skipped_updates": dict(skips)} for number, skips in sorted(found.items())
            if isinstance(skips, Mapping)]


#: The blocks of one rank's evidence or report that say how its observations were made, or what
#: they measured, carried into the all-ranks view as `<block>_by_rank` (WAVE_AF_CONTRACT.md,
#: section 2, EV-04): each rank's are its own, so they are not added up or merged.
RANK_BLOCKS = ("moe_source", *MEASUREMENT_BLOCKS)


def _rank_blocks(found: Mapping[int, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """`{"<block>_by_rank": [{"rank", "<block>"}, ...]}` for each of `RANK_BLOCKS` that some judged
    rank's evidence or report in `found` has, by rank, each block copied unchanged, as
    `skipped_updates_by_rank` carries its key (EV-04). No key for a block no rank has, so a view
    whose ranks have none is what it was."""
    blocks: dict[str, list[dict[str, Any]]] = {}
    for key in RANK_BLOCKS:
        entries = [{"rank": number, key: copy.deepcopy(dict(block))} for number, own in sorted(found.items())
                   if isinstance(own, Mapping) and isinstance(block := own.get(key), Mapping) and block]
        if entries:
            blocks[f"{key}_by_rank"] = entries
    return blocks


def _not_checked_by_rank(found: Mapping[int, Any]) -> list[dict[str, Any]]:
    """Each judged rank's own `not_checked` entries, for the all-ranks view built from one evidence
    per rank (`combine_rank_evidence`), which has no `ranks` list to put them on (EV-02):
    `{"rank", "not_checked"}` for each rank that lists any, by rank, its entries unchanged. The
    view a store's reports give keeps them on each rank's entry of `ranks`, as before."""
    return [{"rank": number, "not_checked": [dict(item) for item in items if isinstance(item, Mapping)]}
            for number, items in sorted(found.items())
            if isinstance(items, (list, tuple)) and any(isinstance(item, Mapping) for item in items)]


def _checkpoint_tensors_by_rank(found: Mapping[int, Any]) -> list[dict[str, Any]]:
    """Each judged rank's `checkpoint_tensors` (CHECKPOINT_TENSOR_CONTRACT.md, section 4), for the
    all-ranks view's `checkpoint_tensors_by_rank`, as `skipped_updates_by_rank` carries its key:
    `{"rank", "checkpoint_tensors"}` for each rank whose evidence or report has the block, by rank,
    unchanged. Each rank loaded its own checkpoint, so the blocks are not added up."""
    return [{"rank": number, "checkpoint_tensors": dict(block)} for number, block in sorted(found.items())
            if isinstance(block, Mapping)]


def checkpoint_tensors_block(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The evidence's `checkpoint_tensors` (CHECKPOINT_TENSOR_CONTRACT.md, section 4): how many
    judged resumes' loads recorded `tensors` that is `compared` or `unavailable`, and the first
    five by step, as `resume_incomplete` lists its changes. None when there is none, so a capture
    written before the key gives the evidence it gave."""
    if not entries:
        return None
    return {"resumes": len(entries), "entries": sorted(entries, key=lambda e: e["step"])[:CHANGES_LISTED]}


def _all_ranks_reported(evidence: Mapping[str, Any]) -> dict[Any, set[int]]:
    """The ranks that show an incident of each kind, from the incident groups."""
    found: dict[Any, set[int]] = {}
    for group in evidence.get("incident_groups") or ():
        if isinstance(group, Mapping):
            found.setdefault(group.get("kind"), set()).update(group.get("ranks") or ())
    return found


def _some(parts: list[str]) -> str:
    """The first `CHECKED_LINE_KINDS` of them, and a count of the rest."""
    rest = len(parts) - CHECKED_LINE_KINDS
    return ", ".join(parts[:CHECKED_LINE_KINDS]) + (f", and {rest} more" if rest > 0 else "")


def _on_ranks(ranks: Iterable[int]) -> str:
    ranks = list(ranks)
    return f"on rank{'s' if len(ranks) != 1 else ''} {_rank_ranges(ranks, 4)}"


def all_ranks_checked_line(evidence: Mapping[str, Any]) -> str | None:
    """The all-ranks summary's count line, or the line that says why there is no count; None
    for evidence that says neither (an older version wrote it).

    `7 of 33 kinds checked on both ranks, 1 of them without enough history yet (grad_spike 5
    of 13 records on rank 1), 25 not configured, 1 of them checked on some ranks only
    (stalled_weights on rank 0)`. A kind is counted as checked only when every judged rank
    checked it; ranks are named wherever the judged ranks differ, and the ranks that are not
    judged are named as not counted. A kind checked on some ranks only says so when those
    ranks are still short of history."""
    counts = checked_counts(evidence)
    if counts is None:
        missing = evidence.get("kind_checks_not_stated")
        if isinstance(missing, Mapping) and missing.get("code") != "no_judged_rank":  # then the first line says it
            return f"What was checked is not stated: {missing.get('detail')}."
        return None
    checks, reported = evidence["kind_checks"], _all_ranks_reported(evidence)
    judged, unknown = list(evidence.get("judged_ranks") or ()), list(evidence.get("unknown_ranks") or ())
    if unknown:
        where = (f"on the {len(judged)} judged rank{'s' if len(judged) != 1 else ''} of {len(judged) + len(unknown)}"
                 f" (rank{'s' if len(unknown) != 1 else ''} {_rank_ranges(unknown, 4)} not judged, so not counted)")
    else:
        where = "on the only rank" if len(judged) == 1 else "on both ranks" if len(judged) == 2 else f"on all {len(judged)} ranks"
    line = f"{counts['checked']} of {counts['kinds']} kinds checked {where}"

    def differing(kind: str, value: str) -> str:
        ranks = (checks[kind].get("ranks") or {}).get(value)
        return f"{kind} {_on_ranks(ranks)}" if ranks else kind

    if counts["not_enough_history"]:
        each = []
        for at, kind in enumerate(counts["not_enough_history_kinds"]):
            seen, needed = checks[kind].get("records"), checks[kind].get("records_needed")
            said = f"{kind} {seen} of {needed}" if seen is not None and needed is not None else kind
            if at == 0 and " of " in said:
                said += " records"
            short = checks[kind].get("not_enough_history_ranks") or ()
            if short and len(short) < len(judged):
                said += f" {_on_ranks(short)}"
            # Only an incident on a rank that is short of history is a faster rule's.
            each.append(said + (" but reported by a faster rule" if reported.get(kind, set()) & set(short) else ""))
        line += f", {counts['not_enough_history']} of them without enough history yet ({_some(each)})"
    if counts["history_not_stated"]:
        line += (f", {counts['history_not_stated']} of them with no word on their history on some rank (an older"
                 " tensorquill-server wrote its report)")
    line += f", {counts['not_configured']} not configured"
    partly = [kind for kind in INCIDENT_CLASSES if isinstance(checks.get(kind), Mapping)
              and checks[kind].get("value") == "not_configured" and (checks[kind].get("ranks") or {}).get("checked")]
    if partly:
        each = []
        for kind in partly:
            # "Checked" there is not yet a verdict there either, when the rank is still warming up.
            there = checks[kind]["ranks"]["checked"]
            short = checks[kind].get("not_enough_history_ranks") or ()
            silent = checks[kind].get("history_not_stated_ranks") or ()
            said = differing(kind, "checked")
            if short:
                said += " without enough history yet" + ("" if len(short) == len(there) else f" {_on_ranks(short)}")
            if silent:
                said += " with no word on its history" + ("" if len(silent) == len(there) else f" {_on_ranks(silent)}")
            each.append(said + (", where it is reported" if reported.get(kind) else ""))
        line += f", {len(partly)} of them checked on some ranks only ({_some(each)})"
    if counts["unavailable"]:
        line += f", {counts['unavailable']} unavailable ({_some([differing(k, 'unavailable') for k in counts['unavailable_kinds']])})"
    return line


def all_ranks_kinds(evidence: Mapping[str, Any]) -> dict[str, Any] | None:
    """The all-ranks evidence's `kind_checks` by value, as `kinds_by_value` gives one rank's,
    with the ranks named as ranges wherever the judged ranks differ. None for evidence
    without the key (see `kind_checks_not_stated` there).

    - `checked`: checked on every judged rank with enough history. "Nothing found" is a
      verdict for these only, and only for the judged ranks.
    - `not_enough_history`: checked on every judged rank, still warming up on those under
      `ranks` (all of them when there is no `ranks`).
    - `not_configured` and `unavailable` are the worst judged rank's value, so a kind listed
      there may be checked on other ranks: `differs_by_rank` gives each such kind's ranks by
      value, and under `not_enough_history` (or `history_not_stated`) those of its checked
      ranks whose rule is still warming up. An incident of such a kind stands for the ranks
      that show it.
    - `judged_ranks` and `ranks_not_counted`: who the values are about, and who they are not.
    """
    found = kinds_by_value(evidence)
    if found is None:
        return None
    checks, reported = evidence["kind_checks"], _all_ranks_reported(evidence)
    judged = list(evidence.get("judged_ranks") or ())
    for kind, entry in found["not_enough_history"].items():
        short = checks[kind].get("not_enough_history_ranks") or ()
        if short and len(short) < len(judged):
            entry["ranks"] = _rank_ranges(short)
        if reported.get(kind, set()) & set(short):
            entry["reported"] = True
    differs = {
        kind: {**{value: _rank_ranges(ranks) for value, ranks in check["ranks"].items()},
               # Of the ranks that checked it: those whose rule could not have fired yet.
               **({"not_enough_history": _rank_ranges(check["not_enough_history_ranks"])}
                  if check.get("not_enough_history_ranks") else {}),
               **({"history_not_stated": _rank_ranges(check["history_not_stated_ranks"])}
                  if check.get("history_not_stated_ranks") else {})}
        for kind, check in checks.items() if isinstance(check, Mapping) and isinstance(check.get("ranks"), Mapping)
    }
    return {
        **found,
        **({"differs_by_rank": differs} if differs else {}),
        "judged_ranks": _rank_ranges(judged),
        "ranks_not_counted": _rank_ranges(evidence.get("unknown_ranks") or ()),
    }


def _all_ranks_evidence(
    source: Path,
    folders: Mapping[int, Path],
    looks: Mapping[int, Mapping[str, Any]],
    status: Mapping[str, Any] | None,
    *,
    traincheck: str | Path | None,
    flight_recorder: str | Path | None,
    flight_recorder_prefix: str | None,
) -> dict[str, Any]:
    """The evidence for the whole run (see `watch_all_ranks`)."""
    judged = [n for n, look in looks.items() if look["state"] == "current"]
    groups = _group_incidents(looks, judged)
    by_rank: dict[int, list[dict[str, Any]]] = {n: [] for n in judged}
    for group in groups:
        for number in group["ranks"]:
            by_rank[number].append({"kind": group["kind"], **_class_key(group["kind"]), "step": group["step"],
                                    "group": group["id"]})
    streams: dict[int, dict[str, Any]] = {}
    for item in (status or {}).get("ranks") or ():
        if isinstance(item, dict) and isinstance(item.get("rank"), int):
            streams[item["rank"]] = {k: item[k] for k in (
                "state", "last_step", "phase", "severity", "seconds_since_progress", "heartbeat_age_seconds",
                "stopped_as",  # a rank that was stopped when its stream dropped (wave AH): what it was
            ) if k in item}
    entries = []
    for number in sorted(looks):
        look = looks[number]
        report = look.get("report") or {}
        detection = {k: v for k, v in _detection(look, "server", number).items() if k not in ("source", "rank")}
        detection["judged"] = number in by_rank
        entry: dict[str, Any] = {"rank": number, "detection": detection}
        if number in by_rank:
            if report.get("moe_flagged") is not None:
                detection["moe_flagged"] = report["moe_flagged"]
            if assumed := thresholds_assumed(report):
                # Not listed by the rank's server: given in `thresholds` as this version's defaults.
                detection["thresholds_assumed"] = assumed
            entry["incidents"] = by_rank[number]
            unchecked = [dict(i) for i in report.get("not_checked") or ()
                         if isinstance(i, dict) and i.get("check") not in ("flight_recorder", "traincheck")]
            if unchecked:
                entry["not_checked"] = unchecked
        elif look["state"] in ("behind", "stale", EARLIER_ATTEMPT) and report:
            # True when it was written and says nothing of what came after: shown as that, apart from the
            # ranks that are current, and never as the rank's incidents.
            entry["last_report"] = {
                "attempt": report.get("attempt"), "last_step": report.get("last_step"), "steps": report.get("steps"),
                "incidents": [dict(i) for i in report.get("incidents") or () if isinstance(i, dict)],
            }
            with_classes(entry["last_report"]["incidents"])
        if number in streams:
            entry["stream"] = streams[number]
        if isinstance(look.get("freshness"), Mapping):
            entry["freshness"] = dict(look["freshness"])  # a rank inside its wait (lane AD1)
        entries.append(entry)

    if status is not None and isinstance(status.get("incidents"), list):
        stream_incidents = [dict(i) for i in status["incidents"]]
    else:  # a status from a server that did not write the list yet
        stream_incidents = detect_watchdog_incidents(status)
    run_incidents = list(stream_incidents)
    # Cross-rank, so only here: each rank's report carries its latest memory reading.
    imbalance = detect_memory_imbalance({n: looks[n]["report"].get("memory") for n in judged})
    if imbalance is not None:
        run_incidents.append(imbalance)
    not_checked: list[dict[str, Any]] = []

    def newest_ts() -> float | None:
        newest = max(((look["stored"], n) for n, look in looks.items() if look.get("stored")), default=None)
        if newest is None or newest[1] not in folders:
            return None
        _, part, _ = _newest_part(folders[newest[1]])
        return _newest_stamp(_read_part(part) if part else ())

    last_step = max((look["stored"][1] for look in looks.values() if look.get("stored")), default=0)
    traincheck_summary, recorder_summary = _check_other_sources(
        run_incidents, stream_incidents, not_checked, last_step=last_step, newest_ts=newest_ts,
        traincheck=traincheck, flight_recorder=flight_recorder, flight_recorder_prefix=flight_recorder_prefix,
    )
    with_classes(run_incidents)
    states: dict[str, list[int]] = {}
    for number in sorted(looks):
        states.setdefault(looks[number]["state"], []).append(number)
    steps = [looks[n]["report"].get("last_step") for n in judged if isinstance(looks[n]["report"].get("last_step"), int)]
    run_ids = [str(r["run_id"]) for n in judged if (r := looks[n]["report"]).get("run_id") is not None]
    # The thresholds each judged rank's server applied. One server judges a run, so they agree,
    # unless it was restarted with others and some ranks have not been rebuilt since.
    by_thresholds = _thresholds_by_rank({number: applied_thresholds(looks[number]["report"]) for number in judged})
    evidence = {
        "view": ALL_RANKS_VIEW,
        "view_version": ALL_RANKS_VIEW_VERSION,
        "run_id": run_ids[0] if run_ids else source.name,
        "run": _manifest.run_block(run_ids[0] if run_ids else source.name, _manifest.find_manifest(source)),
        "run_folder": str(source.resolve()),
        "written_at_ms": int(time.time() * 1000),
        "rank_count": len(looks),
        "world_size": (status or {}).get("world_size"),
        "judged_ranks": judged,
        "unknown_ranks": [n for n in sorted(looks) if n not in by_rank],
        "clean_ranks": [n for n in judged if not by_rank[n]],
        "ranks_with_incidents": [n for n in judged if by_rank[n]],
        "never_reporting_ranks": states.get("nothing_stored", []),
        "ranks_by_state": states,
        "last_step": {"min": min(steps), "max": max(steps)} if steps else None,
        # What the judged ranks were judged by: the first judged rank's five when they differ,
        # with every set and its ranks in `thresholds_by_rank`.
        "thresholds": by_thresholds[0]["thresholds"] if by_thresholds else None,
        **({"thresholds_by_rank": by_thresholds} if len(by_thresholds) > 1 else {}),
        "incident_groups": groups,
        "run_incidents": run_incidents,
        # The stopped ranks' stacks from the status, grouped: present only while a rank is stopped.
        **({"stopped_stacks": stacks} if (stacks := group_stopped_stacks(status)) else {}),
        # Each judged rank's probe head numbers, from its current report (L20): no part is read.
        **({"head_trends_by_rank": heads} if (heads := _head_trends_by_rank(
            {n: looks[n]["report"].get("head_trends") for n in judged})) else {}),
        **({"skipped_updates_by_rank": skips} if (skips := _skipped_updates_by_rank(
            {n: looks[n]["report"].get("skipped_updates") for n in judged})) else {}),
        **({"checkpoint_tensors_by_rank": blocks} if (blocks := _checkpoint_tensors_by_rank(
            {n: looks[n]["report"].get("checkpoint_tensors") for n in judged})) else {}),
        # Each judged rank's `moe_source` and four measurements from its current report (EV-04).
        **_rank_blocks({n: looks[n]["report"] for n in judged}),
        "ranks": entries,
        **({"not_checked": not_checked} if not_checked else {}),
        **({"flight_recorder": recorder_summary} if recorder_summary else {}),
        **({"traincheck": traincheck_summary} if traincheck_summary else {}),
        **({"watchdog": {**{k: status[k] for k in (
            "summary", "written_at_ms", "world_size", "pauses", "recommended_silent_seconds",
            "silent_floor_seconds", "critical_seconds",
        ) if k in status}, **_dropped_rows_said(status)}} if status else {}),
    }
    _all_ranks_kind_checks(evidence, looks, status, flight_recorder=flight_recorder is not None,
                           traincheck=traincheck is not None)
    # The ranks' replica observations, compared by the server: its result is read from the status,
    # as everything here is (contract 8), and no part is opened for it.
    _fold_replica(evidence, *_replica_from_status(status), run_incidents, not_checked)
    with_classes(run_incidents)
    if not_checked:
        evidence["not_checked"] = not_checked
    return evidence


def rows_not_newer_note(rows: int, late: int) -> str:
    """The reader's words for the rows the server did not store because their step was not above
    one it had kept for their attempt (WAVE_AE_CONTRACT.md, section 6), with `rows` the server's
    `rows_not_newer` and `late` its `late_steps_confirmed` (0 when the status has none)."""
    head = (f"The server received {rows} row(s) whose step was not above a step it had already kept for their"
            f" attempt, and did not store them. A row sent again is counted again, so {rows} is not a number of"
            " steps lost.")
    if late:
        middle = (f" {late} distinct step(s) among them are confirmed to have arrived out of order inside one stored"
                  f" part, and are not stored. {late} is a lower bound: a late step that crosses a part boundary is"
                  f" not in it, so the same rows can give another {late} when the parts are cut elsewhere.")
    else:
        middle = " None is confirmed as a step that arrived out of order."
    # With every discarded row confirmed there are no other rows, and the sentence on them is left out.
    return head + middle + ("" if late >= rows else " The other rows are not told apart from rows sent again.")


def _rows_not_newer(entries: Iterable[Any]) -> dict[str, Any] | None:
    """`{"rows", "late_steps_confirmed"?, "ranks"}` over the rank entries of a status: each rank's
    `rows_not_newer` and `late_steps_confirmed` as written, where each is a count above zero, and
    the ranks with `rows_not_newer`. None when no rank has `rows_not_newer`: a status without the
    key reads as before. A `late_steps_confirmed` without `rows_not_newer` on its rank is not
    read, since every confirmed late step is among those rows."""
    rows = late = 0
    ranks: list[int] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        rank, count = entry.get("rank"), entry.get("rows_not_newer")
        if not (type(rank) is int and rank >= 0) or not _count(count):
            continue
        ranks.append(rank)
        rows += count
        confirmed = entry.get("late_steps_confirmed")
        late += confirmed if _count(confirmed) else 0
    if not ranks:
        return None
    return {"rows": rows, **({"late_steps_confirmed": late} if late else {}), "ranks": sorted(ranks)}


def _rows_not_newer_said(found: Mapping[str, Any], *, ranges: bool) -> dict[str, Any]:
    """The `rows_not_newer` block of a `watchdog` block (WAVE_AE_CONTRACT.md, section 6): the
    counts as the status wrote them, the ranks as ranges in the all-ranks view, and the note."""
    return {"rows": found["rows"],
            **({"late_steps_confirmed": found["late_steps_confirmed"]} if "late_steps_confirmed" in found else {}),
            **({"rank_ranges": _rank_ranges(found["ranks"], 8)} if ranges else {}),
            "includes_replays": True,
            "note": rows_not_newer_note(found["rows"], found.get("late_steps_confirmed", 0))}


def _dropped_rows_said(status: Mapping[str, Any] | None) -> dict[str, Any]:
    """`{"dropped_rows": ...}` for the all-ranks view's `watchdog` block, or nothing when no rank
    dropped a row, so such a run's block is as before (L22). The rows the server dropped because
    a key of theirs was null (`runs.dropped_rows`, which reads each rank's `dropped_rows` and
    `dropped_row_keys` in the status): the total, by null key, and the ranks that dropped any,
    as ranges (the first 8 and a count of the rest), so the block stays small however many ranks
    there are; the MCP answers give each rank's count. An upper bound: a
    batch sent again after a lost acknowledgement is counted again.

    And `{"refused_batches": ...}` beside it, or nothing when no rank had a batch refused: the
    batches the server refused whole for a key column that was missing or of another type
    (`runs.refused_batches`, lane AB5), with the same ranges. Its own key, never added to the
    rows, and not an upper bound: the server counts a batch sent again once (wave AB's seam).

    And `{"rows_not_newer": ...}` beside them, or nothing when no rank has the key (L30,
    WAVE_AE_CONTRACT.md, section 6): the rows the server did not store because their step was not
    above one it had kept for their attempt, with `late_steps_confirmed` where the status has it,
    the ranks as ranges, and the page's sentence (`rows_not_newer_note`). Never added to the rows
    dropped."""
    from .runs import dropped_rows, refused_batches

    said: dict[str, Any] = {}
    found = dropped_rows(status)
    if found is not None:
        said["dropped_rows"] = {"rows": found["rows"], **({"by_key": found["by_key"]} if found.get("by_key") else {}),
                                "rank_ranges": _rank_ranges((item["rank"] for item in found["ranks"]), 8),
                                "upper_bound": True, "note": found["note"]}
    refused = refused_batches(status)
    if refused is not None:
        said["refused_batches"] = {"batches": refused["batches"], **({"by_key": refused["by_key"]} if refused.get("by_key") else {}),
                                   "rank_ranges": _rank_ranges((item["rank"] for item in refused["ranks"]), 8),
                                   "note": refused["note"]}
    entries = status.get("ranks") if isinstance(status, Mapping) else None
    not_newer = _rows_not_newer(entries) if isinstance(entries, list) else None
    if not_newer is not None:
        said["rows_not_newer"] = _rows_not_newer_said(not_newer, ranges=True)
    return said


def _rank_dropped_rows_said(status: Mapping[str, Any] | None, rank: Any) -> dict[str, Any]:
    """`{"dropped_rows": ...}` for one rank's evidence built from a server report: the rows the
    server dropped from this rank for a null key, from its entry in the status (`runs.dropped_rows`
    over that entry alone), or nothing when it dropped none, so the block is as before. An upper
    bound, as in the all-ranks view (`_dropped_rows_said`). And `{"refused_batches": ...}` beside
    it for the batches the server refused whole from this rank, on the same terms as there, and
    `{"rows_not_newer": ...}` for this rank's rows not stored because their step was not newer
    (WAVE_AE_CONTRACT.md, section 6), without ranges."""
    from .runs import dropped_rows, refused_batches

    entries = status.get("ranks") if isinstance(status, Mapping) else None
    if not isinstance(rank, int) or not isinstance(entries, list):
        return {}
    own = {"ranks": [e for e in entries if isinstance(e, Mapping) and e.get("rank") == rank]}
    said: dict[str, Any] = {}
    found = dropped_rows(own)
    if found is not None:
        said["dropped_rows"] = {"rows": found["rows"], **({"by_key": found["by_key"]} if found.get("by_key") else {}),
                                "upper_bound": True, "note": found["note"]}
    refused = refused_batches(own)
    if refused is not None:
        said["refused_batches"] = {"batches": refused["batches"], **({"by_key": refused["by_key"]} if refused.get("by_key") else {}),
                                   "note": refused["note"]}
    not_newer = _rows_not_newer(own["ranks"])
    if not_newer is not None:
        said["rows_not_newer"] = _rows_not_newer_said(not_newer, ranges=False)
    return said


def _all_ranks_run_state(evidence: Mapping[str, Any], looks: Mapping[int, Mapping[str, Any]],
                         status: Mapping[str, Any] | None, status_age: float | None) -> dict[str, Any]:
    """The run state of the all-ranks view: every judged rank's incidents and the run's, the
    ranks that are not judged (with their states), and what was not checked on any rank."""
    judged = set(evidence.get("judged_ranks", ()))
    incidents = [group["incident"] for group in evidence.get("incident_groups", ())]
    incidents += list(evidence.get("run_incidents", ()))
    not_checked = [*evidence.get("not_checked", ()), *(item for entry in evidence.get("ranks", ())
                                                        for item in entry.get("not_checked", ()))]
    unknown = {n: f"{look['state']}: {look['reason']}" for n, look in looks.items() if n not in judged}
    steps = any((looks[n].get("report") or {}).get("steps") for n in judged)
    attempts = {n: _whole_attempt((looks[n].get("report") or {}).get("attempt")) for n in judged}
    return run_state(incidents=incidents, steps_judged=steps, status=status, status_age=status_age,
                     not_checked=not_checked, unknown_ranks=unknown, judged_ranks=judged, judged_attempts=attempts)


def _all_ranks_source(source: str | Path) -> Path:
    """The run folder the all-ranks view looks at; `ValueError` for a file or one rank's folder."""
    source = Path(source)
    if source.exists() and not source.is_dir():
        raise ValueError(f"rank='all' needs a run folder in a server's store (runs/<run_id>), not the file {source}")
    if _own_rank(source) is not None:
        raise ValueError(f"{source} is one rank's folder; rank='all' needs the run folder that holds the rank=<n> folders")
    return source


def _all_ranks_status_file(source: Path, watchdog: Any) -> Path:
    return Path(watchdog) if isinstance(watchdog, (str, Path)) else _watchdog_beside(source) or source / "watchdog.json"


def _look_at_ranks(
    source: Path,
    status: Mapping[str, Any] | None,
    memory: dict[int, _RankMemory],
    thresholds: Mapping[str, Any],
    report_grace: float,
) -> tuple[dict[int, Path], dict[int, dict[str, Any]], tuple[Any, ...]]:
    """One look at every rank of a run folder: its `rank=<n>` folders, each rank's look
    (`_rank_look`), and the key the watcher compares with what it last wrote, every rank's.
    A rank inside its wait is given as its look when it was last current, in the looks and in
    the key. `memory` is what the watcher keeps about each rank between looks, updated here;
    an empty one is a first look."""
    folders = _discover_rank_folders(source)
    absent = _ranks_without_folder(folders, status)
    numbers = sorted({*folders, *absent})
    for gone in set(memory) - set(numbers):
        del memory[gone]
    now = time.monotonic()
    looks: dict[int, dict[str, Any]] = {}
    keys: list[tuple[int, Any]] = []
    for number in numbers:
        held = memory.setdefault(number, _RankMemory())
        look, in_grace = _rank_look(folders.get(number), absent.get(number), held, thresholds, report_grace, now)
        if look["state"] == "current":
            held.last_current_look = look
        elif in_grace and held.last_current_look is not None:
            last = held.last_current_look
            if _thresholds_moved(look, last.get("report")):
                # Its server judges by other thresholds since its last current report (lane AD1):
                # that result is not carried over, and the rank is not judged until it is current.
                look = {**look, "freshness": freshness_block(
                    (last.get("report") or {}).get("last_step"), (look.get("stored") or (None, None))[1],
                    "thresholds_changed")}
            else:
                # A rank inside its wait is a moment between a part and its report, as the
                # single-rank watcher treats it: it alone does not cause a write, and in a write
                # that another rank or the status causes it reads as its last current report, as
                # the single-rank watcher writes from its last current report. Before Nov 18, 2026
                # it was written as not judged for that moment. Since lane AD1 the steps stored
                # since are said beside it (`freshness`), and the key compared is the same.
                block = _report_freshness({**last, "state": look["state"], "stored": look.get("stored")}, True)
                look = {**last, **({"freshness": block} if block else {})}
        looks[number] = look
        keys.append((number, _rank_key(look)))
    return folders, looks, tuple(keys)


# The look state the all-ranks view gives a rank whose report is current with its store while
# the server's status shows that the job was started again on it (L6, Nov 19, 2026).
EARLIER_ATTEMPT = "earlier_attempt"


def _earlier_attempt_looks(looks: Mapping[int, Mapping[str, Any]],
                           status: Mapping[str, Any] | None) -> dict[int, Mapping[str, Any]]:
    """`looks` with each current rank that the server's status says was judged by an earlier
    attempt (`_not_current_by_status`: marked `attempt_without_steps`, or shown at another
    attempt than its report's) read as `earlier_attempt`, with why. Such a rank is not judged:
    it is in none of the view's values, as `run_state` calls it unknown, and its report is
    shown as its `last_report`. Only the view's copy changes, never the watcher's memory or the
    key it compares looks by (the status's stamp carries the flag, so it changing is news)."""
    if status is None:
        return dict(looks)
    attempts = {n: _whole_attempt((look.get("report") or {}).get("attempt"))
                for n, look in looks.items() if look["state"] == "current"}
    found = _not_current_by_status(status, attempts)
    return {n: ({**look, "state": EARLIER_ATTEMPT, "reason": found[n]} if n in found else look)
            for n, look in looks.items()}


def _all_ranks_whole(
    source: Path,
    folders: Mapping[int, Path],
    looks: Mapping[int, Mapping[str, Any]],
    status: Mapping[str, Any] | None,
    status_age: float | None,
    other_sources: Mapping[str, Any],
) -> dict[str, Any]:
    """The whole all-ranks evidence of one look: `_all_ranks_evidence`, how old the status is
    when that is too old, and the run state. A rank whose current report the status shows to be
    an earlier attempt's is `earlier_attempt` here (`_earlier_attempt_looks`)."""
    looks = _earlier_attempt_looks(looks, status)
    # A rank whose report was made under other thresholds than the reader gives (`other_thresholds`)
    # was judged, under thresholds not in force: `thresholds_changed` beside the state's own name.
    looks = {n: ({**look, "freshness": freshness_block(
                 (look.get("report") or {}).get("last_step"), (look.get("stored") or (None, None))[1],
                 "thresholds_changed")}
                 if look.get("state") == "other_thresholds" and not look.get("freshness") else look)
             for n, look in looks.items()}
    evidence = _all_ranks_evidence(source, folders, looks, status, **other_sources)
    if status_age is not None and "watchdog" in evidence:
        evidence["watchdog"]["not_written_for_seconds"] = round(status_age, 1)
    if (block := _all_ranks_freshness(evidence, looks)) is not None:
        evidence["freshness"] = block
    evidence["run_state"] = _all_ranks_run_state(evidence, looks, status, status_age)
    return evidence


def _all_ranks_freshness(evidence: Mapping[str, Any], looks: Mapping[int, Mapping[str, Any]]) -> dict[str, Any] | None:
    """The all-ranks view's `freshness` block (CLI-10, lane AD1), or None when every judged rank's
    report covers its newest stored step: `thresholds_changed` when a rank's server judges by other
    thresholds since its last current report or the rank is `other_thresholds`, `nothing_judged` when no rank is judged and steps are
    stored, `records_newer` when a judged rank is inside its wait with steps stored since. The steps
    are the lowest judged and the highest stored over the ranks with a block of their own, on their
    entries in `ranks` (a rank inside its wait, or judged under thresholds not in force). Other
    unjudged ranks beside judged ones give no block: they are unknown by their state."""
    blocks = [look["freshness"] for look in looks.values() if isinstance(look.get("freshness"), Mapping)]
    judged = [b["judged_to_step"] for b in blocks if isinstance(b.get("judged_to_step"), int)]
    stored = [b["records_to_step"] for b in blocks if isinstance(b.get("records_to_step"), int)]
    if any(b.get("why") == "thresholds_changed" for b in blocks):
        return freshness_block(min(judged, default=None), max(stored, default=None), "thresholds_changed")
    if not evidence.get("judged_ranks"):
        steps = [look["stored"][1] for look in looks.values() if look.get("stored")]
        return freshness_block(None, max(steps), "nothing_judged") if steps else None
    if blocks:
        return freshness_block(min(judged, default=None), max(stored, default=None), "records_newer")
    return None


def all_ranks_evidence(
    source: str | Path,
    *,
    watchdog: str | Path | None = None,
    traincheck: str | Path | None = None,
    flight_recorder: str | Path | None = None,
    flight_recorder_prefix: str | None = None,
    overload_threshold: float | None = None,
    drift_tolerance: float | None = None,
    starve_window: int | None = None,
    starve_ratio: float | None = None,
    top_k: int | Mapping[str, int] | None = None,
    update_window: int | None = None,
) -> dict[str, Any]:
    """One look at every rank of a run: the evidence `watch(source, rank="all")` writes on its
    first look, returned and not written.

    `source` is a run folder in a server's store (`runs/<run_id>`, holding `rank=<n>` folders);
    a file or one rank's folder raises `ValueError`. The look reads each rank's
    `incidents.json`, the name of its newest part and the run's `watchdog.json` (`watchdog`
    names another status file). No detector runs, so the server binary is not needed, and no
    part is opened but for one case shared with the watcher: when `flight_recorder` finds a
    stuck collective in complete dumps, the newest stored part is read to date the dump (that
    needs `pyarrow`). A rank counts for or against incidents only while its report is current;
    with nothing stored and no status the evidence has `rank_count` 0.

    The five thresholds are the reader's (`read_server_report`): left at None, each rank's
    report is taken with what its server applied, which the evidence lists under `thresholds`.
    A first look has no memory of earlier ones, so a rank whose report is behind its store is
    `behind` here where a running watcher would wait `report_grace` for it.
    A threshold given here that no server can apply (`check_thresholds`: the Rust detectors'
    contract, so not finite, negative, a window or `top_k` below 1, not a number, a bool) raises
    `ValueError` before anything is read; None is "not given".
    The evidence is `{"view": "all_ranks", ...}`: see `_all_ranks_evidence` and the README.
    """
    given = {
        "overload_threshold": overload_threshold, "drift_tolerance": drift_tolerance,
        "starve_window": starve_window, "starve_ratio": starve_ratio, "top_k": top_k,
        "update_window": update_window,
    }
    check_thresholds(**given)
    source = _all_ranks_source(source)
    status = read_watchdog(_all_ranks_status_file(source, watchdog))
    thresholds = _given_thresholds(given)
    folders, looks, _ = _look_at_ranks(source, status, {}, thresholds, REPORT_GRACE_SECONDS)
    other_sources = {"traincheck": traincheck, "flight_recorder": flight_recorder,
                     "flight_recorder_prefix": flight_recorder_prefix}
    return _all_ranks_whole(source, folders, looks, status, _status_age(status), other_sources)


def watch_all_ranks(
    source: str | Path,
    output: str | Path | None = None,
    *,
    interval: float = 2.0,
    polls: int | None = None,
    on_update: Any = None,
    server_reports: bool = True,
    report_grace: float = REPORT_GRACE_SECONDS,
    compact_window: int | None = None,
    trigger: Any = None,
    **build_options: Any,
) -> Path | None:
    """Keep one evidence file for every rank of a run up to date: `watch(run_folder, rank="all")`.

    `source` is a run folder in a server's store (`runs/<run_id>`, holding `rank=<n>` folders).
    Each look reads, for every rank, its `incidents.json` and the name of its newest part, and
    the run's `watchdog.json`: nothing else, so a look of 64 ranks stays cheap, and no part is
    opened and no detector run here, in any state. The file is replaced by rename, and only
    when something changed: a rank's report (attempt, last step, steps, rebuilds), a rank's
    state, the ranks present, or the status. A rank whose report was current and is then behind,
    missing or unreadable is waited for `report_grace` seconds from that moment, as one rank's
    watcher waits, and reads as its last current report meanwhile; behind for longer, it is
    `stale`. The evidence is `{"view": "all_ranks", ...}`, not
    the per-rank evidence: see `_all_ranks_evidence` and the README for its keys.

    A rank's incidents count only while its report is current (`read_server_report`); a rank
    that is behind, stale, missing, unreadable, unusable, on other thresholds or has nothing
    stored is `unknown`, with its state and why, and is never listed among the ranks with no
    incidents. So is a rank whose report is current while the server's status shows the job
    started again on it (`earlier_attempt`: marked `attempt_without_steps`, or at another
    attempt than its report's; L6). Rank folders may appear or disappear between looks. `server_reports=False`
    (`--recompute`) is refused, because it asks for the detectors to run here, a rank at a
    time and for the whole run each time; use `rank=<n>` for that. Each rank's report is taken
    with the thresholds its server applied, which the evidence lists (`thresholds`); any of the
    five given here (`overload_threshold`, `drift_tolerance`, `starve_window`, `starve_ratio`,
    `top_k`) must be the one a rank's server applied, or that rank is `other_thresholds`.
    `flight_recorder` and `traincheck` are read once for the run and their incidents are run-level.
    A rank inside its wait reads as its last current report, with the steps stored since on its
    entry (`freshness`, `records_newer`), and the view has a `freshness` block of its own in three
    cases: a rank inside its wait (`records_newer`), a rank judged under thresholds not in force
    (`thresholds_changed`: its server lists other thresholds than in its last current report, so it
    is not judged until current again, or the rank is `other_thresholds`), or no rank judged while
    steps are stored (`nothing_judged`) (CLI-10, lane AD1; `_all_ranks_freshness`). A rank that is
    `behind` on a first look, `stale`, `missing` or otherwise unjudged beside judged ranks gets no
    block: it is unknown by its state, so the block's absence does not mean every rank's judgment
    covers its store.
    `detectors` ("auto", "python" or "rust") is accepted and changes nothing: every rank's incidents are
    the Rust detectors' already, from its server's report, and no detector runs here, so the
    server binary is not needed. Returns the evidence path, or None if nothing was written.
    """
    source = _all_ranks_source(source)
    if build_options.pop("detectors", "python") not in ("auto", "python", "rust"):
        raise ValueError("detectors must be 'auto', 'python' or 'rust'")
    build_options.pop("server_binary", None)
    if not server_reports:
        raise ValueError(
            "rank='all' cannot recompute: that would read the parts and run the detectors for every rank, one after"
            " another, on every change (about 8 s a rank at 10,000 steps); recompute one rank with rank=<n>"
        )
    status_file = _all_ranks_status_file(source, build_options.pop("watchdog", None))
    # Said before the first look, as `watch` says it: this one may be called directly.
    check_thresholds(**{name: build_options[name] for name in THRESHOLD_NAMES if name in build_options})
    thresholds = _given_thresholds(build_options)
    other_sources = {
        name: build_options.get(name) for name in ("traincheck", "flight_recorder", "flight_recorder_prefix")
    }
    memory: dict[int, _RankMemory] = {}
    status_clock = _StatusAgeTracker()
    emitted: Any = None
    written: Path | None = None
    done = 0
    try:
        while polls is None or done < polls:
            if done:
                time.sleep(interval)
            done += 1
            if trigger is not None:
                trigger.tick()
            status = read_watchdog(status_file)
            status_age = status_clock.check(status)
            stamp = (_status_stamp(status, status_age), _other_sources_stamp(**other_sources))
            if isinstance(status, Mapping) and "replica" in status:
                # The server's replica result changing is news (contract 7: a verdict is due at the
                # next look). Absent, the stamp is what it was before the key.
                stamp = (*stamp, json.dumps(status["replica"], sort_keys=True, default=str))
            if dropped := _dropped_rows_said(status):
                # So is a change in the rows the server dropped (L22), which the view's `watchdog`
                # block carries. None dropped, the stamp is what it was before.
                stamp = (*stamp, json.dumps(dropped, sort_keys=True))
            folders, looks, keys = _look_at_ranks(source, status, memory, thresholds, report_grace)
            if not looks and status is None:
                continue
            # Compared with what was written, not with the last look: a rank inside its wait is
            # written and compared as its last current look, so one that comes back with the
            # report it had is no news.
            seen = (keys, stamp)
            if seen == emitted:
                continue
            evidence = _all_ranks_whole(source, folders, looks, status, status_age, other_sources)
            path = Path(output) if output else Path(f"{evidence['run_id']}.all-ranks.evidence.json")
            emitted = seen
            # This view has no per-step samples, so `compact_window` only marks and names the file.
            written = _write_output(evidence, path, output is not None, compact_window)
            if on_update is not None:
                on_update(evidence, written)
            if trigger is not None:
                trigger.look(evidence, written)
    except KeyboardInterrupt:
        pass
    return written


def default_evidence_path(source: Path, records: list[dict[str, Any]]) -> Path:
    """`<file>.evidence.json` next to a `.jsonl` file; `<run_id>.evidence.json` in the
    current directory for a directory of Parquet parts."""
    if source.is_dir():
        run_id = str(records[0]["run_id"]) if records and records[0].get("run_id") else source.name
        return Path(f"{run_id}.evidence.json")
    return source.with_suffix(".evidence.json")


def _router_sizes(records: list[dict[str, Any]]) -> dict[str, int]:
    """Each router's largest size, in the order routers first appear with counts the column carries:
    a router whose entry is unreadable on a record (`_router_counts`) takes no layer number there."""
    sizes: dict[str, int] = {}
    for record in records:
        moe = record.get("moe")
        if not isinstance(moe, Mapping):
            continue
        for name, value in moe.items():
            counts = _router_counts(value)
            if counts is None:
                continue
            size = len(counts)
            if name not in sizes or size > sizes[name]:
                sizes[name] = size
    return sizes


MOE_KINDS = ("moe_imbalance", "starved_experts")


def refused_routers_not_checked(records: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The `router` entry of `not_checked`: routers the capture refused to count.

    A router whose output cannot be read as expert indices is not counted (capture.py's
    router guard: the first integer tensor's shape, or what an `indices=` function returned
    or raised) and leaves no entry in `moe`, so the MoE rules never see it. The capture says
    so once, under `moe_refused` on the first record it wrote after the refusal. Without this
    entry a run whose only router was refused would read "no MoE incident", which is not "MoE
    checked". The server writes the same entry (detect.rs).

    The capture writes the whole reason after "router NAME is not counted: ": why, and the
    two ways to have it counted for how that router was named, which only the capture knows.
    It is printed as written, so this entry, the server's and the session's warning say the
    same. Records from before that carry the why alone and read as before.
    """
    first: dict[str, tuple[int, str]] = {}
    for record in records:
        refused = record.get("moe_refused")
        if not isinstance(refused, Mapping):
            continue
        for name, why in refused.items():
            first.setdefault(str(name), (int(record["step"]), str(why)))
    if not first:
        return None
    names = sorted(first)
    return {
        "check": "router",
        "kinds": list(MOE_KINDS),
        "routers": names,
        "reason": "; ".join(f"router {name} is not counted: {first[name][1]}" for name in names),
        "step": min(step for step, _ in first.values()),
    }


MOE_ROUTING_CHECK = "moe_routing"


def routers_not_read_not_checked(records: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The `moe_routing` entry of `not_checked` (WAVE_AF_CONTRACT.md, section 8): routers the records
    name whose counts are 0 on every record, so their routing was not read (a capture that counted
    no forward of a training step, as in eval mode under reentrant checkpointing, wrote zeros).

    `{"check": "moe_routing", "kinds": [the two MoE kinds], "routers": [sorted], "reason"}`, or None
    when every router named has a count above 0 on some record. A router with counts on some records
    and zeros on others is read, as before. While some router of the records was read, the kinds
    are judged on it and the entry only names the others (`kind_checks`); with none read they are
    `unavailable`. The server writes the same entry (lane D).

    A router is read on a record when the sum of its counts is above 0, as `_MoeSamples.feed`,
    `input_counts` and Rust's `Moe::row` read it (wave AJ, A2: before, any one count above 0 was
    enough, so `[5, -5]` read as read here and as not read in Rust). An entry the column cannot carry
    (`_router_counts`) names no router on that record (A5)."""
    # Each router named, and whether a sum above 0 was seen for it. A router once seen read is not
    # looked at again: a whole run of a read router costs one dictionary lookup a record.
    read: dict[str, bool] = {}
    for record in records:
        moe = record.get("moe")
        if not isinstance(moe, Mapping):
            continue
        for name, value in moe.items():
            name = str(name)
            if read.get(name):
                continue
            total = _router_total(value)
            if total is None:
                continue
            read[name] = total > 0
    names = sorted(name for name, seen in read.items() if not seen)
    if not names:
        return None
    return {
        "check": MOE_ROUTING_CHECK,
        "kinds": list(MOE_KINDS),
        "routers": names,
        "reason": "; ".join(f"router {name} recorded no tokens on any step: its routing was not read" for name in names),
    }


def _diversity_from_counts(counts: list[int], n_experts: int) -> float:
    total = sum(counts)
    if total <= 0 or n_experts <= 0:
        return 0.0
    shares = [c / total for c in counts if c > 0]
    entropy = -math.fsum(s * math.log(s) for s in shares)
    return math.exp(entropy) / n_experts


def _non_finite_gradients(record: Mapping[str, Any]) -> list[str]:
    """What is not finite among a record's gradient norms, as a `non_finite` incident names it:
    each parameter, else `<total>`, and `<total before clipping>`."""
    bad = [name for name, norm in _object(record, "grad_norms").items() if _is_non_finite(norm)]
    if _is_non_finite(record.get("grad_norm")) and not bad:
        bad.append("<total>")
    if _is_non_finite(record.get("clip_grad_norm")):
        bad.append("<total before clipping>")
    return bad


def skipped_updates(records: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """One rank's skipped updates by who reported them (STEP_AND_EVIDENCE_CONTRACT.md, section 1),
    the evidence's `skipped_updates`: `{"caller", "scaler", "both"}` counts, then `partial`, the
    steps on which the scaler skipped some optimizers and applied others (`partial_step`,
    WAVE_AE_CONTRACT.md, section 4), counted there and under no other name and only when above
    zero, with
    `non_finite_gradients` (`steps`, `first_step`, and `gradients` named at the first, as
    `non_finite` names them) only when a skip the caller reported, alone or with the scaler, had
    a gradient norm that was not finite. None unless a record carries `step_skipped_by`, so the
    evidence of a capture written before the key is what it was; there, a skip without the key
    counts as the scaler's (`skipped_by`). It raises nothing and changes no kind's verdict."""
    listed = sorted((r for r in records if isinstance(r, Mapping)), key=lambda r: r["step"])
    if not any("step_skipped_by" in r for r in listed):
        return None
    counts = dict.fromkeys(SKIPPED_BY, 0)
    partial = 0
    kept: dict[str, Any] | None = None
    for record in listed:
        by = skipped_by(record)
        if by is None:
            continue
        if partial_step(record):
            partial += 1
        else:
            counts[by] += 1
        bad = _non_finite_gradients(record) if by in ("caller", "both") else []
        if bad:
            if kept is None:
                kept = {"steps": 0, "first_step": int(record["step"]), "gradients": bad}
            kept["steps"] += 1
    return {**counts, **({"partial": partial} if partial else {}),
            **({"non_finite_gradients": kept} if kept is not None else {})}


def detect_non_finite(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first step where a layer statistic or gradient norm is NaN or Inf.

    On a skipped update (`step_skipped`, whoever reported it), Inf or NaN gradients are what
    the skip is for, so only activations are checked there: a layer statistic that is not
    finite is raised with no gradient named. A gradient that is not finite on a skip the caller
    reported is kept in the evidence's `skipped_updates` instead (STEP_AND_EVIDENCE_CONTRACT.md,
    section 1); the scaler's skips are counted in `detect_loss_scale_trouble`.

    The records are judged in step order, records of one step in the order given (wave AJ, A6:
    before, a direct call judged them in list order). `build_evidence` feeds `_NonFinite` records
    already in order and does not come through here.
    """
    state = _NonFinite()
    state.feed(sorted(records, key=lambda r: r["step"]))
    return state.found


class _NonFinite:
    """`detect_non_finite`'s state: the first record found, fed records in their order (lane AD11).
    Once one is found the records after it are not read."""

    __slots__ = ("found",)

    def __init__(self) -> None:
        self.found: dict[str, Any] | None = None

    def feed(self, records: Iterable[Mapping[str, Any]]) -> None:
        if self.found is not None:
            return
        for record in records:
            bad_layers = [
                name
                for name, stats in layer_entries(record)  # every field of each layer that is an object
                if any(_is_non_finite(v) for v in stats.values())
            ]
            bad_grads = [] if record.get("step_skipped") else _non_finite_gradients(record)
            if bad_layers or bad_grads:
                self.found = {
                    "kind": "non_finite",
                    "step": record["step"],
                    "layers": bad_layers,
                    "gradients": bad_grads,
                }
                return

    def result(self) -> dict[str, Any] | None:
        return copy.deepcopy(self.found)


def detect_replica_drift(
    records: list[dict[str, Any]], *, tolerance: float = DRIFT_TOLERANCE
) -> dict[str, Any] | None:
    """Return the first step where cross-GPU parameter spread exceeds `tolerance` or is
    not finite (a NaN or Inf weight on some replica).

    Norm, sum and seeded-projection fingerprints are a screening check, not proof of
    equality: the projection catches changes that keep the norm and sum, such as two
    elements swapping places, but a change that keeps all three is still possible.
    Megatron's `check_param_hashes_across_dp_replicas` compares hashes for exact equality.
    Records written before the projection existed carry two values and are read as before.

    The records are judged in step order, records of one step in the order given (wave AJ, A6).
    A parameter's entry that the `replica_spread` column cannot carry (`_spread_values`: an empty
    list, text other than "nan", "inf" and "-inf", a bool, a number too large for a float, anything
    that is not a list) is not judged, and the other parameters of the record are (A4: before, it
    raised `ValueError`). A null, "nan" and "inf" are values that are not finite, as before.
    """
    return _replica_drift(sorted(records, key=lambda r: r["step"]), tolerance)


def _replica_drift(records: Iterable[Mapping[str, Any]], tolerance: float) -> dict[str, Any] | None:
    """`detect_replica_drift` over records in the order given: `build_evidence`'s, which are in
    order already."""
    for record in records:
        spreads = _spread_table(record)
        if not spreads:
            continue
        diverged = {}
        for name, vals in spreads.items():
            values = _spread_values(vals)
            if not values:  # refused, or `[]`: nothing to judge (Rust finds no drift in it either)
                continue
            detail = dict(zip(("norm_spread", "sum_spread", "projection_spread"), values))
            if not all(math.isfinite(v) for v in values):
                # A NaN or Inf weight on any replica makes the spread non-finite, which
                # compares as no drift; report it instead.
                diverged[name] = {**detail, "non_finite": True}
            elif max(values) > tolerance:
                diverged[name] = detail
        if diverged:
            return {
                "kind": "replica_drift",
                "step": record["step"],
                "tolerance": tolerance,
                "parameters": diverged,
            }
    return None


# --- Replica observations (SDK-22) ---------------------------------------------------------
#
# Each rank records its own observation under `replica` on a scheduled step, and a reader that
# has every rank's record compares them (REPLICA_OBSERVATION_CONTRACT.md, sections 2 to 7). This
# is the Python reader; the server is the other engine, and the two give the same object for
# every case in `tests/fixtures/replica/`.
_SPREAD_NAMES = ("norm_spread", "sum_spread", "projection_spread")  # as `detect_replica_drift` names them
REPLICA_NOT_COMPARED_LISTED = 20  # steps a `kind_checks` coverage entry lists; the rest are counted


def _fingerprint(value: Any) -> float:
    """One fingerprint as a float. Contract 1 writes one that is not finite as the string "nan",
    "inf" or "-inf"; anything that is not a number is read as not finite."""
    if isinstance(value, bool):
        return math.nan
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except OverflowError:  # Rust reads it as text (`detect/extra_json.rs`), and text as NaN
            return math.nan
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return math.nan
    return math.nan


def _observation(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    found = record.get("replica")
    return found if isinstance(found, Mapping) else None


def _observation_read(observation: Mapping[str, Any]) -> dict[str, Any]:
    """An observation as both engines compare it (REPLICA_OBSERVATION_CONTRACT.md section 1, as
    `replica.rs`'s `observed` reads it; BUG-11):

    - a record whose `v` is not the integer 1 names no group it can be compared in (`group` None:
      the step is `not_established`, why `group`);
    - a `launch` that is not text names no launch;
    - an entry of `params` that is not an object, or whose `fp` is not a list of at least two
      fingerprints (`[norm, sum]`), cannot be compared: it is read as `unread`.
    Everything else is the record's own."""
    version = observation.get("v")
    launch = observation.get("launch")
    params = observation.get("params")
    unread = observation.get("unread")
    held: dict[str, Any] = {}
    not_read: dict[str, Any] = dict(unread) if isinstance(unread, Mapping) else {}
    for name, row in (params.items() if isinstance(params, Mapping) else ()):
        prints = row.get("fp") if isinstance(row, Mapping) else None
        if isinstance(prints, list) and len(prints) >= 2:
            held[name] = row
        else:
            not_read.setdefault(name, "fewer than two fingerprints")
    return {**observation, "launch": launch if isinstance(launch, str) else None,
            "group": observation.get("group") if type(version) is int and version == 1 else None,
            "params": held, "unread": not_read}


def _json_key(value: Any) -> Any:
    """`value` as a key that is equal for two JSON values exactly when `serde_json` finds them
    equal: a signature is compared as the value it is, as Rust compares it (BUG-11), so `123` and
    `"123"`, `true` and `1`, and `1` and `1.0` differ."""
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if isinstance(value, str):
        return ("str", value)
    if value is None:
        return ("null",)
    if isinstance(value, (list, tuple)):
        return ("list", tuple(_json_key(v) for v in value))
    if isinstance(value, Mapping):
        return ("object", frozenset((str(k), _json_key(v)) for k, v in value.items()))
    return ("other", repr(value))


def _spread_table(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    found = record.get("replica_spread")
    return found if isinstance(found, Mapping) else None


def _spread_scope(record: Mapping[str, Any]) -> tuple[str, tuple[int, ...]] | None:
    """What a gathered spread says it covers (contract 1): its launch and group, or None when it
    does not say both (an old record, a table the job computed itself, a group not read)."""
    scope = record.get("replica_spread_scope")
    if not isinstance(scope, Mapping):
        return None
    launch, group = scope.get("launch"), scope.get("group")
    if launch is None or not isinstance(group, (list, tuple)):
        return None
    return str(launch), tuple(group)


def _over_tolerance(record: Mapping[str, Any], tolerance: float) -> list[str]:
    """The parameters a gathered spread shows drift on, by today's rule on its own record
    (`detect_replica_drift`, contract 6), sorted."""
    found = _replica_drift([record], tolerance)
    return sorted(found["parameters"]) if found else []


def _compare_one_group(seen: Mapping[int, Mapping[str, Any]], tolerance: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Contract 3 and 4 for one launch, group and step whose every rank sent an observation:
    the step entry's facts, and the spreads of each parameter that drifted (`_SPREAD_NAMES`,
    or `non_finite`) for the incident's detail."""
    ranks = sorted(seen)
    if len(ranks) < 2:  # a group of one has no rank to be compared with
        return {"fingerprints": "not_compared", "fingerprints_why": "one_rank",
                "signatures": "not_compared", "signatures_why": "one_rank"}, {}
    observed = [seen[rank] for rank in ranks]
    devices = {o.get("device") for o in observed}
    projections = {o.get("projection") for o in observed}
    methods = {o.get("signature") for o in observed}
    prints_comparable = len(devices) == 1
    projection_column = len(projections) == 1 and None not in projections
    signatures_comparable = len(methods) == 1 and None not in methods
    params = [o.get("params") if isinstance(o.get("params"), Mapping) else {} for o in observed]
    unread = [o.get("unread") if isinstance(o.get("unread"), Mapping) else {} for o in observed]
    left_out: dict[str, str] = {}
    drifted: dict[str, dict[str, Any]] = {}
    differing: list[str] = []
    prints_made = signatures_made = 0
    for name in sorted({str(n) for table in (*params, *unread) for n in table}):
        if any(name in table for table in unread):
            left_out[name] = "unread"
            continue
        rows = [table.get(name) for table in params]
        if not all(isinstance(row, Mapping) for row in rows):
            left_out[name] = "not_recorded"
            continue
        kinds = {(tuple(row.get("shape") or ()), row.get("dtype")) for row in rows}  # type: ignore[union-attr]
        if len(kinds) > 1:
            left_out[name] = "shape_dtype"
            continue
        if signatures_comparable:
            signatures_made += 1
            if len({_json_key(row.get("sig")) for row in rows}) > 1:  # type: ignore[union-attr]
                differing.append(name)
        if not prints_comparable:
            continue
        prints_made += 1
        width = 3 if projection_column else 2
        table = [[_fingerprint(v) for v in list(row.get("fp") or ())[:width]] for row in rows]  # type: ignore[union-attr]
        if not all(math.isfinite(v) for prints in table for v in prints):
            drifted[name] = {"non_finite": True}
            continue
        norms = [prints[0] for prints in table if prints]
        scale = max(math.fsum(norms) / len(norms), 1e-12) if norms else 1e-12  # `spread_from_fingerprints`
        spreads = [(max(column) - min(column)) / scale for column in zip(*table)]
        if any(spread > tolerance for spread in spreads):
            drifted[name] = dict(zip(_SPREAD_NAMES, spreads))
    facts: dict[str, Any] = {}
    if prints_made:
        facts["fingerprints"] = "drift" if drifted else "agree"
        if drifted:
            facts["drift_parameters"] = sorted(drifted)
    else:
        facts["fingerprints"] = "not_compared"
        facts["fingerprints_why"] = "device" if not prints_comparable else "no_parameters"
    if signatures_made:
        facts["signatures"] = "differ" if differing else "agree"
        if differing:
            facts["signature_parameters"] = differing
    else:
        facts["signatures"] = "not_compared"
        facts["signatures_why"] = "method" if not signatures_comparable else "no_parameters"
    if left_out:
        facts["not_compared"] = left_out
    if not projection_column and projections != {None}:
        facts["projection_compared"] = False
    return facts, drifted


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    half = len(ordered) // 2
    return ordered[half] if len(ordered) % 2 else (ordered[half - 1] + ordered[half]) / 2


def _attributed_by_fingerprints(rows: Mapping[int, Mapping[str, Any]], width: int, tolerance: float) -> dict[str, Any]:
    """D1 by fingerprints for one parameter that drifted (contract 4a): `rows` are each compared
    rank's entry in `params`, `width` the compared columns (2 without the projection)."""
    ranks = sorted(rows)
    if len(ranks) == 2:
        return {"verdict": "inconclusive", "why": "two_ranks"}
    values = {rank: [_fingerprint(v) for v in list(rows[rank].get("fp") or ())[:width]] for rank in ranks}
    columns = min(len(v) for v in values.values())
    finite = [rank for rank in ranks if all(math.isfinite(v) for v in values[rank][:columns])]
    apart = {rank for rank in ranks if rank not in finite}  # a fingerprint that is not finite stands apart
    if finite and columns:
        scale = max(math.fsum(values[rank][0] for rank in finite) / len(finite), 1e-12)
        for column in range(columns):
            middle = _median([values[rank][column] for rank in finite])
            apart |= {rank for rank in finite if abs(values[rank][column] - middle) / scale > tolerance}
    if not apart:
        return {"verdict": "inconclusive", "why": "no_outlier"}
    if 2 * (len(ranks) - len(apart)) <= len(ranks):
        return {"verdict": "inconclusive", "why": "no_majority"}
    return {"verdict": "attributed", "outliers": sorted(apart)}


def _attributed_by_signatures(rows: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """D1 by signatures for one parameter whose signatures differ: one signature held by a strict
    majority names the other ranks (SCOUT's exact classifier)."""
    ranks = sorted(rows)
    if len(ranks) == 2:
        return {"verdict": "inconclusive", "why": "two_ranks"}
    held: dict[Any, list[int]] = {}
    for rank in ranks:
        held.setdefault(_json_key(rows[rank].get("sig")), []).append(rank)
    most = max(held.values(), key=len)
    if 2 * len(most) <= len(ranks):
        return {"verdict": "inconclusive", "why": "no_majority"}
    return {"verdict": "attributed", "outliers": [rank for rank in ranks if rank not in most]}


def _attribution(members: Mapping[int, Mapping[str, Any]], facts: Mapping[str, Any], where: Mapping[str, Any],
                 tolerance: float) -> list[dict[str, Any]]:
    """The attribution entries of one complete step entry (contract 4a), from the rows compared:
    one for each parameter that drifted, by fingerprints, and one for each whose signatures
    differ, by signatures. Each parameter is read on its own."""
    rows_of = {rank: (o.get("params") if isinstance(o.get("params"), Mapping) else {}) for rank, o in members.items()}
    projections = {o.get("projection") for o in members.values()}
    width = 3 if len(projections) == 1 and None not in projections else 2  # the columns `_compare_one_group` compared
    out = []
    for name in facts.get("drift_parameters") or ():
        rows = {rank: params[name] for rank, params in rows_of.items()}
        out.append({**where, "parameter": name, "by": "fingerprints", **_attributed_by_fingerprints(rows, width, tolerance)})
    for name in facts.get("signature_parameters") or ():
        rows = {rank: params[name] for rank, params in rows_of.items()}
        out.append({**where, "parameter": name, "by": "signatures", **_attributed_by_signatures(rows)})
    return out


def _replica_place(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    """Contract 4's order of step entries: by step, then the rows that name no launch, then by
    launch, then by group (a group that could not be named first), a step's own entry before
    the rows set aside from it."""
    launch, group = entry.get("launch"), entry.get("group")
    return (entry["step"], launch is not None, launch or "", group is not None, list(group or ()),
            entry.get("why") == "attempt")


def _replica_judged(
    records: Iterable[Mapping[str, Any]], tolerance: float, attempt_order: Mapping[tuple[str, Any], list[Any]] | None,
) -> tuple[dict[str, Any], dict[tuple[Any, ...], dict[str, Any]]]:
    """`compare_replicas` and, by `(step, launch, group)` of each incident from observations,
    the spreads of its parameters."""
    listed = [dict(r) for r in records]
    check_one_run(listed)
    if attempt_order:
        listed = in_attempt_order(listed, attempt_order)
    legacy: list[dict[str, Any]] = []
    # Every gathered spread, kept under what it says it covers and its step (contract 5).
    covers: dict[tuple[Any, int], dict[str, set[str]]] = {}
    # Observations by launch and step, each rank's; and the rows of a rank's later attempt.
    cells: dict[tuple[Any, int], dict[int, Mapping[str, Any]]] = {}
    keeps: dict[tuple[str, int], Any] = {}  # (launch, rank): the attempt that keeps the launch
    aside: dict[tuple[str, tuple[int, ...], int], set[int]] = {}
    for record in listed:
        step, rank = int(record["step"]), int(record.get("rank", 0))
        observation, table = _observation(record), _held_spread_table(record)
        if table is not None:
            over = _over_tolerance(record, tolerance)
            held = covers.setdefault((_spread_scope(record), step), {"covered": set(), "over": set()})
            # Only a held entry with values says anything of its parameter: a refused one, or `[]`,
            # covers nothing, so it cannot set aside the drift the observations show (wave AJ, A4).
            held["covered"].update(str(name) for name, values in table.items() if _spread_values(values))
            held["over"].update(over)
            if observation is None:  # an old capture: judged as before, on its own (contract 6)
                legacy.append({"rank": rank, "step": step, "drift": bool(over), **({"parameters": over} if over else {})})
        if observation is None:
            continue
        observation = _observation_read(observation)
        launch = observation.get("launch")
        if launch is not None:
            attempt = record.get("attempt", 0)
            if keeps.setdefault((launch, rank), attempt) != attempt:
                # This rank's later attempt under one launch read the launch an earlier start left
                # in the store (contract 2): set aside, never paired with that start's rows.
                group = tuple(observation.get("group") or ())
                aside.setdefault((launch, group, step), set()).add(rank)
                continue
        cells.setdefault((launch, step), {})[rank] = observation
    steps: list[dict[str, Any]] = []
    raised: list[dict[str, Any]] = []
    attribution: list[dict[str, Any]] = []
    details: dict[tuple[Any, ...], dict[str, Any]] = {}
    for (launch, step), seen in cells.items():
        named = {tuple(o.get("group") or ()) for o in seen.values()}
        if launch is None:
            steps.append({"launch": None, "group": list(next(iter(named))) if len(named) == 1 else None, "step": step,
                          "coverage": "not_established", "why": "launch"})
            continue
        unlike = sorted(named)
        overlapping = any(set(a) & set(b) for at, a in enumerate(unlike) for b in unlike[at + 1:])
        outside = any(rank not in (o.get("group") or ()) for rank, o in seen.items())
        if overlapping or outside:
            steps.append({"launch": launch, "group": None, "step": step, "coverage": "not_established", "why": "group"})
            continue
        for group in unlike:
            members = {rank: o for rank, o in seen.items() if tuple(o.get("group") or ()) == group}
            entry: dict[str, Any] = {"launch": launch, "group": list(group), "step": step}
            missing = [rank for rank in group if rank not in members]
            if missing:
                steps.append({**entry, "coverage": "insufficient", "missing": missing})
                continue
            facts, drifted = _compare_one_group(members, tolerance)
            decided = covers.get(((launch, group), step), {}).get("covered", set())
            steps.append({**entry, "coverage": "complete", **({"gathered": sorted(decided)} if decided else {}), **facts})
            attribution.extend(_attribution(members, facts, {"step": step, "launch": launch, "group": list(group)}, tolerance))
            mine = [name for name in facts.get("drift_parameters", ()) if name not in decided]
            if mine:  # a gathered spread of this launch, group and step decides the parameters it holds
                raised.append({"step": step, "parameters": mine, "source": "replica", "launch": launch, "group": list(group)})
                details[(step, launch, group)] = {name: drifted[name] for name in mine}
    for (launch, group, step), ranks in aside.items():
        steps.append({"launch": launch, "group": list(group), "step": step, "coverage": "not_established",
                      "why": "attempt", "ranks": sorted(ranks)})
    for (scope, step), held in covers.items():
        if held["over"]:
            raised.append({"step": step, "parameters": sorted(held["over"]), "source": "replica_spread",
                           **({"launch": scope[0], "group": list(scope[1])} if scope else {})})
    steps.sort(key=_replica_place)
    raised.sort(key=lambda i: (i["step"], i["source"], i.get("launch") or "", list(i.get("group") or ())))
    attribution.sort(key=lambda e: (e["step"], e["launch"], e["group"], e["parameter"], e["by"]))
    return {"steps": steps, "replica_drift": raised, **({"legacy": legacy} if legacy else {}),
            **({"attribution": attribution} if attribution else {})}, details


def compare_replicas(
    records: Iterable[Mapping[str, Any]],
    *,
    tolerance: float = DRIFT_TOLERANCE,
    attempt_order: Mapping[tuple[str, Any], list[Any]] | None = None,
) -> dict[str, Any]:
    """Compare the ranks' replica observations of one run (REPLICA_OBSERVATION_CONTRACT.md,
    sections 2 to 6): `{"steps": [...], "replica_drift": [...]}`, with `legacy` when a record
    holds a gathered spread and no observation, each spelled as the contract's word table
    (section 5) and the fixtures spell them. The server writes the same lists under `replica`
    in a run's `watchdog.json`, with `v` and `tolerance` (contract 8).

    `records` are one run's (`ValueError` otherwise), of any number of ranks. Their order across
    ranks does not matter: the records of several rank files can be given file by file or
    interleaved. Only each rank's own order does, since the first attempt of a rank under a
    launch keeps the launch and a later one's rows are set aside (contract 2): it is contract 1's
    order of the rank's attempts. For a rank in one local file that is the order of its lines;
    for a store, its parts in stored attempt order (`load_records`); for a rank in several local
    files, give `attempt_order` (`attempt_order_across` of that rank's files), which puts each
    rank's attempts in that order first (`in_attempt_order`). Give every record written, not
    `resolve_attempts`' (a later attempt's rollback leaves out rows a restart needs here).

    A gathered spread (`replica_spread`) is judged on its own record by today's rule
    (`detect_replica_drift`) wherever it is. It decides a step's comparison only for the
    parameters it holds and only when its `replica_spread_scope` names that step's launch and
    group.

    `attribution` (contract 4a, D1), a list present only when it has an entry, says for each
    complete step where the ranks differ on a parameter which ranks stand apart from a strict
    majority, by fingerprints for each parameter that drifted and by signatures for each whose
    signatures differ, or why none can be named (`two_ranks`, `no_majority`, `no_outlier`). It
    describes a difference and raises nothing: `replica_drift` is the same with or without it."""
    return _replica_judged(records, tolerance, attempt_order)[0]


def _replica_object(result: Mapping[str, Any], tolerance: float) -> dict[str, Any]:
    """The result as the server carries it in `watchdog.json` (contract 8)."""
    return {"v": 1, "tolerance": tolerance, **result}


def _status_says_no_detect(status: Mapping[str, Any]) -> bool:
    """Whether the server's status says it runs with `--no-detect`. The status has no key for it;
    its `run_state` says it in words, on the reason for the ranks without a current report
    (`ranks_unknown`, `watchdog.rs`), which such a server gives for every rank it has heard of.
    A server older than `run_state`, or one with no rank yet, does not say."""
    state = status.get("run_state")
    reasons = state.get("reasons") if isinstance(state, Mapping) else None
    return any(isinstance(reason, Mapping) and reason.get("code") == "ranks_unknown"
               and "--no-detect" in str(reason.get("detail") or "") for reason in reasons or ())


def _replica_from_status(status: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The server's cross-rank result from its status (`replica` at the top level of
    `watchdog.json`, contract 8), or why there is none: `(object, None)` or `(None, not_stated)`.
    `(None, None)` when there is no status, which the view already says.

    A server with detection on writes `replica` in every status it writes, its last one when it
    stops among them (an empty result for a run without observations). So a status without the
    key is from a server started with `--no-detect` or from one older than the comparison
    (contract 8: the first must not be called the second). The words say which when the status
    says the server runs with `--no-detect` (`_status_says_no_detect`), and both otherwise."""
    if status is None:
        return None, None
    found = status.get("replica")
    if found is None:
        why = ("the server runs with --no-detect, which writes none" if _status_says_no_detect(status)
               else "a server started with --no-detect, or an older tensorquill-server")
        return None, {"code": "status_without_replica",
                      "detail": f"the server's status has no `replica` result ({why}): the ranks' replica observations"
                                " are not compared here"}
    if (not isinstance(found, Mapping) or not isinstance(found.get("steps"), list)
            or not isinstance(found.get("replica_drift"), list) or _finite(found.get("tolerance")) is None):
        return None, {"code": "status_replica_unreadable",
                      "detail": "the server's status has a `replica` value without the `steps` and `replica_drift`"
                                " lists and a finite `tolerance`: the ranks' replica observations are not compared here"}
    return dict(found), None


def _replica_step_compared(entry: Mapping[str, Any]) -> bool:
    """Whether a step entry checked the kind: every rank sent an observation and fingerprints were
    compared, or a gathered spread of that launch, group and step decides some parameters."""
    return entry.get("coverage") == "complete" and (entry.get("fingerprints") in ("agree", "drift")
                                                    or bool(entry.get("gathered")))


def _replica_gap(entry: Mapping[str, Any]) -> dict[str, Any]:
    """A step entry that did not check the kind, with why: the missing ranks, why the records
    cannot say who is compared, or why nothing was compared."""
    gap = {key: entry.get(key) for key in ("step", "launch", "group", "coverage")}
    for key in ("missing", "why", "ranks", "fingerprints_why"):
        if key in entry:
            gap[key] = entry[key]
    return gap


def replica_kind_check(
    check: Mapping[str, Any], replica: Mapping[str, Any], *, all_ranks: bool, records: int | None = None,
    held_ranks: Iterable[int] = (),
) -> dict[str, Any]:
    """The `replica_drift` entry of `kind_checks` where replica observations were read: `check` is
    the entry counted from gathered spreads (`kind_checks`, or a server's report), `replica` the
    cross-rank result (`compare_replicas`, or the server's object).

    The kind is `checked` when a step compared fingerprints (or a gathered spread of its launch,
    group and step decides it), or when a gathered spread was judged on its own record, as
    before. A step whose coverage is `insufficient` or `not_established`, or `complete` with
    nothing compared, does not read as checked: `coverage` lists those steps, the latest
    `REPLICA_NOT_COMPARED_LISTED` with the count of the rest, each with why, and names the
    ranks missing from any of them under `missing_ranks`. With no step compared and no spread,
    one rank's evidence calls the kind `not_configured` (a comparison across ranks, as the two
    other cross-rank kinds are there), and an all-ranks view `unavailable`, with
    `check: "replica_coverage"` and the reason; but a run whose every step entry is a group of
    one rank is `not_configured` in either, as one rank alone is for the other cross-rank kinds
    (`_cross_rank_check`). A result with no step entry at all (a run without observations, or
    before its first scheduled step) leaves `check` as it is. `signatures_differ` lists the steps
    where signatures differ: evidence only, which raises nothing (contract 5). `records` is how
    many records carried an observation or a spread; `held_ranks`, the ranks of a file that one
    rank's evidence was picked from (`ranks_in_source`), lets `detail` tell a rank left out of
    the evidence from one that sent nothing."""
    steps = [entry for entry in replica.get("steps") or () if isinstance(entry, Mapping)]
    if not steps:
        return dict(check)
    compared = [entry for entry in steps if _replica_step_compared(entry)]
    gaps = [_replica_gap(entry) for entry in steps if not _replica_step_compared(entry)]
    missing = sorted({rank for gap in gaps for rank in gap.get("missing") or ()})
    differ = [{"step": e.get("step"), "launch": e.get("launch"), "group": e.get("group"),
               "parameters": list(e.get("signature_parameters") or ())} for e in steps if e.get("signatures") == "differ"]
    alone = all(gap.get("fingerprints_why") == "one_rank" for gap in gaps)
    if compared or check.get("value") == "checked":
        value = "checked"
    else:
        value = "unavailable" if all_ranks and not alone else "not_configured"
    held = gaps[-REPLICA_NOT_COMPARED_LISTED:]
    coverage = {
        "steps": len(steps), "compared": len(compared),
        **({"missing_ranks": missing} if missing else {}),
        **({"not_compared": held} if held else {}),
        **({"not_compared_omitted": len(gaps) - len(held)} if len(gaps) > len(held) else {}),
    }
    if not all_ranks and value != "checked":
        held = set(held_ranks)
        left, absent = [r for r in missing if r in held], [r for r in missing if r not in held]
        coverage["detail"] = (
            "one rank's records: the ranks' replica observations are compared across ranks, in the all-ranks view"
            + (f"; rank{'s' if len(left) != 1 else ''} {_rank_ranges(left, 8)} {'are' if len(left) != 1 else 'is'} in"
               " the source and not judged in this rank's evidence" if left else "")
            + (f"; no observation of rank{'s' if len(absent) != 1 else ''} {_rank_ranges(absent, 8)} is in the"
               " records read" if absent else ""))
    out: dict[str, Any] = {
        "value": value,
        **({"records": records} if records is not None else
           {k: check[k] for k in ("records",) if k in check}),
        **{k: check[k] for k in ("records_needed",) if k in check},
        "enough_history": value == "checked",
        **{k: check[k] for k in ("needs",) if k in check},
        **_noted("replica_drift", value),
    }
    if value == "unavailable":
        out["check"] = "replica_coverage"
        out["reason"] = _replica_reason(coverage)
    elif value == "not_configured" and alone:
        out["note"] = NO_PEER_WORDS
    out["coverage"] = coverage
    if differ:
        out["signatures_differ"] = differ[-REPLICA_NOT_COMPARED_LISTED:]
    return out


def _replica_reason(coverage: Mapping[str, Any]) -> str:
    missing = coverage.get("missing_ranks") or ()
    if not coverage.get("steps"):
        return "no step of the ranks' replica observations is listed yet"
    said = f"none of the {coverage['steps']} replica step entries compared fingerprints"
    return said + (f": rank{'s' if len(missing) != 1 else ''} {_rank_ranges(missing, 8)} sent no observation"
                   " for the steps listed" if missing else "")


def replica_not_checked(check: Mapping[str, Any]) -> dict[str, Any] | None:
    """The `not_checked` entry for a `replica_drift` kind check that is `unavailable` for want of
    coverage (`replica_kind_check` in an all-ranks view), so the run reads `degraded` as for any
    kind that could not be judged; None otherwise."""
    if check.get("value") != "unavailable" or check.get("check") != "replica_coverage":
        return None
    return {"check": "replica_coverage", "kinds": ["replica_drift"], "reason": check.get("reason")}


def _replica_incident(
    replica: Mapping[str, Any], details: Mapping[tuple[Any, ...], Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """The first incident the ranks' observations raise (`source: "replica"`), as evidence lists an
    incident: `parameters` by name (with the spreads and `non_finite` where they were computed
    here; empty from the server's object), the `tolerance`, the `launch` and `group` compared.
    One incident for the kind, as each detector gives its first; every one is in the result's
    `replica_drift`. When a gathered spread also shows drift on that step, it is named under
    `also` with its source: the two cannot be shown to be one comparison (contract 5)."""
    raised = [i for i in replica.get("replica_drift") or () if isinstance(i, Mapping)]
    first = next((i for i in raised if i.get("source") == "replica"), None)
    if first is None:
        return None
    group = list(first.get("group") or ())
    spreads = (details or {}).get((first.get("step"), first.get("launch"), tuple(group))) or {}
    also = [{key: i[key] for key in ("source", "parameters", "launch", "group") if key in i}
            for i in raised if i is not first and i.get("step") == first.get("step")]
    # The ranks named, by the fingerprints' reading only (contract 4a): they raised it. The
    # signatures' reading stays in the result's `attribution`, as evidence.
    names = [str(name) for name in first.get("parameters") or ()]
    attributed = [dict(e) for e in replica.get("attribution") or ()
                  if isinstance(e, Mapping) and e.get("by") == "fingerprints" and e.get("step") == first.get("step")
                  and e.get("launch") == first.get("launch") and list(e.get("group") or ()) == group
                  and str(e.get("parameter")) in names]
    return {
        "kind": "replica_drift", "step": first.get("step"), "tolerance": replica.get("tolerance"),
        "parameters": {name: dict(spreads.get(name) or {}) for name in names},
        "source": "replica", "launch": first.get("launch"), "group": group,
        **({"attribution": attributed} if attributed else {}),
        **({"also": also} if also else {}),
    }


def replica_line(evidence: Mapping[str, Any]) -> str | None:
    """One line on the ranks' replica observations for a summary: how many steps were compared,
    the steps that were not and the ranks missing from them, and the steps whose signatures
    differ (evidence, not an incident). None for evidence without a result. A status without one
    (`replica_not_stated`) gets no line: the view cannot tell whether any rank recorded an
    observation, so the line would be on every run of an older server or one started with
    `--no-detect`; the key says it."""
    check = (evidence.get("kind_checks") or {}).get("replica_drift")
    if not isinstance(evidence.get("replica"), Mapping) or not isinstance(check, Mapping):
        return None
    coverage = check.get("coverage")
    if not isinstance(coverage, Mapping):  # a result with no step entry yet
        return None
    line = f"Replica observations: {coverage.get('compared', 0)} of {coverage.get('steps', 0)} step entries compared"
    gaps = coverage.get("not_compared") or ()
    if gaps:
        last = gaps[-1]
        line += (f"; not compared at {len(gaps) + int(coverage.get('not_compared_omitted') or 0)}, the latest step"
                 f" {last.get('step')} ({last.get('coverage')}"
                 + (f", missing rank{'s' if len(last['missing']) != 1 else ''} {_rank_ranges(last['missing'], 8)}"
                    if last.get("missing") else f", {last.get('why') or last.get('fingerprints_why')}") + ")")
    differ = check.get("signatures_differ") or ()
    if differ:
        line += (f"; signatures differ at step {differ[-1]['step']} on {', '.join(differ[-1]['parameters'][:3])}"
                 " (evidence only: fingerprints decide drift)")
    return line + "."


def replica_across_ranks(
    records: Iterable[Mapping[str, Any]],
    *,
    tolerance: float = DRIFT_TOLERANCE,
    attempt_order: Mapping[tuple[str, Any], list[Any]] | None = None,
) -> dict[str, Any] | None:
    """The cross-rank result of a run's local rank files, as the server carries it (`v`,
    `tolerance`, and `compare_replicas`' lists), for `combine_rank_evidence(replica=...)`; None
    when no record holds an observation. `records` are every rank's records as written, in the
    order `compare_replicas` takes."""
    listed = [dict(r) for r in records]
    if not any(_observation(r) is not None for r in listed):
        return None
    return _replica_object(compare_replicas(listed, tolerance=tolerance, attempt_order=attempt_order), tolerance)


def _fold_replica(
    evidence: dict[str, Any], replica: Mapping[str, Any] | None, not_stated: Mapping[str, Any] | None,
    run_incidents: list[dict[str, Any]], not_checked: list[dict[str, Any]],
    details: Mapping[tuple[Any, ...], Mapping[str, Any]] | None = None,
) -> None:
    """Put a cross-rank replica result into an all-ranks evidence: the result under `replica`, its
    first incident from observations among `run_incidents`, the `replica_drift` entry of
    `kind_checks` from it (`replica_kind_check`), and, when no step could be compared, a
    `not_checked` entry. With no result, `not_stated` says why under `replica_not_stated`."""
    if replica is None:
        if not_stated is not None:
            evidence["replica_not_stated"] = dict(not_stated)
        return
    evidence["replica"] = dict(replica)
    found = _replica_incident(replica, details)
    if found is not None:
        run_incidents.append(found)
    checks = evidence.get("kind_checks")
    if isinstance(checks, dict) and isinstance(checks.get("replica_drift"), Mapping):
        checks["replica_drift"] = replica_kind_check(checks["replica_drift"], replica, all_ranks=True)
        gap = replica_not_checked(checks["replica_drift"])
        if gap is not None:
            not_checked.append(gap)


def _check_other_sources(
    incidents: list[dict[str, Any]],
    stream_incidents: list[dict[str, Any]],
    not_checked: list[dict[str, Any]],
    *,
    last_step: int,
    newest_ts: Any,
    traincheck: str | Path | None,
    flight_recorder: str | Path | None,
    flight_recorder_prefix: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Add what TrainCheck's results and PyTorch's Flight Recorder dumps show to `incidents`.

    Only Python reads these two: the files are on the training nodes and never reach the
    server. `newest_ts()` gives the time of the last recorded step, and is called only when
    a stuck collective has to be dated against it. Returns the two summaries for the evidence.
    """
    traincheck_summary: dict[str, Any] | None = None
    if traincheck is not None:
        from .invariants import read_traincheck, traincheck_incident

        try:
            found = read_traincheck(traincheck)
        except (OSError, ValueError) as error:
            not_checked.append({"check": "traincheck", "reason": str(error)})
        else:
            traincheck_summary = {k: found[k] for k in ("distinct_invariants_violated", "first_violation_step")}
            hit = traincheck_incident(found, step=last_step)
            if hit is not None:
                incidents.append(hit)
    recorder_summary: dict[str, Any] | None = None
    if flight_recorder is not None:
        from .collectives import analyze_flight_recorder, flight_recorder_incident

        try:
            analysis = analyze_flight_recorder(flight_recorder, prefix=flight_recorder_prefix)
        except (ImportError, OSError, ValueError) as error:
            not_checked.append({"check": "flight_recorder", "reason": str(error)})
        else:
            recorder_summary = {
                k: analysis[k]
                for k in ("dumped_ranks", "collectives", "problem_count", "incomplete_dumps", "ignored_before_buffer_start")
            }
            hit = flight_recorder_incident(analysis, step=last_step)
            if hit is not None:
                incidents.append(hit)
                if hit["kind"] == "collective_hang" and not analysis["incomplete_dumps"]:
                    newest = newest_ts()
                    if newest is not None and analysis["newest_dump_time"] >= newest:
                        # A stuck collective in PyTorch's own dump is a second source for a hang the
                        # stream only suggested, if the dump is complete and was written after the last
                        # step TensorQuill recorded. An old or partial dump proves nothing about now.
                        # Only a stall is corroborated: the status also lists `duplicate_batches`,
                        # which has no assessment (lane U7, Nov 19, 2026: it raised KeyError here).
                        for incident in stream_incidents:
                            if incident.get("kind") in ("job_stalled", "rank_silent") \
                                    and incident.get("assessment") == "suspected_hang":
                                incident["assessment"] = "confirmed_hang"
                                incident["corroborated_by"] = "collective_hang"
    return traincheck_summary, recorder_summary


def _moe_samples(
    records: list[dict[str, Any]],
    router_sizes: Mapping[str, int],
    router_top_k: Mapping[str, int],
    overload_threshold: float,
    fixed_top_k: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """The per-step MoE samples as the Python rules judge them: one for every record and
    router, with the experts found overloaded or dead there. `_mark_starved` adds the starved
    ones. The Rust detectors make the same list (`tensorquill-server detect --samples`).
    `router_top_k` is each router's experts per token until a record gives one (`_MoeSamples`)."""
    return _MoeSamples(router_sizes, router_top_k, overload_threshold, fixed_top_k).feed(records)


class _MoeSamples:
    """`_moe_samples`'s state: each router's history and the samples made so far, fed records in step
    order (`feed`, any number of times), as `_GradSpikes` is (lane AD11). The router sizes, their
    threshold are the run's and fixed for the state: a record that changes any of them changes every
    sample, so a state is not fed past such a change (`_Incremental`).

    Experts per token are judged step by step, as Rust's `Moe::row` judges them (BUG-13): a record's
    `moe_top_k` applies to that record and the later ones, until another changes it. Before any,
    a router's value is `router_top_k`'s (1 when it names none). A router `fixed_top_k` names keeps
    that value whatever the records say (`top_k=`, an override)."""

    def __init__(self, router_sizes: Mapping[str, int], router_top_k: Mapping[str, int], overload_threshold: float,
                 fixed_top_k: Mapping[str, int] | None = None) -> None:
        self.router_sizes, self.overload_threshold = router_sizes, overload_threshold
        self.fixed_top_k = dict(fixed_top_k or {})
        self.router_top_k = {**router_top_k, **self.fixed_top_k}  # in force: changed by the records fed
        self.history: dict[str, list[tuple[list[int], int, float]]] = {name: [] for name in router_sizes}
        self.samples: list[dict[str, Any]] = []

    def feed(self, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Make the samples of `records`, keep them, and return them."""
        router_sizes, router_top_k, overload_threshold = self.router_sizes, self.router_top_k, self.overload_threshold
        routers = list(router_sizes)
        history = self.history
        new: list[dict[str, Any]] = []
        for record in sorted(records, key=lambda r: r["step"]):
            step = int(record["step"])
            for name, k in _top_k_entries(record):
                if name not in self.fixed_top_k:
                    router_top_k[name] = k
            moe = record.get("moe")
            if not isinstance(moe, Mapping):
                moe = {}
            moe_stats = record.get("moe_stats")
            if not isinstance(moe_stats, Mapping):  # a value that is not an object holds no statistics (section 1.5)
                moe_stats = {}
            for layer_idx, router_name in enumerate(routers, 1):
                expected_experts = router_sizes[router_name]
                # An entry the column cannot carry is no entry (wave AJ, A5: before, `int(c)` raised
                # over a string's characters), as Rust never sees it.
                given = _router_counts(moe.get(router_name)) if router_name in moe else None
                if given is None:
                    counts = [0] * expected_experts
                    missing = True
                else:
                    counts = list(given)
                    # A sum of 0 or less routed no token: a missing sample, as Rust's `Moe::row` reads
                    # it (A2: a negative sum reached the judged branch's `rule` unset).
                    missing = sum(counts) <= 0
                n_experts = len(counts)
                tokens = sum(counts)
                if missing or tokens <= 0:
                    diversity: float | None = None
                    overloaded: list[int] = []
                    dead: list[int] = []
                else:
                    shares = [c / tokens for c in counts]
                    # The recorded diversity when it is a finite number, else the counts' (section
                    # 1.5): a bool was read as 1.0 here and as absent by Rust.
                    recorded = _recorded_diversity(moe_stats.get(router_name))
                    diversity = recorded if recorded is not None else _diversity_from_counts(counts, n_experts)
                    k = router_top_k.get(router_name, 1)
                    window = history[router_name]
                    if k >= 3:
                        overloaded = _jump_overloaded(counts, k, diversity, window)
                        rule: dict[str, Any] = {
                            "rule": "diversity_drop", "drop": DIVERSITY_DROP, "jump": OVERLOAD_JUMP
                        }
                    else:
                        threshold = effective_overload_threshold(n_experts, tokens, overload_threshold)
                        overloaded = [i for i, share in enumerate(shares, 1) if share > threshold]
                        rule = {"rule": "share", "threshold": threshold}
                    zero_experts = [i for i, c in enumerate(counts, 1) if c == 0]
                    dead = (
                        zero_experts
                        if _dead_experts_significant(
                            n_experts, tokens, overloaded, len(zero_experts), diversity
                        )
                        else []
                    )
                    # During top-k warm-up we can attribute an independently detected dead-expert
                    # collapse using share. With a baseline, only a change from that router's
                    # own past can mark an expert overloaded.
                    if k >= 3 and len(window) < OVERLOAD_MIN_HISTORY and dead:
                        mean_share = 1.0 / n_experts
                        se_share = math.sqrt(mean_share * (1.0 - mean_share) / tokens)
                        collapse_threshold = max(2.0 * mean_share, mean_share + 3.0 * se_share)
                        if collapse_threshold >= 1.0 / k:
                            # A top-k expert's share cannot pass 1/k (each token picks it once at
                            # most), so with n_experts <= 2k that bound names nothing (DET-11): use
                            # the midpoint between an even share and the largest one there is.
                            collapse_threshold = (mean_share + 1.0 / k) / 2.0
                        overloaded = [
                            i for i, share in enumerate(shares, 1) if share > collapse_threshold
                        ]
                        rule = {"rule": "collapse", "threshold": collapse_threshold}
                    if k >= 3 and not (overloaded or dead):
                        # Once a collapse is flagged, compare subsequent samples to the same
                        # pre-collapse baseline. A recovered sample starts updating it again.
                        window.append((counts, tokens, diversity))
                        if len(window) > OVERLOAD_WINDOW:
                            window.pop(0)
                sample: dict[str, Any] = {
                    "step": step,
                    "layer": layer_idx,
                    "router": router_name,
                    "routed_tokens": tokens,
                    "counts": counts,
                    "diversity": diversity,
                    "overloaded": overloaded,
                    "dead": dead,
                }
                if missing:
                    sample["missing"] = True
                else:
                    sample["overload_rule"] = rule
                new.append(sample)
        self.samples.extend(new)
        return new


def _same_record(kept: Mapping[str, Any], given: Mapping[str, Any]) -> bool:
    """Whether `given` is the record `kept` was: the same keys in the same order, each value the same
    object. `build_evidence` copies each record it is handed (`load_records`), so the copies of one
    record of a watcher's `_Tail` share every value; a record read again from a file shares none."""
    return kept is given or (list(kept) == list(given) and all(kept[key] is given[key] for key in kept))


class _Incremental:
    """The detectors that read every record, kept between a watcher's looks (lane AD11, OPEN_BUGS
    section 9: `--watch` judged the whole run again at every look, 8.1 s at 10,000 steps).

    `judge` is handed a look's records as `build_evidence` judges them (resolved, in step order) and
    gives the MoE samples, the `moe_imbalance` sample and the starting state (`_RouterStarts`), and
    the `starved_experts`, `grad_spike`, activation and `non_finite` incidents. When the records
    judged at the last look are the first records of this one, each the same record, and nothing
    the states were made with has changed, only the records after them are fed to the states: this
    look's evidence is what a full judgment would give, since each state is what feeding it every
    record in step order would leave. Otherwise every state starts again from the first record
    (`path` says which, and why):

    - the first look;
    - another option (the six thresholds) or whole-run fact the states read: the routers and their
      sizes, their top-k, and whether the run clips (`clip_grad_norm`), which picks the gradient
      series of every step;
    - a record judged before that is not the same or not in its place: rows that came late and sort
      before the newest step, an attempt that replaced steps already judged, a file that was cut
      short, replaced or rewritten and read again from the start (`_Tail` says which rewrite it does
      not detect);
    - a step that is not a whole number, which the states' own step order might not keep.

    Everything else in the evidence is made again at every look from every record, as before: those
    passes are cheap and some read the arrival order or files outside the records. Each look's
    `samples` is a new list of the samples kept here, which readers must not change in place.
    """

    def __init__(self) -> None:
        self.key: Any = None
        self.records: list[Mapping[str, Any]] = []
        self.path: tuple[str, Any] = ("full", "first look")
        self.texts: list[str] = []  # each sample's JSON text, for the watcher's writer (`samples_text`)
        self.handed: list[dict[str, Any]] | None = None  # the last `samples` list given out

    def _start(self, key: Any, router_sizes: Mapping[str, int], fixed_top_k: Mapping[str, int],
               overload_threshold: float, starve_window: int, starve_ratio: float, grad_key: str) -> None:
        self.key = key
        self.records = []
        self.texts = []
        # Each router's experts per token is read step by step from the records (BUG-13): 1 before
        # any, unless `top_k=` fixes it.
        self.moe = _MoeSamples(dict(router_sizes), {}, overload_threshold, fixed_top_k)
        self.starved = _StarvedMarks(starve_window, starve_ratio)
        self.starved_firsts = _StarvedFirsts()
        self.starts = _RouterStarts()
        self.grad = _GradSpikes(grad_key)
        self.activation = _ActivationAnomalies()
        self.non_finite = _NonFinite()

    def judge(self, records: list[dict[str, Any]], *, router_sizes: Mapping[str, int],
              router_top_k: Mapping[str, int], overload_threshold: float, starve_window: int,
              starve_ratio: float, options: tuple[Any, ...], top_k: int | Mapping[str, int] | None = None) -> dict[str, Any]:
        grad_key = _gradient_key(records)
        key = repr((overload_threshold, starve_window, starve_ratio, options, list(router_sizes.items()),
                    list(router_top_k.items()), grad_key))
        kept = self.records
        why = None
        if self.key is None:
            why = "first look"
        elif key != self.key:
            why = "thresholds or a whole-run fact changed"
        elif len(records) < len(kept) or not all(map(_same_record, kept, records)):
            why = "records judged before changed"
        elif not all(type(r["step"]) is int for r in records):
            why = "a step that is not a whole number"
        if why is None:
            new = records[len(kept):]
            self.path = ("incremental", len(new))
        else:
            self._start(key, router_sizes, _fixed_top_k(router_sizes, top_k), overload_threshold, starve_window,
                        starve_ratio, grad_key)
            new = records
            self.path = ("full", why)
        self.records = list(records)
        made = self.moe.feed(new)
        self.starved.feed(made)
        self.starved_firsts.feed(made)
        self.starts.feed(made)
        self.grad.feed(new)
        self.activation.feed(new)
        self.non_finite.feed(new)
        self.handed = list(self.moe.samples)
        return {
            "samples": self.handed,
            "moe_hit": self.starts.first(),
            "starting": self.starts.starting(),
            "starved_hit": self.starved_firsts.result(),
            "grad_hit": self.grad.result(),
            "activation": self.activation.result(),
            "non_finite_hit": self.non_finite.result(),
        }

    def samples_text(self, evidence: Mapping[str, Any]) -> str | None:
        """The JSON text of `evidence["samples"]` as `write_evidence` writes it, when that is the list
        the last `judge` gave and none of it was replaced; each sample's text is made once. None
        otherwise, and the writer makes it all."""
        samples = evidence.get("samples")
        if samples is None or samples is not self.handed or len(self.texts) > len(samples):
            return None
        for sample in samples[len(self.texts):]:
            self.texts.append(_json_text(sample))
        return "[" + ",".join(self.texts) + "]"


def build_evidence(
    source: str | Path | Iterable[Mapping[str, Any]],
    *,
    rank: int | None = None,
    overload_threshold: float = OVERLOAD_THRESHOLD,
    drift_tolerance: float = DRIFT_TOLERANCE,
    starve_window: int = STARVE_WINDOW,
    starve_ratio: float = STARVE_RATIO,
    top_k: int | Mapping[str, int] | None = None,
    watchdog: str | Path | Mapping[str, Any] | None = None,
    flight_recorder: str | Path | None = None,
    flight_recorder_prefix: str | None = None,
    traincheck: str | Path | None = None,
    detectors: str = "python",
    server_binary: str | os.PathLike[str] | None = None,
    run_manifest: str | Path | Mapping[str, Any] | None = None,
    update_window: int = UPDATE_WINDOW,
    records_file: str | Path | None = None,
    attempt_order: Mapping[tuple[str, Any], list[Any]] | None = None,
) -> dict[str, Any]:
    """Build viewer-neutral detector evidence from captured step records.

    `attempt_order` gives each rank's attempts oldest first by `(run_id, rank)` where the
    records' own order cannot, as for a rank's several local files (`attempt_order_across`,
    contract 1). The records are put in that order first (`in_attempt_order`), so the Python
    detectors and the stream sent to the Rust ones judge the same order.

    `run_end` says whether the session that wrote a local `.jsonl` file closed it
    (`sink.read_end_marker`): read from `source` when that is the file, or from `records_file`
    for a caller that hands over records it has already read. Any state but `closed` means the
    run is not known to have finished. A server's store or a list of records has none.
    `writers`, for that same file, is there only when another process wrote the same run ID and
    rank at once, in a file of its own beside it (`<run_id>.rank<N>.<pid>.jsonl`,
    `sink.writers_block`): `{"state": "several", "files", "detail"}`. The file then holds one of the
    rank's writers, not the rank's whole, whatever its run state says of its own records.

    The evidence's `run` block (project, name, run ID) is read from the run manifest
    (`manifest.py`): the one beside `source` when that is a path, or `run_manifest`, which is
    the manifest itself, its file, or the `.jsonl` file or run folder it sits beside (for a
    caller that hands over records it has already read). A run without a manifest reads as
    project `uncategorized` with its run ID as its name.

    The detectors read one rank's steps. `rank` picks that rank out of a run directory or a
    list of records; records of several ranks with no `rank` given raise `ValueError`,
    because merged they repeat step numbers and give incidents that belong to neither rank.
    The evidence names the rank judged (`rank`), and, when `rank` picked it out of a file or list
    that also held other ranks, every rank that source held (`ranks_in_source`, L21): such an
    evidence file beside a file of several ranks is that one rank's.

    `watchdog` is the server's `watchdog.json` (a path or the parsed document). A run directory
    or one of its `rank=` folders automatically reads the run's status; its progress verdict
    becomes `job_stalled` or `rank_silent`.
    `flight_recorder` is a directory of PyTorch Flight Recorder dumps (one pickle per rank,
    `flight_recorder_prefix` the common start of their names): PyTorch's analyzer runs over
    them and its first problem becomes `collective_hang` or `collective_desync`. When the
    analyzer is missing (PyTorch before 2.10) or the dumps cannot be read, the check is
    listed under `not_checked` with the reason.
    `traincheck` is TrainCheck's `violations_summary.json` (or its results directory):
    the violated invariants become one `traincheck_violation` incident.

    `top_k` overrides the experts-per-token recorded as `moe_top_k`, for captures made
    before it was recorded: one value for every router, or a value per router name.
    `overload_threshold` is the base assignment-share threshold for top-1 and top-2.
    Top-3 and higher use a per-router baseline once ten unflagged samples are available.
    `update_window` is how many of the last step-to-step changes of the update size make the
    baseline of `update_spike` (`detect_update_spikes`), where the rule's paper averages once
    over the run's learning-rate warm-up.

    `detectors="rust"` takes the incidents of the server's kinds (`SERVER_KINDS`) from the
    Rust detectors, run here on the same records through `tensorquill-server detect`
    (`server_binary`, else where `server_detect.find_binary` looks; needs `pyarrow`).
    The six options above go to it as one JSON object (`--thresholds`). A value the Rust
    rules refuse (a negative or non-finite number, a window or `top_k` below 1 or not whole,
    a bool) raises `ValueError` under either detectors, before anything is read
    (`check_thresholds`): the Python ones judge by the same contract.
    The per-step `samples` come from that run too (`detect --samples`), so no MoE rule runs
    here. The weight trends, the router names and sizes and the kinds read from other files
    are still made here. The evidence then carries `"detectors": "rust"`.
    `detectors="auto"` is "rust" when that binary reports the protocol number this package was
    built for and `pyarrow` is installed, and otherwise "python" with one `RuntimeWarning`
    naming what is missing and what installs it. The default stays "python": a library call
    starts no process and gives the same evidence on every machine unless it is asked to. The
    command line's default for one read is "auto" (`main`).

    When one of the six options is not the default, the evidence lists all six as `thresholds`
    (the name a server's report gives them), and `thresholds_line` names those that differ. At
    the defaults there is no such key, so that evidence is what it was before the key.
    """
    return _build_evidence(
        source, None, rank=rank, overload_threshold=overload_threshold, drift_tolerance=drift_tolerance,
        starve_window=starve_window, starve_ratio=starve_ratio, top_k=top_k, watchdog=watchdog,
        flight_recorder=flight_recorder, flight_recorder_prefix=flight_recorder_prefix, traincheck=traincheck,
        detectors=detectors, server_binary=server_binary, run_manifest=run_manifest, update_window=update_window,
        records_file=records_file, attempt_order=attempt_order,
    )


def _build_evidence(
    source: str | Path | Iterable[Mapping[str, Any]],
    memo: "_Incremental | None",
    *,
    rank: int | None = None,
    overload_threshold: float = OVERLOAD_THRESHOLD,
    drift_tolerance: float = DRIFT_TOLERANCE,
    starve_window: int = STARVE_WINDOW,
    starve_ratio: float = STARVE_RATIO,
    top_k: int | Mapping[str, int] | None = None,
    watchdog: str | Path | Mapping[str, Any] | None = None,
    flight_recorder: str | Path | None = None,
    flight_recorder_prefix: str | None = None,
    traincheck: str | Path | None = None,
    detectors: str = "python",
    server_binary: str | os.PathLike[str] | None = None,
    run_manifest: str | Path | Mapping[str, Any] | None = None,
    update_window: int = UPDATE_WINDOW,
    records_file: str | Path | None = None,
    attempt_order: Mapping[tuple[str, Any], list[Any]] | None = None,
    follow: Any = None,
    identity_checked: bool = False,
) -> dict[str, Any]:
    """`build_evidence`, with the states of the detectors that read every record kept in `memo` when
    one is given (`watch`), and else in a state made for this read (lane AD11). With the Rust
    detectors, `follow` (a `server_detect.FollowedDetect`, `watch`) keeps the binary's detectors
    between looks and judges only what was appended; without it, `server_report` judges every record.
    `identity_checked` says the records handed in were checked already (`watch`: its `_Tail` checks
    each as it reads it), so they are copied and not checked again."""
    # "auto" becomes one of the two here. One frame deeper than before lane AD11 put `build_evidence`
    # in front of this function, so the warning still names the caller's file (wave AD's seam).
    detectors = _detectors_option(detectors, server_binary, stacklevel=4)
    # Both detectors judge by one contract, the Rust one's: refused here before anything is read.
    check_thresholds(overload_threshold=overload_threshold, drift_tolerance=drift_tolerance,
                     starve_window=starve_window, starve_ratio=starve_ratio, top_k=top_k,
                     update_window=update_window, none_given=True)
    held_ranks: list[int] | None = None
    # The training records alone are judged: no engine judges a scalar-only record, and none is a
    # step, so the evidence is the training records' (WAVE_AL_CONTRACT.md section 4.6, Q4 a). A file's
    # or a list's scalar-only records are read too, and set apart only once the records are known to
    # be of one run, theirs included, as `scalar_series` and the MCP series check them (BUG-29). A
    # store's `scalars-*` objects are not read for it: a store's rank folder is of one run.
    whole = not (isinstance(source, (str, Path)) and Path(source).is_dir())
    if rank is not None and whole:
        # A file or list is read whole either way: the ranks it holds are kept, so the evidence
        # can say it judged one of several (L21). A store's rank folder holds one rank.
        everything, left_out = _load_counted(source, None, checked=identity_checked, scalars=True)
        held_ranks = sorted({int(r.get("rank", 0)) for r in everything if not is_scalar_record(r)})
        own = [r for r in everything if r.get("rank", 0) == rank]
    else:
        own, left_out = _load_counted(source, rank, checked=identity_checked, scalars=whole)
    every_attempt, scalar_only = set_apart_scalars(own)
    if attempt_order:
        every_attempt = in_attempt_order(every_attempt, attempt_order)
    ranks = sorted({int(r.get("rank", 0)) for r in every_attempt})
    if len(ranks) > 1:
        raise ValueError(f"these records come from ranks {ranks}; pass rank= to read one of them")
    # The judged rank's scalar-only records: another rank's are that rank's.
    check_one_run([*every_attempt, *(r for r in scalar_only if not ranks or r.get("rank", 0) == ranks[0])])
    # Lane W3's step-order rule: the effective run is judged in step order, rows of one step in
    # the order they arrived (a stable sort). Arrival order stays in `every_attempt`, for the
    # analyses that are about arrival (which attempt came first, repeats, resumes).
    records = sorted(resolve_attempts(every_attempt), key=lambda r: int(r["step"]))
    router_sizes = _router_sizes(records)
    routers = list(router_sizes)
    expert_count = max(router_sizes.values(), default=0)
    router_top_k = _router_top_k(records, routers, top_k)
    samples: list[dict[str, Any]]
    incidents: list[dict[str, Any]]
    not_checked: list[dict[str, Any]]
    rank_checks: dict[str, dict[str, Any]]
    from_rust: list[dict[str, Any]] | None = None
    tensor_entries: list[dict[str, Any]] = []  # each judged resume's `tensors` (`checkpoint_tensors`)
    if detectors == "rust" and every_attempt:
        from .server_detect import server_report

        asked = {
            "overload_threshold": overload_threshold, "drift_tolerance": drift_tolerance,
            "starve_window": starve_window, "starve_ratio": starve_ratio, "top_k": _top_k_option(top_k),
            "update_window": update_window,
        }
        report = (follow.report(every_attempt, thresholds=asked) if follow is not None
                  else server_report(every_attempt, thresholds=asked, binary=server_binary, samples=True))
        absent = [kind for kind in SERVER_KINDS if kind not in (report.get("checked") or ())]
        if absent:
            raise RuntimeError(
                f"this tensorquill-server does not check {', '.join(absent)} (an older one): build it from the same version"
            )
        applied = report.get("thresholds") or {}
        expected = {**asked, "spike_sustain": SPIKE_SUSTAIN}
        differing = [name for name, value in expected.items() if not _same_threshold(value, applied.get(name))]
        if differing:
            raise RuntimeError(
                "this tensorquill-server applies other thresholds than were asked for ("
                + ", ".join(f"{name} {applied.get(name)!r}, here {expected[name]!r}" for name in differing)
                + "): build the server from the same version"
            )
        # The samples are judged there too, by the rules that made the incidents.
        samples = report["samples"]
        named = [(s.get("layer"), s.get("router")) for s in samples[:len(routers)]]
        if len(samples) != len(records) * len(routers) or named != list(enumerate(routers, 1)):
            raise RuntimeError(
                f"this tensorquill-server gave {len(samples)} MoE samples, for the routers"
                f" {[name for _, name in named]}, where these {len(records)} records have the routers {routers}:"
                " build the server from the same version"
            )
        rank_checks = report.get("kind_checks")
        if not isinstance(rank_checks, dict) or any(
            not isinstance(rank_checks.get(kind), dict) or rank_checks[kind].get("value") not in KIND_CHECK_VALUES
            for kind in SERVER_KINDS
        ):
            raise RuntimeError(
                "this tensorquill-server does not say what was checked (`kind_checks`, an older one): build it"
                " from the same version"
            )
        if any(not isinstance(rank_checks[kind].get("enough_history"), bool) for kind in SERVER_KINDS):
            raise RuntimeError(
                "this tensorquill-server does not say whether each kind had enough history (`enough_history` in"
                " `kind_checks`, an older one): build it from the same version"
            )
        rank_checks = {kind: dict(rank_checks[kind]) for kind in SERVER_KINDS}
        from_rust = [dict(incident) for incident in report["incidents"]]
        incidents = [i for i in from_rust if i.get("kind") != "replica_drift"]
        # The server lists these two as not checked in every report, since it never has their
        # files. Here they are checked when asked for.
        not_checked = [
            dict(item) for item in report.get("not_checked") or ()
            if isinstance(item, dict) and item.get("check") not in ("flight_recorder", "traincheck")
        ]
        # Unread gradients (`grads_not_read`, L23) are the server's to decide here: it writes the
        # `gradients` entry, makes its kinds `unavailable` and names `parameters_not_judged`, as it
        # names `modules_not_judged` and `starting_state`. Before wave AF a second pass here added
        # Python's entry to a report that had none, for a server that did not read the key yet
        # (WAVE_AF_CONTRACT.md, section 2, "the stale post-pass"): one rule, the server's.
    else:
        # The detectors that read every record are kept in states (`_Incremental`): `watch` keeps one
        # between looks and feeds it each look's new records; a single read starts one here.
        judged = (memo if memo is not None else _Incremental()).judge(
            records, router_sizes=router_sizes, router_top_k=router_top_k, overload_threshold=overload_threshold,
            starve_window=starve_window, starve_ratio=starve_ratio,
            options=(drift_tolerance, _top_k_option(top_k), update_window), top_k=top_k)
        samples = judged["samples"]
        incidents = []
        moe_hit = judged["moe_hit"]
        if moe_hit is not None:
            incidents.append(
                {
                    "kind": "moe_imbalance",
                    "step": moe_hit["step"],
                    "layer": moe_hit["layer"],
                    "router": moe_hit.get("router", f"layer-{moe_hit['layer']:02}"),
                    "diversity": moe_hit["diversity"],
                    "overloaded": moe_hit["overloaded"],
                    "dead": moe_hit["dead"],
                    # A router collapsed since its first step: when it began is `step`, when it
                    # became reportable is `reportable_step` (rules contract, part 3).
                    **{key: moe_hit[key] for key in ("since_start", "reportable_step") if key in moe_hit},
                }
            )
        starved_hit = judged["starved_hit"]
        if starved_hit is not None:
            incidents.append(starved_hit)
        update_judged = _update_spikes_judged(records, window=update_window)
        for detector in (
            lambda found: judged["grad_hit"], lambda found: update_judged[0],
            detect_stalled_weights, detect_repeated_steps, detect_loss_scale_trouble,
        ):
            # Only this detector looks at what a later attempt replaced.
            detector_records = every_attempt if detector is detect_repeated_steps else records
            hit = detector(detector_records)
            if hit is not None:
                incidents.append(hit)
        not_checked = []
        resume_hit = detect_resume_lr_changes(every_attempt, not_checked=not_checked)
        if resume_hit is not None:
            incidents.append(resume_hit)
        incidents.extend(hit for hit in detect_resume_state(
            every_attempt, not_checked=not_checked, tensors=tensor_entries) if hit is not None)
        for data_detector in (detect_data_stalled, detect_epoch_order_repeated):
            data_hit = data_detector(records)
            if data_hit is not None:
                incidents.append(data_hit)
        incidents.extend(judged["activation"])
        incidents.extend(detect_attention_anomalies(records, not_checked=not_checked))
        non_finite_hit = judged["non_finite_hit"]
        if non_finite_hit is not None:
            incidents.append(non_finite_hit)
        incidents.extend(detect_system_incidents(records, not_checked))
        refused = refused_routers_not_checked(records)
        if refused is not None:
            not_checked.append(refused)
        # Routers recorded with no tokens on any step (section 8): right after the refused ones.
        unread_routers = routers_not_read_not_checked(records)
        if unread_routers is not None:
            not_checked.append(unread_routers)
        not_read = update_not_read_not_checked(records)
        if not_read is not None:
            not_checked.append(not_read)
        # Right after `update_size`, the other entry of `update_spike`, and before `gradients`: where
        # the server puts it (WAVE_AE_CONTRACT.md, `clip_groups_beside_other_entries`).
        clip_groups = update_spike_not_checked(update_judged[1])
        if clip_groups is not None:
            not_checked.append(clip_groups)
        grads_unread = grads_not_read_not_checked(records)
        if grads_unread is not None:
            not_checked.append(grads_unread)
        layers_unread = layers_not_read_not_checked(records)
        if layers_unread is not None:
            not_checked.append(layers_unread)
        rank_checks = kind_checks(
            input_counts(records), not_checked, starve_window=starve_window, update_window=update_window)
        _name_starting_state(rank_checks, judged["starting"])
        # The Python detectors' own: under the Rust ones the server's report names them (above).
        name_gradients_not_judged(rank_checks, not_checked)
        name_modules_not_judged(rank_checks, not_checked)
    status = read_watchdog(watchdog if watchdog is not None else _watchdog_beside(source))
    # A status is expected where a server stores the run (a directory of its parts, or one
    # part) or when one was named. A local file or a list of records has no server to miss.
    status_expected = watchdog is not None or (
        isinstance(source, (str, Path)) and (Path(source).is_dir() or Path(source).suffix == ".parquet")
    )
    stream_incidents = detect_watchdog_incidents(status)
    incidents.extend(stream_incidents)
    last_step = max((int(r["step"]) for r in records), default=0)
    def newest_ts() -> float | None:
        return _newest_stamp(records)

    traincheck_summary, recorder_summary = _check_other_sources(
        incidents, stream_incidents, not_checked, last_step=last_step, newest_ts=newest_ts,
        traincheck=traincheck, flight_recorder=flight_recorder, flight_recorder_prefix=flight_recorder_prefix,
    )
    if from_rust is not None:
        drift_hit = next((i for i in from_rust if i.get("kind") == "replica_drift"), None)
    else:
        drift_hit = _replica_drift(records, drift_tolerance)
    if drift_hit is not None:
        incidents.append(drift_hit)
    # The ranks' replica observations (SDK-22). Every record as written, in attempt order: a rank's
    # later attempt under one launch is set aside, which the rollback would hide. One rank's
    # records compare with no other rank, so they raise nothing here and the kind's entry says
    # which ranks were not in the records read; the gathered spreads above are judged as before.
    replica: dict[str, Any] | None = None
    if any(_observation(r) is not None for r in every_attempt):
        result, replica_details = _replica_judged(every_attempt, drift_tolerance, None)
        replica = _replica_object(result, drift_tolerance)
        observed_hit = _replica_incident(replica, replica_details)
        if observed_hit is not None:
            incidents.append(observed_hit)
        rank_checks["replica_drift"] = replica_kind_check(
            rank_checks["replica_drift"], replica, all_ranks=False,
            records=sum(_observation(r) is not None or _held_spread_table(r) is not None for r in records),
            held_ranks=held_ranks or ())

    if any(record.get("scalars") for record in records):
        rust_made = {id(incident) for incident in from_rust or ()}  # these carry their scalars already
        for incident in incidents:
            if id(incident) in rust_made:
                continue
            context = scalar_context(records, incident["step"])
            if context:
                incident["scalars"] = context
    with_classes(incidents)

    run_id = records[0].get("run_id") if records else None
    trends = weight_norm_trends(records)
    head_trends = head_logit_trends(records)
    memory = latest_memory(records)
    evidence = {
        "schema_version": 1,
        "run_id": run_id,
        # The rank judged (lane U7, Nov 19, 2026): a reader that holds records of several ranks
        # beside this file (`mcp_server`) uses its samples only for this rank.
        "rank": ranks[0] if ranks else rank,
        # The ranks the file or list held when it held others than the one judged (L21): its
        # evidence is that one rank's, and the others are not judged in it.
        **({"ranks_in_source": held_ranks} if held_ranks is not None and held_ranks != [rank] else {}),
        "run": _manifest.run_block(run_id, _manifest.find_manifest(
            run_manifest if run_manifest is not None else source, run_id)),
        "layer_count": len(routers),
        "expert_count": expert_count,
        "router_names": routers,
        # How each router's counts reached the capture (rules contract, part 7), where it said.
        **({"moe_source": sourced} if (sourced := moe_source(records)) else {}),
        "overload_threshold": overload_threshold,
        "top_k": router_top_k,
        "starve_window": starve_window,
        "starve_ratio": starve_ratio,
        # The six the detectors ran with, under the name a server's report lists them by, and
        # only when one is not the default: evidence at the defaults is what it was before.
        **({"thresholds": ran_with} if _off_default(ran_with := {
            "overload_threshold": overload_threshold, "drift_tolerance": drift_tolerance,
            "starve_window": starve_window, "starve_ratio": starve_ratio, "top_k": _top_k_option(top_k),
            "update_window": update_window,
        }) else {}),
        "incidents": incidents,
        "samples": samples,
        **({"detectors": "rust"} if from_rust is not None else {}),
        # The ranks' replica observations compared (`compare_replicas`), only where a record holds one.
        **({"replica": replica} if replica is not None else {}),
        **({"run_end": read_end_marker(ended)} if (ended := _records_file(source, records_file)) else {}),
        # Another process's file of the same run ID and rank beside it (`sink.writers_block`): this
        # file holds one of the rank's writers, not the rank's whole.
        **({"not_checked": not_checked} if not_checked else {}),
        # Skipped updates by who reported them: the server's, else counted here (contract, section 1).
        **({"skipped_updates": skipped} if (skipped := (
            dict(report["skipped_updates"]) if from_rust is not None and isinstance(report.get("skipped_updates"), Mapping)
            else skipped_updates(records))) else {}),
        # What each judged resume's load said of the checkpoint's tensors (CHECKPOINT_TENSOR_CONTRACT.md,
        # section 4): the server's block, else built here; only where a load recorded `tensors`.
        **({"checkpoint_tensors": tensors_block} if (tensors_block := (
            (dict(served) if isinstance(served := report.get("checkpoint_tensors"), Mapping) else None)
            if from_rust is not None else checkpoint_tensors_block(tensor_entries))) else {}),
        **({"lines_left_out": left_out} if left_out else {}),
        # Rows of an attempt after a later attempt's first row (contract 1): kept, not progress.
        **({"superseded_rows": late} if (late := superseded_rows(every_attempt)) else {}),
        **({"weight_trends": trends} if trends else {}),
        # The four measurements (rules contract, part 6), made here from the records under either
        # detectors, as `weight_trends` is.
        **measurement_blocks(records),
        **({"head_trends": head_trends} if head_trends else {}),
        **({"memory": memory} if memory else {}),
        **({"flight_recorder": recorder_summary} if recorder_summary else {}),
        **({"traincheck": traincheck_summary} if traincheck_summary else {}),
        **({"watchdog": {k: status[k] for k in (
            "summary", "written_at_ms", "world_size", "pauses", "recommended_silent_seconds",
            "silent_floor_seconds", "critical_seconds",
        ) if k in status}} if status else {}),
    }
    others = other_kind_checks(
        status=status, status_expected=status_expected, not_checked=not_checked,
        flight_recorder=flight_recorder is not None, traincheck=traincheck is not None,
    )
    # Every kind, in the order `INCIDENT_CLASSES` lists them, none called not configured beside its incident.
    evidence["kind_checks"] = checked_where_reported(
        {kind: {**rank_checks, **others}[kind] for kind in INCIDENT_CLASSES}, incidents)
    evidence["run_state"] = _per_rank_run_state(evidence, status, _status_age(status), steps_judged=bool(records),
                                                attempt=newest_attempt(records))
    ended = _records_file(source, records_file)
    return with_writers(evidence, writers_block(ended) if ended else None)


def with_writers(evidence: dict[str, Any], writers: Mapping[str, Any] | None) -> dict[str, Any]:
    """`evidence` of one local records file with `writers` (`sink.writers_block`) when another
    process wrote the same run ID and rank in a file of its own beside it. The file then holds one
    of the rank's writers, not the rank's whole, so a run state that would read healthy or degraded
    reads `unknown`, with the writers as its reason (code `several_writers`); an unhealthy one
    stays so, with the reason added. With no `writers`, `evidence` as it is."""
    if not writers:
        return evidence
    found = dict(evidence.get("run_state") or {})
    state = found.get("state") if found.get("state") in RUN_STATES else "unknown"
    if RUN_STATES.index(state) < RUN_STATES.index("unknown"):
        state = "unknown"
    reason = _reason("unknown", "several_writers", str(writers.get("detail") or "two processes wrote this rank"))
    return {**evidence, "writers": dict(writers),
            "run_state": {**found, "state": state, "reasons": [*(found.get("reasons") or ()), reason]}}


def _records_file(source: Any, records_file: Any) -> Path | None:
    """The local `.jsonl` file whose end marker the evidence reports, if there is one."""
    for candidate in (records_file, source):
        if isinstance(candidate, (str, Path)) and Path(candidate).suffix == ".jsonl":
            return Path(candidate)
    return None


def run_end_line(evidence: Mapping[str, Any]) -> str | None:
    """One line on whether the session that wrote the local file closed it, or None."""
    end = evidence.get("run_end")
    if not isinstance(end, Mapping):
        return None
    if end.get("state") == "closed":
        when = end.get("closed_at")
        at = (" at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(when))
              if isinstance(when, (int, float)) and not isinstance(when, bool) else "")
        if end.get("outcome") == "raised":
            return f"Run end: the session was closed{at} by an exception ({end.get('exception', 'unknown')})."
        return f"Run end: the session closed the file{at}."
    return f"Run end: not known to have finished: {end.get('detail', end.get('state'))}."


# Samples a router may stay flagged in the state it started in, from step 0, before it is
# reported (DETECTOR_RULES_CONTRACT.md, part 3).
MOE_START_GRACE = 10


def _routed_by_router(samples: Iterable[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    """Each router's samples that routed tokens, in step order, routers in their layer's order.
    A missing sample is not one of them: a router's first sample is its first that routed."""
    by_layer: dict[Any, list[Mapping[str, Any]]] = {}
    for sample in samples:
        if isinstance(sample, Mapping) and not sample.get("missing") and sample.get("routed_tokens", 0) > 0:
            by_layer.setdefault(sample.get("layer"), []).append(sample)
    return [sorted(seq, key=lambda s: s["step"]) for _, seq in sorted(
        by_layer.items(), key=lambda item: (0, item[0], "") if isinstance(item[0], int) else (1, 0, repr(item[0])))]


def _flagged_sample(sample: Mapping[str, Any]) -> bool:
    return bool(sample.get("overloaded") or sample.get("dead"))


def _leading_start(seq: list[Mapping[str, Any]]) -> int:
    """How many of a router's first samples are flagged in a row, when its first is at step 0:
    the state it started in. 0 for a router whose first sample is at another step."""
    if not seq or seq[0]["step"] != 0:
        return 0
    leading = 0
    while leading < len(seq) and _flagged_sample(seq[leading]):
        leading += 1
    return leading


def first_incident(evidence: Mapping[str, Any]) -> dict[str, Any] | None:
    """The MoE sample the `moe_imbalance` incident reports, or None (rules contract, part 3).

    The samples are as they were: each says which experts are overloaded or dead. A router whose
    first sample is at step 0 and is flagged is in the state it started in, and that leading run
    of flagged samples is not reported while it is shorter than MOE_START_GRACE samples. Once it
    reaches that many it is: the sample returned is the first of them, with `since_start` true
    and `reportable_step`, the step of the MOE_START_GRACE-th. When it began and when it became
    reportable are two facts. Any flagged sample after the leading run ends is reported as
    before, and a router whose first sample is at another step is judged as before.

    Among routers the one that could be reported first wins: by `reportable_step` for a router
    collapsed since its start, by its step otherwise, then by layer. The sample is a copy; the
    evidence's samples are not changed.
    """
    found: list[tuple[int, Any, dict[str, Any]]] = []
    for seq in _routed_by_router(evidence.get("samples", ())):
        leading = _leading_start(seq)
        if leading >= MOE_START_GRACE:
            reportable = int(seq[MOE_START_GRACE - 1]["step"])
            found.append((reportable, seq[0]["layer"],
                          {**seq[0], "since_start": True, "reportable_step": reportable}))
            continue
        later = next((sample for sample in seq[leading:] if _flagged_sample(sample)), None)
        if later is not None:
            found.append((int(later["step"]), later["layer"], dict(later)))
    if not found:
        return None
    return min(found, key=lambda item: (item[0], item[1]))[2]


def name_starting_state(checks: dict[str, dict[str, Any]], samples: Iterable[Mapping[str, Any]]) -> None:
    """Name the routers still in their starting state (`moe_starting_state`) on the
    `moe_imbalance` entry of `checks`, as `starting_state`, with `enough_history` false: a
    short capture of a router flagged since step 0 reads as checked without enough history, not
    as checked and clean (rules contract, part 3). Not on an entry that is `unavailable`, as
    `parameters_not_judged` is not."""
    _name_starting_state(checks, moe_starting_state(samples))


def _name_starting_state(checks: dict[str, dict[str, Any]], starting: list[dict[str, Any]]) -> None:
    """`name_starting_state` with the routers still in their starting state already found."""
    check = checks.get("moe_imbalance")
    if starting and isinstance(check, dict) and check.get("value") != "unavailable":
        check["starting_state"] = starting
        check["enough_history"] = False


def moe_starting_state(samples: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The routers still in the state they started in when the samples end: flagged on every
    sample since step 0, and fewer than MOE_START_GRACE of them, so nothing is reported for
    them yet. `{"router", "samples"}` each, in layer order (the `starting_state` of the
    `moe_imbalance` entry of `kind_checks`)."""
    return [{"router": seq[0].get("router", f"layer-{seq[0]['layer']:02}"), "samples": len(seq)}
            for seq in _routed_by_router(samples)
            if _leading_start(seq) == len(seq) < MOE_START_GRACE]


class _RouterStarts:
    """`first_incident` and `moe_starting_state` of a run's samples, fed in step order (lane AD11): for
    each router, its first routed sample, how many it has had, the leading run of flagged samples
    while its first is at step 0 (`_leading_start`), the MOE_START_GRACE-th of them, and the first
    flagged sample after that run. `first` and `starting` give what the two functions give on every
    sample fed so far; `build_evidence` reads them here, and the two functions stay for any evidence."""

    def __init__(self) -> None:
        # layer -> [first sample, routed samples, leading run, still leading, grace sample, later flagged]
        self.routers: dict[Any, list[Any]] = {}

    def feed(self, samples: Iterable[Mapping[str, Any]]) -> None:
        routers = self.routers
        for sample in samples:
            if not (isinstance(sample, Mapping) and not sample.get("missing") and sample.get("routed_tokens", 0) > 0):
                continue
            state = routers.get(sample.get("layer"))
            if state is None:
                state = routers[sample.get("layer")] = [sample, 0, 0, sample["step"] == 0, None, None]
            state[1] += 1
            flagged = _flagged_sample(sample)
            if state[3]:
                if flagged:
                    state[2] += 1
                    if state[2] == MOE_START_GRACE:
                        state[4] = sample
                    continue
                state[3] = False
            if flagged and state[5] is None:
                state[5] = sample

    def _in_order(self) -> list[list[Any]]:
        """The routers in their layer's order, as `_routed_by_router` gives them."""
        return [state for _, state in sorted(
            self.routers.items(), key=lambda item: (0, item[0], "") if isinstance(item[0], int) else (1, 0, repr(item[0])))]

    def first(self) -> dict[str, Any] | None:
        found: list[tuple[int, Any, dict[str, Any]]] = []
        for first, _, leading, _, grace, later in self._in_order():
            if leading >= MOE_START_GRACE:
                reportable = int(grace["step"])
                found.append((reportable, first["layer"], {**first, "since_start": True, "reportable_step": reportable}))
            elif later is not None:
                found.append((int(later["step"]), later["layer"], dict(later)))
        if not found:
            return None
        return copy.deepcopy(min(found, key=lambda item: (item[0], item[1]))[2])

    def starting(self) -> list[dict[str, Any]]:
        return [{"router": first.get("router", f"layer-{first['layer']:02}"), "samples": count}
                for first, count, leading, _, _, _ in self._in_order() if leading == count < MOE_START_GRACE]


def _missing_sample(step: int, layer: int, expert_count: int) -> dict[str, Any]:
    return {
        "step": step,
        "layer": layer,
        "routed_tokens": 0,
        "counts": [0] * expert_count,
        "diversity": None,
        "overloaded": [],
        "dead": [],
        "missing": True,
    }


def expert_load_table(evidence: Mapping[str, Any], step: int, layer: int) -> str:
    """Markdown table of one layer's expert load, sorted by largest positive delta first.

    The step is compared with the last earlier one on which the router routed tokens, or else
    the last earlier one. At the layer's first step there is none: the heading says so, every
    delta is `n/a` and the rows are sorted by load (DET-14; until Nov 19, 2026 the first step
    was compared with itself, every delta +0.00). In compact evidence that left out the step it
    is compared with, the heading says so and the deltas are `n/a` the same way (DET-13).

    Raises `compact.SamplesOmitted` for a step that compact evidence left out."""
    from .compact import require_table

    compared_here = require_table(evidence, step, layer)
    lookup = _TABLE_LOOKUP.get()
    if lookup is None or lookup.evidence is not evidence:
        lookup = _SampleLookup(evidence)  # for this table alone
    return _expert_load_text(evidence, lookup.samples, step, layer, lookup.previous(step, layer), compared_here)


class _SampleLookup:
    """The samples of one evidence by `(step, layer)` (the last of a key wins), and each layer's steps
    in order, all of them and those on which the router routed tokens: what `expert_load_table`
    reads. `export_rerun` makes one for the whole recording and puts it in `_TABLE_LOOKUP` while it
    tables every step (lane AD11: each table made it again, so the export was quadratic in the steps)."""

    def __init__(self, evidence: Mapping[str, Any]) -> None:
        self.evidence = evidence
        self.samples = {(int(s["step"]), int(s["layer"])): s for s in evidence.get("samples", ())}
        steps: dict[Any, list[int]] = {}
        for step, layer in self.samples:
            steps.setdefault(layer, []).append(step)
        self.steps = {layer: sorted(found) for layer, found in steps.items()}
        self.routed = {
            layer: [s for s in found if not self.samples[s, layer].get("missing")
                    and self.samples[s, layer].get("routed_tokens", 0) > 0]
            for layer, found in self.steps.items()
        }

    def previous(self, step: int, layer: int) -> int | None:
        """The step `step` is compared with: the last earlier one of the layer on which the router routed
        tokens, else the last earlier one, else None (the layer's first step: nothing to compare it with)."""
        for found in (self.routed.get(layer, ()), self.steps.get(layer, ())):
            index = bisect.bisect_left(found, step)
            if index:
                return found[index - 1]
        return None


# The lookup `export_rerun` made for the evidence it is recording, while it tables its steps.
_TABLE_LOOKUP: contextvars.ContextVar[_SampleLookup | None] = contextvars.ContextVar("tensorquill_table_lookup", default=None)


def _expert_load_text(
    evidence: Mapping[str, Any],
    samples: Mapping[tuple[int, int], Mapping[str, Any]],
    step: int,
    layer: int,
    previous: int | None,
    compared_here: bool,
) -> str:
    """`expert_load_table`'s text, from the samples by `(step, layer)` and the step it is compared
    with (`previous`, None at the layer's first step), as `_SampleLookup` finds them."""
    expert_count = int(evidence.get("expert_count", 0))
    sample = samples.get((step, layer)) or _missing_sample(step, layer, expert_count)
    if not compared_here:
        previous = None  # left out of a compact file: never an earlier window's step instead
    baseline = samples.get((previous, layer)) if previous is not None else None

    tokens = int(sample.get("routed_tokens") or 0)
    if sample.get("missing") or tokens <= 0:
        n_experts = len(sample.get("counts", ())) or expert_count
        rows = [
            (0.0, expert, f"| #{expert:03} | n/a | n/a | Missing |")
            for expert in range(1, n_experts + 1)
        ]
    else:
        base_tokens = int(baseline.get("routed_tokens") or 0) if baseline is not None else 0
        base_counts = baseline.get("counts", ()) if baseline is not None else ()
        rows = []
        for expert, count in enumerate(sample["counts"], 1):
            share = count / tokens
            if expert in sample.get("dead", ()):
                status = "Dead"
            elif expert in sample.get("overloaded", ()):
                status = "Overloaded"
            elif expert in sample.get("starved", ()):
                status = "Starved"
            elif count == 0:
                status = "Idle"
            else:
                status = "Active"
            if base_tokens > 0 and expert - 1 < len(base_counts):
                delta = share - (base_counts[expert - 1] / base_tokens)
                sort_key = -delta
                delta_str = f"{delta * 100:+.2f}"
            else:
                sort_key = -share
                delta_str = "n/a"
            rows.append(
                (sort_key, expert, f"| #{expert:03} | {share * 100:.2f}% | {delta_str} | {status} |")
            )
        rows.sort(key=lambda row: row[:2])
    return "\n".join(
        [
            f"**Layer {layer:02} expert load** · step {step} "
            + (f"vs. {previous}" if previous is not None
               else "(the step it is compared with was left out of this compact file)" if not compared_here
               else "(no earlier step to compare with)"),
            "",
            "| Expert | Load | Δ load (pp) | Status |",
            "| --- | --- | --- | --- |",
            *(row[2] for row in rows),
        ]
    )


def _percent(share: float) -> str:
    """36% stays "36%", while 12.5% is not rounded to "12%"."""
    pct = share * 100
    return f"{pct:.0f}%" if abs(pct - round(pct)) < 0.05 else f"{pct:.1f}%"


def _starved_line(evidence: Mapping[str, Any]) -> str | None:
    hit = next((i for i in evidence.get("incidents", ()) if i["kind"] == "starved_experts"), None)
    if hit is None:
        return None
    listed = hit["experts"][:8]
    text = ", ".join(
        f"Layer {e['layer']:02} expert {e['expert']} ({e['window_share'] * 100:.3f}%)" for e in listed
    )
    more = len(hit["experts"]) - len(listed)
    window = evidence.get("starve_window", STARVE_WINDOW)
    return (
        f"Starved experts (share over {window} steps, from step {hit['step']}): {text}"
        + (f", and {more} more" if more > 0 else "")
    )


def _legend(evidence: Mapping[str, Any]) -> str:
    window = evidence.get("starve_window", STARVE_WINDOW)
    return (
        "Heatmap: blue = active, amber = overloaded, red = dead,"
        f" violet = starved over {window} steps, gray = no tokens this step"
    )


def _how_stopped(rank: Mapping[str, Any]) -> str:
    state, beat = rank.get("state"), rank.get("heartbeat_age_seconds")
    if state == "lost":
        return "its connection dropped and did not return"
    if state == "reconnecting":
        # Listed as stopped only when it was stopped already when its stream dropped (`rank_is_stopped`).
        was = "frozen or gone, with no heartbeat" if rank.get("stopped_as") == "unresponsive" else "silent"
        return f"it was {was}, and its connection has dropped since it stopped and has not come back yet"
    if state == "unresponsive":
        return "its heartbeats stopped too, so the whole process is frozen or gone"
    if beat is None:
        return "no heartbeat, so the phase is unknown and evaluation or a checkpoint look the same"
    return f"its heartbeats continue and it says it is in phase `{rank.get('phase', 'train')}`"


def _where(frame: Mapping[str, Any]) -> str:
    name = str(frame.get("file", "?")).replace("\\", "/").rsplit("/", 1)[-1]
    return f"`{frame.get('function', '?')}` ({name}:{frame.get('line', '?')})"


def _describe_stuck(incident: Mapping[str, Any]) -> str:
    """One sentence on where the stopped ranks' main threads were, ranks at the same place together."""
    places = incident.get("stuck_at") or ()
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for place in places:
        groups.setdefault((place.get("function"), place.get("file"), place.get("line")), []).append(place)
    parts = []
    for members in list(groups.values())[:2]:
        first = members[0]
        ranks = ", ".join(str(m.get("rank")) for m in members[:8])
        text = f"rank{'s' if len(members) != 1 else ''} {ranks}: main thread in {_where(first)}"
        if first.get("caller"):
            text += f", called from {_where(first['caller'])}"
        if first.get("in_tensorquill"):
            text += f", reached through TensorQuill's {_where(first['in_tensorquill'])}"
        held = min((m["unchanged_seconds"] for m in members if isinstance(m.get("unchanged_seconds"), (int, float))),
                   default=None)
        if held is not None:
            text += f", unchanged for {held:.0f} s"
        parts.append(text)
    if not parts:
        return ""
    more = len(groups) - len(parts)
    return " Last stack: " + "; ".join(parts) + (f"; and {more} other place{'s' if more != 1 else ''}" if more > 0 else "") + "."


def _describe_stream(incident: Mapping[str, Any]) -> str:
    assessment = str(incident.get("assessment", "suspected_hang")).replace("_", " ")
    corroboration = incident.get("corroborated_by")
    tail = f" [{incident.get('severity', 'warning')}, {assessment}"
    if corroboration:
        tail += f", corroborated by {corroboration}"
    tail += "]"
    if corroboration == "collective_hang":
        tail = " PyTorch's Flight Recorder shows a stuck collective." + tail
    tail = _describe_stuck(incident) + _capture_stopped_text(incident) + tail
    started_again = [r.get("rank") for r in incident.get("ranks", ()) if r.get("attempt_without_steps")]
    if started_again:
        # The server saw a new attempt in the heartbeats (the job was started again) that has stored
        # no step yet: the step and the quiet time shown are the earlier attempt's.
        tail = (f" Rank{'s' if len(started_again) != 1 else ''} {', '.join(str(r) for r in started_again[:8])}"
                f" {'were' if len(started_again) != 1 else 'was'} started again (a new attempt) and"
                " stored no step of it yet; the step shown is the earlier attempt's." + tail)
    if incident["kind"] == "job_stalled":
        ranks = incident["ranks"]
        quiet = max((r.get("seconds_since_progress") or 0 for r in ranks), default=0)
        reasons = sorted({_how_stopped(r) for r in ranks})
        count = f"none of {len(ranks)} rank{'s' if len(ranks) != 1 else ''}"
        if ranks and all(r.get("last_step") is None for r in ranks):
            # Nothing was ever stored: the capture holds `every` steps before it writes any.
            worked = [r["work"] for r in ranks if isinstance(r.get("work"), int) and r["work"] > 0]
            return (
                f"Job stalled before its first batch was stored: {count} has sent a step in {quiet:.0f} s"
                + (f", and the heartbeats' work counter stands at {max(worked)}" if worked else "")
                + f" ({'; '.join(reasons)})."
                " A hang, a long startup (compilation, a data load), or a killed job can look the same from here." + tail
            )
        return (
            f"Job stalled after step {incident['step']}: {count}"
            f" has sent a new step for at least {quiet:.0f} s ({'; '.join(reasons)})."
            + (_lost_text(incident) if incident.get("assessment") == "lost_connection" else
               " A hang, a long evaluation or checkpoint pause, or a killed job can look the same from here.") + tail
        )
    parts = []
    for r in incident["ranks"][:4]:
        if r.get("last_step") is None:
            parts.append(f"rank {r['rank']} has had no step stored yet ({_how_stopped(r)})")
            continue
        parts.append(f"rank {r['rank']} sent no new step since step {r['last_step']} ({_how_stopped(r)})")
    if incident["missing_ranks"]:
        shown = incident["missing_ranks"][:8]
        omitted = incident.get("missing_ranks_omitted")
        more = (len(incident["missing_ranks"]) - len(shown) + omitted
                if isinstance(omitted, int) and not isinstance(omitted, bool) and omitted > 0 else 0)
        parts.append("ranks " + ", ".join(str(r) for r in shown) + (f" and {more} more" if more else "")
                     + " never connected (if only some ranks call tensorquill.init, ignore this)")
    return (
        f"Ranks out of step with the job: {'; '.join(parts)} while others reached step {incident['newest_step']}."
        + (_lost_text(incident) if incident.get("assessment") == "lost_connection" else
           " A hung rank, a killed process or a stuck telemetry sender look the same to the server.") + tail
    )


def _lost_text(incident: Mapping[str, Any]) -> str:
    """What a stall labelled `lost_connection` is, in place of the sentence that says a hang and a
    killed job look the same: the connection of a rank went and did not come back."""
    lost = [r for r in incident.get("lost_ranks") or () if r is not None]
    who = (f"rank{'s' if len(lost) != 1 else ''} {', '.join(str(r) for r in lost[:8])}"
           + (f" and {len(lost) - 8} more" if len(lost) > 8 else "")) if lost else "the stopped ranks"
    waiting = len(lost) < len([r for r in incident.get("ranks", ()) if rank_is_stopped(r)])
    return (
        f" The connection of {who} dropped and did not come back: read as a lost connection, not a hang."
        " The process was most likely killed or crashed (an out-of-memory kill looks like this), or the network to the"
        " server failed; look at the job's own logs and exit status first."
        + (" The ranks still connected are most likely waiting for it." if waiting else "")
    )


def _describe_attention(incident: Mapping[str, Any]) -> str:
    """An attention incident, with how few probes it rests on."""
    kind, top = incident["kind"], incident["modules"][0]
    more = f", and {incident['affected'] - 1} more modules" if incident["affected"] > 1 else ""
    if kind == "attention_collapse":
        what = (f"Attention entropy collapse at step {incident['step']} in {top['module']}: mean entropy share"
                f" {top['value']:.2f}, {top['ratio']:.2f}× its earlier probes ({top['baseline']:.2f}){_with_score(top)}")
    elif kind == "attention_uniform":
        what = (f"Attention gone uniform at step {incident['step']} in {top['module']}: mean entropy share"
                f" {top['baseline']:.2f} → {top['value']:.2f} (1.0 is uniform over every key)")
    else:
        what = (f"Attention logit growth at step {incident['step']} in {top['module']}: largest logit"
                f" {top['value']:.3g}, {top['ratio']:.2g}× its earlier probes ({top['baseline']:.3g}){_with_score(top)}")
    probes = incident["affected_probes"]
    return (
        f"{what}{more}; held on the probe at step {incident['confirmed_step']} too ({probes} probe"
        f"{'s' if probes != 1 else ''} affected, each layer judged against {top['baseline_probes']} earlier"
        " probe" + ("s" if top["baseline_probes"] != 1 else "") + " of the fixed probe batch)"
    )


# The words of a `resume_incomplete` finding on a load's tensors (CHECKPOINT_TENSOR_CONTRACT.md,
# section 4): singular and plural, then the names, at most five.
_TENSOR_WORDS = {
    "missing_tensors": ("trainable tensor", "trainable tensors", "not in the checkpoint"),
    "shape_differs": ("tensor", "tensors", "with another shape in the checkpoint"),
}
TENSORS_NAMED = 5


def _tensor_words(finding: Mapping[str, Any]) -> str | None:
    """ "2 trainable tensors not in the checkpoint: a, b"; past five names, "and 55 more"."""
    said = _TENSOR_WORDS.get(finding.get("finding"))  # type: ignore[arg-type]
    count = finding.get("count")
    if said is None or not isinstance(count, int) or isinstance(count, bool):
        return None
    names = [str(name) for name in finding.get("tensors") or ()][:TENSORS_NAMED]
    more = count - len(names)
    listed = ", ".join(names) + (f" and {more} more" if more > 0 and names else "")
    return f"{count} {said[0] if count == 1 else said[1]} {said[2]}" + (f": {listed}" if listed else "")


def _describe(incident: Mapping[str, Any]) -> str | None:
    kind = incident["kind"]
    if kind == "update_spike":
        where = ""
        if incident.get("parameters"):
            top = incident["parameters"][0]
            where = f", most in {top['name']} ({top['previous']:.3g} to {top['update_rms']:.3g})"
        count = incident.get("spike_count", 1)
        return (
            f"Update-size jump at step {incident['step']}: the largest RMS update moved {incident['change']:.3g}"
            f" from the step before ({incident['previous']:.3g} to {incident['update_rms']:.3g}), against a usual"
            f" {incident['baseline']:.3g} and a limit of {incident['threshold']:.3g} for a gradient norm that rose"
            f" {incident['grad_rise']:.3g}{where}; {count} step{'s' if count != 1 else ''} flagged"
        )
    if kind == "grad_spike":
        if incident.get("trigger") == "parameter":
            top = incident["parameters"][0]
            others = len(incident["parameters"]) - 1
            ratio = incident.get("ratio")
            total = (
                f"the total norm, {incident['grad_norm']:.3g}, was {ratio:.1f}× its median{_with_score(incident)}"
                if isinstance(ratio, (int, float)) else f"the total norm was {incident['grad_norm']:.3g}"
            )
            return (
                f"Gradient spike in one parameter from step {incident['step']}: {top['name']} at"
                f" {top['ratio']:.0f}× its own median{_with_score(top)}, and still at {SPIKE_FACTOR:g}× or more"
                f" {incident['sustained']} recorded steps in a row"
                + (f" ({others} more parameter{'s' if others > 1 else ''} from the same step)" if others > 0 else "")
                + f"; {total}, so the total alone did not show it"
            )
        where = ""
        if incident["parameters"]:
            top = incident["parameters"][0]
            where = (
                f", first in {top['name']} ({top['ratio']:.0f}× its median"
                + (f", {words}" if (words := score_words(top)) else "")
                + (f", rising since step {top['rising_since']}" if top["rising_since"] != incident["step"] else "")
                + ")"
            )
        more = incident["spike_count"] - 1
        return (
            f"Gradient spike at step {incident['step']}: norm {incident['grad_norm']:.3g},"
            f" {incident['ratio']:.0f}× the median of recent steps{_with_score(incident)}{where}"
            + (f"; {more} more spike{'s' if more > 1 else ''}" if more > 0 else "")
        )
    if kind == "stalled_weights":
        names = ", ".join(incident["parameters"][:3])
        extra = incident["stalled"] - min(3, incident["stalled"])
        optimizer = incident.get("optimizer") or {"state": "unknown", "reason": "not recorded by this capture"}
        dtypes = incident.get("weight_dtypes") or {}
        low = sorted(name for name in dtypes if name in ("bfloat16", "float16"))
        precision = f"the weights are {' and '.join(low)}, so check" if low and len(low) == len(dtypes) else "so check"
        if optimizer["state"] != "known":
            held = f"; whether an optimizer holds them is unknown ({optimizer['reason']})"
        elif optimizer["not_held_count"] == incident["stalled"]:
            held = "; no optimizer that stepped holds any of them, the likely cause"
        elif optimizer["not_held_count"]:
            held = f"; no optimizer holds {optimizer['not_held_count']} of them, the likely cause for those"
        else:
            held = (
                f"; an optimizer holds all of them, {precision} for updates too small for the weight"
                " dtype (bf16) or a zero learning rate"
            )
        onsets = incident.get("onsets") or []
        since = ""
        if onsets:
            trained = onsets[0].get("trained_before", 0)
            since = (
                "; none of them was seen changing at an earlier weight check" if trained == 0
                else f"; {trained} of them changed, or could not be compared, at an earlier weight check"
            )
        later = ""
        if len(onsets) > 1:
            parts = []
            for onset in onsets[1:4]:
                shown = ", ".join(onset["parameters"][:2])
                rest = onset["stalled"] - min(2, onset["stalled"])
                was = (
                    "changed or not compared at an earlier check" if onset["trained_before"] == onset["stalled"]
                    else "never seen changing" if onset["trained_before"] == 0
                    else f"{onset['trained_before']} changed or not compared at an earlier check"
                )
                parts.append(f"step {onset['step']}: {shown}{f', and {rest} more' if rest > 0 else ''} ({was})")
            more_onsets = len(onsets) - 4 + int(incident.get("onsets_omitted") or 0)
            later = (
                ". Stalled later, each parameter on its own history: " + "; ".join(parts)
                + (f"; and {more_onsets} more onset{'s' if more_onsets > 1 else ''}" if more_onsets > 0 else "")
            )
        return (
            f"Weights possibly not updating from step {incident['step']}: {incident['stalled']} of"
            f" {incident['with_gradients']} parameters with gradients had a float64 norm that did not change for"
            f" {incident['consecutive']} weight checks in a row ({names}{f', and {extra} more' if extra > 0 else ''})"
            f"{since}{held}; {incident['stalled_steps']} step{'s' if incident['stalled_steps'] != 1 else ''} affected"
            f"{later}. Equal norms do not prove the weights are equal"
        )
    if kind == "activation_spike":
        top = incident["modules"][0]
        more = ""
        if incident["affected"] > 1:
            # The first module in the model leads; the fault travels down the stack, so the
            # largest ratio is usually further on and is named too.
            largest = max(incident["modules"], key=lambda m: m["ratio"])
            more = f", and {incident['affected'] - 1} more modules"
            if largest["module"] != top["module"]:
                more += (f" (the largest, {largest['stat']} {largest['ratio']:.0f}×"
                         + (f", {words}" if (words := score_words(largest)) else "") + f", is in {largest['module']})")
        return (
            f"Activation spike at step {incident['step']} {'first ' if more else ''}in {top['module']}: {top['stat']}"
            f" {top['ratio']:.0f}× its recent median{_with_score(top)}{more}"
            f"; {incident['affected_steps']} step{'s' if incident['affected_steps'] != 1 else ''} affected"
        )
    if kind == "activation_drift":
        top = incident["modules"][0]
        direction = "grew" if top["ratio"] > 1 else "shrank"
        return (
            f"Activation drift from step {incident['step']} in {top['module']}: {top['stat']}"
            f" {direction} to {top['ratio']:.2g}× its level over the {ACT_WINDOW} records before its last {ACT_RECENT}"
            f"{_with_score(top)}"
            + (f", and {incident['affected'] - 1} more modules" if incident["affected"] > 1 else "")
        )
    if kind == "dead_activations":
        top = incident["modules"][0]
        return (
            f"Outputs going dead from step {incident['step']} in {top['module']}: zero fraction"
            f" {top['baseline']:.0%} → {top['recent']:.0%}"
            + (f", and {incident['affected'] - 1} more modules" if incident["affected"] > 1 else "")
        )
    if kind in ATTN_KINDS:
        return _describe_attention(incident)
    if kind == "repeated_steps":
        return (
            f"Repeated step numbers from step {incident['step']}: {incident['repeated']} steps"
            " recorded more than once; pass start_step= to tensorquill.init when resuming"
        )
    if kind in ("job_stalled", "rank_silent"):
        return _describe_stream(incident)
    if kind == "traincheck_violation":
        shown = incident["violations"][:3]
        names = "; ".join(
            f"{v['display_name']}" + (f" (steps {v['first_step']}-{v['last_step']})" if v.get("first_step") is not None else "")
            for v in shown
        )
        more = incident["distinct_invariants_violated"] - len(shown)
        return (
            f"TrainCheck reports {incident['distinct_invariants_violated']} violated invariant"
            f"{'s' if incident['distinct_invariants_violated'] != 1 else ''}, the first at its step {incident['step']}: "
            f"{names}{f'; and {more} more' if more > 0 else ''}. TrainCheck counts its own iterations, which match"
            " TensorQuill's steps only without gradient accumulation"
        )
    if kind in ("collective_hang", "collective_desync"):
        group = incident["group"]
        where = f"{incident['collective']} #{incident['collective_seq_id']} in group {group['desc'] or group['id']}"
        partial = " (dumps from only some ranks: ranks " + ", ".join(map(str, incident["dumped_ranks"])) + ")" if incident["incomplete_dumps"] else ""
        if kind == "collective_desync":
            return (
                f"Ranks issued different collectives at {where}: {incident['mismatch'] or 'see the analyzer output'}."
                f" The first of {incident['problems']} problem{'s' if incident['problems'] != 1 else ''} in the"
                f" Flight Recorder dumps, after step {incident['step']}{partial}"
            )
        if incident["missing_ranks"]:
            return (
                f"Collective hang at {where}: at dump time rank{'s' if len(incident['missing_ranks']) != 1 else ''}"
                f" {', '.join(map(str, incident['missing_ranks']))} had not entered it (state {incident['state']} on the"
                f" others), after step {incident['step']}. Look at what those ranks were doing instead: a stuck"
                f" data loader or CPU-side branch, an exception, or a dead process{partial}"
            )
        return (
            f"Collective hang at {where}: at dump time every rank had entered it but it had not completed (state {incident['state']}),"
            f" after step {incident['step']}. That points at the network, NCCL or a GPU, not at one rank{partial}"
        )
    if kind == "resume_lr_change":
        change = incident["changes"][0]
        shown = ", ".join(f"{v:.3g}" for v in change["resumed_lr"])
        wanted = ", ".join(f"{v:.3g}" for v in change["expected_lr"])
        how = "at the same step" if change["basis"] == "same_step" else "continuing its recent slope"
        return (
            f"Learning rate after the resume at step {incident['step']} is {shown}; the earlier attempt"
            f" had {wanted} {how}"
            + (", and the resumed value equals the run's first learning rate (as when warmup starts over)"
               if change["equals_first_lr"] else "")
            + ". Expected if the restart changed the schedule on purpose; the records cannot show the intent"
            + (f"; {incident['resumes']} resumes differ" if incident["resumes"] > 1 else "")
        )
    if kind == "resume_data_mismatch":
        change = incident["changes"][0]
        if change["finding"] == "differs":
            what = "a different batch than the earlier attempt drew at that step, one the run drew nowhere else"
        else:
            verb = "repeats the batch drawn" if change["finding"] == "repeated" else "skips ahead to the batch drawn"
            what = f"{verb} at step {change['matched_step']} ({change['batches_apart']} steps apart)"
        return (
            f"Data after the resume at step {incident['step']}: the resumed run draws {what}."
            " The data position was not restored, unless the restart started its epoch over on purpose"
            + (f"; {incident['resumes']} resumes differ" if incident["resumes"] > 1 else "")
        )
    if kind == "resume_incomplete":
        change = incident["changes"][0]
        words = {"missing": "not in the checkpoint", "not_identified": "state could not be identified",
                 "empty": "started empty", "empty_in_checkpoint": "empty in the checkpoint",
                 "step_reset": "step counter back at 1", "position": "last_epoch far from the step",
                 "initial_scale": "back at its initial value"}
        found = "; ".join(f"{f['component']} {_tensor_words(f) or words.get(f['finding'], f['finding'])}"
                          for f in change["findings"])
        source = {"load": " (from the loaded checkpoint's manifest and the first step)",
                  "save": f" (assuming it loaded the save after step {change.get('saved_after_step')})"}.get(change["source"], "")
        return (
            f"Resume at step {incident['step']} did not bring everything back{source}: {found}."
            " Expected only if left out or reset on purpose"
            + (f"; {incident['resumes']} resumes affected" if incident["resumes"] > 1 else "")
        )
    if kind == "data_stalled":
        why = ("drew no new batch" if incident["reason"] == "no_batch" else "drew the same batch again")
        return (
            f"No data progress from step {incident['step']}: steps advanced to {incident['through_step']} but each {why}"
        )
    if kind == "epoch_order_repeated":
        sampler = (
            f"; the sampler's epoch stayed at {incident['sampler_epoch']} (is set_epoch called each epoch?)"
            if incident.get("sampler_epoch") is not None and incident["sampler_epoch"] == incident.get("previous_sampler_epoch")
            else ""
        )
        return (
            f"Epoch {incident['epoch']} (from step {incident['step']}) draws its first {incident['batches_compared']}"
            f" batches in the same order as epoch {incident['previous_epoch']}{sampler}"
        )
    if kind == "duplicate_batches":
        return (
            f"Ranks {', '.join(map(str, incident['ranks']))} drew the same batch at step {incident['step']}"
            f" ({incident['duplicated_steps']} steps duplicated among ranks {_rank_ranges(incident['ranks_compared'])})."
            " Each rank should read its own share: a sampler without the rank, or an IterableDataset not split by rank or worker"
        )
    if kind == "loss_scale_trouble":
        why = (
            f"{incident['consecutive_skips']} consecutive scaler-confirmed skips among observations judged by this rule"
            if incident["reason"] == "skips"
            else f"loss scale fell to {incident['grad_scale']:g}"
        )
        partial = incident.get("partial_steps")
        return (
            f"Loss scaling in trouble at step {incident['step']}: {why}; "
            f"{incident['skipped_steps']} of {incident['recorded_steps']} steps skipped in all"
            + (f", and {PARTIAL_STEP_WORDS.format(n=partial)}" if partial else "")
        )
    if kind == "non_finite":
        parts = [*incident.get("layers", ()), *incident.get("gradients", ())]
        return f"NaN or Inf at step {incident['step']} in {', '.join(parts[:5])}"
    if kind == "replica_drift":
        bad = sum(1 for v in incident["parameters"].values() if v.get("non_finite"))
        text = (
            f"Replicas diverged at step {incident['step']} in"
            f" {len(incident['parameters'])} parameters (tolerance {incident['tolerance']:g})"
        )
        text += f", {bad} with NaN or Inf weights on some replica" if bad else ""
        if incident.get("source") == "replica":
            group = incident.get("group") or ()
            text += (f", from the observations of ranks {_rank_ranges(group, 8)} in launch"
                     f" {str(incident.get('launch'))[:8]}")
            text += _attribution_words(incident.get("attribution") or ())
            for other in incident.get("also") or ():
                text += f"; a gathered spread ({other.get('source')}) shows drift on that step too"
        return text
    return describe_system(incident)


_NO_RANK_NAMED = {
    "two_ranks": "two ranks that differ cannot say which is wrong",
    "no_majority": "no strict majority",
    "no_outlier": "no rank is further than the tolerance from the median",
}


def _attribution_words(attributed: Iterable[Mapping[str, Any]]) -> str:
    """What the incident's `attribution` says, by fingerprints: for each parameter the ranks that
    stand apart from a strict majority, or why none can be named. A named rank stands apart; it is
    not thereby the one at fault (contract 4a)."""
    parts = []
    for entry in attributed:
        if entry.get("verdict") == "attributed":
            ranks = list(entry.get("outliers") or ())
            parts.append(f"{entry.get('parameter')}: rank{'s' if len(ranks) != 1 else ''} {_rank_ranges(ranks, 8)}"
                         " apart from the majority")
        else:
            parts.append(f"{entry.get('parameter')}: no rank named ("
                         f"{_NO_RANK_NAMED.get(str(entry.get('why')), entry.get('why'))})")
    return f"; by fingerprints, {'; '.join(parts[:5])}" if parts else ""


def incident_summary(evidence: Mapping[str, Any]) -> str:
    """Human- and agent-readable summary matching the bake-off's text document format.

    Evidence from the all-ranks view (`"view": "all_ranks"`) gets the cross-rank summary."""
    if evidence.get("view") == ALL_RANKS_VIEW:
        return _all_ranks_summary(evidence)
    lines: list[str] = []
    incidents = evidence.get("incidents", ())
    moe_incident = next((i for i in incidents if i["kind"] == "moe_imbalance"), None)
    # The incident listed, with its sample's counts where the samples hold it; evidence built from
    # the server's report has the incident but no per-step samples. A compacted file keeps only
    # windows of samples, so the incident is read first, not worked out again from what was kept.
    if moe_incident is not None:
        sample = next((s for s in evidence.get("samples", ()) if isinstance(s, Mapping)
                       and s.get("step") == moe_incident.get("step") and s.get("layer") == moe_incident.get("layer")), None)
        moe: Mapping[str, Any] | None = {**sample, **moe_incident} if sample is not None else moe_incident
    else:
        moe = first_incident(evidence)
    if moe is not None:
        lines += _moe_lines(evidence, moe)
        source = moe_source_words(evidence).get(str(moe.get("router")))
        if source is not None:  # what the counts are, only where the capture said (rules contract, part 7)
            lines.append(f"  The counts of router {moe.get('router')} are {source}.")
        if moe_incident is not None:
            lines += _scalar_lines(moe_incident)
    starved = _starved_line(evidence)
    if starved is not None:
        lines.append(starved)
    for incident in incidents:
        if incident["kind"] not in ("moe_imbalance", "starved_experts"):
            text = _describe(incident)
            lines.append(text or f"Incident at step {incident['step']}: {incident['kind']}")
            lines += _scalar_lines(incident)
    classes = _class_line(incidents)
    if classes:
        lines.append(classes)
    pauses = (evidence.get("watchdog") or {}).get("pauses") or {}
    recommended = (evidence.get("watchdog") or {}).get("recommended_silent_seconds")
    if pauses and recommended:
        seen = ", ".join(f"{phase} up to {stats['max_seconds']:.0f} s ({stats['count']}x)" for phase, stats in sorted(pauses.items()))
        lines.append(
            f"Pauses the job reported: {seen}. A silence threshold of {recommended:.0f} s (twice the longest)"
            " would have tolerated them: set the server's --silent-seconds to it or more. This is a measurement of"
            " what happened so far, not a guarantee"
        )
    checked = checked_line(evidence)
    notes = [checked] if checked else []
    notes += [
        f"Not checked ({item['check']}): {item['reason']}"
        for item in evidence.get("not_checked", ())
        if item["check"] != "resume_lr"
    ]
    notes += _source_lines(evidence)
    applied = thresholds_line(evidence)  # only when a threshold applied is not the default
    if applied:
        notes.append(applied)
    ended = run_end_line(evidence)
    if ended:
        notes.append(ended)
    writers = evidence.get("writers")
    if isinstance(writers, Mapping) and writers.get("detail"):
        notes.append(f"Writers: {writers['detail']}.")
    left_out = lines_left_out_line(evidence)
    if left_out:
        notes.append(left_out)
    if not lines:
        return "\n".join(["No incidents detected.", *notes])
    if evidence.get("layer_count"):
        lines.append(_legend(evidence))
    return "\n".join([*lines, *notes])


def _group_line(group: Mapping[str, Any]) -> str:
    """One incident group as a sentence, from the first rank's incident."""
    incident = group["incident"]
    kind = group["kind"]
    if kind == "moe_imbalance":
        overloaded = ", ".join(str(e) for e in incident.get("overloaded", ())[:6]) or "none"
        dead = ", ".join(str(e) for e in incident.get("dead", ())[:6]) or "none"
        when = (f"collapsed since step {group['step']}, reportable at step {incident['reportable_step']}"
                if incident.get("since_start") and incident.get("reportable_step") is not None
                else f"at step {group['step']}")
        text = (f"MoE imbalance {when}, layer {incident.get('layer', 0):02}:"
                f" overloaded experts {overloaded}; dead experts {dead}")
    elif kind == "starved_experts":
        experts = incident.get("experts", ())
        text = (f"Starved experts from step {group['step']}: {len(experts)} in all, first "
                + ", ".join(f"layer {e['layer']:02} expert {e['expert']}" for e in experts[:3]))
    else:
        text = _describe(incident) or f"Incident at step {group['step']}: {kind}"
    if group["numbers_differ"] and len(group["ranks"]) > 1:
        text += f" (rank {group['incident_rank']}'s numbers)"
    return text


def _all_ranks_summary(evidence: Mapping[str, Any]) -> str:
    """The cross-rank picture for an agent: what is common to every judged rank, what only some
    ranks show (named), the ranks with nothing, and what is unknown because a rank's report is
    not current. "No incidents" is said only of ranks that are judged."""
    total = evidence.get("rank_count", 0)
    judged = evidence.get("judged_ranks", [])
    unknown = evidence.get("unknown_ranks", [])
    groups = evidence.get("incident_groups", [])
    clean = evidence.get("clean_ranks", [])
    span = evidence.get("last_step")
    through = (
        "" if not span else f" through step {span['max']}" if span["min"] == span["max"]
        else f" through steps {span['min']}-{span['max']}"
    )
    lines: list[str] = []
    if total == 0:
        lines.append("No rank has stored anything for this run yet, so nothing is known about incidents.")
    elif not unknown:
        if not groups:
            # A stall or a silent rank belongs to the run, not to a rank's steps: say which is empty.
            none = "No incidents in the ranks' step records" if evidence.get("run_incidents") else "No incidents detected"
            lines.append(
                f"{none} on any of the {total} rank{'s' if total != 1 else ''}: every rank's report from"
                f" tensorquill-server is current{through}."
            )
        else:
            with_incidents = evidence.get("ranks_with_incidents", [])
            lines.append(
                f"{len(groups)} distinct incident{'s' if len(groups) != 1 else ''} on {len(with_incidents)} of {total}"
                f" rank{'s' if total != 1 else ''}; every rank's report from tensorquill-server is current{through}."
            )
    else:
        lines.append(
            f"{len(judged)} of {total} ranks are judged from tensorquill-server's current reports{through};"
            f" {len(unknown)} {'is' if len(unknown) == 1 else 'are'} unknown, and nothing is said about incidents on"
            f" {'it' if len(unknown) == 1 else 'them'}."
        )
        if judged and not groups:
            lines.append(f"No incidents on the {len(judged)} judged rank{'s' if len(judged) != 1 else ''}"
                         f" (ranks {_rank_ranges(judged, 8)}).")
    common = [g for g in groups if g["on_all_judged_ranks"]]
    partial = [g for g in groups if not g["on_all_judged_ranks"]]
    if common:
        only = len(judged) == 1
        lines.append(
            f"On the only judged rank ({_rank_ranges(judged)}):" if only
            else f"Common to all {len(judged)} judged ranks ({_rank_ranges(judged, 8)}):"
        )
        lines += [f"  - {_group_line(g)}" for g in common]
    if partial:
        lines.append("Only on some ranks:")
        for group in partial[:SUMMARY_PARTIAL_LINES]:
            ranks = group["ranks"]
            who = (f"rank {ranks[0]} only" if len(ranks) == 1
                   else f"ranks {_rank_ranges(ranks, 8)} ({len(ranks)} of {len(judged)})")
            lines.append(f"  - {who}: {_group_line(group)}")
        rest = partial[SUMMARY_PARTIAL_LINES:]
        if rest:
            kinds: dict[str, int] = {}
            for group in rest:
                kinds[group["kind"]] = kinds.get(group["kind"], 0) + 1
            named = ", ".join(f"{kind} x{count}" for kind, count in sorted(kinds.items()))
            lines.append(f"  - and {len(rest)} more distinct incidents on other ranks ({named}); see incident_groups in"
                         " the evidence file")
    if groups and clean:
        lines.append(f"No incidents on rank{'s' if len(clean) != 1 else ''} {_rank_ranges(clean, 8)}"
                     f" ({len(clean)} of {len(judged)} judged).")
    if unknown:
        by_state = evidence.get("ranks_by_state", {})
        entries = {entry["rank"]: entry for entry in evidence.get("ranks", ())}
        lines.append("Unknown (their reports are not current, so these ranks are not counted as having no incidents):")
        for state, ranks in by_state.items():
            if state == "current":
                continue
            first = entries.get(ranks[0], {})
            reason = (first.get("detection") or {}).get("reason", "")
            line = (f"  - rank{'s' if len(ranks) != 1 else ''} {_rank_ranges(ranks, 8)} ({state}): {reason}")
            held = [(r, entries[r].get("last_report")) for r in ranks if entries.get(r, {}).get("last_report")]
            if held:
                rank, report = held[0]
                kinds = ", ".join(sorted({i.get("kind", "?") for i in report["incidents"]})) or "no incidents"
                line += (f". Rank {rank}'s last report, through step {report['last_step']}, held {kinds}"
                         + (f" (and the same for {len(held) - 1} others)" if len(held) > 1 else ""))
            lines.append(line)
        lines.append("  To judge one of them here, from its stored parts (slower, in proportion to the run's length):"
                     " python -m tensorquill <run folder> --rank <n>")
    differing = evidence.get("thresholds_by_rank") or ()
    if len(differing) > 1:
        usual = differing[0]["thresholds"]
        lines.append("The judged ranks were not all judged by the same thresholds (a server restarted with others"
                     " rebuilds a rank's report only when the rank sends again):")
        for found in differing:
            changed = ", ".join(f"{name} {value!r}" for name, value in found["thresholds"].items()
                                if found is differing[0] or not _same_threshold(value, usual.get(name)))
            lines.append(f"  - rank{'s' if len(found['ranks']) != 1 else ''} {_rank_ranges(found['ranks'], 8)}: {changed}")
    for incident in evidence.get("run_incidents", ()):
        text = _describe(incident)
        lines.append(text or f"Incident at step {incident.get('step')}: {incident['kind']}")
    stacks = stopped_stacks_line(evidence.get("stopped_stacks"))
    if stacks:
        lines.append(stacks)
    classes = _class_line([*groups, *evidence.get("run_incidents", ())])
    if classes:
        lines.append(classes)
    watchdog = evidence.get("watchdog") or {}
    pauses, recommended = watchdog.get("pauses") or {}, watchdog.get("recommended_silent_seconds")
    if pauses and recommended:
        seen = ", ".join(f"{phase} up to {stats['max_seconds']:.0f} s ({stats['count']}x)" for phase, stats in sorted(pauses.items()))
        lines.append(
            f"Pauses the job reported: {seen}. A silence threshold of {recommended:.0f} s (twice the longest)"
            " would have tolerated them: set the server's --silent-seconds to it or more."
        )
    replica = replica_line(evidence)
    if replica:
        lines.append(replica)
    checked = all_ranks_checked_line(evidence)
    if checked:
        lines.append(checked)
    lines += [
        f"Not checked ({item['check']}): {item['reason']}"
        for item in evidence.get("not_checked", ())
        if item["check"] != "resume_lr"
    ]
    # Each judged rank's own (EV-02), once for each entry with the ranks that list it.
    lines += [f"Not checked {_on_ranks(ranks)} ({item.get('check')}): {item.get('reason')}"
              for item, ranks in rank_not_checked(evidence) if item.get("check") != "resume_lr"]
    lines += _source_lines(evidence)
    applied = thresholds_line(evidence)  # only when a threshold applied is not the default
    if applied:
        lines.append(applied)
    return "\n".join(lines)


def rank_not_checked(evidence: Mapping[str, Any]) -> list[tuple[dict[str, Any], list[int]]]:
    """The judged ranks' own `not_checked` entries in an all-ranks view, each once with the ranks
    that list it, in the order they are first listed: from each rank's entry of `ranks` (a store's
    reports) and from `not_checked_by_rank` (`combine_rank_evidence`). Empty when no rank lists any."""
    held: dict[str, tuple[dict[str, Any], list[int]]] = {}
    sources = [*(evidence.get("ranks") or ()), *(evidence.get("not_checked_by_rank") or ())]
    for entry in sources:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("rank"), int):
            continue
        for item in entry.get("not_checked") or ():
            if isinstance(item, Mapping):
                found = held.setdefault(json.dumps(item, sort_keys=True, default=str), (dict(item), []))
                if entry["rank"] not in found[1]:
                    found[1].append(entry["rank"])
    return [(item, sorted(ranks)) for item, ranks in held.values()]


def _class_line(incidents: Iterable[Mapping[str, Any]]) -> str | None:
    """The listed kinds by class (`INCIDENT_CLASSES`), or None when nothing is listed."""
    by_class: dict[str, list[str]] = {}
    for incident in incidents:
        kind = str(incident.get("kind"))
        named = by_class.setdefault(incident_class(kind) or "unclassified", [])
        if kind not in named:
            named.append(kind)
    if not by_class:
        return None
    order = [*INCIDENT_CLASS_NAMES, "unclassified"]
    parts = [f"{name} {', '.join(by_class[name])}" for name in order if name in by_class]
    note = " (observations claim no failure and do not make the run unhealthy)" if "observation" in by_class else ""
    return f"By class: {'; '.join(parts)}{note}"


def _source_lines(evidence: Mapping[str, Any]) -> list[str]:
    """Whether the server is still watching, and whose detectors the incidents come from:
    "no incidents" means something only next to a report that is current."""
    lines = []
    status = evidence.get("watchdog") or {}
    if status.get("summary") == "server_stopped":
        lines.append("The server has stopped: nothing is watching this run for stalls, and no new steps are stored.")
    elif status.get("not_written_for_seconds") is not None:
        lines.append(
            f"The server's status was last written {status['not_written_for_seconds']:.0f} s ago: the server is not"
            " running, or was restarted and no rank of this run has reconnected. Nothing is watching this run for stalls."
        )
    detection = evidence.get("detection")
    if detection:
        reason = detection["reason"]
        if detection["source"] == "server":
            lines.append(
                f"Incidents are from tensorquill-server's report for rank {detection.get('rank')}: {reason}"
                f" ({detection.get('report_steps')} steps judged)."
            )
        else:
            lines.append(f"Incidents were worked out here from the stored parts, because {reason}.")
    return lines


def _scalar_lines(incident: Mapping[str, Any]) -> list[str]:
    context = incident.get("scalars")
    if not context:
        return []

    def fmt(value: Any) -> str:
        if value is None:  # the server's report is strict JSON: a NaN or Inf is null there
            return "not finite"
        return f"{value:.4g}" if isinstance(value, (int, float)) else str(value)

    parts = [
        f"{name} {fmt(entry['value'])}"
        + (f" (median before {fmt(entry['median_before'])})" if entry.get("median_before") is not None else "")
        for name, entry in sorted(context.items())
    ]
    return [f"  Logged at step {incident['step']}: " + ", ".join(parts)]


def _moe_when(incident: Mapping[str, Any], first: str = "First sampled bad step") -> str:
    """When an MoE incident began, and for a router collapsed since its first step when it became
    reportable too (rules contract, part 3): never that it was found at step 0."""
    if incident.get("since_start") and incident.get("reportable_step") is not None:
        return f"Collapsed since step {incident['step']}, reportable at step {incident['reportable_step']}"
    return f"{first} {incident['step']}"


def _moe_lines(evidence: Mapping[str, Any], incident: Mapping[str, Any]) -> list[str]:
    counts, tokens = incident.get("counts"), incident.get("routed_tokens") or 1
    overloaded = ", ".join(
        f"Expert {expert} ({_percent(counts[expert - 1] / tokens)})" if counts else f"Expert {expert}"
        for expert in incident["overloaded"]
    )
    dead = ", ".join(f"{expert:02}" for expert in incident["dead"])
    diversity = float(incident.get("diversity") or 0.0)
    return [
        f"{_moe_when(incident)}  ·  Layer {incident['layer']:02}"
        f"  ·  Diversity {diversity:.2f}",
        f"Overloaded: {overloaded or 'none'}  ·  Dead experts: {dead or 'none'}",
    ]


def routing_colors(sample: Mapping[str, Any], expert_count: int | None = None):
    """RGB colors per expert matching the bake-off status heatmap.

    Missing or skipped router samples (and padded columns when routers have mixed expert
    counts) render as neutral slate gray `MISSING_COLOR` instead of dead-expert red. Only
    experts the detector listed in `sample["dead"]` are red. Experts starved over the
    trailing window are `STARVED_COLOR`, and other zero-token experts are `IDLE_COLOR`.
    """
    counts = sample.get("counts", ())
    target = expert_count if expert_count is not None else len(counts)
    tokens = int(sample.get("routed_tokens") or 0)
    if sample.get("missing") or tokens <= 0:
        for _ in range(target):
            yield MISSING_COLOR
        return
    overloaded = set(sample.get("overloaded", ()))
    dead = set(sample.get("dead", ()))
    starved = set(sample.get("starved", ()))
    for expert, count in enumerate(counts, 1):
        if expert in dead:
            yield DEAD_COLOR
        elif expert in overloaded:
            yield OVERLOADED_COLOR
        elif expert in starved:
            yield STARVED_COLOR
        elif count == 0:
            yield IDLE_COLOR
        else:
            intensity = min(1.0, 100.0 * (count / tokens))
            yield (21 + int(15 * intensity), 44 + int(40 * intensity), 65 + int(54 * intensity))
    for _ in range(max(0, target - len(counts))):
        yield MISSING_COLOR


def export_rerun(
    evidence: Mapping[str, Any],
    path: str | Path,
    *,
    records: Iterable[Mapping[str, Any]] | None = None,
    attempt_order: Mapping[tuple[str, Any], Iterable[Any]] | None = None,
) -> Path:
    """Write `evidence` (and optional Tier 0 `records`) to a Rerun `.rrd` with blueprint.

    At one step, of two attempts' scalar-only points the later attempt's is logged
    (`scalar_points_in_order`): later as `attempt_order` gives each rank's attempts (oldest first,
    by `(run_id, rank)`; `attempts_first_seen` of every record read, before any was resolved away),
    else as the attempts first appear in `records`, training records among them.

    Raises `compact.SamplesOmitted` for compact evidence that left steps out: the recording
    would show the kept windows as if they were the run."""
    from .compact import require_all_samples

    require_all_samples(evidence, "a Rerun recording")
    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hit = first_incident(evidence)
    incident_layer = hit["layer"] if hit is not None else 1
    run_id = str(evidence.get("run_id") or "tensorquill-run")

    rr.init("tensorquill", recording_id=run_id)
    # Let go of the recording however this ends: a failure part way must not leave the
    # process's recording open on a half-written file (DET-14, Nov 19, 2026).
    try:
        rr.save(out_path)
        rr.log("/incident/summary", rr.TextDocument(incident_summary(evidence)), static=True)

        all_samples = list(evidence.get("samples", ()))
        steps = sorted({int(sample["step"]) for sample in all_samples})
        layer_count = int(evidence.get("layer_count") or max((int(s["layer"]) for s in all_samples), default=0))
        expert_count = int(
            evidence.get("expert_count")
            or max((len(s.get("counts", ())) for s in all_samples), default=0)
        )
        by_key = {(int(sample["step"]), int(sample["layer"])): sample for sample in all_samples}

        tabs = []
        if steps and layer_count > 0 and expert_count > 0:
            tabs.append(
                rrb.Horizontal(
                    rrb.Tabs(
                        rrb.Spatial2DView(origin="/routing/status", name="Routing status by layer and expert"),
                        rrb.TensorView(origin="/routing/share", name="Raw routing share"),
                    ),
                    rrb.Vertical(
                        rrb.TextDocumentView(
                            origin="/incident/expert_load", name=f"Layer {incident_layer:02} expert load"
                        ),
                        rrb.TimeSeriesView(origin="/routing/layers", name="Router diversity by layer"),
                        row_shares=[3, 2],
                    ),
                    name="MoE routing",
                )
            )
            # Every step, the first among them: a one-step recording has its table too (DET-14).
            # The tables read one lookup of the samples, made here (lane AD11).
            looked_up = _TABLE_LOOKUP.set(_SampleLookup(evidence))
            try:
                for step in steps:
                    rr.set_time("training_step", sequence=step)
                    rr.log(
                        "/incident/expert_load",
                        rr.TextDocument(
                            expert_load_table(evidence, step, incident_layer),
                            media_type=rr.MediaType.MARKDOWN,
                        ),
                    )
            finally:
                _TABLE_LOOKUP.reset(looked_up)
            for layer in range(1, layer_count + 1):
                rr.log(
                    f"/routing/layers/layer-{layer:02}",
                    rr.SeriesPoints(names=[f"Layer {layer:02}"]),
                    static=True,
                )
            for step in steps:
                rr.set_time("training_step", sequence=step)
                step_samples = [
                    by_key.get((step, layer)) or _missing_sample(step, layer, expert_count)
                    for layer in range(1, layer_count + 1)
                ]
                shares = np.full((layer_count, expert_count), np.nan, dtype=np.float32)
                for row_idx, sample in enumerate(step_samples):
                    tokens = int(sample.get("routed_tokens") or 0)
                    if not sample.get("missing") and tokens > 0:
                        counts = np.asarray(sample["counts"], dtype=np.float64)
                        n = min(len(counts), expert_count)
                        shares[row_idx, :n] = (counts[:n] / tokens).astype(np.float32)
                rr.log(
                    "/routing/share",
                    rr.Tensor(shares, dim_names=("layer", "expert"), value_range=(0.0, 0.36)),
                )
                colors = np.array(
                    [list(routing_colors(sample, expert_count)) for sample in step_samples],
                    dtype=np.uint8,
                )
                rr.log(
                    "/routing/status",
                    rr.Image(
                        bytes=colors.tobytes(),
                        width=expert_count,
                        height=layer_count,
                        color_model="RGB",
                        datatype=np.uint8,
                        magnification_filter="Nearest",
                    ),
                )
                for sample in step_samples:
                    diversity = sample.get("diversity")
                    if not sample.get("missing") and sample.get("routed_tokens", 0) > 0 and diversity is not None:
                        rr.log(
                            f"/routing/layers/layer-{sample['layer']:02}",
                            rr.Scalars(float(diversity)),
                        )

        activation_paths: dict[str, set[str]] = {}
        training_paths: dict[str, str] = {}
        logged_paths: dict[str, str] = {}
        probe_hist: dict[str, str] = {}
        probe_maps: dict[str, str] = {}
        probe_series = False
        probe_invalid = [False]
        if records is not None:
            # Scalar-only records are set apart: none is a training step (WAVE_AL_CONTRACT.md section 4.6).
            # Each of their points is logged under `/logged/<name>` at its step, after the training
            # records', and stands over a training record's value at that step (Q6 a).
            records = list(records)
            order = attempt_order if attempt_order is not None else attempts_first_seen(records)
            records, scalar_only = set_apart_scalars(records)
            standing: dict[str, dict[int, Any]] = {}
            for point in scalar_points_in_order(scalar_only, order=order):
                for name, val in (point.get("scalars") or {}).items() if isinstance(point.get("scalars"), Mapping) else ():
                    if val is not None:
                        standing.setdefault(str(name), {})[int(point["step"])] = val
            overridden = {name: set(at) for name, at in standing.items()}
            for record in records:
                rr.set_time("training_step", sequence=int(record["step"]))
                grad_norm = _finite(record.get("grad_norm"))
                if grad_norm is not None:
                    rr.log("/training/grad_norm", rr.Scalars(grad_norm))
                    training_paths["/training/grad_norm"] = "Gradient norm"
                clip_norm = _finite(record.get("clip_grad_norm"))
                if clip_norm is not None:
                    rr.log("/training/clip_grad_norm", rr.Scalars(clip_norm))
                    training_paths["/training/clip_grad_norm"] = "Pre-clipping gradient norm"
                weights = [v for v in _object(record, "weight_norms").values() if _finite(v) is not None]
                if weights:
                    rr.log("/training/weight_norm", rr.Scalars(math.sqrt(sum(v * v for v in weights))))
                    training_paths["/training/weight_norm"] = "Weight norm"
                lr = record.get("lr")
                if isinstance(lr, list) and lr and _carried(lr[0]):
                    rr.log("/training/lr", rr.Scalars(float(_wire_float(lr[0]))))
                    training_paths["/training/lr"] = "Learning rate"
                scale = _finite(record.get("grad_scale"))
                if scale is not None:
                    rr.log("/training/grad_scale", rr.Scalars(scale))
                    training_paths["/training/grad_scale"] = "Loss scale"
                if "step_skipped" in record:
                    rr.log("/training/step_skipped", rr.Scalars(float(bool(record["step_skipped"]))))
                    training_paths["/training/step_skipped"] = "Update skipped (by the loss scaler or the caller)"
                for name, val in _object(record, "scalars").items():
                    if int(record["step"]) in overridden.get(name, ()):
                        continue  # a scalar-only record's point stands at this step (Q6 a)
                    value = _finite(val)
                    if value is not None:
                        entity = _logged_entity(rr, str(name))
                        rr.log(entity, rr.Scalars(value))
                        logged_paths[entity] = name
                probe = record.get("probe")
                if probe and isinstance(probe, Mapping):
                    probe_series |= _log_probe(rr, probe, probe_hist, probe_maps, probe_invalid)
                for layer_name, stats in layer_entries(record):
                    for key, val in stats.items():
                        val = _finite(val)  # a bool is no value, as the column refuses one (DET-16)
                        if val is not None:
                            entity = (f"/layers/{rr.escape_entity_path_part(str(layer_name))}"
                                      f"/{rr.escape_entity_path_part(str(key))}")
                            rr.log(entity, rr.Scalars(val))
                            activation_paths.setdefault(key, set()).add(entity)
            for name, at in standing.items():
                entity = _logged_entity(rr, name)
                for step, val in sorted(at.items()):
                    value = _finite(val)
                    if value is not None:
                        rr.set_time("training_step", sequence=step)
                        rr.log(entity, rr.Scalars(value))
                        logged_paths[entity] = name

        if activation_paths:
            labels = {
                "abs_max": "Peak magnitude", "mean": "Mean", "std": "Standard deviation",
                "zero_frac": "Zero fraction",
            }
            activation_views = [
                rrb.TimeSeriesView(origin="/layers", contents=sorted(paths), name=labels.get(key, key))
                for key, paths in sorted(activation_paths.items(), key=lambda item: item[0] != "abs_max")
            ]
            tabs.append(rrb.Tabs(*activation_views, name="Activations"))
        if training_paths:
            tabs.append(rrb.Tabs(*(
                rrb.TimeSeriesView(origin=entity, name=name)
                for entity, name in sorted(training_paths.items())
            ), name="Optimizer"))
        if probe_series or probe_hist or probe_maps:
            probe_views = []
            if probe_series:
                probe_views += [
                    rrb.TimeSeriesView(origin="/probe/max_logit", name="Largest attention logit"),
                    rrb.TimeSeriesView(origin="/probe/entropy", name="Attention entropy (share of maximum)"),
                ]
            if probe_invalid[0]:
                probe_views.append(rrb.TimeSeriesView(origin="/probe/invalid_rows", name="Attention rows with NaN or Inf"))
            probe_views += [rrb.BarChartView(origin=e, name=f"Histogram {n}") for e, n in probe_hist.items()]
            probe_views += [rrb.TensorView(origin=e, name=f"Attention {n}") for e, n in probe_maps.items()]
            tabs.append(rrb.Tabs(*probe_views, name="Probe"))
        if logged_paths:
            tabs.append(rrb.Tabs(*(
                rrb.TimeSeriesView(origin=entity, name=name)
                for entity, name in sorted(logged_paths.items(), key=lambda item: item[1])
            ), name="Logged"))

        summary = rrb.TextDocumentView(origin="/incident/summary", name="Training incidents")
        layout = rrb.Vertical(summary, rrb.Tabs(*tabs), row_shares=[1, 4]) if tabs else summary
        rr.send_blueprint(rrb.Blueprint(layout, collapse_panels=True))
    finally:
        rr.disconnect()
    return out_path


def _log_probe(
    rr: Any, probe: Mapping[str, Any], hist_paths: dict, map_paths: dict, invalid_series: list[bool]
) -> bool:
    """Log one probe pass: per-block histograms, attention maps, and per-layer attention
    logit and entropy series. Returns whether any series was logged."""
    import numpy as np

    # A part of the probe that is not what it should be is not logged, and never raised on (BUG-12).
    for layer, hist in _object(probe, "histograms").items():
        if not isinstance(hist, Mapping):
            continue
        counts = hist.get("counts")
        lo, hi = _finite(hist.get("lo")), _finite(hist.get("hi"))
        if not isinstance(counts, list) or not counts or lo is None or hi is None:
            continue
        if any(_finite(count) is None for count in counts):
            continue
        width = (hi - lo) / len(counts)
        entity = f"/probe/histograms/{rr.escape_entity_path_part(layer)}"
        rr.log(entity, rr.BarChart(
            np.asarray(counts, dtype=np.float64),
            abscissa=lo + width * (np.arange(len(counts)) + 0.5),
            widths=np.full(len(counts), width, dtype=np.float32),
        ))
        hist_paths[entity] = layer
    series = False
    for layer, att in _object(probe, "attention").items():
        if not isinstance(att, Mapping):
            continue
        part = rr.escape_entity_path_part(str(layer))
        logits = [v for v in (_finite(x) for x in _listed(att.get("max_logit"))) if v is not None]
        if logits:
            rr.log(f"/probe/max_logit/{part}", rr.Scalars(max(logits)))
            series = True
        shares = [v for v in (_finite(x) for x in _listed(att.get("entropy_share"))) if v is not None]
        if not shares and _finite(att.get("max_entropy")):
            # Probe records from before `entropy_share`: entropy over the log of all keys.
            top = float(att["max_entropy"])
            shares = [v / top for v in (_finite(x) for x in _listed(att.get("entropy"))) if v is not None]
        if shares:
            rr.log(f"/probe/entropy/{part}", rr.Scalars(sum(shares) / len(shares)))
            series = True
        invalid_rows = _finite(att.get("invalid_rows"))
        if invalid_rows:
            # NaN or Inf attention scores: a numerical failure, kept apart from entropy.
            rr.log(f"/probe/invalid_rows/{part}", rr.Scalars(invalid_rows))
            invalid_series[0] = True
        if att.get("map"):
            try:
                values = np.asarray(att["map"], dtype=np.float32)
            except (TypeError, ValueError):
                continue  # not a table of numbers
            if values.ndim != 2:
                continue
            entity = f"/probe/attention/{part}"
            rr.log(entity, rr.Tensor(values, dim_names=("query", "key")))
            map_paths[entity] = layer
    return series


def _logged_entity(rr: Any, name: str) -> str:
    """The `/logged/...` entity of a logged scalar's name: one entity for each name (BUG-28). A name
    whose `/`-separated parts are all non-empty is that path, each part escaped, as before
    (`train/loss` is `/logged/train/loss`). A name with an empty part (`/loss`, `a//b`, `loss/`) is
    one escaped part, its `/` kept inside it, so it never meets another name's path: before, the
    empty parts were dropped and `/loss` was `loss`'s entity."""
    parts = name.split("/")
    if all(parts):
        return "/logged/" + "/".join(rr.escape_entity_path_part(part) for part in parts)
    return "/logged/" + rr.escape_entity_path_part(name)


def _listed(value: Any) -> list[Any]:
    """`value` when it is a list, else an empty one."""
    return value if isinstance(value, list) else []


def _rank_argument(value: str) -> int | str:
    if value == ALL_RANKS:
        return ALL_RANKS
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is neither a rank number nor 'all'") from None


def _main_all_ranks(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """`--rank all`: the all-ranks view, kept up to date with `--watch` and written once without."""
    if args.rerun is not None:
        parser.error("--rank all cannot be combined with --rerun: a Rerun recording is one rank's steps; "
                     "use --rank N (a rank number) with --rerun")
    if args.recompute:
        parser.error("--rank all cannot be combined with --recompute: it would run the detectors here for every rank, "
                     "one after another, on every change; recompute one rank with --rank N --watch --recompute")
    shown: list[str] = []

    def report(evidence: dict[str, Any], path: Path) -> None:
        summary, state = incident_summary(evidence), run_state_line(evidence)
        if [summary, state] != shown:
            shown[:] = [summary, state]
            print(summary)
            if state:
                print(state)
        print(f"updated {path}" if args.watch else f"wrote evidence -> {path}", flush=True)

    if args.watch:
        print(f"watching every rank of {args.input} every {args.interval:g} s; Ctrl-C to stop", flush=True)
    try:
        written = watch(
            args.input, args.output, interval=args.interval, rank=ALL_RANKS, polls=None if args.watch else 1,
            on_update=report, report_grace=args.report_grace, **args.thresholds,
            flight_recorder=args.flight_recorder, flight_recorder_prefix=args.flight_recorder_prefix,
            traincheck=args.traincheck, compact_window=args.compact_window, trigger=args.trigger,
            detectors=args.detectors or "auto",  # nothing runs here, so no binary is asked for
        )
    except MalformedRecordError:
        raise  # said by `main`, never as a usage error (this view reads no step record today)
    except ValueError as error:
        parser.error(str(error))
    if written is None and not args.watch:
        raise SystemExit(f"no rank=<n> folder under {args.input} and no watchdog.json: nothing to show")


def main() -> None:
    """`python -m tensorquill` and the `tensorquill` console script. An input holding a record whose
    identity field is not valid (`MalformedRecordError`, WAVE_AF_CONTRACT.md, section 1) ends the
    command with status 2 and one line, `tensorquill: error: <what, and where>`, and no traceback,
    with either engine and for every verb: nothing was judged and no evidence file was written."""
    try:
        _main()
    except MalformedRecordError as error:
        print(f"tensorquill: error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from None


def _main() -> None:
    # Two verbs come before the path: `runs` (list the runs of a folder or store) and `compare`.
    # The first argument alone decides, whatever is on disk, so the same command means the same
    # thing everywhere; a file or folder with one of those names is read as `./runs` or
    # `./compare`. Both `python -m tensorquill` and the `tensorquill` console script come here.
    if len(sys.argv) > 1 and sys.argv[1] in ("runs", "compare"):
        from . import runs as _runs

        _runs.main(sys.argv[1], sys.argv[2:])
        return
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Two verbs take the place of the path: `python -m tensorquill runs <folder or store>` lists the "
        "runs there by project, and `python -m tensorquill compare A B` compares runs given by ID or name "
        "(each has its own --help). A file or folder named `runs` or `compare` is read as ./runs or ./compare.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="captured .jsonl file from tensorquill.init, or a directory of Parquet parts "
        "written by tensorquill-server (runs/<run_id>, or a rank=<rank> directory in it; "
        "the run directory with --rank all)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="path to write viewer-neutral evidence JSON (default: <input>.evidence.json, "
        "or <run_id>.evidence.json for a directory, <run_id>.all-ranks.evidence.json with --rank all)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="keep running: re-read the input when it grows and replace the evidence file "
        "(with --rank all: every rank's report and the run's status)",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between looks with --watch")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="with --watch on a server's store: do not use the incidents.json the server keeps for the rank; "
        "read the parts and run the detectors here (slower; gives per-step samples). One rank at a time: not with "
        "--rank all",
    )
    parser.add_argument(
        "--report-grace",
        type=float,
        default=REPORT_GRACE_SECONDS,
        help="with --watch on a server's store: seconds the server's report may stay behind the stored parts "
        "before it is called stale and the detectors run here (default: 30; a restarted server rebuilds a rank "
        "from its whole store first, about 6 s for 64 ranks of 10,000 steps). With --rank all it is kept for each "
        "rank, and a stale rank is listed as unknown instead",
    )
    parser.add_argument(
        "--rank",
        type=_rank_argument,
        help="the rank to read (default: rank 0 for a directory holding several ranks), or 'all' for the all-ranks "
        "view of a run folder in a server's store (runs/<run_id>): one evidence file for every rank, taken from "
        "each rank's incidents.json and the run's watchdog.json, with identical incidents grouped across ranks and "
        "ranks whose report is not current listed as unknown. It reads no part and runs no detector, so it works "
        "with or without --watch (one look without) and cannot be combined with --recompute or --rerun; a "
        "threshold option given with it must be the one each rank's server applied; "
        "its default output is <run_id>.all-ranks.evidence.json",
    )
    parser.add_argument("--rerun", type=Path, help="optional path to write a Rerun .rrd file "
                        "(one rank's steps: not with --rank all)")
    parser.add_argument(
        "--flight-recorder",
        type=Path,
        help="directory of PyTorch Flight Recorder dumps (one pickle per rank): the first stuck or "
        "mismatched collective becomes an incident (PyTorch 2.10 or later)",
    )
    parser.add_argument(
        "--traincheck",
        type=Path,
        help="TrainCheck's violations_summary.json, or the results directory traincheck-check wrote: "
        "its violated invariants become one incident",
    )
    parser.add_argument("--flight-recorder-prefix", help="common start of the dump file names, if PyTorch cannot infer it")
    # The five detector thresholds. None of them has a value until it is typed: the detectors
    # run here with the defaults named below, and a server's report is taken with the values the
    # server applied (`tensorquill-server --thresholds`), which the evidence lists.
    parser.add_argument(
        "--overload-threshold",
        type=float,
        help="base assignment-share threshold for top-1 and top-2 routers (default: 0.35). Reading a server's "
        "report, this and the four options below default to what the server applied; one given here that differs "
        "from it makes the report `other_thresholds`",
    )
    parser.add_argument(
        "--drift-tolerance",
        type=float,
        help="largest spread of a replicated parameter's fingerprints across GPUs that is not replica drift "
        "(default: 1e-6)",
    )
    parser.add_argument(
        "--starve-window",
        type=int,
        help="samples of a router over which each expert's assignments are summed for starved experts (default: 50)",
    )
    parser.add_argument(
        "--starve-ratio",
        type=float,
        help="an expert is starved when its share over that window is below this fraction of an even share "
        "(default: 0.1)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="experts chosen per token, for captures made before moe_top_k was recorded",
    )
    parser.add_argument(
        "--update-window",
        type=int,
        help="how many of the last step-to-step changes of the update size make the baseline of update_spike (default: 100)",
    )
    parser.add_argument(
        "--detectors",
        choices=("auto", "python", "rust"),
        help="whose detectors run here. rust: the incidents of the server's kinds come from the Rust detectors, "
        "run on this file through `tensorquill-server detect` (the binary named by TENSORQUILL_SERVER_BIN, else the "
        "one `pip install tensorquill-server` put beside this Python, else the one on PATH; needs pyarrow, "
        "`pip install 'tensorquill[flight]'`). python: the Python detectors, which need neither. auto, the default "
        "for one read: rust when the binary and pyarrow are there, and otherwise python with one warning that says "
        "what to install. --watch needs the Rust detectors and stops before its first look when they are missing; "
        "only `--detectors python` makes it watch with the Python ones. With --watch and the Rust detectors, one "
        "detector process is kept for the watcher and is sent only the records appended since its last look; it reads "
        "everything again after rows that arrive late, a new attempt, a file that was cut short or replaced, or a "
        "change of thresholds. With --watch and the Python detectors, every record read so far is judged again at "
        "each rebuild. With --rank all the option changes nothing: that view "
        "takes every rank's incidents from its server's report and runs no detector, so it needs no binary",
    )
    from . import compact as _compact, trigger as _trigger

    _compact.add_arguments(parser)
    _trigger.add_arguments(parser)
    args = parser.parse_args()
    args.compact_window = _compact.window_from_args(args)  # None: the full file
    args.trigger = _trigger.from_args(args, parser)  # None without --on-incident or --webhook
    args.thresholds = _given_thresholds(vars(args))  # only the ones typed
    try:
        check_thresholds(**args.thresholds)  # a usage error with either detectors, never a traceback
    except ValueError as error:
        parser.error(str(error))

    if args.rank == ALL_RANKS:
        _main_all_ranks(args, parser)
        return
    rank = args.rank
    announced_rank = False
    if rank is None and args.input.is_dir() and len(list(args.input.glob("rank=*"))) > 1:
        rank = 0  # the detectors read one rank's steps
        announced_rank = True
        print("several ranks here; reading rank 0 (choose another with --rank N, or every rank with --rank all)")
    if args.watch:
        if args.rerun is not None:
            parser.error("--rerun cannot be combined with --watch")
        shown: list[str] = []

        def report(evidence: dict[str, Any], path: Path) -> None:
            nonlocal announced_rank
            if args.rank is None and not announced_rank and len(_rank_dirs(args.input)) > 1:
                announced_rank = True
                print("several ranks here; reading rank 0 (choose another with --rank N, or every rank with --rank all)")
            summary, state = incident_summary(evidence), run_state_line(evidence)
            if [summary, state] != shown:
                shown[:] = [summary, state]
                print(summary)
                if state:
                    print(state)
            print(f"updated {path}", flush=True)

        # A watcher judges with the Rust detectors unless it is told `--detectors python`: a missing
        # binary or pyarrow stops it here, before the first look, with what to install.
        detectors = "python" if args.detectors == "python" else "rust"
        if detectors == "rust":
            from .server_detect import rust_ready

            try:
                rust_ready()
            except RuntimeError as error:
                parser.error(f"--watch needs the Rust detectors: {error}. To watch with the Python detectors, "
                             "add --detectors python")
        print(f"watching {args.input} every {args.interval:g} s; Ctrl-C to stop", flush=True)
        try:
            watch(
                args.input, args.output, interval=args.interval, rank=rank,
                default_rank=0 if rank is None else None, on_update=report,
                server_reports=not args.recompute, report_grace=args.report_grace, **args.thresholds,
                flight_recorder=args.flight_recorder, flight_recorder_prefix=args.flight_recorder_prefix,
                traincheck=args.traincheck, compact_window=args.compact_window, trigger=args.trigger,
                detectors=detectors,
            )
        except (RuntimeError, ValueError) as error:
            # The Rust path's own refusals: a binary that fails, thresholds its rules refuse. A record
            # that is not valid is said by `main`, as with the Python detectors.
            if detectors != "rust" or isinstance(error, MalformedRecordError):
                raise
            parser.error(str(error))
        return

    # One read: the Rust detectors when they are installed, and otherwise the Python ones, said once.
    from .server_detect import resolve_detectors

    detectors, fallback = resolve_detectors(args.detectors or "auto")
    if fallback is not None:
        print(f"tensorquill: warning: {fallback}. `--detectors python` asks for the Python detectors without this "
              "warning", file=sys.stderr)
    # A file is read whole and its rank picked by `build_evidence`, which then says which ranks the
    # file held (L21); a directory is read a rank at a time, as before.
    whole_file = not args.input.is_dir()
    # The scalar-only records are read for the Rerun export's series (section 4.6) and, in a file,
    # for the check that the rank's records are of one run, theirs included (BUG-29).
    every, left_out = _load_counted(args.input, None if whole_file else rank, scalars=whole_file or args.rerun is not None)
    records, scalar_only = set_apart_scalars(every)
    held = sorted({int(r.get("rank", 0)) for r in records})
    if whole_file and rank is None and len(held) > 1:
        parser.error(f"{args.input} holds ranks {_rank_ranges(held, 8)}: choose one with --rank N (the evidence is one "
                     "rank's), or compare every rank with python -m tensorquill compare")
    judged = rank if rank is not None else held[0] if len(held) == 1 else None
    own = records if rank is None else [r for r in records if r.get("rank", 0) == rank]
    try:
        # A usage error with either detectors, as the Rust path gives it.
        check_one_run([*own, *(r for r in scalar_only if judged is None or r.get("rank", 0) == judged)])
    except ValueError as error:
        parser.error(str(error))
    try:
        evidence = build_evidence(
            records, rank=rank, **args.thresholds, watchdog=_watchdog_beside(args.input),
            flight_recorder=args.flight_recorder, flight_recorder_prefix=args.flight_recorder_prefix,
            traincheck=args.traincheck, detectors=detectors, run_manifest=args.input, records_file=args.input,
        )
    except (RuntimeError, ValueError) as error:
        if detectors != "rust" or isinstance(error, MalformedRecordError):
            raise
        parser.error(str(error))
    if left_out:
        evidence["lines_left_out"] = left_out
    out_json = _write_output(
        evidence, args.output or default_evidence_path(args.input, records), args.output is not None,
        args.compact_window,
    )
    print(incident_summary(evidence))
    print(run_state_line(evidence))
    print(f"wrote {'evidence' if args.compact_window is None else 'compact evidence'} -> {out_json}")
    if args.rerun is not None:
        # The judged rank's records: another rank's scalar-only points are not this recording's.
        exported = [r for r in every if judged is None or r.get("rank", 0) == judged]
        export_rerun(evidence, args.rerun, records=resolve_attempts(exported), attempt_order=attempts_first_seen(exported))
        print(f"wrote Rerun recording -> {args.rerun}")


if __name__ == "__main__":
    main()
