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
import json
import math
import os
import re
import sys
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import manifest as _manifest
from .data_checks import DATA_STALL_STEPS, detect_data_stalled, detect_epoch_order_repeated, detect_resume_state
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
# step (`update_rms`, with `init(..., updates=True)`). A step is flagged when R changed from
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
ACT_DRIFT_FACTOR = 3.0  # recent median of std or abs_max/std vs the first ACT_WINDOW records
ACT_RECENT = 10
ACT_DEAD_RISE = 0.25  # zero fraction above its baseline, as a recent median
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
    if isinstance(value, str):
        return value in ("nan", "inf", "-inf")
    if isinstance(value, (int, float)):
        return not math.isfinite(value)
    return False


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
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        sample["starved"] = []
        by_layer.setdefault(sample["layer"], []).append(sample)
    for seq in by_layer.values():
        recent: list[dict[str, Any]] = []
        sums: list[int] = []
        tokens = 0
        for sample in seq:  # in step order
            if sample.get("missing") or sample["routed_tokens"] <= 0:
                continue
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
    first: dict[tuple[int, int], dict[str, Any]] = {}
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
    if not first:
        return None
    experts = sorted(first.values(), key=lambda e: (e["window_share"], e["layer"], e["expert"]))
    return {"kind": "starved_experts", "step": min(e["step"] for e in experts), "experts": experts}


def _router_top_k(
    records: list[dict[str, Any]], routers: list[str], override: int | Mapping[str, int] | None
) -> dict[str, int]:
    found: dict[str, int] = {}
    for record in records:
        for name, k in record.get("moe_top_k", {}).items():
            found[name] = int(k)
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


def _finite(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def scalar_context(
    records: list[dict[str, Any]], step: int, *, window: int = SPIKE_WINDOW
) -> dict[str, dict[str, Any]]:
    """Values logged with `session.log` at `step`, each with the median of its finite
    values over the previous `window` records, so an incident shows what the loss did."""
    index = next((i for i, record in enumerate(records) if record.get("step") == step), None)
    if index is None:
        return {}
    context = {}
    for name, value in (records[index].get("scalars") or {}).items():
        before = [
            v for v in (_finite((r.get("scalars") or {}).get(name)) for r in records[max(0, index - window) : index])
            if v is not None
        ]
        context[name] = {"value": value, "median_before": _median(before) if before else None}
    return context


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
    totals: list[float] = []
    per_param: dict[str, list[tuple[int, float]]] = {}
    baselines: dict[str, _Baseline] = {}
    first: dict[str, Any] | None = None
    spikes: list[int] = []
    # One series for the whole run. When the run clips, the pre-clipping norm is used:
    # the per-parameter norms are read after clipping, so a clipped spike stays flat there.
    key = "clip_grad_norm" if any("clip_grad_norm" in r for r in records) else "grad_norm"
    for record in sorted(records, key=lambda r: r["step"]):
        if record.get("step_skipped"):
            continue  # a loss-scaler overflow step: its gradients are Inf or NaN by design
        total = _finite(record.get(key))
        if total is None:
            continue
        step = int(record["step"])
        baseline = _median(totals[-window:]) if len(totals) >= min_history else None
        if baseline is not None and baseline > 0 and total >= factor * baseline:
            spikes.append(step)
            if first is None:
                first = {
                    "step": step,
                    "grad_norm": total,
                    "baseline": baseline,
                    "ratio": total / baseline,
                    "parameters": _spiking_parameters(record, per_param, factor, window),
                }
        totals.append(total)
        lasted: list[dict[str, Any]] = []
        for name, value in record.get("grad_norms", {}).items():
            norm = _finite(value)
            if norm is None:
                continue
            per_param.setdefault(name, []).append((step, norm))
            own = baselines.get(name)
            if own is None:
                own = baselines[name] = _Baseline()
            if len(own.values) >= min_history and norm >= factor * own.low:
                usual = _median(list(own.values))
                if usual > 0 and norm >= factor * usual:
                    if own.count == 0:
                        own.start = {
                            "step": step, "total": total, "total_baseline": baseline,
                            "parameter": {"name": name, "grad_norm": norm, "baseline": usual,
                                          "ratio": norm / usual, "rising_since": step},
                        }
                    own.count += 1
                    if own.count == sustain and own.start is not None:
                        lasted.append(own.start)
                    continue
            own.count = 0
            own.keep(norm, window)
        if lasted and first is None:
            # Usually every excursion that lasts to this record began on the same step; if
            # not, the incident is the earliest, with the parameters that began there.
            began = min(start["step"] for start in lasted)
            starts = [start for start in lasted if start["step"] == began]
            parameters = sorted((start["parameter"] for start in starts), key=lambda p: (-p["ratio"], p["name"]))
            usual = starts[0]["total_baseline"]
            first = {
                "step": began,
                "grad_norm": starts[0]["total"],
                "baseline": usual,
                "ratio": starts[0]["total"] / usual if usual else None,
                "parameters": parameters[:10],
                "trigger": "parameter",
                "sustained": sustain,
            }
    if first is None:
        return None
    return {"kind": "grad_spike", **first, "spike_count": len(spikes), "spike_steps": spikes[:20]}


def detect_update_spikes(
    records: list[dict[str, Any]], *, window: int = UPDATE_WINDOW, alpha: float = UPDATE_ALPHA
) -> dict[str, Any] | None:
    """Return the first jump in the size of the optimizer's update and the parameters whose
    own update moved most, or None.

    Each record's `update_rms` gives the RMS of Adam's update to each parameter; `R` is the
    largest of them. A record is compared with the last one before it that could be judged, in
    the same attempt: `change` is how far `R` moved (either way) and `rise` how far the total
    gradient norm rose (0 when it fell). The first `window` changes are not judged. After that
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
    record before, largest first, so the incident says where the jump is.
    """
    key = "clip_grad_norm" if any("clip_grad_norm" in r for r in records) else "grad_norm"
    before: tuple[Any, float, float, dict[str, float]] | None = None  # attempt, R, total, each parameter's
    recent: deque[float] = deque(maxlen=window)  # the last `window` changes that were not flagged
    first: dict[str, Any] | None = None
    spikes: list[int] = []
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
        largest = max(own.values())
        attempt = record.get("attempt", 0)
        if before is not None and before[0] == attempt:
            change, rise = abs(largest - before[1]), max(0.0, total - before[2])
            if len(recent) < window:
                recent.append(change)
            else:
                baseline = math.fsum(recent) / window
                threshold = baseline / (alpha * math.sqrt(rise)) if rise > 0 and baseline > 0 else None
                if threshold is None or change <= threshold:
                    recent.append(change)
                else:
                    step = int(record["step"])
                    spikes.append(step)
                    if first is None:
                        moved = sorted(
                            ({"name": name, "update_rms": size, "previous": before[3][name],
                              "change": abs(size - before[3][name])}
                             for name, size in own.items() if name in before[3]),
                            key=lambda p: (-p["change"], p["name"]),
                        )
                        first = {
                            "step": step, "update_rms": largest, "previous": before[1], "change": change,
                            "baseline": baseline, "threshold": threshold, "grad_norm": total,
                            "grad_norm_before": before[2], "grad_rise": rise, "parameters": moved[:10],
                        }
        before = (attempt, largest, total, own)
    if first is None:
        return None
    return {"kind": "update_spike", **first, "spike_count": len(spikes), "spike_steps": spikes[:20]}


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


def _spiking_parameters(
    record: dict[str, Any],
    per_param: dict[str, list[tuple[int, float]]],
    factor: float,
    window: int,
) -> list[dict[str, Any]]:
    found = []
    for name, value in record.get("grad_norms", {}).items():
        norm = _finite(value)
        past = per_param.get(name, [])[-window:]
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
             "ratio": norm / baseline, "rising_since": rising}
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
    checked = [r for r in sorted(records, key=lambda r: r["step"]) if r.get("weight_norms")]
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
            "median_ratio": values[len(values) // 2],
            "lowest": {"name": lowest[0], "ratio": lowest[1]},
            "highest": {"name": highest[0], "ratio": highest[1]},
        }
    return {"baseline_step": int(first["step"]), "latest_step": int(last["step"]), "kinds": summary}


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
    return trends


def _watchdog_beside(source: Any) -> Path | None:
    if not isinstance(source, (str, Path)) or not Path(source).is_dir():
        return None
    directory = Path(source).resolve()
    status = directory / "watchdog.json"
    key, _, rank = directory.name.partition("=")
    if key == "rank" and rank.isdecimal() and not status.is_file():
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


def _progress_incidents(status: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not status or status.get("summary") not in ("stalled", "degraded"):
        return []
    full = list(status.get("ranks", ()))
    places = [place for place in map(stuck_at, (r for r in full if r.get("state") in ("silent", "lost", "unresponsive")))
              if place is not None]
    ranks = [{k: v for k, v in r.items() if k != "stack"} for r in full]
    stopped = [r for r in ranks if r.get("state") in ("silent", "lost", "unresponsive")]
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
        return [{
            "kind": "job_stalled",
            "step": max(steps, default=0),
            "ranks": [r for r in ranks if r.get("state") not in ("finished",)],
            "missing_ranks": list(status.get("missing_ranks", ())),
            **common,
        }]
    affected = [r for r in stopped]
    behind = [r["last_step"] for r in affected if r.get("last_step") is not None]
    # The ranks that are not stopped, briefly: what a reader needs to tell that a rank which was
    # silent has stepped on since (the trigger's stall episodes).
    others = [{"rank": r.get("rank"), "state": r.get("state"), "last_step": r.get("last_step"), "attempt": r.get("attempt")}
              for r in ranks if r.get("state") not in ("silent", "lost", "unresponsive")]
    return [{
        "kind": "rank_silent",
        "step": min(behind) if behind else max(steps, default=0),
        "ranks": affected,
        "other_ranks": others,
        "missing_ranks": list(status.get("missing_ranks", ())),
        "newest_step": max(steps, default=0),
        **common,
    }]


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
    once that holds on `steps` checks in a row while its gradient is finite and non-zero.

    There is one incident, dated at the first step any parameter stalled; `step`,
    `parameters`, `stalled` and `optimizer` describe that first onset. Every parameter is
    judged on its own history, so one that stalls later is named too: `onsets` lists each
    step at which parameters first stalled, with the parameters (those that had changed at
    an earlier check first), how many had (`trained_before`), and the optimizer evidence for
    them; `stalled_total` counts every parameter that stalled at any step. A parameter that has never changed since the first check and one that trained and
    then stopped are different findings: with bf16 or fp16 weights and no float32 master
    copy, an update smaller than the weight's spacing rounds away from the first step (norm
    weights near 1.0 at a small learning rate), while a parameter that trained and then
    stopped was frozen during the run. `weight_dtypes` (how many trainable parameters have
    each dtype, when the capture recorded it) says which precision that is.

    `optimizer` says whether an optimizer that stepped holds the parameters: `known` with
    the stalled parameters no optimizer holds, or `unknown` with the reason, for DeepSpeed,
    Megatron, FSDP1, `auto_step=False` or captures that did not record it. A stalled
    parameter that no optimizer holds is the likely cause. Steps with a non-finite gradient
    norm are skipped, because mixed-precision loss scaling skips those optimizer steps on
    purpose. Captures without `weights_unchanged` (older files, or Apple MPS) are not
    checked: a float32 norm stays the same while bf16 weights change by a few elements.
    """
    streak: dict[str, int] = {}
    changed: set[str] = set()  # seen changing at some weight check
    reported: set[str] = set()
    onsets: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    stalled_steps = 0
    onset_steps = 0
    for record in sorted(records, key=lambda r: r["step"]):
        unchanged = record.get("weights_unchanged")
        if unchanged is None:
            continue
        if _finite(record.get("grad_norm")) is None and record.get("grad_norm") is not None:
            continue
        grads = record.get("grad_norms", {})
        unchanged_set = set(unchanged)
        for name, value in grads.items():
            grad = _finite(value)
            if grad is not None and grad > 0 and name in unchanged_set:
                streak[name] = streak.get(name, 0) + 1
            else:
                streak[name] = 0
        stalled = [name for name, count in streak.items() if count >= steps]
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
        changed.update(name for name in grads if name not in unchanged_set)
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
    - `activation_drift`: the median of the last ACT_RECENT records of `std`, or of the
      outlier ratio `abs_max / std`, moved ACT_DRIFT_FACTOR times above or below its
      median over the module's first ACT_WINDOW records.
    - `dead_activations`: the recent median `zero_frac` rose ACT_DEAD_RISE above its
      baseline, as when units stop firing.

    Returns at most one incident per kind: the first step it happened, the modules
    affected at that step, and how many steps were affected in all.
    """
    history: dict[str, list[tuple[float, float, float]]] = {}
    # Each module's baseline for drift, from its first ACT_WINDOW records. A module's history
    # only grows, so those records never change once the drift check starts; they are
    # summarized once, not again for every later record.
    baselines: dict[str, tuple[float, float, float]] = {}
    found: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    outlier = lambda rows: _median([a / s for s, a, _ in rows if s > 0] or [0.0])

    def report(kind: str, step: int, hits: list[dict[str, Any]]) -> None:
        if not hits:
            return
        counts[kind] = counts.get(kind, 0) + 1
        if kind not in found:
            found[kind] = {"kind": kind, "step": step, "modules": hits[:20], "affected": len(hits)}

    for record in sorted(records, key=lambda r: r["step"]):
        step = int(record["step"])
        spikes, drifts, dead = [], [], []
        for name, stats in record.get("layers", {}).items():
            std, amax, zfrac = (_finite(stats.get(k)) for k in ("std", "abs_max", "zero_frac"))
            if std is None or amax is None or zfrac is None:
                continue
            past = history.setdefault(name, [])
            if len(past) >= ACT_MIN_HISTORY:
                recent = past[-ACT_WINDOW:]
                for key, value, base in (
                    ("abs_max", amax, _median([p[1] for p in recent])),
                    ("std", std, _median([p[0] for p in recent])),
                ):
                    if base > 0 and value >= ACT_SPIKE_FACTOR * base:
                        spikes.append({"module": name, "stat": key, "value": value,
                                       "baseline": base, "ratio": value / base})
                        break
            past.append((std, amax, zfrac))
            if len(past) >= ACT_WINDOW + ACT_RECENT:
                last = past[-ACT_RECENT:]
                if name not in baselines:
                    first = past[:ACT_WINDOW]
                    baselines[name] = (_median([p[0] for p in first]), outlier(first), _median([p[2] for p in first]))
                base_std, base_outlier, base_zero = baselines[name]
                for key, base, now in (
                    ("std", base_std, _median([p[0] for p in last])),
                    ("abs_max/std", base_outlier, outlier(last)),
                ):
                    if base > 0 and now > 0 and max(now / base, base / now) >= ACT_DRIFT_FACTOR:
                        drifts.append({"module": name, "stat": key, "baseline": base,
                                       "recent": now, "ratio": now / base})
                        break
                now_zero = _median([p[2] for p in last])
                if now_zero - base_zero >= ACT_DEAD_RISE:
                    dead.append({"module": name, "baseline": base_zero, "recent": now_zero})
        report("activation_spike", step, _first_in_forward_pass_leads(spikes))
        report("activation_drift", step, drifts)
        report("dead_activations", step, dead)
    for kind, incident in found.items():
        incident["affected_steps"] = counts[kind]
    return list(found.values())


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
    entry when any probe failed (`{"error": ...}` in place of its results). Too few probes for
    a module to be judged and confirmed is not an entry: the kinds are then `checked` without
    enough history yet (`kind_checks`, `RECORDS_NEEDED`), and the run is not degraded for it.
    """
    shares: dict[str, _AttentionSeries] = {}
    logits: dict[str, _AttentionSeries] = {}
    found: dict[str, dict[str, Any]] = {}
    affected: dict[str, set[int]] = {}
    probe_steps: list[int] = []
    asked, failed, first_failure = 0, 0, None
    for record in sorted(records, key=lambda r: r["step"]):
        probe = record.get("probe")
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
                            "heads": heads,
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
    return [found[kind] for kind in ATTN_KINDS if kind in found]


def detect_loss_scale_trouble(
    records: list[dict[str, Any]], *, run: int = SCALE_SKIP_RUN, floor: float = SCALE_FLOOR
) -> dict[str, Any] | None:
    """Return the first step where mixed-precision loss scaling is in trouble, or None:
    `run` or more optimizer steps in a row skipped by the scaler, or the loss scale below
    `floor`. Needs `step_skipped` and `grad_scale`, recorded with `init(..., scaler=)`."""
    streak, skipped, total, first = 0, 0, 0, None
    for record in sorted(records, key=lambda r: r["step"]):
        if "step_skipped" not in record:
            continue
        total += 1
        skipped += bool(record["step_skipped"])
        streak = streak + 1 if record["step_skipped"] else 0
        scale = _finite(record.get("grad_scale"))
        if first is None and (streak >= run or (scale is not None and scale < floor)):
            first = {
                "kind": "loss_scale_trouble",
                "step": int(record["step"]),
                "consecutive_skips": streak,
                "grad_scale": scale,
                "reason": "skips" if streak >= run else "scale",
            }
    if first is not None:
        first["skipped_steps"] = skipped
        first["recorded_steps"] = total
    return first


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
        attempts = sorted(steps)
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
    lr = record.get("lr")
    if not lr:
        return None
    values = [_finite(v) for v in lr]
    return None if any(v is None for v in values) else [float(v) for v in values]  # type: ignore[arg-type]


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
        order = sorted(attempts)
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
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            # One rank at a time: a run of many ranks never has to fit in memory at once.
            records = [
                r for part in _parquet_parts(path, rank) for r in _read_part(part)
                if rank is None or r.get("rank", 0) == rank
            ]
        elif path.suffix == ".parquet":
            records = _read_part(path)
        else:
            lines = path.read_text(encoding="utf-8").splitlines()
            records = [json.loads(line) for line in lines if line.strip()]
    else:
        records = [dict(item) for item in source]
    if rank is not None:
        records = [r for r in records if r.get("rank", 0) == rank]
    return records if all_attempts else resolve_attempts(records)


def resolve_attempts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Leave out the records a later attempt replaced.

    Each `tensorquill.init` stamps its records with `attempt`, its start time. A job that
    resumes from the checkpoint at step S writes step S and onwards again as a new
    attempt; from then on, what earlier attempts of that rank recorded at step S or later
    is history that training rolled back, including steps the new attempt has not reached
    yet. Records without `attempt` (written before it existed) count as attempt 0. The
    order of the remaining records is unchanged.
    """
    first_step: dict[tuple[Any, int], int] = {}
    for record in records:
        key = (record.get("rank", 0), record.get("attempt", 0))
        step = int(record["step"])
        if key not in first_step or step < first_step[key]:
            first_step[key] = step
    if len({rank for rank, _ in first_step}) == len(first_step):
        return records  # one attempt per rank: nothing was replaced
    cutoff: dict[tuple[Any, int], float] = {}
    for rank in {rank for rank, _ in first_step}:
        cut = math.inf
        for attempt in sorted((a for r, a in first_step if r == rank), reverse=True):
            cutoff[(rank, attempt)] = cut
            cut = min(cut, first_step[(rank, attempt)])
    return [
        record for record in records
        if int(record["step"]) < cutoff[(record.get("rank", 0), record.get("attempt", 0))]
    ]


_PART_NAME = re.compile(r"part-(-?\d+)-(-?\d+)\.parquet$")


def _other_rank_folder(entry: os.DirEntry[str], rank: int) -> bool:
    name, _, value = entry.name.partition("=")
    return name == "rank" and value.lstrip("-").isdigit() and int(value) != rank


def _parquet_parts(directory: Path, rank: int | None = None) -> list[Path]:
    """The server's part files under `directory`, ordered by rank, attempt and first step.

    With `rank`, the `rank=<n>` folders of other ranks directly under `directory` are not
    entered: the server writes a rank's records only into its own folder, so reading and
    discarding them would cost every other rank's share of the run on each look. Parts
    anywhere else are listed as before and their records filtered by the caller.
    """

    def order(path: Path) -> tuple[int, int, int, str]:
        numbers = {"rank": 0, "attempt": 0}
        for parent in path.relative_to(directory.parent).parts[:-1]:
            name, _, value = parent.partition("=")
            if name in numbers and value.lstrip("-").isdigit():
                numbers[name] = int(value)
        match = _PART_NAME.search(path.name)
        return numbers["rank"], numbers["attempt"], int(match.group(1)) if match else 0, path.name

    if rank is None:
        found = list(directory.rglob("part-*.parquet"))
    else:
        found = list(directory.glob("part-*.parquet"))
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False) and not _other_rank_folder(entry, rank):
                    found.extend((directory / entry.name).rglob("part-*.parquet"))
    return sorted(found, key=order)


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


def write_evidence(evidence: Mapping[str, Any], path: str | Path) -> Path:
    """Write `evidence` as JSON through a temporary file and a rename, so a reader polling
    the file never sees half of it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(evidence, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _write_output(evidence: Mapping[str, Any], path: Path, named: bool, compact_window: int | None) -> Path:
    """`write_evidence`, or with `compact_window` set the compact file instead (`compact.write_compact`)."""
    if compact_window is None:
        return write_evidence(evidence, path)
    from .compact import write_compact

    return write_compact(evidence, path, window=compact_window, named=named)


class _Tail:
    """Reads what is new in a `.jsonl` file or a directory of Parquet parts."""

    def __init__(self, source: Path, rank: int | None) -> None:
        self.source = source
        self.rank = rank
        self.records: list[dict[str, Any]] = []
        self._offset = 0
        self._seen: set[Path] = set()

    def poll(self) -> bool:
        """Add newly written records; return whether there were any."""
        new: list[dict[str, Any]] = []
        if self.source.is_dir():
            for part in _parquet_parts(self.source, self.rank):
                if part not in self._seen:  # a part is written once and never changed
                    new.extend(_read_part(part))
                    self._seen.add(part)
        elif self.source.exists():
            with self.source.open("rb") as file:
                file.seek(self._offset)
                data = file.read()
            end = data.rfind(b"\n") + 1  # a last line with no newline is still being written
            for line in data[:end].splitlines():
                if line.strip():
                    try:
                        new.append(json.loads(line))
                    except ValueError:
                        pass  # a line torn by a killed process
            self._offset += end
        if self.rank is not None:
            new = [r for r in new if r.get("rank", 0) == self.rank]
        self.records.extend(new)
        return bool(new)


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
    "checkpoint": "checkpoint",
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
#   loss_scale_trouble    a scale below the floor 1; SCALE_SKIP_RUN skips in a row
#   data_stalled          no batch: DATA_STALL_STEPS steps in a row, 3; the same batch: the
#                         batch and DATA_STALL_STEPS steps that repeat it
#   epoch_order_repeated  the shortest pass that can be repeated has two different batches,
#                         and the next pass must match both
#   activation_spike      ACT_MIN_HISTORY earlier records and the one judged
#   activation_drift, dead_activations
#                         the first ACT_WINDOW records as the baseline and ACT_RECENT more
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
    "activation_drift": ACT_WINDOW + ACT_RECENT,
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
_STATUS_NEEDS = "the server's progress status (watchdog.json)"
_NO_STATUS = "no progress status (the server's watchdog.json): whether the job is still making progress is not checked"


def input_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """How many of these records carry each input the server's kinds are judged from
    (`INPUT_NEEDS`). Give the records that count (`load_records` without `all_attempts`).

    "Carries" means the field holds something a rule reads: an empty `layers` or
    `grad_norms` does not count, an empty `weights_unchanged` does (a weight check that
    found every parameter changed), and a probe that failed carries no attention.
    """
    counts = dict.fromkeys(INPUT_NEEDS, 0)
    for record in records:
        counts["steps"] += 1
        moe = record.get("moe")
        if isinstance(moe, Mapping) and any(sum(int(c) for c in router) > 0 for router in moe.values()):
            counts["moe"] += 1
        total = record.get("grad_norm") is not None or record.get("clip_grad_norm") is not None
        norms, layers = bool(record.get("grad_norms")), bool(record.get("layers"))
        counts["grad_total"] += total
        counts["update"] += total and isinstance(record.get("update_rms"), Mapping) and bool(record["update_rms"])
        counts["weight_check"] += record.get("weights_unchanged") is not None and norms
        counts["loss_scale"] += record.get("step_skipped") is not None
        counts["lr"] += _lr_values(record) is not None
        counts["data"] += isinstance(record.get("data"), Mapping)
        counts["checkpoint"] += isinstance(record.get("checkpoint"), Mapping)
        counts["layers"] += layers
        probe = record.get("probe")
        counts["attention"] += (
            isinstance(probe, dict) and "error" not in probe
            and isinstance(probe.get("attention"), dict) and bool(probe["attention"])
        )
        counts["numbers"] += layers or total or norms
        counts["replica_spread"] += bool(record.get("replica_spread"))
        system = record.get("system")
        for part in ("cuda", "nvml"):
            counts[part] += isinstance(system, dict) and isinstance(system.get(part), dict)
        wait, took = _finite_number(record.get("data_wait")), _finite_number(record.get("step_time"))
        counts["data_wait"] += wait is not None and took is not None and took > 0 and wait >= 0
    return counts


def _finite_number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _kinds_not_checked(not_checked: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Each kind a `not_checked` entry names, with the first entry that names it."""
    named: dict[str, Mapping[str, Any]] = {}
    for item in not_checked:
        if not isinstance(item, Mapping):
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
    """
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
        each = []
        for at, kind in enumerate(counts["not_enough_history_kinds"]):
            seen, needed = checks[kind].get("records"), checks[kind].get("records_needed")
            said = f"{kind} {seen} of {needed}" if seen is not None and needed is not None else kind
            if at == 0 and " of " in said:
                said += " records"
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
    key, _, value = source.resolve().name.partition("=")
    return int(value) if key == "rank" and value.isdecimal() else None


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
    names alone, with that part's path and how many parts have a name the server did not give."""
    # A walk of names with `os.scandir`, which builds no `Path` per part: with 200 parts a rank
    # (10,000 steps) that was 94% of a look at 64 ranks, and it grows with the run. The parts
    # counted, and the ones left out, are those `folder.rglob("part-*.parquet")` gives.
    newest: tuple[int, int] | None = None
    newest_path: str | None = None
    unknown = 0
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
                    if match is None or depth != 1 or key != "attempt" or not attempt.lstrip("-").isdigit():
                        unknown += 1
                    else:
                        at = (int(attempt), int(match.group(2)))
                        if newest is None or at > newest:
                            newest, newest_path = at, entry.path
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


def _given_thresholds(options: Mapping[str, Any]) -> dict[str, Any]:
    """The thresholds a caller gave: those of the six that are in `options` and not None.
    None is "not given" for every one of them, `top_k` included."""
    return {
        name: _top_k_option(options[name]) if name == "top_k" else options[name]
        for name in THRESHOLD_NAMES if options.get(name) is not None
    }


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def applied_thresholds(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """The six thresholds a server's report says it applied (its `thresholds`), in the shape
    `build_evidence` takes them. One the report does not list reads as the default: a server
    from before `--thresholds` applied the defaults and listed no `top_k`."""
    listed = (report or {}).get("thresholds")
    listed = listed if isinstance(listed, Mapping) else {}
    out = dict(THRESHOLD_DEFAULTS)
    for name in ("overload_threshold", "drift_tolerance", "starve_ratio"):
        value = listed.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            out[name] = value
    for name in ("starve_window", "update_window"):
        if _count(listed.get(name)):
            out[name] = listed[name]
    top_k = listed.get("top_k")
    if _count(top_k):
        out["top_k"] = top_k
    elif isinstance(top_k, Mapping) and top_k and all(_count(k) for k in top_k.values()):
        out["top_k"] = {str(name): k for name, k in top_k.items()}
    return out


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
    """
    if evidence.get("view") == ALL_RANKS_VIEW:
        who = "the server applied"
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
    """
    folder = Path(folder)
    stored, _, unknown = _newest_part(folder)

    def result(state: str, reason: str, report: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"state": state, "reason": reason, "stored": stored, "report": report}

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
    if reported < stored:
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
    `weight_trends` or router sizes: those are per-step data, which stays in the parts.
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
        stamps = [float(r["ts"]) for r in (_read_part(part) if part else ()) if isinstance(r.get("ts"), (int, float))]
        return max(stamps) if stamps else None

    last_step = report["last_step"] if isinstance(report.get("last_step"), int) else 0
    traincheck_summary, recorder_summary = _check_other_sources(
        incidents, stream_incidents, not_checked, last_step=last_step, newest_ts=newest_ts,
        traincheck=traincheck, flight_recorder=flight_recorder, flight_recorder_prefix=flight_recorder_prefix,
    )
    incidents.extend(i for i in record_incidents if i.get("kind") == "replica_drift")
    with_classes(incidents)
    applied = report.get("thresholds") or {}
    evidence = {
        "schema_version": 1,
        "run_id": report.get("run_id"),
        "run": _manifest.run_block(report.get("run_id"), _manifest.find_manifest(folder, report.get("run_id"))),
        "overload_threshold": applied.get("overload_threshold", OVERLOAD_THRESHOLD),
        "starve_window": applied.get("starve_window", STARVE_WINDOW),
        "starve_ratio": applied.get("starve_ratio", STARVE_RATIO),
        "incidents": incidents,
        "samples": [],
        # `thresholds`: the six the server judged by, which a reader takes unless it gave its own.
        "detection": {**_detection(look, "server", None), "moe_flagged": report.get("moe_flagged"),
                      "thresholds": applied_thresholds(report)},
        **({"not_checked": not_checked} if not_checked else {}),
        **({"flight_recorder": recorder_summary} if recorder_summary else {}),
        **({"traincheck": traincheck_summary} if traincheck_summary else {}),
        **({"watchdog": {k: status[k] for k in (
            "summary", "written_at_ms", "world_size", "pauses", "recommended_silent_seconds",
            "silent_floor_seconds", "critical_seconds",
        ) if k in status}} if status else {}),
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
        evidence["kind_checks"] = {
            kind: dict({**rank_checks, **others}[kind]) for kind in INCIDENT_CLASSES
        }
    evidence["run_state"] = _per_rank_run_state(evidence, status, _status_age(status), steps_judged=bool(report.get("steps")))
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


def _reason(state: str, code: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"state": state, "code": code, **extra, "detail": detail}


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
) -> dict[str, Any]:
    """The run's state from what the evidence holds: `{"state", "reasons"}`.

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
    An incident's class is looked up from its kind (`incident_class`): observations go to the
    `observations` reason, whose state is `healthy`, and every other incident, a kind this
    version does not know among them, to `incidents`. Codes are stable: `incidents`;
    `no_steps`, `report_not_current`, `ranks_unknown`, `status_old`, `server_stopped`,
    `status_summary`; `no_status`, `not_checked`; `observations`.
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
        for rank in sorted(expected - judged):
            unknown.setdefault(rank, "the server's status expects it and it has not connected")
    if unknown:
        ranks = sorted(unknown)
        reasons.append(_reason("unknown", "ranks_unknown", f"{len(ranks)} rank{'s' if len(ranks) != 1 else ''} without"
                               f" a current report ({_rank_ranges(ranks, 8)}): nothing is said about incidents on"
                               " them", ranks=ranks))
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
    return f"Run state: {state}" + (f" ({'; '.join(deciding)})" if deciding else "")


def _per_rank_run_state(evidence: Mapping[str, Any], status: Mapping[str, Any] | None,
                        status_age: float | None, steps_judged: bool) -> dict[str, Any]:
    # Evidence that says what was checked says whether a status was expected at all; without
    # the key (the server's report read by a watcher) one is, as before.
    checks = evidence.get("kind_checks")
    stall = checks.get("job_stalled") if isinstance(checks, Mapping) else None
    return run_state(
        incidents=evidence.get("incidents", ()), steps_judged=steps_judged, status=status, status_age=status_age,
        not_checked=evidence.get("not_checked", ()), detection=evidence.get("detection"),
        status_expected=not (isinstance(stall, Mapping) and stall.get("value") == "not_configured"),
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
        tuple(status.get("missing_ranks", ())),
        json.dumps(status.get("pauses", {}), sort_keys=True),
        status.get("recommended_silent_seconds"),
        tuple(
            (r.get("rank"), r.get("state"), r.get("last_step"), r.get("severity"), r.get("phase"))
            for r in status["ranks"]
        ),
        status_age is not None,
    )


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
        tuple(status.get("missing_ranks", ())) if status else (),
        json.dumps(incidents, sort_keys=True),
        status_age is not None,
        other_stamp,
    )


def _detectors_option(detectors: Any, server_binary: Any = None) -> str:
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
        warnings.warn(why, RuntimeWarning, stacklevel=3)
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
    detectors run here, as they do for a `.jsonl` file; that is repeated over the whole run
    at every change. While waiting for a report inside `report_grace`, status updates and
    external sources (`flight_recorder`, `traincheck`) still trigger fresh writes from the
    last current report. `default_rank`, when set alongside `rank=None`, locks onto that
    rank if a watched run directory ends up holding more than one `rank=*` folder.
    `evidence["detection"]` says which happened and why. `server_reports=False` always
    works the incidents out here.

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
    every record read so far, as the Python detectors read them all again. A missing binary or
    `pyarrow` raises `RuntimeError` before the first look, not at the first rebuild.
    `detectors="auto"` is decided once, before the first look, as `build_evidence` decides it.
    The default here is "python"; the command line's `--watch` asks for "rust" (`main`).

    `compact_window` writes the compact file instead of the full one (`compact.compact_evidence`,
    that many recorded steps each side of a MoE incident), as `<name>.compact.json` unless
    `output` names the file; `on_update` still gets the full evidence.
    `trigger` (a `trigger.Trigger`) fires once for each new incident: it is told of every look
    and handed each evidence written, and never raises into the watcher.
    """
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
                    if any(r.get("rank", 0) != picked_rank for r in tail.records):
                        emitted = None
                        emitted_grace = None
                        tail = _Tail(source, picked_rank)
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
                        in_grace = True
                        report_look = {**look, "report": last_current_report}
                    elif look["state"] == "behind":
                        look["state"] = "stale"
            grace_key = _grace_alert_key(status, status_age, other_stamp)
            if report_look is not None:
                report = report_look["report"]
                seen = (
                    "server",
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
                        tail = _Tail(source, picked_rank)
                        grown = tail.poll()
                        folder = _rank_folder(source, picked_rank) if server_reports else None
                        emitted = None
                        emitted_grace = None
                        waiting_since, was_current, last_current_report = None, False, None
                        adopted = {}
                seen = ("python", look and look["state"], stamp, other_stamp)
                if not tail.records or (not grown and seen == emitted):
                    continue
                here = {**adopted, **thresholds}
                evidence = build_evidence(
                    tail.records, **{"run_manifest": source, **build_options, **here,
                                     **({"watchdog": status_file} if status_file else {})}
                )
                if look is not None:
                    evidence["detection"] = {**_detection(look, "python", picked_rank),
                                             "thresholds": {**THRESHOLD_DEFAULTS, **here}}
                path = Path(output) if output else default_evidence_path(source, tail.records)
            if status_age is not None and "watchdog" in evidence:
                evidence["watchdog"]["not_written_for_seconds"] = round(status_age, 1)
            judged = bool(report_look["report"].get("steps")) if report_look is not None else bool(tail.records)
            evidence["run_state"] = _per_rank_run_state(evidence, status, status_age, steps_judged=judged)
            emitted = seen
            emitted_grace = grace_key
            written = _write_output(evidence, path, output is not None, compact_window)
            if on_update is not None:
                on_update(evidence, written)
            if trigger is not None:
                own = _own_rank(folder) if report_look is not None else tail.records[0].get("rank", 0)
                trigger.look(evidence, written, rank=own)
    except KeyboardInterrupt:
        pass
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
MAX_EXPECTED_RANKS = 100_000
# Lines of the text summary given to incidents on only some ranks before the rest are counted.
SUMMARY_PARTIAL_LINES = 8


def _rank_number(name: str) -> int | None:
    key, _, value = name.partition("=")
    return int(value) if key == "rank" and value.isdecimal() else None


def _discover_rank_folders(source: Path) -> dict[int, Path]:
    """The `rank=<n>` folders directly under a run folder, by rank number."""
    found: dict[int, Path] = {}
    try:
        with os.scandir(source) as entries:
            for entry in entries:
                number = _rank_number(entry.name)
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
    return reasons


class _RankMemory:
    """What the all-ranks watcher keeps about one rank between looks, as the single-rank
    watcher keeps it for its one rank: whether the rank's report was ever seen current, since
    when it has been behind, and the key it had the last time it was current."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.was_current = False
        self.waiting_since: float | None = None
        self.last_current_key: Any = None


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
#      `duplicate_batches`, which the server compares and puts in its status, when every
#      judged rank recorded `data` on some step, there are at least two, and a status was
#      read (`unavailable` without one).
#   6. When no rank is judged, or a judged rank's report has no usable `kind_checks` (a
#      server older than the key wrote it), there is no run-level object and no count:
#      `kind_checks_not_stated` says why and names the ranks. A count over the ranks that do
#      say would read as the run's.
#   7. `degraded` is as before and agrees with it: a judged rank's `not_checked` entry makes
#      a kind `unavailable` on that rank, so for the run, and already degrades the run state.
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
    `_report_evidence` applies before it takes them."""
    checks = report.get("kind_checks") if isinstance(report, Mapping) else None
    if isinstance(checks, Mapping) and all(
        isinstance(checks.get(kind), Mapping) and checks[kind].get("value") in KIND_CHECK_VALUES for kind in SERVER_KINDS
    ):
        return checks
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
    duplicates = _cross_rank_check(
        "duplicate_batches", brings(lambda rank: bool(rank_checks[rank][_DATA_KIND].get("records")), _DATA_KIND))
    if duplicates["value"] == "checked":
        if status is None:
            duplicates = {**_unavailable({"check": "status", "reason": _NO_STATUS}, enough_history=False,
                                         needs=duplicates["needs"])}
        else:
            # The status carries the comparison only when it finds a duplicate, so this is what
            # the ranks recorded, not proof that the server held every rank's batches.
            duplicates["note"] = ("the server compares the ranks whose stores it has read since it started, and its"
                                  " status names them only when it finds a duplicate")
    checks["duplicate_batches"] = duplicates
    return {kind: checks[kind] for kind in INCIDENT_CLASSES}


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
) -> dict[str, Any]:
    """The run-level part of the all-ranks view, built from one evidence per judged rank.

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
    """
    judged = sorted(per_rank)
    unknown = dict(unknown_ranks or {})
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
    combined: dict[str, Any] = {
        "rank_count": len(set(judged) | set(unknown)),
        "judged_ranks": judged,
        "unknown_ranks": sorted(unknown),
        "ranks_with_incidents": sorted({n for group in groups for n in group["ranks"]}),
        "last_step": {"min": min(steps), "max": max(steps)} if steps else None,
        "incident_groups": groups,
        "run_incidents": run_incidents,
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
    incidents = [group["incident"] for group in groups] + run_incidents
    combined["run_state"] = run_state(
        incidents=incidents, steps_judged=bool(steps),
        status=status, status_age=_status_age(status), not_checked=not_checked, unknown_ranks=unknown,
        judged_ranks=judged, status_expected=status_expected,
    )
    return combined


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
            entry["incidents"] = by_rank[number]
            unchecked = [dict(i) for i in report.get("not_checked") or ()
                         if isinstance(i, dict) and i.get("check") not in ("flight_recorder", "traincheck")]
            if unchecked:
                entry["not_checked"] = unchecked
        elif look["state"] in ("behind", "stale") and report:
            # True when it was written and says nothing of what came after: shown as that, apart from the
            # ranks that are current, and never as the rank's incidents.
            entry["last_report"] = {
                "attempt": report.get("attempt"), "last_step": report.get("last_step"), "steps": report.get("steps"),
                "incidents": [dict(i) for i in report.get("incidents") or () if isinstance(i, dict)],
            }
            with_classes(entry["last_report"]["incidents"])
        if number in streams:
            entry["stream"] = streams[number]
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
        stamps = [float(r["ts"]) for r in (_read_part(part) if part else ()) if isinstance(r.get("ts"), (int, float))]
        return max(stamps) if stamps else None

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
    judged_by: dict[str, dict[str, Any]] = {}
    for number in judged:
        values = applied_thresholds(looks[number]["report"])
        judged_by.setdefault(json.dumps(values, sort_keys=True), {"ranks": [], "thresholds": values})["ranks"].append(number)
    by_thresholds = [
        {"ranks": found["ranks"], "rank_ranges": _rank_ranges(found["ranks"]), "thresholds": found["thresholds"]}
        for found in judged_by.values()
    ]
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
        "ranks": entries,
        **({"not_checked": not_checked} if not_checked else {}),
        **({"flight_recorder": recorder_summary} if recorder_summary else {}),
        **({"traincheck": traincheck_summary} if traincheck_summary else {}),
        **({"watchdog": {k: status[k] for k in (
            "summary", "written_at_ms", "world_size", "pauses", "recommended_silent_seconds",
            "silent_floor_seconds", "critical_seconds",
        ) if k in status}} if status else {}),
    }
    _all_ranks_kind_checks(evidence, looks, status, flight_recorder=flight_recorder is not None,
                           traincheck=traincheck is not None)
    return evidence


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
    return run_state(incidents=incidents, steps_judged=steps, status=status, status_age=status_age,
                     not_checked=not_checked, unknown_ranks=unknown, judged_ranks=judged)


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
) -> tuple[dict[int, Path], dict[int, dict[str, Any]], tuple[Any, ...], tuple[Any, ...]]:
    """One look at every rank of a run folder: its `rank=<n>` folders, each rank's look
    (`_rank_look`), and the two keys the watcher compares with what it last wrote: every
    rank's own, and the same with a rank inside its wait held at the key it had when last
    current. `memory` is what the watcher keeps about each rank between looks, updated here;
    an empty one is a first look."""
    folders = _discover_rank_folders(source)
    absent = _ranks_without_folder(folders, status)
    numbers = sorted({*folders, *absent})
    for gone in set(memory) - set(numbers):
        del memory[gone]
    now = time.monotonic()
    looks: dict[int, dict[str, Any]] = {}
    full: list[tuple[int, Any]] = []
    calm: list[tuple[int, Any]] = []
    for number in numbers:
        held = memory.setdefault(number, _RankMemory())
        look, in_grace = _rank_look(folders.get(number), absent.get(number), held, thresholds, report_grace, now)
        looks[number] = look
        key = _rank_key(look)
        if look["state"] == "current":
            held.last_current_key = key
        full.append((number, key))
        # A rank inside its wait is a moment between a part and its report, as the
        # single-rank watcher treats it: it alone does not cause a write.
        calm.append((number, held.last_current_key if in_grace else key))
    return folders, looks, tuple(full), tuple(calm)


def _all_ranks_whole(
    source: Path,
    folders: Mapping[int, Path],
    looks: Mapping[int, Mapping[str, Any]],
    status: Mapping[str, Any] | None,
    status_age: float | None,
    other_sources: Mapping[str, Any],
) -> dict[str, Any]:
    """The whole all-ranks evidence of one look: `_all_ranks_evidence`, how old the status is
    when that is too old, and the run state."""
    evidence = _all_ranks_evidence(source, folders, looks, status, **other_sources)
    if status_age is not None and "watchdog" in evidence:
        evidence["watchdog"]["not_written_for_seconds"] = round(status_age, 1)
    evidence["run_state"] = _all_ranks_run_state(evidence, looks, status, status_age)
    return evidence


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
    The evidence is `{"view": "all_ranks", ...}`: see `_all_ranks_evidence` and the README.
    """
    source = _all_ranks_source(source)
    status = read_watchdog(_all_ranks_status_file(source, watchdog))
    thresholds = _given_thresholds({
        "overload_threshold": overload_threshold, "drift_tolerance": drift_tolerance,
        "starve_window": starve_window, "starve_ratio": starve_ratio, "top_k": top_k,
        "update_window": update_window,
    })
    folders, looks, _, _ = _look_at_ranks(source, status, {}, thresholds, REPORT_GRACE_SECONDS)
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
    state, the ranks present, or the status. The evidence is `{"view": "all_ranks", ...}`, not
    the per-rank evidence: see `_all_ranks_evidence` and the README for its keys.

    A rank's incidents count only while its report is current (`read_server_report`); a rank
    that is behind, stale, missing, unreadable, unusable, on other thresholds or has nothing
    stored is `unknown`, with its state and why, and is never listed among the ranks with no
    incidents. Rank folders may appear or disappear between looks. `server_reports=False`
    (`--recompute`) is refused, because it asks for the detectors to run here, a rank at a
    time and for the whole run each time; use `rank=<n>` for that. Each rank's report is taken
    with the thresholds its server applied, which the evidence lists (`thresholds`); any of the
    five given here (`overload_threshold`, `drift_tolerance`, `starve_window`, `starve_ratio`,
    `top_k`) must be the one a rank's server applied, or that rank is `other_thresholds`.
    `flight_recorder` and `traincheck` are read once for the run and their incidents are run-level.
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
            folders, looks, full, calm = _look_at_ranks(source, status, memory, thresholds, report_grace)
            if not looks and status is None:
                continue
            # Compared with what was written, not with the last look: a rank written while it
            # was in a wait, which then comes back with the key it had before, is rewritten.
            seen, calmed = (full, stamp), (calm, stamp)
            if seen == emitted or calmed == emitted:
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
    sizes: dict[str, int] = {}
    for record in records:
        for name, counts in record.get("moe", {}).items():
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


def _diversity_from_counts(counts: list[int], n_experts: int) -> float:
    total = sum(counts)
    if total <= 0 or n_experts <= 0:
        return 0.0
    shares = [c / total for c in counts if c > 0]
    entropy = -math.fsum(s * math.log(s) for s in shares)
    return math.exp(entropy) / n_experts


def detect_non_finite(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first step where a layer statistic or gradient norm is NaN or Inf.

    On steps a loss scaler skipped (`step_skipped`), Inf or NaN gradients are what the
    scaler expects and handles, so only activations are checked there; the skips
    themselves are counted in `detect_loss_scale_trouble`.
    """
    for record in records:
        bad_layers = [
            name
            for name, stats in record.get("layers", {}).items()
            if any(_is_non_finite(v) for v in stats.values())
        ]
        skipped = bool(record.get("step_skipped"))
        bad_grads = [
            name
            for name, norm in ({} if skipped else record.get("grad_norms", {})).items()
            if _is_non_finite(norm)
        ]
        if not skipped and _is_non_finite(record.get("grad_norm")) and not bad_grads:
            bad_grads.append("<total>")
        if not skipped and _is_non_finite(record.get("clip_grad_norm")):
            bad_grads.append("<total before clipping>")
        if bad_layers or bad_grads:
            return {
                "kind": "non_finite",
                "step": record["step"],
                "layers": bad_layers,
                "gradients": bad_grads,
            }
    return None


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
    """
    for record in records:
        spreads = record.get("replica_spread")
        if not spreads:
            continue
        diverged = {}
        for name, vals in spreads.items():
            values = [float(v) for v in vals]
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
                        for incident in stream_incidents:
                            if incident["assessment"] == "suspected_hang":
                                incident["assessment"] = "confirmed_hang"
                                incident["corroborated_by"] = "collective_hang"
    return traincheck_summary, recorder_summary


def _moe_samples(
    records: list[dict[str, Any]],
    router_sizes: Mapping[str, int],
    router_top_k: Mapping[str, int],
    overload_threshold: float,
) -> list[dict[str, Any]]:
    """The per-step MoE samples as the Python rules judge them: one for every record and
    router, with the experts found overloaded or dead there. `_mark_starved` adds the starved
    ones. The Rust detectors make the same list (`tensorquill-server detect --samples`)."""
    routers = list(router_sizes)
    history: dict[str, list[tuple[list[int], int, float]]] = {name: [] for name in routers}
    samples: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda r: r["step"]):
        step = int(record["step"])
        moe = record.get("moe", {})
        moe_stats = record.get("moe_stats", {})
        for layer_idx, router_name in enumerate(routers, 1):
            expected_experts = router_sizes[router_name]
            if router_name not in moe:
                counts = [0] * expected_experts
                missing = True
            else:
                counts = [int(c) for c in moe[router_name]]
                missing = sum(counts) == 0
            n_experts = len(counts)
            tokens = sum(counts)
            if missing or tokens <= 0:
                diversity: float | None = None
                overloaded: list[int] = []
                dead: list[int] = []
            else:
                shares = [c / tokens for c in counts]
                diversity = (
                    float(moe_stats[router_name]["diversity"])
                    if router_name in moe_stats and "diversity" in moe_stats[router_name]
                    else _diversity_from_counts(counts, n_experts)
                )
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
            samples.append(sample)
    return samples


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
) -> dict[str, Any]:
    """Build viewer-neutral detector evidence from captured step records.

    The evidence's `run` block (project, name, run ID) is read from the run manifest
    (`manifest.py`): the one beside `source` when that is a path, or `run_manifest`, which is
    the manifest itself, its file, or the `.jsonl` file or run folder it sits beside (for a
    caller that hands over records it has already read). A run without a manifest reads as
    project `uncategorized` with its run ID as its name.

    The detectors read one rank's steps. `rank` picks that rank out of a run directory or a
    list of records; records of several ranks with no `rank` given raise `ValueError`,
    because merged they repeat step numbers and give incidents that belong to neither rank.

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
    The six options above go to it as one JSON object (`--thresholds`), and a value the
    Rust rules refuse (a negative number, a window or `top_k` below 1) raises `ValueError`.
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
    detectors = _detectors_option(detectors, server_binary)  # "auto" becomes one of the two here
    if isinstance(update_window, bool) or not isinstance(update_window, int) or update_window < 1:
        raise ValueError(f"update_window must be a whole number of at least 1, not {update_window!r}")
    every_attempt = load_records(source, rank=rank, all_attempts=True)
    ranks = sorted({int(r.get("rank", 0)) for r in every_attempt})
    if len(ranks) > 1:
        raise ValueError(f"these records come from ranks {ranks}; pass rank= to read one of them")
    records = resolve_attempts(every_attempt)
    router_sizes = _router_sizes(records)
    routers = list(router_sizes)
    expert_count = max(router_sizes.values(), default=0)
    router_top_k = _router_top_k(records, routers, top_k)
    samples: list[dict[str, Any]]
    incidents: list[dict[str, Any]]
    not_checked: list[dict[str, Any]]
    rank_checks: dict[str, dict[str, Any]]
    from_rust: list[dict[str, Any]] | None = None
    if detectors == "rust" and every_attempt:
        from .server_detect import server_report

        asked = {
            "overload_threshold": overload_threshold, "drift_tolerance": drift_tolerance,
            "starve_window": starve_window, "starve_ratio": starve_ratio, "top_k": _top_k_option(top_k),
            "update_window": update_window,
        }
        report = server_report(every_attempt, thresholds=asked, binary=server_binary, samples=True)
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
    else:
        samples = _moe_samples(records, router_sizes, router_top_k, overload_threshold)
        _mark_starved(samples, starve_window, starve_ratio)
        incidents = []
        moe_hit = first_incident({"samples": samples})
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
                }
            )
        starved_hit = detect_starved_experts(samples)
        if starved_hit is not None:
            incidents.append(starved_hit)
        for detector in (
            detect_grad_spikes, lambda found: detect_update_spikes(found, window=update_window), detect_stalled_weights,
            detect_repeated_steps, detect_loss_scale_trouble,
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
        incidents.extend(hit for hit in detect_resume_state(every_attempt, not_checked=not_checked) if hit is not None)
        for data_detector in (detect_data_stalled, detect_epoch_order_repeated):
            data_hit = data_detector(records)
            if data_hit is not None:
                incidents.append(data_hit)
        incidents.extend(detect_activation_anomalies(records))
        incidents.extend(detect_attention_anomalies(records, not_checked=not_checked))
        non_finite_hit = detect_non_finite(records)
        if non_finite_hit is not None:
            incidents.append(non_finite_hit)
        incidents.extend(detect_system_incidents(records, not_checked))
        refused = refused_routers_not_checked(records)
        if refused is not None:
            not_checked.append(refused)
        not_read = update_not_read_not_checked(records)
        if not_read is not None:
            not_checked.append(not_read)
        rank_checks = kind_checks(
            input_counts(records), not_checked, starve_window=starve_window, update_window=update_window)
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
        stamps = [float(r["ts"]) for r in records if isinstance(r.get("ts"), (int, float))]
        return max(stamps) if stamps else None

    traincheck_summary, recorder_summary = _check_other_sources(
        incidents, stream_incidents, not_checked, last_step=last_step, newest_ts=newest_ts,
        traincheck=traincheck, flight_recorder=flight_recorder, flight_recorder_prefix=flight_recorder_prefix,
    )
    if from_rust is not None:
        drift_hit = next((i for i in from_rust if i.get("kind") == "replica_drift"), None)
    else:
        drift_hit = detect_replica_drift(records, tolerance=drift_tolerance)
    if drift_hit is not None:
        incidents.append(drift_hit)

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
        "run": _manifest.run_block(run_id, _manifest.find_manifest(
            run_manifest if run_manifest is not None else source, run_id)),
        "layer_count": len(routers),
        "expert_count": expert_count,
        "router_names": routers,
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
        **({"not_checked": not_checked} if not_checked else {}),
        **({"weight_trends": trends} if trends else {}),
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
    # Every kind, in the order `INCIDENT_CLASSES` lists them.
    evidence["kind_checks"] = {kind: {**rank_checks, **others}[kind] for kind in INCIDENT_CLASSES}
    evidence["run_state"] = _per_rank_run_state(evidence, status, _status_age(status), steps_judged=bool(records))
    return evidence


def first_incident(evidence: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the earliest MoE sample with an overloaded or dead expert, or None."""
    flagged = [
        sample
        for sample in evidence.get("samples", ())
        if not sample.get("missing")
        and sample.get("routed_tokens", 0) > 0
        and (sample.get("overloaded") or sample.get("dead"))
    ]
    if not flagged:
        return None
    return min(flagged, key=lambda sample: (sample["step"], sample["layer"]))


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

    Raises `compact.SamplesOmitted` for a step, or a step to compare it with, that compact
    evidence left out."""
    from .compact import require_table

    require_table(evidence, step, layer)
    expert_count = int(evidence.get("expert_count", 0))
    samples = {(int(s["step"]), int(s["layer"])): s for s in evidence.get("samples", ())}
    sample = samples.get((step, layer)) or _missing_sample(step, layer, expert_count)
    prior_steps = [s for s, l in samples if l == layer and s < step]
    active_priors = [
        s
        for s in prior_steps
        if not samples[s, layer].get("missing") and samples[s, layer].get("routed_tokens", 0) > 0
    ]
    if active_priors:
        previous = max(active_priors)
    elif prior_steps:
        previous = max(prior_steps)
    else:
        previous = step
    baseline = samples.get((previous, layer))

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
            f"**Layer {layer:02} expert load** · step {step} vs. {previous}",
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
    tail = _describe_stuck(incident) + tail
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
        parts.append("ranks " + ", ".join(str(r) for r in incident["missing_ranks"][:8]) + " never connected"
                     " (if only some ranks call tensorquill.init, ignore this)")
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
    waiting = len(lost) < len([r for r in incident.get("ranks", ()) if r.get("state") in ("silent", "lost", "unresponsive")])
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
                f" {top['value']:.2f}, {top['ratio']:.2f}× its earlier probes ({top['baseline']:.2f})")
    elif kind == "attention_uniform":
        what = (f"Attention gone uniform at step {incident['step']} in {top['module']}: mean entropy share"
                f" {top['baseline']:.2f} → {top['value']:.2f} (1.0 is uniform over every key)")
    else:
        what = (f"Attention logit growth at step {incident['step']} in {top['module']}: largest logit"
                f" {top['value']:.3g}, {top['ratio']:.2g}× its earlier probes ({top['baseline']:.3g})")
    probes = incident["affected_probes"]
    return (
        f"{what}{more}; held on the probe at step {incident['confirmed_step']} too ({probes} probe"
        f"{'s' if probes != 1 else ''} affected, each layer judged against {top['baseline_probes']} earlier"
        " probe" + ("s" if top["baseline_probes"] != 1 else "") + " of the fixed probe batch)"
    )


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
                f"the total norm, {incident['grad_norm']:.3g}, was {ratio:.1f}× its median"
                if isinstance(ratio, (int, float)) else f"the total norm was {incident['grad_norm']:.3g}"
            )
            return (
                f"Gradient spike in one parameter from step {incident['step']}: {top['name']} at"
                f" {top['ratio']:.0f}× its own median, and still at {SPIKE_FACTOR:g}× or more"
                f" {incident['sustained']} recorded steps in a row"
                + (f" ({others} more parameter{'s' if others > 1 else ''} from the same step)" if others > 0 else "")
                + f"; {total}, so the total alone did not show it"
            )
        where = ""
        if incident["parameters"]:
            top = incident["parameters"][0]
            where = (
                f", first in {top['name']} ({top['ratio']:.0f}× its median"
                + (f", rising since step {top['rising_since']}" if top["rising_since"] != incident["step"] else "")
                + ")"
            )
        more = incident["spike_count"] - 1
        return (
            f"Gradient spike at step {incident['step']}: norm {incident['grad_norm']:.3g},"
            f" {incident['ratio']:.0f}× the median of recent steps{where}"
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
                "; none of them has changed since the first weight check" if trained == 0
                else f"; {trained} of them had changed at an earlier check"
            )
        later = ""
        if len(onsets) > 1:
            parts = []
            for onset in onsets[1:4]:
                shown = ", ".join(onset["parameters"][:2])
                rest = onset["stalled"] - min(2, onset["stalled"])
                was = (
                    "after changing at earlier checks" if onset["trained_before"] == onset["stalled"]
                    else "never seen changing" if onset["trained_before"] == 0
                    else f"{onset['trained_before']} after changing at earlier checks"
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
                more += f" (the largest, {largest['stat']} {largest['ratio']:.0f}×, is in {largest['module']})"
        return (
            f"Activation spike at step {incident['step']} {'first ' if more else ''}in {top['module']}: {top['stat']}"
            f" {top['ratio']:.0f}× its recent median{more}"
            f"; {incident['affected_steps']} step{'s' if incident['affected_steps'] != 1 else ''} affected"
        )
    if kind == "activation_drift":
        top = incident["modules"][0]
        direction = "grew" if top["ratio"] > 1 else "shrank"
        return (
            f"Activation drift from step {incident['step']} in {top['module']}: {top['stat']}"
            f" {direction} to {top['ratio']:.2g}× its early level"
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
        words = {"missing": "not in the checkpoint", "empty": "started empty", "empty_in_checkpoint": "empty in the checkpoint",
                 "step_reset": "step counter back at 1", "position": "last_epoch far from the step",
                 "initial_scale": "back at its initial value"}
        found = "; ".join(f"{f['component']} {words.get(f['finding'], f['finding'])}" for f in change["findings"])
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
            f"{incident['consecutive_skips']} optimizer steps in a row skipped on Inf or NaN gradients"
            if incident["reason"] == "skips"
            else f"loss scale fell to {incident['grad_scale']:g}"
        )
        return (
            f"Loss scaling in trouble at step {incident['step']}: {why}; "
            f"{incident['skipped_steps']} of {incident['recorded_steps']} steps skipped in all"
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
        return text + (f", {bad} with NaN or Inf weights on some replica" if bad else "")
    return describe_system(incident)


def incident_summary(evidence: Mapping[str, Any]) -> str:
    """Human- and agent-readable summary matching the bake-off's text document format.

    Evidence from the all-ranks view (`"view": "all_ranks"`) gets the cross-rank summary."""
    if evidence.get("view") == ALL_RANKS_VIEW:
        return _all_ranks_summary(evidence)
    lines: list[str] = []
    incidents = evidence.get("incidents", ())
    moe_incident = next((i for i in incidents if i["kind"] == "moe_imbalance"), None)
    # Evidence built from the server's report has the incident but no per-step samples.
    moe = first_incident(evidence) or moe_incident
    if moe is not None:
        lines += _moe_lines(evidence, moe)
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
        text = (f"MoE imbalance at step {group['step']}, layer {incident.get('layer', 0):02}:"
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
    checked = all_ranks_checked_line(evidence)
    if checked:
        lines.append(checked)
    lines += [
        f"Not checked ({item['check']}): {item['reason']}"
        for item in evidence.get("not_checked", ())
        if item["check"] != "resume_lr"
    ]
    lines += _source_lines(evidence)
    applied = thresholds_line(evidence)  # only when a threshold applied is not the default
    if applied:
        lines.append(applied)
    return "\n".join(lines)


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


def _moe_lines(evidence: Mapping[str, Any], incident: Mapping[str, Any]) -> list[str]:
    counts, tokens = incident.get("counts"), incident.get("routed_tokens") or 1
    overloaded = ", ".join(
        f"Expert {expert} ({_percent(counts[expert - 1] / tokens)})" if counts else f"Expert {expert}"
        for expert in incident["overloaded"]
    )
    dead = ", ".join(f"{expert:02}" for expert in incident["dead"])
    diversity = float(incident.get("diversity") or 0.0)
    return [
        f"First sampled bad step {incident['step']}  ·  Layer {incident['layer']:02}"
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
) -> Path:
    """Write `evidence` (and optional Tier 0 `records`) to a Rerun `.rrd` with blueprint.

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
        for step in steps[1:]:
            rr.set_time("training_step", sequence=step)
            rr.log(
                "/incident/expert_load",
                rr.TextDocument(
                    expert_load_table(evidence, step, incident_layer),
                    media_type=rr.MediaType.MARKDOWN,
                ),
            )
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
        for record in records:
            rr.set_time("training_step", sequence=int(record["step"]))
            grad_norm = record.get("grad_norm")
            if isinstance(grad_norm, (int, float)) and math.isfinite(grad_norm):
                rr.log("/training/grad_norm", rr.Scalars(float(grad_norm)))
                training_paths["/training/grad_norm"] = "Gradient norm"
            clip_norm = _finite(record.get("clip_grad_norm"))
            if clip_norm is not None:
                rr.log("/training/clip_grad_norm", rr.Scalars(clip_norm))
                training_paths["/training/clip_grad_norm"] = "Pre-clipping gradient norm"
            weights = [v for v in record.get("weight_norms", {}).values() if _finite(v) is not None]
            if weights:
                rr.log("/training/weight_norm", rr.Scalars(math.sqrt(sum(v * v for v in weights))))
                training_paths["/training/weight_norm"] = "Weight norm"
            if record.get("lr"):
                rr.log("/training/lr", rr.Scalars(float(record["lr"][0])))
                training_paths["/training/lr"] = "Learning rate"
            scale = _finite(record.get("grad_scale"))
            if scale is not None:
                rr.log("/training/grad_scale", rr.Scalars(scale))
                training_paths["/training/grad_scale"] = "Loss scale"
            if "step_skipped" in record:
                rr.log("/training/step_skipped", rr.Scalars(float(bool(record["step_skipped"]))))
                training_paths["/training/step_skipped"] = "Step skipped by the loss scaler"
            for name, val in (record.get("scalars") or {}).items():
                value = _finite(val)
                if value is not None:
                    parts = "/".join(rr.escape_entity_path_part(p) for p in name.split("/") if p)
                    entity = f"/logged/{parts}"
                    rr.log(entity, rr.Scalars(value))
                    logged_paths[entity] = name
            probe = record.get("probe")
            if probe:
                probe_series |= _log_probe(rr, probe, probe_hist, probe_maps, probe_invalid)
            for layer_name, stats in record.get("layers", {}).items():
                for key, val in stats.items():
                    if isinstance(val, (int, float)) and math.isfinite(val):
                        entity = f"/layers/{layer_name}/{key}"
                        rr.log(entity, rr.Scalars(float(val)))
                        activation_paths.setdefault(key, set()).add(entity)

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

    rr.disconnect()
    return out_path


def _log_probe(
    rr: Any, probe: Mapping[str, Any], hist_paths: dict, map_paths: dict, invalid_series: list[bool]
) -> bool:
    """Log one probe pass: per-block histograms, attention maps, and per-layer attention
    logit and entropy series. Returns whether any series was logged."""
    import numpy as np

    for layer, hist in probe.get("histograms", {}).items():
        counts = hist.get("counts") or []
        lo, hi = _finite(hist.get("lo")), _finite(hist.get("hi"))
        if not counts or lo is None or hi is None:
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
    for layer, att in probe.get("attention", {}).items():
        part = rr.escape_entity_path_part(layer)
        logits = [v for v in (_finite(x) for x in att.get("max_logit", [])) if v is not None]
        if logits:
            rr.log(f"/probe/max_logit/{part}", rr.Scalars(max(logits)))
            series = True
        shares = [v for v in (_finite(x) for x in att.get("entropy_share", [])) if v is not None]
        if not shares and _finite(att.get("max_entropy")):
            # Probe records from before `entropy_share`: entropy over the log of all keys.
            top = float(att["max_entropy"])
            shares = [v / top for v in (_finite(x) for x in att.get("entropy", [])) if v is not None]
        if shares:
            rr.log(f"/probe/entropy/{part}", rr.Scalars(sum(shares) / len(shares)))
            series = True
        if att.get("invalid_rows"):
            # NaN or Inf attention scores: a numerical failure, kept apart from entropy.
            rr.log(f"/probe/invalid_rows/{part}", rr.Scalars(float(att["invalid_rows"])))
            invalid_series[0] = True
        if att.get("map"):
            entity = f"/probe/attention/{part}"
            rr.log(entity, rr.Tensor(np.asarray(att["map"], dtype=np.float32), dim_names=("query", "key")))
            map_paths[entity] = layer
    return series


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
    except ValueError as error:
        parser.error(str(error))
    if written is None and not args.watch:
        raise SystemExit(f"no rank=<n> folder under {args.input} and no watchdog.json: nothing to show")


def main() -> None:
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
        "only `--detectors python` makes it watch with the Python ones. With --watch the detectors run on every "
        "rebuild made here, over every record read so far. With --rank all the option changes nothing: that view "
        "takes every rank's incidents from its server's report and runs no detector, so it needs no binary",
    )
    from . import compact as _compact, trigger as _trigger

    _compact.add_arguments(parser)
    _trigger.add_arguments(parser)
    args = parser.parse_args()
    args.compact_window = _compact.window_from_args(args)  # None: the full file
    args.trigger = _trigger.from_args(args, parser)  # None without --on-incident or --webhook
    args.thresholds = _given_thresholds(vars(args))  # only the ones typed

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
            # The Rust path's own refusals: a binary that fails, thresholds its rules refuse.
            if detectors != "rust":
                raise
            parser.error(str(error))
        return

    # One read: the Rust detectors when they are installed, and otherwise the Python ones, said once.
    from .server_detect import resolve_detectors

    detectors, fallback = resolve_detectors(args.detectors or "auto")
    if fallback is not None:
        print(f"tensorquill: warning: {fallback}. `--detectors python` asks for the Python detectors without this "
              "warning", file=sys.stderr)
    records = load_records(args.input, rank=rank, all_attempts=True)
    try:
        evidence = build_evidence(
            records, **args.thresholds, watchdog=_watchdog_beside(args.input),
            flight_recorder=args.flight_recorder, flight_recorder_prefix=args.flight_recorder_prefix,
            traincheck=args.traincheck, detectors=detectors, run_manifest=args.input,
        )
    except (RuntimeError, ValueError) as error:
        if detectors != "rust":
            raise
        parser.error(str(error))
    out_json = _write_output(
        evidence, args.output or default_evidence_path(args.input, records), args.output is not None,
        args.compact_window,
    )
    print(incident_summary(evidence))
    print(run_state_line(evidence))
    print(f"wrote {'evidence' if args.compact_window is None else 'compact evidence'} -> {out_json}")
    if args.rerun is not None:
        export_rerun(evidence, args.rerun, records=resolve_attempts(records))
        print(f"wrote Rerun recording -> {args.rerun}")


if __name__ == "__main__":
    main()
