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
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

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
# DIVERSITY_DROP times its median over the previous STARVE_WINDOW samples; the overloaded
# experts are then those whose fraction of tokens rose by at least OVERLOAD_JUMP over
# their average in that window. On a healthy Granite run the lowest diversity ratio seen
# was 0.82; the seeded collapse fell to 0.29.
DIVERSITY_DROP = 0.6
OVERLOAD_JUMP = 0.25
OVERLOAD_MIN_HISTORY = 10
# A gradient-norm spike is a step whose total norm is at least SPIKE_FACTOR times the
# median of the previous SPIKE_WINDOW recorded steps (once SPIKE_MIN_HISTORY are known).
SPIKE_FACTOR = 5.0
SPIKE_WINDOW = 50
SPIKE_MIN_HISTORY = 10
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
# A parameter is reported as not updating once its weights are exactly unchanged on this
# many weight checks in a row while it has a gradient. Checks run every 50 steps by default.
STALL_STEPS = 3
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


def detect_grad_spikes(
    records: list[dict[str, Any]],
    *,
    factor: float = SPIKE_FACTOR,
    window: int = SPIKE_WINDOW,
    min_history: int = SPIKE_MIN_HISTORY,
) -> dict[str, Any] | None:
    """Return the first gradient-norm spike and the parameters that spiked most, or None.

    Each parameter's norm is compared with its own median over the same window, so the
    report names where the spike is, not only that the total jumped. Without per-parameter
    norms (router-only capture) no parameter is named. The total is the pre-clipping
    `clip_grad_norm` whenever the run clips, since per-parameter norms are read after
    clipping, and `grad_norm` otherwise. `rising_since` is
    the first step in the window at which that parameter was already above half the
    spike factor, which points at where a spike started when it builds over several steps.
    """
    totals: list[float] = []
    per_param: dict[str, list[tuple[int, float]]] = {}
    spikes: list[dict[str, Any]] = []
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
        if len(totals) >= min_history:
            baseline = _median(totals[-window:])
            if baseline > 0 and total >= factor * baseline:
                spikes.append(
                    {
                        "step": step,
                        "grad_norm": total,
                        "baseline": baseline,
                        "ratio": total / baseline,
                        "parameters": _spiking_parameters(record, per_param, factor, window),
                    }
                )
        totals.append(total)
        for name, value in record.get("grad_norms", {}).items():
            norm = _finite(value)
            if norm is not None:
                per_param.setdefault(name, []).append((step, norm))
    if not spikes:
        return None
    first = spikes[0]
    return {
        "kind": "grad_spike",
        **first,
        "spike_count": len(spikes),
        "spike_steps": [spike["step"] for spike in spikes[:20]],
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


def _watchdog_beside(source: Any) -> Path | None:
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        return Path(source) / "watchdog.json"
    return None


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


def detect_watchdog_incidents(status: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Turn the server's progress status into incidents.

    `job_stalled`: no rank is making progress. `rank_silent`: some ranks are silent, lost
    (their stream dropped and did not return) or never connected while others progress.
    These are observations of the stream the server receives, and the cause is not known:
    a hang, a long evaluation or checkpoint pause, a crashed launcher and a dead telemetry
    sender all look the same. The server's `--silent-seconds` is the pause it tolerates.
    """
    if not status or status.get("summary") not in ("stalled", "degraded"):
        return []
    ranks = list(status.get("ranks", ()))
    stopped = [r for r in ranks if r.get("state") in ("silent", "lost", "unresponsive")]
    # Never "confirmed" from the stream alone: that takes a second source (see build_evidence).
    if any(r.get("state") in ("silent", "unresponsive") for r in stopped):
        assessment = "suspected_hang"
    elif stopped:
        assessment = "lost_connection"
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
    return [{
        "kind": "rank_silent",
        "step": min(behind) if behind else max(steps, default=0),
        "ranks": affected,
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
    """Return the first step where parameters with a gradient stopped changing, or None.

    Uses `weights_unchanged`, which the capture sets when a parameter's float64 norm is
    exactly what it was at the previous weight check. Equal norms are strong evidence
    but not proof that the weights are equal, so this is a warning: it is reported once
    that holds on `steps` checks in a row while the parameter's gradient is finite and
    non-zero, and it says whether an optimizer that stepped holds the parameters
    (`optimizer`: `known` with the stalled parameters no optimizer holds, or `unknown`
    with the reason, for DeepSpeed, Megatron, FSDP1, `auto_step=False` or captures that
    did not record it). A stalled parameter that no optimizer holds is the likely cause. Steps with a non-finite gradient norm are skipped,
    because mixed-precision loss scaling skips those optimizer steps on purpose. Captures
    without `weights_unchanged` (older files, or Apple MPS) are not checked: a float32 norm
    stays the same while bf16 weights change by a few elements.
    """
    streak: dict[str, int] = {}
    first: dict[str, Any] | None = None
    stalled_steps = 0
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
    if first is not None:
        first["stalled_steps"] = stalled_steps
    return first


def detect_activation_anomalies(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Activation spikes, drift and dead outputs, each judged against the module's history.

    - `activation_spike`: `abs_max` or `std` at least ACT_SPIKE_FACTOR times its median
      over the previous ACT_WINDOW records.
    - `activation_drift`: the median of the last ACT_RECENT records of `std`, or of the
      outlier ratio `abs_max / std`, moved ACT_DRIFT_FACTOR times above or below its
      median over the module's first ACT_WINDOW records.
    - `dead_activations`: the recent median `zero_frac` rose ACT_DEAD_RISE above its
      baseline, as when units stop firing.

    Returns at most one incident per kind: the first step it happened, the modules
    affected at that step, and how many steps were affected in all.
    """
    history: dict[str, list[tuple[float, float, float]]] = {}
    found: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}

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
                first, last = past[:ACT_WINDOW], past[-ACT_RECENT:]
                outlier = lambda rows: _median([a / s for s, a, _ in rows if s > 0] or [0.0])
                for key, base, now in (
                    ("std", _median([p[0] for p in first]), _median([p[0] for p in last])),
                    ("abs_max/std", outlier(first), outlier(last)),
                ):
                    if base > 0 and now > 0 and max(now / base, base / now) >= ACT_DRIFT_FACTOR:
                        drifts.append({"module": name, "stat": key, "baseline": base,
                                       "recent": now, "ratio": now / base})
                        break
                base_zero = _median([p[2] for p in first])
                now_zero = _median([p[2] for p in last])
                if now_zero - base_zero >= ACT_DEAD_RISE:
                    dead.append({"module": name, "baseline": base_zero, "recent": now_zero})
        report("activation_spike", step, sorted(spikes, key=lambda h: -h["ratio"]))
        report("activation_drift", step, drifts)
        report("dead_activations", step, dead)
    for kind, incident in found.items():
        incident["affected_steps"] = counts[kind]
    return list(found.values())


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
            records = [r for part in _parquet_parts(path) for r in _read_part(part)]
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


def _parquet_parts(directory: Path) -> list[Path]:
    """The server's part files under `directory`, ordered by rank, attempt and first step."""

    def order(path: Path) -> tuple[int, int, int, str]:
        numbers = {"rank": 0, "attempt": 0}
        for parent in path.relative_to(directory.parent).parts[:-1]:
            name, _, value = parent.partition("=")
            if name in numbers and value.lstrip("-").isdigit():
                numbers[name] = int(value)
        match = _PART_NAME.search(path.name)
        return numbers["rank"], numbers["attempt"], int(match.group(1)) if match else 0, path.name

    return sorted(directory.rglob("part-*.parquet"), key=order)


def _read_part(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError('reading Parquet parts needs pyarrow: pip install pyarrow') from error
    from .flight import row_to_record

    return [row_to_record(row) for row in pq.read_table(path).to_pylist()]


def write_evidence(evidence: Mapping[str, Any], path: str | Path) -> Path:
    """Write `evidence` as JSON through a temporary file and a rename, so a reader polling
    the file never sees half of it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(evidence, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


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
            for part in _parquet_parts(self.source):
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


def watch(
    source: str | Path,
    output: str | Path | None = None,
    *,
    interval: float = 2.0,
    rank: int | None = None,
    polls: int | None = None,
    on_update: Any = None,
    **build_options: Any,
) -> Path | None:
    """Rebuild the evidence whenever `source` grows, until interrupted.

    `source` is a `.jsonl` file a job is writing, or a directory `tensorquill-server` is
    writing Parquet parts into. Every `interval` seconds the new records are read and,
    if there were any, the detectors run over the whole run and the evidence file is
    replaced by rename. `polls` stops after that many looks, for scripts and tests;
    `on_update(evidence, path)` is called after each write. Returns the evidence path,
    or None if nothing was ever written.
    """
    source = Path(source)
    tail = _Tail(source, rank)
    written: Path | None = None
    done = 0
    status_file = build_options.get("watchdog")
    status_file = Path(status_file) if isinstance(status_file, (str, Path)) else _watchdog_beside(source)
    status_seen: Any = None
    try:
        while polls is None or done < polls:
            if done:
                time.sleep(interval)
            done += 1
            grown = tail.poll()
            # A hung job writes no new records, so the server's status file is also watched.
            changed = False
            if status_file is not None:
                # The server rewrites the file every few seconds; only a change of verdict,
                # of a rank's state or of the ranks missing is worth rebuilding for.
                current = read_watchdog(status_file)
                stamp = current and (
                    current["summary"], tuple(current.get("missing_ranks", ())),
                    json.dumps(current.get("pauses", {}), sort_keys=True), current.get("recommended_silent_seconds"),
                    tuple((r.get("rank"), r.get("state"), r.get("last_step"), r.get("severity"), r.get("phase"))
                          for r in current["ranks"]),
                )
                changed, status_seen = stamp is not None and stamp != status_seen, stamp
            if not grown and not (changed and tail.records):
                continue
            evidence = build_evidence(
                tail.records, **{**build_options, **({"watchdog": status_file} if status_file else {})}
            )
            written = write_evidence(evidence, output or default_evidence_path(source, tail.records))
            if on_update is not None:
                on_update(evidence, written)
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


def build_evidence(
    source: str | Path | Iterable[Mapping[str, Any]],
    *,
    overload_threshold: float = OVERLOAD_THRESHOLD,
    drift_tolerance: float = DRIFT_TOLERANCE,
    starve_window: int = STARVE_WINDOW,
    starve_ratio: float = STARVE_RATIO,
    top_k: int | Mapping[str, int] | None = None,
    watchdog: str | Path | Mapping[str, Any] | None = None,
    flight_recorder: str | Path | None = None,
    flight_recorder_prefix: str | None = None,
    traincheck: str | Path | None = None,
) -> dict[str, Any]:
    """Build viewer-neutral detector evidence from captured step records.

    `watchdog` is the server's `watchdog.json` (a path or the parsed document). When
    `source` is a directory of Parquet parts, `watchdog.json` beside the `rank=` folders is
    read without being named; its progress verdict becomes `job_stalled` or `rank_silent`.
    `flight_recorder` is a directory of PyTorch Flight Recorder dumps (one pickle per rank,
    `flight_recorder_prefix` the common start of their names): PyTorch's analyzer runs over
    them and its first problem becomes `collective_hang` or `collective_desync`. When the
    analyzer is missing (PyTorch before 2.10) or the dumps cannot be read, the check is
    listed under `not_checked` with the reason.
    `traincheck` is TrainCheck's `violations_summary.json` (or its results directory):
    the violated invariants become one `traincheck_violation` incident.

    `top_k` overrides the experts-per-token recorded as `moe_top_k`, for captures made
    before it was recorded: one value for every router, or a value per router name.
    """
    every_attempt = load_records(source, all_attempts=True)
    records = resolve_attempts(every_attempt)
    router_sizes = _router_sizes(records)
    routers = list(router_sizes)
    expert_count = max(router_sizes.values(), default=0)
    router_top_k = _router_top_k(records, routers, top_k)
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
                if k >= 3:
                    overloaded = _jump_overloaded(counts, k, diversity, history[router_name])
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
                if dead and not overloaded and n_experts > 1:
                    mean_share = 1.0 / n_experts
                    se_share = math.sqrt(mean_share * (1.0 - mean_share) / tokens)
                    collapse_threshold = max(2.0 * mean_share, mean_share + 3.0 * se_share)
                    overloaded = [
                        i for i, share in enumerate(shares, 1) if share > collapse_threshold
                    ]
                    rule = {"rule": "collapse", "threshold": collapse_threshold}
                window = history[router_name]
                window.append((counts, tokens, diversity))
                if len(window) > starve_window:
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

    incidents: list[dict[str, Any]] = []
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
    _mark_starved(samples, starve_window, starve_ratio)
    starved_hit = detect_starved_experts(samples)
    if starved_hit is not None:
        incidents.append(starved_hit)
    for detector in (detect_grad_spikes, detect_stalled_weights, detect_repeated_steps, detect_loss_scale_trouble):
        # Only this detector looks at what a later attempt replaced.
        detector_records = every_attempt if detector is detect_repeated_steps else records
        hit = detector(detector_records)
        if hit is not None:
            incidents.append(hit)
    not_checked: list[dict[str, Any]] = []
    resume_hit = detect_resume_lr_changes(every_attempt, not_checked=not_checked)
    if resume_hit is not None:
        incidents.append(resume_hit)
    incidents.extend(detect_activation_anomalies(records))
    non_finite_hit = detect_non_finite(records)
    if non_finite_hit is not None:
        incidents.append(non_finite_hit)
    status = read_watchdog(watchdog if watchdog is not None else _watchdog_beside(source))
    stream_incidents = detect_watchdog_incidents(status)
    incidents.extend(stream_incidents)
    last_step = max((int(r["step"]) for r in records), default=0)
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
                stamps = [float(r["ts"]) for r in records if isinstance(r.get("ts"), (int, float))]
                fresh = bool(stamps) and analysis["newest_dump_time"] >= max(stamps)
                if hit["kind"] == "collective_hang" and fresh and not analysis["incomplete_dumps"]:
                    # A stuck collective in PyTorch's own dump is a second source for a hang the
                    # stream only suggested, if the dump is complete and was written after the last
                    # step TensorQuill recorded. An old or partial dump proves nothing about now.
                    for incident in stream_incidents:
                        if incident["assessment"] == "suspected_hang":
                            incident["assessment"] = "confirmed_hang"
                            incident["corroborated_by"] = "collective_hang"
    drift_hit = detect_replica_drift(records, tolerance=drift_tolerance)
    if drift_hit is not None:
        incidents.append(drift_hit)

    if any(record.get("scalars") for record in records):
        for incident in incidents:
            context = scalar_context(records, incident["step"])
            if context:
                incident["scalars"] = context

    run_id = records[0].get("run_id") if records else None
    trends = weight_norm_trends(records)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "layer_count": len(routers),
        "expert_count": expert_count,
        "router_names": routers,
        "overload_threshold": overload_threshold,
        "top_k": router_top_k,
        "starve_window": starve_window,
        "starve_ratio": starve_ratio,
        "incidents": incidents,
        "samples": samples,
        **({"not_checked": not_checked} if not_checked else {}),
        **({"weight_trends": trends} if trends else {}),
        **({"flight_recorder": recorder_summary} if recorder_summary else {}),
        **({"traincheck": traincheck_summary} if traincheck_summary else {}),
        **({"watchdog": {k: status[k] for k in (
            "summary", "written_at_ms", "world_size", "pauses", "recommended_silent_seconds",
            "silent_floor_seconds", "critical_seconds",
        ) if k in status}} if status else {}),
    }


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
    """Markdown table of one layer's expert load, sorted by largest positive delta first."""
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


def _describe_stream(incident: Mapping[str, Any]) -> str:
    assessment = str(incident.get("assessment", "suspected_hang")).replace("_", " ")
    tail = f" [{incident.get('severity', 'warning')}, {assessment}"
    if incident.get("corroborated_by"):
        tail += f", corroborated by {incident['corroborated_by']}"
    tail += "]"
    if incident["kind"] == "job_stalled":
        ranks = incident["ranks"]
        quiet = max((r.get("seconds_since_progress") or 0 for r in ranks), default=0)
        reasons = sorted({_how_stopped(r) for r in ranks})
        return (
            f"Job stalled after step {incident['step']}: none of {len(ranks)} rank{'s' if len(ranks) != 1 else ''}"
            f" has sent a new step for at least {quiet:.0f} s ({'; '.join(reasons)})."
            " A hang, a long evaluation or checkpoint pause, or a killed job can look the same from here;"
            " PyTorch's Flight Recorder shows a stuck collective" + tail
        )
    parts = []
    for r in incident["ranks"][:4]:
        parts.append(f"rank {r['rank']} sent no new step since step {r['last_step']} ({_how_stopped(r)})")
    if incident["missing_ranks"]:
        parts.append("ranks " + ", ".join(str(r) for r in incident["missing_ranks"][:8]) + " never connected"
                     " (if only some ranks call tensorquill.init, ignore this)")
    return (
        f"Ranks out of step with the job: {'; '.join(parts)} while others reached step {incident['newest_step']}."
        " A hung rank, a killed process or a stuck telemetry sender look the same to the server" + tail
    )


def _describe(incident: Mapping[str, Any]) -> str | None:
    kind = incident["kind"]
    if kind == "grad_spike":
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
        if optimizer["state"] != "known":
            held = f"; whether an optimizer holds them is unknown ({optimizer['reason']})"
        elif optimizer["not_held_count"] == incident["stalled"]:
            held = "; no optimizer that stepped holds any of them, the likely cause"
        elif optimizer["not_held_count"]:
            held = f"; no optimizer holds {optimizer['not_held_count']} of them, the likely cause for those"
        else:
            held = (
                "; an optimizer holds all of them, so check for updates too small for the weight"
                " dtype (bf16) or a zero learning rate"
            )
        return (
            f"Weights possibly not updating from step {incident['step']}: {incident['stalled']} of"
            f" {incident['with_gradients']} parameters with gradients had a float64 norm that did not change for"
            f" {incident['consecutive']} weight checks in a row ({names}{f', and {extra} more' if extra > 0 else ''})"
            f"{held}; {incident['stalled_steps']} step{'s' if incident['stalled_steps'] != 1 else ''} affected."
            " Equal norms do not prove the weights are equal"
        )
    if kind == "activation_spike":
        top = incident["modules"][0]
        return (
            f"Activation spike at step {incident['step']} in {top['module']}: {top['stat']}"
            f" {top['ratio']:.0f}× its recent median"
            + (f", and {incident['affected'] - 1} more modules" if incident["affected"] > 1 else "")
            + f"; {incident['affected_steps']} step{'s' if incident['affected_steps'] != 1 else ''} affected"
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
    return None


def incident_summary(evidence: Mapping[str, Any]) -> str:
    """Human- and agent-readable summary matching the bake-off's text document format."""
    lines: list[str] = []
    incidents = evidence.get("incidents", ())
    moe = first_incident(evidence)
    if moe is not None:
        lines += _moe_lines(evidence, moe)
        moe_incident = next((i for i in incidents if i["kind"] == "moe_imbalance"), None)
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
    pauses = (evidence.get("watchdog") or {}).get("pauses") or {}
    recommended = (evidence.get("watchdog") or {}).get("recommended_silent_seconds")
    if pauses and recommended:
        seen = ", ".join(f"{phase} up to {stats['max_seconds']:.0f} s ({stats['count']}x)" for phase, stats in sorted(pauses.items()))
        lines.append(
            f"Pauses the job reported: {seen}. A silence threshold of {recommended:.0f} s (twice the longest)"
            " would have tolerated them: set the server's --silent-seconds to it or more. This is a measurement of"
            " what happened so far, not a guarantee"
        )
    notes = [
        f"Not checked ({item['check']}): {item['reason']}"
        for item in evidence.get("not_checked", ())
        if item["check"] != "resume_lr"
    ]
    if not lines:
        return "\n".join(["No incidents detected.", *notes])
    if evidence.get("layer_count"):
        lines.append(_legend(evidence))
    return "\n".join([*lines, *notes])


def _scalar_lines(incident: Mapping[str, Any]) -> list[str]:
    context = incident.get("scalars")
    if not context:
        return []

    def fmt(value: Any) -> str:
        return f"{value:.4g}" if isinstance(value, (int, float)) else str(value)

    parts = [
        f"{name} {fmt(entry['value'])}"
        + (f" (median before {fmt(entry['median_before'])})" if entry.get("median_before") is not None else "")
        for name, entry in sorted(context.items())
    ]
    return [f"  Logged at step {incident['step']}: " + ", ".join(parts)]


def _moe_lines(evidence: Mapping[str, Any], incident: Mapping[str, Any]) -> list[str]:
    tokens = incident["routed_tokens"] or 1
    overloaded = ", ".join(
        f"Expert {expert} ({_percent(incident['counts'][expert - 1] / tokens)})"
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
    """Write `evidence` (and optional Tier 0 `records`) to a Rerun `.rrd` with blueprint."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="captured .jsonl file from tensorquill.init, or a directory of Parquet parts "
        "written by tensorquill-server (runs/<run_id>, or a rank=<rank> directory in it)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="path to write viewer-neutral evidence JSON (default: <input>.evidence.json, "
        "or <run_id>.evidence.json for a directory)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="keep running: re-read the input when it grows and replace the evidence file",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between looks with --watch")
    parser.add_argument(
        "--rank",
        type=int,
        help="the rank to read (default: rank 0 for a directory holding several ranks)",
    )
    parser.add_argument("--rerun", type=Path, help="optional path to write a Rerun .rrd file")
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
    parser.add_argument(
        "--overload-threshold",
        type=float,
        default=OVERLOAD_THRESHOLD,
        help="expert share threshold for overload detection (default: 0.35)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="experts chosen per token, for captures made before moe_top_k was recorded",
    )
    args = parser.parse_args()

    rank = args.rank
    if rank is None and args.input.is_dir() and len(list(args.input.glob("rank=*"))) > 1:
        rank = 0  # the detectors read one rank's steps
        print("several ranks here; reading rank 0 (choose another with --rank)")
    if args.watch:
        if args.rerun is not None:
            parser.error("--rerun cannot be combined with --watch")
        shown: list[str] = []

        def report(evidence: dict[str, Any], path: Path) -> None:
            summary = incident_summary(evidence)
            if [summary] != shown:
                shown[:] = [summary]
                print(summary)
            print(f"updated {path}", flush=True)

        print(f"watching {args.input} every {args.interval:g} s; Ctrl-C to stop", flush=True)
        watch(
            args.input, args.output, interval=args.interval, rank=rank, on_update=report,
            overload_threshold=args.overload_threshold, top_k=args.top_k,
            flight_recorder=args.flight_recorder, flight_recorder_prefix=args.flight_recorder_prefix,
            traincheck=args.traincheck,
        )
        return

    records = load_records(args.input, rank=rank, all_attempts=True)
    evidence = build_evidence(
        records, overload_threshold=args.overload_threshold, top_k=args.top_k,
        watchdog=_watchdog_beside(args.input),
        flight_recorder=args.flight_recorder, flight_recorder_prefix=args.flight_recorder_prefix,
        traincheck=args.traincheck,
    )
    out_json = write_evidence(evidence, args.output or default_evidence_path(args.input, records))
    print(incident_summary(evidence))
    print(f"wrote evidence -> {out_json}")
    if args.rerun is not None:
        export_rerun(evidence, args.rerun, records=resolve_attempts(records))
        print(f"wrote Rerun recording -> {args.rerun}")


if __name__ == "__main__":
    main()
