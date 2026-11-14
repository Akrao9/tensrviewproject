"""Turn captured JSONL records into detector evidence and a Rerun recording.

The evidence dictionary uses the same schema as `experiments/moe-bakeoff`, so the stock
Rerun blueprint, `verify_rerun.py` and the Rune inspector can all read it directly, while
adding `incidents` for MoE routing imbalance, chronically starved experts, gradient
spikes, weights left unchanged by the optimizer, repeated step numbers, non-finite
activations/gradients and cross-GPU parameter drift.
"""

from __future__ import annotations

import argparse
import json
import math
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
# A parameter is reported as not updating once its weights are exactly unchanged on this
# many weight checks in a row while it has a gradient. Checks run every 50 steps by default.
STALL_STEPS = 3
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


def detect_grad_spikes(
    records: list[dict[str, Any]],
    *,
    factor: float = SPIKE_FACTOR,
    window: int = SPIKE_WINDOW,
    min_history: int = SPIKE_MIN_HISTORY,
) -> dict[str, Any] | None:
    """Return the first gradient-norm spike and the parameters that spiked most, or None.

    Each parameter's norm is compared with its own median over the same window, so the
    report names where the spike is, not only that the total jumped. `rising_since` is
    the first step in the window at which that parameter was already above half the
    spike factor, which points at where a spike started when it builds over several steps.
    """
    totals: list[float] = []
    per_param: dict[str, list[tuple[int, float]]] = {}
    spikes: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda r: r["step"]):
        total = _finite(record.get("grad_norm"))
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


def detect_stalled_weights(
    records: list[dict[str, Any]], *, steps: int = STALL_STEPS
) -> dict[str, Any] | None:
    """Return the first step where parameters with a gradient stopped changing, or None.

    Uses `weights_unchanged`, which the capture sets when a parameter's float64 norm is
    exactly what it was at the previous weight check. A parameter is reported once that
    holds on `steps` checks in a row while its gradient is finite and non-zero, for example because
    it was left out of the optimizer. Steps with a non-finite gradient norm are skipped,
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
                }
    if first is not None:
        first["stalled_steps"] = stalled_steps
    return first


def detect_repeated_steps(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return step numbers recorded more than once, as after a resume without `start_step`."""
    seen: dict[tuple[Any, int], int] = {}
    for record in records:
        key = (record.get("rank"), int(record["step"]))
        seen[key] = seen.get(key, 0) + 1
    repeated = sorted(step for (_, step), n in seen.items() if n > 1)
    if not repeated:
        return None
    return {"kind": "repeated_steps", "step": repeated[0], "repeated": len(repeated),
            "steps": repeated[:20]}


def load_records(source: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Read captured step records from a `.jsonl` file path or an iterable of dicts."""
    if isinstance(source, (str, Path)):
        lines = Path(source).read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    return [dict(item) for item in source]


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
    """Return the first step where a layer statistic or gradient norm is NaN or Inf."""
    for record in records:
        bad_layers = [
            name
            for name, stats in record.get("layers", {}).items()
            if any(_is_non_finite(v) for v in stats.values())
        ]
        bad_grads = [
            name
            for name, norm in record.get("grad_norms", {}).items()
            if _is_non_finite(norm)
        ]
        if _is_non_finite(record.get("grad_norm")) and not bad_grads:
            bad_grads.append("<total>")
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
    """Return the first step where cross-GPU parameter spread exceeds `tolerance`."""
    for record in records:
        spreads = record.get("replica_spread")
        if not spreads:
            continue
        diverged = {
            name: {"norm_spread": float(vals[0]), "sum_spread": float(vals[1])}
            for name, vals in spreads.items()
            if max(float(vals[0]), float(vals[1])) > tolerance
        }
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
) -> dict[str, Any]:
    """Build viewer-neutral detector evidence from captured step records.

    `top_k` overrides the experts-per-token recorded as `moe_top_k`, for captures made
    before it was recorded: one value for every router, or a value per router name.
    """
    records = load_records(source)
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
    for detector in (detect_grad_spikes, detect_stalled_weights, detect_repeated_steps):
        hit = detector(records)
        if hit is not None:
            incidents.append(hit)
    non_finite_hit = detect_non_finite(records)
    if non_finite_hit is not None:
        incidents.append(non_finite_hit)
    drift_hit = detect_replica_drift(records, tolerance=drift_tolerance)
    if drift_hit is not None:
        incidents.append(drift_hit)

    run_id = records[0].get("run_id") if records else None
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
        return (
            f"Weights not updating from step {incident['step']}: {incident['stalled']} of"
            f" {incident['with_gradients']} parameters with gradients were exactly unchanged for"
            f" {incident['consecutive']} weight checks in a row ({names}{f', and {extra} more' if extra > 0 else ''});"
            f" {incident['stalled_steps']} step{'s' if incident['stalled_steps'] != 1 else ''} affected"
        )
    if kind == "repeated_steps":
        return (
            f"Repeated step numbers from step {incident['step']}: {incident['repeated']} steps"
            " recorded more than once; pass start_step= to tensorlens.init when resuming"
        )
    if kind == "non_finite":
        parts = [*incident.get("layers", ()), *incident.get("gradients", ())]
        return f"NaN or Inf at step {incident['step']} in {', '.join(parts[:5])}"
    if kind == "replica_drift":
        return (
            f"Replicas diverged at step {incident['step']} in"
            f" {len(incident['parameters'])} parameters (tolerance {incident['tolerance']:g})"
        )
    return None


def incident_summary(evidence: Mapping[str, Any]) -> str:
    """Human- and agent-readable summary matching the bake-off's text document format."""
    lines: list[str] = []
    moe = first_incident(evidence)
    if moe is not None:
        lines += _moe_lines(evidence, moe)
    starved = _starved_line(evidence)
    if starved is not None:
        lines.append(starved)
    for incident in evidence.get("incidents", ()):
        if incident["kind"] not in ("moe_imbalance", "starved_experts"):
            text = _describe(incident)
            lines.append(text or f"Incident at step {incident['step']}: {incident['kind']}")
    if not lines:
        return "No routing or numerical incidents detected."
    if evidence.get("layer_count"):
        lines.append(_legend(evidence))
    return "\n".join(lines)


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
    run_id = str(evidence.get("run_id") or "tensorlens-run")

    rr.init("tensorlens", recording_id=run_id)
    rr.save(out_path)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.TextDocumentView(origin="/incident/summary", name="MoE routing imbalance"),
                rrb.Horizontal(
                    rrb.Tabs(
                        rrb.Spatial2DView(
                            origin="/routing/status", name="Routing status by layer and expert"
                        ),
                        rrb.TensorView(origin="/routing/share", name="Raw routing share"),
                        active_tab=0,
                    ),
                    rrb.Vertical(
                        rrb.TextDocumentView(
                            origin="/incident/expert_load",
                            name=f"Layer {incident_layer:02} expert load",
                        ),
                        rrb.TimeSeriesView(
                            origin="/routing/layers", name="Router diversity by layer"
                        ),
                        row_shares=[3, 2],
                    ),
                ),
                row_shares=[1, 4],
            ),
            collapse_panels=True,
        )
    )
    rr.log("/incident/summary", rr.TextDocument(incident_summary(evidence)), static=True)

    all_samples = list(evidence.get("samples", ()))
    steps = sorted({int(sample["step"]) for sample in all_samples})
    layer_count = int(evidence.get("layer_count") or max((int(s["layer"]) for s in all_samples), default=0))
    expert_count = int(
        evidence.get("expert_count")
        or max((len(s.get("counts", ())) for s in all_samples), default=0)
    )
    by_key = {(int(sample["step"]), int(sample["layer"])): sample for sample in all_samples}

    if steps and layer_count > 0 and expert_count > 0:
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

    if records is not None:
        for record in records:
            rr.set_time("training_step", sequence=int(record["step"]))
            grad_norm = record.get("grad_norm")
            if isinstance(grad_norm, (int, float)) and math.isfinite(grad_norm):
                rr.log("/training/grad_norm", rr.Scalars(float(grad_norm)))
            weights = [v for v in record.get("weight_norms", {}).values() if _finite(v) is not None]
            if weights:
                rr.log("/training/weight_norm", rr.Scalars(math.sqrt(sum(v * v for v in weights))))
            if record.get("lr"):
                rr.log("/training/lr", rr.Scalars(float(record["lr"][0])))
            for layer_name, stats in record.get("layers", {}).items():
                for key, val in stats.items():
                    if isinstance(val, (int, float)) and math.isfinite(val):
                        rr.log(f"/layers/{layer_name}/{key}", rr.Scalars(float(val)))

    rr.disconnect()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="captured .jsonl file from tensorlens.init")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="path to write viewer-neutral evidence JSON (default: <input>.evidence.json)",
    )
    parser.add_argument("--rerun", type=Path, help="optional path to write a Rerun .rrd file")
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

    records = load_records(args.input)
    evidence = build_evidence(
        records, overload_threshold=args.overload_threshold, top_k=args.top_k
    )
    out_json = args.output or args.input.with_suffix(".evidence.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(evidence, separators=(",", ":")) + "\n", encoding="utf-8")
    print(incident_summary(evidence))
    print(f"wrote evidence -> {out_json}")
    if args.rerun is not None:
        export_rerun(evidence, args.rerun, records=records)
        print(f"wrote Rerun recording -> {args.rerun}")


if __name__ == "__main__":
    main()
