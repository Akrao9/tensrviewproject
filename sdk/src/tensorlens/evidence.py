"""Turn captured JSONL records into detector evidence and a Rerun recording.

The evidence dictionary uses the same schema as `experiments/moe-bakeoff`, so the stock
Rerun blueprint, `verify_rerun.py` and the Rune inspector can all read it directly, while
adding `incidents` for MoE routing imbalance, non-finite activations/gradients and
cross-GPU parameter drift.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

OVERLOAD_THRESHOLD = 0.35
DRIFT_TOLERANCE = 1e-6
MISSING_COLOR = (48, 54, 64)


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
) -> dict[str, Any]:
    """Build viewer-neutral detector evidence from captured step records."""
    records = load_records(source)
    router_sizes = _router_sizes(records)
    routers = list(router_sizes)
    expert_count = max(router_sizes.values(), default=0)

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
                threshold = effective_overload_threshold(n_experts, tokens, overload_threshold)
                overloaded = [i for i, share in enumerate(shares, 1) if share > threshold]
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
            if count == 0:
                status = "Dead"
            elif expert in sample.get("overloaded", ()):
                status = "Overloaded"
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


def incident_summary(evidence: Mapping[str, Any]) -> str:
    """Human- and agent-readable summary matching the bake-off's text document format."""
    incident = first_incident(evidence)
    if incident is not None:
        tokens = incident["routed_tokens"] or 1
        overloaded = ", ".join(
            f"Expert {expert} ({incident['counts'][expert - 1] / tokens:.0%})"
            for expert in incident["overloaded"]
        )
        dead = ", ".join(f"{expert:02}" for expert in incident["dead"])
        diversity = float(incident.get("diversity") or 0.0)
        return (
            f"First sampled bad step {incident['step']}  ·  Layer {incident['layer']:02}"
            f"  ·  Diversity {diversity:.2f}\n"
            f"Overloaded: {overloaded or 'none'}  ·  Dead experts: {dead or 'none'}\n"
            "Heatmap: blue = active, amber = overloaded, red = dead"
        )
    other = evidence.get("incidents", ())
    if other:
        first = other[0]
        return f"Incident at step {first['step']}: {first['kind']}"
    return "No routing or numerical incidents detected."


def routing_colors(sample: Mapping[str, Any], expert_count: int | None = None):
    """RGB colors per expert matching the bake-off status heatmap.

    Missing or skipped router samples (and padded columns when routers have mixed expert
    counts) render as neutral slate gray `MISSING_COLOR` instead of dead-expert red.
    """
    counts = sample.get("counts", ())
    target = expert_count if expert_count is not None else len(counts)
    tokens = int(sample.get("routed_tokens") or 0)
    if sample.get("missing") or tokens <= 0:
        for _ in range(target):
            yield MISSING_COLOR
        return
    overloaded = set(sample.get("overloaded", ()))
    for expert, count in enumerate(counts, 1):
        if count == 0:
            yield (236, 84, 99)
        elif expert in overloaded:
            yield (255, 190, 94)
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
    args = parser.parse_args()

    records = load_records(args.input)
    evidence = build_evidence(records, overload_threshold=args.overload_threshold)
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
