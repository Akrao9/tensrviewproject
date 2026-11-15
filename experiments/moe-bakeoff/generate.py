"""Create shared numeric evidence and the matching Rerun recording."""

import argparse
import json
import math
from pathlib import Path

LAYERS = 60
EXPERTS = 128
TOKENS = 128_000
STEPS = (1750, 1800, 1850, 1900)
OVERLOAD = 0.35


def layer_sample(step, layer):
    counts = [TOKENS // EXPERTS] * EXPERTS
    if step >= 1850 and layer == 31:
        counts[:14] = [0] * 14
        counts[41] = 46_080
        remaining = [i for i, count in enumerate(counts) if count == 1000]
        equal, extra = divmod(TOKENS - counts[41], len(remaining))
        for index, expert in enumerate(remaining):
            counts[expert] = equal + (index < extra)
    assert sum(counts) == TOKENS
    shares = [count / TOKENS for count in counts]
    entropy = -math.fsum(share * math.log(share) for share in shares if share)
    return {
        "step": step,
        "layer": layer,
        "routed_tokens": TOKENS,
        "counts": counts,
        "diversity": math.exp(entropy) / EXPERTS,
        "overloaded": [i for i, share in enumerate(shares, 1) if share > OVERLOAD],
        "dead": [i for i, count in enumerate(counts, 1) if count == 0],
    }


def make_evidence():
    return {
        "schema_version": 1,
        "layer_count": LAYERS,
        "expert_count": EXPERTS,
        "overload_threshold": OVERLOAD,
        "samples": [layer_sample(step, layer) for step in STEPS for layer in range(1, LAYERS + 1)],
    }


def first_incident(evidence):
    return min(
        (sample for sample in evidence["samples"] if sample["overloaded"] or sample["dead"]),
        key=lambda sample: (sample["step"], sample["layer"]),
    )


def expert_load_table(evidence, step, layer):
    """Markdown table of one layer's expert load, matching the Rune inspector's default table.

    Rows are sorted by the change from the previous sample of the same layer, largest first,
    with ties in expert order, as Rune sorts them.
    """
    samples = {(sample["step"], sample["layer"]): sample for sample in evidence["samples"]}
    sample = samples[step, layer]
    previous = max(s for s, l in samples if l == layer and s < step)
    baseline = samples[previous, layer]
    rows = []
    for expert, count in enumerate(sample["counts"], 1):
        share = count / sample["routed_tokens"]
        delta = share - baseline["counts"][expert - 1] / baseline["routed_tokens"]
        if count == 0:
            status = "Dead"
        elif expert in sample["overloaded"]:
            status = "Overloaded"
        else:
            status = "Active"
        rows.append((-delta, expert, f"| #{expert:03} | {share * 100:.2f}% | {delta * 100:+.2f} | {status} |"))
    rows.sort(key=lambda row: row[:2])
    return "\n".join([
        f"**Layer {layer:02} expert load** · step {step} vs. {previous}",
        "",
        "| Expert | Load | Δ load (pp) | Status |",
        "| --- | --- | --- | --- |",
        *(row[2] for row in rows),
    ])


def incident_summary(evidence):
    incident = first_incident(evidence)
    overloaded = ", ".join(
        f"Expert {expert} ({incident['counts'][expert - 1] / incident['routed_tokens']:.0%})"
        for expert in incident["overloaded"]
    )
    dead = ", ".join(f"{expert:02}" for expert in incident["dead"])
    return (
        f"First sampled bad step {incident['step']}  ·  Layer {incident['layer']:02}"
        f"  ·  Diversity {incident['diversity']:.2f}\n"
        f"Overloaded: {overloaded}  ·  Dead experts: {dead}\n"
        "Heatmap: blue = active, amber = overloaded, red = dead"
    )


def routing_colors(sample):
    for expert, count in enumerate(sample["counts"], 1):
        if count == 0:
            yield (236, 84, 99)
        elif expert in sample["overloaded"]:
            yield (255, 190, 94)
        else:
            intensity = min(1.0, 100.0 * (count / sample["routed_tokens"]))
            yield (21 + int(15 * intensity), 44 + int(40 * intensity), 65 + int(54 * intensity))


def export_rerun(evidence, path):
    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb

    incident_layer = first_incident(evidence)["layer"]
    rr.init("tensorquill-moe-bakeoff", recording_id="seeded-moe")
    rr.save(path)
    rr.send_blueprint(rrb.Blueprint(
        rrb.Vertical(
            rrb.TextDocumentView(origin="/incident/summary", name="MoE routing imbalance"),
            rrb.Horizontal(
                rrb.Tabs(
                    rrb.Spatial2DView(origin="/routing/status", name="Routing status by layer and expert"),
                    rrb.TensorView(origin="/routing/share", name="Raw routing share"),
                    active_tab=0,
                ),
                rrb.Vertical(
                    rrb.TextDocumentView(origin="/incident/expert_load",
                                         name=f"Layer {incident_layer:02} expert load"),
                    rrb.TimeSeriesView(origin="/routing/layers", name="Router diversity by layer"),
                    row_shares=[3, 2],
                ),
            ),
            row_shares=[1, 4],
        ),
        collapse_panels=True,
    ))
    rr.log("/incident/summary", rr.TextDocument(incident_summary(evidence)), static=True)
    for step in STEPS[1:]:
        rr.set_time("training_step", sequence=step)
        rr.log("/incident/expert_load", rr.TextDocument(
            expert_load_table(evidence, step, incident_layer), media_type=rr.MediaType.MARKDOWN))
    for layer in range(1, LAYERS + 1):
        rr.log(f"/routing/layers/layer-{layer:02}", rr.SeriesPoints(names=[f"Layer {layer:02}"]), static=True)
    for step in STEPS:
        rr.set_time("training_step", sequence=step)
        samples = [sample for sample in evidence["samples"] if sample["step"] == step]
        shares = np.array([
            np.array(sample["counts"], dtype=np.float64) / sample["routed_tokens"]
            for sample in samples
        ], dtype=np.float32)
        rr.log("/routing/share", rr.Tensor(shares, dim_names=("layer", "expert"), value_range=(0.0, 0.36)))
        colors = np.array([list(routing_colors(sample)) for sample in samples], dtype=np.uint8)
        rr.log("/routing/status", rr.Image(colors, color_model="RGB", magnification_filter="Nearest"))
        for sample in samples:
            rr.log(f"/routing/layers/layer-{sample['layer']:02}", rr.Scalars(sample["diversity"]))
    rr.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path(__file__).parent / "artifacts")
    parser.add_argument("--rerun", action="store_true", help="also write the .rrd using rerun-sdk")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = make_evidence()
    (args.output / "stats.json").write_text(json.dumps(evidence, separators=(",", ":")) + "\n", encoding="utf-8")
    if args.rerun:
        export_rerun(evidence, args.output / "moe.rrd")


if __name__ == "__main__":
    main()
