"""Read a fixture's Rerun recording back and verify it against its evidence.

The tensor, the heatmap pixels, the diversity series and the expert tables are rebuilt from the
counts here. The detector summary and the heatmap colors come from the SDK, which also wrote
them, so those two checks show that the recording holds what the SDK computes, not that its
rules are right: `check_fixtures.py` tests the rules against the planted failures.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from rerun.chunk import RrdReader

from tensorquill import incident_summary, routing_colors

SHARE_ATOL = 3e-8


def read_samples(store, path, component):
    samples = {}
    for chunk in store.stream().filter(content=path, is_static=False).to_chunks():
        batch = chunk.to_record_batch()
        for step, values in zip(batch["training_step"].to_pylist(), batch[component].to_pylist(), strict=True):
            if step in samples or len(values) != 1:
                raise ValueError(f"Expected one {component} value at {path}, step {step}")
            samples[step] = values[0]
    return samples


def require_steps(samples, expected_steps, label):
    if samples.keys() != expected_steps:
        raise ValueError(f"{label}: expected steps {sorted(expected_steps)}, got {sorted(samples)}")


def expected_expert_rows(by_key, step, layer):
    """Recompute the expert-load rows from the counts, independently of the SDK's table code.

    The loads, changes and order come from the counts. The status label is the detector's own
    flag in the evidence, because the rule behind it (share, diversity drop or collapse) is
    not a function of one sample's counts; `test_fixtures.py` checks those flags.
    """
    previous = max(s for s, l in by_key if l == layer and s < step)
    sample, baseline = by_key[step, layer], by_key[previous, layer]
    rows = []
    for expert, (count, before) in enumerate(zip(sample["counts"], baseline["counts"], strict=True), 1):
        share = count / sample["routed_tokens"]
        delta = share - before / baseline["routed_tokens"]
        status = ("Dead" if expert in sample["dead"] else "Overloaded" if expert in sample["overloaded"]
                  else "Starved" if expert in sample["starved"] else "Idle" if count == 0 else "Active")
        rows.append((-delta, expert, [f"#{expert:03}", f"{share * 100:.2f}%", f"{delta * 100:+.2f}", status]))
    return previous, [cells for *_, cells in sorted(rows, key=lambda row: row[:2])]


def verify_expert_load(store, by_key, steps, layer, rune_tree=None):
    tables = read_samples(store, "/incident/expert_load", "TextDocument:text")
    expected_steps = {step for step in steps if any(s < step for s in steps)}
    require_steps(tables, expected_steps, "Expert load table")
    for step in sorted(expected_steps):
        previous, expected = expected_expert_rows(by_key, step, layer)
        lines = tables[step].splitlines()
        if lines[0] != f"**Layer {layer:02} expert load** · step {step} vs. {previous}":
            raise ValueError(f"Expert load heading differs at step {step}: {lines[0]!r}")
        rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[4:]]
        if rows != expected:
            raise ValueError(f"Expert load rows differ from the counts at step {step}")
    if rune_tree is not None:
        elements = json.loads(rune_tree.read_text(encoding="utf-8"))["elements"]
        cells = {}
        for element in elements:
            parts = element["path"].split("/table/viewport/rows/")
            if len(parts) == 2 and "text" in element:
                key, column = parts[1].split("/")[:2]
                cells.setdefault(key, {})[column] = (element["rect"][1], element["text"])
        visible = sorted(cells.values(), key=lambda row: row["expert"][0])
        rune_rows = [[row[column][1] for column in ("expert", "share", "delta", "status")] for row in visible]
        incident_step = min(step for step, l in by_key if by_key[step, l]["overloaded"] or by_key[step, l]["dead"])
        _, expected = expected_expert_rows(by_key, incident_step, layer)
        if not rune_rows or rune_rows != expected[:len(rune_rows)]:
            raise ValueError("Rune's visible expert rows differ from the Rerun table")
        print(f"Verified Rune's {len(rune_rows)} visible expert rows against the Rerun table.")
    return len(expected_steps)


def verify(directory, rune_tree=None):
    name = directory.name
    evidence = json.loads((directory / f"{name}.evidence.json").read_text(encoding="utf-8"))
    layers = range(1, evidence["layer_count"] + 1)
    shape = (evidence["layer_count"], evidence["expert_count"])
    by_key = {(sample["step"], sample["layer"]): sample for sample in evidence["samples"]}
    steps = {step for step, _ in by_key}
    if len(by_key) != len(evidence["samples"]) or set(by_key) != {(s, l) for s in steps for l in layers}:
        raise ValueError("The fixture must have exactly one sample for each step and layer")

    store = RrdReader(directory / f"{name}.rrd").store()
    shares = read_samples(store, "/routing/share", "Tensor:data")
    colors = read_samples(store, "/routing/status", "Image:buffer")
    formats = read_samples(store, "/routing/status", "Image:format")
    for label, samples in (("Routing shares", shares), ("Routing status", colors), ("Image dimensions", formats)):
        require_steps(samples, steps, label)

    largest_error = 0.0
    for step in sorted(steps):
        rows = [by_key[step, layer] for layer in layers]
        tensor = shares[step]
        if tuple(tensor["shape"]) != shape or tensor["names"] != ["layer", "expert"]:
            raise ValueError(f"Incorrect tensor axes at step {step}")
        expected = np.array([
            np.array(row["counts"], dtype=np.float64) / row["routed_tokens"] for row in rows
        ])
        actual = np.asarray(tensor["buffer"], dtype=np.float64).reshape(shape)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=SHARE_ATOL, equal_nan=False, err_msg=f"Step {step}")
        largest_error = max(largest_error, float(np.max(np.abs(actual - expected))))
        if (formats[step]["height"], formats[step]["width"]) != shape:
            raise ValueError(f"Incorrect heatmap dimensions at step {step}")
        expected_colors = np.array([list(routing_colors(row)) for row in rows], dtype=np.uint8)
        np.testing.assert_array_equal(colors[step], expected_colors.ravel(), err_msg=f"Status colors at step {step}")

    for layer in layers:
        path = f"/routing/layers/layer-{layer:02}"
        values = read_samples(store, path, "Scalars:scalars")
        require_steps(values, steps, path)
        for step, value in values.items():
            if value != by_key[step, layer]["diversity"]:
                raise ValueError(f"Diversity differs at step {step}, layer {layer}")

    summary = [
        value
        for chunk in store.stream().filter(content="/incident/summary", is_static=True).to_chunks()
        for value in chunk.to_record_batch()["TextDocument:text"].to_pylist()
    ]
    if summary != [[incident_summary(evidence)]]:
        raise ValueError("Recorded detector summary differs from the fixture")
    if rune_tree is not None:
        elements = json.loads(rune_tree.read_text(encoding="utf-8"))["elements"]
        labels = {element.get("key"): element.get("text") for element in elements}
        if summary[0][0].splitlines()[:2] != [labels["incident-summary"], labels["incident-experts"]]:
            raise ValueError("Rune and Rerun show different detector summaries")
        print("Verified matching summary text in the Rune tree and Rerun recording.")

    incident_layer = min(key for key, sample in by_key.items() if sample["overloaded"] or sample["dead"])[1]
    table_count = verify_expert_load(store, by_key, steps, incident_layer, rune_tree)

    print(f"Verified {len(by_key) * shape[1]:,} shares (max absolute error {largest_error:.3g}, tolerance {SHARE_ATOL:g}), "
          f"{len(by_key)} diversity values, all heatmap pixels, the detector summary, "
          f"and {table_count} layer {incident_layer:02} expert load tables.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?",
                        default=Path(__file__).resolve().parent / "artifacts" / "seeded",
                        help="a fixture's folder, named after the fixture (default: artifacts/seeded)")
    parser.add_argument("--rune-tree", type=Path, help="also compare the visible Rune summary")
    args = parser.parse_args()
    verify(args.directory, args.rune_tree)
