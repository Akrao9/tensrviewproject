"""The recorded GPU captures against protocol 5's prototype layout (WAVE_AL_CONTRACT.md, section 2).

    PYTHONPATH=<worktree>/sdk/src .venv/bin/python -I experiments/protocol-v5/captures.py [--out results/captures.json]

For each of the seven captures in ~/Downloads/runs (read only, never written): the real keys of
`layers`, `moe_stats` and `probe` entries and their value types; the attention maps' shapes, whether
every value is exactly a float32, and whether any is not finite; then every record through
`record_to_row_v5` and `row_to_record_v5` (through a real Arrow batch and back, so Arrow's own
conversion is in the loop), checked with `==` and with the order of every map whose order the
readers use (layers, moe_stats, probe.attention, probe.histograms). Nothing is timed here.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow as pa  # noqa: E402

import prototype as p5  # noqa: E402

RUNS = Path.home() / "Downloads" / "runs"
CAPTURES = (
    "moe-rtx/granite-full.jsonl", "moe-rtx/granite.jsonl", "moe-rtx/granite-no-grads.jsonl",
    "moe-rtx/granite-router-only.jsonl", "dense-layer8/dense.jsonl", "trainer-check/trainer.jsonl",
    "trainer-check-block/trainer.jsonl",
)


def order_of(record: dict) -> dict:
    out = {"layers": list(record.get("layers") or {}), "moe_stats": list(record.get("moe_stats") or {})}
    probe = record.get("probe")
    if isinstance(probe, dict):
        out["probe.attention"] = list(probe.get("attention") or {})
        out["probe.histograms"] = list(probe.get("histograms") or {})
    return out


def f32(value: float) -> bool:
    return struct.unpack("f", struct.pack("f", value))[0] == value


def survey(path: Path) -> dict:
    keys = {"layers": collections.Counter(), "moe_stats": collections.Counter(), "probe": collections.Counter(),
            "probe.attention.entry": collections.Counter(), "probe.histograms.entry": collections.Counter(),
            "probe.head": collections.Counter()}
    types = collections.Counter()
    shapes = collections.Counter()
    maps = {"values": 0, "not_float32": 0, "not_finite": 0}
    records = []
    sizes = {"lines": 0, "bytes": 0, "probe_records": 0, "probe_json_bytes": 0, "probe_maps_json_bytes": 0,
             "probe_histograms_json_bytes": 0}
    empty_layers = 0
    with path.open() as handle:
        for line in handle:
            sizes["lines"] += 1
            sizes["bytes"] += len(line)
            record = json.loads(line)
            records.append(record)
            if record.get("layers") == {}:
                empty_layers += 1
            for key in ("layers", "moe_stats"):
                for entry in (record.get(key) or {}).values():
                    for field, value in entry.items():
                        keys[key][field] += 1
                        types[(key, field, type(value).__name__)] += 1
            probe = record.get("probe")
            if isinstance(probe, dict):
                sizes["probe_records"] += 1
                sizes["probe_json_bytes"] += len(json.dumps(probe, separators=(",", ":")))
                keys["probe"].update(list(probe))
                for entry in (probe.get("histograms") or {}).values():
                    keys["probe.histograms.entry"].update(list(entry))
                    sizes["probe_histograms_json_bytes"] += len(json.dumps(entry, separators=(",", ":")))
                for entry in (probe.get("attention") or {}).values():
                    keys["probe.attention.entry"].update(list(entry))
                    grid = entry.get("map")
                    if grid is not None:
                        sizes["probe_maps_json_bytes"] += len(json.dumps(grid, separators=(",", ":")))
                        shapes[(len(grid), len(grid[0]) if grid else 0, entry.get("query_len"), entry.get("key_len"))] += 1
                        for row in grid:
                            for value in row:
                                maps["values"] += 1
                                if isinstance(value, str) or not math.isfinite(value):
                                    maps["not_finite"] += 1
                                elif not f32(value):
                                    maps["not_float32"] += 1
                if isinstance(probe.get("head"), dict):
                    keys["probe.head"].update(list(probe["head"]))
    return {"records": records, "keys": {k: dict(v) for k, v in keys.items()},
            "types": {"/".join(k): v for k, v in types.items()},
            "map_shapes": {f"{a}x{b} (query_len {c}, key_len {d})": n for (a, b, c, d), n in shapes.items()},
            "maps": maps, "sizes": sizes, "empty_layers": empty_layers}


def round_trip(records: list[dict]) -> dict:
    schema = p5.flight_schema_v5()
    failures, order_failures, extra_rows = [], [], 0
    for start in range(0, len(records), 50):
        chunk = records[start:start + 50]
        rows = [p5.record_to_row_v5(r) for r in chunk]
        extra_rows += sum(1 for r in rows if r.get("extra_json"))
        batch = pa.RecordBatch.from_pylist(rows, schema=schema)
        back = [p5.row_to_record_v5(r) for r in batch.to_pylist()]
        for original, again in zip(chunk, back):
            if original != again:
                failures.append(original["step"])
            elif order_of(original) != order_of(again):
                order_failures.append(original["step"])
    return {"records": len(records), "not_equal": failures[:20], "not_equal_count": len(failures),
            "order_differs": order_failures[:20], "order_differs_count": len(order_failures),
            "rows_with_extra_json": extra_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {}
    for name in CAPTURES:
        path = RUNS / name
        if not path.exists():
            report[name] = {"missing": True}
            continue
        found = survey(path)
        records = found.pop("records")
        found["round_trip"] = round_trip(records)
        report[name] = found
        print(name, json.dumps({k: found[k] for k in ("sizes", "round_trip", "maps", "empty_layers")}))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
