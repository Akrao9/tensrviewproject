"""Read the two nanoMoE arms written again on a later SDK (lane AE3, Nov 21, 2026).

For each arm, build the evidence of the fresh capture and of the first run's capture with the
Python detectors and with the Rust ones (`build_evidence(..., detectors=...)`, the second through
the server named by `--server`), compare the two engines key by key, and set the fresh records
against the first run's step for step. Nothing is fixed or judged here: it prints what it finds
and writes each evidence beside the summary in `--write`.

    PYTHONPATH=sdk/src .venv/bin/python experiments/nanomoe/fresh_compare.py \
        --fresh experiments/nanomoe/out --old <first run's out> --server <tensorquill-server> \
        --write <folder>

The records and evidence are large and are not kept in the repository.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tensorquill import build_evidence

ARMS = ("balance-on", "balance-off")
ENGINE_KEYS = ("incidents", "not_checked", "kind_checks", "run_state", "moe_source",
               "gradient_trend", "attention_trends", "update_trends", "weight_jumps")
# Keys of a record that differ between any two runs by construction.
RUN_KEYS = {"run_id", "attempt", "ts"}


def capture(folder: Path) -> Path:
    found = sorted(folder.glob("*.rank0.jsonl"))
    if len(found) != 1:
        raise SystemExit(f"{folder}: expected one rank-0 capture, found {[p.name for p in found]}")
    return found[0]


def diff(a: Any, b: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    """Every place `a` and `b` differ, exactly (floats too), as (path, a, b)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b), key=str):
            if key not in a or key not in b:
                out.append((f"{path}.{key}", a.get(key, "<absent>"), b.get(key, "<absent>")))
            else:
                out.extend(diff(a[key], b[key], f"{path}.{key}"))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [(f"{path}[len]", len(a), len(b))]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if a == b and type(a) is type(b) else [(path, a, b)]


def short(value: Any, width: int = 160) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= width else text[:width] + "..."


def incident_line(incident: dict[str, Any]) -> str:
    keep = {k: v for k, v in incident.items()
            if k not in ("evidence", "series", "samples", "history", "window", "values")}
    return short(keep, 600)


def engines(label: str, jsonl: Path, server: str, write: Path) -> dict[str, dict[str, Any]]:
    built = {}
    for engine in ("python", "rust"):
        kwargs = {"server_binary": server} if engine == "rust" else {}
        evidence = build_evidence(jsonl, detectors=engine, **kwargs)
        (write / f"{label}.{engine}.evidence.json").write_text(json.dumps(evidence, indent=1, default=str))
        built[engine] = evidence
    py, rs = built["python"], built["rust"]
    print(f"\n=== {label}: {jsonl}")
    print(f"  rust evidence says detectors={rs.get('detectors')!r}; python says {py.get('detectors')!r}")
    for key in ENGINE_KEYS:
        found = diff(py.get(key, "<absent>"), rs.get(key, "<absent>"), key)
        state = "equal" if not found else f"{len(found)} difference(s)"
        print(f"  engines on {key}: {state}" + ("" if key in py or key in rs else " (absent from both)"))
        for path, a, b in found[:10]:
            print(f"    {path}: python={short(a)} rust={short(b)}")
    other = sorted((set(py) | set(rs)) - set(ENGINE_KEYS) - {"detectors"})
    rest = [k for k in other if diff(py.get(k), rs.get(k))]
    print(f"  other keys that differ between engines (not part of the brief's list): {rest}")
    print(f"  run_state (python): {short(py.get('run_state'), 400)}")
    for incident in py.get("incidents", []):
        print(f"  incident: {incident_line(incident)}")
    print(f"  not_checked: {short(py.get('not_checked'), 1200)}")
    print(f"  kind_checks: {short(py.get('kind_checks'), 12000)}")
    print(f"  moe_source: {short(py.get('moe_source'), 600)}")
    for block in ENGINE_KEYS[5:]:
        print(f"  {block}: {short(py.get(block), 3000)}")
    return built


def records(jsonl: Path) -> list[dict[str, Any]]:
    with jsonl.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def against_first_run(arm: str, fresh: Path, old: Path) -> None:
    new_recs, old_recs = records(fresh), records(old)
    print(f"\n=== {arm}: fresh records against the first run's")
    print(f"  records: fresh {len(new_recs)}, first run {len(old_recs)}")
    by_step_old = {r.get("step"): r for r in old_recs}
    keys_new = sorted({k for r in new_recs for k in r})
    keys_old = sorted({k for r in old_recs for k in r})
    print(f"  keys only in fresh records: {sorted(set(keys_new) - set(keys_old))}")
    print(f"  keys only in the first run's: {sorted(set(keys_old) - set(keys_new))}")
    for r in new_recs:
        for key in ("moe_source", "layers_not_read"):
            if key in r:
                print(f"  step {r.get('step')} writes {key}: {short(r[key], 600)}")
    equal_steps, first_part, per_key = 0, None, {}
    sizes: dict[str, list[Any]] = {}  # key -> [abs gap, largest relative gap, where], over every shared step
    for r in new_recs:
        o = by_step_old.get(r.get("step"))
        if o is None:
            continue
        shared = (set(r) & set(o)) - RUN_KEYS
        found = [d for key in sorted(shared) for d in diff(r[key], o[key], key)]
        if not found:
            equal_steps += 1
            continue
        for path, a, b in found:
            top = path.split(".")[0].split("[")[0]
            per_key.setdefault(top, r.get("step"))
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
                gap = abs(a - b)
                rel = gap / max(abs(a), abs(b)) if max(abs(a), abs(b)) > 0 else 0.0
                worst = sizes.setdefault(top, [0.0, 0.0, None])
                if rel > worst[1]:
                    sizes[top] = [gap, rel, (r.get("step"), path)]
            else:
                sizes.setdefault(top, [math.nan, math.nan, (r.get("step"), path, "not a number")])
        if first_part is None:
            first_part = (r.get("step"), found[:5])
    print(f"  steps whose shared keys are equal exactly: {equal_steps} of {len(new_recs)}")
    print(f"  first step at which each shared key differs: {per_key}")
    for top, (gap, rel, where) in sorted(sizes.items()):
        print(f"  {top}: largest relative difference {rel:.3g} (absolute {gap:.3g}) at {where}")
    if first_part:
        step, found = first_part
        print(f"  first step that differs: {step}")
        for path, a, b in found:
            print(f"    {path}: fresh={short(a)} first run={short(b)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args()
    args.write.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        fresh, old = capture(args.fresh / arm), capture(args.old / arm)
        engines(f"{arm}.fresh", fresh, args.server, args.write)
        engines(f"{arm}.first-run", old, args.server, args.write)
        against_first_run(arm, fresh, old)


if __name__ == "__main__":
    main()
