"""What each development input holds and what today's rules say of it. Measurement only."""

from __future__ import annotations

import sys

import inputs
from inputs import ev


def runs_of(flags: list[bool]) -> list[tuple[int, int]]:
    """(start index, length) of each run of True."""
    found, start = [], None
    for at, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = at
        elif not flag and start is not None:
            found.append((start, at - start))
            start = None
    return found


def main() -> None:
    only = set(sys.argv[1:])
    for name, rows, entry in inputs.everything():
        if only and entry["id"] not in only:
            continue
        steps = [int(r["step"]) for r in rows]
        layers = sum(1 for r in rows if r.get("layers"))
        moe = sum(1 for r in rows if r.get("moe"))
        print(f"\n== {name} [{inputs.role(name, entry)}"
              + (f", fault at {entry['fault_step']}" if "fault_step" in entry else "") + "]"
              f" {len(rows)} records, steps {steps[0]}..{steps[-1]}, layers on {layers}, moe on {moe}")
        found = {i["kind"]: i for i in ev.detect_activation_anomalies(rows)}
        for kind, hit in found.items():
            print(f"   {kind}: step {hit['step']}, {hit['affected']} modules, {hit['affected_steps']} steps;"
                  f" first {[(m['module'], m.get('stat'), round(m.get('ratio', 0), 2)) for m in hit['modules'][:2]]}")
        sizes = ev._router_sizes(rows)
        if not sizes:
            continue
        top_k = ev._router_top_k(rows, list(sizes), None)
        samples = ev._moe_samples(rows, sizes, top_k, ev.OVERLOAD_THRESHOLD)
        print(f"   routers {len(sizes)}: sizes {sorted(set(sizes.values()))}, top_k {sorted(set(top_k.values()))}")
        by_router: dict[str, list[dict]] = {}
        for sample in samples:
            by_router.setdefault(sample["router"], []).append(sample)
        for router, seq in by_router.items():
            seq = [s for s in seq if not s.get("missing") and s.get("routed_tokens", 0) > 0]
            flags = [bool(s.get("overloaded") or s.get("dead")) for s in seq]
            if not any(flags):
                continue
            runs = runs_of(flags)
            first = seq[runs[0][0]]
            share = max(first["counts"]) / max(1, sum(first["counts"]))
            print(f"   {router}: {sum(flags)} of {len(seq)} samples flagged, {len(runs)} runs;"
                  f" first at step {first['step']} (sample {runs[0][0]}) for {runs[0][1]} samples,"
                  f" overloaded {first['overloaded']}, dead {first['dead'][:4]}, top share {share:.3f};"
                  f" longest run {max(n for _, n in runs)} from step {seq[max(runs, key=lambda r: r[1])[0]]['step']};"
                  f" runs (step, len) {[(seq[a]['step'], n) for a, n in runs[:6]]}")


if __name__ == "__main__":
    main()
