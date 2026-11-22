"""Measurement only: activation_drift's ratio on recorded captures, at several factors.

The factor table is the rule as it was before wave AD (Nov 21, 2026): a module's baseline is its
first ACT_WINDOW records, its recent value its last ACT_RECENT, with the factor left open. It is
kept so the table in this folder's README can be made again. The current rule is different:
its baseline is the ACT_WINDOW records before the recent ACT_RECENT, and a module is judged
from its ACT_DRIFT_NEEDS-th record, so the baseline holds none of its first ACT_WINDOW records
(DETECTOR_RULES_CONTRACT.md, part 2). The line "detector today" is the current rule, called
through evidence.detect_activation_anomalies.

    python drift_measure.py balance-on=<capture> balance-off=<capture>
"""
import json, sys
from tensorquill import evidence as ev

FACTORS = (2.0, 3.0, 4.0, 5.0, 8.0)

def measure(path, label):
    rows = ev.resolve_attempts([json.loads(l) for l in open(path)])
    rows = [r for r in rows if r.get("rank", 0) == 0]
    rows.sort(key=lambda r: r["step"])
    outlier = lambda rs: ev._median([a / s for s, a in rs if s > 0] or [0.0])
    hist, base = {}, {}
    first = {f: None for f in FACTORS}          # first step any module is at or over the factor
    mods_at_first = {f: 0 for f in FACTORS}
    steps_over = {f: 0 for f in FACTORS}
    worst = {}                                   # module -> (ratio, stat, step)
    judged_steps = 0
    for r in rows:
        best = 0.0; n_over = {f: 0 for f in FACTORS}; judged = False
        for name, st in (r.get("layers") or {}).items():
            std, amax = ev._finite(st.get("std")), ev._finite(st.get("abs_max"))
            if std is None or amax is None or ev._finite(st.get("zero_frac")) is None:
                continue
            past = hist.setdefault(name, []); past.append((std, amax))
            if len(past) >= ev.ACT_WINDOW + ev.ACT_RECENT:
                judged = True
                if name not in base:
                    f0 = past[:ev.ACT_WINDOW]; base[name] = (ev._median([p[0] for p in f0]), outlier(f0))
                last = past[-ev.ACT_RECENT:]
                ratio, stat = 0.0, None
                for key, b, now in (("std", base[name][0], ev._median([p[0] for p in last])),
                                    ("abs_max/std", base[name][1], outlier(last))):
                    if b > 0 and now > 0 and max(now / b, b / now) > ratio:
                        ratio, stat = max(now / b, b / now), key
                if ratio > worst.get(name, (0,))[0]:
                    worst[name] = (ratio, stat, r["step"])
                for f in FACTORS:
                    if ratio >= f:
                        n_over[f] += 1
        judged_steps += judged
        for f in FACTORS:
            if n_over[f]:
                steps_over[f] += 1
                if first[f] is None:
                    first[f], mods_at_first[f] = r["step"], n_over[f]
    found = {i["kind"]: i for i in ev.detect_activation_anomalies(rows)}
    d = found.get("activation_drift")
    top = sorted(worst.items(), key=lambda kv: -kv[1][0])[:3]
    print(f"\n{label}: {len(rows)} records, steps {rows[0]['step']}..{rows[-1]['step']}, {len(hist)} modules, {judged_steps} judged steps")
    print("  detector today:", {k: (v["step"], v["affected"], v["affected_steps"]) for k, v in found.items()} or "nothing")
    print("  the rule before wave AD (baseline: a module's first", ev.ACT_WINDOW, "records):")
    for f in FACTORS:
        print(f"  factor {f:>3}: first step {first[f]}, modules there {mods_at_first[f]}, steps at or over {steps_over[f]}")
    print("  largest ratios:", [(n, round(v[0], 2), v[1], v[2]) for n, v in top])
    ratios = sorted(v[0] for v in worst.values())
    if ratios:
        print(f"  per-module largest ratio: median {ratios[len(ratios)//2]:.2f}, max {ratios[-1]:.2f}")

for path, label in [a.split("=", 1)[::-1] for a in sys.argv[1:]]:
    measure(path, label)
