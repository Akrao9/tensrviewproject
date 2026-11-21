"""Candidate rules for `activation_drift`, each run over the development inputs. Measurement only:
nothing in the SDK is changed, and the arithmetic of today's rule is copied from
`evidence.detect_activation_anomalies` so that candidate D0 is today's rule.

For every candidate and input: the first step at which any module is at or over the candidate's
factor, how many steps are, and the largest ratio any module reaches. A candidate's ratio for a
module at a record is the larger of `std`'s and `abs_max/std`'s, each as max(now/base, base/now).

    D0        today: base = median of the module's first 50 records, now = median of its last 10
    D0@f      the same at another factor
    TRAIL@f   base = median of the 50 records before the last 10 (a baseline that follows the run)
    TRAILX@f  the same, judged only once that baseline window holds none of the module's first 50
              records (from its 110th record): the start of training is not a baseline
    SETTLE:s  base frozen from the first 50-record window whose own first 10 and last 10 medians
              are within a factor s of each other, on both statistics; not judged until one is found
"""

from __future__ import annotations

import json
import sys

import inputs
from inputs import ev

W, R = ev.ACT_WINDOW, ev.ACT_RECENT


def med(values):
    return ev._median(list(values))


def outlier(rows):
    return med([a / s for s, a in rows if s > 0] or [0.0])


def two(base_std, base_out, last):
    best = 0.0
    for base, now in ((base_std, med(p[0] for p in last)), (base_out, outlier(last))):
        if base > 0 and now > 0:
            best = max(best, now / base, base / now)
    return best


class Module:
    def __init__(self):
        self.past: list[tuple[float, float]] = []
        self.frozen = None       # D0's baseline
        self.settled: dict[float, tuple[float, float] | None] = {}

    def ratios(self, settles):
        past = self.past
        out = {}
        if len(past) < W + R:
            return out
        last = past[-R:]
        if self.frozen is None:
            first = past[:W]
            self.frozen = (med(p[0] for p in first), outlier(first))
        out["D0"] = two(*self.frozen, last)
        trail = past[-(W + R):-R]
        out["TRAIL"] = two(med(p[0] for p in trail), outlier(trail), last)
        if len(past) >= 2 * W + R:  # the baseline window holds none of the module's first W records
            out["TRAILX"] = out["TRAIL"]
        for s in settles:
            if self.settled.get(s) is None:
                window = past[-W:]
                head, tail = window[:R], window[-R:]
                quiet = True
                for a, b in ((med(p[0] for p in head), med(p[0] for p in tail)), (outlier(head), outlier(tail))):
                    if not (a > 0 and b > 0 and max(a / b, b / a) <= s):
                        quiet = False
                if quiet:
                    self.settled[s] = (med(p[0] for p in window), outlier(window), len(past))
            base = self.settled.get(s)
            # judged from ACT_RECENT records after the window that settled it
            if base is not None and len(past) >= base[2] + R:
                out[f"SETTLE:{s}"] = two(base[0], base[1], last)
        return out


CANDIDATES = [("D0", "D0", 3.0), ("D0@4", "D0", 4.0), ("D0@5", "D0", 5.0), ("D0@8", "D0", 8.0), ("D0@10", "D0", 10.0),
              ("TRAIL@2", "TRAIL", 2.0), ("TRAIL@3", "TRAIL", 3.0), ("TRAIL@4", "TRAIL", 4.0), ("TRAIL@5", "TRAIL", 5.0),
              ("TRAILX@2", "TRAILX", 2.0), ("TRAILX@3", "TRAILX", 3.0),
              ("SETTLE:1.25@3", "SETTLE:1.25", 3.0), ("SETTLE:1.5@3", "SETTLE:1.5", 3.0), ("SETTLE:2.0@3", "SETTLE:2.0", 3.0)]
SETTLES = (1.25, 1.5, 2.0)


def measure(rows):
    modules: dict[str, Module] = {}
    first = {name: None for name, _, _ in CANDIDATES}
    count = {name: 0 for name, _, _ in CANDIDATES}
    at_first = {name: 0 for name, _, _ in CANDIDATES}
    worst = {key: (0.0, None, None) for key in {k for _, k, _ in CANDIDATES}}
    judged_from: dict[str, int] = {}
    for record in rows:
        step = int(record["step"])
        over = {name: 0 for name, _, _ in CANDIDATES}
        for name, stats in (record.get("layers") or {}).items():
            std, amax, zero = (ev._finite(stats.get(k)) for k in ("std", "abs_max", "zero_frac"))
            if std is None or amax is None or zero is None:
                continue
            module = modules.setdefault(name, Module())
            module.past.append((std, amax))
            for key, ratio in module.ratios(SETTLES).items():
                judged_from.setdefault(key, step)
                if ratio > worst[key][0]:
                    worst[key] = (ratio, name, step)
                for cand, k, factor in CANDIDATES:
                    if k == key and ratio >= factor:
                        over[cand] += 1
        for cand in over:
            if over[cand]:
                count[cand] += 1
                if first[cand] is None:
                    first[cand], at_first[cand] = step, over[cand]
    unsettled = {s: sum(1 for m in modules.values() if m.settled.get(s) is None) for s in SETTLES}
    return {"modules": len(modules), "first": first, "count": count, "at_first": at_first,
            "worst": {k: (round(v[0], 2), v[1], v[2]) for k, v in worst.items()},
            "judged_from": judged_from, "unsettled": unsettled}


def signal(rows, fault):
    """For a seeded input: the largest change of any module's std or abs_max/std from the 10
    records before the fault step to the 10 from it on, whatever rule is used."""
    before: dict[str, list] = {}
    after: dict[str, list] = {}
    for record in rows:
        side = before if int(record["step"]) < fault else after
        for name, stats in (record.get("layers") or {}).items():
            std, amax = ev._finite(stats.get("std")), ev._finite(stats.get("abs_max"))
            if std is not None and amax is not None:
                side.setdefault(name, []).append((std, amax))
    best = (0.0, None)
    for name in before:
        a, b = before[name][-R:], after.get(name, [])[:R]
        if len(a) < R or len(b) < R:
            continue
        ratio = two(med(p[0] for p in a), outlier(a), b)
        if ratio > best[0]:
            best = (round(ratio, 2), name)
    return best


def main():
    only = set(sys.argv[1:])
    results = {}
    for name, rows, entry in inputs.everything():
        if only and entry["id"] not in only:
            continue
        if not any(r.get("layers") for r in rows):
            print(f"{name}: no layers recorded, not an input to this rule")
            continue
        found = measure(rows)
        role = inputs.role(name, entry)
        line = f"\n== {name} [{role}" + (f", fault at {entry['fault_step']}" if "fault_step" in entry else "") + "]"
        print(line, f"{len(rows)} records, {found['modules']} modules")
        if "fault_step" in entry and role == "seeded":
            print("   signal across the fault step (largest module, any rule):", signal(rows, entry["fault_step"]))
        for cand, key, factor in CANDIDATES:
            print(f"   {cand:<14} first step {str(found['first'][cand]):>5}  modules there {found['at_first'][cand]:>3}"
                  f"  steps at or over {found['count'][cand]:>5}  judged from step {found['judged_from'].get(key)}")
        print("   largest ratio by baseline:", found["worst"], " modules never settled:", found["unsettled"])
        results[name] = {"role": role, **found}
    (inputs.HERE / "generated" / "drift.json").write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
