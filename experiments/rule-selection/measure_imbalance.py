"""Candidate rules for `moe_imbalance` on a router that has not been trained yet, each run over
the development inputs. Measurement only. The samples are today's (`evidence._moe_samples`): a
sample is flagged when it has an overloaded or a dead expert. Only which flagged samples are
reported differs between candidates.

    I0          today: the earliest flagged sample
    SKIP0       a sample at the run's first step (step 0, nothing trained yet) is not reported
    SUSTAIN:k   reported once a router is flagged on k samples in a row, dated at the first
    START:g     a router flagged from its first sample at step 0 is in the state it started in:
                that leading run is not reported while it is shorter than g samples, and is
                reported, dated at its first sample, once it reaches g
"""

from __future__ import annotations

import sys

import inputs
from inputs import ev

CANDIDATES = ["I0", "SKIP0", "SUSTAIN:2", "SUSTAIN:3", "SUSTAIN:5", "START:5", "START:10", "START:50"]


def first_hit(seq: list[dict], candidate: str) -> tuple[int, int] | None:
    """(the step the incident is dated at, the step at which it could first be reported)."""
    flags = [bool(s.get("overloaded") or s.get("dead")) for s in seq]
    name, _, arg = candidate.partition(":")
    if name == "I0":
        return next(((s["step"], s["step"]) for s, f in zip(seq, flags) if f), None)
    if name == "SKIP0":
        return next(((s["step"], s["step"]) for s, f in zip(seq, flags) if f and s["step"] != 0), None)
    if name == "SUSTAIN":
        k, run = int(arg), 0
        for at, flag in enumerate(flags):
            run = run + 1 if flag else 0
            if run >= k:
                return seq[at - k + 1]["step"], seq[at]["step"]
        return None
    grace = int(arg)
    leading = 0
    if seq and seq[0]["step"] == 0:
        while leading < len(flags) and flags[leading]:
            leading += 1
    if leading >= grace:
        return seq[0]["step"], seq[grace - 1]["step"]
    return next(((s["step"], s["step"]) for s, f in list(zip(seq, flags))[leading:] if f), None)


def main() -> None:
    only = set(sys.argv[1:])
    for name, rows, entry in inputs.everything():
        if only and entry["id"] not in only:
            continue
        sizes = ev._router_sizes(rows)
        if not sizes:
            continue
        top_k = ev._router_top_k(rows, list(sizes), None)
        samples = ev._moe_samples(rows, sizes, top_k, ev.OVERLOAD_THRESHOLD)
        by_router: dict[str, list[dict]] = {}
        for sample in samples:
            if not sample.get("missing") and sample.get("routed_tokens", 0) > 0:
                by_router.setdefault(sample["router"], []).append(sample)
        print(f"\n== {name} [{entry.get('moe_role', inputs.role(name, entry))}"
              + (f", fault at {entry.get('moe_fault_step', entry.get('fault_step'))}" if "fault_step" in entry or "moe_fault_step" in entry else "")
              + f"] {len(rows)} records, {len(sizes)} routers, top_k {sorted(set(top_k.values()))}")
        for candidate in CANDIDATES:
            hits = {router: first_hit(seq, candidate) for router, seq in by_router.items()}
            hits = {router: hit for router, hit in hits.items() if hit is not None}
            if not hits:
                print(f"   {candidate:<10} nothing")
                continue
            router, (dated, seen) = min(hits.items(), key=lambda kv: (kv[1][1], kv[0]))
            print(f"   {candidate:<10} dated step {dated:>5}, reportable at {seen:>5}, on {router}; routers that fire: {len(hits)}")


if __name__ == "__main__":
    main()
