"""The proposed rules, as the contract's generator states them, run once over the validation
capture: a run that was not used to choose them. Prints today's answer beside the proposed one."""

from __future__ import annotations

import importlib.util
import sys

import inputs
from inputs import ev

spec = importlib.util.spec_from_file_location("rule_cases", inputs.WORKTREE / "sdk/tests/fixtures/rules/make_cases.py")
rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rules)

for name, rows, entry in inputs.load(sys.argv[1] if len(sys.argv) > 1 else "validation-a"):
    steps = [int(r["step"]) for r in rows]
    print(f"{name}: {len(rows)} records, steps {steps[0]}..{steps[-1]}")
    today = {i["kind"]: i for i in ev.detect_activation_anomalies(rows)}
    was = today.get("activation_drift")
    print("  activation_drift today:   ", was and (was["step"], was["affected"], [(m["module"], round(m["ratio"], 2)) for m in was["modules"][:2]]))
    now = rules.drift(rows)
    print("  activation_drift proposed:", now and (now["step"], now["affected"], [(m["module"], round(m["ratio"], 2)) for m in now["modules"][:2]]))
    sizes = ev._router_sizes(rows)
    samples = ev._moe_samples(rows, sizes, ev._router_top_k(rows, list(sizes), None), ev.OVERLOAD_THRESHOLD)
    was = ev.first_incident({"samples": samples})
    flagged = sum(1 for s in samples if s.get("overloaded") or s.get("dead"))
    print("  moe_imbalance today:      ", was and (was["step"], was["router"], was["overloaded"], was["dead"][:4]), f"; {flagged} of {len(samples)} samples flagged")
    now = rules.imbalance(rows)
    print("  moe_imbalance proposed:   ", now and (now["step"], now["router"], now["overloaded"], now.get("since_start")))
    print("  gradient_trend:           ", rules.gradient_trend(rows))
