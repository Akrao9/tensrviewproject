"""The shared cases of `DETECTOR_RULES_CONTRACT.md`, written from reference implementations here.

    PYTHONPATH=sdk/src python sdk/tests/fixtures/rules/make_cases.py

Both engines are held to `cases.json`: the Python detectors and the server build what the page
states and must give each case's `expected`. The functions below are the page's rules written
plainly, once, so that the expected values come from one place; they are not the product's code.
`today` is what `eb5a71b` gave for the same input, where it gave anything.

Only the MoE samples (which experts are overloaded or dead in one sample) are taken from the SDK,
because the contract does not change them: it changes which flagged sample is reported.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from tensorquill import evidence as ev

HERE = Path(__file__).resolve().parent
SCORE_MIN = 10          # earlier finite values a score needs
SCORE_SCALE = 1.4826    # MAD to a standard deviation, for normal data
DRIFT_FACTOR, WINDOW, RECENT = 3.0, 50, 10
DRIFT_NEEDS = 2 * WINDOW + RECENT   # 110: the baseline window holds none of a module's first 50 records
START_GRACE = 10        # samples a router may stay in the state it started in
TREND_MIN = 6           # points a trend needs
LISTED = 10             # entries a measurement lists


# What `eb5a71b` gave for each rule case: the step of its incident, or None. Literals, measured once
# on that commit (Nov 21, 2026), so that this file gives the same `cases.json` after the rules change.
TODAY = {
    "drift_growth_from_scratch_is_not_a_shift": 140, "drift_a_step_change_is_named": 305, "drift_a_fall_is_named_too": 205,
    "drift_just_under_the_factor": None, "drift_a_slow_drift_is_not_seen": 981,
    "drift_a_change_in_the_first_records_is_not_seen": 75, "drift_not_judged_before_110_records": 100,
    "drift_the_outlier_ratio": 205, "drift_one_of_two_modules": 255,
    "start_two_samples_then_even": 0, "start_nine_samples_then_even": 0, "start_ten_samples_is_reported": 0,
    "start_then_a_later_overload": 0, "start_ends_and_comes_back": 0, "not_at_step_0_is_not_a_start": 5,
    "start_in_a_sparse_capture": 0, "one_router_starts_another_fails": 0, "a_healthy_router": None,
    "start_and_the_capture_ends": 0, "a_start_and_an_earlier_reportable_failure": 0,
}


def finite(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def median(values):
    return ev._median(list(values))


# -- part 1: the score -------------------------------------------------------------------------

def score(value, window):
    """{"score": number} or {"score": None, "score_note": why}: `value` against the earlier values."""
    earlier = [v for v in (finite(x) for x in window) if v is not None]
    if len(earlier) < SCORE_MIN:
        return {"score": None, "score_note": "short_history"}
    centre = median(earlier)
    spread = median([abs(v - centre) for v in earlier])
    if spread == 0:
        return {"score": None, "score_note": "no_dispersion"}
    return {"score": (value - centre) / (SCORE_SCALE * spread)}


DISPLAY = {"short_history": f"no score: fewer than {SCORE_MIN} earlier values",
           "no_dispersion": "no score: the earlier values had zero median absolute deviation"}


def display(scored):
    """What a reader shows for a score: the number to one decimal place, or why there is none."""
    return f"score {scored['score']:.1f}" if scored["score"] is not None else DISPLAY[scored["score_note"]]


# -- part 2: activation_drift ------------------------------------------------------------------

def outlier(rows):
    return median([a / s for s, a in rows if s > 0] or [0.0])


def drift(records):
    """The `activation_drift` incident, or None."""
    history: dict[str, list[tuple[float, float]]] = {}
    first, steps = None, 0
    for record in sorted(records, key=lambda r: r["step"]):
        hits = []
        for name, stats in record.get("layers", {}).items():
            std, amax, zero = (finite(stats.get(k)) for k in ("std", "abs_max", "zero_frac"))
            if std is None or amax is None or zero is None:
                continue
            past = history.setdefault(name, [])
            past.append((std, amax))
            if len(past) < DRIFT_NEEDS:
                continue
            last, trail = past[-RECENT:], past[-(WINDOW + RECENT):-RECENT]
            for stat, base, now, window in (
                ("std", median(p[0] for p in trail), median(p[0] for p in last), [p[0] for p in trail]),
                ("abs_max/std", outlier(trail), outlier(last), [a / s for s, a in trail if s > 0]),
            ):
                if base > 0 and now > 0 and max(now / base, base / now) >= DRIFT_FACTOR:
                    hits.append({"module": name, "stat": stat, "baseline": base, "recent": now,
                                 "ratio": now / base, **score(now, window)})
                    break
        if hits:
            steps += 1
            if first is None:
                first = {"kind": "activation_drift", "step": int(record["step"]), "modules": hits[:20], "affected": len(hits)}
    if first is not None:
        first["affected_steps"] = steps
    return first


def layers_series(values, *, module="blk", amax=None):
    """One record per value, steps from 0: a module whose `std` is the value."""
    return [{"step": i, "layers": {module: {"std": v, "abs_max": (amax[i] if amax else 4.0 * v), "zero_frac": 0.0}}}
            for i, v in enumerate(values)]


def wobble(i):
    return 1.0 + 0.01 * ((i * 7) % 5 - 2)  # a fixed pattern, so that no window is without dispersion


def drift_cases():
    def level(n, fn):
        return [fn(i) * wobble(i) for i in range(n)]

    sets = {
        "drift_growth_from_scratch_is_not_a_shift": (
            "a level that grows 1% a record for 700 records, about 1,000 times in all: 1.73 times within any 55 records",
            layers_series(level(700, lambda i: 1.01 ** i))),
        "drift_a_step_change_is_named": (
            "level 1 for 300 records, then 5", layers_series(level(400, lambda i: 1.0 if i < 300 else 5.0))),
        "drift_a_fall_is_named_too": (
            "level 3 for 200 records, then 0.5", layers_series(level(300, lambda i: 3.0 if i < 200 else 0.5))),
        "drift_just_under_the_factor": (
            "level 1 for 200 records, then 2.9", layers_series(level(300, lambda i: 1.0 if i < 200 else 2.9))),
        "drift_a_slow_drift_is_not_seen": (
            "a level that grows tenfold over 2,000 records, evenly: stated as what the rule gives up",
            layers_series(level(2000, lambda i: 10 ** (i / 2000)))),
        "drift_a_change_in_the_first_records_is_not_seen": (
            "level 1 for 70 records, then 5: by the 110th record the 50 before the last 10 are mostly at 5",
            layers_series(level(300, lambda i: 1.0 if i < 70 else 5.0))),
        "drift_not_judged_before_110_records": (
            "109 records with a step change at the 95th", layers_series(level(109, lambda i: 1.0 if i < 95 else 5.0))),
        "drift_the_outlier_ratio": (
            "`std` steady, `abs_max` four times it for 200 records and sixteen times after",
            layers_series(level(300, lambda i: 1.0), amax=[(4.0 if i < 200 else 16.0) * wobble(i + 3) for i in range(300)])),
    }
    two = layers_series(level(300, lambda i: 1.0), module="a")
    for i, record in enumerate(two):
        record["layers"]["b"] = {"std": (1.0 if i < 250 else 4.0) * wobble(i + 1), "abs_max": 4.0, "zero_frac": 0.0}
    sets["drift_one_of_two_modules"] = ("two modules, `b` goes from 1 to 4 at record 250", two)
    out = {}
    for name, (says, records) in sets.items():
        out[name] = {"part": "activation_drift", "says": says, "records": records,
                     "expected": {"incident": drift(records), "enough_history": len(records) >= DRIFT_NEEDS,
                                  "records_needed": DRIFT_NEEDS},
                     "today": {"step": TODAY[name]}}
    return out


# -- part 3: moe_imbalance ---------------------------------------------------------------------

def imbalance(records):
    """The `moe_imbalance` incident, or None: today's flagged samples, with the state a router
    started in excused for up to START_GRACE samples."""
    sizes = ev._router_sizes(records)
    samples = ev._moe_samples(records, sizes, ev._router_top_k(records, list(sizes), None), ev.OVERLOAD_THRESHOLD)
    by_router: dict[str, list[dict]] = {}
    for sample in samples:
        if not sample.get("missing") and sample.get("routed_tokens", 0) > 0:
            by_router.setdefault(sample["router"], []).append(sample)
    found = []
    for seq in by_router.values():
        flags = [bool(s.get("overloaded") or s.get("dead")) for s in seq]
        leading = 0
        if seq[0]["step"] == 0:
            while leading < len(flags) and flags[leading]:
                leading += 1
        if leading >= START_GRACE:
            found.append((seq[0], seq[START_GRACE - 1]["step"]))
            continue
        later = next((s for s, f in list(zip(seq, flags))[leading:] if f), None)
        if later is not None:
            found.append((later, None))
    if not found:
        return None
    # The incident is the one that could be reported first. `step` is when it began; for a router
    # collapsed since its start that is its first sample, and `reportable_step` is its tenth.
    hit, reportable = min(found, key=lambda pair: (pair[1] if pair[1] is not None else pair[0]["step"], pair[0]["layer"]))
    return {"kind": "moe_imbalance", "step": hit["step"], "layer": hit["layer"], "router": hit["router"],
            "diversity": hit["diversity"], "overloaded": hit["overloaded"], "dead": hit["dead"],
            **({"since_start": True, "reportable_step": reportable} if reportable is not None else {})}


def starting(records):
    """The routers still in the state they started in when the records end: flagged from step 0 on
    every sample so far, and fewer than START_GRACE of them. `kind_checks` names them."""
    sizes = ev._router_sizes(records)
    samples = ev._moe_samples(records, sizes, ev._router_top_k(records, list(sizes), None), ev.OVERLOAD_THRESHOLD)
    by_router: dict[str, list[dict]] = {}
    for sample in samples:
        if not sample.get("missing") and sample.get("routed_tokens", 0) > 0:
            by_router.setdefault(sample["router"], []).append(sample)
    return [{"router": router, "samples": len(seq)} for router, seq in by_router.items()
            if seq[0]["step"] == 0 and len(seq) < START_GRACE and all(s.get("overloaded") or s.get("dead") for s in seq)]


EVEN, LOADED = [32] * 8, [102, 22, 22, 22, 22, 22, 22, 22]  # 256 assignments; 102 is 39.8%, over the 35% line


def moe_records(plan, *, steps=None, routers=("r",)):
    """`plan[router]` is a string, one letter a sample: `x` overloaded, `.` even."""
    length = len(plan[routers[0]])
    steps = steps or list(range(length))
    return [{"step": step, "moe": {r: list(LOADED if plan[r][i] == "x" else EVEN) for r in routers},
             "moe_top_k": {r: 1 for r in routers}} for i, step in enumerate(steps)]


def imbalance_cases():
    sets = {
        "start_two_samples_then_even": ("overloaded at steps 0 and 1, then even: nanoMoE's case", moe_records({"r": "xx" + "." * 28})),
        "start_nine_samples_then_even": ("overloaded on its first nine samples, then even", moe_records({"r": "x" * 9 + "." * 21})),
        "start_ten_samples_is_reported": ("overloaded on its first ten samples: reported, dated at the first", moe_records({"r": "x" * 10 + "." * 20})),
        "start_then_a_later_overload": ("overloaded at steps 0 and 1, even to step 19, overloaded at 20", moe_records({"r": "xx" + "." * 18 + "x" * 10})),
        "start_ends_and_comes_back": ("overloaded at steps 0 to 2, even at 3, overloaded from 4", moe_records({"r": "xxx.x" + "x" * 10})),
        "not_at_step_0_is_not_a_start": ("a capture whose first sample is step 5, overloaded there: reported, as today",
                                         moe_records({"r": "xx" + "." * 10}, steps=list(range(5, 17)))),
        "start_in_a_sparse_capture": ("sampled every 50 steps and overloaded on all ten samples", moe_records({"r": "x" * 10}, steps=list(range(0, 500, 50)))),
        "one_router_starts_another_fails": ("router `a` overloaded at steps 0 and 1; router `b` even until step 3",
                                            moe_records({"a": "xx" + "." * 10, "b": "..." + "x" * 9}, routers=("a", "b"))),
        "a_healthy_router": ("even throughout", moe_records({"r": "." * 30})),
        "a_start_and_an_earlier_reportable_failure": (
            "router `a` overloaded from step 0 for good, reportable at its tenth sample (step 9); router `b` even until step 4: `b` is reported, since it could be reported first",
            moe_records({"a": "x" * 14, "b": "...." + "x" * 10}, routers=("a", "b"))),
        "start_and_the_capture_ends": ("overloaded on all five samples of a five-step capture: nothing is reported, and the check names the router as still in its starting state",
                                       moe_records({"r": "x" * 5})),
    }
    out = {}
    for name, (says, records) in sets.items():
        out[name] = {"part": "moe_imbalance", "says": says, "records": records,
                     "expected": {"incident": imbalance(records), "starting_state": starting(records)},
                     "today": {"step": TODAY[name]}}
    return out


# -- part 6: four measurements -----------------------------------------------------------------

def trend(steps, values):
    """The robust local trend, or None: a line through the median of the first third and the
    median of the last third, each at the median of its own steps."""
    n = len(values)
    if n < TREND_MIN:
        return None
    k = max(1, n // 3)
    left, right = (median(steps[:k]), median(values[:k])), (median(steps[-k:]), median(values[-k:]))
    if right[0] - left[0] <= 0:
        return None
    slope = (right[1] - left[1]) / (right[0] - left[0])
    return {"points": n, "first_step": steps[0], "last_step": steps[-1], "first": left[1], "last": right[1],
            "growth": right[1] / left[1] if left[1] > 0 else None,
            "per_100_steps": 100 * slope / left[1] if left[1] > 0 else None}


def gradient_trend(records):
    rows = sorted(records, key=lambda r: r["step"])
    key = "clip_grad_norm" if any("clip_grad_norm" in r for r in rows) else "grad_norm"
    pairs = [(int(r["step"]), finite(r.get(key))) for r in rows if not r.get("step_skipped")]
    pairs = [(s, v) for s, v in pairs if v is not None]
    found = trend([s for s, _ in pairs], [v for _, v in pairs])
    return found and {"series": key, **found}


def attention_trends(records):
    series: dict[str, tuple[list[int], list[float]]] = {}
    for r in sorted(records, key=lambda r: r["step"]):
        for module, stats in ((r.get("probe") or {}).get("attention") or {}).items():
            _, logit, _ = ev._attention_values(stats)
            if logit is not None:
                steps, values = series.setdefault(module, ([], []))
                steps.append(int(r["step"]))
                values.append(logit)
    found = [{"module": m, **t} for m, (s, v) in series.items() if (t := trend(s, v)) is not None]
    if not found:
        return None
    found.sort(key=lambda t: (-(t["growth"] if t["growth"] is not None else -math.inf), t["module"]))
    return {"statistic": "max_logit", "modules": len(found), "listed": found[:LISTED]}


def update_trends(records):
    """Each parameter's lowest update against its own earlier ones: `update_rms` over the median of
    the 50 records before, from 10 earlier finite values on."""
    past: dict[str, list[float]] = {}
    lowest: dict[str, dict] = {}
    for r in sorted(records, key=lambda r: r["step"]):
        for name, value in (r.get("update_rms") or {}).items():
            now = finite(value)
            if now is None:
                continue
            earlier = past.setdefault(name, [])
            if len(earlier) >= SCORE_MIN:
                base = median(earlier[-WINDOW:])
                if base > 0 and (name not in lowest or now / base < lowest[name]["ratio"]):
                    lowest[name] = {"name": name, "ratio": now / base, "step": int(r["step"]), "baseline": base, "update_rms": now}
            earlier.append(now)
    if not past:
        return None
    return {"parameters": len(past), "judged": len(lowest),
            "lowest": sorted(lowest.values(), key=lambda p: (p["ratio"], p["name"]))[:LISTED]}


def weight_jumps(records):
    """The largest change of one tensor's norm between two consecutive weight checks."""
    checks = [r for r in sorted(records, key=lambda r: r["step"]) if r.get("weight_norms")]
    if len(checks) < 2:
        return None
    best: dict[str, dict] = {}
    for before, after in zip(checks, checks[1:]):
        for name, value in after["weight_norms"].items():
            a, b = finite(before["weight_norms"].get(name)), finite(value)
            if a is None or b is None or a <= 0 or b <= 0:
                continue
            ratio = max(b / a, a / b)
            if name not in best or ratio > best[name]["ratio"]:
                best[name] = {"name": name, "ratio": ratio, "step": int(after["step"]),
                              "previous_step": int(before["step"]), "before": a, "after": b}
    if not best:
        return None
    return {"checks": len(checks), "parameters": len(best),
            "largest": sorted(best.values(), key=lambda p: (-p["ratio"], p["name"]))[:LISTED]}


def probe(step, logits):
    return {"step": step, "probe": {"attention": {m: {"entropy_share": [0.5, 0.5], "max_logit": [v, v / 2]} for m, v in logits.items()}}}


def measurement_cases():
    grad_up = [{"step": i, "grad_norm": 1.0 * 1.005 ** i * wobble(i)} for i in range(300)]
    grad_flat = [{"step": i, "grad_norm": 38.45 * wobble(i)} for i in range(100)]
    grad_clip = [{"step": 10 * i, "grad_norm": 1.0, "clip_grad_norm": 2.0 + 0.02 * i, **({"step_skipped": True} if i == 7 else {})} for i in range(30)]
    probes = [probe(1000 * (i + 1), {"attn.0": 10.0 * 1.25 ** i, "attn.1": 10.0}) for i in range(8)]
    updates = [{"step": i, "update_rms": {"w": 1e-3 * wobble(i), "v": (1e-3 if i < 60 else 3e-10) * wobble(i)}} for i in range(100)]
    weights = [{"step": 50 * i, "weight_norms": {"a": 5.3 if i < 4 else 2.1e7, "b": 1.0 + 0.0007 * i}} for i in range(6)]
    sets = {
        "gradient_trend_a_steady_rise": ("B4", "the total gradient norm rising 0.5% a step for 300 steps", grad_up, "gradient_trend", gradient_trend),
        "gradient_trend_a_plateau": ("B4", "a norm that sits at 38.45, as B4's re-creation does once its climb is over: the trend is flat", grad_flat, "gradient_trend", gradient_trend),
        "gradient_trend_reads_the_pre_clip_norm_and_real_steps": ("B4", "records ten steps apart that clip, one of them a skipped step: the pre-clipping norm, the actual step numbers", grad_clip, "gradient_trend", gradient_trend),
        "gradient_trend_under_six_points": ("B4", "five records: no trend, the block is absent", grad_up[:5], "gradient_trend", gradient_trend),
        "attention_trends_a_slow_rise": ("Q1", "eight probes 1,000 steps apart; one module's largest logit rises 25% a probe, which `attention_logit_growth` does not report", probes, "attention_trends", attention_trends),
        "attention_trends_under_six_probes": ("Q1", "four probes: no trend, the block is absent", probes[:4], "attention_trends", attention_trends),
        "update_trends_one_parameter_collapses": ("Q2", "two parameters' update sizes; one falls to 3e-7 of its level at step 60 and stays", updates, "update_trends", update_trends),
        "update_trends_absent_without_updates": ("Q2", "records without `update_rms`: the block is absent", grad_flat[:20], "update_trends", update_trends),
        "weight_jumps_one_tensor_jumps": ("Q3", "six weight checks; one tensor's norm goes from 5.3 to 21 million between two of them", weights, "weight_jumps", weight_jumps),
        "weight_jumps_one_check": ("Q3", "one weight check: nothing to compare, the block is absent", weights[:1], "weight_jumps", weight_jumps),
    }
    return {name: {"part": "measurements", "gap": gap, "says": says, "records": records, "expected": {key: fn(records)}}
            for name, (gap, says, records, key, fn) in sets.items()}


def score_cases():
    base = [10.0 + 0.1 * ((i * 3) % 7) for i in range(50)]
    sets = {
        "score_above": ("50 earlier values near 10, the value 25", base, 25.0),
        "score_below_is_negative": ("the same window, the value 4", base, 4.0),
        "score_ten_earlier_values": ("exactly ten earlier values: scored", base[:10], 25.0),
        "score_nine_earlier_values": ("nine earlier values: no score", base[:9], 25.0),
        "score_values_that_are_not_finite_do_not_count": ("eleven entries, two of them not finite: nine count, no score",
                                                         [*base[:9], "nan", None], 25.0),
        "score_no_dispersion": ("more than half the earlier values equal: no score, and no division", [10.0] * 30 + base[:20], 25.0),
        "score_all_equal": ("every earlier value equal", [10.0] * 50, 25.0),
        "score_six_alike_and_four_different": (
            "six equal values and four that differ: the MAD is zero though the values vary, so the words must not say they did not",
            [10.0] * 6 + [11.0, 12.0, 13.0, 14.0], 25.0),
    }
    return {name: {"part": "score", "says": says, "window": window, "value": value,
                   "expected": {**score(value, window), "display": display(score(value, window))}}
            for name, (says, window, value) in sets.items()}


# -- part 1 over records: the window each ratio is taken against ---------------------------------

def same(a, b):
    return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))


def grad_records(n, total, norms):
    return [{"step": i, "grad_norm": total(i), "grad_norms": {name: fn(i) for name, fn in norms.items()}} for i in range(n)]


def grad_spike_scored(records):
    """Today's `grad_spike` incident with the score each ratio gains. The windows are stated on the
    page; each is checked here against the baseline today's rule reports for it."""
    incident = ev.detect_grad_spikes(records)
    assert incident is not None
    rows = sorted(records, key=lambda r: r["step"])
    at = next(i for i, r in enumerate(rows) if r["step"] == incident["step"])
    totals = [finite(r["grad_norm"]) for r in rows[:at] if not r.get("step_skipped")][-WINDOW:]
    assert incident["baseline"] is None or same(median(totals), incident["baseline"])
    out = {**incident, **(score(incident["grad_norm"], totals) if incident.get("ratio") is not None else {})}
    out["parameters"] = []
    for parameter in incident["parameters"]:
        earlier = [finite(r["grad_norms"][parameter["name"]]) for r in rows if r["step"] < parameter["rising_since"]
                   or (incident.get("trigger") != "parameter" and r["step"] < incident["step"])]
        if incident.get("trigger") != "parameter":
            earlier = [finite(r["grad_norms"][parameter["name"]]) for r in rows[:at]]
        window = earlier[-WINDOW:]
        assert same(median(window), parameter["baseline"]), (parameter, median(window))
        out["parameters"].append({**parameter, **score(parameter["grad_norm"], window)})
    return out


def activation_spike_scored(records):
    incident = next(i for i in ev.detect_activation_anomalies(records) if i["kind"] == "activation_spike")
    rows = sorted(records, key=lambda r: r["step"])
    modules = []
    for hit in incident["modules"]:
        window = [finite(r["layers"][hit["module"]][hit["stat"]]) for r in rows if r["step"] < incident["step"]][-WINDOW:]
        assert same(median(window), hit["baseline"]), hit
        modules.append({**hit, **score(hit["value"], window)})
    return {**incident, "modules": modules}


def score_record_cases():
    total_spike = grad_records(80, lambda i: (10.0 if i == 60 else 1.0) * wobble(i),
                               {"w": lambda i: (14.0 if i == 60 else 0.9) * wobble(i + 1), "v": lambda i: 0.4 * wobble(i + 2)})
    one_parameter = grad_records(80, lambda i: 1.0 * wobble(i),
                                 {"w": lambda i: 0.99 * wobble(i), "tiny": lambda i: (0.08 if i >= 60 else 0.01) * wobble(i + 3)})
    early = grad_records(30, lambda i: (10.0 if i == 12 else 1.0) * wobble(i), {"w": lambda i: (10.0 if i == 12 else 1.0) * wobble(i)})
    spike = layers_series([(1.0) * wobble(i) for i in range(100)],
                          amax=[(40.0 if i == 80 else 4.0) * wobble(i + 2) for i in range(100)])
    return {
        "score_on_grad_spike_by_the_total": {
            "part": "score_records", "says": "the total gradient norm at ten times its level on one step; the incident and each parameter gain a score against the 50 records before",
            "records": total_spike, "expected": {"incident": grad_spike_scored(total_spike)}},
        "score_on_grad_spike_by_one_parameter": {
            "part": "score_records", "says": "the total steady, one small parameter at eight times its own level and staying there: the other parameter path, whose baseline is that parameter's own earlier norms",
            "records": one_parameter, "expected": {"incident": grad_spike_scored(one_parameter)}},
        "score_on_grad_spike_with_twelve_earlier_records": {
            "part": "score_records", "says": "a spike on the thirteenth record: twelve earlier totals, so a score; a ratio and its score always come from the same values",
            "records": early, "expected": {"incident": grad_spike_scored(early)}},
        "score_on_activation_spike": {
            "part": "score_records", "says": "one module's `abs_max` at ten times its level on one record",
            "records": spike, "expected": {"incident": activation_spike_scored(spike)}},
    }


# -- parts 4 and 5 -----------------------------------------------------------------------------

def evidence_cases():
    import sys
    sys.path.insert(0, str(HERE.parents[1]))
    try:
        import test_update_size
    finally:
        sys.path.remove(str(HERE.parents[1]))
    name = next(n for n in test_update_size.UPDATE_CASES if "jump" in n)
    plain = [dict(r) for r in test_update_size.UPDATE_CASES[name]]
    today = ev.detect_update_spikes(plain)
    assert today is not None, name
    rows = sorted(plain, key=lambda r: r["step"])
    at = next(i for i, r in enumerate(rows) if r["step"] == today["step"])
    with_lr = [{**r, "lr": [3e-3, 3e-4] if r["step"] >= today["step"] else [1e-3, 1e-4]} for r in rows]
    without = [{k: v for k, v in r.items() if k != "lr"} for r in rows]
    assert rows[at - 1]["step"] == today["step"] - 1  # the record it was compared with is the one before
    assert ev.detect_update_spikes(with_lr) == today == ev.detect_update_spikes(without)  # the rule does not read `lr`
    unread = [{"step": i, "grad_norm": 1.0, "grad_norms": {"a": 1.0}, "weight_norms": {"a": 2.0 + 0.01 * i},
               **({"grads_not_read": {"c": "a nested tensor", "b": "on the meta device"}} if i == 0 else {})} for i in range(40)]
    entry = ev.build_evidence(unread, detectors="python")["kind_checks"]["stalled_weights"]
    assert entry.get("parameters_not_judged") == ["b", "c"], entry  # the Python detectors do this today
    other = [{**r, "rank": 1, **({"grads_not_read": {"c": "a nested tensor"}} if r["step"] == 0 else {})}
             for r in ({k: v for k, v in r.items() if k != "grads_not_read"} for r in unread)]
    return {
        "update_spike_carries_the_learning_rate": {
            "part": "evidence", "says": f"the parity suite's `{name}` records with a learning rate on every record, which changes at the flagged step: the incident carries the rate there and at the record it was compared with",
            "records": with_lr, "expected": {"incident": {**today, "lr": [3e-3, 3e-4], "lr_before": [1e-3, 1e-4]}}},
        "update_spike_without_a_learning_rate": {
            "part": "evidence", "says": "the same records with no `lr`: neither key is there, and the incident is as today",
            "records": without, "expected": {"incident": today}},
        "stalled_weights_names_what_it_did_not_judge": {
            "part": "evidence", "says": "two parameters whose gradient the capture could not read: the `stalled_weights` entry of `kind_checks` names them, sorted, in both engines",
            "records": unread, "expected": {"kind_checks": {"stalled_weights": {"value": entry["value"], "parameters_not_judged": ["b", "c"]}}}},
        "the_all_ranks_view_names_them_by_rank_when_ranks_differ": {
            "part": "evidence", "says": "rank 0 could not read `b` and `c`, rank 1 only `c`: the run-level entry carries the union, and each rank's list",
            "records_by_rank": {"0": unread, "1": other},
            "expected": {"kind_checks": {"stalled_weights": {"parameters_not_judged": ["b", "c"],
                                                             "parameters_not_judged_by_rank": {"0": ["b", "c"], "1": ["c"]}}}}},
    }


# -- part 7: from the wire to a reader -----------------------------------------------------------

def wire_cases():
    layers = {"blk0": {"std": 1.0, "abs_max": 4.0, "zero_frac": 0.0}}
    partial = [{"step": i, "layers": layers, **({"layers_not_read": {"blk1": "its output is not a tensor"}} if i == 0 else {})} for i in range(20)]
    none = [{"step": i, **({"layers_not_read": {"blk0": "its output is not a tensor", "blk1": "its output is not a tensor"}} if i == 0 else {})} for i in range(20)]
    kinds = ["activation_spike", "activation_drift", "dead_activations"]
    moe = lambda source: [{"step": i, "moe": {"r": [32] * 8}, "moe_top_k": {"r": 1}, **({"moe_source": {"r": source}} if i == 0 else {})} for i in range(3)]
    beat = lambda **more: {"run_id": "run", "rank": 0, "attempt": 1, "phase": "train", "capture_stopped": "sink_error", **more}
    return {
        "layers_not_read_some_modules": {
            "part": "wire", "says": "one module of two could not be read: coverage is partial, the activation kinds stay checked for what was read and name the module",
            "records": partial,
            "expected": {"not_checked": [{"check": "layers", "coverage": "partial", "modules": ["blk1"], "step": 0,
                                          "reason": "layer not read: blk1: its output is not a tensor"}],
                         "kind_checks": {kind: {"value": "checked", "modules_not_judged": ["blk1"]} for kind in kinds}}},
        "layers_not_read_no_module": {
            "part": "wire", "says": "no module could be read on any record: coverage is none, and the activation kinds are unavailable, not clean",
            "records": none,
            "expected": {"not_checked": [{"check": "layers", "coverage": "none", "modules": ["blk0", "blk1"], "step": 0, "kinds": kinds,
                                          "reason": "layer not read: blk0: its output is not a tensor; blk1: its output is not a tensor"}],
                         "kind_checks": {kind: {"value": "unavailable"} for kind in kinds}}},
        "a_module_missing_on_some_records_is_judged_on_the_rest": {
            "part": "wire", "says": "a module read on most records and absent from three, with no `layers_not_read`: nothing is unavailable, and the readable records keep their evidence. v1 does not represent the gap",
            "records": [{"step": i, "layers": {"blk0": layers["blk0"], **({} if i in (5, 6, 7) else {"blk1": layers["blk0"]})}} for i in range(20)],
            "expected": {"not_checked": [], "kind_checks": {kind: {"value": "checked"} for kind in kinds}}},
        "moe_source_scores_are_what_the_router_asked_for": {
            "part": "wire", "says": "counts taken as the top-k of the hooked scores: the router's requests, whatever is applied after",
            "records": moe({"form": "scores", "point": "requested"}),
            "expected": {"moe_source": {"r": {"form": "scores", "point": "requested"}}, "words": {"r": "the assignments the router asked for"}}},
        "moe_source_indices_at_an_unknown_point": {
            "part": "wire", "says": "the model gave indices, and the capture cannot tell whether a capacity limit was applied before that point: a reader does not call them dispatched",
            "records": moe({"form": "indices", "point": "unknown"}),
            "expected": {"moe_source": {"r": {"form": "indices", "point": "unknown"}},
                         "words": {"r": "the assignments the model gave; whether a capacity limit was applied before this point is not known"}}},
        "moe_source_a_map_after_dispatch": {
            "part": "wire", "says": "a routing map taken where the capture point establishes that dispatch is done: only then does a reader say dispatched",
            "records": moe({"form": "map", "point": "dispatched"}),
            "expected": {"moe_source": {"r": {"form": "map", "point": "dispatched"}}, "words": {"r": "the dispatched assignments"}}},
        "moe_source_absent": {
            "part": "wire", "says": "a capture from before the key: no `moe_source`, and a reader makes no claim",
            "records": moe(None)[1:], "expected": {"moe_source": None, "words": {}}},
        "capture_stopped_at_step_0": {
            "part": "wire", "says": "a beat that says capture stopped after committing step 0: the status carries step 0, which is not the same as none",
            "heartbeat": beat(capture_stopped_step=0), "expected": {"status": {"capture_stopped": {"reason": "sink_error", "step": 0}}}},
        "capture_stopped_before_any_step": {
            "part": "wire", "says": "capture stopped before any step was committed: the step is null",
            "heartbeat": beat(capture_stopped_step=None), "expected": {"status": {"capture_stopped": {"reason": "sink_error", "step": None}}}},
        "capture_stopped_from_an_older_sender": {
            "part": "wire", "says": "a sender from before the key: the status has the reason and no `step` key at all",
            "heartbeat": beat(), "expected": {"status": {"capture_stopped": {"reason": "sink_error"}}}},
    }


def jsonable(value):
    """Strict JSON: a float that is not finite never appears in a case."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("a case holds a non-finite float")
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def build():
    cases = {**score_cases(), **score_record_cases(), **drift_cases(), **imbalance_cases(), **measurement_cases(),
             **evidence_cases(), **wire_cases()}
    return {
        "contract": "DETECTOR_RULES_CONTRACT.md",
        "constants": {"score_min": SCORE_MIN, "score_scale": SCORE_SCALE, "drift_factor": DRIFT_FACTOR,
                      "drift_records_needed": DRIFT_NEEDS, "start_grace": START_GRACE, "trend_min": TREND_MIN, "listed": LISTED},
        "cases": jsonable(cases),
    }


if __name__ == "__main__":
    document = build()
    (HERE / "cases.json").write_text(json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n")
    parts: dict[str, int] = {}
    for body in document["cases"].values():
        parts[body["part"]] = parts.get(body["part"], 0) + 1
    print(len(document["cases"]), "cases:", parts)
