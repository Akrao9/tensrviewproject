"""S6 on TensorQuill's own captures: the median-and-MAD score beside the ratio, measurement only.

For the two ratios the spike rules take against a trailing window (the total gradient norm, and a
module's `abs_max` and `std`), every record's value is scored against the same window of earlier
values the ratio uses:

    ratio      value / median(window)
    score      (value - median(window)) / (1.4826 * MAD(window))
    detrended  the same on residuals from the window's robust local trend, read at the record's step

The window is the 50 records before, as the rules have it. A window with fewer than MIN finite
values gives no score, and a zero MAD gives none either: both are counted.
"""

from __future__ import annotations

import sys

import inputs
from inputs import ev
from measure_trend import trend

WINDOW = 50
SCALE = 1.4826


def median(values):
    return ev._median(list(values))


def mad(values, centre):
    return median([abs(v - centre) for v in values])


def scored(steps, values, at, minimum):
    """(ratio, score, detrended score, why no score) of values[at] against the window before it."""
    lo = max(0, at - WINDOW)
    past, past_steps = values[lo:at], steps[lo:at]
    if len(past) < minimum:
        return None, None, None, "short"
    centre = median(past)
    ratio = values[at] / centre if centre > 0 else None
    spread = SCALE * mad(past, centre)
    if spread == 0:
        return ratio, None, None, "zero_mad"
    score = (values[at] - centre) / spread
    detrended = None
    fit = trend(past_steps, past)
    if fit is not None:
        slope, left, _ = fit
        line = lambda x: left[1] + slope * (x - left[0])
        residuals = [v - line(s) for s, v in zip(past_steps, past)]
        res_centre = median(residuals)
        res_spread = SCALE * mad(residuals, res_centre)
        if res_spread > 0:
            detrended = (values[at] - line(steps[at]) - res_centre) / res_spread
    return ratio, score, detrended, None


def series_of(rows):
    """name -> (steps, values): the total gradient norm, and each module's abs_max and std."""
    key = "clip_grad_norm" if any("clip_grad_norm" in r for r in rows) else "grad_norm"
    out: dict[str, tuple[list[int], list[float]]] = {}
    for r in rows:
        step = int(r["step"])
        total = None if r.get("step_skipped") else ev._finite(r.get(key))
        if total is not None:
            s, v = out.setdefault("grad total", ([], []))
            s.append(step); v.append(total)
        for name, stats in (r.get("layers") or {}).items():
            for stat in ("abs_max", "std"):
                value = ev._finite(stats.get(stat))
                if value is not None:
                    s, v = out.setdefault(f"{stat}|{name}", ([], []))
                    s.append(step); v.append(value)
    return out


def main():
    minimum = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"# at least {minimum} earlier finite values in the window of {WINDOW}")
    seen = set()
    for name, rows, entry in inputs.everything():
        if name in seen:
            continue
        seen.add(name)
        role, fault = inputs.role(name, entry), entry.get("fault_step")
        groups: dict[str, dict] = {}
        for label, (steps, values) in series_of(rows).items():
            group = groups.setdefault(label.split("|")[0], {"judged": 0, "short": 0, "zero_mad": 0, "ratio": (0, None),
                                                            "score": (0, None), "detrended": (0, None), "fault": None})
            for at in range(len(values)):
                ratio, score, detrended, why = scored(steps, values, at, minimum)
                if why:
                    group[why] += 1
                    continue
                group["judged"] += 1
                # for a seeded input, the healthy part is what comes before the fault step
                before = fault is None or role == "healthy" or steps[at] < fault
                if before:
                    for key, value in (("ratio", ratio), ("score", score), ("detrended", detrended)):
                        if value is not None and value > group[key][0]:
                            group[key] = (value, steps[at])
                elif steps[at] <= fault + 5 and ratio is not None and (group["fault"] is None or ratio > group["fault"][0]):
                    group["fault"] = (ratio, score, detrended, steps[at])
        if not groups:
            continue
        print(f"\n== {name} [{role}" + (f", fault at {fault}" if fault is not None and role == "seeded" else "") + "]")
        for label, g in groups.items():
            line = (f"   {label:<10} judged {g['judged']:>6}, short {g['short']:>5}, zero MAD {g['zero_mad']:>4};"
                    f" largest before any fault: ratio {g['ratio'][0]:.2f} @{g['ratio'][1]}, score {g['score'][0]:.1f} @{g['score'][1]},"
                    f" detrended {g['detrended'][0]:.1f} @{g['detrended'][1]}")
            if g["fault"]:
                r, s, d, at = g["fault"]
                line += f"; at the fault (step {at}): ratio {r:.1f}, score {s:.0f}, detrended {('%.0f' % d) if d is not None else 'n/a'}"
            print(line)


if __name__ == "__main__":
    main()
