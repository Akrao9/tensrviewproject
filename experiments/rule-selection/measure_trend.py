"""A candidate measurement for two gaps, with no threshold: a robust local trend.

Q1 is a slow rise of an attention module's largest logit, which the moving baseline of
`attention_logit_growth` follows and so never sees. B4 is a total gradient norm that rises steadily.
Both are a level that keeps moving in one direction, so one measurement is tried for both:
`robust_local_trend` of the sibling repository LLM-Training-Debugger
(`src/training_debugger/detectors/base.py`), written again here from its description. It fits a
line through two anchor points, the median of the first third of a window and the median of its
last third, each placed at the median of its own step numbers. So it uses the actual step numbers
(probes 1,000 steps apart and records one step apart are both right), and one outlier moves
neither anchor. Fewer than six points give no trend.

What is printed, per series, is the trend and nothing judged:
    growth      last-third median over first-third median, for the whole series
    per 100     the slope as a share of the first anchor, per 100 steps
    rising      the longest stretch, in steps, over which the trend of the trailing window stayed up
"""

from __future__ import annotations

import sys

import inputs
from inputs import ev

WINDOW = 200  # the trailing window, in points: the sibling's default
MIN_POINTS = 6


def median(values):
    return ev._median(list(values))


def trend(steps, values):
    """(slope per step, first anchor (x, y), last anchor (x, y)), or None under six points or with
    anchors on one step."""
    n = len(values)
    if n < MIN_POINTS:
        return None
    k = max(1, n // 3)
    left = (median(steps[:k]), median(values[:k]))
    right = (median(steps[-k:]), median(values[-k:]))
    if right[0] - left[0] <= 0:
        return None
    return (right[1] - left[1]) / (right[0] - left[0]), left, right


def describe(steps, values):
    whole = trend(steps, values)
    if whole is None:
        return None
    slope, left, right = whole
    out = {"points": len(values), "steps": (steps[0], steps[-1]), "first": left[1], "last": right[1],
           "growth": right[1] / left[1] if left[1] > 0 else None,
           "per_100": 100 * slope / left[1] if left[1] > 0 else None}
    best, start, best_from = 0, None, None
    steepest = (0.0, None)
    for at in range(MIN_POINTS, len(values) + 1):
        local = trend(steps[max(0, at - WINDOW):at], values[max(0, at - WINDOW):at])
        up = local is not None and local[0] > 0
        if up:
            if start is None:
                start = steps[at - 1]
            if steps[at - 1] - start > best:
                best, best_from = steps[at - 1] - start, start
            rate = 100 * local[0] / local[1][1] if local[1][1] > 0 else 0.0
            if at >= min(len(values), 60) and rate > steepest[0]:
                steepest = (rate, steps[at - 1])
        else:
            start = None
    out.update(rising=best, rising_from=best_from, steepest_per_100=steepest[0], steepest_at=steepest[1])
    return out


def show(label, found):
    if found is None:
        print(f"   {label:<34} fewer than {MIN_POINTS} points: no trend")
        return
    growth = "n/a" if found["growth"] is None else f"{found['growth']:.2f}"
    per = "n/a" if found["per_100"] is None else f"{found['per_100']:+.1%}"
    print(f"   {label:<34} {found['points']:>5} points, steps {found['steps'][0]}..{found['steps'][1]}:"
          f" {found['first']:.4g} -> {found['last']:.4g}, growth {growth}, {per} per 100 steps;"
          f" rising {found['rising']} steps from {found['rising_from']};"
          f" steepest trailing trend {found['steepest_per_100']:+.1%} per 100 at step {found['steepest_at']}")


def gradient(rows):
    key = "clip_grad_norm" if any("clip_grad_norm" in r for r in rows) else "grad_norm"
    pairs = [(int(r["step"]), ev._finite(r.get(key))) for r in rows if not r.get("step_skipped")]
    pairs = [(s, v) for s, v in pairs if v is not None]
    return key, [s for s, _ in pairs], [v for _, v in pairs]


def attention(rows):
    """Each attention module's largest finite `max_logit`, probe by probe, as the rules read it."""
    series: dict[str, tuple[list[int], list[float]]] = {}
    for r in rows:
        for module, stats in ((r.get("probe") or {}).get("attention") or {}).items():
            _, logit, _ = ev._attention_values(stats)
            if logit is not None:
                steps, values = series.setdefault(module, ([], []))
                steps.append(int(r["step"]))
                values.append(logit)
    return series


def synthetic():
    """The review's series for Q1 (v3 plan, row Q1): a logit that rises by a fixed share at each
    probe, 1,000 steps apart, from 10. At 25% a probe `attention_logit_growth` says nothing; at 27%
    it fires. And a flat control with the same noise-free shape."""
    out = {}
    for name, rate in (("rising 25% a probe", 1.25), ("rising 27% a probe", 1.27), ("rising 5% a probe", 1.05), ("flat", 1.0)):
        steps = [1000 * (i + 1) for i in range(31)]
        out[name] = (steps, [10.0 * rate ** i for i in range(31)])
    return out


def main():
    only = set(sys.argv[1:])
    print("# B4: the total gradient norm")
    seen = set()
    for name, rows, entry in inputs.everything():
        if only and entry["id"] not in only or name in seen:
            continue
        seen.add(name)
        key, steps, values = gradient(rows)
        if len(values) < MIN_POINTS:
            continue
        show(f"{name[:33]} [{inputs.role(name, entry)[:1]}]", describe(steps, values))
    print("\n# Q1: each attention module's largest logit")
    for name, rows, entry in inputs.everything():
        if only and entry["id"] not in only:
            continue
        series = attention(rows)
        if series:
            counts = sorted({len(v) for _, v in series.values()})
            print(f"   {name}: {len(series)} attention modules, {counts} probes each at steps"
                  f" {next(iter(series.values()))[0]}: " + ("no trend, under six points" if max(counts) < MIN_POINTS else ""))
    print("   the review's series (hand-written, not a capture):")
    for name, (steps, values) in synthetic().items():
        show(name, describe(steps, values))
        early = describe(steps[:8], values[:8])
        show(name + ", first 8 probes", early)


if __name__ == "__main__":
    main()
