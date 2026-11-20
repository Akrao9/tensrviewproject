"""Compact evidence: the evidence file an agent reads.

For an MoE run the per-step `samples` (one per router per step) are nearly the whole of
`evidence.json`: 86 MB of an 86 MB file at 10,000 steps of a 24-router, 32-expert run, where
everything else is under 4 KB. An agent needs the incidents and what the routing looked like
around them. `compact_evidence` keeps every key of the evidence as it is and, of `samples`,
only the steps within a short window of each MoE incident and of each later step a starved
expert's starvation began on; a `compact` key says what was kept, how much was left out and
the last step that was still flagged.

    python -m tensorquill run.jsonl --compact            # writes run.evidence.compact.json
    python -m tensorquill run.jsonl --compact-window 20  # 20 recorded steps each side

`--compact` writes the compact file instead of the full one, under a name of its own
(`<name>.compact.json`). Instead of, because the full file is the cost being avoided: at
10,000 steps it takes most of a second to write, on every look of `--watch`, and it can be
built again from the records at any time by leaving the option out. Under its own name, so a
full file already on disk is never replaced by a smaller one and no reader mistakes the two;
with `-o PATH` it goes to that path.

`incident_summary` and `run_state_line` print the same text for both files: the first flagged
sample, which the summary quotes, is always kept. `expert_load_table` and `export_rerun` need
samples the compact file may not hold, and raise `SamplesOmitted` saying which.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from pathlib import Path
from typing import Any, Mapping

from .evidence import write_evidence

COMPACT_VERSION = 1
# Recorded steps kept on each side of a MoE incident's step. 0 keeps the incident's own step
# alone, the least that leaves `incident_summary` unchanged (it quotes that step's counts).
DEFAULT_WINDOW = 5
# The kinds read from `samples`. The window is taken around each one's `step`: for
# `starved_experts` that is the first step any expert was starved. Each later step an expert's
# starvation began on gets a window too, up to ONSET_WINDOWS of them, the earliest first (lane X5,
# DET-13: on the recorded Granite runs 14 experts start at step 49 and six more at 53 to 131,
# and only 44-54 was kept, so an agent could see none of those six begin). The marker counts the
# onsets past the bound.
MOE_KINDS = ("moe_imbalance", "starved_experts")
ONSET_WINDOWS = 16
_REBUILD = (
    "The full evidence has them: python -m tensorquill <records> without --compact, or build_evidence()"
)


class SamplesOmitted(LookupError):
    """Asked of compact evidence: per-step samples it left out. The message says which."""


def _window(window: Any) -> int:
    if isinstance(window, bool) or not isinstance(window, int) or window < 0:
        raise ValueError(f"the compact window is a number of recorded steps, 0 or more, not {window!r}")
    return window


def _flagged(sample: Mapping[str, Any]) -> bool:
    """As `evidence.first_incident` reads a sample: an overloaded or dead expert on a step that routed tokens."""
    return bool(
        not sample.get("missing")
        and sample.get("routed_tokens", 0) > 0
        and (sample.get("overloaded") or sample.get("dead"))
    )


def _ranges(steps: list[int], kept: list[int]) -> list[list[int]]:
    """Kept positions (sorted) as `[first step, last step]` ranges of steps recorded one after another."""
    found: list[list[int]] = []
    last: int | None = None
    for position in kept:
        if last is not None and position == last + 1:
            found[-1][1] = steps[position]
        else:
            found.append([steps[position], steps[position]])
        last = position
    return found


def compact_evidence(evidence: Mapping[str, Any], window: int = DEFAULT_WINDOW) -> dict[str, Any]:
    """The compact form of `evidence` (what `build_evidence` or a watcher made).

    Every key is kept as it is but `samples`, of which only the steps within `window`
    recorded steps of a MoE incident's step stay (every router of those steps), of each later
    starvation onset (`ONSET_WINDOWS`), and of the first flagged sample when no incident names
    it; and a router's last sample before a window when it routed no token before it
    (`unrouted_before`). A `compact` key is added before `samples`:

        version        1
        window         recorded steps kept on each side of an incident's step
        around         what the windows are around: `[{"kind", "step"}]`, the incidents, then
                       `starved_expert_onset` for a later onset, `first_flagged_sample`
        kept_steps     `[[first, last], ...]`: the ranges of steps whose samples are here
        steps          `{"total", "kept", "omitted"}`: recorded steps with MoE samples
        samples        `{"total", "kept", "omitted"}`: samples (a router on a step)
        sampled_steps  `[first, last]` step with samples in the full evidence, or null
        moe_flagged    `{"samples", "last_step"}`: samples with an overloaded or dead expert in
                       the full evidence and the last step one had (the server's report has the
                       same key); an incident names only the first
        moe_starved    the same for samples with a starved expert
        unrouted_before  `[[step, layer], ...]`: a kept range's first step and a router that
                       routed no token on any step before it in the full evidence, so its
                       expert-load table there compares with a step that routed none (that
                       step's sample is kept, with no token to show)
        starved_onsets_without_window  `{"steps", "first_step"}`, only when later starvation
                       onsets were past `ONSET_WINDOWS`: how many got no window, and the first

    Evidence with no samples (a run without routers, a server's report, the all-ranks view)
    comes back whole, with the marker and nothing left out. Evidence that is already compact
    comes back unchanged: what it left out cannot be put back. `evidence` is not modified.
    """
    window = _window(window)
    if isinstance(evidence.get("compact"), Mapping):
        return dict(evidence)
    samples = evidence.get("samples")
    samples = [s for s in samples if isinstance(s, Mapping)] if isinstance(samples, list) else []
    steps = sorted({int(s["step"]) for s in samples})
    incidents = evidence.get("incidents")
    around = [
        {"kind": i["kind"], "step": i["step"]}
        for i in (incidents if isinstance(incidents, list) else ())
        if isinstance(i, Mapping) and i.get("kind") in MOE_KINDS and isinstance(i.get("step"), int)
    ]
    anchored = {a["step"] for a in around}
    onsets = sorted({
        e["step"] for i in (incidents if isinstance(incidents, list) else ())
        if isinstance(i, Mapping) and i.get("kind") == "starved_experts" and isinstance(i.get("experts"), list)
        for e in i["experts"]
        if isinstance(e, Mapping) and isinstance(e.get("step"), int) and not isinstance(e.get("step"), bool)
    } - anchored)
    around += [{"kind": "starved_expert_onset", "step": step} for step in onsets[:ONSET_WINDOWS]]
    flagged = [s for s in samples if _flagged(s)]
    if flagged:
        # `incident_summary` quotes the earliest flagged sample (`first_incident`), incident or not.
        first = min(flagged, key=lambda s: (s["step"], s["layer"]))
        if not any(a["kind"] == "moe_imbalance" and a["step"] == first["step"] for a in around):
            around.append({"kind": "first_flagged_sample", "step": int(first["step"])})
    positions: set[int] = set()
    for anchor in around:
        at = bisect_left(steps, anchor["step"])
        exact = at < len(steps) and steps[at] == anchor["step"]
        positions.update(range(max(0, at - window), min(len(steps), at + window + (1 if exact else 0))))
    kept_positions = sorted(positions)
    kept_steps = {steps[p] for p in kept_positions}
    ranges = _ranges(steps, kept_positions)
    unrouted, before = _unrouted_before(samples, [first for first, _ in ranges if first != steps[0]])
    kept = [s for s in samples if int(s["step"]) in kept_steps or id(s) in before]
    starved = [s for s in samples if s.get("starved")]
    marker = {
        "version": COMPACT_VERSION,
        "window": window,
        "around": around,
        "kept_steps": ranges,
        "steps": {"total": len(steps), "kept": len(kept_steps), "omitted": len(steps) - len(kept_steps)},
        "samples": {"total": len(samples), "kept": len(kept), "omitted": len(samples) - len(kept)},
        "sampled_steps": [steps[0], steps[-1]] if steps else None,
        "moe_flagged": {"samples": len(flagged), "last_step": max((int(s["step"]) for s in flagged), default=None)},
        "moe_starved": {"samples": len(starved), "last_step": max((int(s["step"]) for s in starved), default=None)},
        "unrouted_before": unrouted,
    }
    if onsets[ONSET_WINDOWS:]:
        marker["starved_onsets_without_window"] = {"steps": len(onsets) - ONSET_WINDOWS, "first_step": onsets[ONSET_WINDOWS]}
    out: dict[str, Any] = {}
    for key, value in evidence.items():
        if key == "samples":
            out["compact"] = marker
            out["samples"] = kept
        else:
            out[key] = value
    out.setdefault("compact", marker)
    return out


def _routed(sample: Mapping[str, Any]) -> bool:
    """As `expert_load_table` picks the step it compares with: not missing, and tokens routed."""
    return not sample.get("missing") and sample.get("routed_tokens", 0) > 0


def _unrouted_before(samples: list[Mapping[str, Any]], firsts: list[int]) -> tuple[list[list[int]], set[int]]:
    """For each kept range's first step (`firsts`) and each router, whether the router routed a
    token on any earlier step of the full evidence. Where it did not, the pair is listed and the
    router's last earlier sample, which routed none, is kept (by `id`): `expert_load_table`
    compares the range's steps with it, as on the full evidence, with no delta to show, and no
    other reader of the samples takes a step with no token routed for one (lane X5, DET-13: the
    table at such a step was refused although nothing was missing for it)."""
    if not firsts:  # no window, or only one at the run's first step: nothing before it
        return [], set()
    by_layer: dict[int, list[Mapping[str, Any]]] = {}
    for sample in sorted(samples, key=lambda s: int(s["step"])):
        by_layer.setdefault(int(sample["layer"]), []).append(sample)
    pairs: list[list[int]] = []
    keep: set[int] = set()
    for layer, held in sorted(by_layer.items()):
        at_steps = [int(s["step"]) for s in held]
        routed_so_far: list[bool] = []
        seen = False
        for sample in held:
            seen = seen or _routed(sample)
            routed_so_far.append(seen)
        for first in firsts:
            at = bisect_left(at_steps, first)
            if at == 0 or not routed_so_far[at - 1]:
                pairs.append([first, layer])
                if at > 0:
                    keep.add(id(held[at - 1]))
    return sorted(pairs), keep


def _span(first: int, last: int) -> str:
    return str(first) if first == last else f"{first}-{last}"


def describe(evidence: Mapping[str, Any]) -> str | None:
    """What a compact evidence file keeps and leaves out, as one clause; None for full evidence."""
    marker = evidence.get("compact")
    if not isinstance(marker, Mapping):
        return None
    samples, steps = marker.get("samples") or {}, marker.get("steps") or {}
    if not samples.get("total"):
        return "compact evidence; it had no per-step samples to leave out"
    if not samples.get("omitted"):
        return f"compact evidence; all {samples.get('total'):,} per-step samples fit the window, none left out"
    ranges = marker.get("kept_steps") or []
    window = marker.get("window")
    onset = any(isinstance(a, Mapping) and a.get("kind") == "starved_expert_onset" for a in marker.get("around") or ())
    where = (
        "steps " + ", ".join(_span(a, b) for a, b in ranges)
        + f" ({window} recorded step{'s' if window != 1 else ''} each side of a MoE incident"
        + (" or of a later starved expert's onset" if onset else "") + ")"
        if ranges else "no step, since no MoE incident is listed"
    )
    return (
        f"compact evidence; it keeps the per-step samples of {where} and leaves out"
        f" {steps.get('omitted'):,} of the run's {steps.get('total'):,} recorded steps"
        f" ({samples.get('omitted'):,} of {samples.get('total'):,} samples)"
    )


def _omits(evidence: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The marker of compact evidence that left samples out; None for full evidence or a compact file that holds all."""
    marker = evidence.get("compact")
    if isinstance(marker, Mapping) and (marker.get("samples") or {}).get("omitted"):
        return marker
    return None


def require_all_samples(evidence: Mapping[str, Any], what: str) -> None:
    """Raise `SamplesOmitted` when `what` needs every step and `evidence` is a compact file that left steps out."""
    if _omits(evidence) is not None:
        raise SamplesOmitted(f"{what} needs every step's samples, and this is {describe(evidence)}. {_REBUILD}.")


def require_table(evidence: Mapping[str, Any], step: int, layer: int) -> None:
    """Raise `SamplesOmitted` when `expert_load_table(evidence, step, layer)` would not be the
    table the full evidence gives: the step was left out, or the step it is compared with was.

    The table compares a step with the last earlier one on which the router routed tokens.
    At the first step a window keeps, that one is among the steps left out (so with
    `window=0` no table can be made, and with any larger window the incident's own step can),
    unless the router routed no token before the window (`unrouted_before`): then nothing is
    left out that the table uses.
    """
    marker = _omits(evidence)
    if marker is None:
        return
    found = next((r for r in marker.get("kept_steps") or () if r[0] <= step <= r[1]), None)
    if found is None:
        raise SamplesOmitted(
            f"the expert load at step {step} is not here: this is {describe(evidence)}. {_REBUILD}."
        )
    sampled = marker.get("sampled_steps") or [None]
    if found[0] == sampled[0]:
        return  # the run's first steps: nothing was left out before them
    if [found[0], layer] in (marker.get("unrouted_before") or ()):
        return  # nothing routed before the window: the step compared with is kept, with no token
    for sample in evidence.get("samples", ()):
        if (
            int(sample["layer"]) == layer and found[0] <= int(sample["step"]) < step
            and not sample.get("missing") and sample.get("routed_tokens", 0) > 0
        ):
            return
    raise SamplesOmitted(
        f"the expert load at step {step} is compared with the last earlier step that routed tokens on layer"
        f" {layer}, and that step is not here: this is {describe(evidence)}."
        f" Ask for a later step of the range, or widen the window (--compact-window). {_REBUILD}."
    )


def compact_path(path: str | Path) -> Path:
    """The compact file's default name beside a full one: `run.evidence.json` -> `run.evidence.compact.json`."""
    path = Path(path)
    return path.with_name(path.stem + ".compact" + path.suffix)


def write_compact(evidence: Mapping[str, Any], path: str | Path, *, window: int = DEFAULT_WINDOW,
                  named: bool = False) -> Path:
    """Write the compact form of `evidence`. `path` is where the full file would go: the
    compact one goes to `compact_path(path)`, or to `path` itself when the caller named it (`named`)."""
    return write_evidence(compact_evidence(evidence, window), path if named else compact_path(path))


# --- The command line (`python -m tensorquill`) ----------------------------------------------


def _window_argument(value: str) -> int:
    try:
        return _window(int(value))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number of recorded steps (0 or more)") from None


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--compact",
        action="store_true",
        help="write the compact evidence file an agent reads instead of the full one: every incident and every "
        "other key, and of the per-step MoE samples only the steps around each MoE incident and each later "
        "starvation onset, with a `compact` key "
        "that counts what was left out (at 10,000 steps of a 24-router run: about 0.2 MB instead of 86 MB). "
        "Default output: <input>.evidence.compact.json, so a full file is never replaced; run again without "
        "--compact for the full file",
    )
    parser.add_argument(
        "--compact-window",
        type=_window_argument,
        metavar="N",
        help=f"with --compact (and implies it): recorded steps kept on each side of a MoE incident's step "
        f"(and of a later starvation onset) "
        f"(default: {DEFAULT_WINDOW}; 0 keeps the incident's own step alone)",
    )


def window_from_args(args: argparse.Namespace) -> int | None:
    """The window `--compact` / `--compact-window` ask for; None when neither was given."""
    if args.compact_window is not None:
        return args.compact_window
    return DEFAULT_WINDOW if args.compact else None
