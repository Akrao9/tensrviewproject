"""Four detector thresholds read against the development inputs, at today's line and the lines
beside it. Measurement only: no rule, threshold or detector is changed, and nothing here proposes
one. Wave AE, lane AE2.

    python measure_thresholds.py                      # from this folder, with the project's .venv
    python measure_thresholds.py --traincheck-from /path/to/checkout   # the sweep is in another checkout
    python measure_thresholds.py --no-traincheck                       # measure without it

The four, as `tensorquill.evidence` applies them:

    grad_spike        a step's total gradient norm at SPIKE_FACTOR (5) times the median of the last
                      SPIKE_WINDOW (50) recorded totals, once SPIKE_MIN_HISTORY (10) are known; or one
                      parameter at 5 times its own median for SPIKE_SUSTAIN (3) recorded steps in a row
    diversity drop    a top-k router (k >= 3) whose diversity falls below DIVERSITY_DROP (0.6) times
                      its median over its last OVERLOAD_WINDOW unflagged samples, with an expert whose
                      share rose by OVERLOAD_JUMP; this marks the sample overloaded (moe_imbalance)
    starved_experts   an expert below STARVE_RATIO (0.1) of an even share over STARVE_WINDOW (50) samples
    stalled_weights   a parameter with a gradient unchanged on STALL_STEPS (3) weight checks in a row

How each line is varied, with the detector's own code throughout:

    grad_spike, total       `detect_grad_spikes(factor=f, sustain=<never>)`: the total's rule alone.
                            The largest ratio comes from a copy of the total's ratio (`total_ratios`),
                            checked against the detector's `spike_count` and `spike_steps` at every
                            factor on every input.
    grad_spike, parameter   `detect_grad_spikes(factor=f, sustain=s)` on copies of the records whose
                            finite totals are all set to 1.0, so the total cannot fire and only the
                            parameter rule can. The longest streak is the largest `s` that fires.
    diversity drop          `evidence.DIVERSITY_DROP` set to the line for the call to `_moe_samples`
                            (a module constant, read when the rule runs: no argument takes it), and
                            put back after. Each sample's `overload_rule["drop"]` is checked to hold
                            the line. The closest a router came is read by wrapping `_jump_overloaded`,
                            which sees the router's own window, so no copy of the rule is made.
    starved_experts         `_mark_starved(samples, window, ratio)`. The smallest share is read with a
                            ratio large enough that every expert not dead is listed with its share.
    stalled_weights         `detect_stalled_weights(steps=n)`. The longest streak is the largest `n`
                            that fires.

At today's lines each record set's four results are compared with `build_evidence`'s incidents,
and `kind_checks` says why a kind could not be judged. Inputs come only through `inputs.py`.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import inputs
from inputs import ev

SPIKE_FACTORS = [3.0, 4.0, 5.0, 8.0]
SPIKE_SUSTAINS = [2, 3, 5]
DIVERSITY_DROPS = [0.5, 0.6, 0.7]
STARVE_RATIOS = [0.05, 0.1, 0.2]
STARVE_WINDOWS = [25, 50, 100]
STALL_STEPS = [2, 3, 5]
NEVER = 10**9  # a sustain no excursion reaches
NOT_SWEEP = ("-probe", "-pytest")  # TrainCheck run folders that are not configurations of the sweep

# The collapse seeded in the Granite job (`install_collapse`, allowlist `role_by_rule`): its steps
# before it are the healthy part, for every rule here.
GRANITE_COLLAPSE = 450

# Which seeded record sets carry each rule's own fault, with the step it was seeded at. Read from
# the allowlist's notes, the TrainCheck catalog (`experiments/traincheck-bugs/catalog.py`) and the
# hand-written record sets' source (`sdk/tests/test_server_detectors.py`), never from a run's output.
FAULTS = {
    "grad_spike": {
        "synthetic-everything": (150, "hand-written: the total jumps at step 150 after building from 147"),
        "B04-SO-60335387-buggy": (0, "B4, a missed zero_grad(): gradients accumulate from the first step"),
    },
    "diversity": {
        "granite-full": (450, "one router collapsed at step 450"),
        "granite": (450, "one router collapsed at step 450"),
        "granite-no-grads": (450, "one router collapsed at step 450"),
        "granite-router-only": (450, "one router collapsed at step 450"),
        "synthetic-everything": (60, "hand-written: router a (top-8 of 32), seven experts take nearly every token from step 60 to 199"),
        "synthetic-moe-edges": (3, "hand-written: router early (top-4 of 16) collapses at step 3, before it has a baseline"),
    },
    "starved": {
        "nanomoe-balance-off": (0, "no balance loss: starved experts expected"),
        "granite-full": (450, "one router collapsed at step 450"),
        "granite": (450, "one router collapsed at step 450"),
        "granite-no-grads": (450, "one router collapsed at step 450"),
        "granite-router-only": (450, "one router collapsed at step 450"),
        "toy-moe": (200, "the router's bias pushed to one expert at step 200"),
        "synthetic-everything": (0, "hand-written: router b's expert 8 gets one token in 2,101 throughout"),
    },
    "stall": {
        "synthetic-everything": (230, "hand-written: w2 unchanged at every weight check from step 230"),
        "B01-PT-84911-buggy": (0, "B1: BatchNorm parameters get gradients that no optimizer holds"),
        "B01-PT-84911-buggy-w5": (0, "B1, weights checked every 5 steps"),
        "B01-PT-84911-buggy-long": (0, "B1, 100 batches"),
        "B01-PT-84911-buggy-initloop": (0, "B1, the capture started in the loop"),
    },
}
# Faults of another kind whose gradient also moves: the three GPU activation faults.
SECONDARY = {"grad_spike": {"dense-layer8": (250, "activation fault in block 8"),
                            "trainer-check": (250, "activation fault"),
                            "trainer-check-block": (253, "activation fault; its spike shows at 250")}}
# Seeded record sets that are healthy for a rule that does not read their fault (allowlist note:
# "starved experts expected, nothing else").
# Where the healthy part of a seeded set ends, when it is not the seeded step.
PREFIX_END = {"trainer-check-block": 250}
HEALTHY_FOR = {"grad_spike": {"nanomoe-balance-off"}, "stall": {"nanomoe-balance-off"},
               "diversity": {"nanomoe-balance-off"}}


def workload(name: str) -> str:
    """What workload a record set is. Ranks and configurations of one script are one workload."""
    if name.startswith("granite"):
        return "Granite 3.0 1B MoE fine-tune (one job)"
    if name.startswith(("dense-layer8", "trainer-check")):
        return "SmolLM2-360M fine-tune on GPU (one model: its own loop, and two Trainer runs)"
    if name.startswith("nanomoe"):
        return "nanoMoE from scratch (one model, one seed)"
    if name.startswith(("B09", "B10", "HW-P2P")):
        return "TrainCheck B9 / B10 / hardware simulation (one model)"
    if name[:3] in {"B01", "B03", "B04", "B06", "B08", "B14", "B19"}:
        return f"TrainCheck {name[:3]}"
    return {"toy-seeded-healthy": "seeded-failure toy", "toy-moe": "toy_moe"}.get(name, name)


def short(name: str) -> str:
    return name.removesuffix(".rank0") if not name.endswith(".rank1") else name


def fault_key(name: str) -> str:
    return name.split(".rank")[0]


# --- inputs -------------------------------------------------------------------------------------

def generate(entry: dict) -> str | None:
    """Make a `generated` entry's capture with the allowlist's own command when it is not there.
    Returns why it could not be made, or None."""
    folder = inputs.GENERATED / entry["id"]
    if folder.exists() and any(folder.glob("*.jsonl")):
        return None
    args = shlex.split(entry["command"])
    script = inputs.WORKTREE / args[0]
    env = {**os.environ, "PYTHONPATH": str(inputs.WORKTREE / "sdk" / "src"),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    python = sys.executable  # the interpreter this script runs under, as the published command gives it
    with tempfile.TemporaryDirectory() as tmp:
        extra = ["--out", str(Path(tmp) / "out")] if "seeded_failures" in args[0] else []
        done = subprocess.run([python, str(script), *args[1:], *extra], cwd=tmp, env=env,
                              capture_output=True, text=True)
        if done.returncode != 0:
            return f"its command failed: {done.stderr.strip().splitlines()[-1:]}"
        made = sorted(Path(tmp).rglob("*.jsonl"))
        if not made:
            return "its command wrote no .jsonl"
        folder.mkdir(parents=True, exist_ok=True)
        for jsonl in made:
            for beside in jsonl.parent.iterdir():  # the capture with its end marker and manifest
                if beside.is_file() and not beside.name.endswith(".evidence.json"):
                    shutil.copy2(beside, folder / beside.name)
    return None


class SweepError(RuntimeError):
    """The TrainCheck sweep read is not the intended one. Raised, never skipped."""


def intended_sweep(checkout: Path) -> list[str]:
    """The 34 run IDs of the TrainCheck sweep: each entry's `run_id` in the committed `results.json`,
    whose keys must be every configuration `catalog.py` names (each bug's `runs`, buggy and healthy)."""
    folder = checkout / "experiments" / "traincheck-bugs"
    results = json.loads((folder / "results.json").read_text())
    catalog = inputs._module(folder / "catalog.py")
    configs = {f"{bug['id']}|{tag}|{variant}" for bug in catalog.BUGS if bug.get("script")
               for tag, _ in bug.get("runs", []) for variant in ("buggy", "healthy")}
    if configs != set(results):
        raise SweepError(f"results.json and catalog.py disagree: only in the catalog {sorted(configs - set(results))},"
                         f" only in results.json {sorted(set(results) - configs)}")
    ids = sorted(entry["run_id"] for entry in results.values())
    if len(set(ids)) != len(ids):
        raise SweepError(f"results.json repeats a run_id: {ids}")
    return ids


def check_sweep(sets: list[tuple[str, list[dict], dict]], root: Path) -> tuple[list, list[dict], list[str]]:
    """The sets of the intended sweep, a row for each of its run IDs, and the folders left out.
    Raises SweepError when a run ID has no records, a set read is not one of them, or a rank file
    of a run is empty (it then gives no records, so the loader returns fewer ranks than files)."""
    out = root / "experiments" / "traincheck-bugs" / "out"
    ids = intended_sweep(inputs.WORKTREE)
    by_id: dict[str, list[tuple[str, list[dict], dict]]] = {}
    for one in sets:
        by_id.setdefault(one[0].split(".rank")[0], []).append(one)
    left_out = sorted(run for run in by_id if run.endswith(NOT_SWEEP))
    read = {run: found for run, found in by_id.items() if not run.endswith(NOT_SWEEP)}
    extra = sorted(set(read) - set(ids))
    missing = sorted(set(ids) - set(read))
    short_of_files = []
    rows = []
    for run in ids:
        files = sorted(path.name for path in (out / run).glob(f"{run}.rank*.jsonl"))  # names only
        found = read.get(run, [])
        if len(files) != len(found) or any(not records for _, records, _ in found):
            short_of_files.append(f"{run} ({len(files)} rank files, {len(found)} with records)")
        manifest_path = out / run / f"{run}.run.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        rows.append({"run_id": run, "ranks": len(found), "records": [len(records) for _, records, _ in found],
                     "sdk_version": manifest.get("sdk_version"), "created_at": manifest.get("created_at"),
                     "commit": next((k for _, records, _ in found for r in records[:1]
                                     for k in r if "commit" in k.lower() or "git" in k.lower()), None)})
    if extra or missing or short_of_files:
        raise SweepError(f"the TrainCheck sweep under {out} is not the intended one: read but not in the sweep {extra};"
                         f" in the sweep but not read {missing}; rank files without records {short_of_files}")
    return [one for run in ids for one in read[run]], rows, left_out


def read_inputs(traincheck_from: Path | None, no_traincheck: bool) -> tuple[list, list[tuple[str, str]], list[dict]]:
    """Every entry `everything()` would give, one entry at a time, with the entries that gave none,
    and the TrainCheck sweep's rows (`check_sweep`)."""
    found, unread, sweep = [], [], []
    for entry_id, entry in inputs.ENTRIES.items():
        if entry["kind"] in ("in-script", "validation"):
            continue
        if entry_id == "traincheck":
            if no_traincheck:
                unread.append((entry_id, "left out with --no-traincheck"))
                continue
            root = traincheck_from or inputs.WORKTREE
            if not (root / "experiments" / "traincheck-bugs" / "out").is_dir():
                raise SweepError(f"no TrainCheck sweep under {root}: give --traincheck-from <a checkout that has"
                                 " experiments/traincheck-bugs/out>, or --no-traincheck to measure without it")
            # The loader's own glob and refusals, with the checkout that holds the sweep's output.
            kept = inputs.WORKTREE
            inputs.WORKTREE = root
            try:
                sets = inputs.load(entry_id)
            finally:
                inputs.WORKTREE = kept
            sets, sweep, left_out = check_sweep(sets, root)
            if left_out:
                unread.append((entry_id, f"left out, not part of the sweep: {', '.join(left_out)}"))
            found.extend(sets)
            continue
        if entry["kind"] == "generated":
            why = generate(entry)
            if why:
                unread.append((entry_id, why))
                continue
        try:
            sets = inputs.load(entry_id)
        except Exception as error:  # noqa: BLE001: a missing file is reported, not raised
            unread.append((entry_id, f"{type(error).__name__}: {error}"))
            continue
        if not sets:
            root = inputs.MAIN if entry.get("root") == "main" else inputs.WORKTREE
            unread.append((entry_id, f"nothing matches {entry.get('path')} under {root}"))
            continue
        found.extend(sets)
    return found, unread, sweep


def before(rows: list[dict], step: int) -> list[dict]:
    return [r for r in rows if int(r["step"]) < step]


def rule_sets(rule: str, sets: list[tuple[str, list[dict], dict]]):
    """(group, name, records, fault) for one rule. Groups: healthy, fault (this rule's own),
    secondary (another kind of fault that moves this rule's series), other (another rule's fault)."""
    for name, rows, entry in sets:
        key = fault_key(name)
        role = inputs.role(name, entry)
        if key.startswith("granite"):
            yield "healthy", f"{name} (before {GRANITE_COLLAPSE})", before(rows, GRANITE_COLLAPSE), None
            if key in FAULTS[rule]:
                yield "fault", name, rows, FAULTS[rule][key]
            else:
                yield "other", name, rows, (GRANITE_COLLAPSE, "router collapse")
            continue
        seeded_at = PREFIX_END.get(key, entry.get("fault_step") or 0)
        if role != "healthy" and seeded_at > 0 and key not in HEALTHY_FOR.get(rule, set()):
            yield "healthy", f"{name} (before {seeded_at})", before(rows, seeded_at), None
        if key in FAULTS[rule]:
            yield "fault", name, rows, FAULTS[rule][key]
        elif key in SECONDARY.get(rule, {}):
            yield "secondary", name, rows, SECONDARY[rule][key]
        elif role == "healthy" or key in HEALTHY_FOR.get(rule, set()):
            yield "healthy", name, rows, None
        else:
            yield "other", name, rows, (entry.get("fault_step"), entry.get("note", ""))


# --- the gradient spike -------------------------------------------------------------------------

def grad_key(rows: list[dict]) -> str:
    return "clip_grad_norm" if any("clip_grad_norm" in r for r in rows) else "grad_norm"


def total_ratios(rows: list[dict]) -> list[tuple[int, float]]:
    """A copy of the total's rule: each judged step's total over the median of the last
    SPIKE_WINDOW totals before it, once SPIKE_MIN_HISTORY are known. Checked against the detector."""
    key, totals, out = grad_key(rows), [], []
    for record in sorted(rows, key=lambda r: r["step"]):
        if record.get("step_skipped"):
            continue
        total = ev._finite(record.get(key))
        if total is None:
            continue
        if len(totals) >= ev.SPIKE_MIN_HISTORY:
            base = ev._median(totals[-ev.SPIKE_WINDOW:])
            if base > 0:
                out.append((int(record["step"]), total / base))
        totals.append(total)
    return out


def flattened(rows: list[dict]) -> list[dict]:
    """Copies of the records whose finite totals are 1.0: the total's rule then never fires."""
    out = []
    for record in rows:
        record = dict(record)
        for key in ("grad_norm", "clip_grad_norm"):
            if ev._finite(record.get(key)) is not None:
                record[key] = 1.0
        out.append(record)
    return out


def longest(fires, limit: int) -> int:
    """The largest n in 1..limit with fires(n), for a rule that fires for every n up to some point."""
    if limit < 1 or not fires(1):
        return 0
    low, high = 1, 2
    while high <= limit and fires(high):
        low, high = high, high * 2
    high = min(high, limit + 1)
    while high - low > 1:
        mid = (low + high) // 2
        low, high = (mid, high) if fires(mid) else (low, mid)
    return low


def measure_spike(rows: list[dict]) -> dict:
    ratios = total_ratios(rows)
    has_params = any(r.get("grad_norms") for r in rows)
    out = {"judged_totals": len(ratios), "params": has_params, "copy_differs": []}
    if not ratios:
        return out
    step, peak = max(ratios, key=lambda sr: sr[1])
    out["peak"] = (round(peak, 2), step)
    flat = flattened(rows) if has_params else None
    for f in SPIKE_FACTORS:
        mine = [s for s, ratio in ratios if ratio >= f]
        theirs = ev.detect_grad_spikes(rows, factor=f, sustain=NEVER)
        got = (theirs or {}).get("spike_count", 0), (theirs or {}).get("spike_steps", [])
        if got != (len(mine), mine[:20]):
            out["copy_differs"].append({"factor": f, "copy": (len(mine), mine[:20]), "detector": got})
        both = ev.detect_grad_spikes(rows, factor=f)
        out[f"total@{f}"] = mine[0] if mine else None
        out[f"total_steps@{f}"] = mine
        out[f"total_count@{f}"] = len(mine)
        out[f"combined@{f}"] = (both["step"], both.get("trigger", "total")) if both else None
        if flat is not None:
            par = ev.detect_grad_spikes(flat, factor=f)
            out[f"param@{f}"] = par["step"] if par else None
            out[f"streak@{f}"] = longest(lambda s: ev.detect_grad_spikes(flat, factor=f, sustain=s) is not None,
                                         len(rows))
    if flat is not None:
        for s in SPIKE_SUSTAINS:
            par = ev.detect_grad_spikes(flat, sustain=s)
            out[f"param_sustain@{s}"] = par["step"] if par else None
    return out


# --- MoE: the diversity drop and starved experts ------------------------------------------------

@contextmanager
def diversity_drop(line: float):
    kept = ev.DIVERSITY_DROP
    ev.DIVERSITY_DROP = line
    try:
        yield
    finally:
        ev.DIVERSITY_DROP = kept


def samples_at(rows: list[dict], drop: float) -> tuple[list[dict], list[float]]:
    """Today's samples with the diversity line at `drop`, and for every call of the diversity rule
    with a full history (OVERLOAD_MIN_HISTORY samples), the router's diversity over its window's
    median, in sample order."""
    sizes = ev._router_sizes(rows)
    top_k = ev._router_top_k(rows, list(sizes), None)
    seen: list[float | None] = []
    rule = ev._jump_overloaded

    def watched(counts, k, diversity, window):
        median = ev._median([d for _, _, d in window]) if len(window) >= ev.OVERLOAD_MIN_HISTORY else None
        seen.append(diversity / median if median else None)
        return rule(counts, k, diversity, window)

    ev._jump_overloaded = watched
    try:
        with diversity_drop(drop):
            samples = ev._moe_samples(rows, sizes, top_k, ev.OVERLOAD_THRESHOLD)
    finally:
        ev._jump_overloaded = rule
    return samples, seen


def measure_diversity(rows: list[dict]) -> dict:
    sizes = ev._router_sizes(rows)
    top_k = ev._router_top_k(rows, list(sizes), None)
    wide = sorted(name for name, k in top_k.items() if k >= 3)
    out = {"routers": len(sizes), "top_k": sorted(set(top_k.values())), "wide": wide, "wrong_line": 0}
    if not wide:
        return out
    for drop in DIVERSITY_DROPS:
        samples, seen = samples_at(rows, drop)
        judged = [s for s in samples if not s.get("missing") and s["routed_tokens"] > 0
                  and s["router"] in wide]
        assert len(judged) == len(seen), "the wrapped rule was called once per judged sample"
        out["wrong_line"] += sum(1 for s in judged if s["overload_rule"].get("rule") == "diversity_drop"
                                 and s["overload_rule"]["drop"] != drop)
        pairs = [(s, r) for s, r in zip(judged, seen) if r is not None]
        out[f"full_history@{drop}"] = len(pairs)
        if pairs and drop == ev.DIVERSITY_DROP:
            low_sample, low = min(pairs, key=lambda p: p[1])
            out["lowest"] = (round(low, 3), low_sample["step"], low_sample["router"])
        drops = [s for s, r in pairs if r < drop]
        flagged = [s for s in drops if s["overload_rule"].get("rule") == "diversity_drop" and s["overloaded"]]
        out[f"below@{drop}"] = len(drops)
        out[f"flagged@{drop}"] = len(flagged)
        out[f"first_flag@{drop}"] = (flagged[0]["step"], flagged[0]["router"]) if flagged else None
        out[f"flags@{drop}"] = [(s["step"], s["router"]) for s in flagged]
        hit = ev.first_incident({"samples": samples})
        out[f"moe_imbalance@{drop}"] = (hit.get("reportable_step", hit["step"]) if hit.get("since_start") else hit["step"],
                                        hit["router"], (hit.get("overload_rule") or {}).get("rule")) if hit else None
    return out


def measure_starved(rows: list[dict]) -> dict:
    sizes = ev._router_sizes(rows)
    out = {"routers": len(sizes)}
    if not sizes:
        return out
    top_k = ev._router_top_k(rows, list(sizes), None)
    samples = ev._moe_samples(rows, sizes, top_k, ev.OVERLOAD_THRESHOLD)
    for window in STARVE_WINDOWS:
        every = copy.deepcopy(samples)
        ev._mark_starved(every, window, 1e9)  # every expert not dead, with its window's share
        shares = [(sample["starved_shares"][str(e)] * len(sample["counts"]), sample["step"], sample["router"], e)
                  for sample in every if "starved_shares" in sample and sample["starved"]
                  for e in sample["starved"]]
        out[f"eligible@{window}"] = sum(1 for sample in every if "starved_shares" in sample)
        if shares:
            low = min(shares)
            out[f"lowest@{window}"] = (round(low[0], 4), low[1], low[2], low[3])
        for ratio in STARVE_RATIOS:
            these = copy.deepcopy(samples)
            ev._mark_starved(these, window, ratio)
            hit = ev.detect_starved_experts(these)
            out[f"fires@{window}/{ratio}"] = (hit["step"], hit["experts"][0]["router"], len(hit["experts"])) if hit else None
            out[f"experts@{window}/{ratio}"] = sorted((e["step"], e["router"], e["expert"]) for e in hit["experts"]) if hit else []
    out["dead_samples"] = [(s["step"], s["router"], len(s["dead"])) for s in samples if s.get("dead")]
    return out


# --- stalled weights ----------------------------------------------------------------------------

def measure_stall(rows: list[dict]) -> dict:
    checks = sum(1 for r in rows if r.get("weights_unchanged") is not None)
    with_grads = sum(1 for r in rows if r.get("weights_unchanged") is not None and r.get("grad_norms"))
    out = {"checks": checks, "checks_with_grads": with_grads}
    if not checks:
        return out
    out["longest"] = longest(lambda n: ev.detect_stalled_weights(rows, steps=n) is not None, checks)
    for n in STALL_STEPS:
        hit = ev.detect_stalled_weights(rows, steps=n)
        out[f"fires@{n}"] = (hit["step"], hit["stalled_total"]) if hit else None
        out[f"onsets@{n}"] = [(o["step"], o["stalled"], o["trained_before"]) for o in hit["onsets"]] if hit else []
    return out


# --- today's lines against build_evidence -------------------------------------------------------

def against_evidence(rows: list[dict], spike: dict, diversity: dict, starved: dict, stall: dict) -> dict:
    evidence = ev.build_evidence(rows)
    steps = {i["kind"]: i.get("step") for i in evidence["incidents"]}
    checks = evidence.get("kind_checks", {})
    mine = {
        "grad_spike": (spike.get(f"combined@{ev.SPIKE_FACTOR}") or (None,))[0],
        "stalled_weights": (stall.get(f"fires@{ev.STALL_STEPS}") or (None,))[0],
        "starved_experts": (starved.get(f"fires@{ev.STARVE_WINDOW}/{ev.STARVE_RATIO}") or (None,))[0],
    }
    moe = diversity.get(f"moe_imbalance@{ev.DIVERSITY_DROP}") if diversity.get("wide") else None
    differ = {kind: (mine[kind], steps.get(kind)) for kind in mine if mine[kind] != steps.get(kind)}
    hit = next((i for i in evidence["incidents"] if i["kind"] == "moe_imbalance"), None)
    if diversity.get("wide"):
        theirs = (hit.get("reportable_step", hit["step"]) if hit.get("since_start") else hit["step"]) if hit else None
        if (moe[0] if moe else None) != theirs:
            differ["moe_imbalance"] = (moe, theirs)
    why = {kind: {k: checks[kind].get(k) for k in ("value", "records", "records_needed", "enough_history", "reason") if k in checks[kind]}
           for kind in ("grad_spike", "stalled_weights", "starved_experts", "moe_imbalance") if kind in checks}
    return {"differ": differ, "checks": why}


# --- printing -----------------------------------------------------------------------------------

def cell(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, tuple):
        return " ".join(str(v) for v in value if v is not None)
    return str(value)


def table(head: list[str], rows: list[list]) -> None:
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for row in rows:
        print("| " + " | ".join(cell(v) for v in row) + " |")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traincheck-from", type=Path,
                        help="a checkout whose experiments/traincheck-bugs/out holds the TrainCheck sweep "
                             "(the allowlist reads it from this checkout by default)")
    parser.add_argument("--no-traincheck", action="store_true",
                        help="measure without the TrainCheck sweep (otherwise its absence is an error)")
    parser.add_argument("--json", type=Path, default=inputs.GENERATED / "thresholds.json")
    args = parser.parse_args()

    sets, unread, sweep = read_inputs(args.traincheck_from, args.no_traincheck)
    print(f"# Record sets read: {len(sets)}\n")
    for entry_id, why in unread:
        print(f"- not read: `{entry_id}`: {why}")
    print()
    if sweep:
        print(f"## The TrainCheck sweep: {len(sweep)} run IDs, each in results.json and catalog.py, each read\n")
        table(["Run ID", "Ranks", "Records per rank", "Manifest sdk_version", "Manifest created_at", "Commit in the records"],
              [[row["run_id"], row["ranks"], ", ".join(map(str, row["records"])), row["sdk_version"], row["created_at"],
                row["commit"] or "not recorded"] for row in sweep])

    results: dict[str, dict] = {"unread": unread, "sweep": sweep, "rules": {}}
    cache: dict[tuple[str, int], dict] = {}

    def measured(rule: str, name: str, rows: list[dict]) -> dict:
        key = (rule, name, len(rows))
        if key not in cache:
            cache[key] = {"grad_spike": measure_spike, "diversity": measure_diversity,
                          "starved": measure_starved, "stall": measure_stall}[rule](rows)
        return cache[key]

    # Today's lines against build_evidence, once per record set as read (Granite's healthy part too).
    checked, differences = 0, []
    for name, rows, entry in sets:
        for part in ([rows, before(rows, GRANITE_COLLAPSE)] if name.startswith("granite") else [rows]):
            label = name if part is rows else f"{name} (before {GRANITE_COLLAPSE})"
            found = against_evidence(part, measured("grad_spike", label, part), measured("diversity", label, part),
                                     measured("starved", label, part), measured("stall", label, part))
            checked += 1
            results.setdefault("evidence", {})[label] = found
            if found["differ"]:
                differences.append((label, found["differ"]))
    copy_diffs = [(name, d) for (rule, name, _), d in cache.items() if rule == "grad_spike" and d.get("copy_differs")]
    wrong_line = [(name, d["wrong_line"]) for (rule, name, _), d in cache.items() if rule == "diversity" and d.get("wrong_line")]
    print(f"## Checks\n\n- build_evidence at today's lines: {checked} record sets, {len(differences)} differ: {differences}")
    print(f"- the total's ratio copy against detect_grad_spikes at {SPIKE_FACTORS}: {len(copy_diffs)} differ: {copy_diffs}")
    print(f"- samples whose diversity rule did not carry the line it was set to: {wrong_line}\n")

    for rule in ("grad_spike", "diversity", "starved", "stall"):
        groups: dict[str, list] = {}
        for group, name, rows, fault in rule_sets(rule, sets):
            groups.setdefault(group, []).append((name, rows, fault, measured(rule, name, rows)))
        results["rules"][rule] = {g: [(n, f, m) for n, _, f, m in items] for g, items in groups.items()}
        globals()[f"print_{rule}"](groups, results)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=1, default=str))
    print(f"(everything above, and the kind_checks of each set: {args.json})")


def judged_line(groups, ok, why) -> None:
    healthy = groups.get("healthy", [])
    judged = [(n, m) for n, _, _, m in healthy if ok(m)]
    loads = sorted({workload(fault_key(n.split(" (")[0])) for n, _ in judged})
    print(f"Healthy record sets: {len(healthy)}; judged: {len(judged)}, from {len(loads)} workloads: {'; '.join(loads)}.\n")
    reasons: dict[str, list[str]] = {}
    for n, _, _, m in healthy:
        if not ok(m):
            reasons.setdefault(why(m), []).append(short(n))
    for reason, names in reasons.items():
        print(f"- not judged, {reason} ({len(names)}): {', '.join(names)}")
    print()


def outcome(fires: list, at: int, judgeable: bool, missing: str, first) -> str:
    """One of three for a seeded fault: it fires (with its step), it is missed though the series
    the rule reads is in the capture, or that series was never in the capture. `fires` lists the
    steps the rule flags, `first` the step its incident is dated at."""
    if not judgeable:
        return f"not in the capture: {missing}"
    after = [f for f in fires if f >= at]
    if first is not None and first < at:
        return (f"fires at {first}, before its seeded step" + (f"; again at {after[0]}" if after else "; nothing at or after it"))
    if after:
        return f"fires at {after[0]}"
    return "missed, though the series is in the capture"


def spike_outcome(m: dict, at: int, factor: float, judgeable: bool) -> str:
    """The incident at this factor (either trigger), or the total's later spikes when it was earlier."""
    hit = m.get(f"combined@{factor}")
    steps = sorted(set(m.get(f"total_steps@{factor}", [])) | ({hit[0]} if hit else set()))
    said = outcome(steps, at, judgeable, "no gradient norm", hit[0] if hit else None)
    return f"{said} ({hit[1]})" if hit else said


def print_grad_spike(groups, results) -> None:
    print("## Gradient spike\n")
    ok = lambda m: m.get("judged_totals", 0) > 0
    judged_line(groups, ok, lambda m: "no gradient norm recorded" if "peak" not in m and not m.get("params")
                else f"no total past the first {ev.SPIKE_MIN_HISTORY}")
    head = ["Record set", "Records", "Largest total ratio (step)"] + [f"Total ≥{f:g}: first (count)" for f in SPIKE_FACTORS] \
        + [f"Longest parameter streak ≥{f:g}" for f in SPIKE_FACTORS] + [f"Parameter rule at 5×, sustain {s}" for s in SPIKE_SUSTAINS]

    def cells(m):
        return ([f"{m.get(f'total@{f}')} ({m.get(f'total_count@{f}')})" if m.get(f"total@{f}") is not None else "—" for f in SPIKE_FACTORS]
                + [m.get(f"streak@{f}") if m.get("params") else "no parameters" for f in SPIKE_FACTORS]
                + [m.get(f"param_sustain@{s}") for s in SPIKE_SUSTAINS]
                + [m.get(f"combined@{f}") for f in SPIKE_FACTORS])
    head += [f"Incident at {f:g}× (trigger)" for f in SPIKE_FACTORS]
    print("### Healthy\n")
    table(head, [[short(n), len(r), m.get("peak")] + cells(m) for n, r, _, m in groups.get("healthy", [])])
    head = ["Record set", "Fault"] + [f"Factor {f:g}" for f in SPIKE_FACTORS]
    for group, title in (("fault", "This rule's faults"), ("secondary", "Activation faults whose gradient also moves"),
                         ("other", "Other faults: does it fire")):
        items = groups.get(group, [])
        if not items:
            continue
        print(f"### {title}\n")
        if group == "other":
            table(["Record set", "Largest total ratio (step)"] + [f"Factor {f:g}" for f in SPIKE_FACTORS],
                  [[short(n), m.get("peak")] + [m.get(f"combined@{f}") for f in SPIKE_FACTORS] for n, _, _, m in items])
            continue
        table(head, [[short(n) + f" (seeded at {f[0]})", f[1]] + [spike_outcome(m, f[0], x, ok(m)) for x in SPIKE_FACTORS]
                     for n, _, f, m in items])
        print("Largest total ratio and longest parameter streak, for the same sets:\n")
        table(["Record set", "Largest total ratio (step)"] + [f"Longest parameter streak ≥{f:g}" for f in SPIKE_FACTORS],
              [[short(n), m.get("peak")] + [m.get(f"streak@{x}") if m.get("params") else "no parameters" for x in SPIKE_FACTORS]
               for n, _, f, m in items])


def print_diversity(groups, results) -> None:
    print("## Top-k diversity drop\n")
    ok = lambda m: bool(m.get("wide")) and m.get(f"full_history@{ev.DIVERSITY_DROP}", 0) > 0
    judged_line(groups, ok, lambda m: "no routers" if not m.get("routers") else
                f"no router with top-k of 3 or more (top-k {m.get('top_k')})" if not m.get("wide")
                else "no sample with 10 earlier unflagged samples")
    head = ["Record set", "Routers with top-k ≥ 3", "Lowest ratio to the window's median (step, router)"] \
        + [f"Line {d:g}: samples below / flagged" for d in DIVERSITY_DROPS] + [f"Line {d:g}: moe_imbalance (step, router, rule)" for d in DIVERSITY_DROPS]
    print("### Healthy\n")
    table(head, [[short(n), len(m["wide"]), m.get("lowest")] + [f"{m.get(f'below@{d}')} / {m.get(f'flagged@{d}')}" for d in DIVERSITY_DROPS]
                 + [m.get(f"moe_imbalance@{d}") for d in DIVERSITY_DROPS] for n, _, _, m in groups.get("healthy", []) if m.get("wide")])
    items = groups.get("fault", [])
    print("### This rule's faults\n")
    table(["Record set", "Fault", "Lowest ratio (step, router)"] + [f"Line {d:g}" for d in DIVERSITY_DROPS] + [f"Line {d:g}: moe_imbalance" for d in DIVERSITY_DROPS],
          [[short(n) + f" (seeded at {f[0]})", f[1], m.get("lowest")]
           + [outcome([st for st, _ in m.get(f"flags@{d}", [])], f[0], ok(m), "no router with top-k of 3 or more",
                      (m.get(f"first_flag@{d}") or (None,))[0]) for d in DIVERSITY_DROPS]
           + [m.get(f"moe_imbalance@{d}") for d in DIVERSITY_DROPS] for n, _, f, m in items])
    other = [i for i in groups.get("other", []) if i[3].get("wide")]
    if other:
        print("### Other faults with a router of top-k 3 or more\n")
        table(head, [[short(n), len(m["wide"]), m.get("lowest")] + [f"{m.get(f'below@{d}')} / {m.get(f'flagged@{d}')}" for d in DIVERSITY_DROPS]
                     + [m.get(f"moe_imbalance@{d}") for d in DIVERSITY_DROPS] for n, _, _, m in other])


def print_starved(groups, results) -> None:
    print("## Starved experts\n")
    w = ev.STARVE_WINDOW
    ok = lambda m: m.get(f"eligible@{w}", 0) > 0
    judged_line(groups, ok, lambda m: "no routers" if not m.get("routers") else
                f"no sample with a full window of {w} and {ev.STARVE_MIN_EXPECTED} expected assignments")
    for window in STARVE_WINDOWS:
        print(f"### Window {window}{' (today)' if window == w else ''}\n")
        head = ["Record set", "Samples judged", "Lowest window share, as a share of even (step, router, expert)"] \
            + [f"Ratio {r:g}: first step (router, experts)" for r in STARVE_RATIOS]
        table(head, [[short(n), m.get(f"eligible@{window}"), m.get(f"lowest@{window}")] + [m.get(f"fires@{window}/{r}") for r in STARVE_RATIOS]
                     for n, _, _, m in groups.get("healthy", []) if m.get("routers")])
        print("This rule's faults:\n")
        table(["Record set", "Fault", "Samples judged", "Lowest share of even"] + [f"Ratio {r:g}" for r in STARVE_RATIOS] + ["Samples with dead experts from the seeded step"],
              [[short(n) + f" (seeded at {f[0]})", f[1], m.get(f"eligible@{window}"), m.get(f"lowest@{window}")]
               + [outcome([e[0] for e in m.get(f"experts@{window}/{r}", [])], f[0], m.get(f"eligible@{window}", 0) > 0,
                          "no sample with a full window", (m.get(f"fires@{window}/{r}") or (None,))[0]) for r in STARVE_RATIOS]
               + [sum(1 for st, _, _ in m.get("dead_samples", []) if st >= f[0])]
               for n, _, f, m in groups.get("fault", [])])
        other = [i for i in groups.get("other", []) if i[3].get("routers")]
        if other:
            print("Other faults with routers:\n")
            table(["Record set", "Samples judged", "Lowest share of even"] + [f"Ratio {r:g}" for r in STARVE_RATIOS],
                  [[short(n), m.get(f"eligible@{window}"), m.get(f"lowest@{window}")] + [m.get(f"fires@{window}/{r}") for r in STARVE_RATIOS]
                   for n, _, _, m in other])


def print_stall(groups, results) -> None:
    print("## Stalled weights\n")
    ok = lambda m: m.get("checks", 0) >= ev.STALL_STEPS
    judged_line(groups, ok, lambda m: f"fewer than {ev.STALL_STEPS} weight checks" if m.get("checks") else "no weight check in the capture")
    head = ["Record set", "Weight checks", "Longest streak"] + [f"{n} checks: first step (parameters)" for n in STALL_STEPS]
    print("### Healthy\n")
    table(head, [[short(n), m.get("checks"), m.get("longest")] + [m.get(f"fires@{s}") for s in STALL_STEPS]
                 for n, _, _, m in groups.get("healthy", [])])
    print("### This rule's faults\n")
    table(["Record set", "Fault", "Weight checks", "Longest streak"] + [f"{n} checks" for n in STALL_STEPS],
          [[short(n) + f" (seeded at {f[0]})", f[1], m.get("checks"), m.get("longest")]
           + [outcome([o[0] for o in m.get(f"onsets@{s}", [])], f[0], bool(m.get("checks")), "no weight check",
                      (m.get(f"fires@{s}") or (None,))[0]) for s in STALL_STEPS]
           for n, _, f, m in groups.get("fault", [])])
    other = [i for i in groups.get("other", []) if i[3].get("longest")]
    print("### Other faults on which it fires at some line\n")
    table(head, [[short(n), m.get("checks"), m.get("longest")] + [m.get(f"fires@{s}") for s in STALL_STEPS] for n, _, _, m in other])


if __name__ == "__main__":
    main()
