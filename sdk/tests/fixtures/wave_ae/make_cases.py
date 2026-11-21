"""Writes `cases.json`, the shared cases of `WAVE_AE_CONTRACT.md`.

    PYTHONPATH=sdk/src python sdk/tests/fixtures/wave_ae/make_cases.py

Run on the base of wave AE (`265311c`), before any rule of the page is built. Each case has
`section`, `status`, `says`, `given` and `expected`:

- a `pin` is behaviour the base already has and the page keeps; its `expected` is checked here
  against the base's Python detectors, and `test_wave_ae_contract.py` runs it again on every tree;
- a `new` case is not built on the base. Its `expected` is the target, written by hand from the
  page. Its `today` is what the base's Python detectors answer on the same input, computed here
  and written as a literal: it is a record of the base, never recomputed by a test, and this
  script refuses to write a `new` case whose `today` already equals its `expected`.

`given.records` are one rank's step records (a detector's input, in any order); `given.window`
is `update_spike`'s window where the case sets it. Sections 3 (the capture's `ends_step=`) and the
reader half of 5 (`runs`, `compare`, the MCP tools) have no case here: their regressions run a
training loop or read files, and are named on the page for the lane that builds them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tensorquill import divergence as dv
from tensorquill import evidence as ev

HERE = Path(__file__).resolve().parent
CASES: dict[str, dict[str, Any]] = {}


def case(name: str, section: int, status: str, says: str, given: dict, expected: dict, today: dict | None = None) -> None:
    assert name not in CASES and status in ("pin", "new")
    body: dict[str, Any] = {"section": section, "status": status, "says": says, "given": given, "expected": expected}
    if today is not None:
        body["today"] = today
    CASES[name] = body


# --- what the base answers ------------------------------------------------------------------

def update_spike(given: dict) -> dict:
    found = ev.detect_update_spikes(given["records"], window=given.get("window", ev.UPDATE_WINDOW))
    entries = [e for e in ev.build_evidence(given["records"], update_window=given.get("window", ev.UPDATE_WINDOW))
               .get("not_checked", []) if e.get("check") == "clip_groups"]
    return {"update_spike_step": None if found is None else found["step"], "clip_groups_entry": entries[0] if entries else None}


def stalled(given: dict) -> dict:
    rows = sorted(ev.resolve_attempts(given["records"]), key=lambda r: int(r["step"]))
    found = ev.detect_stalled_weights(rows)
    return {"stalled_weights_step": None if found is None else found["step"]}


def loss_scale(given: dict) -> dict:
    found, counters = ev._loss_scale_judged(given["records"])
    out = {"loss_scale_trouble_step": None if found is None else found["step"],
           "reason": None if found is None else found["reason"],
           "consecutive_skips": None if found is None else found["consecutive_skips"],
           "counters_at_end": {**counters, "partial_steps": counters.get("partial_steps", 0)}}
    if found is not None and "partial_steps" in found:
        out["incident_partial_steps"] = found["partial_steps"]
    return out


def skipped_block(given: dict) -> dict:
    return {"skipped_updates": ev.skipped_updates(given["records"])}


def divergence(given: dict) -> dict:
    result = dv.first_divergence(given["records_a"], given["records_b"])
    return {"answer": result["answer"], "several_run_ids": result.get("several_run_ids")}


# --- 1. separately clipped groups -------------------------------------------------------------

def clip_rows(calls_at: dict[int, Any] | None = None, *, split: bool = True, late_jump: bool = False) -> list[dict]:
    """Steps 0 to 4 name `a` and `b`, each update a hundredth larger than the last, with a
    pre-clipping total of 1. At step 5 `a`'s update jumps to 2.0 and the total to 2.0. With
    `split`, step 5 also names a third parameter `c` (unfrozen then), so the two records' totals
    are taken over `a` and `b` alone, which needs each record's total split by parameter; without
    it the two records name the same parameters and their totals are compared as recorded.
    `late_jump` adds a step 6 naming what step 5 names, whose update jumps again."""
    rows = []
    for step in range(5):
        rows.append({"step": step, "clip_grad_norm": 1.0, "grad_norms": {"a": 0.6, "b": 0.8},
                     "update_rms": {"a": 1.0 + 0.01 * step, "b": 0.5}})
    names = {"a": 0.6, "b": 0.8, "c": 1.0} if split else {"a": 1.2, "b": 1.6}
    sizes = {"a": 2.0, "b": 0.5, "c": 0.1} if split else {"a": 2.0, "b": 0.5}
    rows.append({"step": 5, "clip_grad_norm": 2.0, "grad_norms": names, "update_rms": sizes})
    if late_jump:
        rows.append({"step": 6, "clip_grad_norm": 4.0, "grad_norms": {k: 2 * v for k, v in names.items()},
                     "update_rms": {**sizes, "a": 6.0}})
    for step, calls in (calls_at or {}).items():
        rows[step]["clip_calls"] = calls
    return rows


ENTRY = {"check": "clip_groups", "kinds": ["update_spike"], "comparisons": 1, "step": 5}

given = {"records": clip_rows({5: 2}), "window": 3}
case("a_split_pair_with_two_clip_calls_is_not_compared", 1, "new",
     "Step 5 names a parameter step 4 does not, so comparing them needs each record's pre-clipping total split by "
     "parameter, and step 5 was clipped by two calls. The pair is not compared: no incident, and one `not_checked` "
     "entry says so. Today the split is made as if one call had clipped everything, and the pair is flagged.",
     given, {"update_spike_step": None, "clip_groups_entry": ENTRY}, update_spike(given))

given = {"records": clip_rows({4: 2}), "window": 3}
case("a_split_pair_whose_earlier_record_had_two_clip_calls", 1, "new",
     "The same, with the two calls on the earlier record of the pair. Either record is enough.",
     given, {"update_spike_step": None, "clip_groups_entry": ENTRY}, update_spike(given))

given = {"records": clip_rows({5: 2}, late_jump=True), "window": 3}
case("after_a_pair_that_is_not_compared_the_next_is", 1, "new",
     "The pair of steps 4 and 5 is not compared, and its change joins no baseline. Step 5 is still the record step 6 "
     "is compared with: they name the same parameters, so their totals are used as recorded, and step 6's jump is "
     "flagged. Today the incident is at step 5.",
     given, {"update_spike_step": 6, "clip_groups_entry": ENTRY}, update_spike(given))

given = {"records": clip_rows(), "window": 3}
case("a_split_pair_with_one_clip_call_is_compared", 1, "pin",
     "The control: the same records with no `clip_calls`. The split is made and step 5 is flagged, as on the base.",
     given, {"update_spike_step": 5, "clip_groups_entry": None})

given = {"records": clip_rows({5: 2, 4: 2}, split=False), "window": 3}
case("a_pair_naming_the_same_parameters_is_compared_whatever_the_clip_calls", 1, "pin",
     "Both records name `a` and `b`, so their totals are compared as recorded and nothing is split. Two clip calls on "
     "both change nothing: step 5 is flagged, and there is no `not_checked` entry.",
     given, {"update_spike_step": 5, "clip_groups_entry": None})

for label, value in (("one", 1), ("text", "2"), ("a_fraction", 2.5), ("true", True)):
    given = {"records": clip_rows({5: value}), "window": 3}
    case(f"clip_calls_that_is_{label}_is_read_as_no_key", 1, "pin",
         f"`clip_calls` is a whole number above 1, or it says nothing. Here it is {value!r}: the pair is compared as if "
         "the key were absent.",
         given, {"update_spike_step": 5, "clip_groups_entry": None})


# --- 2. a stall and a resume ------------------------------------------------------------------

def stall_rows(attempt: int, steps: list[int], listed: dict[int, list[str]]) -> list[dict]:
    """One attempt's rows at `steps`. A step in `listed` is a weight check that compared, with the
    parameters it found unchanged; any other row is no weight check (a process's first check
    compares nothing and writes no list)."""
    rows = []
    for step in steps:
        row: dict[str, Any] = {"rank": 0, "attempt": attempt, "step": step, "grad_norm": 1.0, "grad_norms": {"p": 0.5, "q": 0.5}}
        if step in listed:
            row["weights_unchanged"] = listed[step]
        rows.append(row)
    return rows


P = ["p"]
given = {"records": stall_rows(1, [0, 50, 100], {50: P, 100: P}) + stall_rows(2, [101, 150, 200, 250, 300], {200: P, 250: P, 300: P})}
case("a_stall_does_not_carry_across_a_resume", 2, "new",
     "`p` is unchanged at the checks of steps 50 and 100. The job is resumed at step 101. The new process's first "
     "check, at 150, compares nothing; `p` is unchanged at 200, 250 and 300. The streak starts again at the new "
     "attempt, so the stall is raised at 300. Today the two checks before the resume count, and it is raised at 200, "
     "on three comparisons with the interval from 100 to 150 never compared.",
     given, {"stalled_weights_step": 300}, stalled(given))

given = {"records": stall_rows(1, [0, 50, 100], {50: P, 100: P}) + stall_rows(2, [101, 150, 200, 250], {150: P, 200: P, 250: P})}
case("a_new_attempt_whose_first_check_compared", 2, "new",
     "The same, but the new attempt's first weight check carries a list (a capture that could compare). It counts as "
     "the first check of the new streak, so the stall is raised at 250. Today it is raised at 150.",
     given, {"stalled_weights_step": 250}, stalled(given))

given = {"records": stall_rows(1, [0, 50, 100, 150, 200], {50: P, 100: P, 150: P, 200: P})}
case("a_stall_in_one_attempt", 2, "pin",
     "The control without a resume: unchanged at 50, 100 and 150 raises the stall at 150, as on the base.",
     given, {"stalled_weights_step": 150})

given = {"records": stall_rows(1, [0, 50, 100], {50: [], 100: []}) + stall_rows(2, [101, 150, 200, 250, 300], {200: P, 250: P, 300: P})}
case("a_stall_wholly_inside_the_resumed_attempt", 2, "pin",
     "`p` changed at both checks before the resume and is unchanged at three after it: raised at 300, on the base and "
     "after.",
     given, {"stalled_weights_step": 300})

given = {"records": stall_rows(1, [0, 50, 100, 150], {50: P, 100: P, 150: P}) + stall_rows(2, [151, 200, 250], {250: P})}
case("a_stall_raised_before_the_resume_stands", 2, "pin",
     "The stall is raised at 150, before the resume at 151. What comes after does not take it back or date it again.",
     given, {"stalled_weights_step": 150})


# --- 4. several optimizers, one scaler --------------------------------------------------------

def skip(step: int, by: str = "scaler", **more: Any) -> dict:
    return {"step": step, "step_skipped": True, "step_skipped_by": by, **more}


PART = {"skipped": 1, "of": 2}
full, partial = lambda steps: [skip(s) for s in steps], lambda s, by="scaler": skip(s, by, step_skipped_part=PART)

given = {"records": full(range(5)) + [partial(5)] + full(range(6, 11))}
case("five_full_skips_one_partial_step_five_full_skips", 4, "new",
     "Five updates the scaler skipped for every optimizer, one step on which it skipped one optimizer of two and "
     "applied the other, and five more full skips. An update occurred at the sixth step, so the streak of fully "
     "skipped updates ends there: 11 recorded, 10 skipped, 1 partial, a streak of 5, and nothing is raised. Today the "
     "partial step counts as a skip and the rule is raised at the tenth in a row.",
     given, {"loss_scale_trouble_step": None, "reason": None, "consecutive_skips": None,
             "counters_at_end": {"recorded_steps": 11, "skipped_steps": 10, "streak": 5, "partial_steps": 1}},
     loss_scale(given))

given = {"records": full(range(5)) + [partial(5, "both")] + full(range(6, 11))}
case("a_partial_step_the_caller_also_reported", 4, "new",
     "The same, with the caller reporting a skip on the partial step as well. The scaler confirmed a mixed outcome, "
     "so the counts are the same: 11 recorded, 10 skipped, 1 partial, a streak of 5.",
     given, {"loss_scale_trouble_step": None, "reason": None, "consecutive_skips": None,
             "counters_at_end": {"recorded_steps": 11, "skipped_steps": 10, "streak": 5, "partial_steps": 1}},
     loss_scale(given))

given = {"records": [partial(0)] + full(range(1, 11))}
case("a_partial_step_then_ten_full_skips", 4, "new",
     "A partial step and then ten full skips. The ten raise the rule at the tenth of them, step 10, and the incident "
     "carries `partial_steps: 1`. Today the partial step is the first of the ten and the rule is raised at step 9.",
     given, {"loss_scale_trouble_step": 10, "reason": "skips", "consecutive_skips": 10, "incident_partial_steps": 1,
             "counters_at_end": {"recorded_steps": 11, "skipped_steps": 10, "streak": 10, "partial_steps": 1}},
     loss_scale(given))

given = {"records": full(range(11))}
case("eleven_full_skips_with_one_optimizer", 4, "pin",
     "The single-optimizer control: eleven skips with no `step_skipped_part`. Raised at the tenth, step 9, as on the base.",
     given, {"loss_scale_trouble_step": 9, "reason": "skips", "consecutive_skips": 10,
             "counters_at_end": {"recorded_steps": 11, "skipped_steps": 11, "streak": 11, "partial_steps": 0}})

for label, value in (("every_optimizer_skipped", {"skipped": 2, "of": 2}), ("no_optimizer_skipped", {"skipped": 0, "of": 2}),
                     ("is_not_two_counts", "1 of 2")):
    given = {"records": full(range(5)) + [skip(5, step_skipped_part=value)] + full(range(6, 11))}
    case(f"a_part_key_that_says_{label}_is_a_full_skip" if label != "is_not_two_counts" else "a_part_key_that_is_not_two_counts_is_a_full_skip",
         4, "pin",
         f"`step_skipped_part` states a mixed outcome only as two whole numbers with 0 < skipped < of. Here it is "
         f"{json.dumps(value)} on a skipped step: no update is known to have occurred, so the step is a full skip, and "
         "the rule is raised at the tenth in a row, step 9.",
         given, {"loss_scale_trouble_step": 9, "reason": "skips", "consecutive_skips": 10,
                 "counters_at_end": {"recorded_steps": 11, "skipped_steps": 11, "streak": 11, "partial_steps": 0}})

given = {"records": full(range(3)) + [partial(3) | {"grad_scale": 1e-9}]}
case("a_low_loss_scale_on_a_partial_step", 4, "pin",
     "A loss scale below the floor measured on a partial step raises the rule with reason `scale`: the measurement is "
     "the evidence, as on any step.",
     given, {"loss_scale_trouble_step": 3, "reason": "scale"})

given = {"records": [
    {"step": 0, "step_skipped": False, "weights_unchanged": ["blk.weight"], "grad_norms": {"blk.weight": 0.5}},
    {"step": 1, "step_skipped": False, "weights_unchanged": ["blk.weight"], "grad_norms": {"blk.weight": 0.5}},
    {**partial(2), "weights_unchanged": ["blk.weight"], "grad_norms": {"blk.weight": 0.5}},
    {"step": 3, "step_skipped": False, "weights_unchanged": ["blk.weight"], "grad_norms": {"blk.weight": 0.5}},
]}
case("a_stall_across_a_partial_step", 4, "pin",
     "The record does not say which optimizer's update was applied, so a partial step is read by the per-parameter "
     "kinds as a skipped one: it adds nothing to a stall's streak and does not end it. Unchanged at two applied "
     "steps, at the partial one and at one more applied step raises the stall at the last, step 3.",
     given, {"stalled_weights_step": 3})

given = {"records": full(range(5)) + [partial(5)] + full(range(6, 11))}
case("skipped_updates_counts_a_partial_step_apart", 4, "new",
     "The evidence's `skipped_updates` counts the partial step under `partial` and under no other name. The key "
     "`partial` is there only when its count is above zero.",
     given, {"skipped_updates": {"caller": 0, "scaler": 10, "both": 0, "partial": 1}}, skipped_block(given))

given = {"records": full(range(4)) + [skip(4, "caller"), skip(5, "both")]}
case("skipped_updates_without_a_partial_step", 4, "pin",
     "With no partial step the block is what it is on the base, with no `partial` key.",
     given, {"skipped_updates": {"caller": 1, "scaler": 4, "both": 1}})


# --- 5. several run IDs in one input ----------------------------------------------------------

def run_rows(run_id: str, steps: range, level: float = 1.0) -> list[dict]:
    return [{"run_id": run_id, "rank": 0, "attempt": 1, "step": step, "grad_norm": level + 0.001 * (step % 7),
             "scalars": {"train/loss": 2.0 * level - 0.01 * step}} for step in steps]


def mixed() -> list[dict]:
    """`r1` on steps 0 to 19 and 40 to 59, `r2` on 20 to 39: the first and the last row say `r1`."""
    return run_rows("r1", range(0, 20)) + run_rows("r2", range(20, 40), 5.0) + run_rows("r1", range(40, 60))


given = {"records_a": run_rows("r1", range(60)), "records_b": run_rows("r2", range(60))}
case("two_clean_runs_with_different_ids_are_compared", 5, "pin",
     "Run A is all `r1` and run B all `r2`. That is what the call is for: the two are compared, and the answer is not "
     "`no_answer`.",
     given, {"answer": divergence(given)["answer"], "several_run_ids": None})
assert CASES["two_clean_runs_with_different_ids_are_compared"]["expected"]["answer"] in ("no_divergence", "diverged")

given = {"records_a": mixed(), "records_b": run_rows("r3", range(60))}
case("an_input_with_another_run_id_only_in_the_middle", 5, "new",
     "Run A's rows say `r1` at the start and at the end and `r2` in between; run B is clean. A is not one run, so "
     "there is no answer, and the result names the side and its IDs. A reader that looks only at the first and last "
     "rows would not see it. Today the rows are read as one run.",
     given, {"answer": "no_answer", "several_run_ids": {"a": ["r1", "r2"]}}, divergence(given))

given = {"records_a": run_rows("r3", range(60)), "records_b": mixed()}
case("the_other_input_with_two_run_ids", 5, "new",
     "The same with the mixed rows given as run B: the result names side `b`.",
     given, {"answer": "no_answer", "several_run_ids": {"b": ["r1", "r2"]}}, divergence(given))

given = {"records_a": mixed(), "records_b": run_rows("r3", range(0, 30)) + run_rows("r4", range(30, 60))}
case("both_inputs_with_two_run_ids", 5, "new",
     "Both inputs hold two run IDs: both sides are named.",
     given, {"answer": "no_answer", "several_run_ids": {"a": ["r1", "r2"], "b": ["r3", "r4"]}}, divergence(given))

twice = run_rows("r1", range(60))
given = {"records_a": run_rows("r1", range(60)), "records_b": twice + [dict(twice[30], grad_norm=1.004)]}
case("a_step_written_twice_within_one_run", 5, "pin",
     "Run B holds step 30 twice, both rows of its own run ID. That is one run, and it is compared: the later row is "
     "the one read. (The base's test of this made its second row from another run's, which is now two run IDs.)",
     given, {"answer": divergence(given)["answer"], "several_run_ids": None})
assert CASES["a_step_written_twice_within_one_run"]["expected"]["answer"] != "no_answer"


# --- check against the base, and write --------------------------------------------------------

BASE = {1: update_spike, 2: stalled, 5: divergence}


def base_answer(name: str, body: dict) -> dict:
    if body["section"] == 4:
        if "skipped_updates" in body["expected"]:
            return skipped_block(body["given"])
        if "stalled_weights_step" in body["expected"]:
            return stalled(body["given"])
        return loss_scale(body["given"])
    return BASE[body["section"]](body["given"])


for name, body in CASES.items():
    answer = base_answer(name, body)
    if body["status"] == "pin":
        got = {key: answer.get(key) for key in body["expected"]}
        assert got == body["expected"], f"{name}: the base gives {got}, the pin says {body['expected']}"
    else:
        assert body["today"] == answer
        assert {key: answer.get(key) for key in body["expected"]} != body["expected"], f"{name}: the base already gives the target"

document = {
    "contract": "WAVE_AE_CONTRACT.md",
    "base": "265311c",
    "cases": CASES,
}
(HERE / "cases.json").write_text(json.dumps(document, indent=1) + "\n")
print(f"{len(CASES)} cases: {sum(b['status'] == 'pin' for b in CASES.values())} pins, {sum(b['status'] == 'new' for b in CASES.values())} new")
for name, body in CASES.items():
    if body["status"] == "new":
        print(f"  {name}: today {json.dumps(body['today'])[:150]}")
