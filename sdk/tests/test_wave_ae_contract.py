"""The cases of `WAVE_AE_CONTRACT.md`: separately clipped groups, a stall and a resume, several
optimizers through one scaler, and an input that holds several run IDs.

A `pin` is run here against the Python code of the tree, so behaviour the page keeps cannot move
unnoticed. A `new` case is not built on the base: this file checks only that it is named on the
page and well formed, and that its `today` (a record of the base, never recomputed) is not its
target. The lane that builds a rule turns each new case into a test of its own engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import conftest
import pytest

from tensorquill import divergence as dv
from tensorquill import evidence as ev

ROOT = Path(__file__).parents[2]
DOCUMENT = json.loads((Path(__file__).parent / "fixtures" / "wave_ae" / "cases.json").read_text())
CASES = DOCUMENT["cases"]
PINS = [(name, body) for name, body in CASES.items() if body["status"] == "pin"]
NEW = [(name, body) for name, body in CASES.items() if body["status"] == "new"]


def answer(body: dict) -> dict:
    """What the tree's Python code answers on a case's input, under the keys the cases use."""
    given, expected = body["given"], body["expected"]
    if body["section"] == 5:
        result = dv.first_divergence(given["records_a"], given["records_b"])
        return {"answer": result["answer"], "several_run_ids": result.get("several_run_ids")}
    records = given["records"]
    if "skipped_updates" in expected:
        return {"skipped_updates": ev.skipped_updates(records)}
    if "stalled_weights_step" in expected:
        found = ev.detect_stalled_weights(sorted(ev.resolve_attempts(records), key=lambda r: int(r["step"])))
        return {"stalled_weights_step": None if found is None else found["step"]}
    if body["section"] == 1:
        window = given.get("window", ev.UPDATE_WINDOW)
        found = ev.detect_update_spikes(records, window=window)
        entries = [e for e in ev.build_evidence(records, update_window=window).get("not_checked", [])
                   if e.get("check") == "clip_groups"]
        built = ev.build_evidence(records, update_window=window)
        return {"update_spike_step": None if found is None else found["step"],
                "clip_groups_entry": entries[0] if entries else None,
                "update_spike_check": built["kind_checks"]["update_spike"]["value"],
                "not_checked_checks": [e.get("check") for e in built.get("not_checked", [])]}
    found, counters = ev._loss_scale_judged(records)
    out = {"loss_scale_trouble_step": None if found is None else found["step"],
           "reason": None if found is None else found["reason"],
           "consecutive_skips": None if found is None else found["consecutive_skips"],
           "counters_at_end": {**counters, "partial_steps": counters.get("partial_steps", 0)}}
    if found is not None and "partial_steps" in found:
        out["incident_partial_steps"] = found["partial_steps"]
    return out


def matches(found: dict, expected: dict) -> bool:
    """Every expected key holds. A `clip_groups_entry` is held to the keys the page fixes; its
    `reason` is words, which each engine's own test pins."""
    for key, value in expected.items():
        got = found.get(key)
        if key == "clip_groups_entry" and isinstance(value, dict):
            if not isinstance(got, dict) or {k: got.get(k) for k in value} != value:
                return False
        elif got != value:
            return False
    return True


def test_the_file_holds_the_cases_the_page_names():
    page = conftest.contract_page("WAVE_AE_CONTRACT.md")
    assert len(CASES) == 36 and len(PINS) == 21 and len(NEW) == 15
    assert DOCUMENT["base"] == "265311c"
    for name, body in CASES.items():
        assert f"`{name}`" in page, f"{name} is not named on the page"
        assert body["section"] in (1, 2, 4, 5) and body["says"] and body["given"] and body["expected"]
        assert ("today" in body) == (body["status"] == "new"), name


@pytest.mark.parametrize("name,body", PINS, ids=[n for n, _ in PINS])
def test_a_pin_holds_on_this_tree(name, body):
    found = answer(body)
    assert matches(found, body["expected"]), f"{found} is not {body['expected']}"


@pytest.mark.parametrize("name,body", NEW, ids=[n for n, _ in NEW])
def test_a_new_case_states_a_target_the_base_did_not_meet(name, body):
    assert not matches(body["today"], body["expected"]), "the base already gave the target"


def test_the_paired_regression_of_section_4_is_the_authors():
    """Five full skips, one partial step, then five full skips must end with a streak of five."""
    body = CASES["five_full_skips_one_partial_step_five_full_skips"]
    records = body["given"]["records"]
    assert [bool(r.get("step_skipped_part")) for r in records] == [False] * 5 + [True] + [False] * 5
    assert all(r["step_skipped"] and r["step_skipped_by"] == "scaler" for r in records)
    assert body["expected"]["counters_at_end"] == {"recorded_steps": 11, "skipped_steps": 10, "streak": 5, "partial_steps": 1}
    assert body["expected"]["loss_scale_trouble_step"] is None


def test_the_mixed_input_of_section_5_hides_its_other_run_in_the_middle():
    """The first and last rows say `r1`: a reader that checks only the endpoints sees one run."""
    rows = CASES["an_input_with_another_run_id_only_in_the_middle"]["given"]["records_a"]
    assert rows[0]["run_id"] == rows[-1]["run_id"] == "r1"
    assert {r["run_id"] for r in rows} == {"r1", "r2"}
    assert [r["step"] for r in rows] == sorted(r["step"] for r in rows)
