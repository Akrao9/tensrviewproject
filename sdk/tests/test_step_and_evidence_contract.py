"""The cases of `STEP_AND_EVIDENCE_CONTRACT.md`: skipped steps, unread gradients, a rank with no
peer, a rank silent after its capture stopped, and a checkpoint whose model state is not found.

A `pin` is run here against today's Python detectors, so the behaviour the page relies on cannot
move unnoticed. A `new` case is not built: this file checks only that it is named on the page and
well formed, and the lane that builds it turns its expectation into a test of its own engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorquill import evidence as ev

ROOT = Path(__file__).parents[2]
DOCUMENT = json.loads((Path(__file__).parent / "fixtures" / "behaviour" / "cases.json").read_text())
CASES = DOCUMENT["cases"]
PINS = [(name, body) for name, body in CASES.items() if body["status"] == "pin"]
NEW = [(name, body) for name, body in CASES.items() if body["status"] == "new"]
KNOWN_KINDS = {*ev.SERVER_KINDS, *ev._STATUS_INCIDENT_KINDS}


def _numbers(value):
    """The strings "nan" and "inf" back as numbers: JSON has none."""
    if isinstance(value, dict):
        return {k: _numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_numbers(v) for v in value]
    return float(value) if value in ("nan", "inf", "-inf") else value


def test_the_file_holds_the_cases_the_page_names():
    page = (ROOT / "STEP_AND_EVIDENCE_CONTRACT.md").read_text()
    assert len(CASES) == 45 and len(PINS) == 12
    for name, body in CASES.items():
        assert f"`{name}`" in page, f"{name} is not named on the page"
        assert body["section"] in (1, 2, 3, 4) and body["says"] and body["expected"]


@pytest.mark.parametrize("name,body", PINS, ids=[n for n, _ in PINS])
def test_a_pin_is_what_the_python_detectors_do_today(name, body):
    found = getattr(ev, body["detector"])(_numbers(body["records"]))
    expected = dict(body["expected"])
    if not expected.pop("fires"):
        assert found is None
        return
    assert found is not None, "the detector raised nothing"
    assert {key: found.get(key) for key in expected} == expected


def test_the_skip_changes_the_verdict_and_nothing_else_does():
    """The three spike cases hold the same records but for `step_skipped` on one step."""
    def without_the_flag(name):
        return [{k: v for k, v in r.items() if k != "step_skipped"} for r in CASES[name]["records"]]

    assert without_the_flag("a_spike_on_a_skipped_step") == without_the_flag("a_spike_on_an_applied_step") \
        == CASES["a_spike_with_nothing_said"]["records"]


@pytest.mark.parametrize("name,body", NEW, ids=[n for n, _ in NEW])
def test_a_new_case_states_a_target_in_the_pages_words(name, body):
    assert set(body) - {"today"} == {"section", "status", "says", "given", "expected"}
    allowed = {"incident_kinds_present", "incident_kinds_absent", "incident_kind_one_of", "kind_values", "not_checked",
               "run_state", "status", "words", "words_absent", "incident_carries", "incident_lacks", "findings",
               "records_carry", "records_lack", "records", "evidence_carries", "evidence_lacks", "counters_at_end"}
    assert set(body["expected"]) <= allowed, set(body["expected"]) - allowed
    for kind in (*body["expected"].get("incident_kinds_present", ()), *body["expected"].get("incident_kinds_absent", ()),
                 *body["expected"].get("incident_kind_one_of", ()), *body["expected"].get("kind_values", {})):
        assert kind in KNOWN_KINDS, f"{kind} is not a kind the evidence knows"
    for value in body["expected"].get("kind_values", {}).values():
        assert value in ("checked", "not_configured", "unavailable")


NOT_BUILT = [(name, body) for name, body in NEW if "today" in body]


@pytest.mark.parametrize("name,body", NOT_BUILT, ids=[n for n, _ in NOT_BUILT])
def test_a_new_case_over_records_is_not_what_the_python_detectors_do_today(name, body):
    """`today` is what today's detector answers on the case's records, and it is not the target. So
    the case states something to build. The lane that builds the rule removes `today` from the case
    and makes it a pin."""
    today, expected = dict(body["today"]), body["expected"]
    found = getattr(ev, today.pop("detector"))(_numbers(body["given"]["records"]))
    assert (found is not None) == today.pop("fires")
    assert {key: found.get(key) for key in today} == today
    if found["kind"] in expected.get("incident_kinds_absent", ()):
        return  # raised today, and the target is that it is not
    carried = expected["incident_carries"]
    assert any(found.get(key) != value for key, value in carried.items()), "today's answer is already the target"


def test_the_three_provenances_are_the_same_records_but_for_who_reported():
    """Each family of cases by who reported the skip differs in `step_skipped_by` alone."""
    def bare(name):
        body = CASES[name]
        records = body["records"] if body["status"] == "pin" else body["given"]["records"]
        return [{k: v for k, v in r.items() if k != "step_skipped_by"} for r in records]

    for family in ("weights_unchanged_across_skips_{}_reported", "an_activation_not_finite_on_a_skip_{}_reported",
                   "a_gradient_not_finite_on_a_skip_{}_reported"):
        assert bare(family.format("the_caller")) == bare(family.format("the_scaler")) == bare(family.format("both"))
    assert bare("ten_skips_only_the_caller_reported") == bare("ten_skips_the_scaler_reported") == bare("ten_skips_both_reported")
