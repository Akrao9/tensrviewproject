"""The cases of `DETECTOR_RULES_CONTRACT.md` (wave AD): the score beside a ratio, the two changed
rules, and four measurements.

Nothing of the page is built when this file is written. It checks that every case is named on the
page, that the file is what its generator writes, and what each rule case's `today` records: the
answer of `eb5a71b`. The two tests named `..._records_what_the_rule_says_today` hold the code to that
answer: the Python detector lane (AD7) replaces each with a test of the case's `expected` when it
builds the rule. `today` in the file is a literal, so the generator's output does not move.
"""

from __future__ import annotations

import json
from pathlib import Path

import conftest
import pytest

from tensorquill import evidence as ev

ROOT = Path(__file__).parents[2]
FOLDER = Path(__file__).parent / "fixtures" / "rules"
DOCUMENT = json.loads((FOLDER / "cases.json").read_text())
CASES = DOCUMENT["cases"]
PARTS = {"score": 8, "score_records": 4, "activation_drift": 9, "moe_imbalance": 11, "measurements": 10, "evidence": 4, "wire": 10}


def of(part):
    return [(name, body) for name, body in CASES.items() if body["part"] == part]


def test_the_file_holds_the_cases_the_page_names():
    page = conftest.contract_page("DETECTOR_RULES_CONTRACT.md")
    assert {part: len(of(part)) for part in PARTS} == PARTS and len(CASES) == sum(PARTS.values())
    for name, body in CASES.items():
        assert f"`{name}`" in page, f"{name} is not named on the page"
        assert body["says"] and "expected" in body


def test_the_constants_are_the_pages():
    assert DOCUMENT["constants"] == {"score_min": 10, "score_scale": 1.4826, "drift_factor": 3.0,
                                     "drift_records_needed": 110, "start_grace": 10, "trend_min": 6, "listed": 10}


def _incident(evidence, kind):
    found = next((dict(i) for i in evidence["incidents"] if i["kind"] == kind), None)
    if found is not None:
        found.pop("class", None)
    return json.loads(json.dumps(found))


@pytest.mark.parametrize("name,body", of("activation_drift"), ids=[n for n, _ in of("activation_drift")])
def test_a_drift_case_gives_what_the_page_expects(name, body):
    evidence = ev.build_evidence(body["records"], detectors="python")
    expected = body["expected"]
    assert _incident(evidence, "activation_drift") == expected["incident"], name
    check = evidence["kind_checks"]["activation_drift"]
    assert (check["records_needed"], check["enough_history"]) == (expected["records_needed"], expected["enough_history"]), name


def test_the_reader_words_for_a_score_are_the_ones_the_author_gave():
    """A reader test for the pinned example: the words must be true of a window that varies."""
    for name, body in of("score"):
        expected = body["expected"]
        if expected["score"] is None and expected["score_note"] == "no_dispersion":
            assert expected["display"] == "no score: the earlier values had zero median absolute deviation", name


@pytest.mark.parametrize("name,body", of("moe_imbalance"), ids=[n for n, _ in of("moe_imbalance")])
def test_an_imbalance_case_gives_what_the_page_expects(name, body):
    evidence = ev.build_evidence(body["records"], detectors="python")
    expected = body["expected"]
    assert _incident(evidence, "moe_imbalance") == expected["incident"], name
    # The public function gives the same sample the incident is built from.
    hit = ev.first_incident(evidence)
    assert (hit and {k: hit[k] for k in expected["incident"] if k != "kind"}) == (
        expected["incident"] and {k: v for k, v in expected["incident"].items() if k != "kind"}), name
    check = evidence["kind_checks"]["moe_imbalance"]
    assert check.get("starting_state", []) == expected["starting_state"], name
    if expected["starting_state"]:
        assert check["enough_history"] is False, name


def test_the_cases_show_each_thing_the_page_says_a_rule_gives_up_or_keeps():
    drift = {name: body["expected"]["incident"] for name, body in of("activation_drift")}
    assert drift["drift_growth_from_scratch_is_not_a_shift"] is None and drift["drift_a_slow_drift_is_not_seen"] is None
    assert drift["drift_a_step_change_is_named"]["step"] == 305 and drift["drift_a_fall_is_named_too"]["modules"][0]["ratio"] < 1
    assert not CASES["drift_not_judged_before_110_records"]["expected"]["enough_history"]
    start = {name: body["expected"]["incident"] for name, body in of("moe_imbalance")}
    assert start["start_two_samples_then_even"] is None and start["start_nine_samples_then_even"] is None
    assert start["start_ten_samples_is_reported"] == {**start["start_ten_samples_is_reported"], "step": 0, "since_start": True,
                                                      "reportable_step": 9}
    assert start["start_in_a_sparse_capture"]["reportable_step"] == 450
    assert start["a_start_and_an_earlier_reportable_failure"]["router"] == "b"
    assert "reportable_step" not in start["not_at_step_0_is_not_a_start"]
    assert start["not_at_step_0_is_not_a_start"]["step"] == 5 and "since_start" not in start["not_at_step_0_is_not_a_start"]
    assert start["one_router_starts_another_fails"]["router"] == "b"
    assert start["start_and_the_capture_ends"] is None
    assert CASES["start_and_the_capture_ends"]["expected"]["starting_state"] == [{"router": "r", "samples": 5}]
    assert CASES["start_two_samples_then_even"]["expected"]["starting_state"] == []
    zero_mad = "no score: the earlier values had zero median absolute deviation"
    assert CASES["score_no_dispersion"]["expected"] == {"score": None, "score_note": "no_dispersion", "display": zero_mad}
    # The author's example: six equal values and four different ones vary, and still have a zero MAD.
    six_and_four = CASES["score_six_alike_and_four_different"]
    assert len(set(six_and_four["window"])) == 5 and six_and_four["expected"]["display"] == zero_mad
    assert "did not vary" not in json.dumps(DOCUMENT)
    assert CASES["score_nine_earlier_values"]["expected"]["display"] == "no score: fewer than 10 earlier values"
    assert CASES["score_above"]["expected"]["display"] == "score 49.6"
    assert CASES["capture_stopped_at_step_0"]["expected"]["status"]["capture_stopped"]["step"] == 0
    assert CASES["capture_stopped_before_any_step"]["expected"]["status"]["capture_stopped"]["step"] is None
    assert "step" not in CASES["capture_stopped_from_an_older_sender"]["expected"]["status"]["capture_stopped"]
    assert [body["expected"]["words"].get("r") for name, body in of("wire") if name.startswith("moe_source")].count(
        "the dispatched assignments") == 1
    assert CASES["gradient_trend_a_plateau"]["expected"]["gradient_trend"]["growth"] == 1.0
