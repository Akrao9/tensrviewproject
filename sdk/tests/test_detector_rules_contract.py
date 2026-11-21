"""The cases of `DETECTOR_RULES_CONTRACT.md` (wave AD): the score beside a ratio, the two changed
rules, and four measurements.

Nothing of the page is built when this file is written. It checks that every case is named on the
page, that the file is what its generator writes, and what each rule case's `today` records: the
answer of `eb5a71b`. The two tests named `..._records_what_the_rule_says_today` hold the code to that
answer: the Python detector lane (AD7) replaces each with a test of the case's `expected` when it
builds the rule. `today` in the file is a literal, so the generator's output does not move.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tensorquill import evidence as ev

ROOT = Path(__file__).parents[2]
FOLDER = Path(__file__).parent / "fixtures" / "rules"
DOCUMENT = json.loads((FOLDER / "cases.json").read_text())
CASES = DOCUMENT["cases"]
PARTS = {"score": 7, "activation_drift": 9, "moe_imbalance": 10, "measurements": 10}


def of(part):
    return [(name, body) for name, body in CASES.items() if body["part"] == part]


def test_the_file_holds_the_cases_the_page_names():
    page = (ROOT / "DETECTOR_RULES_CONTRACT.md").read_text()
    assert {part: len(of(part)) for part in PARTS} == PARTS and len(CASES) == sum(PARTS.values())
    for name, body in CASES.items():
        assert f"`{name}`" in page, f"{name} is not named on the page"
        assert body["says"] and "expected" in body


def test_the_file_is_what_its_generator_writes():
    spec = importlib.util.spec_from_file_location("make_rule_cases", FOLDER / "make_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert json.loads(json.dumps(module.build())) == DOCUMENT


def test_the_constants_are_the_pages():
    assert DOCUMENT["constants"] == {"score_min": 10, "score_scale": 1.4826, "drift_factor": 3.0,
                                     "drift_records_needed": 110, "start_grace": 10, "trend_min": 6, "listed": 10}


@pytest.mark.parametrize("name,body", of("activation_drift"), ids=[n for n, _ in of("activation_drift")])
def test_a_drift_case_records_what_the_rule_says_today(name, body):
    today = next((i for i in ev.detect_activation_anomalies(body["records"]) if i["kind"] == "activation_drift"), None)
    assert (today and today["step"]) == body["today"]["step"]


@pytest.mark.parametrize("name,body", of("moe_imbalance"), ids=[n for n, _ in of("moe_imbalance")])
def test_an_imbalance_case_records_what_the_rule_says_today(name, body):
    records = body["records"]
    sizes = ev._router_sizes(records)
    samples = ev._moe_samples(records, sizes, ev._router_top_k(records, list(sizes), None), ev.OVERLOAD_THRESHOLD)
    today = ev.first_incident({"samples": samples})
    assert (today and today["step"]) == body["today"]["step"]


def test_the_cases_show_each_thing_the_page_says_a_rule_gives_up_or_keeps():
    drift = {name: body["expected"]["incident"] for name, body in of("activation_drift")}
    assert drift["drift_growth_from_scratch_is_not_a_shift"] is None and drift["drift_a_slow_drift_is_not_seen"] is None
    assert drift["drift_a_step_change_is_named"]["step"] == 305 and drift["drift_a_fall_is_named_too"]["modules"][0]["ratio"] < 1
    assert not CASES["drift_not_judged_before_110_records"]["expected"]["enough_history"]
    start = {name: body["expected"]["incident"] for name, body in of("moe_imbalance")}
    assert start["start_two_samples_then_even"] is None and start["start_nine_samples_then_even"] is None
    assert start["start_ten_samples_is_reported"] == {**start["start_ten_samples_is_reported"], "step": 0, "since_start": True}
    assert start["not_at_step_0_is_not_a_start"]["step"] == 5 and "since_start" not in start["not_at_step_0_is_not_a_start"]
    assert start["one_router_starts_another_fails"]["router"] == "b"
    assert start["start_and_the_capture_ends"] is None
    assert CASES["start_and_the_capture_ends"]["expected"]["starting_state"] == [{"router": "r", "samples": 5}]
    assert CASES["start_two_samples_then_even"]["expected"]["starting_state"] == []
    assert CASES["score_no_dispersion"]["expected"] == {"score": None, "score_note": "no_dispersion"}
    assert CASES["score_nine_earlier_values"]["expected"] == {"score": None, "score_note": "short_history"}
    assert CASES["gradient_trend_a_plateau"]["expected"]["gradient_trend"]["growth"] == 1.0
