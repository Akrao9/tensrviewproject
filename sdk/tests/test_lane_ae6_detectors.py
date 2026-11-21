"""Lane AE6: the Python detectors' side of sections 1, 2 and 4 of `WAVE_AE_CONTRACT.md`.

1. Separately clipped groups: an `update_spike` pair whose totals would be split by parameter is
   not compared when either record carries `clip_calls` above 1, and `not_checked` says so once.
2. A stalled parameter and a resume: every streak of `stalled_weights` starts again at an
   attempt boundary.
3. Several optimizers through one scaler: a step the scaler skipped for some optimizers and
   applied for others (`step_skipped_part`) is counted apart by `loss_scale_trouble` and by
   `skipped_updates`, and read as a skipped step by the kinds judged parameter by parameter.

Every case of those sections in `fixtures/wave_ae/cases.json` is held here, `new` ones included,
through the detector the contract test reads and through `build_evidence`, which is what both
engines are held to at the merge. The watcher's kept state (`_Incremental`) is held to a fresh
judgment at each look across each of the three conditions.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tensorquill import evidence as ev

from test_wave_ae_contract import answer, matches

DOCUMENT = json.loads((Path(__file__).parent / "fixtures" / "wave_ae" / "cases.json").read_text())
CASES = {name: body for name, body in DOCUMENT["cases"].items() if body["section"] in (1, 2, 4)}
NAMES = list(CASES)
REASON = ("clipped by more than one clip_grad_norm_ call: the pre-clipping total cannot be split by parameter,"
          " so {n} comparison(s) of update_spike were not made")


def _records(name: str) -> list[dict]:
    return copy.deepcopy(CASES[name]["given"]["records"])


def _evidence(name: str, records: list[dict] | None = None) -> dict:
    body = CASES[name]
    return ev.build_evidence(records if records is not None else _records(name),
                             update_window=body["given"].get("window", ev.UPDATE_WINDOW))


def _text(evidence) -> str:
    """An evidence file's text, as `write_evidence` writes it."""
    return json.dumps(ev.json_safe(evidence), separators=(",", ":"), allow_nan=False) + "\n"


def _incident(evidence: dict, kind: str) -> dict | None:
    return next((i for i in evidence["incidents"] if i["kind"] == kind), None)


def test_the_cases_of_the_three_sections_are_all_here():
    assert len(CASES) == 29
    assert sum(body["status"] == "new" for body in CASES.values()) == 10


# --- every case, through the detector and through `build_evidence` ---------------------------


@pytest.mark.parametrize("name", NAMES)
def test_each_case_gives_its_expected_answer(name):
    body = CASES[name]
    found = answer(body)
    assert matches(found, body["expected"]), f"{found} is not {body['expected']}"


@pytest.mark.parametrize("name", NAMES)
def test_each_case_gives_its_expected_answer_through_build_evidence(name):
    body, expected = CASES[name], CASES[name]["expected"]
    evidence = _evidence(name)
    if "update_spike_step" in expected:
        hit = _incident(evidence, "update_spike")
        assert (None if hit is None else hit["step"]) == expected["update_spike_step"]
        entries = [e for e in evidence.get("not_checked", ()) if e.get("check") == "clip_groups"]
        want = expected["clip_groups_entry"]
        if want is None:
            assert entries == []
        else:
            assert entries == [{**want, "reason": REASON.format(n=want["comparisons"])}]
            assert list(entries[0]) == ["check", "kinds", "comparisons", "step", "reason"]
            assert evidence["kind_checks"]["update_spike"]["value"] == "checked"
            assert {"state": "degraded", "code": "not_checked", "checks": ["clip_groups"],
                    "detail": "not checked: clip_groups"} in evidence["run_state"]["reasons"]
    if "stalled_weights_step" in expected:
        hit = _incident(evidence, "stalled_weights")
        assert (None if hit is None else hit["step"]) == expected["stalled_weights_step"]
    if "loss_scale_trouble_step" in expected:
        hit = _incident(evidence, "loss_scale_trouble")
        assert (None if hit is None else hit["step"]) == expected["loss_scale_trouble_step"]
        if hit is not None:
            assert hit["reason"] == expected["reason"]
            if "consecutive_skips" in expected:
                assert hit["consecutive_skips"] == expected["consecutive_skips"]
            counters = expected.get("counters_at_end")
            if counters is not None:
                assert hit["skipped_steps"] == counters["skipped_steps"]
                assert hit["recorded_steps"] == counters["recorded_steps"]
            if "incident_partial_steps" in expected:
                assert hit["partial_steps"] == expected["incident_partial_steps"]
            partial = (expected.get("counters_at_end") or {}).get("partial_steps")
            if partial is not None:
                assert hit.get("partial_steps") == (partial or None)
    if "skipped_updates" in expected:
        assert evidence.get("skipped_updates") == expected["skipped_updates"]


# --- 1. separately clipped groups ------------------------------------------------------------


def test_the_reason_is_the_page_s_words():
    evidence = _evidence("a_split_pair_with_two_clip_calls_is_not_compared")
    entry = next(e for e in evidence["not_checked"] if e["check"] == "clip_groups")
    assert entry["reason"] == (
        "clipped by more than one clip_grad_norm_ call: the pre-clipping total cannot be split by parameter,"
        " so 1 comparison(s) of update_spike were not made")
    assert evidence["run_state"]["reasons"][0]["checks"] == ["clip_groups"]


def test_clip_calls_on_pairs_that_need_no_split_change_nothing_in_the_evidence():
    """A run whose records carry `clip_calls` and whose pairs never need a split has no entry and
    no other difference: its evidence is the text of the same run without the key."""
    for name in ("a_pair_naming_the_same_parameters_is_compared_whatever_the_clip_calls",
                 "a_pair_that_cannot_be_split_is_compared_whatever_the_clip_calls"):
        with_key = _records(name)
        without = [{k: v for k, v in r.items() if k != "clip_calls"} for r in with_key]
        assert any("clip_calls" in r for r in with_key)
        assert _text(_evidence(name, with_key)) == _text(_evidence(name, without)), name


@pytest.mark.parametrize("value", [1, "2", 2.5, True, 2.0, None, -3])
def test_a_clip_calls_that_is_not_a_whole_number_above_1_is_read_as_no_key(value):
    name = "a_split_pair_with_two_clip_calls_is_not_compared"
    records = _records(name)
    records[5]["clip_calls"] = value
    without = _records(name)
    del without[5]["clip_calls"]
    assert _text(_evidence(name, records)) == _text(_evidence(name, without))


def test_a_pair_that_is_not_compared_joins_no_baseline():
    """The change of the pair not compared stays out of `recent`: with it in, the baseline of the
    next judged pair would move. Step 6 is flagged against the four changes before step 5's."""
    evidence = _evidence("after_a_pair_that_is_not_compared_the_next_is")
    hit = _incident(evidence, "update_spike")
    assert hit["step"] == 6 and hit["previous"] == 2.0
    changes = [0.01, 0.01, 0.01]  # steps 2, 3 and 4 (the window is 3); step 1's has left it
    assert hit["baseline"] == pytest.approx(sum(changes) / 3)


def test_a_pair_not_compared_inside_the_first_window_is_counted():
    """Not stated by a case: a split pair with two clip calls among the first `window` changes,
    which are not judged, is still a pair not compared, and is counted."""
    records = _records("a_split_pair_with_two_clip_calls_is_not_compared")[:3]
    records[2] = {**records[2], "grad_norms": {"a": 0.6, "b": 0.8, "c": 1.0},
                  "update_rms": {**records[2]["update_rms"], "c": 0.1}, "clip_calls": 2}
    found, not_compared = ev._update_spikes_judged(records, window=3)
    assert found is None and not_compared == {"comparisons": 1, "step": 2}


def test_a_split_whose_factor_is_1_is_still_a_split():
    """A parameter whose norm is 0 makes the split's factor exactly 1; the pair still needs a
    split, read from the parameters the records name, so it is not compared."""
    records = _records("a_split_pair_with_two_clip_calls_is_not_compared")
    records[5]["grad_norms"] = {"a": 0.6, "b": 0.8, "c": 0.0}
    found, not_compared = ev._update_spikes_judged(records, window=3)
    assert found is None and not_compared == {"comparisons": 1, "step": 5}


def test_detect_update_spikes_keeps_its_return():
    records = _records("a_split_pair_with_one_clip_call_is_compared")
    assert ev.detect_update_spikes(records, window=3) == ev._update_spikes_judged(records, window=3)[0]
    assert ev._update_spikes_judged(records, window=3)[1] is None


def test_the_entry_leaves_update_spike_checked_in_the_all_ranks_view():
    evidence = _evidence("two_pairs_that_are_not_compared")
    combined = ev.combine_rank_evidence({0: evidence})
    assert combined["kind_checks"]["update_spike"]["value"] == "checked"
    assert {"state": "degraded", "code": "not_checked", "checks": ["clip_groups"],
            "detail": "not checked: clip_groups"} in combined["run_state"]["reasons"]


# --- 2. a stall and a resume -----------------------------------------------------------------


def test_only_the_streak_starts_again_at_a_resume():
    """`trained_before` counts a parameter seen changing before the resume: what was seen
    changing is kept across it."""
    records = _records("a_stall_wholly_inside_the_resumed_attempt")
    found = ev.detect_stalled_weights(sorted(ev.resolve_attempts(records), key=lambda r: int(r["step"])))
    assert found["step"] == 300 and found["onsets"][0]["trained_before"] == 1


def test_a_resume_to_the_same_attempt_number_holds_the_streak():
    """No boundary without a change of attempt: the same records, all of attempt 1, raise at 200."""
    records = _records("a_stall_does_not_carry_across_a_resume")
    for record in records:
        record["attempt"] = 1
    assert ev.detect_stalled_weights(records)["step"] == 200


def test_a_record_without_attempt_is_of_attempt_0():
    records = _records("a_stall_does_not_carry_across_a_resume")
    for record in records:
        if record["attempt"] == 1:
            del record["attempt"]
    assert ev.detect_stalled_weights(records)["step"] == 300


def test_the_docstring_no_longer_says_a_streak_holds_across_any_row_without_the_list():
    doc = ev.detect_stalled_weights.__doc__
    assert "A\n    row without the list is no weight check, and every streak holds." not in doc
    assert "A resume starts every streak again" in doc


# --- 4. several optimizers through one scaler --------------------------------------------------


def test_the_counters_carry_partial_steps_only_when_above_zero():
    _, with_part = ev._loss_scale_judged(_records("five_full_skips_one_partial_step_five_full_skips"))
    assert with_part == {"recorded_steps": 11, "skipped_steps": 10, "streak": 5, "partial_steps": 1}
    _, without = ev._loss_scale_judged(_records("eleven_full_skips_with_one_optimizer"))
    assert without == {"recorded_steps": 11, "skipped_steps": 11, "streak": 11}
    hit = ev.detect_loss_scale_trouble(_records("eleven_full_skips_with_one_optimizer"))
    assert "partial_steps" not in hit


def test_the_incident_carries_partial_steps_after_the_counts():
    hit = ev.detect_loss_scale_trouble(_records("a_partial_step_then_ten_full_skips"))
    assert list(hit)[-3:] == ["skipped_steps", "recorded_steps", "partial_steps"] and hit["partial_steps"] == 1


def test_the_summary_says_a_partial_step_in_the_page_s_words():
    line = ev.incident_summary(_evidence("a_partial_step_then_ten_full_skips")).splitlines()[0]
    assert line.endswith(
        "10 of 11 steps skipped in all, and 1 step(s) on which the scaler skipped some optimizers and applied others")
    plain = ev.incident_summary(_evidence("eleven_full_skips_with_one_optimizer")).splitlines()[0]
    assert plain.endswith("11 of 11 steps skipped in all")


def test_a_low_scale_on_a_partial_step_carries_the_partial_count():
    hit = ev.detect_loss_scale_trouble(_records("a_low_loss_scale_on_a_partial_step"))
    assert hit["reason"] == "scale" and hit["partial_steps"] == 1 and hit["skipped_steps"] == 3


def test_a_key_that_states_no_mixed_outcome_changes_nothing_in_the_evidence():
    """A `step_skipped_part` that is not a mixed outcome, or that sits on a skip the scaler did not
    confirm or on an applied step, gives the evidence the same record gives without it."""
    for name, step in (("a_part_key_that_says_every_optimizer_skipped_is_a_full_skip", 5),
                       ("a_part_key_that_says_no_optimizer_skipped_is_a_full_skip", 5),
                       ("a_part_key_that_is_not_two_counts_is_a_full_skip", 5),
                       ("a_part_key_on_a_skip_only_the_caller_reported_says_nothing", 5),
                       ("a_part_key_on_an_applied_step_says_nothing", 5)):
        records = _records(name)
        without = _records(name)
        del without[step]["step_skipped_part"]
        assert _text(_evidence(name, records)) == _text(_evidence(name, without)), name


@pytest.mark.parametrize("part", [{"skipped": True, "of": 2}, {"skipped": 1.0, "of": 2},
                                  {"skipped": 1, "of": "2"}, {"skipped": 3, "of": 2}, [1, 2]])
def test_a_part_key_that_is_not_two_whole_counts_is_a_full_skip(part):
    record = {"step": 0, "step_skipped": True, "step_skipped_by": "scaler", "step_skipped_part": part}
    assert not ev.partial_step(record)


def test_a_partial_step_is_read_by_the_parameter_kinds_as_a_skipped_one():
    """`grad_spike`, `update_spike` and `stalled_weights` read a partial step as they read a full
    skip: a gradient and an update that would be flagged on an applied step are not judged there,
    and the incidents of the per-parameter kinds are the same as with a full skip."""
    rows = []
    for step in range(30):
        rows.append({"step": step, "grad_norm": 1.0 + 0.01 * (step % 3), "grad_norms": {"w": 1.0 + 0.01 * (step % 3)},
                     "update_rms": {"w": 0.01 + 0.0001 * (step % 3)}, "step_skipped": False,
                     "step_skipped_by": "scaler"})
    rows[25] = {**rows[25], "grad_norm": 500.0, "grad_norms": {"w": 500.0}, "update_rms": {"w": 5.0}}
    applied = ev.build_evidence(copy.deepcopy(rows), update_window=10)
    assert {"grad_spike", "update_spike"} <= {i["kind"] for i in applied["incidents"]}
    partial = copy.deepcopy(rows)
    partial[25].update(step_skipped=True, step_skipped_part={"skipped": 1, "of": 2})
    full = copy.deepcopy(rows)
    full[25].update(step_skipped=True)
    per_parameter = ("grad_spike", "update_spike", "stalled_weights")
    got = {i["kind"]: i for i in ev.build_evidence(partial, update_window=10)["incidents"] if i["kind"] in per_parameter}
    want = {i["kind"]: i for i in ev.build_evidence(full, update_window=10)["incidents"] if i["kind"] in per_parameter}
    assert got == want and "grad_spike" not in got and "update_spike" not in got
    assert ev.input_counts(partial) == ev.input_counts(full)


def test_a_partial_step_the_caller_also_reported_keeps_its_non_finite_gradients():
    """Not stated by a case: a partial step reported by both, with a gradient that is not
    finite, is counted under `partial` and still in `non_finite_gradients`, which is about a skip
    the caller reported."""
    records = [{"step": 0, "step_skipped": True, "step_skipped_by": "both",
                "step_skipped_part": {"skipped": 1, "of": 2}, "grad_norms": {"w": float("inf")}}]
    assert ev.skipped_updates(records) == {
        "caller": 0, "scaler": 0, "both": 0, "partial": 1,
        "non_finite_gradients": {"steps": 1, "first_step": 0, "gradients": ["w"]}}


# --- the watcher's kept state ------------------------------------------------------------------


def _looks(records: list[dict], ends: list[int], **options) -> tuple[list[tuple[str, object]], dict]:
    """Each look of one `_Incremental` over growing prefixes, held to a fresh `build_evidence`;
    the paths it took, and the last look's evidence."""
    memo = ev._Incremental()
    paths = []
    for end in ends:
        prefix = copy.deepcopy(records[:end]) if not paths else prefix + copy.deepcopy(records[len(prefix):end])
        fresh = _text(ev.build_evidence(copy.deepcopy(prefix), **options))
        given = ev._build_evidence(prefix, memo, **options)
        assert _text(given) == fresh, (end, memo.path)
        paths.append(memo.path)
    return paths, given


@pytest.mark.parametrize("name,ends,kind,step", [
    ("a_stall_does_not_carry_across_a_resume", [3, 4, 5, 6, 8], "stalled_weights", 300),  # an attempt boundary
    ("a_stall_raised_before_the_resume_stands", [4, 5, 7], "stalled_weights", 150),
    ("five_full_skips_one_partial_step_five_full_skips", [5, 6, 7, 11], "loss_scale_trouble", None),  # a partial step
    ("a_partial_step_then_ten_full_skips", [1, 2, 10, 11], "loss_scale_trouble", 10),
    ("after_a_pair_that_is_not_compared_the_next_is", [5, 6, 7], "update_spike", 6),  # a pair not compared
    ("two_pairs_that_are_not_compared", [5, 6, 7], "update_spike", None),
])
def test_each_look_of_the_watcher_is_a_fresh_judgment(name, ends, kind, step):
    window = CASES[name]["given"].get("window", ev.UPDATE_WINDOW)
    paths, last = _looks(_records(name), ends, update_window=window)
    assert paths[0] == ("full", "first look")
    assert all(path[0] == "incremental" for path in paths[1:]), paths
    hit = _incident(last, kind)
    assert (None if hit is None else hit["step"]) == step
    if kind == "update_spike":
        assert [e["comparisons"] for e in last["not_checked"] if e["check"] == "clip_groups"] == [
            CASES[name]["expected"]["clip_groups_entry"]["comparisons"]]
