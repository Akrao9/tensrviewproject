"""Lane AD7: the Python detectors held to `DETECTOR_RULES_CONTRACT.md` and its cases.

Every case of `fixtures/rules/cases.json` that `test_detector_rules_contract.py` does not hold
(the drift and imbalance cases are held there) is held here: the score's arithmetic and the words a
reader shows for it, the score over records, the four measurements, the learning rate on
`update_spike`, `parameters_not_judged` for one rank and for the run, and the readers' side of the
wire keys. Wherever a case gives records it goes through `build_evidence(records,
detectors="python")`. The tests after the cases hold what the page says and no case shows: the
sentences, the starting state on a longer capture, the measurements changing nothing judged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorquill import evidence as ev

FOLDER = Path(__file__).parent / "fixtures" / "rules"
CASES = json.loads((FOLDER / "cases.json").read_text())["cases"]
ZERO_MAD = "no score: the earlier values had zero median absolute deviation"


def of(part):
    return [(name, body) for name, body in CASES.items() if body["part"] == part]


def ids(part):
    return [name for name, _ in of(part)]


def plain(value):
    """As a JSON reader holds it: the cases are JSON, so a rank key is a string."""
    return json.loads(json.dumps(value))


def build(records):
    return ev.build_evidence(records, detectors="python")


def incident(evidence, kind):
    found = next((dict(i) for i in evidence["incidents"] if i["kind"] == kind), None)
    if found is not None:
        found.pop("class", None)
    return plain(found)


# --- part 1: the score ------------------------------------------------------------------------


@pytest.mark.parametrize("name,body", of("score"), ids=ids("score"))
def test_a_score_case_gives_the_arithmetic_and_the_words(name, body):
    expected = body["expected"]
    scored = ev.robust_score(body["value"], body["window"])
    assert {**scored, "display": ev.score_words(scored)} == expected


def test_a_reader_shows_the_three_displays_in_the_authors_words():
    """The number to one decimal place; for a short history; and for a zero MAD, whose words must
    stay true of the author's example, six equal values and four different ones."""
    assert ev.score_words(ev.robust_score(25.0, CASES["score_above"]["window"])) == "score 49.6"
    assert ev.score_words(ev.robust_score(25.0, [10.0 + 0.1 * i for i in range(9)])) == "no score: fewer than 10 earlier values"
    six_and_four = [10.0] * 6 + [11.0, 12.0, 13.0, 14.0]
    assert len(set(six_and_four)) == 5  # the values vary
    assert ev.score_words(ev.robust_score(25.0, six_and_four)) == ZERO_MAD
    # A null score is never shown as 0, and an entry without the key (older evidence) shows nothing.
    assert ev.score_words({"score": None, "score_note": "no_dispersion"}) == ZERO_MAD
    assert ev.score_words({"ratio": 3.0}) is None


@pytest.mark.parametrize("name,body", of("score_records"), ids=ids("score_records"))
def test_a_score_over_records_is_beside_each_ratio(name, body):
    expected = body["expected"]["incident"]
    assert incident(build(body["records"]), expected["kind"]) == expected


def test_the_summary_prints_the_score_after_each_ratio():
    total = build(CASES["score_on_grad_spike_by_the_total"]["records"])
    text = ev.incident_summary(total)
    assert "10× the median of recent steps (score 593.6), first in w (16× its median, score 981.8)" in text
    one = build(CASES["score_on_grad_spike_by_one_parameter"]["records"])
    text = ev.incident_summary(one)
    assert "tiny at 8× its own median (score 466.7)" in text
    assert "was 1.0× its median (score -1.3)" in text
    spike = build(CASES["score_on_activation_spike"]["records"])
    assert "abs_max 10× its recent median (score 620.5)" in ev.incident_summary(spike)
    drift = build(CASES["drift_a_step_change_is_named"]["records"])
    assert ("Activation drift from step 305 in blk: std grew to 4.9× its level over the 50 records before its last 10"
            " (score 263.1)") in ev.incident_summary(drift)


def _probes(values):
    return [{"step": 10 * i, "probe": {"attention": {"attn": {"entropy_share": [0.6, 0.6], "max_logit": [v, v]}}}}
            for i, v in enumerate(values)]


def test_an_attention_ratio_has_a_score_that_is_always_short_history():
    """At most five earlier probes make an attention baseline, so the score is always null."""
    evidence = build(_probes([2.0, 2.1, 2.0, 2.05, 1.95, 6.0, 6.5]))
    growth = incident(evidence, "attention_logit_growth")
    module = growth["modules"][0]
    assert (module["score"], module["score_note"]) == (None, "short_history") and "ratio" in module
    assert "(no score: fewer than 10 earlier values)" in ev.incident_summary(evidence)
    collapse = [{"step": 10 * i, "probe": {"attention": {"attn": {"entropy_share": [s, s], "max_logit": [2.0, 2.0]}}}}
                for i, s in enumerate([0.6, 0.62, 0.6, 0.2, 0.2])]
    module = incident(build(collapse), "attention_collapse")["modules"][0]
    assert (module["score"], module["score_note"]) == (None, "short_history")
    # `attention_uniform` records a rise, not a ratio: no score.
    uniform = [{"step": 10 * i, "probe": {"attention": {"attn": {"entropy_share": [s, s], "max_logit": [2.0, 2.0]}}}}
               for i, s in enumerate([0.6, 0.6, 0.99, 0.99])]
    assert "score" not in incident(build(uniform), "attention_uniform")["modules"][0]


def test_no_ratio_no_score():
    """The one-parameter path against a total whose median is 0 has no total ratio: no score key
    beside it, while the parameter's ratio has its score."""
    records = [{"step": i, "grad_norm": 0.0, "grad_norms": {"w": 1.0 + 0.01 * (i % 3) if i < 10 else 50.0}} for i in range(13)]
    found = incident(build(records), "grad_spike")
    assert found["trigger"] == "parameter" and found["ratio"] is None and "score" not in found
    assert "score" in found["parameters"][0]


# --- part 2: activation_drift's history, beside dead_activations -------------------------------


def test_drift_needs_110_records_and_dead_activations_keeps_60():
    assert ev.RECORDS_NEEDED["activation_drift"] == 110 and ev.RECORDS_NEEDED["dead_activations"] == 60
    records = [{"step": i, "layers": {"blk": {"std": 1.0, "abs_max": 4.0, "zero_frac": 0.0 if i < 50 else 0.5}}}
               for i in range(60)]
    evidence = build(records)
    assert incident(evidence, "dead_activations")["step"] == 59  # judged against the first 50, as before
    assert evidence["kind_checks"]["dead_activations"]["enough_history"] is True
    assert evidence["kind_checks"]["activation_drift"]["enough_history"] is False


# --- part 3: the sentences and the starting state ------------------------------------------------


def test_a_router_collapsed_since_its_start_is_said_to_have_begun_and_become_reportable():
    body = CASES["start_ten_samples_is_reported"]
    evidence = build(body["records"])
    text = ev.incident_summary(evidence)
    assert "Collapsed since step 0, reportable at step 9  ·  Layer 01" in text
    assert "First sampled bad step 0" not in text
    # The all-ranks view's group line says the same.
    combined = ev.combine_rank_evidence({0: evidence, 1: build([{**r, "rank": 1} for r in body["records"]])})
    line = ev._group_line(combined["incident_groups"][0])
    assert "MoE imbalance collapsed since step 0, reportable at step 9, layer 01" in line
    # A router reported as before is worded as before.
    later = build(CASES["start_then_a_later_overload"]["records"])
    assert "First sampled bad step 20  ·  Layer 01" in ev.incident_summary(later)


def test_first_incident_leaves_the_samples_as_they_were():
    evidence = build(CASES["start_ten_samples_is_reported"]["records"])
    before = json.dumps(evidence["samples"], sort_keys=True)
    hit = ev.first_incident(evidence)
    assert hit["since_start"] is True and hit["reportable_step"] == 9
    assert json.dumps(evidence["samples"], sort_keys=True) == before
    assert not any("since_start" in sample for sample in evidence["samples"])


def test_a_router_still_in_its_starting_state_on_a_capture_long_enough_for_the_kind():
    """Router `a` is flagged on its five samples from step 0 and then absent; router `b` is even on
    all twelve. The kind has its 11 records, but `a` has not been judged: enough history is false."""
    loaded, even = [102] + [22] * 7, [32] * 8
    records = [{"step": i, "moe": {**({"a": loaded} if i < 5 else {}), "b": even}, "moe_top_k": {"a": 1, "b": 1}}
               for i in range(12)]
    evidence = build(records)
    check = evidence["kind_checks"]["moe_imbalance"]
    assert check["records"] == 12 and check["records_needed"] == 11
    assert check["starting_state"] == [{"router": "a", "samples": 5}] and check["enough_history"] is False
    assert incident(evidence, "moe_imbalance") is None
    # The summary says why the kind is short, not "12 of 11 records".
    text = ev.incident_summary(evidence)
    assert "moe_imbalance: router a flagged since step 0 on 5 of 10 samples, starved_experts 12 of 50 records" in text
    assert "12 of 11" not in text
    # Without router `a` the same capture is checked with enough history, and names nothing.
    healthy = build([{"step": i, "moe": {"b": even}, "moe_top_k": {"b": 1}} for i in range(12)])
    assert "starting_state" not in healthy["kind_checks"]["moe_imbalance"]
    assert healthy["kind_checks"]["moe_imbalance"]["enough_history"] is True


def test_a_compacted_file_still_says_what_the_incident_says():
    """`incident_summary` reads the incident listed: a compacted file keeps a window of samples
    around it, from which the incident could not be worked out again."""
    body = CASES["start_ten_samples_is_reported"]
    evidence = build(body["records"])
    kept = {**evidence, "samples": [s for s in evidence["samples"] if s["step"] in (0, 1)]}
    assert "Collapsed since step 0, reportable at step 9" in ev.incident_summary(kept)


# --- part 4 and part 5 -------------------------------------------------------------------------


@pytest.mark.parametrize("name,body", of("evidence"), ids=ids("evidence"))
def test_an_evidence_case_gives_what_the_page_expects(name, body):
    expected = body["expected"]
    if "records_by_rank" in body:
        per_rank = {int(rank): build(records) for rank, records in body["records_by_rank"].items()}
        combined = plain(ev.combine_rank_evidence(per_rank))
        entry = combined["kind_checks"]["stalled_weights"]
        assert {key: entry.get(key) for key in expected["kind_checks"]["stalled_weights"]} == expected["kind_checks"]["stalled_weights"]
        return
    evidence = build(body["records"])
    if "incident" in expected:
        found = incident(evidence, expected["incident"]["kind"])
        found.pop("scalars", None)
        assert found == expected["incident"]
    else:
        entry = evidence["kind_checks"]["stalled_weights"]
        assert {key: entry.get(key) for key in expected["kind_checks"]["stalled_weights"]} == expected["kind_checks"]["stalled_weights"]


def test_the_all_ranks_view_names_no_rank_list_when_the_ranks_agree():
    body = CASES["the_all_ranks_view_names_them_by_rank_when_ranks_differ"]
    same = {0: build(body["records_by_rank"]["0"]), 1: build([{**r, "rank": 1} for r in body["records_by_rank"]["0"]])}
    entry = ev.combine_rank_evidence(same)["kind_checks"]["stalled_weights"]
    assert entry["parameters_not_judged"] == ["b", "c"] and "parameters_not_judged_by_rank" not in entry
    # A rank that read every gradient is not in the map, and the lists still differ.
    clean = [{k: v for k, v in r.items() if k != "grads_not_read"} for r in body["records_by_rank"]["1"]]
    entry = ev.combine_rank_evidence({0: same[0], 1: build(clean)})["kind_checks"]["stalled_weights"]
    assert entry["parameters_not_judged"] == ["b", "c"] and entry["parameters_not_judged_by_rank"] == {"0": ["b", "c"]}
    # Nothing unread anywhere: no key.
    entry = ev.combine_rank_evidence({1: build(clean)})["kind_checks"]["stalled_weights"]
    assert "parameters_not_judged" not in entry


# --- part 6: four measurements -----------------------------------------------------------------


@pytest.mark.parametrize("name,body", of("measurements"), ids=ids("measurements"))
def test_a_measurement_case_gives_its_block_or_none(name, body):
    (key, expected), = body["expected"].items()
    evidence = build(body["records"])
    assert plain(evidence.get(key)) == expected
    # Only its own block, if any: these records hold nothing the other three read, but for a
    # measurement case that shares its records' kind.
    assert (key in evidence) == (expected is not None)


@pytest.mark.parametrize("name", ["gradient_trend_a_steady_rise", "attention_trends_a_slow_rise",
                                  "update_trends_one_parameter_collapses", "weight_jumps_one_tensor_jumps"])
def test_a_measurement_changes_nothing_judged_and_is_worded_nowhere(name, monkeypatch):
    body = CASES[name]
    with_blocks = build(body["records"])
    monkeypatch.setattr(ev, "measurement_blocks", lambda records: {})
    without = build(body["records"])
    for key in ("incidents", "kind_checks", "run_state", "not_checked"):
        assert with_blocks.get(key) == without.get(key), key
    assert ev.incident_summary(with_blocks) == ev.incident_summary(without)
    assert any(key in with_blocks for key in ev.MEASUREMENT_BLOCKS)


# --- part 7: the readers' side of the wire keys ------------------------------------------------


@pytest.mark.parametrize("name,body", [(n, b) for n, b in of("wire") if "records" in b],
                         ids=[n for n, b in of("wire") if "records" in b])
def test_a_wire_case_reads_as_the_page_says(name, body):
    expected = body["expected"]
    evidence = build(body["records"])
    if "not_checked" in expected:
        assert [item for item in evidence.get("not_checked", ()) if item["check"] == "layers"] == expected["not_checked"]
        for kind, holds in expected["kind_checks"].items():
            entry = evidence["kind_checks"][kind]
            assert {key: entry.get(key) for key in holds} == holds, kind
            assert ("modules_not_judged" in entry) == ("modules_not_judged" in holds), kind
    else:
        assert evidence.get("moe_source") == expected["moe_source"]
        assert ev.moe_source_words(evidence) == expected["words"]


def test_layers_coverage_reads_in_the_summary_and_none_is_not_clean():
    none = build(CASES["layers_not_read_no_module"]["records"])
    assert none["run_state"]["state"] == "degraded"
    assert "Not checked (layers): layer not read: blk0: its output is not a tensor" in ev.incident_summary(none)
    partial = build(CASES["layers_not_read_some_modules"]["records"])
    assert {partial["kind_checks"][k]["value"] for k in ev.LAYERS_NOT_READ_KINDS} == {"checked"}


def test_the_summary_says_what_the_counts_are_only_where_the_capture_said():
    flagged = [{"step": i, "moe": {"r": [102] + [22] * 7}, "moe_top_k": {"r": 1}} for i in range(5, 8)]
    dispatched = [{**flagged[0], "moe_source": {"r": {"form": "map", "point": "dispatched"}}}, *flagged[1:]]
    assert "The counts of router r are the dispatched assignments." in ev.incident_summary(build(dispatched))
    unknown = [{**flagged[0], "moe_source": {"r": {"form": "indices", "point": "unknown"}}}, *flagged[1:]]
    text = ev.incident_summary(build(unknown))
    assert "the assignments the model gave; whether a capacity limit was applied before this point is not known" in text
    assert "dispatched" not in text
    assert "The counts of router" not in ev.incident_summary(build(flagged))


def _status(capture_stopped):
    return {"schema_version": 1, "summary": "degraded", "written_at_ms": 1, "world_size": 2,
            "ranks": [{"rank": 0, "state": "silent", "last_step": 7, "attempt": 1, "capture_stopped": capture_stopped},
                      {"rank": 1, "state": "active", "last_step": 9, "attempt": 1}]}


@pytest.mark.parametrize("name,body", [(n, b) for n, b in of("wire") if "heartbeat" in b],
                         ids=[n for n, b in of("wire") if "heartbeat" in b])
def test_the_step_capture_stopped_at_stays_apart_from_none_to_a_reader(name, body):
    """The status as the server writes it from the beat (the cases' `expected`): step 0, null and
    no key read three ways, in the run state's reason and in the silence incident's sentence."""
    said = body["expected"]["status"]["capture_stopped"]
    status = _status(said)
    found = next(i for i in ev.detect_watchdog_incidents(status) if i["kind"] == "rank_silent")
    assert found["capture_stopped"] == said
    text = ev._describe_stream(found) if hasattr(ev, "_describe_stream") else ev._capture_stopped_text(found)
    state = ev.run_state(incidents=[], steps_judged=True, status=status, status_age=0.0, not_checked=[])
    reason = next(r for r in state["reasons"] if r["code"] == "capture_stopped")
    if "step" not in said:
        assert "committing" not in text and "steps" not in reason
        assert "rank 0: sink_error)" in reason["detail"]
    elif said["step"] is None:
        assert "had stopped before committing any step (sink_error)" in text
        assert reason["steps"] == {"0": None} and "rank 0: sink_error, before committing any step" in reason["detail"]
    else:
        assert "had stopped after committing step 0 (sink_error)" in text
        assert reason["steps"] == {"0": 0} and "rank 0: sink_error, after committing step 0" in reason["detail"]
