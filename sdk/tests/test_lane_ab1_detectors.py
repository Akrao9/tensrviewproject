"""Lane AB1 of wave AB: the Python detectors (`evidence.py`, `data_checks.py`).

D1 first: `compare_replicas` gives `attribution` (`REPLICA_OBSERVATION_CONTRACT.md`, section 4a),
checked against all 27 cases of `fixtures/replica/attribution_cases.json` (coverage, attribution
and the number of incidents apart) and against what `earlier_cases` says of the 39 earlier cases.
The rest of the file is `STEP_AND_EVIDENCE_CONTRACT.md`'s rules as this lane builds them.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest

from tensorquill import evidence as ev

REPLICA = Path(__file__).parent / "fixtures" / "replica"
D1 = json.loads((REPLICA / "attribution_cases.json").read_text())
D1_CASES = list(D1["cases"].items())
SAID_OF_EARLIER = D1["earlier_cases"]


def _earlier() -> list[tuple[str, dict, float]]:
    out = []
    for name in ("no_incident_cases.json", "incident_cases.json"):
        document = json.loads((REPLICA / name).read_text())
        out += [(case, body, document["tolerance"]) for case, body in document["cases"].items()]
    return out


EARLIER = _earlier()


def _coverage(result: dict) -> list[dict]:
    """The coverage of each step entry, apart from what was compared (as the contract's test reads it)."""
    return [{k: e[k] for k in ("step", "launch", "group", "coverage", "missing", "why", "ranks") if k in e}
            for e in result["steps"]]


def _interleaved(records: list[dict], seed: str) -> list[dict]:
    """The records with the ranks interleaved at random, each rank's own order kept."""
    rng = random.Random(seed)
    queues = {rank: [r for r in records if r["rank"] == rank] for rank in {r["rank"] for r in records}}
    out = []
    while any(queues.values()):
        out.append(queues[rng.choice([rank for rank, left in queues.items() if left])].pop(0))
    return out


# --- D1: the reader against the cases --------------------------------------------------------------


def test_the_file_holds_27_cases_and_14_earlier_ones():
    assert len(D1_CASES) == 27 and len(SAID_OF_EARLIER) == 14 and len(EARLIER) == 39


@pytest.mark.parametrize("case,body", D1_CASES, ids=[c for c, _ in D1_CASES])
def test_each_d1_case_has_its_coverage(case, body):
    assert _coverage(ev.compare_replicas(body["records"], tolerance=D1["tolerance"])) == body["expected"]["coverage"]


@pytest.mark.parametrize("case,body", D1_CASES, ids=[c for c, _ in D1_CASES])
def test_each_d1_case_has_its_attribution(case, body):
    result = ev.compare_replicas(body["records"], tolerance=D1["tolerance"])
    wanted = body["expected"]["attribution"]
    if wanted:
        assert result["attribution"] == wanted
    else:  # a list present only when it has an entry: a result with no difference is spelled as before
        assert "attribution" not in result


@pytest.mark.parametrize("case,body", D1_CASES, ids=[c for c, _ in D1_CASES])
def test_each_d1_case_raises_as_many_incidents_as_section_5_says(case, body):
    result = ev.compare_replicas(body["records"], tolerance=D1["tolerance"])
    assert len(result["replica_drift"]) == body["expected"]["incidents"]


@pytest.mark.parametrize("case,body", D1_CASES, ids=[c for c, _ in D1_CASES])
def test_each_d1_case_reads_the_same_with_the_ranks_interleaved(case, body):
    given = ev.compare_replicas(body["records"], tolerance=D1["tolerance"])
    assert ev.compare_replicas(_interleaved(body["records"], case), tolerance=D1["tolerance"]) == given


@pytest.mark.parametrize("case,body,tolerance", EARLIER, ids=[c for c, _, _ in EARLIER])
def test_each_earlier_case_keeps_its_object_and_gains_what_earlier_cases_says(case, body, tolerance):
    """Section 4a, "For the engines that build this": everything but `attribution` is compared with
    the earlier file, and `attribution` with `earlier_cases` (an absent key and an empty list are
    the same there)."""
    result = ev.compare_replicas(body["records"], tolerance=tolerance)
    added = result.pop("attribution", [])
    assert result == body["expected"]
    assert added == SAID_OF_EARLIER.get(case, [])


def test_the_two_wordings_part_where_the_page_says():
    """Three ranks within 0.9 of the tolerance of the median and one at 4: fingerprints name rank 3,
    signatures (four different bits) name none."""
    body = D1["cases"]["three_of_four_within_tolerance_of_the_median"]
    found = {e["by"]: e for e in ev.compare_replicas(body["records"], tolerance=D1["tolerance"])["attribution"]}
    assert found["fingerprints"]["outliers"] == [3]
    assert found["signatures"]["verdict"] == "inconclusive"


# --- D1: where a named rank is shown ---------------------------------------------------------------


def _one_rank(records: list[dict], rank: int, tolerance: float) -> dict:
    own = [r for r in records if r["rank"] == rank]
    return {**ev.build_evidence(own, drift_tolerance=tolerance), "last_step": max(r["step"] for r in own)}


def _local_run(records: list[dict], tolerance: float) -> dict:
    per_rank = {rank: _one_rank(records, rank, tolerance) for rank in sorted({r["rank"] for r in records})}
    return ev.combine_rank_evidence(per_rank, replica_records=records, drift_tolerance=tolerance)


def _drift_incidents(evidence: dict, key: str = "run_incidents") -> list[dict]:
    return [i for i in evidence.get(key) or () if i.get("kind") == "replica_drift" and i.get("source") == "replica"]


def _status(ranks: list[int], replica: dict) -> dict:
    return {
        "schema_version": 1, "run_id": "r1", "summary": "active", "server_running": True, "refresh_seconds": 30.0,
        "written_at_ms": int(time.time() * 1000), "world_size": len(ranks), "missing_ranks": [],
        "ranks": [{"rank": r, "attempt": 1, "last_step": 99, "phase": "train", "state": "active", "severity": None,
                   "stream": "open"} for r in ranks],
        "pauses": {}, "recommended_silent_seconds": None, "incidents": [], "ranks_compared": ranks,
        "replica": replica,
    }


def test_the_all_ranks_view_carries_attribution_and_the_incident_names_ranks_by_fingerprints():
    body = D1["cases"]["two_parameters_with_different_outliers"]
    combined = _local_run(body["records"], D1["tolerance"])
    assert combined["replica"]["attribution"] == body["expected"]["attribution"]
    [found] = _drift_incidents(combined)
    assert found["attribution"] == [e for e in body["expected"]["attribution"] if e["by"] == "fingerprints"]
    assert {e["parameter"]: e["outliers"] for e in found["attribution"]} == {"b": [0], "w": [2]}
    text = ev._describe(found)
    assert "b: rank 0 apart from the majority" in text and "w: rank 2 apart from the majority" in text


def test_the_incident_names_no_rank_where_the_fingerprints_cannot():
    """`not_finite_on_two_of_three`: signatures name rank 0, fingerprints none. The incident is the
    fingerprints' reading only; the signatures' stays in the result."""
    body = D1["cases"]["not_finite_on_two_of_three"]
    combined = _local_run(body["records"], D1["tolerance"])
    [found] = _drift_incidents(combined)
    assert [(e["by"], e["verdict"], e["why"]) for e in found["attribution"]] == [("fingerprints", "inconclusive", "no_majority")]
    assert "outliers" not in found["attribution"][0]
    assert any(e["by"] == "signatures" and e["outliers"] == [0] for e in combined["replica"]["attribution"])
    assert "no rank named (no strict majority)" in ev._describe(found)


@pytest.mark.parametrize("case", ["signature_only_two_to_one", "signature_only_between_two_ranks",
                                  "signed_zero_on_one_of_three", "below_tolerance_on_one_of_three"])
def test_a_difference_shown_only_by_signatures_is_attributed_and_raises_nothing(case):
    body = D1["cases"][case]
    combined = _local_run(body["records"], D1["tolerance"])
    assert combined["replica"]["attribution"] == body["expected"]["attribution"]
    assert _drift_incidents(combined) == []


def test_a_servers_attribution_is_passed_through_unchanged():
    body = D1["cases"]["three_ranks_two_to_one"]
    replica = {"v": 1, "tolerance": D1["tolerance"],
               **ev.compare_replicas(body["records"], tolerance=D1["tolerance"])}
    assert replica["attribution"]
    combined = ev.combine_rank_evidence({}, status=_status([0, 1, 2], replica), status_expected=True)
    assert combined["replica"] == replica
    [found] = _drift_incidents(combined)
    assert found["attribution"] == [e for e in replica["attribution"] if e["by"] == "fingerprints"]
    # What the server wrote is what is shown, even where this reader would say otherwise.
    edited = json.loads(json.dumps(replica))
    edited["attribution"][0]["outliers"] = [1]
    combined = ev.combine_rank_evidence({}, status=_status([0, 1, 2], edited), status_expected=True)
    assert combined["replica"]["attribution"] == edited["attribution"]
    assert _drift_incidents(combined)[0]["attribution"][0]["outliers"] == [1]


def test_one_ranks_own_evidence_names_no_rank():
    """No rank's records can say which rank stands apart: one rank's evidence has no attribution."""
    body = D1["cases"]["three_ranks_two_to_one"]
    for rank in (0, 1, 2):
        own = ev.build_evidence([r for r in body["records"] if r["rank"] == rank], drift_tolerance=D1["tolerance"])
        assert "attribution" not in own["replica"]
        assert all("attribution" not in i for i in own["incidents"])


# --- Skipped steps (STEP_AND_EVIDENCE_CONTRACT.md, section 1) --------------------------------------

BEHAVIOUR = json.loads((Path(__file__).parent / "fixtures" / "behaviour" / "cases.json").read_text())["cases"]


def _rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        row = {k: (float(v) if v in ("nan", "inf", "-inf") else v) for k, v in row.items()}
        if isinstance(row.get("grad_norms"), dict):
            row["grad_norms"] = {k: (float(v) if v == "nan" else v) for k, v in row["grad_norms"].items()}
        out.append({"schema_version": 1, "run_id": "r1", "rank": 0, "attempt": 1, **row})
    return out


OVER_RECORDS = [(name, body) for name, body in BEHAVIOUR.items() if body["status"] == "new" and "records" in body["given"]]
DETECTORS = {"stalled_weights": "detect_stalled_weights", "loss_scale_trouble": "detect_loss_scale_trouble",
             "non_finite": "detect_non_finite", "grad_spike": "detect_grad_spikes"}


def test_every_case_over_records_is_built_and_none_keeps_today():
    assert len(OVER_RECORDS) == 18
    assert not [name for name, body in BEHAVIOUR.items() if "today" in body]


@pytest.mark.parametrize("name,body", OVER_RECORDS, ids=[n for n, _ in OVER_RECORDS])
def test_each_new_case_over_records_is_what_the_evidence_gives(name, body):
    """The evidence of the case's records, read in the page's words: which kinds are raised and
    what the first of them carries, each kind's value, the `not_checked` entries (each key given
    is matched), the run state, and the evidence's own keys."""
    rows = _rows(body["given"]["records"])
    evidence = ev.build_evidence(rows)
    expected = body["expected"]
    kinds = [i["kind"] for i in evidence["incidents"]]
    for kind in expected.get("incident_kinds_present", ()):
        assert kind in kinds, f"{kind} is not raised"
    for kind in expected.get("incident_kinds_absent", ()):
        assert kind not in kinds, f"{kind} is raised"
        assert getattr(ev, DETECTORS[kind])(rows) is None  # and the detector alone says the same
    if "incident_carries" in expected:
        kind = expected["incident_kinds_present"][0]
        found = next(i for i in evidence["incidents"] if i["kind"] == kind)
        assert {key: found.get(key) for key in expected["incident_carries"]} == expected["incident_carries"]
    for kind, value in expected.get("kind_values", {}).items():
        assert evidence["kind_checks"][kind]["value"] == value, kind
    for item in expected.get("not_checked", ()):
        assert any({key: entry.get(key) for key in item} == item for entry in evidence.get("not_checked", ())), item
    if "run_state" in expected:
        assert evidence["run_state"]["state"] == expected["run_state"]
    for key, value in expected.get("evidence_carries", {}).items():
        assert evidence.get(key) == value, key
    for key in expected.get("evidence_lacks", ()):
        assert key not in evidence, key


@pytest.mark.parametrize("record,by", [
    ({"step_skipped": False}, None), ({}, None), ({"step_skipped": True}, "scaler"),
    ({"step_skipped": True, "step_skipped_by": "caller"}, "caller"),
    ({"step_skipped": True, "step_skipped_by": "both"}, "both"),
    ({"step_skipped": True, "step_skipped_by": "scaler"}, "scaler"),
    ({"step_skipped": True, "step_skipped_by": "someone"}, "scaler"),
    ({"step_skipped": False, "step_skipped_by": "caller"}, None),
])
def test_who_reported_a_skip(record, by):
    assert ev.skipped_by(record) == by


def test_a_stall_is_held_across_a_skip_and_raised_at_the_next_applied_check():
    """Section 1, rule 1 (signed off by the author): two unchanged checks, a skipped one and a third unchanged check
    raise at the third. Before wave AB the stall was raised at the skipped step."""
    rows = BEHAVIOUR["a_stall_across_a_skipped_step"]["given"]["records"]
    assert ev.detect_stalled_weights(_rows(rows))["step"] == 3
    # A zero gradient on the skipped row still resets: its resets apply.
    reset = [dict(r) for r in rows]
    reset[2]["grad_norms"] = {"blk.weight": 0.0}
    assert ev.detect_stalled_weights(_rows(reset)) is None


def test_caller_skips_do_not_count_toward_loss_scale_trouble_but_their_scale_does():
    caller = ev.detect_loss_scale_trouble(_rows(BEHAVIOUR["ten_skips_only_the_caller_reported"]["given"]["records"]))
    assert caller is None
    inside = ev.detect_loss_scale_trouble(_rows(BEHAVIOUR["caller_skips_inside_a_run_of_scaler_skips"]["given"]["records"]))
    assert (inside["step"], inside["consecutive_skips"], inside["skipped_steps"], inside["recorded_steps"]) == (14, 10, 10, 12)
    low = ev.detect_loss_scale_trouble(_rows([{"step": 3, "step_skipped": True, "step_skipped_by": "caller", "grad_scale": 0.5}]))
    assert low["reason"] == "scale" and low["skipped_steps"] == 0 and low["recorded_steps"] == 0


def test_a_caller_only_skip_is_no_loss_scale_input():
    counts = ev.input_counts(_rows([{"step": s, "step_skipped": True, "step_skipped_by": "caller"} for s in range(3)]
                                   + [{"step": 3, "step_skipped": True, "step_skipped_by": "both"}, {"step": 4, "step_skipped": False}]))
    assert counts["loss_scale"] == 2


def test_skipped_updates_counts_by_who_and_keeps_a_callers_non_finite_gradient():
    rows = _rows([
        {"step": 0, "step_skipped": False, "grad_norm": 1.0},
        {"step": 1, "step_skipped": True, "step_skipped_by": "scaler", "grad_norm": "nan", "grad_norms": {"a": "nan"}},
        {"step": 2, "step_skipped": True, "step_skipped_by": "caller", "grad_norm": 1.0},
        {"step": 3, "step_skipped": True, "step_skipped_by": "caller", "grad_norm": "nan"},
        {"step": 4, "step_skipped": True, "step_skipped_by": "both", "grad_norm": 1.0, "grad_norms": {"b": "nan"},
         "clip_grad_norm": "inf"},
        {"step": 5, "step_skipped": True},  # no key in a capture with it: the scaler's
    ])
    assert ev.skipped_updates(rows) == {
        "caller": 2, "scaler": 2, "both": 1,
        "non_finite_gradients": {"steps": 2, "first_step": 3, "gradients": ["<total>"]},
    }
    evidence = ev.build_evidence(rows)
    assert evidence["skipped_updates"] == ev.skipped_updates(rows)
    assert all(i["kind"] != "non_finite" for i in evidence["incidents"])  # the key raises nothing


def test_a_capture_without_the_key_has_no_skipped_updates_and_reads_as_before():
    rows = _rows([{"step": s, "step_skipped": s >= 5, "grad_norm": 1.0} for s in range(15)])
    evidence = ev.build_evidence(rows)
    assert "skipped_updates" not in evidence
    assert [i["kind"] for i in evidence["incidents"]] == ["loss_scale_trouble"]


def test_the_all_ranks_view_keeps_each_ranks_skipped_updates():
    skipped = _rows([{"step": 0, "step_skipped": True, "step_skipped_by": "caller", "grad_norm": 1.0}])
    plain = [{**{k: v for k, v in r.items() if k != "step_skipped_by"}, "rank": 1, "step_skipped": False} for r in skipped]
    per_rank = {0: ev.build_evidence(skipped), 1: ev.build_evidence(plain)}
    combined = ev.combine_rank_evidence(per_rank)
    assert combined["skipped_updates_by_rank"] == [{"rank": 0, "skipped_updates": {"caller": 1, "scaler": 0, "both": 0}}]
    assert "skipped_updates_by_rank" not in ev.combine_rank_evidence({1: per_rank[1]})


# --- Unread gradients (section 2) ------------------------------------------------------------------


def _unread(rows: list[dict]) -> list[dict]:
    rows = [dict(r) for r in rows]
    rows[0]["grads_not_read"] = {"emb.weight": "a sparse gradient"}
    return _rows(rows)


def test_an_unread_gradient_cannot_hide_a_nan_observed_elsewhere():
    """The author's acceptance test, on a layer and on another parameter's gradient."""
    for nan in ({"layers": {"blk": {"mean": "nan", "std": 1.0, "abs_max": 1.0}}}, {"grad_norms": {"blk.weight": "nan"}}):
        rows = _unread([{"step": 0, "grad_norm": 1.0}, {"step": 1, "grad_norm": 1.0, **nan}])
        if "layers" in nan:
            rows[1]["layers"]["blk"]["mean"] = float("nan")
        else:
            rows[1]["grad_norms"]["blk.weight"] = float("nan")
        evidence = ev.build_evidence(rows)
        [found] = [i for i in evidence["incidents"] if i["kind"] == "non_finite"]
        assert found["step"] == 1
        assert evidence["kind_checks"]["non_finite"]["value"] == "checked"
        assert evidence["run_state"]["state"] == "unhealthy"


def test_an_unread_gradient_leaves_grad_spike_unavailable_and_names_it_for_stalled_weights():
    rows = _unread([{"step": s, "grad_norm": 1.0, "grad_norms": {"blk.weight": 1.0}} for s in range(4)])
    evidence = ev.build_evidence(rows)
    [entry] = [i for i in evidence["not_checked"] if i["check"] == "gradients"]
    assert entry["kinds"] == ["grad_spike"] and entry["parameters"] == ["emb.weight"]
    checks = evidence["kind_checks"]
    assert checks["grad_spike"]["value"] == "unavailable" and checks["grad_spike"]["check"] == "gradients"
    assert checks["non_finite"]["value"] == "checked" and "parameters_not_judged" not in checks["non_finite"]
    assert checks["stalled_weights"]["parameters_not_judged"] == ["emb.weight"]
    assert evidence["run_state"]["state"] == "degraded"


def test_a_spike_in_the_total_as_read_still_fires_beside_an_unread_gradient():
    body = BEHAVIOUR["a_spike_beside_an_unread_gradient"]
    evidence = ev.build_evidence(_rows(body["given"]["records"]))
    assert [i["step"] for i in evidence["incidents"] if i["kind"] == "grad_spike"] == [40]
    assert evidence["kind_checks"]["grad_spike"]["value"] == "unavailable"


# --- A rank with no peer (section 2) ---------------------------------------------------------------


def test_a_lone_rank_with_batch_hashes_is_compared_with_no_one():
    from tensorquill import data_checks as dc

    rows = [{"schema_version": 1, "run_id": "r1", "rank": 0, "attempt": 1, "step": s,
             "data": {"batches": s + 1, "batch_hash": f"sha256:{s:064x}"}} for s in range(3)]
    assert dc._ranks_compared(dc._hashed_steps(rows)) == []
    assert dc.detect_duplicate_batches(rows) is None


def test_one_rank_with_batch_hashes_reads_not_configured_with_the_words():
    checks = {0: ev.kind_checks({**dict.fromkeys(ev.INPUT_NEEDS, 0), "steps": 5, "data": 5}, []),
              1: ev.kind_checks({**dict.fromkeys(ev.INPUT_NEEDS, 0), "steps": 5}, [])}
    status = {"ranks_compared": []}
    for judged in ({0: checks[0]}, checks):
        others = ev.other_kind_checks(status=status, status_expected=True, not_checked=[], flight_recorder=False,
                                      traincheck=False)
        found = ev.run_kind_checks(judged, others, memory={}, status=status)["duplicate_batches"]
        assert found["value"] == "not_configured" and found["note"] == "one rank observed, zero peers compared"
        assert "not_compared_ranks" not in found


def test_a_group_of_one_says_the_same_words_for_replica_fingerprints():
    body = D1["cases"]["group_of_one"]
    combined = _local_run(body["records"], D1["tolerance"])
    check = combined["kind_checks"]["replica_drift"]
    assert check["value"] == "not_configured" and check["note"] == "one rank observed, zero peers compared"


# --- A rank silent after its capture had stopped (section 3) ---------------------------------------


def _silent_status(entry: dict) -> dict:
    others = {"rank": 0, "attempt": 1, "last_step": 300, "state": "active", "phase": "train"}
    return {"schema_version": 1, "run_id": "r1", "summary": "degraded", "world_size": 2, "missing_ranks": [],
            "ranks": [others, {"state": "silent", "phase": "train", "seconds_since_progress": 900.0, **entry}]}


def test_silent_after_capture_stopped_carries_the_two_keys_and_says_so():
    body = BEHAVIOUR["silent_after_capture_stopped"]
    [found] = ev.detect_watchdog_incidents(_silent_status(body["given"]["status_entry"]))
    assert found["kind"] == "rank_silent"
    assert {k: found[k] for k in ("capture_stopped", "capture_last_step")} == body["expected"]["incident_carries"]
    [rank] = [r for r in found["ranks"] if r["rank"] == 1]
    assert rank["capture_last_step"] == 120
    text = ev._describe(found)
    assert body["expected"]["words"] in text and "capture had stopped after step 120 (sink_error, 398 s ago)" in text


def test_silent_after_a_restart_that_stored_nothing_names_no_earlier_step():
    """The author's acceptance test: the incident uses only the current attempt's last recorded
    step. The earlier attempt reached step 100; the new one stored nothing, even where the status
    still shows the earlier attempt's step."""
    body = BEHAVIOUR["silent_after_a_restart_that_stored_nothing"]
    for entry in (body["given"]["status_entry"], {**body["given"]["status_entry"], "last_step": 100}):
        status = _silent_status(entry)
        status["ranks"][0]["last_step"] = 7
        [found] = ev.detect_watchdog_incidents(status)
        assert found["kind"] in body["expected"]["incident_kind_one_of"]
        assert {k: found[k] for k in ("capture_stopped", "capture_last_step")} == body["expected"]["incident_carries"]
        text = ev._describe(found)
        assert "capture had stopped before this attempt stored a step (sink_error, 42 s ago)" in text
        if entry.get("last_step") is None:
            assert body["expected"]["words_absent"] not in text


def test_a_job_stalled_with_capture_stopped_carries_the_keys_too():
    status = _silent_status({"rank": 1, "attempt": 1, "last_step": 120, "capture_stopped": {"reason": "sink_error", "seconds": 5.0}})
    status["summary"] = "stalled"
    status["ranks"][0]["state"] = "silent"
    [found] = ev.detect_watchdog_incidents(status)
    assert found["kind"] == "job_stalled" and found["capture_last_step"] == 120


def test_silent_with_capture_running_has_neither_key():
    body = BEHAVIOUR["silent_with_capture_running"]
    [found] = ev.detect_watchdog_incidents(_silent_status(body["given"]["status_entry"]))
    assert not set(body["expected"]["incident_lacks"]) & set(found)
    assert all("capture_last_step" not in r for r in found["ranks"])
    assert "capture had stopped" not in ev._describe(found)


# --- A checkpoint whose model state is not identified (section 4) ----------------------------------


def _resume(components: dict) -> list[dict]:
    rows = [{"schema_version": 1, "run_id": "r1", "rank": 0, "attempt": 1, "step": s, "lr": [0.1]} for s in range(5)]
    rows += [{"schema_version": 1, "run_id": "r1", "rank": 0, "attempt": 2, "step": 3, "lr": [0.1],
              "checkpoint": {"load": {"components": components}}}]
    return rows


def test_a_checkpoint_with_no_identified_model_state():
    body = BEHAVIOUR["a_checkpoint_with_no_identified_model_state"]
    full = {"optimizer": {"entries": 2}, "rng": {}, "data": {}}
    [found] = [i for i in ev.build_evidence(_resume(full))["incidents"] if i["kind"] == "resume_incomplete"]
    assert found["changes"][0]["findings"] == body["expected"]["findings"]
    text = ev._describe(found)
    assert body["expected"]["words"] in text and body["expected"]["words_absent"] not in text


def test_a_checkpoint_with_no_optimizer_state_keeps_missing():
    body = BEHAVIOUR["a_checkpoint_with_no_optimizer_state"]
    [found] = [i for i in ev.build_evidence(_resume({"model": {}, "rng": {}, "data": {}}))["incidents"]
               if i["kind"] == "resume_incomplete"]
    assert found["changes"][0]["findings"] == body["expected"]["findings"]
    assert "optimizer not in the checkpoint" in ev._describe(found)


# --- One rank's evidence from a server report carries its dropped rows ------------------------------


def _store(tmp_path: Path, rank_entry: dict) -> Path:
    run = tmp_path / "store" / "runs" / "r1"
    folder = run / "rank=0" / "attempt=1"
    folder.mkdir(parents=True)
    (folder / "part-00000000-00000009.parquet").write_bytes(b"")
    report = {"schema_version": 1, "source": "tensorquill-server", "run_id": "r1", "rank": 0, "attempt": 1,
              "last_step": 9, "steps": 10, "incidents": [], "checked": list(ev.SERVER_KINDS),
              "thresholds": {"spike_sustain": ev.SPIKE_SUSTAIN}, "kind_checks": ev.kind_checks(ev.input_counts([]), []),
              "skipped_updates": {"caller": 1, "scaler": 0, "both": 0}}
    (run / "rank=0" / "incidents.json").write_text(json.dumps(report))
    status = {"schema_version": 1, "run_id": "r1", "summary": "active", "server_running": True, "refresh_seconds": 30.0,
              "written_at_ms": int(time.time() * 1000), "world_size": 2, "missing_ranks": [], "incidents": [],
              "ranks": [{"rank": 0, "attempt": 1, "last_step": 9, "state": "active", **rank_entry},
                        {"rank": 1, "attempt": 1, "last_step": 9, "state": "active", "dropped_rows": 7}]}
    (run / "watchdog.json").write_text(json.dumps(status))
    return run


def test_one_ranks_evidence_from_a_server_report_carries_its_own_dropped_rows(tmp_path):
    run = _store(tmp_path, {"dropped_rows": 3, "dropped_row_keys": {"step": 3}})
    status = json.loads((run / "watchdog.json").read_text())
    evidence = ev._report_evidence(run / "rank=0", ev.read_server_report(run / "rank=0"), status)
    dropped = evidence["watchdog"]["dropped_rows"]
    assert dropped["rows"] == 3 and dropped["by_key"] == {"step": 3} and dropped["upper_bound"] is True
    assert dropped["note"]
    # The server's skipped updates are passed on unchanged.
    assert evidence["skipped_updates"] == {"caller": 1, "scaler": 0, "both": 0}
    clean = _store(tmp_path / "b", {})
    evidence = ev._report_evidence(clean / "rank=0", ev.read_server_report(clean / "rank=0"),
                                   json.loads((clean / "watchdog.json").read_text()))
    assert "dropped_rows" not in evidence["watchdog"]  # rank 1's rows are not this rank's


def test_the_loss_scale_streak_is_worded_as_scaler_confirmed_skips():
    found = ev.detect_loss_scale_trouble(_rows(BEHAVIOUR["caller_skips_inside_a_run_of_scaler_skips"]["given"]["records"]))
    text = ev._describe(found)
    assert "10 consecutive scaler-confirmed skips among observations judged by this rule" in text
    assert "10 of 12 steps skipped in all" in text


def test_a_caller_skip_the_scaler_applied_ends_the_streak_and_is_a_recorded_observation():
    """The author's amended rule 2: with `scaler_applied: true` the caller's skip ends the scaler's
    streak and counts in `recorded_steps` (that count awaits the author); with the outcome unknown
    it is passed over."""
    unknown = ev.detect_loss_scale_trouble(_rows(BEHAVIOUR["a_caller_skip_of_unknown_scaler_outcome_inside_scaler_skips"]["given"]["records"]))
    assert (unknown["step"], unknown["consecutive_skips"], unknown["skipped_steps"], unknown["recorded_steps"]) == (12, 10, 10, 12)
    applied = _rows(BEHAVIOUR["a_caller_skip_the_scaler_applied_inside_scaler_skips"]["given"]["records"])
    assert ev.detect_loss_scale_trouble(applied) is None
    after = ev.detect_loss_scale_trouble(_rows(BEHAVIOUR["ten_scaler_skips_after_a_caller_skip_the_scaler_applied"]["given"]["records"]))
    assert (after["step"], after["consecutive_skips"], after["skipped_steps"], after["recorded_steps"]) == (17, 10, 15, 18)
    # The low-scale floor is read on either kind of caller's skip.
    for extra in ({}, {"scaler_applied": True}):
        low = ev.detect_loss_scale_trouble(_rows([{"step": 3, "step_skipped": True, "step_skipped_by": "caller",
                                                   "grad_scale": 0.5, **extra}]))
        assert low["reason"] == "scale" and low["consecutive_skips"] == 0 and low["skipped_steps"] == 0
    # As input: a caller's skip the scaler applied is an observation; one of unknown outcome is not.
    counts = ev.input_counts(_rows([{"step": 0, "step_skipped": True, "step_skipped_by": "caller"},
                                    {"step": 1, "step_skipped": True, "step_skipped_by": "caller", "scaler_applied": True}]))
    assert counts["loss_scale"] == 1
