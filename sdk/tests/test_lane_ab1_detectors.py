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
