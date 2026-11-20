"""D1, which rank a replica drift is attributed to: the cases of contract section 4a, each checked
against a second reading of the rule.

`REPLICA_OBSERVATION_CONTRACT.md`, section 4a, is the rule. `fixtures/replica/attribution_cases.json`
holds its cases, written by hand in `make_attribution_cases.py`: for each, the coverage of every
step, the attribution entries and the number of incidents, as three separate expectations. The
same file says what attribution adds to each of the 39 earlier cases, so no engine has to derive
it. Nothing reads or writes the `attribution` key yet: this file and the page are what the two
engines build against.

The two earlier case files are not changed by D1, and `test_replica_contract.py` still holds for
them: attribution is one more key of the result, present only when it has an entry.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from test_replica_contract import ALL as EARLIER
from test_replica_contract import _number, reference

CASES = Path(__file__).parent / "fixtures" / "replica" / "attribution_cases.json"
DOCUMENT = json.loads(CASES.read_text())
TOLERANCE = DOCUMENT["tolerance"]
NEW = list(DOCUMENT["cases"].items())
SAID_OF_EARLIER = DOCUMENT["earlier_cases"]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    half = len(ordered) // 2
    return ordered[half] if len(ordered) % 2 else (ordered[half - 1] + ordered[half]) / 2


def _by_fingerprints(rows: dict[int, dict], columns: int, tolerance: float) -> dict:
    """Section 4a for one parameter that drifted: `rows` are the compared ranks' entries."""
    ranks = sorted(rows)
    if len(ranks) == 2:
        return {"verdict": "inconclusive", "why": "two_ranks"}
    values = {r: [_number(v) for v in rows[r]["fp"][:columns]] for r in ranks}
    finite = [r for r in ranks if all(math.isfinite(v) for v in values[r])]
    off = {r for r in ranks if r not in finite}
    if finite:
        scale = max(sum(values[r][0] for r in finite) / len(finite), 1e-12)
        for column in range(columns):
            middle = _median([values[r][column] for r in finite])
            off |= {r for r in finite if abs(values[r][column] - middle) / scale > tolerance}
    if not off:
        return {"verdict": "inconclusive", "why": "no_outlier"}
    if 2 * (len(ranks) - len(off)) <= len(ranks):
        return {"verdict": "inconclusive", "why": "no_majority"}
    return {"verdict": "attributed", "outliers": sorted(off)}


def _by_signatures(rows: dict[int, dict]) -> dict:
    ranks = sorted(rows)
    if len(ranks) == 2:
        return {"verdict": "inconclusive", "why": "two_ranks"}
    held: dict[str, list[int]] = {}
    for r in ranks:
        held.setdefault(rows[r]["sig"], []).append(r)
    most = max(held.values(), key=len)
    if 2 * len(most) <= len(ranks):
        return {"verdict": "inconclusive", "why": "no_majority"}
    return {"verdict": "attributed", "outliers": sorted(r for r in ranks if r not in most)}


def attribution(records: list[dict], tolerance: float) -> list[dict]:
    """Section 4a read in code: the attribution entries of one run's records. It starts from what
    sections 2 to 5 already say of each step (`reference`), and reads the rows that were compared."""
    said = reference(records, tolerance)
    keeps: dict[tuple, int] = {}  # (launch, rank) -> the attempt that keeps the launch (contract 2)
    kept: dict[tuple, dict] = {}  # (launch, step, rank) -> the observation that was compared
    for record in records:
        observation = record.get("replica")
        if observation is None or observation.get("launch") is None:
            continue
        launch, rank = observation["launch"], record["rank"]
        if keeps.setdefault((launch, rank), record["attempt"]) == record["attempt"]:
            kept[(launch, record["step"], rank)] = observation
    out = []
    for entry in said["steps"]:
        if entry["coverage"] != "complete":
            continue
        cells = {r: kept[(entry["launch"], entry["step"], r)] for r in entry["group"]}
        columns = 2 if entry.get("projection_compared") is False else 3
        if any(len(cell["params"][name]["fp"]) < 3 for cell in cells.values() for name in entry.get("drift_parameters", ())):
            columns = 2
        where = {"step": entry["step"], "launch": entry["launch"], "group": entry["group"]}
        for name in sorted(entry.get("drift_parameters", ())):
            rows = {r: cells[r]["params"][name] for r in cells}
            out.append({**where, "parameter": name, "by": "fingerprints", **_by_fingerprints(rows, columns, tolerance)})
        for name in sorted(entry.get("signature_parameters", ())):
            rows = {r: cells[r]["params"][name] for r in cells}
            out.append({**where, "parameter": name, "by": "signatures", **_by_signatures(rows)})
    return sorted(out, key=lambda e: (e["step"], e["launch"], e["group"], e["parameter"], e["by"]))


def _coverage(said: dict) -> list[dict]:
    """The coverage of each step entry, apart from what was compared."""
    return [{k: e[k] for k in ("step", "launch", "group", "coverage", "missing", "why", "ranks") if k in e} for e in said["steps"]]


@pytest.mark.parametrize("case,body", NEW, ids=[c for c, _ in NEW])
def test_each_case_has_the_coverage_attribution_and_incident_count_the_contract_gives(case, body):
    said = reference(body["records"], TOLERANCE)
    expected = body["expected"]
    assert _coverage(said) == expected["coverage"]
    assert attribution(body["records"], TOLERANCE) == expected["attribution"]
    assert len(said["replica_drift"]) == expected["incidents"]


def test_the_file_holds_the_cases_the_page_names():
    assert len(NEW) == 26
    page = (Path(__file__).parents[2] / "REPLICA_OBSERVATION_CONTRACT.md").read_text()
    for case, _ in NEW:
        assert f"`{case}`" in page, f"{case} is not named in section 4a"


@pytest.mark.parametrize("file,case,body,tolerance", EARLIER, ids=[c for _, c, _, _ in EARLIER])
def test_what_attribution_adds_to_each_earlier_case_is_said(file, case, body, tolerance):
    """The 39 cases of sections 2 to 6 keep their results. Attribution adds a key to those that
    drift or whose signatures differ, written by hand here, and nothing to the others."""
    assert attribution(body["records"], tolerance) == SAID_OF_EARLIER.get(case, [])


def test_every_earlier_case_named_here_exists():
    assert set(SAID_OF_EARLIER) <= {case for _, case, _, _ in EARLIER}
    assert all(SAID_OF_EARLIER.values()), "a case with no entry is left out of the list"


def test_attribution_never_changes_what_raises():
    """The incident policy of v1 is section 5's and stays: attribution describes a drift, and a
    difference in signatures alone is described and raises nothing."""
    for case, body in NEW:
        said = reference(body["records"], TOLERANCE)
        by = {e["by"] for e in body["expected"]["attribution"]}
        if by == {"signatures"}:
            assert said["replica_drift"] == [], case
        drifted = {(e["step"], e["launch"]) for e in body["expected"]["attribution"] if e["by"] == "fingerprints"}
        raised = {(i["step"], i["launch"]) for i in said["replica_drift"] if i["source"] == "replica"}
        assert drifted <= raised, case


def test_no_rank_is_named_without_a_strict_majority_or_without_every_rank():
    for case, body in NEW:
        complete = {(e["step"], e["launch"]): e["group"] for e in body["expected"]["coverage"] if e["coverage"] == "complete"}
        for entry in body["expected"]["attribution"]:
            group = complete[(entry["step"], entry["launch"])]  # KeyError: attribution on incomplete coverage
            assert entry["group"] == group and len(group) > 1
            if entry["verdict"] == "attributed":
                assert entry["outliers"] and 2 * (len(group) - len(entry["outliers"])) > len(group), case
                assert set(entry["outliers"]) <= set(group)
            else:
                assert "outliers" not in entry and entry["why"] in ("two_ranks", "no_majority", "no_outlier")


# --- Readers that are wrong, each caught by a named case ------------------------------------------


def _wrong(reader, case: str) -> bool:
    body = DOCUMENT["cases"][case]
    return reader(body["records"], TOLERANCE) != body["expected"]["attribution"]


def test_a_reader_that_names_one_outlier_for_the_whole_step_is_caught():
    def one_outlier_a_step(records, tolerance):
        out = attribution(records, tolerance)
        first = next((e for e in out if e["verdict"] == "attributed"), None)
        return [{**e, "outliers": first["outliers"]} if e["verdict"] == "attributed" else e for e in out]

    assert _wrong(one_outlier_a_step, "two_parameters_with_different_outliers")


def test_a_reader_that_takes_the_largest_group_for_a_majority_is_caught():
    def plurality(records, tolerance):
        out = []
        for e in attribution(records, tolerance):
            if e.get("why") == "no_majority":
                e = {k: v for k, v in e.items() if k != "why"} | {"verdict": "attributed", "outliers": [e["group"][-1]]}
            out.append(e)
        return out

    assert _wrong(plurality, "four_ranks_two_one_one") and _wrong(plurality, "four_ranks_two_to_two")


def test_a_reader_that_compares_with_rank_0_is_caught():
    """Megatron compares each replica's hash with replica 0's. When rank 0 is the one that drifted,
    that names every other rank."""
    def against_rank_0(records, tolerance):
        out = []
        for e in attribution(records, tolerance):
            if e["verdict"] == "attributed":
                e = {**e, "outliers": [r for r in e["group"] if r != 0]}
            out.append(e)
        return out

    assert _wrong(against_rank_0, "three_ranks_rank_0_is_the_outlier")


def test_a_reader_that_names_a_rank_of_two_is_caught():
    def names_the_higher(records, tolerance):
        return [{k: v for k, v in e.items() if k != "why"} | {"verdict": "attributed", "outliers": [e["group"][-1]]}
                if e.get("why") == "two_ranks" else e for e in attribution(records, tolerance)]

    assert _wrong(names_the_higher, "two_ranks_differ")


def test_a_reader_that_attributes_among_the_ranks_that_reported_is_caught():
    """Two of three reported and differ; the third is missing. A reader that takes the ranks that
    reported for the group calls it two ranks, where nothing can be said until the third is there."""
    def among_the_reporters(records, tolerance):
        rows: dict[tuple, dict[int, dict]] = {}
        for record in records:
            rows.setdefault((record["replica"]["launch"], record["step"]), {})[record["rank"]] = record["replica"]
        out = []
        for (launch, step), cells in rows.items():
            for name in ("b", "w"):
                held = {r: cell["params"][name] for r, cell in cells.items()}
                if len({json.dumps(row["fp"]) for row in held.values()}) > 1:
                    out.append({"step": step, "launch": launch, "group": sorted(cells), "parameter": name,
                                "by": "fingerprints", **_by_fingerprints(held, 3, tolerance)})
        return out

    assert _wrong(among_the_reporters, "a_peer_is_missing_and_the_reporters_differ")


def test_a_reader_that_lets_a_ranks_later_row_vote_is_caught():
    """Rank 1's later attempt is set aside (contract 2). A reader that takes each rank's last row
    counts it, finds two ranks holding the drifted value, and names the healthy rank."""
    def the_last_row_of_each_rank(records, tolerance):
        return attribution(list(reversed(records)), tolerance)

    assert _wrong(the_last_row_of_each_rank, "a_set_aside_later_attempt_does_not_vote")
    named = [e["outliers"] for e in the_last_row_of_each_rank(
        DOCUMENT["cases"]["a_set_aside_later_attempt_does_not_vote"]["records"], TOLERANCE) if e["verdict"] == "attributed"]
    assert named and all(outliers == [0] for outliers in named)
