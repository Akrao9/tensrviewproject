"""Lane Z2 of the SDK-22 wave: the Python reader of replica observations (`evidence.py`).

`REPLICA_OBSERVATION_CONTRACT.md` says what each rank records under `replica` and how a reader
compares the ranks' records; `fixtures/replica/` holds its 39 cases, each with the object both
engines must give. `evidence.compare_replicas` is the Python engine. It is written from the
contract and does not use the reference reading in `test_replica_contract.py`, which this file
does not import: three readings (the reference, this one, the server's) that agree on every case.

Fail before (on `main` at 633c9da, run with the main checkout's `tensorquill`): every test here
fails there, the case tests with `ImportError` on `compare_replicas`, which did not exist; the
notes on the wiring tests say what each read on `main` where the import alone is not the point.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from tensorquill import evidence as ev

CASES = Path(__file__).parent / "fixtures" / "replica"
FILES = ("no_incident_cases.json", "incident_cases.json")


def _load() -> list[tuple[str, str, dict, float]]:
    out = []
    for name in FILES:
        document = json.loads((CASES / name).read_text())
        out += [(name, case, body, document["tolerance"]) for case, body in document["cases"].items()]
    return out


ALL = _load()
IDS = [f"{f.split('_')[0]}:{c}" for f, c, _, _ in ALL]


def _case(name: str) -> tuple[dict, float]:
    [(body, tolerance)] = [(b, t) for _, c, b, t in ALL if c == name]
    return json.loads(json.dumps(body)), tolerance


# --- 1. The reader against the contract's cases -------------------------------------------------


def test_there_are_39_cases():
    assert len(ALL) == 39 and len({c for _, c, _, _ in ALL}) == 39


@pytest.mark.parametrize("file,case,body,tolerance", ALL, ids=IDS)
def test_each_case_gives_the_object_the_fixtures_spell(file, case, body, tolerance):
    assert ev.compare_replicas(body["records"], tolerance=tolerance) == body["expected"]


def _by_rank(records: list[dict], reverse: bool = False) -> list[dict]:
    """The records as several rank files give them, file by file: each rank's records in their
    own order, the ranks one after another."""
    ranks = sorted({r["rank"] for r in records}, reverse=reverse)
    return [r for rank in ranks for r in records if r["rank"] == rank]


@pytest.mark.parametrize("file,case,body,tolerance", ALL, ids=IDS)
def test_each_case_reads_the_same_fed_file_by_file(file, case, body, tolerance):
    """Only each rank's own order is contract 1's order; how the rank files follow each other
    does not change the object."""
    for order in (_by_rank(body["records"]), _by_rank(body["records"], reverse=True)):
        assert ev.compare_replicas(order, tolerance=tolerance) == body["expected"], case
    # Any interleaving that keeps each rank's own order.
    rng = random.Random(case)
    queues = {rank: [r for r in body["records"] if r["rank"] == rank] for rank in {r["rank"] for r in body["records"]}}
    mixed = []
    while any(queues.values()):
        mixed.append(queues[rng.choice([rank for rank, left in queues.items() if left])].pop(0))
    assert ev.compare_replicas(mixed, tolerance=tolerance) == body["expected"], case


def test_the_stale_launch_case_fed_file_by_file():
    """The case whose result depends on the order of records: rank 1 has two attempts under one
    launch. File by file (rank 0's file, then rank 1's), as the interleaved arrival order."""
    body, tolerance = _case("launch_read_from_an_earlier_start")
    old0, old1, new0, new1 = body["records"]
    for order in ([old0, new0, old1, new1], [old1, new1, old0, new0], [new0, old0, old1, new1]):
        assert ev.compare_replicas(order, tolerance=tolerance) == body["expected"]
    # A rank's own order is what decides: rank 1's lines the other way round make the new
    # attempt the first, and then the old one is the row set aside.
    swapped = ev.compare_replicas([old0, new0, new1, old1], tolerance=tolerance)
    assert swapped != body["expected"]
    assert [s.get("why") for s in swapped["steps"] if s["coverage"] == "not_established"] == ["attempt"]


def test_a_rank_in_several_files_takes_its_attempt_order():
    """A rank whose attempts are in two local files: `attempt_order` (`attempt_order_across` of
    its files, contract 1) orders them, whatever order the files are given in."""
    body, tolerance = _case("launch_read_from_an_earlier_start")
    old0, old1, new0, new1 = body["records"]
    for at, record in enumerate((old0, old1, new0, new1)):
        record["ts"] = 1000.0 + at
    # Rank 1's later file given first: without its order the new attempt would keep the launch.
    given = [old0, new0, new1, old1]
    assert ev.compare_replicas(given, tolerance=tolerance) != body["expected"]
    order = ev.attempt_order_across([[new1], [old1]])
    assert order[("r1", 1)] == [old1["attempt"], new1["attempt"]]
    assert ev.compare_replicas(given, tolerance=tolerance, attempt_order=order) == body["expected"]


def test_records_of_two_runs_are_refused():
    body, tolerance = _case("healthy_peers_with_distinct_attempts")
    body["records"][1]["run_id"] = "another"
    with pytest.raises(ValueError, match="more than one run"):
        ev.compare_replicas(body["records"], tolerance=tolerance)


def test_old_records_keep_todays_verdict_in_the_legacy_list():
    """Contract 6, by the rule the reader uses today (`detect_replica_drift`), on every record of
    every case that holds a gathered spread and no observation."""
    seen = 0
    for _, case, body, tolerance in ALL:
        for record, entry in zip([r for r in body["records"] if "replica_spread" in r and "replica" not in r],
                                 ev.compare_replicas(body["records"], tolerance=tolerance).get("legacy", ())):
            today = ev.detect_replica_drift([record], tolerance=tolerance)
            assert entry["drift"] is (today is not None), case
            assert entry.get("parameters", []) == sorted((today or {}).get("parameters", ())), case
            seen += 1
    assert seen == 2
