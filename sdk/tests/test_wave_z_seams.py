"""Where the lanes of the SDK-22 wave meet, on the tree that holds all of them.

Lane Z2 wrote the reader (`evidence.compare_replicas`, `combine_rank_evidence(replica_records=)`)
and lane Z5 owned `compare.py`, so neither could pass a local run's records to the reader: a run
kept as rank files showed no cross-rank result. Each contract case is written here as the files a
run leaves (one per rank) and opened as a user opens it.

Fail before (the trial tree without the line in `_open_local_all_ranks`): every case that holds
an observation has no `replica` key in its evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorquill import compare as cmp
from tensorquill import evidence as ev

FIXTURES = Path(__file__).parent / "fixtures" / "replica"


def _load() -> list[tuple[str, dict]]:
    out = []
    for name in ("no_incident_cases.json", "incident_cases.json"):
        document = json.loads((FIXTURES / name).read_text())
        assert document["tolerance"] == ev.DRIFT_TOLERANCE  # `open_run` uses the default tolerance
        out += list(document["cases"].items())
    return out


CASES = _load()


def _as_rank_files(folder: Path, records: list[dict]) -> None:
    for rank in sorted({r["rank"] for r in records}):
        lines = [json.dumps(r) for r in records if r["rank"] == rank]
        (folder / f"r1.rank{rank}.jsonl").write_text("\n".join(lines) + "\n")


def test_every_case_is_here():
    assert len(CASES) == 39


@pytest.mark.parametrize("case,body", CASES, ids=[c for c, _ in CASES])
def test_a_local_run_of_several_rank_files_gets_the_cross_rank_result(tmp_path, case, body):
    records = body["records"]
    if len({r["rank"] for r in records}) < 2:
        pytest.skip("one rank file: not the all-ranks view")
    _as_rank_files(tmp_path, records)
    evidence = cmp.open_run("r1", tmp_path)["evidence"]
    if not any("replica" in r for r in records):
        assert "replica" not in evidence  # old records: nothing new is said
        return
    assert evidence["replica"] == {"v": 1, "tolerance": ev.DRIFT_TOLERANCE, **body["expected"]}
    raised = [i for i in evidence["incidents"] if i.get("kind") == "replica_drift" and i.get("source") == "replica"]
    wanted = [i for i in body["expected"]["replica_drift"] if i["source"] == "replica"]
    # One incident for the kind, the first from observations (lane Z2's call), when the contract raises any.
    assert len(raised) == (1 if wanted else 0)
    if wanted:
        assert (raised[0]["step"], raised[0]["launch"], raised[0]["group"]) == \
            (wanted[0]["step"], wanted[0]["launch"], wanted[0]["group"])
