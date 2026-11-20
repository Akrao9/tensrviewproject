"""The replica observation contract's cases, checked against a small reference reading of its rules.

`REPLICA_OBSERVATION_CONTRACT.md` (step 0 of the SDK-22 wave, Nov 20, 2026) says what each rank
records and how a reader compares the ranks' records. The cases in `fixtures/replica/` are its
examples, each with the result both engines must give. Nothing in the SDK or the server reads
the `replica` key yet: this file is not a test of either. It keeps the page and the cases from
disagreeing, by reading the rules a second time in code (`reference`), and it shows that an
observation survives the two ways a record travels.

The Python reader and the server each get their own test over the same two files.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

CASES = Path(__file__).parent / "fixtures" / "replica"
FILES = ("no_incident_cases.json", "incident_cases.json")


def _load() -> list[tuple[str, str, dict, float]]:
    out = []
    for name in FILES:
        document = json.loads((CASES / name).read_text())
        out += [(name, case, body, document["tolerance"]) for case, body in document["cases"].items()]
    return out


ALL = _load()


def _number(value) -> float:
    return float(value) if isinstance(value, str) else value


def _legacy(record: dict, tolerance: float) -> dict:
    """Section 6: a gathered spread is judged on its own record, as before."""
    over = sorted(name for name, row in record["replica_spread"].items()
                  if any(not math.isfinite(v) or v > tolerance for v in row))
    return {"rank": record["rank"], "step": record["step"], "drift": bool(over), **({"parameters": over} if over else {})}


def _compare(observations: dict[int, dict], tolerance: float) -> dict:
    """Sections 3 to 5 for one launch, group and step with complete coverage."""
    ranks = sorted(observations)
    names = sorted({n for o in observations.values() for n in (*o["params"], *o.get("unread", {}))})
    out: dict = {}
    not_compared: dict[str, str] = {}
    same_device = len({o["device"] for o in observations.values()}) == 1
    projection = len({o["projection"] for o in observations.values()}) == 1 and observations[ranks[0]]["projection"] is not None
    if not projection and any(o["projection"] is not None for o in observations.values()):
        out["projection_compared"] = False
    same_method = len({o["signature"] for o in observations.values()}) == 1 and observations[ranks[0]]["signature"] is not None
    drift, differ = [], []
    for name in names:
        if any(name in o.get("unread", {}) for o in observations.values()):
            not_compared[name] = "unread"
            continue
        if any(name not in o["params"] for o in observations.values()):
            not_compared[name] = "not_recorded"
            continue
        rows = [observations[r]["params"][name] for r in ranks]
        if len({(tuple(row["shape"]), row["dtype"]) for row in rows}) > 1:
            not_compared[name] = "shape_dtype"
            continue
        if same_method and len({row["sig"] for row in rows}) > 1:
            differ.append(name)
        if not same_device:
            continue
        columns = 3 if projection else 2
        values = [[_number(v) for v in row["fp"][:columns]] for row in rows]
        if any(not math.isfinite(v) for row in values for v in row):
            drift.append(name)
            continue
        scale = max(sum(row[0] for row in values) / len(values), 1e-12)  # `spread_from_fingerprints`
        if any((max(col) - min(col)) / scale > tolerance for col in zip(*values)):
            drift.append(name)
    if not_compared:
        out["not_compared"] = not_compared
    if same_device:
        out["fingerprints"] = "drift" if drift else "agree"
        if drift:
            out["drift_parameters"] = drift
    else:
        out["fingerprints"], out["why"] = "not_compared", "device"
    if same_method:
        out["signatures"] = "differ" if differ else "agree"
        if differ:
            out["signature_parameters"] = differ
    return out


def reference(records: list[dict], tolerance: float) -> dict:
    """The contract read once more, in code: what a reader says of `records`."""
    steps, incidents = [], []
    legacy = [_legacy(r, tolerance) for r in records if "replica_spread" in r and "replica" not in r]
    gathered = {r["step"]: _legacy(r, tolerance) for r in records if "replica_spread" in r}
    by_step: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for record in records:
        if "replica" in record:
            # one rank's newest attempt of a launch: later in the file wins (contract 1's order)
            by_step[(record["replica"]["launch"], record["step"])][record["rank"]] = record["replica"]
    for (launch, step), seen in sorted(by_step.items(), key=lambda item: (item[0][1], str(item[0][0]))):
        groups = {tuple(o["group"]) for o in seen.values()}
        if launch is None:
            [group] = groups if len(groups) == 1 else [None]
            steps.append({"launch": None, "group": list(group) if group else None, "step": step,
                          "coverage": "not_established", "why": "launch"})
            continue
        for group in sorted(groups):
            members = {r: o for r, o in seen.items() if tuple(o["group"]) == group}
            if any(r in seen and tuple(seen[r]["group"]) != group for r in group):
                steps.append({"launch": launch, "group": None, "step": step, "coverage": "not_established", "why": "group"})
                break
            missing = [r for r in group if r not in members]
            entry = {"launch": launch, "group": list(group), "step": step}
            if missing:
                steps.append({**entry, "coverage": "insufficient", "missing": missing})
                continue
            verdict = _compare(members, tolerance)
            steps.append({**entry, "coverage": "complete", **verdict})
            if verdict.get("fingerprints") == "drift" and step not in gathered:
                incidents.append({"step": step, "parameters": verdict["drift_parameters"], "source": "replica"})
    for step, verdict in sorted(gathered.items()):
        if verdict["drift"]:
            incidents.append({"step": step, "parameters": verdict["parameters"], "source": "replica_spread"})
    out = {"steps": steps, "replica_drift": sorted(incidents, key=lambda i: i["step"])}
    if legacy:
        out["legacy"] = legacy
    return out


@pytest.mark.parametrize("file,case,body,tolerance", ALL, ids=[f"{f.split('_')[0]}:{c}" for f, c, _, _ in ALL])
def test_each_case_has_the_result_the_contract_gives(file, case, body, tolerance):
    assert body["says"]
    assert reference(body["records"], tolerance) == body["expected"]


def test_the_files_hold_the_cases_the_contract_names():
    names = {case for _, case, _, _ in ALL}
    assert len(names) == len(ALL), "a case name is used twice"
    assert {"healthy_peers_with_distinct_attempts", "delayed_rank_before_it_arrives", "delayed_rank_after_it_arrives",
            "rank_that_never_reports", "stale_rows_after_a_restart", "subgroups", "reordered_parameters",
            "signature_only_disagreement", "legacy_replica_spread_below_tolerance", "launch_not_known",
            "one_rank_without_a_launch", "explicit_spread_and_observations_on_one_step",
            "explicit_spread_clean_and_observations_drift"} <= names
    # every no-incident case raises nothing, and every incident case raises exactly one incident
    for file, case, body, _ in ALL:
        raised = body["expected"]["replica_drift"]
        assert (raised == []) if file == FILES[0] else (len(raised) == 1), case


def test_healthy_peers_do_not_share_an_attempt_number():
    [body] = [b for _, c, b, _ in ALL if c == "healthy_peers_with_distinct_attempts"]
    assert len({r["attempt"] for r in body["records"]}) == 3
    assert len({r["replica"]["launch"] for r in body["records"]}) == 1


def test_a_restart_never_pairs_a_new_row_with_an_old_one():
    [(body, tolerance)] = [(b, t) for _, c, b, t in ALL if c == "stale_rows_after_a_restart"]
    # Paired by step alone, rank 0's new row and rank 1's old one would read as drift on w.
    new, old = body["records"][2]["replica"], body["records"][1]["replica"]
    assert _compare({0: new, 1: old}, tolerance)["fingerprints"] == "drift"
    assert body["expected"]["replica_drift"] == []


def test_old_records_get_the_verdict_the_reader_gives_them_today():
    """Section 6 against the real rule: `detect_replica_drift` on each record that holds only a spread."""
    from tensorquill.evidence import detect_replica_drift

    seen = 0
    for _, case, body, tolerance in ALL:
        for wanted in body["expected"].get("legacy", ()):
            [record] = [r for r in body["records"] if r["rank"] == wanted["rank"] and r["step"] == wanted["step"]]
            found = detect_replica_drift([record], tolerance=tolerance)
            assert (found is not None) is wanted["drift"], case
            if found is not None:
                assert found["step"] == wanted["step"] and sorted(found["parameters"]) == wanted["parameters"], case
            seen += 1
    assert seen == 2


def test_no_launch_is_a_launchers_restart_count():
    """torchrun's count starts at 0 on every fresh launch, so two starts of one run ID would share it."""
    for _, case, body, _ in ALL:
        for record in body["records"]:
            launch = (record.get("replica") or {}).get("launch")
            assert launch is None or launch.startswith("rank0:"), case


def _observation() -> dict:
    [body] = [b for _, c, b, _ in ALL if c == "not_finite_on_one_rank"]
    return body["records"][1]  # a signature near the edge of 64 bits beside it, and "nan" fingerprints


def test_an_observation_survives_a_jsonl_line():
    record = _observation()
    line = json.dumps(record, separators=(",", ":"), allow_nan=False)
    assert json.loads(line) == record
    assert isinstance(record["replica"]["params"]["b"]["sig"], str) and int(record["replica"]["params"]["b"]["sig"]) > 2**53


def test_an_observation_survives_the_flight_row_and_parquet(tmp_path):
    from tensorquill.flight import record_to_row, row_to_record

    record = {**_observation(), "ts": 1.0, "grad_norm": 0.5}
    row = record_to_row(record)
    assert "replica" in json.loads(row["extra_json"])  # no column of its own: it travels with the other extra keys
    assert row_to_record(row)["replica"] == record["replica"]
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from tensorquill.flight import flight_schema

    table = pa.Table.from_pylist([row], schema=flight_schema())
    pq.write_table(table, tmp_path / "part.parquet")
    [back] = pq.read_table(tmp_path / "part.parquet").to_pylist()
    assert row_to_record(back)["replica"] == record["replica"]
