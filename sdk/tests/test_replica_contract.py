"""The replica observation contract's cases, checked against a small reference reading of its rules.

`REPLICA_OBSERVATION_CONTRACT.md` (step 0 of the SDK-22 wave, Nov 20, 2026) says what each rank
records and how a reader compares the ranks' records. The cases in `fixtures/replica/` are its
examples, each with the result both engines must give. Nothing in the SDK or the server reads
the `replica` key yet: this file is not a test of either. It keeps the page and the cases from
disagreeing, by reading the rules a second time in code (`reference`), and it shows that an
observation survives the two ways a record travels.

The Python reader and the server each get their own test over the same two files.

Revised the same day after the author's review. Each case added then was first run against the
reference as committed in `89bef4e`, and each read wrongly there:

- `explicit_spread_in_another_launch`, `explicit_spread_of_another_group`,
  `explicit_spread_without_a_scope`, `explicit_spread_covers_one_parameter`: nothing raised, since
  a gathered spread was looked up by step number alone and silenced every comparison of that step.
- `explicit_spread_without_a_scope_and_both_drift`: one entry where two are due.
- `every_parameter_unread`, `group_of_one`: fingerprints and signatures "agree" with nothing compared.
- `signature_method_differs`: no word on signatures at all.
- `two_launches_with_one_attempt_number`: with the launch written as rank 0's attempt, as the
  first draft had it, drift on `w` between a new row and an old one.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from itertools import combinations
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
    prints_compared = signatures_compared = 0  # "agree" needs at least one comparison (contract 4)
    for name in names if len(ranks) > 1 else ():
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
        if same_method:
            signatures_compared += 1
            if len({row["sig"] for row in rows}) > 1:
                differ.append(name)
        if not same_device:
            continue
        prints_compared += 1
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
    if prints_compared:
        out["fingerprints"] = "drift" if drift else "agree"
        if drift:
            out["drift_parameters"] = drift
    else:
        out["fingerprints"] = "not_compared"
        out["fingerprints_why"] = "one_rank" if len(ranks) < 2 else "no_parameters" if same_device else "device"
    if signatures_compared:
        out["signatures"] = "differ" if differ else "agree"
        if differ:
            out["signature_parameters"] = differ
    else:
        out["signatures"] = "not_compared"
        out["signatures_why"] = "one_rank" if len(ranks) < 2 else "no_parameters" if same_method else "method"
    return out


def _scope(record: dict) -> tuple | None:
    """The launch and group a gathered spread says it covers, or None when it says neither
    (an old record, or a table the job computed itself): contract 5."""
    scope = record.get("replica_spread_scope") or {}
    if scope.get("launch") is None or scope.get("group") is None:
        return None
    return scope["launch"], tuple(scope["group"])


def reference(records: list[dict], tolerance: float) -> dict:
    """The contract read once more, in code: what a reader says of `records`, one run's."""
    assert len({r["run_id"] for r in records}) == 1, "a reader's input is one run (contract 2)"
    steps, incidents = [], []
    legacy = [_legacy(r, tolerance) for r in records if "replica_spread" in r and "replica" not in r]
    # Gathered spreads, each kept under what it says it covers. One with no scope covers nothing
    # but itself, whatever step it is on.
    gathered: dict[tuple, dict[str, set]] = defaultdict(lambda: {"covered": set(), "over": set()})
    for record in records:
        if "replica_spread" in record:
            held = gathered[(_scope(record), record["step"])]
            held["covered"] |= set(record["replica_spread"])
            held["over"] |= set(_legacy(record, tolerance).get("parameters", ()))
    by_step: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for record in records:
        if "replica" in record:
            # one rank's newest attempt of a launch: later in the file wins (contract 1's order)
            by_step[(record["replica"]["launch"], record["step"])][record["rank"]] = record["replica"]
    # Contract 4's order: by step, then the rows that name no launch, then by launch, then by group.
    for (launch, step), seen in sorted(by_step.items(), key=lambda item: (item[0][1], item[0][0] is not None, item[0][0] or "")):
        groups = sorted({tuple(o["group"]) for o in seen.values()})
        if launch is None:
            [group] = groups if len(groups) == 1 else [None]
            steps.append({"launch": None, "group": list(group) if group else None, "step": step,
                          "coverage": "not_established", "why": "launch"})
            continue
        if any(set(a) & set(b) for a, b in combinations(groups, 2)) or any(r not in o["group"] for r, o in seen.items()):
            steps.append({"launch": launch, "group": None, "step": step, "coverage": "not_established", "why": "group"})
            continue
        for group in groups:
            members = {r: o for r, o in seen.items() if tuple(o["group"]) == group}
            missing = [r for r in group if r not in members]
            entry = {"launch": launch, "group": list(group), "step": step}
            if missing:
                steps.append({**entry, "coverage": "insufficient", "missing": missing})
                continue
            verdict = _compare(members, tolerance)
            # Only a gathered spread of this launch, this group and this step decides anything
            # here, and only for the parameters it holds.
            covered = gathered[((launch, group), step)]["covered"] if ((launch, group), step) in gathered else set()
            if covered:
                verdict = {"gathered": sorted(covered), **verdict}
            steps.append({**entry, "coverage": "complete", **verdict})
            raised = [name for name in verdict.get("drift_parameters", ()) if name not in covered]
            if raised:
                incidents.append({"step": step, "parameters": raised, "source": "replica", "launch": launch, "group": list(group)})
    for (scope, step), held in gathered.items():
        if held["over"]:
            named = {} if scope is None else {"launch": scope[0], "group": list(scope[1])}
            incidents.append({"step": step, "parameters": sorted(held["over"]), "source": "replica_spread", **named})
    order = lambda i: (i["step"], i["source"], i.get("launch") or "", i.get("group") or [])  # noqa: E731
    out = {"steps": steps, "replica_drift": sorted(incidents, key=order)}
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
            "explicit_spread_clean_and_observations_drift", "two_launches_with_one_attempt_number",
            "explicit_spread_in_another_launch", "explicit_spread_of_another_group", "explicit_spread_without_a_scope",
            "explicit_spread_covers_one_parameter", "explicit_spread_without_a_scope_and_both_drift",
            "every_parameter_unread", "group_of_one", "signature_method_differs"} <= names
    # Every no-incident case raises nothing, and every incident case raises exactly one incident,
    # but for the one case whose point is two comparisons that cannot be shown to be the same one.
    two = {"explicit_spread_without_a_scope_and_both_drift"}
    for file, case, body, _ in ALL:
        raised = body["expected"]["replica_drift"]
        assert (raised == []) if file == FILES[0] else (len(raised) == (2 if case in two else 1)), case
        for incident in raised:
            if incident["source"] == "replica":  # an incident from observations names what was compared
                assert incident["launch"] and incident["group"], case


def test_agreement_is_never_said_of_nothing():
    """Complete coverage means every rank sent a record; "agree" needs at least one comparison."""
    for _, case, body, _ in ALL:
        for step in body["expected"]["steps"]:
            if step["coverage"] != "complete":
                assert "fingerprints" not in step and "signatures" not in step, case
                continue
            assert step["fingerprints"] in ("agree", "drift", "not_compared"), case
            assert step["signatures"] in ("agree", "differ", "not_compared"), case
            assert (step["fingerprints"] == "not_compared") is ("fingerprints_why" in step), case
            assert (step["signatures"] == "not_compared") is ("signatures_why" in step), case
    [body] = [b for _, c, b, _ in ALL if c == "every_parameter_unread"]
    [step] = body["expected"]["steps"]
    assert (step["coverage"], step["fingerprints"], step["signatures"]) == ("complete", "not_compared", "not_compared")
    assert body["expected"]["replica_drift"] == []


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


def test_a_launch_named_by_rank_0s_attempt_would_pair_two_launches():
    """Why a launch is a nonce: an attempt is wall-clock milliseconds, and a clock can be behind."""
    [(body, tolerance)] = [(b, t) for _, c, b, t in ALL if c == "two_launches_with_one_attempt_number"]
    rank0 = [r for r in body["records"] if r["rank"] == 0]
    assert len(rank0) == 2 and rank0[0]["attempt"] == rank0[1]["attempt"]
    assert rank0[0]["replica"]["launch"] != rank0[1]["replica"]["launch"]
    renamed = json.loads(json.dumps(body["records"]))
    for record in renamed:
        record["replica"]["launch"] = f"rank0:{rank0[0]['attempt']}"  # the first draft's form
    assert [i["parameters"] for i in reference(renamed, tolerance)["replica_drift"]] == [["w"]]
    assert reference(body["records"], tolerance)["replica_drift"] == []


def test_a_gathered_spread_decides_only_what_it_covers():
    """The four ways a gathered spread can be of something else: each leaves the drift raised."""
    for case in ("explicit_spread_in_another_launch", "explicit_spread_of_another_group",
                 "explicit_spread_without_a_scope", "explicit_spread_covers_one_parameter"):
        [(body, tolerance)] = [(b, t) for _, c, b, t in ALL if c == case]
        assert all(set(r["replica_spread"]) and _legacy(r, tolerance)["drift"] is False
                   for r in body["records"] if "replica_spread" in r), case  # every gathered spread is clean
        [raised] = reference(body["records"], tolerance)["replica_drift"]
        assert raised["source"] == "replica", case
    # The named regression is two launches on one step number, the clean spread in the earlier one.
    [(body, tolerance)] = [(b, t) for _, c, b, t in ALL if c == "explicit_spread_in_another_launch"]
    assert {r["step"] for r in body["records"]} == {99}
    assert len({r["replica"]["launch"] for r in body["records"]}) == 2
    [raised] = reference(body["records"], tolerance)["replica_drift"]
    assert raised["launch"] not in {r["replica_spread_scope"]["launch"] for r in body["records"] if "replica_spread" in r}


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


def test_a_launch_is_a_nonce_and_holds_no_attempt_or_count():
    """torchrun's count starts at 0 on every fresh launch, and an attempt is a clock reading that
    can repeat on another host. A launch is 32 hex digits from the system's entropy."""
    for _, case, body, _ in ALL:
        attempts = {str(r["attempt"]) for r in body["records"]}
        for record in body["records"]:
            for launch in ((record.get("replica") or {}).get("launch"), (record.get("replica_spread_scope") or {}).get("launch")):
                assert launch is None or re.fullmatch(r"[0-9a-f]{32}", launch), case
                assert launch is None or not any(a in launch for a in attempts if len(a) > 4), case


_M32 = 0xFFFFFFFF


def name_v1_seed(name: str) -> int:
    """Contract 1: the first four bytes of SHA-256 of the name in UTF-8, as a big-endian number."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "big")


def name_v1_weight(index: int, seed: int) -> float:
    h = (index * 0x9E3779B1 + (seed + 1) * 0x7F4A7C15) & _M32
    h = ((h ^ (h >> 16)) * 0x85EBCA6B) & _M32
    h = ((h ^ (h >> 13)) * 0xC2B2AE35) & _M32
    h ^= h >> 16
    return h * (2.0 / 4294967296.0) - 1.0


def test_name_v1_has_one_meaning():
    """The page's test vector, from the standard library alone, so it holds in any process and
    any SDK version. Python's `hash()` of a string would differ between two ranks."""
    assert name_v1_seed("layers.0.weight") == 1749384365 == 0x68457CAD
    assert name_v1_seed("módulo.权重") == 1616250134  # UTF-8, no normalisation
    weights = [name_v1_weight(i, 1749384365) for i in range(4)]
    assert weights == [-0.055502989795058966, -0.19597370875999331, -0.6474399706348777, 0.46693889796733856]
    assert sum(w * x for w, x in zip(weights, (1.0, 2.0, 3.0, 4.0))) == pytest.approx(-0.5220147273503244, rel=1e-12)


def test_todays_weight_function_takes_a_seed_of_full_width():
    """`name-v1` keeps today's weights and changes only the seed, from a place in a list to a
    32-bit number. The function is the SDK's own: the product with the seed must not overflow."""
    torch = pytest.importorskip("torch")
    from tensorquill.distributed import _projection_f64, _projection_weights

    for seed in (name_v1_seed("layers.0.weight"), 0, 0x80000000, 0xFFFFFFFF):
        real = _projection_weights(0, 6, seed, torch.device("cpu")).tolist()
        assert real == [name_v1_weight(i, seed) for i in range(6)], hex(seed)
    got = float(_projection_f64(torch.tensor([1.0, 2.0, 3.0, 4.0]), name_v1_seed("layers.0.weight")))
    assert got == pytest.approx(-0.5220147273503244, rel=1e-12)


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
