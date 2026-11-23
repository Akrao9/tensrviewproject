"""Lane AJ-2 of wave AJ (Nov 22, 2026): items A2, A4, A5, A6 and the Python side of A9 from
`OPEN_BUGS.md` section 15 (the detector correctness audit, checked on `5f6c11b`).

A map entry that the typed column of `flight.py` cannot carry is left unread on its own record. The
rest of its map is judged as before:
- a router's counts must be a list of signed 64-bit integers;
- a replica spread must be a non-empty list of numbers, where None reads as NaN (Rust reads a
  null that way) and "nan", "inf" or "-inf" read as the non-finite numbers the sink writes;
- a weight check needs at least one norm the column carries.

A router counts as read when the sum of its counts is above 0, in both engines. Called directly,
`detect_non_finite` and `detect_replica_drift` judge their records in step order.

Each case runs `build_evidence(detectors="python")`. Where the records can reach the Rust
detectors, the case also runs `detectors="rust"` and the raw report, and the engines must agree.
Some cross-engine tests need lane AJ-1's fix to `flight.py` (an entry the column cannot hold goes
out alone, not the whole map): `test_*_rust_awaits_aj1`. They fail on lane AJ-2's branch alone
and must pass on the joined tree.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from tensorquill import evidence
from tensorquill.server_detect import server_report, server_run_report

from test_flight_e2e import BINARY

needs_server = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
REPLICA_CASES = Path(__file__).parent / "fixtures" / "replica" / "incident_cases.json"
COLLAPSED = [100, 0, 0, 0]


def _short(value):
    return repr(value)[:24]


def rec(step: int, attempt: int = 1, **fields):
    return {"run_id": "r", "rank": 0, "attempt": attempt, "step": step, "grad_norm": 1.0, **fields}


def kinds(ev):
    return [(i["kind"], i.get("step"), i.get("router"), i.get("layer")) for i in ev["incidents"]]


def checks_of(ev, name):
    return [n for n in ev.get("not_checked") or [] if n.get("check") == name]


def agree(records):
    """Python's and the SDK's Rust mode's evidence on `records` agree on incidents, run state,
    `not_checked` and the MoE and replica kind checks; returns Python's."""
    py = evidence.build_evidence(copy.deepcopy(records), detectors="python")
    rs = evidence.build_evidence(copy.deepcopy(records), detectors="rust")
    assert kinds(py) == kinds(rs)
    assert py["run_state"] == rs["run_state"]
    assert py.get("not_checked") == rs.get("not_checked")
    for kind in ("moe_imbalance", "starved_experts", "replica_drift"):
        assert py["kind_checks"].get(kind) == rs["kind_checks"].get(kind), kind
    return py


# --- A2: a router is read when the sum of its counts is above 0 -----------------------------------

@pytest.mark.parametrize("counts", [[5, -5], [1, -5], [0, 0]])
def test_a_router_whose_counts_sum_to_zero_or_less_is_not_read(counts):
    records = [rec(0, moe={"r0": counts}, moe_top_k={"r0": 1})]
    assert evidence.routers_not_read_not_checked(records)["routers"] == ["r0"]
    ev = evidence.build_evidence(records, detectors="python")
    assert [n["routers"] for n in checks_of(ev, "moe_routing")] == [["r0"]]
    assert ev["run_state"]["state"] == "degraded"
    assert evidence.input_counts(records)["moe"] == 0


def test_a_positive_sum_with_a_negative_count_is_read_as_today():
    """`[5, -3]` sums to 2: read, and nothing raised, in both engines (pinned as it is on 212205d)."""
    records = [rec(0, moe={"r0": [5, -3]}, moe_top_k={"r0": 1})]
    assert evidence.routers_not_read_not_checked(records) is None
    ev = evidence.build_evidence(records, detectors="python")
    assert ev["incidents"] == [] and ev["run_state"]["state"] == "healthy"


@needs_server
@pytest.mark.parametrize("counts", [[5, -5], [1, -5], [0, 0], [5, -3]])
def test_the_engines_agree_on_a_signed_count(counts):
    records = [rec(0, moe={"r0": counts}, moe_top_k={"r0": 1})]
    py = agree(records)
    raw = server_report(records)
    assert [n for n in raw["not_checked"] if n["check"] == "moe_routing"] == checks_of(py, "moe_routing")


# --- A5: a router entry the column cannot carry is absent on that record -------------------------

BAD_ENTRIES = ["x", {"a": 1}, 7, None, True, [1.0, 2.0], [True, 3], ["1", "2"], (100, 0, 0, 0), [2**63, 1], [-(2**63) - 1]]


@pytest.mark.parametrize("bad", BAD_ENTRIES, ids=_short)
def test_an_unreadable_router_beside_a_collapsed_one(bad):
    records = [rec(s, moe={"bad": bad, "good": COLLAPSED}, moe_top_k={"good": 1}) for s in range(15)]
    assert evidence._router_sizes(records) == {"good": 4}
    assert evidence.routers_not_read_not_checked(records) is None
    assert evidence.input_counts(records)["moe"] == 15
    ev = evidence.build_evidence(records, detectors="python")
    assert kinds(ev) == [("moe_imbalance", 0, "good", 1)]
    assert checks_of(ev, "moe_routing") == []
    assert {s["router"] for s in evidence._moe_samples(records, {"good": 4}, {"good": 1}, evidence.OVERLOAD_THRESHOLD)} == {"good"}


def test_an_unreadable_router_alone_is_neither_read_nor_warned():
    records = [rec(s, moe={"bad": "x"}) for s in range(3)]
    assert evidence._router_sizes(records) == {}
    assert evidence.routers_not_read_not_checked(records) is None
    assert evidence.input_counts(records)["moe"] == 0
    ev = evidence.build_evidence(records, detectors="python")
    assert checks_of(ev, "moe_routing") == [] and ev["incidents"] == []
    assert ev["kind_checks"]["moe_imbalance"]["state"] != "checked"


def test_a_moe_that_is_not_a_map_is_not_read():
    records = [rec(0, moe=["x"]), rec(1, moe={"good": COLLAPSED}, moe_top_k={"good": 1})]
    assert evidence._router_sizes(records) == {"good": 4}
    evidence.build_evidence(records, detectors="python")


def test_capture_word_that_a_router_is_unreadable_is_kept():
    records = [rec(s, moe={"bad": "x", "good": COLLAPSED}, moe_top_k={"good": 1},
                   **({"moe_refused": {"bad": "its output is not expert indices"}} if s == 0 else {})) for s in range(15)]
    ev = evidence.build_evidence(records, detectors="python")
    assert [(n["routers"], n["step"]) for n in checks_of(ev, "router")] == [(["bad"], 0)]
    assert kinds(ev) == [("moe_imbalance", 0, "good", 1)]


def _later_valid():
    early = [rec(s, moe={"bad": "x", "good": COLLAPSED}, moe_top_k={"good": 1, "bad": 1}) for s in range(5)]
    late = [rec(s, moe={"bad": [25, 25, 25, 25], "good": COLLAPSED}, moe_top_k={"good": 1, "bad": 1}) for s in range(5, 15)]
    return early + late


def test_an_unreadable_router_reserves_no_layer_and_is_judged_once_valid():
    records = _later_valid()
    assert list(evidence._router_sizes(records)) == ["good", "bad"]
    ev = evidence.build_evidence(records, detectors="python")
    assert kinds(ev) == [("moe_imbalance", 0, "good", 1)]
    assert checks_of(ev, "moe_routing") == []
    samples = evidence._moe_samples(records, evidence._router_sizes(records), {"good": 1, "bad": 1}, evidence.OVERLOAD_THRESHOLD)
    layers = {(s["router"], s["layer"]) for s in samples}
    assert layers == {("good", 1), ("bad", 2)}
    bad = [s for s in samples if s["router"] == "bad"]
    assert [s["step"] for s in bad if not s.get("missing")] == list(range(5, 15))
    assert all(s.get("missing") for s in bad if s["step"] < 5)


@needs_server
def test_unreadable_router_beside_a_collapsed_one_rust_awaits_aj1():
    records = [rec(s, moe={"bad": "x", "good": COLLAPSED}, moe_top_k={"good": 1}) for s in range(15)]
    py = agree(records)
    assert kinds(py) == [("moe_imbalance", 0, "good", 1)]
    assert kinds(server_report(records)) == kinds(py)


@needs_server
def test_out_of_range_count_beside_a_valid_router_rust_awaits_aj1():
    records = [rec(s, moe={"bad": [2**63, 1], "good": COLLAPSED}, moe_top_k={"good": 1}) for s in range(15)]
    py = agree(records)
    assert kinds(py) == [("moe_imbalance", 0, "good", 1)]


@needs_server
def test_unreadable_router_layers_rust_awaits_aj1():
    records = _later_valid()
    agree(records)
    raw = server_report(records, samples=True)
    assert {(s["router"], s["layer"]) for s in raw["samples"]} == {("good", 1), ("bad", 2)}
    assert kinds(raw) == [("moe_imbalance", 0, "good", 1)]


@needs_server
def test_capture_word_kept_rust_awaits_aj1():
    records = [rec(s, moe={"bad": "x", "good": COLLAPSED}, moe_top_k={"good": 1},
                   **({"moe_refused": {"bad": "its output is not expert indices"}} if s == 0 else {})) for s in range(15)]
    agree(records)


# --- A4: a replica spread entry that is not a non-empty list of numbers is not judged ------------

DRIFTING = [1e-2, 0.0, 0.0]


@pytest.mark.parametrize("bad", [[], "invalid", {"a": 1}, 3.0, [True, 0.0, 0.0], ["1.5", 0.0, 0.0], ["NaN", 0.0, 0.0],
                                 [10**400, 0.0, 0.0], (1e-2, 0.0, 0.0)], ids=_short)
def test_an_unreadable_spread_beside_a_drifting_one(bad):
    records = [rec(0, replica_spread={"bad_param": bad, "valid_param": DRIFTING})]
    found = evidence.detect_replica_drift(records)
    assert found is not None and list(found["parameters"]) == ["valid_param"]
    ev = evidence.build_evidence(records, detectors="python")
    assert [(i["kind"], sorted(i["parameters"])) for i in ev["incidents"]] == [("replica_drift", ["valid_param"])]


@pytest.mark.parametrize("bad", [[], "invalid", [10**400, 0.0, 0.0]], ids=_short)
def test_an_unreadable_spread_alone_raises_nothing(bad):
    records = [rec(0, replica_spread={"w": bad})]
    assert evidence.detect_replica_drift(records) is None
    assert evidence.build_evidence(records, detectors="python")["incidents"] == []


def test_a_spread_that_is_not_a_map_raises_nothing():
    assert evidence.detect_replica_drift([rec(0, replica_spread=["w"])]) is None


@pytest.mark.parametrize("value", [math.nan, math.inf, "nan", "inf", "-inf", None], ids=_short)
def test_a_non_finite_spread_is_still_reported_as_non_finite(value):
    found = evidence.detect_replica_drift([rec(0, replica_spread={"w": [value, 0.0, 0.0]})])
    assert found["parameters"]["w"]["non_finite"] is True


@needs_server
@pytest.mark.parametrize("table", [{"w": []}, {"w": [None, 0.0, 0.0]}, {"w": ["nan", 0.0, 0.0]}, {"w": [1e-9, 0.0, 0.0]}], ids=_short)
def test_the_engines_agree_on_a_spread_the_column_carries(table):
    agree([rec(0, replica_spread=table)])


@needs_server
@pytest.mark.parametrize("bad", ["invalid", [10**400, 0.0, 0.0]], ids=_short)
def test_unreadable_spread_beside_a_drifting_one_rust_awaits_aj1(bad):
    records = [rec(0, replica_spread={"bad_param": bad, "valid_param": DRIFTING})]
    py = agree(records)
    assert [(i["kind"], sorted(i["parameters"])) for i in py["incidents"]] == [("replica_drift", ["valid_param"])]
    assert [sorted(i["parameters"]) for i in server_report(records)["incidents"]] == [["valid_param"]]


# --- A4, cross-rank: an unreadable entry is no coverage -----------------------------------------

def _scoped_case(bad):
    document = json.loads(REPLICA_CASES.read_text())
    records = copy.deepcopy(document["cases"]["explicit_spread_and_observations_on_one_step"]["records"])
    for record in records:  # the gathered spread says nothing readable of w; the observations show w drift
        record["replica_spread"]["w"] = bad
    return records, document["tolerance"]


EXPECTED_UNCOVERED = {
    "gathered": ["b"],
    "replica_drift": [{"step": 99, "parameters": ["w"], "source": "replica",
                       "launch": "2f6c1a9e4b7d4c1e9a3f6b8d0e1c2a47", "group": [0, 1]}],
}


@pytest.mark.parametrize("bad", ["invalid", []], ids=_short)
def test_an_unreadable_spread_entry_does_not_cover_drift_seen_by_the_observations(bad):
    records, tolerance = _scoped_case(bad)
    result = evidence.compare_replicas(records, tolerance=tolerance)
    assert result["steps"][0]["gathered"] == EXPECTED_UNCOVERED["gathered"]
    assert result["steps"][0]["drift_parameters"] == ["w"]
    assert result["replica_drift"] == EXPECTED_UNCOVERED["replica_drift"]


@needs_server
@pytest.mark.parametrize("bad", ["invalid", []], ids=_short)
def test_unreadable_spread_entry_coverage_raw_rust_awaits_aj3(bad):
    """The raw cross-rank report (`detect --run`) gives Python's result. On 212205d it counts the
    unreadable entry as gathered and raises nothing: lane AJ-3 coordinates the fix."""
    records, tolerance = _scoped_case(bad)
    replica = server_run_report(records)["replica"]
    assert replica["steps"][0]["gathered"] == EXPECTED_UNCOVERED["gathered"]
    assert replica["replica_drift"] == EXPECTED_UNCOVERED["replica_drift"]


# --- A6: the two public detectors judge in step order --------------------------------------------

def test_detect_non_finite_called_directly_gives_the_earliest_step():
    records = [rec(10, grad_norm=math.nan), rec(5, grad_norm=math.inf)]
    assert evidence.detect_non_finite(records)["step"] == 5
    built = next(i for i in evidence.build_evidence(records, detectors="python")["incidents"] if i["kind"] == "non_finite")
    assert built["step"] == 5


def test_detect_non_finite_keeps_list_order_among_equal_steps():
    records = [rec(7, grad_norm=1.0, grad_norms={"a": math.nan}), rec(7, grad_norm=1.0, grad_norms={"b": math.nan}),
               rec(3, grad_norm=1.0)]
    assert evidence.detect_non_finite(records)["gradients"] == ["a"]
    records[0], records[1] = records[1], records[0]
    assert evidence.detect_non_finite(records)["gradients"] == ["b"]


def test_detect_replica_drift_called_directly_gives_the_earliest_step():
    records = [rec(10, replica_spread={"w2": [1e-3, 0.0, 0.0]}), rec(5, replica_spread={"w1": [1e-4, 0.0, 0.0]})]
    found = evidence.detect_replica_drift(records)
    assert (found["step"], list(found["parameters"])) == (5, ["w1"])
    built = next(i for i in evidence.build_evidence(records, detectors="python")["incidents"] if i["kind"] == "replica_drift")
    assert (built["step"], list(built["parameters"])) == (5, ["w1"])


def test_detect_replica_drift_keeps_list_order_among_equal_steps():
    first, second = rec(4, replica_spread={"a": [1e-3, 0.0, 0.0]}), rec(4, replica_spread={"b": [1e-3, 0.0, 0.0]})
    assert list(evidence.detect_replica_drift([first, second, rec(9)])["parameters"]) == ["a"]
    assert list(evidence.detect_replica_drift([second, first, rec(9)])["parameters"]) == ["b"]


def test_the_direct_calls_do_not_reorder_the_callers_list():
    records = [rec(10, grad_norm=math.nan), rec(5, grad_norm=math.inf)]
    evidence.detect_non_finite(records)
    evidence.detect_replica_drift(records)
    assert [r["step"] for r in records] == [10, 5]


# --- A9, the Python side: a weight map the column cannot carry is no weight check ----------------

A9_RECORDS = [
    {"run_id": "r", "rank": 0, "attempt": 1, "step": 0, "weight_norms": {"w": 1.0}},
    {"run_id": "r", "rank": 0, "attempt": 1, "step": 50, "weight_norms": {"w": 2.0}},
    {"run_id": "r", "rank": 0, "attempt": 1, "step": 100, "weight_norms": {"w": "not_a_number"}},
]
A9_EXPECTED = {"checks": 2, "parameters": 1, "largest": [{"name": "w", "ratio": 2.0, "step": 50, "previous_step": 0,
                                                          "before": 1.0, "after": 2.0}]}


def test_a_weight_map_with_no_entry_the_column_carries_is_no_check():
    assert evidence.weight_jumps(copy.deepcopy(A9_RECORDS)) == A9_EXPECTED
    assert evidence.build_evidence(copy.deepcopy(A9_RECORDS), detectors="python")["weight_jumps"] == A9_EXPECTED


def test_a_weight_map_of_nulls_is_still_a_check_as_in_rust():
    records = copy.deepcopy(A9_RECORDS)
    records[2]["weight_norms"] = {"w": None}
    assert evidence.weight_jumps(records)["checks"] == 3


@needs_server
def test_the_a9_fixture_gives_two_checks_in_every_engine():
    """Python, raw Rust and the SDK's Rust mode, which makes the block in Python (A9)."""
    assert server_report(copy.deepcopy(A9_RECORDS))["weight_jumps"] == A9_EXPECTED
    assert evidence.build_evidence(copy.deepcopy(A9_RECORDS), detectors="rust")["weight_jumps"] == A9_EXPECTED
