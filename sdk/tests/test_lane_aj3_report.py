"""Lane AJ-3 (Nov 22, 2026): A8 of `OPEN_BUGS.md` section 15. The raw Rust report's `moe_source`
(`incidents.json`, `detect`, `detect --follow`) says what Python's `evidence.moe_source` says for the
same records: of each router the first entry that is an object, holding only `form` and `point`.

Python's reading is taken twice: `moe_source` over the records as `build_evidence` hands them to it
(attempts resolved, then in step order), and `build_evidence(detectors="python")` itself.
`server/src/detect/lane_aj3_tests.rs` holds the same cases in Rust alone."""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path
from typing import Any

import pytest

from tensorquill import evidence

_ABSENT = object()


@functools.cache
def _rust_missing() -> str | None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return "pyarrow is not installed"
    from tensorquill.server_detect import server_binary

    try:
        server_binary()
    except RuntimeError as error:
        return str(error)
    return None


def row(step: int, attempt: int = 1, **fields: Any) -> dict[str, Any]:
    return {"run_id": "r", "rank": 0, "attempt": attempt, "step": step, "grad_norm": 1.0, **fields}


def sources(*given: Any) -> list[dict[str, Any]]:
    return [row(step, moe_source=source) for step, source in enumerate(given)]


CASES: dict[str, tuple[list[dict[str, Any]], Any]] = {
    # The audit's case and the author's: an extra key dropped, a string dropped, `{}` kept.
    "named": (
        sources({"r0": {"form": "scores", "extra": 123}, "r1": "bad", "r2": {}}),
        {"r0": {"form": "scores"}, "r2": {}},
    ),
    "first_object_wins": (
        sources({"r1": "bad"}, {"r1": {"form": "f", "point": "dispatched", "x": 1}}, {"r1": {"form": "g"}}),
        {"r1": {"form": "f", "point": "dispatched"}},
    ),
    "empty_object_first_wins": (
        sources({"r1": {}}, {"r1": {"form": "f", "point": "dispatched"}}),
        {"r1": {}},
    ),
    "nothing_kept": (sources({"r1": 5, "r2": "bad", "r3": [1], "r4": None}, "not a map"), _ABSENT),
    "superseded_row": (
        [
            row(0, moe_source={"r1": "bad"}),
            row(2, moe_source={"r1": {"form": "a", "point": "requested"}}),
            row(3),
            row(1, attempt=2, moe_source={"r1": {"form": "b", "point": "dispatched", "y": 2}}),
            row(2, attempt=2),
        ],
        {"r1": {"form": "b", "point": "dispatched"}},
    ),
    "well_formed": (
        sources(
            {"router.0": {"form": "scores", "point": "unknown"}, "router.1": {"form": "indices", "point": "dispatched"}},
            {"router.0": {"form": "other", "point": "requested"}},
        ),
        {"router.0": {"form": "scores", "point": "unknown"}, "router.1": {"form": "indices", "point": "dispatched"}},
    ),
}


def _python(records: list[dict[str, Any]]) -> Any:
    resolved = sorted(evidence.resolve_attempts(records), key=lambda r: int(r["step"]))
    found = evidence.moe_source(resolved)
    return _ABSENT if found is None else found


def _key(report: dict[str, Any]) -> Any:
    return report["moe_source"] if "moe_source" in report else _ABSENT


@pytest.mark.parametrize("name", sorted(CASES))
def test_python_reads_the_case_as_expected(name: str) -> None:
    records, expected = CASES[name]
    assert _python(records) == expected
    assert _key(evidence.build_evidence(records, detectors="python")) == expected


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_raw_rust_report_matches_python(name: str) -> None:
    missing = _rust_missing()
    if missing:
        pytest.skip(missing)
    from tensorquill.server_detect import server_report

    records, expected = CASES[name]
    report = server_report(records)
    # An absent key, not a present null: `.get` alone would pass on either.
    assert _key(report) == _python(records) == expected, report.get("moe_source")


@pytest.mark.parametrize("name", ["named", "nothing_kept"])
def test_the_followed_report_matches_python(name: str) -> None:
    missing = _rust_missing()
    if missing:
        pytest.skip(missing)
    from tensorquill.server_detect import FollowedDetect, server_binary

    records, expected = CASES[name]
    followed = FollowedDetect(server_binary())
    try:
        report = followed.report(records)
    finally:
        followed.close()
    assert _key(report) == _python(records) == expected, report.get("moe_source")


# --- an incomplete `grad_norms` (the author, Nov 22, 2026) -----------------------------------
# When either side of an `update_spike` comparison had a refused gradient entry, both engines
# compare the whole totals (the existing fallback). The expected results are written out: the
# same records with the bad entry deleted are no control here, since deleting it also deletes
# the evidence that the map was incomplete. Raw Rust on the typed column with the refused entry
# in `extra_json` (WAVE_AJ_CONTRACT.md, section 1) is in lane_aj3_tests.rs; through
# `server_report` the case runs as the sender on this commit writes it, and once lane AJ-1's
# sender is in, as the contract writes it.

ALL = {"a": 3.0, "b": 4.0, "c": 12.0}
SHARED = {"a": 3.0, "b": 4.0}


def update(step: int, grads: dict[str, Any], grad_norm: float, size: float) -> dict[str, Any]:
    return row(step, grad_norm=grad_norm, grad_norms=grads, update_rms={"a": size})


def steady(*sizes: float) -> list[dict[str, Any]]:
    return [update(step, ALL, 13.0, size) for step, size in enumerate(sizes)]


WHOLE = 2644**0.5
UPDATE_CASES: dict[str, tuple[list[dict[str, Any]], list[tuple[int, float, float]]]] = {
    # Refused c at step 4, whole norm still 13: 13 against 13, no rise, nothing flagged.
    "refused_on_the_later_side": (
        steady(1.0, 1.01, 1.0, 1.01) + [update(4, {**SHARED, "c": True}, 13.0, 2.0)],
        [],
    ),
    # Refused c at step 3: step 4 is compared on its whole norm, sqrt(2644) against 13.
    "refused_on_the_earlier_side": (
        steady(1.0, 1.01, 1.0)
        + [update(3, {**SHARED, "c": "12"}, 13.0, 1.01), update(4, {"a": 30.0, "b": 40.0, "c": 12.0}, WHOLE, 2.0)],
        [(4, WHOLE, 13.0)],
    ),
    # c not recorded at all (no refusal): split as before (B3, unchanged), 13 against 5.
    "left_out_without_a_refusal": (
        steady(1.0, 1.01, 1.0, 1.01) + [update(4, SHARED, 13.0, 2.0)],
        [(4, 13.0, 5.0)],
    ),
}


def _spikes(found: dict[str, Any]) -> list[tuple[int, float, float]]:
    return [
        (incident["step"], round(incident["grad_norm"], 9), round(incident["grad_norm_before"], 9))
        for incident in found.get("incidents", [])
        if incident.get("kind") == "update_spike"
    ]


def _rounded(expected: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
    return [(step, round(now, 9), round(then, 9)) for step, now, then in expected]


@pytest.mark.parametrize("name", sorted(UPDATE_CASES))
def test_python_compares_whole_totals_beside_a_refused_gradient(name: str) -> None:
    records, expected = UPDATE_CASES[name]
    found = evidence.build_evidence(records, detectors="python", update_window=3)
    assert _spikes(found) == _rounded(expected)


@pytest.mark.parametrize("name", sorted(UPDATE_CASES))
def test_the_raw_rust_report_compares_whole_totals_beside_a_refused_gradient(name: str) -> None:
    missing = _rust_missing()
    if missing:
        pytest.skip(missing)
    from tensorquill.server_detect import server_report

    records, expected = UPDATE_CASES[name]
    assert _spikes(server_report(records, thresholds={"update_window": 3})) == _rounded(expected)


SIBLING = [row(0, grad_norms={"good": float("nan"), "bad": True})]


def _non_finite(found: dict[str, Any]) -> list[Any]:
    return [i.get("gradients") for i in found["incidents"] if i.get("kind") == "non_finite"]


def test_python_keeps_a_valid_sibling_of_a_refused_gradient_non_finite() -> None:
    assert _non_finite(evidence.build_evidence(SIBLING, detectors="python")) == [["good"]]


def test_the_raw_rust_report_keeps_a_valid_sibling_of_a_refused_gradient_non_finite() -> None:
    missing = _rust_missing()
    if missing:
        pytest.skip(missing)
    from tensorquill.server_detect import server_report

    assert _non_finite(server_report(SIBLING)) == [["good"]]


def stall_checks(at_2: Any) -> list[dict[str, Any]]:
    """`c` listed unchanged at four weight checks with a gradient of 0.5, `d` always changing;
    `c`'s gradient at step 2 is `at_2` (`ABSENT`: not recorded)."""
    out = []
    for step in range(4):
        grads: dict[str, Any] = {"d": 0.5}
        value = at_2 if step == 2 else 0.5
        if value is not _ABSENT:
            grads["c"] = value
        out.append(row(step, grad_norm=None, grad_norms=grads, weights_unchanged=["c"]))
    return out


# A refused gradient resets the streak (WAVE_AJ_CONTRACT.md section 1), as a null one does; an
# absent one holds it (lane W3's contract 2). Step of the stall, or None.
STALL_CASES = {"refused_true": (True, None), "refused_string": ("0.5", None), "null": (None, None), "absent": (_ABSENT, 3)}


def _stall_step(found: dict[str, Any]) -> int | None:
    steps = [i["step"] for i in found["incidents"] if i.get("kind") == "stalled_weights"]
    return steps[0] if steps else None


@pytest.mark.parametrize("name", sorted(STALL_CASES))
def test_python_stall_streak_beside_a_refused_gradient(name: str) -> None:
    at_2, expected = STALL_CASES[name]
    assert _stall_step(evidence.build_evidence(stall_checks(at_2), detectors="python")) == expected


@pytest.mark.parametrize("name", sorted(STALL_CASES))
def test_the_raw_rust_stall_streak_beside_a_refused_gradient(name: str) -> None:
    missing = _rust_missing()
    if missing:
        pytest.skip(missing)
    from tensorquill.server_detect import server_report

    at_2, expected = STALL_CASES[name]
    found = _stall_step(server_report(stall_checks(at_2)))
    assert found == expected


# --- A7, A10 and the replica `[]` coverage through the raw report ------------------------------
# The cases are lane AJ-1's shared file itself, not a copy: `fixtures/wave_aj/cases.json` exists
# on AJ-1's branch and on the joined tree, and these tests skip where it is not. The blocks are
# compared whole against the case's `expect.blocks`.

WAVE_AJ = Path(__file__).parent / "fixtures" / "wave_aj" / "cases.json"
REPLICA_CASES = Path(__file__).parent / "fixtures" / "replica" / "incident_cases.json"


def _shared(prefixes: tuple[str, ...]) -> list[Any]:
    if not WAVE_AJ.exists():
        return [pytest.param(None, marks=pytest.mark.skip(reason="no wave_aj/cases.json on this tree (lane AJ-1's)"))]
    cases = json.loads(WAVE_AJ.read_text())["cases"]
    return [pytest.param(case, id=case["name"]) for case in cases if case["name"].startswith(prefixes)]


@pytest.mark.parametrize("case", _shared(("a7_", "a10_")))
def test_the_raw_report_gives_the_shared_a7_and_a10_blocks(case: dict[str, Any]) -> None:
    missing = _rust_missing()
    if missing:
        pytest.skip(missing)
    from tensorquill.server_detect import server_report

    report = server_report(copy.deepcopy(case["records"]), thresholds=case.get("thresholds"))
    for name, expected in case["expect"]["blocks"].items():
        assert report.get(name) == expected, name


@pytest.mark.parametrize("bad", ["invalid", []], ids=["text", "empty"])
def test_an_unreadable_spread_entry_does_not_cover_drift_in_the_raw_run_report(bad: Any) -> None:
    """Lane AJ-2's case (`test_lane_aj2_malformed_inputs.py`): with each rank's gathered spread
    saying nothing readable of `w`, the observations' drift on `w` is raised from `replica`."""
    missing = _rust_missing()
    if missing:
        pytest.skip(missing)
    from tensorquill.server_detect import server_run_report

    document = json.loads(REPLICA_CASES.read_text())
    records = copy.deepcopy(document["cases"]["explicit_spread_and_observations_on_one_step"]["records"])
    for record in records:
        record["replica_spread"]["w"] = bad
    replica = server_run_report(records)["replica"]
    assert replica["steps"][0]["gathered"] == ["b"]
    assert replica["replica_drift"] == [{"step": 99, "parameters": ["w"], "source": "replica",
                                         "launch": "2f6c1a9e4b7d4c1e9a3f6b8d0e1c2a47", "group": [0, 1]}]


# --- a number too large for a float in `extra_json` (the author, Nov 22, 2026) ----------------
# "One `10**400` value can make Rust discard every `extra_json` field on that row. Preserve the
# original integer for Python round-trip reading, treat it as unreadable where a float measurement
# is required, and keep valid sibling fields available to Rust. Use the existing ignored regression
# and add a live-server check: a valid replica observation beside the oversized integer must still
# be judged. Verify that Python reads the integer back exactly."

OVERSIZED = 10**400


def test_the_row_round_trip_keeps_the_exact_integer() -> None:
    from tensorquill.flight import record_to_row, row_to_record

    record = row(7, ts=7.0, update_rms={"w": 0.5, "big": OVERSIZED}, replica_spread={"v": [0.0, 0.0, 0.0], "huge": [OVERSIZED]})
    back = row_to_record(record_to_row(record))
    assert back == record
    for got in (back["update_rms"]["big"], back["replica_spread"]["huge"][0]):
        assert got == OVERSIZED and type(got) is int


# Lane AB2's live-server pattern: one file per rank through `send_file`, the run's `replica` read
# from its `watchdog.json`. The records are `drift_between_two_ranks` of the replica fixtures; the
# verdict is written out here, not read from the fixture.
DRIFT_ON_B = {
    "steps": [{"launch": "2f6c1a9e4b7d4c1e9a3f6b8d0e1c2a47", "group": [0, 1], "step": 99, "coverage": "complete",
               "fingerprints": "drift", "drift_parameters": ["b"], "signatures": "differ",
               "signature_parameters": ["b"]}],
    "replica_drift": [{"step": 99, "parameters": ["b"], "source": "replica",
                       "launch": "2f6c1a9e4b7d4c1e9a3f6b8d0e1c2a47", "group": [0, 1]}],
}
LIVE = ("--flush-steps", "50", "--flush-seconds", "0.2", "--watchdog-seconds", "0.5")


def _two_ranks(oversized: bool) -> list[dict[str, Any]]:
    document = json.loads(REPLICA_CASES.read_text())
    records = copy.deepcopy(document["cases"]["drift_between_two_ranks"]["records"])
    if oversized:
        for record in records:  # every entry refused, so the whole map travels in extra_json
            record["update_rms"] = {"big": OVERSIZED}
            record["oversized_note"] = OVERSIZED
    return records


@pytest.mark.parametrize("oversized", [True, False], ids=["beside_10_400", "control"])
def test_a_live_server_judges_a_replica_observation_beside_an_oversized_integer(tmp_path: Path, oversized: bool) -> None:
    missing = _rust_missing()
    if missing:
        pytest.skip(missing)
    pytest.importorskip("pyarrow.flight")
    import time

    from tensorquill.flight import send_file
    from test_flight_e2e import BINARY, Server, stored_records

    if BINARY is None:
        pytest.skip("tensorquill-server is not built")
    records = _two_ranks(oversized)
    run = records[0]["run_id"]
    store = tmp_path / "store"
    server = Server(store, 0, *LIVE)
    try:
        for rank in (0, 1):
            file = tmp_path / f"{run}.rank{rank}.jsonl"
            file.write_text("".join(json.dumps(r) + "\n" for r in records if r["rank"] == rank))
            assert send_file(file, server.endpoint, run, rank, timeout=30.0)
        status_path = store / "runs" / run / "watchdog.json"
        deadline, replica = time.monotonic() + 30.0, None
        while time.monotonic() < deadline:
            try:
                replica = json.loads(status_path.read_text(encoding="utf-8")).get("replica")
            except (FileNotFoundError, ValueError):
                replica = None
            if replica and any(step.get("coverage") == "complete" for step in replica.get("steps", ())):
                break
            time.sleep(0.05)
    finally:
        assert server.terminate() == 0
    assert replica is not None and replica["steps"] == DRIFT_ON_B["steps"], replica
    assert replica["replica_drift"] == DRIFT_ON_B["replica_drift"]
    if oversized:  # the store holds the integer itself, and reads it back exact
        for rank in (0, 1):
            (kept,) = stored_records(store, run, rank)
            assert kept["update_rms"]["big"] == OVERSIZED and type(kept["update_rms"]["big"]) is int
            assert kept["oversized_note"] == OVERSIZED
