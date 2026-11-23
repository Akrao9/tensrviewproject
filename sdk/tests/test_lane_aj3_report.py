"""Lane AJ-3 (Nov 22, 2026): A8 of `OPEN_BUGS.md` section 15. The raw Rust report's `moe_source`
(`incidents.json`, `detect`, `detect --follow`) says what Python's `evidence.moe_source` says for the
same records: of each router the first entry that is an object, holding only `form` and `point`.

Python's reading is taken twice: `moe_source` over the records as `build_evidence` hands them to it
(attempts resolved, then in step order), and `build_evidence(detectors="python")` itself.
`server/src/detect/lane_aj3_tests.rs` holds the same cases in Rust alone."""

from __future__ import annotations

import functools
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


@pytest.mark.xfail(strict=False, reason="A3: until lane AJ-1's sender keeps the typed map, the whole map leaves the column")
def test_the_raw_rust_report_keeps_a_valid_sibling_of_a_refused_gradient_non_finite() -> None:
    missing = _rust_missing()
    if missing:
        pytest.skip(missing)
    from tensorquill.server_detect import server_report

    assert _non_finite(server_report(SIBLING)) == [["good"]]
