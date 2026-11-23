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
