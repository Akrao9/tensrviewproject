"""Lane AS-1 of wave AS: Python's `ranks_unknown` reason has the shape of the server's.

Since lane AR-4 (wave AR) the server's `run_state` (`server/src/watchdog.rs`) lists the lowest 1,000
ranks without a current report in its `ranks_unknown` reason (`LISTED_RANKS`), counts the others
in `ranks_omitted`, a key written only when some are left out, and gives the whole count in its
`detail`. Python's `run_state` in `evidence.py` listed every rank, so the two engines named the
same ranks in different shapes. The owner's answer of Nov 27, 2026 makes Python match.

The cases are shared with `server/src/watchdog/lane_as1_tests.rs`
(`sdk/tests/fixtures/ranks_unknown/cases.json`): under the cap, exactly at it, and over it, with a
gap below the cap and with ranks the status counts and does not list. Every connected rank of a case
is judged, so in both engines the unknown ranks are those the status expects and nobody judged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorquill import evidence
from tensorquill.evidence import run_state

CASES = json.loads((Path(__file__).parent / "fixtures" / "ranks_unknown" / "cases.json").read_text())


def _expand(spans: list[list[int]]) -> list[int]:
    return [rank for first, last in spans for rank in range(first, last + 1)]


def _status(given: dict) -> dict:
    status = {"summary": "degraded", "world_size": given["world_size"],
              "ranks": [{"rank": rank, "attempt": 1, "last_step": 19, "state": "active"} for rank in given["connected"]],
              "missing_ranks": _expand(given["missing_ranks"])}
    if "missing_ranks_omitted" in given:
        status["missing_ranks_omitted"] = given["missing_ranks_omitted"]
    return status


def _reason(given: dict) -> dict:
    found = run_state(incidents=[], steps_judged=True, status=_status(given), judged_ranks=given["judged"],
                      judged_attempts={rank: 1 for rank in given["judged"]})
    (reason,) = [r for r in found["reasons"] if r["code"] == "ranks_unknown"]
    return reason


def test_the_cap_is_the_case_file_s():
    assert evidence.LISTED_RANKS == CASES["cap"] == 1000


@pytest.mark.parametrize("name", sorted(CASES["cases"]))
def test_the_shared_case_gives_the_written_reason(name):
    case = CASES["cases"][name]
    reason = _reason(case["given"])
    expected = case["expected"]
    assert reason["ranks"] == _expand(expected["ranks"]), (len(reason["ranks"]), reason.get("ranks_omitted"))
    if "ranks_omitted" in expected:
        assert reason.get("ranks_omitted") == expected["ranks_omitted"]
    else:
        assert "ranks_omitted" not in reason, reason["ranks_omitted"]
    assert reason["detail"].startswith(expected["detail_starts"] + " ("), reason["detail"]


def test_the_listed_ranks_and_the_count_add_up_to_the_detail_s_count():
    """The count a reader takes is `len(ranks) + ranks_omitted`, and the detail says the same number."""
    for name, case in CASES["cases"].items():
        reason = _reason(case["given"])
        whole = len(reason["ranks"]) + reason.get("ranks_omitted", 0)
        assert len(reason["ranks"]) <= CASES["cap"], name
        assert reason["detail"].startswith(f"{whole} rank"), (name, reason["detail"])


def test_the_detail_names_the_ranges_of_every_unknown_rank_not_only_the_listed_ones():
    """Over the cap, the detail's ranges are those of all the unknown ranks (at most 8, then a count),
    as before the cap: here the third range, past the listed thousand, is still named."""
    reason = _reason(CASES["cases"]["over_the_cap_with_gaps_lists_the_lowest"]["given"])
    assert "(1-499, 501-1199, 1201-1299)" in reason["detail"], reason["detail"]
