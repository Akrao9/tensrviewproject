"""Lane AJ-2 of wave AJ (Nov 22, 2026), the Python side of A7 and A10 (OPEN_BUGS.md section 15, the
author's decisions of Nov 22 in `d774247`).

A7, `update_trends`: a skipped update contributes neither to a parameter's baseline nor to the lowest
ratio. No record names which optimizer applied its update to a parameter: `step_skipped_part` counts
optimizers and names none. So on a partial skip every parameter is ambiguous, and none is judged on
that step. `update_spike` reads a partial step the same way (WAVE_AE_CONTRACT.md, section 4).

A10, `weight_jumps`: a change of attempt between two consecutive weight checks resets the comparison.
The first check after a reload is the new baseline, and a real jump within the new attempt is still
listed. `weight_norm_trends` is unchanged.

Raw Rust must give the same blocks, and lane AJ-3 makes that change: `test_*_raw_rust_awaits_aj3`
fail on lane AJ-2's branch alone and must pass on the joined tree. The SDK's Rust mode is not
compared, since it makes these blocks in Python (A9).
"""

from __future__ import annotations

import copy

import pytest

from tensorquill import evidence
from tensorquill.server_detect import server_report

from test_flight_e2e import BINARY

needs_server = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")


def rec(step: int, attempt: int = 1, **fields):
    return {"run_id": "r", "rank": 0, "attempt": attempt, "step": step, "grad_norm": 1.0, **fields}


# --- A7 -------------------------------------------------------------------------------------------

FULL_SKIP = ([rec(s, update_rms={"w": 1.0}) for s in range(12)]
             + [rec(12, step_skipped=True, step_skipped_by="scaler", update_rms={"w": 0.01}), rec(13, update_rms={"w": 1.0})])
SKIPS_FIRST = ([rec(s, step_skipped=True, step_skipped_by="scaler", update_rms={"w": 5.0}) for s in range(10)]
               + [rec(s, update_rms={"w": 1.0}) for s in range(10, 22)])
PARTIAL_SKIP = ([rec(s, update_rms={"w": 1.0, "u": 1.0}) for s in range(12)]
                + [rec(12, step_skipped=True, step_skipped_by="scaler", step_skipped_part={"skipped": 1, "of": 2},
                       update_rms={"w": 0.01, "u": 1.0})])
CALLER_SKIP = ([rec(s, update_rms={"w": 1.0}) for s in range(12)]
               + [rec(12, step_skipped=True, step_skipped_by="caller", update_rms={"w": 0.01})])


def _lowest(name, step, ratio=1.0, baseline=1.0, update_rms=1.0):
    return {"name": name, "ratio": ratio, "step": step, "baseline": baseline, "update_rms": update_rms}


A7_CASES = {
    "full_skip": (FULL_SKIP, {"parameters": 1, "judged": 1, "lowest": [_lowest("w", 10)]}),
    "skips_first": (SKIPS_FIRST, {"parameters": 1, "judged": 1, "lowest": [_lowest("w", 20)]}),
    "partial_skip": (PARTIAL_SKIP, {"parameters": 2, "judged": 2, "lowest": [_lowest("u", 10), _lowest("w", 10)]}),
    "caller_skip": (CALLER_SKIP, {"parameters": 1, "judged": 1, "lowest": [_lowest("w", 10)]}),
}


@pytest.mark.parametrize("case", sorted(A7_CASES))
def test_a_skipped_update_is_neither_baseline_nor_lowest(case):
    records, expected = A7_CASES[case]
    assert evidence.update_trends(copy.deepcopy(records)) == expected
    assert evidence.build_evidence(copy.deepcopy(records), detectors="python")["update_trends"] == expected


def test_only_skipped_updates_give_no_block():
    records = [rec(s, step_skipped=True, update_rms={"w": 1.0}) for s in range(12)]
    assert evidence.update_trends(records) is None


@needs_server
@pytest.mark.parametrize("case", sorted(A7_CASES))
def test_update_trends_raw_rust_awaits_aj3(case):
    records, expected = A7_CASES[case]
    assert server_report(copy.deepcopy(records))["update_trends"] == expected


# --- A10 ------------------------------------------------------------------------------------------

RELOAD = ([rec(s, attempt=1, weight_norms={"w": 1.0}) for s in range(0, 60, 10)]
          + [rec(30, attempt=2, weight_norms={"w": 50.0}), rec(40, attempt=2, weight_norms={"w": 50.0}),
             rec(50, attempt=2, weight_norms={"w": 150.0})])
RELOAD_EXPECTED = {"checks": 6, "parameters": 1, "largest": [
    {"name": "w", "ratio": 3.0, "step": 50, "previous_step": 40, "before": 50.0, "after": 150.0}]}
RELOAD_ONLY = ([rec(s, attempt=1, weight_norms={"w": 1.0}) for s in range(0, 60, 10)]
               + [rec(s, attempt=2, weight_norms={"w": 50.0}) for s in range(30, 60, 10)])
RELOAD_ONLY_EXPECTED = {"checks": 6, "parameters": 1, "largest": [
    {"name": "w", "ratio": 1.0, "step": 10, "previous_step": 0, "before": 1.0, "after": 1.0}]}


def test_a_reload_resets_weight_jumps_and_a_real_jump_after_it_is_listed():
    assert evidence.build_evidence(copy.deepcopy(RELOAD), detectors="python")["weight_jumps"] == RELOAD_EXPECTED
    resolved = [r for r in RELOAD if r["attempt"] == 2 or r["step"] < 30]
    assert evidence.weight_jumps(copy.deepcopy(resolved)) == RELOAD_EXPECTED


def test_a_difference_across_a_reload_alone_is_not_a_jump():
    assert evidence.build_evidence(copy.deepcopy(RELOAD_ONLY), detectors="python")["weight_jumps"] == RELOAD_ONLY_EXPECTED


def test_weight_norm_trends_still_reads_across_the_reload():
    """Unchanged (the author, Nov 22): from the first check of the run to the last."""
    trends = evidence.build_evidence(copy.deepcopy(RELOAD), detectors="python")["weight_trends"]
    assert (trends["baseline_step"], trends["latest_step"]) == (0, 50)
    assert next(iter(trends["kinds"].values()))["highest"] == {"name": "w", "ratio": 150.0}


@needs_server
@pytest.mark.parametrize("records, expected", [(RELOAD, RELOAD_EXPECTED), (RELOAD_ONLY, RELOAD_ONLY_EXPECTED)],
                         ids=["reload_then_jump", "reload_only"])
def test_weight_jumps_raw_rust_awaits_aj3(records, expected):
    assert server_report(copy.deepcopy(records))["weight_jumps"] == expected
