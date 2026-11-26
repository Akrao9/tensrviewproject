"""Lane AP-1 (wave AP, AO-01): the data and resume detectors judge integers near the ends of int64 alike.

Python's integers do not overflow. Three sums in the Rust detectors (`server/src/detect/data.rs`) did
in `i64`: a debug build panicked, and a release build wrapped to another answer.

- An epoch of `2**63 - 1` followed by `-(2**63)`: `epoch + 1` wrapped, so the two read as one epoch
  after the other and the order as repeated.
- A resume at step `2**63 - 1`: the steps after it were looked up from `resumed_at + 1`, which wrapped,
  so step 0 was taken for the nearest step that drew the batch.
- A scheduler `last_epoch` more than `2**63 - 1` from the resume step: the distance overflowed, so the
  position finding was lost.

Each record is judged by both engines through `build_evidence` (`detectors="python"` and `"rust"`), and
Python's answer is also asserted outright, so two engines wrong together fail. The binary is
`TENSORQUILL_SERVER_BIN` (or one beside the interpreter), built from this tree; without it, or without
pyarrow, the engine comparisons are skipped. The Rust side's own tests are the `wave_ap_` tests in
`server/src/detect/data.rs`, which also cover the negative steps only a store written before wave AG
holds.
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest

from tensorquill import server_detect
from tensorquill.evidence import build_evidence

BINARY = server_detect.find_binary()

needs_rust = [
    pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow is not installed"),
    pytest.mark.skipif(BINARY is None or not os.path.isfile(BINARY),
                       reason="tensorquill-server is not built (TENSORQUILL_SERVER_BIN)"),
]

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def _row(attempt: int, step: int, **fields) -> dict:
    return {"run_id": "r", "rank": 0, "attempt": attempt, "step": step, **fields}


def _two_epochs(first: int, second: int) -> list[dict]:
    return [_row(1, step, data={"batches": 1, "shuffle": True, "batch_hash": f"v2:h{step % 3}",
                                "epoch": first if step < 3 else second})
            for step in range(6)]


def _resume_at_max() -> list[dict]:
    drew = {"batches": 1, "batch_hash": "v2:aaaa"}
    return [_row(1, 0, data=drew), _row(1, 10, data=drew), _row(2, INT64_MAX, data=drew)]


def _scheduler(resumed_at: int, last_epoch: int) -> list[dict]:
    load = {"load": {"components": {"model": {}, "optimizer": {"entries": 1}, "rng": {}, "data": {},
                                    "scheduler": {"last_epoch": last_epoch}}}}
    return [_row(1, 0, lr=[1e-3]), _row(1, 1, lr=[5e-4]), _row(2, resumed_at, lr=[5e-4], checkpoint=load)]


CASES = {
    "epoch_max_then_min": _two_epochs(INT64_MAX, INT64_MIN),
    "epoch_min_then_next": _two_epochs(INT64_MIN, INT64_MIN + 1),
    "resume_at_max": _resume_at_max(),
    "scheduler_min_plus_1_at_1": _scheduler(1, INT64_MIN + 1),
    "scheduler_min_at_1": _scheduler(1, INT64_MIN),
    "scheduler_min_at_max": _scheduler(INT64_MAX, INT64_MIN),
    "scheduler_max_at_2": _scheduler(2, INT64_MAX),
    "scheduler_within_slack": _scheduler(1, 2),
}


def _verdict(evidence: dict) -> dict:
    """What the findings change: the run's state, every incident without its logged scalars, and the
    `not_checked` entries."""
    return {
        "state": evidence["run_state"]["state"],
        "incidents": [{key: value for key, value in incident.items() if key != "scalars"}
                      for incident in evidence["incidents"]],
        "not_checked": sorted(json.dumps(entry, sort_keys=True) for entry in evidence.get("not_checked", [])),
    }


def _python(name: str) -> dict:
    return _verdict(build_evidence(CASES[name], detectors="python"))


def _of_kind(verdict: dict, kind: str) -> list[dict]:
    return [incident for incident in verdict["incidents"] if incident["kind"] == kind]


@needs_rust[0]
@needs_rust[1]
@pytest.mark.parametrize("name", sorted(CASES))
def test_both_engines_judge_alike(name: str) -> None:
    rust = _verdict(build_evidence(CASES[name], detectors="rust", server_binary=BINARY))
    assert rust == _python(name)


def test_python_finds_no_epoch_after_int64_max() -> None:
    verdict = _python("epoch_max_then_min")
    assert _of_kind(verdict, "epoch_order_repeated") == []
    assert verdict["state"] == "healthy"
    found = _of_kind(_python("epoch_min_then_next"), "epoch_order_repeated")
    assert [(i["epoch"], i["previous_epoch"], i["batches_compared"]) for i in found] == [
        (INT64_MIN + 1, INT64_MIN, 3)]


def test_python_matches_the_resume_at_int64_max_with_step_10() -> None:
    (incident,) = _of_kind(_python("resume_at_max"), "resume_data_mismatch")
    (change,) = incident["changes"]
    assert (change["finding"], change["matched_step"], change["batches_apart"]) == (
        "repeated", 10, INT64_MAX - 10)


@pytest.mark.parametrize("name", ["scheduler_min_plus_1_at_1", "scheduler_min_at_1", "scheduler_min_at_max",
                                  "scheduler_max_at_2"])
def test_python_finds_the_scheduler_out_of_place(name: str) -> None:
    (incident,) = _of_kind(_python(name), "resume_incomplete")
    last_epoch = CASES[name][2]["checkpoint"]["load"]["components"]["scheduler"]["last_epoch"]
    assert incident["changes"][0]["findings"] == [
        {"component": "scheduler", "finding": "position", "last_epoch": last_epoch}]
    assert _of_kind(_python("scheduler_within_slack"), "resume_incomplete") == []
