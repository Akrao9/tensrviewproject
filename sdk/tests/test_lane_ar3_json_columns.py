"""Lane AR-3 (Nov 26, 2026): AQ5-D2, an integer too large for a float inside a JSON column.

The author's answer: "The Nov 22 reading of an oversized integer extended to `system_json`,
`data_json`, `probe_json`, `checkpoint_json` in Rust". Rust reads such a row's column with its
other keys, and the oversized value is unreadable where a number is wanted, as Python's `_finite`,
`_number` and `_count` read it. Before, one such integer lost Rust the whole column of that row.

`AGREE` holds the records both engines must now report alike, through the SDK's route and the raw
route (`test_generated_engines.records_differences`): AQ-5's four held cases, moved here from
`HELD`, and the same with valid siblings beside the oversized value.

`STILL_DIFFER` holds what the Rust reading cannot settle, each a strict expected failure: Python
keeps the integer as an integer where Rust's columns hold none, in `data_checks._int` (no bound at
int64, unlike `evidence_system._count`) and where Python writes a value as text (`str(10**400)` is
its digits; Rust writes the unreadable text). The lane's report proposes the Python change.

Needs `pyarrow` and the server binary (`TENSORQUILL_SERVER_BIN`).
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pyarrow")

from test_generated_engines import BINARY, _record, records_differences  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")

BIG = 10**400


def _attempt(attempt: int, step: int, **fields: Any) -> dict[str, Any]:
    return {**_record(step, **fields), "attempt": attempt}


def _probe(share: list[float], logits: list[Any]) -> dict[str, Any]:
    return {"attention": {"blk1": {"entropy_share": share, "max_logit": logits}}}


AGREE: dict[str, list[dict[str, Any]]] = {
    # AQ-5's four held cases. Seeds 93, 340, 731, 750, 1490.
    "AQ5-D2 system with an oversized integer": [_record(39, system={"cuda": {"ooms": BIG}})],
    # Seeds 366, 729, 1352, 1413.
    "AQ5-D2 data with an oversized integer": [_record(37, data={"sampler_epoch": BIG})],
    # Seeds 482, 760, 1424; at 1371 Rust missed an attention_collapse incident Python reports.
    "AQ5-D2 probe with an oversized integer": [
        _record(0, probe={"attention": {"blk1": {"max_logit": [3.943363681, -BIG, 3.956723686]}}})],
    "AQ5-D2 checkpoint with an oversized integer": [_record(0, checkpoint={"save": {"after_step": BIG}})],
    # With valid siblings: Python's incident from the sibling, the oversized value none.
    "system: allocator_retries beside an oversized ooms": [
        _record(39, system={"cuda": {"ooms": BIG, "alloc_retries": 3}})],
    "system: a throttle reason list with an oversized member": [
        _record(5, system={"nvml": {"throttle": ["sw_thermal", BIG], "temperature_c": BIG, "power_w": 250.0}})],
    "data: a stall beside an oversized sampler_epoch": [
        _record(s, data={"batches": 1, "batch_hash": "v2:aa", "sampler_epoch": BIG}) for s in range(4)],
    "data: an oversized batch_hash is no hash": [_record(s, data={"batches": 1, "batch_hash": BIG}) for s in range(4)],
    "probe: an attention collapse beside oversized logits": [
        _record(s, probe=_probe([0.9, 0.9] if s < 6 else [0.1, 0.1], [3.9, -BIG, 3.96])) for s in range(10)],
    "probe: head numbers beside an oversized one": [
        _record(s, probe={"head": {"norm": BIG, "mean": 1.5 + s}}) for s in range(3)],
    "checkpoint: a resume's load manifest beside an oversized integer": [
        *(_attempt(1, s) for s in range(3)),
        _attempt(2, 2, checkpoint={"load": {"components": {"model": {}, "optimizer": {"entries": 0}, "rng": {}},
                                            "note": BIG}})],
    # AQ-5 found these agree, and they must go on agreeing: each fits a float.
    "the integers that fit a float, in all four columns": [
        _record(s, system={"cuda": {"ooms": n, "alloc_retries": 1}}, data={"batches": 1, "batch_hash": "v2:b", "epoch": n},
                probe=_probe([0.5], [n, 3.0]), checkpoint={"save": {"after_step": n}})
        for s, n in enumerate([2**63, 2**64, -(2**63) - 1, 1e308])],
}


STILL_DIFFER: dict[str, tuple[str, list[dict[str, Any]]]] = {
    "data: an oversized batches count": (
        "Python's data_checks._int keeps 10**400 as a count, so the rows drew and stall; Rust's int "
        "reads no integer outside i64, so they drew nothing",
        [_record(s, data={"batches": BIG, "batch_hash": "v2:aa"}) for s in range(4)],
    ),
    "checkpoint: an oversized scheduler last_epoch": (
        "Python's data_checks._int keeps 10**400 and reports the scheduler's position with its digits; "
        "Rust reads no last_epoch",
        [*(_attempt(1, s) for s in range(3)),
         _attempt(2, 2, checkpoint={"load": {"components": {"model": {}, "optimizer": {"entries": 3}, "rng": {},
                                                            "scheduler": {"last_epoch": BIG}}}})],
    ),
    "system: an oversized unavailable reason": (
        "Python writes str(10**400), its digits, as the reason; Rust writes the unreadable text",
        [_record(5, system={"cuda": {"unavailable": BIG}})],
    ),
    "probe: an oversized error": (
        "Python writes str(10**400), its digits, in the failed probe's reason; Rust writes the unreadable text",
        [_record(0, probe={"error": BIG})],
    ),
}


@pytest.mark.parametrize("name", sorted(AGREE))
def test_both_engines_read_a_json_column_with_an_oversized_integer_alike(name, tmp_path):
    found = records_differences(AGREE[name], tmp_path)
    assert found == [], f"{name}:\n" + "\n".join(found[:10])


@pytest.mark.parametrize("name", [pytest.param(name, marks=pytest.mark.xfail(strict=True, raises=AssertionError,
                                                                             reason=STILL_DIFFER[name][0]))
                                  for name in sorted(STILL_DIFFER)])
def test_what_the_rust_reading_cannot_settle_still_differs(name, tmp_path):
    reason, records = STILL_DIFFER[name]
    found = records_differences(records, tmp_path)
    assert found == [], f"{reason}:\n" + "\n".join(found[:10])
