"""Lane AR-3 (Nov 26, 2026): AQ5-D2, an integer too large for a float inside a JSON column.

The author's answer: "The Nov 22 reading of an oversized integer extended to `system_json`,
`data_json`, `probe_json`, `checkpoint_json` in Rust". Rust reads such a row's column with its
other keys, and the oversized value is unreadable where a number is wanted, as Python's `_finite`,
`_number` and `_count` read it. Before, one such integer lost Rust the whole column of that row.

`AGREE` holds the records both engines must now report alike, through the SDK's route and the raw
route (`test_generated_engines.records_differences`): AQ-5's four held cases, moved here from
`HELD`, and the same with valid siblings beside the oversized value.

Nov 27, 2026, the author: "bound Python's `data_checks._int` at int64, as its sibling `_count`
already is. No capture writes such a number." With it the same bound on `overflow_positions` in
`evidence.head_logit_trends`. Rust's `int` (`detect/data.rs`) reads a whole number that is an `i64`
and nothing else, so Python is brought to Rust: the three `_int` cases moved from `STILL_DIFFER`
to `AGREE`, and `BOUNDS` checks each integer `_int` reads that a rule reports from at `2**63 - 1`,
`2**63`, `-(2**63)`, `-(2**63) - 1` and `2**64`. Rust reads `overflow_positions` as a `u64`
(`HeadTrends::row`), so there the int64 bound leaves `2**63` to `2**64 - 1` read by Rust only
(`test_head_trends_past_int64_still_differ_on_the_raw_route`; the lane's report).

`STILL_DIFFER` holds what the author left as it is (Nov 27, 2026), each a strict expected failure:
where Python writes an oversized value as text, `str(10**400)` is its digits and Rust writes the
unreadable text. A capture never writes an integer there.

Needs `pyarrow` and the server binary (`TENSORQUILL_SERVER_BIN`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow")

import tensorquill  # noqa: E402
from tensorquill.data_checks import _int  # noqa: E402
from tensorquill.evidence import head_logit_trends  # noqa: E402
from tensorquill.server_detect import server_report  # noqa: E402
from test_generated_engines import BINARY, _record, gen, records_differences, run_differences  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")

BIG = 10**400
INT64_MAX = 2**63 - 1
INT64_MIN = -(2**63)


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
    "checkpoint: an oversized integer is no tensor name": [
        *(_attempt(1, s) for s in range(3)),
        _attempt(2, 2, checkpoint={"load": {
            "components": {"model": {}, "optimizer": {"entries": 3}, "rng": {}, "data": {}},
            "tensors": {"state": "compared", "missing": {"count": 2, "names": ["a", BIG]},
                        "unexpected": {"count": 1, "names": [BIG]}}}})],
    # AQ-5 found these agree, and they must go on agreeing: each fits a float.
    "the integers that fit a float, in all four columns": [
        _record(s, system={"cuda": {"ooms": n, "alloc_retries": 1}}, data={"batches": 1, "batch_hash": "v2:b", "epoch": n},
                probe=_probe([0.5], [n, 3.0]), checkpoint={"save": {"after_step": n}})
        for s, n in enumerate([2**63, 2**64, -(2**63) - 1, 1e308])],
    # Nov 27, 2026: `_int` bounded at int64, as Rust's `int` reads a whole number. Until then Python
    # kept each of these as an integer and the engines differed.
    "data: an oversized batches count": [_record(s, data={"batches": BIG, "batch_hash": "v2:aa"}) for s in range(4)],
    "data: a batches count past int64, which differed before D2 too": [
        _record(s, data={"batches": 2**64, "batch_hash": "v2:aa"}) for s in range(4)],
    "checkpoint: an oversized scheduler last_epoch": [
        *(_attempt(1, s) for s in range(3)),
        _attempt(2, 2, checkpoint={"load": {"components": {"model": {}, "optimizer": {"entries": 3}, "rng": {},
                                                           "scheduler": {"last_epoch": BIG}}}})],
}


# Records that read one integer at `n`, for each integer `_int` reads that a rule reports from or
# acts on. `epoch` is given as a pass and the pass after it, `lo` and `lo + 1`, with `n` one of them.
def _batches(n: int) -> list[dict[str, Any]]:
    return [_record(s, data={"batches": n, "batch_hash": "v2:aa"}) for s in range(4)]


def _passes(first: int, second: int, sampler_epoch: Any = None) -> list[dict[str, Any]]:
    out = []
    for s in range(6):
        data = {"batches": 1, "batch_hash": f"v2:{s % 3}", "epoch": first if s < 3 else second}
        if sampler_epoch is not None:
            data["sampler_epoch"] = sampler_epoch
        out.append(_record(s, data=data))
    return out


def _epoch(n: int) -> list[dict[str, Any]]:
    return _passes(n - 1, n) if n > 0 else _passes(n, n + 1)


def _last_epoch(n: int) -> list[dict[str, Any]]:
    return [*(_attempt(1, s) for s in range(3)),
            _attempt(2, 2, checkpoint={"load": {"components": {"model": {}, "optimizer": {"entries": 3}, "rng": {},
                                                               "scheduler": {"last_epoch": n}}}})]


def _entries_after(n: int) -> list[dict[str, Any]]:
    return [*(_attempt(1, s) for s in range(3)),
            _attempt(2, 2, checkpoint={"optimizer": {"entries_before": 0, "entries_after": n}})]


def _after_step(n: int) -> list[dict[str, Any]]:
    return [_attempt(1, 0, checkpoint={"save": {"after_step": n}}), *(_attempt(1, s) for s in range(1, 3)),
            _attempt(2, 2, data={"batches": 1, "batch_hash": "v2:q"})]


BOUNDARY = {"2**63 - 1": INT64_MAX, "2**63": 2**63, "-(2**63)": INT64_MIN, "-(2**63) - 1": INT64_MIN - 1,
            "2**64": 2**64}
FIELDS = {"batches": _batches, "epoch": _epoch, "sampler_epoch": lambda n: _passes(0, 1, n),
          "last_epoch": _last_epoch, "entries_after": _entries_after, "after_step": _after_step}
BOUNDS = {f"{field} at {name}": (make, n) for field, make in FIELDS.items() for name, n in BOUNDARY.items()}


STILL_DIFFER: dict[str, tuple[str, list[dict[str, Any]]]] = {
    "system: an oversized unavailable reason": (
        "Python writes str(10**400), its digits, as the reason; Rust writes the unreadable text. The "
        "author left it as it is (Nov 27, 2026): a capture never writes an integer there",
        [_record(5, system={"cuda": {"unavailable": BIG}})],
    ),
    "probe: an oversized error": (
        "Python writes str(10**400), its digits, in the failed probe's reason; Rust writes the unreadable "
        "text. The author left it as it is (Nov 27, 2026): a capture never writes an integer there",
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


@pytest.mark.parametrize("name", sorted(BOUNDS))
def test_both_engines_read_an_integer_only_within_int64(name, tmp_path):
    make, n = BOUNDS[name]
    found = records_differences(make(n), tmp_path)
    assert found == [], f"{name}:\n" + "\n".join(found[:10])


@pytest.mark.parametrize("n", list(BOUNDARY.values()) + [BIG], ids=list(BOUNDARY) + ["10**400"])
def test_both_engines_read_a_data_rank_only_within_int64(n, tmp_path):
    # Rank 1's share is `n`, or its rank (1) where `n` is not read: then it shares rank 0's data, and
    # the same batches are no duplicate.
    paths = {}
    for rank, share in ((0, 1), (1, n)):
        path = tmp_path / f"held.rank{rank}.jsonl"
        path.write_text("".join(gen.line({**_record(s, data={"batches": 1, "batch_hash": f"v2:{s}",
                                                              "data_rank": share}), "rank": rank})
                                for s in range(3)), encoding="utf-8")
        paths[rank] = path
    found = run_differences(paths)
    assert found == [], "\n".join(found[:10])


@pytest.mark.parametrize(("value", "read"), [
    (0, 0), (INT64_MAX, INT64_MAX), (INT64_MIN, INT64_MIN), (2**63, None), (INT64_MIN - 1, None),
    (2**64, None), (BIG, None), (-BIG, None), (True, None), (1.0, None), ("3", None), (None, None)],
    ids=["0", "2**63 - 1", "-(2**63)", "2**63", "-(2**63) - 1", "2**64", "10**400", "-10**400", "True", "1.0",
         "text", "None"])
def test_int_reads_a_whole_number_that_is_an_int64(value, read):
    assert _int(value) == read
    assert _int(value) is None or type(_int(value)) is int


def _head(n: Any) -> list[dict[str, Any]]:
    return [{"step": 3, "probe": {"head": {"mean": 1.0, "overflow_positions": n}}}]


@pytest.mark.parametrize("n", [1, INT64_MAX], ids=["1", "2**63 - 1"])
def test_head_trends_read_overflow_positions_within_int64(n):
    assert head_logit_trends(_head(n))["overflow_positions"] == {
        "probes": 1, "positions": n, "most": n, "first_step": 3, "latest_step": 3}


@pytest.mark.parametrize("n", [0, -1, 2**63, 2**64, BIG, True, 2.0],
                         ids=["0", "-1", "2**63", "2**64", "10**400", "True", "2.0"])
def test_head_trends_read_no_overflow_positions_outside_int64(n):
    assert "overflow_positions" not in head_logit_trends(_head(n))


def _raw_head_trends(records: list[dict[str, Any]], tmp_path: Path) -> tuple[Any, Any]:
    path = tmp_path / "held.rank0.jsonl"
    path.write_text("".join(gen.line(r) for r in records), encoding="utf-8")
    python = tensorquill.build_evidence(path).get("head_trends")
    rust = server_report(tensorquill.load_records(path, all_attempts=True), binary=BINARY).get("head_trends")
    return python, rust


@pytest.mark.parametrize("n", [1, INT64_MAX, 2**64, BIG], ids=["1", "2**63 - 1", "2**64", "10**400"])
def test_head_trends_within_int64_or_past_u64_agree_on_the_raw_route(n, tmp_path):
    python, rust = _raw_head_trends([_record(0, probe={"head": {"mean": 1.0, "overflow_positions": n}})], tmp_path)
    assert python == rust


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "Rust's HeadTrends::row reads overflow_positions as a u64, so 2**63 to 2**64 - 1 is read there and "
    "not by Python's int64 bound (the author's, Nov 27, 2026); the lane's report gives the two ways to close it"))
def test_head_trends_past_int64_still_differ_on_the_raw_route(tmp_path):
    python, rust = _raw_head_trends([_record(0, probe={"head": {"mean": 1.0, "overflow_positions": 2**63}})],
                                    tmp_path)
    assert python == rust
