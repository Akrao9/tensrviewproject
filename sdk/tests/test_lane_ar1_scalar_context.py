"""Lane AR-1 (wave AR, Nov 26, 2026): an incident's scalar context in both engines, AQ5-D4 and AQ5-D7.

- **D4.** The author: "Rust brought to Python: "before" leaves out the same step's earlier writing". An
  incident on a later writing of a step shows what Python's `scalar_context` shows: the scalars of the
  step's first writing, with the median of the records before that writing.
- **D7.** The author: "A refused entry of `scalars` is absent and its siblings are kept, in both engines".
  `WAVE_AJ_CONTRACT.md` section 3 changed with it: lane W3's whole-map rule is gone. The sender keeps the
  entries the column holds in the column and puts the refused ones aside in `extra_json` under
  `scalars`, as wave AJ's maps do; Python's scalar context leaves a refused entry out; Rust reads a row
  stored before the change (its column null, the whole map in `extra_json`) as Python reads it back.

Each route a record takes: the Python engine on a file, the raw Rust report (`server_report`, built by
the sender's own `_row`), the sender's rows and what a live server stores and reports, a row stored
before the change, and (with `TENSORQUILL_AR1_OLDER_SERVER` set to a server built before this
change) the new rows read by an old server. The two AQ5-D4 and two AQ5-D7 cases held as expected failures
in `test_generated_engines.py` pass here."""

from __future__ import annotations

import io
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")

import tensorquill  # noqa: E402
from tensorquill import evidence  # noqa: E402
from tensorquill.flight import _plain, flight_schema, record_to_row, row_to_record, rows_to_send  # noqa: E402
from tensorquill.server_detect import records_stream, server_report  # noqa: E402

from test_generated_engines import BINARY, records_differences  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
# Not named TENSORQUILL_SERVER_BIN_...: CI's server job fails on a skip whose reason names that variable,
# which is right for the binary the job builds and wrong for an older build that no job has.
BEFORE_ENV = "TENSORQUILL_AR1_OLDER_SERVER"
HUGE = 10**400


def _record(step: int, **fields: Any) -> dict[str, Any]:
    return {"schema_version": 1, "run_id": "ar1", "rank": 0, "attempt": 1, "step": step, **fields}


def _starved_twice() -> list[dict[str, Any]]:
    starved = {"r0": [256, 128, 127, 1]}
    last = _record(48, moe=starved, scalars={"loss": 2.0})
    return [_record(step, moe=starved) for step in range(48)] + [last, dict(last)]


def _back(record: dict[str, Any]) -> dict[str, Any]:
    """`record` as `row_to_record` gives it back: a step row reads `ts` and `grad_norm` as present."""
    return {"ts": None, "grad_norm": None, **record}


IMBALANCED = {"r0": [1, 1, 10_000, 1]}

# --- AQ5-D4 -------------------------------------------------------------------------------------------

D4_CASES = {
    # The two held in `test_generated_engines.py` (seeds 1863 and 41).
    "AQ5-D4 a MoE incident at a step written twice": [
        _record(26, scalars={"loss": 2.4}), _record(26, scalars={"loss": 2.4}, moe=IMBALANCED)],
    "AQ5-D4 a MoE incident at a step written twice, starved_experts": _starved_twice(),
    # The same with other values, each of which today's Rust read from the incident's own row.
    "the two writings log different values": [
        _record(24, scalars={"loss": 1.0}), _record(25, scalars={"loss": 2.0}),
        _record(26, scalars={"loss": 2.4}), _record(26, scalars={"loss": 3.0}, moe=IMBALANCED)],
    "only the first writing logs a scalar": [_record(26, scalars={"loss": 2.4}), _record(26, moe=IMBALANCED)],
    "only the second writing logs a scalar": [_record(26), _record(26, scalars={"loss": 3.0}, moe=IMBALANCED)],
    "a step written many times": (
        [_record(s, scalars={"loss": 5.0}) for s in range(10)] + [_record(10, scalars={"loss": 1.0})]
        + [_record(10, scalars={"loss": float(at)}) for at in range(2, 201)]
        + [_record(11, scalars={"loss": 9.0}, moe=IMBALANCED)]),
    "a step written many times, the incident on its last writing": (
        [_record(s, scalars={"loss": 5.0}) for s in range(10)] + [_record(10, scalars={"loss": 1.0})]
        + [_record(10, scalars={"loss": float(at)}) for at in range(2, 200)]
        + [_record(10, scalars={"loss": 200.0}, moe=IMBALANCED)]),
    "a gradient spike on a step's second writing": (
        [_record(s, grad_norm=1.0, scalars={"loss": 1.0 + s}) for s in range(30)]
        + [_record(30, grad_norm=1.0, scalars={"loss": 2.4}), _record(30, grad_norm=100.0, scalars={"loss": 3.0})]),
}


@pytest.mark.parametrize("name", sorted(D4_CASES))
def test_d4_both_engines_give_an_incident_on_a_later_writing_the_same_context(name, tmp_path):
    assert records_differences(D4_CASES[name], tmp_path) == []


def test_d4_the_context_is_the_first_writings_value_and_the_median_before_it():
    records = D4_CASES["the two writings log different values"]
    want = {"loss": {"value": 2.4, "median_before": 1.5}}
    python = evidence.build_evidence(records)
    rust = server_report(records)
    for incidents in (python["incidents"], rust["incidents"]):
        moe = next(i for i in incidents if i["kind"] == "moe_imbalance")
        assert moe["scalars"] == want, moe


# --- AQ5-D7: the engines ------------------------------------------------------------------------------

D7_CASES = {
    # The two held in `test_generated_engines.py` (seed 4082): the refused entry alone.
    "AQ5-D7 a scalars entry the column refuses, in an incident's context": [
        _record(115, scalars={"train/loss": -HUGE}), _record(115)],
    "AQ5-D7 a scalars entry the column refuses, a text": [_record(115, scalars={"train/loss": "text"}), _record(115)],
    # With siblings, which both engines keep.
    "a text beside a number": [_record(115, scalars={"train/loss": "text", "acc": 0.5}), _record(115)],
    "a number too large for a float beside a number": [
        _record(115, scalars={"train/loss": -HUGE, "acc": 0.5}), _record(115)],
    "a text and a bool beside a number": [
        _record(115, scalars={"flag": True, "train/loss": "text", "acc": 0.5}), _record(115)],
    "a text beside the sink's nan": [_record(115, scalars={"train/loss": "text", "acc": "nan"}), _record(115)],
    # Seed 3569, which AQ-5 filed under D7: a null, which the column holds as null and Rust's context
    # skips. Python leaves it out too (lane AR-1's choice: a null is no value, as in its median).
    "a null beside a number": [_record(115, scalars={"train/loss": None, "acc": 0.5}), _record(115)],
    "the records before the step hold a text too": (
        [_record(s, scalars={"acc": float(s), "train/loss": "text"}) for s in range(5)]
        + [_record(5, scalars={"acc": 9.0, "train/loss": [1]}), _record(5)]),
}


@pytest.mark.parametrize("name", sorted(D7_CASES))
def test_d7_both_engines_leave_a_refused_entry_out_and_keep_its_siblings(name, tmp_path):
    assert records_differences(D7_CASES[name], tmp_path) == []


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_d7_the_context_holds_the_sibling_and_not_the_refused_entry(engine):
    def context(records: list[dict[str, Any]]) -> Any:
        doc = evidence.build_evidence(records) if engine == "python" else server_report(records)
        return next(i for i in doc["incidents"] if i["kind"] == "repeated_steps").get("scalars")

    sibling = {"acc": {"value": 0.5, "median_before": None}}
    assert context(D7_CASES["a text beside a number"]) == sibling
    assert context(D7_CASES["a number too large for a float beside a number"]) == sibling
    assert context(D7_CASES["a text and a bool beside a number"]) == sibling
    assert context(D7_CASES["a null beside a number"]) == sibling
    assert context(D7_CASES["AQ5-D7 a scalars entry the column refuses, a text"]) is None
    assert context(D7_CASES["the records before the step hold a text too"]) == {
        "acc": {"value": 9.0, "median_before": 2.0}}


def test_d7_python_reads_what_the_column_holds_under_scalars():
    """`scalar_context` alone: a number and the sink's "nan" are shown as before; a text, a list, an
    object, a bool and a number too large for a float are absent, and so is a null (no value)."""
    records = [_record(0, scalars={"a": 1.0, "b": None, "c": "nan", "d": "NaN", "e": [1], "f": {}, "g": True,
                                   "h": HUGE, "i": 3})]
    assert evidence.scalar_context(records, 0) == {
        "a": {"value": 1.0, "median_before": None}, "c": {"value": "nan", "median_before": None},
        "i": {"value": 3, "median_before": None}}


# --- AQ5-D7: the sender's row and its read-back -------------------------------------------------------


def _pairs(column: Any) -> Any:
    return None if column is None else {k: _plain(v) for k, v in column}


def test_d7_a_refused_entry_travels_aside_and_its_siblings_stay_in_the_column():
    record = _record(3, scalars={"train/loss": "text", "acc": 0.5, "flag": True})
    row = record_to_row(record)
    assert _pairs(row["scalars"]) == {"acc": 0.5}
    assert json.loads(row["extra_json"]) == {"scalars": {"train/loss": "text", "flag": True}}
    assert row_to_record(row) == _back(record)


def test_d7_an_overflow_and_a_name_arrow_refuses_are_counted_in_moved_and_a_text_is_not():
    records = [_record(1, scalars={"loss": HUGE, "acc": 0.5}), _record(2, scalars={"l\ud800": 1.0, "acc": 0.5}),
               _record(3, scalars={"loss": "text", "acc": 0.5})]
    moved: list = []
    batch, steps, skipped = rows_to_send(records, flight_schema(), moved)
    assert steps == [1, 2, 3] and skipped == 0
    assert moved == [(1, "scalars"), (2, "scalars")]
    rows = batch.to_pylist()
    assert [_pairs(r["scalars"]) for r in rows] == [{"acc": 0.5}] * 3
    assert [json.loads(r["extra_json"]) for r in rows] == [
        {"scalars": {"loss": HUGE}}, {"scalars": {"l\ud800": 1.0}}, {"scalars": {"loss": "text"}}]
    assert [row_to_record(r) for r in rows] == [_back(r) for r in records]


@pytest.mark.parametrize("scalars", [
    {"loss": "text"},                      # no entry held: the whole map aside, the column null, as before
    {"loss": "text", "flag": True},
    {"loss": HUGE},
], ids=["a text", "a text and a bool", "an oversized integer"])
def test_d7_a_map_with_no_held_entry_travels_whole_as_before(scalars):
    record = _record(4, scalars=scalars)
    row = record_to_row(record)
    assert row["scalars"] is None and json.loads(row["extra_json"]) == {"scalars": scalars}
    assert row_to_record(row) == _back(record)


@pytest.mark.parametrize("scalars", [
    {"loss": 2.0, "acc": "nan"},
    {"loss": 2.0, "flag": True},
    {"flag": True},
    {"loss": 2**53 + 1, "acc": 0.5},
    {},
], ids=["numbers", "a bool", "a bool alone", "past 2**53", "empty"])
def test_d7_a_map_with_nothing_refused_but_a_bool_makes_the_row_it_made_before(scalars):
    """The rows lane W3 and wave AO made: a bool and an integer past 2**53 in `extra_json`, the rest in
    the column (an empty map when nothing else is logged)."""
    row = record_to_row(_record(4, scalars=scalars))
    assert _pairs(row["scalars"]) == {k: float(v) if isinstance(v, int) else v
                                      for k, v in scalars.items() if not isinstance(v, bool)}
    aside = {k: v for k, v in scalars.items() if isinstance(v, bool) or (isinstance(v, int) and v > 2**53)}
    assert (json.loads(row["extra_json"]) if row["extra_json"] else {}) == ({"scalars": aside} if aside else {})


# --- AQ5-D7: a row stored before this change, read by both engines -------------------------------------


def _old_row(record: dict[str, Any]) -> dict[str, Any]:
    """The row a sender built before this change: lane W3's whole-map rule (any refusal but a bool's
    moves the whole map to `extra_json`, the column null)."""
    row = record_to_row(record)
    extra = json.loads(row["extra_json"]) if row["extra_json"] else {}
    extra["scalars"] = record["scalars"]
    return {**row, "scalars": None, "extra_json": json.dumps(extra, sort_keys=True, separators=(",", ":"))}


def _stream(rows: list[dict[str, Any]]) -> bytes:
    schema = flight_schema()
    out = io.BytesIO()
    with pa.ipc.new_stream(out, schema) as writer:
        writer.write_batch(pa.RecordBatch.from_pylist(rows, schema=schema))
    return out.getvalue()


def _detect(binary: Any, stream: bytes) -> dict[str, Any]:
    done = subprocess.run([str(binary), "detect"], input=stream, capture_output=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


OLD_SHAPED = (
    [_record(s, scalars={"acc": float(s), "train/loss": "text"}) for s in range(4)]
    + [_record(4, scalars={"acc": 9.0, "train/loss": "text", "flag": True}), _record(4)]
)


def test_d7_a_row_stored_before_the_change_reads_the_same_in_both_engines():
    """Python reads the old row back whole (`row_to_record`) and leaves the text out; Rust, whose column
    is null, reads the held entries from `extra_json`. Both show the sibling, with its median before."""
    rows = [_old_row(r) if "scalars" in r else record_to_row(r) for r in OLD_SHAPED]
    assert all(r["scalars"] is None for r in rows)
    assert [row_to_record(r) for r in rows] == [_back(r) for r in OLD_SHAPED]
    want = {"acc": {"value": 9.0, "median_before": 1.5}}
    python = evidence.build_evidence([row_to_record(r) for r in rows])
    rust = _detect(BINARY, _stream(rows))
    for incidents in (python["incidents"], rust["incidents"]):
        assert next(i for i in incidents if i["kind"] == "repeated_steps")["scalars"] == want, incidents


def test_d7_an_old_row_with_an_oversized_integer_reads_the_same_in_both_engines():
    """`extra_json` holding `10**400` parses in Rust with the number read as unreadable
    (`detect/extra_json.rs`), so the sibling is kept there too."""
    records = [_record(4, scalars={"acc": 0.5, "train/loss": -HUGE}), _record(4)]
    rows = [_old_row(records[0]), record_to_row(records[1])]
    want = {"acc": {"value": 0.5, "median_before": None}}
    python = evidence.build_evidence([row_to_record(r) for r in rows])
    rust = _detect(BINARY, _stream(rows))
    for incidents in (python["incidents"], rust["incidents"]):
        assert next(i for i in incidents if i["kind"] == "repeated_steps")["scalars"] == want, incidents


@pytest.mark.skipif(not os.environ.get(BEFORE_ENV), reason=f"{BEFORE_ENV} names no server built before this change")
def test_d7_the_new_senders_rows_read_by_a_server_built_before_the_change():
    """A server built before this change reads the `scalars` column alone, so it sees the siblings the new
    sender keeps there and never the refused entry: what the new engines report. Before, it saw no scalars."""
    binary = Path(os.environ[BEFORE_ENV])
    records = D7_CASES["the records before the step hold a text too"]
    old = _detect(binary, records_stream(records))
    assert next(i for i in old["incidents"] if i["kind"] == "repeated_steps")["scalars"] == {
        "acc": {"value": 9.0, "median_before": 2.0}}


# --- AQ5-D7: the live server ---------------------------------------------------------------------------

SPIKE = ([_record(s, grad_norm=1.0, scalars={"acc": 0.4, "train/loss": "text"}) for s in range(30)]
         + [_record(30, grad_norm=100.0, scalars={"acc": 0.5, "train/loss": "text"})])


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    from test_lane_aj1_rust_paths import _Live

    server = _Live(tmp_path_factory.mktemp("ar1-store"))
    yield server
    server.close()


def test_d7_the_live_server_stores_the_siblings_and_reports_what_python_reports(live, tmp_path):
    report = live.report(SPIKE, "ar1-live", tmp_path)
    spike = next(i for i in report["incidents"] if i["kind"] == "grad_spike")
    assert spike["scalars"] == {"acc": {"value": 0.5, "median_before": 0.4}}
    stored = tensorquill.load_records(live.store / "runs" / "ar1-live")
    assert stored == [_back({**r, "run_id": "ar1-live"}) for r in SPIKE]
    python = evidence.build_evidence(stored)
    assert next(i for i in python["incidents"] if i["kind"] == "grad_spike")["scalars"] == spike["scalars"]
