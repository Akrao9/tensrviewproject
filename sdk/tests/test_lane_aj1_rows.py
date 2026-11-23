"""Lane AJ-1, A3: a map entry its column cannot hold leaves the typed column alone (WAVE_AJ_CONTRACT.md,
section 1). The rows of the shared cases (`fixtures/wave_aj/cases.json`): what the typed column and
`extra_json` hold, and `row_to_record(record_to_row(r)) == r`. Fail-before: on 212205d one refused entry
sent the whole map to `extra_json`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from tensorquill.flight import _FIELDS, _plain, flight_schema, record_to_row, row_to_record, rows_to_send  # noqa: E402

CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_aj" / "cases.json").read_text())["cases"]
ROW_CASES = [c for c in CASES if "row" in c["paths"]]
SENDER_CASES = [c for c in CASES if "row_sender" in c["paths"]]


def _at(case: dict, step: int) -> dict:
    return next(r for r in case["records"] if r["step"] == step)


def _check_row(row: dict, want: dict) -> None:
    for key, typed in want["typed"].items():
        column = _FIELDS[key][0]
        if typed is None:
            assert row[column] is None, (key, row[column])
        else:
            assert {k: _plain(v) for k, v in row[column]} == typed, (key, row[column])
    if want["extra"] is None:
        assert row["extra_json"] is None, row["extra_json"]
    else:
        assert json.loads(row["extra_json"]) == want["extra"]


@pytest.mark.parametrize("case", ROW_CASES, ids=[c["name"] for c in ROW_CASES])
def test_the_typed_column_keeps_every_held_entry(case):
    want = case["row"]
    _check_row(record_to_row(_at(case, want["step"])), want)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_every_record_reads_back_as_it_was_sent(case):
    for record in case["records"] + case["clean_records"]:
        if "row" in case["paths"]:
            assert row_to_record(record_to_row(record)) == record
        batch, steps, skipped = rows_to_send([record], flight_schema())
        assert steps == [record["step"]] and skipped == 0
        assert row_to_record(batch.to_pylist()[0]) == record


@pytest.mark.parametrize("case", SENDER_CASES, ids=[c["name"] for c in SENDER_CASES])
def test_an_entry_arrow_refuses_travels_alone_and_moved_names_its_key(case):
    want = case["row"]
    moved: list = []
    batch, steps, skipped = rows_to_send([_at(case, want["step"])], flight_schema(), moved)
    assert skipped == 0 and steps == [want["step"]]
    _check_row(batch.to_pylist()[0], want)
    assert moved == [(want["step"], key) for key in want["moved"]]


def test_a_clean_records_row_has_nothing_in_extra_json():
    """The control of every case: with the refused entries removed, nothing travels in `extra_json`."""
    for case in CASES:
        for record in case["clean_records"]:
            assert record_to_row(record)["extra_json"] is None, (case["name"], record["step"])


def test_a_name_arrow_cannot_encode_is_refused_alone_on_the_senders_path():
    """Lane AD4's case with a sibling: the name with a lone surrogate travels in `extra_json`, the other
    parameter stays in the column, and `moved` names `grad_norms` once."""
    record = {"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 1, "step": 5, "ts": 5.0, "grad_norm": 1.0,
              "grad_norms": {"w\ud800": 1.0, "b": "nan"}}
    moved: list = []
    batch, steps, skipped = rows_to_send([record], flight_schema(), moved)
    row = batch.to_pylist()[0]
    assert steps == [5] and skipped == 0 and moved == [(5, "grad_norms")]
    assert {k: _plain(v) for k, v in row["grad_norms"]} == {"b": "nan"}
    assert json.loads(row["extra_json"]) == {"grad_norms": {"w\ud800": 1.0}}
    assert row_to_record(row) == record


def test_none_under_a_list_map_is_refused_alone():
    record = {"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 1, "step": 0, "ts": 0.0, "grad_norm": 1.0,
              "moe": {"good": [3, 1], "gone": None}, "replica_spread": {"w": [0.0, None], "x": None}}
    row = record_to_row(record)
    assert dict(row["moe_counts"]) == {"good": [3, 1]}
    assert dict(row["replica_spread"]) == {"w": [0.0, None]}
    assert json.loads(row["extra_json"]) == {"moe": {"gone": None}, "replica_spread": {"x": None}}
    assert row_to_record(row) == record
