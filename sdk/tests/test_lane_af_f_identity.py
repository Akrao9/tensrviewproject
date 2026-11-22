"""Lane F of wave AF: `tensorquill.identity.check`, by the table "What is valid" of
`WAVE_AF_CONTRACT.md` section 1, on every section 1 case of `fixtures/wave_af/cases.json`."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tensorquill import identity
from tensorquill.identity import MalformedRecordError, check

DOCUMENT = json.loads((Path(__file__).parent / "fixtures" / "wave_af" / "cases.json").read_text())
SECTION_1 = [(name, body) for name, body in DOCUMENT["cases"].items() if body["section"] == 1]
GOOD = {"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 0, "step": 0, "grad_norm": 1.0}


@pytest.mark.parametrize("name,body", SECTION_1, ids=[name for name, _ in SECTION_1])
def test_every_section_1_case_at_the_check(name, body):
    record = body["given"]["record"]
    before = json.dumps(record, sort_keys=True)
    expected = body["expected"]["python"]
    if expected.get("reads"):
        assert check(record, "here") is None
    else:
        with pytest.raises(MalformedRecordError) as raised:
            check(record, "here")
        error = raised.value
        assert isinstance(error, ValueError)
        assert error.field == expected["field"] == body["field"]
        assert error.where == "here"
        assert body["field"] in str(error) and "here" in str(error)
    assert json.dumps(record, sort_keys=True) == before  # nothing changed, nothing filled in


def test_the_error_says_the_field_what_was_found_what_is_read_and_where():
    with pytest.raises(MalformedRecordError) as raised:
        check({**GOOD, "rank": True}, identity.in_file("one.jsonl", 3))
    assert str(raised.value) == "rank is true at one.jsonl, line 3: tensorquill reads an integer of 0 or more"
    assert raised.value.value is True and raised.value.reason == "tensorquill reads an integer of 0 or more"
    with pytest.raises(MalformedRecordError) as raised:
        check({**GOOD, "schema_version": 99}, identity.in_part("part-0-9.parquet", 4))
    assert str(raised.value) == ("schema_version is 99 at part-0-9.parquet, row 4: "
                                 "this tensorquill reads schema versions 0 and 1")
    without_step = dict(GOOD)
    del without_step["step"]
    with pytest.raises(MalformedRecordError) as raised:
        check(without_step, identity.in_list(2))
    assert raised.value.value is MalformedRecordError.ABSENT
    assert str(raised.value).startswith("step is absent at record 2 of the records passed: ")
    with pytest.raises(MalformedRecordError) as raised:
        check({**GOOD, "attempt": "1"})
    assert str(raised.value) == 'attempt is "1": tensorquill reads an integer of 0 or more'
    assert raised.value.where is None


def test_the_first_field_in_the_fixed_order_is_named():
    with pytest.raises(MalformedRecordError) as raised:
        check({"step": None, "attempt": -1, "rank": 1.0, "schema_version": "1"})
    assert raised.value.field == "schema_version"
    with pytest.raises(MalformedRecordError) as raised:
        check({"step": None, "attempt": -1, "rank": 1.0})
    assert raised.value.field == "rank"


def test_no_upper_bound_and_run_id_is_not_looked_at():
    assert check({**GOOD, "step": 2**70, "rank": 2**40, "attempt": 2**63}) is None
    assert check({"step": 5, "run_id": None}) is None
    assert check({"step": 0, "schema_version": 0}) is None


def test_the_newest_schema_version_written_is_read():
    from tensorquill import flight

    assert flight.SCHEMA_VERSION in identity.SCHEMA_VERSIONS


def test_importing_it_loads_neither_torch_nor_pyarrow():
    """The module on its own: the package's `__init__` imports the capture, which imports torch."""
    code = ("import importlib.util, sys; spec = importlib.util.spec_from_file_location('identity', sys.argv[1]); "
            "spec.loader.exec_module(importlib.util.module_from_spec(spec)); "
            "print(sorted(m for m in ('torch', 'pyarrow', 'numpy') if m in sys.modules))")
    done = subprocess.run([sys.executable, "-c", code, identity.__file__], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"


# --- Item 2: the row for the Rust detectors and for the sender (the page's boundary, lane F) -------

import hashlib  # noqa: E402

from tensorquill import flight, server_detect  # noqa: E402

pytest.importorskip("pyarrow")

NEW_1 = [(name, body) for name, body in SECTION_1 if body["status"] == "new"]
PINS_1 = [(name, body) for name, body in SECTION_1 if body["status"] == "pin"]
STREAMS = {"records_stream": server_detect.records_stream, "run_stream": server_detect.run_stream}


@pytest.mark.parametrize("name,body", NEW_1, ids=[name for name, _ in NEW_1])
def test_record_to_row_refuses_a_new_case_with_the_named_error(name, body):
    with pytest.raises(MalformedRecordError) as raised:
        flight.record_to_row(body["given"]["record"])
    assert raised.value.field == body["field"]


@pytest.mark.parametrize("stream", sorted(STREAMS))
@pytest.mark.parametrize("name,body", NEW_1, ids=[name for name, _ in NEW_1])
def test_a_stream_refuses_a_new_case_with_the_named_error(name, body, stream):
    with pytest.raises(MalformedRecordError) as raised:
        STREAMS[stream]([GOOD, body["given"]["record"]])
    assert raised.value.field == body["field"]
    assert raised.value.where == identity.in_list(1)


@pytest.mark.parametrize("stream", sorted(STREAMS))
@pytest.mark.parametrize("name,body", PINS_1, ids=[name for name, _ in PINS_1])
def test_a_pinned_case_still_makes_a_stream(name, body, stream):
    import pyarrow as pa

    table = pa.ipc.open_stream(STREAMS[stream]([body["given"]["record"]])).read_all()
    assert table.num_rows == 1


@pytest.mark.parametrize("name,body", PINS_1, ids=[name for name, _ in PINS_1])
def test_record_to_row_reads_a_pinned_case_as_on_the_base(name, body):
    """An absent `schema_version`, `rank` or `attempt` makes a row. `run_id` is left as it is: the
    base's `record_to_row` raises a KeyError or TypeError on a missing or non-text run ID (its
    callers fill one in), and it still does, never the new error."""
    record = body["given"]["record"]
    if body["field"] == "run_id":
        with pytest.raises((KeyError, TypeError)) as raised:
            flight.record_to_row(record)
        assert not isinstance(raised.value, MalformedRecordError)
    else:
        row = flight.record_to_row(record)
        assert (row["schema_version"], row["rank"], row["attempt"], row["step"]) == (
            record.get("schema_version", 0), record.get("rank", 0), record.get("attempt", 0), record["step"])


def test_sdk06_a_refusal_leaves_moved_as_it_was_passed():
    """On the base `moved` held `grad_norm` after the call raised on `rank` (SDK-06)."""
    moved = ["kept"]
    with pytest.raises(MalformedRecordError) as raised:
        flight.record_to_row({**GOOD, "grad_norm": 10**400, "rank": True}, None, moved)
    assert raised.value.field == "rank"
    assert moved == ["kept"]
    moved = []
    flight.record_to_row({**GOOD, "grad_norm": 10**400}, None, moved)
    assert moved == ["grad_norm"]  # a valid record still says what it moved


def test_the_stream_names_the_place_in_the_order_the_records_were_passed():
    """`stream_order` sorts by rank, attempt and step; the error names the caller's position."""
    records = [{**GOOD, "step": 9}, {**GOOD, "step": 8}, {**GOOD, "step": 1, "attempt": "1"}, {**GOOD, "step": 0}]
    for stream in STREAMS.values():
        with pytest.raises(MalformedRecordError) as raised:
            stream(records)
        assert raised.value.where == "record 2 of the records passed"


# Valid records of many shapes: identity fields absent, moved values, a logged bool, nested JSON.
VALID = [
    {"run_id": "v", "step": 0, "ts": 1.0, "grad_norm": None},
    {"schema_version": 0, "run_id": "v", "step": 1, "grad_norm": 2.5, "lr": [0.1, 0.2]},
    {"schema_version": 1, "run_id": "v", "rank": 0, "step": 2, "grad_norm": 10**400, "scalars": {"loss": 1.0, "flag": True}},
    {"schema_version": 1, "run_id": "v", "rank": 0, "attempt": 1, "step": 3, "layers": {"b": {"x": 1}, "a": {"x": "nan"}},
     "moe": {"r": [1, 2]}, "moe_top_k": {"r": 2**40}, "unknown": [1, None]},
    {"schema_version": 1, "run_id": "v", "rank": 0, "attempt": 1, "step": 4, "step_skipped": True, "data": {"h": "x"}},
]
# sha256 of the rows `record_to_row` made of VALID on the base (1ca26d4), with and without the schema.
BASE_ROWS = "cb6c98b556d0d49ef742ec9f8e9ee605ac9393b8934c76194139108554c8d954"
BASE_ROWS_SCHEMA = "631f74a7c8c31069a728894f5c85e054b217247b9bf550a45515643960728ee1"


def _rows_digest(schema) -> str:
    out = []
    for record in VALID:
        moved: list[str] = []
        out.append([flight.record_to_row(record, schema, moved), moved])
    return hashlib.sha256(json.dumps(out, sort_keys=True, default=repr).encode()).hexdigest()


def test_a_valid_record_s_row_is_what_it_is_on_the_base():
    assert _rows_digest(None) == BASE_ROWS
    assert _rows_digest(flight.flight_schema()) == BASE_ROWS_SCHEMA



@pytest.mark.parametrize("stream", sorted(STREAMS))
def test_a_valid_stream_is_byte_for_byte_the_stream_of_the_base_rows(stream):
    """The stream is the base's construction over the rows: built here from `record_to_row`'s rows
    in `stream_order`, as the base wrote it."""
    import io

    import pyarrow as pa

    # Without `moe_top_k` past int32: the streams build rows without the schema's check, so Arrow
    # refuses that value there with ArrowInvalid, on the base as after (not this page's).
    third = {k: v for k, v in VALID[3].items() if k != "moe_top_k"}
    given = [third, VALID[0], VALID[4], {k: v for k, v in VALID[1].items() if k != "run_id"}, VALID[2]]
    rows = [flight.record_to_row(r if isinstance(r.get("run_id"), str) else {**r, "run_id": ""})
            for r in server_detect.stream_order(given)]
    schema = flight.flight_schema()
    out = io.BytesIO()
    with pa.ipc.new_stream(out, schema) as writer:
        writer.write_batch(pa.RecordBatch.from_pylist(rows, schema=schema))
    assert STREAMS[stream](given) == out.getvalue()
