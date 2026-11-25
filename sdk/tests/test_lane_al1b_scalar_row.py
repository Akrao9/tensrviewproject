"""Wave AL step 2, lane AL-1': the scalar-only record's row (`WAVE_AL_CONTRACT.md` sections 3, 4.4 and 4.5).

- The row of case `scalar_record_keys`: `seq` set, `record_version` 2, row `schema_version` 2, `scalars`
  in the `scalars` column under wave W3's rule, nothing in `extra_json` (`record` comes back from `seq`).
- `row_to_record(record_to_row(r)) == r` for every record of every section 4 case, scalar-only and step
  records alike, in a plain row and through a real Arrow batch; a step row's `seq` is null.
- A store holding `scalars-<first_seq:08>-<last_seq:08>.parquet` objects beside its parts (section 4.5's
  names; lane AL-2' writes them, so the store here is built with pyarrow) is read back record for record.

`identity.check` takes record version 2 only with lane AL-4''s change (`identity.py`, not this lane's).
Until the trees are joined, the tests that need it take the `reads_version_2` fixture, which widens
`identity.SCHEMA_VERSIONS` for the test when 2 is not in it and does nothing once it is. The tests that wait
for the joined tree are strict expected failures that name it. Tests that need pyarrow skip one by one
without it (CI's plain jobs have none); none needs the server binary or git history."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tensorquill import flight, identity

CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_al" / "cases.json").read_text(encoding="utf-8"))["cases"]
SECTION_4 = [c for c in CASES if c["section"] == 4]
KEYS = next(c for c in SECTION_4 if c["name"] == "scalar_record_keys")
JOINED = ("the joined tree: identity.check takes record version 2 with lane AL-4''s change to identity.py "
          "(contract section 4.6), not in this lane's tree")
JOINED_LOADER = ("the joined tree: evidence.load_records (lane AL-4''s file) reads a store's scalars-* objects "
                 "through flight.load_scalar_records")


def _scalar_record() -> dict:
    return copy.deepcopy(next(r for r in KEYS["records"] if r.get("record") == "scalars"))


def _is_scalar(record: dict) -> bool:
    return record.get("record") == "scalars"


@pytest.fixture
def reads_version_2(monkeypatch):
    """`identity.check` taking record version 2, as lane AL-4' makes it (section 4.6). A no-op on a tree
    where it already does."""
    if 2 not in identity.SCHEMA_VERSIONS:
        monkeypatch.setattr(identity, "SCHEMA_VERSIONS", (*identity.SCHEMA_VERSIONS, 2))


# --- (a) The row of `scalar_record_keys` ---------------------------------------------------------


def _summary(row: dict) -> dict:
    return {"seq": row["seq"], "record_version": row["record_version"], "schema_version": row["schema_version"],
            "extra": json.loads(row["extra_json"]) if row.get("extra_json") else None}


def test_the_case_s_record_has_the_nine_keys_and_its_row_is_the_case_s():
    record = _scalar_record()
    assert list(record) == KEYS["expect"]["keys"]
    row = flight._row(copy.deepcopy(record))  # the sender's row (`rows_to_send`), with no identity check
    want = {k: v for k, v in KEYS["row"].items() if k != "round_trip"}
    assert _summary(row) == want == {"seq": 0, "record_version": 2, "schema_version": 2, "extra": None}
    assert row["scalars"] == [("eval_loss", 1.8)]
    assert (row["run_id"], row["rank"], row["attempt"], row["step"], row["ts"]) == ("al_sc", 0, 1, 500, 100.0)
    assert row["grad_norm"] is None and row["layers"] is None


def test_record_to_row_gives_the_case_s_row_where_version_2_is_read(reads_version_2):
    record = _scalar_record()
    row = flight.record_to_row(copy.deepcopy(record))
    assert _summary(row) == {"seq": 0, "record_version": 2, "schema_version": 2, "extra": None}
    assert flight.row_to_record(row) == record


@pytest.mark.xfail(strict=True, raises=identity.MalformedRecordError, reason=JOINED)
def test_record_to_row_takes_the_scalar_only_record():
    record = _scalar_record()
    assert _summary(flight.record_to_row(copy.deepcopy(record)))["seq"] == 0


def test_the_schema_has_seq_as_a_nullable_int64_before_record_version():
    pa = pytest.importorskip("pyarrow")
    schema = flight.flight_schema()
    field = schema.field("seq")
    assert field.type == pa.int64() and field.nullable
    names = schema.names
    assert names.index("seq") + 1 == names.index("record_version") and names[-1] == "extra_json"


def test_scalars_follow_wave_w3_s_rule_in_a_scalar_only_row():
    """A logged bool is left out of the column, as on a step row; any other value the column cannot hold
    moves the whole map to `extra_json`, and the row reads back as the record."""
    record = _scalar_record()
    step = {k: v for k, v in record.items() if k not in ("record", "seq")} | {"schema_version": 1}
    with_bool = {**record, "scalars": {"eval_loss": 1.8, "flag": True}}
    assert flight._row(with_bool)["scalars"] == flight._row({**step, "scalars": with_bool["scalars"]})["scalars"] \
        == [("eval_loss", 1.8)]
    with_text = {**record, "scalars": {"eval_loss": 1.8, "note": "high"}}
    row = flight._row(with_text)
    assert row["scalars"] is None and json.loads(row["extra_json"]) == {"scalars": with_text["scalars"]}
    assert row["seq"] == 0
    assert flight.row_to_record(row) == with_text
    non_finite = {**record, "scalars": {"eval_loss": "nan", "eval_ppl": "inf"}}
    assert flight.row_to_record(flight._row(non_finite)) == non_finite


# --- (b) The round trip -------------------------------------------------------------------------


def _through_arrow(records: list[dict]) -> list[dict]:
    pytest.importorskip("pyarrow")
    rows = [flight._row(copy.deepcopy(r)) for r in records]
    batch = flight.rows_batch(records, rows, flight.flight_schema())
    return [flight.row_to_record(row) for row in batch.to_pylist()]


@pytest.mark.parametrize("case", SECTION_4, ids=[c["name"] for c in SECTION_4])
def test_every_section_4_record_comes_back_exactly_from_its_row(case):
    for record in case["records"]:
        row = flight._row(copy.deepcopy(record))
        assert (row["seq"] is not None) == _is_scalar(record), record
        back = flight.row_to_record(row)
        assert back == record
        if _is_scalar(record):
            assert list(back) == list(record)  # the capture's order of the nine keys


@pytest.mark.parametrize("case", SECTION_4, ids=[c["name"] for c in SECTION_4])
def test_every_section_4_record_comes_back_exactly_through_an_arrow_batch(case):
    records = copy.deepcopy(case["records"])
    assert _through_arrow(records) == case["records"]


@pytest.mark.parametrize("case", SECTION_4, ids=[c["name"] for c in SECTION_4])
def test_every_section_4_record_comes_back_through_record_to_row(case, reads_version_2):
    for record in case["records"]:
        assert flight.row_to_record(flight.record_to_row(copy.deepcopy(record))) == record


def test_a_step_row_has_a_null_seq_and_a_stray_seq_or_record_key_travels_in_extra_json():
    step = next(r for r in KEYS["records"] if not _is_scalar(r))
    assert flight._row(copy.deepcopy(step))["seq"] is None
    for stray in ({"seq": 3}, {"record": "other", "seq": 3}, {"seq": 3, "schema_version": 2}):
        record = {**step, **stray}
        row = flight._row(copy.deepcopy(record))
        assert row["seq"] is None, stray
        assert {k: v for k, v in json.loads(row["extra_json"]).items()} == {k: v for k, v in stray.items()
                                                                           if k != "schema_version"}, stray
        assert flight.row_to_record(row) == record


def test_a_scalar_only_row_gives_back_an_explicit_null_and_never_adds_a_key():
    """A step row reads a null `ts` or `grad_norm` back as a present null (`flight._ALWAYS`); a scalar-only
    row gives back exactly the keys its record had."""
    record = _scalar_record()
    assert "grad_norm" not in flight.row_to_record(flight._row(copy.deepcopy(record)))
    for changed in ({**record, "ts": None}, {**record, "grad_norm": None},
                    {k: v for k, v in record.items() if k != "ts"}, {**record, "attempt": 7, "rank": 3}):
        assert flight.row_to_record(flight._row(copy.deepcopy(changed))) == changed, changed
        assert _through_arrow([changed]) == [changed], changed


@pytest.mark.parametrize("seq", [True, -1, "0", 1.0, None, 2**63], ids=repr)
def test_a_scalar_only_record_whose_seq_the_column_cannot_hold_is_not_made_a_step_row(seq):
    """Its row would carry a null `seq`, which the server stores as a training row in a part and which moves
    the step watermark (D8). So the key is refused: by `record_to_row`, naming `seq`, and on the sender's path
    the line is skipped and counted, like a line whose key column cannot be stored."""
    record = {**_scalar_record(), "seq": seq}
    with pytest.raises(identity.MalformedRecordError) as raised:
        flight._row(copy.deepcopy(record))
    assert raised.value.field == "seq"
    missing = {k: v for k, v in _scalar_record().items() if k != "seq"}
    with pytest.raises(identity.MalformedRecordError):
        flight._row(missing)
    with pytest.raises(identity.MalformedRecordError):  # the record's mark, whatever its version
        flight._row({**missing, "schema_version": 1})


def test_the_sender_skips_and_names_a_scalar_only_line_whose_seq_cannot_be_stored():
    pytest.importorskip("pyarrow")
    good, step = _scalar_record(), next(r for r in KEYS["records"] if not _is_scalar(r))
    bad = {**good, "seq": -1}
    refused: list = []
    batch, steps, skipped = flight.rows_to_send([step, bad, good], flight.flight_schema(), refused=refused)
    assert skipped == 1 and refused == [(1, "seq", -1)]
    assert batch.column("seq").to_pylist() == [None, 0]
    assert batch.column("step").to_pylist() == [step["step"], 500]
    assert flight.SENDER_FIELDS == ("rank", "attempt", "step")


def test_a_row_without_a_seq_column_reads_as_before():
    """Rows of step 1's schema and of protocol 4's have no `seq`: read as step rows, as before."""
    step = next(r for r in KEYS["records"] if not _is_scalar(r))
    row = flight._row(copy.deepcopy(step))
    del row["seq"]
    assert flight.row_to_record(row) == step


# --- (c) A store with `scalars-*` objects beside its parts ---------------------------------------


def _write(path: Path, records: list[dict]) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flight._row(copy.deepcopy(r)) for r in records]
    pq.write_table(pa.Table.from_batches([flight.rows_batch(records, rows, flight.flight_schema())]), path)


def _store(tmp_path: Path) -> tuple[Path, dict]:
    """runs/al_sc: rank 0 with a part of steps 0 to 5 and two `scalars-*` objects (seq 0, then 1 and 2),
    rank 1 with a part and one object, each named as section 4.5 names them."""
    run = tmp_path / "store" / "runs" / "al_sc"
    steps = [r for r in KEYS["records"] if not _is_scalar(r)]
    first = _scalar_record()
    later = [{**first, "seq": 1, "step": 3, "ts": 101.0, "scalars": {"eval_loss": 1.7}},
             {**first, "seq": 2, "step": 900, "ts": 102.0, "scalars": {"eval_loss": 1.6, "eval_acc": 0.5}}]
    other = [{**r, "rank": 1} for r in steps]
    other_scalar = [{**first, "rank": 1, "seq": 0, "step": 5}]
    attempt0, attempt1 = run / "rank=0" / "attempt=1", run / "rank=1" / "attempt=1"
    _write(attempt0 / "part-00000000-00000005.parquet", steps)
    _write(attempt0 / "scalars-00000001-00000002.parquet", later)
    _write(attempt0 / "scalars-00000000-00000000.parquet", [first])
    _write(attempt1 / "part-00000000-00000005.parquet", other)
    _write(attempt1 / "scalars-00000000-00000000.parquet", other_scalar)
    return run, {"steps": steps, "rank0": [first, *later], "rank1": other_scalar}


def test_the_store_s_scalars_objects_are_listed_by_rank_attempt_and_first_seq(tmp_path):
    run, _ = _store(tmp_path)
    names = [p.relative_to(run).as_posix() for p in flight.scalar_objects(run)]
    assert names == ["rank=0/attempt=1/scalars-00000000-00000000.parquet",
                     "rank=0/attempt=1/scalars-00000001-00000002.parquet",
                     "rank=1/attempt=1/scalars-00000000-00000000.parquet"]
    assert [p.name for p in flight.scalar_objects(run, rank=1)] == ["scalars-00000000-00000000.parquet"]
    assert [p.name for p in flight.scalar_objects(run / "rank=0")] == ["scalars-00000000-00000000.parquet",
                                                                      "scalars-00000001-00000002.parquet"]
    assert flight.scalar_objects(run / "rank=0" / "attempt=1" / "part-00000000-00000005.parquet") == []


def test_the_store_s_scalar_only_records_read_back_exactly(tmp_path, reads_version_2):
    run, made = _store(tmp_path)
    assert flight.load_scalar_records(run, rank=0) == made["rank0"]
    assert flight.load_scalar_records(run, rank=1) == made["rank1"]
    assert flight.load_scalar_records(run) == [*made["rank0"], *made["rank1"]]


def test_a_scalars_object_is_read_with_its_rows_checked(tmp_path, reads_version_2):
    run, _ = _store(tmp_path)
    [path] = flight.scalar_objects(run, rank=1)
    _write(path, [{**_scalar_record(), "rank": 1, "attempt": -4}])
    with pytest.raises(identity.MalformedRecordError) as raised:
        flight.load_scalar_records(run, rank=1)
    assert str(raised.value) == (f"attempt is -4 at {identity.in_part(path, 0)}: "
                                 "tensorquill reads an integer of 0 or more")


def test_load_records_reads_one_scalars_object_as_its_records(tmp_path, reads_version_2):
    from tensorquill import evidence

    run, made = _store(tmp_path)
    path = run / "rank=0" / "attempt=1" / "scalars-00000001-00000002.parquet"
    assert evidence.load_records(path, all_attempts=True) == made["rank0"][1:]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=JOINED_LOADER)
def test_load_records_reads_a_store_s_scalars_objects_beside_its_parts(tmp_path):
    from tensorquill import evidence

    run, made = _store(tmp_path)
    read = evidence.load_records(run, rank=0, all_attempts=True)
    assert [r for r in read if not _is_scalar(r)] == made["steps"]
    assert [r for r in read if _is_scalar(r)] == made["rank0"]
