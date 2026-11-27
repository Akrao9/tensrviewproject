"""Wave AL step 2, lane AL-1': the scalar-only record's row (`WAVE_AL_CONTRACT.md` sections 3, 4.4 and 4.5).

- The row of case `scalar_record_keys`: `seq` set, `record_version` 2, row `schema_version` 2, `scalars`
  in the `scalars` column entry by entry (AQ5-D7, Nov 26, 2026; before, wave W3's rule), nothing in `extra_json` (`record` comes back from `seq`).
- `row_to_record(record_to_row(r)) == r` for every record of every section 4 case, scalar-only and step
  records alike, in a plain row and through a real Arrow batch; a step row's `seq` is null.
- A store holding `scalars-<first_seq:08>-<last_seq:08>.parquet` objects beside its parts (section 4.5's
  names; the server writes them, so the store here is built with pyarrow) is listed by the one listing,
  `evidence._parquet_parts(..., scalars=True)`, and read back record for record by
  `load_records(..., scalars=True)`.
- One rule of what a scalar-only record is, `identity.is_scalar_record`, which `flight` asks too.

Tests that need pyarrow skip one by one without it (CI's plain jobs have none); none needs the server
binary or git history. A server's own store is step 2's seam test (`test_wave_al_step2_seams.py`)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tensorquill import flight, identity

CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_al" / "cases.json").read_text(encoding="utf-8"))["cases"]
SECTION_4 = [c for c in CASES if c["section"] == 4]
KEYS = next(c for c in SECTION_4 if c["name"] == "scalar_record_keys")


def _scalar_record() -> dict:
    return copy.deepcopy(next(r for r in KEYS["records"] if r.get("record") == "scalars"))


def _is_scalar(record: dict) -> bool:
    return record.get("record") == "scalars"


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


def test_record_to_row_takes_the_scalar_only_record():
    """`identity.check` takes record version 2 for a scalar-only record (section 4.6), so `record_to_row`
    gives the case's row through the check."""
    assert 2 in identity.SCHEMA_VERSIONS
    record = _scalar_record()
    row = flight.record_to_row(copy.deepcopy(record))
    assert _summary(row) == {"seq": 0, "record_version": 2, "schema_version": 2, "extra": None}
    assert flight.row_to_record(row) == record


def test_the_schema_has_seq_as_a_nullable_int64_before_record_version():
    pa = pytest.importorskip("pyarrow")
    schema = flight.flight_schema()
    field = schema.field("seq")
    assert field.type == pa.int64() and field.nullable
    names = schema.names
    assert names.index("seq") + 1 == names.index("record_version") and names[-1] == "extra_json"


def test_scalars_are_held_entry_by_entry_in_a_scalar_only_row():
    """A logged bool is left out of the column, as on a step row; so is any other value the column cannot
    hold, alone, which travels in `extra_json` while its siblings stay in the column (AQ5-D7, the author,
    Nov 26, 2026; under wave W3's rule it moved the whole map), and the row reads back as the record."""
    record = _scalar_record()
    step = {k: v for k, v in record.items() if k not in ("record", "seq")} | {"schema_version": 1}
    with_bool = {**record, "scalars": {"eval_loss": 1.8, "flag": True}}
    assert flight._row(with_bool)["scalars"] == flight._row({**step, "scalars": with_bool["scalars"]})["scalars"] \
        == [("eval_loss", 1.8)]
    with_text = {**record, "scalars": {"eval_loss": 1.8, "note": "high"}}
    row = flight._row(with_text)
    assert row["scalars"] == [("eval_loss", 1.8)] and json.loads(row["extra_json"]) == {"scalars": {"note": "high"}}
    only_text = {**record, "scalars": {"note": "high"}}  # no entry held: the whole map, the column null
    assert flight._row(only_text)["scalars"] is None
    assert json.loads(flight._row(only_text)["extra_json"]) == {"scalars": {"note": "high"}}
    assert flight.row_to_record(flight._row(only_text)) == only_text
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
def test_every_section_4_record_comes_back_through_record_to_row(case):
    for record in case["records"]:
        assert flight.row_to_record(flight.record_to_row(copy.deepcopy(record))) == record


def test_a_step_row_has_a_null_seq_and_a_stray_seq_or_record_key_travels_in_extra_json():
    step = next(r for r in KEYS["records"] if not _is_scalar(r))
    assert flight._row(copy.deepcopy(step))["seq"] is None
    for stray in ({"seq": 3}, {"record": "other", "seq": 3}, {"seq": 3, "schema_version": 2}, {"record": "scalars"},
                  {"record": "scalars", "seq": 3}):
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


@pytest.mark.parametrize("extra", [{}, {"seq": 3}, {"seq": -1}], ids=repr)
def test_a_version_1_record_marked_scalars_is_a_step_record_whose_record_key_travels_as_any_unknown_key(extra):
    """One rule for what a scalar-only record is: `identity.is_scalar_record` (`schema_version` the integer 2
    and `"record": "scalars"`), which `identity.check`, every reader and `flight` ask. A version 1 record that
    carries the mark is an ordinary step record: its `record` (and any `seq`) travel in `extra_json`, its
    `seq` column is null, and the Rust paths are sent it as a training record."""
    from tensorquill import server_detect

    step = next(r for r in KEYS["records"] if not _is_scalar(r))
    record = {**copy.deepcopy(step), "record": "scalars", **extra}
    assert not identity.is_scalar_record(record)
    row = flight.record_to_row(copy.deepcopy(record))  # identity.check takes a version 1 record
    assert row["seq"] is None and (row["schema_version"], row["record_version"]) == (2, 1)
    assert json.loads(row["extra_json"]) == {"record": "scalars", **extra}
    assert flight.row_to_record(row) == record
    assert server_detect.stream_order([record]) == [record]
    assert not hasattr(flight, "scalar_only")  # the rule is identity's, asked by name


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


def _names(run: Path, paths: list[Path]) -> list[str]:
    return [p.relative_to(run).as_posix() for p in paths]


def test_the_store_s_scalars_objects_are_listed_after_their_attempt_s_parts_by_first_seq(tmp_path):
    """One listing, `evidence._parquet_parts`: with `scalars=True` each attempt's `scalars-*` objects follow
    its parts, by the first `seq` of their names; without it they are never listed, as before step 2."""
    from tensorquill import evidence

    run, _ = _store(tmp_path)
    assert _names(run, evidence._parquet_parts(run, scalars=True)) == [
        "rank=0/attempt=1/part-00000000-00000005.parquet",
        "rank=0/attempt=1/scalars-00000000-00000000.parquet",
        "rank=0/attempt=1/scalars-00000001-00000002.parquet",
        "rank=1/attempt=1/part-00000000-00000005.parquet",
        "rank=1/attempt=1/scalars-00000000-00000000.parquet"]
    assert _names(run, evidence._parquet_parts(run, 1, scalars=True)) == [
        "rank=1/attempt=1/part-00000000-00000005.parquet", "rank=1/attempt=1/scalars-00000000-00000000.parquet"]
    assert _names(run, evidence._parquet_parts(run)) == ["rank=0/attempt=1/part-00000000-00000005.parquet",
                                                         "rank=1/attempt=1/part-00000000-00000005.parquet"]
    assert not hasattr(flight, "scalar_objects") and not hasattr(flight, "read_scalar_object")


def test_load_records_reads_a_store_s_scalars_objects_beside_its_parts(tmp_path):
    from tensorquill import evidence

    run, made = _store(tmp_path)
    read = evidence.load_records(run, rank=0, all_attempts=True, scalars=True)
    assert read == [*made["steps"], *made["rank0"]]  # each record exactly as written
    assert evidence.load_records(run, rank=1, scalars=True) == [*[{**r, "rank": 1} for r in made["steps"]],
                                                                *made["rank1"]]
    assert evidence.load_records(run, rank=0, all_attempts=True) == made["steps"]  # the default, as before


def test_a_scalars_object_s_rows_are_checked_naming_the_object_and_row(tmp_path):
    from tensorquill import evidence

    run, _ = _store(tmp_path)
    path = run / "rank=1" / "attempt=1" / "scalars-00000000-00000000.parquet"
    _write(path, [{**_scalar_record(), "rank": 1, "attempt": -4}])
    with pytest.raises(identity.MalformedRecordError) as raised:
        evidence.load_records(run, rank=1, scalars=True)
    assert str(raised.value) == (f"attempt is -4 at {identity.in_part(path, 0)}: "
                                 "tensorquill reads an integer of 0 or more")
    assert len(evidence.load_records(run, rank=1)) == 6  # never listed by default, so never read


def test_a_scalars_name_that_does_not_parse_is_listed_last_in_its_attempt(tmp_path):
    """Section 4.5's names give an object's first and last `seq`; the server reads the footer for a name that
    does not parse. The listing keeps such an object, after the named ones of its attempt, by name, and its
    rows read as any object's."""
    from tensorquill import evidence

    run, made = _store(tmp_path)
    extra = {**_scalar_record(), "seq": 3, "step": 950}
    _write(run / "rank=0" / "attempt=1" / "scalars-recovered.parquet", [extra])
    assert _names(run, evidence._parquet_parts(run, 0, scalars=True)) == [
        "rank=0/attempt=1/part-00000000-00000005.parquet",
        "rank=0/attempt=1/scalars-00000000-00000000.parquet",
        "rank=0/attempt=1/scalars-00000001-00000002.parquet",
        "rank=0/attempt=1/scalars-recovered.parquet"]
    read = evidence.load_records(run, rank=0, all_attempts=True, scalars=True)
    assert [r for r in read if _is_scalar(r)] == [*made["rank0"], extra]


def test_load_records_reads_one_scalars_object_as_its_records(tmp_path):
    """Given one object's path, `load_records` reads it as any part: with `scalars=True` its records come
    back exactly; by default it gives the training records alone, so none."""
    from tensorquill import evidence

    run, made = _store(tmp_path)
    path = run / "rank=0" / "attempt=1" / "scalars-00000001-00000002.parquet"
    assert evidence.load_records(path, all_attempts=True, scalars=True) == made["rank0"][1:]
    assert evidence.load_records(path, all_attempts=True) == []


def test_a_rank_s_listing_never_enters_another_rank_s_folder(tmp_path, monkeypatch):
    """With `rank`, the `rank=<n>` folders of other ranks are not read, scalars objects included, so one
    rank's read never costs the rest of the run's listing."""
    import os

    from tensorquill import evidence

    run, _ = _store(tmp_path)
    opened: list[str] = []
    scandir = os.scandir

    def watched(path="."):
        opened.append(os.fspath(path))
        return scandir(path)

    monkeypatch.setattr(os, "scandir", watched)
    assert len(evidence._parquet_parts(run, 0, scalars=True)) == 3
    assert opened and not [p for p in opened if "rank=1" in Path(p).parts], opened


@pytest.mark.parametrize("seq", [-1, None, "absent"], ids=repr)
def test_the_sender_says_which_scalar_only_line_it_left_out_for_its_seq(seq, capsys):
    """What `rows_to_send` gives the sender for such a line is what the sender says, once (wave AG's message)."""
    pytest.importorskip("pyarrow")
    good, step = _scalar_record(), next(r for r in KEYS["records"] if not _is_scalar(r))
    bad = {k: v for k, v in good.items() if k != "seq"} if seq == "absent" else {**good, "seq": seq}
    records = [step, bad]
    lines = [json.dumps(r).encode() for r in records]
    refused: list = []
    flight.rows_to_send(records, flight.flight_schema(), refused=refused)
    sender = flight.FlightSender.__new__(flight.FlightSender)
    sender._said_skipped, sender.path, sender.endpoint = False, "run.rank0.jsonl", "grpc://server"
    sender._say_skipped(0, lines, [0, 1], refused, [r.get("step") for r in records])
    found = "absent" if seq == "absent" else identity._shown(seq)
    assert capsys.readouterr().err.startswith(
        f"tensorquill: the line at byte {len(lines[0]) + 1} of run.rank0.jsonl (step 500) is not sent to grpc://server: "
        f"its seq is {found}, and its column in the stream cannot hold it.")
