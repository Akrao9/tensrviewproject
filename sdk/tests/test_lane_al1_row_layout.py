"""Lane AL-1 of wave AL: protocol 5's row layout in `flight.py` (`WAVE_AL_CONTRACT.md` sections 1 to 3).

- Section 1: `layers` and `moe_stats` as typed maps of structs, in the record's order, a refused field
  or entry in `extra_json`, `moved` naming the map once per row for an overflow or Arrow's refusal.
- Section 2: the probe's maps in `probe_maps`, its histograms in `probe_histograms`, the rest in `probe_json`.
- Section 3: row `schema_version` 2 with the record's own version in `record_version`; batch metadata 2.
- `row_to_record` reads a row of either layout, so `load_records` reads a store of protocol 4 parts,
  protocol 5 parts, or both.

The row tests run on CI's plain jobs (no pyarrow): `record_to_row` and `row_to_record` need none. Those
that build an Arrow batch or a Parquet part skip there.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from tensorquill import flight
from tensorquill.flight import record_to_row, row_to_record

CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_al" / "cases.json").read_text())["cases"]
ROWS = [c for c in CASES if c["section"] in (1, 2)]
TYPED = ("layers", "moe_stats", "probe_maps", "probe_histograms")


def _at(case: dict) -> dict:
    return copy.deepcopy(next(r for r in case["records"] if r["step"] == case["row_step"]))


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _orders_kept(back: dict, record: dict) -> None:
    for key in ("layers", "moe_stats"):
        if isinstance(record.get(key), dict):
            assert list(back[key]) == list(record[key]), key
    probe = record.get("probe")
    if isinstance(probe, dict):
        for key in ("attention", "histograms"):
            if isinstance(probe.get(key), dict):
                assert list(back["probe"][key]) == list(probe[key]), key


# --- the cases of sections 1 and 2, pure Python ---------------------------------------------------


@pytest.mark.parametrize("case", ROWS, ids=[c["name"] for c in ROWS])
def test_record_to_row_gives_the_cases_row(case):
    record = _at(case)
    row = record_to_row(copy.deepcopy(record))
    want = case["row"]
    for column in TYPED:
        if column in want:
            got = None if row[column] is None else [[name, _jsonable(v)] for name, v in row[column]]
            assert got == want[column], column
    if "probe_json" in want:
        assert row["probe_json"] == want["probe_json"]
    assert (json.loads(row["extra_json"]) if row["extra_json"] else None) == want["extra"]
    assert (row["schema_version"], row["record_version"]) == (want["schema_version"], want["record_version"])
    assert "layers_json" not in row and "moe_stats_json" not in row


@pytest.mark.parametrize("case", ROWS, ids=[c["name"] for c in ROWS])
def test_row_to_record_gives_every_record_back_with_each_maps_order(case):
    for record in case["records"] + (case.get("clean_records") or []):
        back = row_to_record(record_to_row(copy.deepcopy(record)))
        assert back == record, record["step"]
        _orders_kept(back, record)


def test_moved_names_moe_stats_once_for_an_integer_past_int64():
    """Case `moe_stats_int_past_int64`: an overflow is counted; a type refusal is not (section 1.4)."""
    case = next(c for c in ROWS if c["name"] == "moe_stats_int_past_int64")
    record = _at(case)
    record["moe_stats"]["second.router"] = dict(record["moe_stats"]["model.layers.0.block_sparse_moe.router"])
    moved: list = []
    record_to_row(record, moved=moved)
    assert moved == ["moe_stats"]  # two refused fields in one row: named once
    for name in ("moe_stats_float_in_int_field", "moe_stats_diversity_refused", "layers_field_bool",
                 "layers_field_extra", "layers_entry_not_object", "layers_value_not_object", "layers_field_null"):
        moved = []
        record_to_row(_at(next(c for c in ROWS if c["name"] == name)), moved=moved)
        assert moved == [], name


def test_moved_names_layers_for_a_name_arrow_cannot_encode_and_a_float_too_large():
    for name in ("layers_name_arrow_cannot_encode",):
        moved: list = []
        record_to_row(_at(next(c for c in ROWS if c["name"] == name)), moved=moved)
        assert moved == ["layers"], name
    record = {"schema_version": 1, "run_id": "r", "rank": 0, "step": 0, "ts": 1.0, "grad_norm": None,
              "layers": {"a": {"mean": 10**400, "std": 1.0}, "b": {"std": 2.0}}}
    moved = []
    row = record_to_row(record, moved=moved)
    assert moved == ["layers"]
    assert row["layers"] == [("a", {"std": 1.0}), ("b", {"std": 2.0})]
    assert json.loads(row["extra_json"]) == {"layers": {"a": {"mean": 10**400}}}
    assert row_to_record(row) == record


# --- the versions (section 3) -------------------------------------------------------------------------


def _plain_record(**more) -> dict:
    return {"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 1, "step": 3, "ts": 1.0, "grad_norm": 2.0,
            **more}


def test_the_row_carries_version_2_and_the_records_own_version():
    assert flight.SCHEMA_VERSION == 2  # the batch metadata (D6)
    row = record_to_row(_plain_record())
    assert (row["schema_version"], row["record_version"]) == (2, 1)
    assert row_to_record(row) == _plain_record()
    legacy = {k: v for k, v in _plain_record().items() if k != "schema_version"}
    row = record_to_row(legacy)
    assert (row["schema_version"], row["record_version"]) == (2, None)
    assert row_to_record(row) == legacy  # no `schema_version` key comes back
    zero = _plain_record(schema_version=0)
    row = record_to_row(zero)
    assert (row["schema_version"], row["record_version"]) == (2, 0)
    assert row_to_record(row) == zero


def test_a_version_this_package_does_not_read_is_the_rows_own_so_the_server_counts_it():
    """RS10 kept: a record of a version no reader takes travels with that version as its row's, so a
    server drops and counts it (`unsupported_schema_version`) instead of storing it as a protocol 5 row."""
    row = flight._row(_plain_record(schema_version=99))
    assert (row["schema_version"], row["record_version"]) == (99, 99)


def test_the_sender_still_sends_an_unread_version_and_skips_one_that_is_not_an_integer():
    pytest.importorskip("pyarrow")
    good = _plain_record()
    records = [good, {**good, "step": 4, "schema_version": 99}, {**good, "step": 5, "schema_version": 1.0}]
    refused: list = []
    batch, steps, skipped = flight.rows_to_send(records, flight.flight_schema(), refused=refused)
    assert steps == [3, 4] and skipped == 1
    assert [(field, value) for _, field, value in refused] == [("schema_version", 1.0)]
    rows = batch.to_pylist()
    assert [(r["schema_version"], r["record_version"]) for r in rows] == [(2, 1), (99, 99)]


# --- the edges section 1.4 names, pure Python ------------------------------------------------------------


@pytest.mark.parametrize("layers", [
    {},
    {"a": {}},
    {"a": None, "b": {"std": 1.0}},
    {"a": "x", "b": [1], "c": {"std": "inf", "zero_frac": 0}},
    {"\ud800": {"std": 1.0}},  # no entry held: the whole value travels in extra_json, the column is null
    {"a": {"std": True, "p99": "nan", "mean": None}},
    [1, 2],
    "text",
])
def test_every_shape_of_layers_reads_back_as_written(layers):
    record = _plain_record(layers=layers)
    back = row_to_record(record_to_row(copy.deepcopy(record)))
    assert back == record
    if isinstance(layers, dict):
        assert list(back["layers"]) == list(layers)


def test_an_empty_map_is_an_empty_column_and_null_layers_travels_in_extra_json():
    row = record_to_row(_plain_record(layers={}))
    assert row["layers"] == [] and row["extra_json"] is None
    row = record_to_row(_plain_record(layers=None))
    assert row["layers"] is None and json.loads(row["extra_json"]) == {"layers": None}
    assert row_to_record(row) == _plain_record(layers=None)


def test_a_layer_with_no_encodable_name_held_sends_the_whole_value():
    row = record_to_row(_plain_record(layers={"\ud800": {"std": 1.0}}))
    assert row["layers"] is None
    assert json.loads(row["extra_json"]) == {"layers": {"\ud800": {"std": 1.0}}}


@pytest.mark.parametrize("probe", [
    {"histograms": {"a": {"lo": 0, "hi": "inf", "counts": [0, 1], "nonfinite": 2}},
     "attention": {"a": {"rows": 2, "map": [["nan", 0.5, 1], []]}, "b": {"map": [[10**400]]}}},
    {"attention": {"a": {"map": "not a grid"}, "b": {"rows": 1}}, "head": {"module": "lm_head", "not_read": "x"}},
    {"histograms": {"a": {"hi": 1.0, "lo": 0.0, "counts": [1], "nonfinite": 0}}},  # another key order: stays
    {"histograms": {"a": {"lo": 0.0, "hi": 1.0, "counts": [2**63], "nonfinite": 0}}},  # past int64: stays
    {"attention": {"\ud800": {"map": [[1.0]]}}},
    "a probe that is text",
    [1, 2],
])
def test_every_shape_of_probe_reads_back_as_written(probe):
    record = _plain_record(probe=probe)
    back = row_to_record(record_to_row(copy.deepcopy(record)))
    assert back == record


def test_the_probe_split_by_column():
    probe = {"histograms": {"a": {"lo": 0, "hi": "inf", "counts": [0, 1], "nonfinite": 2}},
             "attention": {"a": {"rows": 2, "map": [["nan", 0.5, 1], []]}, "b": {"map": [[0.1]]}}}
    row = record_to_row(_plain_record(probe=probe))
    assert json.loads(row["probe_json"]) == {"attention": {"a": {"rows": 2}, "b": {"map": [[0.1]]}}}
    assert _jsonable(row["probe_maps"]) == [["a", [["nan", 0.5, 1.0], []]]]
    assert _jsonable(row["probe_histograms"]) == [["a", {"lo": 0.0, "hi": "inf", "counts": [0, 1], "nonfinite": 2}]]
    back = row_to_record(row)
    assert back == _plain_record(probe=probe)
    assert list(back["probe"]) == ["histograms", "attention"]


# --- through Arrow and Parquet ----------------------------------------------------------------------


def test_the_schema_is_protocol_5s():
    pa = pytest.importorskip("pyarrow")
    schema = flight.flight_schema()
    names = schema.names
    for gone in ("layers_json", "moe_stats_json"):
        assert gone not in names
    f64, f32, i64 = pa.float64(), pa.float32(), pa.int64()
    layer = pa.struct([pa.field(n, f64) for n in ("mean", "std", "abs_max", "zero_frac")])
    moe = pa.struct([pa.field("routed_tokens", i64), pa.field("entropy", f64), pa.field("diversity", f64),
                     pa.field("dead_experts", i64), pa.field("dead_frac", f64)])
    hist = pa.struct([pa.field("lo", f64), pa.field("hi", f64), pa.field("counts", pa.list_(i64)),
                      pa.field("nonfinite", i64)])
    assert schema.field("layers").type == pa.map_(pa.string(), layer)
    assert schema.field("moe_stats").type == pa.map_(pa.string(), moe)
    assert schema.field("probe_maps").type == pa.map_(pa.string(), pa.list_(pa.list_(f32)))
    assert schema.field("probe_histograms").type == pa.map_(pa.string(), hist)
    assert schema.field("record_version").type == pa.int32() and schema.field("record_version").nullable
    for name in ("layers", "moe_stats", "probe_maps", "probe_histograms"):
        assert schema.field(name).type.keys_sorted is False, name
    assert names.index("probe_json") < names.index("probe_maps") < names.index("probe_histograms")
    assert "seq" not in names  # the scalar-only record's column is step 2's (section 4)


def test_every_case_record_round_trips_through_an_arrow_batch_with_order_kept():
    pytest.importorskip("pyarrow")
    for case in ROWS:
        records = copy.deepcopy(case["records"])
        batch = flight.records_to_batch(records)
        for record, row in zip(records, batch.to_pylist()):
            assert row["schema_version"] == 2
            back = row_to_record(row)
            assert back == record, (case["name"], record["step"])
            _orders_kept(back, record)


def test_the_sender_sends_every_case_record_with_moved_named_once():
    pytest.importorskip("pyarrow")
    for case in ROWS:
        records = copy.deepcopy(case["records"])
        moved: list = []
        batch, steps, skipped = flight.rows_to_send(records, flight.flight_schema(), moved)
        assert skipped == 0 and steps == [r["step"] for r in records], case["name"]
        assert [row_to_record(r) for r in batch.to_pylist()] == records, case["name"]
        if case["name"] == "moe_stats_int_past_int64":  # once per row that holds the overflow
            past = [r["step"] for r in records
                    if any(isinstance(v, int) and v > 2**63 - 1 for e in r["moe_stats"].values() for v in e.values())]
            assert case["row_step"] in past and moved == [(step, "moe_stats") for step in past]
        else:
            assert {key for _, key in moved} <= {"layers"}, case["name"]  # the lone surrogate's name only


V4_COLUMNS = ("layers_json", "moe_stats_json", "probe_json")


def _v4_schema():
    """The protocol 4 layout, written out here (this test needs no git history)."""
    import pyarrow as pa

    f64 = pa.float64()
    return pa.schema([
        pa.field("schema_version", pa.int32(), nullable=False), pa.field("run_id", pa.string(), nullable=False),
        pa.field("rank", pa.int32(), nullable=False), pa.field("attempt", pa.int64(), nullable=False),
        pa.field("step", pa.int64(), nullable=False), pa.field("ts", f64), pa.field("lr", pa.list_(f64)),
        pa.field("grad_norm", f64), pa.field("clip_grad_norm", f64), pa.field("weight_norm", f64),
        pa.field("weights_unchanged", pa.list_(pa.string())), pa.field("step_skipped", pa.bool_()),
        pa.field("grad_scale", f64), pa.field("scalars", pa.map_(pa.string(), f64)),
        pa.field("grad_norms", pa.map_(pa.string(), f64)), pa.field("weight_norms", pa.map_(pa.string(), f64)),
        pa.field("update_rms", pa.map_(pa.string(), f64)), pa.field("moe_top_k", pa.map_(pa.string(), pa.int32())),
        pa.field("moe_counts", pa.map_(pa.string(), pa.list_(pa.int64()))),
        pa.field("replica_spread", pa.map_(pa.string(), pa.list_(f64))), pa.field("layers_json", pa.string()),
        pa.field("moe_stats_json", pa.string()), pa.field("probe_json", pa.string()), pa.field("data_json", pa.string()),
        pa.field("checkpoint_json", pa.string()), pa.field("data_wait", f64), pa.field("step_time", f64),
        pa.field("system_json", pa.string()), pa.field("extra_json", pa.string()),
    ])


def _v4_row(record: dict) -> dict:
    """A protocol 4 row as today's sender makes it, for the records this test writes (every value held)."""
    row = {"schema_version": record.get("schema_version", 0), "run_id": record["run_id"], "rank": record["rank"],
           "attempt": record.get("attempt", 0), "step": record["step"], "ts": record.get("ts"),
           "grad_norm": record.get("grad_norm"), "lr": record.get("lr")}
    if "layers" in record:
        row["layers_json"] = json.dumps(record["layers"], separators=(",", ":"))
    if "moe_stats" in record:
        row["moe_stats_json"] = json.dumps(record["moe_stats"], separators=(",", ":"), sort_keys=True)
    if "probe" in record:
        row["probe_json"] = json.dumps(record["probe"], separators=(",", ":"))
    return row


def _records(steps, attempt: int = 1) -> list[dict]:
    layers = {"model.layers.2": {"mean": 0.0, "std": 1.0, "abs_max": 3.0, "zero_frac": 0.0},
              "model.layers.10": {"mean": 0.0, "std": 1.5, "abs_max": 4.0, "zero_frac": 0.0}}
    moe = {"r": {"routed_tokens": 8, "entropy": 1.0, "diversity": 0.5, "dead_experts": 0, "dead_frac": 0.0}}
    probe = {"histograms": {"a": {"lo": -1.0, "hi": 1.0, "counts": [1, 2], "nonfinite": 0}},
             "attention": {"a": {"max_logit": [1.0], "rows": 2, "map": [[0.5, 0.25]]}}}
    return [{"schema_version": 1, "run_id": "mixed", "rank": 0, "attempt": attempt, "step": s, "ts": float(s),
             "grad_norm": 1.0, "lr": [0.01], "layers": copy.deepcopy(layers), "moe_stats": copy.deepcopy(moe),
             **({"probe": copy.deepcopy(probe)} if s % 2 else {})} for s in steps]


def _write_part(folder: Path, rows: list[dict], schema) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder.mkdir(parents=True, exist_ok=True)
    steps = [r["step"] for r in rows]
    path = folder / f"part-{min(steps):08d}-{max(steps):08d}.parquet"
    pq.write_table(pa.Table.from_batches([pa.RecordBatch.from_pylist(rows, schema=schema)]), path)
    return path


def test_load_records_reads_a_store_of_either_layout_and_of_both():
    """Section 3: a layout is told by its columns. A rank whose protocol 4 backlog and protocol 5 rows
    meet has parts of both layouts, never a part of two; `load_records` reads every one through
    `row_to_record`."""
    pytest.importorskip("pyarrow")
    from tensorquill.evidence import load_records

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "runs" / "mixed"
        folder = run / "rank=0" / "attempt=1"
        old, new = _records(range(0, 4)), _records(range(4, 8))
        _write_part(folder, [_v4_row(r) for r in old], _v4_schema())
        _write_part(folder, [record_to_row(r) for r in new], flight.flight_schema())
        loaded = load_records(run)
        assert loaded == old + new
        for back, record in zip(loaded, old + new):
            _orders_kept(back, record)
        alone = load_records(sorted(folder.glob("part-*.parquet"))[1])
        assert alone == new


def test_a_parquet_part_keeps_every_case_record_and_its_order(tmp_path):
    pytest.importorskip("pyarrow")
    from tensorquill.evidence import load_records

    for index, case in enumerate(ROWS):
        records = [{**r, "run_id": "cases", "attempt": 1} for r in copy.deepcopy(case["records"])]
        folder = tmp_path / str(index)
        batch = flight.records_to_batch(records)
        import pyarrow as pa
        import pyarrow.parquet as pq

        folder.mkdir()
        part = folder / "part-00000000-00000099.parquet"
        pq.write_table(pa.Table.from_batches([batch]), part)
        loaded = load_records(part)
        assert loaded == records, case["name"]
        for back, record in zip(loaded, records):
            _orders_kept(back, record)
