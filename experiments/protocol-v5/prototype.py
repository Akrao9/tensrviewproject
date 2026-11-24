"""A prototype of protocol 5's row layout, for WAVE_AL_CONTRACT.md's measurements. Not product code.

It builds on protocol 4's row functions: `record_to_row_v5` takes the v4 row of the record without
`layers`, `moe_stats` and `probe`, and adds the typed columns the contract proposes;
`row_to_record_v5` is its inverse. `flight_schema_v5` is the Arrow schema. Since lane AL-1 merged
(`92e471a`) `tensorquill.flight` writes protocol 5's layout itself, so the v4 row comes from
`flight_v4.py` beside this file, a frozen copy of `flight.py`'s row functions at `5c286ed`; the
prototype's rows are unchanged. Lane AL-1's `record_to_row` gives the prototype's row on every record
of the section 1 and 2 cases (`test_wave_al_contract.py`).

What the prototype holds, and where a value it cannot hold goes (the contract's section 2):

- `layers`: `Map<Utf8, Struct<mean, std, abs_max, zero_frac: Float64>>`, the map's values nullable.
- `moe_stats`: `Map<Utf8, Struct<routed_tokens: Int64, entropy: Float64, diversity: Float64,
  dead_experts: Int64, dead_frac: Float64>>`, the values nullable.
- A field holds what `flight._float` (Float64) or an int64 integer (Int64) takes: a number that is
  not a bool, or one of the strings "nan", "inf", "-inf" for a Float64 field. A field the struct
  does not name, an explicit `null`, a bool, any other string and a number its type cannot hold are
  refused: the field travels in `extra_json` under `<key>.<entry>.<field>`, and the struct's field is
  null. A missing field is null and reads back absent.
- An entry that is not an object is held as a null struct (its place in the map's order) and
  travels whole in `extra_json` under `<key>.<entry>`. A name Arrow cannot encode cannot be a key: the
  entry travels only in `extra_json`, and so loses its place (it is read after the column's entries).
- A value that is not an object travels whole in `extra_json` and the column is null. An empty
  object is an empty map.
- `probe`: the light part stays JSON in `probe_json`; each attention module's `map` goes to
  `probe_maps: Map<Utf8, List<List<Float32>>>` (keyed by the module's label) when every value is a
  number exactly representable as float32 (or "nan"/"inf"/"-inf"), and `histograms` goes whole to
  `probe_histograms: Map<Utf8, Struct<lo, hi: Float64, counts: List<Int64>, nonfinite: Int64>>`
  when every entry has exactly those four keys with those types. Anything else stays in `probe_json`
  where it was.
- `seq` (Int64) and `record_version` (Int32) are the scalar-only record's columns (contract section
  4); a step record leaves `seq` null.

`layout` selects the row's `schema_version`: "v5" writes 2 (the contract's proposal), "v4-accepted"
writes the record's own version, so a protocol 4 server stores the rows (used by the measurements,
which run against the protocol 4 binary).
"""

from __future__ import annotations

import json
import math
import struct
from typing import Any, Mapping


def _flight_v4() -> Any:
    """`flight_v4.py` beside this file, loaded by path: the prototype is loaded by path too, from the
    tests and from the scripts here, so it adds nothing to `sys.path`."""
    import importlib.util
    import sys
    from pathlib import Path

    name = "tensorquill_protocol_v5_experiment_flight_v4"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name("flight_v4.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


v4 = _flight_v4()

LAYER_FIELDS = ("mean", "std", "abs_max", "zero_frac")
MOE_FIELDS = ("routed_tokens", "entropy", "diversity", "dead_experts", "dead_frac")
MOE_INT = frozenset({"routed_tokens", "dead_experts"})
HIST_KEYS = ("lo", "hi", "counts", "nonfinite")
INT64 = (-(2**63), 2**63 - 1)
_MOVED = ("layers", "moe_stats", "probe")


def flight_schema_v5(token_entropy: bool = False) -> Any:
    import pyarrow as pa

    f64, f32, i64 = pa.float64(), pa.float32(), pa.int64()
    base = v4.flight_schema()
    layer = pa.struct([pa.field(name, f64) for name in LAYER_FIELDS])
    moe_fields = [pa.field(name, i64 if name in MOE_INT else f64) for name in MOE_FIELDS]
    if token_entropy:
        moe_fields.append(pa.field("token_entropy", f64))
    moe = pa.struct(moe_fields)
    hist = pa.struct([pa.field("lo", f64), pa.field("hi", f64), pa.field("counts", pa.list_(i64)),
                      pa.field("nonfinite", i64)])
    fields = []
    for field in base:
        if field.name == "layers_json":
            fields.append(pa.field("layers", pa.map_(pa.string(), layer)))
        elif field.name == "moe_stats_json":
            fields.append(pa.field("moe_stats", pa.map_(pa.string(), moe)))
        elif field.name == "probe_json":
            fields.append(field)
            fields.append(pa.field("probe_maps", pa.map_(pa.string(), pa.list_(pa.list_(f32)))))
            fields.append(pa.field("probe_histograms", pa.map_(pa.string(), hist)))
        elif field.name == "extra_json":
            fields.append(pa.field("seq", i64))
            fields.append(pa.field("record_version", pa.int32()))
            fields.append(field)
        else:
            fields.append(field)
    return pa.schema(fields)


_SCHEMA: Any = None


def _schema_v5() -> Any:
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = flight_schema_v5()
    return _SCHEMA


def _encodable(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _field(name: str, value: Any, integer: bool) -> Any:
    """The struct field's value, or raise for a refused one."""
    if value is None:
        raise TypeError("an explicit null is refused")
    if integer:
        if isinstance(value, bool) or not isinstance(value, int) or not INT64[0] <= value <= INT64[1]:
            raise TypeError("not an int64")
        return value
    converted = v4._float(value)  # raises for a bool, another string; OverflowError for 10**400
    return converted


def _split_structs(value: Any, fields: tuple[str, ...], ints: frozenset[str]) -> tuple[Any, Any]:
    """(the column's pairs or None, what travels in extra_json under the key or None)."""
    if not isinstance(value, dict):
        return None, value
    held: list[tuple[str, Any]] = []
    refused: dict[str, Any] = {}
    for name, entry in value.items():
        if not _encodable(name):
            refused[name] = entry
            continue
        if not isinstance(entry, dict):
            held.append((name, None))
            refused[name] = entry
            continue
        row: dict[str, Any] = {}
        bad: dict[str, Any] = {}
        for key, field_value in entry.items():
            if key not in fields:
                bad[key] = field_value
                continue
            try:
                row[key] = _field(key, field_value, key in ints)
            except Exception:  # noqa: BLE001  refused: a type, an overflow
                bad[key] = field_value
        held.append((name, row))
        if bad:
            refused[name] = bad
    if not held and value:
        return None, value
    return held, refused or None


def _float32_exact(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return isinstance(value, str) and value in v4._NON_FINITE
    if not math.isfinite(value):
        return True
    try:
        return struct.unpack("f", struct.pack("f", value))[0] == value
    except OverflowError:
        return False


def _map_holdable(grid: Any) -> bool:
    return isinstance(grid, list) and all(
        isinstance(row, list) and all(_float32_exact(v) for v in row) for row in grid
    )


def _hist_holdable(entry: Any) -> bool:
    if not isinstance(entry, dict) or tuple(entry) != HIST_KEYS:
        return False
    lo, hi, counts, nonfinite = (entry[k] for k in HIST_KEYS)
    number = lambda v: not isinstance(v, bool) and (isinstance(v, (int, float)) or v in v4._NON_FINITE)  # noqa: E731
    integer = lambda v: not isinstance(v, bool) and isinstance(v, int) and INT64[0] <= v <= INT64[1]  # noqa: E731
    return number(lo) and number(hi) and isinstance(counts, list) and all(integer(c) for c in counts) and integer(nonfinite)


def split_probe(probe: Any) -> tuple[Any, Any, Any]:
    """(light probe for probe_json, probe_maps pairs or None, probe_histograms pairs or None)."""
    if not isinstance(probe, dict):
        return probe, None, None
    light = dict(probe)
    maps = None
    attention = probe.get("attention")
    if isinstance(attention, dict):
        pairs, kept = [], {}
        for label, entry in attention.items():
            if isinstance(entry, dict) and "map" in entry and _encodable(label) and _map_holdable(entry["map"]):
                pairs.append((label, [[v4._float(v) for v in row] for row in entry["map"]]))
                kept[label] = {k: v for k, v in entry.items() if k != "map"}
            else:
                kept[label] = entry
        if pairs:
            maps, light["attention"] = pairs, kept
    hists = None
    histograms = probe.get("histograms")
    if isinstance(histograms, dict) and histograms and all(
            _encodable(n) and _hist_holdable(h) for n, h in histograms.items()):
        hists = [(name, {"lo": v4._float(h["lo"]), "hi": v4._float(h["hi"]), "counts": list(h["counts"]),
                         "nonfinite": h["nonfinite"]}) for name, h in histograms.items()]
        del light["histograms"]
    return light, maps, hists


def record_to_row_v5(record: Mapping[str, Any], layout: str = "v5", token_entropy: bool = False) -> dict[str, Any]:
    scalar_record = record.get("schema_version") == 2
    # A scalar-only record's `record` and `seq` are its row's `seq` column (contract section 4.4).
    rest = {k: v for k, v in record.items() if k not in _MOVED and not (scalar_record and k in ("seq", "record"))}
    # With Arrow's check, as `flight.rows_batch` redoes a row Arrow refuses (wave AJ): the result for a
    # row Arrow takes is the same either way.
    row = v4._row(rest, _schema_v5())
    extra = json.loads(row.pop("extra_json")) if row.get("extra_json") else {}
    for column in ("layers_json", "moe_stats_json"):
        row.pop(column)
    row["layers"] = row["moe_stats"] = row["probe_json"] = row["probe_maps"] = row["probe_histograms"] = None
    moe_fields = MOE_FIELDS + (("token_entropy",) if token_entropy else ())
    for key, fields, ints in (("layers", LAYER_FIELDS, frozenset()), ("moe_stats", moe_fields, MOE_INT)):
        if key not in record:
            continue
        if record[key] is None:
            extra[key] = None
            continue
        held, refused = _split_structs(record[key], fields, ints)
        row[key] = held
        if refused is not None:
            extra[key] = refused
    if "probe" in record:
        if record["probe"] is None:
            extra["probe"] = None
        else:
            light, maps, hists = split_probe(record["probe"])
            row["probe_json"] = v4._ordered_json(light)
            row["probe_maps"], row["probe_histograms"] = maps, hists
    version = record.get("schema_version", 0)
    row["record_version"] = record["schema_version"] if "schema_version" in record else None
    row["seq"] = record.get("seq") if record.get("schema_version") == 2 else None
    if layout == "v5":
        row["schema_version"] = 2
    else:
        row["schema_version"] = version
    row["extra_json"] = v4._json(extra) if extra else None
    return row


def _plain(value: Any) -> Any:
    return v4._plain(value)


def _struct_back(value: Any) -> Any:
    if value is None:
        return None
    return {k: _plain(v) for k, v in value.items() if v is not None}


def row_to_record_v5(row: Mapping[str, Any]) -> dict[str, Any]:
    plain = {k: v for k, v in row.items()
             if k not in ("layers", "moe_stats", "probe_json", "probe_maps", "probe_histograms", "seq",
                          "record_version", "extra_json")}
    plain["layers_json"] = plain["moe_stats_json"] = plain["probe_json"] = None
    extra = json.loads(row["extra_json"]) if row.get("extra_json") else {}
    # Everything but the struct maps' refused fields goes through today's reader, which merges wave
    # AJ's refused map entries and puts every other key of extra_json on the record.
    rest = {k: v for k, v in extra.items() if k not in ("layers", "moe_stats")}
    plain["extra_json"] = json.dumps(rest) if rest else None
    extra = {k: v for k, v in extra.items() if k in ("layers", "moe_stats")}
    plain["schema_version"] = row.get("record_version") or 0
    record = v4.row_to_record(plain)
    if row.get("record_version") is None:
        record.pop("schema_version", None)
    rebuilt: dict[str, Any] = {}
    for key in ("layers", "moe_stats"):
        pairs = row.get(key)
        if pairs is None:
            continue
        value = {name: _struct_back(entry) for name, entry in pairs}
        refused = extra.pop(key, None)
        if isinstance(refused, dict):
            for name, more in refused.items():
                if name in value and value[name] is not None and isinstance(more, dict):
                    value[name] = {**value[name], **more}
                else:
                    value[name] = more  # a placeholder's place, or appended
        rebuilt[key] = value
    if row.get("probe_json") is not None:
        probe = json.loads(row["probe_json"])
        if row.get("probe_histograms") is not None and isinstance(probe, dict):
            hists = {name: {"lo": _plain(h["lo"]), "hi": _plain(h["hi"]), "counts": list(h["counts"]),
                            "nonfinite": h["nonfinite"]} for name, h in row["probe_histograms"]}
            probe = {"histograms": hists, **probe}
        if row.get("probe_maps") is not None and isinstance(probe, dict):
            attention = probe.get("attention")
            for label, grid in row["probe_maps"]:
                attention[label] = {**attention[label], "map": [[_plain(v) for v in r] for r in grid]}
        rebuilt["probe"] = probe
    # Put the moved keys back where the capture writes them: after `ts` for layers; the rest at the end.
    out: dict[str, Any] = {}
    for key, value in record.items():
        out[key] = value
        if key == "ts" and "layers" in rebuilt:
            out["layers"] = rebuilt.pop("layers")
    out.update(rebuilt)
    for key, value in extra.items():  # a struct map's whole value, when the column held none of it
        out[key] = value
    if row.get("seq") is not None:
        # A scalar-only record (section 4.4): `record` comes back from `seq`, and the columns a step
        # record always carries (`flight._ALWAYS`) come back only when the row holds a value.
        for key in v4._ALWAYS:
            if out.get(key) is None and row.get(key) is None:
                out.pop(key, None)
        out = {"schema_version": out.pop("schema_version"), "record": "scalars", **out, "seq": row["seq"]}
    return out


def batch_v5(records: list[Mapping[str, Any]], layout: str = "v5", schema: Any = None) -> Any:
    import pyarrow as pa

    schema = schema or flight_schema_v5()
    with_te = "token_entropy" in schema.field("moe_stats").type.item_type.names
    return pa.RecordBatch.from_pylist([record_to_row_v5(r, layout, with_te) for r in records], schema=schema)


def same(a: Any, b: Any) -> bool:
    """Equal as `==`, and every object's keys in the same order at every depth."""
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b and type(a) is type(b) or (a == b and {type(a), type(b)} <= {int, float})
