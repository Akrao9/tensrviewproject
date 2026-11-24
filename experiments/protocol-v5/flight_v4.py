"""Protocol 4's row layout, frozen for WAVE_AL_CONTRACT.md's experiments. Not product code.

A verbatim copy of the record <-> Arrow row functions of `sdk/src/tensorquill/flight.py` at
`5c286ed` (trial/ai-wave before lane AL-1 merged): `flight_schema`, `record_to_row`, `_row`,
`records_to_batch`, `rows_batch`, `row_to_record` and their helpers, from `_NON_FINITE` to
`_MAP_COLUMNS`. Nothing else of that file is copied, and nothing here is changed but this docstring
and the imports below.

Since lane AL-1 (`92e471a`), `tensorquill.flight` writes protocol 5's layout. This module keeps
protocol 4's for what the experiments still need it for:

- `prototype.py`, the layout the contract states and measured, which is built on protocol 4's row
  (it takes that row without `layers`, `moe_stats` and `probe`, and adds the typed columns);
- the "today" arm of `measure.py` (`v4`): protocol 4's rows, sent with batch metadata version 1, as
  an SDK of before the wave sends them to a protocol 5 server;
- `captures.py --fixtures` and `live.py`, whose "today" is the base's.

It imports `identity.check` and `sink._finite` from the SDK as that file did; neither changed in
step 1 of wave AL.
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Iterable, Mapping

from tensorquill.identity import check
from tensorquill.sink import _finite

_NON_FINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


# --- Record <-> Arrow row conversion -------------------------------------------------


def _float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("bool is not a float")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value in _NON_FINITE:
        return _NON_FINITE[value]  # JsonlSink writes non-finite numbers as these strings
    raise TypeError(f"not a number: {value!r}")


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"not an integer: {value!r}")
    return value


def _str(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"not a string: {value!r}")
    return value


def _bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"not a bool: {value!r}")
    return value


def _list(item: Callable[[Any], Any]) -> Callable[[Any], list[Any]]:
    def convert(value: Any) -> list[Any]:
        if not isinstance(value, list):
            raise TypeError(f"not a list: {value!r}")
        return [item(v) for v in value]

    return convert


def _map(item: Callable[[Any], Any]) -> Callable[[Any], list[tuple[str, Any]]]:
    def convert(value: Any) -> list[tuple[str, Any]]:
        if not isinstance(value, dict):
            raise TypeError(f"not an object: {value!r}")
        return [(_str(k), item(v)) for k, v in value.items()]

    return convert


def _json(value: Any) -> str:
    """`value` as standard JSON text. A non-finite float inside it (a line holding a bare `NaN` or
    `Infinity`, which a JSON reader accepts but `JsonlSink` never writes) is written as the string
    `JsonlSink` writes, "nan", "inf" or "-inf", which both engines read (L20): before, the whole
    record was refused and never sent."""
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)
    except ValueError:
        return json.dumps(_finite(value), separators=(",", ":"), sort_keys=True, allow_nan=False)


def _ordered_json(value: Any) -> str:
    """`_json` with the object's key order kept. For `layers` that order is the model's, which the
    detectors use to say which block showed a change first; sorted, `model.layers.10` would
    come before `model.layers.2`."""
    try:
        return json.dumps(value, separators=(",", ":"), allow_nan=False)
    except ValueError:
        return json.dumps(_finite(value), separators=(",", ":"), allow_nan=False)


# JSONL key -> (Arrow column, converter). A value that does not fit its column's type, and
# any key not listed here, travels in `extra_json`, so no field is ever dropped. For the maps of
# `_SPLIT`, only the entries their column cannot hold do (wave AJ).
_FIELDS: dict[str, tuple[str, Callable[[Any], Any]]] = {
    "ts": ("ts", _float),
    "lr": ("lr", _list(_float)),
    "grad_norm": ("grad_norm", _float),
    "clip_grad_norm": ("clip_grad_norm", _float),
    "weight_norm": ("weight_norm", _float),
    "weights_unchanged": ("weights_unchanged", _list(_str)),
    "step_skipped": ("step_skipped", _bool),
    "grad_scale": ("grad_scale", _float),
    "scalars": ("scalars", _map(_float)),
    "grad_norms": ("grad_norms", _map(_float)),
    "weight_norms": ("weight_norms", _map(_float)),
    "update_rms": ("update_rms", _map(_float)),
    "moe_top_k": ("moe_top_k", _map(_int)),
    "moe": ("moe_counts", _map(_list(_int))),
    "replica_spread": ("replica_spread", _map(_list(_float))),
    "layers": ("layers_json", _ordered_json),
    "moe_stats": ("moe_stats_json", _json),
    "probe": ("probe_json", _ordered_json),  # attention modules in the model's order, as for layers
    "data": ("data_json", _json),
    "checkpoint": ("checkpoint_json", _json),
    "data_wait": ("data_wait", _float),
    "step_time": ("step_time", _float),
    "system": ("system_json", _json),
}
_KEY_COLUMNS = ("schema_version", "run_id", "rank", "attempt", "step")
# Capture always writes these two, as null when there is no value, so a null column reads
# back as a present `None`. Every other null column reads back as an absent key.
_ALWAYS = ("ts", "grad_norm")
_COLUMNS = (*_KEY_COLUMNS, *(column for column, _ in _FIELDS.values()), "extra_json")


def flight_schema() -> Any:
    """The Arrow schema of one step per row (`schema_version` 1)."""
    import pyarrow as pa

    f64 = pa.float64()
    return pa.schema(
        [
            pa.field("schema_version", pa.int32(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("rank", pa.int32(), nullable=False),
            pa.field("attempt", pa.int64(), nullable=False),  # 0: written before attempts
            pa.field("step", pa.int64(), nullable=False),
            pa.field("ts", f64),
            pa.field("lr", pa.list_(f64)),
            pa.field("grad_norm", f64),
            pa.field("clip_grad_norm", f64),
            pa.field("weight_norm", f64),
            pa.field("weights_unchanged", pa.list_(pa.string())),
            pa.field("step_skipped", pa.bool_()),
            pa.field("grad_scale", f64),
            pa.field("scalars", pa.map_(pa.string(), f64)),
            pa.field("grad_norms", pa.map_(pa.string(), f64)),
            pa.field("weight_norms", pa.map_(pa.string(), f64)),
            pa.field("update_rms", pa.map_(pa.string(), f64)),
            pa.field("moe_top_k", pa.map_(pa.string(), pa.int32())),
            pa.field("moe_counts", pa.map_(pa.string(), pa.list_(pa.int64()))),
            pa.field("replica_spread", pa.map_(pa.string(), pa.list_(f64))),
            pa.field("layers_json", pa.string()),
            pa.field("moe_stats_json", pa.string()),
            pa.field("probe_json", pa.string()),
            pa.field("data_json", pa.string()),
            pa.field("checkpoint_json", pa.string()),
            pa.field("data_wait", f64),
            pa.field("step_time", f64),
            pa.field("system_json", pa.string()),
            pa.field("extra_json", pa.string()),
        ]
    )


def record_to_row(
    record: Mapping[str, Any], schema: Any = None, moved: list[str] | None = None, where: str | None = None,
) -> dict[str, Any]:
    """One JSONL step record as a row of `flight_schema()`.

    The record's identity fields are checked first (`identity.check`, `WAVE_AF_CONTRACT.md`
    section 1): a `schema_version`, `rank`, `attempt` or `step` that is not valid raises
    `identity.MalformedRecordError`, naming `where` when it is given, before `moved` is touched
    (SDK-06). A valid record's row is what it was before the check. The sender's own path
    (`rows_to_send`) makes its rows without this check, as before: a line the server refuses and
    counts stays the server's to refuse and count (live ingest is unchanged, RS10).

    The rest is `_row`'s."""
    check(record, where)
    return _row(record, schema, moved)


def _row(
    record: Mapping[str, Any], schema: Any = None, moved: list[str] | None = None,
) -> dict[str, Any]:
    """`record_to_row` without the identity check.

    A value that does not fit its column's type travels in `extra_json`, as always. So does one the
    column's converter or Arrow cannot hold (lane AD4): an integer too large for a float
    (`float(10**400)` raises `OverflowError`), past int32 or int64, or a string with a lone
    surrogate; Arrow's check runs only when `schema` is given, which `rows_to_send` does for a row
    Arrow refused. Before, any of these cost the whole step. Each such key is added to `moved`.

    A map of `_SPLIT` is held entry by entry when it is not held whole (wave AJ, A3,
    `WAVE_AJ_CONTRACT.md` section 1): the entries its column can hold stay in the column, and only
    the others travel in `extra_json`, under the map's own key. Under `grad_norms` (`_NULL_REFUSED`)
    a refused entry also stays in the column, as null. With no entry held the whole value travels
    there and the column is null, as before. A map held whole is converted as before, so its row is
    unchanged."""
    row: dict[str, Any] = dict.fromkeys(_COLUMNS)
    extra: dict[str, Any] = {}
    for key, value in record.items():
        if key in _KEY_COLUMNS:
            continue
        spec = _FIELDS.get(key)
        if spec is None or (value is None and key not in _ALWAYS):
            extra[key] = value
            continue
        if key == "scalars" and isinstance(value, dict):
            # A logged bool is not a number in either engine (lane W3, DET-16): that one value is left
            # out of the row, and the rest of the map stays in its column for the Rust detectors.
            value = {k: v for k, v in value.items() if not isinstance(v, bool)}
        counted = False  # refused by an overflow or by Arrow, not by its type: `moved` names the key
        try:
            converted = spec[1](value)
        except (TypeError, ValueError):
            converted = _UNHELD
        except Exception:  # noqa: BLE001  OverflowError and the like: the value's, not the step's
            converted, counted = _UNHELD, True
        if converted is not _UNHELD:
            if schema is None or _fits(converted, schema.field(spec[0]).type):
                row[spec[0]] = converted
                continue
            counted = True
        item = _SPLIT.get(key)
        if item is not None and isinstance(value, dict):
            # Wave AJ (A3, WAVE_AJ_CONTRACT.md section 1): an entry the column cannot hold leaves it
            # alone; the rest of the map stays in the column, where the Rust detectors read it.
            held, refused, counted = _split(value, item, None if schema is None else schema.field(spec[0]).type,
                                            key in _NULL_REFUSED)
            if held:
                row[spec[0]] = held
                if refused:
                    extra[key] = refused
            else:  # no entry held: the whole value, as before, and the column stays null
                extra[key] = value
        else:
            extra[key] = value
        if counted and moved is not None:
            moved.append(key)
    row["schema_version"] = _int(record.get("schema_version", 0))
    row["run_id"] = _str(record["run_id"])
    row["rank"] = _int(record.get("rank", 0))
    row["attempt"] = _int(record.get("attempt", 0))
    row["step"] = _int(record["step"])
    if extra:
        row["extra_json"] = _json(extra)
    return row


# The maps held entry by entry (wave AJ, A3), each with its entries' converter. `scalars` keeps
# lane W3's rule (a bool dropped, any other refusal moving the whole map) and is not among them.
_SPLIT: dict[str, Callable[[Any], Any]] = {
    "grad_norms": _float,
    "weight_norms": _float,
    "update_rms": _float,
    "moe_top_k": _int,
    "moe": _list(_int),
    "replica_spread": _list(_float),
}
# The maps whose refused entry stays in the column as null, not absent (WAVE_AJ_CONTRACT.md, section 1):
# a gradient that is listed with no value, as Python reads the raw `true` or string. So an update
# comparison takes the whole-total fallback in both engines (a non-finite norm ends the split), and
# a stalled weight's streak resets on it in both, where an unlisted gradient would leave both alone.
_NULL_REFUSED = frozenset({"grad_norms"})
_UNHELD = object()


def _split(
    value: Mapping[str, Any], item: Callable[[Any], Any], map_type: Any, null_refused: bool = False,
) -> tuple[list[tuple[str, Any]], dict[str, Any], bool]:
    """`value`'s entries as the column's pairs, those the column cannot hold, and whether any of
    those was refused by an overflow or by Arrow (`map_type`, when given: the name is checked with
    the value, as one entry of the map) and not by its type. With `null_refused`, a refused entry
    whose name the column can hold is also kept in the pairs, as null."""
    held: list[tuple[str, Any]] = []
    refused: dict[str, Any] = {}
    counted = False

    def fits(pair: tuple[str, Any]) -> bool:
        return map_type is None or _fits([pair], map_type)

    for name, entry in value.items():
        try:
            pair = (_str(name), item(entry))
        except (TypeError, ValueError):
            pair = None
        except Exception:  # noqa: BLE001  OverflowError and the like
            pair, counted = None, True
        if pair is not None and not fits(pair):
            pair, counted = None, True
        if pair is not None:
            held.append(pair)
            continue
        refused[name] = entry
        if null_refused and isinstance(name, str) and fits((name, None)):
            held.append((name, None))
    return held, refused, counted


def _fits(value: Any, arrow_type: Any) -> bool:
    """Whether Arrow can hold `value` in a column of `arrow_type`."""
    import pyarrow as pa

    try:
        pa.array([value], type=arrow_type)
    except Exception:  # noqa: BLE001  out of range, a lone surrogate, any refusal
        return False
    return True


def records_to_batch(records: Iterable[Mapping[str, Any]]) -> Any:
    records = list(records)
    return rows_batch(records, [record_to_row(r) for r in records], flight_schema())


def rows_batch(records: list[Mapping[str, Any]], rows: list[dict[str, Any]], schema: Any) -> Any:
    """`rows` (`_row`'s of `records`, made without Arrow's check) as one record batch of `schema`.
    When Arrow refuses the batch, each row it refuses alone is made again with the check, so a
    value only Arrow refuses (an integer past its column's type, a name with a lone surrogate)
    travels in `extra_json` as on the sender's path (wave AJ), and the rest of its row is kept. A
    key column Arrow refuses still raises, as before."""
    import pyarrow as pa

    try:
        return pa.RecordBatch.from_pylist(rows, schema=schema)
    except Exception:  # noqa: BLE001  find the rows Arrow refuses, one at a time
        pass
    remade = []
    for record, row in zip(records, rows):
        try:
            pa.RecordBatch.from_pylist([row], schema=schema)
        except Exception:  # noqa: BLE001
            row = _row(record, schema)
        remade.append(row)
    return pa.RecordBatch.from_pylist(remade, schema=schema)


def _plain(value: Any) -> Any:
    """Undo the Arrow representation: maps come back as key-value pairs and non-finite
    floats go back to the strings `JsonlSink` writes."""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, list):
        if value and all(isinstance(v, tuple) and len(v) == 2 for v in value):
            return {k: _plain(v) for k, v in value}
        return [_plain(v) for v in value]
    return value


def row_to_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """The inverse of `record_to_row`: a row read back from the server's Parquet parts
    (`table.to_pylist()`) as the dict that `evidence.py` reads from a `.jsonl` file."""
    record: dict[str, Any] = {}
    if row["schema_version"]:
        record["schema_version"] = row["schema_version"]
    record.update(run_id=row["run_id"], rank=row["rank"])
    if row.get("attempt"):
        record["attempt"] = row["attempt"]
    record["step"] = row["step"]
    for key, (column, convert) in _FIELDS.items():
        value = row.get(column)
        if value is None:
            if key in _ALWAYS:
                record[key] = None
        elif convert in (_json, _ordered_json):
            record[key] = json.loads(value)
        elif column in _MAP_COLUMNS:
            record[key] = {k: _plain(v) for k, v in value}  # an empty map stays a dict
        else:
            record[key] = _plain(value)
    if row.get("extra_json"):
        extra = json.loads(row["extra_json"])
        for key in _SPLIT:
            # The entries the column could not hold, beside those it did (wave AJ, A3).
            if key in record and isinstance(extra.get(key), dict):
                record[key] = {**record[key], **extra.pop(key)}
        record.update(extra)
    return record


_MAP_COLUMNS = frozenset(
    {"scalars", "grad_norms", "weight_norms", "update_rms", "moe_top_k", "moe_counts", "replica_spread"}
)
