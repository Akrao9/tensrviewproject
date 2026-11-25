"""Opt-in streaming of a rank's local `.jsonl` file to `tensorquill-server` over Arrow Flight.

The local file is always written first and stays the source of truth. A daemon thread
tails it from the byte offset saved in `<run_id>.rank<rank>.offset`, sends complete lines
as Arrow record batches on one long-lived `DoPut` stream, and advances the offset file
only when the server acknowledges that the bytes are stored. Nothing here runs on the
training thread, and a server that is down, slow or full only delays the stream: the
sender keeps no backlog in memory and re-reads the file from the saved offset.

`pyarrow` is imported only when an endpoint is set (`pip install pyarrow`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .identity import SCHEMA_VERSIONS, MalformedRecordError, _shown, check, is_scalar_record
from .manifest import METADATA_KEY as MANIFEST_KEY, _say, manifest_to_send
from .sink import _finite

# The numbers of `WAVE_AL_CONTRACT.md` section 3, kept apart. `SCHEMA_VERSION` is the batch metadata's
# (`app_metadata`): 2 for protocol 5's layout. A server that reads only 1 refuses the stream with
# INVALID_ARGUMENT and acknowledges nothing (D6), which the sender says once (`OlderServer`).
# `ROW_VERSION` is a row's `schema_version` column in protocol 5's layout (D7), and the record's own
# version travels beside it in `record_version` (Q1 a). Readers tell a layout by its columns.
SCHEMA_VERSION = 2
ROW_VERSION = 2
BATCH_STEPS = 50
BATCH_SECONDS = 2.0
BATCH_BYTES = 4 * 1024 * 1024
BACKOFF_BASE = 0.1
BACKOFF_CAP = 30.0
HEARTBEAT_SECONDS = 15.0
LAST_BEAT_SECONDS = 1.5  # the bound on `Heartbeat.close()`'s last beat, under the join's 2 s
# At most this long, a new process's first heartbeat waits for the session to name its attempt,
# and its stream waits for that beat, so the server knows a job started again before the stream
# opens (see `Heartbeat`). Background threads only: training never waits for it.
FIRST_BEAT_HOLD_SECONDS = 2.0
STACK_SECONDS = 60.0  # no work for this long: the heartbeat carries where each thread is
STACK_FRAMES = 40  # frames kept of the main thread, innermost first
STACK_OTHER_FRAMES = 4  # and of each other thread
STACK_THREADS = 8  # at most 68 frames in all: under 10 KB a rank in the server's status file
CLOSE_TIMEOUT = 5.0
# At most this long, a stream that failed waits for its ack reader to finish, so a last answer the
# server sent before the status that ended the stream (an ack, or a refusal's range, RS10) is taken
# before the next stream chooses where it starts.
ACK_DRAIN_SECONDS = 1.0
POLL_SECONDS = 0.2
ENDPOINT_ENV = "TENSORQUILL_ENDPOINT"
TOKEN_ENV = "TENSORQUILL_TOKEN"
TLS_CA_ENV = "TENSORQUILL_TLS_CA"

_NON_FINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


# --- Record <-> Arrow row conversion -------------------------------------------------


def _float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("bool is not a float")
    if isinstance(value, (int, float)):
        value = float(value)
        # One NaN for every NaN, so a line holding a bare `NaN` makes the same row as the sink's "nan"
        # (L20), now that a struct field holds the float and not JSON text.
        return _NON_FINITE["nan"] if math.isnan(value) else value
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
    # Protocol 5's typed columns (`WAVE_AL_CONTRACT.md` sections 1 and 2): converted by `_structs` and
    # `_split_probe`, never by the converter named here, and read back by `_structs_back` and `_probe_back`.
    "layers": ("layers", _ordered_json),
    "moe_stats": ("moe_stats", _json),
    "probe": ("probe_json", _ordered_json),  # its light part; the maps and histograms have columns of their own
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
# A scalar-only record's keys that its row holds in the `seq` column alone (section 4.4).
_SCALAR_KEYS = ("record", "seq")
_COLUMNS = (*_KEY_COLUMNS, *(column for column, _ in _FIELDS.values()), "probe_maps", "probe_histograms",
            "seq", "record_version", "extra_json")

# The struct maps of section 1.2: their fields in the capture's order (`capture.py` `STATS`,
# `_router_stats`), and the fields that are Int64. Readers look a field up by name, so a field added
# later (K3's `token_entropy`) is absent in an older part and needs no row version (D3).
LAYER_FIELDS = ("mean", "std", "abs_max", "zero_frac")
MOE_FIELDS = ("routed_tokens", "entropy", "diversity", "dead_experts", "dead_frac")
_STRUCT_MAPS: dict[str, tuple[tuple[str, ...], frozenset[str]]] = {
    "layers": (LAYER_FIELDS, frozenset()),
    "moe_stats": (MOE_FIELDS, frozenset({"routed_tokens", "dead_experts"})),
}
# Protocol 4's JSON columns for the same keys: never written now, still read (section 3).
_JSON_COLUMNS = {"layers": "layers_json", "moe_stats": "moe_stats_json"}
_HIST_KEYS = ("lo", "hi", "counts", "nonfinite")
_INT64 = (-(2**63), 2**63 - 1)


def flight_schema() -> Any:
    """The Arrow schema of one step per row, in protocol 5's layout (row `schema_version` 2,
    `WAVE_AL_CONTRACT.md` sections 1 to 3): `layers` and `moe_stats` as maps of structs,
    the probe's maps and histograms in `probe_maps` and `probe_histograms` beside `probe_json`, the
    scalar-only record's `seq` (null on a step row; section 4.4), and the record's own version in
    `record_version`. No map is marked `keys_sorted`: entries keep the record's order (D1)."""
    import pyarrow as pa

    f64, f32, i64 = pa.float64(), pa.float32(), pa.int64()
    layer = pa.struct([pa.field(name, f64) for name in LAYER_FIELDS])
    ints = _STRUCT_MAPS["moe_stats"][1]
    moe = pa.struct([pa.field(name, i64 if name in ints else f64) for name in MOE_FIELDS])
    histogram = pa.struct([pa.field("lo", f64), pa.field("hi", f64), pa.field("counts", pa.list_(i64)),
                           pa.field("nonfinite", i64)])
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
            pa.field("layers", pa.map_(pa.string(), layer)),
            pa.field("moe_stats", pa.map_(pa.string(), moe)),
            pa.field("probe_json", pa.string()),
            pa.field("probe_maps", pa.map_(pa.string(), pa.list_(pa.list_(f32)))),
            pa.field("probe_histograms", pa.map_(pa.string(), histogram)),
            pa.field("data_json", pa.string()),
            pa.field("checkpoint_json", pa.string()),
            pa.field("data_wait", f64),
            pa.field("step_time", f64),
            pa.field("system_json", pa.string()),
            pa.field("seq", pa.int64()),  # a scalar-only record's; a server stores such rows apart (section 4.5)
            pa.field("record_version", pa.int32()),
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

    A scalar-only record (`identity.is_scalar_record`) whose `seq` the column cannot hold raises the same error,
    naming `seq` (section 4.4).

    The rest is `_row`'s."""
    check(record, where)
    if is_scalar_record(record):
        _seq(record, where)
    return _row(record, schema, moved)


def _seq(record: Mapping[str, Any], where: str | None = None) -> int:
    """A scalar-only record's `seq`, which its row's `seq` column holds: an integer from 0 that Int64
    holds. Else `MalformedRecordError` naming `seq`: such a row would carry a null `seq`, which a server
    stores as a training row in a part, moving the step watermark (D8), so the line is refused instead,
    and on the sender's path skipped and counted like a line whose key column cannot be stored."""
    value = record.get("seq", MalformedRecordError.ABSENT)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _INT64[1]:
        raise MalformedRecordError("seq", value, where,
                                   'a scalar-only record ("record": "scalars") needs a seq, an integer of 0 or more '
                                   "that Int64 holds")
    return value


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
    unchanged.

    `layers` and `moe_stats` are maps of structs and the probe is split by column (protocol 5,
    `WAVE_AL_CONTRACT.md` sections 1 and 2; `_structs`, `_split_probe`). The row's `schema_version`
    is `ROW_VERSION` and the record's own version is in `record_version` (section 3, Q1 a): null when
    the record has no `schema_version` key. A record of a version this package does not read keeps
    that version as its row's too, so a server drops and counts it as before (RS10) and never stores
    it as a row it reads.

    A scalar-only record (`identity.is_scalar_record`: version 2 and `"record": "scalars"`, the one rule
    every reader asks; section 4.4) converts like any line, with its `seq` in the
    `seq` column (null on every other row) and its `record` key in no column: `row_to_record` restores
    it from the non-null `seq`. Its `scalars` follow wave W3's rule, as a step record's. A null travels
    in its row's columns only as an absent key: an explicit `null`, even for `ts` or `grad_norm`
    (`_ALWAYS`, which a step row reads back as present), travels in `extra_json`, so its row gives the
    record back exactly. A `seq` the column cannot hold raises `MalformedRecordError` (`_seq`)."""
    row: dict[str, Any] = dict.fromkeys(_COLUMNS)
    extra: dict[str, Any] = {}
    scalar = is_scalar_record(record)
    if scalar:
        row["seq"] = _seq(record)
    always = () if scalar else _ALWAYS
    for key, value in record.items():
        if key in _KEY_COLUMNS or (scalar and key in _SCALAR_KEYS):
            continue
        spec = _FIELDS.get(key)
        if spec is None or (value is None and key not in always):
            extra[key] = value
            continue
        if key in _STRUCT_MAPS:
            held, refused, counted = _structs(value, *_STRUCT_MAPS[key])
            row[spec[0]] = held
            if refused is not _UNHELD:
                extra[key] = refused
            if counted and moved is not None:
                moved.append(key)
            continue
        if key == "probe":
            row["probe_json"], row["probe_maps"], row["probe_histograms"] = _split_probe(value)
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
    version = _int(record.get("schema_version", 0))
    row["schema_version"] = ROW_VERSION if version in SCHEMA_VERSIONS else version
    row["record_version"] = version if "schema_version" in record else None
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


def _encodable(name: Any) -> bool:
    """Whether `name` can be an Arrow map's key: text that UTF-8 can encode (not a lone surrogate)."""
    if not isinstance(name, str):
        return False
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _int64(value: Any) -> int:
    """An Int64 struct field's value: an integer that is not a bool (TypeError), within int64
    (OverflowError, counted as `moved`)."""
    value = _int(value)
    if not _INT64[0] <= value <= _INT64[1]:
        raise OverflowError(f"{value} is past int64")
    return value


def _structs(
    value: Any, fields: tuple[str, ...], ints: frozenset[str],
) -> tuple[list[tuple[str, Any]] | None, Any, bool]:
    """A struct map (`layers`, `moe_stats`) as the column's pairs, what travels in `extra_json` under
    its key (`_UNHELD` for nothing), and whether a field was refused by an overflow or by Arrow, which
    `moved` names once per row (`WAVE_AL_CONTRACT.md` section 1.4, wave AJ's rule extended field by field):

    - every field held: the struct;
    - a field refused (a bool, a string but the sink's three, an explicit null, a number its type
      cannot hold, a float in an Int64 field, a field the struct does not name): null in the struct,
      and only that field under the entry's name;
    - a field missing: null in the struct, nothing else, so it reads back absent;
    - an entry that is not an object: a null struct in its place, and the entry whole;
    - a name Arrow cannot encode: not in the map, and the entry whole (read after the column's entries);
    - a value that is not an object, or one no entry of which is held: no column, and the value whole;
    - `{}`: an empty map.

    Entries keep the record's order, and nothing sorts them. Decided here, in Python, so a row is the
    same with or without Arrow's check."""
    if not isinstance(value, dict):
        return None, value, False
    held: list[tuple[str, Any]] = []
    refused: dict[str, Any] = {}
    counted = False
    for name, entry in value.items():
        if not _encodable(name):
            refused[name] = entry
            counted = counted or isinstance(name, str)  # Arrow's refusal of the text, not a type's
            continue
        if not isinstance(entry, dict):
            held.append((name, None))
            refused[name] = entry
            continue
        struct_value: dict[str, Any] = {}
        bad: dict[str, Any] = {}
        for field, field_value in entry.items():
            if field not in fields or field_value is None:
                bad[field] = field_value
                continue
            try:
                struct_value[field] = _int64(field_value) if field in ints else _float(field_value)
            except (TypeError, ValueError):
                bad[field] = field_value
            except Exception:  # noqa: BLE001  OverflowError: a number its type cannot hold
                bad[field] = field_value
                counted = True
        held.append((name, struct_value))
        if bad:
            refused[name] = bad
    if value and not held:
        return None, value, counted
    return held, refused if refused else _UNHELD, counted


def _float32_exact(value: Any) -> bool:
    """Whether a map value is exactly a float32 (D10), or one of the sink's non-finite strings."""
    if isinstance(value, str):
        return value in _NON_FINITE
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        as_float = float(value)
        if not math.isfinite(as_float):
            return True
        return struct.unpack("f", struct.pack("f", as_float))[0] == value
    except (OverflowError, struct.error):
        return False


def _number(value: Any) -> bool:
    try:
        return _float(value) is not None
    except Exception:  # noqa: BLE001  a bool, other text, a number too large for a float
        return False


def _whole64(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and _INT64[0] <= value <= _INT64[1]


def _histogram_held(entry: Any) -> bool:
    """Whether `probe_histograms` holds a histogram as written: its four keys in the probe's order, `lo`
    and `hi` numbers, `counts` and `nonfinite` int64 integers."""
    if not isinstance(entry, dict) or tuple(entry) != _HIST_KEYS:
        return False
    counts = entry["counts"]
    return (_number(entry["lo"]) and _number(entry["hi"]) and isinstance(counts, list)
            and all(_whole64(c) for c in counts) and _whole64(entry["nonfinite"]))


def _split_probe(probe: Any) -> tuple[str, list[tuple[str, Any]] | None, list[tuple[str, Any]] | None]:
    """The probe as `probe_json`, `probe_maps` and `probe_histograms` (`WAVE_AL_CONTRACT.md` section 2.4).

    - `probe_maps`, by the module's label: a module's `map` when every value is exactly a float32 or one
      of the sink's non-finite strings, and the label can be a key. Any other map stays where it was.
    - `probe_histograms`: every histogram or none, so their order is never split. One that is not held
      as written keeps them all in `probe_json`, and so does an empty `histograms`.
    - `probe_json`: the rest, in the probe's order, with each moved `map` and a moved `histograms` left
      out. A probe that is not an object is all here."""
    if not isinstance(probe, dict):
        return _ordered_json(probe), None, None
    light = dict(probe)
    maps: list[tuple[str, Any]] | None = None
    attention = probe.get("attention")
    if isinstance(attention, dict):
        pairs: list[tuple[str, Any]] = []
        kept: dict[str, Any] = {}
        for label, entry in attention.items():
            grid = entry.get("map") if isinstance(entry, dict) else None
            if (isinstance(grid, list) and _encodable(label)
                    and all(isinstance(r, list) and all(_float32_exact(v) for v in r) for r in grid)):
                pairs.append((label, [[_float(v) for v in r] for r in grid]))
                kept[label] = {k: v for k, v in entry.items() if k != "map"}
            else:
                kept[label] = entry
        if pairs:
            maps, light["attention"] = pairs, kept
    histograms_held: list[tuple[str, Any]] | None = None
    histograms = probe.get("histograms")
    if (isinstance(histograms, dict) and histograms
            and all(_encodable(name) and _histogram_held(h) for name, h in histograms.items())):
        histograms_held = [(name, {"lo": _float(h["lo"]), "hi": _float(h["hi"]), "counts": list(h["counts"]),
                                   "nonfinite": h["nonfinite"]}) for name, h in histograms.items()]
        del light["histograms"]
    return _ordered_json(light), maps, histograms_held


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
    (`table.to_pylist()`) as the dict that `evidence.py` reads from a `.jsonl` file.

    A row of either layout (`WAVE_AL_CONTRACT.md` section 3), told by its columns, not its number:
    protocol 5's (`record_version`, `layers` and `moe_stats` as maps, `probe_maps`, `probe_histograms`)
    or protocol 4's (`layers_json`, `moe_stats_json`, the whole probe in `probe_json`). So `load_records`
    reads a store of either layout, or a rank with parts of both. A struct's null field reads back
    absent, and the fields it could not hold come back from `extra_json` (section 1.4).

    A row with a non-null `seq` is a scalar-only record's (section 4.4): `record: "scalars"` comes back
    from it, with the keys in the capture's order, and a null column reads back absent. A row with no
    `seq` column (step 1's schema, protocol 4's) or a null one is a step row, read as before."""
    record: dict[str, Any] = {}
    seq = row.get("seq")
    if "record_version" in row:  # protocol 5's layout: the record's own version, null for none
        if row["record_version"] is not None:
            record["schema_version"] = row["record_version"]
    elif row["schema_version"]:
        record["schema_version"] = row["schema_version"]
    if seq is not None:
        record["record"] = "scalars"
    record.update(run_id=row["run_id"], rank=row["rank"])
    if row.get("attempt"):
        record["attempt"] = row["attempt"]
    record["step"] = row["step"]
    if seq is not None:
        record["seq"] = seq
    extra = json.loads(row["extra_json"]) if row.get("extra_json") else {}
    for key, (column, convert) in _FIELDS.items():
        if key in _STRUCT_MAPS:
            pairs = row.get(column)
            if pairs is not None:
                refused = extra.pop(key) if isinstance(extra.get(key), dict) else None
                record[key] = _structs_back(pairs, refused)
            elif row.get(_JSON_COLUMNS[key]) is not None:
                record[key] = json.loads(row[_JSON_COLUMNS[key]])
            continue
        if key == "probe":
            if row.get(column) is not None:
                record[key] = _probe_back(row)
            continue
        value = row.get(column)
        if value is None:
            if key in _ALWAYS and seq is None:
                record[key] = None
        elif convert in (_json, _ordered_json):
            record[key] = json.loads(value)
        elif column in _MAP_COLUMNS:
            record[key] = {k: _plain(v) for k, v in value}  # an empty map stays a dict
        else:
            record[key] = _plain(value)
    if extra:
        for key in _SPLIT:
            # The entries the column could not hold, beside those it did (wave AJ, A3).
            if key in record and isinstance(extra.get(key), dict):
                record[key] = {**record[key], **extra.pop(key)}
        record.update(extra)
    return record


def _structs_back(pairs: Iterable[tuple[str, Any]], refused: Mapping[str, Any] | None) -> dict[str, Any]:
    """A struct map read back in its stored order: each struct without its null fields, an entry's
    refused fields merged into it, a null struct replaced by the entry it held the place of, and a name
    the map does not hold appended (section 1.4)."""
    value: dict[str, Any] = {}
    for name, entry in pairs:
        value[name] = None if entry is None else {k: _plain(v) for k, v in entry.items() if v is not None}
    for name, more in (refused or {}).items():
        if isinstance(value.get(name), dict) and isinstance(more, dict):
            value[name] = {**value[name], **more}
        else:
            value[name] = more
    return value


def _probe_back(row: Mapping[str, Any]) -> Any:
    """The probe from `probe_json`, with `histograms` put back first, as the capture writes it, and each
    module's `map` back on its entry (section 2.4). A row of protocol 4's layout has the whole probe in
    `probe_json`."""
    probe = json.loads(row["probe_json"])
    if not isinstance(probe, dict):
        return probe
    histograms = row.get("probe_histograms")
    if histograms is not None:
        probe = {"histograms": {name: {"lo": _plain(h["lo"]), "hi": _plain(h["hi"]), "counts": list(h["counts"]),
                                       "nonfinite": h["nonfinite"]} for name, h in histograms}, **probe}
    maps = row.get("probe_maps")
    if maps is not None:
        attention = probe.get("attention")
        if not isinstance(attention, dict):
            attention = probe["attention"] = {}
        for label, grid in maps:
            entry = attention.get(label)
            attention[label] = {**(entry if isinstance(entry, dict) else {}), "map": [[_plain(v) for v in r] for r in grid]}
    return probe


_MAP_COLUMNS = frozenset(
    {"scalars", "grad_norms", "weight_norms", "update_rms", "moe_top_k", "moe_counts", "replica_spread"}
)


# --- The local file and its offset ----------------------------------------------------


def offset_path(jsonl_path: str | Path) -> Path:
    """`<run_id>.rank<rank>.jsonl` -> `<run_id>.rank<rank>.offset`."""
    return Path(jsonl_path).with_suffix(".offset")


# The offset file says how many bytes of the local file the server stored, and also which
# server (its address, the run and the rank) and which file: a digest of the file's first bytes
# and of the bytes just before the offset. An append-only file never changes those bytes, so a
# sender resumes at the offset only when the server (its store, or for a server that names none its
# address; see below), the run, the rank and both digests match;
# anything else (another server, a file deleted and written again, a file rewritten in place,
# a file shorter than the offset, an offset file from before the binding) sends the file again
# from its start, which is safe because the server drops the steps it already has. The file's
# inode is not part of it: a copy of the same bytes resumes, which is right, and a new file under
# the same inode is told apart by its bytes.
#
# Which store it is (L3). A server that names its store returns `store_id` with each ack
# (`{"end_byte_offset": N, "store_id": "..."}`), and the offset file keeps it. The saved offset is
# then bound to the store, not to the address: before a stream reuses a saved offset its first
# message is an empty batch, and the answer to it names the store. The same store reached at
# another address resumes; another store (wiped or replaced behind the same address), or an offset
# saved before the server named its store, sends the file again from its start, said once per
# sender. A server whose ack has no `store_id` (built before it) is bound to the address, as before.
# W&B asks the server for its record of the run before it resumes a file stream
# (`wandb/core/internal/runbranch/resume.go`, `processAllOffsets`); the empty first batch is how
# this sender asks, with nothing but the ack the server already sends.
OFFSET_VERSION = 2
HEAD_BYTES = 64 * 1024
TAIL_BYTES = 4 * 1024


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _windows(path: Path, offset: int) -> tuple[bytes, bytes]:
    """`path`'s first `min(offset, HEAD_BYTES)` bytes and the `TAIL_BYTES` before `offset`.
    Raises OSError, or ValueError when the file is shorter than `offset`."""
    with open(path, "rb") as file:
        head = file.read(min(offset, HEAD_BYTES))
        start = max(0, offset - TAIL_BYTES)
        file.seek(start)
        tail = file.read(offset - start)
    if len(tail) != offset - start or len(head) != min(offset, HEAD_BYTES):
        raise ValueError(f"{path.name} is shorter than {offset} bytes")
    return head, tail


def file_identity(path: Path, offset: int) -> dict[str, str]:
    """What the offset file keeps of `path` at `offset`: the digests of its first bytes and of
    the bytes just before the offset."""
    head, tail = _windows(path, offset)
    return {"head": _digest(head), "tail": _digest(tail)}


def read_binding(path: Path) -> dict[str, Any]:
    """The offset file as written by `write_offset`: `{"offset": int, ...}`, with the
    destination and file identity when it has them. `{"offset": 0}` when there is none."""
    try:
        text = path.read_text(encoding="ascii").strip()
    except (OSError, ValueError):
        return {"offset": 0}
    try:
        document = json.loads(text) if text else 0
    except ValueError:
        return {"offset": 0}
    if isinstance(document, dict):
        offset = document.get("offset")
        if isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0:
            return document
        return {"offset": 0}
    if isinstance(document, int) and not isinstance(document, bool):
        return {"offset": max(0, document)}  # an offset file from before the binding
    return {"offset": 0}


def _store_of(document: Mapping[str, Any]) -> str | None:
    """The store ID that an ack or an offset file names: a non-empty string, else None."""
    store_id = document.get("store_id")
    return store_id if isinstance(store_id, str) and store_id else None


def read_offset(path: Path) -> int:
    return read_binding(path)["offset"]


def write_offset(path: Path, offset: int, binding: Mapping[str, Any] | None = None) -> None:
    """Write and rename, so a crash leaves either the old offset or the new one. The scratch
    file is named for this process and thread, so two writers in one directory (a live sender
    and a `send_file` of the same file, say) never write into each other's scratch file.
    Without `binding` the offset names no server and no file, and a sender reading it starts
    again from the start of the file."""
    text = json.dumps({"version": OFFSET_VERSION, "offset": offset, **binding}) if binding else f"{offset}"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text + "\n", encoding="ascii")
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def read_lines(path: Path, offset: int, max_lines: int, max_bytes: int) -> tuple[list[bytes], int]:
    """Up to `max_lines` complete lines starting at byte `offset`, and the offset just after
    them. A trailing line with no newline yet is never returned: the writer may still be in
    the middle of it. One line longer than `max_bytes` is returned alone."""
    try:
        with open(path, "rb") as file:
            file.seek(offset)
            data = file.read(max_bytes)
            if b"\n" not in data:
                if len(data) < max_bytes:
                    return [], offset
                while True:  # one line longer than the cap: read through its newline
                    more = file.read(1024 * 1024)
                    if not more:
                        return [], offset
                    cut = more.find(b"\n")
                    if cut >= 0:
                        data += more[: cut + 1]
                        break
                    data += more
    except OSError:
        return [], offset
    lines = data.split(b"\n")[:-1][:max_lines]  # the last piece is the incomplete tail
    return lines, offset + sum(len(line) + 1 for line in lines)


def parse_lines(
    lines: Iterable[bytes], run_id: str, places: list[int] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """The step records among `lines`, and how many lines were skipped. A line is skipped
    when it is not a JSON object of this run: for example what a killed process left before the
    next process appended to the same file. `places`, when given, gets the index in `lines` of
    each record returned.

    Whether a record's `step` is valid is `rows_to_send`'s to decide, by `identity.check` (wave AG):
    before, a line of this run whose `step` was not an integer was skipped here, and nothing said."""
    records: list[dict[str, Any]] = []
    skipped = 0
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            ok = isinstance(record, dict) and record.get("run_id") == run_id
        except ValueError:
            ok = False
        if ok:
            records.append(record)
            if places is not None:
                places.append(index)
        else:
            skipped += 1
    return records, skipped


# The identity fields the sender checks before it sends a line (`WAVE_AF_CONTRACT.md` section 10).
# Not `schema_version`: a version the server does not read is sent, for the server to count (RS10).
SENDER_FIELDS = ("rank", "attempt", "step")
_KEY_DEFAULTS = {"schema_version": 0, "rank": 0, "attempt": 0}


def _unheld_key(record: Mapping[str, Any], schema: Any) -> tuple[str, Any]:
    """The key column whose value made Arrow refuse `record`'s row, and that value: a value that
    is not an integer (or not text for `run_id`), or one too large for its column."""
    for key in _KEY_COLUMNS:
        value = record.get(key, _KEY_DEFAULTS.get(key, MalformedRecordError.ABSENT))
        if key == "run_id":
            if not isinstance(value, str) or not _fits(value, schema.field(key).type):
                return key, value
        elif isinstance(value, bool) or not isinstance(value, int) or not _fits(value, schema.field(key).type):
            return key, value
    if is_scalar_record(record):
        try:
            _seq(record)
        except MalformedRecordError as error:
            return error.field, error.value
    return "a key column", None


def rows_to_send(
    records: Iterable[Mapping[str, Any]], schema: Any, moved: list[tuple[int, str]] | None = None,
    refused: list[tuple[int, str, Any]] | None = None,
) -> tuple[Any, list[int], int]:
    """`records` as one record batch of `schema`, the steps it holds, and how many records were
    left out for a key: a `rank`, `attempt` or `step` that `identity.check` does not take (negative,
    not an integer, or a missing `step`; wave AG, `WAVE_AF_CONTRACT.md` section 10), or a key
    column whose value cannot be stored (a null or a string where a key column needs an integer,
    an integer too large for its column). Such a line is skipped and counted like a line that is
    not JSON, and the lines after it are sent. Were it kept, every retry would start again at it
    and the sender would never get past it: the server refuses a whole batch holding a negative
    `attempt` or `rank`, with no range to move past. `refused`, when given, gets `(index in
    records, field, value)` for each record left out, so the sender can say which line it was. A
    `schema_version` the server does not read is sent, for the server to count (RS10).
    (A NaN inside a nested object is sent as "nan", L20.)

    Any other field whose value cannot be stored in its column goes in `extra_json` with the rest
    of the row sent (lane AD4; before, the whole step was left out), and `(step, key)` is added to
    `moved`: the server reads such a value as absent from its column, and the sender says so.

    The rows are `_row`'s, so a valid record's row is what `record_to_row` makes of it."""
    import pyarrow as pa

    def strict(record: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]] | None:
        """The row with every field Arrow refuses moved to `extra_json`, and those fields' keys;
        None if a key column is refused."""
        keys: list[str] = []
        try:
            row = _row(record, schema, keys)
            pa.RecordBatch.from_pylist([row], schema=schema)
        except Exception:  # noqa: BLE001  a key column the schema cannot hold: the line is skipped
            return None
        return row, keys

    def leave_out(index: int, record: Mapping[str, Any], field: str | None = None, value: Any = None) -> None:
        if refused is not None:
            try:
                if field is None:
                    field, value = _unheld_key(record, schema)
                refused.append((index, field, value))
            except Exception:  # noqa: BLE001  naming the line never costs the batch
                refused.append((index, "a key column", None))

    kept: list[tuple[int, Mapping[str, Any], dict[str, Any], list[str]]] = []
    skipped = 0
    for index, record in enumerate(records):
        try:
            check(record, None, SENDER_FIELDS)
        except MalformedRecordError as error:
            skipped += 1
            leave_out(index, record, error.field, error.value)
            continue
        keys: list[str] = []
        try:
            kept.append((index, record, _row(record, None, keys), keys))
        except Exception:  # noqa: BLE001  any value that cannot be converted is the line's, not the stream's
            found = strict(record)
            if found is None:
                skipped += 1
                leave_out(index, record)
            else:
                kept.append((index, record, *found))
    try:
        batch = pa.RecordBatch.from_pylist([row for _, _, row, _ in kept], schema=schema)
    except Exception:  # noqa: BLE001  find the rows Arrow refuses, one at a time
        alone = []
        for index, record, row, keys in kept:
            try:
                pa.RecordBatch.from_pylist([row], schema=schema)
            except Exception:  # noqa: BLE001
                found = strict(record)
                if found is None:
                    skipped += 1
                    leave_out(index, record)
                    continue
                row, keys = found
            alone.append((index, record, row, keys))
        kept = alone
        batch = pa.RecordBatch.from_pylist([row for _, _, row, _ in kept], schema=schema)
    if refused is not None:
        refused.sort(key=lambda item: item[0])
    rows = [row for _, _, row, _ in kept]
    if moved is not None:
        moved.extend((row["step"], key) for _, _, row, keys in kept for key in keys)
    return batch, [row["step"] for row in rows], skipped


class BatchRefused(ConnectionError):
    """The stream ended because the server refused a batch for good and named its byte range (RS10):
    the sender's `last_error` after such a stream. Its bytes are not sent again."""


class OlderServer(ConnectionError):
    """The server refused the stream's batch metadata version (`SCHEMA_VERSION`): it speaks an older
    protocol than this tensorquill (`WAVE_AL_CONTRACT.md` section 5; D6). The sender's `last_error` while
    it does, said once on stderr. Nothing is acknowledged, so every line stays in the local file and
    goes once the server is upgraded; the sender keeps trying."""


# A server's refusal of a batch metadata version it does not read, in either wording: a protocol 4
# server's singular "this server reads version 1", or a protocol 5 server's plural "this server reads
# versions 1 and 2" (both pinned in `tests/fixtures/wave_al/refusal.json`, which the server's own test
# reads too).
_REFUSED_VERSION = re.compile(r"schema_version (-?\d+) is not supported; this server reads versions? "
                              r"(\d+(?:(?:, | and )\d+)*)")


def _older_server_reads(text: str, sent: int = SCHEMA_VERSION) -> str | None:
    """The batch metadata versions a server reads, as its refusal lists them ("1", "1 and 2"), when
    `text` holds its refusal of version `sent` (`ingest.rs`: "schema_version N is not supported; this
    server reads versions 1 and 2", or a protocol 4 server's "... reads version 1", sent as
    INVALID_ARGUMENT); else None."""
    for found in _REFUSED_VERSION.finditer(text):
        if int(found.group(1)) == sent:
            return found.group(2)
    return None


class LocalFileUnreadable(OSError):
    """The local `.jsonl` file cannot be read: it was deleted or moved, or is not readable.
    This is not "nothing left to send": `close()` returns False and says so."""


# --- The sender ------------------------------------------------------------------------


def _location(endpoint: str) -> str:
    if endpoint.startswith("grpc://"):
        return endpoint
    if endpoint.startswith("grpcs://"):
        return "grpc+tls://" + endpoint[len("grpcs://") :]
    raise ValueError(f"endpoint must be grpc://host:port or grpcs://host:port, got {endpoint!r}")


def _read_tls_root_certs(tls_root_certs: bytes | str | os.PathLike[str] | None) -> bytes | None:
    if tls_root_certs is None:
        env_path = os.environ.get(TLS_CA_ENV)
        if not env_path:
            return None
        tls_root_certs = env_path
    if isinstance(tls_root_certs, (bytes, bytearray, memoryview)):
        return bytes(tls_root_certs)
    if isinstance(tls_root_certs, (str, os.PathLike)):
        return Path(tls_root_certs).read_bytes()
    raise TypeError(f"tls_root_certs must be bytes or a file path, got {type(tls_root_certs).__name__}")


class FlightSender:
    """Tails one rank's `.jsonl` file and streams it to the ingest server.

    One `DoPut` stream stays open. Each batch carries `end_byte_offset`, the file position
    just after its last line; the server answers with the offset it has stored durably, and
    only then does the `.offset` file move. After any failure the sender sleeps (full-jitter
    exponential backoff) and starts again from the saved offset, so at most one batch is in
    memory at a time and the server's `(run_id, rank, attempt, step)` filter drops what it already has.
    """

    def __init__(
        self,
        path: str | Path,
        endpoint: str,
        run_id: str,
        rank: int = 0,
        *,
        token: str | None = None,
        tls_root_certs: bytes | str | os.PathLike[str] | None = None,
        world_size: int | None = None,
        manifest: Mapping[str, Any] | None = None,
        batch_steps: int = BATCH_STEPS,
        batch_seconds: float = BATCH_SECONDS,
        batch_bytes: int = BATCH_BYTES,
        poll_seconds: float = POLL_SECONDS,
        backoff_base: float = BACKOFF_BASE,
        backoff_cap: float = BACKOFF_CAP,
        hold: threading.Event | None = None,
        await_attempt: bool = False,
    ) -> None:
        import pyarrow.flight  # noqa: F401  fail here, at init, when pyarrow is missing

        self.path = Path(path)
        self.offset_path = offset_path(self.path)
        self.endpoint = endpoint
        self.run_id = run_id
        self.rank = rank
        self.world_size = world_size  # lets the server name ranks that never connect
        # The run manifest (`manifest.py`), given on rank 0 only. It rides on the first batch
        # of each stream, and the server writes `runs/<run_id>/run.json` from it.
        # A copy too large for a batch's metadata goes without its config, or not at all.
        self.manifest, left_out = manifest_to_send(manifest)
        if left_out:
            # Said, never raised (lane Y6, XC3): under `python -W error` it is logged instead.
            _say(f"tensorquill: {left_out}. The local file has all of it; the run streams as usual.", stacklevel=4)
        self.last_error: BaseException | None = None
        self.skipped_lines = 0
        # Fields sent in `extra_json` because their column cannot hold the value (`rows_to_send`):
        # the step is sent and the server reads that value as absent. Said once per sender.
        self.moved_fields = 0
        self._said_moved = False
        self._said_skipped = False
        # What explains the last stream that ended without this sender having closed it (lane AD4;
        # the plan's unexplained endings, "h2 protocol error", 3 of 22 synthetic runs): set beside
        # `last_error` when a stream fails before `done_writing`, and kept after later progress,
        # which clears `last_error`. See `_note_stream_end`. Holds no token and no header.
        self.last_stream_end: dict[str, Any] | None = None
        # RS10 (detector rules contract, part 8): batches the server refused for good, naming the
        # exact byte range this sender sent. Their bytes are passed over: the saved offset moves to
        # the refused range's end. `last_refused` is the last such range, with its column.
        self.refused_batches = 0
        self.last_refused: dict[str, Any] | None = None
        self._said_refused = False
        self._said_older = False  # a server of an older protocol (`OlderServer`) is said once per sender
        self._life: dict[str, Any] | None = None  # the stream now open: when, and what it carried
        self._location = _location(endpoint)
        self._token = token if token is not None else os.environ.get(TOKEN_ENV)
        self._tls_root_certs = (
            _read_tls_root_certs(tls_root_certs)
            if (tls_root_certs is not None or self._location.startswith("grpc+tls://"))
            else None
        )
        self._batch_steps = batch_steps
        self._batch_seconds = batch_seconds
        self._batch_bytes = batch_bytes
        self._poll_seconds = poll_seconds
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        # The jitter's own generator: drawing from the global `random` during an outage would
        # change what a training script's seeded sampler draws next.
        self._random = random.Random()
        # Where the offset file binds its offset: this server, run and rank (see `write_offset`).
        self._destination = {"endpoint": self._location, "run_id": run_id, "rank": rank}
        self.replayed: str | None = None  # why the last stream started again at byte 0, if it did
        self._binding: dict[str, Any] = {"offset": 0}  # the offset file as the stream started
        self._store_id: str | None = None  # the store this stream's server named, once it has
        self._said_store = False  # a replay for want of the same store is said once per sender
        # Set from the offset file once the server has named its store (`_stream`), never before:
        # an offset that store did not confirm is not counted as confirmed.
        self._acked = 0
        self._head = b""  # the file's first bytes and the bytes just before `sent`, to bind acks
        self._tail = b""
        self._pending: dict[int, dict[str, str]] = {}  # batch end offset -> the file's identity there
        self._starts: dict[int, int] = {}  # batch end offset -> its start, for a refusal's range (RS10)
        self._pending_lock = threading.Lock()
        self._progress = False
        self._reported = False
        self._pid = os.getpid()
        self._closing = threading.Event()
        self._deadline = math.inf
        self._hold = hold  # set once the heartbeat has named the attempt (or given up waiting)
        # The attempt of the session whose process sends this stream (contract 1): every batch names
        # it as `sender_attempt`, so the server can tell a claim from backlog rows of an earlier
        # attempt. Set by the session as it starts (`capture.Session`); `attach` makes the first
        # connection wait for it, `FIRST_BEAT_HOLD_SECONDS` at most, so the first batch names it too.
        self._attempt: int | None = None
        self._named = threading.Event()
        self._await_attempt = await_attempt
        self._thread = threading.Thread(
            target=self._run, name=f"tensorquill-flight-rank{rank}", daemon=True
        )
        self._thread.start()

    @property
    def attempt(self) -> int | None:
        return self._attempt

    @attempt.setter
    def attempt(self, attempt: int | None) -> None:
        self._attempt = attempt
        if attempt is not None:
            self._named.set()

    @property
    def acked_offset(self) -> int:
        """Bytes of the local file that the server has stored."""
        return self._acked

    def unacked_bytes(self) -> int | None:
        """Bytes of the local file the server has not confirmed; None when the file cannot be
        read, which is not "nothing left to send"."""
        try:
            return max(0, self.path.stat().st_size - self._acked)
        except OSError:
            return None

    def close(self, timeout: float = CLOSE_TIMEOUT) -> bool:
        """Send the complete lines that are left and wait up to `timeout` seconds for the
        server to store them. Returns whether the whole file was acknowledged; if not, says
        where the local file is. Call after the local file's writer has been closed.

        In a forked child (a `DataLoader` worker, a `multiprocessing` process) this does
        nothing and returns False: a fork copies this object but not its thread, and the
        process that called `init` is the one that streams the file."""
        if os.getpid() != self._pid:
            return False
        self._deadline = time.monotonic() + timeout
        self._closing.set()
        self._thread.join(timeout)
        left = self.unacked_bytes()
        if left is None:
            print(
                f"tensorquill: {self.path} cannot be read, so whether {self.endpoint} has all of it is not "
                f"known ({self.last_error or 'the file is missing'}).",
                file=sys.stderr,
            )
            return False
        if left:
            reason = f" ({self.last_error})" if self.last_error is not None else ""
            print(
                f"tensorquill: {left} bytes of {self.path} were not confirmed by "
                f"{self.endpoint}{reason}. The local file is complete; streaming resumes from "
                f"{self.offset_path.name} the next time this run ID streams.",
                file=sys.stderr,
            )
        return left == 0

    # Everything below runs on the sender thread.

    def _run(self) -> None:
        # The first connection waits for the heartbeat that names the attempt, and for the session to
        # name its attempt to this sender, a moment at most.
        deadline = time.monotonic() + FIRST_BEAT_HOLD_SECONDS + 1.0
        while (
            ((self._hold is not None and not self._hold.is_set())
             or (self._await_attempt and not self._named.is_set()))
            and not self._closing.is_set() and time.monotonic() < deadline
        ):
            (self._hold if self._hold is not None and not self._hold.is_set() else self._named).wait(0.05)
        ceiling = self._backoff_base  # the next delay's upper bound, doubled after each failure up to the cap
        retried_on_close = False
        while True:
            was_closing = self._closing.is_set()
            refused = self.refused_batches
            try:
                self._stream()
                return
            except BaseException as error:  # noqa: BLE001  the stream must never raise into training
                self.last_error = _without_traceback(error)
                if self.refused_batches != refused and self.last_refused is not None:
                    # RS10: the stream ended because the server refused a batch for good, and said so.
                    named = self.last_refused
                    column = f" (column {named['column']!r})" if named.get("column") else ""
                    self.last_error = BatchRefused(
                        f"the server refused bytes {named['start_byte_offset']} to {named['end_byte_offset']} for good"
                        f"{column}, {self.refused_batches} batch(es) so far; the stream goes on from byte "
                        f"{named['end_byte_offset']} ({self._scrub(str(error))[:200]})"
                    )
                older = self._older(error)
                if older is not None:
                    self.last_error = older
                self._note_stream_end(error)
            if self._progress:
                self._progress = False
                ceiling = self._backoff_base
                self._reported = False
            elif isinstance(self.last_error, OlderServer):
                # Said once per sender, in place of the outage's line: retrying cannot help until the
                # server is upgraded, and the sender keeps trying for when it is.
                self._reported = True
                if not self._said_older:
                    self._said_older = True
                    try:
                        print(f"tensorquill: {self.last_error}. Training continues; said once.", file=sys.stderr)
                    except Exception:  # noqa: BLE001  a closed stderr is not the stream's failure
                        pass
            elif not self._reported and not self._closing.is_set():
                # Say once per outage that nothing is arriving: a wrong token or address
                # would otherwise stay silent until close(). close() reports on its own.
                self._reported = True
                if isinstance(self.last_error, LocalFileUnreadable):
                    message = (f"tensorquill: {self.last_error}, so nothing more of it can be streamed to "
                               f"{self.endpoint}. Training continues; the sender tries the file again later.")
                else:
                    message = (f"tensorquill: streaming to {self.endpoint} is failing ({self.last_error}). Training "
                               f"continues and {self.path} stays complete; the sender keeps retrying.")
                print(message, file=sys.stderr)
            if self._closing.is_set():
                if time.monotonic() >= self._deadline or isinstance(self.last_error, LocalFileUnreadable):
                    return  # close() says the file cannot be read; waiting would not change that
                if not was_closing and not retried_on_close:
                    retried_on_close = True
                    ceiling = self._backoff_base
                    continue
                delay, ceiling = self._backoff(ceiling)
                remaining = self._deadline - time.monotonic()
                if remaining <= 0:
                    return
                time.sleep(min(delay, remaining))
                if time.monotonic() >= self._deadline:
                    return
                continue
            delay, ceiling = self._backoff(ceiling)
            if self._closing.wait(delay):
                if time.monotonic() >= self._deadline:
                    return
                retried_on_close = True
                ceiling = self._backoff_base

    def _backoff(self, ceiling: float) -> tuple[float, float]:
        """Full jitter: a delay drawn below `ceiling`, and the next ceiling, doubled and held at
        the cap, so a long outage keeps retrying every `backoff_cap` seconds at most."""
        ceiling = min(self._backoff_cap, max(0.0, ceiling))
        return self._random.uniform(0.0, ceiling), min(self._backoff_cap, ceiling * 2.0)

    def _resume(self) -> int:
        """Where this stream may start as far as this side can tell: the saved offset when the
        offset file names this run and rank and the file's bytes match its digests, else 0 (the
        server drops what it already has), with `replayed` saying why. A saved offset above 0 is
        used only once the server has named its store (`_bound`), so the address is checked here
        only for an offset saved without a store ID. Raises `LocalFileUnreadable` when the file
        cannot be read."""
        binding = read_binding(self.offset_path)
        self._binding = binding
        offset = binding["offset"]
        try:
            head, tail = _windows(self.path, 0)
            reason = None
            checked = ("run_id", "rank") if _store_of(binding) else ("endpoint", "run_id", "rank")
            if offset and any(binding.get(key) != self._destination[key] for key in checked):
                reason = "the offset file names another server, run or rank" if "endpoint" in binding else (
                    "the offset file names no server or file")
            elif offset:
                try:
                    found_head, found_tail = _windows(self.path, offset)
                except ValueError:
                    reason = f"the file is shorter than its saved offset {offset}"
                else:
                    if (_digest(found_head), _digest(found_tail)) != (binding.get("head"), binding.get("tail")):
                        reason = "the file's bytes are not the ones the offset was saved for"
                    else:
                        head, tail = found_head, found_tail
        except OSError as error:
            raise LocalFileUnreadable(f"{self.path} cannot be read ({error})") from None
        self.replayed = reason
        self._head, self._tail = head, tail
        with self._pending_lock:
            self._pending.clear()
            self._starts.clear()
        return 0 if reason else offset

    def _bound(self, store_id: str | None) -> str | None:
        """Why the saved offset that passed `_resume` must not be used with the store this stream's
        server named in its first ack (`store_id`, None from a server that names none), or None."""
        if store_id is None:  # a server built before the store ID: bound to its address, as before
            if self._binding.get("endpoint") != self._destination["endpoint"]:
                return "the offset file names another server, run or rank"
            return None
        saved = _store_of(self._binding)
        if saved is None:
            return "the offset was saved before the server named its store"
        if saved != store_id:
            return "the server's store is not the one the offset was saved for"
        return None  # the same store, at this address or another

    def _replay(self, reason: str) -> None:
        """Start this stream again at byte 0, for `reason`. A replay for want of the same store is
        said once per sender: the stream may be opened again before the first ack is stored."""
        self.replayed = reason
        self._head, self._tail = b"", b""
        with self._pending_lock:
            self._pending.clear()
            self._starts.clear()
        if "store" in reason and not self._said_store:
            self._said_store = True
            saved, named = _store_of(self._binding), self._store_id
            which = f" (store {named}; the offset was saved for store {saved})" if saved else f" (store {named})"
            print(
                f"tensorquill: {reason}{which}, so {self.path} is sent to {self.endpoint} again from its start, "
                f"once. The server keeps each step once; training is not affected.",
                file=sys.stderr,
            )

    def _note_stream_end(self, error: BaseException) -> None:
        """Keep what would explain a stream that ended before this sender said it was done writing,
        as `last_stream_end`: the error and the gRPC status it names (code and name, from the text
        pyarrow gives, "gRPC client debug context: UNAVAILABLE:..."), its details, what the ack
        reader and `writer.close()` raised (one can hide the other), how long the stream had been
        open, the file bytes, batches and Arrow bytes it sent, the acks it got and the last offset
        acknowledged. Not for a file that cannot be read (that is not the stream's ending) or a
        stream the sender closed. The token is cut out of every text kept; no header is kept.
        Never raises, and changes nothing the sender does."""
        try:
            life = self._life
            self._life = None
            if isinstance(error, LocalFileUnreadable) or (life is not None and life.get("done_writing")):
                return
            text = self._scrub(f"{error}")
            name = _grpc_status_name(text, error)
            opened = life.get("opened") if life else None
            self.last_stream_end = {
                "at": time.time(),
                "error": type(error).__name__,
                "grpc_code": _GRPC_CODES.get(name) if name else None,
                "grpc_status": name,
                "details": text[:500],
                "ack_reader_error": life.get("ack_error") if life else None,
                "writer_close_error": life.get("close_error") if life else None,
                "stream_opened": opened is not None,
                "seconds_open": round(time.monotonic() - opened, 3) if opened is not None else None,
                "start_offset": life.get("start", 0) if life else 0,
                "bytes_sent": (life.get("sent", 0) - life.get("start", 0)) if life else 0,
                "batches_sent": life.get("batches", 0) if life else 0,
                "arrow_bytes_sent": life.get("arrow_bytes", 0) if life else 0,
                "acks": life.get("acks", 0) if life else 0,
                "last_acked_offset": self._acked,
                "sender_closing": self._closing.is_set(),
            }
        except Exception:  # noqa: BLE001  explaining a failure must never be one
            pass

    def _older(self, error: BaseException) -> OlderServer | None:
        """An `OlderServer` when the stream that ended with `error` was refused for its batch metadata
        version, else None. The refusal can be the error itself (the server's status, raised by a write
        or the close), or what the ack reader or `writer.close()` met, behind another error (a stream
        that asked for the store with an empty batch and got no answer). Never raises."""
        try:
            life = self._life or {}
            texts = [f"{error}", f"{life.get('ack_error') or ''}", f"{life.get('close_error') or ''}"]
            reads = next((found for found in map(_older_server_reads, texts) if found is not None), None)
            if reads is None:
                return None
            return OlderServer(
                f"the server at {self.endpoint} speaks an older protocol: upgrade tensorquill-server "
                f"(pip install -U tensorquill-server). It reads batch schema_version{'s' if ' ' in reads else ''} "
                f"{reads} and this tensorquill "
                f"sends version {SCHEMA_VERSION}, so it acknowledges nothing; {self.path} keeps every line, and "
                "they are sent once the server is upgraded")
        except Exception:  # noqa: BLE001  explaining a failure must never be one
            return None

    def _scrub(self, text: str) -> str:
        """`text` with the token cut out, should a server or library ever echo it."""
        if self._token:
            text = text.replace(self._token, "<token>")
        return text

    def _brief(self, error: BaseException | None) -> str | None:
        """`error` as short text, the token cut out. Only text is kept of an error here: its traceback
        holds the stream's frames, the cycle `_without_traceback` is there to avoid. Never raises."""
        if error is None:
            return None
        try:
            return self._scrub(f"{type(error).__name__}: {error}")[:300]
        except Exception:  # noqa: BLE001  an error whose str() raises
            return type(error).__name__

    def _refusal(self, refused: Any) -> None:
        """A batch the server refused for good (RS10, detector rules contract part 8), told on the
        acknowledgement channel before the status that ends the stream:
        `{"start_byte_offset": S, "end_byte_offset": E, "column": ...}`.

        The saved offset moves for two reasons only: the server stored the bytes (`_ack`), or it
        refused, naming exactly the range this sender sent, a batch: S is the last offset the server
        confirmed and E the end of a batch this stream sent that began at S. Then the batch is
        counted (`refused_batches`), E is saved as the offset, as an ack of E would be, and the next
        stream goes on from E without sending those bytes again. Anything else (no range, a start or
        an end that is not this sender's, an answer that cannot be read) moves nothing: the bytes go
        again from the last confirmed offset, as before. Runs on the ack thread."""
        try:
            start, end = refused["start_byte_offset"], refused["end_byte_offset"]
            column = refused.get("column")
        except Exception:  # noqa: BLE001  not a range: nothing moves
            return
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (start, end)) or end <= start:
            return
        with self._pending_lock:
            identity = self._pending.get(end)
            began = self._starts.get(end)
        if identity is None or began != start or start != self._acked:
            return
        store = {"store_id": self._store_id} if self._store_id is not None else {}
        write_offset(self.offset_path, end, {**self._destination, **identity, **store})
        self._acked = end
        with self._pending_lock:
            for done in [done for done in self._pending if done <= end]:
                del self._pending[done]
                self._starts.pop(done, None)
        self.refused_batches += 1
        self.last_refused = {"start_byte_offset": start, "end_byte_offset": end,
                             "column": column if isinstance(column, str) else None}
        self._progress = True  # the offset moved: the next stream goes at once, not after a backoff
        if not self._said_refused:
            self._said_refused = True
            try:
                print(
                    f"tensorquill: {self.endpoint} refused bytes {start} to {end} of {self.path} for good"
                    f"{f' (column {column!r})' if isinstance(column, str) else ''}, so those steps are not stored "
                    f"and are not sent again; the stream goes on from byte {end}. The local file has them. "
                    "Said once; sender.refused_batches counts them.",
                    file=sys.stderr,
                )
            except Exception:  # noqa: BLE001
                pass

    def _say_moved(self, moved: list[tuple[int, str]]) -> None:
        """Count fields sent in `extra_json` because their column cannot hold the value, and say the
        first once per sender. Never raises."""
        self.moved_fields += len(moved)
        if self._said_moved:
            return
        self._said_moved = True
        step, key = moved[0]
        try:
            print(
                f"tensorquill: step {step}'s {key!r} cannot be held by its column in the stream to {self.endpoint} "
                f"(a number out of range, or text Arrow cannot encode), so it is sent in extra_json with the rest of "
                f"the step, and the server reads it as absent. {self.path} has it as written. Said once; "
                "sender.moved_fields counts them.",
                file=sys.stderr,
            )
        except Exception:  # noqa: BLE001  a closed stderr is not the stream's failure
            pass

    def _say_skipped(
        self, start: int, lines: list[bytes], places: list[int], refused: list[tuple[int, str, Any]],
        steps: list[Any],
    ) -> None:
        """Say the first line left out for a key once per sender (wave AG): its byte offset in the
        file, its step, and the field. `start` is the byte offset of `lines[0]`; `places` maps each
        record's index to its line, and `steps` holds each record's `step` as written. Every such
        line is counted in `skipped_lines`. Never raises."""
        if self._said_skipped or not refused:
            return
        self._said_skipped = True
        try:
            index, field, value = refused[0]
            line_index = places[index]
            offset = start + sum(len(line) + 1 for line in lines[:line_index])
            step = steps[index]
            found = "absent" if value is MalformedRecordError.ABSENT else _shown(value)
            why = ("tensorquill reads an integer of 0 or more; the server would refuse the whole batch holding it"
                   if field in SENDER_FIELDS and isinstance(value, int) and not isinstance(value, bool)
                   and value < 0 else "its column in the stream cannot hold it")
            shown_step = "" if step is MalformedRecordError.ABSENT else f" (step {_shown(step)})"
            print(
                f"tensorquill: the line at byte {offset} of {self.path}{shown_step} is not sent to {self.endpoint}: "
                f"its {field} is {found}, and {why}. The lines after it are sent, and the file keeps it as written. "
                "Said once; sender.skipped_lines counts such lines.",
                file=sys.stderr,
            )
        except Exception:  # noqa: BLE001  a closed stderr is not the stream's failure
            pass

    def _sent(self, data: bytes, end: int) -> int:
        """Move the identity windows past `data`, the bytes just sent, which end at `end`, and return
        where they begin: the start a refusal of them must name (RS10), which the batch's metadata
        states too (`start_byte_offset`)."""
        start = end - len(data)
        if len(self._head) < HEAD_BYTES:
            self._head += data[: HEAD_BYTES - len(self._head)]
        self._tail = (self._tail + data[-TAIL_BYTES:])[-TAIL_BYTES:]
        with self._pending_lock:
            self._pending[end] = {"head": _digest(self._head), "tail": _digest(self._tail)}
            self._starts[end] = start  # where the batch that ends here began (RS10)
        return start

    def _ack(self, offset: int, store_id: str | None = None) -> None:
        """Take the server's ack of `offset`, which names `store_id` when the server names its store.
        An ack naming another store than this stream's first ack did ends the stream: its offset
        is not this store's to confirm."""
        if store_id is not None:
            if self._store_id is None:
                self._store_id = store_id
            elif store_id != self._store_id:
                raise ConnectionError(
                    f"the server named store {store_id} after naming store {self._store_id} on the same stream")
        if offset > self._acked:
            with self._pending_lock:
                identity = self._pending.get(offset)
            if identity is None:
                # An offset this stream did not send: the server keeps the highest offset of a
                # stream that ended before it stored it, and acks it on the next stream. When this
                # stream started again at byte 0 because the file was replaced, that offset is the
                # old file's, so it is not taken. Otherwise it is this file's, if the file reaches it.
                if self.replayed is not None:
                    return
                try:
                    identity = file_identity(self.path, offset)
                except (OSError, ValueError):
                    return  # later acks name the batches this stream sends
            store = {"store_id": self._store_id} if self._store_id is not None else {}
            write_offset(self.offset_path, offset, {**self._destination, **identity, **store})
            self._acked = offset
            with self._pending_lock:
                for end in [end for end in self._pending if end <= offset]:
                    del self._pending[end]
                    self._starts.pop(end, None)
            self._progress = True
            self.last_error = None

    def _stream(self) -> None:
        import pyarrow as pa
        import pyarrow.flight as flight

        life: dict[str, Any] = {"opened": None, "start": 0, "sent": 0, "batches": 0, "arrow_bytes": 0, "acks": 0}
        self._life = life  # read by `_note_stream_end` if this stream fails
        saved = self._resume()  # 0 when the file is not the one the offset was saved for
        # A saved offset is used once the server has named its store, in answer to an empty first
        # batch (L3): until then nothing past byte 0 is sent and nothing is counted as confirmed.
        probing = saved > 0
        self._acked = 0
        self._store_id = None
        sent = 0
        schema = flight_schema()
        headers = [(b"authorization", f"Bearer {self._token}".encode())] if self._token else []
        client_kwargs: dict[str, Any] = {
            "generic_options": [("grpc.keepalive_time_ms", 30_000), ("grpc.keepalive_timeout_ms", 10_000)],
        }
        if self._tls_root_certs is not None:
            client_kwargs["tls_root_certs"] = self._tls_root_certs
        client = flight.FlightClient(self._location, **client_kwargs)
        try:
            descriptor = flight.FlightDescriptor.for_path("runs", self.run_id, f"rank={self.rank}")
            writer, reader = client.do_put(descriptor, schema, flight.FlightCallOptions(headers=headers))
            life["opened"] = time.monotonic()
            failure: list[BaseException] = []
            ended = threading.Event()
            answered = threading.Event()  # the first ack has come: it names the store, or none
            decided = threading.Event()  # and this thread has chosen where the stream starts
            named: list[str | None] = []
            chose: list[bool] = []

            def read_acks() -> None:
                try:
                    while True:
                        buffer = reader.read()
                        if buffer is None:
                            return
                        life["acks"] += 1
                        document = json.loads(buffer.to_pybytes())
                        offset, store_id = int(document["end_byte_offset"]), _store_of(document)
                        if probing and not answered.is_set():
                            named.append(store_id)
                            answered.set()
                            decided.wait()  # its offset is taken once the stream knows where it starts
                            if not chose:
                                return  # the stream failed before it chose: nothing of this store is taken
                        self._ack(offset, store_id)
                        if "refused" in document:
                            self._refusal(document["refused"])
                except BaseException as error:  # noqa: BLE001
                    failure.append(error)
                    life["ack_error"] = self._brief(error)  # text only: an error's traceback holds this stream
                finally:
                    ended.set()

            acks = threading.Thread(target=read_acks, name="tensorquill-flight-acks", daemon=True)
            acks.start()
            waiting_since: float | None = None
            first_batch = True

            def metadata_for(end: int, steps: list[int], asking: bool = False, start: int | None = None) -> bytes:
                """A batch's metadata. The empty batch that asks for the store (`asking`) carries no
                manifest: that rides on the stream's first batch of rows, as before the question.
                A batch of rows states where its bytes begin (`start`, RS10, wave AE): a server that
                has acknowledged nothing for the rank since it started names a refusal of the batch
                from there. It is the same number `_starts` holds, so a refusal that names it is one
                `_refusal` can confirm; the asking batch claims no bytes and states none."""
                nonlocal first_batch
                metadata = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "rank": self.rank,
                    "min_step": min(steps) if steps else None,
                    "max_step": max(steps) if steps else None,
                    "end_byte_offset": end,
                    # Optional, and unknown to a server older than it, which ignores it.
                    **({"start_byte_offset": start} if start is not None and not asking else {}),
                    **({"world_size": self.world_size} if self.world_size else {}),
                    # Optional, and unknown to a server older than it, which ignores it.
                    **({"sender_attempt": self._attempt} if self._attempt is not None else {}),
                    # Optional, and unknown to a server older than it, which ignores it.
                    **({MANIFEST_KEY: self.manifest} if first_batch and self.manifest and not asking else {}),
                }
                first_batch = first_batch and asking
                return json.dumps(metadata).encode()

            try:
                if probing:
                    try:
                        # No rows and no bytes claimed: the server answers it at once with what it
                        # already holds (an offset of 0 on this stream) and, when it has one, its store ID.
                        empty = pa.RecordBatch.from_pylist([], schema=schema)
                        writer.write_with_metadata(empty, pa.py_buffer(metadata_for(0, [], asking=True)))
                        while not answered.is_set() and not ended.is_set():
                            if self._closing.is_set() and time.monotonic() >= self._deadline:
                                break
                            answered.wait(self._poll_seconds)
                        if not answered.is_set():
                            if failure:
                                raise failure[0]
                            raise ConnectionError(
                                "the server did not answer the stream's first batch, so the saved offset was not used")
                        self._store_id = named[0]
                        reason = self._bound(named[0])
                        if reason is None:
                            self._acked = sent = saved
                            life["start"] = life["sent"] = saved
                        else:
                            self._replay(reason)
                        chose.append(True)
                    finally:
                        decided.set()
                # The ack reader returning is how a stream the server ended shows up here;
                # the server's status (such as "ingest buffer full") is raised by close().
                while not ended.is_set():
                    closing = self._closing.is_set()  # read before the file, so nothing is missed
                    lines, end = read_lines(self.path, sent, self._batch_steps, self._batch_bytes)
                    now = time.monotonic()
                    if lines and waiting_since is None:
                        waiting_since = now
                    full = len(lines) >= self._batch_steps or end - sent >= self._batch_bytes
                    if lines and (full or closing or now - waiting_since >= self._batch_seconds):
                        places: list[int] = []
                        records, skipped = parse_lines(lines, self.run_id, places)
                        moved: list[tuple[int, str]] = []
                        refused: list[tuple[int, str, Any]] = []
                        batch, steps, unstorable = rows_to_send(records, schema, moved, refused)
                        self.skipped_lines += skipped + unstorable
                        if moved:
                            self._say_moved(moved)
                        if refused:
                            self._say_skipped(sent, lines, places, refused,
                                              [r.get("step", MalformedRecordError.ABSENT) for r in records])
                        start = self._sent(b"".join(line + b"\n" for line in lines), end)
                        metadata = metadata_for(end, steps, start=start)
                        writer.write_with_metadata(batch, pa.py_buffer(metadata))
                        sent = life["sent"] = end
                        life["batches"] += 1
                        life["arrow_bytes"] += batch.nbytes
                        waiting_since = None
                        continue
                    if closing and not lines:
                        break
                    ended.wait(self._poll_seconds) if closing else self._closing.wait(self._poll_seconds)
                if not ended.is_set():
                    life["done_writing"] = True  # from here an ending is the sender's own close
                    writer.done_writing()  # the server stores what it holds and sends the last ack
                    ended.wait(max(0.0, self._deadline - time.monotonic()))
            except BaseException:
                try:
                    writer.close()
                except BaseException as error:  # noqa: BLE001  the first failure is the one to report
                    life["close_error"] = self._brief(error)
                ended.wait(ACK_DRAIN_SECONDS)  # the answers sent before the status are taken first
                raise
            try:
                writer.close()
            except BaseException as error:
                life["close_error"] = self._brief(error)
                ended.wait(ACK_DRAIN_SECONDS)
                raise
            if failure:
                raise failure[0]
            if self._acked < sent or not self._closing.is_set():
                raise ConnectionError("the server ended the stream before confirming every batch")
        finally:
            client.close()


# gRPC's status codes by name (`grpc/status.h`), for the name pyarrow puts in an error's text.
_GRPC_CODES = {
    "OK": 0, "CANCELLED": 1, "UNKNOWN": 2, "INVALID_ARGUMENT": 3, "DEADLINE_EXCEEDED": 4, "NOT_FOUND": 5,
    "ALREADY_EXISTS": 6, "PERMISSION_DENIED": 7, "RESOURCE_EXHAUSTED": 8, "FAILED_PRECONDITION": 9, "ABORTED": 10,
    "OUT_OF_RANGE": 11, "UNIMPLEMENTED": 12, "INTERNAL": 13, "UNAVAILABLE": 14, "DATA_LOSS": 15, "UNAUTHENTICATED": 16,
}
# pyarrow's Flight errors by the status they stand for, when the text names none.
_FLIGHT_ERRORS = {
    "FlightUnavailableError": "UNAVAILABLE", "FlightInternalError": "INTERNAL", "FlightTimedOutError": "DEADLINE_EXCEEDED",
    "FlightCancelledError": "CANCELLED", "FlightUnauthenticatedError": "UNAUTHENTICATED",
    "FlightUnauthorizedError": "PERMISSION_DENIED",
}


def _grpc_status_name(text: str, error: BaseException) -> str | None:
    """The gRPC status an error from pyarrow's Flight client stands for: from its class where pyarrow
    gives the status one, else "grpc_status:N" in its text, else the name that opens pyarrow's
    "gRPC client debug context: UNAVAILABLE:..." (a text can name another status further in, "last
    error: UNKNOWN: ..."); None for an error that is not gRPC's (a server that ended the stream with
    OK, a local error)."""
    name = _FLIGHT_ERRORS.get(type(error).__name__)
    if name is not None:
        return name
    at = text.find("grpc_status:")
    if at >= 0:
        digits = ""
        for c in text[at + len("grpc_status:"):].lstrip():
            if not c.isdigit():
                break
            digits += c
        for name, code in _GRPC_CODES.items():
            if digits and code == int(digits):
                return name
    marker = "gRPC client debug context: "
    at = text.find(marker)
    if at >= 0:
        name = text[at + len(marker):].split(":", 1)[0].strip()
        if name in _GRPC_CODES:
            return name
    return None


def _without_traceback(error: BaseException) -> BaseException:
    """`error` and the exceptions it was raised from, without their tracebacks. A kept
    traceback holds the sender thread's frames, and through them the sender and the gRPC
    stream: a reference cycle. Freed by the garbage collector inside a forked DataLoader
    worker, the stream's gRPC channel then waits forever for threads the fork did not copy."""
    seen: set[int] = set()
    item: BaseException | None = error
    while item is not None and id(item) not in seen:
        seen.add(id(item))
        item.__traceback__ = None
        item = item.__cause__ or item.__context__
    return error


def _close_quietly(client: Any) -> None:
    if client is None:
        return
    try:
        client.close()
    except BaseException:  # noqa: BLE001  a client that cannot close is dropped as before
        pass


def send_file(
    path: str | Path,
    endpoint: str,
    run_id: str,
    rank: int = 0,
    *,
    timeout: float = 60.0,
    token: str | None = None,
    tls_root_certs: bytes | str | os.PathLike[str] | None = None,
) -> bool:
    """Send what a finished run's `.jsonl` file still owes the server. Returns whether the
    server confirmed the whole file."""
    return FlightSender(
        path, endpoint, run_id, rank, token=token, tls_root_certs=tls_root_certs
    ).close(timeout)


# --- The hook used by `tensorquill.init` ------------------------------------------------


_PACKAGE = os.path.dirname(os.path.abspath(__file__)) + os.sep


def thread_stacks() -> list[dict[str, Any]]:
    """Where each Python thread of this process is now, the main thread first and the calling
    thread and TensorQuill's own left out: `{"name", "main", "frames"}` with frames innermost
    first as `{"file", "line", "function"}`, with `"own": True` on a frame of TensorQuill's
    own code. File names, line numbers and function names only:
    no source text is read (the file system may be what hangs) and no local values are taken."""
    main = threading.main_thread().ident
    names = {t.ident: t.name for t in threading.enumerate()}
    me = threading.get_ident()
    threads = []
    for ident, frame in sys._current_frames().items():
        name = names.get(ident, f"thread-{ident}")
        if ident == me or name.startswith("tensorquill-"):
            continue
        keep = STACK_FRAMES if ident == main else STACK_OTHER_FRAMES
        frames = []
        while frame is not None and len(frames) < keep:
            code = frame.f_code
            entry = {"file": code.co_filename[-200:], "line": frame.f_lineno, "function": code.co_name[:80]}
            if code.co_filename.startswith(_PACKAGE):
                entry["own"] = True
            frames.append(entry)
            frame = frame.f_back
        threads.append({"name": name[:80], "main": ident == main, "frames": frames})
    threads.sort(key=lambda t: (not t["main"], t["name"]))
    return threads[:STACK_THREADS]


class Heartbeat:
    """Tells the server, from a thread of its own, that this rank's process is alive and which
    phase it is in, whether or not it is stepping.

    That is what lets the server tell a rank that is evaluating or saving a checkpoint (its
    heartbeats go on, its phase is not `train`) from one that is stuck in training (its
    heartbeats go on, its phase is `train`, no step arrives) and from a frozen or vanished
    process (the heartbeats stop). The thread never raises into training and never prints:
    the sender reports a failing connection. A forked child does nothing.

    When the work counter has not moved for `stack_seconds`, each beat also carries where the
    process's threads are (`thread_stacks`) and for how long the main thread has been at that
    same place, so a silent rank is reported with the call it is stuck in. A main thread that
    hangs while holding the interpreter lock stops this thread too: that rank reads as
    `unresponsive` and has no stack.

    Each beat also carries the session's `attempt` (the one its records carry), from the moment
    the session hands over its work counter, and that beat goes at once. The first beat waits for
    it, `FIRST_BEAT_HOLD_SECONDS` at most, and the stream's first connection waits for the first
    beat (`first_sent`), so a new process names its attempt before its stream opens. So the server knows a
    job that was started again before the new attempt stores any step, and gives it a start
    window of its own (see `watchdog.rs`). It is read once, so a beat costs nothing more.

    After the session stops capture (`capture.Session._stop`; contract 3), every beat carries
    `"capture_stopped": <reason>`, a short word (`sink_error`, `writer_behind`, `close_timeout`, ...),
    and no `stack`: the work counter stands still while training goes on, so a stack would read as a
    stuck step that is not one. The stop sends a beat at once, and `close()` sends one last beat,
    bounded by `LAST_BEAT_SECONDS`, when no beat carrying the stop has been taken by the server yet.
    If every one of them is lost and the stream then closes cleanly, the server reads the rank as
    finished: the documented limit. A server built before this key ignores it.

    A beat the server refuses (a retryable UNAVAILABLE when it cannot write the attempt order,
    contract 1) is caught like any failed beat: the next beat, one interval later, names the same
    attempt. Nothing is printed or raised."""

    def __init__(
        self,
        endpoint: str,
        run_id: str,
        rank: int,
        *,
        token: str | None = None,
        tls_root_certs: bytes | str | os.PathLike[str] | None = None,
        interval: float = HEARTBEAT_SECONDS,
        stack_seconds: float | None = STACK_SECONDS,
    ) -> None:
        import pyarrow.flight  # noqa: F401  fail at init, as the sender does, when pyarrow is missing

        self.interval = interval
        self.stack_seconds = stack_seconds
        self._work: int | None = None
        self._work_moved = time.monotonic()
        self._place: Any = None  # the main thread's frames at the last look, and since when
        self._place_since = 0.0
        self.phase = "train"
        self.capture_stopped: str | None = None  # the stop's reason, once the session stopped capture
        # The last step the session committed before it stopped, or None when it committed none
        # (detector rules contract, part 7). Set before `capture_stopped`, so a beat that reads the
        # reason reads the step that goes with it.
        self.capture_stopped_step: int | None = None
        self._stop_told = False  # a beat carrying the stop was taken by the server
        self._counter: Any = None  # a callable returning the work done so far (steps + ticks)
        self.attempt: int | None = None  # the session's attempt, once it has handed over its counter
        self._location = _location(endpoint)
        self._run_id = run_id
        self._rank = rank
        self._token = token if token is not None else os.environ.get(TOKEN_ENV)
        self._tls_root_certs = (
            _read_tls_root_certs(tls_root_certs)
            if (tls_root_certs is not None or self._location.startswith("grpc+tls://"))
            else None
        )
        self._pid = os.getpid()
        self._changed = threading.Event()
        self._stopped = threading.Event()
        self.first_sent = threading.Event()  # the first beat was tried
        self.sent = 0
        self._thread = threading.Thread(target=self._run, name=f"tensorquill-heartbeat-rank{rank}", daemon=True)
        self._thread.start()

    @property
    def counter(self) -> Any:
        return self._counter

    @counter.setter
    def counter(self, counter: Any) -> None:
        """Set by the session (`capture.py`) as it starts, just after it takes its attempt: the
        attempt is read here, and the next beat is sent now rather than after an interval."""
        self._counter = counter
        if counter is not None:
            self.attempt = self._take_attempt()
            self._changed.set()

    @staticmethod
    def _take_attempt() -> int | None:
        """The attempt of the session starting now: the one the session handing over its counter on
        this thread names as it does (`capture._attempt_handed_over`), so a session started at the
        same moment on another thread cannot lend its own. A counter set by other code gets the last
        attempt `capture` gave out in this process, as before."""
        try:
            from .capture import _attempt_handed_over, _last_attempt
        except Exception:  # noqa: BLE001  a beat without an attempt is a beat as before
            return None
        attempt = _attempt_handed_over()
        return attempt if attempt is not None else (_last_attempt or None)

    def set_phase(self, phase: str) -> None:
        """Name what the process is doing (`train`, `eval`, `checkpoint`, or any short word);
        the server is told at once, not at the next beat."""
        if os.getpid() != self._pid:
            return
        self.phase = str(phase)[:40] or "train"
        self._changed.set()

    def stop_capture(self, reason: str, step: int | None = None) -> None:
        """The session stopped capture (contract 3): from now on every beat says so, with `reason`,
        and the next beat goes at once. Each such beat also carries `capture_stopped_step`
        (detector rules contract, part 7): `step`, the last step the session committed before it
        stopped, or null when it committed none or the caller gives none. Step 0 is a step, not
        null. A value that is not a whole number of 0 or more (a bool among them) is sent as null."""
        if os.getpid() != self._pid or self.capture_stopped is not None:
            return
        ok = isinstance(step, int) and not isinstance(step, bool) and step >= 0
        self.capture_stopped_step = int(step) if ok else None
        self.capture_stopped = str(reason)[:40] or "unknown"
        self._changed.set()

    def close(self) -> None:
        if os.getpid() != self._pid:
            return
        self._stopped.set()
        self._changed.set()
        self._thread.join(2.0)

    def _run(self) -> None:
        import pyarrow.flight as flight

        headers = [(b"authorization", f"Bearer {self._token}".encode())] if self._token else []
        client = None
        if self._counter is None:
            self._changed.wait(FIRST_BEAT_HOLD_SECONDS)  # the session names its attempt in a moment
        while not self._stopped.is_set():
            # Clear first, so a phase change made while this beat is being sent wakes the next wait.
            self._changed.clear()
            client = self._beat(flight, client, headers, timeout=10.0)
            self.first_sent.set()
            self._changed.wait(self.interval)
        if self.capture_stopped is not None and not self._stop_told:
            # The stop was not delivered yet: one last beat, bounded, so a session closed right after
            # its capture stopped does not read as finished.
            client = self._beat(flight, client, headers, timeout=LAST_BEAT_SECONDS)
        _close_quietly(client)

    def _beat(self, flight: Any, client: Any, headers: list[Any], timeout: float) -> Any:
        """Send one beat; the client to use next (None after a failure, so the next beat reconnects)."""
        try:
            if client is None:
                kwargs: dict[str, Any] = {}
                if self._tls_root_certs is not None:
                    kwargs["tls_root_certs"] = self._tls_root_certs
                client = flight.FlightClient(self._location, **kwargs)
            stopped = self.capture_stopped  # read once: the same value is sent and marked told
            stopped_step = self.capture_stopped_step  # set before the reason (`stop_capture`)
            payload: dict[str, Any] = {"run_id": self._run_id, "rank": self._rank, "phase": self.phase,
                                       "interval_seconds": self.interval}
            if self.attempt is not None:
                payload["attempt"] = self.attempt
            if stopped is not None:
                payload["capture_stopped"] = stopped
                payload["capture_stopped_step"] = stopped_step  # null: no step committed (part 7)
            if self.counter is not None:
                payload["work"] = int(self.counter())
                stack = None if stopped is not None else self._stalled_stack(payload["work"])
                if stack is not None:
                    payload["stack"] = stack
            body = json.dumps(payload).encode()
            for _ in client.do_action(
                flight.Action("heartbeat", body), flight.FlightCallOptions(headers=headers, timeout=timeout)
            ):
                pass
            self.sent += 1
            if stopped is not None:
                self._stop_told = True
            return client
        except BaseException:  # noqa: BLE001  a missed beat is the server's to notice
            _close_quietly(client)  # its channel is closed now, not when the collector gets to it
            return None


    def _stalled_stack(self, work: int) -> dict[str, Any] | None:
        """The threads' stacks once `work` has stood still for `stack_seconds`, else None."""
        now = time.monotonic()
        if work != self._work:
            self._work, self._work_moved, self._place = work, now, None
            return None
        still = now - self._work_moved
        if not self.stack_seconds or still < self.stack_seconds:
            return None
        try:
            threads = thread_stacks()
        except Exception:  # noqa: BLE001  a beat without a stack is still a beat
            return None
        place = next((t["frames"] for t in threads if t["main"]), None)
        if place != self._place:
            self._place, self._place_since = place, now
        return {
            "still_seconds": round(still, 1),
            "unchanged_seconds": round(now - self._place_since, 1),
            "threads": threads,
        }


class StreamingSink:
    """A sink that also stops the sender, after the local file is closed and complete."""

    def __init__(self, sink: Any, sender: FlightSender, heartbeat: Heartbeat | None = None) -> None:
        self._sink = sink
        self.sender = sender
        self.heartbeat = heartbeat
        self.path = sink.path

    def write(self, records: list[dict[str, Any]]) -> None:
        self._sink.write(records)

    def close(self) -> None:
        """The local file first, then the heartbeat and the sender. A local file whose close raises
        stops capture: the heartbeat is told (`sink_error`) before it ends, so its last beat carries
        the stop, and the error is raised after the stream is closed."""
        failure: BaseException | None = None
        try:
            self._sink.close()
        except BaseException as error:  # noqa: BLE001  raised again below, once the stream is closed
            failure = error
            if self.heartbeat is not None:
                self.heartbeat.stop_capture("sink_error")
        if self.heartbeat is not None:
            self.heartbeat.close()
        self.sender.close()
        if failure is not None:
            raise failure


def attach(
    sink: Any,
    run_id: str,
    rank: int,
    endpoint: str | None = None,
    *,
    token: str | None = None,
    tls_root_certs: bytes | str | os.PathLike[str] | None = None,
    world_size: int | None = None,
    heartbeat_seconds: float | None = HEARTBEAT_SECONDS,
    stack_seconds: float | None = STACK_SECONDS,
    manifest: Mapping[str, Any] | None = None,
) -> Any:
    """Start streaming `sink`'s file when `endpoint` or `TENSORQUILL_ENDPOINT` is set;
    otherwise return `sink` unchanged, with no socket opened and `pyarrow` not imported."""
    endpoint = endpoint or os.environ.get(ENDPOINT_ENV)
    if not endpoint:
        return sink
    _location(endpoint)
    path = getattr(sink, "path", None)
    if path is None:
        raise ValueError(
            "endpoint= streams the local .jsonl file, so the sink must be a JsonlSink "
            f"(or left as the default), not {type(sink).__name__}"
        )
    heartbeat = None
    try:
        # The heartbeat first: the stream's first connection waits for its first beat.
        if heartbeat_seconds:
            heartbeat = Heartbeat(
                endpoint, run_id, rank, token=token, tls_root_certs=tls_root_certs, interval=heartbeat_seconds,
                stack_seconds=stack_seconds,
            )
        sender = FlightSender(
            path, endpoint, run_id, rank, token=token, tls_root_certs=tls_root_certs,
            world_size=world_size, manifest=manifest, hold=heartbeat.first_sent if heartbeat else None,
            await_attempt=True,
        )
    except ImportError as error:
        if heartbeat is not None:
            heartbeat.close()
        raise ImportError(
            'endpoint= needs pyarrow: pip install pyarrow'
        ) from error
    return StreamingSink(sink, sender, heartbeat)
