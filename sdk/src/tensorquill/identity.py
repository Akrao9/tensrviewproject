"""Whether a record's identity fields are valid: `WAVE_AF_CONTRACT.md`, section 1 (L32, SDK-06).

`check(record, where)` is the only Python code that decides whether a record's `schema_version`,
`rank`, `attempt` and `step` are valid. A reader calls it where records are loaded, before either
engine is chosen, and refuses the whole input on the first record that is not valid with
`MalformedRecordError`. It never runs inside a training step or the capture.

| Field            | Absent                    | Valid when present                     |
|------------------|---------------------------|----------------------------------------|
| `schema_version` | read as 0, the legacy one | the integer 0, 1 or 2 (`SCHEMA_VERSIONS`); 2 for a scalar-only record alone |
| `rank`           | read as on the base       | an integer of 0 or more                |
| `attempt`        | read as 0                 | an integer of 0 or more                |
| `step`           | not valid                 | an integer of 0 or more                |

A Boolean, a float (also a whole one, `1.0`), text and `null` are not integers. `run_id` is not
looked at. There is no upper bound here: a value too large for its column is Arrow's to refuse.

Record version 2 is the scalar-only record (`WAVE_AL_CONTRACT.md` sections 3 and 4.2): a record of
version 2 is valid only with `"record": "scalars"` and a `seq` that is an integer of 0 or more,
checked after `step`. A step record stays version 0 or 1, and a key `record` on one is read as
before. `is_scalar_record` says which a record is; every reader sets such records apart where
records are loaded, after this check and before any engine (section 4.6). Other keys of a
scalar-only record are not looked at here: a reader that does not know one ignores it.

`where` is a string the caller supplies; `in_file`, `in_part` and `in_list` write it the same way
everywhere. This module imports only the standard library.
"""

from __future__ import annotations

from typing import Any, Mapping

# The record versions (a record's own `schema_version`, as in the `.jsonl` file) this package reads.
# They are neither the batch metadata's version (`flight.SCHEMA_VERSION`) nor a stored row's
# (`flight.ROW_VERSION`, with the record's own in `record_version`): `WAVE_AL_CONTRACT.md` section 3
# keeps the four numbers apart.
SCHEMA_VERSIONS = (0, 1, 2)
# The scalar-only record's version and kind (`WAVE_AL_CONTRACT.md` section 4.2).
SCALAR_VERSION = 2
SCALAR_RECORD = "scalars"
# Checked in this order; the first field that is not valid is the one named.
FIELDS = ("schema_version", "rank", "attempt", "step")
_ABSENT = object()


class MalformedRecordError(ValueError):
    """A record whose identity field is not valid. `field` is the field's name, `value` what was
    found (`None` for a null; `MalformedRecordError.ABSENT` for a missing `step`), `where` the place
    the caller named (or None), and `reason` what is read there."""

    ABSENT = _ABSENT

    def __init__(self, field: str, value: Any, where: str | None, reason: str) -> None:
        self.field = field
        self.value = value
        self.where = where
        self.reason = reason
        found = "absent" if value is _ABSENT else f"{_shown(value)}"
        place = f" at {where}" if where else ""
        super().__init__(f"{field} is {found}{place}: {reason}")


def _shown(value: Any) -> str:
    """A value as the record's JSON writes it: `null`, `true`, `1.5`, `"1"`."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return repr(value).replace("'", '"') if '"' not in value else repr(value)
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def in_file(path: Any, line: int) -> str:
    """A line of a `.jsonl` file, `line` counted from 1."""
    return f"{path}, line {line}"


def in_part(part: Any, row: int) -> str:
    """A row of a Parquet part of a store, `row` counted from 0 as Arrow counts it."""
    return f"{part}, row {row}"


def in_list(index: int) -> str:
    """A record passed in memory, by its position in the list, counted from 0."""
    return f"record {index} of the records passed"


def is_scalar_record(record: Any) -> bool:
    """Whether `record` is a scalar-only record: `schema_version` the integer 2 and `record`
    `"scalars"`. Records that passed `check` are one or the other kind, so this is the one test
    every reader makes to set them apart; on records not checked it is the same test."""
    if not isinstance(record, Mapping):
        return False
    version = record.get("schema_version")
    return type(version) is int and version == SCALAR_VERSION and record.get("record") == SCALAR_RECORD


def _versions_read() -> str:
    *first, last = (str(v) for v in SCHEMA_VERSIONS)
    return f"{', '.join(first)} and {last}" if first else last


def check(record: Mapping[str, Any], where: str | None = None, fields: tuple[str, ...] = FIELDS) -> None:
    """Raise `MalformedRecordError` if `record`'s identity fields are not valid; else return None.
    The record is not changed and nothing is filled in.

    `fields` limits the check to some of `FIELDS`, in `FIELDS`' order, by the same rules: the
    sender's own file checks only `rank`, `attempt` and `step` (`flight.SENDER_FIELDS`), since a
    `schema_version` the server does not read is sent for the server to count (RS10).

    With `schema_version` checked, a record of version 2 must be a scalar-only record: `record`
    `"scalars"`, else `schema_version` is named; and, after `step`, `seq` an integer of 0 or more,
    else `seq` is named."""
    scalar = False
    for field in (name for name in FIELDS if name in fields):
        value = record.get(field, _ABSENT) if isinstance(record, Mapping) else _ABSENT
        if value is _ABSENT:
            if field == "step":
                raise MalformedRecordError(field, _ABSENT, where, "every record needs a step, an integer of 0 or more")
            continue
        if field == "schema_version":
            if isinstance(value, bool) or not isinstance(value, int) or value not in SCHEMA_VERSIONS:
                raise MalformedRecordError(field, value, where, f"this tensorquill reads schema versions {_versions_read()}")
            if value == SCALAR_VERSION:
                if record.get("record") != SCALAR_RECORD:
                    raise MalformedRecordError(
                        field, value, where,
                        f'schema version {SCALAR_VERSION} is the scalar-only record, which has "record": "{SCALAR_RECORD}"')
                scalar = True
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MalformedRecordError(field, value, where, "tensorquill reads an integer of 0 or more")
    if scalar:
        value = record.get("seq", _ABSENT)
        if value is _ABSENT or isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MalformedRecordError("seq", value, where, "a scalar-only record needs a seq, an integer of 0 or more")
