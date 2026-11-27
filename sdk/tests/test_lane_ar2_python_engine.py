"""Wave AR, lane AR-2: three places where the Python engine departed from the contract (lane AQ-5's
generated runs, `test_generated_engines.py`).

- AQ5-D1. A refused `update_rms` map is an absent key (the author: "Python brought to Rust"):
  `input_counts` counted a map with no entry its column holds toward `update_spike`'s records.
- AQ5-D3. A top-level `grad_norm` that is not a number is absent in Python too: the stalled-weights
  rule let one break its streak, so a malformed `grad_norm` hid a stall from the default engine.
- AQ5-D6. `identity.check` is bounded at int64 (`step`, `attempt`) and int32 (`rank`), the columns'
  types, and never raises into training.

Each comparison with Rust needs `pyarrow` and the server binary (`TENSORQUILL_SERVER_BIN`), and is
skipped without them, one by one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import tensorquill
from tensorquill import evidence
from tensorquill.server_detect import find_binary

try:  # each test is collected and skipped without pyarrow
    import pyarrow  # noqa: F401
except ImportError:
    pyarrow = None

BINARY = find_binary()


def needs_rust() -> None:
    if pyarrow is None:
        pytest.skip("pyarrow is not installed")
    if BINARY is None:
        pytest.skip("tensorquill-server is not built")


def _record(step: int, **fields: Any) -> dict[str, Any]:
    return {"schema_version": 1, "run_id": "ar2", "rank": 0, "attempt": 1, "step": step, **fields}


def _write(folder: Path, records: list[dict[str, Any]]) -> Path:
    path = folder / "ar2.rank0.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _both(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    needs_rust()
    return (tensorquill.build_evidence(path),
            tensorquill.build_evidence(path, detectors="rust", server_binary=BINARY))


def _check(found: dict[str, Any], kind: str) -> tuple[Any, Any]:
    entry = found["kind_checks"][kind]
    return entry.get("value"), entry.get("records")


# --- AQ5-D1: a refused update_rms map is an absent key ------------------------------------------

# Each `update_rms` value, and whether its column holds the map (an entry it carries: a number, the
# sink's "nan", or a null, which Rust's map column keeps as an entry).
UPDATE_MAPS = {
    "a text that is not the sink's spelling": ({"p0.weight": "nan "}, False),
    "the text NaN": ({"p0.weight": "NaN"}, False),
    "an integer too large for a float": ({"p0.weight": 10**400}, False),
    "a bool": ({"p0.weight": True}, False),
    "two refused entries": ({"a": "x", "b": [1.0]}, False),
    "a held entry beside a refused one": ({"a": 0.01, "b": "x"}, True),
    "the sink's nan": ({"p0.weight": "nan"}, True),
    "a null": ({"p0.weight": None}, True),
    "a number": ({"p0.weight": 0.01}, True),
    "an empty map": ({}, False),
    "not a map": ("x", False),
}


@pytest.mark.parametrize("name", sorted(UPDATE_MAPS))
def test_d1_an_update_rms_map_counts_only_when_its_column_holds_an_entry(name):
    sizes, held = UPDATE_MAPS[name]
    counts = evidence.input_counts([_record(47, grad_norm=1.05, update_rms=sizes)])
    assert counts["update"] == int(held)


@pytest.mark.parametrize("name", sorted(UPDATE_MAPS))
def test_d1_both_engines_count_update_spike_records_alike(name, tmp_path):
    sizes, held = UPDATE_MAPS[name]
    python, rust = _both(_write(tmp_path, [_record(47, grad_norm=1.048992069, update_rms=sizes)]))
    assert _check(python, "update_spike") == _check(rust, "update_spike")
    assert _check(python, "update_spike") == (("checked", 1) if held else ("not_configured", 0))


def test_d1_a_refused_map_is_absent_to_the_update_readers_too(tmp_path):
    """`update_spike` itself and `update_trends` already skip each refused entry (`_finite`), so a
    record whose map is refused gives no update point in either engine; a run of them beside held ones
    judges as the held ones alone."""
    held = [_record(step, grad_norm=1.0 + 0.01 * step, update_rms={"w": 1e-3 * (1 + 0.01 * step)})
            for step in range(12)]
    refused = [_record(step, grad_norm=1.0, update_rms={"w": "nan "}) for step in range(12, 16)]
    path = _write(tmp_path, held + refused)
    python, rust = _both(path)
    assert python.get("update_trends") == rust.get("update_trends")
    (tmp_path / "alone").mkdir()
    alone, _ = _both(_write(tmp_path / "alone", held))
    assert python.get("update_trends") == alone.get("update_trends")
    assert _check(python, "update_spike")[1] == _check(alone, "update_spike")[1] == 12
