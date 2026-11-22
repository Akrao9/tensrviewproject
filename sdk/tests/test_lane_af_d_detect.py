"""Lane D of wave AF (Nov 21, 2026): the offline `tensorquill-server detect`, with and without
`--run`, against the `detect` entries of section 1's cases (`fixtures/wave_af/cases.json`).

Each case is one record whose identity field holds the case's value. It is put in an Arrow stream of
the schema the SDK sends (`flight.flight_schema`), from a valid row with that one column set, as the
page's base run made it, and given to the binary. Expected, per the page: exit status 1 and one line
that names the column (for a value, the value and the row too). A null in a column the schema
declares non-nullable is Arrow's own refusal, which names the column, as on the base.

Fails on the base (`1ca26d4`) for the negative values and `schema_version` 99: each was judged, exit 0.
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")

from tensorquill import flight  # noqa: E402

from test_flight_e2e import BINARY  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")

CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_af" / "cases.json").read_text())["cases"]
DETECT = {name: body for name, body in CASES.items() if (body.get("expected") or {}).get("detect")}
GOOD = {"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 0, "step": 0, "grad_norm": 1.0}


def _stream(field: str, value) -> bytes:
    row = flight.record_to_row(dict(GOOD))
    row[field] = value
    batch = pa.RecordBatch.from_pylist([row], schema=flight.flight_schema())
    out = io.BytesIO()
    with pa.ipc.new_stream(out, batch.schema) as writer:
        writer.write_batch(batch)
    return out.getvalue()


def test_the_detect_cases_are_the_ones_lane_d_targets():
    assert sorted(DETECT) == sorted([
        "attempt_negative", "attempt_null", "rank_negative", "rank_null", "schema_version_99",
        "schema_version_negative", "schema_version_null", "step_negative", "step_null",
    ])


@pytest.mark.parametrize("mode", [[], ["--run"]], ids=["detect", "detect --run"])
@pytest.mark.parametrize("name", sorted(DETECT))
def test_a_malformed_identity_field_is_refused_with_one_line_naming_it(name, mode):
    body = DETECT[name]
    field, value = body["field"], body["given"]["record"][body["field"]]
    expected = body["expected"]["detect"]
    done = subprocess.run([BINARY, "detect", *mode], input=_stream(field, value), capture_output=True, timeout=120)
    lines = done.stderr.decode().strip().splitlines()
    assert done.returncode == expected["exit"], (done.returncode, done.stdout[:200], lines)
    assert done.stdout == b""
    assert len(lines) == 1, lines
    assert expected["names"] in lines[0], lines
    if value is not None:
        assert lines[0].startswith(f"tensorquill-server: row 1: {field} is {value}"), lines


def test_valid_rows_are_judged_as_before_in_both_modes():
    """A pin: the same stream with every identity field valid is judged, exit 0."""
    stream = _stream("step", 0)
    for mode in ([], ["--run"]):
        done = subprocess.run([BINARY, "detect", *mode], input=stream, capture_output=True, timeout=120)
        assert done.returncode == 0, done.stderr
        report = json.loads(done.stdout)
        report = report["reports"][0] if mode else report
        assert (report["rank"], report["steps"]) == (0, 1)
