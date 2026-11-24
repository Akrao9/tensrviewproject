"""Lane AL-3 (wave AL, step 1): `tensorquill-server detect` on protocol 5's typed rows.

`WAVE_AL_CONTRACT.md` sections 1 and 2: the Rust detectors read `layers` and `moe_stats` from the
typed map columns where a batch has them (`layers_json` and `moe_stats_json` where it does not), a
refused field and an entry that is not an object are absent to every detector, `non_finite` also
reads the refused fields kept in `extra_json`, and `probe_maps` and `probe_histograms` are never
read. Every section 1 and 2 case of `fixtures/wave_al/cases.json` is sent to the binary here:

- as protocol 5's rows, which must give the case's `expect["rust_raw"]`. Until lane AL-1's sender
  is in the tree the rows are the prototype's (`experiments/protocol-v5/prototype.py`, the layout
  the contract states); once `flight` writes protocol 5's layout itself, they are the SDK's own
  (`server_detect.records_stream`), so this file keeps its meaning after the seam;
- as protocol 4's rows (the JSON columns, through `server_detect.server_report`), while `flight`
  still writes them: the same report wherever the case expects no difference, and today's report
  for the three targets of section 1.5, whose JSON row the contract leaves as it is.

The binary is `TENSORQUILL_SERVER_BIN` (or one beside the interpreter); without it, or without
pyarrow, everything here is skipped.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorquill import server_detect

ROOT = Path(__file__).parents[2]
CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_al" / "cases.json").read_text())["cases"]
ROWS = [c for c in CASES if c["section"] in (1, 2) and "rust_raw" in c["expect"]]
TARGETS = {"layers_entry_not_object", "layers_name_arrow_cannot_encode", "layers_value_not_object"}
BINARY = server_detect.find_binary()

# Marks, not a module-level skip, so that a run of this file alone on CI's plain jobs (no pyarrow, no
# binary) collects every test and skips each, as the whole suite does.
pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow is not installed"),
    pytest.mark.skipif(BINARY is None or not os.path.isfile(BINARY),
                       reason="tensorquill-server is not built (TENSORQUILL_SERVER_BIN)"),
]


def _flight_is_protocol_4() -> bool:
    from tensorquill import flight

    return "layers_json" in flight.flight_schema().names


def _prototype():
    spec = importlib.util.spec_from_file_location("prototype_v5_al3",
                                                  ROOT / "experiments" / "protocol-v5" / "prototype.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _typed_stream(records: list[dict], token_entropy: bool = False) -> bytes:
    """One rank's records as protocol 5's rows, in the order `detect` is sent them."""
    import pyarrow as pa

    if not _flight_is_protocol_4():
        return server_detect.records_stream(records)
    p5 = _prototype()
    ordered = [r if isinstance(r.get("run_id"), str) else {**r, "run_id": ""}
               for r in server_detect.stream_order(copy.deepcopy(records))]
    schema = p5.flight_schema_v5(token_entropy=token_entropy)
    batch = p5.batch_v5(ordered, schema=schema)
    assert batch.column("schema_version").to_pylist() == [2] * len(ordered)
    assert "layers" in batch.schema.names and "layers_json" not in batch.schema.names
    out = io.BytesIO()
    with pa.ipc.new_stream(out, schema) as writer:
        writer.write_batch(batch)
    return out.getvalue()


def _detect(stream: bytes, *flags: str) -> dict:
    done = subprocess.run([BINARY, "detect", *flags], input=stream, capture_output=True, timeout=120)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    return json.loads(done.stdout)


def _brief(incident: dict) -> dict:
    """As `test_wave_al_contract._brief`."""
    out = {"kind": incident["kind"], "step": incident.get("step")}
    for key in ("layers", "modules", "gradients", "routers", "router", "parameters", "experts"):
        if key in incident:
            value = incident[key]
            out[key] = sorted(value) if isinstance(value, dict) else value
    return out


def _seen(report: dict, want: dict) -> dict:
    return {"incidents": [_brief(i) for i in report["incidents"]],
            "kind_checks": {k: (report["kind_checks"].get(k) or {}).get("value") for k in sorted(want["kind_checks"])}}


@pytest.mark.parametrize("case", ROWS, ids=[c["name"] for c in ROWS])
def test_detect_on_the_typed_rows_reports_the_rust_expectation(case):
    want = case["expect"]["rust_raw"]
    assert _seen(_detect(_typed_stream(case["records"])), want) == want


def test_a_part_with_token_entropy_is_read_as_one_without():
    """D3: K3's field is not reserved; a part that has it is read alike."""
    if not _flight_is_protocol_4():
        pytest.skip("flight writes protocol 5's layout: its schema has no token_entropy yet")
    case = next(c for c in ROWS if c["name"] == "moe_stats_token_entropy_by_name")
    want = case["expect"]["rust_raw"]
    without = _detect(_typed_stream(case["records"]))
    with_field = _detect(_typed_stream(case["records"], token_entropy=True))
    assert _seen(with_field, want) == _seen(without, want) == want
    assert with_field["incidents"] == without["incidents"] and with_field["kind_checks"] == without["kind_checks"]


@pytest.mark.parametrize("case", ROWS, ids=[c["name"] for c in ROWS])
def test_the_json_rows_report_the_same_where_the_case_expects_no_difference(case):
    if not _flight_is_protocol_4():
        pytest.skip("flight writes protocol 5's layout: the JSON rows are judged by lane_al3_tests.rs")
    want = case["expect"]["rust_raw"]
    typed = _detect(_typed_stream(case["records"]))
    json_rows = server_detect.server_report(copy.deepcopy(case["records"]), binary=BINARY)
    if case["name"] in TARGETS:
        # Protocol 4's reader is unchanged: the JSON row reports today's result.
        assert _seen(json_rows, want) == case["today"]["rust_raw"]
        if case["today"]["rust_raw"] != want:
            assert _seen(typed, want) != _seen(json_rows, want)
        return
    assert case["today"]["rust_raw"] == want
    assert _seen(json_rows, want) == want
    for key in ("incidents", "kind_checks", "not_checked", "head_trends", "moe_flagged"):
        assert typed[key] == json_rows[key], key


@pytest.mark.parametrize("case", [c for c in ROWS if c["name"].startswith("moe_stats")],
                         ids=[c["name"] for c in ROWS if c["name"].startswith("moe_stats")])
def test_the_per_step_moe_samples_are_the_same_on_both_layouts(case):
    """`detect --samples` lists each router's diversity as it was judged: read from the typed struct
    by name, and from the counts where the field was refused, as on the JSON row."""
    if not _flight_is_protocol_4():
        pytest.skip("flight writes protocol 5's layout: the JSON rows are judged by lane_al3_tests.rs")
    typed = _detect(_typed_stream(case["records"]), "--samples")
    json_rows = server_detect.server_report(copy.deepcopy(case["records"]), binary=BINARY, samples=True)
    assert typed["samples"] and typed["samples"] == json_rows["samples"]
