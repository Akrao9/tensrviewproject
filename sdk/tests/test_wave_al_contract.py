"""The cases of `WAVE_AL_CONTRACT.md` (protocol 5): `fixtures/wave_al/cases.json`.

What runs here, on the base and on every lane's tree:

- The document: every case is named on the page, in its section, with the fields its paths need.
- The cases against the prototype (`experiments/protocol-v5/prototype.py`): each case's `row` is
  what the prototype gives, its round trip holds and keeps each map's order. This keeps the cases and
  the page's layout one thing; the prototype is not product code.
- Pins: where the base already reports on `records` what the contract asks (`expect`), the engine
  paths are ordinary tests, so no lane can move them.
- Targets: where the base does not (a refused entry today, the row layout, the scalar-only record),
  the test is a strict expected failure (`xfail(strict=True)`, as the repository marks work a lane
  has not landed). The lane named in its reason turns it into an ordinary test by removing the mark
  when its change makes it pass; a target that passes before that fails the run (strict).

The Rust paths need the server binary (`TENSORQUILL_SERVER_BIN`, or one beside the interpreter) and
are skipped without it; anything that needs pyarrow is skipped without it (CI's plain jobs have none).
The live server, `runs` and MCP paths are the lanes' own tests (contract section 9).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

from tensorquill import evidence

ROOT = Path(__file__).parents[2]
DOCUMENT = json.loads((Path(__file__).parent / "fixtures" / "wave_al" / "cases.json").read_text())
CASES = DOCUMENT["cases"]
PAGE = (ROOT / DOCUMENT["contract"]).read_text()
EXPERIMENTS = ROOT / "experiments" / "protocol-v5"
ROWS = [c for c in CASES if c["section"] in (1, 2)]
SCALAR = [c for c in CASES if c["section"] == 4]
LANE_ROW = "lane AL-1 (flight.py, server_detect.py): protocol 5's row layout, contract sections 1 and 2"
LANE_RUST = "lane AL-3 (detect.rs, detect/): the typed columns and the refused-field rule, contract sections 1 and 2"
LANE_PY = "lane AL-4 (evidence.py): an entry the column cannot hold is absent to every detector, contract section 1"
LANE_SCALAR = "step 2 of the wave (contract section 4): the scalar-only record"


def _prototype():
    spec = importlib.util.spec_from_file_location("prototype_v5", EXPERIMENTS / "prototype.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_document_and_the_page():
    assert DOCUMENT["contract"] == "WAVE_AL_CONTRACT.md"
    assert len(DOCUMENT["base"]) == 40
    names = [c["name"] for c in CASES]
    assert len(names) == len(set(names))
    for case in CASES:
        assert f"`{case['name']}`" in PAGE, case["name"]
        assert case["section"] in (1, 2, 4), case["name"]
        assert case["paths"], case["name"]
        if case["section"] in (1, 2):
            assert {"row", "expect", "today", "records"} <= set(case), case["name"]
        else:
            assert {"expect", "records"} <= set(case), case["name"]
    assert "Where each decision stands" in PAGE


def test_the_measurements_the_page_cites_are_recorded():
    for name in ("captures.json", "live.json", "ack.json", "live64.json", "decode.json"):
        assert (EXPERIMENTS / "results" / name).is_file(), name
    captures = json.loads((EXPERIMENTS / "results" / "captures.json").read_text())
    assert len(captures) == 7
    for name, found in captures.items():
        trip = found["round_trip"]
        assert trip["not_equal_count"] == 0 and trip["order_differs_count"] == 0, name
    live = json.loads((EXPERIMENTS / "results" / "live.json").read_text())
    assert live["order"]["kept"] is True
    assert live["watermark"]["lost"] == [4, 5, 6, 7, 8, 9]


@pytest.mark.parametrize("case", ROWS, ids=[c["name"] for c in ROWS])
def test_the_prototype_gives_each_row(case):
    """The case's `row` is the prototype's: the layout the page states, in one place."""
    pa = pytest.importorskip("pyarrow")

    p5 = _prototype()
    record = next(r for r in case["records"] if r["step"] == case["row_step"])
    row = p5.record_to_row_v5(copy.deepcopy(record))
    want = case["row"]
    for column in ("layers", "moe_stats", "probe_maps", "probe_histograms"):
        if column in want:
            got = None if row[column] is None else [[name, _jsonable(v)] for name, v in row[column]]
            assert got == want[column], column
    if "probe_json" in want:
        assert row["probe_json"] == want["probe_json"]
    assert (json.loads(row["extra_json"]) if row.get("extra_json") else None) == want["extra"]
    back = p5.row_to_record_v5(pa.RecordBatch.from_pylist([row], schema=p5.flight_schema_v5()).to_pylist()[0])
    assert back == record and want["round_trip"] is True
    for key in ("layers", "moe_stats"):
        assert list(back.get(key) or {}) == list(record.get(key) or {}), key
    probe = record.get("probe")
    if isinstance(probe, dict):
        for key in ("attention", "histograms"):
            assert list((back["probe"] or {}).get(key) or {}) == list(probe.get(key) or {}), key


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


@pytest.mark.xfail(strict=True, reason=LANE_ROW)
@pytest.mark.parametrize("case", ROWS, ids=[c["name"] for c in ROWS])
def test_flight_gives_the_row(case):
    """Lane AL-1's target: `flight.record_to_row` gives the case's row in protocol 5's layout, and
    `row_to_record` gives the record back."""
    from tensorquill import flight

    record = next(r for r in case["records"] if r["step"] == case["row_step"])
    names = flight.flight_schema().names
    assert "layers" in names and "layers_json" not in names and "probe_maps" in names
    row = flight.record_to_row(copy.deepcopy(record))
    want = case["row"]
    for column in ("layers", "moe_stats", "probe_maps", "probe_histograms"):
        if column in want:
            got = None if row[column] is None else [[name, _jsonable(v)] for name, v in row[column]]
            assert got == want[column], column
    assert (json.loads(row["extra_json"]) if row.get("extra_json") else None) == want["extra"]
    assert flight.row_to_record(row) == record


def _brief(incident: dict) -> dict:
    out = {"kind": incident["kind"], "step": incident.get("step")}
    for key in ("layers", "modules", "gradients", "routers", "router", "parameters", "experts"):
        if key in incident:
            value = incident[key]
            out[key] = sorted(value) if isinstance(value, dict) else value
    return out


def _engine(path: str, records: list[dict], kinds: list[str]) -> dict:
    if path == "python":
        built = evidence.build_evidence(copy.deepcopy(records), detectors="python")
        return {"incidents": [_brief(i) for i in built["incidents"]],
                "kind_checks": {k: (built["kind_checks"].get(k) or {}).get("value") for k in kinds},
                "run_state": (built.get("run_state") or {}).get("state")}
    from tensorquill.server_detect import server_report

    report = server_report(copy.deepcopy(records))
    return {"incidents": [_brief(i) for i in report["incidents"]],
            "kind_checks": {k: (report["kind_checks"].get(k) or {}).get("value") for k in kinds}}


def _engine_params():
    out = []
    for case in ROWS:
        for path in ("python", "rust_raw"):
            if path not in case["paths"] or path not in case["expect"]:
                continue
            pinned = case["today"].get(path) == case["expect"][path]
            marks = []
            if path == "rust_raw":
                from tensorquill.server_detect import find_binary

                marks.append(pytest.mark.skipif(find_binary() is None or importlib.util.find_spec("pyarrow") is None,
                                                reason="tensorquill-server is not built, or pyarrow is not installed"))
            if not pinned:
                marks.append(pytest.mark.xfail(strict=True, reason=LANE_PY if path == "python" else LANE_RUST))
            out.append(pytest.param(case, path, marks=marks, id=f"{case['name']}-{path}-{'pin' if pinned else 'target'}"))
    return out


@pytest.mark.parametrize(("case", "path"), _engine_params())
def test_each_engine_reports_what_the_clean_records_give(case, path):
    """On `records` each engine must report what today's engines report on `clean_records` (section
    1 and 2's invariant: the evidence does not change shape). A pin holds on the base already."""
    want = case["expect"][path]
    if "raises" in want:
        pytest.fail(f"the clean records raise: {want['raises']}")
    got = _engine(path, case["records"], sorted(want.get("kind_checks", {})))
    assert got["incidents"] == want["incidents"]
    assert got["kind_checks"] == want["kind_checks"]
    if path == "python":
        assert got["run_state"] == want["run_state"]


@pytest.mark.xfail(strict=True, reason=LANE_SCALAR)
@pytest.mark.parametrize("case", [c for c in SCALAR if "python" in c["paths"]],
                         ids=[c["name"] for c in SCALAR if "python" in c["paths"]])
def test_a_scalar_record_is_read_and_never_a_training_step(case):
    """The readers take a file with scalar-only records (section 4): never a step judged, never
    `repeated_steps`; the evidence equals the evidence of the training records alone."""
    records = copy.deepcopy(case["records"])
    training = [r for r in records if r.get("schema_version") != 2]
    built = evidence.build_evidence(records, detectors="python")
    alone = evidence.build_evidence(copy.deepcopy(training), detectors="python")
    assert [i["kind"] for i in built["incidents"]] == [i["kind"] for i in alone["incidents"]]
    assert built["kind_checks"] == alone["kind_checks"]
    assert "repeated_steps" not in [i["kind"] for i in built["incidents"]]


def test_a_scalar_record_is_refused_by_the_base_readers():
    """Pinned as the base does it, for section 5's matrix: today's `load_records` refuses a file with a
    scalar-only record, naming its version. Kept only until step 2 of the wave lands (its lane removes
    this test with the target above)."""
    case = next(c for c in SCALAR if c["name"] == "scalar_old_reader")
    assert case["today"]["python"].startswith("MalformedRecordError: schema_version is 2")
