"""The cases of `WAVE_AL_CONTRACT.md` (protocol 5): `fixtures/wave_al/cases.json`.

What runs here, on the base and on every lane's tree:

- The document: every case is named on the page, in its section, with the fields its paths need.
- The cases against the prototype (`experiments/protocol-v5/prototype.py`): each case's `row` is
  what the prototype gives, its round trip holds and keeps each map's order. This keeps the cases and
  the page's layout one thing; the prototype is not product code. It is built on protocol 4's row
  functions, frozen beside it in `flight_v4.py` (`flight.py` at `5c286ed`), as it was measured.
- `flight` itself (lane AL-1, merged at `92e471a`): each case's `row` and round trip, and the
  prototype's row on every record of every section 1 and 2 case, `clean_records` included.
- The `rust_sdk` path (`build_evidence(detectors="rust")`) of every case that names it, against the
  case's `rust_raw` expectation: ordinary tests since the joined tree (92e471a).
- The `rust_raw_v4` path (the owner, Nov 24): `detect` given protocol 4's rows, as an older SDK sends
  them, reads them as the base did. `layers_entry_not_object`'s is lane AL-3′'s target (step 2);
  `layers_name_arrow_cannot_encode`'s is a known difference, pinned at today's report.
- Pins: where the base already reports on `records` what the contract asks (`expect`), the engine
  paths are ordinary tests, so no lane can move them.
- Targets: where the base does not (a refused entry today, the scalar-only record; the row layout
  until lane AL-1 landed), the test is a strict expected failure (`xfail(strict=True)`, as the repository marks work a lane
  has not landed). The lane named in its reason turns it into an ordinary test by removing the mark
  when its change makes it pass; a target that passes before that fails the run (strict).
- Section 4.8, on `tensorquill.init` and `Session.log` themselves: a value logged late inside a compiled
  function before the flush for a newly trainable parameter (a target), and its eager twin (a pin).

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
LANE_RUST = "lanes AL-1 and AL-3 joined (flight.py's layout, detect.rs' reader): contract sections 1 and 2"
LANE_PY = "lane AL-4' (evidence.py, step 2): an entry the column cannot hold is absent to every detector, contract section 1.5"
LANE_PY_NO_LAYER = ("lane AL-4' (evidence.py, step 2): an entry that is not an object is absent, and a row whose only "
                    "layers are such entries holds no layer (the owner's approved change, Nov 24), contract sections 1.5 and 7")
LANE_RUST_V4 = ("lane AL-3' (detect.rs, step 2): protocol 4's layers_json read entry by entry, so an entry that is not "
                "an object is absent and the rest are read (the owner, Nov 24), contract sections 1.5 and 9")
LANE_SCALAR = "step 2 of the wave (contract section 4): the scalar-only record"
# Targets whose lanes have landed: they pass on the joined tree and are ordinary tests now. A case's
# `today` stays the base's (f71d8e0), so the mark is decided here, not by `today == expect`.
LANDED = {
    ("layers_entry_not_object", "rust_raw"),  # lanes AL-1 and AL-3, merged at 92e471a
    ("layers_name_arrow_cannot_encode", "rust_raw"),  # lanes AL-1 and AL-3, merged at 92e471a
}


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
    for name in ("captures.json", "fixtures.json", "live.json", "ack.json", "live64.json", "decode.json"):
        assert (EXPERIMENTS / "results" / name).is_file(), name
    captures = json.loads((EXPERIMENTS / "results" / "captures.json").read_text())
    assert len(captures) == 7
    for name, found in captures.items():
        trip = found["round_trip"]
        assert trip["not_equal_count"] == 0 and trip["order_differs_count"] == 0, name
    fixtures = json.loads((EXPERIMENTS / "results" / "fixtures.json").read_text())
    for name, found in fixtures.items():
        assert found["differs_count"] == found["raised_count"] == found["order_differs_count"] == 0, name
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


def test_the_prototype_gives_the_scalar_record_row():
    """Section 4.4: a scalar-only record's row carries `seq` and `record_version` 2, nothing in
    `extra_json` (`record` comes back from `seq`), and reads back as the record."""
    pa = pytest.importorskip("pyarrow")
    p5 = _prototype()
    case = next(c for c in SCALAR if c["name"] == "scalar_record_keys")
    record = next(r for r in case["records"] if r.get("schema_version") == 2)
    assert list(record) == case["expect"]["keys"]
    row = p5.record_to_row_v5(copy.deepcopy(record))
    assert (row["seq"], row["record_version"], row["schema_version"], row["extra_json"]) == (0, 2, 2, None)
    assert case["row"] == {"seq": 0, "record_version": 2, "schema_version": 2, "extra": None, "round_trip": True}
    back = p5.row_to_record_v5(pa.RecordBatch.from_pylist([row], schema=p5.flight_schema_v5()).to_pylist()[0])
    assert back == record


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


@pytest.mark.parametrize("case", ROWS, ids=[c["name"] for c in ROWS])
def test_flight_gives_the_row(case):
    """Lane AL-1's (landed at 92e471a): `flight.record_to_row` gives the case's row in protocol 5's
    layout, and `row_to_record` gives the record back."""
    pytest.importorskip("pyarrow")  # flight_schema() imports it; CI's plain jobs have none
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


@pytest.mark.parametrize("case", ROWS, ids=[c["name"] for c in ROWS])
def test_flight_gives_the_prototype_s_row_on_every_record(case):
    """`flight.record_to_row` gives the measured prototype's row, every column, on every record of the
    case (`records` and `clean_records`), so section 6's prototype figures are the product's layout.
    The prototype's `seq` is step 2's (section 4.4): not in step 1's schema, compared once it is."""
    pytest.importorskip("pyarrow")
    from tensorquill import flight

    p5 = _prototype()
    assert set(p5.flight_schema_v5().names) ^ set(flight.flight_schema().names) <= {"seq"}
    for record in [*case["records"], *(case.get("clean_records") or [])]:
        row = flight.record_to_row(copy.deepcopy(record))
        prototype = p5.record_to_row_v5(copy.deepcopy(record))
        assert set(prototype) ^ set(row) <= {"seq"}, record["step"]
        for column in set(row) & set(prototype):
            assert _jsonable(row[column]) == _jsonable(prototype[column]), (record["step"], column)


def _brief(incident: dict) -> dict:
    out = {"kind": incident["kind"], "step": incident.get("step")}
    for key in ("layers", "modules", "gradients", "routers", "router", "parameters", "experts"):
        if key in incident:
            value = incident[key]
            out[key] = sorted(value) if isinstance(value, dict) else value
    return out


def _protocol_4_report(records: list[dict]) -> dict:
    """`server_report` on protocol 4's rows (`flight_v4.py`, frozen at 5c286ed), as an older SDK sends them:
    its stream takes `_row`, `flight_schema` and `rows_batch` from `tensorquill.flight` when called."""
    from tensorquill import flight
    from tensorquill.server_detect import server_report

    v4 = _prototype().v4
    saved = {name: getattr(flight, name) for name in ("_row", "flight_schema", "rows_batch")}
    try:
        for name in saved:
            setattr(flight, name, getattr(v4, name))
        return server_report(records)
    finally:
        for name, value in saved.items():
            setattr(flight, name, value)


def _engine(path: str, records: list[dict], kinds: list[str]) -> dict:
    if path == "python":
        built = evidence.build_evidence(copy.deepcopy(records), detectors="python")
        return {"incidents": [_brief(i) for i in built["incidents"]],
                "kind_checks": {k: (built["kind_checks"].get(k) or {}).get("value") for k in kinds},
                "run_state": (built.get("run_state") or {}).get("state")}
    from tensorquill.server_detect import server_report

    report = _protocol_4_report(copy.deepcopy(records)) if path == "rust_raw_v4" else server_report(copy.deepcopy(records))
    return {"incidents": [_brief(i) for i in report["incidents"]],
            "kind_checks": {k: (report["kind_checks"].get(k) or {}).get("value") for k in kinds}}


def _engine_params():
    out = []
    for case in ROWS:
        for path in ("python", "rust_raw", "rust_raw_v4"):
            if path not in case["paths"] or path not in case["expect"]:
                continue
            pinned = case["today"].get(path) == case["expect"][path]
            landed = (case["name"], path) in LANDED
            marks = []
            if path in ("rust_raw", "rust_raw_v4"):
                from tensorquill.server_detect import find_binary

                marks.append(pytest.mark.skipif(find_binary() is None or importlib.util.find_spec("pyarrow") is None,
                                                reason="tensorquill-server is not built, or pyarrow is not installed"))
            if not pinned and not landed:
                reason = {"python": LANE_PY_NO_LAYER if case["name"] == "layers_only_entries_not_object" else LANE_PY,
                          "rust_raw": LANE_RUST, "rust_raw_v4": LANE_RUST_V4}[path]
                marks.append(pytest.mark.xfail(strict=True, reason=reason))
            kind = "pin" if pinned else "landed" if landed else "target"
            out.append(pytest.param(case, path, marks=marks, id=f"{case['name']}-{path}-{kind}"))
    return out


RUST_SDK = [c for c in ROWS if "rust_sdk" in c["paths"] and "rust_raw" in c["expect"]]


@pytest.mark.parametrize("case", RUST_SDK, ids=[c["name"] for c in RUST_SDK])
def test_the_sdk_s_rust_path_reports_the_rust_expectation(case):
    """The case's `rust_sdk` path, `build_evidence(detectors="rust")` (lane AL-3's reader fed by lane
    AL-1's rows), reports on `records` what `expect["rust_raw"]` says. The cases record no `today` for
    this path; it holds on the joined tree (92e471a), so it is an ordinary test."""
    pytest.importorskip("pyarrow")
    from tensorquill.server_detect import find_binary

    if find_binary() is None:
        pytest.skip("tensorquill-server is not built")
    want = case["expect"]["rust_raw"]
    built = evidence.build_evidence(copy.deepcopy(case["records"]), detectors="rust")
    assert built["detectors"] == "rust"
    assert [_brief(i) for i in built["incidents"]] == want["incidents"]
    assert {k: (built["kind_checks"].get(k) or {}).get("value") for k in sorted(want["kind_checks"])} == want["kind_checks"]


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


LANE_LATE = ("lane AL-5 (capture.py, step 2): a value logged late inside a compiled function is drained before the "
             "flush for a newly trainable parameter, contract section 4.8")
LEFT_OUT = "left out the scalar"
UNFROZEN = "frozen when the session started"


def _late_before_unfreeze(compiled: bool, every: int):
    """Section 4.8's scenario: the first layer frozen at `init`; steps 0 and 1, each followed by an
    evaluation under "eval" that logs `eval/x`; the layer is made trainable, so step 2's pre-step hook
    flushes the ring (`_take_unfrozen`) before step 2 ends. Returns step 1's scalars, the value logged
    after step 1 and the warnings. The evaluation is mirrored on wave AK's compiled tests, which run
    unguarded on CI's whole matrix (torch 2.6 to 2.15).

    Raises `RuntimeError`, never `AssertionError`, when the scenario itself is not reached (the
    evaluation not compiled as one frame, or no unfreeze flush), so the strict target below cannot
    pass for a wrong reason or fail for one."""
    import warnings

    import torch
    from torch import nn

    import tensorquill

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    model[0].requires_grad_(False)
    sink = tensorquill.MemorySink()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    session = None

    def evaluate(x):
        with torch.no_grad():
            value = model(x).pow(2).mean()
        session.log({"eval/x": value})
        return value

    counter = None
    if compiled:
        from torch._dynamo.testing import CompileCounter

        torch._dynamo.reset()
        counter = CompileCounter()
        evaluate = torch.compile(evaluate, backend=counter, fullgraph=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, sink=sink, every=every, weights=1, updates=True, run_id="al_late",
                                   heartbeat_seconds=None, stack_seconds=None)
        try:
            for step in range(3):
                if step == 2:
                    model[0].requires_grad_(True)  # trainable now: this step's pre-step hook flushes the ring
                model(torch.randn(8, 4)).pow(2).mean().backward()
                opt.step()  # the TensorQuill step ends here (auto_step)
                opt.zero_grad()
                if step in (0, 1):  # twice, so one frame for both shows nothing recompiled (section 4.8, item 3)
                    with session.phase("eval"):
                        logged = evaluate(torch.full((8, 4), float(step + 1))).item()
        finally:
            session.close()
            if compiled:
                torch._dynamo.reset()
    messages = [str(w.message) for w in caught]
    if compiled and counter.frame_count != 1:
        raise RuntimeError(f"the evaluation compiled {counter.frame_count} frames, not one: the traced path was not taken")
    if sum(UNFROZEN in m for m in messages) != 1:
        raise RuntimeError(f"no flush for a newly trainable parameter: {messages}")
    step_1 = next(r for r in sink.records if r["step"] == 1).get("scalars", {})
    return step_1, logged, messages


@pytest.mark.parametrize("every", [1, 4])
def test_the_same_value_logged_eagerly_reaches_its_step(every):
    """Case `scalar_eager_late_before_unfreeze`, pinned: the reference the compiled run must match."""
    step_1, logged, messages = _late_before_unfreeze(False, every)
    assert step_1.get("eval/x") == pytest.approx(logged)
    assert not [m for m in messages if LEFT_OUT in m]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_LATE)
@pytest.mark.parametrize("every", [1, 4])
def test_a_value_logged_late_inside_a_compiled_function_reaches_its_step(every):
    """Case `scalar_compiled_late_before_unfreeze`, the target: on step 1's own record, as eagerly, and no
    left-out warning. Today the flush for the unfrozen layer writes step 1 without draining
    `_late_traced`, and the value is left out with a warning that names the wrong cause."""
    step_1, logged, messages = _late_before_unfreeze(True, every)
    assert not [m for m in messages if LEFT_OUT in m]
    assert step_1.get("eval/x") == pytest.approx(logged)


def test_a_scalar_record_is_refused_by_the_base_readers():
    """Pinned as the base does it, for section 5's matrix: today's `load_records` refuses a file with a
    scalar-only record, naming its version. Kept only until step 2 of the wave lands (its lane removes
    this test with the target above)."""
    case = next(c for c in SCALAR if c["name"] == "scalar_old_reader")
    assert case["today"]["python"].startswith("MalformedRecordError: schema_version is 2")
