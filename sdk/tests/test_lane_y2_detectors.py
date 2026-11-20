"""Lane Y2's tests (Nov 20, 2026): the detector engines.

- L13: `update_spike` compares only the parameters both of the records it compares name, in
  both engines (Python's `detect_update_spikes`, Rust's `Updates` in `server/src/detect.rs`),
  held to each other on the same records.
- DET-13, the open part: a step a compact file keeps has its expert load table when the step
  it is compared with was left out, with every delta `n/a`.
- L15, the evidence half: `read_server_report` refuses the thresholds the detectors refuse,
  and evidence built from a server's report carries the report's `head_trends`.

The Rust halves need `pyarrow` and the server binary (`TENSORQUILL_SERVER_BIN`, or one built
in `server/target`); they are skipped without them.
"""

from __future__ import annotations

import json
import math

import pytest

import tensorquill
from tensorquill import evidence as ev

try:
    import pyarrow  # noqa: F401

    from test_flight_e2e import BINARY
except ImportError:  # pragma: no cover - the `flight` extra is not installed
    BINARY = None

needs_rust = pytest.mark.skipif(BINARY is None, reason="pyarrow or tensorquill-server is not here")


@pytest.fixture
def rust(monkeypatch):
    from tensorquill.server_detect import BINARY_ENV

    monkeypatch.setenv(BINARY_ENV, str(BINARY))


# --- L13: update_spike over the parameters both records name ---------------------------------

def _row(at: int, sizes: dict[str, float], total: float = 1.0, attempt: int = 1) -> dict:
    return {"schema_version": 1, "run_id": "y2-updates", "rank": 0, "attempt": attempt, "step": at,
            "ts": float(at), "grad_norm": total, "update_rms": sizes}


def _quiet(count: int, start: int = 0, **extra: float) -> list[dict]:
    """The largest update moves 0.001 on every step; `extra` are more parameters, held still."""
    return [_row(at, {"big": 0.010 + 0.001 * (at % 2), "small": 0.002, **extra}) for at in range(start, start + count)]


def _spikes(evidence: dict) -> list[dict]:
    return [i for i in evidence["incidents"] if i["kind"] == "update_spike"]


def _both_engines(records: list[dict], **options) -> list[tuple[str, list[dict]]]:
    engines = [("python", tensorquill.build_evidence(records, **options))]
    if BINARY is not None:
        engines.append(("rust", tensorquill.build_evidence(records, detectors="rust", server_binary=BINARY, **options)))
    return [(name, _spikes(found)) for name, found in engines]


def _appears(window: int) -> list[dict]:
    """`window + 2` quiet records of two parameters, then one that names a third with a far larger
    update, on a step whose gradient norm rose (it now counts the third's gradient too)."""
    return _quiet(window + 2) + [_row(window + 2, {"big": 0.010, "small": 0.002, "late": 0.300}, total=5.0)]


@pytest.mark.parametrize("window", [4, 100])
def test_l13_a_parameter_that_appears_raises_no_update_spike_in_python(window):
    records = _appears(window)
    # Without the third parameter the step is quiet; with it the largest update went 0.011 -> 0.3.
    assert ev.detect_update_spikes(records, window=window) is None
    assert _spikes(tensorquill.build_evidence(records, update_window=window)) == []


@needs_rust
@pytest.mark.parametrize("window", [4, 100])
def test_l13_a_parameter_that_appears_raises_no_update_spike_in_rust(window, rust):
    records = _appears(window)
    assert _spikes(tensorquill.build_evidence(records, detectors="rust", update_window=window)) == []


def test_l13_a_parameter_that_goes_away_raises_none_either():
    # The records before name a parameter with the largest update, moving 0.001 a step (so the
    # baseline is 0.001 over all of them); this one does not name it.
    records = [_row(at, {"big": 0.010 + 0.001 * (at % 2), "small": 0.002, "late": 0.300 + 0.001 * (at % 2)})
               for at in range(6)]
    records.append(_row(6, {"big": 0.010, "small": 0.002}, total=5.0))
    for engine, found in _both_engines(records, update_window=4):
        assert found == [], engine


def test_l13_a_spike_on_a_parameter_both_records_name_still_fires():
    # The third parameter appears on the same step as a jump of `small`, which both records name.
    records = _quiet(6) + [_row(6, {"big": 0.010, "small": 0.300, "late": 0.900}, total=5.0)]
    found = _both_engines(records, update_window=4)
    for engine, spikes in found:
        assert len(spikes) == 1, (engine, spikes)
        hit = spikes[0]
        assert (hit["step"], hit["spike_steps"], hit["spike_count"]) == (6, [6], 1), engine
        # R is the largest update over the two parameters both records name: 0.3, not 0.9.
        assert (hit["update_rms"], hit["previous"]) == (0.3, 0.011), engine
        assert hit["change"] == pytest.approx(0.289) and hit["baseline"] == pytest.approx(0.001), engine
        assert [p["name"] for p in hit["parameters"]] == ["small", "big"], engine
    if len(found) == 2:
        assert json.dumps(found[0][1], sort_keys=True) == json.dumps(found[1][1], sort_keys=True)


def test_l13_after_it_appears_the_new_parameter_is_compared_like_any_other():
    # From the record after it appeared on, both records name it: a jump of its own is a spike.
    # It moves 0.001 a step, and is then the largest on both sides.
    records = _appears(4) + [_row(at, {"big": 0.010, "small": 0.002, "late": 0.300 + 0.001 * (at % 2)})
                             for at in range(7, 11)]
    records.append(_row(11, {"big": 0.010, "small": 0.002, "late": 0.900}, total=5.0))
    for engine, spikes in _both_engines(records, update_window=4):
        assert [(s["step"], s["update_rms"], s["previous"]) for s in spikes] == [(11, 0.9, 0.3)], engine
        assert spikes[0]["parameters"][0]["name"] == "late", engine


def test_l13_records_that_share_no_parameter_are_not_compared():
    # A record whose parameters are all new is not judged and adds no change to the baseline;
    # the next record is compared with it.
    records = _quiet(6)
    records.append(_row(6, {"other": 5.0}, total=5.0))
    records.append(_row(7, {"other": 5.001}, total=1.0))  # a change of 0.001 joins the baseline
    records.append(_row(8, {"other": 5.3}, total=5.0))  # 0.299 past a baseline of 0.001: flagged
    for engine, spikes in _both_engines(records, update_window=4):
        assert [(s["step"], s["previous"]) for s in spikes] == [(8, 5.001)], engine


def _grads_row(at: int, sizes: dict[str, float], grads: dict[str, float], clip: float | None = None) -> dict:
    out = _row(at, sizes, total=math.sqrt(math.fsum(v * v for v in grads.values())))
    out["grad_norms"] = grads
    if clip is not None:
        out["clip_grad_norm"] = clip
    return out


@pytest.mark.parametrize("clipped", [False, True])
def test_l13_the_gradient_rise_is_taken_over_the_shared_parameters_too(clipped):
    # The new parameter carries almost all of the gradient: the whole total rises by about 500,
    # which with an ordinary change of R (0.001) would pass the limit on its own. Over the two
    # parameters both records name it did not rise at all.
    def quiet_row(at):
        grads = {"big": 0.6, "small": 0.8}
        return _grads_row(at, {"big": 0.010 + 0.001 * (at % 2), "small": 0.002}, grads,
                          clip=10.0 if clipped else None)

    records = [quiet_row(at) for at in range(6)]
    grads = {"big": 0.6, "small": 0.8, "late": 500.0}
    # Clipped to a norm of 1: every gradient scaled by one factor, the total before clipping kept.
    if clipped:
        whole = math.sqrt(math.fsum(v * v for v in grads.values()))
        grads = {name: v / whole for name, v in grads.items()}
    records.append(_grads_row(6, {"big": 0.010, "small": 0.002, "late": 0.010}, grads,
                              clip=10.0 * 500.001 if clipped else None))
    for engine, spikes in _both_engines(records, update_window=4):
        assert spikes == [], engine
    # The same jump in the whole total with the parameter set unchanged is still a rise.
    same = [quiet_row(at) for at in range(6)]
    same.append(_grads_row(6, {"big": 0.010, "small": 0.002}, {"big": 0.6, "small": 500.0},
                           clip=10.0 * 500.0 if clipped else None))
    for engine, spikes in _both_engines(same, update_window=4):
        assert [s["step"] for s in spikes] == [6], engine


def test_l13_a_spike_with_a_new_parameter_reports_the_shared_gradient_totals():
    records = [_grads_row(at, {"big": 0.010 + 0.001 * (at % 2), "small": 0.002}, {"big": 0.6, "small": 0.8})
               for at in range(6)]
    # `small` jumps and its own gradient rises (0.8 -> 4.8): the shared total goes 1 -> sqrt(0.36 + 23.04).
    records.append(_grads_row(6, {"big": 0.010, "small": 0.300, "late": 0.010},
                              {"big": 0.6, "small": 4.8, "late": 100.0}))
    found = _both_engines(records, update_window=4)
    for engine, spikes in found:
        (hit,) = spikes
        assert hit["grad_norm"] == pytest.approx(math.sqrt(0.36 + 23.04), rel=1e-12), engine
        assert hit["grad_norm_before"] == pytest.approx(1.0, rel=1e-12), engine
        assert hit["grad_rise"] == pytest.approx(math.sqrt(0.36 + 23.04) - 1.0, rel=1e-12), engine
    if len(found) == 2:
        assert json.dumps(found[0][1], sort_keys=True) == json.dumps(found[1][1], sort_keys=True)


def test_l13_without_per_parameter_gradients_the_whole_totals_are_compared():
    # A record with no `grad_norms` (or a norm that is not finite) cannot be split: the whole
    # totals are compared, as before the fix.
    records = _quiet(6) + [_row(6, {"big": 0.010, "small": 0.002, "late": 0.010}, total=500.0)]
    for engine, spikes in _both_engines(records, update_window=4):
        assert [s["step"] for s in spikes] == [6], engine
    with_nan = [_grads_row(at, {"big": 0.010 + 0.001 * (at % 2), "small": 0.002}, {"big": 0.6, "small": 0.8})
                for at in range(6)]
    with_nan.append(_grads_row(6, {"big": 0.010, "small": 0.002, "late": 0.010},
                               {"big": 0.6, "small": 0.8, "late": 500.0}))
    with_nan[-1]["grad_norms"]["small"] = math.nan
    for engine, spikes in _both_engines(with_nan, update_window=4):
        assert [s["step"] for s in spikes] == [6], engine


def test_l13_the_known_cases_are_judged_as_before():
    """Records whose parameters never change keep every result: the update-size golden cases and
    parity cases are checked by `test_golden.py` and `test_server_detect.py`; this is the short
    form of the rule's own first test (`test_update_size.py`)."""
    records = _quiet(6) + [_row(6, {"big": 0.010, "small": 0.300}, total=5.0)]
    hit = ev.detect_update_spikes(records, window=4)
    assert (hit["step"], hit["update_rms"], hit["previous"], hit["grad_rise"]) == (6, 0.3, 0.011, 4.0)
    assert hit["threshold"] == pytest.approx(0.01)


# --- DET-13: a kept step's expert load when the step it is compared with was left out --------

def _moe(incidents: list[dict], quiet_until: int = 0) -> dict:
    """Hand-built evidence of two routers of four experts over 200 steps (as lane X5's tests):
    router 1 routes no token before `quiet_until`, router 2 routes from step 0."""
    samples = []
    for step in range(200):
        for layer, counts in ((1, [0, 0, 0, 0] if step < quiet_until else [10 + step % 3, 10, 10, 10]),
                              (2, [12 + 6 * (step % 2), 9 + step % 2, 10, 10])):
            samples.append({"step": step, "layer": layer, "router": f"model.layers.{layer}.mlp.router",
                            "counts": counts, "routed_tokens": sum(counts), "diversity": 0.9,
                            "overloaded": [], "dead": [], "starved": []})
    return {"run_id": "y2-moe", "expert_count": 4, "layer_count": 2, "samples": samples, "incidents": incidents}


def _compact(full: dict, window: int) -> dict:
    from tensorquill.compact import compact_evidence

    return json.loads(json.dumps(compact_evidence(full, window=window)))  # as read back from the file


def _deltas(table: str) -> list[str]:
    return [line.split("|")[3].strip() for line in table.splitlines()[4:]]


def _loads(table: str) -> list[str]:
    return [line.split("|")[2].strip() for line in table.splitlines()[4:]]


def test_det13_a_kept_step_whose_baseline_was_left_out_shows_its_loads_with_no_delta():
    full = _moe([{"kind": "moe_imbalance", "step": 100, "layer": 2}])
    small = _compact(full, window=3)
    assert small["compact"]["kept_steps"] == [[97, 103]]
    table = tensorquill.expert_load_table(small, 97, 2)
    heading = table.splitlines()[0]
    assert heading.startswith("**Layer 02 expert load** · step 97 ") and "left out" in heading, heading
    assert "vs." not in heading and "no earlier step" not in heading
    # The loads are the full evidence's; every delta is n/a, never +0.00, and rows go by load.
    whole = tensorquill.expert_load_table(full, 97, 2)
    assert sorted(_loads(table)) == sorted(_loads(whole)) and _deltas(whole) != ["n/a"] * 4
    assert _deltas(table) == ["n/a"] * 4
    shares = [float(load.rstrip("%")) for load in _loads(table)]
    assert shares == sorted(shares, reverse=True)
    # The next kept step is compared with 97, which is here: the full evidence's table.
    assert tensorquill.expert_load_table(small, 98, 2) == tensorquill.expert_load_table(full, 98, 2)


def test_det13_with_a_window_of_0_the_incidents_own_step_has_its_table():
    full = _moe([{"kind": "moe_imbalance", "step": 100, "layer": 2}])
    small = _compact(full, window=0)
    assert small["compact"]["kept_steps"] == [[100, 100]]
    table = tensorquill.expert_load_table(small, 100, 2)
    assert "left out" in table.splitlines()[0] and _deltas(table) == ["n/a"] * 4


def test_det13_a_later_window_is_never_compared_with_an_earlier_one():
    # Two windows: the first step of the second is compared, on the full evidence, with the step
    # before it, which was left out. The routed steps of the first window are not that step.
    full = _moe([{"kind": "moe_imbalance", "step": 50, "layer": 2}, {"kind": "moe_imbalance", "step": 150, "layer": 2}])
    small = _compact(full, window=3)
    assert small["compact"]["kept_steps"] == [[47, 53], [147, 153]]
    table = tensorquill.expert_load_table(small, 147, 2)
    assert "left out" in table.splitlines()[0] and "vs. 53" not in table and _deltas(table) == ["n/a"] * 4
    assert tensorquill.expert_load_table(small, 148, 2) == tensorquill.expert_load_table(full, 148, 2)


def test_det13_a_step_that_was_left_out_still_says_so():
    from tensorquill.compact import SamplesOmitted

    small = _compact(_moe([{"kind": "moe_imbalance", "step": 100, "layer": 2}]), window=3)
    for step in (50, 96, 104):
        with pytest.raises(SamplesOmitted, match=rf"step {step} is not here"):
            tensorquill.expert_load_table(small, step, 2)


def test_det13_an_old_compact_file_without_the_unrouted_marker_shows_the_table_too():
    # Router 1 routed nothing before step 97. A file written before `unrouted_before` cannot tell
    # that from a step left out, so it shows the loads with no delta rather than refusing.
    full = _moe([{"kind": "moe_imbalance", "step": 100, "layer": 2}], quiet_until=97)
    old = _compact(full, window=3)
    old["compact"].pop("unrouted_before")
    table = tensorquill.expert_load_table(old, 97, 1)
    assert "left out" in table.splitlines()[0] and _deltas(table) == ["n/a"] * 4
    assert sorted(_loads(table)) == sorted(_loads(tensorquill.expert_load_table(full, 97, 1)))


def test_det13_a_compact_file_that_kept_every_step_is_the_full_evidence():
    full = _moe([{"kind": "moe_imbalance", "step": 100, "layer": 2}])
    whole = _compact(full, window=500)
    for step in (0, 97, 150):
        assert tensorquill.expert_load_table(whole, step, 2) == tensorquill.expert_load_table(full, step, 2)


# --- L15: a server report's thresholds and head trends ---------------------------------------

@pytest.mark.parametrize("bad", [
    {"overload_threshold": -0.1}, {"drift_tolerance": math.nan}, {"starve_ratio": math.inf},
    {"starve_window": 0}, {"update_window": 2.5}, {"top_k": 0}, {"top_k": True}, {"starve_window": True},
    {"top_k": {"router": 0}}, {"overload_threshold": "0.5"},
])
def test_l15_read_server_report_refuses_the_thresholds_the_detectors_refuse(tmp_path, bad):
    folder = tmp_path / "store" / "runs" / "r" / "rank=0"
    with pytest.raises(ValueError, match="the detectors refuse these thresholds") as error:
        ev.read_server_report(folder, **bad)
    # The words build_evidence uses for the same value.
    with pytest.raises(ValueError) as same:
        tensorquill.build_evidence([], **bad)
    assert str(error.value) == str(same.value)


def test_l15_read_server_report_takes_none_as_not_given(tmp_path):
    folder = tmp_path / "store" / "runs" / "r" / "rank=0"
    assert ev.read_server_report(folder)["state"] == "nothing_stored"
    given = dict.fromkeys(ev.THRESHOLD_NAMES)
    assert ev.read_server_report(folder, **given)["state"] == "nothing_stored"
    assert ev.read_server_report(folder, starve_window=10, top_k={"router": 2})["state"] == "nothing_stored"


HEAD = {"module": "lm_head", "positions": 8, "nonfinite_positions": 0, "vocab": 32, "calls": 1,
        "input_dtype": "float32", "weight_dtype": "float32"}


def _probed_run(run: str) -> list[dict]:
    """Two attempts of one rank: attempt 2 resumes at step 30, so attempt 1's probes from there
    on were rolled back. Among the probes: one that did not read the head, one that failed, one
    whose numbers include a bool and a whole number, and two that counted overflowing positions."""
    def row(step: int, attempt: int, probe: dict | None = None) -> dict:
        out = {"schema_version": 1, "run_id": run, "rank": 0, "attempt": attempt, "step": step,
               "ts": float(step), "grad_norm": 1.0}
        return {**out, "probe": probe} if probe is not None else out

    def head(step: int, **more) -> dict:
        return {"head": {**HEAD, "norm": 4.0 + step / 10, "mean": 0.1 - step / 1000, "log_z": 3.0,
                         "gap": 0.01 * (1 + step % 3), "gap_relative": 0.002, **more}}

    probes = {
        0: head(0),
        5: head(5, overflow_positions=2),
        10: {"head": {**HEAD, "positions": 0, "not_read": "no position had finite FP32 logits",
                      "overflow_positions": 3}},
        15: {"error": "the probe raised"},
        20: head(20, gap=True, log_z=7),
        35: head(35, module="model.lm_head"),  # rolled back by attempt 2
    }
    first = [row(step, 1, probes.get(step)) for step in range(40)]
    later = {30: head(30, norm=9.0), 45: head(45, overflow_positions=1)}
    second = [row(step, 2, later.get(step)) for step in range(30, 50)]
    return first + second


@needs_rust
def test_l15_evidence_from_a_servers_report_carries_head_trends(tmp_path, rust):
    from test_server_detectors import FAST, Stream

    run = "y2-heads"
    records = _probed_run(run)
    python = tensorquill.build_evidence(records)["head_trends"]
    assert python["probes"] == 6 and python["read"] == 5 and python["module"] == "lm_head"
    assert python["overflow_positions"]["positions"] == 6 and python["latest_step"] == 45
    stream = Stream(tmp_path, *FAST, run=run)
    try:
        stream.send([r for r in records if r["attempt"] == 1])
        document = stream.send([r for r in records if r["attempt"] == 2])
    finally:
        assert stream.server.terminate() == 0
    assert document["head_trends"] == python
    folder = stream.incidents_file.parent
    assert ev.read_server_report(folder)["state"] == "current"
    out = ev.watch(folder, tmp_path / "live.json", interval=0.0, polls=1)
    evidence = json.loads(out.read_text())
    assert evidence["detection"]["source"] == "server"
    assert evidence["head_trends"] == python
    # The same as the evidence the server's store gives when its records are judged here.
    assert tensorquill.build_evidence(folder)["head_trends"] == python


@needs_rust
def test_l15_a_report_with_no_probe_head_gives_no_head_trends(tmp_path, rust):
    from test_server_detectors import FAST, Stream

    run = "y2-no-heads"
    records = [{"schema_version": 1, "run_id": run, "rank": 0, "attempt": 1, "step": s, "ts": float(s),
                "grad_norm": 1.0} for s in range(10)]
    stream = Stream(tmp_path, *FAST, run=run)
    try:
        document = stream.send(records)
    finally:
        assert stream.server.terminate() == 0
    assert document["head_trends"] is None
    out = ev.watch(stream.incidents_file.parent, tmp_path / "live.json", interval=0.0, polls=1)
    evidence = json.loads(out.read_text())
    assert evidence["detection"]["source"] == "server" and "head_trends" not in evidence
    assert "head_trends" not in tensorquill.build_evidence(records)
