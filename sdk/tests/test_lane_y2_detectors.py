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
