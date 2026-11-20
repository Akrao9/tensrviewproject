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
