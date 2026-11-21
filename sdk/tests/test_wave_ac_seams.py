"""Where the lanes of wave AC meet, on the tree that holds all of them.

Lane AC1 built `CHECKPOINT_TENSOR_CONTRACT.md` (TrainCheck B14) in the SDK and the Python
detectors and lane AC2 built section 4 in the server, each against the case file alone and
neither seeing the other's code. Lane AC1 ran the server's path only with a stand-in report, and
lane AC2 read its block from the server's own file, not from the evidence. So no test of either
ran one record through both engines and through `build_evidence`.

Here every case over records goes through both, and both are held to the case and to each other.
So do the records no case states (section 4, "Records no case states"), which lane AC2 settled
while building and lane AC1 was then given as rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorquill import evidence as ev
from tensorquill import server_detect

_CASES = json.loads((Path(__file__).parent / "fixtures" / "checkpoint" / "cases.json").read_text())["cases"]
_RECORDS = [(name, body) for name, body in _CASES.items() if body["group"] == "records"]
_KEYS = ("incidents", "not_checked", "checkpoint_tensors", "kind_checks", "run_state")


def _both(records):
    try:
        server_detect.rust_ready()
    except RuntimeError as error:
        pytest.skip(str(error))
    records = [{"schema_version": 1, "run_id": "case", **record} for record in records]
    return ev.build_evidence(records, detectors="python"), ev.build_evidence(records, detectors="rust")


def _findings(evidence):
    found = [i for i in evidence["incidents"] if i["kind"] == "resume_incomplete"]
    return found[0]["changes"][0]["findings"] if found else None


def test_the_cases_over_records_are_all_here():
    assert len(_RECORDS) == 17


@pytest.mark.parametrize("name,body", _RECORDS, ids=[n for n, _ in _RECORDS])
def test_both_engines_say_the_same_of_a_checkpoint_case_and_what_the_case_says(name, body):
    python, rust = _both(body["records"])
    expected = body["expected"]
    for label, evidence in (("python", python), ("rust", rust)):
        assert _findings(evidence) == (expected["findings"] if expected["fires"] else None), label
        assert evidence.get("checkpoint_tensors") == expected["checkpoint_tensors"], label
    for key in _KEYS:
        assert python.get(key) == rust.get(key), key


def _resumed(tensors, **first):
    """The cases' own shape: ten steps, then a second attempt resuming at step 10 from a load whose
    manifest carries `tensors`; `first` is added to the resumed attempt's first record."""
    components = {"model": {"tensors": 7, "keys": ["model"]}, "optimizer": {"entries": 7, "param_groups": 1, "step": 10},
                  "rng": {}, "data": {}}
    rows = [{"step": s, "rank": 0, "attempt": 1000, "lr": [1e-3], "grad_norm": 1.0} for s in range(10)]
    rows += [{"step": s, "rank": 0, "attempt": 2000, "lr": [1e-3], "grad_norm": 1.0} for s in range(10, 15)]
    rows[10]["checkpoint"] = {"load": {"components": components, "tensors": tensors}, **first.pop("checkpoint", {})}
    rows[10].update(first)
    return rows


_BASE = {"state": "compared", "entry": "model", "by": "name", "model": 7, "trainable": 7, "checkpoint": 5, "matched": 5}
_TWO = ["lstm.bias_hh_l0", "lstm.weight_hh_l0"]
_ODD = {
    "a_count_that_is_a_float": ({**_BASE, "missing": {"count": 2.0, "names": _TWO}}, None),
    "a_count_that_is_a_boolean": ({**_BASE, "missing": {"count": True, "names": _TWO}}, None),
    "a_count_that_is_a_string": ({**_BASE, "missing": {"count": "2", "names": _TWO}}, None),
    "a_count_below_one": ({**_BASE, "missing": {"count": -1, "names": _TWO}}, None),
    "a_count_the_server_cannot_hold": ({**_BASE, "missing": {"count": 2**63, "names": _TWO}}, None),
    "the_largest_count_read": ({**_BASE, "missing": {"count": 2**63 - 1, "names": _TWO}},
                               [{"component": "model", "finding": "missing_tensors", "count": 2**63 - 1, "tensors": _TWO}]),
    "a_list_that_is_not_an_object": ({**_BASE, "missing": _TWO}, None),
    "names_that_are_not_a_list": ({**_BASE, "missing": {"count": 2, "names": "lstm.bias_hh_l0"}},
                                  [{"component": "model", "finding": "missing_tensors", "count": 2, "tensors": []}]),
    "names_with_things_that_are_not_strings": ({**_BASE, "missing": {"count": 3, "names": ["a", 7, None, "b"]}},
                                               [{"component": "model", "finding": "missing_tensors", "count": 3, "tensors": ["a", "b"]}]),
    "scalar_keys_of_other_types": ({"state": "compared", "entry": None, "by": 3, "model": "seven", "trainable": None,
                                    "checkpoint": 5.5, "matched": [1]}, None),
    "an_unavailable_record_with_lists": ({"state": "unavailable", "why": "ambiguous", "missing": {"count": 2, "names": _TWO},
                                          "unexpected": {"count": 1, "names": ["x"]}}, None),
    "a_state_that_is_neither_word": ({**_BASE, "state": "partly", "missing": {"count": 2, "names": _TWO}}, None),
    "no_state": ({"missing": {"count": 2, "names": _TWO}}, None),
    "keys_the_block_does_not_carry": ({**_BASE, "tied": {"count": 1, "names": ["t"]}, "prefix": {"side": "model", "removed": "net."},
                                       "closest": {"entry": "x"}, "candidates": ["a", "b"],
                                       "shape_differs": {"count": 1, "names": ["head.weight"], "shapes": {"head.weight": {}}}},
                                      [{"component": "model", "finding": "shape_differs", "count": 1, "tensors": ["head.weight"]}]),
}


@pytest.mark.parametrize("name", list(_ODD))
def test_both_engines_say_the_same_of_a_record_no_case_states(name):
    tensors, findings = _ODD[name]
    python, rust = _both(_resumed(tensors))
    assert _findings(python) == findings
    for key in _KEYS:
        assert python.get(key) == rust.get(key), key
    block = python.get("checkpoint_tensors")
    if name in ("a_state_that_is_neither_word", "no_state"):
        assert block is None
    else:
        assert block["resumes"] == 1 and not {"tied", "prefix", "closest", "candidates", "shapes"} & {
            *block["entries"][0], *(block["entries"][0].get("shape_differs") or {})}


def test_tensor_findings_come_after_the_snapshots_and_the_scalers_in_both_engines():
    """Today's findings from the optimizer snapshot and the loss scale come after the manifest's;
    the tensors come after all of them (section 4)."""
    tensors = {**_BASE, "missing": {"count": 2, "names": _TWO}}
    rows = _resumed(tensors, checkpoint={"optimizer": {"entries_before": 0, "entries_after": 7, "step_after": 11}})
    python, rust = _both(rows)
    assert [f["finding"] for f in _findings(python)] == ["empty", "missing_tensors"]
    assert _findings(python) == _findings(rust)
