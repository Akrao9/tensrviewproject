"""The cases of `CHECKPOINT_TENSOR_CONTRACT.md` (TrainCheck B14): whether a checkpoint holds the
model's tensors, and what a resume says of it.

It checks that every case is named on the page and well formed. A coverage case's `today` is what
the manifest says without a model (`checkpoint_manifest(state)`, as for a save), which the page
keeps as it was, so it is still run here as a pin of that rule. The records cases are built in the
Python detectors (lane AC1, Nov 21, 2026): each is held to its `expected`, and its `today`, what
the detector answered before, is kept in `cases.json` as the record of what changed. Each coverage
case is run through a session against its `expected` in `test_lane_ac1_checkpoint.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import conftest
import pytest
import torch

from tensorquill import evidence as ev
from tensorquill.data import checkpoint_manifest, files_manifest
from tensorquill.data_checks import detect_resume_state

ROOT = Path(__file__).parents[2]
DOCUMENT = json.loads((Path(__file__).parent / "fixtures" / "checkpoint" / "cases.json").read_text())
CASES = DOCUMENT["cases"]
COVERAGE = [(name, body) for name, body in CASES.items() if body["group"] == "coverage"]
RECORDS = [(name, body) for name, body in CASES.items() if body["group"] == "records"]
LISTS = ("missing", "missing_frozen", "missing_buffers", "unexpected", "shape_differs", "tied")


def test_the_file_holds_the_cases_the_page_names():
    page = conftest.contract_page("CHECKPOINT_TENSOR_CONTRACT.md")
    assert len(COVERAGE) == 39 and len(RECORDS) == 17
    assert sum(body["status"] == "pin" for body in CASES.values()) == 4
    for name, body in CASES.items():
        assert f"`{name}`" in page, f"{name} is not named on the page"
        assert body["status"] in ("pin", "new") and body["says"] and body["expected"]


def test_no_list_names_more_than_the_page_allows_and_each_is_sorted():
    def lists(value):
        if isinstance(value, dict):
            for key, inner in value.items():
                if key in LISTS and isinstance(inner, dict) and "names" in inner:
                    yield inner
                yield from lists(inner)
        elif isinstance(value, list):
            for inner in value:
                yield from lists(inner)

    found = list(lists(CASES))
    assert found
    for one in found:
        assert len(one["names"]) <= DOCUMENT["names_listed"] == 50 and one["names"] == sorted(one["names"])
        assert isinstance(one["count"], int) and (one["count"] >= len(one["names"]) or one["count"] == 0)


def _tensors(entry):
    return {name: torch.zeros(shape) for name, shape in entry.items()}


def _manifest_today(given):
    if "folder" in given:
        return files_manifest(given["folder"])
    if "bare" in given:
        return checkpoint_manifest(_tensors(given["bare"]))
    return checkpoint_manifest({
        key: ({"state": {0: {"step": torch.tensor(9.0)}}, "param_groups": [{"lr": 1e-3}]}
              if isinstance(value, str) else _tensors(value))
        for key, value in given["checkpoint"].items()  # `loader` and `model_note` are not read today
    })


@pytest.mark.parametrize("name,body", COVERAGE, ids=[n for n, _ in COVERAGE])
def test_a_coverage_case_states_what_the_manifest_says_today(name, body):
    manifest = _manifest_today(body["given"])
    assert body["today"] == {"model_component": "model" in manifest["components"], "tensors": "tensors" in manifest}
    built = "tensors" in body["expected"] and body["today"]["tensors"]
    same = body["today"]["model_component"] == body["expected"]["model_component"] and (
        built or body["expected"].get("tensors_key") is False)
    assert same == (body["status"] == "pin")


@pytest.mark.parametrize("name,body", RECORDS, ids=[n for n, _ in RECORDS])
def test_a_records_case_is_what_the_python_detector_says(name, body):
    found = detect_resume_state([dict(record) for record in body["records"]])[1]
    answer = {"fires": found is not None, **({"findings": found["changes"][0]["findings"]} if found else {})}
    expected = body["expected"]
    assert answer == {"fires": expected["fires"], **({"findings": expected["findings"]} if expected["fires"] else {})}
    # A pin's `today` was already its expectation; a new case's was not, so this lane changed it.
    was = body["today"]["fires"] == expected["fires"] and body["today"].get("findings", []) == expected.get(
        "findings", []) and expected["checkpoint_tensors"] is None
    assert was == (body["status"] == "pin")
    assert ev.build_evidence(body["records"], detectors="python").get("checkpoint_tensors") == expected["checkpoint_tensors"]


def test_a_finding_names_what_its_record_names():
    """A `missing_tensors` or `shape_differs` finding carries its list's count and names, unchanged."""
    seen = 0
    for _, body in RECORDS:
        load = next((r["checkpoint"].get("load") for r in body["records"] if "checkpoint" in r), None)
        for finding in body["expected"].get("findings", []):
            key = {"missing_tensors": "missing", "shape_differs": "shape_differs"}.get(finding["finding"])
            if key is not None:
                assert finding["count"] == load["tensors"][key]["count"]
                assert finding["tensors"] == load["tensors"][key]["names"]
                seen += 1
    assert seen == 4
