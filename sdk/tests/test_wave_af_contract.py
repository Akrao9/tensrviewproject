"""The cases of `WAVE_AF_CONTRACT.md`: a record whose identity field is not valid (section 1), and
the provenance keys `first_divergence` lists as series (section 2).

A `pin` is run here against the tree, so behaviour the page keeps cannot move unnoticed. A `new`
case is not built on the base: this file checks only that it is well formed, that its field is
named on the page, and that its `today` (a record of the base, never recomputed) is not already
its target. The lane that builds a rule turns each new case into a test of its own boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorquill import evidence as ev

ROOT = Path(__file__).parents[2]
DOCUMENT = json.loads((Path(__file__).parent / "fixtures" / "wave_af" / "cases.json").read_text())
CASES = DOCUMENT["cases"]
PINS = [(name, body) for name, body in CASES.items() if body["status"] == "pin"]
NEW = [(name, body) for name, body in CASES.items() if body["status"] == "new"]
BOUNDARIES = {"python", "rust", "cli_python", "cli_rust", "detect"}


def satisfied(today: dict, expected: dict) -> bool:
    """Whether what the base answered at one boundary is already what the page asks there."""
    if expected.get("reads"):
        return today.get("reads") is True
    if "raises" in expected:
        return today.get("raises") == expected["raises"]
    return (today.get("exit") == expected["exit"] and not today.get("traceback", False)
            and expected["names"] in today.get("last_line", ""))


def test_the_cases_are_those_of_the_page_and_its_base():
    assert DOCUMENT["contract"] == "WAVE_AF_CONTRACT.md"
    assert len(DOCUMENT["base"]) == 40
    assert len(CASES) == 33 and len(PINS) == 6 and len(NEW) == 27
    page = (ROOT / DOCUMENT["contract"]).read_text()
    for name, body in CASES.items():
        assert body["section"] in (1, 2), name
        if body["section"] == 1:
            assert f"`{body['field']}`" in page, name
            assert set(body["expected"]) == set(body["today"]) <= BOUNDARIES, name
    assert "divergence_provenance_keys_are_not_series" in page


@pytest.mark.parametrize("name,body", NEW, ids=[name for name, _ in NEW])
def test_a_new_case_is_not_what_the_base_already_does(name, body):
    if body["section"] == 2:
        assert body["today"] != body["expected"]
        return
    assert not all(satisfied(body["today"][b], body["expected"][b]) for b in body["today"])


@pytest.mark.parametrize("name,body", PINS, ids=[name for name, _ in PINS])
@pytest.mark.parametrize("engine", ["python", "rust"])
def test_a_pinned_record_is_still_read(name, body, engine, tmp_path):
    """An absent `schema_version`, `rank` or `attempt`, and a run ID of any kind, are read by both
    engines as on the base. The Rust half needs the server binary the tests are given."""
    if engine == "rust":
        from tensorquill import server_detect

        if server_detect.find_binary() is None:
            pytest.skip("no tensorquill-server binary")
    path = tmp_path / "one.jsonl"
    path.write_text(json.dumps(body["given"]["record"]) + "\n")
    assert isinstance(ev.build_evidence(path, detectors=engine), dict)
