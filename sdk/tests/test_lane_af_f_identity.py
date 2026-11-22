"""Lane F of wave AF: `tensorquill.identity.check`, by the table "What is valid" of
`WAVE_AF_CONTRACT.md` section 1, on every section 1 case of `fixtures/wave_af/cases.json`."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tensorquill import identity
from tensorquill.identity import MalformedRecordError, check

DOCUMENT = json.loads((Path(__file__).parent / "fixtures" / "wave_af" / "cases.json").read_text())
SECTION_1 = [(name, body) for name, body in DOCUMENT["cases"].items() if body["section"] == 1]
GOOD = {"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 0, "step": 0, "grad_norm": 1.0}


@pytest.mark.parametrize("name,body", SECTION_1, ids=[name for name, _ in SECTION_1])
def test_every_section_1_case_at_the_check(name, body):
    record = body["given"]["record"]
    before = json.dumps(record, sort_keys=True)
    expected = body["expected"]["python"]
    if expected.get("reads"):
        assert check(record, "here") is None
    else:
        with pytest.raises(MalformedRecordError) as raised:
            check(record, "here")
        error = raised.value
        assert isinstance(error, ValueError)
        assert error.field == expected["field"] == body["field"]
        assert error.where == "here"
        assert body["field"] in str(error) and "here" in str(error)
    assert json.dumps(record, sort_keys=True) == before  # nothing changed, nothing filled in


def test_the_error_says_the_field_what_was_found_what_is_read_and_where():
    with pytest.raises(MalformedRecordError) as raised:
        check({**GOOD, "rank": True}, identity.in_file("one.jsonl", 3))
    assert str(raised.value) == "rank is true at one.jsonl, line 3: tensorquill reads an integer of 0 or more"
    assert raised.value.value is True and raised.value.reason == "tensorquill reads an integer of 0 or more"
    with pytest.raises(MalformedRecordError) as raised:
        check({**GOOD, "schema_version": 99}, identity.in_part("part-0-9.parquet", 4))
    assert str(raised.value) == ("schema_version is 99 at part-0-9.parquet, row 4: "
                                 "this tensorquill reads schema versions 0 and 1")
    without_step = dict(GOOD)
    del without_step["step"]
    with pytest.raises(MalformedRecordError) as raised:
        check(without_step, identity.in_list(2))
    assert raised.value.value is MalformedRecordError.ABSENT
    assert str(raised.value).startswith("step is absent at record 2 of the records passed: ")
    with pytest.raises(MalformedRecordError) as raised:
        check({**GOOD, "attempt": "1"})
    assert str(raised.value) == 'attempt is "1": tensorquill reads an integer of 0 or more'
    assert raised.value.where is None


def test_the_first_field_in_the_fixed_order_is_named():
    with pytest.raises(MalformedRecordError) as raised:
        check({"step": None, "attempt": -1, "rank": 1.0, "schema_version": "1"})
    assert raised.value.field == "schema_version"
    with pytest.raises(MalformedRecordError) as raised:
        check({"step": None, "attempt": -1, "rank": 1.0})
    assert raised.value.field == "rank"


def test_no_upper_bound_and_run_id_is_not_looked_at():
    assert check({**GOOD, "step": 2**70, "rank": 2**40, "attempt": 2**63}) is None
    assert check({"step": 5, "run_id": None}) is None
    assert check({"step": 0, "schema_version": 0}) is None


def test_the_newest_schema_version_written_is_read():
    from tensorquill import flight

    assert flight.SCHEMA_VERSION in identity.SCHEMA_VERSIONS


def test_importing_it_loads_neither_torch_nor_pyarrow():
    """The module on its own: the package's `__init__` imports the capture, which imports torch."""
    code = ("import importlib.util, sys; spec = importlib.util.spec_from_file_location('identity', sys.argv[1]); "
            "spec.loader.exec_module(importlib.util.module_from_spec(spec)); "
            "print(sorted(m for m in ('torch', 'pyarrow', 'numpy') if m in sys.modules))")
    done = subprocess.run([sys.executable, "-c", code, identity.__file__], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"
