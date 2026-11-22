"""Wave AF, where the lanes meet: every case of `WAVE_AF_CONTRACT.md`, section 1, at every boundary
the page names, on the merged tree with its own server binary.

No lane could run this: lane F built the check, lane C the readers and the command line, lane D
the offline `detect`, each against the base's other halves. Each case's `expected` is the page's;
nothing here is recomputed from the base.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tensorquill import divergence as dv
from tensorquill import evidence as ev
from tensorquill import flight, identity, server_detect

CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_af" / "cases.json").read_text())["cases"]
ONE = [(name, body) for name, body in CASES.items() if body["section"] == 1]
# The command line runs in a process of its own: it must import this tree's package, not another
# one the interpreter has installed.
ENV = {**os.environ, "PYTHONPATH": str(Path(ev.__file__).resolve().parents[1])}
GOOD = {"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 0, "step": 0, "grad_norm": 1.0}


def binary() -> str:
    found = server_detect.find_binary()
    if found is None:
        pytest.skip("no tensorquill-server binary")
    return found


def one_file(tmp_path: Path, body: dict) -> Path:
    path = tmp_path / "one.jsonl"
    path.write_text(json.dumps(body["given"]["record"]) + "\n")
    return path


@pytest.mark.parametrize("name,body", ONE, ids=[n for n, _ in ONE])
@pytest.mark.parametrize("engine", ["python", "rust"])
def test_a_reader_reads_a_valid_record_and_refuses_another_by_name(name, body, engine, tmp_path):
    if engine == "rust":
        binary()
    expected = body["expected"][engine]
    path = one_file(tmp_path, body)
    if expected.get("reads"):
        assert isinstance(ev.build_evidence(path, detectors=engine), dict)
        return
    with pytest.raises(identity.MalformedRecordError) as caught:
        ev.build_evidence(path, detectors=engine)
    assert caught.value.field == expected["field"]
    assert expected["field"] in str(caught.value) and "one.jsonl" in str(caught.value)


@pytest.mark.parametrize("name,body", ONE, ids=[n for n, _ in ONE])
@pytest.mark.parametrize("engine", ["python", "rust"])
def test_the_command_line_exits_2_with_one_line_and_no_traceback(name, body, engine, tmp_path):
    if engine == "rust":
        binary()
    expected = body["expected"][f"cli_{engine}"]
    path, out = one_file(tmp_path, body), tmp_path / "out.json"
    done = subprocess.run([sys.executable, "-m", "tensorquill", str(path), "--detectors", engine, "--output", str(out)],
                          capture_output=True, text=True, timeout=120, env=ENV)
    if expected.get("reads"):
        assert done.returncode == 0, done.stderr
        return
    assert done.returncode == expected["exit"], done.stderr
    assert "Traceback" not in done.stderr
    assert expected["names"] in done.stderr.strip().splitlines()[-1]
    assert not out.exists()


@pytest.mark.parametrize("name,body", ONE, ids=[n for n, _ in ONE])
def test_the_row_is_refused_before_moved_is_touched(name, body):
    expected = body["expected"]["python"]
    moved: list[str] = []
    if expected.get("reads"):
        if isinstance(body["given"]["record"].get("run_id"), str):
            flight.record_to_row(body["given"]["record"], moved=moved)
        return
    with pytest.raises(identity.MalformedRecordError):
        flight.record_to_row({**body["given"]["record"], "moe_top_k": 10**400}, moved=moved)
    assert moved == []


@pytest.mark.parametrize("name,body", [(n, b) for n, b in ONE if "detect" in b["expected"]],
                         ids=[n for n, b in ONE if "detect" in b["expected"]])
@pytest.mark.parametrize("run", [False, True])
def test_the_offline_detect_refuses_the_row_by_name(name, body, run):
    import pyarrow as pa

    expected, field = body["expected"]["detect"], body["field"]
    schema = flight.flight_schema()
    row = flight.record_to_row(GOOD)
    row[field] = body["given"]["record"][field]
    out = io.BytesIO()
    with pa.ipc.new_stream(out, schema) as writer:
        writer.write_batch(pa.RecordBatch.from_pylist([row], schema=schema))
    done = subprocess.run([binary(), "detect", *(["--run"] if run else [])], input=out.getvalue(),
                          capture_output=True, timeout=60)
    assert done.returncode == expected["exit"], done.stderr
    assert expected["names"] in done.stderr.decode().strip().splitlines()[-1]


def test_first_divergence_refuses_a_malformed_record_and_lists_no_provenance_key():
    body = CASES["divergence_provenance_keys_are_not_series"]
    result = dv.first_divergence(body["given"]["records_a"], body["given"]["records_b"])
    listed = (result.get("left_out") or {}).get("not_compared") or ()
    assert not {"layers_not_read", "moe_source"} & set(listed)
    bad = [{**GOOD, "attempt": None}]
    with pytest.raises(identity.MalformedRecordError):
        dv.first_divergence(bad, [dict(GOOD)])
