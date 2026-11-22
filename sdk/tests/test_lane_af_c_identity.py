"""Lane C of wave AF: the readers refuse a record whose identity field is not valid
(WAVE_AF_CONTRACT.md, section 1, the rows owned by lane C).

The check is lane F's `identity.check`; the readers call it where records are loaded, before
either engine is chosen, so the two engines cannot disagree. A refusal is one named error,
`MalformedRecordError`, that says the field, the value and where. No evidence file is written or
replaced. The command line exits with status 2 and one line, `tensorquill: error: ...`. The MCP
tools that read records answer a `ToolError` with the error's text, and the server answers the
next call. `runs.discover_runs` and `list_runs` do not raise.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tensorquill import evidence as ev
from tensorquill import mcp_server
from tensorquill.divergence import first_divergence
from tensorquill.identity import MalformedRecordError

DOCUMENT = json.loads((Path(__file__).parent / "fixtures" / "wave_af" / "cases.json").read_text())
SECTION_1 = [(name, body) for name, body in DOCUMENT["cases"].items() if body["section"] == 1]
SRC = Path(__file__).parents[1] / "src"


def _rust_or_skip() -> None:
    from tensorquill import server_detect

    if server_detect.find_binary() is None:
        pytest.skip("no tensorquill-server binary")
    pytest.importorskip("pyarrow")


def good(step: int, run: str = "r", **extra) -> dict:
    return {"schema_version": 1, "run_id": run, "rank": 0, "attempt": 0, "step": step, "grad_norm": 1.0, **extra}


def lines(records: list[dict]) -> str:
    return "".join(json.dumps(r) + "\n" for r in records)


def one_case(tmp_path: Path, body: dict) -> Path:
    path = tmp_path / "one.jsonl"
    path.write_text(json.dumps(body["given"]["record"]) + "\n")
    return path


def cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run([sys.executable, "-m", "tensorquill", *args], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=120)


# ---- The section 1 cases at each of lane C's four boundaries --------------------------------


@pytest.mark.parametrize("engine", ["python", "rust"])
@pytest.mark.parametrize("name,body", SECTION_1, ids=[name for name, _ in SECTION_1])
def test_each_case_with_build_evidence(name, body, engine, tmp_path):
    if engine == "rust":
        _rust_or_skip()
    expected = body["expected"][engine]
    path = one_case(tmp_path, body)
    if expected.get("reads"):
        assert isinstance(ev.build_evidence(path, detectors=engine), dict)
        return
    with pytest.raises(MalformedRecordError) as raised:
        ev.build_evidence(path, detectors=engine)
    assert raised.value.field == expected["field"]
    assert raised.value.where == f"{path}, line 1"
    assert expected["field"] in str(raised.value) and f"{path}, line 1" in str(raised.value)


@pytest.mark.parametrize("engine", ["python", "rust"])
@pytest.mark.parametrize("name,body", SECTION_1, ids=[name for name, _ in SECTION_1])
def test_each_case_on_the_command_line(name, body, engine, tmp_path):
    if engine == "rust":
        _rust_or_skip()
    expected = body["expected"][f"cli_{engine}"]
    path = one_case(tmp_path, body)
    out = tmp_path / "out.json"
    done = cli(str(path), "--detectors", engine, "-o", str(out), cwd=tmp_path)
    if expected.get("reads"):
        assert done.returncode == 0, done.stderr
        assert out.exists()
        return
    assert done.returncode == expected["exit"], (done.stdout, done.stderr)
    assert "Traceback" not in done.stderr and "Traceback" not in done.stdout
    said = done.stderr.strip().splitlines()
    assert len(said) == 1 and said[0].startswith("tensorquill: error: ") and expected["names"] in said[0], said
    assert not out.exists(), "an evidence file was written for a refused input"


# ---- Where the refusal says the record is, and that the rest is not judged -------------------


def test_the_line_named_is_the_one_in_the_file_counting_blank_and_torn_lines(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(lines([good(0), good(1)]) + "\n" + "{not json\n" + json.dumps(good(2, rank=True)) + "\n")
    with pytest.raises(MalformedRecordError) as raised:
        ev.build_evidence(path)
    assert raised.value.field == "rank" and raised.value.where == f"{path}, line 5"


def test_records_in_memory_are_named_by_their_place_in_the_list():
    with pytest.raises(MalformedRecordError) as raised:
        ev.build_evidence([good(0), good(1), {**good(2), "attempt": -1}])
    assert raised.value.field == "attempt" and raised.value.where == "record 2 of the records passed"
    with pytest.raises(MalformedRecordError) as raised:
        ev.load_records([good(0), {**good(1), "step": "1"}])
    assert raised.value.where == "record 1 of the records passed"


def test_an_invalid_record_of_another_rank_is_refused_before_a_rank_is_picked(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(lines([good(0), good(1), {**good(0), "rank": 1, "step": 1.0}]))
    with pytest.raises(MalformedRecordError) as raised:
        ev.build_evidence(path, rank=0)
    assert raised.value.field == "step" and raised.value.where == f"{path}, line 3"


def test_a_store_part_is_named_with_its_row(tmp_path, monkeypatch):
    folder = tmp_path / "runs" / "r" / "rank=0" / "attempt=0"
    folder.mkdir(parents=True)
    part = folder / "part-000001.parquet"
    part.write_bytes(b"")
    monkeypatch.setattr(ev, "_read_part", lambda path: [good(0), {**good(1), "schema_version": 7}])
    with pytest.raises(MalformedRecordError) as raised:
        ev.load_records(tmp_path / "runs" / "r", rank=0)
    assert raised.value.field == "schema_version" and raised.value.where == f"{part}, row 1"


# ---- The watcher: nothing written or replaced ----------------------------------------------


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_the_watcher_refuses_and_writes_nothing(tmp_path, engine):
    if engine == "rust":
        _rust_or_skip()
    path = tmp_path / "r.jsonl"
    path.write_text(lines([good(0), {**good(1), "attempt": None}]))
    out = tmp_path / "out.json"
    with pytest.raises(MalformedRecordError) as raised:
        ev.watch(path, out, interval=0.0, polls=1, detectors=engine)
    assert raised.value.where == f"{path}, line 2"
    assert not out.exists()


def test_the_watcher_leaves_the_last_evidence_as_it_was_when_a_bad_line_arrives(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(lines([good(step) for step in range(5)]))
    out = tmp_path / "out.json"
    written: list[bytes] = []

    def update(evidence, where):
        written.append(where.read_bytes())
        with path.open("a") as file:
            file.write(lines([good(5), {**good(6), "rank": -3}]))

    with pytest.raises(MalformedRecordError) as raised:
        ev.watch(path, out, interval=0.0, polls=3, on_update=update)
    assert raised.value.field == "rank" and raised.value.where == f"{path}, line 7"
    assert len(written) == 1 and out.read_bytes() == written[0]


def test_the_watch_command_exits_2_with_one_line(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(lines([good(0), {**good(1), "step": True}]))
    done = cli(str(path), "--watch", "--detectors", "python", "--interval", "0", "-o", str(tmp_path / "o.json"),
               cwd=tmp_path)
    assert done.returncode == 2 and "Traceback" not in done.stderr
    assert done.stderr.strip().splitlines() == [f"tensorquill: error: step is true at {path}, line 2: "
                                                "tensorquill reads an integer of 0 or more"]


# ---- first_divergence and compare ----------------------------------------------------------


def test_first_divergence_refuses_and_names_the_run():
    a = [good(step) for step in range(30)]
    b = [good(step, run="b") for step in range(30)]
    b[4] = {**b[4], "rank": 0.0}
    with pytest.raises(MalformedRecordError) as raised:
        first_divergence(a, b)
    assert raised.value.field == "rank" and raised.value.where == "record 4 of the records passed, run B"


def _two_runs(tmp_path: Path) -> Path:
    (tmp_path / "a.jsonl").write_text(lines([good(step, run="a") for step in range(30)]))
    bad = [good(step, run="b") for step in range(30)]
    bad[10] = {**bad[10], "schema_version": 99}
    (tmp_path / "b.jsonl").write_text(lines(bad))
    return tmp_path / "b.jsonl"


def test_compare_refuses_the_run_whose_file_holds_the_record(tmp_path):
    from tensorquill.compare import compare_runs

    bad = _two_runs(tmp_path)
    with pytest.raises(MalformedRecordError) as raised:
        compare_runs("a", "b", source=tmp_path)
    assert raised.value.field == "schema_version" and raised.value.where == f"{bad}, line 11"


def test_the_compare_command_exits_2_with_one_line(tmp_path):
    bad = _two_runs(tmp_path)
    done = cli("compare", "a", "b", "--source", str(tmp_path), "-o", str(tmp_path / "c.json"), cwd=tmp_path)
    assert done.returncode == 2 and "Traceback" not in done.stderr
    said = done.stderr.strip().splitlines()
    assert said == [f"tensorquill: error: schema_version is 99 at {bad}, line 11: "
                    "this tensorquill reads schema versions 0 and 1"]
    assert not (tmp_path / "c.json").exists()


# ---- The MCP tools that read records ----------------------------------------------------------


def _reply(source: mcp_server.Source, name: str, arguments: dict | None = None) -> dict:
    return mcp_server.call_tool(source, 1, {"name": name, "arguments": arguments or {}})["result"]


@pytest.mark.parametrize("tool,arguments", [
    ("list_incidents", {}), ("get_incident", {"id": "grad_spike@3"}),
    ("get_series", {"name": "grad_norm", "kind": "scalar"}),
])
def test_an_mcp_tool_on_the_capture_answers_the_error_and_the_next_call(tmp_path, tool, arguments):
    path = tmp_path / "r.jsonl"
    path.write_text(lines([good(0), good(1), {**good(2), "attempt": "0"}]))
    source = mcp_server.Source(path)
    reply = _reply(source, tool, arguments)
    assert reply["isError"] is True
    assert reply["content"][0]["text"] == f'attempt is "0" at {path}, line 3: tensorquill reads an integer of 0 or more'
    # The server answers the next call: once the file is put right, from the same Source.
    path.write_text(lines([good(0), good(1), good(2)]))
    assert _reply(source, "list_incidents")["isError"] is False


def test_mcp_compare_runs_answers_the_error(tmp_path):
    bad = _two_runs(tmp_path)
    reply = _reply(mcp_server.Source(tmp_path / "a.jsonl"), "compare_runs", {"runs": ["a", "b"]})
    assert reply["isError"] is True
    assert reply["content"][0]["text"].startswith(f"schema_version is 99 at {bad}, line 11")


# ---- discover_runs and list_runs: as on the base ------------------------------------------------


def test_listing_the_runs_does_not_raise_on_such_a_file(tmp_path):
    from tensorquill import runs

    _two_runs(tmp_path)
    found = runs.discover_runs(tmp_path)
    assert sorted(run["run_id"] for run in found) == ["a", "b"]
    rows = runs.list_runs(tmp_path)
    assert sorted(row["run_id"] for row in rows) == ["a", "b"]
    reply = _reply(mcp_server.Source(tmp_path / "a.jsonl"), "list_runs")
    assert reply["isError"] is False
