"""Lane AL-4′ (wave AL, step 2): the Python readers take the scalar-only record.

`WAVE_AL_CONTRACT.md` section 4.6. `identity.check` takes record version 2 for a scalar-only record.
Every reader sets those records apart where records are loaded, after that check and before any
engine. Attempt resolution applies to them with the training records' cutoffs. A scalar's series is
the training records' values and the scalar-only records' points by step, and at one step the
highest `seq` wins (Q6 a). No reader makes a record, a step or a sample of one.

The inputs are the shared cases' records (`fixtures/wave_al/cases.json`, section 4), written as
`.jsonl` lines as section 4.2 states them. The capture does not write the record in this lane's tree
(lane AL-5) and `flight` has no `seq` here (lane AL-1′), so a store's `scalars-*` objects are read
only on the joined tree: that test skips where `flight`'s schema has no `seq`.

Needs nothing beyond the SDK and pytest, except the store test (pyarrow) and the Rerun export test
(NumPy), which skip test by test without them.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

from tensorquill import evidence, identity, mcp_server
from tensorquill.identity import MalformedRecordError

HERE = Path(__file__).parent
DOCUMENT = json.loads((HERE / "fixtures" / "wave_al" / "cases.json").read_text())
SCALAR = {c["name"]: c for c in DOCUMENT["cases"] if c["section"] == 4}
SRC = Path(evidence.__file__).parents[1]


def _case(name: str) -> dict:
    return copy.deepcopy(SCALAR[name])


def _records(name: str) -> list[dict]:
    return _case(name)["records"]


def _training(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("schema_version") != 2]


def _write(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def _file(tmp_path: Path, name: str) -> Path:
    records = _records(name)
    return _write(tmp_path / f"{records[0]['run_id']}.rank0.jsonl", records)


def _series_subset(got: dict, want: dict) -> dict:
    return {name: got.get(name) for name in want}


PYTHON = [n for n, c in SCALAR.items() if "python" in c["paths"]]
SERIES = [n for n, c in SCALAR.items() if "series" in c["expect"] and "python" in c["paths"]]
RUNS = [n for n, c in SCALAR.items() if "runs" in c["paths"]]
MCP = [n for n, c in SCALAR.items() if "mcp_get_series" in c["paths"]]


# ---- identity.check ---------------------------------------------------------------------------


def test_check_takes_record_version_2_for_a_scalar_only_record():
    record = next(r for r in _records("scalar_record_keys") if r["schema_version"] == 2)
    assert identity.check(record, "here") is None
    assert identity.is_scalar_record(record) is True
    assert 2 in identity.SCHEMA_VERSIONS
    for training in _training(_records("scalar_record_keys")):
        assert identity.is_scalar_record(training) is False


@pytest.mark.parametrize("change,field", [
    ({"record": None}, "schema_version"),  # a version 2 record that is not a scalar-only record
    ({"record": "steps"}, "schema_version"),
    ({"seq": None}, "seq"),
    ({"seq": True}, "seq"),
    ({"seq": -1}, "seq"),
    ({"seq": 1.0}, "seq"),
    ({"seq": "0"}, "seq"),
])
def test_check_refuses_a_version_2_record_that_is_not_a_whole_scalar_only_record(change, field):
    record = next(r for r in _records("scalar_record_keys") if r["schema_version"] == 2)
    for key, value in change.items():
        if value is None:
            del record[key]
        else:
            record[key] = value
    with pytest.raises(MalformedRecordError) as raised:
        identity.check(record, identity.in_file("x.jsonl", 5))
    assert raised.value.field == field and raised.value.where == "x.jsonl, line 5"
    assert identity.is_scalar_record(record) is (field == "seq")


def test_a_version_no_reader_takes_names_the_versions_read():
    with pytest.raises(MalformedRecordError) as raised:
        identity.check({"schema_version": 99, "step": 0}, "here")
    assert str(raised.value) == "schema_version is 99 at here: this tensorquill reads schema versions 0, 1 and 2"


def test_a_step_record_of_version_1_with_a_record_key_is_still_a_step_record():
    """Only version 2 is the scalar-only record: a key `record` on a version 1 line is read as before."""
    record = {**_training(_records("scalar_record_keys"))[0], "record": "scalars"}
    identity.check(record, "here")
    assert identity.is_scalar_record(record) is False


def test_identity_imports_the_standard_library_alone():
    tree = ast.parse((SRC / "tensorquill" / "identity.py").read_text())
    names = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    names |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert names <= {"__future__", "typing"}, names


def test_the_sender_s_fields_are_checked_as_before():
    """`flight.SENDER_FIELDS` checks `rank`, `attempt` and `step` only: a scalar-only record with
    no `seq` passes there, as any line's version does (RS10)."""
    record = next(r for r in _records("scalar_record_keys") if r["schema_version"] == 2)
    del record["seq"]
    identity.check(record, None, fields=("rank", "attempt", "step"))


# ---- Attempt resolution ------------------------------------------------------------------------


def test_a_later_attempt_rolls_a_scalar_back_with_the_training_records_cutoffs():
    records = _records("scalar_rolled_back_by_a_later_attempt")
    kept = evidence.resolve_attempts(records)
    assert [r for r in kept if identity.is_scalar_record(r)] == []
    assert [(r["attempt"], r["step"]) for r in kept] == [(1, 0), (1, 1), (1, 2), (1, 3), (2, 4), (2, 5), (2, 6), (2, 7)]
    assert len(evidence.load_records(records, all_attempts=True, scalars=True)) == len(records)


def test_a_baseline_scalar_of_a_resumed_attempt_does_not_roll_back_the_earlier_one():
    """A resumed attempt whose first line is a scalar-only record at step 0 cuts at its first
    training step (4), not at the scalar's step."""
    first = [{"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 1, "step": s, "ts": float(s)} for s in range(6)]
    baseline = {"schema_version": 2, "record": "scalars", "run_id": "r", "rank": 0, "attempt": 2, "step": 0, "seq": 0,
                "ts": 50.0, "scalars": {"eval/loss": 3.0}}
    second = [{"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 2, "step": s, "ts": float(s)} for s in range(4, 8)]
    kept = evidence.resolve_attempts([*first, baseline, *second])
    assert [(r["attempt"], r["step"]) for r in kept if not identity.is_scalar_record(r)] == [
        (1, 0), (1, 1), (1, 2), (1, 3), (2, 4), (2, 5), (2, 6), (2, 7)]
    assert baseline in kept
    assert [(r["attempt"], r["step"]) for r in evidence.load_records([*first, baseline, *second])] == [
        (1, 0), (1, 1), (1, 2), (1, 3), (2, 4), (2, 5), (2, 6), (2, 7)]


def test_an_attempt_holding_only_scalar_records_cuts_nothing():
    first = [{"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 1, "step": s, "ts": float(s)} for s in range(6)]
    alone = {"schema_version": 2, "record": "scalars", "run_id": "r", "rank": 0, "attempt": 2, "step": 2, "seq": 0,
             "ts": 50.0, "scalars": {"eval/loss": 3.0}}
    kept = evidence.resolve_attempts([*first, alone])
    assert kept == [*first, alone]
    # ... and an earlier attempt's scalar is cut by a later attempt's training records alone.
    late = [{"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 3, "step": s, "ts": float(s)} for s in range(1, 3)]
    kept = evidence.resolve_attempts([*first, alone, *late])
    assert alone not in kept and [(r["attempt"], r["step"]) for r in kept] == [(1, 0), (3, 1), (3, 2)]


# ---- build_evidence: the training records alone -------------------------------------------------


@pytest.mark.parametrize("name", PYTHON)
def test_build_evidence_judges_the_training_records_alone(name, tmp_path):
    """`kind_checks`' records, `repeated_steps` and the run state are what the training records
    alone give (case `scalar_not_a_training_step`): the whole evidence is theirs, from a list and
    from the `.jsonl` file."""
    case = _case(name)
    alone = evidence.build_evidence(copy.deepcopy(_training(case["records"])), detectors="python")
    built = evidence.build_evidence(copy.deepcopy(case["records"]), detectors="python")
    assert json.dumps(built, sort_keys=True) == json.dumps(alone, sort_keys=True)
    path = _file(tmp_path, name)
    from_file = evidence.build_evidence(path, detectors="python")
    trained = evidence.build_evidence(_write(tmp_path / "training.jsonl", _training(case["records"])), detectors="python")
    from_file.pop("run_end", None), trained.pop("run_end", None)
    assert json.dumps(from_file, sort_keys=True) == json.dumps(trained, sort_keys=True)
    assert "repeated_steps" not in [i["kind"] for i in built["incidents"]]
    if "steps_judged" in case["expect"]:
        assert built["kind_checks"]["repeated_steps"]["records"] == case["expect"]["steps_judged"]


def test_load_records_gives_the_training_records_by_default(tmp_path):
    path = _file(tmp_path, "scalar_first_and_last_line")
    records = _records("scalar_first_and_last_line")
    assert evidence.load_records(path) == _training(records)
    assert evidence.load_records(path, scalars=True) == records
    assert evidence.load_records(records) == _training(records)
    every, left_out = evidence.read_jsonl(path)
    assert every == records and left_out == 0  # the raw read: every whole line


def test_an_incident_s_scalar_context_is_the_training_record_s():
    """Q4 (a): case `scalar_baseline_and_step_0_share_a_point`'s step 0 record keeps its own
    `eval/loss` (2.8), which the scalar-only record at step 0 (3.0) does not change."""
    case = _case("scalar_baseline_and_step_0_share_a_point")
    loaded = evidence.load_records(case["records"])
    assert [r["step"] for r in loaded] == [0, 1]
    assert loaded[0]["scalars"]["eval/loss"] == case["expect"]["step_0_record_eval_loss"]
    built = evidence.build_evidence(case["records"], detectors="python")
    assert built["kind_checks"]["repeated_steps"]["records"] == case["expect"]["steps_judged"]


# ---- The series --------------------------------------------------------------------------------


@pytest.mark.parametrize("name", SERIES)
def test_a_scalar_s_series_merges_both_kinds_by_step(name, tmp_path):
    case = _case(name)
    want = case["expect"]["series"]
    assert _series_subset(evidence.scalar_series(case["records"]), want) == want
    assert _series_subset(evidence.scalar_series(_file(tmp_path, name)), want) == want


def test_a_series_of_a_rolled_back_scalar():
    case = _case("scalar_rolled_back_by_a_later_attempt")
    assert evidence.scalar_series(case["records"]) == case["expect"]["series_default"]
    assert evidence.scalar_series(case["records"], all_attempts=True) == case["expect"]["series_all_attempts"]


def test_the_highest_seq_wins_whatever_the_order_of_the_lines():
    records = _records("scalar_same_step_twice")
    reordered = [records[1], *records[2:], records[0]]  # seq 1 first, seq 0 last
    assert evidence.scalar_series(reordered)["eval/loss"] == [[4, 1.25]]


def test_a_scalar_only_point_wins_over_the_training_record_s_value_at_its_step():
    records = _records("scalar_baseline_and_step_0_share_a_point")
    assert evidence.scalar_series(records)["eval/loss"] == [[0, 3.0]]
    assert evidence.scalar_series(_training(records))["eval/loss"] == [[0, 2.8]]


def test_a_series_needs_one_rank():
    records = _records("scalar_same_step_twice")
    other = [{**r, "rank": 1} for r in records]
    with pytest.raises(ValueError, match="pass rank="):
        evidence.scalar_series([*records, *other])
    assert evidence.scalar_series([*records, *other], rank=1)["eval/loss"] == [[4, 1.25]]


# ---- `python -m tensorquill runs` -----------------------------------------------------------


def _runs(folder: Path) -> list[dict]:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    done = subprocess.run([sys.executable, "-m", "tensorquill", "runs", str(folder), "--json"], capture_output=True,
                          text=True, env=env, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.mark.parametrize("name", RUNS)
def test_runs_takes_the_run_id_from_the_first_line_and_the_last_step_from_the_last_training_record(name, tmp_path):
    case = _case(name)
    _file(tmp_path, name)
    [row] = _runs(tmp_path)
    assert row["run_id"] == case["expect"].get("run_id", case["records"][0]["run_id"])
    assert row["last_step"] == case["expect"]["last_training_step"]


def test_runs_reads_a_scalar_record_past_the_last_step_in_a_file_of_two_attempts(tmp_path):
    """The file read whole for a rank in two attempts (XC4): the scalar-only record of step 1000
    is not the last step."""
    records = _records("scalar_rolled_back_by_a_later_attempt")
    records.append({"schema_version": 2, "record": "scalars", "run_id": "al_roll", "rank": 0, "attempt": 2,
                    "step": 1000, "seq": 0, "ts": 200.0, "scalars": {"eval/loss": 0.5}})
    _write(tmp_path / "al_roll.rank0.jsonl", records)
    [row] = _runs(tmp_path)
    assert row["last_step"] == 7


def test_runs_reads_each_rank_s_last_training_step_in_a_file_of_two_ranks(tmp_path):
    records = _records("scalar_step_never_written")
    other = [{**r, "rank": 1} for r in records]
    _write(tmp_path / "shared.jsonl", [x for pair in zip(records, other) for x in pair])
    [row] = _runs(tmp_path)
    assert row["last_step"] == 5


# ---- MCP -------------------------------------------------------------------------------------


def _tool(source: mcp_server.Source, name: str, arguments: dict) -> tuple[bool, dict | str]:
    reply = mcp_server.call_tool(source, 1, {"name": name, "arguments": arguments})["result"]
    text = reply["content"][0]["text"]
    return reply["isError"], (text if reply["isError"] else json.loads(text))


@pytest.mark.parametrize("name", MCP)
def test_mcp_get_series_gives_the_merged_series(name, tmp_path):
    case = _case(name)
    source = mcp_server.Source(_file(tmp_path, name))
    want = case["expect"].get("series") or case["expect"]["series_default"]
    for scalar, points in want.items():
        error, got = _tool(source, "get_series", {"name": scalar, "kind": "scalar"})
        assert not error, got
        assert [[s, v] for s, v in zip(got["steps"], got["values"]["value"])] == points
    if name == "scalar_rolled_back_by_a_later_attempt":
        error, said = _tool(source, "get_series", {"name": "eval/loss", "kind": "scalar"})
        assert error and "eval/loss" in said


def test_mcp_names_and_steps_are_the_training_records_and_the_scalar_names(tmp_path):
    source = mcp_server.Source(_file(tmp_path, "scalar_step_never_written"))
    error, got = _tool(source, "get_series", {"name": "eval/loss"})  # found by name alone, as a scalar
    assert not error and got["kind"] == "scalar" and got["steps"] == [7]
    error, got = _tool(source, "get_series", {"name": "train/loss"})
    assert not error and got["steps"] == [0, 1, 2, 3, 4, 5]
    opened = source.open(None)
    assert opened["source"]["steps"] == {"first": 0, "last": 5, "records": 6}
    assert "steps_all_attempts" not in opened["source"]


def test_mcp_list_runs_lists_the_last_training_step(tmp_path):
    path = _file(tmp_path, "scalar_step_never_written")
    error, got = _tool(mcp_server.Source(path, runs=tmp_path), "list_runs", {})
    assert not error, got
    [row] = [r for r in got["runs"] if r["run_id"] == "al_never"]
    assert row["last_step"] == 5


# ---- compare (Q8 a) ------------------------------------------------------------------------------


def _pair_of_runs() -> tuple[list[dict], list[dict]]:
    def run(run_id: str, leaves: bool) -> list[dict]:
        out = []
        for step in range(50):  # first_divergence needs 33 common steps
            out.append({"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": 1, "step": step, "ts": float(step),
                        "grad_norm": 1.0, "scalars": {"train/loss": 2.0 - 0.01 * step}})
            eval_loss = 3.0 - 0.01 * step + (5.0 if leaves and step >= 40 else 0.0)
            out.append({"schema_version": 2, "record": "scalars", "run_id": run_id, "rank": 0, "attempt": 1, "step": step,
                        "seq": step, "ts": 100.0 + step, "scalars": {"eval/loss": eval_loss}})
        return out
    return run("a", False), run("b", True)


def test_first_divergence_reads_scalar_records_as_series_points():
    from tensorquill.divergence import first_divergence

    a, b = _pair_of_runs()
    found = first_divergence(a, b)
    assert found["answer"] == "diverged", found["reason"]
    assert (found["divergence"]["kind"], found["divergence"]["name"], found["divergence"]["step"]) == ("scalars", "eval/loss", 40)
    assert found["runs"]["a"]["records"] == found["runs"]["b"]["records"] == 50  # never a record of their own
    assert found["runs"]["b"]["last_step"] == 49
    found = first_divergence(_training(a), _training(b))
    assert found["answer"] == "no_divergence"


def test_compare_runs_reads_scalar_records_as_series_points(tmp_path):
    from tensorquill.compare import compare_runs

    a, b = _pair_of_runs()
    _write(tmp_path / "a.rank0.jsonl", a)
    _write(tmp_path / "b.rank0.jsonl", b)
    found = compare_runs("a", "b", source=tmp_path)["divergence"]
    assert found["state"] == "diverged", found
    assert (found["divergence"]["name"], found["divergence"]["step"]) == ("eval/loss", 40)
    error, said = _tool(mcp_server.Source(tmp_path / "a.rank0.jsonl", runs=tmp_path), "compare_runs", {"runs": ["a", "b"]})
    assert not error, said
    assert said["divergence"]["state"] == "diverged"


# ---- The Rerun export --------------------------------------------------------------------------


def test_the_rerun_export_logs_each_point_at_its_step(tmp_path, monkeypatch):
    """`/logged/<name>` at each point's step, the merged series (Q6), and no step logged for a
    scalar-only record but its scalars. Rerun itself is replaced by a recorder."""
    pytest.importorskip("numpy")
    logged: list[tuple[int | None, str, object]] = []
    now: list[int | None] = [None]
    rr = mock.MagicMock(name="rerun")
    rr.escape_entity_path_part.side_effect = lambda part: part
    rr.set_time.side_effect = lambda _timeline, sequence: now.__setitem__(0, sequence)
    rr.Scalars.side_effect = lambda value: ("scalars", value)
    rr.log.side_effect = lambda entity, value, **_: logged.append((now[0], entity, value))
    blueprint = mock.MagicMock(name="rerun.blueprint")
    rr.blueprint = blueprint
    monkeypatch.setitem(sys.modules, "rerun", rr)
    monkeypatch.setitem(sys.modules, "rerun.blueprint", blueprint)
    records = [*_records("scalar_baseline_and_step_0_share_a_point"),
               {"schema_version": 2, "record": "scalars", "run_id": "al_share", "rank": 0, "attempt": 1, "step": 9,
                "seq": 1, "ts": 101.0, "scalars": {"eval/loss": 1.0}}]
    built = evidence.build_evidence(records, detectors="python")
    evidence.export_rerun(built, tmp_path / "r.rrd", records=evidence.load_records(records, scalars=True))
    points = [(step, entity, value[1]) for step, entity, value in logged if entity.startswith("/logged/")]
    assert sorted(points) == [(0, "/logged/eval/loss", 3.0), (0, "/logged/train/loss", 2.9), (1, "/logged/train/loss", 2.7),
                              (9, "/logged/eval/loss", 1.0)]
    assert not [x for x in logged if x[0] == 9 and not x[1].startswith("/logged/")]


# ---- A store's scalars-* objects (the joined tree) ----------------------------------------------


def test_load_records_reads_a_store_s_scalar_objects_only_when_asked(tmp_path):
    """Section 4.5's objects, `attempt=<a>/scalars-<first>-<last>.parquet`, beside the parts. Their
    rows are lane AL-1′'s (`seq` in the schema): judged on the joined tree."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from tensorquill import flight

    if "seq" not in flight.flight_schema().names:
        pytest.skip("flight's schema has no seq in this tree (lane AL-1′'s): judged on the joined tree")
    records = _records("scalar_first_and_last_line")
    training, scalars = _training(records), [r for r in records if r["schema_version"] == 2]
    folder = tmp_path / "runs" / "al_edges" / "rank=0" / "attempt=1"
    folder.mkdir(parents=True)
    schema = flight.flight_schema()
    pq.write_table(pa.Table.from_pylist([flight.record_to_row(r) for r in training], schema=schema),
                   folder / "part-00000000-00000005.parquet")
    pq.write_table(pa.Table.from_pylist([flight.record_to_row(r) for r in scalars], schema=schema),
                   folder / "scalars-00000000-00000001.parquet")
    run = tmp_path / "runs" / "al_edges"
    assert evidence.load_records(run) == training
    both = evidence.load_records(run, scalars=True)
    assert [r for r in both if not identity.is_scalar_record(r)] == training
    assert [r for r in both if identity.is_scalar_record(r)] == scalars
    assert evidence.scalar_series(run)["eval/final"] == [[5, 1.0]]


# ---- The Rust path, the watcher's tail and a file still being written --------------------------


def _binary():
    import importlib.util

    if importlib.util.find_spec("pyarrow") is None:
        return None
    from tensorquill.server_detect import find_binary

    return find_binary()


@pytest.mark.parametrize("name", PYTHON)
def test_build_evidence_with_the_rust_detectors_judges_the_training_records_alone(name):
    """The `rust_sdk` path: `build_evidence(detectors="rust")` never sends a scalar-only record to the
    binary, so the evidence is the training records' under the Rust detectors too."""
    binary = _binary()
    if binary is None:
        pytest.skip("tensorquill-server is not built (TENSORQUILL_SERVER_BIN), or pyarrow is not installed")
    case = _case(name)
    built = evidence.build_evidence(copy.deepcopy(case["records"]), detectors="rust", server_binary=binary)
    alone = evidence.build_evidence(copy.deepcopy(_training(case["records"])), detectors="rust", server_binary=binary)
    assert built["detectors"] == "rust"
    assert json.dumps(built, sort_keys=True) == json.dumps(alone, sort_keys=True)


def test_the_watcher_s_tail_sets_scalar_records_apart(tmp_path):
    records = _records("scalar_first_and_last_line")
    path = _write(tmp_path / "w.jsonl", records[:3])
    tail = evidence._Tail(path, None)
    assert tail.poll() is True
    assert tail.records == _training(records[:3]) and tail.scalars == records[:1]
    with path.open("a") as file:
        file.write(json.dumps(records[-1]) + "\n")  # a scalar-only line alone
    assert tail.poll() is False  # no news: the evidence of the training records is the same
    assert tail.scalars == [records[0], records[-1]] and tail.records == _training(records[:3])


def _torn(tmp_path: Path, name: str, records: list[dict]) -> Path:
    path = _write(tmp_path / name, records)
    with path.open("a") as file:
        file.write('{"schema_version": 1, "run_id": "')  # a last line still being written
    return path


def test_a_file_still_being_written_gives_the_merged_series_and_compares(tmp_path):
    """The readers' fallbacks for a last line with no end: MCP's `_complete_lines` and `compare`'s
    `_Tail` read the whole lines, the scalar-only records among them."""
    source = mcp_server.Source(_torn(tmp_path, "edges.jsonl", _records("scalar_same_step_twice")))
    error, got = _tool(source, "get_series", {"name": "eval/loss", "kind": "scalar"})
    assert not error, got
    assert (got["steps"], got["values"]["value"]) == ([4], [1.25])
    from tensorquill.compare import compare_runs

    a, b = _pair_of_runs()
    _write(tmp_path / "a.rank0.jsonl", a)
    _torn(tmp_path, "b.rank0.jsonl", b)
    found = compare_runs("a", "b", source=tmp_path)["divergence"]
    assert found["state"] == "diverged", found
    assert (found["divergence"]["name"], found["divergence"]["step"]) == ("eval/loss", 40)
