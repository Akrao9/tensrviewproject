"""Wave AS, lane AS-6: a scalar-only record's `seq` past Int64 is refused by every reader of a records file.

`identity.check` bounds `seq` at Int64, the `seq` column's type, as it bounds `step` and `attempt`
(`WAVE_AF_CONTRACT.md` section 12, AQ5-D6). Before, it bounded `step`, `attempt` and `rank` only, so a
local file whose scalar-only record (`schema_version` 2, `"record": "scalars"`) held a `seq` of 2**63 was
read by `load_records`, `build_evidence`, the command line, the watcher, `first_divergence`, `compare`
and the MCP tools, while the sender (`flight._seq`) left the same line out. The rule is the owner's
(Nov 27, 2026): one line past an identity bound refuses the whole local file, as a negative step does;
live, the sender skips that line and sends the rest.

The pins at the end held before the change too: the sender names and counts such a line as it did, a
`seq` of 2**63 - 1 is read and sent, and a version 1 record marked `"record": "scalars"` is a step
record whose `seq` is not looked at.

Tests that need `pyarrow` or the server binary skip one by one without them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import tensorquill
from tensorquill import evidence
from tensorquill.identity import MalformedRecordError, check
from tensorquill.server_detect import find_binary

BINARY = find_binary()
INT64_MAX = 2**63 - 1
PAST = 2**63
REASON = "a scalar-only record needs a seq, an integer of 0 or more that Int64 holds"


def needs_rust() -> None:
    pytest.importorskip("pyarrow")
    if BINARY is None:
        pytest.skip("tensorquill-server is not built")


def _step(step: int, **fields: Any) -> dict[str, Any]:
    return {"schema_version": 1, "run_id": "as6", "rank": 0, "attempt": 0, "step": step, "grad_norm": 1.0, **fields}


def _scalar(step: int, seq: Any, **fields: Any) -> dict[str, Any]:
    return {"schema_version": 2, "record": "scalars", "run_id": "as6", "rank": 0, "attempt": 0, "step": step,
            "seq": seq, "scalars": {"eval/loss": 1.5}, **fields}


def _records(seq: Any, run_id: str = "as6", steps: int = 6) -> list[dict[str, Any]]:
    """Steps 0 to steps - 1, with a scalar-only record logged at step 2 after step 2's record (line 4)."""
    records = [_step(s, run_id=run_id) for s in range(steps)]
    records.insert(3, _scalar(2, seq, run_id=run_id))
    return records


def _write(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


# --- identity.check ------------------------------------------------------------------------------

def test_check_refuses_a_seq_past_int64_and_names_it():
    with pytest.raises(MalformedRecordError) as raised:
        check(_scalar(2, PAST), "here")
    error = raised.value
    assert (error.field, error.value, error.where) == ("seq", PAST, "here")
    assert str(error) == f"seq is {PAST} at here: {REASON}"


def test_check_takes_the_largest_seq_int64_holds():
    assert check(_scalar(2, INT64_MAX)) is None


def test_the_sender_fields_check_bounds_seq_on_a_version_2_line():
    """`rows_to_send` checks a version 2 line with `schema_version` too (`flight._sender_fields`), so the
    bound holds there by the same rule."""
    with pytest.raises(MalformedRecordError) as raised:
        check(_scalar(2, PAST), None, ("schema_version", "rank", "attempt", "step"))
    assert raised.value.field == "seq"


def test_the_other_seq_refusals_and_the_order_are_as_before():
    with pytest.raises(MalformedRecordError) as raised:
        check(_scalar(2, -1))
    assert str(raised.value) == "seq is -1: a scalar-only record needs a seq, an integer of 0 or more"
    with pytest.raises(MalformedRecordError) as raised:
        check(_scalar(PAST, PAST))
    assert raised.value.field == "step"  # `seq` is checked after `step`


# --- the readers of a local file refuse the whole file --------------------------------------------

def test_load_records_refuses_the_file_by_its_line(tmp_path):
    path = _write(tmp_path / "as6.rank0.jsonl", _records(PAST))
    with pytest.raises(MalformedRecordError) as raised:
        tensorquill.load_records(path)
    assert (raised.value.field, raised.value.where) == ("seq", f"{path}, line 4")


def test_scalar_series_refuses_the_file_by_its_line(tmp_path):
    path = _write(tmp_path / "as6.rank0.jsonl", _records(PAST))
    with pytest.raises(MalformedRecordError) as raised:
        evidence.scalar_series(path)
    assert (raised.value.field, raised.value.where) == ("seq", f"{path}, line 4")


def test_load_records_refuses_records_in_memory_by_their_place():
    with pytest.raises(MalformedRecordError) as raised:
        tensorquill.load_records(_records(PAST))
    assert (raised.value.field, raised.value.where) == ("seq", "record 3 of the records passed")


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_build_evidence_refuses_the_file_with_either_engine(engine, tmp_path):
    if engine == "rust":
        needs_rust()
    path = _write(tmp_path / "as6.rank0.jsonl", _records(PAST))
    with pytest.raises(MalformedRecordError) as raised:
        tensorquill.build_evidence(path, detectors=engine, **({"server_binary": BINARY} if engine == "rust" else {}))
    assert (raised.value.field, raised.value.where) == ("seq", f"{path}, line 4")


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_both_engines_read_the_largest_seq_int64_holds(engine, tmp_path):
    if engine == "rust":
        needs_rust()
    path = _write(tmp_path / "as6.rank0.jsonl", _records(INT64_MAX))
    found = tensorquill.build_evidence(path, detectors=engine, **({"server_binary": BINARY} if engine == "rust" else {}))
    assert found["kind_checks"]["grad_spike"]["value"] == "checked"
    assert evidence.scalar_series(path)["eval/loss"] == [[2, 1.5]]


def test_the_rust_route_refuses_by_place():
    pytest.importorskip("pyarrow")
    from tensorquill.server_detect import records_stream

    with pytest.raises(MalformedRecordError) as raised:
        records_stream(_records(PAST))
    assert (raised.value.field, raised.value.where) == ("seq", "record 3 of the records passed")


def test_the_command_line_exits_2_with_one_line(tmp_path):
    import os
    import subprocess
    import sys

    path = _write(tmp_path / "as6.rank0.jsonl", _records(PAST))
    src = Path(__file__).parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(src), "PYTHONWARNINGS": "ignore:Failed to initialize NumPy"}
    done = subprocess.run([sys.executable, "-m", "tensorquill", str(path), "--detectors", "python",
                           "-o", str(tmp_path / "out.json")], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 2 and "Traceback" not in done.stderr
    said = done.stderr.strip().splitlines()
    assert said == [f"tensorquill: error: seq is {PAST} at {path}, line 4: {REASON}"], said
    assert not (tmp_path / "out.json").exists()


def test_the_watcher_refuses_and_writes_nothing(tmp_path):
    path = _write(tmp_path / "as6.rank0.jsonl", _records(PAST))
    out = tmp_path / "out.json"
    with pytest.raises(MalformedRecordError) as raised:
        evidence.watch(path, out, interval=0.0, polls=1)
    assert (raised.value.field, raised.value.where) == ("seq", f"{path}, line 4")
    assert not out.exists()


def test_first_divergence_and_compare_refuse(tmp_path):
    from tensorquill.compare import compare_runs
    from tensorquill.divergence import first_divergence

    a = [_step(s, run_id="a") for s in range(30)]
    b = _records(PAST, run_id="b", steps=30)
    with pytest.raises(MalformedRecordError) as raised:
        first_divergence(a, b)
    assert (raised.value.field, raised.value.where) == ("seq", "record 3 of the records passed, run B")
    _write(tmp_path / "a.jsonl", a)
    _write(tmp_path / "b.jsonl", b)
    with pytest.raises(MalformedRecordError) as raised:
        compare_runs("a", "b", source=tmp_path)
    assert (raised.value.field, raised.value.where) == ("seq", f"{tmp_path / 'b.jsonl'}, line 4")


def test_an_mcp_tool_answers_the_error_and_the_next_call(tmp_path):
    from tensorquill import mcp_server

    path = _write(tmp_path / "as6.rank0.jsonl", _records(PAST))
    source = mcp_server.Source(path)

    def reply(name: str) -> dict[str, Any]:
        return mcp_server.call_tool(source, 1, {"name": name, "arguments": {}})["result"]

    answer = reply("list_incidents")
    assert answer["isError"] is True
    assert answer["content"][0]["text"].startswith(f"seq is {PAST} at {path}, line 4")
    _write(path, _records(5))
    assert reply("list_incidents")["isError"] is False


def test_the_run_listing_does_not_raise(tmp_path):
    """`runs.discover_runs` does not call the check, as for a step past Int64 (wave AR): it lists the run."""
    from tensorquill.runs import discover_runs

    _write(tmp_path / "as6.rank0.jsonl", _records(PAST))
    assert [run["run_id"] for run in discover_runs(tmp_path)] == ["as6"]


def test_record_to_row_refuses_by_the_check_before_moved_is_touched():
    """`record_to_row` runs `identity.check` first, so the refusal is the check's, the one every reader gives."""
    pytest.importorskip("pyarrow")
    from tensorquill import flight

    moved = ["kept"]
    with pytest.raises(MalformedRecordError) as raised:
        flight.record_to_row(_scalar(2, PAST), moved=moved, where="w")
    assert str(raised.value) == f"seq is {PAST} at w: {REASON}" and moved == ["kept"]


# --- pins: what held before the change and still holds --------------------------------------------

def test_a_version_1_record_marked_scalars_is_a_step_record_whose_seq_is_not_looked_at(tmp_path):
    record = {**_step(2), "record": "scalars", "seq": 2**64}
    assert check(record) is None
    path = _write(tmp_path / "as6.rank0.jsonl", [_step(0), _step(1), record])
    assert len(tensorquill.load_records(path)) == 3


def test_rows_to_send_leaves_the_line_out_and_names_it_as_before():
    """Before the change the refusal came from `_row` (`flight._seq`) and was named by `_unheld_key`; now
    it comes from `identity.check`. The count and the name are the same."""
    pytest.importorskip("pyarrow")
    from tensorquill.flight import flight_schema, rows_to_send

    records = [_step(0), _scalar(0, PAST), _step(1), _scalar(1, INT64_MAX)]
    refused: list[Any] = []
    batch, steps, skipped = rows_to_send(records, flight_schema(), refused=refused)
    assert (steps, skipped, refused) == ([0, 1], 1, [(1, "seq", PAST)])
    assert batch.column("seq").to_pylist() == [None, None, INT64_MAX]


def test_the_sender_skips_the_line_says_it_once_and_stores_the_rest(tmp_path, capfd):
    needs_rust()
    import pyarrow.parquet as pq
    from tensorquill.flight import FlightSender
    from test_flight_e2e import Server, stored_records

    run_id = "as6-seq"
    path = _write(tmp_path / "rank0.jsonl",
                  [_step(0, run_id=run_id), _scalar(0, PAST, run_id=run_id), _step(1, run_id=run_id),
                   _scalar(1, INT64_MAX, run_id=run_id), _step(2, run_id=run_id)])
    server = Server(tmp_path / "store")
    try:
        sender = FlightSender(path, server.endpoint, run_id, 0, batch_steps=50, batch_seconds=0.05,
                              poll_seconds=0.02, backoff_cap=0.2)
        closed = sender.close(20.0)
    finally:
        assert server.terminate() == 0
    assert closed is False and (sender.skipped_lines, sender.left_out_lines) == (1, 1)
    assert sender.acked_offset == path.stat().st_size
    assert [r["step"] for r in stored_records(tmp_path / "store", run_id)] == [0, 1, 2]
    scalars = sorted((tmp_path / "store" / "runs" / run_id / "rank=0").glob("attempt=*/scalars-*.parquet"))
    assert [s for part in scalars for s in pq.read_table(part, partitioning=None).column("seq").to_pylist()] \
        == [INT64_MAX]
    said = [s for s in capfd.readouterr().err.splitlines() if s.startswith("tensorquill:") and "is not sent" in s]
    assert len(said) == 1 and f"its seq is {PAST}, and its column in the stream cannot hold it" in said[0], said
