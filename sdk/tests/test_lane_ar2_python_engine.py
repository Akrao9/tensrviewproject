"""Wave AR, lane AR-2: three places where the Python engine departed from the contract (lane AQ-5's
generated runs, `test_generated_engines.py`).

- AQ5-D1. A refused `update_rms` map is an absent key (the author: "Python brought to Rust"):
  `input_counts` counted a map with no entry its column holds toward `update_spike`'s records.
- AQ5-D3. A top-level `grad_norm` that is not a number is absent in Python too: the stalled-weights
  rule let one break its streak, so a malformed `grad_norm` hid a stall from the default engine.
- AQ5-D6. `identity.check` is bounded at int64 (`step`, `attempt`) and int32 (`rank`), the columns'
  types, and never raises into training.

Each comparison with Rust needs `pyarrow` and the server binary (`TENSORQUILL_SERVER_BIN`), and is
skipped without them, one by one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import tensorquill
from tensorquill import evidence
from tensorquill.server_detect import find_binary

try:  # each test is collected and skipped without pyarrow
    import pyarrow  # noqa: F401
except ImportError:
    pyarrow = None

BINARY = find_binary()


def needs_rust() -> None:
    if pyarrow is None:
        pytest.skip("pyarrow is not installed")
    if BINARY is None:
        pytest.skip("tensorquill-server is not built")


def _record(step: int, **fields: Any) -> dict[str, Any]:
    return {"schema_version": 1, "run_id": "ar2", "rank": 0, "attempt": 1, "step": step, **fields}


def _write(folder: Path, records: list[dict[str, Any]]) -> Path:
    path = folder / "ar2.rank0.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _both(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    needs_rust()
    return (tensorquill.build_evidence(path),
            tensorquill.build_evidence(path, detectors="rust", server_binary=BINARY))


def _check(found: dict[str, Any], kind: str) -> tuple[Any, Any]:
    entry = found["kind_checks"][kind]
    return entry.get("value"), entry.get("records")


# --- AQ5-D1: a refused update_rms map is an absent key ------------------------------------------

# Each `update_rms` value, and whether its column holds the map (an entry it carries: a number, the
# sink's "nan", or a null, which Rust's map column keeps as an entry).
UPDATE_MAPS = {
    "a text that is not the sink's spelling": ({"p0.weight": "nan "}, False),
    "the text NaN": ({"p0.weight": "NaN"}, False),
    "an integer too large for a float": ({"p0.weight": 10**400}, False),
    "a bool": ({"p0.weight": True}, False),
    "two refused entries": ({"a": "x", "b": [1.0]}, False),
    "a held entry beside a refused one": ({"a": 0.01, "b": "x"}, True),
    "the sink's nan": ({"p0.weight": "nan"}, True),
    "a null": ({"p0.weight": None}, True),
    "a number": ({"p0.weight": 0.01}, True),
    "an empty map": ({}, False),
    "not a map": ("x", False),
}


@pytest.mark.parametrize("name", sorted(UPDATE_MAPS))
def test_d1_an_update_rms_map_counts_only_when_its_column_holds_an_entry(name):
    sizes, held = UPDATE_MAPS[name]
    counts = evidence.input_counts([_record(47, grad_norm=1.05, update_rms=sizes)])
    assert counts["update"] == int(held)


@pytest.mark.parametrize("name", sorted(UPDATE_MAPS))
def test_d1_both_engines_count_update_spike_records_alike(name, tmp_path):
    sizes, held = UPDATE_MAPS[name]
    python, rust = _both(_write(tmp_path, [_record(47, grad_norm=1.048992069, update_rms=sizes)]))
    assert _check(python, "update_spike") == _check(rust, "update_spike")
    assert _check(python, "update_spike") == (("checked", 1) if held else ("not_configured", 0))


def test_d1_a_refused_map_is_absent_to_the_update_readers_too(tmp_path):
    """`update_spike` itself and `update_trends` already skip each refused entry (`_finite`), so a
    record whose map is refused gives no update point in either engine; a run of them beside held ones
    judges as the held ones alone."""
    held = [_record(step, grad_norm=1.0 + 0.01 * step, update_rms={"w": 1e-3 * (1 + 0.01 * step)})
            for step in range(12)]
    refused = [_record(step, grad_norm=1.0, update_rms={"w": "nan "}) for step in range(12, 16)]
    path = _write(tmp_path, held + refused)
    python, rust = _both(path)
    assert python.get("update_trends") == rust.get("update_trends")
    (tmp_path / "alone").mkdir()
    alone, _ = _both(_write(tmp_path / "alone", held))
    assert python.get("update_trends") == alone.get("update_trends")
    assert _check(python, "update_spike")[1] == _check(alone, "update_spike")[1] == 12


# --- AQ5-D3: a top-level grad_norm that is not a number is absent -------------------------------

_ABSENT = object()


def _stall(grad_norm: Any) -> list[dict[str, Any]]:
    middle = {} if grad_norm is _ABSENT else {"grad_norm": grad_norm}
    return [_record(35, grad_norms={"p0.weight": 0.49}, weights_unchanged=["p0.weight"]),
            _record(36, **middle, grad_norms={"p0.weight": 0.48}, weights_unchanged=["p0.weight"]),
            _record(37, grad_norms={"p0.weight": 0.51}, weights_unchanged=["p0.weight"])]


# The middle record's `grad_norm`, and the incidents both engines report. A value the float64 column
# refuses is absent, as when the key is left out; a non-finite value the column carries (the sink's
# "nan", "inf") is `non_finite`'s, and the stall's evidence is not taken from that step.
STALL_GRAD_NORMS = {
    "absent": (_ABSENT, ["stalled_weights"]),
    "null": (None, ["stalled_weights"]),
    "a number": (1.0, ["stalled_weights"]),
    "the sink's nan": ("nan", ["non_finite"]),
    "the sink's inf": ("inf", ["non_finite"]),
    "the text NAN": ("NAN", ["stalled_weights"]),
    "a text": ("x", ["stalled_weights"]),
    "a bool": (True, ["stalled_weights"]),
    "a list": ([1.0], ["stalled_weights"]),
    "an object": ({"a": 1}, ["stalled_weights"]),
    "an integer too large for a float": (10**400, ["stalled_weights"]),
}


@pytest.mark.parametrize("name", sorted(STALL_GRAD_NORMS))
def test_d3_a_refused_grad_norm_does_not_break_a_stall_in_python(name):
    grad_norm, kinds = STALL_GRAD_NORMS[name]
    found = evidence.detect_stalled_weights(_stall(grad_norm))
    assert (found is not None) == ("stalled_weights" in kinds)


@pytest.mark.parametrize("name", sorted(STALL_GRAD_NORMS))
def test_d3_both_engines_report_the_stall_alike(name, tmp_path):
    grad_norm, kinds = STALL_GRAD_NORMS[name]
    python, rust = _both(_write(tmp_path, _stall(grad_norm)))
    assert sorted(i["kind"] for i in python["incidents"]) == sorted(i["kind"] for i in rust["incidents"]) == kinds


def _gradient_run(refused: Any) -> list[dict[str, Any]]:
    """Forty steps of a gradient norm and update sizes with a spike at step 30, and the given value in
    place of `grad_norm` at steps 10, 20 and 31 (`_ABSENT` leaves the key out)."""
    records = []
    for step in range(40):
        norm: Any = 50.0 if step == 30 else 1.0 + 0.01 * (step % 5)
        fields: dict[str, Any] = {"grad_norms": {"w": norm}, "update_rms": {"w": 1e-3 * (1 + 0.02 * (step % 3))}}
        if step in (10, 20, 31):
            norm = refused
        if norm is not _ABSENT:
            fields["grad_norm"] = norm
        records.append(_record(step, **fields))
    return records


@pytest.mark.parametrize("refused", ["NAN", True, [1.0], 10**400], ids=["text", "bool", "list", "big"])
def test_d3_every_reader_of_grad_norm_reads_a_refused_value_as_absent(refused, tmp_path):
    """`grad_spike`, `update_spike`, `gradient_trend`, `non_finite` and the input counts read a refused
    `grad_norm` as they read a left-out one, in both engines."""
    (tmp_path / "refused").mkdir()
    (tmp_path / "absent").mkdir()
    python, rust = _both(_write(tmp_path / "refused", _gradient_run(refused)))
    left_out, _ = _both(_write(tmp_path / "absent", _gradient_run(_ABSENT)))
    for document in (python, rust):
        assert ([(i["kind"], i.get("step")) for i in document["incidents"]]
                == [(i["kind"], i.get("step")) for i in left_out["incidents"]])
        assert document["kind_checks"] == left_out["kind_checks"]
        assert document.get("gradient_trend") == left_out.get("gradient_trend")


# --- AQ5-D6: identity.check bounded at the columns' types, never raising into training -----------

from tensorquill.identity import MalformedRecordError, check  # noqa: E402

INT64_MAX, INT32_MAX = 2**63 - 1, 2**31 - 1
# Each identity field past its column's bound (Int64 for `step` and `attempt`, Int32 for `rank`), and
# its largest valid value.
PAST = {"step": 2**63, "attempt": 2**63, "rank": 2**31}
EDGE = {"step": INT64_MAX, "attempt": INT64_MAX, "rank": INT32_MAX}
FIELDS = sorted(PAST)


def _line(step: int, **fields: Any) -> dict[str, Any]:
    return {"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 0, "step": step, "grad_norm": 1.0, **fields}


def _one(tmp_path: Path, field: str, value: int) -> Path:
    path = tmp_path / "one.jsonl"
    path.write_text(json.dumps({**_line(3), field: value}) + "\n")
    return path


@pytest.mark.parametrize("field", FIELDS)
def test_d6_check_refuses_a_value_past_the_column_and_names_the_field(field):
    with pytest.raises(MalformedRecordError) as raised:
        check({**_line(3), field: PAST[field]}, "here")
    error = raised.value
    assert (error.field, error.value, error.where) == (field, PAST[field], "here")
    holds = "Int32" if field == "rank" else "Int64"
    assert str(error) == f"{field} is {PAST[field]} at here: tensorquill reads an integer of 0 or more that {holds} holds"


@pytest.mark.parametrize("field", FIELDS)
def test_d6_check_takes_the_largest_value_the_column_holds(field):
    assert check({**_line(3), field: EDGE[field]}) is None


def test_d6_the_order_and_the_other_refusals_are_as_before():
    with pytest.raises(MalformedRecordError) as raised:
        check({"step": 2**63, "attempt": 2**63, "rank": 2**31})
    assert raised.value.field == "rank"
    with pytest.raises(MalformedRecordError) as raised:
        check({"step": 2**63, "attempt": -1})
    assert raised.value.field == "attempt" and str(raised.value).endswith("tensorquill reads an integer of 0 or more")
    # The fields limited to the sender's own, as `rows_to_send` checks them.
    with pytest.raises(MalformedRecordError):
        check({"step": 2**63}, None, ("rank", "attempt", "step"))


# The readers: each refuses the record as it refuses any malformed one.

@pytest.mark.parametrize("engine", ["python", "rust"])
@pytest.mark.parametrize("field", FIELDS)
def test_d6_build_evidence_refuses_the_file_with_either_engine(field, engine, tmp_path):
    if engine == "rust":
        needs_rust()
    path = _one(tmp_path, field, PAST[field])
    with pytest.raises(MalformedRecordError) as raised:
        tensorquill.build_evidence(path, detectors=engine, **({"server_binary": BINARY} if engine == "rust" else {}))
    assert raised.value.field == field and raised.value.where == f"{path}, line 1"


@pytest.mark.parametrize("engine", ["python", "rust"])
@pytest.mark.parametrize("field", FIELDS)
def test_d6_both_engines_judge_the_largest_valid_value(field, engine, tmp_path):
    if engine == "rust":
        needs_rust()
    path = _one(tmp_path, field, EDGE[field])
    found = tensorquill.build_evidence(path, detectors=engine, **({"server_binary": BINARY} if engine == "rust" else {}))
    assert found["kind_checks"]["grad_spike"]["value"] == "checked"


@pytest.mark.parametrize("field", FIELDS)
def test_d6_records_in_memory_are_refused_by_their_place(field):
    with pytest.raises(MalformedRecordError) as raised:
        tensorquill.load_records([_line(0), {**_line(1), field: PAST[field]}])
    assert raised.value.field == field and raised.value.where == "record 1 of the records passed"


@pytest.mark.parametrize("field", FIELDS)
def test_d6_the_rust_route_refuses_before_arrow(field):
    """`server_detect.records_stream` (and so `server_report` and `server_run_report`): a named
    refusal, where a bare `OverflowError` or `ArrowInvalid` came from building the row."""
    pytest.importorskip("pyarrow")
    from tensorquill.server_detect import records_stream

    with pytest.raises(MalformedRecordError) as raised:
        records_stream([_line(0), {**_line(1), field: PAST[field]}])
    assert raised.value.field == field and raised.value.where == "record 1 of the records passed"


@pytest.mark.parametrize("field", FIELDS)
def test_d6_record_to_row_refuses_before_moved_is_touched(field):
    pytest.importorskip("pyarrow")
    from tensorquill import flight

    moved = ["kept"]
    with pytest.raises(MalformedRecordError) as raised:
        flight.record_to_row({**_line(1), field: PAST[field]}, moved=moved, where="w")
    assert raised.value.field == field and moved == ["kept"]
    with pytest.raises(MalformedRecordError):
        flight.records_to_batch([_line(0), {**_line(1), field: PAST[field]}])


@pytest.mark.parametrize("field", FIELDS)
def test_d6_the_command_line_exits_2_with_one_line(field, tmp_path):
    import os
    import subprocess
    import sys

    path = _one(tmp_path, field, PAST[field])
    src = Path(__file__).parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(src), "PYTHONWARNINGS": "ignore:Failed to initialize NumPy"}
    done = subprocess.run([sys.executable, "-m", "tensorquill", str(path), "--detectors", "python",
                           "-o", str(tmp_path / "out.json")], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 2 and "Traceback" not in done.stderr
    said = done.stderr.strip().splitlines()
    assert len(said) == 1 and said[0].startswith(f"tensorquill: error: {field} is {PAST[field]} at {path}, line 1"), said
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("field", FIELDS)
def test_d6_the_watcher_refuses_and_writes_nothing(field, tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(_line(0)) + "\n" + json.dumps({**_line(1), field: PAST[field]}) + "\n")
    out = tmp_path / "out.json"
    with pytest.raises(MalformedRecordError) as raised:
        evidence.watch(path, out, interval=0.0, polls=1)
    assert raised.value.field == field and raised.value.where == f"{path}, line 2"
    assert not out.exists()


@pytest.mark.parametrize("field", FIELDS)
def test_d6_first_divergence_and_compare_refuse(field, tmp_path):
    from tensorquill.compare import compare_runs
    from tensorquill.divergence import first_divergence

    a = [_line(step, run_id="a") for step in range(30)]
    b = [_line(step, run_id="b") for step in range(30)]
    b[4] = {**b[4], field: PAST[field]}
    with pytest.raises(MalformedRecordError) as raised:
        first_divergence(a, b)
    assert raised.value.field == field and raised.value.where == "record 4 of the records passed, run B"
    (tmp_path / "a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in a))
    (tmp_path / "b.jsonl").write_text("".join(json.dumps(r) + "\n" for r in b))
    with pytest.raises(MalformedRecordError) as raised:
        compare_runs("a", "b", source=tmp_path)
    assert raised.value.field == field and raised.value.where == f"{tmp_path / 'b.jsonl'}, line 5"


@pytest.mark.parametrize("field", FIELDS)
def test_d6_an_mcp_tool_answers_the_error_and_the_next_call(field, tmp_path):
    from tensorquill import mcp_server

    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in [_line(0), _line(1), {**_line(2), field: PAST[field]}]))
    source = mcp_server.Source(path)

    def reply(name: str) -> dict[str, Any]:
        return mcp_server.call_tool(source, 1, {"name": name, "arguments": {}})["result"]

    answer = reply("list_incidents")
    assert answer["isError"] is True
    assert answer["content"][0]["text"].startswith(f"{field} is {PAST[field]} at {path}, line 3")
    path.write_text("".join(json.dumps(_line(step)) + "\n" for step in range(3)))
    assert reply("list_incidents")["isError"] is False


@pytest.mark.parametrize("field", FIELDS)
def test_d6_offline_detect_cannot_be_given_such_a_value_and_refuses_its_wrapped_bits(field):
    """`tensorquill-server detect` reads Arrow, whose Int64 and Int32 columns cannot hold the value: the
    only route to it from a record is the SDK's, which refuses first (above). The same bits read as the
    column's type are negative, which `detect` refuses with one line naming the field, so the two engines
    refuse alike (a pin: Rust is unchanged)."""
    needs_rust()
    import io
    import subprocess

    import pyarrow as pa
    from tensorquill import flight

    row = flight.record_to_row(_line(0))
    with pytest.raises((OverflowError, pa.ArrowInvalid)):
        pa.RecordBatch.from_pylist([{**row, field: PAST[field]}], schema=flight.flight_schema())
    wrapped = -PAST[field]  # 2**63 as Int64, 2**31 as Int32
    batch = pa.RecordBatch.from_pylist([{**row, field: wrapped}], schema=flight.flight_schema())
    out = io.BytesIO()
    with pa.ipc.new_stream(out, batch.schema) as writer:
        writer.write_batch(batch)
    done = subprocess.run([BINARY, "detect"], input=out.getvalue(), capture_output=True, timeout=120)
    lines = done.stderr.decode().strip().splitlines()
    assert done.returncode == 1 and done.stdout == b"" and len(lines) == 1, lines
    assert lines[0].startswith(f"tensorquill-server: row 1: {field} is {wrapped}"), lines


# The sender and the training process: nothing raises, and the valid lines are sent.

@pytest.mark.parametrize("field", FIELDS)
def test_d6_rows_to_send_leaves_the_line_out_and_names_it(field):
    pytest.importorskip("pyarrow")
    from tensorquill.flight import flight_schema, rows_to_send

    records = [_line(0), {**_line(1), field: PAST[field]}, _line(2), {**_line(3), field: EDGE[field]}]
    refused: list[Any] = []
    batch, steps, skipped = rows_to_send(records, flight_schema(), refused=refused)
    assert (steps, skipped, refused) == ([0, 2, records[3]["step"]], 1, [(1, field, PAST[field])])
    assert batch.num_rows == 3


def _server():
    needs_rust()
    from test_flight_e2e import Server

    return Server


@pytest.mark.parametrize("field", FIELDS)
def test_d6_the_sender_skips_the_line_and_stores_every_valid_one(field, tmp_path, capfd):
    Server = _server()
    from tensorquill.flight import FlightSender
    from test_flight_e2e import stored_records

    run_id = f"ar2-{field}"
    path = tmp_path / "rank0.jsonl"
    path.write_text("".join(json.dumps({**_line(s, run_id=run_id), **({field: PAST[field]} if s == 5 else {})}) + "\n"
                            for s in range(10)))
    server = Server(tmp_path / "store")
    try:
        sender = FlightSender(path, server.endpoint, run_id, 0, batch_steps=50, batch_seconds=0.05,
                              poll_seconds=0.02, backoff_cap=0.2)
        closed = sender.close(20.0)
    finally:
        assert server.terminate() == 0
    assert closed is False and (sender.skipped_lines, sender.left_out_lines) == (1, 1)
    assert sender.acked_offset == path.stat().st_size
    assert sorted(r["step"] for r in stored_records(tmp_path / "store", run_id)) == [0, 1, 2, 3, 4, 6, 7, 8, 9]
    said = [s for s in capfd.readouterr().err.splitlines() if s.startswith("tensorquill:")]
    assert f"its {field} is {PAST[field]}, and its column in the stream cannot hold it" in said[0], said


def _train(session: Any, model: Any, steps: int, log_step: int | None = None) -> None:
    import torch

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for i in range(steps):
        optimizer.zero_grad()
        model(torch.randn(4, 8)).pow(2).mean().backward()
        optimizer.step()
        if log_step is None:
            session.log({"loss": 1.0})
            session.step()
        else:
            session.log({"loss": 1.0}, step=log_step + i)
    session.flush()


@pytest.mark.parametrize("how", ["start_step", "log_step"])
def test_d6_a_session_past_int64_never_raises_and_sends_the_steps_it_can(how, tmp_path, monkeypatch):
    """`init(start_step=)` near the bound and `log(step=)` past it (both take any integer of 0 or more):
    `step()`, `log()`, `flush()` and `close()` return, the file keeps every record as written, and the
    sender sends the steps Int64 holds and goes on. A pin: no caller of `identity.check` is in the
    training thread, and the sender's thread catches its refusal."""
    torch = pytest.importorskip("torch")
    Server = _server()
    import pyarrow.parquet as pq

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_TOKEN", raising=False)
    monkeypatch.delenv("TENSORQUILL_ENDPOINT", raising=False)
    server = Server(tmp_path / "store")
    try:
        model = torch.nn.Linear(8, 4)
        start = {"start_step": INT64_MAX - 1} if how == "start_step" else {}
        session = tensorquill.init(model, run_id=f"ar2-{how}", auto_step=False, endpoint=server.endpoint, **start)
        _train(session, model, 4, log_step=None if how == "start_step" else INT64_MAX - 1)
        session.close()
    finally:
        assert server.terminate() == 0
    # `log(step=)` with no `step()` writes scalar-only records, stored as `scalars-*.parquet`.
    written = [json.loads(line) for line in (tmp_path / "tensorquill-out" / f"ar2-{how}.rank0.jsonl").read_text().splitlines()]
    steps = sorted({r["step"] for r in written})
    assert steps[:2] == [INT64_MAX - 1, INT64_MAX] and max(steps) > INT64_MAX
    stored = sorted({step for part in (tmp_path / "store").rglob("*.parquet")
                     for step in pq.read_table(part, partitioning=None).column("step").to_pylist()})
    assert stored == [INT64_MAX - 1, INT64_MAX]


def test_d6_a_session_whose_rank_is_past_int32_never_raises(tmp_path, monkeypatch):
    """`RANK` past Int32: the session trains, flushes and closes without raising and writes its file;
    the server refuses the stream's descriptor (`rank=2147483648`), which the sender says on close and
    which stops nothing in training. A pin."""
    torch = pytest.importorskip("torch")
    Server = _server()

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_TOKEN", raising=False)
    monkeypatch.delenv("TENSORQUILL_ENDPOINT", raising=False)
    monkeypatch.setenv("RANK", str(2**31))
    monkeypatch.setenv("WORLD_SIZE", "4")
    server = Server(tmp_path / "store")
    try:
        model = torch.nn.Linear(8, 4)
        with pytest.warns(RuntimeWarning):  # the rank is not below the world size
            session = tensorquill.init(model, run_id="ar2-rank", auto_step=False, endpoint=server.endpoint)
        _train(session, model, 3)
        session.close()
    finally:
        assert server.terminate() == 0
    written = (tmp_path / "tensorquill-out" / f"ar2-rank.rank{2**31}.jsonl").read_text().splitlines()
    assert [json.loads(line)["rank"] for line in written] == [2**31] * len(written) and len(written) >= 3
    assert not list((tmp_path / "store").rglob("*.parquet"))
    with pytest.raises(MalformedRecordError) as raised:
        tensorquill.build_evidence(tmp_path / "tensorquill-out" / f"ar2-rank.rank{2**31}.jsonl")
    assert raised.value.field == "rank"
