"""Wave AO, lane AO-2: the sender (`flight.py`, `sink.py`, `identity.py`).

BUG-02, the sender's side: a line of record version 2 that is not a whole scalar-only record (no
`"record": "scalars"`) was sent as a step row (`record_version` 2, a null `seq`), and every Python
reader then refused the stored part. Such a line is now skipped and counted like any other line the
sender cannot send (`identity.check`'s rule for version 2, the one `is_scalar_record` mirrors).

Each test that needs pyarrow or the server binary (`TENSORQUILL_SERVER_BIN`, or one beside the
interpreter) is skipped without it, one by one.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path

import pytest

from tensorquill import flight
from tensorquill.identity import is_scalar_record
from tensorquill.server_detect import find_binary

try:  # each test is collected and skipped without pyarrow
    import pyarrow  # noqa: F401
except ImportError:
    pyarrow = None

BINARY = find_binary()
needs_arrow = pytest.mark.skipif(pyarrow is None, reason="pyarrow is not installed")


def skip_without_server() -> None:
    if pyarrow is None:
        pytest.skip("pyarrow is not installed")
    if not BINARY or not os.path.isfile(BINARY):
        pytest.skip("tensorquill-server is not built")


class Server:
    """A server on a loopback port with its own store; `close()` stops it and returns its stderr."""

    def __init__(self, store: Path) -> None:
        env = {k: v for k, v in os.environ.items() if k not in ("TENSORQUILL_TOKEN", "TENSORQUILL_ENDPOINT")}
        self.process = subprocess.Popen(
            [BINARY, "--listen", "127.0.0.1:0", "--store", str(store), "--flush-steps", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        line = self.process.stdout.readline()
        assert "listening on" in line, line + self.process.stderr.read()
        self.endpoint = f"grpc://127.0.0.1:{int(line.rsplit(':', 1)[1])}"
        self.store = store

    def close(self) -> str:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
        self.process.wait(30)
        err = self.process.stderr.read()
        self.process.stdout.close()
        self.process.stderr.close()
        return err


def write_lines(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def step_record(run_id: str, step: int, **more) -> dict:
    return {"schema_version": 1, "run_id": run_id, "rank": 0, "step": step, "ts": float(step), "grad_norm": 1.0,
            **more}


def scalar_record(run_id: str, step: int, seq: int, scalars: dict) -> dict:
    return {"schema_version": 2, "record": "scalars", "run_id": run_id, "rank": 0, "step": step, "seq": seq,
            "ts": float(step), "scalars": scalars}


# The line BUG-02 names: version 2 with no `"record": "scalars"`.
def bare_v2(run_id: str, step: int) -> dict:
    return {"schema_version": 2, "run_id": run_id, "rank": 0, "step": step, "loss": 1.0}


# --- BUG-02: a version 2 line that is not a whole scalar-only record ------------------------


@needs_arrow
def test_bug02_a_version_2_line_without_the_record_marker_is_left_out_and_counted():
    refused: list = []
    batch, steps, skipped = flight.rows_to_send([bare_v2("p", 4)], flight.flight_schema(), refused=refused)
    assert batch.num_rows == 0, batch.to_pylist()
    assert steps == [] and skipped == 1
    assert [(index, field, value) for index, field, value in refused] == [(0, "schema_version", 2)]


@needs_arrow
def test_bug02_the_audit_record_with_a_seq_but_no_marker_is_left_out_too():
    record = {"schema_version": 2, "run_id": "local", "rank": 0, "attempt": 1, "step": 4, "seq": 0,
              "scalars": {"loss": 1.0}}
    assert not is_scalar_record(record)
    batch, steps, skipped = flight.rows_to_send([record], flight.flight_schema())
    assert (batch.num_rows, steps, skipped) == (0, [], 1)


@needs_arrow
def test_bug02_pins_what_is_still_sent():
    """A whole scalar-only record is sent with its seq; a step record of version 0 or 1 as a step row;
    a version this package does not read (3) is sent for the server to count (RS10); a scalar-only
    record with no seq is left out naming seq, as before."""
    records = [
        scalar_record("p", 7, 0, {"eval/loss": 1.5}),
        step_record("p", 0),
        {"run_id": "p", "rank": 0, "step": 1},
        {**step_record("p", 2), "schema_version": 3},
        {"schema_version": 2, "record": "scalars", "run_id": "p", "rank": 0, "step": 9, "scalars": {"x": 1.0}},
    ]
    refused: list = []
    batch, steps, skipped = flight.rows_to_send(records, flight.flight_schema(), refused=refused)
    rows = batch.to_pylist()
    assert [(r["step"], r["seq"], r["record_version"], r["schema_version"]) for r in rows] == [
        (7, 0, 2, 2), (0, None, 1, 2), (1, None, None, 2), (2, None, 3, 3)]
    assert steps == [0, 1, 2] and skipped == 1
    assert [(i, f) for i, f, _ in refused] == [(4, "seq")]


def test_bug02_through_a_real_server_the_store_stays_readable(tmp_path, capsys):
    skip_without_server()
    from tensorquill.evidence import load_records

    running = Server(tmp_path / "store")
    try:
        path = write_lines(tmp_path / "p.jsonl", [step_record("p", 0), bare_v2("p", 1), step_record("p", 2)])
        sender = flight.FlightSender(path, running.endpoint, "p", 0)
        sender.close(20.0)
        assert sender.skipped_lines == 1
    finally:
        running.close()
    stored = load_records(tmp_path / "store" / "runs" / "p")
    assert [r["step"] for r in stored] == [0, 2]
    said = capsys.readouterr().err
    assert "(step 1) is not sent" in said and '"record": "scalars"' in said, said


# --- BUG-24 (a): an integer scalar a float64 cannot hold exactly ----------------------------
#
# Capture keeps a logged Python integer as it is (`capture.Session._scalar_values`), and the
# `.jsonl` file holds it exactly. The `scalars` column is Float64: the column keeps the nearest
# float, which is what the Rust detectors judge, and the exact integer travels beside it in
# `extra_json`, so a record read back from the store is the file's.

EXACT_EDGE = 2**53
PAST_EDGE = [2**53 + 1, -(2**53) - 1, 2**60 + 1, 2**63 + 1]


def _rows(records: list[dict], sender: bool) -> list[dict]:
    if sender:
        batch, _, skipped = flight.rows_to_send(records, flight.flight_schema())
        assert skipped == 0
        return batch.to_pylist()
    return flight.records_to_batch(records).to_pylist()


@needs_arrow
@pytest.mark.parametrize("sender", [False, True], ids=["record_to_row", "rows_to_send"])
def test_bug24a_an_integer_scalar_past_2_53_comes_back_exact(sender):
    scalars = {"loss": 1.5, **{f"n{i}": v for i, v in enumerate(PAST_EDGE)}}
    record = step_record("r", 0, scalars=scalars)
    row = _rows([record], sender)[0]
    assert dict(row["scalars"]) == {name: float(v) for name, v in scalars.items()}  # the column: the Rust view
    back = flight.row_to_record(row)
    assert back["scalars"] == scalars
    assert list(back["scalars"]) == list(scalars)  # in the record's order
    assert all(type(back["scalars"][f"n{i}"]) is int for i in range(len(PAST_EDGE)))


@needs_arrow
def test_bug24a_a_scalar_only_record_keeps_its_exact_integer_too():
    record = scalar_record("r", 3, 0, {"counter": 2**53 + 1})
    row = _rows([record], True)[0]
    assert row["seq"] == 0
    back = flight.row_to_record(row)
    assert back["scalars"] == {"counter": 2**53 + 1} and is_scalar_record(back)


@needs_arrow
def test_bug24a_pins_an_integer_a_float_holds_and_a_float():
    """2**53, a small integer and a float stay in the column alone, with no `extra_json`, as before."""
    scalars = {"edge": EXACT_EDGE, "small": 3, "loss": 0.25, "neg": -(2**53)}
    row = _rows([step_record("r", 0, scalars=scalars)], True)[0]
    assert row["extra_json"] is None
    assert flight.row_to_record(row)["scalars"] == scalars


def test_bug24a_through_a_real_server_the_stored_series_is_the_files(tmp_path):
    skip_without_server()
    import torch

    import tensorquill as tq
    from tensorquill.evidence import load_records, scalar_series

    path = tmp_path / "metrics.jsonl"
    with tq.init(torch.nn.Linear(1, 1), sink=tq.JsonlSink(path), run_id="metrics", mirror_scalars=False) as session:
        session.log(counter=2**53 + 1, step=0)
        session.log(eval_loss=1.5, step=1)
    running = Server(tmp_path / "store")
    try:
        assert flight.FlightSender(path, running.endpoint, "metrics", 0).close(20.0)
    finally:
        running.close()
    stored = tmp_path / "store" / "runs" / "metrics"
    local = scalar_series(path)
    assert local["counter"] == [[0, 2**53 + 1]]
    assert scalar_series(stored) == local
    assert [r["scalars"] for r in load_records(stored, scalars=True) if "scalars" in r] == \
        [r["scalars"] for r in load_records(path, scalars=True) if "scalars" in r]


# --- BUG-24 (b): a bool among `scalars` -----------------------------------------------------
#
# A bool is not a number in either engine (lane W3, DET-16): the `scalars` column leaves it out, as
# before, so the Rust detectors read the same map. It was dropped from the row altogether, so a
# record read back from the store lacked what the file holds; now it travels in `extra_json` and
# comes back after the column's entries.


@needs_arrow
@pytest.mark.parametrize("sender", [False, True], ids=["record_to_row", "rows_to_send"])
def test_bug24b_a_bool_scalar_comes_back_from_the_row(sender):
    scalars = {"loss": 1.0, "ok": True, "lr": 0.5, "off": False}
    row = _rows([step_record("r", 0, scalars=scalars)], sender)[0]
    assert dict(row["scalars"]) == {"loss": 1.0, "lr": 0.5}  # the column, as the Rust detectors read it
    back = flight.row_to_record(row)
    assert back["scalars"] == scalars
    assert back["scalars"]["ok"] is True and back["scalars"]["off"] is False


@needs_arrow
def test_bug24b_a_map_of_bools_alone_and_a_scalar_only_record():
    rows = _rows([step_record("r", 0, scalars={"ok": True}), scalar_record("r", 5, 0, {"done": True, "x": 2.0})], True)
    assert [dict(r["scalars"]) for r in rows] == [{}, {"x": 2.0}]
    back = [flight.row_to_record(r) for r in rows]
    assert [r["scalars"] for r in back] == [{"ok": True}, {"done": True, "x": 2.0}]


@needs_arrow
def test_bug24b_an_entry_its_column_refuses_travels_alone_with_the_bools():
    """Any other refused entry travels alone in `extra_json`, beside the bools and the exact integers,
    and the rest of the map stays in the column (AQ5-D7, the author, Nov 26, 2026; under lane W3's rule
    it moved the whole map). A map with no held entry still travels whole."""
    scalars = {"loss": 1.0, "note": "text", "ok": True, "n": 2**53 + 1}
    row = _rows([step_record("r", 0, scalars=scalars)], True)[0]
    assert row["scalars"] == [("loss", 1.0), ("n", float(2**53 + 1))]
    assert json.loads(row["extra_json"])["scalars"] == {"note": "text", "ok": True, "n": 2**53 + 1}
    assert flight.row_to_record(row)["scalars"] == scalars
    alone = {"note": "text", "ok": True}
    row = _rows([step_record("r", 0, scalars=alone)], True)[0]
    assert row["scalars"] is None
    assert flight.row_to_record(row)["scalars"] == alone


def test_bug24b_through_a_real_server_the_stored_record_is_the_files(tmp_path):
    skip_without_server()
    from tensorquill.evidence import load_records

    path = write_lines(tmp_path / "b.jsonl", [step_record("b", 0, scalars={"loss": 1.0, "ok": True}),
                                               scalar_record("b", 9, 0, {"flag": False, "eval": 2.0})])
    running = Server(tmp_path / "store")
    try:
        assert flight.FlightSender(path, running.endpoint, "b", 0).close(20.0)
    finally:
        running.close()
    stored = tmp_path / "store" / "runs" / "b"
    assert [r["scalars"] for r in load_records(stored, scalars=True)] == \
        [r["scalars"] for r in load_records(path, scalars=True)]


# --- BUG-24 (c): what close() and send_file return over a line left out ----------------------
#
# A line of this run that the sender leaves out (a step past Int64, a refused key, BUG-02's line)
# is passed over: the saved offset moves past it, so every byte is confirmed. close() and
# send_file returned True, "the whole file was acknowledged", although the server never had that
# line, and a later send_file of the same file said True again. Recommended, and built: they
# return False while any line of this run in the confirmed part of the file was left out, and the
# offset file keeps the count, so a later sender of the same file says the same.
# `left_out_lines` is that count. Lines that are not this run's records (not JSON, another run's)
# are counted in `skipped_lines` as before and do not change what close() returns.


def _send(path: Path, endpoint: str, run_id: str) -> flight.FlightSender:
    sender = flight.FlightSender(path, endpoint, run_id, 0)
    sender.result = sender.close(20.0)
    return sender


def test_bug24c_a_line_past_int64_makes_close_and_send_file_false(tmp_path, capsys):
    skip_without_server()
    from tensorquill.evidence import load_records

    path = write_lines(tmp_path / "bs.jsonl", [step_record("bs", 0), scalar_record("bs", 2**63, 0, {"x": 1.0}),
                                                step_record("bs", 1)])
    running = Server(tmp_path / "store")
    try:
        first = _send(path, running.endpoint, "bs")
        assert first.acked_offset == path.stat().st_size  # every byte is confirmed
        assert first.result is False
        assert (first.skipped_lines, first.left_out_lines) == (1, 1)
        assert "1 line of run bs" in capsys.readouterr().err
        again = _send(path, running.endpoint, "bs")  # resumes at the saved offset: nothing is owed
        assert (again.result, again.left_out_lines, again.skipped_lines) == (False, 1, 0)
        assert flight.send_file(path, running.endpoint, "bs", timeout=20.0) is False
    finally:
        running.close()
    assert [r["step"] for r in load_records(tmp_path / "store" / "runs" / "bs")] == [0, 1]


def test_bug24c_bug02s_line_makes_close_false_and_the_count_survives_more_lines(tmp_path):
    skip_without_server()
    path = write_lines(tmp_path / "p.jsonl", [step_record("p", 0), bare_v2("p", 1)])
    running = Server(tmp_path / "store")
    try:
        assert _send(path, running.endpoint, "p").result is False
        with path.open("a") as file:
            file.write(json.dumps(step_record("p", 2)) + "\n")
        later = _send(path, running.endpoint, "p")
        assert (later.result, later.left_out_lines) == (False, 1)
        assert later.acked_offset == path.stat().st_size
    finally:
        running.close()


def test_bug24c_sent_again_from_byte_0_counts_each_line_once(tmp_path):
    skip_without_server()
    path = write_lines(tmp_path / "bs.jsonl", [scalar_record("bs", 2**63, 0, {"x": 1.0}), step_record("bs", 0)])
    running = Server(tmp_path / "store")
    try:
        assert _send(path, running.endpoint, "bs").result is False
        flight.offset_path(path).unlink()
        replayed = _send(path, running.endpoint, "bs")
        assert (replayed.result, replayed.left_out_lines) == (False, 1)
    finally:
        running.close()


def test_bug24c_the_offset_file_keeps_the_count(tmp_path):
    skip_without_server()
    path = write_lines(tmp_path / "bs.jsonl", [scalar_record("bs", 2**63, 0, {"x": 1.0}), step_record("bs", 0)])
    running = Server(tmp_path / "store")
    try:
        _send(path, running.endpoint, "bs")
    finally:
        running.close()
    document = flight.read_binding(flight.offset_path(path))
    assert document["offset"] == path.stat().st_size and document["left_out"] == 1


def test_bug24c_pins_a_whole_file_and_lines_that_are_not_this_runs(tmp_path):
    """True as before for a file whose every line of this run was sent, also beside a line that is
    not JSON and a line of another run (counted in `skipped_lines`, as before)."""
    skip_without_server()
    path = write_lines(tmp_path / "ok.jsonl", [step_record("ok", 0), scalar_record("ok", 5, 0, {"x": 1.0})])
    with path.open("a") as file:
        file.write("not json\n" + json.dumps(step_record("other", 3)) + "\n" + json.dumps(step_record("ok", 1)) + "\n")
    running = Server(tmp_path / "store")
    try:
        sender = _send(path, running.endpoint, "ok")
        assert (sender.result, sender.skipped_lines, getattr(sender, "left_out_lines", 0)) == (True, 2, 0)
        assert _send(path, running.endpoint, "ok").result is True
        assert flight.send_file(path, running.endpoint, "ok", timeout=20.0) is True
    finally:
        running.close()


# --- BUG-24 (d): the stop's step when the local file's close fails --------------------------
#
# `StreamingSink.close()` tells the heartbeat `sink_error` before it ends, so its last beat carries
# the stop; it gave no step, so `capture_stopped_step` was None although step 0 (or more) was
# committed, and the session's own stop, after it, does nothing once a stop is set. Step 0 and
# null are different (detector rules contract, part 7). The sink now names the last training step
# it wrote: at a close, every step the session committed has been written through it.


class _Beat:
    def __init__(self) -> None:
        self.stops: list = []
        self.closed = False

    def stop_capture(self, reason: str, step: int | None = None) -> None:
        self.stops.append((reason, step))

    def close(self) -> None:
        self.closed = True


class _Sender:
    def __init__(self) -> None:
        self.closed = False

    def close(self, timeout: float = 0.0) -> bool:
        self.closed = True
        return True


class _FailingSink:
    def __init__(self, path: Path, fail_write_at: int | None = None) -> None:
        self.path = path
        self.writes = 0
        self.fail_write_at = fail_write_at

    def write(self, records: list) -> None:
        self.writes += 1
        if self.writes == self.fail_write_at:
            raise OSError("disk full")

    def close(self) -> None:
        raise OSError("disk error")


def _closed_with(sink, writes: list[list[dict]]) -> list:
    beat, sender = _Beat(), _Sender()
    streaming = flight.StreamingSink(sink, sender, beat)
    for records in writes:
        try:
            streaming.write(records)
        except OSError:
            pass
    with pytest.raises(OSError, match="disk error"):
        streaming.close()
    assert beat.closed and sender.closed
    return beat.stops


@pytest.mark.parametrize("steps, wanted", [([0], 0), ([0, 1, 2], 2), ([5, 9], 9)])
def test_bug24d_the_stop_names_the_last_step_written(tmp_path, steps, wanted):
    writes = [[step_record("d", s)] for s in steps] + [[scalar_record("d", 500, 0, {"x": 1.0})]]
    assert _closed_with(_FailingSink(tmp_path / "d.jsonl"), writes) == [("sink_error", wanted)]


def test_bug24d_no_step_written_is_null(tmp_path):
    assert _closed_with(_FailingSink(tmp_path / "d.jsonl"), []) == [("sink_error", None)]
    only_scalars = [[scalar_record("d", 3, 0, {"x": 1.0})]]
    assert _closed_with(_FailingSink(tmp_path / "e.jsonl"), only_scalars) == [("sink_error", None)]


def test_bug24d_a_write_that_raised_is_not_counted(tmp_path):
    writes = [[step_record("d", 0)], [step_record("d", 1)]]
    assert _closed_with(_FailingSink(tmp_path / "d.jsonl", fail_write_at=2), writes) == [("sink_error", 0)]


def test_bug24d_through_init_and_a_real_server_step_0_is_kept(tmp_path):
    skip_without_server()
    import warnings

    import torch

    import tensorquill as tq
    from tensorquill.sink import JsonlSink

    class FailClose(JsonlSink):
        def close(self) -> None:
            super().close()
            raise OSError("disk error")

    running = Server(tmp_path / "store")
    try:
        for steps, every, wanted in ((1, 1, 0), (3, 1, 2), (3, 50, 2)):
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run_id = f"sc{steps}-{every}"
                session = tq.init(model, run_id=run_id, sink=FailClose(tmp_path / f"{run_id}.jsonl"),
                                  endpoint=running.endpoint, every=every, heartbeat_seconds=60.0, mirror_scalars=False)
                for _ in range(steps):
                    model(torch.ones(1, 2)).sum().backward()
                    optimizer.step()
                    optimizer.zero_grad()
                session.close()
            beat = session._sink.heartbeat
            assert (beat.capture_stopped, beat.capture_stopped_step) == ("sink_error", wanted)
    finally:
        running.close()
