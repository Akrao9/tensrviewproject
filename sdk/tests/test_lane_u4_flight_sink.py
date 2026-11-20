"""Lane U4: captured steps lost between the local file and the server (audit C07, C08, C09,
SDK-21, SDK-26's offset scratch file, SDK-14 and C24's end marker half).

Most tests run against the in-process stand-in server of `test_flight.py`. The two marked
`needs_server` run the real `tensorquill-server` (`TENSORQUILL_SERVER_BIN`) and skip without it.
"""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")
import pyarrow as pa  # noqa: E402
import pyarrow.flight as pa_flight  # noqa: E402

import tensorquill  # noqa: E402
from tensorquill import flight  # noqa: E402
from tensorquill.flight import (  # noqa: E402
    FlightSender,
    Heartbeat,
    flight_schema,
    offset_path,
    read_binding,
    read_offset,
    rows_to_send,
    send_file,
    write_offset,
)
from tensorquill.sink import end_marker_path, read_end_marker, write_end_marker  # noqa: E402
from test_flight import FakeServer, Store, append, record, wait_for  # noqa: E402
from test_flight_e2e import BINARY, Server, stored_records  # noqa: E402

needs_server = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built (TENSORQUILL_SERVER_BIN)")


def fast(path: Path, endpoint: str, run_id: str = "run-a", **options) -> FlightSender:
    options = {"batch_seconds": 0.05, "poll_seconds": 0.02, "backoff_cap": 0.2, **options}
    return FlightSender(path, endpoint, run_id, 0, **options)


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "store-a")


@pytest.fixture
def server(store):
    running = FakeServer(store)
    yield running
    running.stop()


@pytest.fixture
def second(tmp_path):
    """Another server, with a store of its own."""
    other = Store(tmp_path / "store-b")
    running = FakeServer(other)
    yield running, other
    running.stop()


def lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").split("\n")


def readable_steps(path: Path) -> list[int]:
    """What a plain line-by-line reader gets: the steps of the lines that are whole JSON."""
    steps = []
    for line in lines(path):
        try:
            steps.append(json.loads(line)["step"])
        except ValueError:
            continue
    return steps


# --- C07: the writer appending after a torn last line ------------------------------------

TORN = '{"schema_version":1,"run_id":"run-a","rank":0,"step":2,"gra'


def test_c07_sink_ends_a_torn_last_line_so_the_first_new_step_is_read(tmp_path):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(1)])
    with path.open("a", encoding="utf-8") as file:
        file.write(TORN)  # the process was killed in the middle of step 2
    sink = tensorquill.JsonlSink(path)
    sink.write([record(3)])
    sink.close()
    assert readable_steps(path) == [1, 3]  # step 3, the first after the restart, is kept
    # The fragment is still there, a line of its own that is not a whole object: incomplete history.
    assert lines(path)[1] == TORN
    with pytest.raises(ValueError):
        json.loads(lines(path)[1])
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_c07_a_fragment_that_is_a_whole_object_becomes_a_whole_line(tmp_path):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(1)])
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record(2), separators=(",", ":")))  # killed just before the newline
    sink = tensorquill.JsonlSink(path)
    sink.write([record(3)])
    sink.close()
    assert readable_steps(path) == [1, 2, 3]


def test_c07_a_clean_or_empty_file_is_not_touched_and_its_end_marker_still_matches(tmp_path):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(1)])
    before = path.read_bytes()
    assert write_end_marker(path, run_id="run-a", rank=0, attempt=1)
    tensorquill.JsonlSink(path).close()
    assert path.read_bytes() == before
    assert read_end_marker(path)["state"] == "closed"
    empty = tmp_path / "empty.rank0.jsonl"
    tensorquill.JsonlSink(empty).close()
    assert empty.read_bytes() == b""


def test_c07_sender_waits_at_a_torn_tail_and_reads_the_step_the_next_writer_appends(tmp_path, store, server):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(0), record(1)])
    whole = path.stat().st_size
    with path.open("a", encoding="utf-8") as file:
        file.write(TORN)
    sender = fast(path, server.endpoint)
    # The plan's torn-line rule: no newline yet, so the sender does not pass the fragment.
    wait_for(lambda: sender.acked_offset == whole)
    time.sleep(0.2)
    assert store.steps() == [0, 1]
    sink = tensorquill.JsonlSink(path)  # the restarted process
    sink.write([record(3), record(4)])
    sink.close()
    assert sender.close() is True
    assert store.steps() == [0, 1, 3, 4]
    assert sender.skipped_lines == 1  # the fragment
    assert read_offset(offset_path(path)) == path.stat().st_size


@needs_server
def test_c07_real_server_stores_the_first_step_after_a_torn_tail(tmp_path):
    path = tmp_path / "torn.rank0.jsonl"
    append(path, [record(0, run_id="torn"), record(1, run_id="torn")])
    with path.open("a", encoding="utf-8") as file:
        file.write('{"schema_version":1,"run_id":"torn","rank":0,"step":2,"gra')
    sink = tensorquill.JsonlSink(path)
    sink.write([record(3, run_id="torn"), record(4, run_id="torn")])
    sink.close()
    real = Server(tmp_path / "store")
    try:
        assert send_file(path, real.endpoint, "torn", timeout=30.0) is True
    finally:
        assert real.terminate() == 0
    assert [r["step"] for r in stored_records(tmp_path / "store", "torn")] == [0, 1, 3, 4]


# --- C08: the saved offset bound to the server and to the file ---------------------------


def test_c08_offset_saved_for_one_server_is_not_used_for_another(tmp_path, store, server, second):
    other, other_store = second
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(step) for step in range(5)])
    assert send_file(path, server.endpoint, "run-a", timeout=10) is True
    assert store.steps() == list(range(5))
    # An empty second store: the offset that the first server confirmed says nothing about it.
    assert send_file(path, other.endpoint, "run-a", timeout=10) is True
    assert other_store.steps() == list(range(5))  # before the fix: [], and send_file said True
    binding = read_binding(offset_path(path))
    assert binding["endpoint"] == other.endpoint and binding["offset"] == path.stat().st_size
    # Back to the first: sent again from the start, and its store keeps each step once.
    assert send_file(path, server.endpoint, "run-a", timeout=10) is True
    assert store.steps() == list(range(5))
    assert store.received_rows == 10


@pytest.mark.parametrize("how", ["deleted_and_written_again", "rewritten_in_place"])
def test_c08_replaced_file_of_equal_size_is_sent_from_its_start(tmp_path, store, server, how):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(step, ts=1.5) for step in range(5)])
    size = path.stat().st_size
    assert send_file(path, server.endpoint, "run-a", timeout=10) is True
    if how == "deleted_and_written_again":
        path.unlink()
    else:
        path.write_bytes(b"")  # the same inode, emptied
    append(path, [record(step, ts=2.5) for step in range(5, 10)])
    assert path.stat().st_size == size  # the only guard before the fix was the size
    sender = fast(path, server.endpoint)
    assert sender.close() is True
    assert store.steps() == list(range(10))  # before the fix: [0, 1, 2, 3, 4], and close() said True
    assert sender.replayed == "the file's bytes are not the ones the offset was saved for"


def test_c08_unchanged_file_resumes_and_sends_nothing_twice(tmp_path, store, server):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(step) for step in range(60)])
    assert fast(path, server.endpoint).close() is True
    append(path, [record(step) for step in range(60, 100)])
    sender = fast(path, server.endpoint)
    assert sender.close() is True
    assert store.received_rows == 100  # the second sender started at the saved offset
    assert store.steps() == list(range(100))
    assert sender.replayed is None


def test_c08_offset_without_a_binding_is_sent_again_from_the_start(tmp_path, store, server):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(step) for step in range(5)])
    write_offset(offset_path(path), path.stat().st_size)  # an offset file from before the binding
    sender = fast(path, server.endpoint)
    assert sender.close() is True
    assert store.steps() == list(range(5))
    assert sender.replayed == "the offset file names no server or file"


def test_c08_missing_file_is_an_error_and_not_nothing_to_send(tmp_path, server, capfd):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(0)])
    assert send_file(path, server.endpoint, "run-a", timeout=10) is True
    path.unlink()
    started = time.monotonic()
    assert send_file(path, server.endpoint, "run-a", timeout=60.0) is False
    assert time.monotonic() - started < 5.0  # it says so at once, without waiting out the timeout
    assert "cannot be read" in capfd.readouterr().err


@needs_server
def test_c08_real_servers_with_two_stores_each_get_the_whole_file(tmp_path):
    path = tmp_path / "two.rank0.jsonl"
    append(path, [record(step, run_id="two") for step in range(30)])
    first, second_server = Server(tmp_path / "store-1"), Server(tmp_path / "store-2")
    try:
        assert send_file(path, first.endpoint, "two", timeout=30.0) is True
        assert send_file(path, second_server.endpoint, "two", timeout=30.0) is True
    finally:
        assert first.terminate() == 0 and second_server.terminate() == 0
    for store_root in (tmp_path / "store-1", tmp_path / "store-2"):
        assert [r["step"] for r in stored_records(store_root, "two")] == list(range(30))


def idle_sender(path: Path, monkeypatch, endpoint: str = "grpc://127.0.0.1:9") -> FlightSender:
    """A sender whose thread does nothing, to drive its methods from the test."""
    with monkeypatch.context() as patched:
        patched.setattr(FlightSender, "_run", lambda self: None)
        return FlightSender(path, endpoint, "run-a", 0)


def test_c08_ack_of_an_offset_not_sent_after_a_replay_is_not_taken(tmp_path, monkeypatch):
    # The server keeps the highest offset of a stream that ended before it stored it
    # (`ingest.rs`, `detach` clears only the ack channel) and acks it on the next stream.
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(step, ts=1.5) for step in range(5)])
    old_end = path.stat().st_size
    sender = idle_sender(path, monkeypatch)
    assert sender._resume() == 0
    sender._sent(path.read_bytes(), old_end)
    sender._ack(old_end)  # the old file's offset, bound to the old file
    path.unlink()  # replaced by a file with fewer, longer lines
    append(path, [record(step, ts=2.5, note="x" * 200) for step in range(5, 7)])
    first = len(lines(path)[0]) + 1
    sender = idle_sender(path, monkeypatch)
    assert sender._resume() == 0 and sender.replayed is not None
    sender._sent(path.read_bytes()[:first], first)
    sender._ack(old_end)  # max(this stream's batch end, the old stream's held offset)
    assert sender.acked_offset == 0  # not taken: it names the old file's bytes
    assert read_offset(offset_path(path)) == old_end and sender.unacked_bytes() == path.stat().st_size
    sender._ack(first)
    assert sender.acked_offset == first
    assert read_binding(offset_path(path))["head"] == flight.file_identity(path, first)["head"]


def test_c08_ack_beyond_the_end_of_the_file_is_not_taken(tmp_path, monkeypatch):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(step) for step in range(3)])
    sender = idle_sender(path, monkeypatch)
    assert sender._resume() == 0 and sender.replayed is None
    sender._ack(path.stat().st_size + 100)  # an offset no stream of this file sent
    assert sender.acked_offset == 0 and not offset_path(path).exists()
    middle = len(lines(path)[0]) + 1
    sender._ack(middle)  # one this stream did not send but the file has: bound to its bytes
    assert read_binding(offset_path(path)) == {
        "version": 2, "offset": middle, "endpoint": "grpc://127.0.0.1:9", "run_id": "run-a", "rank": 0,
        **flight.file_identity(path, middle)}


# --- C09: the retry loop after a long outage ---------------------------------------------


class FakeClosing:
    """The sender's closing event on a fake clock: a wait returns at once and is recorded."""

    def __init__(self) -> None:
        self.flag = False
        self.waits: list[float] = []

    def is_set(self) -> bool:
        return self.flag

    def set(self) -> None:
        self.flag = True

    def wait(self, timeout: float | None = None) -> bool:
        self.waits.append(timeout)
        return self.flag


def outage(tmp_path, monkeypatch, failures: int, **options) -> FakeClosing:
    """Run the real retry loop on this thread against `failures` connection failures in a row."""
    path = tmp_path / "run-a.rank0.jsonl"
    path.touch()
    with monkeypatch.context() as patched:
        patched.setattr(FlightSender, "_run", lambda self: None)  # no thread: the loop runs below
        sender = FlightSender(path, "grpc://127.0.0.1:9", "run-a", 0, **options)
    closing = FakeClosing()
    sender._closing = closing
    tries = [0]

    def down() -> None:
        tries[0] += 1
        if tries[0] > failures:
            sender._deadline = 0.0
            closing.set()
        raise ConnectionError("the server is down")

    sender._stream = down
    FlightSender._run(sender)  # returns: the deadline has passed
    assert tries[0] == failures + 1
    return closing


def test_c09_thousands_of_failures_keep_retrying_at_the_cap(tmp_path, monkeypatch):
    closing = outage(tmp_path, monkeypatch, 5000)
    delays = closing.waits
    assert len(delays) == 5000
    assert all(0.0 <= delay <= flight.BACKOFF_CAP for delay in delays)
    assert max(delays[-1000:]) > flight.BACKOFF_CAP / 2  # still drawn up to the cap, long after attempt 1,025


def test_c09_an_outage_leaves_the_global_random_untouched(tmp_path, monkeypatch):
    random.seed(1234)
    expected = [random.random() for _ in range(5)]
    random.seed(1234)
    outage(tmp_path, monkeypatch, 200)
    assert [random.random() for _ in range(5)] == expected


def test_c09_backoff_ceiling_doubles_and_holds_at_the_cap(tmp_path, monkeypatch):
    path = tmp_path / "run-a.rank0.jsonl"
    with monkeypatch.context() as patched:
        patched.setattr(FlightSender, "_run", lambda self: None)
        sender = FlightSender(path, "grpc://127.0.0.1:9", "run-a", 0, backoff_base=0.1, backoff_cap=30.0)
    ceiling, seen = 0.1, []
    for _ in range(3000):
        delay, ceiling = sender._backoff(ceiling)
        seen.append(ceiling)
        assert 0.0 <= delay <= 30.0
    assert seen[:3] == [0.2, 0.4, 0.8] and seen[-1] == 30.0


# --- SDK-21: a line the schema cannot hold -----------------------------------------------


def test_sdk21_unstorable_lines_are_skipped_and_counted_and_the_sender_moves_on(tmp_path, store, server):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(0)])
    bad = [
        {**record(1), "attempt": None},  # the audit's case: an explicit null in a key column
        {**record(2), "rank": "0"},
        {**record(3), "schema_version": None},
        {**record(4), "step": 2**70},  # too large for the int64 column: Arrow refuses it
        {**record(5), "moe_top_k": {"router": 2**40}},  # too large for its int32 map
    ]
    append(path, bad)
    with path.open("a", encoding="utf-8") as file:
        # Stored since wave AA (lane AA4, L20), its NaN sent as "nan": skipped before.
        file.write('{"schema_version":1,"run_id":"run-a","rank":0,"step":6,"layers":{"x":NaN}}\n')
    append(path, [record(7), record(8)])
    sender = fast(path, server.endpoint)
    assert sender.close() is True
    assert store.steps() == [0, 6, 7, 8]
    assert sender.skipped_lines == 5
    assert read_offset(offset_path(path)) == path.stat().st_size


def test_sdk21_rows_to_send_keeps_the_good_rows_of_a_batch_arrow_refuses():
    good = [record(0), record(2)]
    batch, steps, skipped = rows_to_send([good[0], {**record(1), "step": 2**70}, good[1]], flight_schema())
    assert steps == [0, 2] and skipped == 1
    assert isinstance(batch, pa.RecordBatch) and batch.num_rows == 2


# --- SDK-26: the offset file's scratch file ----------------------------------------------


def test_offset_writers_in_one_directory_do_not_share_a_scratch_file(tmp_path, monkeypatch):
    target = tmp_path / "run-a.rank0.offset"
    real_replace = flight.os.replace
    scratch: list[str] = []
    other_error: list[BaseException] = []

    def other_writer() -> None:
        try:
            write_offset(target, 99)
        except BaseException as error:  # noqa: BLE001
            other_error.append(error)

    def replace(source, destination):
        scratch.append(Path(source).name)
        if len(scratch) == 1:  # the first writer is between its write and its rename
            other = threading.Thread(target=other_writer)
            other.start()
            other.join()
        return real_replace(source, destination)

    monkeypatch.setattr(flight.os, "replace", replace)
    write_offset(target, 1)  # before the fix: FileNotFoundError, the other writer took its scratch file
    assert other_error == []
    assert len(set(scratch)) == 2
    assert read_offset(target) == 1
    assert not list(tmp_path.glob("*.tmp"))


# --- SDK-14: the heartbeat's client ------------------------------------------------------


def test_heartbeat_closes_each_client_it_drops_after_a_failed_beat(monkeypatch):
    made: list[object] = []
    closed: list[object] = []

    class Down:
        def __init__(self, *args, **kwargs) -> None:
            made.append(self)

        def do_action(self, *args, **kwargs):
            raise pa_flight.FlightUnavailableError("the server is down")

        def close(self) -> None:
            closed.append(self)

    monkeypatch.setattr(pa_flight, "FlightClient", Down)
    beat = Heartbeat("grpc://127.0.0.1:9", "run-a", 0, interval=0.01, stack_seconds=None)
    beat.counter = lambda: 0  # the session hands over its counter: beats start at once
    wait_for(lambda: len(made) >= 5)
    beat.close()
    assert len(made) >= 5
    assert closed == made  # every client was closed, in order, none left to the collector


# --- C24's end-marker half: a marker without its records file ----------------------------


def test_end_marker_without_its_records_file_is_not_closed(tmp_path):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(0)])
    assert write_end_marker(path, run_id="run-a", rank=0, attempt=1)
    assert read_end_marker(path)["state"] == "closed"
    path.unlink()
    assert end_marker_path(path).exists()
    found = read_end_marker(path)
    assert found["state"] == "unreadable"
    assert "cannot be read" in found["detail"]
