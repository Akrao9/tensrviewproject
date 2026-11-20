"""Lane AA4, the sender (`flight.py`).

L3: the saved offset is bound to the store the server names in its ack
(`{"end_byte_offset": N, "store_id": "..."}`), not to the server's address. Before a stream reuses a
saved offset, its first message is an empty batch, and the server's answer names the store. Most tests
run against `StandIn`, an in-process stand-in whose ack carries a `store_id` or none (an older server).
The acceptance test that the author asked for runs the real `tensorquill-server`. It skips, saying why,
when that binary's ack carries no `store_id` (before lane AA3's server half).

L20: a record whose nested value holds a bare `NaN` or `Infinity` was dropped by the sender. It is
now sent as `JsonlSink` writes the same values, `"nan"`, `"inf"` or `"-inf"`.
"""

from __future__ import annotations

import json
import math
import shutil
import threading
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")
import pyarrow as pa  # noqa: E402
import pyarrow.flight as pa_flight  # noqa: E402

from tensorquill.flight import (  # noqa: E402
    FlightSender,
    file_identity,
    flight_schema,
    offset_path,
    parse_lines,
    read_binding,
    record_to_row,
    row_to_record,
    rows_to_send,
    send_file,
    write_offset,
)
from tensorquill.sink import JsonlSink  # noqa: E402
from test_flight import append, record  # noqa: E402
from test_flight_e2e import BINARY, Server, stored_records  # noqa: E402

needs_server = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built (TENSORQUILL_SERVER_BIN)")

STORE_REPLACED = "the server's store is not the one the offset was saved for"
SAVED_BEFORE_STORE_ID = "the offset was saved before the server named its store"


class Shelf:
    """What a stand-in store holds: kept rows and every batch received, by any server in front of it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.rows: list[dict] = []
        self.batches: list[dict] = []
        self.watermark: dict[tuple, int] = {}

    def steps(self) -> list[int]:
        return [row["step"] for row in self.rows]

    @property
    def received_rows(self) -> int:
        return sum(batch["rows"] for batch in self.batches)


class StandIn(pa_flight.FlightServerBase):
    """The server's `DoPut` contract in process: rows whose step is not above the last one kept are
    dropped, and each batch is answered with its `end_byte_offset`, and with `store_id` when given.
    `refuse_rows` batches with rows are refused first, ending their streams."""

    def __init__(self, shelf: Shelf, store_id: str | None, port: int = 0, refuse_rows: int = 0) -> None:
        super().__init__(f"grpc://127.0.0.1:{port}")
        self.shelf, self.store_id, self.refuse_rows = shelf, store_id, refuse_rows
        self._thread = threading.Thread(target=self.serve, daemon=True)
        self._thread.start()

    @property
    def endpoint(self) -> str:
        return f"grpc://127.0.0.1:{self.port}"

    def do_put(self, context, descriptor, reader, writer):
        while True:
            try:
                chunk = reader.read_chunk()
            except StopIteration:
                return
            metadata = json.loads(chunk.app_metadata.to_pybytes())
            rows = chunk.data.to_pylist() if chunk.data is not None else []
            with self.shelf.lock:
                if rows and self.refuse_rows > 0:
                    self.refuse_rows -= 1
                    raise pa_flight.FlightUnavailableError("refused once, as a stand-in for a failing store")
                self.shelf.batches.append(
                    {"end": metadata["end_byte_offset"], "rows": len(rows), "first": rows[0]["step"] if rows else None})
                for row in rows:
                    key = (row["run_id"], row["rank"], row["attempt"])
                    if row["step"] > self.shelf.watermark.get(key, -1):
                        self.shelf.watermark[key] = row["step"]
                        self.shelf.rows.append(row)
            ack = {"end_byte_offset": metadata["end_byte_offset"]}
            if self.store_id is not None:
                ack["store_id"] = self.store_id
            writer.write(pa.py_buffer(json.dumps(ack).encode()))

    def stop(self) -> None:
        self.shutdown()
        self._thread.join(10)


@pytest.fixture
def servers():
    running: list[StandIn] = []

    def start(shelf: Shelf, store_id: str | None, port: int = 0, **options) -> StandIn:
        server = StandIn(shelf, store_id, port, **options)
        running.append(server)
        return server

    yield start
    for server in running:
        try:
            server.stop()
        except Exception:  # noqa: BLE001  stopped by the test already
            pass


def fast(path: Path, endpoint: str, run_id: str = "run-a", **options) -> FlightSender:
    options = {"batch_seconds": 0.05, "poll_seconds": 0.02, "backoff_cap": 0.2, **options}
    return FlightSender(path, endpoint, run_id, 0, **options)


def write_steps(path: Path, steps: range) -> None:
    append(path, [record(step) for step in steps])


def first_message(shelf: Shelf, since: int) -> dict:
    return shelf.batches[since]


# --- L3: the saved offset bound to the store ---------------------------------------------


def test_l3_store_replaced_behind_the_same_address_is_seen_before_the_offset_is_used(tmp_path, servers, capfd):
    path = tmp_path / "run-a.rank0.jsonl"
    write_steps(path, range(10))
    old = Shelf()
    first = servers(old, "store-A")
    assert fast(path, first.endpoint).close() is True
    saved = read_binding(offset_path(path))
    assert saved["offset"] == path.stat().st_size
    write_steps(path, range(10, 20))  # steps past the saved offset, written before the store is replaced
    port = first.port
    first.stop()
    new = Shelf()  # the same address, another store behind it
    second = servers(new, "store-B", port)
    assert second.endpoint == first.endpoint
    capfd.readouterr()
    sender = fast(path, second.endpoint)
    assert sender.close() is True
    # Had the offset been used first, steps 10 to 19 would have been stored first and the replay's
    # steps 0 to 9 dropped by the step filter: every step stored shows the change was seen first.
    assert new.steps() == list(range(20))  # before the fix: [10, ..., 19]
    assert [b for b in new.batches if b["rows"]][0]["first"] == 0
    assert not any(b["end"] > saved["offset"] and b["first"] == 10 for b in new.batches[:2])
    assert new.batches[0] == {"end": 0, "rows": 0, "first": None}  # the empty first batch asked first
    assert sender.replayed == STORE_REPLACED
    assert saved["store_id"] == "store-A"
    err = capfd.readouterr().err
    assert err.count(STORE_REPLACED) == 1 and "store-A" in err and "store-B" in err
    assert read_binding(offset_path(path))["store_id"] == "store-B"


def test_l3_the_same_store_at_another_address_resumes_at_its_offset(tmp_path, servers):
    path = tmp_path / "run-a.rank0.jsonl"
    write_steps(path, range(10))
    shelf = Shelf()  # one store, reached at two addresses
    one, two = servers(shelf, "store-S"), servers(shelf, "store-S")
    assert fast(path, one.endpoint).close() is True
    write_steps(path, range(10, 15))
    sender = fast(path, two.endpoint)
    assert sender.close() is True
    assert sender.replayed is None
    assert shelf.received_rows == 15  # before the fix: 25, the whole file sent again
    assert shelf.steps() == list(range(15))
    binding = read_binding(offset_path(path))
    assert binding["endpoint"] == two.endpoint and binding["store_id"] == "store-S"


def test_l3_offset_saved_without_a_store_id_is_sent_again_once_and_said_once(tmp_path, servers, capfd):
    path = tmp_path / "run-a.rank0.jsonl"
    write_steps(path, range(8))
    shelf = Shelf()
    older = servers(shelf, None)  # a server built before the store ID
    assert fast(path, older.endpoint).close() is True
    assert "store_id" not in read_binding(offset_path(path))
    port = older.port
    older.stop()
    # The same store behind the same address, now naming itself. Its first batch with rows is refused,
    # so the replay is cut off once and the sender opens the stream again.
    newer = servers(shelf, "store-N", port, refuse_rows=1)
    capfd.readouterr()
    sender = fast(path, newer.endpoint)
    assert sender.close() is True
    assert sender.replayed == SAVED_BEFORE_STORE_ID
    assert shelf.received_rows == 16  # before the fix: 8, the offset taken from the address
    assert shelf.steps() == list(range(8))
    assert capfd.readouterr().err.count(SAVED_BEFORE_STORE_ID) == 1  # said once, across two streams
    assert read_binding(offset_path(path))["store_id"] == "store-N"
    # Once bound to the store, the next sender resumes.
    write_steps(path, range(8, 10))
    again = fast(path, newer.endpoint)
    assert again.close() is True and again.replayed is None
    assert shelf.received_rows == 18 and shelf.steps() == list(range(10))
    assert capfd.readouterr().err.count(SAVED_BEFORE_STORE_ID) == 0


def test_l3_an_older_server_is_asked_first_and_stays_bound_to_its_address(tmp_path, servers):
    path = tmp_path / "run-a.rank0.jsonl"
    write_steps(path, range(6))
    shelf = Shelf()
    one, two = servers(shelf, None), servers(shelf, None)
    assert fast(path, one.endpoint).close() is True
    write_steps(path, range(6, 9))
    asked = len(shelf.batches)
    sender = fast(path, one.endpoint)
    assert sender.close() is True and sender.replayed is None
    assert first_message(shelf, asked) == {"end": 0, "rows": 0, "first": None}  # before the fix: rows first
    assert shelf.received_rows == 9
    # Another address of a server that names no store: the offset is not used there, as before.
    other = fast(path, two.endpoint)
    assert other.close() is True
    assert other.replayed == "the offset file names another server, run or rank"
    assert shelf.received_rows == 18 and shelf.steps() == list(range(9))
    assert "store_id" not in read_binding(offset_path(path))


def test_l3_a_saved_store_id_met_by_an_ack_without_one_is_bound_to_the_address(tmp_path, servers):
    # Not pinned by the author: a server that no longer names its store (an older binary put back).
    # The rule for an ack without `store_id` is taken literally: bound to the address, as before.
    path = tmp_path / "run-a.rank0.jsonl"
    write_steps(path, range(5))
    shelf = Shelf()
    server = servers(shelf, "store-A")
    assert fast(path, server.endpoint).close() is True
    port = server.port
    server.stop()
    older = servers(shelf, None, port)
    write_steps(path, range(5, 7))
    asked = len(shelf.batches)
    sender = fast(path, older.endpoint)
    assert sender.close() is True and sender.replayed is None
    assert first_message(shelf, asked) == {"end": 0, "rows": 0, "first": None}
    assert shelf.received_rows == 7 and shelf.steps() == list(range(7))


def test_l3_a_saved_offset_the_server_never_answered_for_is_not_counted_as_confirmed(tmp_path, servers):
    path = tmp_path / "run-a.rank0.jsonl"
    write_steps(path, range(4))
    server = servers(Shelf(), "store-A")
    assert send_file(path, server.endpoint, "run-a", timeout=10) is True
    endpoint = server.endpoint
    server.stop()  # nothing listens at the address now
    sender = fast(path, endpoint)
    assert sender.close(timeout=1.0) is False  # before the fix: True, from the offset file alone
    assert sender.acked_offset == 0 and sender.unacked_bytes() == path.stat().st_size
    assert read_binding(offset_path(path))["store_id"] == "store-A"  # the offset file is left as it was


def test_l3_an_ack_naming_another_store_on_the_same_stream_is_not_taken(tmp_path, monkeypatch):
    path = tmp_path / "run-a.rank0.jsonl"
    write_steps(path, range(3))
    with monkeypatch.context() as patched:
        patched.setattr(FlightSender, "_run", lambda self: None)
        sender = FlightSender(path, "grpc://127.0.0.1:9", "run-a", 0)
    assert sender._resume() == 0
    data = path.read_bytes()
    sender._sent(data, len(data))
    sender._ack(0, "store-A")
    with pytest.raises(ConnectionError):
        sender._ack(len(data), "store-B")
    assert sender.acked_offset == 0 and not offset_path(path).exists()
    sender._ack(len(data), "store-A")
    assert read_binding(offset_path(path)) == {
        "version": 2, "offset": len(data), "endpoint": "grpc://127.0.0.1:9", "run_id": "run-a", "rank": 0,
        **file_identity(path, len(data)), "store_id": "store-A"}


def test_l3_a_store_id_that_is_not_a_string_is_no_store_id(tmp_path, servers):
    path = tmp_path / "run-a.rank0.jsonl"
    write_steps(path, range(3))
    shelf = Shelf()
    server = servers(shelf, "")  # an empty name is not a name
    assert fast(path, server.endpoint).close() is True
    binding = read_binding(offset_path(path))
    assert "store_id" not in binding
    write_offset(offset_path(path), binding["offset"], {**{k: v for k, v in binding.items()
                                                           if k not in ("version", "offset")}, "store_id": 7})
    write_steps(path, range(3, 5))
    named = servers(shelf, "store-A")
    sender = fast(path, named.endpoint)
    assert sender.close() is True
    assert sender.replayed == "the offset file names another server, run or rank"  # 7 is no store ID
    assert read_binding(offset_path(path))["store_id"] == "store-A"


def store_named_by(endpoint: str) -> str | None:
    """The `store_id` the server at `endpoint` puts in its ack, asked without the sender: one empty
    batch of a run of its own, and the first ack read. None when the ack names no store."""
    client = pa_flight.FlightClient(endpoint)
    try:
        descriptor = pa_flight.FlightDescriptor.for_path("runs", "ask-store-id", "rank=0")
        writer, reader = client.do_put(descriptor, flight_schema(), pa_flight.FlightCallOptions(timeout=10.0))
        metadata = {"schema_version": 1, "run_id": "ask-store-id", "rank": 0, "end_byte_offset": 0}
        writer.write_with_metadata(pa.RecordBatch.from_pylist([], schema=flight_schema()),
                                   pa.py_buffer(json.dumps(metadata).encode()))
        ack = json.loads(reader.read().to_pybytes())
        writer.done_writing()
        writer.close()
    finally:
        client.close()
    store_id = ack.get("store_id")
    return store_id if isinstance(store_id, str) and store_id else None


@needs_server
def test_l3_real_server_store_replaced_behind_the_same_address_is_seen_before_the_offset_is_used(tmp_path):
    """The author's acceptance test: keep the address, replace the store behind it, and show the
    sender sees the change before it uses its offset."""
    path = tmp_path / "real.rank0.jsonl"
    append(path, [record(step, run_id="real") for step in range(12)])
    root = tmp_path / "store"
    server = Server(root)
    port, endpoint = server.port, server.endpoint
    try:
        first_store = store_named_by(endpoint)  # decided by the server's ack alone, not by the sender
        if first_store is None:
            pytest.skip("this tensorquill-server's ack carries no store_id (lane AA3's store ID is not in this "
                        "binary), so there is no store to bind the offset to; the sender stays bound to the address")
        assert send_file(path, endpoint, "real", timeout=30.0) is True
    finally:
        assert server.terminate() == 0
    saved = read_binding(offset_path(path))
    append(path, [record(step, run_id="real") for step in range(12, 20)])
    shutil.rmtree(root)  # the store wiped behind the same address
    server = Server(root, port)
    try:
        assert server.endpoint == endpoint
        sender = FlightSender(path, endpoint, "real", 0, batch_seconds=0.05, poll_seconds=0.02)
        assert sender.close(30.0) is True
    finally:
        assert server.terminate() == 0
    # Steps 0 to 11 are stored only if the sender started at byte 0 before it sent steps 12 to 19.
    assert sorted(r["step"] for r in stored_records(root, "real")) == list(range(20))
    assert sender.replayed == STORE_REPLACED
    assert saved.get("store_id") == first_store
    assert read_binding(offset_path(path))["store_id"] not in (None, first_store)


@needs_server
def test_l3_real_server_answers_the_empty_first_batch_and_the_offset_resumes_on_the_same_store(tmp_path, monkeypatch):
    # The server must answer the empty first batch at once, with its store ID or none: the sender
    # waits for that answer before it uses its offset. Measured on `2e2f4dc`'s binary: about 2 ms.
    answers: list = []
    bound = FlightSender._bound
    monkeypatch.setattr(FlightSender, "_bound", lambda self, store_id: answers.append(store_id) or bound(self, store_id))
    path = tmp_path / "same.rank0.jsonl"
    append(path, [record(step, run_id="same") for step in range(10)])
    root = tmp_path / "store"
    server = Server(root)
    try:
        assert send_file(path, server.endpoint, "same", timeout=30.0) is True
        assert answers == []  # nothing saved yet: no question to ask
        append(path, [record(step, run_id="same") for step in range(10, 15)])
        sender = FlightSender(path, server.endpoint, "same", 0, batch_seconds=0.05, poll_seconds=0.02)
        assert sender.close(30.0) is True
    finally:
        assert server.terminate() == 0
    assert len(answers) == 1 and (answers[0] is None or isinstance(answers[0], str))
    assert sender.replayed is None
    assert sorted(r["step"] for r in stored_records(root, "same")) == list(range(15))


# --- L20: a NaN inside a nested value -----------------------------------------------------

NESTED = {
    "layers": {"model.layers.0": {"mean": math.nan, "std": math.inf}},
    "probe": {"head": {"max_logit": [math.nan, 1.0, -math.inf]}},
    "replica": {"params": {"w": {"fp": [3.5, math.inf, math.nan]}}},  # an unknown key: `extra_json`
    "system": {"gpu": [{"temp": math.nan}]},
}


def bare_and_sink_lines(tmp_path: Path) -> tuple[bytes, bytes]:
    """The same record as `JsonlSink` writes it and with bare `NaN` / `Infinity` tokens."""
    sink = JsonlSink(tmp_path / "sink.rank0.jsonl")
    sink.write([{**record(0), **NESTED}])
    sink.close()
    written = (tmp_path / "sink.rank0.jsonl").read_bytes().rstrip(b"\n")
    bare = json.dumps({**record(0), **NESTED}, separators=(",", ":")).encode()
    assert b"NaN" in bare and b"Infinity" in bare and b"NaN" not in written
    return bare, written


def test_l20_a_bare_nan_inside_a_nested_value_is_sent_as_the_sink_writes_it(tmp_path):
    bare, written = bare_and_sink_lines(tmp_path)
    records, skipped = parse_lines([bare, written], "run-a")
    assert skipped == 0 and len(records) == 2
    # Before the fix: `record_to_row` raised ValueError on the bare line, and `rows_to_send` dropped it.
    assert record_to_row(records[0]) == record_to_row(records[1])
    batch, steps, unstorable = rows_to_send(records, flight_schema())
    assert unstorable == 0 and steps == [0, 0]
    rows = batch.to_pylist()
    assert json.loads(rows[0]["probe_json"]) == {"head": {"max_logit": ["nan", 1.0, "-inf"]}}
    assert json.loads(rows[0]["extra_json"])["replica"]["params"]["w"]["fp"] == [3.5, "inf", "nan"]
    assert row_to_record(rows[0]) == json.loads(written)  # read back as the sink's own line reads


def test_l20_a_probe_with_a_nan_reaches_the_server(tmp_path, servers):
    path = tmp_path / "run-a.rank0.jsonl"
    bare, written = bare_and_sink_lines(tmp_path)
    with path.open("ab") as file:
        file.write(written + b"\n")
        file.write(bare.replace(b'"step":0', b'"step":1') + b"\n")
    shelf = Shelf()
    server = servers(shelf, None)
    sender = fast(path, server.endpoint)
    assert sender.close() is True
    assert shelf.steps() == [0, 1]  # before the fix: [0]
    assert sender.skipped_lines == 0
    assert json.loads(shelf.rows[1]["probe_json"]) == json.loads(shelf.rows[0]["probe_json"])


@needs_server
def test_l20_real_server_stores_a_probe_whose_nested_value_is_nan(tmp_path):
    path = tmp_path / "nan.rank0.jsonl"
    probe = {"head": {"max_logit": [math.nan, 2.0]}}
    with path.open("w", encoding="utf-8") as file:
        file.write(json.dumps({**record(0, run_id="nan"), "probe": probe}) + "\n")
        file.write(json.dumps(record(1, run_id="nan")) + "\n")
    server = Server(tmp_path / "store")
    try:
        assert send_file(path, server.endpoint, "nan", timeout=30.0) is True
    finally:
        assert server.terminate() == 0
    stored = stored_records(tmp_path / "store", "nan")
    assert [r["step"] for r in stored] == [0, 1]  # before the fix: [1]
    assert stored[0]["probe"] == {"head": {"max_logit": ["nan", 2.0]}}
