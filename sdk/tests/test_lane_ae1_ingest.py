"""Lane AE1 of wave AE (Nov 21, 2026): the server's ingest path.

Item 1, RS10's open part. A restarted server holds no acknowledged offset for a rank, so its refusal
named the refused bytes from byte 0 and the returning sender, whose range begins at its saved offset,
held there and sent the batch again at every stream. A batch of rows now states where it begins
(`start_byte_offset` in its metadata), and a server that holds nothing for the rank names the range
from there. The sender's rule is unchanged: its offset moves only for bytes the server stored, or
for a refusal that names exactly the range it sent. The tests here are the pins around that change:
what the sender states, and that a sender without the claim, or a server that does not read it,
keep the safe behaviour of before (the sender holds, nothing is lost, nothing raises). The run itself,
across a restart and a kill of the real server, is in `test_wave_ad_trial.py` section 2.
"""

from __future__ import annotations

import json
import threading
import types
from typing import Any

import pytest

pytest.importorskip("pyarrow.flight")

import pyarrow as pa  # noqa: E402

from tensorquill import flight  # noqa: E402
from tensorquill.flight import FlightSender, offset_path, read_binding  # noqa: E402

from test_flight import append, record, wait_for  # noqa: E402
from test_flight_e2e import BINARY, Server, stored_records  # noqa: E402
from test_rs10_sender import A, B, C, REFUSED, Refusing  # noqa: E402
from test_server_detectors import FAST  # noqa: E402


def _saved(path) -> int | None:
    return (read_binding(offset_path(path)) or {}).get("offset")


def _sender(path, endpoint) -> FlightSender:
    return FlightSender(path, endpoint, "run-a", 0, batch_seconds=0.05, poll_seconds=0.02, backoff_cap=0.2)


class NamesByteZero(Refusing):
    """A stand-in for a server older than the claim, across its own restart: it stores what it is
    sent and refuses the batch that holds step `refuse`, naming the refused range from byte 0 and
    acknowledging 0, whatever the batch states. It keeps every batch's metadata."""

    def __init__(self, refuse: int | None) -> None:
        super().__init__(refuse if refuse is not None else -1, "exact")
        self.metadata: list[dict] = []

    def do_put(self, context, descriptor, reader, writer):
        while True:
            try:
                chunk = reader.read_chunk()
            except StopIteration:
                return
            metadata = json.loads(chunk.app_metadata.to_pybytes())
            steps = [row["step"] for row in chunk.data.to_pylist()] if chunk.data is not None else []
            end = metadata["end_byte_offset"]
            with self.lock:
                self.metadata.append(metadata)
            if not steps:
                writer.write(pa.py_buffer(json.dumps({"end_byte_offset": 0}).encode()))
                continue
            with self.lock:
                self.batches.append({"steps": steps, "start": metadata.get("start_byte_offset"), "end": end})
                if self.refuse in steps:
                    writer.write(pa.py_buffer(json.dumps({"end_byte_offset": 0, "refused": {
                        "start_byte_offset": 0, "end_byte_offset": end, "column": "step"}}).encode()))
                    raise pa.ArrowInvalid(REFUSED)
                self.stored += [s for s in steps if s not in self.stored]
                self.held = end
            writer.write(pa.py_buffer(json.dumps({"end_byte_offset": end}).encode()))


@pytest.fixture
def stand_in():
    running: list[Refusing] = []

    def start(refuse):
        one = NamesByteZero(refuse)
        running.append(one)
        return one

    yield start
    for one in running:
        one.stop()


def test_every_batch_of_rows_states_where_it_begins_and_the_store_query_states_nothing(tmp_path, stand_in):
    """The claim the server reads: each batch of rows states `start_byte_offset`, the end of the
    batch before it on the stream (the saved offset for a stream's first), which is the start a
    refusal of it must name for `_refusal` to confirm it. The store query, which claims no bytes,
    states none. Fails before lane AE1: no batch stated a start."""
    one = stand_in(None)
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(step) for step in A])
    first = _sender(path, one.endpoint)
    wait_for(lambda: one.held == path.stat().st_size)
    after_a = path.stat().st_size
    append(path, [record(step) for step in B])
    wait_for(lambda: one.held == path.stat().st_size)
    assert first.close(timeout=5.0) is True
    append(path, [record(step) for step in C])
    assert _sender(path, one.endpoint).close(timeout=5.0) is True       # returns with a saved offset: asks first
    rows = [m for m in one.metadata if m["max_step"] is not None]
    asked = [m for m in one.metadata if m["max_step"] is None]
    assert [m["start_byte_offset"] for m in rows] == [0, after_a, one.batches[1]["end"]]
    assert [m["end_byte_offset"] for m in rows] == [b["end"] for b in one.batches]
    assert asked and all("start_byte_offset" not in m for m in asked)
    assert one.stored == [*A, *B, *C]


def test_the_new_sender_holds_when_a_server_names_the_refused_bytes_from_byte_0(tmp_path, stand_in):
    """A server that still names byte 0 (one older than the claim, which a restart left holding
    nothing for the rank): the range is not the one the sender sent, so it moves nothing. The sender
    keeps its saved offset before B, counts no refusal, sends B again, raises nothing, and loses
    nothing: once the server can take B, B and C are stored."""
    one = stand_in(B[0])
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(step) for step in A])
    first = _sender(path, one.endpoint)
    wait_for(lambda: one.held == path.stat().st_size)
    after_a = path.stat().st_size
    wait_for(lambda: _saved(path) == after_a)
    assert first.close(timeout=5.0) is True
    append(path, [record(step) for step in B])
    sender = _sender(path, one.endpoint)
    try:
        wait_for(lambda: one.seen(B) >= 3)
        assert sender.refused_batches == 0 and sender.last_refused is None
        assert _saved(path) == after_a
        assert all(b["start"] == after_a for b in one.batches if b["steps"][0] == B[0])   # each told where it began
        assert one.stored == list(A)
        with one.lock:
            one.refuse = -1                                                  # the server can take B now
        append(path, [record(step) for step in C])
        wait_for(lambda: one.held == path.stat().st_size)
    finally:
        sender.close(timeout=5.0)
    assert one.stored == [*A, *B, *C] and _saved(path) == path.stat().st_size


def _without_the_claim(monkeypatch) -> None:
    """Makes the production sender send the metadata of a sender older than lane AE1: every batch
    without `start_byte_offset`. Only the JSON it writes is changed; it reads as before."""
    def dumps(value, *args, **kwargs):
        if isinstance(value, dict) and "start_byte_offset" in value and "run_id" in value:
            value = {k: v for k, v in value.items() if k != "start_byte_offset"}
        return json.dumps(value, *args, **kwargs)

    monkeypatch.setattr(flight, "json", types.SimpleNamespace(dumps=dumps, loads=json.loads))


@pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
@pytest.mark.parametrize("how", ["terminate", "kill"])
def test_a_sender_without_the_claim_across_a_restart_of_the_real_server_holds_and_loses_nothing(tmp_path, monkeypatch, how):
    """A sender older than the claim, against the new server, across a restart (or a kill) of the
    server: the refusal is named from byte 0, as before, so the sender counts nothing and keeps its
    offset before B, sends B again, and raises nothing; the server counts B once; and a sender that
    returns with the real schema stores B and C, so nothing is lost. Today's safe behaviour, pinned."""
    store, path = tmp_path / "store", tmp_path / "run-a.rank0.jsonl"
    good_schema, good_rows = flight.flight_schema, flight.rows_to_send
    text = pa.schema([pa.field("step", pa.string(), nullable=False) if f.name == "step" else f for f in good_schema()])
    named: list[Any] = []
    refusal = FlightSender._refusal

    def text_rows(records, schema, *more):
        batch, steps, *rest = good_rows(records, good_schema(), *more)
        at = batch.schema.get_field_index("step")
        table = pa.Table.from_batches([batch]).set_column(at, text.field("step"), pa.array([str(s) for s in steps], pa.string()))
        return (table.to_batches()[0], steps, *rest)

    def heard(self, refused):
        named.append(refused)
        return refusal(self, refused)

    _without_the_claim(monkeypatch)
    monkeypatch.setattr(FlightSender, "_refusal", heard)
    server = Server(store, 0, *FAST)
    try:
        append(path, [record(step) for step in A])
        assert _sender(path, server.endpoint).close(timeout=10.0) is True
        after_a = _saved(path)
        assert after_a == path.stat().st_size
        getattr(server, how)()
        server = Server(store, 0, *FAST)
        monkeypatch.setattr(flight, "flight_schema", lambda: text)
        monkeypatch.setattr(flight, "rows_to_send", text_rows)
        append(path, [record(step) for step in B])
        second = _sender(path, server.endpoint)
        try:
            wait_for(lambda: len(named) >= 2, timeout=15.0)                 # B goes again
        finally:
            closed = second.close(timeout=2.0)
        assert closed is False                                              # B is not confirmed, and nothing raised
        assert all(r.get("start_byte_offset") == 0 for r in named), named
        assert second.refused_batches == 0 and _saved(path) == after_a
        monkeypatch.setattr(flight, "flight_schema", good_schema)
        monkeypatch.setattr(flight, "rows_to_send", good_rows)
        append(path, [record(step) for step in C])
        assert _sender(path, server.endpoint).close(timeout=10.0) is True
        assert _saved(path) == path.stat().st_size
    finally:
        server.terminate()
    assert sorted(int(r["step"]) for r in stored_records(store, "run-a")) == [*A, *B, *C]
    status = json.loads((store / "runs" / "run-a" / "watchdog.json").read_text())
    assert next(r for r in status["ranks"] if r["rank"] == 0).get("refused_batches") == 1
