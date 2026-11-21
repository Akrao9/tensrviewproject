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


# --- Item 2, L30 (WAVE_AE_CONTRACT.md section 6): the server's two counts, against the real server --


def _put(endpoint: str, run_id: str, batches: list[list[dict]]) -> None:
    """One `DoPut` stream holding `batches`, each with the metadata the contract names; ended by the
    client, so the server stores what it holds before the stream ends."""
    import pyarrow.flight as pa_flight

    from tensorquill.flight import records_to_batch

    client = pa_flight.FlightClient(endpoint)
    try:
        first = records_to_batch(batches[0])
        writer, reader = client.do_put(pa_flight.FlightDescriptor.for_path("runs", run_id, "rank=0"), first.schema,
                                       pa_flight.FlightCallOptions(timeout=60.0))
        end = 0
        for records in batches:
            steps = [r["step"] for r in records]
            start, end = end, end + 100 * len(records)
            metadata = {"schema_version": 1, "run_id": run_id, "rank": 0, "start_byte_offset": start,
                        "end_byte_offset": end, "min_step": min(steps), "max_step": max(steps), "sender_attempt": 1}
            writer.write_with_metadata(records_to_batch(records).cast(first.schema) if records is not batches[0] else first,
                                       pa.py_buffer(json.dumps(metadata).encode()))
        writer.done_writing()
        while reader.read() is not None:
            pass
        writer.close()
    finally:
        client.close()


def _entry(store, run_id: str) -> tuple[dict, dict]:
    status = json.loads((store / "runs" / run_id / "watchdog.json").read_text())
    return (next(r for r in status["ranks"] if r["rank"] == 0),
            next(r for r in status.get("restart") or [] if r["rank"] == 0))


def _counts(entry: dict) -> tuple[Any, Any]:
    return entry.get("rows_not_newer"), entry.get("late_steps_confirmed")


@pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
def test_the_seams_part_gives_12_rows_not_newer_and_12_steps_confirmed_late(tmp_path):
    """The author's first test. The 60-row part of `test_wave_ad_seams.py`, as that test sends it:
    48 stored, and 12 rows not stored, all 12 distinct steps confirmed to have arrived out of order
    inside the part, on the rank's entry and in `restart`. Fails before lane AE1: no key."""
    from test_wave_ad_seams import LATE, RUN as SEAM_RUN, _out_of_order, _send

    store = tmp_path / "store"
    server = Server(store)
    try:
        _send(server.endpoint, _out_of_order())
    finally:
        server.terminate()
    assert sorted(int(r["step"]) for r in stored_records(store, SEAM_RUN)) == [s for s in range(60) if s not in LATE]
    entry, carry = _entry(store, SEAM_RUN)
    assert _counts(entry) == (12, 12) and _counts(carry) == (12, 12), (entry, carry)


@pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
def test_a_stored_part_sent_again_adds_rows_not_newer_and_no_step_confirmed_late(tmp_path):
    """The author's second test: the same 60-row part sent again on a new stream, after the first was
    stored, adds its 60 rows to `rows_not_newer` and nothing to `late_steps_confirmed`."""
    from test_wave_ad_seams import _out_of_order

    store, sent = tmp_path / "store", [dict(r, run_id="ae1-resent") for r in _out_of_order()]
    server = Server(store)
    try:
        _put(server.endpoint, "ae1-resent", [sent])
        _put(server.endpoint, "ae1-resent", [sent])
    finally:
        server.terminate()
    assert len(stored_records(store, "ae1-resent")) == 48
    entry, carry = _entry(store, "ae1-resent")
    assert _counts(entry) == (72, 12) and _counts(carry) == (72, 12), (entry, carry)


@pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
def test_the_same_discarded_step_sent_again_adds_a_row_not_newer_and_no_second_late_step(tmp_path):
    """The author's third test: step 2, discarded as late in the 60-row part, sent again in the same
    open part, and again on a new stream after the part was stored: each adds one to
    `rows_not_newer`, and `late_steps_confirmed` stays at 12."""
    from test_wave_ad_seams import _out_of_order

    run_id = "ae1-same-step"
    sent = [dict(r, run_id=run_id) for r in _out_of_order()]
    again = [r for r in sent if r["step"] == 2]
    store = tmp_path / "store"
    server = Server(store)          # parts every 500 steps or 10 s: both batches of the first stream are one part
    try:
        _put(server.endpoint, run_id, [sent, again])
        _put(server.endpoint, run_id, [again])
    finally:
        server.terminate()
    assert 2 not in {int(r["step"]) for r in stored_records(store, run_id)}
    entry, carry = _entry(store, run_id)
    assert _counts(entry) == (14, 12) and _counts(carry) == (14, 12), (entry, carry)


@pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
def test_a_run_with_nothing_discarded_writes_neither_key(tmp_path):
    """Both counts zero: neither key on the entry or in `restart`, so the status file is as before."""
    store = tmp_path / "store"
    server = Server(store)
    try:
        _put(server.endpoint, "ae1-in-order", [[dict(record(s), run_id="ae1-in-order", attempt=1) for s in range(10)]])
    finally:
        server.terminate()
    entry, carry = _entry(store, "ae1-in-order")
    assert _counts(entry) == (None, None) and _counts(carry) == (None, None), (entry, carry)
