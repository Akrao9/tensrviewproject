"""Lane AR-4 of wave AR, the sender's side: what the production `FlightSender` does when the server
ends its stream because a record batch was malformed (AQ5-R1, AQ5-R2).

Since lane AR-4 the server guards Arrow's decode: a record batch its IPC reader panics on ends the
stream with `INVALID_ARGUMENT` and the message `MALFORMED` below (`server/src/ingest.rs`,
`MALFORMED_BATCH`), and nothing on the acknowledgement channel names the bytes. It is not a refusal
for good (RS10): the server cannot read the batch's rows, so it cannot say they are rows it will never
store, and the same records encoded again may well decode (bytes damaged on the way).

The real sender cannot be made to send a malformed batch (pyarrow encodes every batch it writes), so
the refusal is made by a stand-in server in this process, as in `test_rs10_sender.py`: it stores what
it is sent, and ends the stream with the server's status on the batch that holds a chosen step, for as
long as it is told to. The file is written in three parts so that each is one batch: A, B and C.

What these tests hold, as the sender does it today (no change to `flight.py`):
- the sender keeps its last confirmed offset and sends the same bytes again, at its backoff, for as
  long as the server refuses them; nothing past them is sent; the batch is not counted as refused;
- it says once on stderr that streaming is failing, with the server's message;
- once the server takes the bytes (the damage was on the way), the rest follows and the offset moves.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa  # noqa: E402
import pyarrow.flight as pa_flight  # noqa: E402

from tensorquill.flight import BatchRefused, FlightSender, OlderServer, offset_path, read_binding  # noqa: E402
from test_flight import append, record, wait_for  # noqa: E402

A, B, C = range(0, 3), range(3, 5), range(5, 8)

# `MALFORMED_BATCH` in `server/src/ingest.rs`, the status message of a stream whose batch Arrow's reader
# panicked on, sent as INVALID_ARGUMENT.
MALFORMED = ("a malformed record batch, which the Arrow IPC reader cannot decode (its header names bytes or a "
             "layout its body does not hold); nothing of it is stored and the stream ends here")


class Malformed(pa_flight.FlightServerBase):
    """Stores each batch and answers it with its end offset, but ends the stream with the server's
    malformed-batch status on the batch that holds step `refuse`, while `refusing` is set."""

    def __init__(self, refuse: int) -> None:
        super().__init__("grpc://127.0.0.1:0")
        self.refuse = refuse
        self.refusing = threading.Event()
        self.refusing.set()
        self.lock = threading.Lock()
        self.stored: list[int] = []
        self.batches: list[dict] = []
        self.held = 0
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
            steps = [row["step"] for row in chunk.data.to_pylist()] if chunk.data is not None else []
            end = metadata["end_byte_offset"]
            if not steps:
                writer.write(pa.py_buffer(json.dumps({"end_byte_offset": 0}).encode()))
                continue
            with self.lock:
                self.batches.append({"steps": steps, "end": end})
                if self.refuse in steps and self.refusing.is_set():
                    raise pa.ArrowInvalid(MALFORMED)
                self.stored += [s for s in steps if s not in self.stored]
                self.held = end
            writer.write(pa.py_buffer(json.dumps({"end_byte_offset": end}).encode()))

    def seen(self, steps: range) -> int:
        """How many batches arrived that begin at the first of `steps`."""
        with self.lock:
            return sum(1 for batch in self.batches if batch["steps"][0] == steps[0])

    def stop(self) -> None:
        self.shutdown()
        self._thread.join(10)


@pytest.fixture
def server():
    one = Malformed(B[0])
    yield one
    one.stop()


def test_a_batch_refused_as_malformed_is_sent_again_from_the_last_confirmed_offset(tmp_path: Path, server, capfd):
    path = tmp_path / "run-a.rank0.jsonl"
    sender = FlightSender(path, server.endpoint, "run-a", 0, batch_seconds=0.05, poll_seconds=0.02, backoff_cap=0.2)
    append(path, [record(step) for step in A])
    wait_for(lambda: server.held == path.stat().st_size)
    after_a = path.stat().st_size
    wait_for(lambda: (read_binding(offset_path(path)) or {}).get("offset") == after_a)
    append(path, [record(step) for step in B])
    wait_for(lambda: server.seen(B) >= 1)
    append(path, [record(step) for step in C])
    after_c = path.stat().st_size

    # While the server refuses B, the same bytes go again at the backoff, and nothing past them.
    wait_for(lambda: server.seen(B) >= 3)
    assert read_binding(offset_path(path))["offset"] == after_a
    assert server.stored == list(A) and server.seen(C) == 0
    assert sender.refused_batches == 0 and sender.last_refused is None
    error = sender.last_error
    assert not isinstance(error, (BatchRefused, OlderServer)), repr(error)
    assert "a malformed record batch" in str(error), repr(error)
    said = capfd.readouterr().err
    assert said.count("streaming to") == 1, said
    assert "a malformed record batch" in said, said

    # The damage was on the way: the server takes B now, and C follows.
    server.refusing.clear()
    wait_for(lambda: server.held == after_c, timeout=5.0)
    assert sender.close(timeout=5.0) is True
    assert server.stored == [*A, *B, *C]
    assert read_binding(offset_path(path))["offset"] == after_c
    assert sender.refused_batches == 0
