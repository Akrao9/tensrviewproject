"""RS10, the sender's side (`DETECTOR_RULES_CONTRACT.md`, part 8): the production `FlightSender`'s
saved offset when the server refuses a batch for good.

The real server cannot be made to refuse a batch of the real sender, which builds every batch from
one schema. So the refusal is injected by a stand-in server in this process: it stores what it is
sent, and refuses the batch that holds a chosen step, with or without the answer the contract
states. The file is written in three parts so that each is one batch: A, B and C.

The rule under test: the sender moves its saved offset for two reasons only. The server confirmed
it stored the bytes, or the server confirmed, naming exactly the byte range the sender sent, that it
refused them for good. A refusal with no answer, or with an answer that names another range, moves
nothing: the sender keeps its last confirmed offset.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa  # noqa: E402
import pyarrow.flight as pa_flight  # noqa: E402

from tensorquill.flight import FlightSender, offset_path, read_binding  # noqa: E402
from test_flight import append, record, wait_for  # noqa: E402

A, B, C = range(0, 3), range(3, 5), range(5, 8)
REFUSED = "column `step` is Utf8, not Int64; the batch is refused"


class Refusing(pa_flight.FlightServerBase):
    """Stores each batch and answers it with its end offset, but refuses the batch that holds
    step `refuse`. `answer` says what it tells the sender first: `"exact"` names the byte range it
    was sent; `"other"` names another range, `"start"` a range whose start alone is wrong, `"end"` one
    whose end alone is wrong; and `None` says nothing before the status."""

    def __init__(self, refuse: int, answer: str | None) -> None:
        super().__init__("grpc://127.0.0.1:0")
        self.refuse, self.answer = refuse, answer
        self.lock = threading.Lock()
        self.stored: list[int] = []
        self.batches: list[dict] = []   # every batch with rows that arrived: its steps and byte range
        self.held = 0                   # the end of the last bytes stored
        self._thread = threading.Thread(target=self.serve, daemon=True)
        self._thread.start()

    @property
    def endpoint(self) -> str:
        return f"grpc://127.0.0.1:{self.port}"

    def do_put(self, context, descriptor, reader, writer):
        start = None
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
                begin = self.held if start is None else start
                self.batches.append({"steps": steps, "start": begin, "end": end})
                if self.refuse in steps:
                    if self.answer is not None:
                        named = {"exact": (begin, end), "other": (begin + 1, end + 1),
                                 "start": (begin + 1, end), "end": (begin, end + 1)}[self.answer]
                        writer.write(pa.py_buffer(json.dumps({"end_byte_offset": self.held, "refused": {
                            "start_byte_offset": named[0], "end_byte_offset": named[1], "column": "step"}}).encode()))
                    raise pa.ArrowInvalid(REFUSED)
                self.stored += [s for s in steps if s not in self.stored]
                self.held = start = end
            writer.write(pa.py_buffer(json.dumps({"end_byte_offset": end}).encode()))

    def seen(self, steps: range) -> int:
        """How many batches arrived that begin at the first of `steps`. A batch sent again after
        more lines were written holds those lines too, so it is known by where it begins."""
        with self.lock:
            return sum(1 for batch in self.batches if batch["steps"][0] == steps[0])

    def stop(self) -> None:
        self.shutdown()
        self._thread.join(10)


@pytest.fixture
def server(request):
    running: list[Refusing] = []

    def start(answer):
        one = Refusing(B[0], answer)
        running.append(one)
        return one

    yield start
    for one in running:
        one.stop()


def the_run(tmp_path: Path, server: Refusing) -> dict:
    """Write A and wait for it to be confirmed; write B and wait until it has been refused; write C.
    Returns the byte offsets and the sender."""
    path = tmp_path / "run-a.rank0.jsonl"
    sender = FlightSender(path, server.endpoint, "run-a", 0, batch_seconds=0.05, poll_seconds=0.02, backoff_cap=0.2)
    append(path, [record(step) for step in A])
    wait_for(lambda: server.held == path.stat().st_size)
    after_a = path.stat().st_size
    wait_for(lambda: (read_binding(offset_path(path)) or {}).get("offset") == after_a)
    append(path, [record(step) for step in B])
    after_b = path.stat().st_size
    wait_for(lambda: server.seen(B) >= 1)
    append(path, [record(step) for step in C])
    return {"path": path, "sender": sender, "after_a": after_a, "after_b": after_b, "after_c": path.stat().st_size}


@pytest.mark.xfail(strict=True, reason="wave AD, lane AD4: the sender does not read a refusal's answer yet")
def test_a_confirmed_refusal_moves_the_offset_to_the_end_of_the_refused_bytes(tmp_path, server):
    one = server("exact")
    run = the_run(tmp_path, one)
    wait_for(lambda: one.held == run["after_c"], timeout=5.0)
    assert run["sender"].close(timeout=5.0) is True
    assert one.stored == [*A, *C]                       # only A and C are stored
    assert one.seen(B) == 1                             # B was sent once and not again
    assert [b["steps"] for b in one.batches] == [list(A), list(B), list(C)]
    assert one.batches[2]["start"] == run["after_b"]    # C was sent from the end of the refused bytes
    assert read_binding(offset_path(run["path"]))["offset"] == run["after_c"]
    assert run["sender"].refused_batches == 1


@pytest.mark.parametrize("answer", [None, "other", "start", "end"],
                         ids=["no_answer", "an_answer_for_another_range", "only_the_start_is_wrong", "only_the_end_is_wrong"])
def test_a_refusal_that_is_lost_or_names_another_range_keeps_the_last_confirmed_offset(tmp_path, server, answer):
    """Holds today, and must go on holding: nothing but a confirmation that names the sender's own
    byte range moves its offset past bytes the server did not store."""
    one = server(answer)
    run = the_run(tmp_path, one)
    wait_for(lambda: one.seen(B) >= 3)                  # the same bytes go again, at the backoff
    assert read_binding(offset_path(run["path"]))["offset"] == run["after_a"]
    assert one.stored == list(A) and one.seen(C) == 0   # no batch began past the refused bytes
    assert all(b["start"] == run["after_a"] for b in one.batches if b["steps"][0] == B[0])
    assert run["sender"].close(timeout=1.0) is False
    assert read_binding(offset_path(run["path"]))["offset"] == run["after_a"]
