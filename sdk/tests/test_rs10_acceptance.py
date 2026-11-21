"""RS10's acceptance run (`DETECTOR_RULES_CONTRACT.md`, part 8), against the real server.

One rank sends a batch the server can store (A), a batch whose key column `step` is of another
type (B), and another it can store (C); then it disconnects and comes back. A `DoPut` stream has
one schema for all its batches, so B travels in a stream of its own: within one stream a batch
that is refused for its columns is followed only by batches that would be refused the same way.

The client here is a raw Flight client that plays the sender by the contract's rules and keeps a
byte offset as the sender does. Each batch claims 100 bytes a row: A ends at 300, B at 500, C at
800.

`TODAY` is what the server of `eb5a71b` answers, pinned so that the change is seen. `EXPECTED` is
what the contract states. Lane AD2 builds the server's side and lane AD4 the sender's; the test of
`EXPECTED` is marked as an expected failure until then, strictly, so that it cannot pass unseen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow")

from test_flight_e2e import BINARY, Server, stored_records  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
RUN = "rs10-acceptance"
A, B, C = (0, 1, 2), (3, 4), (5, 6, 7)

# What the server holds after the whole run, and what it told the client on the way.
TODAY = {
    "ack_after_a": 300,
    "refusal_answer": None,          # the stream ends with INVALID_ARGUMENT and no answer first
    "refusal_status": "step",        # the status names the column
    "probe_names_an_offset": False,  # a new stream's first answer names the store, and offset 0
    "restart_before_c": 300,         # so the sender restarts from its own saved offset: B again
    "b_sent_again_refused": True,
    "ack_after_c": None,             # C waits behind B for as long as B is refused
    "restart_after_the_run": 300,
    "stored_steps": [0, 1, 2],
    "refused_batches": 1,            # B sent twice is counted once
    "refused_batch_keys": {"step": 1},
}
EXPECTED = {
    **TODAY,
    # Before it ends the stream, the server says on the acknowledgement channel which bytes it
    # refused for good and why, beside the offset it holds. A sender that reads it does not send
    # those bytes again: it counts them, saves their end as its offset, and goes on from there.
    "refusal_answer": {"end_byte_offset": 300, "refused": {"end_byte_offset": 500, "column": "step"}},
    "restart_before_c": 500,
    "ack_after_c": 800,
    "restart_after_the_run": 800,
    "stored_steps": [0, 1, 2, 5, 6, 7],
}


def _record(step: int) -> dict[str, Any]:
    return {"schema_version": 1, "run_id": RUN, "rank": 0, "attempt": 1, "step": step, "ts": 1.0 + step, "grad_norm": 1.0}


class Client:
    """One sender's offset over several streams."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint, self.offset = endpoint, 0

    def _put(self, batch, end: int, steps: tuple[int, ...] | None):
        import pyarrow as pa
        import pyarrow.flight as flight

        client = flight.FlightClient(self.endpoint)
        descriptor = flight.FlightDescriptor.for_path("runs", RUN, "rank=0")
        writer, reader = client.do_put(descriptor, batch.schema, flight.FlightCallOptions(timeout=30.0))
        metadata = {"schema_version": 1, "run_id": RUN, "rank": 0, "end_byte_offset": end,
                    "min_step": min(steps) if steps else None, "max_step": max(steps) if steps else None,
                    "sender_attempt": 1}
        answers, status = [], None
        try:
            writer.write_with_metadata(batch, pa.py_buffer(json.dumps(metadata).encode()))
            if steps is None:  # the empty first batch that asks what the server holds
                answers.append(json.loads(reader.read().to_pybytes()))
            writer.done_writing()
            while True:
                buffer = reader.read()
                if buffer is None:
                    break
                answers.append(json.loads(buffer.to_pybytes()))
        except Exception as error:  # noqa: BLE001  the status the stream ended with
            status = str(error)
        finally:
            try:
                writer.close()
            except Exception as error:  # noqa: BLE001
                status = status or str(error)
            client.close()
        return answers, status

    def send(self, steps: tuple[int, ...], *, step_as_text: bool = False):
        import pyarrow as pa
        from tensorquill.flight import records_to_batch

        batch = records_to_batch([_record(step) for step in steps])
        if step_as_text:
            at = batch.schema.get_field_index("step")
            table = pa.Table.from_batches([batch]).set_column(at, pa.field("step", pa.string(), nullable=False),
                                                              pa.array([str(s) for s in steps], pa.string()))
            batch = table.to_batches()[0]
        return self._put(batch, self.offset + 100 * len(steps), steps)

    def ask(self) -> int:
        """A new stream's first answer: the offset the server names for it."""
        import pyarrow as pa
        from tensorquill.flight import flight_schema

        answers, status = self._put(pa.RecordBatch.from_pylist([], schema=flight_schema()), 0, None)
        assert answers, f"the server gave no answer to the first batch: {status}"
        return int(answers[0]["end_byte_offset"])


def _refused(store: Path) -> tuple[int, dict]:
    status = json.loads((store / "runs" / RUN / "watchdog.json").read_text())
    entry = next(r for r in status["ranks"] if r["rank"] == 0)
    return entry.get("refused_batches", 0), entry.get("refused_batch_keys", {})


def the_run(tmp_path: Path) -> dict[str, Any]:
    """The client follows the sender's rules: it goes on from its saved offset, which moves when
    the server acknowledges stored rows and when the server says bytes were refused for good."""
    store = tmp_path / "store"
    server = Server(store)
    seen: dict[str, Any] = {}
    try:
        client = Client(server.endpoint)
        answers, status = client.send(A)
        assert status is None, status
        seen["ack_after_a"] = client.offset = int(answers[-1]["end_byte_offset"])

        answers, status = client.send(B, step_as_text=True)
        refusal = next((a for a in answers if "refused" in a), None)
        seen["refusal_answer"] = refusal and {"end_byte_offset": refusal["end_byte_offset"], "refused": {
            key: refusal["refused"][key] for key in ("end_byte_offset", "column")}}
        seen["refusal_status"] = "step" if status and "`step`" in status and "refused" in status else status
        if refusal:
            client.offset = int(refusal["refused"]["end_byte_offset"])

        # The disconnect and the return. The server's first answer on a new stream names no offset
        # of its own, so the sender restarts from the offset it saved.
        seen["probe_names_an_offset"] = client.ask() != 0
        seen["restart_before_c"] = client.offset
        if client.offset == 300:  # not told the refusal is for good: the same bytes go again
            _, status = client.send(B, step_as_text=True)
            seen["b_sent_again_refused"] = bool(status and "refused" in status)
            seen["ack_after_c"] = None
        else:
            seen["b_sent_again_refused"] = TODAY["b_sent_again_refused"]  # sent once more below, to count it
            client.send(B, step_as_text=True)
            answers, status = client.send(C)
            assert status is None, status
            seen["ack_after_c"] = client.offset = int(answers[-1]["end_byte_offset"])
        seen["restart_after_the_run"] = client.offset
    finally:
        server.terminate()
    seen["refused_batches"], seen["refused_batch_keys"] = _refused(store)  # the status as the server left it
    seen["stored_steps"] = sorted(int(r["step"]) for r in stored_records(store, RUN))
    return seen


def test_the_run_as_the_server_answers_today(tmp_path):
    assert the_run(tmp_path) == TODAY


@pytest.mark.xfail(strict=True, reason="wave AD: the server's side is lane AD2's and the sender's lane AD4's")
def test_the_run_as_the_contract_states_it(tmp_path):
    assert the_run(tmp_path) == EXPECTED
