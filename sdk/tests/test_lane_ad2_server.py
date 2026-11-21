"""Lane AD2 of wave AD (Nov 21, 2026): the server says why every `DoPut` stream ended.

The v3 plan records streams that ended with `Unknown: h2 protocol error: error reading a body
from connection` and no line in the server's log, in 3 of 22 synthetic scaling runs, and nobody
could say which side ended them. The server now writes one line on stderr per ended stream: the
run, the rank, the gRPC code and message, the cause the transport gave, the batches, rows and bytes
received, the last offset acknowledged on the stream, whether the request had ended first (a
half-close, or a reset the server cannot tell from one: a call whose deadline passed), whether the
response was open as the line was written, and how long the stream was open. Clean ends of a rank are
held to one line per 10 s, and so are its errors, apart; another rank has limits of its own.

Every test here runs the real binary (`TENSORQUILL_SERVER_BIN`, or the newest build under
`server/target`) with its stderr in a file.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

flight = pytest.importorskip("pyarrow.flight")
import pyarrow as pa  # noqa: E402

from tensorquill.flight import flight_schema, records_to_batch  # noqa: E402
from test_flight_e2e import BINARY  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")

LINE = re.compile(r"^tensorquill-server: (the stream of (?P<run>\S+) rank (?P<rank>\d+)|a stream that named no run) "
                  r"(?P<how>ended \(code Ok\)|ended in an error \(code (?P<code>\w+)\): .*); "
                  r"received (?P<batches>\d+) batches, (?P<rows>\d+) rows, (?P<bytes>\d+) bytes; "
                  r"(?P<acked>acknowledged to offset (?P<offset>\d+)|nothing acknowledged); "
                  r"the request (?P<half>had|had not) ended first; the response was (?P<response>open|closed); "
                  r"open (?P<seconds>[\d.]+) s(?P<held>; .*)?$")


class Server:
    """The binary with its stderr in a file, every batch stored and acknowledged as it comes."""

    def __init__(self, tmp_path: Path, *extra: str) -> None:
        self.stderr_path = tmp_path / "stderr.txt"
        env = {k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"}
        self.stderr = self.stderr_path.open("w")
        self.process = subprocess.Popen(
            [str(BINARY), "--listen", "127.0.0.1:0", "--store", str(tmp_path / "store"),
             "--flush-steps", "1", "--no-detect", "--watchdog-seconds", "0", *extra],
            stdout=subprocess.PIPE, stderr=self.stderr, text=True, env=env,
        )
        line = self.process.stdout.readline()
        self.endpoint = f"grpc://127.0.0.1:{int(line.rsplit(':', 1)[1])}"

    def lines(self) -> list[str]:
        return [line for line in self.stderr_path.read_text().splitlines() if " stream" in line]

    def wait_lines(self, count: int, seconds: float = 20.0) -> list[str]:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and len(self.lines()) < count:
            time.sleep(0.05)
        return self.lines()

    def stop(self) -> int:
        self.process.send_signal(signal.SIGTERM)
        code = self.process.wait(20)
        self.process.stdout.close()
        self.stderr.close()
        return code


def rec(run_id: str, rank: int, step: int) -> dict[str, Any]:
    return {"schema_version": 1, "run_id": run_id, "rank": rank, "attempt": 0, "step": step,
            "ts": float(step), "grad_norm": 1.0, "lr": [0.1]}


class Stream:
    def __init__(self, endpoint: str, run_id: str, rank: int = 0) -> None:
        self.run_id, self.rank = run_id, rank
        self.client = flight.FlightClient(endpoint)
        descriptor = flight.FlightDescriptor.for_path("runs", run_id, f"rank={rank}")
        self.writer, self.reader = self.client.do_put(descriptor, flight_schema(), flight.FlightCallOptions(timeout=30.0))

    def send(self, steps: list[int], offset: Any) -> None:
        metadata = {"schema_version": 1, "run_id": self.run_id, "rank": self.rank,
                    "min_step": min(steps), "max_step": max(steps), "end_byte_offset": offset}
        batch = records_to_batch([rec(self.run_id, self.rank, step) for step in steps])
        self.writer.write_with_metadata(batch, pa.py_buffer(json.dumps(metadata).encode()))

    def ack(self) -> int:
        return int(json.loads(self.reader.read().to_pybytes())["end_byte_offset"])

    def finish(self) -> None:
        self.writer.done_writing()
        self.writer.close()
        self.client.close()

    def fail(self) -> str:
        """The error the stream ended in, as the sender reads it."""
        try:
            self.reader.read()
            self.writer.close()
        except (flight.FlightError, pa.ArrowException) as error:
            return type(error).__name__ + ": " + str(error)
        finally:
            self.client.close()
        return "no error"


def parsed(line: str) -> dict[str, str]:
    found = LINE.match(line)
    assert found, line
    return found.groupdict()


def test_a_clean_end_says_what_was_received_and_acknowledged_and_that_the_sender_half_closed(tmp_path):
    server = Server(tmp_path)
    try:
        stream = Stream(server.endpoint, "ad2-clean")
        stream.send([0, 1], 100)
        assert stream.ack() == 100
        stream.send([2], 250)
        assert stream.ack() == 250
        stream.finish()
        lines = server.wait_lines(1)
    finally:
        assert server.stop() == 0
    assert len(lines) == 1, lines
    line = parsed(lines[0])
    assert (line["run"], line["rank"], line["how"]) == ("ad2-clean", "0", "ended (code Ok)"), lines
    assert (line["batches"], line["rows"], line["offset"], line["half"], line["response"]) == ("2", "3", "250", "had", "open"), lines
    assert int(line["bytes"]) > 0


def test_a_refused_batch_ends_its_stream_with_its_code_the_offset_acknowledged_before_and_no_payload(tmp_path):
    server = Server(tmp_path)
    try:
        stream = Stream(server.endpoint, "ad2-refused")
        stream.send([0], 100)
        assert stream.ack() == 100
        stream.send([1], "SECRET-PAYLOAD")
        assert stream.fail() != "no error"
        lines = server.wait_lines(1)
    finally:
        assert server.stop() == 0
    assert len(lines) == 1, lines
    line = parsed(lines[0])
    assert (line["run"], line["code"], line["batches"], line["rows"]) == ("ad2-refused", "InvalidArgument", "2", "2"), lines
    assert (line["offset"], line["half"]) == ("100", "had not"), lines
    assert "SECRET-PAYLOAD" not in server.stderr_path.read_text()
    assert "(cause:" not in lines[0]  # a refused request's words stop at its first ": " (RUST-18)


SENDER = textwrap.dedent("""
    import json, sys, time
    import pyarrow as pa
    import pyarrow.flight as flight
    from tensorquill.flight import flight_schema, records_to_batch
    client = flight.FlightClient(sys.argv[1])
    writer, reader = client.do_put(flight.FlightDescriptor.for_path("runs", "ad2-killed", "rank=0"), flight_schema())
    row = {"schema_version": 1, "run_id": "ad2-killed", "rank": 0, "attempt": 0, "step": 0, "ts": 0.0,
           "grad_norm": 1.0, "lr": [0.1]}
    meta = {"schema_version": 1, "run_id": "ad2-killed", "rank": 0, "min_step": 0, "max_step": 0, "end_byte_offset": 100}
    writer.write_with_metadata(records_to_batch([row]), pa.py_buffer(json.dumps(meta).encode()))
    print(json.loads(reader.read().to_pybytes())["end_byte_offset"], flush=True)
    time.sleep(60)
""")


def test_a_sender_killed_mid_stream_is_said_to_have_ended_it_without_half_closing(tmp_path):
    server = Server(tmp_path)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    sender = subprocess.Popen([sys.executable, "-c", SENDER, server.endpoint], stdout=subprocess.PIPE, text=True, env=env)
    try:
        assert sender.stdout.readline().strip() == "100"
        sender.kill()  # this test's own child, by its PID
        sender.wait(20)
        lines = server.wait_lines(1)
    finally:
        if sender.poll() is None:
            sender.kill()
            sender.wait(20)
        sender.stdout.close()
        assert server.stop() == 0
    assert len(lines) == 1, lines
    line = parsed(lines[0])
    assert line["run"] == "ad2-killed" and line["how"].startswith("ended in an error"), lines
    assert (line["batches"], line["rows"], line["offset"], line["half"]) == ("1", "1", "100", "had not"), lines
    assert "(cause: " in lines[0], lines
    cause = lines[0].split("(cause: ", 1)[1].split("); received", 1)[0]
    assert "code: '" not in cause, lines  # the transport's own words, not the status again
    assert "broken pipe" in cause or "connection reset" in cause, lines


def test_a_call_whose_deadline_passed_is_timed_out_for_the_sender_and_gets_one_line_at_its_offset(tmp_path):
    """Each side is held to what it can establish. The sender: its batch was acknowledged at its
    offset, and the call then reached its deadline (the response ends about when the deadline
    passes, with the writer and reader still held so no half-close was sent, and closing the writer
    afterwards is refused as timed out). The server: exactly one line for the
    stream, for that run, at that offset. `the response was open|closed` is what the server saw as
    it wrote the line: after a reset the transport closes the response on its own task, so either
    is right here, and the line's pattern admits both. A real half-close is the clean-end test's."""
    server = Server(tmp_path)
    try:
        client = flight.FlightClient(server.endpoint)
        writer, reader = client.do_put(flight.FlightDescriptor.for_path("runs", "ad2-deadline", "rank=0"), flight_schema(),
                                       flight.FlightCallOptions(timeout=1.0))
        opened = time.monotonic()
        metadata = {"schema_version": 1, "run_id": "ad2-deadline", "rank": 0, "min_step": 0, "max_step": 0, "end_byte_offset": 7}
        writer.write_with_metadata(records_to_batch([rec("ad2-deadline", 0, 0)]), pa.py_buffer(json.dumps(metadata).encode()))
        assert json.loads(reader.read().to_pybytes())["end_byte_offset"] == 7
        assert reader.read() is None  # nothing more is sent: the response ends when the deadline passes
        waited = time.monotonic() - opened
        assert 0.9 <= waited < 10.0, waited
        server.wait_lines(1, seconds=10.0)  # writer and reader still held: no half-close was sent
        with pytest.raises(flight.FlightTimedOutError):  # and the call says why it ended
            writer.close()
        del writer, reader
        client.close()
    finally:
        assert server.stop() == 0
    lines = server.lines()  # read after the server stopped: every line it wrote
    assert len(lines) == 1, lines
    line = parsed(lines[0])
    assert (line["run"], line["rank"], line["offset"]) == ("ad2-deadline", "0", "7"), lines
    assert line["response"] in ("open", "closed"), lines
    # The request ended with no error, which is how hyper reads the reset (200 of 200 under load).
    assert (line["how"], line["half"]) == ("ended (code Ok)", "had"), lines


def test_each_rank_has_its_own_limit_and_a_ranks_clean_ends_do_not_hold_back_its_error(tmp_path):
    server = Server(tmp_path)
    try:
        for step in range(3):  # three clean ends of rank 0 inside 10 s: one line
            stream = Stream(server.endpoint, "ad2-limit", 0)
            stream.send([step], 100 * (step + 1))
            assert stream.ack() == 100 * (step + 1)
            stream.finish()
        other = Stream(server.endpoint, "ad2-limit", 1)  # rank 1: a line of its own
        other.send([0], 100)
        assert other.ack() == 100
        other.finish()
        refused = Stream(server.endpoint, "ad2-limit", 0)  # rank 0's error: a line of its own
        refused.send([5], "not-a-number")
        refused.fail()
        lines = server.wait_lines(3)
        time.sleep(0.5)  # a fourth line, if one were written, would be there by now
        lines = server.lines()
    finally:
        assert server.stop() == 0
    found = [parsed(line) for line in lines]
    assert [(f["rank"], f["code"]) for f in found] == [("0", None), ("1", None), ("0", "InvalidArgument")], lines


def test_a_stream_that_names_no_run_is_said_to_have_ended_too(tmp_path):
    server = Server(tmp_path)
    try:
        client = flight.FlightClient(server.endpoint)
        writer, reader = client.do_put(flight.FlightDescriptor.for_path("not", "runs"), flight_schema(),
                                       flight.FlightCallOptions(timeout=30.0))
        with pytest.raises((flight.FlightError, pa.ArrowException)):
            writer.write_batch(records_to_batch([rec("x", 0, 0)]))
            reader.read()
            writer.close()
        client.close()
        lines = server.wait_lines(1)
    finally:
        assert server.stop() == 0
    assert len(lines) == 1, lines
    line = parsed(lines[0])
    assert (line["run"], line["code"], line["acked"]) == (None, "InvalidArgument", "nothing acknowledged"), lines


# RS10's transport (`DETECTOR_RULES_CONTRACT.md` part 8). The author's open question: does a pyarrow
# Flight client read the server's last message, the one naming the refused bytes, before the error
# status ends the call? Each pattern below refuses a batch REPEATS times, each on a stream of its own,
# and counts how often the message was read first.

REPEATS = 250


def _refused_batch(run_id: str, steps: list[int]):
    batch = records_to_batch([rec(run_id, 0, step) for step in steps])
    at = batch.schema.get_field_index("step")
    table = pa.Table.from_batches([batch]).set_column(at, pa.field("step", pa.string(), nullable=False),
                                                      pa.array([str(s) for s in steps], pa.string()))
    return table.to_batches()[0]


def _refusal(endpoint: str, run_id: str, end: int, pattern: str) -> tuple[dict | None, str | None]:
    """One refused batch on a stream of its own, read as `pattern`: `half_close` (write, half-close,
    read until the status, as the acceptance run does), `read` (write and read with the request
    still open) or `thread` (a reader thread from the start and the batch written after, as the
    production sender reads). The refusal message, if read before the status, and the status."""
    import threading

    client = flight.FlightClient(endpoint)
    batch = _refused_batch(run_id, [3, 4])
    writer, reader = client.do_put(flight.FlightDescriptor.for_path("runs", run_id, "rank=0"), batch.schema,
                                   flight.FlightCallOptions(timeout=30.0))
    metadata = {"schema_version": 1, "run_id": run_id, "rank": 0, "min_step": 3, "max_step": 4, "end_byte_offset": end}
    answers: list[dict] = []
    status: list[str] = []

    def read() -> None:
        try:
            while (buffer := reader.read()) is not None:
                answers.append(json.loads(buffer.to_pybytes()))
        except (flight.FlightError, pa.ArrowException) as error:
            status.append(str(error))

    try:
        if pattern == "thread":
            thread = threading.Thread(target=read)
            thread.start()
            writer.write_with_metadata(batch, pa.py_buffer(json.dumps(metadata).encode()))
            thread.join(30)
        else:
            writer.write_with_metadata(batch, pa.py_buffer(json.dumps(metadata).encode()))
            if pattern == "half_close":
                writer.done_writing()
            read()
        try:
            writer.close()
        except (flight.FlightError, pa.ArrowException) as error:
            status.append(str(error))
    finally:
        client.close()
    return next((a for a in answers if "refused" in a), None), (status[0] if status else None)


@pytest.mark.parametrize("pattern", ["half_close", "read", "thread"])
def test_a_pyarrow_client_reads_the_refusal_before_the_status_every_time(tmp_path, pattern):
    server = Server(tmp_path)
    run_id = f"ad2-refusal-{pattern}"
    read_first = 0
    wrong: list[Any] = []
    try:
        stream = Stream(server.endpoint, run_id)
        stream.send([0, 1, 2], 300)
        assert stream.ack() == 300
        stream.finish()
        for _ in range(REPEATS):
            refusal, status = _refusal(server.endpoint, run_id, 500, pattern)
            assert status and "`step`" in status and "refused" in status, status
            if refusal is not None:
                read_first += 1
                if refusal["end_byte_offset"] != 300 or {k: refusal["refused"][k] for k in ("start_byte_offset", "end_byte_offset", "column")} != {
                        "start_byte_offset": 300, "end_byte_offset": 500, "column": "step"}:
                    wrong.append(refusal)
    finally:
        assert server.stop() == 0
    print(f"{pattern}: the refusal was read before the status {read_first} of {REPEATS} times")
    assert (read_first, wrong) == (REPEATS, [])
