"""Lane AL-1 of wave AL: the sender against a server of an older protocol (`WAVE_AL_CONTRACT.md` section 5).

A protocol 4 server refuses a stream whose batch metadata is `schema_version` 2 with INVALID_ARGUMENT,
"schema_version 2 is not supported; this server reads version 1", and acknowledges nothing (D6,
measured against the protocol 4 binary: `results/live.json`, `old_server_meta_v2`). The sender says
once, on stderr and in `last_error`, that the server speaks an older protocol and to upgrade
`tensorquill-server`, and keeps every line in its file.

The servers here are in-process stand-ins (`pyarrow.flight`), which refuse as the protocol 4 binary
does, with its text; no binary is run.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
flight = pytest.importorskip("pyarrow.flight")

from tensorquill import flight as tq_flight  # noqa: E402
from tensorquill.flight import FlightSender, file_identity, offset_path, write_offset  # noqa: E402

# The protocol 4 binary's own words, as recorded against it (`results/live.json`, `old_server_meta_v2`).
RECORDED = ("ArrowInvalid: Flight returned invalid argument error, with message: schema_version 2 is not supported; "
            "this server reads version 1. gRPC client debug context: INVALID_ARGUMENT:schema_version 2 is not "
            "supported; this server reads version 1. Client context: OK")
SAID = "speaks an older protocol: upgrade tensorquill-server"


class Server(flight.FlightServerBase):
    """Reads the batch metadata versions in `reads`, refusing any other as `ingest.rs` does; acknowledges
    each batch it reads, and keeps its metadata and rows."""

    def __init__(self, reads: tuple[int, ...]) -> None:
        super().__init__("grpc://127.0.0.1:0")
        self.reads = reads
        self.metadata: list[dict] = []
        self.rows: list[dict] = []
        self.refused = 0
        self.lock = threading.Lock()

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
            version = metadata.get("schema_version")
            if version not in self.reads:
                with self.lock:
                    self.refused += 1
                raise pa.ArrowInvalid(f"schema_version {version} is not supported; this server reads version 1")
            with self.lock:
                self.metadata.append(metadata)
                if chunk.data is not None:
                    self.rows.extend(chunk.data.to_pylist())
            writer.write(pa.py_buffer(json.dumps({"end_byte_offset": metadata["end_byte_offset"]}).encode()))


def _lines(path: Path, steps) -> None:
    with path.open("a", encoding="utf-8") as file:
        for step in steps:
            record = {"schema_version": 1, "run_id": "old", "rank": 0, "attempt": 1, "step": step, "ts": 1.0,
                      "grad_norm": 1.0, "layers": {"b.2": {"std": 1.0}, "b.10": {"std": 2.0}}}
            file.write(json.dumps(record, separators=(",", ":")) + "\n")


def _sender(path: Path, endpoint: str) -> FlightSender:
    return FlightSender(path, endpoint, "old", 0, batch_seconds=0.05, poll_seconds=0.02, backoff_base=0.01,
                        backoff_cap=0.05)


def _wait(condition, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def _check_said_once(sender: FlightSender, server: Server, path: Path, capfd) -> None:
    _wait(lambda: server.refused >= 3)  # refused again and again: still said once
    _wait(lambda: isinstance(sender.last_error, tq_flight.OlderServer))
    assert sender.close(timeout=0.3) is False
    text = str(sender.last_error)
    assert f"the server at {server.endpoint} {SAID}" in text
    assert "reads batch schema_version 1" in text and "sends version 2" in text
    err = capfd.readouterr().err
    said = [line for line in err.splitlines() if line.startswith("tensorquill: the server at") and SAID in line]
    assert len(said) == 1, err
    assert "is failing" not in err, err  # the general outage line is not said for it
    assert server.metadata == [] and server.rows == []


def test_a_fresh_stream_to_an_older_server_says_so_once(tmp_path, capfd):
    server = Server(reads=(1,))
    try:
        path = tmp_path / "old.rank0.jsonl"
        _lines(path, range(3))
        sender = _sender(path, server.endpoint)
        _check_said_once(sender, server, path, capfd)
        assert sender.acked_offset == 0 and not offset_path(path).exists()
    finally:
        server.shutdown()


def test_a_resumed_stream_to_an_older_server_says_so_once(tmp_path, capfd):
    """A saved offset makes the stream's first message an empty batch that asks for the store (L3); the
    older server refuses that one, by its metadata version, before any row."""
    server = Server(reads=(1,))
    try:
        path = tmp_path / "old.rank0.jsonl"
        _lines(path, range(2))
        saved = path.stat().st_size
        destination = {"endpoint": server.endpoint, "run_id": "old", "rank": 0}
        write_offset(offset_path(path), saved, {**destination, **file_identity(path, saved)})
        _lines(path, range(2, 4))
        sender = _sender(path, server.endpoint)
        _check_said_once(sender, server, path, capfd)
        assert json.loads(offset_path(path).read_text())["offset"] == saved  # nothing moved
    finally:
        server.shutdown()


def test_a_server_of_protocol_5_is_sent_metadata_version_2_and_typed_rows(tmp_path, capfd):
    server = Server(reads=(1, 2))
    try:
        path = tmp_path / "old.rank0.jsonl"
        _lines(path, range(3))
        sender = _sender(path, server.endpoint)
        assert sender.close(timeout=10) is True
        assert {m["schema_version"] for m in server.metadata} == {2}
        assert [r["step"] for r in server.rows] == [0, 1, 2]
        assert all(r["schema_version"] == 2 and r["record_version"] == 1 for r in server.rows)
        assert [name for name, _ in server.rows[0]["layers"]] == ["b.2", "b.10"]
        assert SAID not in capfd.readouterr().err
    finally:
        server.shutdown()


def test_the_older_servers_refusal_is_told_by_its_text():
    assert tq_flight._older_server_reads(RECORDED) == "1"
    assert tq_flight._older_server_reads("schema_version 1 is not supported; this server reads version 0") is None
    assert tq_flight._older_server_reads("ingest buffer full") is None
