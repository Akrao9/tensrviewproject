"""The Flight sender against an in-process stand-in for `tensorquill-server`.

The stand-in follows the same `DoPut` contract as the Rust server: it drops rows whose step
is not above the last one it kept for `(run_id, rank)` and answers each batch with the
`end_byte_offset` it now holds. `test_flight_e2e.py` runs the real server.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
flight = pytest.importorskip("pyarrow.flight")

import tensorquill  # noqa: E402
from tensorquill.flight import (  # noqa: E402
    FlightSender,
    flight_schema,
    offset_path,
    read_lines,
    read_offset,
    records_to_batch,
    row_to_record,
    write_offset,
)


class Store:
    """What the stand-in server keeps on disk. It outlives a server restart, like the
    object store behind the real server."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir()

    def _read(self, name: str) -> list[dict]:
        path = self.root / name
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    @property
    def rows(self) -> list[dict]:
        return self._read("rows.jsonl")

    @property
    def batches(self) -> list[dict]:
        return self._read("batches.jsonl")

    @property
    def received_rows(self) -> int:
        return sum(batch["rows"] for batch in self.batches)

    def steps(self) -> list[int]:
        return [row["step"] for row in self.rows]


class FakeServer:
    """`flight_fake_server.py` in a subprocess, so stopping it is a real disconnect."""

    def __init__(self, store: Store, port: int = 0) -> None:
        script = Path(__file__).with_name("flight_fake_server.py")
        self.process = subprocess.Popen(
            [sys.executable, str(script), str(store.root), str(port)], stdout=subprocess.PIPE, text=True
        )
        self.port = int(self.process.stdout.readline())
        self.endpoint = f"grpc://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self.process.kill()
        self.process.wait(10)
        self.process.stdout.close()


def record(step: int, run_id: str = "run-a", **more) -> dict:
    return {"schema_version": 1, "run_id": run_id, "rank": 0, "step": step, "ts": 1.5 + step,
            "layers": {}, "grad_norm": None, "grad_norms": {}, "lr": [0.01], **more}


def append(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as file:
        for item in records:
            file.write(json.dumps(item, separators=(",", ":")) + "\n")


def wait_for(condition, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting")
        time.sleep(0.02)


def fast_sender(path: Path, endpoint: str, **options) -> FlightSender:
    options = {"batch_seconds": 0.05, "poll_seconds": 0.02, "backoff_cap": 0.2, **options}
    return FlightSender(path, endpoint, "run-a", 0, **options)


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "server-store")


@pytest.fixture
def server(store):
    server = FakeServer(store)
    yield server
    server.stop()


def test_offset_file_matches_file_size_after_close(tmp_path, store, server):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(step) for step in range(120)])
    sender = fast_sender(path, server.endpoint)
    assert sender.close() is True
    assert store.steps() == list(range(120))
    assert read_offset(offset_path(path)) == path.stat().st_size
    assert offset_path(path).name == "run-a.rank0.offset"
    assert not (tmp_path / "run-a.rank0.offset.tmp").exists()
    # 50 steps per batch, and the metadata the server contract names.
    assert [b["max_step"] for b in store.batches] == [49, 99, 119]
    assert store.batches[0] == {"schema_version": 1, "run_id": "run-a", "rank": 0, "min_step": 0,
                                "max_step": 49, "rows": 50,
                                "end_byte_offset": store.batches[0]["end_byte_offset"]}
    assert store.batches[-1]["end_byte_offset"] == path.stat().st_size


def test_partial_trailing_line_is_not_sent_until_completed(tmp_path, store, server):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(0), record(1)])
    whole = json.dumps(record(2), separators=(",", ":"))
    with path.open("a", encoding="utf-8") as file:
        file.write(whole[:25])  # a writer in the middle of a line
    complete_size = path.stat().st_size - 25
    sender = fast_sender(path, server.endpoint)
    wait_for(lambda: sender.acked_offset == complete_size)
    time.sleep(0.3)
    assert store.steps() == [0, 1]
    assert read_offset(offset_path(path)) == complete_size
    with path.open("a", encoding="utf-8") as file:
        file.write(whole[25:] + "\n")
    wait_for(lambda: store.steps() == [0, 1, 2])
    assert sender.close() is True
    assert read_offset(offset_path(path)) == path.stat().st_size


def test_line_torn_by_a_crash_is_skipped_and_other_runs_lines_are_not_sent(tmp_path, store, server):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(0)])
    with path.open("a", encoding="utf-8") as file:
        file.write('{"schema_version":1,"run_id":"run-a","ra')  # the process died here
    append(path, [record(1), record(7, run_id="someone-else"), record(2)])
    sender = fast_sender(path, server.endpoint)
    assert sender.close() is True
    assert store.steps() == [0, 2]  # the torn line swallowed step 1's line; both are skipped
    assert sender.skipped_lines == 2
    assert read_offset(offset_path(path)) == path.stat().st_size


def test_resumes_from_the_offset_file(tmp_path, store, server):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(step) for step in range(60)])
    assert fast_sender(path, server.endpoint).close() is True
    assert store.received_rows == 60
    append(path, [record(step) for step in range(60, 100)])
    assert fast_sender(path, server.endpoint).close() is True
    # The second sender started at the saved offset: nothing was read or sent twice.
    assert store.received_rows == 100
    assert store.steps() == list(range(100))


def test_lost_offset_resends_and_the_server_drops_the_duplicates(tmp_path, store, server):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(step) for step in range(50)])
    assert fast_sender(path, server.endpoint).close() is True
    # The ack was lost: the offset file still says nothing was stored.
    write_offset(offset_path(path), 0)
    append(path, [record(step) for step in range(50, 60)])
    assert fast_sender(path, server.endpoint).close() is True
    assert store.received_rows == 110  # steps 0-49 really were sent twice
    assert store.steps() == list(range(60))  # and stored once
    assert read_offset(offset_path(path)) == path.stat().st_size


def test_server_down_for_200_steps_delivers_everything_in_order_after_restart(tmp_path, store):
    first = FakeServer(store)
    port, endpoint = first.port, first.endpoint
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(step) for step in range(100)])
    sender = fast_sender(path, endpoint)
    wait_for(lambda: store.steps() == list(range(100)))
    first.stop()

    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for step in range(100, 300):  # 200 steps while the server is away
        append(path, [record(step, scalars={"loss": 1.0 / (step + 1)})])
        time.sleep(0.002)
    time.sleep(0.5)  # several failed attempts and backoff sleeps
    after, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert after - before < 8 * 1024 * 1024
    assert sender.last_error is not None
    assert store.steps() == list(range(100))
    assert read_offset(offset_path(path)) < path.stat().st_size

    second = FakeServer(store, port=port)
    try:
        wait_for(lambda: len(store.rows) == 300)
        assert sender.close() is True
        assert store.steps() == list(range(300))  # in order, none lost, none twice
        assert read_offset(offset_path(path)) == path.stat().st_size
    finally:
        second.stop()


def test_close_with_the_server_down_returns_in_time_and_names_the_local_file(tmp_path, capsys):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(step) for step in range(10)])
    sender = fast_sender(path, "grpc://127.0.0.1:9")  # nothing listens on the discard port
    started = time.monotonic()
    assert sender.close(timeout=1.0) is False
    assert time.monotonic() - started < 3.0
    assert str(path) in capsys.readouterr().err
    assert read_offset(offset_path(path)) == 0


def test_resource_exhausted_backs_off_and_retries(tmp_path, store):
    (store.root / "full").touch()
    full = FakeServer(store)
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(step) for step in range(20)])
    sender = fast_sender(path, full.endpoint)
    wait_for(lambda: sender.last_error is not None)
    assert store.rows == [] and read_offset(offset_path(path)) == 0
    (store.root / "full").unlink()  # the server has room again
    assert sender.close() is True
    assert store.steps() == list(range(20))
    full.stop()


def test_read_lines_never_returns_an_incomplete_tail(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_bytes(b"aa\nbbb\ncc")
    assert read_lines(path, 0, 50, 1024) == ([b"aa", b"bbb"], 7)
    assert read_lines(path, 7, 50, 1024) == ([], 7)
    assert read_lines(path, 0, 1, 1024) == ([b"aa"], 3)
    # A line longer than the byte cap is returned alone, not split.
    path.write_bytes(b"x" * 100 + b"\nyy\n")
    assert read_lines(path, 0, 50, 16) == ([b"x" * 100], 101)
    assert read_lines(tmp_path / "missing.jsonl", 0, 50, 16) == ([], 0)


def test_records_round_trip_through_the_arrow_schema():
    records = [
        record(0),
        record(
            1,
            layers={"blocks.0": {"mean": 0.5, "max_abs": "inf"}},
            grad_norm="nan",
            grad_norms={"w": "nan", "b": 0.25},
            weight_norms={"w": 3.0},
            weights_unchanged=["w"],
            clip_grad_norm=2.5,
            step_skipped=True,
            grad_scale=65536.0,
            scalars={"train/loss": "inf"},
            replica_spread={"norm.weight": [0.0, 1e-9]},
            moe={"router": [3, 0, 5]},
            moe_top_k={"router": 2},
            moe_stats={"router": {"routed_tokens": 8, "entropy": 0.6}},
            probe={"blocks.0": {"hist": [1, 2, 3]}},
            something_new={"added": "later"},
        ),
        {"run_id": "run-a", "rank": 0, "step": 2, "lr": "not a list"},  # an old or odd record
    ]
    batch = records_to_batch(records)
    assert batch.schema.equals(flight_schema())
    assert batch.column("grad_norm").to_pylist()[1] != batch.column("grad_norm").to_pylist()[1]  # NaN
    back = [row_to_record(row) for row in batch.to_pylist()]
    assert back[0] == records[0]
    assert back[1] == records[1]
    assert back[2] == {**records[2], "ts": None, "grad_norm": None}


def test_init_without_an_endpoint_opens_no_sender(tmp_path, toy_moe, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_ENDPOINT", raising=False)
    session = tensorquill.init(toy_moe, run_id="plain", routers={"router": 4})
    assert isinstance(session._sink, tensorquill.JsonlSink)
    session.close()
    assert not list(tmp_path.rglob("*.offset"))


@pytest.mark.parametrize("how", ["argument", "environment"])
def test_init_with_an_endpoint_streams_the_capture_file(tmp_path, toy_moe, monkeypatch, store, server, how):
    import torch

    monkeypatch.chdir(tmp_path)
    if how == "environment":
        monkeypatch.setenv("TENSORQUILL_ENDPOINT", server.endpoint)
        session = tensorquill.init(toy_moe, run_id="streamed", every=5, routers={"router": 4})
    else:
        monkeypatch.delenv("TENSORQUILL_ENDPOINT", raising=False)
        session = tensorquill.init(
            toy_moe, run_id="streamed", every=5, routers={"router": 4}, endpoint=server.endpoint
        )
    optimizer = torch.optim.SGD(toy_moe.parameters(), lr=0.01)
    for _ in range(12):
        optimizer.zero_grad()
        toy_moe(torch.randn(16, 8)).sum().backward()
        session.log(loss=1.0)
        optimizer.step()
    session.close()
    path = tmp_path / "tensorquill-out" / "streamed.rank0.jsonl"
    written = tensorquill.load_records(path)
    assert len(written) == 12
    assert [row_to_record(row) for row in store.rows] == written
    assert read_offset(offset_path(path)) == path.stat().st_size


def test_init_rejects_a_bad_endpoint_and_a_sink_with_no_file(toy_moe, monkeypatch):
    monkeypatch.delenv("TENSORQUILL_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="grpc://"):
        tensorquill.init(toy_moe, endpoint="http://localhost:1", sink=tensorquill.MemorySink())
    with pytest.raises(ValueError, match="JsonlSink"):
        tensorquill.init(toy_moe, endpoint="grpc://localhost:1", sink=tensorquill.MemorySink())


def test_a_failing_stream_is_reported_once_while_training_runs(tmp_path, capsys):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(step) for step in range(10)])
    sender = fast_sender(path, "grpc://127.0.0.1:9")  # nothing listens on the discard port
    deadline = time.monotonic() + 5.0
    while sender.last_error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.3)  # several more failed attempts
    during = capsys.readouterr().err
    assert during.count("is failing") == 1 and "Training continues" in during
    sender.close(timeout=0.5)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_a_forked_child_leaves_the_stream_to_its_parent(tmp_path, store, server):
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(step) for step in range(10)])
    sender = fast_sender(path, server.endpoint)
    append(path, [record(10)])  # not yet sent when the child is forked
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # the child has a copy of the sender but not its thread
        sys.stderr = os.fdopen(write, "w")  # pytest's captured stderr would stay in the child
        started = time.monotonic()
        result = sender.close(timeout=30.0)
        sys.stderr.write(f"|{result}|{time.monotonic() - started < 1.0}")
        sys.stderr.flush()
        os._exit(0)
    os.close(write)
    os.waitpid(pid, 0)
    with os.fdopen(read) as pipe:
        assert pipe.read() == "|False|True"  # at once, and nothing printed about lost bytes
    assert sender.close(timeout=10.0) is True
    assert store.steps() == list(range(11))


def test_close_wakes_backoff_immediately_to_flush(tmp_path, store, monkeypatch):
    # First get an unused loopback port, then let the sender fail and enter a long backoff.
    first = FakeServer(store)
    port, endpoint = first.port, first.endpoint
    first.stop()

    monkeypatch.setattr("tensorquill.flight.random.uniform", lambda _a, b: b)
    path = tmp_path / "run-a.rank0.jsonl"
    append(path, [record(step) for step in range(10)])
    sender = fast_sender(path, endpoint, backoff_base=10.0, backoff_cap=10.0)
    wait_for(lambda: sender.last_error is not None)
    time.sleep(0.05)  # sender is now waiting in its 10s backoff

    restarted = FakeServer(store, port=port)
    try:
        started = time.monotonic()
        assert sender.close(timeout=3.0) is True
        assert time.monotonic() - started < 2.0
    finally:
        restarted.stop()
    assert store.steps() == list(range(10))


def test_tls_ca_is_loaded_at_init_and_ignored_for_plaintext(tmp_path, monkeypatch):
    path = tmp_path / "run-a.rank0.jsonl"
    missing = tmp_path / "missing-ca.pem"
    monkeypatch.setenv("TENSORQUILL_TLS_CA", str(missing))
    with pytest.raises(FileNotFoundError):
        FlightSender(path, "grpcs://127.0.0.1:9", "run-a", 0)
    monkeypatch.delenv("TENSORQUILL_TLS_CA")
    with pytest.raises(FileNotFoundError):
        FlightSender(path, "grpcs://127.0.0.1:9", "run-a", 0, tls_root_certs=missing)
    monkeypatch.setenv("TENSORQUILL_TLS_CA", str(missing))
    plain = FlightSender(path, "grpc://127.0.0.1:9", "run-a", 0)
    assert plain._tls_root_certs is None
    assert plain.close(timeout=0.05) is True  # empty missing file has 0 unacked bytes
