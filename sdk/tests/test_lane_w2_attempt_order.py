"""Lane W2: contract 1 (which attempt owns a rank, RS05) and contract 3 (capture stopped, C03's
server half), end to end. The contracts are the signed-off notes `contract-1-attempt-ownership.md`
and `contract-3-capture-stopped.md` (session notes, wave V/W).

The RS05 fixture numbers its attempts going DOWN, A = 900 then B = 2, so it fails under the
numeric order every reader used before contract 1. Each reader is asserted against what it can
know: the server (which saw B's heartbeat, and keeps the order in `attempts.json` over a restart)
and a reader of the local JSONL file alone (which never saw that heartbeat).

The server tests drive the real `tensorquill-server` with raw Flight calls, so every step is
deterministic. They need the binary (`TENSORQUILL_SERVER_BIN`, or `server/target`) and pyarrow,
and are skipped without them. Waits are on a condition in the status file, bounded, never a sleep.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import pytest

from tensorquill.evidence import resolve_attempts

RUN = "w2-attempts"
A, B = 900, 2  # A starts first; B is the replacement, with a smaller number (a clock behind)


def _server_binary() -> Path | None:
    named = os.environ.get("TENSORQUILL_SERVER_BIN")
    if named:
        return Path(named)
    target = Path(__file__).resolve().parents[2] / "server" / "target"
    built = [p for p in (target / "debug" / "tensorquill-server", target / "release" / "tensorquill-server") if p.exists()]
    return max(built, key=lambda p: p.stat().st_mtime) if built else None


BINARY = _server_binary()
needs_server = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")


def rec(attempt: int, step: int, run_id: str = RUN) -> dict[str, Any]:
    return {"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": attempt, "step": step,
            "ts": float(step), "grad_norm": 1.0, "lr": [0.1]}


def steps_of(records: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [(r["attempt"], r["step"]) for r in records]


# ---------------------------------------------------------------------------------------------
# The server, driven with raw Flight calls


class Server:
    FLAGS = ("--silent-seconds", "600", "--watchdog-seconds", "0.1", "--flush-seconds", "0.1")

    def __init__(self, store: Path, *flags: str) -> None:
        env = {k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"}
        self.started_ms = time.time() * 1000
        self.process = subprocess.Popen(
            [str(BINARY), "--listen", "127.0.0.1:0", "--store", str(store), *(flags or self.FLAGS)],
            stdout=subprocess.PIPE, text=True, env=env,
        )
        line = self.process.stdout.readline()  # "tensorquill-server listening on 127.0.0.1:PORT"
        self.endpoint = f"grpc://127.0.0.1:{int(line.rsplit(':', 1)[1])}"

    def kill(self) -> None:
        """SIGKILL: what was only in the server's memory is gone, and nothing more is written."""
        self.process.kill()
        self.process.wait(20)
        self.process.stdout.close()

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(20)
        self.process.stdout.close()


def beat(endpoint: str, attempt: int | None, *, run_id: str = RUN, phase: str = "train",
         interval: float = 0.2, work: int | None = None, **extra: Any) -> None:
    """One heartbeat; raises what the server answers (a refusal is a `pyarrow.flight` error)."""
    flight = pytest.importorskip("pyarrow.flight")
    payload: dict[str, Any] = {"run_id": run_id, "rank": 0, "phase": phase, "interval_seconds": interval}
    if attempt is not None:
        payload["attempt"] = attempt
    if work is not None:
        payload["work"] = work
    payload.update(extra)
    client = flight.FlightClient(endpoint)
    try:
        for _ in client.do_action(flight.Action("heartbeat", json.dumps(payload).encode()),
                                  flight.FlightCallOptions(timeout=10.0)):
            pass
    finally:
        client.close()


class Stream:
    """One `DoPut` stream of rank 0, as a sender of `sender_attempt` opens it. `send` waits for
    the server's acknowledgement, which comes only once the rows are stored."""

    def __init__(self, endpoint: str, sender_attempt: int | None, run_id: str = RUN) -> None:
        flight = pytest.importorskip("pyarrow.flight")
        from tensorquill.flight import flight_schema

        self.run_id, self.sender_attempt, self.offset = run_id, sender_attempt, 0
        self.client = flight.FlightClient(endpoint)
        descriptor = flight.FlightDescriptor.for_path("runs", run_id, "rank=0")
        self.writer, self.reader = self.client.do_put(descriptor, flight_schema(),
                                                      flight.FlightCallOptions(timeout=60.0))

    def send(self, records: list[dict[str, Any]]) -> int:
        import pyarrow as pa
        from tensorquill.flight import records_to_batch

        self.offset += 100 * len(records)
        metadata: dict[str, Any] = {
            "schema_version": 1, "run_id": self.run_id, "rank": 0,
            "min_step": min(r["step"] for r in records), "max_step": max(r["step"] for r in records),
            "end_byte_offset": self.offset,
        }
        if self.sender_attempt is not None:
            metadata["sender_attempt"] = self.sender_attempt
        self.writer.write_with_metadata(records_to_batch(records), pa.py_buffer(json.dumps(metadata).encode()))
        ack = self.reader.read()
        return int(json.loads(ack.to_pybytes())["end_byte_offset"])

    def close(self) -> None:
        try:
            self.writer.done_writing()
            self.writer.close()
        except Exception:  # noqa: BLE001  a stream the server dropped (it was killed) closes as it can
            pass
        self.client.close()


def run_dir(store: Path, run_id: str = RUN) -> Path:
    return store / "runs" / run_id


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def wait_status(store: Path, ready: Callable[[dict[str, Any]], bool], seconds: float = 20.0,
                run_id: str = RUN) -> dict[str, Any]:
    """The first `watchdog.json` that `ready` accepts, or the last one read when none did within
    `seconds`: the caller's asserts then say what the server showed instead."""
    path, last, deadline = run_dir(store, run_id) / "watchdog.json", None, time.monotonic() + seconds
    while time.monotonic() < deadline:
        status = read_json(path)
        if status is not None:
            last = status
            try:
                if ready(status):
                    return status
            except (KeyError, IndexError, TypeError, StopIteration):
                pass
        time.sleep(0.05)
    assert last is not None, f"no watchdog.json under {path.parent}"
    return last


def rank0(status: dict[str, Any]) -> dict[str, Any]:
    return next(r for r in status["ranks"] if r["rank"] == 0)


def wait_report(store: Path, ready: Callable[[dict[str, Any]], bool], seconds: float = 20.0) -> dict[str, Any]:
    path, last, deadline = run_dir(store) / "rank=0" / "incidents.json", None, time.monotonic() + seconds
    while time.monotonic() < deadline:
        report = read_json(path)
        if report is not None:
            last = report
            if ready(report):
                return report
        time.sleep(0.05)
    assert last is not None, f"no incidents.json at {path}"
    return last


def attempt_order(store: Path) -> list[int] | None:
    document = read_json(run_dir(store) / "rank=0" / "attempts.json")
    return None if document is None else [entry["attempt"] for entry in document["attempts"]]


@pytest.fixture
def servers(tmp_path):
    started: list[Server] = []

    def start(*flags: str) -> Server:
        server = Server(tmp_path / "store", *flags)
        started.append(server)
        return server

    yield start
    for server in started:
        if server.process.poll() is None:
            server.stop()


# ---------------------------------------------------------------------------------------------
# Contract 1, the fixture: the server's column


@needs_server
def test_rs05_fixture_the_server_keeps_the_claimed_owner_over_a_restart_and_a_late_step(tmp_path, servers):
    pytest.importorskip("pyarrow.flight")
    store = tmp_path / "store"
    server = servers()

    # 1. A reaches step 100.
    beat(server.endpoint, A)
    a_stream = Stream(server.endpoint, A)
    a_stream.send([rec(A, s) for s in range(0, 101)])
    status = wait_status(store, lambda s: rank0(s)["last_step"] == 100)
    assert (rank0(status)["attempt"], rank0(status)["last_step"]) == (A, 100)
    before_b = (run_dir(store) / "watchdog.json").read_bytes()

    # 2. B claims the rank by heartbeat only, then the server dies before B's first row. Its last
    #    watchdog.json is from before B's beat: only attempts.json can know that B owns the rank.
    beat(server.endpoint, B)
    server.kill()
    a_stream.close()
    (run_dir(store) / "watchdog.json").write_bytes(before_b)
    server = servers()
    status = wait_status(store, lambda s: s["written_at_ms"] >= server.started_ms)
    entry = rank0(status)
    assert (entry["attempt"], entry["last_step"], entry.get("attempt_without_steps")) == (B, None, True), entry
    assert entry["severity"] is None, entry
    assert attempt_order(store) == [A, B]

    # 3. A's late step 101, on A's own stream: stored, and not progress.
    a_late = Stream(server.endpoint, A)
    a_late.send([rec(A, 101)])
    status = wait_status(store, lambda s: rank0(s)["attempt"] != B or rank0(s).get("superseded_rows")
                         or rank0(s)["last_step"] is not None)
    entry = rank0(status)
    assert (entry["attempt"], entry["last_step"], entry.get("attempt_without_steps")) == (B, None, True), entry
    assert entry.get("superseded_rows") == 1, entry
    assert any((run_dir(store) / "rank=0" / f"attempt={A}").glob("part-*00000101.parquet")), \
        "A's late step is stored under its own attempt"

    # 4. B writes step 30.
    b_stream = Stream(server.endpoint, B)
    b_stream.send([rec(B, 30)])
    status = wait_status(store, lambda s: rank0(s)["last_step"] == 30)
    entry = rank0(status)
    assert (entry["attempt"], entry["last_step"], entry["state"]) == (B, 30, "active"), entry
    assert "attempt_without_steps" not in entry
    report = wait_report(store, lambda r: r["attempt"] == B and r["last_step"] == 30)
    # A's 0..29 and B's 30 count; A's 100 and 101 are rolled back by B, which started at 30.
    assert (report["attempt"], report["last_step"], report["steps"]) == (B, 30, 31), report
    assert report.get("attempt_order") == "first_seen", report
    # The report covers the newest part of the owner: the rank is judged, not `behind`.
    status = wait_status(store, lambda s: all(r.get("code") != "behind" for r in s["run_state"]["reasons"]))
    assert all(r.get("code") != "behind" for r in status["run_state"]["reasons"]), status["run_state"]
    for stream in (a_late, b_stream):
        stream.close()


@needs_server
def test_rs05_a_backlog_of_a_never_seen_attempt_sent_by_the_new_process_is_older(tmp_path, servers):
    """The server never saw A (it was down). B's process claims the rank, then uploads A's unsent
    rows from the rank's file, then its own. A is history, not the owner."""
    pytest.importorskip("pyarrow.flight")
    store = tmp_path / "store"
    server = servers()
    beat(server.endpoint, B)
    stream = Stream(server.endpoint, B)
    stream.send([rec(A, s) for s in range(0, 51)])
    stream.send([rec(B, s) for s in range(30, 41)])
    status = wait_status(store, lambda s: rank0(s)["attempt"] == B and rank0(s)["last_step"] == 40)
    entry = rank0(status)
    assert (entry["attempt"], entry["last_step"]) == (B, 40), entry
    report = wait_report(store, lambda r: r["attempt"] == B and r["last_step"] == 40)
    assert (report["attempt"], report["last_step"], report["steps"]) == (B, 40, 30 + 11), report
    assert attempt_order(store) == [A, B]
    stream.close()


@needs_server
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes into a read-only folder")
def test_rs05_a_claim_whose_order_cannot_be_stored_is_refused_and_retried(tmp_path, servers):
    """A failed write of `attempts.json` places nothing: the beat is refused retryably, the rank
    stays A's, and once the store takes the write again B's next beat places it."""
    flight = pytest.importorskip("pyarrow.flight")
    store = tmp_path / "store"
    server = servers()
    beat(server.endpoint, A)
    stream = Stream(server.endpoint, A)
    stream.send([rec(A, s) for s in range(0, 11)])
    wait_status(store, lambda s: rank0(s)["last_step"] == 10)
    rank_folder = run_dir(store) / "rank=0"
    mode = rank_folder.stat().st_mode
    rank_folder.chmod(0o555)  # attempts.json beside the attempt folders cannot be replaced
    try:
        with pytest.raises(flight.FlightUnavailableError):
            beat(server.endpoint, B)
        status = wait_status(store, lambda s: rank0(s)["attempt"] == A, seconds=2.0)
        assert rank0(status)["attempt"] == A
        assert "attempt_without_steps" not in rank0(status)
    finally:
        rank_folder.chmod(mode)
    beat(server.endpoint, B)
    status = wait_status(store, lambda s: rank0(s)["attempt"] == B)
    assert (rank0(status)["attempt"], rank0(status).get("attempt_without_steps")) == (B, True)
    assert attempt_order(store) == [A, B]
    stream.close()


# ---------------------------------------------------------------------------------------------
# Contract 1, the fixture: the column of a reader with only the local JSONL file


def _local_file_at_each_stage() -> list[list[dict[str, Any]]]:
    stage1 = [rec(A, s) for s in range(0, 101)]
    stage2 = list(stage1)  # B's heartbeat never reaches the file
    stage3 = stage2 + [rec(A, 101)]
    stage4 = stage3 + [rec(B, 30)]
    return [stage1, stage2, stage3, stage4]


def test_rs05_fixture_a_local_file_orders_attempts_by_first_appearance():
    stage1, stage2, stage3, stage4 = _local_file_at_each_stage()
    assert resolve_attempts(stage1) == stage1  # A owns, at 100
    assert resolve_attempts(stage2) == stage2  # A is the only attempt observed
    assert resolve_attempts(stage3)[-1] == rec(A, 101)  # A at 101 is the latest visible progress
    kept = steps_of(resolve_attempts(stage4))
    # B at 30; A's history below 30 is kept, A's 100 and 101 are out of the effective run.
    assert kept == [(A, s) for s in range(0, 30)] + [(B, 30)]


def test_rs05_backlog_in_a_local_file_is_older_than_the_attempt_after_it():
    records = [rec(A, s) for s in range(0, 51)] + [rec(B, s) for s in range(30, 41)]
    assert steps_of(resolve_attempts(records)) == [(A, s) for s in range(0, 30)] + [(B, s) for s in range(30, 41)]


@needs_server
def test_rs05_fixture_the_offline_detectors_read_a_local_file_in_its_own_order():
    pytest.importorskip("pyarrow")
    from tensorquill.server_detect import server_report

    *_, stage4 = _local_file_at_each_stage()
    report = server_report(stage4, binary=BINARY)
    assert (report["attempt"], report["last_step"], report["steps"]) == (B, 30, 31), report


# ---------------------------------------------------------------------------------------------
# Contract 3: capture stopped


def _stopped_entry(status: dict[str, Any]) -> dict[str, Any] | None:
    return rank0(status).get("capture_stopped")


@needs_server
def test_c03_capture_stopped_lifecycle_close_and_a_new_attempt(tmp_path, servers, monkeypatch):
    """Capture stops while training goes on; the stop reaches the server; the session closes; a
    new attempt starts. The stopped attempt never reads finished, and the new one takes over."""
    pytest.importorskip("pyarrow.flight")
    import torch

    import tensorquill

    store, run_id = tmp_path / "store", "w2-c03"
    server = servers("--silent-seconds", "2", "--silent-factor", "0", "--watchdog-seconds", "0.1",
                     "--flush-seconds", "0.1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TENSORQUILL_ENDPOINT", server.endpoint)
    for name in ("TENSORQUILL_TOKEN", "MLFLOW_RUN_ID", "TORCHELASTIC_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    def step() -> None:
        opt.zero_grad()
        model(torch.ones(2, 4)).sum().backward()
        opt.step()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = tensorquill.init(model, run_id=run_id, every=1, heartbeat_seconds=0.2, stack_seconds=None)
        inner, writes = session._sink._sink, [0]
        real = inner.write

        def failing(records):
            writes[0] += 1
            if writes[0] > 3:
                raise OSError("disk gone")
            real(records)

        inner.write = failing
        deadline = time.monotonic() + 30
        while not session._stopped and time.monotonic() < deadline:
            step()
        assert session._stopped, "the injected sink failure stopped capture"
        first = session.attempt if hasattr(session, "attempt") else None

        # Training goes on; the server must show the stop, not a suspected hang.
        ready = lambda s: rank0(s)["state"] == "capture_stopped"  # noqa: E731
        status = wait_status(store, ready, run_id=run_id)
        for _ in range(3):
            step()
        entry = rank0(status)
        assert entry["state"] == "capture_stopped", entry
        assert entry["severity"] is None, entry
        assert _stopped_entry(status) and _stopped_entry(status)["reason"] == "sink_error", entry
        assert "stack" not in entry

        # Closed: still not finished.
        session.close()
        status = wait_status(store, lambda s: rank0(s)["stream"] == "finished", run_id=run_id)
        entry = rank0(status)
        assert (entry["stream"], entry["state"], status["summary"]) == ("finished", "capture_stopped",
                                                                         "capture_stopped"), entry
        assert status["run_state"]["state"] != "healthy"

        # A new attempt on the same run and rank takes the rank and is watched again.
        again = tensorquill.init(model, run_id=run_id, every=1, heartbeat_seconds=0.2, stack_seconds=None)
        try:
            for _ in range(5):
                step()
            status = wait_status(store, lambda s: rank0(s)["state"] == "active"
                                 and rank0(s).get("capture_stopped") is None, run_id=run_id)
            entry = rank0(status)
            assert entry["state"] == "active" and "capture_stopped" not in entry, entry
            if first is not None:
                assert entry["attempt"] != first
        finally:
            again.close()


@needs_server
def test_c03_the_stop_and_its_reason_survive_a_restart_while_the_rank_reads_unresponsive(tmp_path, servers):
    """Once seen, the stop is kept in every state: a restart while the rank reads `unresponsive`
    (stream open, beats stopped) keeps the reason, and the run never reads finished or healthy."""
    pytest.importorskip("pyarrow.flight")
    store, attempt = tmp_path / "store", 7
    flags = ("--silent-seconds", "2", "--silent-factor", "0", "--watchdog-seconds", "0.1", "--flush-seconds", "0.1")
    server = servers(*flags)
    beat(server.endpoint, attempt, work=0)
    stream = Stream(server.endpoint, attempt)
    stream.send([rec(attempt, s) for s in range(0, 5)])
    beat(server.endpoint, attempt, work=5, capture_stopped="sink_error")
    status = wait_status(store, lambda s: rank0(s)["state"] == "capture_stopped")
    assert rank0(status)["state"] == "capture_stopped", rank0(status)

    # The beats stop with the stream open: a frozen process.
    status = wait_status(store, lambda s: rank0(s)["state"] == "unresponsive")
    entry = rank0(status)
    assert entry["state"] == "unresponsive", entry
    assert _stopped_entry(status) and _stopped_entry(status)["reason"] == "sink_error", entry

    server.kill()
    stream.close()
    server = servers(*flags)
    status = wait_status(store, lambda s: s["written_at_ms"] >= server.started_ms)
    entry = rank0(status)
    assert _stopped_entry(status) and _stopped_entry(status)["reason"] == "sink_error", entry
    assert status["summary"] != "finished"
    assert status["run_state"]["state"] != "healthy"
    assert any(reason.get("code") == "capture_stopped" for reason in status["run_state"]["reasons"]), \
        status["run_state"]

    # The process's beats come back: it reads capture_stopped again, not a hang.
    beat(server.endpoint, attempt, work=5, capture_stopped="sink_error")
    status = wait_status(store, lambda s: rank0(s)["state"] == "capture_stopped")
    assert rank0(status)["state"] == "capture_stopped", rank0(status)
