"""Lane AR-7, item D (item 7 of the author's list, widened by lane AR-4's limit): a session the server
would refuse does not stream.

The author: "A rank past Int32 or an attempt past Int64 opens a stream the server refuses: the sender
says so once at start and does not stream." Lane AR-4 then made the server refuse a stream whose rank
is 100000 or more, a batch whose `world_size` is over 100000 and a heartbeat of such a rank ("over
this server's limit of 100000 ranks a run"). The unchanged sender, given such a rank or world size,
had its batches refused and sent the same bytes again at its backoff for the rest of the run, and its
heartbeats were refused without a word.

So `flight.attach` checks before it starts anything: `rank >= 100000`, or `world_size > 100000`, or
the session's attempt past Int64 (the Int32 bound on `rank` is inside the first). Then it returns the
sink unchanged, starts no sender and no heartbeat, and says once which number is over which limit
and that the file is written here as usual; nothing more at `flush()` or `close()`. The limit is the
SDK's one number, `identity.MAX_EXPECTED_RANKS`, held equal to the server's `watchdog::MAX_RANKS`.

The tests marked `needs_server` run the real `tensorquill-server` (`TENSORQUILL_SERVER_BIN`) behind a
relay that counts the connections made to it, and skip without it.
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("pyarrow.flight")

import tensorquill  # noqa: E402
from tensorquill import capture, evidence, identity  # noqa: E402
from tensorquill.flight import offset_path  # noqa: E402
from test_flight_e2e import BINARY, Server  # noqa: E402

needs_server = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built (TENSORQUILL_SERVER_BIN)")

NOWHERE = "grpc://127.0.0.1:9"  # nothing listens here
LIMIT = 100_000
INT64_MAX = 2**63 - 1
SERVER_SOURCE = Path(__file__).resolve().parents[2] / "server" / "src"


@pytest.fixture
def here(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name in ("RANK", "LOCAL_RANK", "WORLD_SIZE") or name.startswith(
                ("SLURM_", "OMPI_", "PMI_", "TORCHELASTIC_", "TENSORQUILL_", "MLFLOW_", "WANDB_")):
            if name != "TENSORQUILL_SERVER_BIN":
                monkeypatch.delenv(name, raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


def _stream_threads() -> int:
    return sum(1 for thread in threading.enumerate() if thread.name.startswith(("tensorquill-flight-",
                                                                                "tensorquill-heartbeat-")))


def _said(caught) -> list[str]:
    return [str(w.message) for w in caught if "tensorquill" in str(w.message).lower()]


def _over(caught) -> list[str]:
    return [m for m in _said(caught) if "over the server's limit" in m or "is past Int64" in m]


def _run(run_id: str, endpoint: str, steps: int = 3):
    """A session started, trained, flushed and closed; what it said at each of those."""
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    threads = _stream_threads()
    with warnings.catch_warnings(record=True) as at_init:
        warnings.simplefilter("always")
        session = tensorquill.init(model, run_id=run_id, every=1, endpoint=endpoint)
    started = _stream_threads() - threads
    with warnings.catch_warnings(record=True) as later:
        warnings.simplefilter("always")
        for _ in range(steps):
            model(torch.randn(3, 2)).pow(2).mean().backward()
            optimizer.step()
            optimizer.zero_grad()
        session.flush()
        session.close()
    return session, at_init, later, started


def _steps(path: Path) -> list[int]:
    return [json.loads(line)["step"] for line in path.read_text().splitlines() if '"record"' not in line]


# --- One limit, in both languages -----------------------------------------------------------------


def test_the_sdks_limit_is_the_servers_max_ranks():
    source = (SERVER_SOURCE / "watchdog.rs").read_text(encoding="utf-8")
    (number,) = re.findall(r"^pub const MAX_RANKS: i32 = ([0-9_]+);$", source, re.M)
    assert int(number.replace("_", "")) == identity.MAX_EXPECTED_RANKS == LIMIT
    assert evidence.MAX_EXPECTED_RANKS is identity.MAX_EXPECTED_RANKS  # one number, not two


def test_the_server_refuses_with_the_limit_the_sdk_checks():
    """The refusals the local check stands in for (lane AR-4): a rank and a world size over the limit."""
    source = (SERVER_SOURCE / "ingest.rs").read_text(encoding="utf-8")
    assert "world_size {world_size} is over this server's limit of {} ranks a run" in source
    assert "rank {rank} is over this server's limit of {limit} ranks a run (ranks 0 to {})" in source


# --- No server ------------------------------------------------------------------------------------


@pytest.mark.parametrize("rank, world, what", [
    (LIMIT, LIMIT, f"rank {LIMIT} is over the server's limit of {LIMIT} ranks a run (ranks 0 to {LIMIT - 1})"),
    (2**31, 4, f"rank {2**31} is over the server's limit of {LIMIT} ranks a run (ranks 0 to {LIMIT - 1})"),
    (0, LIMIT + 1, f"world size {LIMIT + 1} is over the server's limit of {LIMIT} ranks a run"),
], ids=["rank-at-the-limit", "rank-past-int32", "world-over-the-limit"])
def test_a_session_over_the_limit_starts_no_sender_and_no_heartbeat_and_says_so_once(here, monkeypatch, rank, world,
                                                                                    what):
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", str(world))
    session, at_init, later, started = _run("over", NOWHERE)
    assert type(session._sink) is tensorquill.JsonlSink, type(session._sink)
    assert started == 0
    (said,) = _over(at_init)
    path = Path("tensorquill-out") / f"over.rank{rank}.jsonl"
    assert said.startswith(f"tensorquill.init: {what}, so nothing of {path.name} is streamed to {NOWHERE}: "), said
    assert "no sender and no heartbeat are started" in said and "The file is written here as usual." in said
    assert not _said(later)  # nothing at flush() or close()
    assert _steps(path) == [0, 1, 2]
    assert not offset_path(path).exists()


def test_a_session_whose_attempt_is_past_int64_starts_no_sender(here, monkeypatch):
    monkeypatch.setattr(capture, "_last_attempt", INT64_MAX)  # the next attempt is 2**63
    session, at_init, later, started = _run("late", NOWHERE)
    assert session.attempt == 2**63
    assert type(session._sink) is tensorquill.JsonlSink and started == 0
    (said,) = _over(at_init)
    assert said.startswith(f"tensorquill.init: this session's attempt {2**63} is past Int64 ({INT64_MAX}), the most "
                           "the server's attempt column holds, so nothing of late.rank0.jsonl is streamed"), said
    assert not _said(later)


def test_every_number_over_its_limit_is_named_in_the_one_statement(here, monkeypatch):
    monkeypatch.setenv("RANK", str(LIMIT + 5))
    monkeypatch.setenv("WORLD_SIZE", str(LIMIT + 9))
    monkeypatch.setattr(capture, "_last_attempt", INT64_MAX)
    _, at_init, _, started = _run("all", NOWHERE)
    (said,) = _over(at_init)
    assert started == 0
    assert f"rank {LIMIT + 5} is over" in said and f"world size {LIMIT + 9} is over" in said, said
    assert f"attempt {2**63} is past Int64" in said, said


@pytest.mark.parametrize("rank, world", [(LIMIT - 1, LIMIT), (0, LIMIT), (3, 0)])
def test_at_the_edges_the_session_streams_as_before(here, monkeypatch, rank, world):
    """Rank 99,999 and a world size of 100,000 are taken by the server; a world size of 0 names none."""
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", str(world))
    model = nn.Linear(2, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, run_id="edge", every=1, endpoint=NOWHERE, heartbeat_seconds=None)
    try:
        assert hasattr(session._sink, "sender"), type(session._sink)
        assert not _over(caught)
    finally:
        session.close()


def test_with_no_endpoint_nothing_is_said(here, monkeypatch):
    monkeypatch.setenv("RANK", str(LIMIT))
    monkeypatch.setenv("WORLD_SIZE", str(LIMIT + 1))
    model = nn.Linear(2, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, run_id="quiet", every=1)
    session.close()
    assert not _over(caught)


def test_a_second_writer_over_the_limit_is_said_as_a_second_writer_only(here, monkeypatch):
    """Both rules hold: the second writer's statement (lane AR-5) is the one said about the stream."""
    monkeypatch.setenv("RANK", str(LIMIT))
    monkeypatch.setenv("WORLD_SIZE", str(LIMIT + 1))
    one = tensorquill.init(nn.Linear(2, 1), run_id="both", every=1)
    try:
        session, at_init, later, started = _run("both", NOWHERE)
    finally:
        one.close()
    assert started == 0 and not _over(at_init)
    assert [m for m in _said(at_init) if "another writer holds" in m]


def test_under_warnings_as_errors_the_statement_is_logged_and_init_returns(here, monkeypatch, caplog):
    monkeypatch.setenv("RANK", str(LIMIT))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        session = tensorquill.init(nn.Linear(2, 1), run_id="strict", every=1, endpoint=NOWHERE)
    session.close()
    assert [r for r in caplog.records if "over the server's limit" in r.getMessage()]


# --- A live server receives nothing ---------------------------------------------------------------


class Relay:
    """A TCP relay in front of the server that counts the connections made through it."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.connections = 0
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.endpoint = f"grpc://127.0.0.1:{self.listener.getsockname()[1]}"
        self._closed = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while not self._closed:
            try:
                client, _ = self.listener.accept()
            except OSError:
                return
            self.connections += 1
            upstream = socket.create_connection(("127.0.0.1", self.port))
            for a, b in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._pipe, args=(a, b), daemon=True).start()

    @staticmethod
    def _pipe(source: socket.socket, target: socket.socket) -> None:
        try:
            while True:
                data = source.recv(65536)
                if not data:
                    break
                target.sendall(data)
        except OSError:
            pass
        finally:
            for end in (source, target):
                try:
                    end.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def close(self) -> None:
        self._closed = True
        self.listener.close()


@needs_server
@pytest.mark.parametrize("over", ["rank", "world_size", "attempt"])
def test_a_live_server_receives_nothing_from_a_session_over_the_limit(here, monkeypatch, tmp_path, over):
    if over == "rank":
        monkeypatch.setenv("RANK", str(LIMIT))
        monkeypatch.setenv("WORLD_SIZE", str(LIMIT + 1))
    elif over == "world_size":
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", str(LIMIT + 1))
    else:
        monkeypatch.setattr(capture, "_last_attempt", INT64_MAX)
    store = tmp_path / "store"
    server = Server(store)
    relay = Relay(server.port)
    try:
        session, at_init, later, started = _run(f"live-{over}", relay.endpoint, steps=4)
    finally:
        relay.close()
        assert server.terminate() == 0
    assert started == 0
    assert relay.connections == 0  # not a heartbeat, not a stream
    assert not (store / "runs" / f"live-{over}").exists()
    assert len(_over(at_init)) == 1 and not _said(later)


@needs_server
def test_through_the_relay_a_session_under_the_limit_reaches_the_server(here, monkeypatch, tmp_path):
    """The guard: the relay passes a stream through, so a count of 0 above means nothing was sent."""
    from test_flight_e2e import stored_records

    monkeypatch.setenv("RANK", str(LIMIT - 1))
    monkeypatch.setenv("WORLD_SIZE", str(LIMIT))
    store = tmp_path / "store"
    server = Server(store)
    relay = Relay(server.port)
    try:
        session, at_init, _, started = _run("live-under", relay.endpoint, steps=4)
    finally:
        relay.close()
        assert server.terminate() == 0
    assert started == 2  # the sender and the heartbeat
    assert relay.connections > 0
    assert sorted(r["step"] for r in stored_records(store, "live-under", rank=LIMIT - 1)) == [0, 1, 2, 3]
