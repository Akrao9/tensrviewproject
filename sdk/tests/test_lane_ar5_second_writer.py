"""Lane AR-5, part 1: a second writer of a rank does not stream.

A second writer of the same run ID and rank (a second process started by mistake with the same
run ID, or a second session in this one) finds the rank's file locked and writes
`<run_id>.rank<N>.<pid>.jsonl` beside it (`sink.JsonlSink`, lane AQ-2). Its records carry the same
`run_id` and `rank` as the first writer's. Before this lane, with `endpoint=` (or
`TENSORQUILL_ENDPOINT`) set, `flight.attach` streamed that file too, under the same run ID and
rank: the server got two streams of one rank and stored both writers' rows in that rank, each
under its own attempt, so the second writer's rows read as a newer attempt of the first's rank.

The author's answer: "With `endpoint=` set, a second writer writes its file, does not stream, and
says so once". So such a session writes its file as before, starts no sender and no heartbeat,
and says once, at `init`, that nothing of its file is streamed; nothing more at `flush()` or
`close()`. The first writer streams as before. Where the lock is quiet (no `fcntl`, or a lock
that fails) no second writer is detected and nothing changes.

The tests marked `needs_server` run the real `tensorquill-server` (`TENSORQUILL_SERVER_BIN`) and
skip without it; the others need no server.
"""

from __future__ import annotations

import errno
import json
import os
import threading
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("pyarrow.flight")

import tensorquill  # noqa: E402
from tensorquill import flight  # noqa: E402
from tensorquill import sink as _sink  # noqa: E402
from tensorquill.flight import offset_path, send_file  # noqa: E402
from tensorquill.sink import end_marker_path  # noqa: E402
from test_flight_e2e import BINARY, Server, stored_records  # noqa: E402

needs_server = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built (TENSORQUILL_SERVER_BIN)")

# Nothing listens here: a session given it starts a sender and a heartbeat that never connect.
NOWHERE = "grpc://127.0.0.1:9"


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


def _said(caught) -> list[str]:
    return [str(w.message) for w in caught if "tensorquill" in str(w.message).lower()]


def _init(run_id: str, **options):
    model = nn.Linear(2, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, run_id=run_id, every=1, **options)
    return session, _said(caught), model, caught


def _train(model, steps: int) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(steps):
        model(torch.randn(3, 2)).pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad()


def _steps(path: Path) -> list[int]:
    return [json.loads(line)["step"] for line in path.read_text().splitlines() if '"step"' in line]


def _stream_threads(rank: int = 0) -> dict[str, int]:
    names = [thread.name for thread in threading.enumerate()]
    return {kind: names.count(f"tensorquill-{kind}-rank{rank}") for kind in ("flight", "heartbeat")}


def _streams(session) -> bool:
    return hasattr(session._sink, "sender") or hasattr(session._sink, "heartbeat")


def _assert_not_streamed_said(message: str, path: Path, first: str) -> None:
    """The second writer's one statement about the stream: its file, that another writer holds the
    run's rank, that nothing of the file is streamed, and that there is no supported way to send it."""
    assert path.name in message and first in message, message
    assert "another writer" in message and "nothing of" in message and "streamed" in message, message
    assert "send_file" in message, message


# --- No server ------------------------------------------------------------------------------


@pytest.mark.parametrize("given", ["endpoint=", "TENSORQUILL_ENDPOINT"])
def test_a_second_writer_with_an_endpoint_starts_no_sender_and_no_heartbeat(here, monkeypatch, given):
    """Fail-before: the second session's sink was a `StreamingSink` with a sender of the pid file
    under the first's run ID and rank, and a heartbeat of that rank."""
    one, said_one, model_one, _ = _init("same")  # the first writer holds the lock; it does not stream
    options = {"endpoint": NOWHERE} if given == "endpoint=" else {}
    if given == "TENSORQUILL_ENDPOINT":
        monkeypatch.setenv("TENSORQUILL_ENDPOINT", NOWHERE)
    before = _stream_threads()
    two, said_two, model_two, caught = _init("same", **options)
    try:
        pid_file = Path("tensorquill-out") / f"same.rank0.{os.getpid()}.jsonl"
        assert two._sink.path == pid_file
        assert type(two._sink) is tensorquill.JsonlSink, type(two._sink)
        assert not _streams(two)
        assert _stream_threads() == before == {"flight": 0, "heartbeat": 0}
        # Two statements at init: the sink's (which file, and why), and the stream's (nothing of
        # the file is streamed). The stream's points where the sink's does (`init`'s caller, or the
        # `torch.compiler.disable` frame around `init`, as every warning of `init` does).
        assert said_one == [] and len(said_two) == 2, said_two
        assert "held by another writer" in said_two[0]
        _assert_not_streamed_said(said_two[1], pid_file, "same.rank0.jsonl")
        assert len({(w.filename, w.lineno) for w in caught}) == 1, [(w.filename, w.lineno) for w in caught]
        with warnings.catch_warnings(record=True) as later:
            warnings.simplefilter("always")
            _train(model_two, 3)
            two.flush()
            two.close()
        assert _said(later) == []  # once: nothing more at flush() or close()
    finally:
        two.close()
        _train(model_one, 2)
        one.close()
    out = here / "tensorquill-out"
    assert _steps(out / pid_file.name) == [0, 1, 2]
    assert end_marker_path(out / pid_file.name).exists()
    assert not offset_path(out / pid_file.name).exists()
    assert _steps(out / "same.rank0.jsonl") == [0, 1]


def test_two_writers_of_one_rank_both_given_an_endpoint_make_one_stream_of_that_rank(here):
    """No server. Fail-before: two senders and two heartbeats of rank 0 ran, one of each per writer,
    the second's sender tailing the pid file under the first's run ID and rank."""
    before = _stream_threads()
    one, _, _, _ = _init("same", endpoint=NOWHERE)
    two, said, _, _ = _init("same", endpoint=NOWHERE)
    try:
        assert isinstance(one._sink, flight.StreamingSink) and not _streams(two)
        now = _stream_threads()
        assert {kind: now[kind] - before[kind] for kind in now} == {"flight": 1, "heartbeat": 1}
        assert len(said) == 2, said
    finally:
        for session in (two, one):
            sender = getattr(session._sink, "sender", None)
            if sender is not None:
                sender.close(0.0)  # nothing listens: do not wait the close's five seconds
            session.close()


def test_a_third_writer_in_this_process_does_not_stream_either(here):
    """The third writer writes into the second's pid file, unlocked (lane AQ-2's choice 6): it is
    still not the rank's first writer, so it streams nothing."""
    one, _, _, _ = _init("same")
    two, _, _, _ = _init("same", endpoint=NOWHERE)
    three, said, model, _ = _init("same", endpoint=NOWHERE)
    try:
        assert three._sink.path == two._sink.path
        assert not _streams(two) and not _streams(three)
        assert _stream_threads() == {"flight": 0, "heartbeat": 0}
        assert len(said) == 2 and "lock" in said[0], said
        _assert_not_streamed_said(said[1], three._sink.path, "same.rank0.jsonl")
        _train(model, 1)
    finally:
        for session in (three, two, one):
            session.close()


def test_the_first_writer_is_still_streamed(here):
    """A guard: the writer that holds the lock streams as before."""
    one, said, _, _ = _init("first", endpoint=NOWHERE, heartbeat_seconds=None)
    try:
        assert one._sink.path.name == "first.rank0.jsonl"
        assert isinstance(one._sink, flight.StreamingSink)
        assert one._sink.sender.path == one._sink.path and one._sink.sender.rank == 0
        assert said == []
    finally:
        one._sink.sender.close(0.0)  # nothing listens: do not wait the close's five seconds
        one.close()


def test_a_sink_made_by_hand_with_a_pid_files_name_is_streamed(tmp_path):
    """A guard: the second writer is known by what its sink found at open, not by its file's name."""
    path = tmp_path / "same.rank0.4242.jsonl"
    sink = tensorquill.JsonlSink(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        streamed = flight.attach(sink, "same", 0, NOWHERE, heartbeat_seconds=None)
    try:
        assert isinstance(streamed, flight.StreamingSink) and streamed.sender.path == path
        assert _said(caught) == []
    finally:
        streamed.sender.close(0.0)
        streamed.close()


def test_where_there_is_no_fcntl_nothing_changes(here, monkeypatch):
    """Where Python has no `fcntl` (Windows) the lock is quiet: no second writer is detected, both
    sessions write the same file and both stream it, as before this lane. A guard."""
    monkeypatch.setattr(_sink, "_fcntl", None)
    one, said_one, _, _ = _init("same", endpoint=NOWHERE, heartbeat_seconds=None)
    two, said_two, _, _ = _init("same", endpoint=NOWHERE, heartbeat_seconds=None)
    try:
        assert one._sink.path == two._sink.path == Path("tensorquill-out") / "same.rank0.jsonl"
        assert _streams(one) and _streams(two)
        assert said_one == said_two == []
    finally:
        for session in (two, one):
            session._sink.sender.close(0.0)
            session.close()


def test_where_the_lock_fails_nothing_changes(here, monkeypatch):
    """A lock that fails for another reason (a file system without locking) is no lock here: no
    second writer is detected, and each session streams the file it writes, as before. A guard."""
    def fails(fd, operation):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(_sink._fcntl, "flock", fails)
    one, said, _, _ = _init("same", endpoint=NOWHERE, heartbeat_seconds=None)
    try:
        assert one._sink.path.name == "same.rank0.jsonl" and _streams(one)
        assert len(said) == 1 and "could not lock" in said[0], said
    finally:
        one._sink.sender.close(0.0)
        one.close()


# --- The real server ------------------------------------------------------------------------


@needs_server
def test_two_writers_of_one_rank_reach_the_server_as_the_first_writer_alone(here, tmp_path):
    """Fail-before: the server stored both writers' rows in rank 0, the second's under an attempt of
    its own, newer than the first's (seen on the base commit: steps 0-3 under the first's attempt
    and steps 0-2 under the second's, and a `.offset` file beside the pid file)."""
    store = tmp_path / "store"
    server = Server(store)
    try:
        one, said_one, model_one, _ = _init("same", endpoint=server.endpoint)
        two, said_two, model_two, _ = _init("same", endpoint=server.endpoint)
        try:
            assert said_one == [] and len(said_two) == 2, said_two
            assert _stream_threads() == {"flight": 1, "heartbeat": 1}
            _train(model_one, 4)
            _train(model_two, 3)
        finally:
            two.close()
            one.close()
        attempt_one, pid_file = one.attempt, here / two._sink.path
    finally:
        assert server.terminate() == 0
    stored = stored_records(store, "same")
    assert sorted((r["attempt"], r["step"]) for r in stored) == [(attempt_one, step) for step in range(4)]
    assert sorted(p.name for p in (store / "runs" / "same" / "rank=0").glob("attempt=*")) == [
        f"attempt={attempt_one}"]
    assert _steps(pid_file) == [0, 1, 2]
    assert end_marker_path(pid_file).exists()
    assert not offset_path(pid_file).exists()


@needs_server
def test_send_file_of_a_second_writers_file_is_stored_as_another_attempt_of_the_rank(here, tmp_path):
    """What the second writer's statement says of `send_file`: it is not a way to send the file on
    its own, since it stores the file's rows in the first writer's run and rank, under the second
    writer's attempt. A guard of that statement's truth."""
    store = tmp_path / "store"
    server = Server(store)
    try:
        one, _, model_one, _ = _init("same", endpoint=server.endpoint)
        two, _, model_two, _ = _init("same", endpoint=server.endpoint)
        _train(model_one, 2)
        _train(model_two, 2)
        two.close()
        one.close()
        assert send_file(here / two._sink.path, server.endpoint, "same", 0, timeout=20.0)
    finally:
        assert server.terminate() == 0
    stored = sorted((r["attempt"], r["step"]) for r in stored_records(store, "same"))
    assert stored == sorted([(one.attempt, 0), (one.attempt, 1), (two.attempt, 0), (two.attempt, 1)])
