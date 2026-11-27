"""Lane AR-7, item D later: the sender itself, not only the one `init` starts, sends nothing the server
would refuse.

The author's answer for item 7: "A rank past Int32 or an attempt past Int64 opens a stream the server
refuses: the sender says so once at start and does not stream." Item D made `flight.attach` check it
(`test_lane_ar7_over_the_limit.py`). `flight.send_file` and a `FlightSender` built by hand did not:
observed before this change, `send_file(path, endpoint, run_id, rank=100000, timeout=10)` against the
real server opened 9 connections in its 10 s, each refused ("rank 100000 is over this server's limit
of 100000 ranks a run"), stored nothing, said only the server's refusal at its close, and returned
False after the whole timeout.

Now a `FlightSender` with a rank of `MAX_EXPECTED_RANKS` (100,000) or more, or a `world_size` over it,
opens no connection and starts no thread, says once which number is over which limit, and `close()`,
so `send_file`, returns False at once. Nothing is raised: `send_file` answers with a bool.

A line whose attempt is past Int64 needed no change: the sender already leaves it out, says so once
and counts it, and `send_file` returns False at once (pinned below).

The tests marked `needs_server` run the real `tensorquill-server` behind the counting relay of
`test_lane_ar7_over_the_limit.py`.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import warnings
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")

from tensorquill import flight  # noqa: E402
from tensorquill.flight import FlightSender, offset_path, send_file  # noqa: E402
from test_flight_e2e import BINARY, Server  # noqa: E402
from test_lane_ar7_over_the_limit import Relay  # noqa: E402

needs_server = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built (TENSORQUILL_SERVER_BIN)")

NOWHERE = "grpc://127.0.0.1:9"
LIMIT = 100_000
QUICK = 2.0  # seconds: what "at once" is allowed here; the timeout given is 30 s


def _file(folder: Path, run_id: str, rank: int, attempt: int = 1, steps: int = 4) -> Path:
    path = folder / f"{run_id}.rank{rank}.jsonl"
    lines = [{"schema_version": 1, "run_id": run_id, "rank": rank, "attempt": attempt, "step": step,
              "ts": float(step), "grad_norm": 1.0} for step in range(steps)]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def _sender_threads() -> int:
    return sum(1 for thread in threading.enumerate() if thread.name.startswith("tensorquill-flight-"))


def _said(caught) -> list[str]:
    return [str(w.message) for w in caught if "tensorquill" in str(w.message).lower()]


@pytest.mark.parametrize("rank, world, what", [
    (LIMIT, None, f"rank {LIMIT} is over the server's limit of {LIMIT} ranks a run (ranks 0 to {LIMIT - 1})"),
    (2**31, None, f"rank {2**31} is over the server's limit of {LIMIT} ranks a run (ranks 0 to {LIMIT - 1})"),
    (0, LIMIT + 1, f"world size {LIMIT + 1} is over the server's limit of {LIMIT} ranks a run"),
], ids=["rank-at-the-limit", "rank-past-int32", "world-over-the-limit"])
def test_a_sender_over_the_limit_starts_no_thread_says_so_once_and_closes_false_at_once(tmp_path, capfd, rank,
                                                                                      world, what):
    path = _file(tmp_path, "hand", rank)
    before = _sender_threads()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sender = FlightSender(path, NOWHERE, "hand", rank, world_size=world)
        started = _sender_threads() - before
        begun = time.monotonic()
        closed = sender.close(timeout=30.0)
        took = time.monotonic() - begun
    assert started == 0
    assert closed is False and took < QUICK
    (said,) = _said(caught)
    assert said == (f"tensorquill: {what}, so nothing of {path.name} is sent to {NOWHERE}: the server would refuse "
                    "it, so no connection is made. The file is kept as written."), said
    assert isinstance(sender.last_error, flight.OverTheLimit) and what in str(sender.last_error)
    assert "tensorquill" not in capfd.readouterr().err  # nothing more at close()
    assert not offset_path(path).exists()
    assert sender.acked_offset == 0 and sender.unacked_bytes() == path.stat().st_size


def test_send_file_over_the_limit_returns_false_at_once_and_raises_nothing(tmp_path):
    path = _file(tmp_path, "file", LIMIT)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        begun = time.monotonic()
        sent = send_file(path, NOWHERE, "file", LIMIT, timeout=30.0)
        took = time.monotonic() - begun
    assert sent is False and took < QUICK
    assert len(_said(caught)) == 1
    # The statement points at the caller of send_file, as other warnings of the package do.
    (said,) = [w for w in caught if "over the server's limit" in str(w.message)]
    assert Path(said.filename) == Path(__file__), said.filename


def test_under_warnings_as_errors_the_statement_is_logged_and_nothing_raised(tmp_path, caplog):
    path = _file(tmp_path, "strict", LIMIT)
    with warnings.catch_warnings(), caplog.at_level(logging.WARNING, logger="tensorquill"):
        warnings.simplefilter("error")
        assert send_file(path, NOWHERE, "strict", LIMIT, timeout=30.0) is False
    assert [r for r in caplog.records if "over the server's limit" in r.getMessage()]


@needs_server
def test_a_live_server_receives_nothing_from_send_file_over_the_limit(tmp_path):
    path = _file(tmp_path, "live", LIMIT)
    store = tmp_path / "store"
    server = Server(store)
    relay = Relay(server.port)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            begun = time.monotonic()
            sent = send_file(path, relay.endpoint, "live", LIMIT, timeout=30.0)
            took = time.monotonic() - begun
    finally:
        relay.close()
        assert server.terminate() == 0
    assert sent is False and took < QUICK
    assert relay.connections == 0
    assert not (store / "runs" / "live").exists()
    assert len(_said(caught)) == 1


@needs_server
def test_through_the_relay_send_file_under_the_limit_is_stored(tmp_path):
    """The guard: rank 99,999 is taken, so a count of 0 above means nothing was sent."""
    from test_flight_e2e import stored_records

    path = _file(tmp_path, "under", LIMIT - 1)
    store = tmp_path / "store"
    server = Server(store)
    relay = Relay(server.port)
    try:
        sent = send_file(path, relay.endpoint, "under", LIMIT - 1, timeout=30.0)
    finally:
        relay.close()
        assert server.terminate() == 0
    assert sent is True and relay.connections > 0
    assert [r["step"] for r in stored_records(store, "under", rank=LIMIT - 1)] == [0, 1, 2, 3]


@needs_server
def test_a_line_whose_attempt_is_past_int64_is_left_out_said_once_and_send_file_returns_false(tmp_path, capfd):
    """A pin, true before this change too: the attempt is on the file's lines, and the sender leaves
    such a line out, says so once and counts it; nothing of it is stored."""
    path = _file(tmp_path, "late", 0, attempt=2**63)
    store = tmp_path / "store"
    server = Server(store)
    try:
        begun = time.monotonic()
        sent = send_file(path, server.endpoint, "late", 0, timeout=30.0)
        took = time.monotonic() - begun
    finally:
        assert server.terminate() == 0
    err = capfd.readouterr().err
    assert sent is False and took < QUICK
    assert err.count(f"its attempt is {2**63}, and its column in the stream cannot hold it") == 1, err
    assert "4 lines of run late" in err and "left_out_lines" in err, err
    assert not list(store.rglob("*.parquet"))
