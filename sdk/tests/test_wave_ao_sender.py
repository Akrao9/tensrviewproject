"""Wave AO, lane AO-2: the sender (`flight.py`, `sink.py`, `identity.py`).

BUG-02, the sender's side: a line of record version 2 that is not a whole scalar-only record (no
`"record": "scalars"`) was sent as a step row (`record_version` 2, a null `seq`), and every Python
reader then refused the stored part. Such a line is now skipped and counted like any other line the
sender cannot send (`identity.check`'s rule for version 2, the one `is_scalar_record` mirrors).

Each test that needs pyarrow or the server binary (`TENSORQUILL_SERVER_BIN`, or one beside the
interpreter) is skipped without it, one by one.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path

import pytest

from tensorquill import flight
from tensorquill.identity import is_scalar_record
from tensorquill.server_detect import find_binary

try:  # each test is collected and skipped without pyarrow
    import pyarrow  # noqa: F401
except ImportError:
    pyarrow = None

BINARY = find_binary()
needs_arrow = pytest.mark.skipif(pyarrow is None, reason="pyarrow is not installed")


def skip_without_server() -> None:
    if pyarrow is None:
        pytest.skip("pyarrow is not installed")
    if not BINARY or not os.path.isfile(BINARY):
        pytest.skip("tensorquill-server is not built")


class Server:
    """A server on a loopback port with its own store; `close()` stops it and returns its stderr."""

    def __init__(self, store: Path) -> None:
        env = {k: v for k, v in os.environ.items() if k not in ("TENSORQUILL_TOKEN", "TENSORQUILL_ENDPOINT")}
        self.process = subprocess.Popen(
            [BINARY, "--listen", "127.0.0.1:0", "--store", str(store), "--flush-steps", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        line = self.process.stdout.readline()
        assert "listening on" in line, line + self.process.stderr.read()
        self.endpoint = f"grpc://127.0.0.1:{int(line.rsplit(':', 1)[1])}"
        self.store = store

    def close(self) -> str:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
        self.process.wait(30)
        err = self.process.stderr.read()
        self.process.stdout.close()
        self.process.stderr.close()
        return err


def write_lines(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def step_record(run_id: str, step: int, **more) -> dict:
    return {"schema_version": 1, "run_id": run_id, "rank": 0, "step": step, "ts": float(step), "grad_norm": 1.0,
            **more}


def scalar_record(run_id: str, step: int, seq: int, scalars: dict) -> dict:
    return {"schema_version": 2, "record": "scalars", "run_id": run_id, "rank": 0, "step": step, "seq": seq,
            "ts": float(step), "scalars": scalars}


# The line BUG-02 names: version 2 with no `"record": "scalars"`.
def bare_v2(run_id: str, step: int) -> dict:
    return {"schema_version": 2, "run_id": run_id, "rank": 0, "step": step, "loss": 1.0}


# --- BUG-02: a version 2 line that is not a whole scalar-only record ------------------------


@needs_arrow
def test_bug02_a_version_2_line_without_the_record_marker_is_left_out_and_counted():
    refused: list = []
    batch, steps, skipped = flight.rows_to_send([bare_v2("p", 4)], flight.flight_schema(), refused=refused)
    assert batch.num_rows == 0, batch.to_pylist()
    assert steps == [] and skipped == 1
    assert [(index, field, value) for index, field, value in refused] == [(0, "schema_version", 2)]


@needs_arrow
def test_bug02_the_audit_record_with_a_seq_but_no_marker_is_left_out_too():
    record = {"schema_version": 2, "run_id": "local", "rank": 0, "attempt": 1, "step": 4, "seq": 0,
              "scalars": {"loss": 1.0}}
    assert not is_scalar_record(record)
    batch, steps, skipped = flight.rows_to_send([record], flight.flight_schema())
    assert (batch.num_rows, steps, skipped) == (0, [], 1)


@needs_arrow
def test_bug02_pins_what_is_still_sent():
    """A whole scalar-only record is sent with its seq; a step record of version 0 or 1 as a step row;
    a version this package does not read (3) is sent for the server to count (RS10); a scalar-only
    record with no seq is left out naming seq, as before."""
    records = [
        scalar_record("p", 7, 0, {"eval/loss": 1.5}),
        step_record("p", 0),
        {"run_id": "p", "rank": 0, "step": 1},
        {**step_record("p", 2), "schema_version": 3},
        {"schema_version": 2, "record": "scalars", "run_id": "p", "rank": 0, "step": 9, "scalars": {"x": 1.0}},
    ]
    refused: list = []
    batch, steps, skipped = flight.rows_to_send(records, flight.flight_schema(), refused=refused)
    rows = batch.to_pylist()
    assert [(r["step"], r["seq"], r["record_version"], r["schema_version"]) for r in rows] == [
        (7, 0, 2, 2), (0, None, 1, 2), (1, None, None, 2), (2, None, 3, 3)]
    assert steps == [0, 1, 2] and skipped == 1
    assert [(i, f) for i, f, _ in refused] == [(4, "seq")]


def test_bug02_through_a_real_server_the_store_stays_readable(tmp_path):
    skip_without_server()
    from tensorquill.evidence import load_records

    running = Server(tmp_path / "store")
    try:
        path = write_lines(tmp_path / "p.jsonl", [step_record("p", 0), bare_v2("p", 1), step_record("p", 2)])
        sender = flight.FlightSender(path, running.endpoint, "p", 0)
        sender.close(20.0)
        assert sender.skipped_lines == 1
    finally:
        running.close()
    stored = load_records(tmp_path / "store" / "runs" / "p")
    assert [r["step"] for r in stored] == [0, 2]
