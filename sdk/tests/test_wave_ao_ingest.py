"""Lane AO-3 of wave AO, through a real server: the ingest and storage findings on `453f1e2`.

- BUG-07 (TQ-01): three sessions under one run ID on rank 0. A logs `eval_loss` at an explicit
  step 50 and closes, B writes training step 0, C writes training step 100 and is the sender's
  attempt. The local reader rolls A's point back at B's first step, so the default series is empty.
  The same file sent one line a batch and as one batch must give the store the same attempt order,
  A, B, C, and the same empty series. On `453f1e2` one batch gave B, A, C and the point came back.
- BUG-08: an attempt the rank's `attempts.json` does not list goes just before the newest, by
  number among such, whatever placed the newest. The same table is asserted against the server's
  reader in `server/tests/wave_ao_ingest_test.rs`; here it pins Python's side to it.
- BUG-02, the server side: a row whose `record_version` is 2 and whose `seq` is null is dropped and
  counted under `seq`, so the store stays readable by every Python reader. The batch is built here
  with pyarrow, not by the sender, so this test stands on the server alone.

Needs pyarrow and the server binary (`TENSORQUILL_SERVER_BIN`, or one beside the interpreter); each
test that needs them is skipped without them.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from tensorquill.evidence import load_records, scalar_series
from tensorquill.server_detect import find_binary

try:  # each test is collected and skipped without pyarrow (a module skipped whole collects nothing)
    import pyarrow as pa
    import pyarrow.flight as pa_flight
    import pyarrow.parquet as pq
except ImportError:
    pa = pa_flight = pq = None

BINARY = find_binary()
needs_pyarrow = pytest.mark.skipif(pa is None, reason="pyarrow is not installed")
needs_binary = pytest.mark.skipif(not BINARY or not os.path.isfile(BINARY), reason="tensorquill-server is not built")

A, B, C = 1_732_000_000_000, 1_732_000_100_000, 1_732_000_200_000


class Server:
    def __init__(self, store: Path) -> None:
        env = {k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"}
        self.process = subprocess.Popen(
            [BINARY, "--listen", "127.0.0.1:0", "--store", str(store), "--flush-steps", "1",
             "--watchdog-seconds", "0.5"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        line = self.process.stdout.readline()
        assert "listening on" in line, line + self.process.stderr.read()
        self.endpoint = f"grpc://127.0.0.1:{int(line.rsplit(':', 1)[1])}"

    def close(self) -> str:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
        self.process.wait(30)
        err = self.process.stderr.read()
        self.process.stdout.close()
        self.process.stderr.close()
        return err


# --- BUG-07 ---------------------------------------------------------------------------------------

def tq01_lines(run_id: str) -> list[dict]:
    """The three sessions' lines as they stand in the rank's file, in the capture's record shapes."""
    return [
        {"schema_version": 2, "record": "scalars", "run_id": run_id, "rank": 0, "attempt": A, "step": 50,
         "seq": 0, "ts": 10.0, "scalars": {"eval_loss": 99.0}},
        {"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": B, "step": 0, "ts": 20.0,
         "grad_norm": 1.0},
        {"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": C, "step": 100, "ts": 30.0,
         "grad_norm": 1.0},
    ]


def stored_order(store: Path, run_id: str) -> list[int]:
    document = json.loads((store / "runs" / run_id / "rank=0" / "attempts.json").read_text())
    return [entry["attempt"] for entry in document["attempts"]]


@needs_pyarrow
@needs_binary
@pytest.mark.parametrize("batch_steps", [1, 64])
def test_bug07_the_stored_attempt_order_and_series_do_not_depend_on_the_batch_size(tmp_path, batch_steps):
    from tensorquill import flight

    run_id = f"ao3_bug07_{batch_steps}"
    capture = tmp_path / "capture.jsonl"
    capture.write_text("".join(json.dumps(line) + "\n" for line in tq01_lines(run_id)))
    # The local reader: B's first step (0) rolls back A's point at 50.
    assert scalar_series(capture) == {}
    store = tmp_path / "store"
    server = Server(store)
    try:
        sender = flight.FlightSender(capture, server.endpoint, run_id, 0, batch_steps=batch_steps,
                                     await_attempt=True)
        sender.attempt = C
        assert sender.close(timeout=20.0) is True
        assert sender.acked_offset == capture.stat().st_size
    finally:
        server.close()
    assert stored_order(store, run_id) == [A, B, C]
    assert scalar_series(store / "runs" / run_id, rank=0) == {}
    # With every attempt's points, A's is there: it was stored, then rolled back by the order.
    assert scalar_series(store / "runs" / run_id, rank=0, all_attempts=True) == {"eval_loss": [[50, 99.0]]}


# --- BUG-08 ---------------------------------------------------------------------------------------

# (case, the file's entries, the attempts stored but not listed, the order read). The same table as
# `bug08_table` in `server/tests/wave_ao_ingest_test.rs`.
BUG08_TABLE = [
    ("claimed_newest_unlisted_above", [(200, "claim")], [300], [300, 200]),
    ("claimed_newest_unlisted_below", [(200, "claim")], [100], [100, 200]),
    ("rows_newest_unlisted_above", [(100, "claim"), (200, "rows")], [300], [100, 300, 200]),
    ("store_newest_unlisted_above", [(100, "store"), (200, "store")], [300], [100, 300, 200]),
    ("one_entry_by_rows_unlisted_below", [(200, "rows")], [50], [50, 200]),
    ("two_unlisted_either_side", [(100, "claim"), (200, "claim")], [300, 50], [100, 50, 300, 200]),
    ("an_empty_list", [], [5, 3], [3, 5]),
]


@pytest.mark.parametrize("case, entries, unlisted, expected", BUG08_TABLE, ids=[row[0] for row in BUG08_TABLE])
def test_bug08_python_places_an_unlisted_attempt_as_the_server_does(case, entries, unlisted, expected):
    from tensorquill.evidence import _attempt_place  # here, so a rename in that module fails these tests alone

    order = [attempt for attempt, _ in entries]
    attempts = order + unlisted
    assert sorted(attempts, key=lambda attempt: _attempt_place(order, attempt)) == expected


@pytest.mark.parametrize("case, entries, unlisted, expected", BUG08_TABLE, ids=[row[0] for row in BUG08_TABLE])
def test_bug08_python_lists_a_stores_parts_in_that_order(tmp_path, case, entries, unlisted, expected):
    from tensorquill.evidence import _parquet_parts

    rank = tmp_path / "runs" / "r" / "rank=0"
    for attempt in [a for a, _ in entries] + unlisted:
        folder = rank / f"attempt={attempt}"
        folder.mkdir(parents=True)
        (folder / "part-00000000-00000000.parquet").write_bytes(b"")
    rank.mkdir(parents=True, exist_ok=True)
    (rank / "attempts.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "r", "rank": 0,
        "attempts": [{"attempt": a, "by": by, "first_seen_ms": 0} for a, by in entries]}))
    parts = _parquet_parts(tmp_path / "runs" / "r", rank=0)
    assert [int(part.parent.name.split("=")[1]) for part in parts] == expected


# --- BUG-02, the server side ------------------------------------------------------------------------

def send_rows(server: Server, run_id: str, rows: list[dict], sender: int) -> list[int]:
    """`rows` (protocol 5's layout, `flight.flight_schema()`) as one batch, metadata version 2."""
    from tensorquill import flight

    schema = flight.flight_schema()
    client = pa_flight.FlightClient(server.endpoint)
    try:
        writer, reader = client.do_put(pa_flight.FlightDescriptor.for_path("runs", run_id, "rank=0"), schema)
        steps = [row["step"] for row in rows]
        metadata = {"schema_version": 2, "run_id": run_id, "rank": 0, "min_step": min(steps),
                    "max_step": max(steps), "end_byte_offset": len(rows), "sender_attempt": sender}
        writer.write_with_metadata(pa.RecordBatch.from_pylist(rows, schema=schema),
                                   pa.py_buffer(json.dumps(metadata).encode()))
        acked = [json.loads(reader.read().to_pybytes())["end_byte_offset"]]
        writer.done_writing()
        writer.close()
    finally:
        client.close()
    return acked


@needs_pyarrow
@needs_binary
def test_bug02_a_version_2_row_with_a_null_seq_is_dropped_and_the_store_stays_readable(tmp_path):
    from tensorquill import flight

    run_id, attempt = "ao3_bug02", 7
    records = [{"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": attempt, "step": step,
                "ts": float(step), "loss": 1.0} for step in (0, 1, 2, 4, 5)]
    batch, _, skipped = flight.rows_to_send(records, flight.flight_schema())
    assert skipped == 0
    rows = batch.to_pylist()
    # What an encoder makes of `{"schema_version": 2, ..., "step": 3, "loss": 1.0}` with no
    # `"record": "scalars"` (TQ-10): the row of a training step, with the record's version 2.
    bad = dict(rows[2], step=3, record_version=2, seq=None)
    rows.insert(3, bad)
    store = tmp_path / "store"
    server = Server(store)
    run = store / "runs" / run_id
    try:
        assert send_rows(server, run_id, rows, attempt) == [6]
        # A watchdog pass writes the rank's status (every 0.5 s here) before the server stops.
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                if json.loads((run / "watchdog.json").read_text())["ranks"][0].get("dropped_rows"):
                    break
            except (OSError, ValueError, KeyError, IndexError):
                pass
            time.sleep(0.05)
    finally:
        err = server.close()
    loaded = load_records(run)
    assert [record["step"] for record in loaded] == [0, 1, 2, 4, 5]
    assert f"{run_id} rank 0" in err and "1 with a null seq" in err, err
    status = json.loads((run / "watchdog.json").read_text())
    entry = status["ranks"][0]
    assert entry["dropped_rows"] == 1 and entry["dropped_row_keys"] == {"seq": 1}, entry
    steps = [s for part in sorted(run.rglob("part-*.parquet"))
             for s in pq.read_table(part, partitioning=None).column("step").to_pylist()]
    assert steps == [0, 1, 2, 4, 5]
