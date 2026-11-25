"""Lane AL-2′, the owner's fix of Nov 25, 2026, through a real server: under batch metadata
`schema_version` 1 the server reads row versions 0 and 1 only.

A protocol 4 sender given a `.jsonl` that a protocol 5 capture wrote sends its scalar-only record as a
step row of row version 2, with no `seq` column, under metadata 1 (lane AL-0's run across a real
protocol 4 build). With training steps 0 to 9, that record at step 500, then steps 10 to 29, the
server must store all 30 training steps and nothing at 500, give `last_step` 29, count the row once
under `unsupported_schema_version` with the stderr line every dropped row gets, acknowledge the
sender to the end, and judge steps 0 to 29. The owner's words: "Drop and count."

The batches are protocol 4's row layout built here with pyarrow (the columns of
`experiments/protocol-v5/flight_v4.py` these tests need). Needs pyarrow and the server binary
(`TENSORQUILL_SERVER_BIN`, or one beside the interpreter); each test is skipped without either.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from tensorquill.server_detect import find_binary

try:  # each test is collected and skipped without pyarrow (a module skipped whole collects nothing)
    import pyarrow as pa
    import pyarrow.flight as flight
    import pyarrow.parquet as pq
except ImportError:
    pa = flight = pq = None

BINARY = find_binary()
pytestmark = [
    pytest.mark.skipif(pa is None, reason="pyarrow is not installed"),
    pytest.mark.skipif(not BINARY or not os.path.isfile(BINARY), reason="tensorquill-server is not built"),
]

ATTEMPT = 1_732_460_000_000
DROPPED_LINE = "dropped rows that cannot be stored: 1 of a schema_version this server does not read"


def schema_v4():
    """Protocol 4's row layout: the keys not nullable, `scalars` a map, the statistics JSON text."""
    return pa.schema([
        pa.field("schema_version", pa.int32(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("rank", pa.int32(), nullable=False),
        pa.field("attempt", pa.int64(), nullable=False),
        pa.field("step", pa.int64(), nullable=False),
        pa.field("ts", pa.float64()),
        pa.field("grad_norm", pa.float64()),
        pa.field("scalars", pa.map_(pa.string(), pa.float64())),
        pa.field("layers_json", pa.string()),
        pa.field("moe_stats_json", pa.string()),
        pa.field("probe_json", pa.string()),
        pa.field("extra_json", pa.string()),
    ])


def step_row(run_id: str, step: int) -> dict:
    return {"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": ATTEMPT, "step": step,
            "ts": float(step), "grad_norm": 1.0}


def scalar_record_row(run_id: str, step: int) -> dict:
    """What a protocol 4 sender makes of a protocol 5 capture's scalar-only record: row version 2 (the
    record's own), its `scalars`; no column for `seq` or `record`."""
    return {"schema_version": 2, "run_id": run_id, "rank": 0, "attempt": ATTEMPT, "step": step, "ts": 500.0,
            "scalars": [("eval/loss", 1.4)]}


def scenario(run_id: str) -> list[dict]:
    return [step_row(run_id, s) for s in range(10)] + [scalar_record_row(run_id, 500)] + \
        [step_row(run_id, s) for s in range(10, 30)]


class Server:
    def __init__(self, store: Path) -> None:
        env = {k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"}
        self.process = subprocess.Popen(
            [BINARY, "--listen", "127.0.0.1:0", "--store", str(store), "--watchdog-seconds", "0.5",
             "--flush-seconds", "0.2"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        line = self.process.stdout.readline()
        assert "listening on" in line, line + self.process.stderr.read()
        self.endpoint = f"grpc://127.0.0.1:{int(line.rsplit(':', 1)[1])}"

    def close(self) -> str:
        self.process.send_signal(signal.SIGTERM)
        self.process.wait(30)
        err = self.process.stderr.read()
        self.process.stdout.close()
        self.process.stderr.close()
        return err


def send(server: Server, run_id: str, batches: list[list[dict]]) -> list[int]:
    """Sends each list of rows as one batch of protocol 4's layout on one stream, metadata version 1,
    each batch's end offset its place in the file (one line a row); returns the offsets acknowledged."""
    client = flight.FlightClient(server.endpoint)
    acked, end = [], 0
    try:
        descriptor = flight.FlightDescriptor.for_path("runs", run_id, "rank=0")
        writer, reader = client.do_put(descriptor, schema_v4())
        for rows in batches:
            end += len(rows)
            steps = [row["step"] for row in rows]
            metadata = {"schema_version": 1, "run_id": run_id, "rank": 0, "min_step": min(steps),
                        "max_step": max(steps), "end_byte_offset": end}
            batch = pa.RecordBatch.from_pylist(rows, schema=schema_v4())
            writer.write_with_metadata(batch, pa.py_buffer(json.dumps(metadata).encode()))
            acked.append(json.loads(reader.read().to_pybytes())["end_byte_offset"])
        writer.done_writing()
        writer.close()
    finally:
        client.close()
    return acked


def stored_steps(store: Path, run_id: str) -> list[int]:
    folder = store / "runs" / run_id / "rank=0" / f"attempt={ATTEMPT}"
    steps = []
    for part in sorted(folder.glob("part-*.parquet")):
        steps.extend(pq.read_table(part, partitioning=None).column("step").to_pylist())
    return steps


def wait_for(path: Path, wanted, what: str) -> dict:
    deadline = time.monotonic() + 20
    document = None
    while time.monotonic() < deadline:
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            document = None
        if document is not None and wanted(document):
            return document
        time.sleep(0.05)
    raise AssertionError(f"{what} never showed it: {document}")


def run_scenario(tmp_path: Path, run_id: str, batches: list[list[dict]]) -> None:
    store = tmp_path / "store"
    server = Server(store)
    try:
        acked = send(server, run_id, batches)
        assert acked[-1] == 31, f"acknowledged to the end of the file: {acked}"
        report = wait_for(store / "runs" / run_id / "rank=0" / "incidents.json",
                          lambda report: report.get("last_step") == 29, "incidents.json")
        status = wait_for(store / "runs" / run_id / "watchdog.json",
                          lambda status: status["ranks"] and status["ranks"][0].get("last_step") == 29,
                          "watchdog.json")
    finally:
        err = server.close()
    assert stored_steps(store, run_id) == list(range(30)), "every training step and nothing at 500"
    assert report["steps"] == 30, report
    entry = status["ranks"][0]
    assert entry["dropped_rows"] == 1, entry
    assert entry["dropped_row_keys"] == {"unsupported_schema_version": 1}, entry
    assert "rows_not_newer" not in entry, entry
    assert f"{run_id} rank 0: {DROPPED_LINE}" in err, err


def test_the_owners_scenario_in_one_batch_drops_and_counts_the_version_2_row(tmp_path):
    run_id = "al2b_v2v1_py_one"
    run_scenario(tmp_path, run_id, [scenario(run_id)])


def test_the_version_2_row_alone_in_its_batch_is_dropped_and_counted(tmp_path):
    run_id = "al2b_v2v1_py_alone"
    rows = scenario(run_id)
    run_scenario(tmp_path, run_id, [rows[:10], rows[10:11], rows[11:]])


def test_the_version_2_row_first_and_last_in_its_batch_is_dropped_and_counted(tmp_path):
    for where, split in (("first", 10), ("last", 11)):
        run_id = f"al2b_v2v1_py_{where}"
        rows = scenario(run_id)
        run_scenario(tmp_path / where, run_id, [rows[:split], rows[split:]])
