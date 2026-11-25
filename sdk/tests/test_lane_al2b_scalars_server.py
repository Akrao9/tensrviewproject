"""Lane AL-2′ of wave AL (`WAVE_AL_CONTRACT.md` section 4.5): the real server binary stores
scalar-only rows apart from the training rows.

- `scalar_future_step_keeps_training_rows` (D8): a scalar-only row of step 1000 among training steps
  is stored in `scalars-00000000-00000000.parquet`, never in a `part-*` file, so it does not move the
  step watermark: every training step is stored, across a restart too.
- `scalar_replay_from_byte_0`: the whole file sent again stores nothing twice, and the scalar rows at
  or below the attempt's `seq` watermark are counted as `scalars_not_newer` in `watchdog.json`.
- `scalar_seq_watermark_across_restart`: the `seq` watermark is read back from the objects' names.
- A row of a future step with `seq` set, alone in its batch, leaves the step watermark and the
  rank's last step alone (D8).
- A batch without a `seq` column is handled as before: every row a step row, in a part.

The rows are the cases' records laid out as section 4.4 says, built here: a training record as
`flight.record_to_row` makes it, with a null `seq`; a scalar-only record as the same row of its
fields, with `record_version` 2 and its `seq`. Lane AL-1′ adds `seq` to the sender in parallel; this
file does not use the sender.

Needs pyarrow and the server binary (`TENSORQUILL_SERVER_BIN`, or one beside the interpreter); each
test is skipped without either, as on CI's plain jobs.
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

CASES = {case["name"]: case for case in
         json.loads((Path(__file__).parent / "fixtures" / "wave_al" / "cases.json").read_text())["cases"]}


def _schema(with_seq: bool = True):
    from tensorquill import flight as tq_flight

    fields = [field for field in tq_flight.flight_schema() if field.name != "seq"]
    if with_seq:
        fields.append(pa.field("seq", pa.int64()))
    return pa.schema(fields)


def _row(record: dict) -> dict:
    """A record of the cases as its row (section 4.4)."""
    from tensorquill import flight as tq_flight

    if record.get("record") == "scalars":
        fields = {key: value for key, value in record.items() if key not in ("record", "seq")}
        row = tq_flight.record_to_row({**fields, "schema_version": 1})
        row["record_version"] = record["schema_version"]
        row["seq"] = record["seq"]
        return row
    row = tq_flight.record_to_row(record)
    row["seq"] = None
    return row


def batch(records: list[dict], with_seq: bool = True):
    rows = [_row(record) for record in records]
    if not with_seq:
        for row in rows:
            row.pop("seq")
    return pa.RecordBatch.from_pylist(rows, schema=_schema(with_seq))


def step_record(run_id: str, step: int) -> dict:
    return {"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": 1, "step": step, "ts": float(step),
            "grad_norm": 1.0}


def scalar_record(run_id: str, step: int, seq: int) -> dict:
    return {"schema_version": 2, "record": "scalars", "run_id": run_id, "rank": 0, "attempt": 1, "step": step,
            "seq": seq, "ts": 100.0 + seq, "scalars": {"eval/loss": 1.0 + seq}}


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


def put(server: Server, run_id: str, rows, end: int) -> dict:
    """Sends `rows` on a stream of its own, metadata version 2, and returns the acknowledgement's JSON."""
    client = flight.FlightClient(server.endpoint)
    try:
        descriptor = flight.FlightDescriptor.for_path("runs", run_id, "rank=0")
        writer, reader = client.do_put(descriptor, rows.schema)
        steps = rows.column("step").to_pylist()
        metadata = {"schema_version": 2, "run_id": run_id, "rank": 0, "min_step": min(steps),
                    "max_step": max(steps), "end_byte_offset": end}
        writer.write_with_metadata(rows, pa.py_buffer(json.dumps(metadata).encode()))
        ack = reader.read()
        writer.done_writing()
        writer.close()
    finally:
        client.close()
    return json.loads(ack.to_pybytes())


def attempt_dir(store: Path, run_id: str) -> Path:
    return store / "runs" / run_id / "rank=0" / "attempt=1"


def names(store: Path, run_id: str, prefix: str) -> list[str]:
    folder = attempt_dir(store, run_id)
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.name.startswith(prefix) and p.name.endswith(".parquet"))


def column(store: Path, run_id: str, prefix: str, name: str) -> list:
    values = []
    for file in names(store, run_id, prefix):
        values.extend(pq.read_table(attempt_dir(store, run_id) / file, partitioning=None).column(name).to_pylist())
    return values


def rank_status(store: Path, run_id: str, wanted) -> dict:
    """The rank's entry in `watchdog.json` once `wanted(entry)` holds."""
    path = store / "runs" / run_id / "watchdog.json"
    deadline = time.monotonic() + 20
    entry = None
    while time.monotonic() < deadline:
        try:
            entry = json.loads(path.read_text())["ranks"][0]
        except (OSError, ValueError, KeyError, IndexError):
            entry = None
        if entry is not None and wanted(entry):
            return entry
        time.sleep(0.05)
    raise AssertionError(f"watchdog.json never showed it: {entry}")


def test_scalar_future_step_keeps_training_rows(tmp_path):
    case = CASES["scalar_future_step_keeps_training_rows"]
    records = case["records"]
    run_id = records[0]["run_id"]
    store = tmp_path / "store"
    server = Server(store)
    try:
        assert put(server, run_id, batch(records[:5]), 5)["end_byte_offset"] == 5
        assert put(server, run_id, batch(records[5:8]), 8)["end_byte_offset"] == 8
        entry = rank_status(store, run_id, lambda entry: entry.get("last_step") is not None)
        assert entry["last_step"] == 6, entry
    finally:
        server.close()
    server = Server(store)
    try:
        assert put(server, run_id, batch(records[5:]), 11)["end_byte_offset"] == 11
        entry = rank_status(store, run_id, lambda entry: entry.get("last_step") == 9)
    finally:
        server.close()
    expect = case["expect"]
    assert column(store, run_id, "part-", "step") == expect["stored_training_steps"]
    assert column(store, run_id, "scalars-", "seq") == expect["stored_scalar_seqs"]
    assert set(column(store, run_id, "part-", "seq")) == {None}, "no scalar row is in a part (D8)"
    assert names(store, run_id, "part-"), "the training rows are in part-* files"
    assert names(store, run_id, "scalars-") == [name for name in expect["objects"] if name != "part-*"]
    assert "scalars_not_newer" not in entry, entry


def test_scalar_replay_from_byte_0(tmp_path):
    case = CASES["scalar_replay_from_byte_0"]
    records = case["records"]
    run_id = records[0]["run_id"]
    store = tmp_path / "store"
    server = Server(store)
    try:
        assert put(server, run_id, batch(records), 5)["end_byte_offset"] == 5
        assert case["expect"]["sent_twice"]
        assert put(server, run_id, batch(records), 5)["end_byte_offset"] == 5
        entry = rank_status(store, run_id, lambda entry: "scalars_not_newer" in entry)
    finally:
        server.close()
    assert entry["scalars_not_newer"] == case["expect"]["scalars_not_newer"], entry
    assert entry["rows_not_newer"] == 3, entry
    assert column(store, run_id, "scalars-", "seq") == case["expect"]["stored_scalar_seqs"]
    assert column(store, run_id, "part-", "step") == [0, 1, 2]


def test_scalar_seq_watermark_across_restart(tmp_path):
    case = CASES["scalar_seq_watermark_across_restart"]
    records = case["records"]
    run_id = records[0]["run_id"]
    store = tmp_path / "store"
    server = Server(store)
    try:
        assert put(server, run_id, batch(records[:2]), 2)["end_byte_offset"] == 2
    finally:
        server.close()
    server = Server(store)
    try:
        assert put(server, run_id, batch(records), 3)["end_byte_offset"] == 3
        entry = rank_status(store, run_id, lambda entry: "scalars_not_newer" in entry)
    finally:
        server.close()
    assert column(store, run_id, "scalars-", "seq") == case["expect"]["stored_scalar_seqs"]
    assert names(store, run_id, "scalars-") == ["scalars-00000000-00000001.parquet",
                                                "scalars-00000002-00000002.parquet"]
    assert names(store, run_id, "part-") == []
    assert entry["scalars_not_newer"] == 2, entry


def test_a_row_of_a_future_step_with_seq_set_leaves_the_step_watermark_alone(tmp_path):
    run_id = "al2b_d8_py"
    store = tmp_path / "store"
    server = Server(store)
    try:
        assert put(server, run_id, batch([step_record(run_id, s) for s in range(4)]), 4)["end_byte_offset"] == 4
        assert put(server, run_id, batch([scalar_record(run_id, 1000, 0)]), 5)["end_byte_offset"] == 5
        entry = rank_status(store, run_id, lambda entry: entry.get("last_step") is not None)
        assert entry["last_step"] == 3, entry
    finally:
        server.close()
    server = Server(store)
    try:
        assert put(server, run_id, batch([step_record(run_id, s) for s in (4, 5)]), 7)["end_byte_offset"] == 7
        rank_status(store, run_id, lambda entry: entry.get("last_step") == 5)
    finally:
        server.close()
    assert column(store, run_id, "part-", "step") == [0, 1, 2, 3, 4, 5]
    assert column(store, run_id, "scalars-", "step") == [1000]


def test_a_batch_without_a_seq_column_is_handled_as_before(tmp_path):
    """Every row a step row, stored as received in a part, under the step watermark, as a step 1
    sender's always was: a row of a future step sent so raises the watermark."""
    run_id = "al2b_no_seq_py"
    store = tmp_path / "store"
    server = Server(store)
    try:
        first = batch([step_record(run_id, s) for s in (0, 1, 2, 1000)], with_seq=False)
        assert "seq" not in first.schema.names
        assert put(server, run_id, first, 4)["end_byte_offset"] == 4
        assert put(server, run_id, batch([step_record(run_id, s) for s in (3, 4)], with_seq=False), 6)[
            "end_byte_offset"] == 6
        entry = rank_status(store, run_id, lambda entry: entry.get("rows_not_newer") == 2)
    finally:
        server.close()
    assert names(store, run_id, "part-") == ["part-00000000-00001000.parquet"]
    stored = pq.read_table(attempt_dir(store, run_id) / "part-00000000-00001000.parquet", partitioning=None)
    assert stored.to_batches()[0].equals(first), "stored as received"
    assert names(store, run_id, "scalars-") == []
    assert entry["last_step"] == 1000 and "scalars_not_newer" not in entry, entry
