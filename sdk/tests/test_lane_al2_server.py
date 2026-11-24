"""Lane AL-2 of wave AL (`WAVE_AL_CONTRACT.md` sections 2.4, 3 and 5): the real server binary takes
protocol 5's layout beside protocol 4's, and keeps the `layers` map's entry order across a restart.

- `layers_model_order` on a live server: the case's records in protocol 5's layout (typed `layers`,
  row version 2) under batch metadata 2 are stored with the model's entry order on the part; after a
  restart the server takes the next steps, drops the ones already stored, rebuilds the rank from
  every part, and every part still holds that order.
- A protocol 4 sender (metadata 1, JSON columns) on the same server, followed by protocol 5's rows:
  each layout is stored as received in parts of its own, and a restart replays them all.

Both layouts are built here from the contract's column definitions (sections 1.2 and 2.4), not from
`flight.py`, whose layout lane AL-1 changes: a protocol 4 sender is what senders send today. No test
here depends on a layer being judged from a typed column (lane AL-3's reader).

Needs pyarrow and the server binary (`TENSORQUILL_SERVER_BIN`, or one beside the interpreter); skipped
without either, as on CI's plain jobs.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
flight = pytest.importorskip("pyarrow.flight")
pq = pytest.importorskip("pyarrow.parquet")

from tensorquill.server_detect import find_binary  # noqa: E402

BINARY = find_binary()
pytestmark = pytest.mark.skipif(
    not BINARY or not os.path.isfile(BINARY), reason="tensorquill-server is not built"
)

CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_al" / "cases.json").read_text())["cases"]
ORDER_CASE = next(case for case in CASES if case["name"] == "layers_model_order")
LAYER_FIELDS = ("mean", "std", "abs_max", "zero_frac")


def _base_fields() -> list:
    return [
        pa.field("schema_version", pa.int32()),
        pa.field("run_id", pa.string()),
        pa.field("rank", pa.int32()),
        pa.field("attempt", pa.int64()),
        pa.field("step", pa.int64()),
        pa.field("ts", pa.float64()),
        pa.field("grad_norm", pa.float64()),
    ]


def schema_v5():
    """Protocol 5's layout (sections 1.2, 2.4 and 3)."""
    f64, i64 = pa.float64(), pa.int64()
    layer = pa.struct([pa.field(name, f64) for name in LAYER_FIELDS])
    moe = pa.struct([pa.field("routed_tokens", i64), pa.field("entropy", f64), pa.field("diversity", f64),
                     pa.field("dead_experts", i64), pa.field("dead_frac", f64)])
    histogram = pa.struct([pa.field("lo", f64), pa.field("hi", f64), pa.field("counts", pa.list_(i64)),
                           pa.field("nonfinite", i64)])
    return pa.schema(_base_fields() + [
        pa.field("layers", pa.map_(pa.string(), layer)),
        pa.field("moe_stats", pa.map_(pa.string(), moe)),
        pa.field("probe_json", pa.string()),
        pa.field("probe_maps", pa.map_(pa.string(), pa.list_(pa.list_(pa.float32())))),
        pa.field("probe_histograms", pa.map_(pa.string(), histogram)),
        pa.field("seq", i64),
        pa.field("record_version", pa.int32()),
        pa.field("extra_json", pa.string()),
    ])


def schema_v4():
    """Protocol 4's layout: the statistics as JSON text."""
    return pa.schema(_base_fields() + [
        pa.field("layers_json", pa.string()),
        pa.field("moe_stats_json", pa.string()),
        pa.field("probe_json", pa.string()),
        pa.field("extra_json", pa.string()),
    ])


def _identity(record: dict, step: int) -> dict:
    return {"run_id": record["run_id"], "rank": record["rank"], "attempt": record["attempt"], "step": step,
            "ts": float(step), "grad_norm": record["grad_norm"]}


def batch_v5(records: list[dict]):
    rows = []
    for record in records:
        rows.append({
            "schema_version": 2, **_identity(record, record["step"]),
            "layers": [(name, {f: stats[f] for f in LAYER_FIELDS}) for name, stats in record["layers"].items()],
            "moe_stats": None, "probe_json": None, "probe_maps": None, "probe_histograms": None,
            "seq": None, "record_version": record["schema_version"], "extra_json": None,
        })
    return pa.RecordBatch.from_pylist(rows, schema=schema_v5())


def batch_v4(records: list[dict]):
    rows = [{"schema_version": record["schema_version"], **_identity(record, record["step"]),
             "layers_json": json.dumps(record["layers"]), "moe_stats_json": None, "probe_json": None,
             "extra_json": None} for record in records]
    return pa.RecordBatch.from_pylist(rows, schema=schema_v4())


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


def put(server: Server, run_id: str, batch, metadata_version: int, end: int) -> dict:
    """Sends `batch` on a stream of its own and returns the acknowledgement's JSON."""
    client = flight.FlightClient(server.endpoint)
    try:
        descriptor = flight.FlightDescriptor.for_path("runs", run_id, "rank=0")
        writer, reader = client.do_put(descriptor, batch.schema)
        steps = batch.column("step").to_pylist()
        metadata = {"schema_version": metadata_version, "run_id": run_id, "rank": 0, "min_step": min(steps),
                    "max_step": max(steps), "end_byte_offset": end}
        writer.write_with_metadata(batch, pa.py_buffer(json.dumps(metadata).encode()))
        ack = reader.read()
        writer.done_writing()
        writer.close()
    finally:
        client.close()
    return json.loads(ack.to_pybytes())


def parts(store: Path, run_id: str) -> list[Path]:
    return sorted((store / "runs" / run_id).rglob("part-*.parquet"))


def layer_keys(path: Path) -> list[list[str]]:
    table = pq.read_table(path, partitioning=None)
    return [[name for name, _ in row] for row in table.column("layers").to_pylist()]


def report_through(store: Path, run_id: str, last_step: int) -> dict:
    path = store / "runs" / run_id / "rank=0" / "incidents.json"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError):
            report = None
        if report and report.get("last_step") == last_step:
            return report
        time.sleep(0.05)
    raise AssertionError(f"no report through step {last_step}: {path.read_text() if path.exists() else None}")


def test_layers_model_order_holds_on_the_part_and_across_a_restart(tmp_path):
    records = ORDER_CASE["records"]
    run_id = records[0]["run_id"]
    order = list(records[0]["layers"])
    assert order != sorted(order), "the case's order is a model's, not a sorted one"
    store = tmp_path / "store"
    server = Server(store)
    try:
        assert put(server, run_id, batch_v5(records[:25]), 2, 25)["end_byte_offset"] == 25
        report_through(store, run_id, 24)
    finally:
        server.close()
    stored = parts(store, run_id)
    assert len(stored) == 1
    assert pq.read_schema(stored[0]).equals(schema_v5(), check_metadata=False)
    assert layer_keys(stored[0]) == [order] * 25
    # The restart: the next batch repeats steps 20 to 24, which are dropped (the watermark is read from
    # the part's name and footer), and the rank's first part since the start rebuilds from every part.
    server = Server(store)
    try:
        assert put(server, run_id, batch_v5(records[20:]), 2, 40)["end_byte_offset"] == 40
        report = report_through(store, run_id, 39)
    finally:
        server.close()
    assert report["rebuilds"] == 1, report
    assert report["steps"] == 40, report
    stored = parts(store, run_id)
    assert len(stored) == 2
    assert pq.read_table(stored[1], partitioning=None).column("step").to_pylist() == list(range(25, 40))
    for part in stored:
        assert all(keys == order for keys in layer_keys(part)), part


def test_a_protocol_4_sender_and_protocol_5_rows_have_parts_of_their_own_and_a_restart_reads_both(tmp_path):
    records = ORDER_CASE["records"]
    run_id = records[0]["run_id"]
    store = tmp_path / "store"
    server = Server(store)
    try:
        old = batch_v4(records[:10])
        assert put(server, run_id, old, 1, 10)["end_byte_offset"] == 10
        assert put(server, run_id, batch_v5(records[10:20]), 2, 20)["end_byte_offset"] == 20
        report_through(store, run_id, 19)
    finally:
        server.close()
    stored = parts(store, run_id)
    assert len(stored) == 2
    first = pq.read_table(stored[0], partitioning=None)
    assert first.schema.equals(schema_v4(), check_metadata=False)
    assert first.to_batches()[0].equals(old), "a protocol 4 batch is stored as received"
    assert pq.read_schema(stored[1]).equals(schema_v5(), check_metadata=False)
    server = Server(store)
    try:
        assert put(server, run_id, batch_v5(records[15:30]), 2, 30)["end_byte_offset"] == 30
        report = report_through(store, run_id, 29)
    finally:
        server.close()
    assert report["rebuilds"] == 1, report
    assert report["steps"] == 30, report
    steps = [s for part in parts(store, run_id) for s in pq.read_table(part, partitioning=None).column("step").to_pylist()]
    assert steps == list(range(30))
