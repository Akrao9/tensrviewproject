"""What the protocol 4 tree does today with protocol 5's data, run against a real release server.

    PYTHONPATH=<worktree>/sdk/src TENSORQUILL_SERVER_BIN=<worktree>/server/target/release/tensorquill-server \
        .venv/bin/python experiments/protocol-v5/live.py --out experiments/protocol-v5/results/live.json

Each check starts its own server on a fresh store in a temporary folder, sends with a plain
pyarrow Flight client (`put`), and reads what the server stored. Nothing under sdk/src or
server/src is changed. The checks (WAVE_AL_CONTRACT.md, the "today" of each cell):

- `order`: layers in the model's order (`model.layers.2` before `model.layers.10`) as the prototype's
  typed map, sent by Flight, stored by the server as Parquet, read back by pyarrow and by the Rust
  reader in `rust-reader/` (the server's own `Storage::read_part`), then the server restarted on the
  store and a second batch sent, so the rank is rebuilt from the stored parts, and read again.
- `old_server_typed`: a protocol 4 server given the prototype layout with a v4-accepted row version
  (1): is it stored, and what `incidents.json` says of the layers.
- `old_server_row_v2`: the same rows with row `schema_version` 2: dropped and counted?
- `old_server_meta_v2`: batch metadata `schema_version` 2: what the sender is told.
- `old_python_on_v5_part`: today's `load_records` on a stored part of the prototype layout, with row
  version 1 and with 2.
- `watermark`: a row with a future step (as a scalar-only record would carry) among `part-*` files:
  what happens to the training rows sent after it.
- `struct_field_added`: two parts, `moe_stats` with and without a `token_entropy` field, read by one
  pyarrow reader and by the Rust reader by field name.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow as pa  # noqa: E402
import pyarrow.flight as flight  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import prototype as p5  # noqa: E402
from tensorquill import evidence  # noqa: E402
from tensorquill import flight as v4  # noqa: E402

HERE = Path(__file__).resolve().parent
BINARY = os.environ.get("TENSORQUILL_SERVER_BIN")
READER = HERE / "rust-reader" / "target" / "release" / "part-reader"


class Server:
    def __init__(self, store: Path, *extra: str) -> None:
        env = {k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"}
        self.store = store
        self.process = subprocess.Popen(
            [BINARY, "--listen", "127.0.0.1:0", "--store", str(store), "--watchdog-seconds", "0.5",
             "--flush-seconds", "0.3", *extra], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        line = self.process.stdout.readline()
        self.endpoint = f"grpc://127.0.0.1:{int(line.rsplit(':', 1)[1])}"

    def close(self) -> str:
        self.process.send_signal(signal.SIGTERM)
        self.process.wait(20)
        err = self.process.stderr.read()
        self.process.stdout.close()
        self.process.stderr.close()
        return err


def put(endpoint: str, run_id: str, rank: int, batches: list[pa.RecordBatch], meta_version: int = 1,
        attempt: int | None = None) -> list:
    """Send `batches` on one DoPut stream, one acknowledgement awaited per batch. Returns, per batch,
    the acknowledgement's JSON and the seconds from write to acknowledgement, or the error's text."""
    client = flight.FlightClient(endpoint)
    descriptor = flight.FlightDescriptor.for_path("runs", run_id, f"rank={rank}")
    writer, reader = client.do_put(descriptor, batches[0].schema)
    out, end = [], 0
    try:
        for batch in batches:
            steps = batch.column("step").to_pylist()
            end += 1000
            metadata = {"schema_version": meta_version, "run_id": run_id, "rank": rank, "min_step": min(steps),
                        "max_step": max(steps), "end_byte_offset": end, "start_byte_offset": end - 1000}
            if attempt is not None:
                metadata["sender_attempt"] = attempt
            started = time.perf_counter()
            writer.write_with_metadata(batch, pa.py_buffer(json.dumps(metadata).encode()))
            ack = reader.read()
            out.append({"ack": json.loads(ack.to_pybytes()) if ack is not None else None,
                        "seconds": time.perf_counter() - started})
        writer.done_writing()
        writer.close()
    except Exception as error:  # noqa: BLE001
        out.append({"error": f"{type(error).__name__}: {str(error).splitlines()[0]}"})
    finally:
        client.close()
    return out


def rank_entry(status: dict | None, rank: int = 0) -> dict:
    """The rank's entry of a run's `watchdog.json` (`ranks` is a list of entries)."""
    for entry in (status or {}).get("ranks") or []:
        if isinstance(entry, dict) and entry.get("rank") == rank:
            return entry
    return {}


def parts(store: Path, run_id: str) -> list[Path]:
    return sorted((store / "runs" / run_id).rglob("part-*.parquet"))


def rust_keys(path: Path, column: str) -> list:
    if not READER.exists():
        return ["rust reader not built"]
    text = subprocess.run([str(READER), "keys", str(path.parent.parent.parent.parent.parent), str(path), column],
                          capture_output=True, text=True, check=False)
    return json.loads(text.stdout) if text.returncode == 0 else [f"exit {text.returncode}: {text.stderr.strip()}"]


def layer_records(run_id: str, steps: range, names: list[str]) -> list[dict]:
    return [{"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": 1, "step": s, "ts": float(s),
             "layers": {n: {"mean": 0.0, "std": 1.0 + i, "abs_max": 3.0, "zero_frac": 0.0} for i, n in enumerate(names)},
             "grad_norm": 1.0} for s in steps]


FORWARD = [f"model.layers.{i}" for i in (0, 1, 2, 10, 11, 3, 20)]  # a model's order, not sorted


def check_order(tmp: Path) -> dict:
    store = tmp / "order"
    server = Server(store)
    run_id = "order"
    schema = p5.flight_schema_v5()
    first = pa.RecordBatch.from_pylist([p5.record_to_row_v5(r, "v4-accepted") for r in layer_records(run_id, range(0, 5), FORWARD)], schema=schema)
    acks = put(server.endpoint, run_id, 0, [first], attempt=1)
    time.sleep(1.0)
    server.close()
    stored = parts(store, run_id)
    on_part = {"pyarrow": [[k for k, _ in row] for row in pq.read_table(stored[0], partitioning=None).column("layers").to_pylist()][:1],
               "rust": rust_keys(stored[0], "layers")}
    # Restart on the same store: the rank's first part since the start rebuilds from every stored part.
    server = Server(store)
    second = pa.RecordBatch.from_pylist([p5.record_to_row_v5(r, "v4-accepted") for r in layer_records(run_id, range(5, 10), FORWARD)], schema=schema)
    acks += put(server.endpoint, run_id, 0, [second], attempt=1)
    time.sleep(1.5)
    report = json.loads(next((store / "runs" / run_id).rglob("incidents.json")).read_text())
    stderr = server.close()
    stored = parts(store, run_id)
    after = {str(p.name): {"pyarrow": [[k for k, _ in row] for row in pq.read_table(p, partitioning=None).column("layers").to_pylist()][:1],
                           "rust": rust_keys(p, "layers")} for p in stored}
    kept = all(v["pyarrow"] == [FORWARD] and v["rust"] == [FORWARD] for v in after.values()) and on_part["pyarrow"] == [FORWARD]
    return {"sent_order": FORWARD, "on_part": on_part, "after_restart": after, "acks": acks, "kept": kept,
            "rebuilds": report.get("rebuilds"), "report_rows": report.get("steps"), "stderr": stderr.strip()[-400:]}


def spike_records(run_id: str) -> list[dict]:
    """110 healthy steps, then a step where two blocks spike, the later-in-forward-order one sorted first."""
    out = []
    for s in range(120):
        spike = s == 115
        layers = {n: {"mean": 0.0, "std": 1.0 * (1 + 0.01 * (s % 3)), "abs_max": (50.0 if spike and n in ("model.layers.2", "model.layers.10") else 3.0 * (1 + 0.01 * (s % 5))),
                      "zero_frac": 0.0} for n in FORWARD}
        out.append({"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": 1, "step": s, "ts": float(s), "layers": layers, "grad_norm": 1.0})
    return out


def check_old_server_typed(tmp: Path) -> dict:
    """The v5 layout with a v4-accepted row version, to a v4 server: stored, but what is judged?"""
    store = tmp / "typed"
    server = Server(store)
    run_id = "typed"
    records = spike_records(run_id)
    schema = p5.flight_schema_v5()
    batches = [pa.RecordBatch.from_pylist([p5.record_to_row_v5(r, "v4-accepted") for r in records[i:i + 50]], schema=schema)
               for i in range(0, len(records), 50)]
    acks = put(server.endpoint, run_id, 0, batches, attempt=1)
    time.sleep(1.5)
    report = json.loads(next((store / "runs" / run_id).rglob("incidents.json")).read_text())
    server.close()
    python = evidence.build_evidence(records, detectors="python")
    return {"acks": [a.get("ack") or a.get("error") for a in acks], "parts": [p.name for p in parts(store, run_id)],
            "server_incidents": [(i["kind"], i.get("step"), i.get("layers")) for i in report.get("incidents", [])],
            "server_kind_checks_activation_spike": report.get("kind_checks", {}).get("activation_spike"),
            "python_on_the_records": [(i["kind"], i.get("step"), i.get("layers")) for i in python["incidents"]],
            "python_kind_checks_activation_spike": python["kind_checks"].get("activation_spike")}


def check_old_server_row_v2(tmp: Path) -> dict:
    store = tmp / "rowv2"
    server = Server(store)
    run_id = "rowv2"
    records = layer_records(run_id, range(0, 5), FORWARD)
    batch = pa.RecordBatch.from_pylist([p5.record_to_row_v5(r, "v5") for r in records], schema=p5.flight_schema_v5())
    acks = put(server.endpoint, run_id, 0, [batch], attempt=1)
    time.sleep(1.0)
    status = json.loads((store / "runs" / run_id / "watchdog.json").read_text()) if (store / "runs" / run_id / "watchdog.json").exists() else None
    stderr = server.close()
    rank = rank_entry(status)
    return {"acks": acks, "parts": [p.name for p in parts(store, run_id)],
            "dropped_rows": rank.get("dropped_rows"), "dropped_row_keys": rank.get("dropped_row_keys"),
            "stderr": stderr.strip()[-300:]}


def check_old_server_meta_v2(tmp: Path) -> dict:
    store = tmp / "metav2"
    server = Server(store)
    run_id = "metav2"
    records = layer_records(run_id, range(0, 5), FORWARD)
    batch = pa.RecordBatch.from_pylist([p5.record_to_row_v5(r, "v5") for r in records], schema=p5.flight_schema_v5())
    acks = put(server.endpoint, run_id, 0, [batch], meta_version=2, attempt=1)
    time.sleep(0.5)
    stderr = server.close()
    return {"acks": acks, "parts": [p.name for p in parts(store, run_id)], "stderr": stderr.strip()[-300:]}


def check_old_python_on_v5_part(tmp: Path) -> dict:
    out = {}
    records = spike_records("oldpy")
    for layout in ("v4-accepted", "v5"):
        folder = tmp / f"oldpy-{layout}" / "runs" / "oldpy" / "rank=0" / "attempt=1"
        folder.mkdir(parents=True)
        table = pa.Table.from_batches([p5.batch_v5(records, layout)])
        pq.write_table(table, folder / "part-00000000-00000119.parquet", compression="zstd")
        try:
            loaded = evidence.load_records(folder.parent.parent)
            built = evidence.build_evidence(folder.parent.parent, detectors="python")
            out[layout] = {"loaded": len(loaded), "layers_in_first_record": "layers" in loaded[0],
                           "incidents": [(i["kind"], i.get("step")) for i in built["incidents"]],
                           "kind_checks_activation_spike": built["kind_checks"].get("activation_spike")}
        except Exception as error:  # noqa: BLE001
            out[layout] = {"raised": f"{type(error).__name__}: {error}"}
    out["the_same_records_from_jsonl"] = [(i["kind"], i.get("step")) for i in evidence.build_evidence(records, detectors="python")["incidents"]]
    return out


def check_watermark(tmp: Path) -> dict:
    """A row with a future step in a part (a scalar-only record logged with step=1000 at step 3), then
    the training rows of steps 4 to 9: what is stored and acknowledged."""
    store = tmp / "watermark"
    server = Server(store)
    run_id = "watermark"
    rows = layer_records(run_id, range(0, 4), FORWARD)
    scalar = {"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": 1, "step": 1000, "ts": 3.5,
              "scalars": {"eval/loss": 1.5}, "grad_norm": None}
    later = layer_records(run_id, range(4, 10), FORWARD)
    acks = put(server.endpoint, run_id, 0, [v4.records_to_batch(rows + [scalar])], attempt=1)
    time.sleep(1.0)
    stderr1 = server.close()
    server = Server(store)  # a restart: the watermark is read back from the parts' footers
    acks += put(server.endpoint, run_id, 0, [v4.records_to_batch(later)], attempt=1)
    time.sleep(1.0)
    status = json.loads((store / "runs" / run_id / "watchdog.json").read_text())
    stderr2 = server.close()
    stored = [r["step"] for p in parts(store, run_id) for r in pq.read_table(p, partitioning=None).select(["step"]).to_pylist()]
    rank = rank_entry(status)
    return {"acks": [a.get("ack") or a.get("error") for a in acks], "stored_steps": stored,
            "rank_counts": {k: v for k, v in rank.items() if any(w in k for w in ("newer", "late", "dropped", "step"))},
            "lost": [s for s in range(4, 10) if s not in stored], "stderr": (stderr1 + stderr2).strip()[-300:]}


def check_struct_field_added(tmp: Path) -> dict:
    folder = tmp / "fieldadded" / "runs" / "f" / "rank=0" / "attempt=1"
    folder.mkdir(parents=True)
    rec = lambda s, te: {"schema_version": 1, "run_id": "f", "rank": 0, "attempt": 1, "step": s, "ts": 0.0,  # noqa: E731
                         "moe_stats": {"r": {"routed_tokens": 8, "entropy": 1.0, "diversity": 0.5, "dead_experts": 0,
                                             "dead_frac": 0.0, **({"token_entropy": 0.7} if te else {})}}}
    without = pa.Table.from_batches([p5.batch_v5([rec(0, False)], "v4-accepted", p5.flight_schema_v5(False))])
    with_te = pa.Table.from_batches([p5.batch_v5([rec(1, True)], "v4-accepted", p5.flight_schema_v5(True))])
    pq.write_table(without, folder / "part-00000000-00000000.parquet", compression="zstd")
    pq.write_table(with_te, folder / "part-00000001-00000001.parquet", compression="zstd")
    read = {}
    for path in sorted(folder.glob("*.parquet")):
        table = pq.read_table(path, partitioning=None)
        entry = table.column("moe_stats").to_pylist()[0][0][1]
        read[path.name] = {"pyarrow_by_name": {k: entry.get(k, "absent") for k in ("diversity", "token_entropy")},
                           "rust_by_name": rust_keys(path, "moe_stats.token_entropy")}
    return read


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--only")
    args = parser.parse_args()
    if not BINARY:
        raise SystemExit("set TENSORQUILL_SERVER_BIN to the protocol 4 release binary built from this worktree")
    protocol = subprocess.run([BINARY, "protocol"], capture_output=True, text=True).stdout.strip()
    checks = {"order": check_order, "old_server_typed": check_old_server_typed, "old_server_row_v2": check_old_server_row_v2,
              "old_server_meta_v2": check_old_server_meta_v2, "old_python_on_v5_part": check_old_python_on_v5_part,
              "watermark": check_watermark, "struct_field_added": check_struct_field_added}
    report = {"binary_protocol": protocol}
    with tempfile.TemporaryDirectory() as tmp:
        for name, check in checks.items():
            if args.only and name not in args.only.split(","):
                continue
            report[name] = check(Path(tmp))
            print(name, json.dumps(report[name], default=str)[:1500])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
