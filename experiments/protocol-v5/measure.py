"""Protocol 5's before-numbers on this machine's CPU (WAVE_AL_CONTRACT.md, section 6).

    cd <worktree>
    cargo build --release --manifest-path server/Cargo.toml
    cargo build --release --offline --manifest-path experiments/protocol-v5/rust-reader/Cargo.toml
    PYTHONPATH=$PWD/sdk/src TENSORQUILL_SERVER_BIN=$PWD/server/target/release/tensorquill-server \
        .venv/bin/python experiments/protocol-v5/measure.py ack --rounds 5 --out experiments/protocol-v5/results/ack.json
    ... measure.py decode --rounds 5 --out experiments/protocol-v5/results/decode.json

Every timing is repeated `--rounds` times (5 or more), the arms interleaved within each round, and
each round records the one-minute load average before it starts. Medians are reported with the
lowest and highest value. No product code is changed: the "v5" arm is `prototype.py`'s layout,
sent to and stored by the protocol 4 release binary with a row version that binary accepts (1).

`ack`: the real probe records of `dense-layer8/dense.jsonl` (400 steps, a probe at steps 99, 199,
299 and 399, each 1.7 MB of JSON of which 98% is the 32 attention maps). Three arms: `v4` (the
probe in `probe_json`, as today), `v5` (maps and histograms in the prototype's typed columns, the
rest in `probe_json`), `none` (the same records with no probe at all). Per round and arm: a fresh
server (`--flush-steps 50`, so each 50-step batch is one part, written and acknowledged before the
next is taken) and one DoPut stream of eight 50-step batches; the time from writing a batch to its
acknowledgement. A probe batch's acknowledgement waits for the part's Parquet write; the batch after
it also waits for the detectors to have read the probe part (they run after the acknowledgement,
holding the rank). Then, for the stored parts, the Parquet write alone, as `Storage::write_part`
does it, timed in `rust-reader` (`part-reader write`), and the parts' bytes.

`decode`: what the Rust detectors pay to read the layer and router statistics, from the JSON columns
(`detect.rs`' own visitors, copied into `rust-reader`) or from the prototype's typed maps, on the
same synthetic Granite-shaped records `sdk/examples/benchmark_detectors.py` writes (24 blocks, 32
experts, top-8, from a real capture of its toy model); plus `detect_store --bench` (the server's own
example) on a 64-rank store of those records for the detectors' whole time, as the README measured it;
and Python's `_read_part` (`load_records` on a store) for both layouts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "sdk" / "examples"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import prototype as p5  # noqa: E402
from live import Server, put  # noqa: E402
from tensorquill import flight as v4  # noqa: E402

READER = HERE / "rust-reader" / "target" / "release" / "part-reader"
DENSE = Path.home() / "Downloads" / "runs" / "dense-layer8" / "dense.jsonl"


def spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}


def load() -> float:
    return round(os.getloadavg()[0], 2)


def reader(*args: str) -> object:
    done = subprocess.run([str(READER), *args], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


# --- ack ------------------------------------------------------------------------------------------


def ack_round(records: list[dict], arm: str, tmp: Path, index: int) -> dict:
    run_id = f"ack-{arm}-{index}"
    rows = [{**r, "run_id": run_id} for r in records]
    if arm == "none":
        rows = [{k: v for k, v in r.items() if k != "probe"} for r in rows]
    if arm == "v5":
        schema = p5.flight_schema_v5()
        batches = [pa.RecordBatch.from_pylist([p5.record_to_row_v5(r, "v4-accepted") for r in rows[i:i + 50]], schema=schema)
                   for i in range(0, len(rows), 50)]
    else:
        batches = [v4.records_to_batch(rows[i:i + 50]) for i in range(0, len(rows), 50)]
    store = tmp / run_id
    server = Server(store, "--flush-steps", "50")
    try:
        acks = put(server.endpoint, run_id, 0, batches, attempt=int(rows[0]["attempt"]))
        time.sleep(0.5)
    finally:
        server.close()
    has_probe = ["probe" in r for r in rows]
    kinds = []
    for i, a in enumerate(acks[:len(batches)]):
        probe = any(has_probe[i * 50:(i + 1) * 50])
        after = i > 0 and any(has_probe[(i - 1) * 50:i * 50])
        kinds.append(("probe" if probe else "after_probe" if after else "plain", a.get("seconds"), a.get("error")))
    stored = sorted((store / "runs" / run_id).rglob("part-*.parquet"))
    return {"acks": kinds, "parts": [str(p) for p in stored], "arrow_bytes": [b.get_total_buffer_size() for b in batches]}


def measure_ack(rounds: int, write_rounds: int) -> dict:
    records = [json.loads(line) for line in DENSE.open()]
    arms = ("v4", "v5", "none")
    result: dict = {"source": str(DENSE), "records": len(records), "rounds": [], "arms": {}}
    with tempfile.TemporaryDirectory() as tmp:
        kept_parts: dict[str, list[str]] = {}
        for index in range(rounds):
            entry = {"load_before": load(), "arms": {}}
            for arm in arms:  # interleaved within the round
                got = ack_round(records, arm, Path(tmp), index)
                entry["arms"][arm] = got["acks"]
                kept_parts[arm] = got["parts"]
                entry.setdefault("arrow_bytes", {})[arm] = got["arrow_bytes"]
            entry["load_after"] = load()
            result["rounds"].append(entry)
        for arm in arms:
            by_kind: dict[str, list[float]] = {}
            for entry in result["rounds"]:
                for kind, seconds, error in entry["arms"][arm]:
                    if error is None and seconds is not None:
                        by_kind.setdefault(kind, []).append(seconds * 1e3)
            result["arms"][arm] = {kind: spread(values) for kind, values in by_kind.items()}
        # The Parquet write alone, of the last round's parts, with the server's writer settings.
        writes = {}
        for arm in arms:
            loads = load()
            parts = reader("write", str(write_rounds), *kept_parts[arm])
            probe_parts = [p for p in parts if pq.read_table(p["part"], columns=["probe_json"]).column(0).null_count < 50]
            plain_parts = [p for p in parts if p not in probe_parts]
            writes[arm] = {
                "load_before": loads,
                "probe_part": {"write_ms_median": [p["write"]["median_ms"] for p in probe_parts],
                               "write_ms_min": [p["write"]["min_ms"] for p in probe_parts],
                               "write_ms_max": [p["write"]["max_ms"] for p in probe_parts],
                               "bytes": [p["bytes"] for p in probe_parts], "arrow_bytes": [p["arrow_bytes"] for p in probe_parts]},
                "plain_part": {"write_ms_median": [p["write"]["median_ms"] for p in plain_parts],
                               "bytes": [p["bytes"] for p in plain_parts]},
            }
        result["parquet_write"] = writes
    return result


# --- decode ---------------------------------------------------------------------------------------


def measure_decode(rounds: int, flushes: int, ranks: int, steps: int, keep: Path | None) -> dict:
    import benchmark_detectors as bd

    from tensorquill import evidence

    result: dict = {}
    base = Path(tempfile.mkdtemp(prefix="tq-al0-decode-")) if keep is None else keep
    try:
        shape = {"blocks": 24, "experts": 32, "top_k": 8}
        bench_store = base / "bench-store"
        template, manifest = bd.prepare_store(bench_store, shape, ranks, steps, max(1, (os.cpu_count() or 2) - 1))
        result["bench_store"] = manifest
        # One stream (variant 0, rank 0) in both layouts, `flushes` 50-step parts each, written as
        # the benchmark writes its parts (Zstd level 1).
        layouts = {"v4": base / "v4", "v5": base / "v5"}
        for layout, folder in layouts.items():
            folder = folder / "runs" / "bench" / "rank=0" / f"attempt={template['attempt']}"
            folder.mkdir(parents=True, exist_ok=True)
            for flush in range(flushes):
                low = flush * 50
                records = [bd.synth_record(template, 0, 0, s) for s in range(low, low + 50)]
                if layout == "v4":
                    table = pa.Table.from_batches([v4.records_to_batch(records)])
                else:
                    table = pa.Table.from_batches([p5.batch_v5(records, "v4-accepted")])
                pq.write_table(table, folder / f"part-{low:08}-{low + 49:08}.parquet", compression="zstd", compression_level=1)
        parts = {layout: sorted(str(p) for p in folder.rglob("part-*.parquet")) for layout, folder in layouts.items()}
        result["part_bytes"] = {layout: sum(Path(p).stat().st_size for p in ps) for layout, ps in parts.items()}
        runs = []
        for index in range(rounds):
            entry = {"load_before": load()}
            for layout in ("v4", "v5"):
                entry[layout] = reader("bench", "1", *parts[layout])
            for layout in ("v4", "v5"):
                folder = layouts[layout] / "runs" / "bench"
                started = time.perf_counter()
                loaded = evidence.load_records(folder)
                entry[f"python_load_records_{layout}_ms"] = (time.perf_counter() - started) * 1e3
                entry[f"python_records_{layout}"] = len(loaded)
            runs.append(entry)
        result["rounds"] = runs
        result["summary"] = {}
        for layout in ("v4", "v5"):
            result["summary"][layout] = {
                key: spread([r[layout][key]["median_ms"] for r in runs])
                for key in ("read_part", "layers_decode", "moe_stats_decode")
            }
            result["summary"][layout]["python_load_records_ms"] = spread([r[f"python_load_records_{layout}_ms"] for r in runs])
        result["rows"] = runs[0]["v4"]["rows"]
        # The detectors' whole time on the 64-rank store, as the README measured it.
        example = ROOT / "server" / "target" / "release" / "examples" / "detect_store"
        bench = []
        for index in range(rounds):
            loads = load()
            done = subprocess.run([str(example), str(bench_store), "bench", "--bench"], capture_output=True, text=True)
            bench.append({"load_before": loads, "exit": done.returncode, "stdout": done.stdout[-1500:], "stderr": done.stderr[-500:]})
        result["detect_store_bench"] = bench
    finally:
        if keep is None:
            shutil.rmtree(base, ignore_errors=True)
    return result


# --- live64 ---------------------------------------------------------------------------------------


def measure_live64(rounds: int, ranks: int, step_ms: float) -> dict:
    """64 ranks' streams at once against a fresh protocol 4 server (`--flush-steps 50`), replayed by
    the server's own `live_streams` example from a source store of the dense capture's 400 records
    in one layout (8 source ranks, each simulated rank replaying source rank r mod 8). Every other
    50-step part holds a probe (the capture's probe_every was 100). Arms: v4, v5, none."""
    records = [json.loads(line) for line in DENSE.open()]
    example = ROOT / "server" / "target" / "release" / "examples" / "live_streams"
    result: dict = {"ranks": ranks, "step_ms": step_ms, "rounds": [], "arms": {}}
    with tempfile.TemporaryDirectory() as tmp:
        sources = {}
        for arm in ("v4", "v5", "none"):
            source = Path(tmp) / f"source-{arm}"
            for rank in range(8):
                rows = [{**r, "run_id": "live", "rank": rank} for r in records]
                if arm == "none":
                    rows = [{k: v for k, v in r.items() if k != "probe"} for r in rows]
                folder = source / "runs" / "live" / f"rank={rank}" / f"attempt={rows[0]['attempt']}"
                folder.mkdir(parents=True, exist_ok=True)
                for low in range(0, len(rows), 50):
                    chunk = rows[low:low + 50]
                    table = pa.Table.from_batches([p5.batch_v5(chunk, "v4-accepted") if arm == "v5" else v4.records_to_batch(chunk)])
                    pq.write_table(table, folder / f"part-{low:08}-{low + 49:08}.parquet", compression="zstd")
            sources[arm] = source
        for index in range(rounds):
            entry = {"load_before": load(), "arms": {}}
            for arm in ("v4", "v5", "none"):
                store = Path(tmp) / f"store-{arm}-{index}"
                server = Server(store, "--flush-steps", "50")
                try:
                    done = subprocess.run([str(example), str(sources[arm]), "live", server.endpoint.replace("grpc://", "http://"),
                                           "--store", str(store), "--ranks", str(ranks), "--step-ms", str(step_ms)],
                                          capture_output=True, text=True, timeout=600)
                finally:
                    server.close()
                last = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else "{}"
                try:
                    entry["arms"][arm] = json.loads(last)
                except ValueError:
                    entry["arms"][arm] = {"exit": done.returncode, "stdout": done.stdout[-800:], "stderr": done.stderr[-800:]}
                shutil.rmtree(store, ignore_errors=True)
            entry["load_after"] = load()
            result["rounds"].append(entry)
            print(json.dumps(entry)[:1500], flush=True)
    for arm in ("v4", "v5", "none"):
        got = [r["arms"][arm] for r in result["rounds"] if isinstance(r["arms"][arm].get("ack"), dict)]
        summary = {}
        for key in got[0]["ack"] if got else ():
            if isinstance(got[0]["ack"][key], (int, float)):
                summary[key] = spread([g["ack"][key] for g in got])
        result["arms"][arm] = {"ack": summary, "failed_ranks": [g.get("failed_ranks") for g in got]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("ack", "decode", "live64"))
    parser.add_argument("--step-ms", type=float, default=65.0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--write-rounds", type=int, default=7)
    parser.add_argument("--flushes", type=int, default=20)
    parser.add_argument("--ranks", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--keep", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.rounds < 5:
        raise SystemExit("--rounds must be 5 or more")
    started = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "load_at_start": os.getloadavg(),
               "cpu_count": os.cpu_count(), "python": sys.version.split()[0], "pyarrow": pa.__version__}
    if args.what == "ack":
        report = {**started, **measure_ack(args.rounds, args.write_rounds)}
    elif args.what == "live64":
        report = {**started, **measure_live64(args.rounds, args.ranks, args.step_ms)}
    else:
        report = {**started, **measure_decode(args.rounds, args.flushes, args.ranks, args.steps, args.keep)}
    report["load_at_end"] = os.getloadavg()
    text = json.dumps(report, indent=1, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(json.dumps({k: report[k] for k in report if k in ("arms", "parquet_write", "summary", "part_bytes")}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
