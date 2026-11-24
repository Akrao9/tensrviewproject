"""Section 6's acceptance figures on this tree's own protocol 5 server (WAVE_AL_CONTRACT.md, section 6).

    cd <worktree>
    cargo build --release --manifest-path server/Cargo.toml --bin tensorquill-server --example live_streams --example detect_store
    cargo build --release --offline --manifest-path experiments/protocol-v5/rust-reader/Cargo.toml
    TENSORQUILL_SERVER_BIN=$PWD/server/target/release/tensorquill-server \\
        .venv/bin/python experiments/protocol-v5/measure.py ack --rounds 7 --out experiments/protocol-v5/results/ack.json
    ... measure.py live64 --rounds 7 --out experiments/protocol-v5/results/live64.json
    ... measure.py decode --rounds 7 --out experiments/protocol-v5/results/decode.json

`--smoke` runs one small round of any of the three (fewer ranks, steps and parts) to show the command
works; its figures are not measurements. The script puts this tree's `sdk/src` first on `sys.path`.

Every timing is repeated `--rounds` times (5 or more), the arms interleaved within each round, and
each round records the one-minute load average before it starts. Medians are reported with the
lowest and highest value.

The arms, since step 1 merged (`92e471a`):
- `v4`: today's rows, protocol 4's layout (`flight_v4.py`, `flight.py`'s row functions frozen at
  `5c286ed`), sent with batch metadata version 1, as an SDK of before the wave sends them to a
  protocol 5 server. The server stores them as sent and its detectors read their JSON columns.
- `v5`: protocol 5's layout, the SDK's own `flight` (lane AL-1), metadata version 2.
- `none`: the same records with no probe at all, in protocol 5's layout, metadata version 2: the arm
  the acceptance ratios are taken against (a no-probe arm in protocol 5's layout, same server, same run).
Before step 1 (the figures of `results/*.json` up to `eab810a`), the "v5" arm was `prototype.py`'s
layout sent to the protocol 4 binary as row version 1, and `none` was protocol 4's; lane AL-1's rows
equal the prototype's on every case record and capture (`captures.py`, the contract test).

`ack`: the real probe records of `dense-layer8/dense.jsonl` (400 steps, a probe at steps 99, 199,
299 and 399, each 1.7 MB of JSON of which 98% is the 32 attention maps). Per round and arm: a fresh
server (`--flush-steps 50`, so each 50-step batch is one part, written and acknowledged before the
next is taken) and one DoPut stream of eight 50-step batches; the time from writing a batch to its
acknowledgement. A probe batch's acknowledgement waits for the part's Parquet write; the batch after
it also waits for the detectors to have read the probe part (they run after the acknowledgement,
holding the rank). Then, for the stored parts, the Parquet write alone, as `Storage::write_part`
does it, timed in `rust-reader` (`part-reader write`, which reads the part whole), and the parts' bytes.

`live64`: 64 ranks' streams at once, replayed by the server's own `live_streams` example (it names
metadata version 2 for a source part in protocol 5's layout, 1 otherwise).

`decode`: what the Rust detectors pay to read the layer and router statistics, from protocol 4's JSON
columns (`detect.rs`' visitors of `f71d8e0`, copied into `rust-reader`) or from protocol 5's typed maps,
on the synthetic Granite-shaped records `sdk/examples/benchmark_detectors.py` writes (24 blocks, 32
experts, top-8, from a real capture of its toy model), read with `Storage::read_part` (since lane AL-2
the projected read replay makes); Python's `load_records` on both layouts; and `detect_store --bench`
(the server's own example, the detectors' whole time) on a 64-rank store in protocol 5's layout and
on the same records in protocol 4's layout, interleaved within each round.
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
sys.path.insert(0, str(ROOT / "sdk" / "src"))  # this tree's SDK first, whatever the interpreter has installed

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import prototype as p5  # noqa: E402
from live import Server, put  # noqa: E402
from tensorquill import flight  # noqa: E402

v4 = p5.v4  # protocol 4's row functions, frozen (flight_v4.py)
READER = HERE / "rust-reader" / "target" / "release" / "part-reader"
EXAMPLES = ROOT / "server" / "target" / "release" / "examples"
DENSE = Path.home() / "Downloads" / "runs" / "dense-layer8" / "dense.jsonl"
ARMS = ("v4", "v5", "none")
METADATA = {"v4": 1, "v5": 2, "none": 2}


def spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}


def load() -> float:
    return round(os.getloadavg()[0], 2)


def reader(*args: str) -> object:
    done = subprocess.run([str(READER), *args], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def arm_rows(records: list[dict], arm: str) -> list[dict]:
    if arm == "none":
        return [{k: v for k, v in r.items() if k != "probe"} for r in records]
    return records


def arm_batch(records: list[dict], arm: str) -> pa.RecordBatch:
    """One batch of `records` in the arm's layout: protocol 4's frozen rows, or the SDK's own."""
    return (v4 if arm == "v4" else flight).records_to_batch(arm_rows(records, arm))


def check_layouts() -> None:
    """Refuse to measure a tree whose `flight` is not protocol 5's, or a binary that is not this tree's."""
    if "layers" not in flight.flight_schema().names or "layers_json" not in v4.flight_schema().names:
        raise SystemExit(f"{flight.__file__} does not write protocol 5's layout: run this on the joined tree")


# --- ack ------------------------------------------------------------------------------------------


def ack_round(records: list[dict], arm: str, tmp: Path, index: int) -> dict:
    run_id = f"ack-{arm}-{index}"
    rows = [{**r, "run_id": run_id} for r in records]
    batches = [arm_batch(rows[i:i + 50], arm) for i in range(0, len(rows), 50)]
    store = tmp / run_id
    server = Server(store, "--flush-steps", "50")
    try:
        acks = put(server.endpoint, run_id, 0, batches, meta_version=METADATA[arm], attempt=int(rows[0]["attempt"]))
        time.sleep(0.5)
    finally:
        server.close()
    has_probe = ["probe" in r for r in arm_rows(rows, arm)]
    kinds = []
    for i, a in enumerate(acks[:len(batches)]):
        probe = any(has_probe[i * 50:(i + 1) * 50])
        after = i > 0 and any(has_probe[(i - 1) * 50:i * 50])
        kinds.append(("probe" if probe else "after_probe" if after else "plain", a.get("seconds"), a.get("error")))
    errors = [a["error"] for a in acks if a.get("error")]
    stored = sorted((store / "runs" / run_id).rglob("part-*.parquet"))
    return {"acks": kinds, "errors": errors, "parts": [str(p) for p in stored],
            "arrow_bytes": [b.get_total_buffer_size() for b in batches]}


def measure_ack(rounds: int, write_rounds: int, steps: int | None) -> dict:
    records = [json.loads(line) for line in DENSE.open()][:steps]
    result: dict = {"source": "dense-layer8/dense.jsonl", "records": len(records), "metadata_versions": METADATA,
                    "rounds": [], "arms": {}}
    with tempfile.TemporaryDirectory() as tmp:
        kept_parts: dict[str, list[str]] = {}
        for index in range(rounds):
            entry = {"load_before": load(), "arms": {}}
            for arm in ARMS:  # interleaved within the round
                got = ack_round(records, arm, Path(tmp), index)
                entry["arms"][arm] = got["acks"]
                if got["errors"]:
                    entry.setdefault("errors", {})[arm] = got["errors"]
                kept_parts[arm] = got["parts"]
                entry.setdefault("arrow_bytes", {})[arm] = got["arrow_bytes"]
            entry["load_after"] = load()
            result["rounds"].append(entry)
            print(json.dumps({"round": index, "load_before": entry["load_before"], "errors": entry.get("errors")}), flush=True)
        for arm in ARMS:
            by_kind: dict[str, list[float]] = {}
            for entry in result["rounds"]:
                for kind, seconds, error in entry["arms"][arm]:
                    if error is None and seconds is not None:
                        by_kind.setdefault(kind, []).append(seconds * 1e3)
            result["arms"][arm] = {kind: spread(values) for kind, values in by_kind.items()}
            if "probe" in result["arms"][arm] and "plain" in result["arms"][arm]:
                result["arms"][arm]["probe_over_plain"] = (result["arms"][arm]["probe"]["median"]
                                                           / result["arms"][arm]["plain"]["median"])
        # The Parquet write alone, of the last round's parts, with the server's writer settings.
        writes = {}
        for arm in ARMS:
            loads = load()
            parts = reader("write", str(write_rounds), *kept_parts[arm])
            probe_parts = [p for p in parts if pq.read_table(p["part"], columns=["probe_json"]).column(0).null_count
                           < pq.read_metadata(p["part"]).num_rows]
            plain_parts = [p for p in parts if p not in probe_parts]
            writes[arm] = {
                "load_before": loads,
                "columns": pq.read_schema(kept_parts[arm][0]).names if kept_parts[arm] else [],
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


def write_store_v4(store: Path, template: dict, ranks: int, steps: int) -> dict:
    """`benchmark_detectors.write_store`'s store, the same records and parts, in protocol 4's layout
    (`flight_v4.records_to_batch`), for `detect_store --bench` beside protocol 5's."""
    import benchmark_detectors as bd

    store.mkdir(parents=True, exist_ok=True)
    (store / "template.json").write_text(json.dumps(template))
    variants = min(bd.VARIANTS, ranks)
    size = 0
    for low in range(0, steps, bd.FLUSH_STEPS):
        high = low + bd.FLUSH_STEPS - 1
        for variant in range(variants):
            shared = [r for r in range(ranks) if r % variants == variant]
            records = [bd.synth_record(template, variant, shared[0], step) for step in range(low, high + 1)]
            table = pa.Table.from_batches([v4.records_to_batch(records)])
            column = table.schema.get_field_index("rank")
            for rank in shared:
                path = bd.part_path(store, rank, template["attempt"], low, high)
                path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table.set_column(column, table.schema.field("rank"), pa.array([rank] * table.num_rows, pa.int32())),
                               path, compression="zstd", compression_level=1)
                size += path.stat().st_size
    manifest = {"ranks": ranks, "steps": steps, "shape": template["shape"], "parts": ranks * (steps // bd.FLUSH_STEPS),
                "bytes": size, "layout": "protocol 4 (flight_v4.py)"}
    (store / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def detect_store_bench(store: Path) -> dict:
    loads = load()
    done = subprocess.run([str(EXAMPLES / "detect_store"), str(store), "bench", "--bench"], capture_output=True, text=True)
    return {"load_before": loads, "exit": done.returncode, "stdout": done.stdout[-1500:], "stderr": done.stderr[-500:]}


def measure_decode(rounds: int, flushes: int, ranks: int, steps: int, keep: Path | None) -> dict:
    import benchmark_detectors as bd

    from tensorquill import evidence

    result: dict = {}
    base = Path(tempfile.mkdtemp(prefix="tq-al0-decode-")) if keep is None else keep
    try:
        shape = {"blocks": 24, "experts": 32, "top_k": 8}
        stores = {"v5": base / "bench-store", "v4": base / "bench-store-v4"}
        # benchmark_detectors writes its store with the SDK's own flight: protocol 5's layout.
        template, manifest = bd.prepare_store(stores["v5"], shape, ranks, steps, max(1, (os.cpu_count() or 2) - 1))
        result["bench_store"] = {"v5": manifest}
        if not (stores["v4"] / "manifest.json").exists():
            result["bench_store"]["v4"] = write_store_v4(stores["v4"], template, ranks, steps)
        else:
            result["bench_store"]["v4"] = json.loads((stores["v4"] / "manifest.json").read_text())
        part = next(stores["v5"].rglob("part-*.parquet"))
        result["bench_store"]["v5"]["columns"] = pq.read_schema(part).names
        # One stream (variant 0, rank 0) in both layouts, `flushes` 50-step parts each, written as
        # the benchmark writes its parts (Zstd level 1).
        layouts = {"v4": base / "v4", "v5": base / "v5"}
        for layout, folder in layouts.items():
            folder = folder / "runs" / "bench" / "rank=0" / f"attempt={template['attempt']}"
            folder.mkdir(parents=True, exist_ok=True)
            for flush_index in range(flushes):
                low = flush_index * 50
                records = [bd.synth_record(template, 0, 0, s) for s in range(low, low + 50)]
                table = pa.Table.from_batches([arm_batch(records, layout)])
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
        # The detectors' whole time on the 64-rank store, protocol 5's layout beside protocol 4's,
        # interleaved within each round.
        bench = []
        for index in range(rounds):
            bench.append({layout: detect_store_bench(stores[layout]) for layout in ("v5", "v4")})
            print(json.dumps({"round": index, **{k: {"exit": v["exit"], "load_before": v["load_before"]} for k, v in bench[-1].items()}}), flush=True)
        result["detect_store_bench"] = bench
    finally:
        if keep is None:
            shutil.rmtree(base, ignore_errors=True)
    return result


# --- live64 ---------------------------------------------------------------------------------------


def measure_live64(rounds: int, ranks: int, step_ms: float, steps: int | None) -> dict:
    """64 ranks' streams at once against a fresh server of this tree (`--flush-steps 50`), replayed by
    the server's own `live_streams` example from a source store of the dense capture's 400 records
    in one layout (8 source ranks, each simulated rank replaying source rank r mod 8). Every other
    50-step part holds a probe (the capture's probe_every was 100). Arms: v4, v5, none."""
    records = [json.loads(line) for line in DENSE.open()][:steps]
    example = EXAMPLES / "live_streams"
    result: dict = {"ranks": ranks, "step_ms": step_ms, "metadata_versions": METADATA, "rounds": [], "arms": {}}
    with tempfile.TemporaryDirectory() as tmp:
        sources = {}
        for arm in ARMS:
            source = Path(tmp) / f"source-{arm}"
            for rank in range(8):
                rows = [{**r, "run_id": "live", "rank": rank} for r in records]
                folder = source / "runs" / "live" / f"rank={rank}" / f"attempt={rows[0]['attempt']}"
                folder.mkdir(parents=True, exist_ok=True)
                for low in range(0, len(rows), 50):
                    chunk = rows[low:low + 50]
                    table = pa.Table.from_batches([arm_batch(chunk, arm)])
                    pq.write_table(table, folder / f"part-{low:08}-{low + len(chunk) - 1:08}.parquet", compression="zstd")
            sources[arm] = source
        for index in range(rounds):
            entry = {"load_before": load(), "arms": {}}
            for arm in ARMS:
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
                if done.returncode != 0:
                    entry["arms"][arm].setdefault("exit", done.returncode)
                    entry["arms"][arm].setdefault("stderr", done.stderr[-800:])
                shutil.rmtree(store, ignore_errors=True)
            entry["load_after"] = load()
            result["rounds"].append(entry)
            print(json.dumps(entry)[:1500], flush=True)
    for arm in ARMS:
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
    parser.add_argument("--smoke", action="store_true",
                        help="one small round, to show the command works: not a measurement")
    args = parser.parse_args()
    capture_steps = None
    if args.smoke:
        args.rounds, args.write_rounds, args.flushes = 1, 1, 2
        args.ranks, args.steps, capture_steps = 8, 100, 200  # 200 of the capture's steps hold two probes
    elif args.rounds < 5:
        raise SystemExit("--rounds must be 5 or more")
    check_layouts()
    started = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "load_at_start": os.getloadavg(),
               "cpu_count": os.cpu_count(), "python": sys.version.split()[0], "pyarrow": pa.__version__,
               "smoke": args.smoke, "flight_schema_version": flight.SCHEMA_VERSION, "row_version": flight.ROW_VERSION}
    if args.what == "ack":
        report = {**started, **measure_ack(args.rounds, args.write_rounds, capture_steps)}
    elif args.what == "live64":
        report = {**started, **measure_live64(args.rounds, args.ranks, args.step_ms, capture_steps)}
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
