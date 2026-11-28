"""A restart's cost, as WAVE_AN_CONTRACT.md section 13.1 states it: after the server restarts on a store
of R ranks of N steps, how long each rank's next part waits.

    TENSORQUILL_SERVER_BIN=<binary> .venv/bin/python experiments/read-api/restart.py \\
        --ranks 64 --steps 10000 --rounds 5 --store <a folder kept between runs> --out restart-64x10k.json
    ... --ranks 8 --steps 100000 ...

The store is `sdk/examples/benchmark_detectors.py`'s (synthetic records from a real capture of its toy
model: 24 blocks, 32 experts, top-8), in 50-step parts, built once in `--store` and reused (about 4 KB a
row on disk: 64 x 10,000 is about 3 GB, 8 x 100,000 about 3.5 GB). Each arm works on its own copy
(`--work`, cloned on APFS), so one arm's snapshots never reach another.

Per round and arm: a server is started on the arm's copy (`--flush-steps 50` and the arm's arguments),
and every rank, at once, sends two 50-step batches above what is stored, each waiting for its
acknowledgement, as the SDK's sender does. Recorded per rank: the first batch's acknowledgement (stored,
not yet judged), the second's (it waits behind the rebuild of the rank's detectors, which runs after the
first is acknowledged, holding the rank), and the time from the first acknowledgement until the rank's
`incidents.json` covers the first batch. Then the server is stopped (SIGTERM), and the next round sends
the next 100 steps. The figure is the second batch's wait.

Arms: `--arm NAME=ARGS`, repeatable; the default is one arm, `today=` (no extra arguments). For the
wave's server, for example `--arm replay="--snapshot-every-parts 0" --arm snapshot=` (section 7.3).
Each round runs every arm, in turn. Also printed: the bytes of the rank 0 copy's parts and snapshots
after the last round (the 10% figure at 100,000 steps).

`--smoke`: 4 ranks, 600 steps, one round, in a temporary folder: it shows the command works, and its
figures are not measurements. Take no timing on a shared machine.
"""

from __future__ import annotations

import argparse
import json
import shlex
import tempfile
import threading
import time
from pathlib import Path

import common


def run_round(store: Path, extra: list[str], ranks: int, first: int, tmpl: dict) -> dict:
    import benchmark_detectors as bd

    loads = common.load()
    batches = {rank: common.batches_for(tmpl, rank, first, 2) for rank in range(ranks)}
    started = time.perf_counter()
    server = common.Server(store, "--flush-steps", "50", *extra)
    up = time.perf_counter() - started
    results: dict[int, dict] = {}

    def one(rank: int) -> None:
        acks = common.put(server.flight, bd.RUN_ID, rank, batches[rank], start=first * 100)
        ready = common.report_reaches(store / "runs" / bd.RUN_ID / f"rank={rank}", first + 49)
        first_ack = acks[0].get("acked") if acks and "seconds" in acks[0] else None
        results[rank] = {
            "first": acks[0].get("seconds") if acks else None,
            "second": acks[1].get("seconds") if len(acks) > 1 else None,
            "report_after_first": (ready - first_ack) if ready and first_ack else None,
            "errors": [a["error"] for a in acks if "error" in a],
        }

    threads = [threading.Thread(target=one, args=(rank,)) for rank in range(ranks)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    err = server.close()
    return {
        "load_before": loads, "start_seconds": up,
        "first_ack": common.spread([r["first"] for r in results.values()]),
        "second_ack": common.spread([r["second"] for r in results.values()]),
        "report_after_first_ack": common.spread([r["report_after_first"] for r in results.values()]),
        "errors": sorted({e for r in results.values() for e in r["errors"]})[:5],
        "stderr_tail": err[-400:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ranks", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--store", type=Path, help="the benchmark store, built here once and reused")
    parser.add_argument("--work", type=Path, help="where each arm's copy goes (default: beside --store)")
    parser.add_argument("--arm", action="append", default=[], help="NAME=ARGS for the server, repeatable")
    parser.add_argument("--workers", type=int, default=4, help="processes that write the store")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    import benchmark_detectors as bd

    arms = dict(arm.split("=", 1) for arm in args.arm) or {"today": ""}
    shape = {"blocks": 24, "experts": 32, "top_k": 8}
    scratch = tempfile.TemporaryDirectory() if args.smoke else None
    if args.smoke:
        args.ranks, args.steps, args.rounds = 4, 600, 1
        args.store = Path(scratch.name) / "store"
        args.work = Path(scratch.name)
    if args.store is None:
        raise SystemExit("--store is needed (the store is large and is kept between runs)")
    tmpl, manifest = bd.prepare_store(args.store, shape, args.ranks, args.steps, args.workers)
    work = args.work or args.store.parent
    copies = {}
    for name in arms:
        copies[name] = work / f"{args.store.name}-{name}"
        common.clone_tree(args.store, copies[name])
    result = {"what": "restart", "machine": common.machine(), "ranks": args.ranks, "steps": args.steps,
              "store": manifest, "arms": {name: {"args": extra, "rounds": []} for name, extra in arms.items()}}
    for index in range(args.rounds):
        first = args.steps + 100 * index
        for name, extra in arms.items():
            entry = run_round(copies[name], shlex.split(extra), args.ranks, first, tmpl)
            result["arms"][name]["rounds"].append(entry)
            print(json.dumps({"round": index, "arm": name, "second_ack": entry["second_ack"],
                              "load_before": entry["load_before"], "errors": entry["errors"]}), flush=True)
    for name, arm in result["arms"].items():
        medians = [r["second_ack"].get("median") for r in arm["rounds"]]
        arm["second_ack_median_of_rounds"] = common.spread(medians)
        rank0 = copies[name] / "runs" / bd.RUN_ID / "rank=0"
        arm["rank0_bytes"] = {"parts": common.bytes_under(rank0, "part-*.parquet"),
                              "snapshots": common.bytes_under(rank0, "state-*"),
                              "compacted": common.bytes_under(rank0, "compact-*.parquet")}
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(json.dumps({name: {"second_ack": arm["second_ack_median_of_rounds"], "rank0_bytes": arm["rank0_bytes"]}
                      for name, arm in result["arms"].items()}, indent=1))
    if args.smoke:
        print("smoke run: these figures are not measurements")
        scratch.cleanup()


if __name__ == "__main__":
    main()
