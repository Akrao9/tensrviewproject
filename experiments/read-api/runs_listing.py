"""`runs` on a store of many runs, as WAVE_AN_CONTRACT.md section 13.2 states it.

    TENSORQUILL_SERVER_BIN=<binary> .venv/bin/python experiments/read-api/runs_listing.py \\
        --runs 10000 --store <a folder kept between runs> --rounds 5 --out runs-10k.json
    # the wave's server, through its read API as well:
    ... runs_listing.py --runs 10000 --store <folder> --rounds 5 --api

The store: one real run made by the server itself (two 50-step batches of the benchmark's synthetic
records, so its `incidents.json`, `watchdog.json` and `attempts.json` are the server's own), then copied
under `--runs` run IDs, each with its JSON files rewritten for its ID and its parts hard-linked. Built
once in `--store` and reused.

Timed per round: `python -m tensorquill runs <store> --json` in a fresh process (today's path: it reads
every run's files from the folder). With `--api` (a server of wave AN): a server on the store with
`--http-listen`, then the first page of `GET /v1/runs` and every page to the end, each timed, and the
`list` count of `/metrics` before and after the first page (the figure says no store listing is made by
the request); and `python -m tensorquill runs --endpoint <url> --json`. A server before wave AN refuses
`--http-listen`, and the script says so and records only today's path.

`--smoke`: 200 runs, one round; not a measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import common

PYTHON = sys.executable


def build(store: Path, runs: int) -> dict:
    import benchmark_detectors as bd

    marker = store / "listing.json"
    if marker.exists():
        made = json.loads(marker.read_text())
        if made["runs"] >= runs:
            return made
        raise SystemExit(f"--store {store} holds {made['runs']} runs; give a new or empty one. Nothing was deleted.")
    if store.exists() and any(store.iterdir()):
        raise SystemExit(f"--store {store} is not empty and holds no listing store. Nothing was deleted.")
    seed = store / "seed"
    tmpl = common.template({"blocks": 4, "experts": 4, "top_k": 2})
    server = common.Server(seed, "--flush-steps", "50")
    acks = common.put(server.flight, bd.RUN_ID, 0, common.batches_for(tmpl, 0, 0, 2))
    common.report_reaches(seed / "runs" / bd.RUN_ID / "rank=0", 99, deadline=60)
    server.close()
    source = seed / "runs" / bd.RUN_ID
    for index in range(runs):
        run_id = f"run-{index:06}"
        for path in source.rglob("*"):
            target = store / "runs" / run_id / path.relative_to(source)
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif path.suffix == ".json":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(path.read_text().replace(f'"{bd.RUN_ID}"', f'"{run_id}"'))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(path, target)
    (store / "store.json").write_bytes((seed / "store.json").read_bytes())
    made = {"runs": runs, "seed_acks": [a.get("seconds") for a in acks]}
    marker.write_text(json.dumps(made))
    return made


def timed_cli(*arguments: str) -> dict:
    env = {**os.environ, "PYTHONPATH": str(common.ROOT / "sdk" / "src")}
    began = time.perf_counter()
    done = subprocess.run([PYTHON, "-m", "tensorquill", "runs", *arguments, "--json"], capture_output=True,
                          text=True, env=env)
    seconds = time.perf_counter() - began
    try:
        rows = json.loads(done.stdout)  # `runs --json` prints a list of rows
        listed = len(rows) if isinstance(rows, list) else None
    except ValueError:
        listed = None
    return {"seconds": seconds, "exit": done.returncode, "listed": listed, "stderr_tail": done.stderr[-300:]}


def get(url: str) -> tuple[float, bytes]:
    began = time.perf_counter()
    with urllib.request.urlopen(url, timeout=120) as response:
        body = response.read()
    return time.perf_counter() - began, body


def lists(metrics: bytes) -> float | None:
    found = re.findall(rb'store_requests_total\{[^}]*op="list"[^}]*\} ([0-9.e+]+)', metrics)
    return sum(float(x) for x in found) if found else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=10_000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--api", action="store_true", help="also through the read API (a server of wave AN)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    scratch = tempfile.TemporaryDirectory() if args.smoke else None
    if args.smoke:
        args.runs, args.rounds, args.store = 200, 1, Path(scratch.name) / "store"
    if args.store is None:
        raise SystemExit("--store is needed (the store is kept between runs)")
    made = build(args.store, args.runs)
    result = {"what": "runs_listing", "machine": common.machine(), "store": made, "rounds": []}
    for index in range(args.rounds):
        entry = {"load_before": common.load(), "cli_folder": timed_cli(str(args.store))}
        if args.api:
            try:
                server = common.Server(args.store, http=True)
            except RuntimeError as error:
                entry["api"] = {"not_available": str(error)[:300]}
            else:
                try:
                    _, before = get(server.http + "/metrics")
                    first, body = get(server.http + "/v1/runs?limit=100")
                    _, after = get(server.http + "/metrics")
                    pages, token, seconds = 1, json.loads(body).get("next_page_token"), first
                    while token:
                        took, body = get(server.http + f"/v1/runs?limit=1000&page_token={token}")
                        pages, seconds, token = pages + 1, seconds + took, json.loads(body).get("next_page_token")
                    entry["api"] = {"first_page_seconds": first, "all_pages_seconds": seconds, "pages": pages,
                                    "store_lists_by_first_page": (None if lists(before) is None or lists(after) is None
                                                                  else lists(after) - lists(before)),
                                    "cli_endpoint": timed_cli("--endpoint", server.http)}
                finally:
                    server.close()
        result["rounds"].append(entry)
        print(json.dumps(entry), flush=True)
    result["cli_folder_seconds"] = common.spread([r["cli_folder"]["seconds"] for r in result["rounds"]])
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"cli_folder_seconds": result["cli_folder_seconds"]}, indent=1))
    if args.smoke:
        print("smoke run: these figures are not measurements")
        scratch.cleanup()


if __name__ == "__main__":
    main()
