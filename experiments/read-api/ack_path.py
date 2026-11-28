"""What the catalog (and a snapshot) adds to an acknowledgement, as WAVE_AN_CONTRACT.md section 13.3
states it: one rank's 50-step batches, sent one at a time, each waiting for its acknowledgement.

    TENSORQUILL_SERVER_BIN=<binary> .venv/bin/python experiments/read-api/ack_path.py --rounds 7 --out ack-path.json
    # the wave's server, the arms interleaved in each round:
    ... ack_path.py --rounds 7 --arm off="--catalog off" --arm on= --arm snapshots="--snapshot-every-parts 1"

Per round and arm: a fresh store in a temporary folder, a server on it (`--flush-steps 50` and the
arm's arguments), one stream of `--batches` batches of the benchmark's synthetic records
(`sdk/examples/benchmark_detectors.py`), each its own part. Every batch after the first waits behind the
detectors of the one before, which run after its acknowledgement holding the rank, so these acks carry
whatever the catalog event, the report or a snapshot adds there. Printed per arm: the median and p99 of
those acknowledgements over each round, and the median of the rounds.

The figure: with the catalog on, the median and p99 are each at most 1.05 times the same server's with
`--catalog off`; the same for a snapshot after every part. A server before wave AN has neither option:
run it with the default arm (`today=`) for the base's figures.

`--smoke`: one round of 6 batches; not a measurement.
"""

from __future__ import annotations

import argparse
import json
import shlex
import tempfile
from pathlib import Path

import common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--arm", action="append", default=[], help="NAME=ARGS for the server, repeatable")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    import benchmark_detectors as bd

    if args.smoke:
        args.rounds, args.batches = 1, 6
    arms = dict(arm.split("=", 1) for arm in args.arm) or {"today": ""}
    tmpl = common.template({"blocks": 24, "experts": 32, "top_k": 8})
    batches = common.batches_for(tmpl, 0, 0, args.batches)
    result = {"what": "ack_path", "machine": common.machine(), "batches": args.batches,
              "arms": {name: {"args": extra, "rounds": []} for name, extra in arms.items()}}
    for index in range(args.rounds):
        for name, extra in arms.items():
            with tempfile.TemporaryDirectory() as tmp:
                loads = common.load()
                server = common.Server(Path(tmp) / "store", "--flush-steps", "50", *shlex.split(extra))
                acks = common.put(server.flight, bd.RUN_ID, 0, batches)
                err = server.close()
            later = [a["seconds"] for a in acks[1:] if "seconds" in a]
            entry = {"load_before": loads, "first": acks[0].get("seconds") if acks else None,
                     "after_first": common.spread(later), "errors": [a["error"] for a in acks if "error" in a],
                     "stderr_tail": err[-300:]}
            result["arms"][name]["rounds"].append(entry)
            print(json.dumps({"round": index, "arm": name, **{k: entry[k] for k in ("load_before", "after_first", "errors")}}),
                  flush=True)
    for arm in result["arms"].values():
        arm["median_of_round_medians"] = common.spread([r["after_first"].get("median") for r in arm["rounds"]])
        arm["median_of_round_p99s"] = common.spread([r["after_first"].get("p99") for r in arm["rounds"]])
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({name: {"median": arm["median_of_round_medians"], "p99": arm["median_of_round_p99s"]}
                      for name, arm in result["arms"].items()}, indent=1))
    if args.smoke:
        print("smoke run: these figures are not measurements")


if __name__ == "__main__":
    main()
