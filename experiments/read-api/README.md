# Phase 4's measurements (`WAVE_AN_CONTRACT.md` section 13)

Scripts for the coordinator to run on a quiet machine. None was run for a figure when they were written: each was run once with `--smoke` against the release server of `a6c1e2c`, to show it works, and those outputs are not measurements. No product code is here.

Every script drives the binary that `TENSORQUILL_SERVER_BIN` names, so the same script measures today's server and the wave's with no cargo build, and puts this tree's `sdk/src` first on `sys.path`. They need the project's `.venv` (torch, to capture the benchmark's record template, and pyarrow with Flight). Each records the one-minute load average before every round; arms are interleaved within a round; medians are given with the lowest, highest and p99.

| Script | The figure (section) | What it prints |
|---|---|---|
| `restart.py` | 13.1: each rank's next part after a restart waits at most 1.0 s at 64 ranks of 10,000 steps and 2.0 s at 8 ranks of 100,000; snapshots at most 10% of a rank's part bytes at 100,000 | per round and arm, one JSON line with the second batch's wait across ranks (the figure) and errors; at the end, per arm, the median of the rounds' medians and rank 0's bytes of parts, snapshots and compacted files |
| `ack_path.py` | 13.3: with the catalog on, and with a snapshot after every part, the median and p99 of the acknowledgements at most 1.05 times the same server's with `--catalog off` | per round and arm, one JSON line with the median and p99 of every acknowledgement after the first; at the end, per arm, the median of the rounds' medians and of their p99s |
| `runs_listing.py` | 13.2: `runs --endpoint` lists 10,000 runs in at most 2 s, the first page of `/v1/runs` answers in at most 50 ms, with no store listing made by the request | per round, one JSON line with `python -m tensorquill runs <store>`'s seconds and exit status and, with `--api`, the read API's first page, all pages, the `list` count of `/metrics` across the first page, and `runs --endpoint`; at the end, the folder command's median |
| `binary_size.py` | 13.4: at most 2.5 MB of growth over the base's binary | each binary's bytes and its growth over the first |

`common.py` is what they share: the server started on port 0 (and `--http-listen` when the read API is asked for), a DoPut stream that waits for each acknowledgement as the SDK's sender does, the benchmark's synthetic records (`sdk/examples/benchmark_detectors.py`), and the summaries.

## Today's figures first

```bash
export TENSORQUILL_SERVER_BIN=<the release binary of a6c1e2c>
.venv/bin/python experiments/read-api/restart.py --ranks 64 --steps 10000 --rounds 5 --store <big disk>/an-restart-64x10k --out restart-64x10k-base.json
.venv/bin/python experiments/read-api/restart.py --ranks 8 --steps 100000 --rounds 5 --store <big disk>/an-restart-8x100k --out restart-8x100k-base.json
.venv/bin/python experiments/read-api/ack_path.py --rounds 7 --out ack-path-base.json
.venv/bin/python experiments/read-api/runs_listing.py --runs 10000 --rounds 5 --store <disk>/an-runs-10k --out runs-10k-base.json
.venv/bin/python experiments/read-api/binary_size.py base=$TENSORQUILL_SERVER_BIN
```

The stores are built once and kept (`--store`): about 3 GB for 64 ranks of 10,000 steps and 3.5 GB for 8 ranks of 100,000; `restart.py` clones the store for each arm (`--work`, beside it by default; a clone on APFS costs no space until written). Nothing is deleted by any script: a store that does not fit is refused with a message.

## The wave's figures, beside today's

Run the same commands with the wave's binary, adding the arms the contract names: `restart.py --arm replay="--snapshot-every-parts 0" --arm snapshot=`; `ack_path.py --arm off="--catalog off" --arm on= --arm snapshots="--snapshot-every-parts 1"`; `runs_listing.py --api`; `binary_size.py base=<a6c1e2c's> wave=<the wave's>`. The option names are the contract's proposals (sections 1.2, 7.3 and 13.3); the lanes may name them otherwise, and these arms change with them.
