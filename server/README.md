# tensorquill-server

The server and offline detector binary for TensorQuill (the `tensorquill` package),
a debugger for PyTorch training runs. It takes a run's readings over Arrow Flight and writes
partitioned Parquet to local disk, S3 or GCS, and its `detect` subcommand reads a recorded run and
reports what went wrong in it. The `tensorquill` Python package runs that subcommand for you.

```
pip install tensorquill-server
tensorquill-server --help
```

This package holds no Python code. On the platforms below the wheel carries one program,
`tensorquill-server`, which pip puts in the environment's scripts directory (`bin/` in a virtual
environment):

| Platform | Wheel |
|---|---|
| Linux x86_64, glibc 2.17 or newer | `manylinux_2_17_x86_64` |
| Linux arm64, glibc 2.17 or newer | `manylinux_2_17_aarch64` |
| macOS arm64 | `macosx_11_0_arm64` |

On every other platform (Windows, Intel macOS, Alpine and other musl Linux) pip installs a
fallback wheel that holds the license files and nothing else: the install succeeds and there is
no `tensorquill-server` command. The `tensorquill` package then uses its Python detectors for
one-shot reads. To get the program there, build it from the source with `cargo build --release`
in the `server` directory.

The `tensorquill` package and this program must speak the same protocol number
(`tensorquill-server protocol` prints it; 5 since Nov 24, 2026, when the stored row's layout
changed; 4 from Nov 21, 2026, when `detect --follow` was added; 3 before). The package refuses a
binary with another number for offline detection and says which to install.

Streaming across protocols 4 and 5 works in one direction only. A sender older than protocol 5
streams to a protocol 5 server as before, and its batches are stored as sent; but in a file that a
protocol 5 capture wrote, its scalar-only records (values logged at an explicit `step=`) reach the
server as rows it cannot store, which it drops and counts under `unsupported_schema_version` in the
rank's entry of `watchdog.json`; they stay in the local `.jsonl`. A server older than protocol 5
refuses a protocol 5 sender's stream and stores nothing; the sender says so and keeps its lines
until the server is upgraded. A protocol 4 server started on a store that holds protocol 5 parts
reads them without their layers, MoE statistics and scalar-only records: once a store holds
protocol 5 parts, keep the server and the package at protocol 5 or later.

`detect` reads one rank's rows as an Arrow IPC stream on standard input and prints that rank's
report; `--samples` adds the per-step MoE samples. `detect --samples --follow` is what
`python -m tensorquill <file or store> --watch` runs with the Rust detectors: one process for the
whole watch, which keeps the rank's detectors between looks as the server keeps them between
flushes. Its input is frames, each an 8-byte little-endian length and one Arrow IPC stream: the
first holds every row of the rank, and each later one only rows appended to its newest attempt,
which are judged without judging the earlier rows again. It answers each frame with one line, and
refuses a frame that is not such an append (another attempt, a step not above the last, a router
not seen before or with more experts) and ends, and the package then starts it again with every
row. The process holds the rank's detectors and samples in memory for the life of the watch, where
`detect` holds them only while it runs: for 10,000 steps of 24 routers of 32 experts, about 386 MB
at its peak during the first look and about 240 MB resident between looks. It ends when the
watcher stops, and by itself when its input ends, so a watcher that is killed leaves none behind.

Each attempt's folder holds its training steps as Parquet parts (`part-*.parquet`) and, since
protocol 5, its scalar-only records as `scalars-*.parquet` objects of their own, which no detector
reads. Beside them, each rank's folder in the store holds `incidents.json` (what the
detectors found) and `attempts.json` (the order in which the job's attempts were first seen,
which decides which attempt owns the rank after a restart). The run's `watchdog.json` says
whether each rank is making progress; a rank whose capture stopped while training went on
reads `capture_stopped`, which is neither a hang nor finished.

A run has at most 100,000 ranks (`watchdog::MAX_RANKS`). A stream of rank 100,000 or more, a batch
whose `world_size` is over 100,000 and a heartbeat of such a rank are refused with
`INVALID_ARGUMENT` ("over this server's limit of 100000 ranks a run") and nothing of them is
stored or walked; the SDK checks the same limit before it streams and starts no stream past it.
A status names at most the lowest 1,000 ranks that never connected in `missing_ranks` and counts
the others in `missing_ranks_omitted` (absent when none is left out), and the run state's
`ranks_unknown` lists its lowest 1,000 and counts the rest in `ranks_omitted`.

A rank reads `reconnecting` for `--silent-seconds` after its stream drops without ending, and
`lost` after that. `reconnecting` is the state of the connection. A rank that was healthy at the
drop has no severity for that time and counts as active. A rank that was already `silent` or
`unresponsive` at the drop keeps its severity, on the clock it had, and stays in the run's
`rank_silent` or `job_stalled`; its entry says what it was in `stopped_as`. Opening a stream
again clears nothing, a new step recovers the rank, and a newer attempt that takes the rank
starts it afresh. A restart of the server does not carry the kept stall over: every unfinished
rank gets a fresh window from the restart.

At the store root the server keeps `store.json`, which names the store with a random ID written
once, by the first server to find none. Every acknowledgement carries it,
`{"end_byte_offset": N, "store_id": "..."}`, so a sender can tell a store wiped and made again
behind the same address from the one its saved offset belongs to; a restarted server, and any
other address that reaches the same store, return the same ID. Do not copy `store.json` from one
store to another.

The ID is written with a create that fails if the file already exists, so servers that start at
once on an empty store agree on one. Local disk and GCS always support that, and so does S3 unless
`AWS_CONDITIONAL_PUT=disabled` is set. On a backend that cannot, the server will not write
`store.json` (two servers starting at once could give one store two IDs) and refuses to start until
the ID is provisioned: put a `store.json` at the store root holding
`{"store_id": "<32 lowercase hex digits>"}` (for example from `openssl rand -hex 16`), once, or turn
conditional writes back on. An existing `store.json` is used on any backend.

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option.
