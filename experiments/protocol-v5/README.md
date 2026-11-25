# Protocol 5's measurements (wave AL, lane AL-0)

What `WAVE_AL_CONTRACT.md` cites. Nothing here is product code, and nothing here changes `sdk/src` or `server/src`. Every run was made on an 8-core Apple M1 Pro with Python 3.12.3 and pyarrow 26.0.1. Each result file records the load average before each round.

**Two trees.** The result files up to `eab810a` were measured on the base, against its protocol 4 release binary, with the prototype standing in for protocol 5's layout; they are kept in the history at that commit. Since then `ack.json`, `ack-run2.json`, `live64.json`, `live64-run2.json`, `decode.json` and `captures.json` are the joined tree's (`ce504e4`, protocol 5's own server, measured by the coordinator on Nov 24 from 19:09 to 19:36, with no session working; the contract's section 6 gives the figures beside the base's). `ack` and `live64` were each run a second time, `*-run2.json`, only after the first came out at or over its line. Since step 1 merged (`92e471a`), `tensorquill.flight` writes protocol 5's layout and the tree's binary reads it. So `captures.py` and `measure.py` now run the SDK's own `flight` against this tree's server. Protocol 4's rows, for the "today" arm and the prototype, come from `flight_v4.py`, a frozen copy. `live.py` describes the base and runs only there. `build_cases.py` runs on both trees alike (below).

| File | What it is |
|---|---|
| `flight_v4.py` | Protocol 4's row functions, a verbatim copy of `sdk/src/tensorquill/flight.py`'s at `5c286ed`, from `_NON_FINITE` to `_MAP_COLUMNS` |
| `prototype.py` | Protocol 5's row layout as the contract states it (sections 1 to 4), built on `flight_v4.py`. `layout="v4-accepted"` writes a row version the protocol 4 server stores. Lane AL-1's `flight` gives its row on every case record and every capture record (its `seq` is step 2's) |
| `captures.py` → `results/captures.json` | The seven GPU captures in `~/Downloads/runs` (read only): their real keys and map shapes; the round trip through the SDK's own `flight` and a real Arrow batch; each row against the prototype's |
| `captures.py --fixtures` → `results/fixtures.json` | Every record of the repository's shared case files (`sdk/tests/fixtures/*/*.json`; no Gate 1 held-out file), its round trip through `flight` against protocol 4's (`flight_v4.py`) |
| `live.py` → `results/live.json` | The base's behaviour with protocol 5's data, against the base's protocol 4 binary: map order through Flight, Parquet and a restart; an old server and old Python readers given the new layout and versions; the step watermark and a future step; a struct field added later. To reproduce it, point `TENSORQUILL_SERVER_BIN` at a binary built from the base, and `PYTHONPATH` at the base's SDK |
| `measure.py ack` → `results/ack.json`, `results/ack-run2.json` | One rank: a probe batch's acknowledgement, its part's bytes and its Parquet write. Arms: `v4` (protocol 4's rows, metadata 1), `v5` (the SDK's protocol 5 rows, metadata 2), `none` (no probe, protocol 5's layout) |
| `measure.py live64` → `results/live64.json`, `results/live64-run2.json` | 64 ranks at 65 ms a step through the server's own `live_streams` example, for the same three arms |
| `measure.py decode` → `results/decode.json` | The detectors' JSON decoding against the typed columns (`rust-reader bench`); `detect_store --bench` on 64 ranks × 1,000 steps in protocol 5's layout and in protocol 4's, interleaved; Python's `load_records` on both layouts |
| `cross_version.py` → `results/cross_version.json` | Section 5's matrix across a real protocol 4 build (below) |
| `build_cases.py` → `sdk/tests/fixtures/wave_al/cases.json` | The shared cases: `row` from the prototype, `expect` and `today` from the base's engines. Its Rust reports are made from protocol 4's rows (`flight_v4.py`) through the tree's binary, which reads them as the base's did, so it gives the same file on the base and on step 1's joined tree, all but the `base` field (the commit it ran on). Step 2 changes the Python readers (AL-4′) and protocol 4's `layers_json` read (AL-3′); from then on `today` is the file's, and the builder refuses a tree whose readers take record version 2 |
| `rust-reader/` | Reads a part whole with `Storage::read_part_whole` (`keys`), times reading with `Storage::read_part` (since lane AL-2, the projected read replay makes) and decoding (`bench`), and times a whole part's Parquet write with the server's writer settings (`write`) |

## Running them on this tree

```bash
cd <worktree>
cargo build --release --manifest-path server/Cargo.toml --bin tensorquill-server --example detect_store --example live_streams
cargo build --release --offline --manifest-path experiments/protocol-v5/rust-reader/Cargo.toml
export TENSORQUILL_SERVER_BIN=$PWD/server/target/release/tensorquill-server
PY=/path/to/.venv/bin/python
$PY -I experiments/protocol-v5/captures.py --out experiments/protocol-v5/results/captures.json
$PY -I experiments/protocol-v5/captures.py --fixtures experiments/protocol-v5/results/fixtures.json
PYTHONPATH=$PWD/sdk/src $PY experiments/protocol-v5/build_cases.py
$PY experiments/protocol-v5/measure.py ack --rounds 7 --out experiments/protocol-v5/results/ack.json
$PY experiments/protocol-v5/measure.py live64 --rounds 7 --out experiments/protocol-v5/results/live64.json
$PY experiments/protocol-v5/measure.py decode --rounds 7 --out experiments/protocol-v5/results/decode.json
```

`captures.py` and `measure.py` put this tree's `sdk/src` first on `sys.path`, so an interpreter with another checkout's SDK installed still runs this tree's. `measure.py ... --smoke` runs one small round of any of the three, to show the command works; its figures are not measurements.

Run the timings with nothing else heavy on the machine, and compare a lane's figures with today's measured beside them on the same machine: the acceptance figures in the contract's section 6 are ratios for that reason.

## Across a real protocol 4 build

`cross_version.py` runs section 5's matrix with no stand-in. It uses a clone of the repository at `85a477a` (the last protocol 4 commit, on `main`) with its own release server, and its `sdk/src` as the old SDK. The new side is this tree, with its own release server. Each SDK call runs in a child process with that SDK first on `sys.path`, against real servers on real stores. It records the messages each side gives and checks each claim of section 5 and the README; a claim that does not hold is reported as a finding, not adjusted. It takes no timing. Run on Nov 24 against `ce504e4`, every claim held but one:
- An old `runs` does not refuse a store of protocol 5 parts. It lists the run with the evidence the protocol 5 server wrote (`incidents.json`, `watchdog.json`), as the new `runs` does.
- The old `load_records` and the old one read refuse it, naming the part and the row.

```bash
git clone <this repository> <old> && git -C <old> checkout 85a477a
cargo build --release --manifest-path <old>/server/Cargo.toml --bin tensorquill-server
$PY experiments/protocol-v5/cross_version.py --old-checkout <old> --old-binary <old>/server/target/release/tensorquill-server \
    --new-binary server/target/release/tensorquill-server --out experiments/protocol-v5/results/cross_version.json
```
