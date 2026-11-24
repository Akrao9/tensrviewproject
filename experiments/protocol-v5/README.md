# Protocol 5's measurements (wave AL, lane AL-0)

What `WAVE_AL_CONTRACT.md` cites. Nothing here is product code, and nothing here changes `sdk/src` or `server/src`. Every run was made on an 8-core Apple M1 Pro with Python 3.12.3 and pyarrow 26.0.1. Each result file records the load average before each round.

**Two trees.** The result files up to `eab810a` were measured on the base, against its protocol 4 release binary, with the prototype standing in for protocol 5's layout. Since step 1 merged (`92e471a`), `tensorquill.flight` writes protocol 5's layout and the tree's binary reads it. So `captures.py` and `measure.py` now run the SDK's own `flight` against this tree's server. Protocol 4's rows, for the "today" arm and the prototype, come from `flight_v4.py`, a frozen copy. `live.py` describes the base and runs only there. `build_cases.py` runs on both trees alike (below).

| File | What it is |
|---|---|
| `flight_v4.py` | Protocol 4's row functions, a verbatim copy of `sdk/src/tensorquill/flight.py`'s at `5c286ed`, from `_NON_FINITE` to `_MAP_COLUMNS` |
| `prototype.py` | Protocol 5's row layout as the contract states it (sections 1 to 4), built on `flight_v4.py`. `layout="v4-accepted"` writes a row version the protocol 4 server stores. Lane AL-1's `flight` gives its row on every case record and every capture record (its `seq` is step 2's) |
| `captures.py` → `results/captures.json` | The seven GPU captures in `~/Downloads/runs` (read only): their real keys and map shapes; the round trip through the SDK's own `flight` and a real Arrow batch; each row against the prototype's |
| `captures.py --fixtures` → `results/fixtures.json` | Every record of the repository's shared case files (`sdk/tests/fixtures/*/*.json`; no Gate 1 held-out file), its round trip through `flight` against protocol 4's (`flight_v4.py`) |
| `live.py` → `results/live.json` | The base's behaviour with protocol 5's data, against the base's protocol 4 binary: map order through Flight, Parquet and a restart; an old server and old Python readers given the new layout and versions; the step watermark and a future step; a struct field added later. To reproduce it, point `TENSORQUILL_SERVER_BIN` at a binary built from the base, and `PYTHONPATH` at the base's SDK |
| `measure.py ack` → `results/ack.json` | One rank: a probe batch's acknowledgement, its part's bytes and its Parquet write. Arms: `v4` (protocol 4's rows, metadata 1), `v5` (the SDK's protocol 5 rows, metadata 2), `none` (no probe, protocol 5's layout) |
| `measure.py live64` → `results/live64.json` | 64 ranks at 65 ms a step through the server's own `live_streams` example, for the same three arms |
| `measure.py decode` → `results/decode.json` | The detectors' JSON decoding against the typed columns (`rust-reader bench`); `detect_store --bench` on 64 ranks × 1,000 steps in protocol 5's layout and in protocol 4's, interleaved; Python's `load_records` on both layouts |
| `build_cases.py` → `sdk/tests/fixtures/wave_al/cases.json` | The shared cases: `row` from the prototype, `expect` and `today` from the base's engines. Its Rust reports are made from protocol 4's rows (`flight_v4.py`) through the tree's binary, which reads them as the base's did, so it gives the same file on the base and on step 1's joined tree, all but the `base` field (the commit it ran on). Step 2 changes the Python readers; from then on `today` is the file's |
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
