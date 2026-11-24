# Protocol 5's measurements (wave AL, lane AL-0)

What `WAVE_AL_CONTRACT.md` cites. Nothing here is product code, and nothing here changes `sdk/src` or `server/src`. Every run was made on an 8-core Apple M1 Pro with Python 3.12.3 and pyarrow 26.0.1, against the protocol 4 release binary built from this tree. Each result file records the load average before each round.

| File | What it is |
|---|---|
| `prototype.py` | Protocol 5's row layout as the contract states it (sections 1 to 4), built on today's `flight.py`. `layout="v4-accepted"` writes a row version the protocol 4 server stores |
| `captures.py` → `results/captures.json` | The seven GPU captures in `~/Downloads/runs` (read only): their real keys, map shapes, and the round trip through the prototype |
| `live.py` → `results/live.json` | Today's behaviour with protocol 5's data, against a real server: map order through Flight, Parquet and a restart; an old server and old Python readers given the new layout and versions; the step watermark and a future step; a struct field added later |
| `measure.py ack` → `results/ack.json` | One rank: a probe batch's acknowledgement, its part's bytes and its Parquet write, for today, the prototype, and no probe |
| `measure.py live64` → `results/live64.json` | 64 ranks at 65 ms a step through the server's own `live_streams` example, for the same three arms |
| `measure.py decode` → `results/decode.json` | The detectors' JSON decoding against the typed columns (`rust-reader bench`), `detect_store --bench` on 64 ranks × 1,000 steps, and Python's `load_records` on both layouts |
| `build_cases.py` → `sdk/tests/fixtures/wave_al/cases.json` | The shared cases: `row` from the prototype, `expect` and `today` from the base's engines |
| `rust-reader/` | Reads a part with the server's own `Storage::read_part` (`keys`), times decoding (`bench`), and times a part's Parquet write with the server's writer settings (`write`) |

## Running them

```bash
cd <worktree>
cargo build --release --manifest-path server/Cargo.toml --bin tensorquill-server --example detect_store --example live_streams
cargo build --release --offline --manifest-path experiments/protocol-v5/rust-reader/Cargo.toml
export PYTHONPATH=$PWD/sdk/src TENSORQUILL_SERVER_BIN=$PWD/server/target/release/tensorquill-server
PY=/path/to/.venv/bin/python
$PY experiments/protocol-v5/captures.py --out experiments/protocol-v5/results/captures.json
$PY experiments/protocol-v5/live.py --out experiments/protocol-v5/results/live.json
$PY experiments/protocol-v5/measure.py ack --rounds 7 --out experiments/protocol-v5/results/ack.json
$PY experiments/protocol-v5/measure.py live64 --rounds 5 --out experiments/protocol-v5/results/live64.json
$PY experiments/protocol-v5/measure.py decode --rounds 5 --out experiments/protocol-v5/results/decode.json
$PY experiments/protocol-v5/build_cases.py
```

Run the timings with nothing else heavy on the machine, and compare a lane's figures with today's measured beside them on the same machine: the acceptance figures in the contract's section 6 are ratios for that reason.
