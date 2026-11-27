# Fuzz targets of the server's decode

Two `cargo fuzz` targets, from the bytes a client sends up to the rows the detectors receive:

- `decode_put`: a `DoPut` stream's messages through `ingest::decode_put`, the decode the server's
  `run_stream` and `accept` do before anything is placed or kept, and the step rows on through
  `detect::judge`.
- `detect_stream`: any bytes as the Arrow IPC stream `tensorquill-server detect` reads.

Each must give rows (or a report) or an error and never panic. The ordinary suite runs a bounded,
seeded version of both in `../tests/aq5_generated_decode_test.rs`, with nothing added to the
server's dependencies.

This folder is a package with its own workspace, so the server's `Cargo.toml` and `Cargo.lock` are
not touched by it. Running it needs `cargo-fuzz`, a nightly toolchain and the `libfuzzer-sys` crate:

    cargo install cargo-fuzz
    rustup toolchain install nightly
    cd server && cargo +nightly fuzz run detect_stream
    cd server && cargo +nightly fuzz run decode_put

Since wave AR (lane AR-4) the decode is guarded (`src/decode_guard.rs`). The panics `arrow-ipc` 60.1.0
raises on a record batch message whose header names buffers its body does not hold (`AQ5-R1` and
`AQ5-R2`) end the live stream with `INVALID_ARGUMENT` naming a malformed record batch, and `detect` with
its error exit; the guard keeps the panic hook quiet for them. So:

- `decode_put` treats that status as an error like any other. An `INTERNAL` status saying the decode
  panicked is a panic the guard caught outside Arrow: the target panics on it, so the run stops there.
- `detect_stream` runs `detect::detect_stream` inside `decode_guard::offline`, as `main` runs
  `detect`. The reads are in `src/detect.rs`, so `detect_stream` alone still panics where Arrow's
  reader does; the command, and this target, do not.

libFuzzer's own hook aborts on a panic. The guard's hook is installed after it, at the first guarded
call, and is kept quiet only for a panic raised in an Arrow crate during a guarded call, which it lets
unwind to the guard; any other panic reaches libFuzzer's hook and stops the run.

Neither target has been built or run: there is no nightly toolchain, cargo-fuzz or libfuzzer-sys on
the machine where they were written (lane AQ-5) or changed (lane AR-4).
