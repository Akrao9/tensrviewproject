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

Known today: both targets reach a panic inside `arrow-ipc` 60.1.0 within seconds (`AQ5-R1` and
`AQ5-R2` in the test file above), on a record batch message whose header names buffers its body does
not hold. Until that is settled, a run stops at the first such input.
