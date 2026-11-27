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
not touched by it. Running it needs `cargo-fuzz` and a nightly toolchain (`libfuzzer-sys` comes as a
crate). Its own lock file is not kept in git, and a fresh resolve takes the newest Arrow 60 release
(60.2.0 on Nov 27, 2026), not the 60.1.0 the server's lock holds; so start it from the server's lock,
and the decode under test is the one the server runs. From `server/`:

    cargo install cargo-fuzz
    rustup toolchain install nightly
    cp Cargo.lock fuzz/Cargo.lock
    cargo +nightly fuzz build
    cargo run --manifest-path fuzz/Cargo.toml --example seeds -- fuzz/corpus
    cargo +nightly fuzz run decode_put fuzz/corpus/decode_put -- -max_total_time=600 -timeout=30
    cargo +nightly fuzz run detect_stream fuzz/corpus/detect_stream -- -max_total_time=600 -timeout=30

`examples/seeds.rs` writes five seeds a target, each checked before it is written: four valid runs
as the SDK sends them (a small one of 4 steps in one batch, 30 steps in batches of 7, a resumed run of
two attempts, 40 steps with a gradient spike), and one that Arrow panics on and the guard refuses (for
`decode_put` the small run's batch with its body cut to half; for `detect_stream` the small run's
stream with one byte set). `-timeout=30` reports an input that runs over 30 seconds; libFuzzer's own
default is 1200 seconds, longer than a run of ten minutes. `corpus/`, `artifacts/`, `target/` and
`Cargo.lock` are left out of git (`.gitignore`).

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

Both targets were first built and run on Nov 27, 2026 (lane AS-8 of wave AS), on an Apple M1 Pro
(8 cores, 16 GB, macOS 27.2) shared with other builds and tests, with rustc 1.102.0-nightly
(2026-11-25), cargo-fuzz 0.13.2 and libfuzzer-sys 0.4.13, in cargo-fuzz's default build (release with
debug assertions, AddressSanitizer), against Arrow 60.1.0. Each ran once for ten minutes, one after
the other, with one job, from the five seeds above:

| target | runs | runs a second | coverage (`cov`, `ft`) at the start | at the end | corpus at the end | peak memory |
|---|---|---|---|---|---|---|
| `decode_put` | 317,778 | 528 | 7,354, 13,301 | 13,670, 41,895 | 1,594 files, 71 MB | 653 MB |
| `detect_stream` | 463,806 | 771 | 6,239, 11,520 | 12,608, 37,700 | 1,781 files, 74 MB | 878 MB |

Neither run found a crash, a timeout (over 30 seconds) or an out-of-memory stop (libFuzzer's limit of
2048 MB); the slowest input took under a second. Both seeds Arrow panics on ran through the guard to
an error, inside the fuzz build, without stopping the run. Ten minutes a target is a smoke run: coverage
was still rising at the end of both, and no input longer than the largest seed (65,662 and 65,160
bytes) was tried. It shows the harness works and the guarded paths hold for this much input, not that
the decode is free of faults.
