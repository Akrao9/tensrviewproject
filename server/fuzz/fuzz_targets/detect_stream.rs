//! `tensorquill-server detect`'s input: any bytes as an Arrow IPC stream, read and judged as `main` runs
//! `detect`, inside `decode_guard::offline` (wave AR, lane AR-4). It must return a report or an error,
//! never panic. The ordinary suite's bounded version is `tests/aq5_generated_decode_test.rs`.
#![no_main]

use std::io::Cursor;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use libfuzzer_sys::fuzz_target;
use tensorquill_server::decode_guard;
use tensorquill_server::detect::{detect_stream, Thresholds};

fuzz_target!(|data: &[u8]| {
    let thresholds = Arc::new(Thresholds::default());
    let _ = decode_guard::offline(AssertUnwindSafe(|| detect_stream(Cursor::new(data), &thresholds)));
});
