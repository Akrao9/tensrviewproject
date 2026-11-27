//! `tensorquill-server detect`'s input: any bytes as an Arrow IPC stream, read and judged. It must
//! return a report or an error, never panic. The ordinary suite's bounded version is
//! `tests/aq5_generated_decode_test.rs`.
#![no_main]

use std::io::Cursor;
use std::sync::Arc;

use libfuzzer_sys::fuzz_target;
use tensorquill_server::detect::{detect_stream, Thresholds};

fuzz_target!(|data: &[u8]| {
    let thresholds = Arc::new(Thresholds::default());
    let _ = detect_stream(Cursor::new(data), &thresholds);
});
