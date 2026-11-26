//! A `DoPut` stream's messages, decoded as the server decodes them (`ingest::decode_put`), and the step
//! rows that come out judged (`detect::judge`). It must return rows or an error, never panic.
//!
//! The input is cut into parts, each a 4-byte little-endian length and that many bytes (a length past
//! the end takes the rest), and every three parts make one message: its data header, its data body and
//! its `app_metadata`. The first message carries the descriptor `["runs", "fuzz", "rank=0"]`, so the
//! fuzzer spends its time on the batches and not on the descriptor's one check.
#![no_main]

use std::sync::Arc;

use arrow_flight::{FlightData, FlightDescriptor};
use libfuzzer_sys::fuzz_target;
use tensorquill_server::detect::{judge, report, Thresholds};
use tensorquill_server::ingest::decode_put;

fn parts(mut data: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    while data.len() >= 4 {
        let length = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        data = &data[4..];
        let length = length.min(data.len());
        out.push(data[..length].to_vec());
        data = &data[length..];
    }
    out
}

fuzz_target!(|data: &[u8]| {
    let parts = parts(data);
    let mut messages = Vec::new();
    for (at, three) in parts.chunks(3).enumerate() {
        let mut message = FlightData::new()
            .with_data_header(three[0].clone())
            .with_data_body(three.get(1).cloned().unwrap_or_default())
            .with_app_metadata(three.get(2).cloned().unwrap_or_default());
        if at == 0 {
            message = message.with_descriptor(FlightDescriptor::new_path(vec![
                "runs".into(),
                "fuzz".into(),
                "rank=0".into(),
            ]));
        }
        messages.push(message);
    }
    let Ok(decoded) = decode_put(messages) else {
        return;
    };
    let steps: Vec<_> = decoded
        .iter()
        .flat_map(|batch| batch.step_rows.iter().map(|(_, rows)| rows.clone()))
        .collect();
    if let Ok(detectors) = judge(0, &steps, &Arc::new(Thresholds::default())) {
        let _ = report("fuzz", 0, &detectors, 0);
    }
});
