//! Writes seed inputs for both fuzz targets: valid streams as the SDK sends them, so libFuzzer starts
//! from batches the server decodes and judges and not from random bytes alone.
//!
//!     cargo run --manifest-path server/fuzz/Cargo.toml --example seeds -- server/fuzz/corpus
//!
//! writes `<folder>/detect_stream/seed-*` (Arrow IPC streams, as `tensorquill-server detect` reads
//! them) and `<folder>/decode_put/seed-*` (the same batches as `DoPut` messages, in the
//! `decode_put` target's layout: each message's data header, data body and `app_metadata`, each part
//! a 4-byte little-endian length and its bytes). Before writing, it checks that every seed decodes
//! (`ingest::decode_put` gives step rows, `detect::detect_stream` a report), so a seed that the
//! server would refuse is an error here and not a silent waste of the fuzzer's time.
//!
//! The rows follow `flight.py`'s protocol 5 layout, as `tests/aq5_generated_decode_test.rs` builds
//! them, with no randomness: a seed is the same bytes on every run.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::{FlightData, FlightDescriptor};
use futures::TryStreamExt;
use serde_json::{json, Value};
use tensorquill_server::decode_guard::{self, Caught, MALFORMED_INPUT};
use tensorquill_server::detect::{detect_stream, Thresholds};
use tensorquill_server::ingest::decode_put;

const RUN: &str = "fuzz";

fn map_of(value: DataType) -> DataType {
    DataType::Map(
        Arc::new(Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", value, true),
            ])),
            false,
        )),
        false,
    )
}

fn list_of(value: DataType) -> DataType {
    DataType::List(Arc::new(Field::new_list_field(value, true)))
}

/// `flight.py`'s `flight_schema()`: protocol 5's row layout, every column.
fn schema() -> Arc<Schema> {
    let f64s = |names: &[&str]| -> Vec<Field> {
        names
            .iter()
            .map(|name| Field::new(*name, DataType::Float64, true))
            .collect()
    };
    let layer = DataType::Struct(Fields::from(f64s(&["mean", "std", "abs_max", "zero_frac"])));
    let moe = DataType::Struct(Fields::from(vec![
        Field::new("routed_tokens", DataType::Int64, true),
        Field::new("entropy", DataType::Float64, true),
        Field::new("diversity", DataType::Float64, true),
        Field::new("dead_experts", DataType::Int64, true),
        Field::new("dead_frac", DataType::Float64, true),
    ]));
    let histogram = DataType::Struct(Fields::from(vec![
        Field::new("lo", DataType::Float64, true),
        Field::new("hi", DataType::Float64, true),
        Field::new("counts", list_of(DataType::Int64), true),
        Field::new("nonfinite", DataType::Int64, true),
    ]));
    let f64 = || DataType::Float64;
    Arc::new(Schema::new(vec![
        Field::new("schema_version", DataType::Int32, false),
        Field::new("run_id", DataType::Utf8, false),
        Field::new("rank", DataType::Int32, false),
        Field::new("attempt", DataType::Int64, false),
        Field::new("step", DataType::Int64, false),
        Field::new("ts", f64(), true),
        Field::new("lr", list_of(f64()), true),
        Field::new("grad_norm", f64(), true),
        Field::new("clip_grad_norm", f64(), true),
        Field::new("weight_norm", f64(), true),
        Field::new("weights_unchanged", list_of(DataType::Utf8), true),
        Field::new("step_skipped", DataType::Boolean, true),
        Field::new("grad_scale", f64(), true),
        Field::new("scalars", map_of(f64()), true),
        Field::new("grad_norms", map_of(f64()), true),
        Field::new("weight_norms", map_of(f64()), true),
        Field::new("update_rms", map_of(f64()), true),
        Field::new("moe_top_k", map_of(DataType::Int32), true),
        Field::new("moe_counts", map_of(list_of(DataType::Int64)), true),
        Field::new("replica_spread", map_of(list_of(f64())), true),
        Field::new("layers", map_of(layer), true),
        Field::new("moe_stats", map_of(moe), true),
        Field::new("probe_json", DataType::Utf8, true),
        Field::new("probe_maps", map_of(list_of(list_of(DataType::Float32))), true),
        Field::new("probe_histograms", map_of(histogram), true),
        Field::new("data_json", DataType::Utf8, true),
        Field::new("checkpoint_json", DataType::Utf8, true),
        Field::new("data_wait", f64(), true),
        Field::new("step_time", f64(), true),
        Field::new("system_json", DataType::Utf8, true),
        Field::new("seq", DataType::Int64, true),
        Field::new("record_version", DataType::Int32, true),
        Field::new("extra_json", DataType::Utf8, true),
    ]))
}

/// `count` step rows of one attempt; every fifth carries the probe columns, every tenth the system,
/// checkpoint and extra columns, and after every seventh comes a scalar-only row (`seq` set).
/// `spike` puts a gradient spike at that step, so the detectors have something to find.
fn rows(count: i64, attempt: i64, first: i64, spike: Option<i64>) -> Vec<Value> {
    let mut out = Vec::new();
    for step in first..first + count {
        let wobble = |centre: f64| centre * (0.95 + 0.1 * (((step * 37) % 11) as f64 / 10.0));
        let grad = if spike == Some(step) { 1000.0 } else { wobble(1.0) };
        let mut row = json!({
            "schema_version": 2, "run_id": RUN, "rank": 0, "attempt": attempt, "step": step,
            "ts": 1.7e9 + step as f64, "grad_norm": grad, "lr": [1e-3],
            "clip_grad_norm": grad * 1.2, "step_skipped": false, "grad_scale": 65536.0,
            "scalars": {"train/loss": 2.5 - step as f64 / 400.0},
            "grad_norms": {"p0.weight": wobble(0.5), "p1.weight": wobble(0.4)},
            "update_rms": {"p0.weight": wobble(1e-3), "p1.weight": wobble(1e-3)},
            "weights_unchanged": [],
            "moe_top_k": {"r0": 2},
            "moe_counts": {"r0": [256, 250, 262, 256]},
            "layers": {"blk0": {"mean": 0.01, "std": wobble(1.0), "abs_max": wobble(3.0), "zero_frac": 0.0}},
            "moe_stats": {"r0": {"routed_tokens": 512, "entropy": 1.38, "diversity": 0.99, "dead_experts": 0, "dead_frac": 0.0}},
            "data_json": format!("{{\"batches\":1,\"batch_hash\":\"h{step}\",\"epoch\":0,\"shuffle\":true}}"),
            "data_wait": 0.002, "step_time": 0.1,
            "record_version": 1,
        });
        if step % 5 == 0 {
            row["probe_json"] = json!(
                r#"{"attention":{"blk0":{"max_logit":[4.0,4.1],"entropy":[2.0,2.0],"entropy_share":[0.7,0.7],"masked_rows":0,"invalid_rows":0}}}"#
            );
            row["probe_histograms"] = json!({"blk0": {"lo": 0.0, "hi": 1.0, "counts": [1, 2, 3], "nonfinite": 0}});
            row["probe_maps"] = json!({"blk0": [[0.25, 0.75], [0.5, 0.5]]});
        }
        if step % 10 == 9 {
            row["system_json"] = json!(
                r#"{"cuda":{"allocated":2147483648,"reserved":4294967296,"peak_allocated":3221225472,"total":17179869184,"alloc_retries":0,"ooms":0}}"#
            );
            row["weight_norms"] = json!({"p0.weight": 10.0, "p1.weight": 9.0});
            row["replica_spread"] = json!({"p0.weight": [0.0, 0.0, 0.0]});
            row["checkpoint_json"] = json!(r#"{"save":{"after_step":9,"components":{"model":{"tensors":4}}}}"#);
            row["extra_json"] = json!(r#"{"optimizer_ownership":{"state":"known","unheld":[],"unheld_count":0}}"#);
        }
        out.push(row);
        if step % 7 == 6 {
            out.push(json!({
                "schema_version": 2, "run_id": RUN, "rank": 0, "attempt": attempt, "step": step,
                "ts": 1.7e9 + step as f64, "scalars": {"eval/loss": 1.5}, "seq": step, "record_version": 2,
            }));
        }
    }
    out
}

fn batch(rows: &[Value]) -> RecordBatch {
    let text: String = rows.iter().map(|row| format!("{row}\n")).collect();
    let mut reader = arrow::json::ReaderBuilder::new(schema())
        .with_batch_size(rows.len().max(1))
        .build(Cursor::new(text))
        .expect("a reader of the rows");
    let batch = reader.next().expect("one batch").expect("the rows read");
    assert!(reader.next().is_none());
    batch
}

/// The batches as one Arrow IPC stream: what `tensorquill-server detect` reads.
fn ipc(batches: &[RecordBatch]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut writer = StreamWriter::try_new(&mut out, &schema()).expect("a stream writer");
    for batch in batches {
        writer.write(batch).expect("a batch written");
    }
    writer.finish().expect("the stream finished");
    drop(writer);
    out
}

fn metadata(end: i64) -> Vec<u8> {
    format!(
        "{{\"schema_version\":2,\"run_id\":\"{RUN}\",\"rank\":0,\"end_byte_offset\":{end},\"sender_attempt\":1,\"world_size\":1}}"
    )
    .into_bytes()
}

/// The batches as the messages of one `DoPut` stream, as the SDK's sender cuts them: the descriptor
/// and schema first, then each batch with its `app_metadata`.
fn put_messages(batches: &[RecordBatch]) -> Vec<FlightData> {
    futures::executor::block_on(async {
        let descriptor = FlightDescriptor::new_path(vec!["runs".into(), RUN.into(), "rank=0".into()]);
        let mut out: Vec<FlightData> = FlightDataEncoderBuilder::new()
            .with_flight_descriptor(Some(descriptor))
            .with_schema(schema())
            .build(futures::stream::empty())
            .try_collect()
            .await
            .expect("the schema message");
        for (at, batch) in batches.iter().enumerate() {
            let encoded: Vec<FlightData> = FlightDataEncoderBuilder::new()
                .with_max_flight_data_size(usize::MAX)
                .build(futures::stream::iter([Ok(batch.clone())]))
                .try_collect()
                .await
                .expect("a batch's messages");
            for mut message in encoded.into_iter().skip(1) {
                message.app_metadata = metadata(1000 * (at as i64 + 1)).into();
                out.push(message);
            }
        }
        out
    })
}

/// The messages in the `decode_put` target's layout. The target sets the descriptor on the first
/// message itself, so it is not written.
fn put_bytes(messages: &[FlightData]) -> Vec<u8> {
    let mut out = Vec::new();
    for message in messages {
        for part in [&message.data_header, &message.data_body, &message.app_metadata] {
            out.extend_from_slice(&(part.len() as u32).to_le_bytes());
            out.extend_from_slice(part);
        }
    }
    out
}

/// What the `decode_put` target makes of `bytes`, without the descriptor: the same cut into parts.
fn put_from_bytes(mut data: &[u8]) -> Vec<FlightData> {
    let mut parts = Vec::new();
    while data.len() >= 4 {
        let length = (u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize).min(data.len() - 4);
        parts.push(data[4..4 + length].to_vec());
        data = &data[4 + length..];
    }
    parts
        .chunks(3)
        .enumerate()
        .map(|(at, three)| {
            let mut message = FlightData::new()
                .with_data_header(three[0].clone())
                .with_data_body(three.get(1).cloned().unwrap_or_default())
                .with_app_metadata(three.get(2).cloned().unwrap_or_default());
            if at == 0 {
                message = message.with_descriptor(FlightDescriptor::new_path(vec![
                    "runs".into(),
                    RUN.into(),
                    "rank=0".into(),
                ]));
            }
            message
        })
        .collect()
}

/// Whether Arrow's own stream reader, unguarded by TensorQuill, panics on `bytes`; run inside the
/// guard only so that the panic is caught and the hook stays quiet for it.
fn arrow_reader_panics(bytes: &[u8]) -> bool {
    let read = std::panic::AssertUnwindSafe(|| {
        arrow::ipc::reader::StreamReader::try_new(Cursor::new(bytes), None)
            .map(|reader| reader.collect::<Result<Vec<_>, _>>().is_ok())
            .is_ok()
    });
    matches!(decode_guard::guard(read), Err(Caught::Arrow(_)))
}

/// The first stream, of `valid` with one byte set (SplitMix64 from seed 1, at most 20000 tries), on
/// which Arrow's reader panics: a seed next to the guarded code (AQ5-R1 and AQ5-R2).
fn ipc_that_panics_arrow(valid: &[u8]) -> Vec<u8> {
    let mut state: u64 = 1;
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    for _ in 0..20_000 {
        let mut bytes = valid.to_vec();
        let at = (next() % bytes.len() as u64) as usize;
        bytes[at] = next() as u8;
        if arrow_reader_panics(&bytes) {
            return bytes;
        }
    }
    panic!("no stream of one byte set panics Arrow's reader in 20000 tries")
}

fn write(folder: &Path, name: &str, bytes: &[u8]) {
    std::fs::create_dir_all(folder).expect("the seed folder");
    std::fs::write(folder.join(name), bytes).expect("a seed written");
}

fn main() {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("the corpus folder, as the first argument"),
    );
    let thresholds = Arc::new(Thresholds::default());
    // (name, the batches): one batch of a few rows; a run cut into batches of 7; a resumed run (two
    // attempts, the second rewriting steps); and a run with a gradient spike.
    let mut runs: Vec<(&str, Vec<RecordBatch>)> = vec![("small", vec![batch(&rows(4, 1, 0, None))])];
    runs.push(("batches", rows(30, 1, 0, None).chunks(7).map(batch).collect()));
    let mut resumed = rows(20, 1, 0, None);
    resumed.extend(rows(15, 2, 12, None));
    runs.push(("resumed", resumed.chunks(10).map(batch).collect()));
    runs.push(("spike", rows(40, 1, 0, Some(30)).chunks(16).map(batch).collect()));
    for (name, batches) in &runs {
        let stream = ipc(batches);
        detect_stream(Cursor::new(&stream), &thresholds).unwrap_or_else(|error| panic!("{name}: detect: {error}"));
        write(&out.join("detect_stream"), &format!("seed-{name}"), &stream);
        let put = put_bytes(&put_messages(batches));
        let decoded =
            decode_put(put_from_bytes(&put)).unwrap_or_else(|status| panic!("{name}: decode_put: {status:?}"));
        let steps: usize = decoded
            .iter()
            .flat_map(|batch| batch.step_rows.iter().map(|(_, rows)| rows.num_rows()))
            .sum();
        assert!(steps > 0, "{name}: decode_put gave no step rows");
        write(&out.join("decode_put"), &format!("seed-{name}"), &put);
        println!(
            "{name}: {} batches, IPC stream {} bytes, DoPut seed {} bytes, {steps} step rows",
            batches.len(),
            stream.len(),
            put.len()
        );
    }
    // Two seeds Arrow panics on, which the guard must turn into errors: the small run's batch with its
    // body cut to half (the suite's `cut_body`) for `decode_put`, and the small run's IPC stream
    // with one byte set for `detect_stream`.
    let small = &runs[0].1;
    let mut messages = put_messages(small);
    let body = messages[1].data_body.to_vec();
    messages[1].data_body = body[..body.len() / 2].to_vec().into();
    let put = put_bytes(&messages);
    match decode_put(put_from_bytes(&put)) {
        Err(status) if status.message().starts_with("a malformed record batch") => {}
        other => panic!("the cut body: decode_put gave {:?}", other.map(|batches| batches.len())),
    }
    write(&out.join("decode_put"), "seed-malformed", &put);
    let stream = ipc_that_panics_arrow(&ipc(small));
    match decode_guard::offline(std::panic::AssertUnwindSafe(|| {
        detect_stream(Cursor::new(&stream), &thresholds)
    })) {
        Err(error) if error.starts_with(MALFORMED_INPUT) => {}
        other => panic!("the set byte: detect gave {other:?}"),
    }
    write(&out.join("detect_stream"), "seed-malformed", &stream);
    println!(
        "malformed: DoPut seed {} bytes, IPC stream {} bytes",
        put.len(),
        stream.len()
    );
}
