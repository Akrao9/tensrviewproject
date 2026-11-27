//! Lane AQ-5 (SV7): the decode of what a client sends, driven with seeded random and mutated input.
//!
//! Two entries, each from bytes up to the rows the detectors receive, and on to the detectors:
//!
//! - `ingest::decode_put`: the `DoPut` stream's messages in (a descriptor, a schema, record batches
//!   with their `app_metadata`), each batch's rows out as the server takes them apart before placing or
//!   keeping anything. The step rows that come out are then judged (`detect::judge`).
//! - `detect::detect_stream`: an Arrow IPC stream in, the report out (`tensorquill-server detect`,
//!   which the SDK's Rust detectors use).
//!
//! Each is given seeded random bytes and valid input mutated byte by byte, and must never panic: it
//! returns rows (or a report) or an error. A panic is caught (`catch_unwind`) and named with its seed
//! and iteration, so one failing case is enough to reproduce it. An abort (an allocation that cannot be
//! made) cannot be caught: the seed and iteration are printed before each call when
//! `TENSORQUILL_DECODE_TRACE` is set, so the last line printed names the input.
//!
//! The randomness is a few lines of SplitMix64: nothing is added to the server's dependencies.
//! `TENSORQUILL_DECODE_ITERATIONS` raises the number of inputs per test (default `ITERATIONS`).

use std::collections::BTreeMap;
use std::io::Cursor;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::{FlightData, FlightDescriptor};
use futures::TryStreamExt;
use serde_json::{json, Value};
use tensorquill_server::detect::{self, Thresholds};
use tensorquill_server::ingest::decode_put;

const ITERATIONS: u64 = 300;
const RUN: &str = "aq5";

fn iterations() -> u64 {
    std::env::var("TENSORQUILL_DECODE_ITERATIONS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(ITERATIONS)
}

fn trace(test: &str, seed: u64, iteration: u64, what: &str) {
    if std::env::var_os("TENSORQUILL_DECODE_TRACE").is_some() {
        eprintln!("{test} seed {seed} iteration {iteration}: {what}");
    }
}

/// SplitMix64: a seeded, repeatable stream of numbers.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// A number below `bound` (which must not be 0).
    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
    fn bytes(&mut self, length: usize) -> Vec<u8> {
        (0..length).map(|_| self.next() as u8).collect()
    }
    /// `plus` to `plus + below - 1` random bytes.
    fn some_bytes(&mut self, plus: usize, below: u64) -> Vec<u8> {
        let length = plus + self.below(below) as usize;
        self.bytes(length)
    }
    fn float(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// --- valid input, as the SDK sends it ------------------------------------------------------------

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

/// A run's rows as `record_to_row` makes them, drawn from `rng`: steps of one or two attempts with the
/// columns the detectors read, and now and then a scalar-only row.
fn rows(rng: &mut Rng, count: usize) -> Vec<Value> {
    let attempts: i64 = if rng.chance(30) { 2 } else { 1 };
    let mut out = Vec::new();
    for at in 0..count {
        let attempt = if attempts == 2 && at >= count / 2 { 2 } else { 1 };
        let step = if attempt == 2 {
            (at - count / 2 + count / 4) as i64
        } else {
            at as i64
        };
        let wobble = |rng: &mut Rng, centre: f64| centre * (0.95 + 0.1 * rng.float());
        let grad = wobble(rng, 1.0);
        let mut row = json!({
            "schema_version": 2, "run_id": RUN, "rank": 0, "attempt": attempt, "step": step,
            "ts": 1.7e9 + step as f64, "grad_norm": grad, "lr": [1e-3],
            "clip_grad_norm": grad * 1.2, "step_skipped": false, "grad_scale": 65536.0,
            "scalars": {"train/loss": 2.5 - step as f64 / 400.0},
            "grad_norms": {"p0.weight": wobble(rng, 0.5), "p1.weight": wobble(rng, 0.4)},
            "update_rms": {"p0.weight": wobble(rng, 1e-3), "p1.weight": wobble(rng, 1e-3)},
            "weights_unchanged": [],
            "moe_top_k": {"r0": 2},
            "moe_counts": {"r0": [256, 250, 262, 256]},
            "layers": {"blk0": {"mean": 0.01, "std": wobble(rng, 1.0), "abs_max": wobble(rng, 3.0), "zero_frac": 0.0}},
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
        if rng.chance(5) {
            out.push(json!({
                "schema_version": 2, "run_id": RUN, "rank": 0, "attempt": attempt, "step": step,
                "ts": 1.7e9 + step as f64, "scalars": {"eval/loss": 1.5}, "seq": at as i64, "record_version": 2,
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
        .unwrap();
    let batch = reader.next().unwrap().unwrap();
    assert!(reader.next().is_none());
    batch
}

/// The batches as one Arrow IPC stream: what `tensorquill-server detect` reads.
fn ipc(batches: &[RecordBatch]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut writer = StreamWriter::try_new(&mut out, &schema()).unwrap();
    for batch in batches {
        writer.write(batch).unwrap();
    }
    writer.finish().unwrap();
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
            .unwrap();
        for (at, batch) in batches.iter().enumerate() {
            let encoded: Vec<FlightData> = FlightDataEncoderBuilder::new()
                .with_max_flight_data_size(usize::MAX)
                .build(futures::stream::iter([Ok(batch.clone())]))
                .try_collect()
                .await
                .unwrap();
            for mut message in encoded.into_iter().skip(1) {
                message.app_metadata = metadata(1000 * (at as i64 + 1)).into();
                out.push(message);
            }
        }
        out
    })
}

fn valid_batches(rng: &mut Rng) -> Vec<RecordBatch> {
    let count = 20 + rng.below(60) as usize;
    let all = rows(rng, count);
    let size = 1 + rng.below(30) as usize;
    all.chunks(size).map(batch).collect()
}

// --- mutation ------------------------------------------------------------------------------------

/// `bytes` changed in one of several ways, drawn from `rng`; what was done, for the message.
fn mutate(rng: &mut Rng, bytes: &mut Vec<u8>) -> String {
    if bytes.is_empty() {
        bytes.extend(rng.some_bytes(1, 16));
        return "filled".into();
    }
    let len = bytes.len() as u64;
    match rng.below(8) {
        0 => {
            let flips = 1 + rng.below(4);
            for _ in 0..flips {
                let at = rng.below(len) as usize;
                bytes[at] ^= 1 << rng.below(8);
            }
            format!("{flips} bits flipped")
        }
        1 => {
            let at = rng.below(len) as usize;
            bytes[at] = rng.next() as u8;
            format!("byte {at} set")
        }
        2 => {
            let at = rng.below(len) as usize;
            bytes[at] = *[0x00, 0xff, 0x7f, 0x80].get(rng.below(4) as usize).unwrap();
            format!("byte {at} set to an edge")
        }
        3 => {
            let to = rng.below(len) as usize;
            bytes.truncate(to);
            format!("cut at {to}")
        }
        4 => {
            let at = rng.below(len) as usize;
            let extra = rng.some_bytes(1, 64);
            bytes.splice(at..at, extra);
            format!("bytes put in at {at}")
        }
        5 => {
            let from = rng.below(len) as usize;
            let to = (from + 1 + rng.below(64) as usize).min(bytes.len());
            bytes.drain(from..to);
            format!("bytes {from}..{to} taken out")
        }
        6 => {
            // A little-endian length or offset made large, as a flatbuffer or IPC prefix holds it.
            let at = rng.below(len.saturating_sub(4).max(1)) as usize;
            let value: u32 = *[u32::MAX, 0x7fff_ffff, 0x8000_0000, 0xffff]
                .get(rng.below(4) as usize)
                .unwrap();
            for (offset, byte) in value.to_le_bytes().iter().enumerate() {
                if let Some(slot) = bytes.get_mut(at + offset) {
                    *slot = *byte;
                }
            }
            format!("u32 {value:#x} written at {at}")
        }
        _ => {
            let from = rng.below(len) as usize;
            let to = (from + 1 + rng.below(128) as usize).min(bytes.len());
            let copy = bytes[from..to].to_vec();
            let at = rng.below(len) as usize;
            bytes.splice(at..at, copy);
            format!("bytes {from}..{to} copied to {at}")
        }
    }
}

/// A valid IPC stream with one to three mutations.
fn mutated_ipc(rng: &mut Rng, _: u64) -> (Vec<u8>, String) {
    let mut bytes = ipc(&valid_batches(rng));
    let times = 1 + rng.below(3);
    let what = (0..times)
        .map(|_| mutate(rng, &mut bytes))
        .collect::<Vec<_>>()
        .join(", ");
    (bytes, what)
}

/// A valid `DoPut` stream with one part of one message mutated.
fn mutated_put(rng: &mut Rng, _: u64) -> (Vec<FlightData>, String) {
    let mut messages = put_messages(&valid_batches(rng));
    let at = rng.below(messages.len() as u64) as usize;
    let part = rng.below(3);
    let mut bytes = match part {
        0 => messages[at].data_header.to_vec(),
        1 => messages[at].data_body.to_vec(),
        _ => messages[at].app_metadata.to_vec(),
    };
    let what = mutate(rng, &mut bytes);
    match part {
        0 => messages[at].data_header = bytes.into(),
        1 => messages[at].data_body = bytes.into(),
        _ => messages[at].app_metadata = bytes.into(),
    }
    let name = ["data_header", "data_body", "app_metadata"][part as usize];
    (messages, format!("message {at}, {name}: {what}"))
}

// --- the checks ----------------------------------------------------------------------------------

fn thresholds() -> Arc<Thresholds> {
    Arc::new(Thresholds::default())
}

/// The decode of a `DoPut` stream, and the detectors over its step rows: never a panic.
fn put_then_judge(messages: Vec<FlightData>) {
    let Ok(decoded) = decode_put(messages) else {
        return;
    };
    let steps: Vec<RecordBatch> = decoded
        .iter()
        .flat_map(|batch| batch.step_rows.iter().map(|(_, rows)| rows.clone()))
        .collect();
    if let Ok(detectors) = detect::judge(0, &steps, &thresholds()) {
        let _ = detect::report(RUN, 0, &detectors, 0);
    }
}

fn detect_bytes(bytes: &[u8]) {
    let _ = detect::detect_stream(Cursor::new(bytes), &thresholds());
}

/// The panics found by this lane, each held by an ignored test below and listed in the lane's report.
/// Both are raised inside arrow's IPC reader (`arrow-ipc` 60.1.0 reading a record batch message whose
/// header names buffers its body does not hold, through `arrow-buffer`), on both entries: the server's
/// `DoPut` decode and `tensorquill-server detect`. A panic whose message holds none of these fails the
/// test it happens in.
const KNOWN_PANICS: [(&str, &str); 2] = [
    (
        "AQ5-R1",
        "the offset of the new Buffer cannot exceed the existing length",
    ),
    ("AQ5-R2", "assertion failed: prefix.is_empty() && suffix.is_empty()"),
];

thread_local! {
    /// Set while a check runs, so the panic hook stays quiet for the panics `catch_unwind` takes.
    static CHECKING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn quiet_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !CHECKING.with(std::cell::Cell::get) {
                default(info);
            }
        }));
    });
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|text| text.to_string()))
        .unwrap_or_else(|| "a panic without a message".into())
}

/// The finding of a known panic, or None.
fn known(text: &str) -> Option<&'static str> {
    KNOWN_PANICS
        .iter()
        .find(|(_, words)| text.contains(words))
        .map(|(finding, _)| *finding)
}

/// `check` on `input`: Ok, or the text of the panic.
fn run<T>(check: &impl Fn(T), input: T) -> Result<(), String> {
    quiet_hook();
    CHECKING.with(|flag| flag.set(true));
    let result = catch_unwind(AssertUnwindSafe(|| check(input)));
    CHECKING.with(|flag| flag.set(false));
    result.map_err(|payload| panic_text(payload.as_ref()))
}

/// Runs `check` on each input `make` gives for `iterations()` iterations. A panic of `KNOWN_PANICS`
/// is counted and printed; any other fails the test, naming its seed and iteration.
fn never_panics<T>(test: &str, seed: u64, mut make: impl FnMut(&mut Rng, u64) -> (T, String), check: impl Fn(T)) {
    let mut rng = Rng::new(seed);
    let mut unknown = Vec::new();
    let mut held: BTreeMap<&str, Vec<u64>> = BTreeMap::new();
    for iteration in 0..iterations() {
        let (input, what) = make(&mut rng, iteration);
        trace(test, seed, iteration, &what);
        if let Err(text) = run(&check, input) {
            match known(&text) {
                Some(finding) => held.entry(finding).or_default().push(iteration),
                None => unknown.push(format!("seed {seed} iteration {iteration}: {what}: {text}")),
            }
        }
    }
    for (finding, iterations) in &held {
        eprintln!(
            "{test}: {} inputs gave the known panic {finding} (seed {seed}, iterations {:?})",
            iterations.len(),
            &iterations[..iterations.len().min(10)]
        );
    }
    assert!(
        unknown.is_empty(),
        "{test}: {} inputs panicked:\n{}",
        unknown.len(),
        unknown.join("\n")
    );
}

#[test]
fn the_valid_input_decodes_to_its_rows_on_both_entries() {
    let mut rng = Rng::new(1);
    let batches = valid_batches(&mut rng);
    let sent: usize = batches.iter().map(RecordBatch::num_rows).sum();
    let decoded = decode_put(put_messages(&batches)).expect("a valid stream decodes");
    assert_eq!(decoded.len(), batches.len());
    let rows: usize = decoded
        .iter()
        .map(|batch| {
            batch.step_rows.iter().map(|(_, rows)| rows.num_rows()).sum::<usize>()
                + batch.scalar_rows.iter().map(|(_, rows)| rows.num_rows()).sum::<usize>()
        })
        .sum();
    assert_eq!(
        rows, sent,
        "every row of a valid stream is a step row or a scalar-only row"
    );
    assert!(decoded.iter().all(|batch| batch.dropped.is_empty()));
    let report = detect::detect_stream(Cursor::new(ipc(&batches)), &thresholds()).expect("a valid stream is judged");
    assert!(report["steps"].as_u64().unwrap() > 0, "{report}");
}

#[test]
fn a_put_stream_refused_by_its_metadata_or_descriptor_is_an_error_not_a_panic() {
    let mut rng = Rng::new(2);
    let batches = valid_batches(&mut rng);
    let mut messages = put_messages(&batches);
    messages[1].app_metadata = b"{\"schema_version\":9,\"run_id\":\"aq5\",\"rank\":0,\"end_byte_offset\":1}"
        .to_vec()
        .into();
    assert!(decode_put(messages).unwrap_err().message().contains("schema_version 9"));
    let mut messages = put_messages(&batches);
    messages[0].flight_descriptor = Some(FlightDescriptor::new_path(vec![
        "runs".into(),
        "../x".into(),
        "rank=0".into(),
    ]));
    assert!(decode_put(messages).is_err());
    assert!(decode_put(Vec::new()).unwrap().is_empty());
}

#[test]
fn random_bytes_never_panic_the_detect_stream_decode() {
    never_panics(
        "random_bytes_detect",
        11,
        |rng, _| {
            let length = rng.below(4096) as usize;
            (rng.bytes(length), format!("{length} random bytes"))
        },
        |bytes| detect_bytes(&bytes),
    );
}

#[test]
fn mutated_ipc_streams_never_panic_the_detect_stream_decode_or_the_detectors() {
    never_panics("mutated_ipc_detect", 12, mutated_ipc, |bytes| detect_bytes(&bytes));
}

#[test]
fn random_flight_messages_never_panic_the_put_decode() {
    never_panics(
        "random_messages_put",
        13,
        |rng, _| {
            let count = 1 + rng.below(4) as usize;
            let mut messages = Vec::new();
            for at in 0..count {
                let mut message = FlightData::new()
                    .with_data_header(rng.some_bytes(0, 512))
                    .with_data_body(rng.some_bytes(0, 2048))
                    .with_app_metadata(rng.some_bytes(0, 128));
                if at == 0 && rng.chance(70) {
                    message = message.with_descriptor(FlightDescriptor::new_path(vec![
                        "runs".into(),
                        RUN.into(),
                        "rank=0".into(),
                    ]));
                }
                messages.push(message);
            }
            (messages, format!("{count} random messages"))
        },
        put_then_judge,
    );
}

#[test]
fn mutated_put_streams_never_panic_the_put_decode_or_the_detectors() {
    never_panics("mutated_put", 14, mutated_put, put_then_judge);
}

#[test]
fn mutated_column_values_never_panic_the_detectors() {
    // The body of a record batch holds the values: a mutation there keeps the stream decodable more
    // often than one in a header, so more of these inputs reach the detectors.
    never_panics(
        "mutated_body_put",
        15,
        |rng, _| {
            let mut messages = put_messages(&valid_batches(rng));
            let mut what = Vec::new();
            for message in messages.iter_mut().skip(1) {
                if message.data_body.is_empty() || !rng.chance(60) {
                    continue;
                }
                let mut body = message.data_body.to_vec();
                for _ in 0..1 + rng.below(8) {
                    let at = rng.below(body.len() as u64) as usize;
                    body[at] = rng.next() as u8;
                }
                message.data_body = body.into();
                what.push("body bytes set");
            }
            (messages, what.join(", "))
        },
        put_then_judge,
    );
}

// --- the panics found, each held at a case that shows it ------------------------------------------

/// A record batch message whose body is cut short of what its header names.
fn cut_body() -> Vec<FlightData> {
    let mut rng = Rng::new(1);
    let mut messages = put_messages(&valid_batches(&mut rng));
    let body = messages[1].data_body.to_vec();
    messages[1].data_body = body[..body.len() / 2].to_vec().into();
    messages
}

#[test]
#[ignore = "AQ5-R1: arrow-ipc 60.1.0 panics (arrow-buffer Buffer::slice) on a record batch message whose \
            body is shorter than its header says, so a client's malformed batch panics the server's \
            stream task in place of ending the stream with an error; held for the author, not fixed here"]
fn a_put_batch_whose_body_is_cut_short_is_refused_without_a_panic() {
    let found = run(&|messages| put_then_judge(messages), cut_body());
    assert_eq!(found, Ok(()), "the decode panicked");
    assert!(decode_put(cut_body()).is_err(), "a batch cut short is refused");
}

#[test]
fn the_cut_body_case_still_panics_with_the_known_message() {
    // What keeps the ignored test above honest: the case still gives AQ5-R1 today. When arrow or the
    // server stops panicking here, this fails, and the ignored test is to be turned on.
    let found = run(&|messages| put_then_judge(messages), cut_body()).unwrap_err();
    assert_eq!(known(&found), Some("AQ5-R1"), "{found}");
}

#[test]
#[ignore = "AQ5-R1 and AQ5-R2 on `tensorquill-server detect`: arrow-ipc 60.1.0's stream reader panics \
            on a mutated record batch header, so a malformed Arrow stream given to `detect` panics in \
            place of an error; held for the author, not fixed here"]
fn mutated_ipc_streams_that_panic_today_give_an_error() {
    let mut rng = Rng::new(12);
    let mut found = Vec::new();
    for iteration in 0..iterations() {
        let (bytes, what) = mutated_ipc(&mut rng, iteration);
        if let Err(text) = run(&|bytes: Vec<u8>| detect_bytes(&bytes), bytes) {
            found.push(format!("iteration {iteration}: {what}: {text}"));
        }
    }
    assert!(found.is_empty(), "{}", found.join("\n"));
}

#[test]
fn a_known_detect_case_still_panics_with_a_known_message() {
    let mut rng = Rng::new(12);
    let first = (0..iterations()).find_map(|iteration| {
        let (bytes, _) = mutated_ipc(&mut rng, iteration);
        run(&|bytes: Vec<u8>| detect_bytes(&bytes), bytes).err()
    });
    let text = first.expect("seed 12 still gives a panic among its mutated streams");
    assert!(known(&text).is_some(), "{text}");
}
