//! Lane AO-3 of wave AO: the ingest and storage findings on `453f1e2`.
//!
//! - BUG-07: the order a rank's attempts are placed in (`attempts.json`) is the order they first
//!   appear in the rows as sent, whatever the batch size. On `453f1e2` a batch holding training and
//!   scalar-only rows placed the training rows' attempts first, so the same rows sent as one batch
//!   and one row at a time gave different orders, and a scalar that a later attempt rolled back
//!   came back.
//! - BUG-08: an attempt stored but not listed in `attempts.json` goes just before the newest, by
//!   number among such, as Python's `evidence._attempt_place` reads it; and a folder holding only
//!   `scalars-*` objects is one of the rank's attempts.
//! - BUG-02, the server side: a row whose `record_version` is 2 and whose `seq` is null (a version 2
//!   record that is not a scalar-only record) is dropped and counted, never stored as a training
//!   step in a part that every Python reader then refuses.

mod lane_aa3_support;

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Array, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::record_batch::RecordBatch;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::{FlightClient, FlightData, FlightDescriptor, PutResult};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use lane_aa3_support::{json_file, start, TestServer};
use object_store::local::LocalFileSystem;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{json, Value};
use tensorquill_server::detect::{replay, Thresholds};
use tensorquill_server::ingest::Ingest;
use tensorquill_server::storage::Storage;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Channel;

// --- protocol 5's rows ------------------------------------------------------------------------------

fn scalars_map() -> DataType {
    DataType::Map(
        Arc::new(Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Float64, true),
            ])),
            false,
        )),
        false,
    )
}

/// Protocol 5's row layout for the columns these tests need (`WAVE_AL_CONTRACT.md` sections 3 and
/// 4.4): row version 2 on every row, the record's own version in `record_version`, `scalars` as a
/// map, and `seq`, null on a step row.
fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("schema_version", DataType::Int32, true),
        Field::new("run_id", DataType::Utf8, true),
        Field::new("rank", DataType::Int32, true),
        Field::new("attempt", DataType::Int64, true),
        Field::new("step", DataType::Int64, true),
        Field::new("ts", DataType::Float64, true),
        Field::new("grad_norm", DataType::Float64, true),
        Field::new("scalars", scalars_map(), true),
        Field::new("seq", DataType::Int64, true),
        Field::new("record_version", DataType::Int32, true),
        Field::new("extra_json", DataType::Utf8, true),
    ]))
}

/// `schema()` without the columns named.
fn schema_without(names: &[&str]) -> Arc<Schema> {
    Arc::new(Schema::new(
        schema()
            .fields()
            .iter()
            .filter(|field| !names.contains(&field.name().as_str()))
            .map(|field| field.as_ref().clone())
            .collect::<Vec<_>>(),
    ))
}

/// A training row: row version 2, the record's version 1.
fn step_row(run: &str, attempt: i64, step: i64) -> Value {
    json!({"schema_version": 2, "run_id": run, "rank": 0, "attempt": attempt, "step": step,
           "ts": step as f64, "grad_norm": 1.0, "record_version": 1})
}

/// A whole scalar-only row (section 4.4): the record's version 2 and its `seq`.
fn scalar_row(run: &str, attempt: i64, step: i64, seq: i64, value: f64) -> Value {
    json!({"schema_version": 2, "run_id": run, "rank": 0, "attempt": attempt, "step": step,
           "ts": 1000.0 + seq as f64, "scalars": {"eval_loss": value}, "seq": seq, "record_version": 2})
}

/// What an encoder makes of a version 2 line with no `"record": "scalars"` (BUG-02, TQ-10): the
/// record's version 2 in `record_version`, its values, and a null `seq`.
fn v2_row_without_seq(run: &str, attempt: i64, step: i64) -> Value {
    json!({"schema_version": 2, "run_id": run, "rank": 0, "attempt": attempt, "step": step,
           "ts": step as f64, "scalars": {"loss": 1.0}, "record_version": 2})
}

fn decode(schema: Arc<Schema>, rows: &[Value]) -> RecordBatch {
    let text: String = rows.iter().map(|row| format!("{row}\n")).collect();
    let mut reader = arrow::json::ReaderBuilder::new(schema)
        .with_batch_size(rows.len().max(1))
        .build(Cursor::new(text))
        .unwrap();
    let batch = reader.next().unwrap().unwrap();
    assert!(reader.next().is_none());
    batch
}

fn batch(rows: &[Value]) -> RecordBatch {
    decode(schema(), rows)
}

// --- one DoPut stream, batch metadata 2, with or without `sender_attempt` --------------------------

struct Put {
    run: String,
    sender: Option<i64>,
    messages: mpsc::UnboundedSender<Result<FlightData, FlightError>>,
    acks: BoxStream<'static, Result<PutResult, FlightError>>,
}

async fn open(address: &str, run: &str, schema: Arc<Schema>, sender: Option<i64>) -> Put {
    let channel = Channel::from_shared(address.to_string())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightClient::new(channel);
    let (messages, outgoing) = mpsc::unbounded_channel();
    let descriptor = FlightDescriptor::new_path(vec!["runs".into(), run.into(), "rank=0".into()]);
    let opening: Vec<FlightData> = FlightDataEncoderBuilder::new()
        .with_flight_descriptor(Some(descriptor))
        .with_schema(schema)
        .build(futures::stream::empty())
        .try_collect()
        .await
        .unwrap();
    for message in opening {
        messages.send(Ok(message)).unwrap();
    }
    let acks = match client.do_put(UnboundedReceiverStream::new(outgoing)).await {
        Ok(acks) => acks,
        Err(error) => futures::stream::once(async move { Err(error) }).boxed(),
    };
    Put {
        run: run.to_string(),
        sender,
        messages,
        acks,
    }
}

impl Put {
    async fn send(&mut self, batch: RecordBatch, end_byte_offset: i64) {
        let mut metadata = json!({"schema_version": 2, "run_id": self.run, "rank": 0, "min_step": null,
                                  "max_step": null, "end_byte_offset": end_byte_offset});
        if let Some(sender) = self.sender {
            metadata["sender_attempt"] = json!(sender);
        }
        let metadata = metadata.to_string();
        let encoded: Vec<FlightData> = FlightDataEncoderBuilder::new()
            .with_max_flight_data_size(usize::MAX)
            .build(futures::stream::iter([Ok(batch)]))
            .try_collect()
            .await
            .unwrap();
        for mut message in encoded.into_iter().skip(1) {
            message.app_metadata = metadata.clone().into();
            let _ = self.messages.send(Ok(message));
        }
    }

    async fn reply(&mut self) -> Result<i64, tonic::Status> {
        match tokio::time::timeout(Duration::from_secs(20), self.acks.next())
            .await
            .expect("no reply")
        {
            Some(Ok(result)) => {
                let reply: Value = serde_json::from_slice(&result.app_metadata).unwrap();
                Ok(reply["end_byte_offset"].as_i64().unwrap())
            }
            Some(Err(FlightError::Tonic(status))) => Err(*status),
            Some(Err(other)) => panic!("{other}"),
            None => panic!("the stream ended without a reply"),
        }
    }

    async fn finish(self) {
        drop(self.messages);
        let mut acks = self.acks;
        while acks.next().await.is_some() {}
    }
}

/// Sends `batches` on one stream, each batch's end offset its last row's place (one line a row),
/// and returns the offsets acknowledged.
async fn send(server: &TestServer, run: &str, sender: Option<i64>, batches: Vec<RecordBatch>) -> Vec<i64> {
    let mut put = open(&server.address, run, batches[0].schema(), sender).await;
    let (mut end, mut acked) = (0, Vec::new());
    for batch in batches {
        end += batch.num_rows() as i64;
        put.send(batch, end).await;
        acked.push(put.reply().await.expect("an acknowledgement"));
    }
    put.finish().await;
    acked
}

// --- what the store holds --------------------------------------------------------------------------

fn rank_dir(root: &Path, run: &str) -> PathBuf {
    root.join("runs").join(run).join("rank=0")
}

/// The attempts of the rank's `attempts.json`, in its order.
fn attempts_json(root: &Path, run: &str) -> Vec<i64> {
    json_file(&rank_dir(root, run).join("attempts.json"))["attempts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["attempt"].as_i64().unwrap())
        .collect()
}

fn read_file(path: &Path) -> Vec<RecordBatch> {
    let file = std::fs::File::open(path).unwrap();
    ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

/// The files of an attempt whose names start with `prefix` and end with `.parquet`, sorted.
fn files(root: &Path, run: &str, attempt: i64, prefix: &str) -> Vec<PathBuf> {
    let dir = rank_dir(root, run).join(format!("attempt={attempt}"));
    let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|entries| entries.map(|entry| entry.unwrap().path()).collect())
        .unwrap_or_default();
    found.retain(|path| {
        let name = path.file_name().unwrap().to_string_lossy();
        name.starts_with(prefix) && name.ends_with(".parquet")
    });
    found.sort();
    found
}

/// Every value of the Int64 column `column` in `paths`, in file and row order; a null is -1.
fn column_of(paths: &[PathBuf], column: &str) -> Vec<i64> {
    let mut out = Vec::new();
    for path in paths {
        for batch in read_file(path) {
            let values = batch.column_by_name(column).unwrap();
            let values = values.as_any().downcast_ref::<Int64Array>().unwrap();
            out.extend(values.iter().map(|value| value.unwrap_or(-1)));
        }
    }
    out
}

/// The steps of an attempt's parts.
fn part_steps(root: &Path, run: &str, attempt: i64) -> Vec<i64> {
    column_of(&files(root, run, attempt, "part-"), "step")
}

/// The steps of an attempt's `scalars-*` objects.
fn scalar_steps(root: &Path, run: &str, attempt: i64) -> Vec<i64> {
    column_of(&files(root, run, attempt, "scalars-"), "step")
}

fn rank_entry(ingest: &Ingest, run: &str) -> Value {
    let statuses = ingest.watchdog_statuses();
    let status = statuses.get(run).unwrap_or_else(|| panic!("no status of {run}"));
    status["ranks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|rank| rank["rank"] == 0)
        .cloned()
        .unwrap()
}

fn storage(root: &Path) -> Storage {
    Storage::new(Arc::new(LocalFileSystem::new_with_prefix(root).unwrap()), "".into())
}

// --- BUG-07: attempts are placed in the order their rows were sent -----------------------------------

/// TQ-01's three sessions under one run ID on rank 0, as their lines stand in the rank's file: A
/// logs `eval_loss` at an explicit step 50 and closes (one scalar-only record), B writes training
/// step 0, C writes training step 100 and is the sender's attempt. The numbers are the sessions'
/// start times, so A < B < C.
const A: i64 = 1_732_000_000_000;
const B: i64 = 1_732_000_100_000;
const C: i64 = 1_732_000_200_000;

fn tq01_rows(run: &str) -> Vec<Value> {
    vec![
        scalar_row(run, A, 50, 0, 99.0),
        step_row(run, B, 0),
        step_row(run, C, 100),
    ]
}

/// The rows sent as one batch and one row a batch give the same order: the order the attempts first
/// appear in the rows. Fails on `453f1e2`: one batch gives B, A, C (the training rows' attempts were
/// placed before the scalar rows'), so A's point at step 50 is read as rolled back only by C's first
/// step, 100, and comes back in the default series.
#[tokio::test]
async fn bug07_attempts_are_placed_in_row_order_whatever_the_batch_size_with_a_sender_attempt() {
    for (cut, name) in [(3, "one_batch"), (1, "one_row_a_batch")] {
        let run = format!("ao3_bug07_sender_{name}");
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let server = start(root).await;
        let rows = tq01_rows(&run);
        let batches: Vec<RecordBatch> = rows.chunks(cut).map(batch).collect();
        let acked = send(&server, &run, Some(C), batches).await;
        assert_eq!(acked.last(), Some(&3), "{name}: acknowledged to the end: {acked:?}");
        assert_eq!(attempts_json(root, &run), [A, B, C], "{name}");
        // Every row is where it belongs: A's point in a scalars object, B's and C's steps in parts.
        assert_eq!(scalar_steps(root, &run, A), [50], "{name}");
        assert_eq!(part_steps(root, &run, B), [0], "{name}");
        assert_eq!(part_steps(root, &run, C), [100], "{name}");
        server.ingest.shutdown().await.unwrap();
    }
}

/// The same with no `sender_attempt` (a sender older than claims): its attempts are placed by number
/// against the newest, one at a time, so here too in the order they appear. Fails on `453f1e2` for
/// the one batch (B, A, C: A, placed last, is below C's number and goes just before it).
#[tokio::test]
async fn bug07_attempts_are_placed_in_row_order_whatever_the_batch_size_without_a_sender_attempt() {
    for (cut, name) in [(3, "one_batch"), (1, "one_row_a_batch")] {
        let run = format!("ao3_bug07_nosender_{name}");
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let server = start(root).await;
        let batches: Vec<RecordBatch> = tq01_rows(&run).chunks(cut).map(batch).collect();
        send(&server, &run, None, batches).await;
        assert_eq!(attempts_json(root, &run), [A, B, C], "{name}");
        server.ingest.shutdown().await.unwrap();
    }
}

/// Interleaved: B's step, A's point, B's next step, C's step, in one batch and one row a batch. The
/// first appearance decides (B before A), as the local reader orders a file's attempts.
#[tokio::test]
async fn bug07_an_attempt_that_appears_again_later_in_the_batch_keeps_its_first_place() {
    for (cut, name) in [(4, "one_batch"), (1, "one_row_a_batch")] {
        let run = format!("ao3_bug07_interleaved_{name}");
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let server = start(root).await;
        let rows = vec![
            step_row(&run, B, 0),
            scalar_row(&run, A, 50, 0, 99.0),
            step_row(&run, B, 1),
            step_row(&run, C, 100),
        ];
        let batches: Vec<RecordBatch> = rows.chunks(cut).map(batch).collect();
        send(&server, &run, Some(C), batches).await;
        assert_eq!(attempts_json(root, &run), [B, A, C], "{name}");
        assert_eq!(part_steps(root, &run, B), [0, 1], "{name}");
        server.ingest.shutdown().await.unwrap();
    }
}

/// A pin: a batch refused whole (a negative step) places no attempt of it, and a row dropped for a
/// null key never places its attempt.
#[tokio::test]
async fn bug07_a_refused_batch_places_nothing_and_a_dropped_rows_attempt_is_not_placed() {
    let run = "ao3_bug07_refused";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    let mut negative = step_row(run, B, 0);
    negative["step"] = json!(-1);
    let mut put = open(&server.address, run, schema(), Some(C)).await;
    put.send(batch(&[scalar_row(run, A, 50, 0, 1.0), negative]), 2).await;
    assert!(put.reply().await.is_err(), "a negative step refuses the batch");
    // The stream's claim is placed as it opens; no attempt of the refused rows is.
    assert_eq!(attempts_json(root, run), [C], "nothing of a refused batch is placed");
    // A row whose step is null is dropped and counted; its attempt (B) is not placed.
    let mut null_step = step_row(run, B, 0);
    null_step["step"] = Value::Null;
    let acked = send(&server, run, Some(C), vec![batch(&[null_step, step_row(run, C, 100)])]).await;
    assert_eq!(acked, [2]);
    assert_eq!(attempts_json(root, run), [C]);
    assert_eq!(rank_entry(&server.ingest, run)["dropped_row_keys"], json!({"step": 1}));
    server.ingest.shutdown().await.unwrap();
}

// --- BUG-08: an attempt the order does not list, as Python reads it ----------------------------------

/// Writes the rank's `attempts.json` with `entries` (attempt, by).
fn write_attempts(root: &Path, run: &str, entries: &[(i64, &str)]) {
    let dir = rank_dir(root, run);
    std::fs::create_dir_all(&dir).unwrap();
    let document = json!({
        "schema_version": 1, "run_id": run, "rank": 0,
        "attempts": entries.iter().map(|(attempt, by)| json!({"attempt": attempt, "by": by, "first_seen_ms": 0})).collect::<Vec<_>>(),
    });
    std::fs::write(dir.join("attempts.json"), document.to_string()).unwrap();
}

/// One part of `attempt` at step 0, with `grad_norm`.
async fn put_part(storage: &Storage, run: &str, attempt: i64, grad_norm: f64) {
    // Set as an array: JSON has no NaN.
    let rows = batch(&[step_row(run, attempt, 0)]);
    let at = rows.schema().index_of("grad_norm").unwrap();
    let mut columns = rows.columns().to_vec();
    columns[at] = Arc::new(Float64Array::from(vec![grad_norm]));
    let rows = RecordBatch::try_new(rows.schema(), columns).unwrap();
    storage.write_part(run, 0, attempt, &[rows], 0, 0).await.unwrap();
}

/// The table both engines must agree on (`evidence._attempt_place`: an attempt the file does not
/// list goes just before the newest, by number among such, whatever placed the newest; with no
/// entry at all, by number). The same table is asserted against Python in
/// `sdk/tests/test_wave_ao_ingest.py`.
fn bug08_table() -> Vec<(&'static str, Vec<(i64, &'static str)>, Vec<i64>, Vec<i64>)> {
    vec![
        // (case, the file's entries, the attempts stored but not listed, the order read)
        (
            "claimed_newest_unlisted_above",
            vec![(200, "claim")],
            vec![300],
            vec![300, 200],
        ),
        (
            "claimed_newest_unlisted_below",
            vec![(200, "claim")],
            vec![100],
            vec![100, 200],
        ),
        (
            "rows_newest_unlisted_above",
            vec![(100, "claim"), (200, "rows")],
            vec![300],
            vec![100, 300, 200],
        ),
        (
            "store_newest_unlisted_above",
            vec![(100, "store"), (200, "store")],
            vec![300],
            vec![100, 300, 200],
        ),
        (
            "one_entry_by_rows_unlisted_below",
            vec![(200, "rows")],
            vec![50],
            vec![50, 200],
        ),
        (
            "two_unlisted_either_side",
            vec![(100, "claim"), (200, "claim")],
            vec![300, 50],
            vec![100, 50, 300, 200],
        ),
        ("an_empty_list", vec![], vec![5, 3], vec![3, 5]),
    ]
}

/// Fails on `453f1e2`: an unlisted attempt numbered above the newest was put after it (the live
/// rule for rows of a sender that names no attempt, which `order_of` reused), where Python puts it
/// before.
#[tokio::test]
async fn bug08_an_unlisted_attempt_goes_just_before_the_newest_as_python_reads_it() {
    for (case, entries, unlisted, expected) in bug08_table() {
        let run = format!("ao3_bug08_{case}");
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let storage = storage(root);
        for (attempt, _) in &entries {
            put_part(&storage, &run, *attempt, 1.0).await;
        }
        for attempt in &unlisted {
            put_part(&storage, &run, *attempt, 1.0).await;
        }
        write_attempts(root, &run, &entries);
        let order = storage.attempt_order(&run, 0).await.unwrap();
        assert_eq!(order.attempts(), expected, "{case}");
        let parts = storage.rank_parts(&run, 0).await.unwrap();
        let attempts: Vec<i64> = parts.iter().map(|part| part.attempt).collect();
        assert_eq!(attempts, expected, "{case}: the parts in that order");
    }
}

/// The audit's case through the detectors: claimed attempt 200 has a finite `grad_norm` at step 0,
/// and an attempt the file does not list, 300 (a higher number: a clock ahead), a NaN at step 0.
/// Python places 300 before 200, so 200's step 0 rolls it back and `non_finite` is not raised; the
/// Rust replay must read the same. Fails on `453f1e2`, where it fires `non_finite`.
#[tokio::test]
async fn bug08_the_replay_rolls_back_an_unlisted_attempt_as_python_does() {
    let run = "ao3_bug08_replay";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let storage = storage(root);
    put_part(&storage, run, 200, 1.0).await;
    put_part(&storage, run, 300, f64::NAN).await;
    write_attempts(root, run, &[(200, "claim")]);
    let detectors = replay(&storage, run, 0, &Arc::new(Thresholds::default()))
        .await
        .unwrap();
    let kinds: Vec<String> = detectors
        .incidents()
        .iter()
        .map(|incident| incident["kind"].as_str().unwrap_or_default().to_string())
        .collect();
    assert!(!kinds.iter().any(|kind| kind == "non_finite"), "{kinds:?}");
}

/// A folder holding only `scalars-*` objects is one of the rank's attempts: with no readable
/// `attempts.json` the order is numeric and lists it, and a later claim goes after it. Fails on
/// `453f1e2`, which listed only attempts with parts: the order became [100, 200, 400], and Python
/// then put 300 just before 400, after 200, which rolls back no point of 300 that 200 should.
#[tokio::test]
async fn bug08_an_attempt_with_only_scalars_objects_is_in_the_rank_order() {
    let run = "ao3_bug08_scalars_only";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let store = storage(root);
    put_part(&store, run, 100, 1.0).await;
    store
        .write_scalars(run, 0, 300, &[batch(&[scalar_row(run, 300, 50, 0, 99.0)])], 0, 0)
        .await
        .unwrap();
    // No order the server can read (a file that is not one is taken as none).
    std::fs::write(rank_dir(root, run).join("attempts.json"), "not an order").unwrap();
    assert_eq!(store.attempt_order(run, 0).await.unwrap().attempts(), [100, 300]);
    // A pin: the parts and the step watermark are the parts' alone (D8).
    let parts = store.rank_parts(run, 0).await.unwrap();
    assert_eq!(parts.iter().map(|part| part.attempt).collect::<Vec<_>>(), [100]);
    // Through the server: two claims after the store was read.
    let server = start(root).await;
    send(&server, run, Some(200), vec![batch(&[step_row(run, 200, 0)])]).await;
    send(&server, run, Some(400), vec![batch(&[step_row(run, 400, 1)])]).await;
    assert_eq!(attempts_json(root, run), [100, 300, 200, 400]);
    server.ingest.shutdown().await.unwrap();
}

// --- BUG-02, the server side: a version 2 row with a null `seq` -------------------------------------

/// Training steps 0 to 3, then the version 2 row without `seq` at step 4, then steps 5 and 6, in one
/// batch under metadata 2. The row is dropped and counted under `seq`, with the stderr line other
/// dropped rows get, and the rest of the batch is stored. Fails on `453f1e2`: it is stored as a
/// training step, in a part that every Python reader refuses ("schema version 2 is the scalar-only
/// record").
#[tokio::test]
async fn bug02_a_version_2_row_with_a_null_seq_is_dropped_and_counted_and_the_rest_stored() {
    let run = "ao3_bug02_mixed";
    let attempt = 7;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    let mut rows: Vec<Value> = (0..4).map(|step| step_row(run, attempt, step)).collect();
    rows.push(v2_row_without_seq(run, attempt, 4));
    rows.extend((5..7).map(|step| step_row(run, attempt, step)));
    let acked = send(&server, run, Some(attempt), vec![batch(&rows)]).await;
    assert_eq!(acked, [7], "acknowledged to the end");
    assert_eq!(part_steps(root, run, attempt), [0, 1, 2, 3, 5, 6]);
    assert!(files(root, run, attempt, "scalars-").is_empty());
    // No stored row is a version 2 record: every Python reader can read the part.
    assert!(!column_of(&files(root, run, attempt, "part-"), "seq").contains(&0));
    let entry = rank_entry(&server.ingest, run);
    assert_eq!(entry["dropped_rows"], 1, "{entry}");
    assert_eq!(entry["dropped_row_keys"], json!({"seq": 1}), "{entry}");
    assert_eq!(entry["last_step"], 6, "{entry}");
    server.ingest.shutdown().await.unwrap();
}

/// The row alone in its batch: acknowledged, counted, nothing stored.
#[tokio::test]
async fn bug02_a_batch_of_only_such_a_row_is_acknowledged_counted_and_stores_nothing() {
    let run = "ao3_bug02_alone";
    let attempt = 7;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    let acked = send(
        &server,
        run,
        Some(attempt),
        vec![batch(&[v2_row_without_seq(run, attempt, 4)])],
    )
    .await;
    assert_eq!(acked, [1]);
    assert!(files(root, run, attempt, "part-").is_empty());
    assert!(files(root, run, attempt, "scalars-").is_empty());
    let entry = rank_entry(&server.ingest, run);
    assert_eq!(entry["dropped_row_keys"], json!({"seq": 1}), "{entry}");
    // Steps after it are stored: it raised no watermark.
    let acked = send(&server, run, Some(attempt), vec![batch(&[step_row(run, attempt, 0)])]).await;
    assert_eq!(acked, [1]);
    assert_eq!(part_steps(root, run, attempt), [0]);
    server.ingest.shutdown().await.unwrap();
}

/// A batch with no `seq` column at all under metadata 2: a row whose record is version 2 has no
/// `seq` there either, and is dropped the same way.
#[tokio::test]
async fn bug02_with_no_seq_column_a_version_2_record_row_is_dropped_too() {
    let run = "ao3_bug02_noseq";
    let attempt = 7;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    let schema = schema_without(&["seq"]);
    let rows = vec![
        step_row(run, attempt, 0),
        v2_row_without_seq(run, attempt, 1),
        step_row(run, attempt, 2),
    ];
    let acked = send(&server, run, Some(attempt), vec![decode(schema, &rows)]).await;
    assert_eq!(acked, [3]);
    assert_eq!(part_steps(root, run, attempt), [0, 2]);
    assert_eq!(rank_entry(&server.ingest, run)["dropped_row_keys"], json!({"seq": 1}));
    server.ingest.shutdown().await.unwrap();
}

/// Pins: training rows of `record_version` 0, 1 and null, and a whole scalar-only row, are stored as
/// before; a batch with no `record_version` column (a layout without it) is stored as sent.
#[tokio::test]
async fn bug02_pins_training_rows_and_whole_scalar_rows_are_stored_as_before() {
    let run = "ao3_bug02_pins";
    let attempt = 7;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    let mut rows: Vec<Value> = (0..3).map(|step| step_row(run, attempt, step)).collect();
    rows[0]["record_version"] = json!(0);
    rows[2]["record_version"] = Value::Null;
    rows.insert(1, scalar_row(run, attempt, 500, 0, 1.5));
    let acked = send(&server, run, Some(attempt), vec![batch(&rows)]).await;
    assert_eq!(acked, [4]);
    assert_eq!(part_steps(root, run, attempt), [0, 1, 2]);
    assert_eq!(scalar_steps(root, run, attempt), [500]);
    let entry = rank_entry(&server.ingest, run);
    assert!(entry.get("dropped_rows").is_none(), "{entry}");
    // No `record_version` column: nothing says a row is a version 2 record.
    let schema = schema_without(&["record_version"]);
    let mut put = open(&server.address, run, schema.clone(), Some(attempt)).await;
    put.send(decode(schema, &[step_row(run, attempt, 3)]), 5).await;
    assert_eq!(put.reply().await.unwrap(), 5);
    put.finish().await;
    assert_eq!(part_steps(root, run, attempt), [0, 1, 2, 3]);
    assert!(rank_entry(&server.ingest, run).get("dropped_rows").is_none());
    server.ingest.shutdown().await.unwrap();
}
