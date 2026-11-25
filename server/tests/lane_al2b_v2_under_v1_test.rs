//! Lane AL-2′, the owner's fix of Nov 25, 2026: under batch metadata `schema_version` 1 the server
//! reads row versions 0 and 1 only.
//!
//! The finding (lane AL-0's run across a real protocol 4 build): a protocol 4 sender given a `.jsonl`
//! that a protocol 5 capture wrote sends its scalar-only record as a step row, of row version 2, with
//! no `seq` column, under batch metadata 1. The server stored it as a training step: with steps 0 to
//! 9, the record at step 500, then steps 10 to 29, it kept 0 to 9 and 500, dropped 10 to 29 as not
//! newer, said nothing and acknowledged everything. The owner's words: "Drop and count. In a batch
//! whose metadata schema_version is 1, the server reads row versions 0 and 1 only; a version 2 row
//! there is dropped and counted as unsupported_schema_version, with the stderr line other dropped
//! rows get, and the rest of the batch is stored."
//!
//! The batches here are what a protocol 4 sender sends: protocol 4's row layout (the JSON columns, no
//! `seq`, no `record_version`), batch metadata 1. The stderr line is shown by the Python test through
//! a real server (`sdk/tests/test_lane_al2b_v2_under_v1.py`); here the count is read from the
//! server's own counter and the rank's entry.

mod lane_aa3_support;

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Array, Int64Array};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::record_batch::RecordBatch;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::{FlightClient, FlightData, FlightDescriptor, PutResult};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use lane_aa3_support::{json_file, start, TestServer};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{json, Value};
use tensorquill_server::ingest::Ingest;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Channel;

const ATTEMPT: i64 = 1_732_460_000_000;

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

/// Protocol 4's row layout (`experiments/protocol-v5/flight_v4.py`), the columns these tests need:
/// the keys not nullable, `scalars` as a map, the statistics as JSON text, no `seq`.
fn schema_v4() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("schema_version", DataType::Int32, false),
        Field::new("run_id", DataType::Utf8, false),
        Field::new("rank", DataType::Int32, false),
        Field::new("attempt", DataType::Int64, false),
        Field::new("step", DataType::Int64, false),
        Field::new("ts", DataType::Float64, true),
        Field::new("grad_norm", DataType::Float64, true),
        Field::new("scalars", scalars_map(), true),
        Field::new("layers_json", DataType::Utf8, true),
        Field::new("moe_stats_json", DataType::Utf8, true),
        Field::new("probe_json", DataType::Utf8, true),
        Field::new("extra_json", DataType::Utf8, true),
    ]))
}

/// Protocol 4's layout with a `seq` column added (case (c)).
fn schema_v4_with_seq() -> Arc<Schema> {
    let mut fields: Vec<Field> = schema_v4()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect();
    fields.push(Field::new("seq", DataType::Int64, true));
    Arc::new(Schema::new(fields))
}

/// Protocol 5's row layout for the pins of case (d): row version 2, `record_version`, `seq`.
fn schema_v5() -> Arc<Schema> {
    let mut fields: Vec<Field> = schema_v4()
        .fields()
        .iter()
        .filter(|field| !field.name().ends_with("_json"))
        .map(|field| field.as_ref().clone().with_nullable(true))
        .collect();
    fields.push(Field::new("seq", DataType::Int64, true));
    fields.push(Field::new("record_version", DataType::Int32, true));
    fields.push(Field::new("extra_json", DataType::Utf8, true));
    Arc::new(Schema::new(fields))
}

/// A training row of `run` at `step`, row version `version`.
fn step_row(run: &str, step: i64, version: i32) -> Value {
    json!({"schema_version": version, "run_id": run, "rank": 0, "attempt": ATTEMPT, "step": step,
           "ts": step as f64, "grad_norm": 1.0})
}

/// The row a protocol 4 sender makes of a protocol 5 capture's scalar-only record at `step`: its
/// version, 2, as the row version, and its `scalars`; `seq` and `record` have no column there.
fn scalar_record_row(run: &str, step: i64) -> Value {
    json!({"schema_version": 2, "run_id": run, "rank": 0, "attempt": ATTEMPT, "step": step,
           "ts": 500.0, "scalars": {"eval/loss": 1.4}})
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

fn v4(rows: &[Value]) -> RecordBatch {
    decode(schema_v4(), rows)
}

fn steps(run: &str, range: std::ops::Range<i64>) -> Vec<Value> {
    range.map(|step| step_row(run, step, 1)).collect()
}

/// The owner's scenario: training steps 0 to 9, the version 2 row at step 500, steps 10 to 29.
fn scenario(run: &str) -> Vec<Value> {
    let mut rows = steps(run, 0..10);
    rows.push(scalar_record_row(run, 500));
    rows.extend(steps(run, 10..30));
    rows
}

// --- one DoPut stream ------------------------------------------------------------------------------

struct Put {
    run: String,
    messages: mpsc::UnboundedSender<Result<FlightData, FlightError>>,
    acks: BoxStream<'static, Result<PutResult, FlightError>>,
}

async fn open(address: &str, run: &str, schema: Arc<Schema>) -> Put {
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
        messages,
        acks,
    }
}

impl Put {
    async fn send(&mut self, batch: RecordBatch, metadata_version: i64, end_byte_offset: i64) {
        let metadata = format!(
            "{{\"schema_version\":{metadata_version},\"run_id\":\"{}\",\"rank\":0,\"min_step\":null,\"max_step\":null,\"end_byte_offset\":{end_byte_offset}}}",
            self.run
        );
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

    async fn ack(&mut self) -> i64 {
        match tokio::time::timeout(Duration::from_secs(20), self.acks.next())
            .await
            .expect("no reply")
        {
            Some(Ok(result)) => {
                let reply: Value = serde_json::from_slice(&result.app_metadata).unwrap();
                reply["end_byte_offset"].as_i64().unwrap()
            }
            other => panic!("not an acknowledgement: {other:?}"),
        }
    }

    async fn finish(self) {
        drop(self.messages);
        let mut acks = self.acks;
        while acks.next().await.is_some() {}
    }
}

/// Sends `batches` on one stream with metadata `version`, each batch's end offset its place in the
/// file (one line a row), and returns the offsets acknowledged.
async fn send(server: &TestServer, run: &str, batches: Vec<RecordBatch>, version: i64) -> Vec<i64> {
    let mut put = open(&server.address, run, batches[0].schema()).await;
    let mut end = 0;
    let mut acked = Vec::new();
    for batch in batches {
        end += batch.num_rows() as i64;
        put.send(batch, version, end).await;
        acked.push(put.ack().await);
    }
    put.finish().await;
    acked
}

// --- what the store and the server hold --------------------------------------------------------------

fn attempt_dir(root: &Path, run: &str) -> PathBuf {
    root.join("runs")
        .join(run)
        .join("rank=0")
        .join(format!("attempt={ATTEMPT}"))
}

fn files(root: &Path, run: &str, prefix: &str) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(attempt_dir(root, run))
        .map(|entries| entries.map(|entry| entry.unwrap().path()).collect())
        .unwrap_or_default();
    found.retain(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(prefix) && name.ends_with(".parquet"))
    });
    found.sort();
    found
}

fn read_file(path: &Path) -> Vec<RecordBatch> {
    ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(path).unwrap())
        .unwrap()
        .build()
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

fn stored_steps(root: &Path, run: &str) -> Vec<i64> {
    let mut out = Vec::new();
    for part in files(root, run, "part-") {
        for batch in read_file(&part) {
            let column = batch.column_by_name("step").unwrap();
            out.extend(column.as_any().downcast_ref::<Int64Array>().unwrap().values().iter());
        }
    }
    out
}

fn rank_entry(ingest: &Ingest, run: &str) -> Value {
    ingest.watchdog_statuses()[run]["ranks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|rank| rank["rank"] == 0)
        .cloned()
        .unwrap()
}

async fn report_through(root: &Path, run: &str, last_step: i64) -> Value {
    let path = root.join("runs").join(run).join("rank=0").join("incidents.json");
    for _ in 0..400 {
        if path.exists() {
            let report = json_file(&path);
            if report["last_step"] == last_step {
                return report;
            }
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!(
        "no report through step {last_step}: {:?}",
        std::fs::read_to_string(&path).ok()
    );
}

/// What the owner's scenario must leave, however its rows were split into batches: every training
/// step stored and none at 500, `last_step` 29, one row counted under `unsupported_schema_version`,
/// and the detectors judging steps 0 to 29.
async fn assert_the_scenario_kept(server: &TestServer, root: &Path, run: &str) {
    assert_eq!(
        stored_steps(root, run),
        (0..30).collect::<Vec<_>>(),
        "every training step, none at 500"
    );
    let entry = rank_entry(&server.ingest, run);
    assert_eq!(entry["last_step"], 29, "{entry}");
    assert_eq!(entry["dropped_rows"], 1, "{entry}");
    assert_eq!(
        entry["dropped_row_keys"],
        json!({"unsupported_schema_version": 1}),
        "{entry}"
    );
    assert!(entry.get("rows_not_newer").is_none(), "{entry}");
    assert_eq!(
        server.ingest.dropped_rows().get("unsupported_schema_version"),
        Some(&1),
        "{:?}",
        server.ingest.dropped_rows()
    );
    let report = report_through(root, run, 29).await;
    assert_eq!(report["steps"], 30, "the detectors judge steps 0 to 29: {report}");
}

// --- (a) the owner's scenario, one batch ----------------------------------------------------------------

/// (a) The owner's scenario exactly, as one batch a protocol 4 sender sends: the version 2 row is
/// dropped and counted, the rest of the batch is stored, and the sender's offset is acknowledged to
/// the end. Fails on `4e31e2b`: the row is stored at step 500 and steps 10 to 29 are dropped.
#[tokio::test]
async fn a_version_2_row_under_metadata_1_is_dropped_and_counted_and_every_training_step_is_stored() {
    let run = "al2b_v2v1_one";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    let acked = send(&server, run, vec![v4(&scenario(run))], 1).await;
    assert_eq!(acked, [31], "acknowledged to the end of the file");
    assert_the_scenario_kept(&server, root, run).await;
    server.ingest.shutdown().await.unwrap();
}

// --- (b) the same rows over several batches -----------------------------------------------------------

/// (b) The version 2 row alone in its batch: that batch is acknowledged with nothing stored of it.
#[tokio::test]
async fn a_version_2_row_alone_in_its_batch_under_metadata_1_is_dropped_and_counted() {
    let run = "al2b_v2v1_alone";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    let rows = scenario(run);
    let batches = vec![v4(&rows[..10]), v4(&rows[10..11]), v4(&rows[11..])];
    assert_eq!(send(&server, run, batches, 1).await, [10, 11, 31]);
    assert_the_scenario_kept(&server, root, run).await;
    server.ingest.shutdown().await.unwrap();
}

/// (b) The version 2 row first in its batch, before steps 10 to 29.
#[tokio::test]
async fn a_version_2_row_first_in_its_batch_under_metadata_1_is_dropped_and_counted() {
    let run = "al2b_v2v1_first";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    let rows = scenario(run);
    let batches = vec![v4(&rows[..10]), v4(&rows[10..])];
    assert_eq!(send(&server, run, batches, 1).await, [10, 31]);
    assert_the_scenario_kept(&server, root, run).await;
    server.ingest.shutdown().await.unwrap();
}

/// (b) The version 2 row last in its batch, after steps 0 to 9.
#[tokio::test]
async fn a_version_2_row_last_in_its_batch_under_metadata_1_is_dropped_and_counted() {
    let run = "al2b_v2v1_last";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    let rows = scenario(run);
    let batches = vec![v4(&rows[..11]), v4(&rows[11..])];
    assert_eq!(send(&server, run, batches, 1).await, [11, 31]);
    assert_the_scenario_kept(&server, root, run).await;
    server.ingest.shutdown().await.unwrap();
}

// --- (c) a version 2 row with a seq column under metadata 1 ---------------------------------------------

/// (c) A batch of metadata 1 that has a `seq` column, the version 2 row's `seq` set: the row is
/// dropped and counted as in (a), before the scalar rows are taken out, so no `scalars-*` object is
/// made. Fails on `4e31e2b`: the row is stored as a scalars object.
#[tokio::test]
async fn a_version_2_row_with_seq_under_metadata_1_is_dropped_and_never_a_scalars_object() {
    let run = "al2b_v2v1_seq";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    let rows: Vec<Value> = scenario(run)
        .into_iter()
        .map(|mut row| {
            row["seq"] = if row["schema_version"] == 2 {
                json!(0)
            } else {
                Value::Null
            };
            row
        })
        .collect();
    let acked = send(&server, run, vec![decode(schema_v4_with_seq(), &rows)], 1).await;
    assert_eq!(acked, [31]);
    assert!(
        files(root, run, "scalars-").is_empty(),
        "no scalars object under metadata 1"
    );
    assert_the_scenario_kept(&server, root, run).await;
    server.ingest.shutdown().await.unwrap();
}

// --- (d) unchanged -------------------------------------------------------------------------------------

/// (d) Under metadata 2, a version 2 step row is stored, and a scalar-only row (its `seq` set) is
/// stored apart, as on `4e31e2b`. A pin.
#[tokio::test]
async fn under_metadata_2_a_version_2_step_row_is_stored_and_a_scalar_row_is_stored_apart() {
    let run = "al2b_v2v2";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    let mut rows: Vec<Value> = (0..3)
        .map(|step| {
            let mut row = step_row(run, step, 2);
            row["record_version"] = json!(1);
            row
        })
        .collect();
    let mut scalar = scalar_record_row(run, 500);
    scalar["seq"] = json!(0);
    scalar["record_version"] = json!(2);
    rows.insert(2, scalar);
    let acked = send(&server, run, vec![decode(schema_v5(), &rows)], 2).await;
    assert_eq!(acked, [4]);
    assert_eq!(stored_steps(root, run), [0, 1, 2]);
    let scalars = files(root, run, "scalars-");
    assert_eq!(scalars.len(), 1, "{scalars:?}");
    let entry = rank_entry(&server.ingest, run);
    assert_eq!(entry["last_step"], 2, "{entry}");
    assert!(entry.get("dropped_rows").is_none(), "{entry}");
    server.ingest.shutdown().await.unwrap();
}

/// (d) Under metadata 1, rows of versions 0 and 1 are stored exactly as received. A pin.
#[tokio::test]
async fn under_metadata_1_rows_of_versions_0_and_1_are_stored_as_before() {
    let run = "al2b_v1_rows";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    let rows = vec![
        step_row(run, 0, 0),
        step_row(run, 1, 1),
        step_row(run, 2, 0),
        step_row(run, 3, 1),
    ];
    let sent = v4(&rows);
    assert_eq!(send(&server, run, vec![sent.clone()], 1).await, [4]);
    let parts = files(root, run, "part-");
    assert_eq!(parts.len(), 1);
    assert_eq!(read_file(&parts[0]), [sent], "stored as received");
    let entry = rank_entry(&server.ingest, run);
    assert!(entry.get("dropped_rows").is_none(), "{entry}");
    server.ingest.shutdown().await.unwrap();
}

/// (d) A row of version 99 is dropped and counted under `unsupported_schema_version`, under either
/// metadata version, and the rest of its batch is stored. A pin.
#[tokio::test]
async fn a_row_of_version_99_is_dropped_and_counted_under_either_metadata_version() {
    for (metadata, layout) in [(1, schema_v4()), (2, schema_v5())] {
        let run = format!("al2b_v99_meta{metadata}");
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let server = start(root).await;
        let row_version = if metadata == 1 { 1 } else { 2 };
        let rows = vec![
            step_row(&run, 0, row_version),
            step_row(&run, 1, 99),
            step_row(&run, 2, row_version),
        ];
        assert_eq!(send(&server, &run, vec![decode(layout, &rows)], metadata).await, [3]);
        assert_eq!(stored_steps(root, &run), [0, 2], "metadata {metadata}");
        let entry = rank_entry(&server.ingest, &run);
        assert_eq!(
            entry["dropped_row_keys"],
            json!({"unsupported_schema_version": 1}),
            "metadata {metadata}: {entry}"
        );
        server.ingest.shutdown().await.unwrap();
    }
}

// --- (e) a resend from byte 0 after a restart ------------------------------------------------------------

/// (e) After a restart, the sender sends the same file again from byte 0 (a replaced store's ID,
/// rewritten bytes). Nothing is stored twice: the 30 training rows are dropped as not newer (the
/// watermark is read from the parts) and counted in `rows_not_newer`; the version 2 row is dropped
/// again and counted again under `unsupported_schema_version`, as any dropped row sent again is
/// (RS10: a count of rows received, a resend counted again). The restarted server counts from 0, so
/// its own count is 1; its status entry starts from the status written before only when it takes
/// the run up from `watchdog.json`, which this server in a test does not.
#[tokio::test]
async fn after_a_restart_a_resend_from_byte_0_stores_nothing_twice_and_counts_the_row_again() {
    let run = "al2b_v2v1_restart";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    {
        let server = start(root).await;
        assert_eq!(send(&server, run, vec![v4(&scenario(run))], 1).await, [31]);
        assert_the_scenario_kept(&server, root, run).await;
        server.ingest.shutdown().await.unwrap();
    }
    let parts_before = files(root, run, "part-");
    let server = start(root).await;
    assert_eq!(send(&server, run, vec![v4(&scenario(run))], 1).await, [31]);
    assert_eq!(files(root, run, "part-"), parts_before, "no new part");
    assert_eq!(stored_steps(root, run), (0..30).collect::<Vec<_>>());
    let entry = rank_entry(&server.ingest, run);
    assert_eq!(entry["rows_not_newer"], 30, "{entry}");
    assert_eq!(
        entry["dropped_row_keys"],
        json!({"unsupported_schema_version": 1}),
        "{entry}"
    );
    assert_eq!(server.ingest.dropped_rows().get("unsupported_schema_version"), Some(&1));
    server.ingest.shutdown().await.unwrap();
}
