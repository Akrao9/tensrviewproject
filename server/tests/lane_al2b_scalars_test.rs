//! Lane AL-2′ of wave AL (`WAVE_AL_CONTRACT.md` section 4.5): the server stores scalar-only rows.
//!
//! - A row whose `seq` is not null is taken out of its batch before the step watermark: it is never
//!   in a `part-*` file and never progress (not `last_step`, not the step watermark; decision D8).
//! - It is deduplicated by `(run_id, rank, attempt, seq)` against the attempt's `seq` watermark; a
//!   row at or below it is dropped and counted as `scalars_not_newer` on the rank's entry.
//! - It is stored as `attempt=<a>/scalars-<first_seq:08>-<last_seq:08>.parquet`, created only if no
//!   such object exists; the watermark across a restart is read from those names (a name that does
//!   not parse is read by its footer).
//! - An offset is acknowledged only when every row before it is stored, of both kinds: the step part,
//!   then the scalars object, then the acknowledgement. A create that meets an existing object is
//!   read for its `seq` range: the same range is stored, another keeps the rows and acknowledges
//!   nothing.
//! - The detectors never read a scalar row, live or in a replay.
//! - A batch without a `seq` column (step 1's sender, a protocol 4 sender) is handled as before.
//!
//! The rows are the cases' records (`sdk/tests/fixtures/wave_al/cases.json`) laid out as section 4.4
//! says: protocol 5's row version 2, the record's own version in `record_version`, `scalars` in its
//! map column, and a nullable Int64 `seq`, null on a step row. Lane AL-1′ adds `seq` to the Python
//! sender in parallel; nothing here uses it.

mod lane_aa3_support;

use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow::array::{Array, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::record_batch::RecordBatch;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::{FlightClient, FlightData, FlightDescriptor, PutResult};
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt, TryStreamExt};
use lane_aa3_support::{json_file, start};
use object_store::local::LocalFileSystem;
use object_store::path::Path as StorePath;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore, PutMultipartOptions,
    PutOptions, PutPayload,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use serde_json::{json, Value};
use tensorquill_server::ingest::{Ingest, IngestConfig};
use tensorquill_server::server::serve;
use tensorquill_server::storage::Storage;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Channel;

// --- the cases' records as rows -------------------------------------------------------------------

fn cases() -> Vec<Value> {
    let text =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../sdk/tests/fixtures/wave_al/cases.json"))
            .unwrap();
    // One case's layer name is the JSON escape of a lone surrogate (`layers_name_arrow_cannot_encode`),
    // which serde_json refuses; none of the cases read here holds one.
    // An escaped backslash before `ud800` (the same name inside a JSON text) is valid and kept.
    let text = text
        .replace("\\\\ud800", "\u{0}")
        .replace("\\ud800", "?")
        .replace('\u{0}', "\\\\ud800");
    let document: Value = serde_json::from_str(&text).unwrap();
    document["cases"].as_array().unwrap().clone()
}

fn case(name: &str) -> Value {
    cases()
        .into_iter()
        .find(|case| case["name"] == name)
        .unwrap_or_else(|| panic!("no case {name}"))
}

fn records(case: &Value) -> Vec<Value> {
    case["records"].as_array().unwrap().clone()
}

/// Protocol 5's row for the columns these tests need (sections 3 and 4.4): row version 2, the
/// record's own version in `record_version`, `scalars` as a map, and `seq`, null on a step row.
fn schema() -> Arc<Schema> {
    let scalars = DataType::Map(
        Arc::new(Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Float64, true),
            ])),
            false,
        )),
        false,
    );
    Arc::new(Schema::new(vec![
        Field::new("schema_version", DataType::Int32, true),
        Field::new("run_id", DataType::Utf8, true),
        Field::new("rank", DataType::Int32, true),
        Field::new("attempt", DataType::Int64, true),
        Field::new("step", DataType::Int64, true),
        Field::new("ts", DataType::Float64, true),
        Field::new("grad_norm", DataType::Float64, true),
        Field::new("scalars", scalars, true),
        Field::new("seq", DataType::Int64, true),
        Field::new("record_version", DataType::Int32, true),
        Field::new("extra_json", DataType::Utf8, true),
    ]))
}

/// A record of the cases as its row (section 4.4).
fn row(record: &Value) -> Value {
    json!({
        "schema_version": 2,
        "run_id": record["run_id"],
        "rank": record["rank"],
        "attempt": record["attempt"],
        "step": record["step"],
        "ts": record["ts"],
        "grad_norm": record.get("grad_norm").cloned().unwrap_or(Value::Null),
        "scalars": record.get("scalars").cloned().unwrap_or(Value::Null),
        "seq": record.get("seq").cloned().unwrap_or(Value::Null),
        "record_version": record["schema_version"],
        "extra_json": null,
    })
}

fn decode(schema: Arc<Schema>, rows: &[Value]) -> RecordBatch {
    let text: String = rows.iter().map(|row| format!("{row}\n")).collect();
    let mut reader = arrow::json::ReaderBuilder::new(schema.clone())
        .with_batch_size(rows.len().max(1))
        .build(Cursor::new(text))
        .unwrap();
    let batch = reader.next().unwrap().unwrap();
    assert!(reader.next().is_none());
    batch
}

/// The records' rows as one batch.
fn batch(records: &[Value]) -> RecordBatch {
    decode(schema(), &records.iter().map(row).collect::<Vec<_>>())
}

/// A step record of `run` at `step` (attempt 1, rank 0).
fn step_record(run: &str, step: i64) -> Value {
    json!({"schema_version": 1, "run_id": run, "rank": 0, "attempt": 1, "step": step, "ts": step as f64, "grad_norm": 1.0})
}

/// A scalar-only record of `run` at `step` with `seq` (section 4.2).
fn scalar_record(run: &str, step: i64, seq: i64) -> Value {
    json!({"schema_version": 2, "record": "scalars", "run_id": run, "rank": 0, "attempt": 1, "step": step,
           "seq": seq, "ts": 100.0 + seq as f64, "scalars": {"eval/loss": 1.0 + seq as f64}})
}

// --- one DoPut stream with batch metadata 2 --------------------------------------------------------

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

/// Sends `batch` on a stream of its own, metadata version 2, and returns its acknowledgement or the
/// status that ended the stream.
async fn send_one(address: &str, run: &str, batch: RecordBatch, end: i64) -> Result<i64, tonic::Status> {
    send_versioned(address, run, batch, 2, end).await
}

async fn send_versioned(
    address: &str,
    run: &str,
    batch: RecordBatch,
    version: i64,
    end: i64,
) -> Result<i64, tonic::Status> {
    let mut put = open(address, run, batch.schema()).await;
    put.send(batch, version, end).await;
    let reply = put.reply().await;
    if reply.is_ok() {
        put.finish().await;
    }
    reply
}

// --- what the store holds --------------------------------------------------------------------------

fn attempt_dir(root: &Path, run: &str) -> PathBuf {
    root.join("runs").join(run).join("rank=0").join("attempt=1")
}

/// The names of the attempt's files that start with `prefix` and end with `.parquet`, sorted.
fn files(root: &Path, run: &str, prefix: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(attempt_dir(root, run))
        .map(|entries| {
            entries
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    names.retain(|name| name.starts_with(prefix) && name.ends_with(".parquet"));
    names.sort();
    names
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

/// Every value of the Int64 column `name` in the files named, in file and row order; a null is -1.
fn column_of(root: &Path, run: &str, names: &[String], column: &str) -> Vec<i64> {
    let mut out = Vec::new();
    for name in names {
        for batch in read_file(&attempt_dir(root, run).join(name)) {
            let values = batch
                .column_by_name(column)
                .unwrap_or_else(|| panic!("{name}: no {column}"));
            let values = values.as_any().downcast_ref::<Int64Array>().unwrap();
            out.extend(values.iter().map(|value| value.unwrap_or(-1)));
        }
    }
    out
}

/// The steps of the attempt's `part-*` files, in name and row order.
fn stored_steps(root: &Path, run: &str) -> Vec<i64> {
    column_of(root, run, &files(root, run, "part-"), "step")
}

/// The `seq` of every row of the attempt's `scalars-*` files, in name and row order.
fn stored_seqs(root: &Path, run: &str) -> Vec<i64> {
    column_of(root, run, &files(root, run, "scalars-"), "seq")
}

/// The `seq` of every row of the attempt's `part-*` files that has one: none, by D8.
fn seqs_in_parts(root: &Path, run: &str) -> Vec<i64> {
    let names = files(root, run, "part-");
    let mut out = Vec::new();
    for name in &names {
        for batch in read_file(&attempt_dir(root, run).join(name)) {
            if let Some(values) = batch.column_by_name("seq") {
                let values = values.as_any().downcast_ref::<Int64Array>().unwrap();
                out.extend(values.iter().flatten());
            }
        }
    }
    out
}

/// The rank's entry in the run's status as the server holds it now.
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

/// The rank's `incidents.json` once it covers `last_step`.
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

fn storage(root: &Path) -> Storage {
    Storage::new(Arc::new(LocalFileSystem::new_with_prefix(root).unwrap()), "".into())
}

/// Writes `batch` as a Parquet file at `path`, as some other writer might have.
fn write_parquet(path: &Path, batch: &RecordBatch) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, batch.schema(), None).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    std::fs::write(path, bytes).unwrap();
}

// --- the cases ------------------------------------------------------------------------------------

/// Case `scalar_future_step_keeps_training_rows` (D8): a scalar-only record for step 1000 logged at
/// step 3, then steps 4 to 9 across a server restart. Every training row is stored, the scalar row
/// is in a `scalars-*` object of its own and in no part, and the rank's `last_step` is a training
/// step. Fails on the base: the row of step 1000 is stored in a part and raises the attempt's step
/// watermark, so steps 4 to 9 are dropped as not newer.
#[tokio::test]
async fn scalar_future_step_keeps_training_rows() {
    let case = case("scalar_future_step_keeps_training_rows");
    let records = records(&case);
    let run = records[0]["run_id"].as_str().unwrap().to_string();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    {
        let server = start(root).await;
        assert_eq!(
            send_one(&server.address, &run, batch(&records[..5]), 5).await.unwrap(),
            5
        );
        assert_eq!(
            send_one(&server.address, &run, batch(&records[5..8]), 8).await.unwrap(),
            8
        );
        let entry = rank_entry(&server.ingest, &run);
        assert_eq!(entry["last_step"], 6, "a scalar row is never progress: {entry}");
        assert!(entry.get("rows_not_newer").is_none(), "{entry}");
        server.ingest.shutdown().await.unwrap();
    }
    // The restart: the sender sends from its offset again, steps 4 to 6 once more and 7 to 9.
    let server = start(root).await;
    assert_eq!(
        send_one(&server.address, &run, batch(&records[5..]), 11).await.unwrap(),
        11
    );
    let expect = &case["expect"];
    let steps: Vec<i64> = expect["stored_training_steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|step| step.as_i64().unwrap())
        .collect();
    assert_eq!(stored_steps(root, &run), steps);
    assert_eq!(stored_seqs(root, &run), [0]);
    assert!(seqs_in_parts(root, &run).is_empty());
    assert_eq!(files(root, &run, "scalars-"), ["scalars-00000000-00000000.parquet"]);
    let entry = rank_entry(&server.ingest, &run);
    assert_eq!(entry["last_step"], 9, "{entry}");
    assert_eq!(entry["rows_not_newer"], 3, "steps 4 to 6 sent again: {entry}");
    assert!(entry.get("scalars_not_newer").is_none(), "{entry}");
    server.ingest.shutdown().await.unwrap();
}

/// Decision D8: a row of a future step with `seq` set, alone in its batch, leaves the step watermark
/// alone. The store's `max_step` is the last training step, the parts the replay lists hold no such
/// row, and the rank's `last_step` does not move. Fails on the base: the row is stored in a part.
#[tokio::test]
async fn a_row_of_a_future_step_with_seq_set_leaves_the_step_watermark_alone() {
    let run = "al2b_d8";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let training: Vec<Value> = (0..4).map(|step| step_record(run, step)).collect();
    {
        let server = start(root).await;
        assert_eq!(send_one(&server.address, run, batch(&training), 4).await.unwrap(), 4);
        let future = [scalar_record(run, 1000, 0)];
        assert_eq!(send_one(&server.address, run, batch(&future), 5).await.unwrap(), 5);
        let entry = rank_entry(&server.ingest, run);
        assert_eq!(entry["last_step"], 3, "{entry}");
        server.ingest.shutdown().await.unwrap();
    }
    let store = storage(root);
    assert_eq!(store.max_step(run, 0, 1).await.unwrap(), Some(3));
    let listed = store.rank_parts(run, 0).await.unwrap();
    assert!(
        listed
            .iter()
            .all(|part| part.location.filename().unwrap().starts_with("part-")),
        "{listed:?}"
    );
    assert_eq!(stored_steps(root, run), [0, 1, 2, 3]);
    assert_eq!(files(root, run, "scalars-"), ["scalars-00000000-00000000.parquet"]);
    // After a restart, steps 4 and 5 are new.
    let server = start(root).await;
    let next: Vec<Value> = (4..6).map(|step| step_record(run, step)).collect();
    assert_eq!(send_one(&server.address, run, batch(&next), 7).await.unwrap(), 7);
    assert_eq!(stored_steps(root, run), [0, 1, 2, 3, 4, 5]);
    server.ingest.shutdown().await.unwrap();
}

/// Case `scalar_replay_from_byte_0`: the sender sends its whole file again from byte 0 (a replaced
/// store's ID, rewritten bytes). The scalar rows at or below the attempt's `seq` watermark are
/// dropped and counted as `scalars_not_newer`; nothing is stored twice. Fails on the base: the scalar
/// rows are step rows there, stored in parts, and counted in `rows_not_newer` when sent again.
#[tokio::test]
async fn scalar_replay_from_byte_0() {
    let case = case("scalar_replay_from_byte_0");
    let records = records(&case);
    let run = records[0]["run_id"].as_str().unwrap().to_string();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    assert_eq!(send_one(&server.address, &run, batch(&records), 5).await.unwrap(), 5);
    assert_eq!(send_one(&server.address, &run, batch(&records), 5).await.unwrap(), 5);
    assert_eq!(stored_steps(root, &run), [0, 1, 2]);
    assert_eq!(stored_seqs(root, &run), [0, 1]);
    let entry = rank_entry(&server.ingest, &run);
    assert_eq!(
        entry["scalars_not_newer"], case["expect"]["scalars_not_newer"],
        "{entry}"
    );
    assert_eq!(
        entry["rows_not_newer"], 3,
        "the three training rows sent again: {entry}"
    );
    assert_eq!(entry["last_step"], 2, "{entry}");
    server.ingest.shutdown().await.unwrap();
    // And again after a restart, where the watermark is read from the objects' names.
    let server = start(root).await;
    assert_eq!(send_one(&server.address, &run, batch(&records), 5).await.unwrap(), 5);
    assert_eq!(stored_steps(root, &run), [0, 1, 2]);
    assert_eq!(stored_seqs(root, &run), [0, 1]);
    // Dropped on arrival, not written again and met as an existing object: this server counts both.
    let entry = rank_entry(&server.ingest, &run);
    assert_eq!(entry["scalars_not_newer"], 2, "{entry}");
    server.ingest.shutdown().await.unwrap();
}

/// Case `scalar_seq_watermark_across_restart`: the server restarts between `seq` 1 and `seq` 2. The
/// watermark is read back from the `scalars-*` objects' names, so `seq` 2 is stored and a resend of
/// 0 and 1 is not. Fails on the base: no `scalars-*` object is written.
#[tokio::test]
async fn scalar_seq_watermark_across_restart() {
    let case = case("scalar_seq_watermark_across_restart");
    let records = records(&case);
    let run = records[0]["run_id"].as_str().unwrap().to_string();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    {
        let server = start(root).await;
        assert_eq!(
            send_one(&server.address, &run, batch(&records[..2]), 2).await.unwrap(),
            2
        );
        server.ingest.shutdown().await.unwrap();
    }
    let server = start(root).await;
    assert_eq!(send_one(&server.address, &run, batch(&records), 3).await.unwrap(), 3);
    assert_eq!(stored_seqs(root, &run), [0, 1, 2]);
    assert_eq!(
        files(root, &run, "scalars-"),
        ["scalars-00000000-00000001.parquet", "scalars-00000002-00000002.parquet"]
    );
    assert!(files(root, &run, "part-").is_empty(), "no training row was sent");
    let entry = rank_entry(&server.ingest, &run);
    assert_eq!(entry["scalars_not_newer"], 2, "{entry}");
    assert!(entry.get("last_step").is_none_or(Value::is_null), "{entry}");
    server.ingest.shutdown().await.unwrap();
}

/// Section 4.5: the watermark comes from the objects' names, and an object whose name does not parse
/// is read by its footer, as `max_step` reads a part. Here `scalars-legacy.parquet` holds `seq` 0 and
/// 1: after a restart only `seq` 2 of a resend of 0 to 2 is new, and the object is left as it was.
#[tokio::test]
async fn a_scalars_object_whose_name_does_not_parse_is_read_by_its_footer() {
    let run = "al2b_unnamed";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let legacy = attempt_dir(root, run).join("scalars-legacy.parquet");
    let held: Vec<Value> = (0..2).map(|seq| scalar_record(run, 5, seq)).collect();
    write_parquet(&legacy, &batch(&held));
    let before = std::fs::read(&legacy).unwrap();
    let server = start(root).await;
    let sent: Vec<Value> = (0..3).map(|seq| scalar_record(run, 5, seq)).collect();
    assert_eq!(send_one(&server.address, run, batch(&sent), 3).await.unwrap(), 3);
    assert_eq!(
        files(root, run, "scalars-"),
        ["scalars-00000002-00000002.parquet", "scalars-legacy.parquet"]
    );
    assert_eq!(
        column_of(root, run, &["scalars-00000002-00000002.parquet".to_string()], "seq"),
        [2]
    );
    assert_eq!(
        std::fs::read(&legacy).unwrap(),
        before,
        "the server never replaces an object"
    );
    assert_eq!(rank_entry(&server.ingest, run)["scalars_not_newer"], 2);
    server.ingest.shutdown().await.unwrap();
}

/// A batch without a `seq` column (a protocol 4 sender, step 1's sender) is handled exactly as before:
/// every row is a step row, stored as received in a part, under the step watermark, and no
/// `scalars-*` object is written. Here a row of a future step sent so raises the watermark and the
/// steps after it are dropped as not newer, which is what such a sender always got. A pin: it passes
/// on the base.
#[tokio::test]
async fn a_batch_without_a_seq_column_is_handled_as_before() {
    let run = "al2b_no_seq";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    let full = schema();
    let without: Vec<Field> = full
        .fields()
        .iter()
        .filter(|field| !matches!(field.name().as_str(), "seq" | "record_version"))
        .map(|field| field.as_ref().clone())
        .collect();
    let without = Arc::new(Schema::new(without));
    let rows = |steps: &[i64], version: i32| -> RecordBatch {
        let rows: Vec<Value> = steps
            .iter()
            .map(|step| {
                let mut row = row(&step_record(run, *step));
                let row_object = row.as_object_mut().unwrap();
                row_object.remove("seq");
                row_object.remove("record_version");
                row_object.insert("schema_version".into(), json!(version));
                row
            })
            .collect();
        decode(without.clone(), &rows)
    };
    let first = rows(&[0, 1, 2, 1000], 1);
    assert_eq!(
        send_versioned(&server.address, run, first.clone(), 1, 4).await.unwrap(),
        4
    );
    assert_eq!(
        send_versioned(&server.address, run, rows(&[3, 4], 1), 1, 6)
            .await
            .unwrap(),
        6
    );
    assert_eq!(files(root, run, "part-"), ["part-00000000-00001000.parquet"]);
    let stored = read_file(&attempt_dir(root, run).join("part-00000000-00001000.parquet"));
    assert_eq!(stored, [first], "stored as received");
    assert!(files(root, run, "scalars-").is_empty());
    let entry = rank_entry(&server.ingest, run);
    assert_eq!(entry["last_step"], 1000, "{entry}");
    assert_eq!(entry["rows_not_newer"], 2, "{entry}");
    assert!(entry.get("scalars_not_newer").is_none(), "{entry}");
    server.ingest.shutdown().await.unwrap();
}

/// A batch whose `seq` column is all null (step 1's layout with the column) is step rows only, stored
/// as received with the column, and no `scalars-*` object is written. A pin.
#[tokio::test]
async fn a_batch_whose_seq_is_all_null_is_step_rows_stored_as_received() {
    let run = "al2b_null_seq";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    let sent = batch(&(0..3).map(|step| step_record(run, step)).collect::<Vec<_>>());
    assert_eq!(send_one(&server.address, run, sent.clone(), 3).await.unwrap(), 3);
    let stored = read_file(&attempt_dir(root, run).join("part-00000000-00000002.parquet"));
    assert_eq!(stored, [sent]);
    assert!(files(root, run, "scalars-").is_empty());
    server.ingest.shutdown().await.unwrap();
}

/// Q4 (a): the detectors never read a scalar row, live or in a replay. A scalar row of step 1000
/// whose `grad_norm` is not finite sits among training steps 0 to 4: the report reads five steps
/// through step 4 with no incident; after a restart, the rebuild from the store reads the same. The
/// same row with no `seq` (a training row) is reported, so the row would be seen if it were read.
#[tokio::test]
async fn the_detectors_never_read_a_scalar_row_live_or_in_a_replay() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    // `grad_norm` not finite on the row of step 1000; `seq` 0 on it, or null for the control.
    let rows = |run: &str, seq: Option<i64>| -> RecordBatch {
        let mut records: Vec<Value> = (0..5).map(|step| step_record(run, step)).collect();
        let mut odd = step_record(run, 1000);
        if let Some(seq) = seq {
            odd = scalar_record(run, 1000, seq);
        }
        records.insert(3, odd);
        if seq.is_none() {
            // A training row of step 1000 among 0 to 4 would drop the rows after it; the control
            // puts it last instead.
            let odd = records.remove(3);
            records.push(odd);
        }
        let batch = batch(&records);
        let index = batch.schema().index_of("grad_norm").unwrap();
        let steps = batch
            .column_by_name("step")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .clone();
        let norms: Float64Array = steps
            .iter()
            .map(|step| Some(if step == Some(1000) { f64::NAN } else { 1.0 }))
            .collect();
        let mut columns = batch.columns().to_vec();
        columns[index] = Arc::new(norms);
        RecordBatch::try_new(batch.schema(), columns).unwrap()
    };
    // The control: the row read as a training row is reported.
    {
        let control = "al2b_detect_control";
        let server = start(root).await;
        assert_eq!(
            send_one(&server.address, control, rows(control, None), 6)
                .await
                .unwrap(),
            6
        );
        let report = report_through(root, control, 1000).await;
        let kinds: Vec<&str> = report["incidents"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|incident| incident["kind"].as_str())
            .collect();
        assert!(kinds.contains(&"non_finite"), "{report}");
        server.ingest.shutdown().await.unwrap();
    }
    let run = "al2b_detect";
    {
        let server = start(root).await;
        assert_eq!(send_one(&server.address, run, rows(run, Some(0)), 6).await.unwrap(), 6);
        let report = report_through(root, run, 4).await;
        assert_eq!(report["steps"], 5, "{report}");
        assert_eq!(report["incidents"], json!([]), "{report}");
        server.ingest.shutdown().await.unwrap();
    }
    // The restart: the rank's first part since the start rebuilds from every stored part.
    let server = start(root).await;
    assert_eq!(
        send_one(&server.address, run, batch(&[step_record(run, 5)]), 7)
            .await
            .unwrap(),
        7
    );
    let report = report_through(root, run, 5).await;
    assert_eq!(report["rebuilds"], 1, "{report}");
    assert_eq!(report["steps"], 6, "{report}");
    assert_eq!(report["incidents"], json!([]), "{report}");
    server.ingest.shutdown().await.unwrap();
}

// --- the acknowledgement, and a create that meets an existing object --------------------------------

/// A local store that can be told to fail the `PUT` of a `scalars-*` object, before or after storing
/// it, and that records the file name of every `PUT` in order.
#[derive(Debug)]
struct FaultStore {
    inner: Arc<dyn ObjectStore>,
    fail_scalars: AtomicBool,
    store_then_fail: AtomicBool,
    puts: Mutex<Vec<String>>,
}

impl std::fmt::Display for FaultStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FaultStore({})", self.inner)
    }
}

impl FaultStore {
    fn new(root: &Path) -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(LocalFileSystem::new_with_prefix(root).unwrap()),
            fail_scalars: AtomicBool::new(false),
            store_then_fail: AtomicBool::new(false),
            puts: Mutex::new(Vec::new()),
        })
    }

    /// The names put, parts and scalars objects only, in order.
    fn data_puts(&self) -> Vec<String> {
        self.puts
            .lock()
            .unwrap()
            .iter()
            .filter(|name| name.ends_with(".parquet"))
            .cloned()
            .collect()
    }
}

impl ObjectStore for FaultStore {
    fn put_opts<'life0, 'life1, 'async_trait>(
        &'life0 self,
        location: &'life1 StorePath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> BoxFuture<'async_trait, object_store::Result<object_store::PutResult>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        let name = location.filename().unwrap_or_default().to_string();
        self.puts.lock().unwrap().push(name.clone());
        let scalars = name.starts_with("scalars-");
        let failure = || object_store::Error::Generic {
            store: "FaultStore",
            source: "the test fails this put".into(),
        };
        if scalars && self.fail_scalars.load(Ordering::SeqCst) {
            return futures::future::err(failure()).boxed();
        }
        if scalars && self.store_then_fail.swap(false, Ordering::SeqCst) {
            let stored = self.inner.put_opts(location, payload, opts);
            return async move {
                stored.await?;
                Err(failure())
            }
            .boxed();
        }
        self.inner.put_opts(location, payload, opts)
    }

    fn put_multipart_opts<'life0, 'life1, 'async_trait>(
        &'life0 self,
        location: &'life1 StorePath,
        opts: PutMultipartOptions,
    ) -> BoxFuture<'async_trait, object_store::Result<Box<dyn MultipartUpload>>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        self.inner.put_multipart_opts(location, opts)
    }

    fn get_opts<'life0, 'life1, 'async_trait>(
        &'life0 self,
        location: &'life1 StorePath,
        options: GetOptions,
    ) -> BoxFuture<'async_trait, object_store::Result<GetResult>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        self.inner.get_opts(location, options)
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<StorePath>>,
    ) -> BoxStream<'static, object_store::Result<StorePath>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&StorePath>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_delimiter<'life0, 'life1, 'async_trait>(
        &'life0 self,
        prefix: Option<&'life1 StorePath>,
    ) -> BoxFuture<'async_trait, object_store::Result<ListResult>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        self.inner.list_with_delimiter(prefix)
    }

    fn copy_opts<'life0, 'life1, 'life2, 'async_trait>(
        &'life0 self,
        from: &'life1 StorePath,
        to: &'life2 StorePath,
        options: CopyOptions,
    ) -> BoxFuture<'async_trait, object_store::Result<()>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        'life2: 'async_trait,
        Self: 'async_trait,
    {
        self.inner.copy_opts(from, to, options)
    }
}

/// A server in this process over a `FaultStore`.
struct FaultServer {
    address: String,
    ingest: Arc<Ingest>,
    _stop: oneshot::Sender<()>,
}

/// A server in this process over `store`, as `lane_aa3_support::start` makes one.
async fn start_over(store: Arc<FaultStore>) -> FaultServer {
    start_with(store, 1, Duration::from_secs(3600)).await
}

/// A server in this process over `store` that writes a rank's buffer once it holds `flush_steps`
/// steps or its oldest row has waited `flush_interval`.
async fn start_with(store: Arc<FaultStore>, flush_steps: usize, flush_interval: Duration) -> FaultServer {
    let config = IngestConfig {
        flush_steps,
        flush_interval,
        ..IngestConfig::default()
    };
    let ingest = Ingest::new(Storage::new(store, "".into()), config);
    ingest.spawn_flusher();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let (stop, stopped) = oneshot::channel::<()>();
    let served = ingest.clone();
    tokio::spawn(async move {
        serve(listener, served, None, None, async {
            let _ = stopped.await;
        })
        .await
        .unwrap();
    });
    FaultServer {
        address,
        ingest,
        _stop: stop,
    }
}

/// Section 4.5: an offset is acknowledged only when every row before it is stored, of both kinds. A
/// batch of steps 0 to 2 and one scalar row: the step part is written first, then the scalars
/// object, then the offset is acknowledged. While the scalars object cannot be written, the part is
/// stored and nothing is acknowledged; the sender sends the batch again, and once the object is
/// stored the offset is. Fails on the base: the scalar row is a step row there, and the offset is
/// acknowledged with the part.
#[tokio::test]
async fn an_offset_is_acknowledged_only_when_the_part_and_then_the_scalars_object_are_stored() {
    let run = "al2b_ack";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let store = FaultStore::new(root);
    let server = start_over(store.clone()).await;
    let mut sent: Vec<Value> = (0..3).map(|step| step_record(run, step)).collect();
    sent.insert(2, scalar_record(run, 1, 0));
    store.fail_scalars.store(true, Ordering::SeqCst);
    let refused = send_one(&server.address, run, batch(&sent), 4)
        .await
        .expect_err("nothing is acknowledged while the scalars object is not stored");
    assert_eq!(refused.code(), tonic::Code::Unavailable, "{refused:?}");
    assert_eq!(files(root, run, "part-"), ["part-00000000-00000002.parquet"]);
    assert!(files(root, run, "scalars-").is_empty());
    store.fail_scalars.store(false, Ordering::SeqCst);
    assert_eq!(send_one(&server.address, run, batch(&sent), 4).await.unwrap(), 4);
    assert_eq!(stored_steps(root, run), [0, 1, 2]);
    assert_eq!(stored_seqs(root, run), [0]);
    let puts = store.data_puts();
    assert_eq!(
        puts,
        [
            "part-00000000-00000002.parquet",
            "scalars-00000000-00000000.parquet",
            "scalars-00000000-00000000.parquet",
        ],
        "the part first, then the scalars object, tried again under its name"
    );
    server.ingest.shutdown().await.unwrap();
}

/// Section 4.5: a create that meets an existing object is read for its `seq` range. A `PUT` that
/// stored the object and then reported failure is tried again under the same name; it meets the
/// object, whose range is the rows' own, so the rows are stored and the offset is acknowledged, and
/// the object is written once. Fails on the base: no `scalars-*` object is written.
#[tokio::test]
async fn a_create_that_meets_an_object_of_the_same_seq_range_counts_the_rows_stored() {
    let run = "al2b_same_range";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let store = FaultStore::new(root);
    let server = start_over(store.clone()).await;
    let sent: Vec<Value> = (0..2).map(|seq| scalar_record(run, 7, seq)).collect();
    store.store_then_fail.store(true, Ordering::SeqCst);
    let refused = send_one(&server.address, run, batch(&sent), 2)
        .await
        .expect_err("the put reported failure");
    assert_eq!(refused.code(), tonic::Code::Unavailable, "{refused:?}");
    assert_eq!(files(root, run, "scalars-"), ["scalars-00000000-00000001.parquet"]);
    let before = std::fs::read(attempt_dir(root, run).join("scalars-00000000-00000001.parquet")).unwrap();
    assert_eq!(send_one(&server.address, run, batch(&sent), 2).await.unwrap(), 2);
    assert_eq!(files(root, run, "scalars-"), ["scalars-00000000-00000001.parquet"]);
    assert_eq!(stored_seqs(root, run), [0, 1]);
    assert_eq!(
        std::fs::read(attempt_dir(root, run).join("scalars-00000000-00000001.parquet")).unwrap(),
        before
    );
    server.ingest.shutdown().await.unwrap();
}

/// Section 4.5: a create that meets an object whose `seq` range is not the rows' own keeps the rows
/// buffered, logs the error and acknowledges nothing; the object is left as it was (the server never
/// replaces or deletes one). Here another writer put `scalars-00000001-00000002.parquet` holding
/// `seq` 7 and 8 after the server read the attempt's watermark. Fails on the base: no `scalars-*`
/// object is written, and the offset is acknowledged with the rows in a part.
#[tokio::test]
async fn a_create_that_meets_an_object_of_another_seq_range_keeps_the_rows_and_acknowledges_nothing() {
    let run = "al2b_other_range";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    assert_eq!(
        send_one(&server.address, run, batch(&[scalar_record(run, 1, 0)]), 1)
            .await
            .unwrap(),
        1
    );
    let foreign = attempt_dir(root, run).join("scalars-00000001-00000002.parquet");
    write_parquet(&foreign, &batch(&[scalar_record(run, 1, 7), scalar_record(run, 1, 8)]));
    let before = std::fs::read(&foreign).unwrap();
    let sent = batch(&[scalar_record(run, 2, 1), scalar_record(run, 2, 2)]);
    let refused = send_one(&server.address, run, sent, 3)
        .await
        .expect_err("nothing is acknowledged");
    assert_eq!(refused.code(), tonic::Code::Unavailable, "{refused:?}");
    assert_eq!(std::fs::read(&foreign).unwrap(), before);
    assert_eq!(
        files(root, run, "scalars-"),
        ["scalars-00000000-00000000.parquet", "scalars-00000001-00000002.parquet"]
    );
    server.ingest.shutdown().await.expect_err("the rows are still buffered");
}

/// The counts the rank's entry carries are what the tests above read; `scalars_not_newer` is absent
/// while none was dropped, so a status that dropped none is as it was.
#[tokio::test]
async fn scalars_not_newer_is_absent_until_a_row_is_dropped() {
    let run = "al2b_absent";
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    let sent = batch(&[step_record(run, 0), scalar_record(run, 0, 0)]);
    assert_eq!(send_one(&server.address, run, sent.clone(), 2).await.unwrap(), 2);
    let entry = rank_entry(&server.ingest, run);
    assert!(entry.get("scalars_not_newer").is_none(), "{entry}");
    assert_eq!(send_one(&server.address, run, sent, 2).await.unwrap(), 2);
    let entry = rank_entry(&server.ingest, run);
    assert_eq!(entry["scalars_not_newer"], 1, "{entry}");
    let counted: BTreeMap<String, Value> = ["rows_not_newer", "scalars_not_newer"]
        .iter()
        .map(|key| (key.to_string(), entry[*key].clone()))
        .collect();
    assert_eq!(counted["rows_not_newer"], 1, "{entry}");
    server.ingest.shutdown().await.unwrap();
}

/// Section 4.5 with a `PUT` that stored the scalars object and then reported failure: until the object
/// is stored, the rank takes no new scalar row, so the write is tried again under the same name and
/// no second object over the same `seq` is made. Here the sender sends `seq` 0 and 1 again with a new
/// `seq` 2: the first object is met and taken as stored, and `seq` 2 gets one of its own.
#[tokio::test]
async fn after_a_failed_scalars_write_no_new_scalar_row_is_taken_until_it_is_stored() {
    let run = "al2b_sealed";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let store = FaultStore::new(root);
    let server = start_over(store.clone()).await;
    store.store_then_fail.store(true, Ordering::SeqCst);
    let first: Vec<Value> = (0..2).map(|seq| scalar_record(run, 3, seq)).collect();
    send_one(&server.address, run, batch(&first), 2)
        .await
        .expect_err("the put reported failure");
    let again: Vec<Value> = (0..3).map(|seq| scalar_record(run, 3, seq)).collect();
    assert_eq!(send_one(&server.address, run, batch(&again), 3).await.unwrap(), 3);
    assert_eq!(
        files(root, run, "scalars-"),
        ["scalars-00000000-00000001.parquet", "scalars-00000002-00000002.parquet"]
    );
    assert_eq!(stored_seqs(root, run), [0, 1, 2], "no seq is stored twice");
    server.ingest.shutdown().await.unwrap();
}

/// Section 4.5: an object holds one attempt, under that attempt's folder. A batch whose scalar rows
/// are of two attempts (the end of one session's file and the start of the next) gives each its own
/// object, each with its own `seq` watermark.
#[tokio::test]
async fn the_scalar_rows_of_two_attempts_are_stored_apart() {
    let run = "al2b_attempts";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    let mut later = scalar_record(run, 0, 0);
    later["attempt"] = json!(2);
    let sent = batch(&[scalar_record(run, 4, 0), scalar_record(run, 4, 1), later]);
    assert_eq!(send_one(&server.address, run, sent, 3).await.unwrap(), 3);
    assert_eq!(stored_seqs(root, run), [0, 1]);
    let second = root.join("runs").join(run).join("rank=0").join("attempt=2");
    let names: Vec<String> = std::fs::read_dir(&second)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["scalars-00000000-00000000.parquet"]);
    server.ingest.shutdown().await.unwrap();
}

/// Section 4.5: an object has one schema. Scalar rows buffered beside step rows, then a batch of
/// scalar rows only in another layout (a column more): the buffered rows are stored first, as a part
/// is when its schema changes, then the new ones, and the offset is acknowledged.
#[tokio::test]
async fn scalar_rows_of_another_schema_begin_an_object_of_their_own() {
    let run = "al2b_schemas";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let store = FaultStore::new(root);
    let server = start_with(store, 100, Duration::from_secs(3600)).await;
    let mut put = open(&server.address, run, schema()).await;
    put.send(batch(&[step_record(run, 0), scalar_record(run, 0, 0)]), 2, 2)
        .await;
    let mut fields: Vec<Field> = schema().fields().iter().map(|field| field.as_ref().clone()).collect();
    fields.push(Field::new("lr", DataType::Float64, true));
    let wider = Arc::new(Schema::new(fields));
    let mut row = row(&scalar_record(run, 0, 1));
    row["lr"] = json!(0.1);
    let mut wide = open(&server.address, run, wider.clone()).await;
    wide.send(decode(wider, &[row]), 2, 3).await;
    assert_eq!(wide.reply().await.unwrap(), 3);
    assert_eq!(
        files(root, run, "scalars-"),
        ["scalars-00000000-00000000.parquet", "scalars-00000001-00000001.parquet"]
    );
    assert_eq!(stored_steps(root, run), [0]);
    wide.finish().await;
    put.finish().await;
    server.ingest.shutdown().await.unwrap();
}

/// Section 4.5: scalar rows whose object could not be written are tried again by the timed flush, as
/// a part is, with no new batch from the sender.
#[tokio::test]
async fn the_timed_flush_tries_a_failed_scalars_write_again() {
    let run = "al2b_timed";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let store = FaultStore::new(root);
    let server = start_with(store.clone(), 1, Duration::from_millis(100)).await;
    store.fail_scalars.store(true, Ordering::SeqCst);
    send_one(&server.address, run, batch(&[scalar_record(run, 0, 0)]), 1)
        .await
        .expect_err("not stored");
    store.fail_scalars.store(false, Ordering::SeqCst);
    // The object is on disk before the flush gives its bytes back (`flush_scalars` writes it, then
    // subtracts them), so both are waited for in one loop with one deadline: reading the count as
    // soon as the object appeared failed about 1 run in 6 alone.
    for _ in 0..200 {
        if !files(root, run, "scalars-").is_empty() && server.ingest.buffered_bytes() == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(files(root, run, "scalars-"), ["scalars-00000000-00000000.parquet"]);
    assert_eq!(server.ingest.buffered_bytes(), 0, "nothing is held once it is stored");
    server.ingest.shutdown().await.unwrap();
}

/// Q4 (a), the replay side, read directly: `detect::replay`, the read a restart and a rebuild make,
/// over a store that holds a part of training steps 0 to 2 and a `scalars-*` object whose row is of
/// step 1000 with a `grad_norm` that is not finite. The report reads the part alone: three steps,
/// through step 2, no incident. The object is the server's own, as the live path writes it.
#[tokio::test]
async fn the_replay_reads_the_parts_alone_and_never_a_scalars_object() {
    let run = "al2b_replay_read";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let store = storage(root);
    let training = batch(&(0..3).map(|step| step_record(run, step)).collect::<Vec<_>>());
    store.write_part(run, 0, 1, &[training], 0, 2).await.unwrap();
    let scalar = batch(&[scalar_record(run, 1000, 0)]);
    let index = scalar.schema().index_of("grad_norm").unwrap();
    let mut columns = scalar.columns().to_vec();
    columns[index] = Arc::new(Float64Array::from(vec![f64::NAN]));
    let scalar = RecordBatch::try_new(scalar.schema(), columns).unwrap();
    store.write_scalars(run, 0, 1, &[scalar], 0, 0).await.unwrap();
    assert_eq!(files(root, run, "scalars-"), ["scalars-00000000-00000000.parquet"]);
    let thresholds = Arc::new(tensorquill_server::detect::Thresholds::default());
    let detectors = tensorquill_server::detect::replay(&store, run, 0, &thresholds)
        .await
        .unwrap();
    let report = tensorquill_server::detect::report(run, 0, &detectors, 1);
    assert_eq!(report["steps"], 3, "{report}");
    assert_eq!(report["last_step"], 2, "{report}");
    assert_eq!(report["incidents"], json!([]), "{report}");
}
