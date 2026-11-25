//! Lane AL-2 of wave AL (`WAVE_AL_CONTRACT.md` sections 2.4, 3 and 5): the server takes protocol 5's
//! layout beside protocol 4's.
//!
//! - Batch metadata `schema_version` 1 and 2 are both taken; any other is refused, naming both.
//! - Row `schema_version` 0, 1 and 2 are stored; any other is dropped and counted (RS10).
//! - A protocol 4 sender works unchanged: its batches are stored as received, in parts of their own.
//! - The detectors' read of a part (`Storage::read_part_for_detect`, what `detect::replay` calls)
//!   leaves out `probe_maps` and `probe_histograms`, so a replay never decodes them. `Storage::read_part`
//!   reads every column (the owner's choice at the seam, Nov 24, 2026: a read loses nothing by default).
//! - A restart over a store of protocol 4 parts, of protocol 5 parts, or of both, replays every part
//!   by its columns; the step watermark is from the parts' names and footers, as before.
//! - `layers_model_order`: the `layers` map keeps the model's entry order on the part and in the
//!   replay read after a restart.
//!
//! The typed columns are built from the contract's sections 1.2 and 2.4 with arrow's JSON decoder,
//! as `experiments/protocol-v5/prototype.py` lays them out. No test here depends on a layer being
//! judged from a typed column: that is lane AL-3's reader.

mod lane_aa3_support;

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Array, MapArray, StringArray};
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
use tensorquill_server::storage::Storage;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Channel;

const RUN: &str = "al2";
const ATTEMPT: i64 = 1_732_460_000_000;

/// A model's order of blocks, not sorted (case `layers_model_order`).
const MODEL_ORDER: [&str; 6] = [
    "model.layers.0",
    "model.layers.1",
    "model.layers.2",
    "model.layers.10",
    "model.layers.11",
    "model.layers.3",
];

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

fn floats(names: &[&str]) -> Vec<Field> {
    names
        .iter()
        .map(|name| Field::new(*name, DataType::Float64, true))
        .collect()
}

/// Protocol 5's layout (sections 1.2, 2.4 and 3): the typed `layers` and `moe_stats` maps in place
/// of their JSON columns, the probe's maps and histograms beside `probe_json`, and `seq` and
/// `record_version`.
fn schema_v5() -> Arc<Schema> {
    let layer = DataType::Struct(Fields::from(floats(&["mean", "std", "abs_max", "zero_frac"])));
    let moe = DataType::Struct(Fields::from(vec![
        Field::new("routed_tokens", DataType::Int64, true),
        Field::new("entropy", DataType::Float64, true),
        Field::new("diversity", DataType::Float64, true),
        Field::new("dead_experts", DataType::Int64, true),
        Field::new("dead_frac", DataType::Float64, true),
    ]));
    let grid = DataType::List(Arc::new(Field::new_list_field(
        DataType::List(Arc::new(Field::new_list_field(DataType::Float32, true))),
        true,
    )));
    let histogram = DataType::Struct(Fields::from(vec![
        Field::new("lo", DataType::Float64, true),
        Field::new("hi", DataType::Float64, true),
        Field::new(
            "counts",
            DataType::List(Arc::new(Field::new_list_field(DataType::Int64, true))),
            true,
        ),
        Field::new("nonfinite", DataType::Int64, true),
    ]));
    Arc::new(Schema::new(vec![
        Field::new("schema_version", DataType::Int32, true),
        Field::new("run_id", DataType::Utf8, true),
        Field::new("rank", DataType::Int32, true),
        Field::new("attempt", DataType::Int64, true),
        Field::new("step", DataType::Int64, true),
        Field::new("ts", DataType::Float64, true),
        Field::new("grad_norm", DataType::Float64, true),
        Field::new("layers", map_of(layer), true),
        Field::new("moe_stats", map_of(moe), true),
        Field::new("probe_json", DataType::Utf8, true),
        Field::new("probe_maps", map_of(grid), true),
        Field::new("probe_histograms", map_of(histogram), true),
        Field::new("seq", DataType::Int64, true),
        Field::new("record_version", DataType::Int32, true),
        Field::new("extra_json", DataType::Utf8, true),
    ]))
}

/// Protocol 4's layout, as `flight.py` writes it today: the same statistics as JSON text.
fn schema_v4() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("schema_version", DataType::Int32, true),
        Field::new("run_id", DataType::Utf8, true),
        Field::new("rank", DataType::Int32, true),
        Field::new("attempt", DataType::Int64, true),
        Field::new("step", DataType::Int64, true),
        Field::new("ts", DataType::Float64, true),
        Field::new("grad_norm", DataType::Float64, true),
        Field::new("layers_json", DataType::Utf8, true),
        Field::new("moe_stats_json", DataType::Utf8, true),
        Field::new("probe_json", DataType::Utf8, true),
        Field::new("extra_json", DataType::Utf8, true),
    ]))
}

/// The `layers` object of a record at `step`, as JSON text with its entries in `MODEL_ORDER`
/// (`serde_json::Map` would sort them).
fn layers_text(step: i64) -> String {
    let entries: Vec<String> = MODEL_ORDER
        .iter()
        .enumerate()
        .map(|(i, name)| {
            format!(
                "\"{name}\":{{\"mean\":0.0,\"std\":{},\"abs_max\":{},\"zero_frac\":0.0}}",
                1.0 + i as f64 / 100.0,
                3.0 + step as f64 / 100.0
            )
        })
        .collect();
    format!("{{{}}}", entries.join(","))
}

/// Where a row's `layers` text goes, put in after the rest of the row is written.
const LAYERS: &str = "\"<layers>\"";

fn decode(schema: Arc<Schema>, rows: &[serde_json::Value]) -> RecordBatch {
    let text: String = rows.iter().map(|row| format!("{row}\n")).collect();
    let text = text.replace(LAYERS, &layers_text(0));
    let mut reader = arrow::json::ReaderBuilder::new(schema.clone())
        .with_batch_size(rows.len().max(1))
        .build(Cursor::new(text))
        .unwrap();
    let batch = reader.next().unwrap().unwrap();
    assert!(reader.next().is_none());
    assert_eq!(batch.schema(), schema);
    batch
}

/// Rows of protocol 5's layout at `steps`, of row version `version`, each with a probe whose map and
/// histogram are in their own columns (section 2.4).
fn batch_v5(steps: std::ops::Range<i64>, version: i32) -> RecordBatch {
    let rows: Vec<serde_json::Value> = steps
        .map(|step| {
            serde_json::json!({
                "schema_version": version, "run_id": RUN, "rank": 0, "attempt": ATTEMPT, "step": step,
                "ts": step as f64, "grad_norm": 1.0,
                "layers": "<layers>",
                "moe_stats": {"model.layers.0.mlp.router": {"routed_tokens": 512, "entropy": 2.1, "diversity": 0.9, "dead_experts": 0, "dead_frac": 0.0}},
                "probe_json": "{\"attention\":{\"model.layers.0.self_attn\":{\"max_logit\":[3.5],\"rows\":2}}}",
                "probe_maps": {"model.layers.0.self_attn": [[0.25, 0.75], [0.5, 0.5]]},
                "probe_histograms": {"model.layers.0": {"lo": -1.0, "hi": 1.0, "counts": [1, 2, 3], "nonfinite": 0}},
                "seq": null, "record_version": 1, "extra_json": null,
            })
        })
        .collect();
    decode(schema_v5(), &rows)
}

/// Rows of protocol 4's layout at `steps`, of row version `version`.
fn batch_v4(steps: std::ops::Range<i64>, version: i32) -> RecordBatch {
    let rows: Vec<serde_json::Value> = steps
        .map(|step| {
            serde_json::json!({
                "schema_version": version, "run_id": RUN, "rank": 0, "attempt": ATTEMPT, "step": step,
                "ts": step as f64, "grad_norm": 1.0,
                "layers_json": layers_text(step),
                "moe_stats_json": null, "probe_json": null, "extra_json": null,
            })
        })
        .collect();
    decode(schema_v4(), &rows)
}

/// One open `DoPut` stream whose batches carry metadata of a chosen `schema_version`.
struct Put {
    messages: mpsc::UnboundedSender<Result<FlightData, FlightError>>,
    acks: BoxStream<'static, Result<PutResult, FlightError>>,
}

async fn open(address: &str, schema: Arc<Schema>) -> Put {
    let channel = Channel::from_shared(address.to_string())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightClient::new(channel);
    let (messages, outgoing) = mpsc::unbounded_channel();
    let descriptor = FlightDescriptor::new_path(vec!["runs".into(), RUN.into(), "rank=0".into()]);
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
    Put { messages, acks }
}

impl Put {
    async fn send(&mut self, batch: RecordBatch, metadata_version: i64, end_byte_offset: i64) {
        let metadata = format!(
            "{{\"schema_version\":{metadata_version},\"run_id\":\"{RUN}\",\"rank\":0,\"min_step\":null,\"max_step\":null,\"end_byte_offset\":{end_byte_offset}}}"
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

    /// The next acknowledged `end_byte_offset`, or the status that ended the stream.
    async fn reply(&mut self) -> Result<i64, tonic::Status> {
        match tokio::time::timeout(Duration::from_secs(20), self.acks.next())
            .await
            .expect("no reply")
        {
            Some(Ok(result)) => {
                let reply: serde_json::Value = serde_json::from_slice(&result.app_metadata).unwrap();
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

/// Sends `batch` on a stream of its own with metadata `version`, and returns its acknowledgement.
async fn send_one(server: &TestServer, batch: RecordBatch, version: i64, end: i64) -> Result<i64, tonic::Status> {
    let mut put = open(&server.address, batch.schema()).await;
    put.send(batch, version, end).await;
    let reply = put.reply().await;
    if reply.is_ok() {
        put.finish().await;
    }
    reply
}

/// The rank's parts, sorted by name.
fn parts(root: &Path) -> Vec<PathBuf> {
    let dir = root
        .join("runs")
        .join(RUN)
        .join("rank=0")
        .join(format!("attempt={ATTEMPT}"));
    let mut parts: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| entries.map(|entry| entry.unwrap().path()).collect())
        .unwrap_or_default();
    parts.retain(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("part-") && name.ends_with(".parquet"))
    });
    parts.sort();
    parts
}

/// Every row of a part file, every column, read with no projection.
fn read_whole(path: &Path) -> Vec<RecordBatch> {
    let file = std::fs::File::open(path).unwrap();
    ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

fn storage(root: &Path) -> Storage {
    Storage::new(Arc::new(LocalFileSystem::new_with_prefix(root).unwrap()), "".into())
}

/// The part's location in `storage`, from its file path under `root`.
fn location(root: &Path, path: &Path) -> object_store::path::Path {
    object_store::path::Path::from(path.strip_prefix(root).unwrap().to_str().unwrap())
}

/// The keys of every row's `layers` map, in stored order.
fn layer_keys(batches: &[RecordBatch]) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for batch in batches {
        let layers = batch.column_by_name("layers").expect("a `layers` column");
        let layers = layers.as_any().downcast_ref::<MapArray>().unwrap();
        for row in 0..layers.len() {
            let entries = layers.value(row);
            let keys = entries.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            out.push(keys.iter().map(|key| key.unwrap().to_string()).collect());
        }
    }
    out
}

fn steps_of(batches: &[RecordBatch]) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|batch| {
            let steps = batch.column_by_name("step").unwrap();
            let steps = steps.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
            steps.values().to_vec()
        })
        .collect()
}

fn names(batch: &RecordBatch) -> Vec<String> {
    batch
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect()
}

/// The rank's `incidents.json` once it covers `last_step`.
async fn report_through(root: &Path, last_step: i64) -> serde_json::Value {
    let path = root.join("runs").join(RUN).join("rank=0").join("incidents.json");
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

/// The server's refusal of a batch metadata version it does not read is the text both sides pin in
/// `sdk/tests/fixtures/wave_al/refusal.json` (`this_server`), the plural form a later sender's
/// `_older_server_reads` must recognise beside a protocol 4 server's singular (the owner's seam note,
/// Nov 24, 2026). The Python side reads the same file (`test_wave_al_step1_seams.py`).
#[tokio::test]
async fn the_refusal_of_an_unread_metadata_version_is_the_shared_fixtures_text() {
    let fixture: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../sdk/tests/fixtures/wave_al/refusal.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let this_server = &fixture["this_server"];
    let sent = this_server["sent"].as_i64().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    let refused = send_one(&server, batch_v5(0..5, 2), sent, 100)
        .await
        .expect_err("refused");
    assert_eq!(refused.code(), tonic::Code::InvalidArgument, "{refused:?}");
    assert_eq!(refused.message(), this_server["text"].as_str().unwrap());
}

/// Section 3: a protocol 5 server takes batch metadata versions 1 and 2, and refuses any other with
/// `INVALID_ARGUMENT` naming both versions it reads, before storing or acknowledging anything. Fails
/// on the base (`5c286ed`): version 2 is refused, "this server reads version 1" (D6).
#[tokio::test]
async fn batch_metadata_versions_1_and_2_are_taken_and_any_other_is_refused_naming_both() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    assert_eq!(send_one(&server, batch_v4(0..5, 1), 1, 100).await.unwrap(), 100);
    assert_eq!(send_one(&server, batch_v5(5..10, 2), 2, 200).await.unwrap(), 200);
    for version in [0, 3, 5, -1] {
        let refused = send_one(&server, batch_v5(10..15, 2), version, 300)
            .await
            .expect_err("refused");
        assert_eq!(refused.code(), tonic::Code::InvalidArgument, "{refused:?}");
        assert_eq!(
            refused.message(),
            format!("schema_version {version} is not supported; this server reads versions 1 and 2")
        );
    }
    let stored: Vec<i64> = parts(dir.path())
        .iter()
        .flat_map(|part| steps_of(&read_whole(part)))
        .collect();
    assert_eq!(
        stored,
        (0..10).collect::<Vec<_>>(),
        "nothing of a refused batch is stored"
    );
}

/// Section 3, RS10: under batch metadata 2, rows of `schema_version` 0, 1 and 2 are stored and
/// acknowledged; a row of any other version is dropped and counted under `unsupported_schema_version`, and the rest of its batch
/// is kept. Fails on the base: a row of version 2 is dropped and counted (D6, `old_server_row_v2`).
#[tokio::test]
async fn rows_of_versions_0_1_and_2_are_stored_and_any_other_is_dropped_and_counted() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    let mut rows = batch_v5(0..7, 2);
    let versions = arrow::array::Int32Array::from(vec![2, 0, 1, 3, 2, -1, 99]);
    let mut columns = rows.columns().to_vec();
    columns[0] = Arc::new(versions);
    rows = RecordBatch::try_new(rows.schema(), columns).unwrap();
    // Under metadata 2, protocol 5's: the versions its batch may hold are 0, 1 and 2. Under metadata 1
    // a row of version 2 is dropped too (the owner's decision of Nov 25, 2026, tested in
    // `lane_al2b_v2_under_v1_test.rs`).
    assert_eq!(send_one(&server, rows, 2, 100).await.unwrap(), 100);
    let stored: Vec<i64> = parts(dir.path())
        .iter()
        .flat_map(|part| steps_of(&read_whole(part)))
        .collect();
    assert_eq!(stored, vec![0, 1, 2, 4]);
    let statuses = server.ingest.watchdog_statuses();
    let entry = &statuses[RUN]["ranks"][0];
    assert_eq!(entry["dropped_rows"], 3, "{entry}");
    assert_eq!(
        entry["dropped_row_keys"],
        serde_json::json!({"unsupported_schema_version": 3}),
        "{entry}"
    );
}

/// Sections 2.4 and 3, and case `layers_model_order`: a batch of protocol 5's layout (metadata 2,
/// row version 2) is stored as received, every typed column with it, and its `layers` map keeps the
/// model's entry order on the part. Fails on the base: the batch is refused.
#[tokio::test]
async fn a_protocol_5_batch_is_stored_as_received_with_its_typed_columns_in_entry_order() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    let sent = batch_v5(0..5, 2);
    assert_eq!(send_one(&server, sent.clone(), 2, 100).await.unwrap(), 100);
    let parts = parts(dir.path());
    assert_eq!(parts.len(), 1);
    let stored = read_whole(&parts[0]);
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].schema().fields(), sent.schema().fields());
    assert_eq!(stored[0].columns(), sent.columns(), "stored as received");
    let order: Vec<String> = MODEL_ORDER.iter().map(|name| name.to_string()).collect();
    assert_eq!(layer_keys(&stored), vec![order; 5]);
}

/// Section 5's first row and section 3: a protocol 4 sender (metadata 1, rows of version 1, JSON
/// columns) works unchanged on a protocol 5 server, and its rows are stored as received, with no
/// conversion. When the same rank's protocol 5 rows follow, each layout has parts of its own: a part
/// has one schema. Fails on the base: the protocol 5 rows are refused.
#[tokio::test]
async fn a_protocol_4_sender_is_stored_as_received_and_each_layout_has_parts_of_its_own() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    let old = batch_v4(0..5, 1);
    assert_eq!(send_one(&server, old.clone(), 1, 100).await.unwrap(), 100);
    assert_eq!(send_one(&server, batch_v5(5..10, 2), 2, 200).await.unwrap(), 200);
    let parts = parts(dir.path());
    assert_eq!(parts.len(), 2, "{parts:?}");
    let first = read_whole(&parts[0]);
    assert_eq!(first[0].schema().fields(), old.schema().fields());
    assert_eq!(
        first[0].columns(),
        old.columns(),
        "a protocol 4 batch is stored as received"
    );
    assert_eq!(steps_of(&read_whole(&parts[1])), (5..10).collect::<Vec<_>>());
    assert!(names(&read_whole(&parts[1])[0]).contains(&"layers".to_string()));
}

/// Section 2.4: the detectors' read of a part (`Storage::read_part_for_detect`, which `detect::replay`
/// calls) leaves out `probe_maps` and `probe_histograms` and keeps every other column, in order, with
/// its values, `probe_json` and the typed `layers` included. A protocol 4 part, which has neither, is
/// read whole. `Storage::read_part` reads every column of either. (Lane AL-2 built the projection as
/// `read_part` and the whole read as `read_part_whole`; the owner swapped the names at the seam.)
#[tokio::test]
async fn the_detectors_read_of_a_part_leaves_out_the_probe_arrays_and_read_part_keeps_them() {
    let dir = tempfile::tempdir().unwrap();
    let storage = storage(dir.path());
    let sent = batch_v5(0..5, 2);
    let written = storage
        .write_part(RUN, 0, ATTEMPT, std::slice::from_ref(&sent), 0, 4)
        .await
        .unwrap();
    let read = storage.read_part_for_detect(&written).await.unwrap();
    let kept: Vec<String> = names(&sent)
        .into_iter()
        .filter(|name| name != "probe_maps" && name != "probe_histograms")
        .collect();
    assert_eq!(read.len(), 1);
    assert_eq!(names(&read[0]), kept);
    for name in &kept {
        assert_eq!(
            read[0].column_by_name(name).unwrap(),
            sent.column_by_name(name).unwrap(),
            "{name}"
        );
    }
    // The plain read, what a tool that sends stored parts again reads (`examples/live_streams.rs`):
    // every column.
    assert_eq!(storage.read_part(&written).await.unwrap(), vec![sent]);
    let old = batch_v4(5..10, 1);
    let written = storage
        .write_part(RUN, 0, ATTEMPT, std::slice::from_ref(&old), 5, 9)
        .await
        .unwrap();
    assert_eq!(storage.read_part_for_detect(&written).await.unwrap(), vec![old.clone()]);
    assert_eq!(storage.read_part(&written).await.unwrap(), vec![old]);
}

/// The layouts a store holds before the restart.
#[derive(Clone, Copy, Debug)]
enum Stored {
    Protocol4,
    Protocol5,
    Both,
}

/// Section 5's third row, and `layers_model_order` across a restart: a server that starts again on a
/// store of protocol 4 parts, of protocol 5 parts, or of both, takes the rank's next batch, drops the
/// rows already stored (the watermark is from the parts' names and footers, as before), and rebuilds
/// its detectors from every part, each read by its columns: the report counts every stored row. The
/// replay read of every part keeps the `layers` map's entry order. Fails on the base for a store with
/// protocol 5 parts: they are refused before they are stored.
async fn restart_over(layout: Stored) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    {
        let server = start(root).await;
        let (first, second): (RecordBatch, (RecordBatch, i64)) = match layout {
            Stored::Protocol4 => (batch_v4(0..5, 1), (batch_v4(5..10, 1), 1)),
            Stored::Protocol5 => (batch_v5(0..5, 2), (batch_v5(5..10, 2), 2)),
            Stored::Both => (batch_v4(0..5, 1), (batch_v5(5..10, 2), 2)),
        };
        let version = if matches!(layout, Stored::Protocol5) { 2 } else { 1 };
        assert_eq!(send_one(&server, first, version, 100).await.unwrap(), 100, "{layout:?}");
        assert_eq!(
            send_one(&server, second.0, second.1, 200).await.unwrap(),
            200,
            "{layout:?}"
        );
        report_through(root, 9).await;
        server.ingest.shutdown().await.unwrap();
    }
    // The restart: a new server on the same store, which holds nothing of the rank in memory.
    let server = start(root).await;
    let next = match layout {
        Stored::Protocol4 => batch_v4(8..12, 1),
        _ => batch_v5(8..12, 2),
    };
    let version = if matches!(layout, Stored::Protocol4) { 1 } else { 2 };
    assert_eq!(send_one(&server, next, version, 300).await.unwrap(), 300, "{layout:?}");
    let report = report_through(root, 11).await;
    assert_eq!(report["rebuilds"], 1, "{layout:?}: {report}");
    assert_eq!(
        report["steps"], 12,
        "{layout:?}: every stored row is read once: {report}"
    );
    let stored = parts(root);
    assert_eq!(stored.len(), 3, "{layout:?}: {stored:?}");
    let steps: Vec<i64> = stored.iter().flat_map(|part| steps_of(&read_whole(part))).collect();
    assert_eq!(
        steps,
        (0..12).collect::<Vec<_>>(),
        "{layout:?}: steps 8 and 9 are not stored twice"
    );
    let storage = storage(root);
    let order: Vec<String> = MODEL_ORDER.iter().map(|name| name.to_string()).collect();
    for part in &stored {
        let read = storage.read_part_for_detect(&location(root, part)).await.unwrap();
        if names(&read[0]).contains(&"layers".to_string()) {
            assert!(
                layer_keys(&read).iter().all(|keys| *keys == order),
                "{layout:?}: {part:?}"
            );
        }
    }
}

#[tokio::test]
async fn a_restart_over_a_store_of_protocol_4_parts_replays_every_part() {
    restart_over(Stored::Protocol4).await;
}

#[tokio::test]
async fn a_restart_over_a_store_of_protocol_5_parts_replays_every_part_in_entry_order() {
    restart_over(Stored::Protocol5).await;
}

#[tokio::test]
async fn a_restart_over_a_store_of_both_layouts_replays_every_part_by_its_columns() {
    restart_over(Stored::Both).await;
}
