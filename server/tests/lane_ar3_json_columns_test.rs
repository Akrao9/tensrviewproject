//! Lane AR-3 (Nov 26, 2026), AQ5-D2 on the server's own routes: rows whose `system_json`,
//! `data_json`, `probe_json` and `checkpoint_json` hold an integer too large for a float, sent as
//! the SDK sends them.
//!
//! - The stored part holds each column's text as it was sent, the integer's digits included: the
//!   server stores the batch as received and does not parse the JSON columns.
//! - The live detectors read each row with its valid siblings, and the oversized value is
//!   unreadable where a number is wanted (`detect/extra_json.rs`).
//! - After a restart, the detectors rebuilt from the stored parts read the same.

mod lane_aa3_support;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Array, ArrayRef, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use lane_aa3_support::{json_file, open, start};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{json, Value};

const RUN: &str = "ar3";
const ATTEMPT: i64 = 1_732_600_000_000;
const COLUMNS: [&str; 4] = ["system_json", "data_json", "probe_json", "checkpoint_json"];

fn big() -> String {
    format!("1{}", "0".repeat(400))
}

fn schema() -> Arc<Schema> {
    let mut fields = vec![
        Field::new("schema_version", DataType::Int32, false),
        Field::new("run_id", DataType::Utf8, false),
        Field::new("rank", DataType::Int32, false),
        Field::new("attempt", DataType::Int64, false),
        Field::new("step", DataType::Int64, false),
        Field::new("ts", DataType::Float64, true),
    ];
    fields.extend(COLUMNS.iter().map(|name| Field::new(*name, DataType::Utf8, true)));
    Arc::new(Schema::new(fields))
}

/// Each column's text at `step`: a stalled loader's batch beside an oversized epoch, a cuda reading
/// with one oversized count beside a valid one, a probe whose logits hold an oversized member, and on
/// the first step a save manifest with an oversized step.
fn texts(step: i64) -> [Option<String>; 4] {
    let big = big();
    [
        Some(format!(r#"{{"cuda": {{"ooms": {big}, "alloc_retries": 3}}}}"#)),
        Some(format!(
            r#"{{"batches": 1, "batch_hash": "v2:aa", "sampler_epoch": {big}}}"#
        )),
        Some(format!(
            r#"{{"attention": {{"blk1": {{"entropy_share": [0.5, 0.5], "max_logit": [3.9, -{big}, 3.96]}}}}}}"#
        )),
        (step == 0).then(|| format!(r#"{{"save": {{"after_step": {big}}}}}"#)),
    ]
}

fn batch(steps: std::ops::Range<i64>) -> RecordBatch {
    let steps: Vec<i64> = steps.collect();
    let n = steps.len();
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(vec![1; n])),
        Arc::new(StringArray::from(vec![RUN; n])),
        Arc::new(Int32Array::from(vec![0; n])),
        Arc::new(Int64Array::from(vec![ATTEMPT; n])),
        Arc::new(Int64Array::from(steps.clone())),
        Arc::new(Float64Array::from(
            steps.iter().map(|step| *step as f64).collect::<Vec<_>>(),
        )),
    ];
    for at in 0..COLUMNS.len() {
        let column: Vec<Option<String>> = steps.iter().map(|step| texts(*step)[at].clone()).collect();
        columns.push(Arc::new(StringArray::from(column)));
    }
    RecordBatch::try_new(schema(), columns).unwrap()
}

/// The rank's `incidents.json` once it covers `last_step`.
async fn report_through(root: &Path, last_step: i64) -> Value {
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

fn parts(root: &Path) -> Vec<PathBuf> {
    let dir = root
        .join("runs")
        .join(RUN)
        .join("rank=0")
        .join(format!("attempt={ATTEMPT}"));
    let mut parts: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| entries.map(|entry| entry.unwrap().path()).collect())
        .unwrap_or_default();
    parts.retain(|path| path.extension().is_some_and(|extension| extension == "parquet"));
    parts.sort();
    parts
}

/// Every stored text of `column`, in stored order.
fn stored(root: &Path, column: &str) -> Vec<Option<String>> {
    let mut out = Vec::new();
    for part in parts(root) {
        let file = std::fs::File::open(part).unwrap();
        for batch in ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap() {
            let batch = batch.unwrap();
            let texts = batch.column_by_name(column).unwrap();
            let texts = texts.as_any().downcast_ref::<StringArray>().unwrap();
            out.extend((0..texts.len()).map(|row| texts.is_valid(row).then(|| texts.value(row).to_string())));
        }
    }
    out
}

/// What the report must hold, live or rebuilt: every row's columns read, with their siblings.
fn assert_read(report: &Value, rows: u64) {
    let checks = &report["kind_checks"];
    for kind in [
        "memory_growth",
        "memory_pressure",
        "allocator_retries",
        "data_stalled",
        "epoch_order_repeated",
        "attention_collapse",
        "attention_uniform",
        "attention_logit_growth",
    ] {
        assert_eq!(
            (&checks[kind]["value"], &checks[kind]["records"]),
            (&json!("checked"), &json!(rows)),
            "{kind}: {report}"
        );
    }
    assert_eq!(checks["resume_incomplete"]["records"], json!(1), "{report}");
    let incident = |kind: &str| {
        report["incidents"]
            .as_array()
            .unwrap()
            .iter()
            .find(|incident| incident["kind"] == kind)
            .cloned()
            .unwrap_or_else(|| panic!("no {kind}: {report}"))
    };
    let retries = incident("allocator_retries");
    assert_eq!(
        (&retries["step"], &retries["alloc_retries"], &retries["ooms"]),
        (&json!(0), &json!(3), &Value::Null)
    );
    let stalled = incident("data_stalled");
    assert_eq!(
        (&stalled["step"], &stalled["batch_hash"]),
        (&json!(1), &json!("v2:aa")),
        "{stalled}"
    );
    let trend = &report["attention_trends"]["listed"][0];
    assert_eq!(
        (&trend["module"], &trend["first"]),
        (&json!("blk1"), &json!(3.96)),
        "{report}"
    );
}

#[tokio::test]
async fn d2_oversized_integers_in_the_json_columns_are_stored_as_sent_and_read_live_and_after_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    {
        let server = start(root).await;
        let mut put = open(&server.address, RUN, 0, schema()).await;
        put.send(batch(0..6), 100).await;
        assert_eq!(put.ack().await, 100);
        put.finish().await;
        assert_read(&report_through(root, 5).await, 6);
        server.ingest.shutdown().await.unwrap();
    }
    // The stored rows hold the text as sent: the integer's 401 digits, not a float or a placeholder.
    for (at, column) in COLUMNS.iter().enumerate() {
        let expected: Vec<Option<String>> = (0..6).map(|step| texts(step)[at].clone()).collect();
        assert_eq!(stored(root, column), expected, "{column}");
    }
    assert!(stored(root, "system_json")[0].as_deref().unwrap().contains(&big()));

    // A restart: the detectors are rebuilt from the stored parts, then read the new rows.
    let server = start(root).await;
    let mut put = open(&server.address, RUN, 0, schema()).await;
    put.send(batch(6..8), 200).await;
    assert_eq!(put.ack().await, 200);
    put.finish().await;
    let report = report_through(root, 7).await;
    assert_eq!(report["rebuilds"], 1, "{report}");
    assert_read(&report, 8);
}
