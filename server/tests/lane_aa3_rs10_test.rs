//! Lane AA3 of wave AA (Nov 20, 2026), RS10 as the author decided it: a batch whose required key
//! column is absent or of another type is refused and counted, with the column named; where the
//! column is sound, a row whose `schema_version` is not one this server reads (0, a record that
//! named none, as `flight.py` writes it, and 1) is dropped and counted, and the batch's valid rows
//! are stored and acknowledged.

mod lane_aa3_support;

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use lane_aa3_support::{open, start, stored_steps};
use tonic::Code;

const RUN: &str = "run-rs10";

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("schema_version", DataType::Int32, false),
        Field::new("run_id", DataType::Utf8, false),
        Field::new("rank", DataType::Int32, false),
        Field::new("attempt", DataType::Int64, false),
        Field::new("step", DataType::Int64, false),
        Field::new("ts", DataType::Float64, true),
    ]))
}

/// Rows of rank 0, attempt 1: `(schema_version, step)`.
fn rows(rows: &[(i32, i64)]) -> RecordBatch {
    let n = rows.len();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(rows.iter().map(|row| row.0).collect::<Vec<_>>())),
            Arc::new(StringArray::from(vec![RUN; n])),
            Arc::new(Int32Array::from(vec![0; n])),
            Arc::new(Int64Array::from(vec![1; n])),
            Arc::new(Int64Array::from(rows.iter().map(|row| row.1).collect::<Vec<_>>())),
            Arc::new(Float64Array::from(vec![0.5; n])),
        ],
    )
    .unwrap()
}

/// Rank 0's entry in the run's status, as `watchdog.json` carries it.
fn rank_entry(server: &lane_aa3_support::TestServer) -> serde_json::Value {
    let statuses = server.ingest.watchdog_statuses();
    statuses[RUN]["ranks"][0].clone()
}

/// The rows of versions 0 and 1 are stored and the batch is acknowledged; the rows of any other
/// version are dropped and counted in `dropped_rows` and `dropped_row_keys`, under their own key.
/// Until Nov 20, 2026 the value was not looked at: every row was stored, whatever it named. Version 2
/// is protocol 5's row (wave AL, stored from then on), so the dropped rows here are of version 3.
#[tokio::test]
async fn rows_of_an_unsupported_schema_version_are_dropped_and_the_rest_of_the_batch_is_stored() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    let mut put = open(&server.address, RUN, 0, schema()).await;
    put.send(rows(&[(1, 0), (0, 1), (3, 2), (1, 3), (-1, 4), (99, 5), (0, 6)]), 100)
        .await;
    assert_eq!(put.ack().await, 100, "the batch's valid rows are acknowledged");
    // A batch whose every row is of another version is acknowledged too: nothing of it can be
    // stored, and its sender must not send it forever.
    put.send(rows(&[(3, 7)]), 200).await;
    assert_eq!(put.ack().await, 200);
    // The stream goes on.
    put.send(rows(&[(1, 8)]), 300).await;
    assert_eq!(put.ack().await, 300);
    let entry = rank_entry(&server);
    put.finish().await;
    assert_eq!(stored_steps(dir.path(), RUN, 0), vec![0, 1, 3, 6, 8]);
    assert_eq!(entry["dropped_rows"], 4, "{entry}");
    assert_eq!(
        entry["dropped_row_keys"],
        serde_json::json!({"unsupported_schema_version": 4}),
        "{entry}"
    );
    assert_eq!(
        server.ingest.dropped_rows(),
        std::collections::BTreeMap::from([("unsupported_schema_version".to_string(), 4)])
    );
}

/// A batch whose `step` column is missing, or whose `attempt` is text: no row of it can be read. It
/// is refused, its stream ends with `INVALID_ARGUMENT` naming the column, and the refusal is counted
/// in the rank's status (`refused_batches`, `refused_batch_keys`). The same batch sent again (the
/// sender sends from its offset after the stream ended) is not counted again. Until Nov 20, 2026
/// a refusal reached only the stream and stderr.
#[tokio::test]
async fn a_batch_with_a_missing_or_mistyped_key_column_is_refused_and_counted_naming_the_column() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    let mut put = open(&server.address, RUN, 0, schema()).await;
    put.send(rows(&[(1, 0), (1, 1)]), 100).await;
    assert_eq!(put.ack().await, 100);
    put.finish().await;

    let no_step = Arc::new(Schema::new(vec![
        Field::new("schema_version", DataType::Int32, false),
        Field::new("run_id", DataType::Utf8, false),
        Field::new("rank", DataType::Int32, false),
        Field::new("attempt", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        no_step.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1])) as ArrayRef,
            Arc::new(StringArray::from(vec![RUN])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Int64Array::from(vec![1])),
        ],
    )
    .unwrap();
    for _ in 0..2 {
        let mut put = open(&server.address, RUN, 0, no_step.clone()).await;
        put.send(batch.clone(), 200).await;
        // RS10's transport (DETECTOR_RULES_CONTRACT.md part 8): the refused bytes are named first.
        let refusal = put.reply().await.expect("the refusal before the status");
        assert_eq!(
            refusal["refused"],
            serde_json::json!({"start_byte_offset": 100, "end_byte_offset": 200, "column": "step"})
        );
        let refused = put.reply().await.expect_err("refused");
        assert_eq!(refused.code(), Code::InvalidArgument);
        assert!(refused.message().contains("`step`"), "{}", refused.message());
    }
    let entry = rank_entry(&server);
    assert_eq!(entry["refused_batches"], 1, "{entry}");
    assert_eq!(entry["refused_batch_keys"], serde_json::json!({"step": 1}), "{entry}");

    let text_attempt = Arc::new(Schema::new(vec![
        Field::new("schema_version", DataType::Int32, false),
        Field::new("run_id", DataType::Utf8, false),
        Field::new("rank", DataType::Int32, false),
        Field::new("attempt", DataType::Utf8, false),
        Field::new("step", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        text_attempt.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1])) as ArrayRef,
            Arc::new(StringArray::from(vec![RUN])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(StringArray::from(vec!["1"])),
            Arc::new(Int64Array::from(vec![5])),
        ],
    )
    .unwrap();
    let mut put = open(&server.address, RUN, 0, text_attempt).await;
    put.send(batch, 300).await;
    let refusal = put.reply().await.expect("the refusal before the status");
    assert_eq!(
        refusal["refused"],
        serde_json::json!({"start_byte_offset": 100, "end_byte_offset": 300, "column": "attempt"})
    );
    let refused = put.reply().await.expect_err("refused");
    assert_eq!(refused.code(), Code::InvalidArgument);
    assert!(refused.message().contains("`attempt`"), "{}", refused.message());
    let entry = rank_entry(&server);
    assert_eq!(entry["refused_batches"], 2, "{entry}");
    assert_eq!(
        entry["refused_batch_keys"],
        serde_json::json!({"attempt": 1, "step": 1}),
        "{entry}"
    );
    // Nothing of the refused batches was stored, and the rank takes well-formed batches after them.
    let mut put = open(&server.address, RUN, 0, schema()).await;
    put.send(rows(&[(1, 2)]), 400).await;
    assert_eq!(put.ack().await, 400);
    put.finish().await;
    assert_eq!(stored_steps(dir.path(), RUN, 0), vec![0, 1, 2]);
}

/// The refusal count survives a restart, as `dropped_rows` does: it is carried in `restart`.
#[tokio::test]
async fn the_refusal_count_survives_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let server = start(dir.path()).await;
        let mut put = open(&server.address, RUN, 0, schema()).await;
        put.send(rows(&[(1, 0)]), 100).await;
        assert_eq!(put.ack().await, 100);
        put.finish().await;
        let no_rank = Arc::new(Schema::new(vec![
            Field::new("schema_version", DataType::Int32, false),
            Field::new("run_id", DataType::Utf8, false),
            Field::new("step", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            no_rank.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1])) as ArrayRef,
                Arc::new(StringArray::from(vec![RUN])),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        let mut put = open(&server.address, RUN, 0, no_rank).await;
        put.send(batch, 200).await;
        let refusal = put.reply().await.expect("the refusal before the status");
        assert_eq!(
            refusal["refused"],
            serde_json::json!({"start_byte_offset": 100, "end_byte_offset": 200, "column": "rank"})
        );
        assert_eq!(put.reply().await.expect_err("refused").code(), Code::InvalidArgument);
        server.ingest.write_watchdog().await;
    }
    let server = start(dir.path()).await;
    server.ingest.restore_watchdog().await.unwrap();
    let entry = rank_entry(&server);
    assert_eq!(entry["refused_batches"], 1, "{entry}");
    assert_eq!(entry["refused_batch_keys"], serde_json::json!({"rank": 1}), "{entry}");
}
