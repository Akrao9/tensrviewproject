//! Lane D of wave AF (Nov 21, 2026), section 4 of `WAVE_AF_CONTRACT.md`, the second half of RS-01:
//! a status that only gained a refused batch, a row that was not newer, or a step confirmed late is
//! written at once, by the path the other counts take (`Ingest::write_statuses` and the values it
//! compares), not at the next periodic write.
//!
//! Each test runs a real server in this process whose watchdog passes often (every 50 ms) and whose
//! periodic rewrite of an unchanged status (`status_refresh`) is an hour away. The status is first
//! written with the rank's progress; then the one event happens, with the rank's stream held open so
//! its state and last step do not move. The count can only reach `watchdog.json` by a write that
//! compares it: nothing else in the status changed, and the periodic write is an hour off.

mod lane_aa3_support;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{ArrayRef, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use lane_aa3_support::{json_file, open};
use object_store::local::LocalFileSystem;
use tensorquill_server::ingest::{Ingest, IngestConfig};
use tensorquill_server::server::serve;
use tensorquill_server::storage::Storage;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

const RUN: &str = "run-af-d-status";
/// The periodic rewrite of an unchanged status: far beyond any wait here.
const REFRESH: Duration = Duration::from_secs(3600);
/// How long a test waits for a write; only a failing test waits this long.
const BOUND: Duration = Duration::from_secs(60);

struct TestServer {
    address: String,
    ingest: Arc<Ingest>,
    _stop: oneshot::Sender<()>,
}

fn schema(step: DataType) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("schema_version", DataType::Int32, false),
        Field::new("run_id", DataType::Utf8, false),
        Field::new("rank", DataType::Int32, false),
        Field::new("attempt", DataType::Int64, false),
        Field::new("step", step, false),
    ]))
}

/// Rows of rank 0, attempt 1, at `steps`.
fn rows(steps: &[i64]) -> RecordBatch {
    let n = steps.len();
    RecordBatch::try_new(
        schema(DataType::Int64),
        vec![
            Arc::new(Int32Array::from(vec![1; n])) as ArrayRef,
            Arc::new(StringArray::from(vec![RUN; n])),
            Arc::new(Int32Array::from(vec![0; n])),
            Arc::new(Int64Array::from(vec![1; n])),
            Arc::new(Int64Array::from(steps.to_vec())),
        ],
    )
    .unwrap()
}

/// A batch the server refuses for good: its `step` is text.
fn text_steps(steps: &[i64]) -> RecordBatch {
    let n = steps.len();
    let text: Vec<String> = steps.iter().map(|step| step.to_string()).collect();
    RecordBatch::try_new(
        schema(DataType::Utf8),
        vec![
            Arc::new(Int32Array::from(vec![1; n])) as ArrayRef,
            Arc::new(StringArray::from(vec![RUN; n])),
            Arc::new(Int32Array::from(vec![0; n])),
            Arc::new(Int64Array::from(vec![1; n])),
            Arc::new(StringArray::from(text)),
        ],
    )
    .unwrap()
}

/// A server in this process over `root` whose watchdog writes the status every 50 ms when it
/// changed, and rewrites an unchanged one only after `REFRESH`. Without `flusher`, nothing stores a
/// buffered part on time: the test does, with `Ingest::flush_due`.
async fn start(root: &Path, flush_steps: usize, flush_interval: Duration, flusher: bool) -> TestServer {
    let config = IngestConfig {
        flush_steps,
        flush_interval,
        detect: false,
        watchdog_interval: Duration::from_millis(50),
        status_refresh: REFRESH,
        silent_floor: Duration::from_secs(3600),
        ..IngestConfig::default()
    };
    let store = LocalFileSystem::new_with_prefix(root).unwrap();
    let ingest = Ingest::new(Storage::new(Arc::new(store), "".into()), config);
    if flusher {
        ingest.spawn_flusher();
    }
    ingest.spawn_watchdog();
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
    TestServer {
        address,
        ingest,
        _stop: stop,
    }
}

/// Rank 0's entry in the `watchdog.json` on disk, once there is one.
fn entry(root: &Path) -> Option<serde_json::Value> {
    let path = root.join("runs").join(RUN).join("watchdog.json");
    if !path.exists() {
        return None;
    }
    let status = json_file(&path);
    status["ranks"]
        .as_array()?
        .iter()
        .find(|rank| rank["rank"] == 0)
        .cloned()
}

/// Waits until rank 0's entry on disk satisfies `ready`, and returns it. Fails after `BOUND`.
async fn until(root: &Path, what: &str, ready: impl Fn(&serde_json::Value) -> bool) -> serde_json::Value {
    let started = std::time::Instant::now();
    loop {
        if let Some(entry) = entry(root).filter(|entry| ready(entry)) {
            return entry;
        }
        assert!(
            started.elapsed() < BOUND,
            "watchdog.json did not show {what} within {BOUND:?}; the rank's entry on disk: {:?}",
            entry(root)
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// What the test knows of the run: the status is written with the rank at `last`, its stream open.
async fn progress_written(root: &Path, last: i64) -> serde_json::Value {
    until(root, &format!("last_step {last} on an open stream"), |entry| {
        entry["last_step"] == last && entry["stream"] == "open"
    })
    .await
}

/// A refused batch, alone: the status gains `refused_batches` and `refused_batch_keys` at once.
/// The refused stream is a second one of the rank; the first stays open, so the rank's stream state
/// and progress are those already written. Fails on the base (`1ca26d4`): nothing the status
/// compares changed, so the count waited for the periodic write an hour away.
#[tokio::test]
async fn a_refused_batch_alone_reaches_the_status_before_the_periodic_write() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), 1, Duration::from_secs(3600), true).await;
    let mut held = open(&server.address, RUN, 0, schema(DataType::Int64)).await;
    held.send(rows(&[0, 1, 2]), 300).await;
    assert_eq!(held.ack().await, 300);
    let before = progress_written(dir.path(), 2).await;
    assert!(before.get("refused_batches").is_none(), "{before}");

    let mut refused = open(&server.address, RUN, 0, schema(DataType::Utf8)).await;
    refused.send(text_steps(&[3, 4]), 500).await;
    let answer = refused.reply().await.expect("the refusal before the status");
    assert_eq!(answer["refused"]["column"], "step", "{answer}");
    refused.reply().await.expect_err("the stream ends refused");
    // The server's own view has the count at once (as on the base).
    let live = server.ingest.watchdog_statuses();
    assert_eq!(
        live[RUN]["ranks"][0]["refused_batch_keys"],
        serde_json::json!({"step": 1})
    );

    let after = until(dir.path(), "refused_batch_keys", |entry| {
        entry.get("refused_batch_keys").is_some()
    })
    .await;
    assert_eq!(
        (&after["refused_batches"], &after["refused_batch_keys"]),
        (&serde_json::json!(1), &serde_json::json!({"step": 1}))
    );
    // Nothing else of the rank moved: the write was the count's.
    assert_eq!(
        (&after["last_step"], &after["stream"], &after["state"]),
        (&before["last_step"], &before["stream"], &before["state"])
    );
    drop(held);
}

/// A row that is not newer than what its attempt stored, alone: the status gains `rows_not_newer`
/// at once. Sent on the rank's open stream; the row is not stored and moves no progress. Fails on
/// the base.
#[tokio::test]
async fn a_row_not_newer_alone_reaches_the_status_before_the_periodic_write() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), 1, Duration::from_secs(3600), true).await;
    let mut put = open(&server.address, RUN, 0, schema(DataType::Int64)).await;
    put.send(rows(&[0, 1, 2]), 300).await;
    assert_eq!(put.ack().await, 300);
    let before = progress_written(dir.path(), 2).await;
    assert!(before.get("rows_not_newer").is_none(), "{before}");

    put.send(rows(&[1]), 400).await;
    assert_eq!(put.ack().await, 400);
    let after = until(dir.path(), "rows_not_newer", |entry| {
        entry.get("rows_not_newer").is_some()
    })
    .await;
    assert_eq!(after["rows_not_newer"], 1, "{after}");
    assert!(
        after.get("late_steps_confirmed").is_none(),
        "a replay of a stored step is not late: {after}"
    );
    assert_eq!(
        (&after["last_step"], &after["stream"], &after["state"]),
        (&before["last_step"], &before["stream"], &before["state"])
    );
    drop(put);
}

/// A step confirmed late inside the buffered part, alone: once the part is stored, the status gains
/// `late_steps_confirmed` at once. The late row's own arrival already counted `rows_not_newer`; the
/// part is stored only when the test calls `flush_due`, after that count is on disk, so what is
/// written after it is the late count alone, however loaded the machine. Fails on the base.
#[tokio::test]
async fn a_step_confirmed_late_alone_reaches_the_status_before_the_periodic_write() {
    let dir = tempfile::tempdir().unwrap();
    // The part stays open until the test stores it: the late step is behind a row of it.
    let server = start(dir.path(), 1000, Duration::ZERO, false).await;
    let mut put = open(&server.address, RUN, 0, schema(DataType::Int64)).await;
    put.send(rows(&[0, 2]), 200).await;
    put.send(rows(&[1]), 300).await;
    let counted = until(dir.path(), "rows_not_newer", |entry| {
        entry.get("rows_not_newer").is_some()
    })
    .await;
    assert_eq!(counted["last_step"], 2, "{counted}");
    assert!(
        counted.get("late_steps_confirmed").is_none(),
        "the part is not stored yet: {counted}"
    );
    server.ingest.flush_due().await;
    let after = until(dir.path(), "late_steps_confirmed", |entry| {
        entry.get("late_steps_confirmed").is_some()
    })
    .await;
    assert_eq!(
        (&after["late_steps_confirmed"], &after["rows_not_newer"]),
        (&serde_json::json!(1), &serde_json::json!(1))
    );
    assert_eq!(
        (&after["last_step"], &after["stream"], &after["state"]),
        (&counted["last_step"], &counted["stream"], &counted["state"])
    );
    drop(put);
}
