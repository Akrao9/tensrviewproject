//! Lane D of wave AF (Nov 21, 2026), section 5 of `WAVE_AF_CONTRACT.md` (RS-04): the offline
//! `tensorquill-server detect` against live ingest, on the same valid rows.
//!
//! The shared cases: rows of one attempt, and of two (a resume), in step order. They are sent to a
//! live server in this process, which stores and judges them and writes the rank's
//! `incidents.json`; and the same rows, as one Arrow IPC stream, are given to the real binary's
//! `detect` (and `detect --run`, whose `reports` hold the rank's report). The two reports must be
//! equal once the keys only a live server can have are removed (`LIVE_ONLY`). A key that differs and
//! is not on that list fails the test.

mod lane_aa3_support;

use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{
    ArrayRef, Float64Array, Float64Builder, Int32Array, Int64Array, MapBuilder, StringArray, StringBuilder,
};
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use lane_aa3_support::{json_file, open};
use object_store::local::LocalFileSystem;
use serde_json::Value;
use tensorquill_server::ingest::{Ingest, IngestConfig};
use tensorquill_server::server::serve;
use tensorquill_server::storage::Storage;
use tokio::net::TcpListener;

const RUN: &str = "run-af-d-parity";

/// The keys of a rank's report that only a live server can have, each with why. Every other key is
/// the same in both, value for value.
///
/// - `rebuilds`: how often the server built the rank's detectors from its store: when it first takes
///   the rank up (also on a fresh store), after a restart, a new attempt, or rows of an earlier
///   attempt arriving late. The offline modes read the rows in hand once and build from no store:
///   always 0. Live it counts the server's own history, not the rows: 1 for one attempt sent here,
///   2 for two (run on this branch).
const LIVE_ONLY: [&str; 1] = ["rebuilds"];

/// Rows of rank 0 for `attempt` at `steps`: a gradient norm of 1, `spike` times larger at
/// `spike_at`, and a training loss.
fn rows(attempt: i64, steps: std::ops::Range<i64>, spike_at: Option<i64>) -> RecordBatch {
    let steps: Vec<i64> = steps.collect();
    let n = steps.len();
    let norms: Vec<f64> = steps
        .iter()
        .map(|step| {
            if Some(*step) == spike_at {
                1e4
            } else {
                1.0 + (*step as f64) * 1e-3
            }
        })
        .collect();
    let mut scalars = MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
    for step in &steps {
        scalars.keys().append_value("train/loss");
        scalars.values().append_value(2.0 - (*step as f64) * 1e-3);
        scalars.append(true).unwrap();
    }
    RecordBatch::try_from_iter(vec![
        ("schema_version", Arc::new(Int32Array::from(vec![1; n])) as ArrayRef),
        ("run_id", Arc::new(StringArray::from(vec![RUN; n])) as ArrayRef),
        ("rank", Arc::new(Int32Array::from(vec![0; n])) as ArrayRef),
        ("attempt", Arc::new(Int64Array::from(vec![attempt; n])) as ArrayRef),
        ("step", Arc::new(Int64Array::from(steps)) as ArrayRef),
        ("grad_norm", Arc::new(Float64Array::from(norms)) as ArrayRef),
        ("scalars", Arc::new(scalars.finish()) as ArrayRef),
    ])
    .unwrap()
}

/// The live report of `batches`: sent in order, one batch per part, on one stream, to a server in
/// this process over `root`; then the server is stopped, which stores and judges what it holds.
async fn live(root: &Path, batches: &[RecordBatch]) -> Value {
    let config = IngestConfig {
        flush_steps: 1,
        flush_interval: Duration::from_secs(3600),
        watchdog_interval: Duration::ZERO,
        ..IngestConfig::default()
    };
    let store = LocalFileSystem::new_with_prefix(root).unwrap();
    let ingest = Ingest::new(Storage::new(Arc::new(store), "".into()), config);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let served = ingest.clone();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        serve(listener, served, None, None, async {
            let _ = stopped.await;
        })
        .await
        .unwrap();
    });
    let mut put = open(&address, RUN, 0, batches[0].schema()).await;
    let mut end = 0;
    for batch in batches {
        end += 100 * batch.num_rows() as i64;
        put.send(batch.clone(), end).await;
        // Each batch is stored and judged before it is acknowledged (`flush_steps` 1).
        assert_eq!(put.ack().await, end);
    }
    put.finish().await;
    ingest.stop(std::iter::empty()).await.unwrap();
    let _ = stop.send(());
    let _ = server.await;
    json_file(&root.join("runs").join(RUN).join("rank=0").join("incidents.json"))
}

/// The offline reports of `batches`: the real binary's `detect`, and the rank's report from
/// `detect --run`.
fn offline(batches: &[RecordBatch]) -> (Value, Value) {
    let schema: Arc<Schema> = batches[0].schema();
    let mut stream = Vec::new();
    {
        let mut writer = arrow::ipc::writer::StreamWriter::try_new(&mut stream, &schema).unwrap();
        for batch in batches {
            writer.write(batch).unwrap();
        }
        writer.finish().unwrap();
    }
    let run = |args: &[&str]| -> Value {
        use std::io::Write;
        let mut child = Command::new(env!("CARGO_BIN_EXE_tensorquill-server"))
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(&stream).unwrap();
        let done = child.wait_with_output().unwrap();
        assert!(done.status.success(), "{}", String::from_utf8_lossy(&done.stderr));
        serde_json::from_slice(&done.stdout).unwrap()
    };
    let one = run(&["detect"]);
    let all = run(&["detect", "--run"]);
    assert_eq!(all["reports"].as_array().unwrap().len(), 1, "{all}");
    (one, all["reports"][0].clone())
}

/// Every top-level key where the two reports differ, after `LIVE_ONLY` is removed from both.
fn differences(live: &Value, offline: &Value) -> Vec<String> {
    let (live, offline) = (live.as_object().unwrap(), offline.as_object().unwrap());
    let mut keys: Vec<&String> = live.keys().chain(offline.keys()).collect();
    keys.sort();
    keys.dedup();
    keys.into_iter()
        .filter(|key| !LIVE_ONLY.contains(&key.as_str()))
        .filter(|key| live.get(*key) != offline.get(*key))
        .map(|key| format!("{key}: live {:?}, offline {:?}", live.get(key), offline.get(key)))
        .collect()
}

async fn the_reports_agree(batches: Vec<RecordBatch>, rebuilds: u64) -> Value {
    // `rebuilds` is what this server did, which the test reads to show the key is the server's.
    let dir = tempfile::tempdir().unwrap();
    let live = live(dir.path(), &batches).await;
    let (one, run) = offline(&batches);
    assert_eq!(differences(&live, &one), Vec::<String>::new(), "detect against live");
    assert_eq!(
        differences(&live, &run),
        Vec::<String>::new(),
        "detect --run against live"
    );
    // What `LIVE_ONLY` says of each key it removes.
    assert_eq!(
        (&live["rebuilds"], &one["rebuilds"]),
        (&serde_json::json!(rebuilds), &serde_json::json!(0))
    );
    live
}

/// One attempt. Fails on the base (`1ca26d4`): the offline reports had no `attempt_order`.
#[tokio::test]
async fn one_attempt_in_step_order_reads_the_same_offline_and_live() {
    let live = the_reports_agree(vec![rows(1, 0..20, None), rows(1, 20..40, Some(30))], 1).await;
    assert_eq!(live["attempt_order"], "first_seen");
    assert_eq!(live["steps"], 40);
    let kinds: Vec<&str> = live["incidents"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|incident| incident["kind"].as_str())
        .collect();
    assert!(kinds.contains(&"grad_spike"), "the case judges something: {kinds:?}");
}

/// Two attempts, the second resuming at step 25 of the first's 40. Fails on the base: no
/// `attempt_order` offline.
#[tokio::test]
async fn two_attempts_in_step_order_read_the_same_offline_and_live() {
    let live = the_reports_agree(
        vec![rows(1, 0..20, None), rows(1, 20..40, None), rows(2, 25..60, Some(50))],
        2,
    )
    .await;
    assert_eq!(
        (&live["attempt_order"], &live["attempt"]),
        (&serde_json::json!("first_seen"), &serde_json::json!(2))
    );
    assert_eq!(live["last_step"], 59);
}
