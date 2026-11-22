//! Lane D of wave AF (Nov 21, 2026): the live report's trend memoization, through a server in this
//! process with two ranks.
//!
//! A report finds a series' trend (`gradient_trend`, and each module's of `attention_trends`) again
//! only when the series received a point since the last report. `Ingest::trend_computations` is
//! the count each test reads; nothing here is timed.
//!
//! 1. Rows for rank A compute nothing for rank B. A part for B that gives none of its series a point
//!    rewrites B's report and finds none of B's trends again.
//! 2. Rows for B that do give its series points: the trends found are those of `detect` on the same rows.
//! 3. The same across a restart of the server, and across a resume below the last step (a second
//!    attempt): each rebuilds B from the store, and the report is `detect`'s.

mod lane_aa3_support;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use lane_aa3_support::{json_file, open};
use object_store::local::LocalFileSystem;
use serde_json::{json, Value};
use tensorquill_server::detect::{detect_stream, Thresholds};
use tensorquill_server::ingest::{Ingest, IngestConfig};
use tensorquill_server::server::serve;
use tensorquill_server::storage::Storage;
use tokio::net::TcpListener;

const RUN: &str = "run-af-d-memo";

/// Rows of `rank` and `attempt` at `steps`. With `points`, each has a total norm and every fifth a
/// probe of two attention modules; without, each has neither (nothing a trend reads).
fn rows(rank: i32, attempt: i64, steps: std::ops::Range<i64>, points: bool, scale: f64) -> RecordBatch {
    let steps: Vec<i64> = steps.collect();
    let n = steps.len();
    let norms: Vec<Option<f64>> = steps
        .iter()
        .map(|step| points.then(|| scale * (1.0 + *step as f64 / 50.0) * (1.0 + ((step * 37) % 21) as f64 / 100.0)))
        .collect();
    let probes: Vec<Option<String>> = steps
        .iter()
        .map(|step| {
            (points && step % 5 == 0).then(|| {
                let x = *step as f64;
                json!({"attention": {"a.0": {"max_logit": [scale * (2.0 + x / 20.0)]},
                                     "a.1": {"max_logit": [scale * (3.0 + (step % 7) as f64)]}}})
                .to_string()
            })
        })
        .collect();
    RecordBatch::try_from_iter_with_nullable(vec![
        (
            "schema_version",
            Arc::new(Int32Array::from(vec![1; n])) as ArrayRef,
            false,
        ),
        ("run_id", Arc::new(StringArray::from(vec![RUN; n])) as ArrayRef, false),
        ("rank", Arc::new(Int32Array::from(vec![rank; n])) as ArrayRef, false),
        (
            "attempt",
            Arc::new(Int64Array::from(vec![attempt; n])) as ArrayRef,
            false,
        ),
        ("step", Arc::new(Int64Array::from(steps)) as ArrayRef, false),
        // Nullable whatever the rows hold, so every batch has one schema.
        ("grad_norm", Arc::new(Float64Array::from(norms)) as ArrayRef, true),
        ("probe_json", Arc::new(StringArray::from(probes)) as ArrayRef, true),
    ])
    .unwrap()
}

struct Live {
    address: String,
    ingest: Arc<Ingest>,
    stop: tokio::sync::oneshot::Sender<()>,
    served: tokio::task::JoinHandle<()>,
}

/// A server in this process over `root` that stores and judges every batch as it comes.
async fn start(root: &Path) -> Live {
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
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let serving = ingest.clone();
    let served = tokio::spawn(async move {
        serve(listener, serving, None, None, async {
            let _ = stopped.await;
        })
        .await
        .unwrap();
    });
    Live {
        address,
        ingest,
        stop,
        served,
    }
}

impl Live {
    /// Sends `batch` for its rank on a stream of its own, ending at byte `end`, and waits for its
    /// acknowledgement and then for its report (the count waits for the rank's buffer, which the
    /// report is made under).
    async fn send(&self, batch: RecordBatch, end: i64) -> u64 {
        let rank = batch
            .column_by_name("rank")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .value(0);
        let mut put = open(&self.address, RUN, rank, batch.schema()).await;
        put.send(batch, end).await;
        assert_eq!(put.ack().await, end);
        put.finish().await;
        self.ingest
            .trend_computations(RUN, rank)
            .await
            .expect("the rank has detectors")
    }

    async fn stop(self) {
        self.ingest.stop(std::iter::empty()).await.unwrap();
        let _ = self.stop.send(());
        let _ = self.served.await;
    }
}

fn report(root: &Path, rank: i32) -> Value {
    json_file(
        &root
            .join("runs")
            .join(RUN)
            .join(format!("rank={rank}"))
            .join("incidents.json"),
    )
}

/// `detect` on `batches`, as one Arrow stream.
fn offline(batches: &[RecordBatch]) -> Value {
    let mut stream = Vec::new();
    {
        let mut writer = arrow::ipc::writer::StreamWriter::try_new(&mut stream, &batches[0].schema()).unwrap();
        for batch in batches {
            writer.write(batch).unwrap();
        }
        writer.finish().unwrap();
    }
    detect_stream(stream.as_slice(), &Arc::new(Thresholds::default())).unwrap()
}

/// The live report and `detect`'s, without `rebuilds` (the key only a live server has: section 5).
fn same(live: &Value, offline: &Value) {
    let strip = |report: &Value| {
        let mut report = report.clone();
        report.as_object_mut().unwrap().remove("rebuilds");
        report
    };
    assert_eq!(strip(live), strip(offline));
    for block in ["gradient_trend", "attention_trends"] {
        assert!(live.get(block).is_some(), "{block} is in the report: {live}");
    }
}

#[tokio::test]
async fn an_unchanged_series_is_not_found_again_and_a_changed_one_is_detects_across_a_restart_and_a_resume() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let (a, b) = (0, 1);
    let mut b_rows = Vec::new();

    let live = start(root).await;
    live.send(rows(a, 1, 0..40, true, 1.0), 100).await;
    b_rows.push(rows(b, 1, 0..40, true, 2.0));
    let b_first = live.send(b_rows[0].clone(), 100).await;
    // The first report of B finds its three series: the norm and two modules.
    assert_eq!(b_first, 3);
    let b_report = report(root, b);

    // 1. Rows for A: nothing is computed for B, and B's report is as it was.
    let a_after = live.send(rows(a, 1, 40..80, true, 1.0), 200).await;
    assert_eq!(live.ingest.trend_computations(RUN, b).await, Some(b_first));
    assert_eq!(report(root, b), b_report);
    assert_eq!(a_after, 6, "A's three series grew and were found again");

    // A part for B with nothing a trend reads rewrites B's report and finds none of its trends again.
    b_rows.push(rows(b, 1, 40..60, false, 2.0));
    assert_eq!(live.send(b_rows[1].clone(), 200).await, b_first);
    let rewritten = report(root, b);
    assert_eq!(rewritten["last_step"], 59, "the report was rewritten");
    for block in ["gradient_trend", "attention_trends"] {
        assert_eq!(rewritten[block], b_report[block], "{block}");
    }
    same(&rewritten, &offline(&b_rows));

    // 2. Rows for B that give its series points: found again, and `detect`'s.
    b_rows.push(rows(b, 1, 60..100, true, 2.0));
    assert_eq!(live.send(b_rows[2].clone(), 300).await, b_first + 3);
    same(&report(root, b), &offline(&b_rows));
    live.stop().await;

    // 3. A restart: B's next part rebuilds it from the store, with every trend found afresh.
    let live = start(root).await;
    b_rows.push(rows(b, 1, 100..120, true, 2.0));
    assert_eq!(live.send(b_rows[3].clone(), 400).await, 3);
    let restarted = report(root, b);
    assert_eq!(restarted["rebuilds"], 1);
    same(&restarted, &offline(&b_rows));
    // Then a part with nothing a trend reads: nothing is found again after the rebuild either.
    b_rows.push(rows(b, 1, 120..130, false, 2.0));
    assert_eq!(live.send(b_rows[4].clone(), 500).await, 3);
    same(&report(root, b), &offline(&b_rows));

    // A resume below the last step (129): attempt 2 from step 70, with other values. It rebuilds,
    // and the report is `detect`'s on all the rows, attempt 1 then attempt 2.
    b_rows.push(rows(b, 2, 70..110, true, 3.0));
    assert_eq!(live.send(b_rows[5].clone(), 600).await, 3);
    let resumed = report(root, b);
    assert_eq!((&resumed["rebuilds"], &resumed["attempt"]), (&json!(2), &json!(2)));
    assert_eq!(resumed["gradient_trend"]["last_step"], 109);
    same(&resumed, &offline(&b_rows));
    live.stop().await;
}
