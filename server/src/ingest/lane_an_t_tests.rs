//! Lane AN-T (SV10 a, the lock rule of `WAVE_AN_CONTRACT.md` section 11): a panic while any of
//! `Ingest`'s standard mutexes is held poisons it, and the server still serves the next request.
//!
//! Each test poisons one mutex the way a panic under it would (a thread that holds it panics), then
//! makes the requests that take that mutex: a `DoPut` stream with a run manifest, rows and one row
//! with no step, ended as `close()` ends it; the readers (`dropped_rows`, `runs_held`,
//! `watchdog_statuses`, `trend_computations`); the watchdog's pass, which writes the run's status
//! and lets go of settled runs; a timed flush; the clean-up of a run nothing holds, which takes every
//! map; the end of a stream that named no run; and a stream of another run. The poisoning is done
//! from the test, not from a panic inside product code: no input reaches a panic under these locks,
//! and adding one would change product code.
//!
//! On the base (`ddcb0a0`), before the helper `crate::lock::locked`, the ten mutexes taken with
//! `.lock().unwrap()` somewhere (`written`, `data_windows`, `replicas`, `reports`, `stored`,
//! `manifests`, `dropped_rows`, `dropped_log`, `settled`, `order_locks`) failed these tests;
//! the four already taken through `PoisonError::into_inner` (`buffers`, `error_log`, `end_log`,
//! `streams`) passed, and are held here so that they stay so.

use std::io::Cursor;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::record_batch::RecordBatch;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::{FlightClient, FlightData, FlightDescriptor, PutResult};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use object_store::local::LocalFileSystem;
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Channel;
use tonic::Status;

use super::{Ingest, IngestConfig, StreamTally};
use crate::server::serve;
use crate::storage::Storage;

/// The longest a stream's replies are waited for.
const BOUND: Duration = Duration::from_secs(20);

/// What the panic of each test says.
const MESSAGE: &str = "a panic while the lock was held, from a lane AN-T test";

/// Poisons `mutex` as a panic under it does: a thread takes it and panics while it holds it.
fn poison<T: Send>(mutex: &Mutex<T>) {
    std::thread::scope(|scope| {
        let holder = scope.spawn(|| {
            let _held = mutex.lock();
            panic!("{MESSAGE}");
        });
        assert!(holder.join().is_err(), "the holding thread did not panic");
    });
    assert!(mutex.is_poisoned(), "the mutex was not poisoned");
}

// --- protocol 5's rows -----------------------------------------------------------------------------

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

/// Rows of rank 0 for `steps`, and one more row with no step, which is dropped and counted.
fn batch(run: &str, steps: std::ops::Range<i64>) -> RecordBatch {
    let mut rows: Vec<Value> = steps
        .map(|step| {
            json!({"schema_version": 2, "run_id": run, "rank": 0, "attempt": 1, "step": step,
                   "ts": step as f64, "grad_norm": 1.0, "record_version": 1})
        })
        .collect();
    rows.push(
        json!({"schema_version": 2, "run_id": run, "rank": 0, "attempt": 1, "step": null,
                     "ts": 0.0, "grad_norm": 1.0, "record_version": 1}),
    );
    let text: String = rows.iter().map(|row| format!("{row}\n")).collect();
    let mut reader = arrow::json::ReaderBuilder::new(schema())
        .with_batch_size(64)
        .build(Cursor::new(text))
        .unwrap();
    reader.next().unwrap().unwrap()
}

/// The record batch message of `steps`, with the `app_metadata` the SDK's sender gives a batch
/// ending at `end_byte_offset`, a world of one rank and a run manifest.
async fn messages(run: &str, steps: std::ops::Range<i64>, end_byte_offset: i64) -> Vec<FlightData> {
    let metadata = json!({"schema_version": 2, "run_id": run, "rank": 0, "min_step": null,
                          "max_step": null, "end_byte_offset": end_byte_offset, "sender_attempt": 1,
                          "world_size": 1, "run_manifest": {"run_id": run, "project": "lane_an_t"}})
    .to_string();
    let encoded: Vec<FlightData> = FlightDataEncoderBuilder::new()
        .with_max_flight_data_size(usize::MAX)
        .build(futures::stream::iter([Ok(batch(run, steps))]))
        .try_collect()
        .await
        .unwrap();
    encoded
        .into_iter()
        .skip(1)
        .map(|mut message| {
            message.app_metadata = metadata.clone().into();
            message
        })
        .collect()
}

// --- a server in this process ----------------------------------------------------------------------

struct Server {
    address: String,
    ingest: Arc<Ingest>,
    _stop: oneshot::Sender<()>,
}

/// A batch stays in the buffer until its stream ends, and a settled run is let go of at the first
/// pass after it settles.
async fn start(root: &Path) -> Server {
    let store = Arc::new(LocalFileSystem::new_with_prefix(root).unwrap());
    let config = IngestConfig {
        flush_steps: 100_000,
        flush_interval: Duration::from_secs(3600),
        evict_after: Duration::ZERO,
        ..IngestConfig::default()
    };
    let ingest = Ingest::new(Storage::new(store, "".into()), config);
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
    Server {
        address,
        ingest,
        _stop: stop,
    }
}

/// Opens a stream of `run`'s rank 0, sends `steps` and ends the stream as `close()` does; the
/// replies until the server ends it.
async fn put(
    address: &str,
    run: &str,
    steps: std::ops::Range<i64>,
    end_byte_offset: i64,
) -> Vec<Result<i64, tonic::Status>> {
    let channel = Channel::from_shared(address.to_string())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightClient::new(channel);
    let (sender, outgoing) = tokio::sync::mpsc::unbounded_channel();
    let descriptor = FlightDescriptor::new_path(vec!["runs".into(), run.into(), "rank=0".into()]);
    let opening: Vec<FlightData> = FlightDataEncoderBuilder::new()
        .with_flight_descriptor(Some(descriptor))
        .with_schema(schema())
        .build(futures::stream::empty())
        .try_collect()
        .await
        .unwrap();
    for message in opening.into_iter().chain(messages(run, steps, end_byte_offset).await) {
        sender.send(Ok(message)).unwrap();
    }
    drop(sender);
    let mut acks: BoxStream<'static, Result<PutResult, FlightError>> =
        match client.do_put(UnboundedReceiverStream::new(outgoing)).await {
            Ok(acks) => acks,
            Err(error) => futures::stream::once(async move { Err(error) }).boxed(),
        };
    let mut replies = Vec::new();
    tokio::time::timeout(BOUND, async {
        while let Some(reply) = acks.next().await {
            replies.push(match reply {
                Ok(result) => {
                    let reply: Value = serde_json::from_slice(&result.app_metadata).unwrap();
                    Ok(reply["end_byte_offset"].as_i64().unwrap())
                }
                Err(FlightError::Tonic(status)) => Err(*status),
                Err(other) => panic!("{other}"),
            });
        }
    })
    .await
    .expect("the stream never ended");
    replies
}

/// The steps of every row stored for rank 0 of `run`, read from its part files, in order.
fn stored_steps(root: &Path, run: &str) -> Vec<i64> {
    let mut steps = Vec::new();
    let mut folders = vec![root.join("runs").join(run).join("rank=0")];
    while let Some(folder) = folders.pop() {
        for entry in std::fs::read_dir(&folder).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                folders.push(path);
            } else if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("part-") && name.ends_with(".parquet"))
            {
                let file = std::fs::File::open(&path).unwrap();
                let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
                    .unwrap()
                    .build()
                    .unwrap();
                for batch in reader {
                    let batch = batch.unwrap();
                    let column = batch.column_by_name("step").unwrap();
                    let column = column.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
                    steps.extend(column.values().iter().copied());
                }
            }
        }
    }
    steps.sort_unstable();
    steps
}

/// The requests that take every one of `Ingest`'s standard mutexes, each answered as on a server
/// where nothing panicked. `lock` names the mutex poisoned, for the messages.
async fn serves(server: &Server, root: &Path, lock: &str) {
    let run = format!("an_t_{lock}");
    let replies = put(&server.address, &run, 0..3, 10).await;
    assert!(
        matches!(replies.as_slice(), [Ok(10)]),
        "{lock} poisoned: the stream was answered {replies:?}"
    );
    assert_eq!(stored_steps(root, &run), [0, 1, 2], "{lock} poisoned");
    assert!(
        root.join("runs").join(&run).join("run.json").is_file(),
        "{lock} poisoned: the run manifest was not stored"
    );
    assert_eq!(
        server.ingest.dropped_rows().values().sum::<u64>(),
        1,
        "{lock} poisoned: the row with no step was not counted"
    );
    assert!(server.ingest.runs_held() >= 1, "{lock} poisoned");
    assert!(
        server.ingest.watchdog_statuses().contains_key(&run),
        "{lock} poisoned: the run has no status"
    );
    let _ = server.ingest.trend_computations(&run, 0).await;
    // The watchdog's pass: the run's status is written, and the run, finished, is let go of.
    server.ingest.write_watchdog().await;
    assert!(
        root.join("runs").join(&run).join("watchdog.json").is_file(),
        "{lock} poisoned: the run's status was not written"
    );
    server.ingest.flush_due().await;
    // A run nothing holds is let go of at once: every map is taken.
    assert!(
        server.ingest.evict_run("an_t_held_by_none", Instant::now()),
        "{lock} poisoned"
    );
    // The end of a stream that named no run.
    server.ingest.stream_closed(
        None,
        Some(&Status::invalid_argument("lane AN-T: a stream that named no run")),
        &StreamTally::default(),
        Duration::ZERO,
        true,
    );
    // And the next stream, of another run, is taken and stored.
    let next = format!("{run}_next");
    let replies = put(&server.address, &next, 0..2, 20).await;
    assert!(
        matches!(replies.as_slice(), [Ok(20)]),
        "{lock} poisoned: the next stream was answered {replies:?}"
    );
    assert_eq!(stored_steps(root, &next), [0, 1], "{lock} poisoned");
}

/// One test a mutex: poisoned, then every request that takes it served.
macro_rules! served_after_a_panic_under {
    ($($name:ident: $field:ident,)*) => {$(
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn $name() {
            let dir = tempfile::tempdir().unwrap();
            let server = start(dir.path()).await;
            poison(&server.ingest.$field);
            serves(&server, dir.path(), stringify!($field)).await;
        }
    )*};
}

served_after_a_panic_under! {
    served_after_a_panic_under_buffers: buffers,
    served_after_a_panic_under_written: written,
    served_after_a_panic_under_data_windows: data_windows,
    served_after_a_panic_under_replicas: replicas,
    served_after_a_panic_under_reports: reports,
    served_after_a_panic_under_stored: stored,
    served_after_a_panic_under_manifests: manifests,
    served_after_a_panic_under_error_log: error_log,
    served_after_a_panic_under_end_log: end_log,
    served_after_a_panic_under_dropped_rows: dropped_rows,
    served_after_a_panic_under_dropped_log: dropped_log,
    served_after_a_panic_under_settled: settled,
    served_after_a_panic_under_order_locks: order_locks,
    served_after_a_panic_under_streams: streams,
}

/// The same requests on a server where nothing was poisoned: the scenario itself passes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn served_with_nothing_poisoned() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    serves(&server, dir.path(), "none").await;
}
