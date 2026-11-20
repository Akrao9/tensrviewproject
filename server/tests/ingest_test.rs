//! The ingest server over a real socket, with a local directory as the object store.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Array, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::{Action, FlightClient, FlightData, FlightDescriptor, PutResult};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use object_store::local::LocalFileSystem;
use object_store::{ObjectStore, ObjectStoreExt};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use tensorquill_server::ingest::{Ingest, IngestConfig, DEFAULT_FLUSH_STEPS, DEFAULT_QUOTA_BYTES};
use tensorquill_server::server::serve;
use tensorquill_server::storage::Storage;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Channel;
use tonic::Code;

const RUN: &str = "run-a";

struct TestServer {
    address: String,
    ingest: Arc<Ingest>,
    _stop: oneshot::Sender<()>,
}

/// A server in this process, storing under `root`.
async fn start(root: &Path, config: IngestConfig, token: Option<&str>) -> TestServer {
    let store = LocalFileSystem::new_with_prefix(root).unwrap();
    start_on(Arc::new(store), config, token).await
}

/// A server in this process, storing in `store`.
async fn start_on(store: Arc<dyn ObjectStore>, config: IngestConfig, token: Option<&str>) -> TestServer {
    let ingest = Ingest::new(Storage::new(store, "".into()), config);
    ingest.spawn_flusher();
    serve_ingest(ingest, token).await
}

async fn serve_ingest(ingest: Arc<Ingest>, token: Option<&str>) -> TestServer {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let (stop, stopped) = oneshot::channel::<()>();
    let token = token.map(String::from);
    let served = ingest.clone();
    tokio::spawn(async move {
        serve(listener, served, token, None, async {
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

/// Never flushes on its own: only the end of a stream or shutdown writes a part.
fn manual() -> IngestConfig {
    IngestConfig {
        flush_steps: usize::MAX,
        flush_interval: Duration::from_secs(3600),
        ..Default::default()
    }
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("schema_version", DataType::Int32, false),
        Field::new("run_id", DataType::Utf8, false),
        Field::new("rank", DataType::Int32, false),
        Field::new("step", DataType::Int64, false),
        Field::new("ts", DataType::Float64, true),
        Field::new("layers_json", DataType::Utf8, true),
    ]))
}

fn batch_with(rank: i32, steps: impl IntoIterator<Item = i64>, layers_json: &str) -> RecordBatch {
    let steps: Vec<i64> = steps.into_iter().collect();
    let n = steps.len();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(vec![1; n])),
            Arc::new(StringArray::from(vec![RUN; n])),
            Arc::new(Int32Array::from(vec![rank; n])),
            Arc::new(Int64Array::from(steps.clone())),
            Arc::new(Float64Array::from(
                steps.iter().map(|step| *step as f64 + 0.5).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(vec![layers_json; n])),
        ],
    )
    .unwrap()
}

fn batch(rank: i32, steps: impl IntoIterator<Item = i64>) -> RecordBatch {
    batch_with(rank, steps, "{}")
}

/// One open `DoPut` stream, as the Python sender holds it.
struct Put {
    rank: i32,
    messages: mpsc::UnboundedSender<Result<FlightData, FlightError>>,
    acks: BoxStream<'static, Result<PutResult, FlightError>>,
}

async fn open_as(server: &TestServer, run_id: &str, rank: i32, token: Option<&str>) -> Put {
    open_at(&server.address, run_id, rank, token).await
}

async fn open_at(address: &str, run_id: &str, rank: i32, token: Option<&str>) -> Put {
    open_with(address, run_id, rank, token, schema()).await
}

async fn open_with(address: &str, run_id: &str, rank: i32, token: Option<&str>, schema: Arc<Schema>) -> Put {
    let channel = Channel::from_shared(address.to_string())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightClient::new(channel);
    if let Some(token) = token {
        client.add_header("authorization", &format!("Bearer {token}")).unwrap();
    }
    let (messages, outgoing) = mpsc::unbounded_channel();
    // The schema message opens the stream, as pyarrow's `do_put` sends it before any batch.
    let descriptor = FlightDescriptor::new_path(vec!["runs".into(), run_id.into(), format!("rank={rank}")]);
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
    Put { rank, messages, acks }
}

async fn open(server: &TestServer, rank: i32) -> Put {
    open_as(server, RUN, rank, None).await
}

impl Put {
    /// Sends one batch with the `app_metadata` the contract names.
    async fn send(&mut self, batch: RecordBatch, end_byte_offset: i64) {
        let metadata = format!(
            "{{\"schema_version\":1,\"run_id\":\"{RUN}\",\"rank\":{},\"min_step\":null,\"max_step\":null,\"end_byte_offset\":{end_byte_offset}}}",
            self.rank
        );
        self.send_with_metadata(batch, metadata).await;
    }

    async fn send_with_metadata(&mut self, batch: RecordBatch, metadata: String) {
        let encoded: Vec<FlightData> = FlightDataEncoderBuilder::new()
            .with_max_flight_data_size(usize::MAX)
            .build(futures::stream::iter([Ok(batch)]))
            .try_collect()
            .await
            .unwrap();
        // Each encoder starts with a schema message; the stream already has one.
        for mut message in encoded.into_iter().skip(1) {
            message.app_metadata = metadata.clone().into();
            let _ = self.messages.send(Ok(message));
        }
    }

    /// The next acknowledged `end_byte_offset`, or the status that ended the stream.
    async fn ack(&mut self) -> Result<i64, Code> {
        match tokio::time::timeout(Duration::from_secs(20), self.acks.next())
            .await
            .expect("no reply")
        {
            Some(Ok(result)) => {
                let value: serde_json::Value = serde_json::from_slice(&result.app_metadata).unwrap();
                Ok(value["end_byte_offset"].as_i64().unwrap())
            }
            Some(Err(FlightError::Tonic(status))) => Err(status.code()),
            Some(Err(other)) => panic!("{other}"),
            None => panic!("the stream ended without a reply"),
        }
    }

    async fn no_ack_within(&mut self, wait: Duration) -> bool {
        tokio::time::timeout(wait, self.acks.next()).await.is_err()
    }

    /// Ends the stream and returns every remaining acknowledgement.
    async fn finish(mut self) -> Vec<i64> {
        drop(self.messages);
        let mut offsets = Vec::new();
        while let Some(result) = self.acks.next().await {
            let value: serde_json::Value = serde_json::from_slice(&result.unwrap().app_metadata).unwrap();
            offsets.push(value["end_byte_offset"].as_i64().unwrap());
        }
        offsets
    }
}

fn part_files(root: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, found: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, found);
            } else if path.extension().is_some_and(|extension| extension == "parquet") {
                found.push(path);
            }
        }
    }
    let mut found = Vec::new();
    walk(root, &mut found);
    found.sort();
    found
}

/// Part paths relative to the store root, with `/` separators.
fn part_names(root: &Path) -> Vec<String> {
    part_files(root)
        .iter()
        .map(|path| path.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/"))
        .collect()
}

/// Every stored step per rank, in file order.
fn stored_steps(root: &Path) -> BTreeMap<i32, Vec<i64>> {
    let mut steps: BTreeMap<i32, Vec<i64>> = BTreeMap::new();
    for path in part_files(root) {
        let reader = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&path).unwrap()).unwrap();
        for batch in reader.build().unwrap() {
            let batch = batch.unwrap();
            let rank = batch
                .column_by_name("rank")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let step = batch
                .column_by_name("step")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for row in 0..batch.num_rows() {
                steps.entry(rank.value(row)).or_default().push(step.value(row));
            }
        }
    }
    steps
}

#[tokio::test]
async fn resending_steps_writes_no_duplicate_rows() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), manual(), None).await;

    let mut put = open(&server, 0).await;
    put.send(batch(0, 0..50), 1000).await;
    assert_eq!(put.finish().await, vec![1000]);

    // The acknowledgement was lost: the sender starts over from byte 0.
    let mut put = open(&server, 0).await;
    put.send(batch(0, 0..50), 1000).await;
    assert_eq!(
        put.ack().await,
        Ok(1000),
        "nothing new to store, so it is confirmed at once"
    );
    put.send(batch(0, 40..60), 1200).await; // overlaps what is stored
    assert_eq!(put.finish().await, vec![1200]);

    assert_eq!(stored_steps(dir.path())[&0], (0..60).collect::<Vec<_>>());
    assert_eq!(
        part_names(dir.path()),
        [
            "runs/run-a/rank=0/attempt=0/part-00000000-00000049.parquet",
            "runs/run-a/rank=0/attempt=0/part-00000050-00000059.parquet"
        ]
    );
    assert_eq!(server.ingest.buffered_bytes(), 0);
}

fn attempt_schema() -> Arc<Schema> {
    let mut fields: Vec<Field> = schema().fields().iter().map(|field| field.as_ref().clone()).collect();
    fields.insert(3, Field::new("attempt", DataType::Int64, false));
    Arc::new(Schema::new(fields))
}

async fn open_attempts(server: &TestServer) -> Put {
    open_with(&server.address, RUN, 0, None, attempt_schema()).await
}

/// Rows as a sender with attempts writes them: `rows` is `(attempt, step)` in file order.
fn attempt_batch(rows: &[(i64, i64)]) -> RecordBatch {
    let n = rows.len();
    RecordBatch::try_new(
        attempt_schema(),
        vec![
            Arc::new(Int32Array::from(vec![1; n])),
            Arc::new(StringArray::from(vec![RUN; n])),
            Arc::new(Int32Array::from(vec![0; n])),
            Arc::new(Int64Array::from(rows.iter().map(|row| row.0).collect::<Vec<_>>())),
            Arc::new(Int64Array::from(rows.iter().map(|row| row.1).collect::<Vec<_>>())),
            Arc::new(Float64Array::from(vec![0.5; n])),
            Arc::new(StringArray::from(vec!["{}"; n])),
        ],
    )
    .unwrap()
}

fn rows(attempt: i64, steps: std::ops::Range<i64>) -> Vec<(i64, i64)> {
    steps.map(|step| (attempt, step)).collect()
}

#[tokio::test]
async fn a_resumed_attempt_keeps_the_steps_it_writes_again() {
    let dir = tempfile::tempdir().unwrap();
    let first = rows(100, 0..150);
    let resumed = rows(200, 100..180); // restarted from the checkpoint at step 100
    let whole_file: Vec<_> = first.iter().chain(&resumed).copied().collect();
    let expected = [
        "runs/run-a/rank=0/attempt=100/part-00000000-00000149.parquet",
        "runs/run-a/rank=0/attempt=200/part-00000100-00000179.parquet",
    ];
    {
        let server = start(dir.path(), manual(), None).await;
        let mut put = open_attempts(&server).await;
        put.send(attempt_batch(&first), 1000).await;
        assert_eq!(put.finish().await, vec![1000]);
        let mut put = open_attempts(&server).await;
        put.send(attempt_batch(&resumed), 1600).await;
        assert_eq!(put.finish().await, vec![1600]);
        assert_eq!(part_names(dir.path()), expected);

        // The offset file was lost: the sender resends the file from byte 0, both attempts
        // in one batch. Each row is checked against its own attempt, so nothing is new.
        let mut put = open_attempts(&server).await;
        put.send(attempt_batch(&whole_file), 1600).await;
        assert_eq!(put.ack().await, Ok(1600));
        put.finish().await;
    }
    // A restarted server finds each attempt's watermark in that attempt's parts.
    let server = start(dir.path(), manual(), None).await;
    let mut put = open_attempts(&server).await;
    put.send(attempt_batch(&whole_file), 1600).await;
    assert_eq!(put.ack().await, Ok(1600));
    // One batch that ends one attempt and starts the next is written as two parts.
    let mut tail = rows(200, 180..185);
    tail.extend(rows(300, 50..60));
    put.send(attempt_batch(&tail), 1900).await;
    assert_eq!(put.finish().await, vec![1900]);

    let names = part_names(dir.path());
    assert_eq!(names[..2], expected);
    assert_eq!(
        names[2..],
        [
            "runs/run-a/rank=0/attempt=200/part-00000180-00000184.parquet",
            "runs/run-a/rank=0/attempt=300/part-00000050-00000059.parquet",
        ]
    );
    assert_eq!(stored_steps(dir.path())[&0].len(), 150 + 80 + 5 + 10);
    assert_eq!(server.ingest.buffered_bytes(), 0);
}

/// `(attempt, last_step, gaps)` of rank 0 in the status, once its last step is `step`.
async fn progress_at(server: &TestServer, step: i64) -> (i64, i64, usize) {
    until_within(&format!("last_step {step}"), SLOW_BOUND, || {
        server
            .ingest
            .watchdog_statuses()
            .get(RUN)
            .is_some_and(|status| status["ranks"][0]["last_step"] == step)
    })
    .await;
    let status = &server.ingest.watchdog_statuses()[RUN];
    (
        status["ranks"][0]["attempt"].as_i64().unwrap(),
        status["ranks"][0]["last_step"].as_i64().unwrap(),
        status["restart"][0]["gaps_seconds"].as_array().unwrap().len(),
    )
}

/// The status's `last_step` is the newest attempt's once that attempt stores rows, also when it
/// resumed from a checkpoint below the steps an earlier attempt reached, and when one batch ends
/// an attempt and starts the next. (Until Nov 18, 2026 it was the highest step of any attempt,
/// so a resumed job read as the old attempt's step until it passed it.) Each batch is one gap,
/// however many attempts it holds, so the stall timing is what it was.
#[tokio::test]
async fn the_status_follows_the_newest_attempts_step_once_it_stores_rows() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), watch_config(60_000), None).await;
    let mut put = open_attempts(&server).await;
    put.send(attempt_batch(&rows(100, 0..150)), 1000).await;
    assert_eq!(progress_at(&server, 149).await, (100, 149, 1));
    // Resumed from the checkpoint at step 100.
    put.send(attempt_batch(&rows(200, 100..110)), 1100).await;
    assert_eq!(progress_at(&server, 109).await, (200, 109, 2));
    // One batch that ends attempt 200 and starts attempt 300 from step 50.
    let mut tail = rows(200, 110..120);
    tail.extend(rows(300, 50..60));
    put.send(attempt_batch(&tail), 1200).await;
    assert_eq!(progress_at(&server, 59).await, (300, 59, 3));
    // A part holds one attempt: each change of attempt wrote the rows before it.
    assert_eq!(put.finish().await, vec![1000, 1100, 1200]);
}

#[tokio::test]
async fn a_restarted_server_reads_its_watermark_from_the_parquet_footers() {
    let dir = tempfile::tempdir().unwrap();
    {
        let server = start(dir.path(), manual(), None).await;
        let mut put = open(&server, 3).await;
        put.send(batch(3, 0..50), 500).await;
        put.send(batch(3, 50..120), 900).await;
        assert_eq!(put.finish().await, vec![900]);
    }
    let server = start(dir.path(), manual(), None).await; // knows nothing in memory
    let mut put = open(&server, 3).await;
    put.send(batch(3, 100..130), 1000).await;
    assert_eq!(put.finish().await, vec![1000]);
    let mut other = open(&server, 4).await; // another rank has its own watermark
    other.send(batch(4, 0..10), 77).await;
    assert_eq!(other.finish().await, vec![77]);

    let steps = stored_steps(dir.path());
    assert_eq!(steps[&3], (0..130).collect::<Vec<_>>());
    assert_eq!(steps[&4], (0..10).collect::<Vec<_>>());
    assert_eq!(part_names(dir.path()).len(), 3);
}

#[tokio::test]
async fn repeated_and_out_of_order_steps_inside_one_batch_are_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), manual(), None).await;
    let mut put = open(&server, 0).await;
    put.send(batch(0, [0, 1, 1, 2, 0, 3]), 10).await;
    put.finish().await;
    assert_eq!(stored_steps(dir.path())[&0], [0, 1, 2, 3]);
}

#[tokio::test]
async fn exceeding_the_128_mib_quota_returns_resource_exhausted() {
    assert_eq!(DEFAULT_QUOTA_BYTES, 128 * 1024 * 1024);
    let dir = tempfile::tempdir().unwrap();
    let config = IngestConfig {
        quota_bytes: DEFAULT_QUOTA_BYTES,
        ..manual()
    };
    let server = start(dir.path(), config, None).await;
    let wide = "x".repeat(50 * 1024 * 1024); // one 50 MiB row

    let mut first = open(&server, 0).await;
    first.send(batch_with(0, [0], &wide), 1).await;
    let mut second = open(&server, 1).await;
    second.send(batch_with(1, [0], &wide), 1).await;
    // Both rows are buffered, however long reading 100 MiB takes on a loaded machine (a fixed
    // wait here failed under load).
    until_within("both 50 MiB rows buffered", SLOW_BOUND, || {
        server.ingest.buffered_bytes() >= 100 * 1024 * 1024
    })
    .await;

    // A third rank's 50 MiB would pass 128 MiB: refused, and nothing of it is kept.
    let mut third = open(&server, 2).await;
    third.send(batch_with(2, [0], &wide), 1).await;
    assert_eq!(third.ack().await, Err(Code::ResourceExhausted));
    assert!(server.ingest.buffered_bytes() <= DEFAULT_QUOTA_BYTES);
    // While the third was refused, neither of the first two was written or acknowledged.
    assert!(first.no_ack_within(Duration::from_millis(100)).await);
    assert!(second.no_ack_within(Duration::from_millis(100)).await);

    // Once a buffer is written there is room, and the retry is accepted.
    assert_eq!(first.finish().await, vec![1]);
    let mut third = open(&server, 2).await;
    third.send(batch_with(2, [0], &wide), 1).await;
    assert_eq!(third.finish().await, vec![1]);
    assert_eq!(second.finish().await, vec![1]);
    assert_eq!(server.ingest.buffered_bytes(), 0);
    assert_eq!(stored_steps(dir.path()).len(), 3);
}

/// The status that ended a refused stream as the client read it, with the errors under it.
async fn refusal(put: &mut Put) -> String {
    match tokio::time::timeout(SLOW_BOUND, put.acks.next())
        .await
        .expect("no reply")
    {
        Some(Err(FlightError::Tonic(status))) => {
            let mut text = format!("{:?}: {}", status.code(), status.message());
            let mut source = std::error::Error::source(&*status);
            while let Some(error) = source {
                text += &format!(" <- {error}");
                source = error.source();
            }
            text
        }
        other => format!("not a status: {other:?}"),
    }
}

/// A refused stream's sender is told why, read through a real client, while it still has
/// batches queued behind the refused one: its DATA is in flight when the server stops reading
/// and drops the request stream, before the error is sent. The review of Nov 18, 2026 (RUST-01)
/// held that this order would show as `Unknown: h2 protocol error: error reading a body from
/// connection`. It does not: dropping the request body alone resets nothing while the response
/// is open (h2 0.4.20, `RecvStream::drop`), the status goes out as trailers, and the reset that
/// follows is `NO_ERROR`, which hyper reads as the end of the body. Many streams at once, on a
/// runtime with several threads as the binary has.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_refused_stream_is_told_its_status_while_more_batches_are_in_flight() {
    let dir = tempfile::tempdir().unwrap();
    let wide = "x".repeat(1024 * 1024); // one 1 MiB row
    let config = IngestConfig {
        quota_bytes: 3 * 1024 * 1024 / 2,
        ..manual()
    };
    let server = start(dir.path(), config, None).await;
    // One rank holds 1 MiB, so every other rank's 1 MiB row is over the quota.
    let mut holder = open(&server, 1000).await;
    holder.send(batch_with(1000, [0], &wide), 1).await;
    until_within("the holder's row buffered", SLOW_BOUND, || {
        server.ingest.buffered_bytes() >= 1024 * 1024
    })
    .await;

    let (rounds, at_once, queued): (i32, i32, i64) = (25, 8, 8);
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for round in 0..rounds {
        let mut streams = Vec::new();
        for slot in 0..at_once {
            let rank = round * at_once + slot;
            let address = server.address.clone();
            let wide = wide.clone();
            streams.push(tokio::spawn(async move {
                let mut put = open_at(&address, RUN, rank, None).await;
                if slot % 2 == 0 {
                    // Refused for its quota.
                    put.send(batch_with(rank, [0], &wide), 1).await;
                } else {
                    // Refused as a bad batch: rows of another rank.
                    put.send(batch_with(rank + 1, [0], &wide), 1).await;
                }
                for step in 1..=queued {
                    put.send(batch_with(rank, [step], &wide), step + 1).await;
                }
                refusal(&mut put).await
            }));
        }
        for stream in streams {
            *seen.entry(stream.await.unwrap()).or_default() += 1;
        }
    }
    let half = (rounds * at_once / 2) as usize;
    let expected = BTreeMap::from([
        (
            "InvalidArgument: a row belongs to another run or rank than the descriptor".to_string(),
            half,
        ),
        ("ResourceExhausted: ingest buffer full".to_string(), half),
    ]);
    assert_eq!(seen, expected);
    assert_eq!(holder.finish().await, vec![1]);
    assert_eq!(server.ingest.buffered_bytes(), 0);
}

#[tokio::test]
async fn a_full_quota_writes_the_ranks_own_buffer_before_refusing() {
    let dir = tempfile::tempdir().unwrap();
    let one: usize = batch(0, 0..50)
        .columns()
        .iter()
        .map(|c| c.to_data().get_slice_memory_size().unwrap())
        .sum();
    let config = IngestConfig {
        quota_bytes: one * 5 / 2,
        ..manual()
    };
    let server = start(dir.path(), config, None).await;
    let mut put = open(&server, 0).await;
    put.send(batch(0, 0..50), 1).await;
    put.send(batch(0, 50..100), 2).await;
    assert!(put.no_ack_within(Duration::from_millis(300)).await);
    put.send(batch(0, 100..150), 3).await; // does not fit beside the first two
    assert_eq!(
        put.ack().await,
        Ok(2),
        "the first two batches were written to make room"
    );
    assert_eq!(put.finish().await, vec![3]);
    assert_eq!(stored_steps(dir.path())[&0], (0..150).collect::<Vec<_>>());
}

#[tokio::test]
async fn a_part_is_written_every_500_steps_and_acknowledged_only_then() {
    assert_eq!(DEFAULT_FLUSH_STEPS, 500);
    let dir = tempfile::tempdir().unwrap();
    let config = IngestConfig {
        flush_interval: Duration::from_secs(3600),
        ..Default::default()
    };
    let server = start(dir.path(), config, None).await;
    let mut put = open(&server, 0).await;
    for i in 0..9 {
        put.send(batch(0, i * 50..(i + 1) * 50), (i + 1) * 100).await;
    }
    assert!(
        put.no_ack_within(Duration::from_millis(300)).await,
        "450 steps are only buffered"
    );
    assert!(part_files(dir.path()).is_empty());
    put.send(batch(0, 450..500), 1000).await;
    assert_eq!(put.ack().await, Ok(1000));
    assert_eq!(
        part_names(dir.path()),
        ["runs/run-a/rank=0/attempt=0/part-00000000-00000499.parquet"]
    );

    let file = std::fs::File::open(&part_files(dir.path())[0]).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let column = reader.metadata().row_group(0).column(0);
    assert!(matches!(column.compression(), Compression::ZSTD(_)));
    assert_eq!(reader.metadata().file_metadata().num_rows(), 500);
    drop(put);
}

#[tokio::test]
async fn a_part_is_written_after_the_flush_interval_while_the_stream_stays_open() {
    let dir = tempfile::tempdir().unwrap();
    let config = IngestConfig {
        flush_interval: Duration::from_millis(200),
        ..Default::default()
    };
    let server = start(dir.path(), config, None).await;
    let mut put = open(&server, 0).await;
    put.send(batch(0, 0..7), 70).await;
    assert_eq!(put.ack().await, Ok(70));
    assert_eq!(
        part_names(dir.path()),
        ["runs/run-a/rank=0/attempt=0/part-00000000-00000006.parquet"]
    );
    put.send(batch(0, 7..9), 90).await;
    assert_eq!(put.ack().await, Ok(90));
    assert_eq!(stored_steps(dir.path())[&0], (0..9).collect::<Vec<_>>());
}

#[tokio::test]
async fn a_wrong_bearer_token_is_unauthenticated() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), manual(), Some("s3cret")).await;
    for token in [None, Some("wrong"), Some("s3cret-and-more")] {
        let mut put = open_as(&server, RUN, 0, token).await;
        put.send(batch(0, 0..5), 5).await;
        assert_eq!(put.ack().await, Err(Code::Unauthenticated));
    }
    let mut put = open_as(&server, RUN, 0, Some("s3cret")).await;
    put.send(batch(0, 0..5), 5).await;
    assert_eq!(put.finish().await, vec![5]);
    assert_eq!(stored_steps(dir.path())[&0], [0, 1, 2, 3, 4]);
}

#[tokio::test]
async fn bad_requests_are_invalid_argument() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), manual(), None).await;

    let mut put = open_as(&server, "../escape", 0, None).await;
    put.send(batch(0, 0..5), 5).await;
    assert_eq!(put.ack().await, Err(Code::InvalidArgument));

    let mut put = open(&server, 0).await;
    let version_2 = format!("{{\"schema_version\":2,\"run_id\":\"{RUN}\",\"rank\":0,\"end_byte_offset\":5}}");
    put.send_with_metadata(batch(0, 0..5), version_2).await;
    assert_eq!(put.ack().await, Err(Code::InvalidArgument));

    let mut put = open(&server, 0).await;
    put.send(batch(1, 0..5), 5).await; // rows of rank 1 on rank 0's stream
    assert_eq!(put.ack().await, Err(Code::InvalidArgument));

    assert!(part_files(dir.path()).is_empty());
}

/// A JSON file the server writes, or `Null` while there is none.
fn json_file(path: &Path) -> serde_json::Value {
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

/// The `code` of each reason in a status's `run_state`.
fn run_state_codes(status: &serde_json::Value) -> Vec<String> {
    status["run_state"]["reasons"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|reason| reason["code"].as_str().map(str::to_string))
        .collect()
}

/// The real binary: `SIGTERM` must write what is buffered before the process exits, and only
/// then the last status, so the last `watchdog.json` and `incidents.json` both cover the last
/// part stored. (Until Nov 18, 2026 the stopped status was written before the buffers were.)
#[cfg(unix)]
#[tokio::test]
async fn sigterm_flushes_buffered_rows_before_exit() {
    use std::io::{BufRead, BufReader};

    let dir = tempfile::tempdir().unwrap();
    // A part only at the end of a stream or at shutdown; the status rewritten every 0.1 s.
    let mut child = Command::new(env!("CARGO_BIN_EXE_tensorquill-server"))
        .args([
            "--listen",
            "127.0.0.1:0",
            "--flush-seconds",
            "3600",
            "--watchdog-seconds",
            "0.1",
        ])
        .arg("--store")
        .arg(dir.path())
        .env_remove("TENSORQUILL_TOKEN")
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    let address = line.trim().rsplit(' ').next().unwrap().to_string();

    let mut put = open_at(&format!("http://{address}"), RUN, 0, None).await;
    put.send(batch(0, 0..50), 4242).await;
    // The server has the rows once its status names their last step. Waited for, not slept on:
    // a signal sent before the rows arrive would find nothing to write.
    let status_path = dir.path().join("runs").join(RUN).join("watchdog.json");
    until_within("the batch taken", SLOW_BOUND, || {
        json_file(&status_path)["ranks"][0]["last_step"] == 49
    })
    .await;
    assert!(
        put.no_ack_within(Duration::from_millis(200)).await,
        "50 steps are only buffered"
    );
    assert!(part_files(dir.path()).is_empty());
    let live = json_file(&status_path);
    assert_eq!(live["server_running"], true);
    assert!(run_state_codes(&live).contains(&"no_steps".to_string()), "{live}");

    assert!(Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .unwrap()
        .success());
    let status = tokio::task::spawn_blocking(move || child.wait().unwrap())
        .await
        .unwrap();
    assert!(status.success(), "{status}");
    assert_eq!(stored_steps(dir.path())[&0], (0..50).collect::<Vec<_>>());
    assert_eq!(
        part_names(dir.path()),
        ["runs/run-a/rank=0/attempt=0/part-00000000-00000049.parquet"]
    );
    // The detectors judged that part ...
    let incidents = json_file(&dir.path().join("runs").join(RUN).join("rank=0").join("incidents.json"));
    assert_eq!(incidents["last_step"], 49, "{incidents}");
    // ... before the last status was written: it says the server stopped, and rank 0 judged.
    let stopped = json_file(&status_path);
    assert_eq!(
        (stopped["summary"].as_str(), stopped["server_running"].as_bool()),
        (Some("server_stopped"), Some(false))
    );
    let codes = run_state_codes(&stopped);
    assert!(
        !codes.iter().any(|code| code == "no_steps" || code == "ranks_unknown"),
        "{stopped}"
    );
}

/// The binary refuses to listen beyond loopback with no token and no explicit flag.
#[test]
fn the_binary_refuses_an_unauthenticated_public_bind() {
    let dir = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_tensorquill-server"))
        .args(["--listen", "0.0.0.0:0", "--store"])
        .arg(dir.path())
        .env_remove("TENSORQUILL_TOKEN")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("TENSORQUILL_TOKEN"));
}

fn watch_config(floor_ms: u64) -> IngestConfig {
    IngestConfig {
        silent_floor: Duration::from_millis(floor_ms),
        silent_factor: 5.0,
        watchdog_interval: Duration::ZERO,
        ..manual()
    }
}

fn rank_states(server: &TestServer) -> Vec<(i64, String, String)> {
    server.ingest.watchdog_statuses()[RUN]["ranks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                r["rank"].as_i64().unwrap(),
                r["state"].as_str().unwrap().to_string(),
                r["stream"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

async fn until(what: &str, ready: impl FnMut() -> bool) {
    until_within(what, Duration::from_secs(5), ready).await;
}

/// How long to wait for something a loaded machine is slow to do (reading 100 MiB in a debug
/// build beside other builds). Only a failing test waits this long.
const SLOW_BOUND: Duration = Duration::from_secs(120);

/// Waits until `ready`, looking every 50 ms, and fails after `bound`. A state the server reaches
/// on its own time is waited for this way, never with a fixed sleep.
async fn until_within(what: &str, bound: Duration, mut ready: impl FnMut() -> bool) {
    let started = std::time::Instant::now();
    while !ready() {
        assert!(
            started.elapsed() < bound,
            "timed out after {bound:?} waiting for {what}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn world_metadata(rank: i32, world_size: i32, end: i64) -> String {
    format!(
        "{{\"schema_version\":1,\"run_id\":\"{RUN}\",\"rank\":{rank},\"min_step\":null,\"max_step\":null,\"end_byte_offset\":{end},\"world_size\":{world_size}}}"
    )
}

#[tokio::test]
async fn the_watchdog_sees_progress_a_hang_and_a_clean_finish() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), watch_config(400), None).await;
    let mut ranks = Vec::new();
    for rank in 0..2 {
        let mut put = open(&server, rank).await;
        put.send_with_metadata(batch(rank, 0..50), world_metadata(rank, 3, 100))
            .await;
        ranks.push(put);
    }
    until("both ranks active", || {
        let states = rank_states(&server);
        states.len() == 2 && states.iter().all(|(_, state, _)| state == "active")
    })
    .await;
    let status = &server.ingest.watchdog_statuses()[RUN];
    assert_eq!(status["world_size"], 3);
    assert_eq!(status["ranks"][0]["last_step"], 49);
    // Rank 2 has not connected, but ranks start a moment apart: not missing yet.
    assert_eq!(status["missing_ranks"], serde_json::json!([]));
    assert_eq!(status["summary"], "active");

    // Rank 1 keeps its stream open and sends nothing new; rank 0 goes on.
    let mut rank0 = ranks.remove(0);
    let mut rank1 = ranks.remove(0);
    for step in [50, 100, 150] {
        tokio::time::sleep(Duration::from_millis(250)).await;
        rank0
            .send_with_metadata(batch(0, step..step + 50), world_metadata(0, 3, 100 + step))
            .await;
        // A resend of rows already kept is not progress.
        rank1
            .send_with_metadata(batch(1, 0..50), world_metadata(1, 3, 100))
            .await;
    }
    until("rank 1 silent", || {
        rank_states(&server)
            .iter()
            .any(|(rank, state, _)| *rank == 1 && state == "silent")
    })
    .await;
    assert_eq!(rank_states(&server)[0].1, "active");
    assert_eq!(server.ingest.watchdog_statuses()[RUN]["ranks"][1]["last_step"], 49);
    // Past the floor, the rank that never connected is named.
    assert_eq!(
        server.ingest.watchdog_statuses()[RUN]["missing_ranks"],
        serde_json::json!([2])
    );

    // The status file is what readers see. Wait for rank 0's last batch to land first.
    until("rank 0's last batch", || {
        server.ingest.watchdog_statuses()[RUN]["ranks"][0]["last_step"] == 199
    })
    .await;
    server.ingest.write_watchdog().await;
    let written: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("runs").join(RUN).join("watchdog.json")).unwrap())
            .unwrap();
    assert_eq!(written["summary"], "degraded");
    assert_eq!(written["ranks"][1]["state"], "silent");
    // An unchanged status is not written again, even though its clock fields moved.
    let path = dir.path().join("runs").join(RUN).join("watchdog.json");
    let first = std::fs::metadata(&path).unwrap().modified().unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    server.ingest.write_watchdog().await;
    assert_eq!(std::fs::metadata(&path).unwrap().modified().unwrap(), first);

    // A clean end is finished; a stream that is cut is not.
    let _ = rank0.finish().await;
    until("rank 0 finished", || rank_states(&server)[0].1 == "finished").await;
    // A stream that ends in an error, as a cut connection does, is dropped, not finished. A
    // bad batch is the way to make the server end one here.
    rank1
        .send_with_metadata(batch(1, 50..60), world_metadata(1, 3, 999).replace(RUN, "another-run"))
        .await;
    assert_eq!(rank1.ack().await, Err(Code::InvalidArgument));
    until("rank 1 dropped", || rank_states(&server)[1].2 == "dropped").await;
}

#[tokio::test]
async fn an_unchanged_status_is_written_again_after_the_refresh_time_until_the_run_finishes() {
    let dir = tempfile::tempdir().unwrap();
    let config = IngestConfig {
        status_refresh: Duration::from_millis(150),
        ..watch_config(60_000)
    };
    let server = start(dir.path(), config, None).await;
    let mut put = open(&server, 0).await;
    put.send_with_metadata(batch(0, 0..10), world_metadata(0, 1, 50)).await;
    until("the rank's batch", || {
        rank_states(&server).len() == 1 && server.ingest.watchdog_statuses()[RUN]["ranks"][0]["last_step"] == 9
    })
    .await;
    let path = dir.path().join("runs").join(RUN).join("watchdog.json");
    let read = || -> serde_json::Value { serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap() };
    server.ingest.write_watchdog().await;
    let first = read();
    assert_eq!(
        (first["summary"].as_str(), first["refresh_seconds"].as_f64()),
        (Some("active"), Some(0.15))
    );
    // Nothing moves in the run. Within the refresh time the file is left alone; after it
    // the same verdict is written again with a new time, which is how a reader knows the
    // server is still there.
    server.ingest.write_watchdog().await;
    assert_eq!(read()["written_at_ms"], first["written_at_ms"]);
    tokio::time::sleep(Duration::from_millis(200)).await;
    server.ingest.write_watchdog().await;
    let second = read();
    assert_eq!(second["summary"], "active");
    assert!(second["written_at_ms"].as_i64() > first["written_at_ms"].as_i64());

    // A finished run needs no watching, and its status is not written again.
    let _ = put.finish().await;
    until("the rank finished", || rank_states(&server)[0].1 == "finished").await;
    server.ingest.write_watchdog().await;
    let finished = read();
    assert_eq!(finished["summary"], "finished");
    tokio::time::sleep(Duration::from_millis(200)).await;
    server.ingest.write_watchdog().await;
    assert_eq!(read()["written_at_ms"], finished["written_at_ms"]);
}

#[tokio::test]
async fn the_last_write_before_shutdown_says_the_server_stopped() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), watch_config(60_000), None).await;
    let mut put = open(&server, 0).await;
    put.send_with_metadata(batch(0, 0..10), world_metadata(0, 1, 50)).await;
    until("the rank is seen", || rank_states(&server).len() == 1).await;
    server.ingest.write_watchdog().await;
    let path = dir.path().join("runs").join(RUN).join("watchdog.json");
    let live: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(
        (live["summary"].as_str(), live["server_running"].as_bool()),
        (Some("active"), Some(true))
    );
    server.ingest.write_watchdog_stopped().await;
    let stopped: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(
        (stopped["summary"].as_str(), stopped["server_running"].as_bool()),
        (Some("server_stopped"), Some(false))
    );
}

#[tokio::test]
async fn a_finished_run_keeps_its_verdict_when_the_server_stops() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), watch_config(60_000), None).await;
    let mut put = open(&server, 0).await;
    put.send_with_metadata(batch(0, 0..10), world_metadata(0, 1, 50)).await;
    let _ = put.finish().await;
    until("the rank finished", || rank_states(&server)[0].1 == "finished").await;
    server.ingest.write_watchdog_stopped().await;
    let path = dir.path().join("runs").join(RUN).join("watchdog.json");
    let status: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(status["summary"], "finished");
}

async fn client(server: &TestServer) -> FlightClient {
    let channel = Channel::from_shared(server.address.clone())
        .unwrap()
        .connect()
        .await
        .unwrap();
    FlightClient::new(channel)
}

#[tokio::test]
async fn a_heartbeat_names_the_phase_and_a_bad_one_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), watch_config(60_000), None).await;
    let mut flight = client(&server).await;
    let beat = |body: &str| Action::new("heartbeat", body.to_string());
    flight
        .do_action(beat(&format!(
            "{{\"run_id\":\"{RUN}\",\"rank\":0,\"phase\":\"checkpoint\",\"interval_seconds\":15}}"
        )))
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let ranks = &server.ingest.watchdog_statuses()[RUN]["ranks"];
    assert_eq!(
        (ranks[0]["phase"].as_str(), ranks[0]["rank"].as_i64()),
        (Some("checkpoint"), Some(0))
    );
    assert!(ranks[0]["heartbeat_age_seconds"].is_number());
    // The phase comes back to training and the pause is on record.
    flight
        .do_action(beat(&format!(
            "{{\"run_id\":\"{RUN}\",\"rank\":0,\"phase\":\"train\"}}"
        )))
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(
        server.ingest.watchdog_statuses()[RUN]["pauses"]["checkpoint"]["count"],
        1
    );
    // Refused: another action, bad JSON, a run ID that is not a plain name.
    for action in [
        Action::new("shutdown", "{}".to_string()),
        beat("not json"),
        beat("{\"run_id\":\"../x\",\"rank\":0}"),
        beat(&format!("{{\"run_id\":\"{RUN}\",\"rank\":-1}}")),
    ] {
        assert!(flight.do_action(action).await.is_err());
    }
}

// ---------------------------------------------------------------------------------------------
// Object-store behaviour that differs from local disk: a write that reports failure, latency,
// the listings a restart costs. `TestStore` is an in-memory store with knobs.

mod store {
    use std::fmt;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use futures::future::BoxFuture;
    use futures::stream::BoxStream;
    use futures::FutureExt;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::{
        CopyOptions, GetOptions, GetRange, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
        PutMultipartOptions, PutOptions, PutPayload, PutResult,
    };

    #[derive(Default)]
    pub struct Knobs {
        /// Every request waits this long first, as a round trip to a remote store would.
        pub delay_ms: AtomicU64,
        /// The next this-many writes fail.
        pub fail_puts: AtomicUsize,
        /// A failing write is stored first: it landed, and the reply was lost.
        pub commit_failed_puts: AtomicBool,
        pub puts: AtomicUsize,
        pub lists: AtomicUsize,
        pub gets: AtomicUsize,
        pub range_gets: AtomicUsize,
        pub in_flight: AtomicUsize,
        pub max_in_flight: AtomicUsize,
        /// While set, a write of an `incidents.json` waits (the detectors' report is held back).
        pub hold_incidents: AtomicBool,
        /// How many writes of an `incidents.json` have waited on `hold_incidents`.
        pub held_incidents: AtomicUsize,
        /// While set, every read of an object fails.
        pub fail_gets: AtomicBool,
        /// Whole-object reads (a part read by a rebuild) in progress, and the most at once.
        pub gets_in_flight: AtomicUsize,
        pub max_gets_in_flight: AtomicUsize,
        /// Reads and writes of a rank's `attempts.json` (contract 1), kept apart from the counts and
        /// knobs above, which are about parts, statuses and reports: the order is written once per
        /// new attempt and read once per rank a server takes up.
        pub order_gets: AtomicUsize,
        pub order_puts: AtomicUsize,
        /// The next this-many writes of an `attempts.json` fail (and are not stored).
        pub fail_order_puts: AtomicUsize,
    }

    #[derive(Debug)]
    pub struct TestStore {
        pub inner: Arc<InMemory>,
        pub knobs: Arc<Knobs>,
    }

    impl fmt::Debug for Knobs {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Knobs")
        }
    }

    impl TestStore {
        pub fn new() -> Arc<Self> {
            Arc::new(Self {
                inner: Arc::new(InMemory::new()),
                knobs: Arc::new(Knobs::default()),
            })
        }

        async fn wait(&self) {
            let millis = self.knobs.delay_ms.load(Ordering::SeqCst);
            if millis > 0 {
                tokio::time::sleep(Duration::from_millis(millis)).await;
            }
        }
    }

    impl fmt::Display for TestStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "TestStore")
        }
    }

    impl ObjectStore for TestStore {
        fn put_opts<'life0, 'life1, 'async_trait>(
            &'life0 self,
            location: &'life1 Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> BoxFuture<'async_trait, object_store::Result<PutResult>>
        where
            'life0: 'async_trait,
            'life1: 'async_trait,
            Self: 'async_trait,
        {
            async move {
                if location.as_ref().ends_with("attempts.json") {
                    self.knobs.order_puts.fetch_add(1, Ordering::SeqCst);
                    self.wait().await;
                    let fail = self
                        .knobs
                        .fail_order_puts
                        .try_update(Ordering::SeqCst, Ordering::SeqCst, |left| left.checked_sub(1))
                        .is_ok();
                    return if fail {
                        Err(object_store::Error::Generic {
                            store: "TestStore",
                            source: "injected failure of the attempt order".into(),
                        })
                    } else {
                        self.inner.put_opts(location, payload, opts).await
                    };
                }
                self.knobs.puts.fetch_add(1, Ordering::SeqCst);
                let now = self.knobs.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                self.knobs.max_in_flight.fetch_max(now, Ordering::SeqCst);
                self.wait().await;
                if location.as_ref().ends_with("incidents.json") && self.knobs.hold_incidents.load(Ordering::SeqCst) {
                    self.knobs.held_incidents.fetch_add(1, Ordering::SeqCst);
                    while self.knobs.hold_incidents.load(Ordering::SeqCst) {
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                }
                let fail = self
                    .knobs
                    .fail_puts
                    .try_update(Ordering::SeqCst, Ordering::SeqCst, |left| left.checked_sub(1))
                    .is_ok();
                let result = if fail {
                    if self.knobs.commit_failed_puts.load(Ordering::SeqCst) {
                        let _ = self.inner.put_opts(location, payload, opts).await;
                    }
                    Err(object_store::Error::Generic {
                        store: "TestStore",
                        source: "injected failure".into(),
                    })
                } else {
                    self.inner.put_opts(location, payload, opts).await
                };
                self.knobs.in_flight.fetch_sub(1, Ordering::SeqCst);
                result
            }
            .boxed()
        }

        fn put_multipart_opts<'life0, 'life1, 'async_trait>(
            &'life0 self,
            location: &'life1 Path,
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
            location: &'life1 Path,
            options: GetOptions,
        ) -> BoxFuture<'async_trait, object_store::Result<GetResult>>
        where
            'life0: 'async_trait,
            'life1: 'async_trait,
            Self: 'async_trait,
        {
            async move {
                if location.as_ref().ends_with("attempts.json") {
                    self.knobs.order_gets.fetch_add(1, Ordering::SeqCst);
                    self.wait().await;
                    return self.inner.get_opts(location, options).await;
                }
                let whole = !matches!(options.range, Some(GetRange::Suffix(_) | GetRange::Bounded(_)));
                if whole {
                    self.knobs.gets.fetch_add(1, Ordering::SeqCst);
                    let now = self.knobs.gets_in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    self.knobs.max_gets_in_flight.fetch_max(now, Ordering::SeqCst);
                } else {
                    self.knobs.range_gets.fetch_add(1, Ordering::SeqCst);
                }
                self.wait().await;
                let result = if self.knobs.fail_gets.load(Ordering::SeqCst) {
                    Err(object_store::Error::Generic {
                        store: "TestStore",
                        source: "injected read failure".into(),
                    })
                } else {
                    self.inner.get_opts(location, options).await
                };
                if whole {
                    self.knobs.gets_in_flight.fetch_sub(1, Ordering::SeqCst);
                }
                result
            }
            .boxed()
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.knobs.lists.fetch_add(1, Ordering::SeqCst);
            self.inner.list(prefix)
        }

        fn list_with_delimiter<'life0, 'life1, 'async_trait>(
            &'life0 self,
            prefix: Option<&'life1 Path>,
        ) -> BoxFuture<'async_trait, object_store::Result<ListResult>>
        where
            'life0: 'async_trait,
            'life1: 'async_trait,
            Self: 'async_trait,
        {
            async move {
                self.knobs.lists.fetch_add(1, Ordering::SeqCst);
                self.wait().await;
                self.inner.list_with_delimiter(prefix).await
            }
            .boxed()
        }

        fn copy_opts<'life0, 'life1, 'life2, 'async_trait>(
            &'life0 self,
            from: &'life1 Path,
            to: &'life2 Path,
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
}

use store::TestStore;

/// The part names stored in `store`, in order.
async fn names_in(store: &TestStore) -> Vec<String> {
    let mut names: Vec<String> = store
        .inner
        .list(None)
        .try_collect::<Vec<_>>()
        .await
        .unwrap()
        .into_iter()
        .filter(|object| object.location.extension() == Some("parquet"))
        .map(|object| object.location.to_string())
        .collect();
    names.sort();
    names
}

/// Every stored step of every part in `store`, in part order.
async fn steps_in(store: &TestStore) -> Vec<i64> {
    let mut steps = Vec::new();
    for name in names_in(store).await {
        let bytes = store
            .inner
            .get(&object_store::path::Path::from(name))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        for batch in ParquetRecordBatchReaderBuilder::try_new(bytes)
            .unwrap()
            .build()
            .unwrap()
        {
            let batch = batch.unwrap();
            let column = batch
                .column_by_name("step")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            steps.extend(column.values().iter().copied());
        }
    }
    steps
}

/// Writes a part every 50 steps, nothing else.
fn every_50() -> IngestConfig {
    IngestConfig {
        flush_steps: 50,
        ..manual()
    }
}

#[tokio::test]
async fn a_put_that_failed_after_landing_cannot_leave_an_overlapping_part() {
    use std::sync::atomic::Ordering;
    let store = TestStore::new();
    let server = start_on(store.clone(), every_50(), None).await;
    for rank in [0, 1] {
        // The write of steps 0-49 lands, then the store reports failure: no acknowledgement, and the
        // rows stay buffered, so the buffer takes nothing else until a write of it succeeds.
        store.knobs.fail_puts.store(1, Ordering::SeqCst);
        store.knobs.commit_failed_puts.store(true, Ordering::SeqCst);
        let mut put = open(&server, rank).await;
        put.send(batch(rank, 0..50), 500).await;
        assert_eq!(put.ack().await, Err(Code::Unavailable));
        let mut put = open(&server, rank).await;
        if rank == 0 {
            // The sender comes back with the next rows: the earlier part is written again, with
            // the same name, before they are taken.
            put.send(batch(rank, 50..100), 1000).await;
            assert_eq!(put.ack().await, Ok(1000));
        } else {
            // It resends from its last acknowledged offset: the same rows, which are acknowledged
            // once the retried part is stored, then the next ones.
            put.send(batch(rank, 0..50), 500).await;
            assert_eq!(put.ack().await, Ok(500));
            put.send(batch(rank, 50..100), 1000).await;
            assert_eq!(put.ack().await, Ok(1000));
        }
    }
    let names = names_in(&store).await;
    assert_eq!(
        names,
        [
            "runs/run-a/rank=0/attempt=0/part-00000000-00000049.parquet",
            "runs/run-a/rank=0/attempt=0/part-00000050-00000099.parquet",
            "runs/run-a/rank=1/attempt=0/part-00000000-00000049.parquet",
            "runs/run-a/rank=1/attempt=0/part-00000050-00000099.parquet",
        ]
    );
    // Every step once: no duplicate rows beside the rewritten part.
    let mut steps = steps_in(&store).await;
    steps.sort();
    assert_eq!(steps.len(), 200);
    assert_eq!(
        steps[..100].iter().sum::<i64>() + steps[100..].iter().sum::<i64>(),
        2 * (0..100).sum::<i64>()
    );
    steps.dedup();
    assert_eq!(steps.len(), 100);
}

#[tokio::test]
async fn nothing_is_acknowledged_before_the_put_succeeds_and_the_quota_stays_held() {
    use std::sync::atomic::Ordering;
    let store = TestStore::new();
    store.knobs.fail_puts.store(2, Ordering::SeqCst);
    let server = start_on(store.clone(), every_50(), None).await;
    let mut put = open(&server, 0).await;
    put.send(batch(0, 0..50), 500).await;
    assert_eq!(put.ack().await, Err(Code::Unavailable));
    assert!(names_in(&store).await.is_empty());
    let held = server.ingest.buffered_bytes();
    assert!(
        held > 0,
        "the rows are still buffered and still count against the quota"
    );
    // A second failure changes nothing; the third write goes through.
    let mut put = open(&server, 0).await;
    put.send(batch(0, 0..50), 500).await;
    assert_eq!(put.ack().await, Err(Code::Unavailable));
    assert_eq!(server.ingest.buffered_bytes(), held);
    let mut put = open(&server, 0).await;
    put.send(batch(0, 0..50), 500).await;
    assert_eq!(put.ack().await, Ok(500));
    assert_eq!(server.ingest.buffered_bytes(), 0);
    assert_eq!(steps_in(&store).await, (0..50).collect::<Vec<_>>());
}

/// 32 ranks with one 10-step batch each, buffered and not yet written.
async fn buffered_ranks(server: &TestServer, ranks: i32) -> Vec<Put> {
    let mut puts = Vec::new();
    for rank in 0..ranks {
        let mut put = open(server, rank).await;
        put.send(batch(rank, 0..10), 100).await;
        puts.push(put);
    }
    // Every rank's batch is taken (the watchdog hears of a batch once its rows are buffered),
    // however slow the machine; then every buffer is older than a 50 ms flush interval.
    until_within("every batch is buffered", SLOW_BOUND, || {
        server.ingest.watchdog_statuses().get(RUN).is_some_and(|status| {
            let seen = status["ranks"].as_array().unwrap();
            seen.len() == ranks as usize && seen.iter().all(|rank| rank["last_step"] == 9)
        })
    })
    .await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    puts
}

#[tokio::test]
async fn ranks_are_written_concurrently_so_a_slow_store_does_not_serialise_them() {
    use std::sync::atomic::Ordering;
    let store = TestStore::new();
    store.knobs.delay_ms.store(100, Ordering::SeqCst);
    // No background flusher: the timed flush is driven by hand.
    let config = IngestConfig {
        flush_interval: Duration::from_millis(50),
        detect: false,
        ..manual()
    };
    let ingest = Ingest::new(Storage::new(store.clone(), "".into()), config.clone());
    let server = serve_ingest(ingest, None).await;
    let puts = buffered_ranks(&server, 32).await;
    let started = std::time::Instant::now();
    server.ingest.flush_due().await;
    let timed = started.elapsed();
    assert_eq!(names_in(&store).await.len(), 32);
    // 32 writes of 100 ms one after another would take 3.2 s.
    assert!(timed < Duration::from_millis(1500), "{timed:?}");
    assert!(store.knobs.max_in_flight.load(Ordering::SeqCst) >= 8);
    drop(puts);

    // The same at shutdown, and with a store that fails every write: the time is one retry
    // period for all ranks together, not one per rank, and the rows stay buffered.
    let store = TestStore::new();
    store.knobs.delay_ms.store(100, Ordering::SeqCst);
    store.knobs.fail_puts.store(10_000, Ordering::SeqCst);
    let ingest = Ingest::new(
        Storage::new(store.clone(), "".into()),
        IngestConfig {
            detect: false,
            ..manual()
        },
    );
    let server = serve_ingest(ingest, None).await;
    let puts = buffered_ranks(&server, 32).await;
    let started = std::time::Instant::now();
    assert!(server.ingest.shutdown().await.is_err());
    let stopped = started.elapsed();
    assert!(stopped < Duration::from_millis(1500), "{stopped:?}");
    assert!(server.ingest.buffered_bytes() > 0);
    assert!(names_in(&store).await.is_empty());
    drop(puts);
}

/// A JSON object in `store`, or `Null` while there is none.
async fn json_in(store: &TestStore, path: &str) -> serde_json::Value {
    match store.inner.get(&object_store::path::Path::from(path)).await {
        Ok(found) => serde_json::from_slice(&found.bytes().await.unwrap()).unwrap(),
        Err(_) => serde_json::Value::Null,
    }
}

/// A stop that comes while the timed flush is judging a part it has just stored: the flush is let
/// finish, so that part's `incidents.json` is written, and the stopped status is written last,
/// with the rank judged. (Until Nov 18, 2026 the flusher was aborted, which could leave the
/// report behind the part for good, and the stopped status was written before the buffers.)
#[tokio::test]
async fn a_stop_lets_a_flush_in_progress_finish_and_writes_the_stopped_status_last() {
    use std::sync::atomic::Ordering;
    let store = TestStore::new();
    let config = IngestConfig {
        flush_interval: Duration::from_millis(50),
        watchdog_interval: Duration::from_millis(50),
        ..manual()
    };
    let ingest = Ingest::new(Storage::new(store.clone(), "".into()), config);
    let tasks = [ingest.spawn_flusher(), ingest.spawn_watchdog()];
    let server = serve_ingest(ingest, None).await;
    // Every request to the store takes 300 ms, so the detectors' read of a part and the write of
    // its incidents.json take the better part of a second after the part is acknowledged.
    store.knobs.delay_ms.store(300, Ordering::SeqCst);
    let mut put = open(&server, 0).await;
    put.send(batch(0, 0..10), 100).await;
    assert_eq!(put.ack().await, Ok(100), "stored by the timed flush");
    assert!(json_in(&store, "runs/run-a/rank=0/incidents.json").await.is_null());

    server.ingest.stop(tasks).await.unwrap();
    let incidents = json_in(&store, "runs/run-a/rank=0/incidents.json").await;
    assert_eq!(incidents["last_step"], 9, "{incidents}");
    let status = json_in(&store, "runs/run-a/watchdog.json").await;
    assert_eq!(
        (status["summary"].as_str(), status["server_running"].as_bool()),
        (Some("server_stopped"), Some(false))
    );
    let codes = run_state_codes(&status);
    assert!(
        !codes.iter().any(|code| code == "no_steps" || code == "ranks_unknown"),
        "{status}"
    );
    // Nothing is taken after the stop.
    put.send(batch(0, 10..20), 200).await;
    assert_eq!(put.ack().await, Err(Code::Unavailable));
}

#[tokio::test]
async fn a_restart_costs_one_listing_and_one_footer_read_per_rank() {
    use std::sync::atomic::Ordering;
    let store = TestStore::new();
    let config = IngestConfig {
        detect: false,
        ..every_50()
    };
    {
        let server = start_on(store.clone(), config.clone(), None).await;
        for rank in 0..64 {
            let mut put = open(&server, rank).await;
            for first in [0, 50, 100] {
                put.send(batch(rank, first..first + 50), 1000).await;
                put.ack().await.unwrap();
            }
        }
    }
    assert_eq!(names_in(&store).await.len(), 64 * 3);
    // A new server knows nothing: each rank's first batch reads its watermark.
    let (lists, range_gets, gets) = (&store.knobs.lists, &store.knobs.range_gets, &store.knobs.gets);
    lists.store(0, Ordering::SeqCst);
    range_gets.store(0, Ordering::SeqCst);
    gets.store(0, Ordering::SeqCst);
    store.knobs.order_gets.store(0, Ordering::SeqCst);
    let server = start_on(store.clone(), config, None).await;
    for rank in 0..64 {
        let mut put = open(&server, rank).await;
        put.send(batch(rank, 150..200), 2000).await;
        assert_eq!(put.ack().await, Ok(2000));
    }
    assert_eq!(lists.load(Ordering::SeqCst), 64, "one listing of the attempt per rank");
    assert_eq!(
        range_gets.load(Ordering::SeqCst),
        64,
        "one suffix read of the newest part per rank"
    );
    assert_eq!(gets.load(Ordering::SeqCst), 0, "no part is read whole");
    // And one small read of each rank's `attempts.json`, which the first server wrote (contract 1):
    // the rank's order is known, so nothing is listed for it.
    assert_eq!(store.knobs.order_gets.load(Ordering::SeqCst), 64);
}

// ---------------------------------------------------------------------------------------------
// The watchdog across a restart, on local disk.

fn read_status(root: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(root.join("runs").join(RUN).join("watchdog.json")).unwrap()).unwrap()
}

/// Two ranks of a run of two: rank 0 steps on, rank 1 sent one batch and went quiet. Returns once
/// rank 1 is silent and the file says so.
async fn a_run_with_a_silent_rank(server: &TestServer) -> (Put, Put) {
    let mut rank0 = open(server, 0).await;
    let mut rank1 = open(server, 1).await;
    for rank in [&mut rank0, &mut rank1] {
        let n = rank.rank;
        rank.send_with_metadata(batch(n, 0..50), world_metadata(n, 2, 100))
            .await;
    }
    for step in [50, 100, 150, 200, 250, 300] {
        tokio::time::sleep(Duration::from_millis(150)).await;
        rank0
            .send_with_metadata(batch(0, step..step + 50), world_metadata(0, 2, 100 + step))
            .await;
    }
    until("rank 1 silent", || {
        rank_states(server)
            .iter()
            .any(|(rank, state, _)| *rank == 1 && state == "silent")
    })
    .await;
    (rank0, rank1)
}

#[tokio::test]
async fn a_killed_servers_silent_rank_gets_a_fresh_window_and_keeps_the_quiet_time_it_shows() {
    let dir = tempfile::tempdir().unwrap();
    let old = start(dir.path(), watch_config(400), None).await;
    let (_rank0, _rank1) = a_run_with_a_silent_rank(&old).await;
    old.ingest.write_watchdog().await;
    let before = read_status(dir.path());
    assert_eq!(before["summary"], "degraded");
    assert_eq!(before["incidents"][0]["kind"], "rank_silent");
    let quiet_before = before["ranks"][1]["seconds_since_progress"].as_f64().unwrap();
    assert!(quiet_before >= 0.4, "{quiet_before}");
    // The old server is killed: its last file stays, and nothing more is written by it. (Its open
    // streams are not the new server's.)
    drop(old);
    tokio::time::sleep(Duration::from_millis(300)).await; // an outage

    let new = start(dir.path(), watch_config(400), None).await;
    assert_eq!(new.ingest.restore_watchdog().await.unwrap(), 1);
    // The first status after the restart, with no sender reconnected: every rank has a fresh
    // window from the restart (decided Nov 17, 2026), so no stall yet; the quiet time it shows is
    // the one it had (not reset, and not grown by the outage).
    let status = &new.ingest.watchdog_statuses()[RUN];
    assert_eq!(status["summary"], "active");
    assert!(status["incidents"].as_array().unwrap().is_empty(), "{status}");
    assert_eq!(status["ranks"][1]["state"], "reconnecting");
    assert_eq!(status["ranks"][1]["stream"], "dropped");
    assert_eq!(status["ranks"][1]["last_step"], 49);
    assert_eq!(status["ranks"][0]["last_step"], before["ranks"][0]["last_step"]);
    let quiet_after = status["ranks"][1]["seconds_since_progress"].as_f64().unwrap();
    assert!(
        quiet_after >= quiet_before && quiet_after < quiet_before + 0.25,
        "{quiet_before} -> {quiet_after}"
    );
    assert_eq!(status["world_size"], 2);
    // The file now says the new server is watching.
    new.ingest.write_watchdog().await;
    let after = read_status(dir.path());
    assert_eq!(after["server_running"], true);
    assert!(after["written_at_ms"].as_i64() > before["written_at_ms"].as_i64());

    // The silent rank's sender returns with nothing new: silent again once its window from the
    // restart has passed. The other rank's batch makes it active again.
    let _rank1 = open(&new, 1).await;
    until("rank 1 silent again", || rank_states(&new)[1].1 == "silent").await;
    let mut rank0 = open(&new, 0).await;
    rank0
        .send_with_metadata(batch(0, 400..450), world_metadata(0, 2, 500))
        .await;
    until("rank 0's batch", || {
        new.ingest.watchdog_statuses()[RUN]["ranks"][0]["last_step"] == 449
    })
    .await;
    assert_eq!(rank_states(&new)[0].1, "active");
}

#[tokio::test]
async fn a_clean_stop_says_server_stopped_and_the_stall_comes_back_after_the_fresh_window() {
    let dir = tempfile::tempdir().unwrap();
    let old = start(dir.path(), watch_config(400), None).await;
    let (_rank0, _rank1) = a_run_with_a_silent_rank(&old).await;
    old.ingest.write_watchdog_stopped().await;
    let stopped = read_status(dir.path());
    assert_eq!(
        (
            stopped["summary"].as_str(),
            stopped["incidents"].as_array().map(Vec::len)
        ),
        (Some("server_stopped"), Some(0))
    );
    drop(old);

    let new = start(dir.path(), watch_config(400), None).await;
    assert_eq!(new.ingest.restore_watchdog().await.unwrap(), 1);
    let status = &new.ingest.watchdog_statuses()[RUN];
    assert_eq!(
        (status["summary"].as_str(), status["server_running"].as_bool()),
        (Some("active"), Some(true))
    );
    assert_eq!(status["ranks"][1]["state"], "reconnecting");
    // No sender comes back: after the window both ranks are lost and the run is stalled.
    until("the stall back", || {
        new.ingest.watchdog_statuses()[RUN]["summary"] == "stalled"
    })
    .await;
    let status = &new.ingest.watchdog_statuses()[RUN];
    assert_eq!(
        (&status["incidents"][0]["kind"], &status["incidents"][0]["assessment"]),
        (&serde_json::json!("job_stalled"), &serde_json::json!("lost_connection"))
    );
    new.ingest.write_watchdog().await;
    assert_eq!(read_status(dir.path())["summary"], "stalled");
}

#[tokio::test]
async fn a_finished_run_is_left_alone_by_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let old = start(dir.path(), watch_config(60_000), None).await;
    let mut put = open(&old, 0).await;
    put.send_with_metadata(batch(0, 0..10), world_metadata(0, 1, 50)).await;
    let _ = put.finish().await;
    until("finished", || rank_states(&old)[0].1 == "finished").await;
    old.ingest.write_watchdog_stopped().await;
    let path = dir.path().join("runs").join(RUN).join("watchdog.json");
    let written = std::fs::metadata(&path).unwrap().modified().unwrap();
    drop(old);
    tokio::time::sleep(Duration::from_millis(50)).await;

    let new = start(dir.path(), watch_config(60_000), None).await;
    assert_eq!(new.ingest.restore_watchdog().await.unwrap(), 0);
    assert!(new.ingest.watchdog_statuses().is_empty());
    new.ingest.write_watchdog().await;
    new.ingest.write_watchdog_stopped().await;
    assert_eq!(std::fs::metadata(&path).unwrap().modified().unwrap(), written);
    assert_eq!(read_status(dir.path())["summary"], "finished");
}

#[tokio::test]
async fn an_abandoned_run_stops_being_refreshed_and_is_not_taken_up_again() {
    let dir = tempfile::tempdir().unwrap();
    let config = IngestConfig {
        status_refresh: Duration::from_millis(100),
        abandon_after: Duration::from_millis(300),
        ..watch_config(60_000)
    };
    let old = start(dir.path(), config.clone(), None).await;
    let mut put = open(&old, 0).await;
    put.send_with_metadata(batch(0, 0..10), world_metadata(0, 1, 50)).await;
    until("the rank is seen", || rank_states(&old).len() == 1).await;
    // The stream is cut and nothing comes back.
    put.send_with_metadata(batch(0, 10..20), world_metadata(0, 1, 9).replace(RUN, "another-run"))
        .await;
    assert_eq!(put.ack().await, Err(Code::InvalidArgument));
    until("dropped", || rank_states(&old)[0].2 == "dropped").await;
    tokio::time::sleep(Duration::from_millis(350)).await;
    old.ingest.write_watchdog().await;
    let abandoned = read_status(dir.path());
    assert_eq!(abandoned["abandoned"], true);
    // From here on the status is not rewritten, although its refresh time has long passed.
    tokio::time::sleep(Duration::from_millis(250)).await;
    old.ingest.write_watchdog().await;
    assert_eq!(read_status(dir.path())["written_at_ms"], abandoned["written_at_ms"]);
    drop(old);

    let new = start(dir.path(), config, None).await;
    assert_eq!(new.ingest.restore_watchdog().await.unwrap(), 0);
    assert!(new.ingest.watchdog_statuses().is_empty());
}

#[tokio::test]
async fn a_status_that_cannot_be_read_does_not_stop_the_server_or_the_other_runs() {
    let dir = tempfile::tempdir().unwrap();
    {
        let old = start(dir.path(), watch_config(60_000), None).await;
        let mut put = open(&old, 0).await;
        put.send_with_metadata(batch(0, 0..10), world_metadata(0, 1, 50)).await;
        until("the rank is seen", || rank_states(&old).len() == 1).await;
        old.ingest.write_watchdog().await;
    }
    // Another run's file is not JSON, a third is JSON of the wrong shape.
    for (run, content) in [
        ("broken", "{not json"),
        (
            "other",
            "{\"run_id\":\"other\",\"schema_version\":1,\"ranks\":[{}],\"restart\":[{}]}",
        ),
    ] {
        let directory = dir.path().join("runs").join(run);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("watchdog.json"), content).unwrap();
    }
    let new = start(dir.path(), watch_config(60_000), None).await;
    assert_eq!(new.ingest.restore_watchdog().await.unwrap(), 1);
    assert_eq!(new.ingest.watchdog_statuses().keys().collect::<Vec<_>>(), [RUN]);
}

/// `schema()` with a `data_json` column.
fn data_schema() -> Arc<Schema> {
    let mut fields = schema().fields().to_vec();
    fields.push(Arc::new(Field::new("data_json", DataType::Utf8, true)));
    Arc::new(Schema::new(fields))
}

/// Rows in `data_schema()` that drew one batch per step, its hash `hash(step)`; no `data_json`
/// where that is nothing.
fn data_batch(rank: i32, steps: std::ops::Range<i64>, hash: impl Fn(i64) -> Option<String>) -> RecordBatch {
    let steps: Vec<i64> = steps.collect();
    let n = steps.len();
    let data: Vec<Option<String>> = steps
        .iter()
        .map(|step| hash(*step).map(|hash| format!(r#"{{"batches":1,"batch_hash":"{hash}"}}"#)))
        .collect();
    RecordBatch::try_new(
        data_schema(),
        vec![
            Arc::new(Int32Array::from(vec![1; n])),
            Arc::new(StringArray::from(vec![RUN; n])),
            Arc::new(Int32Array::from(vec![rank; n])),
            Arc::new(Int64Array::from(steps.clone())),
            Arc::new(Float64Array::from(
                steps.iter().map(|step| *step as f64).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(vec!["{}"; n])),
            Arc::new(StringArray::from(data)),
        ],
    )
    .unwrap()
}

/// The status says which ranks' batch hashes the server compared, found or not, so that
/// "compared, none found" can be told from "not compared": before, it named them only inside a
/// finding, and a restarted server's status read the same as a clean comparison. A restarted
/// server has compared nothing until a rank's store is read again.
#[tokio::test]
async fn the_status_names_the_ranks_whose_batches_were_compared_found_or_not() {
    let dir = tempfile::tempdir().unwrap();
    let config = IngestConfig {
        flush_steps: 10,
        ..watch_config(60_000)
    };
    let own = |rank: i32| move |step: i64| Some(format!("r{rank}-{step}"));
    let compared = |server: &TestServer| server.ingest.watchdog_statuses()[RUN]["ranks_compared"].clone();
    let old = start(dir.path(), config.clone(), None).await;
    let mut rank0 = open_with(&old.address, RUN, 0, None, data_schema()).await;
    let mut rank1 = open_with(&old.address, RUN, 1, None, data_schema()).await;
    rank0.send(data_batch(0, 0..10, |_| None), 1).await; // no data: nothing to compare
    assert_eq!(rank0.ack().await, Ok(1));
    assert_eq!(compared(&old), serde_json::json!([]));
    rank0.send(data_batch(0, 10..20, own(0)), 2).await;
    assert_eq!(rank0.ack().await, Ok(2));
    until_within("rank 0 compared", SLOW_BOUND, || {
        compared(&old) == serde_json::json!([0])
    })
    .await;
    rank1.send(data_batch(1, 0..10, own(1)), 1).await;
    assert_eq!(rank1.ack().await, Ok(1));
    until_within("both ranks compared", SLOW_BOUND, || {
        compared(&old) == serde_json::json!([0, 1])
    })
    .await;
    let status = &old.ingest.watchdog_statuses()[RUN];
    assert_eq!(status["duplicate_batches"], serde_json::Value::Null, "{status}");
    assert!(status["incidents"].as_array().unwrap().is_empty(), "{status}");
    old.ingest.write_watchdog().await;
    assert_eq!(read_status(dir.path())["ranks_compared"], serde_json::json!([0, 1]));
    // The server is killed: its last file stays. (Its open streams are not the new server's.)
    let _old_streams = (rank0, rank1);
    drop(old);

    let new = start(dir.path(), config, None).await;
    assert_eq!(new.ingest.restore_watchdog().await.unwrap(), 1);
    assert_eq!(
        compared(&new),
        serde_json::json!([]),
        "nothing read since this server started"
    );
    new.ingest.write_watchdog().await;
    assert_eq!(read_status(dir.path())["ranks_compared"], serde_json::json!([]));
    // A rank that sends again has its store read, and its batches are compared again.
    let mut rank1 = open_with(&new.address, RUN, 1, None, data_schema()).await;
    rank1.send(data_batch(1, 10..20, own(1)), 2).await;
    assert_eq!(rank1.ack().await, Ok(2));
    until_within("rank 1 compared", SLOW_BOUND, || {
        compared(&new) == serde_json::json!([1])
    })
    .await;
    // Rank 0 draws rank 1's batches on 20..25: a duplicate, named with the same ranks compared.
    let mut rank0 = open_with(&new.address, RUN, 0, None, data_schema()).await;
    rank1.send(data_batch(1, 20..30, own(1)), 3).await;
    assert_eq!(rank1.ack().await, Ok(3));
    rank0
        .send(
            data_batch(0, 20..30, |step| if step < 25 { own(1)(step) } else { own(0)(step) }),
            3,
        )
        .await;
    assert_eq!(rank0.ack().await, Ok(3));
    until_within("the duplicate found", SLOW_BOUND, || {
        new.ingest.watchdog_statuses()[RUN]["duplicate_batches"].is_object()
    })
    .await;
    let status = &new.ingest.watchdog_statuses()[RUN];
    assert_eq!(status["ranks_compared"], serde_json::json!([0, 1]));
    assert_eq!(
        status["duplicate_batches"],
        serde_json::json!({"step": 20, "ranks": [0, 1], "duplicated_steps": 5, "ranks_compared": [0, 1]})
    );
    assert_eq!(status["incidents"][0]["kind"], "duplicate_batches");
}

/// A server process that is killed when dropped, so a failed assertion leaves none running.
struct Reaped(std::process::Child);

impl Reaped {
    fn kill(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl Drop for Reaped {
    fn drop(&mut self) {
        self.kill();
    }
}

/// The real binary, killed with SIGKILL: its last file stays, and the next one takes the run up
/// with a fresh window for its rank.
#[cfg(unix)]
#[tokio::test]
async fn the_binary_killed_with_sigkill_takes_its_run_up_with_a_fresh_window() {
    use std::io::{BufRead, BufReader};
    let dir = tempfile::tempdir().unwrap();
    let launch = || {
        let mut child = Command::new(env!("CARGO_BIN_EXE_tensorquill-server"))
            .args([
                "--listen",
                "127.0.0.1:0",
                "--silent-seconds",
                "1",
                "--critical-seconds",
                "2",
            ])
            .args(["--watchdog-seconds", "0.2", "--store"])
            .arg(dir.path())
            .env_remove("TENSORQUILL_TOKEN")
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut line = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let address = format!("http://{}", line.trim().rsplit(' ').next().unwrap());
        (Reaped(child), address)
    };
    let (mut child, address) = launch();
    let mut put = open_at(&address, RUN, 0, None).await;
    put.send_with_metadata(batch(0, 0..10), world_metadata(0, 1, 50)).await;
    let path = dir.path().join("runs").join(RUN).join("watchdog.json");
    // Quiet for longer than the floor: stalled. Wait for the file to say so.
    let mut before = serde_json::Value::Null;
    for _ in 0..100 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if let Ok(bytes) = std::fs::read(&path) {
            before = serde_json::from_slice(&bytes).unwrap_or_default();
            if before["summary"] == "stalled" {
                break;
            }
        }
    }
    assert_eq!(before["summary"], "stalled", "{before}");
    child.kill();
    drop(put);
    // The file as the dead server left it: the first newer one is the new server's, and its quiet
    // time is compared with this one (a dead server writes nothing more).
    let before: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(before["summary"], "stalled");
    let quiet_before = before["ranks"][0]["seconds_since_progress"].as_f64().unwrap();
    tokio::time::sleep(Duration::from_secs(3)).await; // an outage longer than the floor and than the bound below

    let (_child, address) = launch(); // restores before it listens
    let mut first = serde_json::Value::Null;
    for _ in 0..100 {
        if let Ok(bytes) = std::fs::read(&path) {
            let status: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_default();
            if status["written_at_ms"].as_i64() > before["written_at_ms"].as_i64() {
                first = status;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // The first file the new server wrote has no stall: the rank has a fresh window from the
    // restart (decided Nov 17, 2026). It shows the quiet time it had.
    assert_eq!(first["server_running"], true, "{first}");
    assert_eq!(first["summary"], "active");
    assert_eq!(first["incidents"], serde_json::json!([]));
    assert_eq!(first["ranks"][0]["state"], "reconnecting");
    assert_eq!(first["ranks"][0]["last_step"], 9);
    let quiet = first["ranks"][0]["seconds_since_progress"].as_f64().unwrap();
    assert!(
        quiet >= quiet_before && quiet < quiet_before + 1.5,
        "{quiet_before} -> {quiet}: the outage must not count"
    );
    // A sender comes back and steps on: the run is active again.
    let mut put = open_at(&address, RUN, 0, None).await;
    put.send_with_metadata(batch(0, 10..20), world_metadata(0, 1, 90)).await;
    let mut active = false;
    for _ in 0..100 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        active = std::fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .is_some_and(|status| status["summary"] == "active" && status["ranks"][0]["last_step"] == 19);
        if active {
            break;
        }
    }
    assert!(active);
    drop(put);
}

/// Statuses for `runs` unfinished runs of 8 ranks each, written to `store` as a server would.
async fn write_unfinished_runs(store: &Arc<TestStore>, runs: usize) {
    use tensorquill_server::watchdog::{Watchdog, WatchdogConfig};
    let storage = Storage::new(store.clone(), "".into());
    let dog = Watchdog::new(WatchdogConfig::default());
    let t0 = std::time::Instant::now();
    for run in 0..runs {
        for rank in 0..8 {
            let key = (format!("run-{run}"), rank);
            dog.stream_opened(&key, t0);
            dog.batch(&key, 1, Some(49), Some(8), t0);
        }
    }
    for (run, status) in dog.statuses(t0, true) {
        storage.write_status(&run, &status).await.unwrap();
    }
}

#[tokio::test]
async fn restoring_many_runs_is_one_listing_and_concurrent_small_reads() {
    use std::sync::atomic::Ordering;
    let store = TestStore::new();
    write_unfinished_runs(&store, 160).await;
    store.knobs.delay_ms.store(25, Ordering::SeqCst);
    store.knobs.lists.store(0, Ordering::SeqCst);
    let ingest = Ingest::new(Storage::new(store.clone(), "".into()), IngestConfig::default());
    let started = std::time::Instant::now();
    assert_eq!(ingest.restore_watchdog().await.unwrap(), 160);
    let took = started.elapsed();
    assert_eq!(
        store.knobs.lists.load(Ordering::SeqCst),
        1,
        "one delimiter listing of runs/"
    );
    assert_eq!(
        store.knobs.gets.load(Ordering::SeqCst),
        160,
        "one read of one status per run, no part"
    );
    // And one small read per rank taken up, of its `attempts.json` (contract 1): none of these
    // ranks has one, so none is listed now; that waits for the rank's first beat or batch.
    assert_eq!(store.knobs.order_gets.load(Ordering::SeqCst), 160 * 8);
    // 160 + 1,280 reads of 25 ms one after another would take 36 s; 32 at a time, about 1.1 s.
    assert!(took < Duration::from_millis(3000), "{took:?}");
    assert_eq!(ingest.watchdog_statuses().len(), 160);
}

/// Prints what a remote store's latency costs (simulated with sleeps, not measured against S3).
/// `cargo test --release --test ingest_test simulated_latency -- --ignored --nocapture`
#[tokio::test]
#[ignore = "a measurement, not a check"]
async fn simulated_latency_report() {
    use std::sync::atomic::Ordering;
    for latency in [20u64, 50, 100] {
        // Restore of 1,000 runs.
        let store = TestStore::new();
        write_unfinished_runs(&store, 1000).await;
        store.knobs.delay_ms.store(latency, Ordering::SeqCst);
        let ingest = Ingest::new(Storage::new(store.clone(), "".into()), IngestConfig::default());
        let started = std::time::Instant::now();
        let restored = ingest.restore_watchdog().await.unwrap();
        println!(
            "{latency} ms per request: restore of {restored} runs {:?}",
            started.elapsed()
        );

        // 64 ranks, 20 parts of 500 steps each (10,000 steps), a restart: the first batch of each rank
        // reads its watermark (1 listing + 1 suffix read), and with detectors on the rank is replayed:
        // 1 listing + 20 whole-part reads, one after another per rank.
        let config = IngestConfig {
            flush_steps: 500,
            ..manual()
        };
        for (detect, label) in [(false, "watermark only"), (true, "watermark + detector replay")] {
            let store = TestStore::new();
            {
                let server = start_on(store.clone(), config.clone(), None).await;
                for rank in 0..64 {
                    let mut put = open(&server, rank).await;
                    for first in (0..10_000).step_by(500) {
                        put.send(batch(rank, first..first + 500), 1).await;
                        put.ack().await.unwrap();
                    }
                }
            }
            store.knobs.delay_ms.store(latency, Ordering::SeqCst);
            let server = start_on(
                store.clone(),
                IngestConfig {
                    detect,
                    ..config.clone()
                },
                None,
            )
            .await;
            let started = std::time::Instant::now();
            let mut puts = Vec::new();
            for rank in 0..64 {
                let mut put = open(&server, rank).await;
                put.send(batch(rank, 10_000..10_500), 2).await;
                puts.push(put);
            }
            for put in &mut puts {
                put.ack().await.unwrap();
            }
            let first = started.elapsed();
            // The detectors replay after the first part is acknowledged, and the rank's next part waits for it.
            for (rank, put) in puts.iter_mut().enumerate() {
                put.send(batch(rank as i32, 10_500..11_000), 3).await;
            }
            for put in &mut puts {
                put.ack().await.unwrap();
            }
            let second = started.elapsed() - first;
            println!(
                "{latency} ms per request: 64 ranks after a restart ({label}): first part acknowledged after {first:?}, the next one {second:?} later"
            );
        }
    }
}

// --- The run manifest -------------------------------------------------------------------

fn manifest_metadata(end: i64, manifest: &serde_json::Value) -> String {
    serde_json::json!({
        "schema_version": 1, "run_id": RUN, "rank": 0, "min_step": null, "max_step": null,
        "end_byte_offset": end, "run_manifest": manifest,
    })
    .to_string()
}

fn stored_manifest(root: &Path) -> Option<serde_json::Value> {
    let bytes = std::fs::read(root.join("runs").join(RUN).join("run.json")).ok()?;
    Some(serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn a_manifest_on_a_batch_is_written_as_run_json_and_replaced_when_it_changes() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), manual(), None).await;
    let first = serde_json::json!({
        "schema_version": 1, "project": "granite", "name": "baseline", "run_id": RUN,
        "config": {"lr": 0.001}, "a_field_this_server_does_not_know": [1, 2],
    });

    let mut put = open(&server, 0).await;
    put.send_with_metadata(batch(0, 0..5), manifest_metadata(5, &first))
        .await;
    // Written while the stream is still open, before any part is: a listing can name a run
    // from its first batch.
    until("run.json", || stored_manifest(dir.path()).is_some()).await;
    assert_eq!(
        stored_manifest(dir.path()).unwrap(),
        first,
        "stored as the sender gave it"
    );
    assert_eq!(put.finish().await, vec![5]);
    let path = dir.path().join("runs").join(RUN).join("run.json");
    let written = std::fs::metadata(&path).unwrap().modified().unwrap();

    // The same manifest on a later stream is not written again.
    tokio::time::sleep(Duration::from_millis(20)).await;
    let mut put = open(&server, 0).await;
    put.send_with_metadata(batch(0, 5..10), manifest_metadata(10, &first))
        .await;
    assert_eq!(put.finish().await, vec![10]);
    assert_eq!(std::fs::metadata(&path).unwrap().modified().unwrap(), written);

    // The run starts again under its run ID with another name: the file follows.
    let mut second = first.clone();
    second["name"] = "after-the-fix".into();
    let mut put = open(&server, 0).await;
    put.send_with_metadata(batch(0, 10..15), manifest_metadata(15, &second))
        .await;
    assert_eq!(put.finish().await, vec![15]);
    assert_eq!(stored_manifest(dir.path()).unwrap()["name"], "after-the-fix");
    assert_eq!(stored_steps(dir.path())[&0], (0..15).collect::<Vec<_>>());
}

#[tokio::test]
async fn a_sender_older_than_the_manifest_and_one_newer_than_this_server_both_store_their_rows() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), manual(), None).await;

    // Older: the five keys and no manifest.
    let mut put = open(&server, 0).await;
    put.send(batch(0, 0..5), 5).await;
    assert_eq!(put.finish().await, vec![5]);
    assert_eq!(
        stored_manifest(dir.path()),
        None,
        "no manifest was sent, so none is written"
    );

    // Newer: a key this server has never heard of. An older server meets `run_manifest` this way.
    let mut put = open(&server, 0).await;
    let newer = format!(
        "{{\"schema_version\":1,\"run_id\":\"{RUN}\",\"rank\":0,\"end_byte_offset\":10,\"a_key_from_a_later_sender\":{{\"x\":1}}}}"
    );
    put.send_with_metadata(batch(0, 5..10), newer).await;
    assert_eq!(put.finish().await, vec![10]);
    assert_eq!(stored_steps(dir.path())[&0], (0..10).collect::<Vec<_>>());
}

#[tokio::test]
async fn a_manifest_that_cannot_be_stored_never_refuses_the_batch() {
    use std::sync::atomic::Ordering;
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), manual(), None).await;
    let mut end = 0;
    let too_large = serde_json::json!({"run_id": RUN, "config": {"blob": "x".repeat(70_000)}});
    for bad in [
        serde_json::json!("not an object"),
        serde_json::json!({"run_id": "another-run", "project": "p"}),
        serde_json::json!({"run_id": "../../escape", "project": "p"}),
        too_large,
    ] {
        let mut put = open(&server, 0).await;
        put.send_with_metadata(batch(0, end..end + 5), manifest_metadata(end + 5, &bad))
            .await;
        end += 5;
        assert_eq!(put.finish().await, vec![end], "{bad:.80}");
    }
    assert_eq!(stored_manifest(dir.path()), None);
    assert_eq!(stored_steps(dir.path())[&0], (0..end).collect::<Vec<_>>());
    let listed: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(listed, ["runs"], "nothing is written outside the run's folder");

    // The store refuses the manifest's write: the rows are stored and acknowledged all the same,
    // and the next stream's manifest is written.
    let store = TestStore::new();
    let server = start_on(store.clone(), manual(), None).await;
    let good = serde_json::json!({"run_id": RUN, "project": "p", "name": "n"});
    let manifest_path = object_store::path::Path::from(format!("runs/{RUN}/run.json"));
    let mut put = open(&server, 0).await;
    put.send(batch(0, 0..5), 5).await;
    assert_eq!(put.finish().await, vec![5]);
    store.knobs.fail_puts.store(1, Ordering::SeqCst);
    // Steps already stored: nothing to write but the manifest, whose write fails.
    let mut put = open(&server, 0).await;
    put.send_with_metadata(batch(0, 0..5), manifest_metadata(5, &good))
        .await;
    assert_eq!(put.finish().await.last(), Some(&5));
    assert_eq!(
        store.knobs.fail_puts.load(Ordering::SeqCst),
        0,
        "the manifest's write was tried and failed"
    );
    assert!(store.inner.get(&manifest_path).await.is_err());
    let mut put = open(&server, 0).await;
    put.send_with_metadata(batch(0, 5..10), manifest_metadata(10, &good))
        .await;
    assert_eq!(put.finish().await, vec![10]);
    let stored = store.inner.get(&manifest_path).await.unwrap().bytes().await.unwrap();
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&stored).unwrap(), good);
    assert_eq!(steps_in(&store).await, (0..10).collect::<Vec<_>>());
}

// --- Lane U6 (Nov 19, 2026): a run read as healthy on a report of other rows -------------------

/// Writes the status and reads it back from `store`.
async fn written_status(server: &TestServer, store: &TestStore) -> serde_json::Value {
    server.ingest.write_watchdog().await;
    json_in(store, &format!("runs/{RUN}/watchdog.json")).await
}

/// Whether the status's run state counts rank 0 as judged by a current report.
fn judged(status: &serde_json::Value) -> bool {
    let codes = run_state_codes(status);
    !codes.iter().any(|code| code == "ranks_unknown" || code == "no_steps")
}

/// Writes the status until `ready` holds for it, for at most `SLOW_BOUND`.
async fn status_until(
    server: &TestServer,
    store: &TestStore,
    what: &str,
    ready: impl Fn(&serde_json::Value) -> bool,
) -> serde_json::Value {
    let deadline = tokio::time::Instant::now() + SLOW_BOUND;
    loop {
        let status = written_status(server, store).await;
        if ready(&status) {
            return status;
        }
        assert!(tokio::time::Instant::now() < deadline, "{what}: {status}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// RS04. A rank's cached report judges it only while it covers what the status shows: the
/// attempt the status names (not one seen in the beats alone) and the last part this server
/// stored (here with no grace). Two flushes with the second report's write held back and the
/// attempt changed, then a third with the attempt the same: while the report is held the rank is
/// not judged. Rows waiting in the buffer, not stored yet, do not make a current report stale.
#[tokio::test]
async fn a_cached_report_of_other_rows_than_the_status_shows_does_not_judge_the_rank() {
    use std::sync::atomic::Ordering;
    let store = TestStore::new();
    let config = IngestConfig {
        report_grace: Duration::ZERO,
        ..every_50()
    };
    let server = start_on(store.clone(), config, None).await;
    let mut put = open_attempts(&server).await;
    put.send(attempt_batch(&rows(1, 0..50)), 100).await;
    assert_eq!(put.ack().await, Ok(100));
    status_until(&server, &store, "attempt 1 judged", judged).await;

    // Attempt 2 resumes at step 30; its report's write is held back.
    store.knobs.hold_incidents.store(true, Ordering::SeqCst);
    put.send(attempt_batch(&rows(2, 30..80)), 200).await;
    assert_eq!(put.ack().await, Ok(200)); // acknowledged once stored, before it is judged
    until_within("the second report held", SLOW_BOUND, || {
        store.knobs.held_incidents.load(Ordering::SeqCst) == 1
    })
    .await;
    let status = written_status(&server, &store).await;
    assert_eq!(
        (&status["ranks"][0]["attempt"], &status["ranks"][0]["last_step"]),
        (&serde_json::json!(2), &serde_json::json!(79))
    );
    assert!(!judged(&status), "{}", status["run_state"]);
    assert_eq!(status["run_state"]["state"], "unknown");
    store.knobs.hold_incidents.store(false, Ordering::SeqCst);
    status_until(&server, &store, "attempt 2 judged", judged).await;

    // The same attempt goes on; the report of its next part is held back.
    store.knobs.hold_incidents.store(true, Ordering::SeqCst);
    put.send(attempt_batch(&rows(2, 80..130)), 300).await;
    assert_eq!(put.ack().await, Ok(300));
    until_within("the third report held", SLOW_BOUND, || {
        store.knobs.held_incidents.load(Ordering::SeqCst) == 2
    })
    .await;
    let status = written_status(&server, &store).await;
    assert_eq!(status["ranks"][0]["last_step"], 129);
    assert!(!judged(&status), "{}", status["run_state"]);
    store.knobs.hold_incidents.store(false, Ordering::SeqCst);
    status_until(&server, &store, "the third part judged", judged).await;

    // Rows in the buffer, not yet stored: the report is still current for what is stored.
    put.send(attempt_batch(&rows(2, 130..140)), 400).await;
    until_within("the buffered rows seen", SLOW_BOUND, || {
        server.ingest.watchdog_statuses()[RUN]["ranks"][0]["last_step"] == 139
    })
    .await;
    let status = written_status(&server, &store).await;
    assert!(judged(&status), "{}", status["run_state"]);
    assert_eq!(put.finish().await, vec![400]);
}

/// RS04, with the usual grace: a report a moment behind the part just stored still judges its
/// rank, so a run that streams normally does not read unknown after each part; a report of an
/// earlier attempt than the status shows never does.
#[tokio::test]
async fn a_report_a_moment_behind_the_store_judges_its_rank_for_the_grace_and_one_of_another_attempt_never() {
    use std::sync::atomic::Ordering;
    let store = TestStore::new();
    let server = start_on(store.clone(), every_50(), None).await; // the usual 30 s grace
    let mut put = open_attempts(&server).await;
    put.send(attempt_batch(&rows(1, 0..50)), 100).await;
    assert_eq!(put.ack().await, Ok(100));
    status_until(&server, &store, "attempt 1 judged", judged).await;
    store.knobs.hold_incidents.store(true, Ordering::SeqCst);
    put.send(attempt_batch(&rows(1, 50..100)), 200).await;
    assert_eq!(put.ack().await, Ok(200));
    until_within("the second report held", SLOW_BOUND, || {
        store.knobs.held_incidents.load(Ordering::SeqCst) == 1
    })
    .await;
    let status = written_status(&server, &store).await;
    assert_eq!(status["ranks"][0]["last_step"], 99);
    assert!(judged(&status), "{}", status["run_state"]);
    store.knobs.hold_incidents.store(false, Ordering::SeqCst);
    // The held report is written (the rank reads judged throughout, so wait for the file).
    let deadline = tokio::time::Instant::now() + SLOW_BOUND;
    while json_in(&store, &format!("runs/{RUN}/rank=0/incidents.json")).await["last_step"] != 99 {
        assert!(tokio::time::Instant::now() < deadline, "the second report written");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    store.knobs.hold_incidents.store(true, Ordering::SeqCst);
    put.send(attempt_batch(&rows(2, 0..50)), 300).await;
    assert_eq!(put.ack().await, Ok(300));
    until_within("the third report held", SLOW_BOUND, || {
        store.knobs.held_incidents.load(Ordering::SeqCst) == 2
    })
    .await;
    let status = written_status(&server, &store).await;
    assert_eq!(status["ranks"][0]["attempt"], 2);
    assert!(!judged(&status), "{}", status["run_state"]);
    store.knobs.hold_incidents.store(false, Ordering::SeqCst);
    status_until(&server, &store, "attempt 2 judged", judged).await;
    put.finish().await;
}

fn data_attempt_schema() -> Arc<Schema> {
    let mut fields = attempt_schema().fields().to_vec();
    fields.push(Arc::new(Field::new("data_json", DataType::Utf8, true)));
    Arc::new(Schema::new(fields))
}

/// Rows of rank 0 in `data_attempt_schema()` that drew one batch per step, hashed by step.
fn data_attempt_batch(attempt: i64, steps: std::ops::Range<i64>) -> RecordBatch {
    let steps: Vec<i64> = steps.collect();
    let n = steps.len();
    RecordBatch::try_new(
        data_attempt_schema(),
        vec![
            Arc::new(Int32Array::from(vec![1; n])),
            Arc::new(StringArray::from(vec![RUN; n])),
            Arc::new(Int32Array::from(vec![0; n])),
            Arc::new(Int64Array::from(vec![attempt; n])),
            Arc::new(Int64Array::from(steps.clone())),
            Arc::new(Float64Array::from(vec![0.5; n])),
            Arc::new(StringArray::from(vec!["{}"; n])),
            Arc::new(StringArray::from(
                steps
                    .iter()
                    .map(|step| Some(format!(r#"{{"batches":1,"batch_hash":"a{attempt}-{step}"}}"#)))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// RS04. A rebuild of a rank's detectors that fails drops the rank's report and its batch hashes:
/// the status no longer says its batches were compared, from hashes of rows before the rebuild.
#[tokio::test]
async fn a_failed_rebuild_drops_the_ranks_batch_hashes_from_the_comparison() {
    use std::sync::atomic::Ordering;
    let store = TestStore::new();
    let config = IngestConfig {
        flush_steps: 10,
        ..manual()
    };
    let server = start_on(store.clone(), config, None).await;
    let compared = |server: &TestServer| server.ingest.watchdog_statuses()[RUN]["ranks_compared"].clone();
    let mut put = open_with(&server.address, RUN, 0, None, data_attempt_schema()).await;
    put.send(data_attempt_batch(1, 0..10), 1).await;
    assert_eq!(put.ack().await, Ok(1));
    until_within("rank 0 compared", SLOW_BOUND, || {
        compared(&server) == serde_json::json!([0])
    })
    .await;
    status_until(&server, &store, "rank 0 judged", judged).await;
    // A new attempt needs a rebuild from the store, which cannot be read.
    let failed = store.knobs.gets.load(Ordering::SeqCst);
    store.knobs.fail_gets.store(true, Ordering::SeqCst);
    put.send(data_attempt_batch(2, 0..10), 2).await;
    assert_eq!(put.ack().await, Ok(2));
    until_within("the rebuild tried and failed", SLOW_BOUND, || {
        store.knobs.gets.load(Ordering::SeqCst) > failed && compared(&server) == serde_json::json!([])
    })
    .await;
    let status = written_status(&server, &store).await;
    assert_eq!(status["ranks_compared"], serde_json::json!([]), "{status}");
    assert!(!judged(&status), "{}", status["run_state"]);
    store.knobs.fail_gets.store(false, Ordering::SeqCst);
    put.finish().await;
}

// --- Lane V6 (Nov 19, 2026): the server's watchdog and ingest ----------------------------------

/// L4. An offset is acknowledged only on the stream that sent it. A stream that ended in an error
/// before its rows were stored left its offset held, and the next stream of the rank was told
/// that offset (here 1,000, where it had sent 300). Two streams of one rank open at once are the
/// same: each is told only its own offsets.
#[tokio::test]
async fn an_offset_is_acknowledged_only_on_the_stream_that_sent_it() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), manual(), None).await;
    let mut dead = open(&server, 0).await;
    dead.send(batch(0, 0..10), 1000).await; // buffered, not stored
    dead.send_with_metadata(batch(0, 10..20), world_metadata(0, 1, 2000).replace(RUN, "another-run"))
        .await;
    assert_eq!(dead.ack().await, Err(Code::InvalidArgument));
    let mut next = open(&server, 0).await;
    next.send(batch(0, 10..15), 300).await;
    assert_eq!(next.finish().await, [300]);
    assert_eq!(stored_steps(dir.path())[&0], (0..15).collect::<Vec<_>>());

    let mut first = open(&server, 1).await;
    first.send(batch(1, 0..10), 1000).await;
    until("rank 1's first rows taken", || {
        server.ingest.watchdog_statuses()[RUN]["ranks"][1]["last_step"] == 9
    })
    .await;
    let mut second = open(&server, 1).await;
    second.send(batch(1, 10..15), 300).await;
    assert_eq!(second.finish().await, [300]);
    // The first stream's rows were stored with the second's; it is told so once it sends again
    // (a resend of rows the server has is acknowledged at once).
    first.send(batch(1, 0..10), 1000).await;
    assert_eq!(first.ack().await, Ok(1000));
    assert_eq!(first.finish().await, Vec::<i64>::new());
    assert_eq!(stored_steps(dir.path())[&1], (0..15).collect::<Vec<_>>());
}

/// RS14. With the watchdog's interval at zero (the status file off), a stop writes and judges
/// what is buffered and writes no status: it wrote one at shutdown.
#[tokio::test]
async fn with_the_status_file_off_a_stop_writes_no_status() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), watch_config(60_000), None).await;
    let mut flight = client(&server).await;
    flight
        .do_action(Action::new(
            "heartbeat",
            format!("{{\"run_id\":\"{RUN}\",\"rank\":0,\"phase\":\"train\",\"interval_seconds\":15}}"),
        ))
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let mut put = open(&server, 1).await;
    put.send(batch(1, 0..10), 50).await;
    until("rank 1's rows taken", || {
        server.ingest.watchdog_statuses()[RUN]["ranks"][1]["last_step"] == 9
    })
    .await;
    server.ingest.stop([]).await.unwrap();
    assert_eq!(stored_steps(dir.path())[&1], (0..10).collect::<Vec<_>>());
    let run = dir.path().join("runs").join(RUN);
    assert!(run.join("rank=1").join("incidents.json").exists());
    assert!(!run.join("watchdog.json").exists());
}

/// RS15. A duration flag that is not a number of seconds the server can hold (NaN, negative,
/// infinite, too large, or too large once turned into the refresh time or hours) and a quota whose
/// bytes overflow are refused with a message naming the flag. They panicked with exit code 101.
#[test]
fn a_duration_flag_out_of_range_is_refused_with_its_name() {
    let dir = tempfile::tempdir().unwrap();
    for (flag, named) in [
        ("--flush-seconds=NaN", "--flush-seconds"),
        ("--store-retry-seconds=-1", "--store-retry-seconds"),
        ("--silent-seconds=inf", "--silent-seconds"),
        ("--critical-seconds=1e20", "--critical-seconds"),
        ("--pause-seconds=-0.5", "--pause-seconds"),
        ("--start-seconds=nan", "--start-seconds"),
        ("--watchdog-seconds=1e19", "--watchdog-seconds"), // three times it is too long
        ("--abandon-hours=1e300", "--abandon-hours"),
        ("--quota-mib=18446744073709551615", "--quota-mib"),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_tensorquill-server"))
            .args(["--listen", "127.0.0.1:0", flag, "--store"])
            .arg(dir.path())
            .env_remove("TENSORQUILL_TOKEN")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            matches!(output.status.code(), Some(1 | 2)),
            "{flag}: {:?} {stderr}",
            output.status
        );
        assert!(
            stderr.contains(named) && !stderr.contains("panicked"),
            "{flag}: {stderr}"
        );
    }
}

/// RS15. Zero is a valid value of every duration flag: the server starts and stops cleanly.
#[cfg(unix)]
#[tokio::test]
async fn zero_is_a_valid_duration_for_every_flag() {
    use std::io::{BufRead, BufReader};
    let dir = tempfile::tempdir().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_tensorquill-server"))
        .args(["--listen", "127.0.0.1:0"])
        .args([
            "--flush-seconds=0",
            "--silent-seconds=0",
            "--critical-seconds=0",
            "--pause-seconds=0",
            "--start-seconds=0",
            "--watchdog-seconds=0",
            "--store-retry-seconds=0",
            "--abandon-hours=0",
        ])
        .arg("--store")
        .arg(dir.path())
        .env_remove("TENSORQUILL_TOKEN")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    assert!(line.starts_with("tensorquill-server listening on"), "{line:?}");
    assert!(Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .unwrap()
        .success());
    let output = tokio::task::spawn_blocking(move || child.wait_with_output().unwrap())
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "{:?} {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

/// RUST-16. A rebuild of a rank's detectors that fails is not tried again at every part: parts
/// stored while the store cannot be read were each a whole replay of the rank.
#[tokio::test]
async fn a_failed_rebuild_is_not_tried_again_at_every_part() {
    use std::sync::atomic::Ordering;
    let store = TestStore::new();
    let config = IngestConfig {
        flush_steps: 10,
        replay_backoff: Duration::from_secs(3600),
        ..manual()
    };
    let server = start_on(store.clone(), config, None).await;
    let mut put = open_with(&server.address, RUN, 0, None, data_attempt_schema()).await;
    put.send(data_attempt_batch(1, 0..10), 1).await;
    assert_eq!(put.ack().await, Ok(1));
    status_until(&server, &store, "rank 0 judged", judged).await;
    let before = store.knobs.gets.load(Ordering::SeqCst);
    store.knobs.fail_gets.store(true, Ordering::SeqCst);
    // A new attempt needs a rebuild from the store, which cannot be read.
    put.send(data_attempt_batch(2, 0..10), 2).await;
    assert_eq!(put.ack().await, Ok(2));
    // The next part waits for that rebuild (the rank's buffer is held while it runs).
    put.send(data_attempt_batch(2, 10..20), 3).await;
    assert_eq!(put.ack().await, Ok(3));
    let tried = store.knobs.gets.load(Ordering::SeqCst);
    assert!(tried > before, "the rebuild read the store");
    for (offset, first) in [(4, 20), (5, 30), (6, 40)] {
        put.send(data_attempt_batch(2, first..first + 10), offset).await;
        assert_eq!(put.ack().await, Ok(offset));
    }
    put.finish().await; // every part judged, or not, by now
    assert_eq!(
        store.knobs.gets.load(Ordering::SeqCst),
        tried,
        "each part replayed the rank again"
    );
    store.knobs.fail_gets.store(false, Ordering::SeqCst);
}

/// RUST-16. Once the wait after a failed rebuild has passed and the store reads again, the next
/// part rebuilds the rank and it is judged again.
#[tokio::test]
async fn a_rank_whose_rebuild_failed_is_judged_again_once_the_store_reads() {
    use std::sync::atomic::Ordering;
    let store = TestStore::new();
    let config = IngestConfig {
        flush_steps: 10,
        replay_backoff: Duration::from_millis(100),
        ..manual()
    };
    let server = start_on(store.clone(), config, None).await;
    let mut put = open_with(&server.address, RUN, 0, None, data_attempt_schema()).await;
    put.send(data_attempt_batch(1, 0..10), 1).await;
    assert_eq!(put.ack().await, Ok(1));
    status_until(&server, &store, "rank 0 judged", judged).await;
    store.knobs.fail_gets.store(true, Ordering::SeqCst);
    put.send(data_attempt_batch(2, 0..10), 2).await;
    assert_eq!(put.ack().await, Ok(2));
    status_until(&server, &store, "rank 0 not judged", |status| !judged(status)).await;
    store.knobs.fail_gets.store(false, Ordering::SeqCst);
    let deadline = tokio::time::Instant::now() + SLOW_BOUND;
    let mut first = 10;
    loop {
        put.send(data_attempt_batch(2, first..first + 10), first + 10).await;
        assert_eq!(put.ack().await, Ok(first + 10));
        first += 10;
        if judged(&written_status(&server, &store).await) {
            break;
        }
        assert!(tokio::time::Instant::now() < deadline, "never judged again");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    put.finish().await;
}

/// RUST-18. A stream that ends in an error is counted, and the binary says so on stderr with the
/// run, rank and code, without the bytes the sender sent.
#[cfg(unix)]
#[tokio::test]
async fn a_stream_error_is_counted_and_logged_without_its_payload() {
    use std::io::{BufRead, BufReader};
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), manual(), None).await;
    assert_eq!(server.ingest.stream_errors(), 0);
    let mut put = open(&server, 0).await;
    put.send_with_metadata(batch(0, 0..10), "{\"end_byte_offset\":\"SECRET-PAYLOAD\"}".into())
        .await;
    assert_eq!(put.ack().await, Err(Code::InvalidArgument));
    assert_eq!(server.ingest.stream_errors(), 1);

    let mut child = Command::new(env!("CARGO_BIN_EXE_tensorquill-server"))
        .args(["--listen", "127.0.0.1:0", "--watchdog-seconds", "0", "--store"])
        .arg(dir.path())
        .env_remove("TENSORQUILL_TOKEN")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    let address = line.trim().rsplit(' ').next().unwrap().to_string();
    let mut put = open_at(&format!("http://{address}"), RUN, 3, None).await;
    put.send_with_metadata(batch(3, 0..10), "{\"end_byte_offset\":\"SECRET-PAYLOAD\"}".into())
        .await;
    assert_eq!(put.ack().await, Err(Code::InvalidArgument));
    assert!(Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .unwrap()
        .success());
    let output = tokio::task::spawn_blocking(move || child.wait_with_output().unwrap())
        .await
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let logged: Vec<&str> = stderr.lines().filter(|line| line.contains("stream")).collect();
    assert_eq!(logged.len(), 1, "{stderr}");
    assert!(
        logged[0].contains("run-a rank 3") && logged[0].contains("code InvalidArgument"),
        "{stderr}"
    );
    assert!(!stderr.contains("SECRET-PAYLOAD"), "{stderr}");
}

fn batch_of(run_id: &str, rank: i32, steps: std::ops::Range<i64>) -> RecordBatch {
    let steps: Vec<i64> = steps.collect();
    let n = steps.len();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(vec![1; n])),
            Arc::new(StringArray::from(vec![run_id; n])),
            Arc::new(Int32Array::from(vec![rank; n])),
            Arc::new(Int64Array::from(steps.clone())),
            Arc::new(Float64Array::from(vec![0.5; n])),
            Arc::new(StringArray::from(vec!["{}"; n])),
        ],
    )
    .unwrap()
}

fn metadata_of(run_id: &str, rank: i32, end: i64) -> String {
    format!(
        "{{\"schema_version\":1,\"run_id\":\"{run_id}\",\"rank\":{rank},\"end_byte_offset\":{end},\"world_size\":1}}"
    )
}

/// One rank of `run_id` sends `steps` and closes its stream; the offsets it was told.
async fn a_short_run(server: &TestServer, run_id: &str, steps: std::ops::Range<i64>) -> Vec<i64> {
    let mut put = open_as(server, run_id, 0, None).await;
    let end = steps.end;
    put.send_with_metadata(batch_of(run_id, 0, steps), metadata_of(run_id, 0, end))
        .await;
    put.finish().await
}

/// RS13. A long-lived server lets a run go once its status reads finished, is written so, and
/// the grace has passed: twenty short runs one after another leave nothing held. A rank of a run
/// let go that comes back is taken as after a restart: its watermark from the store (a resend
/// stores nothing twice) and its detectors rebuilt from the store.
#[tokio::test]
async fn finished_runs_are_let_go_and_one_that_comes_back_is_taken_as_after_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let held = IngestConfig {
        evict_after: Duration::from_secs(3600),
        ..watch_config(60_000)
    };
    let server = start(dir.path(), held, None).await;
    assert_eq!(a_short_run(&server, "kept", 0..10).await, [10]);
    server.ingest.write_watchdog().await;
    assert_eq!(server.ingest.runs_held(), 1, "inside the grace a finished run is kept");

    let dir = tempfile::tempdir().unwrap();
    let config = IngestConfig {
        evict_after: Duration::ZERO,
        ..watch_config(60_000)
    };
    let server = start(dir.path(), config, None).await;
    for n in 0..20 {
        let run_id = format!("run-{n}");
        assert_eq!(a_short_run(&server, &run_id, 0..10).await, [10]);
        // A pass that finds a buffer still in use leaves the run to the next one.
        let deadline = tokio::time::Instant::now() + SLOW_BOUND;
        loop {
            server.ingest.write_watchdog().await;
            if server.ingest.runs_held() == 0 {
                break;
            }
            assert!(tokio::time::Instant::now() < deadline, "{run_id} never let go of");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let status = json_file(&dir.path().join("runs").join(&run_id).join("watchdog.json"));
        assert_eq!(status["summary"], "finished", "{status}");
    }
    // A run that is not finished is kept.
    let mut open_run = open_as(&server, "still-open", 0, None).await;
    open_run
        .send_with_metadata(batch_of("still-open", 0, 0..10), metadata_of("still-open", 0, 10))
        .await;
    until("the open run's rows taken", || {
        server.ingest.watchdog_statuses()["still-open"]["ranks"][0]["last_step"] == 9
    })
    .await;
    server.ingest.write_watchdog().await;
    assert_eq!(server.ingest.runs_held(), 1);

    // run-3 comes back: a resend of its rows and ten new ones.
    assert_eq!(a_short_run(&server, "run-3", 0..20).await, [20]);
    let steps: Vec<i64> = part_files(&dir.path().join("runs").join("run-3"))
        .iter()
        .flat_map(|path| {
            let reader = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(path).unwrap()).unwrap();
            reader
                .build()
                .unwrap()
                .flat_map(|batch| {
                    let batch = batch.unwrap();
                    let steps = batch.column_by_name("step").unwrap();
                    steps.as_any().downcast_ref::<Int64Array>().unwrap().values().to_vec()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(steps, (0..20).collect::<Vec<_>>());
    let incidents = json_file(
        &dir.path()
            .join("runs")
            .join("run-3")
            .join("rank=0")
            .join("incidents.json"),
    );
    assert_eq!(incidents["last_step"], 19, "{incidents}");
    server.ingest.write_watchdog().await;
    let status = json_file(&dir.path().join("runs").join("run-3").join("watchdog.json"));
    assert_eq!(status["summary"], "finished", "{status}");
    assert_eq!(server.ingest.runs_held(), 1, "only the open run");
    // A heartbeat for a rank let go of finds nothing: the rank is watched afresh, as after a
    // restart (a rank first seen by a beat is `starting`). The SDK stops its heartbeat before it
    // closes its stream, so only a job started again, or another sender, can send one this late.
    let mut flight = client(&server).await;
    flight
        .do_action(Action::new(
            "heartbeat",
            "{\"run_id\":\"run-5\",\"rank\":0,\"phase\":\"train\",\"interval_seconds\":15}".to_string(),
        ))
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let rank = &server.ingest.watchdog_statuses()["run-5"]["ranks"][0];
    assert_eq!(
        (rank["state"].as_str(), rank["last_step"].as_i64()),
        (Some("starting"), None),
        "{rank}"
    );
    // A restarted server takes up the open run only.
    drop(open_run);
    let restarted = Ingest::new(
        Storage::new(
            Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap()),
            "".into(),
        ),
        watch_config(60_000),
    );
    assert_eq!(restarted.restore_watchdog().await.unwrap(), 1);
}

/// RS13. With the status file off a settled run is let go of all the same, and no status is written.
#[tokio::test]
async fn with_the_status_file_off_a_finished_run_is_let_go_of() {
    let dir = tempfile::tempdir().unwrap();
    let config = IngestConfig {
        evict_after: Duration::ZERO,
        ..watch_config(60_000)
    };
    let server = start(dir.path(), config, None).await;
    assert_eq!(a_short_run(&server, "run-0", 0..10).await, [10]);
    assert_eq!(server.ingest.runs_held(), 1);
    let task = server.ingest.spawn_watchdog(); // its first round is at once
    until("the finished run let go of", || server.ingest.runs_held() == 0).await;
    server.ingest.stop([task]).await.unwrap();
    assert!(!dir.path().join("runs").join("run-0").join("watchdog.json").exists());
    assert!(dir
        .path()
        .join("runs")
        .join("run-0")
        .join("rank=0")
        .join("incidents.json")
        .exists());
}

/// RUST-17. After a restart the first part of each rank rebuilds its detectors from the store,
/// and parts written by the streams themselves (not the timed flush) had no shared bound: forty
/// ranks coming back at once read forty ranks at once. The rebuilds are bounded across streams.
#[tokio::test]
async fn rebuilds_from_the_streams_after_a_restart_are_bounded() {
    use std::sync::atomic::Ordering;
    const RANKS: i32 = 40;
    let store = TestStore::new();
    {
        let server = start_on(store.clone(), manual(), None).await;
        for rank in 0..RANKS {
            let mut put = open(&server, rank).await;
            put.send(batch(rank, 0..10), 10).await;
            assert_eq!(put.finish().await, [10]);
        }
    }
    store.knobs.delay_ms.store(20, Ordering::SeqCst);
    store.knobs.max_gets_in_flight.store(0, Ordering::SeqCst);
    let server = start_on(store.clone(), manual(), None).await;
    let mut puts = Vec::new();
    for rank in 0..RANKS {
        let mut put = open(&server, rank).await;
        put.send(batch(rank, 10..20), 20).await;
        puts.push(put);
    }
    let acks = futures::future::join_all(puts.into_iter().map(Put::finish)).await;
    assert!(acks.iter().all(|acks| acks == &[20]), "{acks:?}");
    let most = store.knobs.max_gets_in_flight.load(Ordering::SeqCst);
    assert!(most <= 16, "{most} parts read at once");
}

// ---------------------------------------------------------------------------------------------
// Contract 1 (RS05, decided Nov 19, 2026): the order of a rank's attempts, through Flight.

async fn beat_attempt(server: &TestServer, attempt: i64) -> Result<(), tonic::Status> {
    let mut flight = client(server).await;
    let body =
        format!("{{\"run_id\":\"{RUN}\",\"rank\":0,\"phase\":\"train\",\"interval_seconds\":5,\"attempt\":{attempt}}}");
    match flight.do_action(Action::new("heartbeat", body)).await {
        Ok(replies) => {
            replies.try_collect::<Vec<_>>().await.unwrap();
            Ok(())
        }
        Err(FlightError::Tonic(status)) => Err(*status),
        Err(other) => panic!("{other}"),
    }
}

fn sender_metadata(sender_attempt: i64, end: i64) -> String {
    format!(
        "{{\"schema_version\":1,\"run_id\":\"{RUN}\",\"rank\":0,\"min_step\":null,\"max_step\":null,\"end_byte_offset\":{end},\"sender_attempt\":{sender_attempt}}}"
    )
}

#[tokio::test]
async fn w2_a_claim_whose_order_cannot_be_stored_is_refused_and_leaves_nothing_behind() {
    use std::sync::atomic::Ordering;
    let store = TestStore::new();
    store.knobs.fail_order_puts.store(1, Ordering::SeqCst);
    let server = start_on(store.clone(), watch_config(60_000), None).await;
    let refused = beat_attempt(&server, 7).await.unwrap_err();
    assert_eq!(refused.code(), Code::Unavailable, "{refused:?}");
    assert!(
        !server.ingest.watchdog_statuses().contains_key(RUN),
        "a refused beat places nothing and leaves no rank"
    );
    // The sender's next beat goes through: the order is stored, then the attempt owns the rank.
    beat_attempt(&server, 7).await.unwrap();
    assert_eq!(server.ingest.watchdog_statuses()[RUN]["ranks"][0]["attempt"], 7);
    assert_eq!(store.knobs.order_puts.load(Ordering::SeqCst), 2);
    // A batch that would place a new attempt is refused the same way, and its rows are not taken.
    store.knobs.fail_order_puts.store(1, Ordering::SeqCst);
    let mut put = open_attempts(&server).await;
    put.send_with_metadata(attempt_batch(&rows(9, 0..5)), sender_metadata(9, 100))
        .await;
    assert_eq!(put.ack().await, Err(Code::Unavailable));
    assert_eq!(server.ingest.buffered_bytes(), 0);
    assert_eq!(server.ingest.watchdog_statuses()[RUN]["ranks"][0]["attempt"], 7);
}

#[tokio::test]
async fn w2_an_earlier_attempts_stream_that_ends_cleanly_after_a_new_claim_does_not_finish_the_rank() {
    // RS01's open case: attempt 1's stream is open when attempt 2's first beat comes, then it
    // closes cleanly. Until Nov 19, 2026 the rank read finished.
    let dir = tempfile::tempdir().unwrap();
    let config = IngestConfig {
        flush_steps: 1,
        ..watch_config(60_000)
    };
    let server = start(dir.path(), config, None).await;
    beat_attempt(&server, 1).await.unwrap();
    let mut put = open_attempts(&server).await;
    put.send_with_metadata(attempt_batch(&rows(1, 0..10)), sender_metadata(1, 100))
        .await;
    assert_eq!(put.ack().await, Ok(100));
    beat_attempt(&server, 2).await.unwrap();
    assert_eq!(put.finish().await, Vec::<i64>::new());
    let mut entry = serde_json::Value::Null;
    until("the earlier stream's end to be seen", || {
        entry = server.ingest.watchdog_statuses()[RUN]["ranks"][0].clone();
        entry["stream"] != "open"
    })
    .await;
    assert_eq!(
        (
            entry["attempt"].as_i64(),
            entry["state"].as_str(),
            entry["last_step"].clone()
        ),
        (Some(2), Some("reconnecting"), serde_json::Value::Null),
        "{entry}"
    );
    // A late row of attempt 1 is stored and counted, and not progress.
    let mut late = open_attempts(&server).await;
    late.send_with_metadata(attempt_batch(&rows(1, 10..11)), sender_metadata(1, 200))
        .await;
    assert_eq!(late.ack().await, Ok(200));
    let entry = &server.ingest.watchdog_statuses()[RUN]["ranks"][0];
    assert_eq!(
        (entry["attempt"].as_i64(), entry["superseded_rows"].as_i64()),
        (Some(2), Some(1)),
        "{entry}"
    );
    let order: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.path().join("runs").join(RUN).join("rank=0").join("attempts.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(order["attempts"][0]["attempt"], 1);
    assert_eq!(order["attempts"][1]["attempt"], 2);
}

#[tokio::test]
async fn w2_a_new_attempt_whose_rows_beat_its_first_heartbeat_owns_the_rank_with_a_sender_that_names_none() {
    // Every SDK before Nov 19, 2026 names no `sender_attempt`. Attempt 1 is claimed and steps; the
    // job starts again as attempt 2, whose rows reach the server before any beat naming it.
    let dir = tempfile::tempdir().unwrap();
    let config = IngestConfig {
        flush_steps: 1,
        ..watch_config(60_000)
    };
    let server = start(dir.path(), config, None).await;
    beat_attempt(&server, 1).await.unwrap();
    let mut put = open_attempts(&server).await;
    put.send(attempt_batch(&rows(1, 0..10)), 100).await;
    assert_eq!(put.ack().await, Ok(100));
    put.send(attempt_batch(&rows(2, 5..8)), 200).await;
    assert_eq!(put.ack().await, Ok(200));
    beat_attempt(&server, 2).await.unwrap();
    let entry = &server.ingest.watchdog_statuses()[RUN]["ranks"][0];
    assert_eq!(
        (
            entry["attempt"].as_i64(),
            entry["last_step"].as_i64(),
            entry["state"].as_str()
        ),
        (Some(2), Some(7), Some("active")),
        "{entry}"
    );
}

// --- Lane X6 (Nov 19, 2026): the server's watchdog and ingest ----------------------------------

/// `attempt_schema()` with every key column nullable, as a sender other than the SDK may send it.
fn nullable_key_schema() -> Arc<Schema> {
    let fields: Vec<Field> = attempt_schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone().with_nullable(true))
        .collect();
    Arc::new(Schema::new(fields))
}

/// Rows of rank 0 in `nullable_key_schema()`: `(run_id, rank, attempt, step)`, any of them null.
#[allow(clippy::type_complexity)]
fn nullable_key_batch(rows: &[(Option<&str>, Option<i32>, Option<i64>, Option<i64>)]) -> RecordBatch {
    let n = rows.len();
    RecordBatch::try_new(
        nullable_key_schema(),
        vec![
            Arc::new(Int32Array::from(vec![1; n])),
            Arc::new(StringArray::from(rows.iter().map(|row| row.0).collect::<Vec<_>>())),
            Arc::new(Int32Array::from(rows.iter().map(|row| row.1).collect::<Vec<_>>())),
            Arc::new(Int64Array::from(rows.iter().map(|row| row.2).collect::<Vec<_>>())),
            Arc::new(Int64Array::from(rows.iter().map(|row| row.3).collect::<Vec<_>>())),
            Arc::new(Float64Array::from(vec![0.5; n])),
            Arc::new(StringArray::from(vec!["{}"; n])),
        ],
    )
    .unwrap()
}

/// RS10, live half. A row whose run, rank, step or attempt is null is dropped and counted by the
/// key, and the other rows of its batch are stored and acknowledged: the stream goes on. Until
/// Nov 19, 2026 the whole batch was refused with `INVALID_ARGUMENT`, which ended the stream, and
/// the sender, which sends the same rows again from its offset, could store nothing more for the
/// rank. A column that is missing or of another type is still refused: no row of it can be read.
#[tokio::test]
async fn a_row_with_a_null_key_is_dropped_and_counted_and_the_rest_of_its_batch_is_stored() {
    let dir = tempfile::tempdir().unwrap();
    let config = IngestConfig {
        flush_steps: 1,
        ..manual()
    };
    let server = start(dir.path(), config, None).await;
    let mut put = open_with(&server.address, RUN, 0, None, nullable_key_schema()).await;
    let good = |step: i64| (Some(RUN), Some(0), Some(1), Some(step));
    put.send(
        nullable_key_batch(&[
            good(0),
            good(1),
            (Some(RUN), Some(0), Some(1), None), // no step
            good(3),
            (None, Some(0), Some(1), Some(4)), // no run
            good(5),
            (Some(RUN), None, Some(1), Some(6)), // no rank
            (Some(RUN), Some(0), None, Some(7)), // no attempt
            (None, Some(0), Some(1), None),      // no run and no step: counted once, by the run
            good(9),
        ]),
        100,
    )
    .await;
    assert_eq!(put.ack().await, Ok(100), "the batch was refused");
    // A batch whose every row is dropped is acknowledged, so its sender does not send it forever.
    put.send(nullable_key_batch(&[(Some(RUN), Some(0), Some(1), None)]), 200)
        .await;
    assert_eq!(put.ack().await, Ok(200));
    // The stream goes on.
    put.send(nullable_key_batch(&[good(10), good(11)]), 300).await;
    assert_eq!(put.ack().await, Ok(300));
    put.finish().await;
    assert_eq!(stored_steps(dir.path())[&0], vec![0, 1, 3, 5, 9, 10, 11]);
    assert_eq!(
        server.ingest.dropped_rows(),
        BTreeMap::from([
            ("attempt".to_string(), 1),
            ("rank".to_string(), 1),
            ("run_id".to_string(), 2),
            ("step".to_string(), 2),
        ])
    );
    // A key column of another type is refused, as before, naming the column.
    let mut put = open_with(&server.address, RUN, 0, None, put_schema_of_mistyped_attempt()).await;
    let mistyped = RecordBatch::try_new(
        put_schema_of_mistyped_attempt(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(StringArray::from(vec![RUN])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(StringArray::from(vec!["1"])),
            Arc::new(Int64Array::from(vec![20])),
        ],
    )
    .unwrap();
    put.send(mistyped, 400).await;
    assert_eq!(put.ack().await, Err(Code::InvalidArgument));
    assert_eq!(stored_steps(dir.path())[&0], vec![0, 1, 3, 5, 9, 10, 11]);
}

/// The key columns, with `attempt` as text.
fn put_schema_of_mistyped_attempt() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("schema_version", DataType::Int32, false),
        Field::new("run_id", DataType::Utf8, false),
        Field::new("rank", DataType::Int32, false),
        Field::new("attempt", DataType::Utf8, false),
        Field::new("step", DataType::Int64, false),
    ]))
}

/// Starts the binary with `flags`. `None` when it started (it printed the line it listens on):
/// it is then killed. Otherwise its exit code and stderr.
#[cfg(unix)]
fn exit_of_the_binary_with(flags: &[&str]) -> Option<(Option<i32>, String)> {
    use std::io::{BufRead, BufReader, Read};
    let dir = tempfile::tempdir().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_tensorquill-server"))
        .args(["--listen", "127.0.0.1:0", "--watchdog-seconds", "0"])
        .args(flags)
        .arg("--store")
        .arg(dir.path())
        .env_remove("TENSORQUILL_TOKEN")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    if line.starts_with("tensorquill-server listening on") {
        let _ = child.kill();
        let _ = child.wait();
        return None;
    }
    let status = child.wait().unwrap();
    let mut stderr = String::new();
    child.stderr.take().unwrap().read_to_string(&mut stderr).unwrap();
    Some((status.code(), stderr))
}

/// RS15's open part. A `--silent-factor` the watchdog cannot use is refused at startup, naming the
/// flag. Until Nov 19, 2026 the server started with any number: NaN or a negative one left the
/// floor alone, as 0 does, and `inf` made every rank with a cadence never silent, never critical.
/// 0 (the floor alone, which the SDK's tests use) and any finite number above 1 start.
#[cfg(unix)]
#[test]
fn a_silent_factor_the_watchdog_cannot_use_is_refused_at_startup() {
    for bad in ["NaN", "inf", "-inf", "-1", "-0.5", "0.5", "1", "1e400"] {
        let flag = format!("--silent-factor={bad}");
        let Some((code, stderr)) = exit_of_the_binary_with(&[&flag]) else {
            panic!("{flag}: the server started");
        };
        assert_eq!(code, Some(2), "{flag}: {stderr}");
        assert!(
            stderr.contains("--silent-factor") && !stderr.contains("panicked"),
            "{flag}: {stderr}"
        );
    }
    for good in ["0", "1.5", "5", "1000"] {
        let flag = format!("--silent-factor={good}");
        assert_eq!(
            exit_of_the_binary_with(&[&flag]),
            None,
            "{flag}: the server did not start"
        );
    }
}

/// RUST-16's open part. A rank whose rebuild failed is rebuilt once its wait has passed and the
/// store reads again, though it sends nothing more: the timed pass tries it. Until Nov 19, 2026
/// only a later part of the rank tried again, so a rank that sent nothing more (a job that ended,
/// or one that hung right then) stayed without a report and unjudged for good.
#[tokio::test]
async fn a_rank_whose_rebuild_failed_is_rebuilt_without_a_later_part() {
    use std::sync::atomic::Ordering;
    let store = TestStore::new();
    let config = IngestConfig {
        flush_steps: 10,
        replay_backoff: Duration::from_millis(100),
        ..manual()
    };
    let server = start_on(store.clone(), config, None).await;
    let mut put = open_with(&server.address, RUN, 0, None, data_attempt_schema()).await;
    put.send(data_attempt_batch(1, 0..10), 1).await;
    assert_eq!(put.ack().await, Ok(1));
    status_until(&server, &store, "rank 0 judged", judged).await;
    store.knobs.fail_gets.store(true, Ordering::SeqCst);
    // A new attempt needs a rebuild from the store, which cannot be read.
    put.send(data_attempt_batch(2, 0..10), 2).await;
    assert_eq!(put.ack().await, Ok(2));
    status_until(&server, &store, "rank 0 not judged", |status| !judged(status)).await;
    store.knobs.fail_gets.store(false, Ordering::SeqCst);
    // Nothing more is sent. The flusher's pass, about once a second here, tries the rebuild again.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    loop {
        let status = written_status(&server, &store).await;
        if judged(&status) {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "never judged again without a later part: {}",
            status["run_state"]
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let report = json_in(&store, &format!("runs/{RUN}/rank=0/incidents.json")).await;
    assert_eq!(report["attempt"], serde_json::json!(2), "{report}");
    put.finish().await;
}
