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
    assert!(first.no_ack_within(Duration::from_millis(300)).await);
    assert!(second.no_ack_within(Duration::from_millis(300)).await);
    assert!(server.ingest.buffered_bytes() >= 100 * 1024 * 1024);

    // A third rank's 50 MiB would pass 128 MiB: refused, and nothing of it is kept.
    let mut third = open(&server, 2).await;
    third.send(batch_with(2, [0], &wide), 1).await;
    assert_eq!(third.ack().await, Err(Code::ResourceExhausted));
    assert!(server.ingest.buffered_bytes() <= DEFAULT_QUOTA_BYTES);

    // Once a buffer is written there is room, and the retry is accepted.
    assert_eq!(first.finish().await, vec![1]);
    let mut third = open(&server, 2).await;
    third.send(batch_with(2, [0], &wide), 1).await;
    assert_eq!(third.finish().await, vec![1]);
    assert_eq!(second.finish().await, vec![1]);
    assert_eq!(server.ingest.buffered_bytes(), 0);
    assert_eq!(stored_steps(dir.path()).len(), 3);
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

/// The real binary: `SIGTERM` must write what is buffered before the process exits.
#[cfg(unix)]
#[tokio::test]
async fn sigterm_flushes_buffered_rows_before_exit() {
    use std::io::{BufRead, BufReader};

    let dir = tempfile::tempdir().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_tensorquill-server"))
        .args(["--listen", "127.0.0.1:0", "--store"])
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
    assert!(
        put.no_ack_within(Duration::from_millis(500)).await,
        "50 steps are only buffered"
    );
    assert!(part_files(dir.path()).is_empty());

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

async fn until(what: &str, mut ready: impl FnMut() -> bool) {
    for _ in 0..100 {
        if ready() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("timed out waiting for {what}");
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
                self.knobs.puts.fetch_add(1, Ordering::SeqCst);
                let now = self.knobs.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                self.knobs.max_in_flight.fetch_max(now, Ordering::SeqCst);
                self.wait().await;
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
                if matches!(options.range, Some(GetRange::Suffix(_) | GetRange::Bounded(_))) {
                    self.knobs.range_gets.fetch_add(1, Ordering::SeqCst);
                } else {
                    self.knobs.gets.fetch_add(1, Ordering::SeqCst);
                }
                self.wait().await;
                self.inner.get_opts(location, options).await
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
    until("every batch is buffered", || server.ingest.buffered_bytes() > 0).await;
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
async fn a_killed_servers_silent_rank_and_incident_survive_a_restart_with_their_clock() {
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
    // The first status after the restart, with no sender reconnected: the stall is still there,
    // with the quiet time it had (not reset, and not grown by the outage).
    let status = &new.ingest.watchdog_statuses()[RUN];
    assert_eq!(status["summary"], "degraded");
    assert_eq!(status["incidents"][0]["kind"], "rank_silent");
    assert_eq!(status["incidents"][0]["assessment"], "suspected_hang");
    assert_eq!(status["ranks"][1]["state"], "silent"); // held as it was while its sender comes back
    assert_eq!(status["ranks"][1]["stream"], "dropped");
    assert_eq!(status["ranks"][1]["last_step"], 49);
    assert_eq!(status["ranks"][0]["last_step"], before["ranks"][0]["last_step"]);
    let quiet_after = status["ranks"][1]["seconds_since_progress"].as_f64().unwrap();
    assert!(
        quiet_after >= quiet_before && quiet_after < quiet_before + 0.25,
        "{quiet_before} -> {quiet_after}"
    );
    assert_eq!(status["world_size"], 2);
    // The file now says the new server is watching, with the incident in it.
    new.ingest.write_watchdog().await;
    let after = read_status(dir.path());
    assert_eq!(after["server_running"], true);
    assert_eq!(after["incidents"][0]["kind"], "rank_silent");
    assert!(after["written_at_ms"].as_i64() > before["written_at_ms"].as_i64());

    // The silent rank's sender returns with nothing new: silent, from its old last progress. The
    // other rank's batch makes it active again.
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
async fn a_clean_stop_says_server_stopped_and_a_restart_brings_the_stall_back() {
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
        (Some("degraded"), Some(true))
    );
    assert_eq!(status["incidents"][0]["kind"], "rank_silent");
    assert_eq!(status["ranks"][1]["state"], "silent");
    new.ingest.write_watchdog().await;
    assert_eq!(read_status(dir.path())["summary"], "degraded");
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

/// The real binary, killed with SIGKILL: its last file stays, and the next one takes the run up.
#[cfg(unix)]
#[tokio::test]
async fn the_binary_killed_with_sigkill_resumes_the_stall_it_was_watching() {
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
    // The first file the new server wrote already carries the incident, with the old clock.
    assert_eq!(first["server_running"], true, "{first}");
    assert_eq!(first["summary"], "stalled");
    assert_eq!(first["incidents"][0]["kind"], "job_stalled");
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
    // 160 reads of 25 ms one after another would take 4 s.
    assert!(took < Duration::from_millis(1500), "{took:?}");
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
