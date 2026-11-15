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
use arrow_flight::{FlightClient, FlightData, FlightDescriptor, PutResult};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use object_store::local::LocalFileSystem;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use tensorlens_server::ingest::{Ingest, IngestConfig, DEFAULT_FLUSH_STEPS, DEFAULT_QUOTA_BYTES};
use tensorlens_server::server::serve;
use tensorlens_server::storage::Storage;
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
    let ingest = Ingest::new(Storage::new(Arc::new(store), "".into()), config);
    ingest.spawn_flusher();
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
        .with_schema(schema())
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
            "runs/run-a/rank=0/part-00000000-00000049.parquet",
            "runs/run-a/rank=0/part-00000050-00000059.parquet"
        ]
    );
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
        ["runs/run-a/rank=0/part-00000000-00000499.parquet"]
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
        ["runs/run-a/rank=0/part-00000000-00000006.parquet"]
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
    let mut child = Command::new(env!("CARGO_BIN_EXE_tensorlens-server"))
        .args(["--listen", "127.0.0.1:0", "--store"])
        .arg(dir.path())
        .env_remove("TENSORLENS_TOKEN")
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
        ["runs/run-a/rank=0/part-00000000-00000049.parquet"]
    );
}

/// The binary refuses to listen beyond loopback with no token and no explicit flag.
#[test]
fn the_binary_refuses_an_unauthenticated_public_bind() {
    let dir = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_tensorlens-server"))
        .args(["--listen", "0.0.0.0:0", "--store"])
        .arg(dir.path())
        .env_remove("TENSORLENS_TOKEN")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("TENSORLENS_TOKEN"));
}
