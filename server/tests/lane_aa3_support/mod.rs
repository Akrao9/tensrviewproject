//! What lane AA3's tests share (wave AA, Nov 20, 2026): a server in this process, a `DoPut` stream
//! as the Python sender holds one, and the real binary. Not a test target of its own: each
//! `lane_aa3_*.rs` takes it with `mod lane_aa3_support;`.

#![allow(dead_code)] // each test file uses part of it

use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::{FlightClient, FlightData, FlightDescriptor, PutResult};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use object_store::local::LocalFileSystem;
use tensorquill_server::ingest::{Ingest, IngestConfig};
use tensorquill_server::server::serve;
use tensorquill_server::storage::Storage;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Channel;

pub struct TestServer {
    pub address: String,
    pub ingest: Arc<Ingest>,
    _stop: oneshot::Sender<()>,
}

/// A server in this process, storing under `root`; `flush_steps` 1, so each batch is stored and
/// acknowledged as it arrives.
pub async fn start(root: &Path) -> TestServer {
    let config = IngestConfig {
        flush_steps: 1,
        flush_interval: Duration::from_secs(3600),
        ..IngestConfig::default()
    };
    let store = LocalFileSystem::new_with_prefix(root).unwrap();
    let ingest = Ingest::new(Storage::new(Arc::new(store), "".into()), config);
    ingest.spawn_flusher();
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

/// One open `DoPut` stream.
pub struct Put {
    run_id: String,
    rank: i32,
    messages: mpsc::UnboundedSender<Result<FlightData, FlightError>>,
    acks: BoxStream<'static, Result<PutResult, FlightError>>,
}

/// Opens a stream for `run_id` and `rank` whose batches have `schema`.
pub async fn open(address: &str, run_id: &str, rank: i32, schema: Arc<Schema>) -> Put {
    let channel = Channel::from_shared(address.to_string())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightClient::new(channel);
    let (messages, outgoing) = mpsc::unbounded_channel();
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
    Put {
        run_id: run_id.to_string(),
        rank,
        messages,
        acks,
    }
}

impl Put {
    /// Sends one batch with the `app_metadata` the contract names.
    pub async fn send(&mut self, batch: RecordBatch, end_byte_offset: i64) {
        let metadata = format!(
            "{{\"schema_version\":1,\"run_id\":\"{}\",\"rank\":{},\"min_step\":null,\"max_step\":null,\"end_byte_offset\":{end_byte_offset}}}",
            self.run_id, self.rank
        );
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

    /// The next acknowledgement as the JSON it carries, or the status that ended the stream.
    pub async fn reply(&mut self) -> Result<serde_json::Value, tonic::Status> {
        match tokio::time::timeout(Duration::from_secs(20), self.acks.next())
            .await
            .expect("no reply")
        {
            Some(Ok(result)) => Ok(serde_json::from_slice(&result.app_metadata).unwrap()),
            Some(Err(FlightError::Tonic(status))) => Err(*status),
            Some(Err(other)) => panic!("{other}"),
            None => panic!("the stream ended without a reply"),
        }
    }

    /// The next acknowledged `end_byte_offset`.
    pub async fn ack(&mut self) -> i64 {
        let reply = self.reply().await.expect("an acknowledgement");
        reply["end_byte_offset"].as_i64().unwrap()
    }

    /// Ends the stream and waits for the server to end it.
    pub async fn finish(mut self) -> Vec<serde_json::Value> {
        drop(self.messages);
        let mut replies = Vec::new();
        while let Some(result) = self.acks.next().await {
            replies.push(serde_json::from_slice(&result.unwrap().app_metadata).unwrap());
        }
        replies
    }
}

/// The real binary, killed when dropped.
pub struct Binary {
    child: Child,
    pub address: String,
}

impl Binary {
    /// Starts `tensorquill-server` on `listen` (`127.0.0.1:0` for any port) over the store `store`,
    /// and waits for the line that names its address.
    pub fn start(store: &Path, listen: &str) -> Binary {
        use std::io::{BufRead, BufReader};
        let mut child = Command::new(env!("CARGO_BIN_EXE_tensorquill-server"))
            .args(["--listen", listen, "--watchdog-seconds", "0", "--store"])
            .arg(store)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut line = String::new();
        BufReader::new(child.stdout.as_mut().unwrap())
            .read_line(&mut line)
            .unwrap();
        let address = line
            .trim()
            .strip_prefix("tensorquill-server listening on ")
            .unwrap_or_else(|| panic!("the binary did not start: {line:?}"))
            .to_string();
        Binary { child, address }
    }

    pub fn url(&self) -> String {
        format!("http://{}", self.address)
    }

    pub fn stop(mut self) {
        self.kill();
    }

    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Binary {
    fn drop(&mut self) {
        self.kill();
    }
}

/// `path` as JSON.
pub fn json_file(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))).unwrap()
}

/// Every stored step of `run_id` and `rank`, in step order, from its Parquet parts.
pub fn stored_steps(root: &Path, run_id: &str, rank: i32) -> Vec<i64> {
    use arrow::array::Int64Array;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let dir = root.join("runs").join(run_id).join(format!("rank={rank}"));
    let mut steps = Vec::new();
    let mut pending = vec![dir];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "parquet") {
                let file = std::fs::File::open(&path).unwrap();
                for batch in ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap() {
                    let batch = batch.unwrap();
                    let column = batch.column_by_name("step").unwrap();
                    let column = column.as_any().downcast_ref::<Int64Array>().unwrap();
                    steps.extend(column.values().iter().copied());
                }
            }
        }
    }
    steps.sort_unstable();
    steps
}
