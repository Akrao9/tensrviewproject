//! Lane AR-4 of wave AR, follow-up 2: a panic anywhere in a `DoPut` stream's task, not only in the
//! decode, ends the client's call with a status, runs the stream's clean-up, leaves the rank as a
//! restart leaves it, and the server goes on.
//!
//! The panics are injected (`super::injected`, compiled into this crate's unit tests alone), at two
//! points: inside `accept`, once the rank's step watermark and counts were moved for a batch and
//! before its rows are held (the rank's buffer half changed), and at a stream's first message while
//! the `streams` mutex is held (the mutex poisoned).
//!
//! On the commit before the fix, a panic there skipped the clean-up, as the decode's did before it
//! was guarded: a client whose stream had a batch held got no reply and no status, the rank's stream
//! read `open`, the half-changed buffer refused the rows sent again as not newer, so they were never
//! stored though they were acknowledged, and a poisoned `streams` mutex made every later stream's
//! task panic too.

use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

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

use super::injected::{self, Point};
use super::{Ingest, IngestConfig};
use crate::server::serve;
use crate::storage::Storage;

/// The longest any condition here is waited for.
const BOUND: Duration = Duration::from_secs(20);
/// How long a reply that should come at once is waited for before the test says none came.
const PROMPT: Duration = Duration::from_secs(5);

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

fn batch(run: &str, steps: std::ops::Range<i64>) -> RecordBatch {
    let text: String = steps
        .map(|step| {
            let row = json!({"schema_version": 2, "run_id": run, "rank": 0, "attempt": 1, "step": step,
                             "ts": step as f64, "grad_norm": 1.0, "record_version": 1});
            format!("{row}\n")
        })
        .collect();
    let mut reader = arrow::json::ReaderBuilder::new(schema())
        .with_batch_size(64)
        .build(Cursor::new(text))
        .unwrap();
    reader.next().unwrap().unwrap()
}

/// The record batch message of `steps`, with the `app_metadata` the SDK's sender gives a batch
/// ending at `end_byte_offset`.
async fn messages(run: &str, steps: std::ops::Range<i64>, end_byte_offset: i64) -> Vec<FlightData> {
    let metadata = json!({"schema_version": 2, "run_id": run, "rank": 0, "min_step": null,
                          "max_step": null, "end_byte_offset": end_byte_offset, "sender_attempt": 1})
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

/// A batch stays in the buffer, not acknowledged, until its stream ends.
async fn start(root: &Path) -> Server {
    let store = Arc::new(LocalFileSystem::new_with_prefix(root).unwrap());
    let config = IngestConfig {
        flush_steps: 100_000,
        flush_interval: Duration::from_secs(3600),
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

struct Put {
    messages: tokio::sync::mpsc::UnboundedSender<Result<FlightData, FlightError>>,
    acks: BoxStream<'static, Result<PutResult, FlightError>>,
}

/// What the client's call gave next.
#[derive(Debug)]
enum Reply {
    Ack(i64),
    Error(tonic::Status),
    /// The response ended with no message and no status.
    Ended,
    /// Nothing came within the wait.
    Nothing,
}

async fn open(address: &str, run: &str) -> Put {
    let channel = Channel::from_shared(address.to_string())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightClient::new(channel);
    let (messages, outgoing) = tokio::sync::mpsc::unbounded_channel();
    let descriptor = FlightDescriptor::new_path(vec!["runs".into(), run.into(), "rank=0".into()]);
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
    Put { messages, acks }
}

impl Put {
    fn send(&self, messages: Vec<FlightData>) {
        for message in messages {
            let _ = self.messages.send(Ok(message));
        }
    }

    async fn reply(&mut self, wait: Duration) -> Reply {
        match tokio::time::timeout(wait, self.acks.next()).await {
            Err(_) => Reply::Nothing,
            Ok(None) => Reply::Ended,
            Ok(Some(Ok(result))) => {
                let reply: Value = serde_json::from_slice(&result.app_metadata).unwrap();
                Reply::Ack(reply["end_byte_offset"].as_i64().unwrap())
            }
            Ok(Some(Err(FlightError::Tonic(status)))) => Reply::Error(*status),
            Ok(Some(Err(other))) => panic!("{other}"),
        }
    }

    /// Closes the stream as `close()` does and gives its replies until the server ends it.
    async fn finish(self) -> Vec<Reply> {
        drop(self.messages);
        let mut acks = self.acks;
        let mut replies = Vec::new();
        tokio::time::timeout(BOUND, async {
            while let Some(reply) = acks.next().await {
                replies.push(match reply {
                    Ok(result) => {
                        let reply: Value = serde_json::from_slice(&result.app_metadata).unwrap();
                        Reply::Ack(reply["end_byte_offset"].as_i64().unwrap())
                    }
                    Err(FlightError::Tonic(status)) => Reply::Error(*status),
                    Err(other) => panic!("{other}"),
                });
            }
        })
        .await
        .expect("the stream never ended");
        replies
    }
}

/// The steps each stored part of rank 0 holds, as their names say (`part-<first>-<last>.parquet`).
fn parts(root: &Path, run: &str) -> Vec<(i64, i64)> {
    let mut found = Vec::new();
    let mut folders = vec![root.join("runs").join(run).join("rank=0")];
    while let Some(folder) = folders.pop() {
        for entry in std::fs::read_dir(&folder).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                folders.push(path);
            } else if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
                if let Some(range) = name
                    .strip_prefix("part-")
                    .and_then(|rest| rest.strip_suffix(".parquet"))
                {
                    let (first, last) = range.split_once('-').unwrap();
                    found.push((first.parse().unwrap(), last.parse().unwrap()));
                }
            }
        }
    }
    found.sort();
    found
}

/// The steps of every row stored for rank 0, read from its part files, in order: a part's name is
/// taken from the buffer's bounds, which a half-changed buffer can make wrong.
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

/// What rank 0's entry of the run's status says of its stream: `open`, `finished` or `dropped`.
fn stream_state(ingest: &Ingest, run: &str) -> Value {
    let statuses = ingest.watchdog_statuses();
    let Some(status) = statuses.get(run) else {
        return Value::Null;
    };
    status["ranks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["rank"] == 0)
        .map_or(Value::Null, |entry| entry["stream"].clone())
}

/// What the client got after the batch whose task panicked, and what the server shows of the
/// stream once its end has had time to be seen. Printed by every assertion below.
async fn after_the_panic(put: &mut Put, ingest: &Ingest, run: &str) -> (Reply, Reply, String) {
    let first = put.reply(PROMPT).await;
    let after = put.reply(PROMPT).await;
    let deadline = tokio::time::Instant::now() + PROMPT;
    while stream_state(ingest, run) == "open" && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let seen = format!(
        "the client got {first:?}, then {after:?} (each waited for at most {PROMPT:?}); rank 0's stream reads {}; {} bytes buffered",
        stream_state(ingest, run),
        ingest.buffered_bytes()
    );
    (first, after, seen)
}

/// The status a stream whose task panicked ends with: `INTERNAL`, saying the server failed on it,
/// with nothing of the panic (its message, "panicked", a source path).
fn assert_the_server_failed(reply: &Reply, seen: &str) {
    let Reply::Error(status) = reply else {
        panic!("wanted INTERNAL for a fault of the server; {seen}");
    };
    assert_eq!(status.code(), tonic::Code::Internal, "{seen}");
    let message = status.message();
    assert!(message.starts_with("the server failed on this stream"), "{seen}");
    for leak in [injected::MESSAGE, "panick", ".rs", "/", "\\"] {
        assert!(!message.contains(leak), "the message shows {leak:?}: {seen}");
    }
}

// --- the tests -------------------------------------------------------------------------------------

/// A stream whose second batch's task panics inside `accept`, with its first batch held in the
/// buffer. The call ends with `INTERNAL`, the rank's stream reads `dropped`, what was held is
/// dropped and its bytes given back to the quota, and a stream that sends the rows again from
/// offset 0, as the sender does, has every one of them stored.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_panic_inside_accept_ends_the_call_drops_what_was_held_and_the_rows_sent_again_are_stored() {
    let run = "ar4_panic_accept";
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    let mut put = open(&server.address, run).await;
    put.send(messages(run, 0..3, 10).await);
    let deadline = tokio::time::Instant::now() + BOUND;
    while server.ingest.buffered_bytes() == 0 {
        assert!(tokio::time::Instant::now() < deadline, "the first batch was never held");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    injected::arm(run, Point::Accept);
    put.send(messages(run, 3..6, 20).await);
    let (first, after, seen) = after_the_panic(&mut put, &server.ingest, run).await;
    let (stream, held, stored_then) = (
        stream_state(&server.ingest, run),
        server.ingest.buffered_bytes(),
        parts(dir.path(), run),
    );
    // The sender sends again from the offset it last had acknowledged, 0: every row is stored.
    let again = open(&server.address, run).await;
    again.send(messages(run, 0..3, 10).await);
    again.send(messages(run, 3..6, 20).await);
    let replies = again.finish().await;
    let (stored, steps) = (parts(dir.path(), run), stored_steps(dir.path(), run));
    let seen = format!(
        "{seen}; the stream sent again was answered {replies:?}, and then the parts were named {stored:?} and held the steps {steps:?}"
    );
    assert_the_server_failed(&first, &seen);
    assert!(matches!(after, Reply::Ended), "the call ends after its status: {seen}");
    assert_eq!(stream, "dropped", "{seen}");
    assert_eq!(held, 0, "{seen}");
    assert!(stored_then.is_empty(), "{seen}");
    assert_eq!(steps, (0..6).collect::<Vec<_>>(), "{seen}");
    assert!(matches!(replies.last(), Some(Reply::Ack(20))), "{seen}");
}

/// Another stream of the same rank, open when the panic came, with a batch held: what it held is
/// dropped with the rest, so it is ended at its next batch with `UNAVAILABLE`, as a restart would
/// end it, and is not told an offset past rows the server no longer has. Sent again, all is stored.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn another_open_stream_of_the_rank_is_ended_after_the_panic_and_nothing_it_sent_is_lost() {
    let run = "ar4_panic_other";
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    let mut other = open(&server.address, run).await;
    other.send(messages(run, 0..3, 10).await);
    let deadline = tokio::time::Instant::now() + BOUND;
    while server.ingest.buffered_bytes() == 0 {
        assert!(tokio::time::Instant::now() < deadline, "the first batch was never held");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let mut put = open(&server.address, run).await;
    injected::arm(run, Point::Accept);
    put.send(messages(run, 3..6, 20).await);
    let (first, _, seen) = after_the_panic(&mut put, &server.ingest, run).await;
    assert_the_server_failed(&first, &seen);
    assert_eq!(server.ingest.buffered_bytes(), 0, "{seen}");
    other.send(messages(run, 6..9, 30).await);
    let ended = other.reply(PROMPT).await;
    let Reply::Error(status) = &ended else {
        panic!("wanted UNAVAILABLE on the other stream, got {ended:?}; {seen}");
    };
    assert_eq!(status.code(), tonic::Code::Unavailable, "{status}");
    assert!(
        status.message().contains("after a fault on another of its streams"),
        "{status}"
    );
    assert!(matches!(other.reply(PROMPT).await, Reply::Ended));
    let again = open(&server.address, run).await;
    again.send(messages(run, 0..9, 30).await);
    let replies = again.finish().await;
    let (stored, steps) = (parts(dir.path(), run), stored_steps(dir.path(), run));
    assert_eq!(
        steps,
        (0..9).collect::<Vec<_>>(),
        "replies {replies:?}, parts {stored:?}"
    );
}

/// A panic while the `streams` mutex is held poisons it. The call still ends with `INTERNAL`, the
/// rank's stream reads `dropped`, and the server goes on: the next stream of the rank is taken and
/// stored, and so is a stream of another run.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_panic_holding_the_streams_mutex_ends_the_call_and_the_server_goes_on() {
    let run = "ar4_panic_streams";
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    injected::arm(run, Point::HoldingStreams);
    let mut put = open(&server.address, run).await;
    put.send(messages(run, 0..3, 10).await);
    let (first, after, seen) = after_the_panic(&mut put, &server.ingest, run).await;
    assert_the_server_failed(&first, &seen);
    assert!(matches!(after, Reply::Ended), "the call ends after its status: {seen}");
    assert_eq!(stream_state(&server.ingest, run), "dropped", "{seen}");
    for run in [run, "ar4_panic_streams_next"] {
        let again = open(&server.address, run).await;
        again.send(messages(run, 0..3, 10).await);
        let replies = again.finish().await;
        assert!(
            matches!(replies.as_slice(), [Reply::Ack(10)]),
            "{run}: {replies:?}; {seen}"
        );
        assert_eq!(parts(dir.path(), run), [(0, 2)], "{run}");
        assert_eq!(stored_steps(dir.path(), run), [0, 1, 2], "{run}");
        assert_eq!(stream_state(&server.ingest, run), "finished", "{run}");
    }
}
