//! Lane AR-4 of wave AR: a record batch that Arrow's IPC reader cannot decode (AQ5-R1, AQ5-R2).
//!
//! Lane AQ-5 found that a record batch message whose header names buffers its body does not hold
//! panics inside `arrow-ipc` 60.1.0 (through `arrow-buffer`), on the live `DoPut` decode and in
//! `tensorquill-server detect`. Its look at what a client then sees was one probe, run once and
//! deleted. These tests are that probe kept, on a live server and a real Flight client, and they hold
//! the rule the author set: the stream ends with an error the client sees (`INVALID_ARGUMENT`, naming
//! a malformed batch), and the stream's clean-up runs as for any stream that ends in an error.
//!
//! What each test looked for on the base commit (`ce9da4a`), where the decode was not guarded, is
//! written above it. Every wait is for a condition, with a deadline.

use std::io::{BufRead, Cursor, Read};
use std::path::Path;
use std::process::{Command, Stdio};
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
use tensorquill_server::ingest::{Ingest, IngestConfig};
use tensorquill_server::server::serve;
use tensorquill_server::storage::Storage;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Channel;

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

fn steps(run: &str, steps: std::ops::Range<i64>) -> Vec<Value> {
    steps
        .map(|step| {
            json!({"schema_version": 2, "run_id": run, "rank": 0, "attempt": 1, "step": step,
                   "ts": step as f64, "grad_norm": 1.0, "record_version": 1})
        })
        .collect()
}

fn batch(rows: &[Value]) -> RecordBatch {
    let text: String = rows.iter().map(|row| format!("{row}\n")).collect();
    let mut reader = arrow::json::ReaderBuilder::new(schema())
        .with_batch_size(rows.len().max(1))
        .build(Cursor::new(text))
        .unwrap();
    reader.next().unwrap().unwrap()
}

/// The record batch messages of `rows`, each with the `app_metadata` the SDK's sender gives a batch
/// ending at `end_byte_offset`.
async fn messages(run: &str, rows: &[Value], end_byte_offset: i64) -> Vec<FlightData> {
    let metadata = json!({"schema_version": 2, "run_id": run, "rank": 0, "min_step": null,
                          "max_step": null, "end_byte_offset": end_byte_offset, "sender_attempt": 1})
    .to_string();
    let encoded: Vec<FlightData> = FlightDataEncoderBuilder::new()
        .with_max_flight_data_size(usize::MAX)
        .build(futures::stream::iter([Ok(batch(rows))]))
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

/// AQ5-R1's case: a record batch message whose body is cut to half of what its header names.
async fn cut_short(run: &str, rows: &[Value], end_byte_offset: i64) -> Vec<FlightData> {
    let mut out = messages(run, rows, end_byte_offset).await;
    let last = out.last_mut().unwrap();
    let body = last.data_body.to_vec();
    last.data_body = body[..body.len() / 2].to_vec().into();
    out
}

// --- a server in this process ----------------------------------------------------------------------

struct Server {
    address: String,
    ingest: Arc<Ingest>,
    _stop: oneshot::Sender<()>,
}

async fn start(root: &Path, config: IngestConfig) -> Server {
    let store = Arc::new(LocalFileSystem::new_with_prefix(root).unwrap());
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

/// A part is written at every batch.
fn stored_at_once() -> IngestConfig {
    IngestConfig {
        flush_steps: 1,
        flush_interval: Duration::from_secs(3600),
        ..IngestConfig::default()
    }
}

/// A batch stays in the buffer, not acknowledged, until its stream ends or a test flushes.
fn held_in_the_buffer() -> IngestConfig {
    IngestConfig {
        flush_steps: 100_000,
        flush_interval: Duration::from_secs(3600),
        ..IngestConfig::default()
    }
}

struct Put {
    messages: mpsc::UnboundedSender<Result<FlightData, FlightError>>,
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
    let (messages, outgoing) = mpsc::unbounded_channel();
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

    /// Closes the stream as `close()` does; every reply until the server ends it must be an
    /// acknowledgement.
    async fn finish(self) {
        drop(self.messages);
        let mut acks = self.acks;
        tokio::time::timeout(BOUND, async {
            while let Some(reply) = acks.next().await {
                reply.expect("the stream ends cleanly");
            }
        })
        .await
        .expect("the stream never ended");
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

/// What rank 0's entry of the run's status says of its stream: `open`, `finished` or `dropped`.
fn stream_state(ingest: &Ingest, run: &str) -> Value {
    let statuses = ingest.watchdog_statuses();
    let status = statuses.get(run).unwrap_or_else(|| panic!("no status of {run}"));
    status["ranks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["rank"] == 0)
        .map(|entry| entry["stream"].clone())
        .unwrap_or_else(|| panic!("no rank 0 in {status}"))
}

/// The status a malformed batch must end its stream with: `INVALID_ARGUMENT`, naming a malformed
/// batch, and with nothing of a panic (no "panicked", no source path).
fn assert_refused_as_malformed(reply: &Reply, seen: &str) {
    let Reply::Error(status) = reply else {
        panic!("wanted INVALID_ARGUMENT for a malformed batch; {seen}");
    };
    assert_eq!(status.code(), tonic::Code::InvalidArgument, "{seen}");
    let message = status.message();
    assert!(message.contains("malformed"), "{seen}");
    for leak in ["panick", ".rs", "/", "\\", "Buffer::slice", "offset of the new Buffer"] {
        assert!(!message.contains(leak), "the message shows {leak:?}: {seen}");
    }
}

/// The clean-up of a stream that ended in an error: the call ended after its status, the rank's
/// stream reads `dropped`, not `open`, and the error is counted.
fn assert_cleaned_up(after: &Reply, ingest: &Ingest, run: &str, errors: u64, seen: &str) {
    assert!(matches!(after, Reply::Ended), "the call ends after its status: {seen}");
    assert_eq!(stream_state(ingest, run), "dropped", "{seen}");
    assert_eq!(ingest.stream_errors(), errors, "{seen}");
}

/// What the client got after its malformed batch, and what the server shows of the stream once its
/// end has had time to be seen: the reply, what the call gave after it, rank 0's stream state and the
/// count of stream errors. Printed by every assertion below, so a failure says all of it.
async fn after_malformed(put: &mut Put, ingest: &Ingest, run: &str) -> (Reply, Reply, String) {
    let refused = put.reply(PROMPT).await;
    let after = put.reply(PROMPT).await;
    let deadline = tokio::time::Instant::now() + PROMPT;
    while stream_state(ingest, run) == "open" && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let seen = format!(
        "the client got {refused:?}, then {after:?} (each waited for at most {PROMPT:?}); rank 0's stream reads {}; stream errors counted {}",
        stream_state(ingest, run),
        ingest.stream_errors()
    );
    (refused, after, seen)
}

/// Waits until `ready` holds, for at most `BOUND`.
async fn until(what: &str, mut ready: impl FnMut() -> bool) {
    let deadline = tokio::time::Instant::now() + BOUND;
    while !ready() {
        assert!(tokio::time::Instant::now() < deadline, "never: {what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

// --- the live path ---------------------------------------------------------------------------------

/// A stream whose second batch is cut short. Its first batch was stored and acknowledged. On
/// `ce9da4a` the stream's task panicked: the client got no reply and no status (its response just
/// ended), the rank's stream stayed `open` in the status, and no stream error was counted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_batch_cut_short_ends_its_stream_with_invalid_argument_and_its_clean_up_runs() {
    let run = "ar4_cut_after_stored";
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), stored_at_once()).await;
    let mut put = open(&server.address, run).await;
    put.send(messages(run, &steps(run, 0..3), 10).await);
    let first = put.reply(BOUND).await;
    assert!(matches!(first, Reply::Ack(10)), "{first:?}");
    put.send(cut_short(run, &steps(run, 3..6), 20).await);
    let (refused, after, seen) = after_malformed(&mut put, &server.ingest, run).await;
    assert_refused_as_malformed(&refused, &seen);
    assert_cleaned_up(&after, &server.ingest, run, 1, &seen);
    let ingest = server.ingest.clone();
    // Nothing of the malformed batch is stored.
    assert_eq!(parts(dir.path(), run), vec![(0, 2)]);
    // The server goes on serving the rank, and nothing it holds was left locked.
    let mut again = open(&server.address, run).await;
    again.send(messages(run, &steps(run, 3..6), 20).await);
    let taken = again.reply(BOUND).await;
    assert!(matches!(taken, Reply::Ack(20)), "{taken:?}");
    again.finish().await;
    assert_eq!(parts(dir.path(), run), vec![(0, 2), (3, 5)]);
    until("the second stream reads as finished", || {
        stream_state(&ingest, run) == "finished"
    })
    .await;
}

/// A stream whose first batch is still in the buffer, not yet acknowledged, when its second batch is
/// cut short. On `ce9da4a` the client was left waiting: the rank's buffer held the stream's
/// acknowledgement channel (the panic skipped `detach`), so the call neither ended nor got a status.
/// The rows buffered before are treated as those of any stream cut off by an error: kept, never
/// acknowledged on the stream that ended, and the batch sent again is taken without a row twice.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_batch_cut_short_after_one_still_buffered_does_not_leave_the_client_waiting() {
    let run = "ar4_cut_after_buffered";
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), held_in_the_buffer()).await;
    let mut put = open(&server.address, run).await;
    put.send(messages(run, &steps(run, 0..3), 10).await);
    let ingest = server.ingest.clone();
    until("the first batch is buffered", || ingest.buffered_bytes() > 0).await;
    put.send(cut_short(run, &steps(run, 3..6), 20).await);
    let (refused, after, seen) = after_malformed(&mut put, &server.ingest, run).await;
    assert_refused_as_malformed(&refused, &seen);
    assert_cleaned_up(&after, &server.ingest, run, 1, &seen);
    // The sender, told nothing past offset 0, sends both batches again from there.
    let again = open(&server.address, run).await;
    again.send(messages(run, &steps(run, 0..3), 10).await);
    again.send(messages(run, &steps(run, 3..6), 20).await);
    again.finish().await;
    assert_eq!(parts(dir.path(), run), vec![(0, 5)], "every step stored once");
}

/// A stream whose very first batch is cut short. On `ce9da4a` the client's response ended with no
/// status, as AQ-5's probe saw, and the rank's stream stayed `open`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_first_batch_cut_short_is_refused_the_same_way() {
    let run = "ar4_cut_first";
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path(), stored_at_once()).await;
    let mut put = open(&server.address, run).await;
    put.send(cut_short(run, &steps(run, 0..3), 10).await);
    let (refused, after, seen) = after_malformed(&mut put, &server.ingest, run).await;
    assert_refused_as_malformed(&refused, &seen);
    assert_cleaned_up(&after, &server.ingest, run, 1, &seen);
    assert!(parts(dir.path(), run).is_empty());
    let mut again = open(&server.address, run).await;
    again.send(messages(run, &steps(run, 0..3), 10).await);
    let taken = again.reply(BOUND).await;
    assert!(matches!(taken, Reply::Ack(10)), "{taken:?}");
    again.finish().await;
}

// --- the binary --------------------------------------------------------------------------------------

/// The server binary started on a free port over `store`; its standard error is read on a thread.
struct Binary {
    child: std::process::Child,
    address: String,
    stderr: std::thread::JoinHandle<String>,
}

fn spawn_binary(store: &Path) -> Binary {
    let mut child = Command::new(env!("CARGO_BIN_EXE_tensorquill-server"))
        .args(["--listen", "127.0.0.1:0", "--store"])
        .arg(store)
        .args(["--flush-steps", "1", "--watchdog-seconds", "0"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout = std::io::BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    stdout.read_line(&mut line).unwrap();
    let address = line
        .trim()
        .strip_prefix("tensorquill-server listening on ")
        .unwrap_or_else(|| panic!("no listening line: {line:?}"))
        .to_string();
    let mut stderr = child.stderr.take().unwrap();
    let stderr = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = stderr.read_to_string(&mut text);
        text
    });
    Binary {
        child,
        address: format!("http://{address}"),
        stderr,
    }
}

impl Binary {
    /// Stops the server and gives what it wrote on standard error.
    fn stop(mut self) -> String {
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.stderr.join().unwrap()
    }
}

/// The binary as shipped (the release profile unwinds; `server/Cargo.toml` sets no `panic`). On
/// `ce9da4a` its standard error held the panic hook's "thread ... panicked at" line with the path of
/// arrow-buffer's source in the cargo registry, then the stream's line "ended in an error (code
/// Internal): the stream's task panicked". Wanted: the stream's ordinary line for a refusal, with the
/// malformed batch named, and no panic at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_binary_refuses_a_batch_cut_short_without_a_panic_in_its_log() {
    let run = "ar4_binary";
    let dir = tempfile::tempdir().unwrap();
    let binary = spawn_binary(dir.path());
    let mut put = open(&binary.address, run).await;
    put.send(messages(run, &steps(run, 0..3), 10).await);
    let first = put.reply(BOUND).await;
    assert!(matches!(first, Reply::Ack(10)), "{first:?}");
    put.send(cut_short(run, &steps(run, 3..6), 20).await);
    let refused = put.reply(PROMPT).await;
    let after = put.reply(PROMPT).await;
    // The server goes on serving: a second stream is taken.
    let mut again = open(&binary.address, run).await;
    again.send(messages(run, &steps(run, 3..6), 20).await);
    let taken = again.reply(BOUND).await;
    again.finish().await;
    drop(put);
    let log = binary.stop();
    let seen = format!("the client got {refused:?}, then {after:?}; the server's log:\n{log}");
    assert_refused_as_malformed(&refused, &seen);
    assert!(matches!(after, Reply::Ended), "{seen}");
    assert!(matches!(taken, Reply::Ack(20)), "{taken:?}\n{log}");
    assert!(!log.contains("panicked"), "the log shows a panic:\n{log}");
    assert!(
        log.contains(&format!(
            "the stream of {run} rank 0 ended in an error (code InvalidArgument)"
        )),
        "{log}"
    );
    assert!(log.contains("malformed"), "{log}");
    assert_eq!(parts(dir.path(), run), vec![(0, 2), (3, 5)]);
}
