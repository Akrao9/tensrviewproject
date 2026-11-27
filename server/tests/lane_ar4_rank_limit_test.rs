//! Lane AR-4 of wave AR, follow-up 1: the server takes no rank and no world size over its limit of
//! 100,000 ranks a run, at any of the three places a sender names one, and a restarted server takes
//! up a store that already holds such a run in bounded time and memory.
//!
//! The rule, at its edges:
//! - a `DoPut` stream whose descriptor names rank 100,000 or more ends at its first message with
//!   `INVALID_ARGUMENT` and the message `rank_refusal` below; nothing of it is stored and the watchdog
//!   holds nothing of it. Rank 99,999 is taken;
//! - a batch whose `app_metadata` names a `world_size` over 100,000 ends its stream with
//!   `INVALID_ARGUMENT` and `world_refusal`; nothing of that batch is stored. 100,000 is taken;
//! - a heartbeat from rank 100,000 or more is refused with `INVALID_ARGUMENT` and `beat_refusal`;
//!   rank 99,999's is taken.
//!
//! On `ce9da4a` all three were taken: AR-2 stored a run with `rank=2147483647`, and a world size of
//! 2,147,483,647 made every status of its run walk two thousand million ranks. The tests here use
//! values a few times the limit, which the base takes without a long walk, and `i32::MAX` once the
//! walk is bounded (added once it was).

use std::io::{BufRead, Cursor, Read};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::record_batch::RecordBatch;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::{Action, FlightClient, FlightData, FlightDescriptor, PutResult};
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

/// The most bytes a status file of these runs may take, whatever world size or rank was stored.
const SMALL: u64 = 64 * 1024;

fn rank_refusal(rank: i64) -> String {
    format!(
        "rank {rank} is over this server's limit of 100000 ranks a run (ranks 0 to 99999); nothing of this stream is stored"
    )
}

fn world_refusal(world_size: i64) -> String {
    format!(
        "world_size {world_size} is over this server's limit of 100000 ranks a run; nothing of this batch is stored"
    )
}

fn beat_refusal(rank: i64) -> String {
    format!(
        "rank {rank} is over this server's limit of 100000 ranks a run (ranks 0 to 99999); the heartbeat is not taken"
    )
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

/// Steps 0, 1 and 2 of `rank`, as one record batch.
fn batch(run: &str, rank: i64) -> RecordBatch {
    let text: String = (0..3)
        .map(|step| {
            let row = json!({"schema_version": 2, "run_id": run, "rank": rank, "attempt": 1, "step": step,
                             "ts": step as f64, "grad_norm": 1.0, "record_version": 1});
            format!("{row}\n")
        })
        .collect();
    let mut reader = arrow::json::ReaderBuilder::new(schema())
        .with_batch_size(3)
        .build(Cursor::new(text))
        .unwrap();
    reader.next().unwrap().unwrap()
}

/// The messages of a stream of `rank` holding steps 0 to 2, ending at byte 10, naming `world_size`
/// when it is given, as the SDK's sender sends them: the descriptor and schema, then the batch.
async fn stream(run: &str, rank: i64, world_size: Option<i64>) -> Vec<FlightData> {
    let mut metadata = json!({"schema_version": 2, "run_id": run, "rank": rank, "min_step": 0,
                              "max_step": 2, "end_byte_offset": 10, "sender_attempt": 1});
    if let Some(world_size) = world_size {
        metadata["world_size"] = json!(world_size);
    }
    let descriptor = FlightDescriptor::new_path(vec!["runs".into(), run.into(), format!("rank={rank}")]);
    let mut messages: Vec<FlightData> = FlightDataEncoderBuilder::new()
        .with_flight_descriptor(Some(descriptor))
        .with_max_flight_data_size(usize::MAX)
        .build(futures::stream::iter([Ok(batch(
            run,
            rank.clamp(0, i64::from(i32::MAX)),
        ))]))
        .try_collect()
        .await
        .unwrap();
    for message in messages.iter_mut().skip(1) {
        message.app_metadata = metadata.to_string().into();
    }
    messages
}

// --- a server in this process ----------------------------------------------------------------------

struct Server {
    address: String,
    ingest: Arc<Ingest>,
    _stop: oneshot::Sender<()>,
}

async fn start(root: &Path) -> Server {
    let store = Arc::new(LocalFileSystem::new_with_prefix(root).unwrap());
    let config = IngestConfig {
        flush_steps: 1,
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

async fn client(address: &str) -> FlightClient {
    let channel = Channel::from_shared(address.to_string())
        .unwrap()
        .connect()
        .await
        .unwrap();
    FlightClient::new(channel)
}

/// What a stream's call gave: the acknowledged offsets, then the status it ended with, if any.
async fn put(address: &str, messages: Vec<FlightData>) -> (Vec<i64>, Option<tonic::Status>) {
    let (sender, outgoing) = mpsc::unbounded_channel::<Result<FlightData, FlightError>>();
    for message in messages {
        sender.send(Ok(message)).unwrap();
    }
    drop(sender);
    let mut flight = client(address).await;
    let mut acks: BoxStream<'static, Result<PutResult, FlightError>> =
        match flight.do_put(UnboundedReceiverStream::new(outgoing)).await {
            Ok(acks) => acks,
            Err(error) => futures::stream::once(async move { Err(error) }).boxed(),
        };
    let mut offsets = Vec::new();
    let ended = tokio::time::timeout(BOUND, async {
        while let Some(reply) = acks.next().await {
            match reply {
                Ok(result) => {
                    let reply: Value = serde_json::from_slice(&result.app_metadata).unwrap();
                    offsets.push(reply["end_byte_offset"].as_i64().unwrap());
                }
                Err(FlightError::Tonic(status)) => return Some(*status),
                Err(other) => panic!("{other}"),
            }
        }
        None
    })
    .await
    .expect("the call never ended");
    (offsets, ended)
}

/// A stream sent and left open once its batch is acknowledged: its sender and its call, kept.
async fn open_held(
    address: &str,
    messages: Vec<FlightData>,
) -> (
    mpsc::UnboundedSender<Result<FlightData, FlightError>>,
    BoxStream<'static, Result<PutResult, FlightError>>,
) {
    let (sender, outgoing) = mpsc::unbounded_channel::<Result<FlightData, FlightError>>();
    for message in messages {
        sender.send(Ok(message)).unwrap();
    }
    let mut flight = client(address).await;
    let mut acks = flight.do_put(UnboundedReceiverStream::new(outgoing)).await.unwrap();
    let first = tokio::time::timeout(BOUND, acks.next())
        .await
        .expect("no acknowledgement");
    let reply: Value = serde_json::from_slice(&first.unwrap().unwrap().app_metadata).unwrap();
    assert_eq!(reply["end_byte_offset"], 10);
    (sender, acks)
}

async fn beat(address: &str, run: &str, rank: i64) -> Result<(), tonic::Status> {
    let body = json!({"run_id": run, "rank": rank, "phase": "train", "interval_seconds": 15}).to_string();
    let mut flight = client(address).await;
    match flight.do_action(Action::new("heartbeat", body)).await {
        Ok(replies) => replies
            .try_collect::<Vec<_>>()
            .await
            .map(|_| ())
            .map_err(|error| match error {
                FlightError::Tonic(status) => *status,
                other => panic!("{other}"),
            }),
        Err(FlightError::Tonic(status)) => Err(*status),
        Err(other) => panic!("{other}"),
    }
}

fn assert_refused(ended: &Option<tonic::Status>, message: &str, seen: &str) {
    let status = ended
        .as_ref()
        .unwrap_or_else(|| panic!("wanted INVALID_ARGUMENT; {seen}"));
    assert_eq!(status.code(), tonic::Code::InvalidArgument, "{seen}");
    assert_eq!(status.message(), message, "{seen}");
}

/// The rank folders of `run` in the store.
fn rank_folders(root: &Path, run: &str) -> Vec<String> {
    let mut found: Vec<String> = std::fs::read_dir(root.join("runs").join(run))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| entry.file_name().to_str().map(str::to_string))
        .filter(|name| name.starts_with("rank="))
        .collect();
    found.sort();
    found
}

fn status_ranks(ingest: &Ingest, run: &str) -> Vec<i64> {
    ingest.watchdog_statuses().get(run).map_or_else(Vec::new, |status| {
        status["ranks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["rank"].as_i64().unwrap())
            .collect()
    })
}

// --- the three doors -------------------------------------------------------------------------------

/// Rank 100,000, and higher, is refused at the stream's first message, with nothing stored and no
/// rank held by the watchdog. Rank 99,999 is taken.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stream_from_a_rank_over_the_limit_is_refused_and_one_below_it_taken() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    for rank in [100_000, 400_000, i64::from(i32::MAX)] {
        let run = format!("ar4_rank_{rank}");
        let (acks, ended) = put(&server.address, stream(&run, rank, None).await).await;
        let seen = format!(
            "acks {acks:?}, ended {ended:?}; folders {:?}",
            rank_folders(dir.path(), &run)
        );
        assert!(acks.is_empty(), "{seen}");
        assert_refused(&ended, &rank_refusal(rank), &seen);
        assert!(rank_folders(dir.path(), &run).is_empty(), "{seen}");
        assert!(status_ranks(&server.ingest, &run).is_empty(), "{seen}");
    }
    let run = "ar4_rank_99999";
    let (acks, ended) = put(&server.address, stream(run, 99_999, None).await).await;
    assert_eq!((acks, ended.map(|status| status.to_string())), (vec![10], None));
    assert_eq!(rank_folders(dir.path(), run), ["rank=99999"]);
    assert_eq!(status_ranks(&server.ingest, run), [99_999]);
}

/// A world size over 100,000 refuses its batch, and the stream ends; 100,000 is taken and written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_batch_naming_a_world_size_over_the_limit_is_refused_and_one_at_it_taken() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    for world in [100_001, 400_000, i64::from(i32::MAX)] {
        let run = format!("ar4_world_{world}");
        let (acks, ended) = put(&server.address, stream(&run, 0, Some(world)).await).await;
        let seen = format!(
            "acks {acks:?}, ended {ended:?}; folders {:?}",
            rank_folders(dir.path(), &run)
        );
        assert!(acks.is_empty(), "{seen}");
        assert_refused(&ended, &world_refusal(world), &seen);
        let statuses = server.ingest.watchdog_statuses();
        let world_seen = statuses.get(&run).map(|status| status["world_size"].clone());
        assert!(world_seen.as_ref().is_none_or(Value::is_null), "{seen}; {world_seen:?}");
    }
    let run = "ar4_world_100000";
    let (acks, ended) = put(&server.address, stream(run, 0, Some(100_000)).await).await;
    assert_eq!((acks, ended.map(|status| status.to_string())), (vec![10], None));
    assert_eq!(server.ingest.watchdog_statuses()[run]["world_size"], 100_000);
}

/// A heartbeat from rank 100,000 is refused, and the watchdog holds nothing of it; rank 99,999's
/// is taken.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_heartbeat_from_a_rank_over_the_limit_is_refused_and_one_below_it_taken() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    for rank in [100_000, i64::from(i32::MAX)] {
        let run = format!("ar4_beat_{rank}");
        let refused = beat(&server.address, &run, rank).await;
        let seen = format!("{refused:?}; ranks held {:?}", status_ranks(&server.ingest, &run));
        let status = refused.err().unwrap_or_else(|| panic!("taken: {seen}"));
        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{seen}");
        assert_eq!(status.message(), beat_refusal(rank), "{seen}");
        assert!(status_ranks(&server.ingest, &run).is_empty(), "{seen}");
    }
    let run = "ar4_beat_99999";
    beat(&server.address, run, 99_999).await.unwrap();
    assert_eq!(status_ranks(&server.ingest, run), [99_999]);
}

// --- a restart over a store that holds such a run ---------------------------------------------------

/// The server binary over `store`, writing statuses every 0.2 s; its standard error is read on a
/// thread.
struct Binary {
    child: std::process::Child,
    address: String,
    stderr: std::thread::JoinHandle<String>,
}

fn spawn_binary(store: &Path) -> Binary {
    let mut child = Command::new(env!("CARGO_BIN_EXE_tensorquill-server"))
        .args(["--listen", "127.0.0.1:0", "--store"])
        .arg(store)
        .args([
            "--flush-steps",
            "1",
            "--watchdog-seconds",
            "0.2",
            "--silent-seconds",
            "1",
        ])
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
    fn stop(mut self) -> String {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let stderr = std::mem::replace(&mut self.stderr, std::thread::spawn(String::new));
        stderr.join().unwrap()
    }
}

impl Drop for Binary {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn now_ms() -> i64 {
    let since = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    i64::try_from(since.as_millis()).unwrap()
}

fn status_file(root: &Path, run: &str) -> std::path::PathBuf {
    root.join("runs").join(run).join("watchdog.json")
}

/// The run's status file once `ready` holds for it, waited for at most `BOUND`.
async fn status_when(root: &Path, run: &str, what: &str, ready: impl Fn(&Value) -> bool) -> Value {
    let deadline = tokio::time::Instant::now() + BOUND;
    let mut last = None;
    loop {
        if let Some(status) = std::fs::read(status_file(root, run))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        {
            if ready(&status) {
                return status;
            }
            last = Some(status);
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the status never showed {what} within {BOUND:?}; last {}",
            last.map_or("none".to_string(), |status| {
                let text = status.to_string();
                text.chars().take(2_000).collect()
            })
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// A store holding a run of ranks 0 and 1 whose status was then made to name `world_size` and
/// whose rank 1 was made rank `rank`, in its status and its folder: as a server before the limit
/// could have left it. A restarted server takes the run up, writes its status within the bound and
/// small, with no world size and without that rank, and refuses that rank's stream.
async fn restart_over(world_size: i64, rank: i64) {
    let run = "ar4_restore";
    let dir = tempfile::tempdir().unwrap();
    let first = spawn_binary(dir.path());
    // The streams stay open, as a job's do while it runs: the server is stopped under them.
    let mut held = Vec::new();
    for one in [0, 1] {
        held.push(open_held(&first.address, stream(run, one, Some(2)).await).await);
    }
    status_when(dir.path(), run, "both ranks", |status| {
        status["ranks"].as_array().is_some_and(|ranks| ranks.len() == 2)
    })
    .await;
    first.stop();
    drop(held);
    let path = status_file(dir.path(), run);
    let mut status: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    status["world_size"] = json!(world_size);
    for list in ["ranks", "restart"] {
        for entry in status[list].as_array_mut().unwrap() {
            if entry["rank"] == 1 {
                entry["rank"] = json!(rank);
            }
        }
    }
    let written_before = status["written_at_ms"].as_i64().unwrap();
    std::fs::write(&path, serde_json::to_vec_pretty(&status).unwrap()).unwrap();
    let folder = dir.path().join("runs").join(run);
    std::fs::rename(folder.join("rank=1"), folder.join(format!("rank={rank}"))).unwrap();

    // A status written past the silence floor (1 s) after the start, when every rank expected and
    // not seen is missing.
    let started = std::time::Instant::now();
    let settled_ms = written_before.max(now_ms()) + 1_500;
    let second = spawn_binary(dir.path());
    let status = status_when(dir.path(), run, "a status written 1.5 s after the restart", |status| {
        status["written_at_ms"]
            .as_i64()
            .is_some_and(|written| written > settled_ms)
    })
    .await;
    let took = started.elapsed();
    let bytes = std::fs::metadata(&path).unwrap().len();
    let (acks, ended) = put(&second.address, stream(run, rank, None).await).await;
    let log = second.stop();
    let seen = format!(
        "{bytes} bytes, written {took:?} after the start; world_size {}, ranks {}, missing_ranks of {} entries; log:\n{log}",
        status["world_size"],
        json!(status["ranks"].as_array().unwrap().iter().map(|entry| entry["rank"].clone()).collect::<Vec<_>>()),
        status["missing_ranks"].as_array().map_or(0, Vec::len),
    );
    assert!(bytes < SMALL, "{seen}");
    assert_eq!(status["world_size"], Value::Null, "{seen}");
    let ranks: Vec<i64> = status["ranks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["rank"].as_i64().unwrap())
        .collect();
    assert_eq!(ranks, [0], "{seen}");
    assert_eq!(status["missing_ranks"], json!([]), "{seen}");
    assert!(acks.is_empty(), "{seen}");
    assert_refused(&ended, &rank_refusal(rank), &seen);
    assert!(!log.contains("panicked"), "{seen}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_restarted_server_takes_up_a_stored_run_over_the_limit_in_bounded_time() {
    restart_over(400_000, 400_000).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_restarted_server_takes_up_a_stored_run_of_world_size_and_rank_i32_max_in_bounded_time() {
    restart_over(i64::from(i32::MAX), i64::from(i32::MAX)).await;
}
