//! Lane Z3 of the SDK-22 wave (Nov 20, 2026): replica observations sent through a real server,
//! compared across its ranks, and carried in the run's `watchdog.json` under `replica`
//! (`REPLICA_OBSERVATION_CONTRACT.md`, sections 7 and 8).
//!
//! The clock of contract 7: it starts when the server acknowledges the batch with the last required
//! observation and stops at the first `watchdog.json` whose `replica` object holds the step's verdict.
//! The bound is one status interval. Each test measures that time for several steps, the last rank
//! sent at a different moment of the status writer's cycle each time.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
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
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Channel;

const RUN: &str = "run-replica";
const LAUNCH: &str = "2f6c1a9e4b7d4c1e9a3f6b8d0e1c2a47";
/// The status interval (`--watchdog-seconds`) these tests run with.
const INTERVAL: Duration = Duration::from_millis(400);

struct Server {
    address: String,
    ingest: Arc<Ingest>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

/// Every batch is stored and acknowledged as it arrives (`flush_steps` 1), and the status is
/// written every `INTERVAL`.
fn config() -> IngestConfig {
    IngestConfig {
        flush_steps: 1,
        flush_interval: Duration::from_secs(3600),
        watchdog_interval: INTERVAL,
        ..IngestConfig::default()
    }
}

async fn start(root: &Path) -> Server {
    let store = LocalFileSystem::new_with_prefix(root).unwrap();
    let ingest = Ingest::new(Storage::new(Arc::new(store), "".into()), config());
    ingest.restore_watchdog().await.unwrap();
    let tasks = vec![ingest.spawn_flusher(), ingest.spawn_watchdog()];
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let served = ingest.clone();
    tokio::spawn(async move {
        serve(listener, served, None, None, std::future::pending())
            .await
            .unwrap();
    });
    Server { address, ingest, tasks }
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("schema_version", DataType::Int32, false),
        Field::new("run_id", DataType::Utf8, false),
        Field::new("rank", DataType::Int32, false),
        Field::new("attempt", DataType::Int64, false),
        Field::new("step", DataType::Int64, false),
        Field::new("ts", DataType::Float64, true),
        Field::new("extra_json", DataType::Utf8, true),
    ]))
}

/// A rank's attempt: its own clock, so no two ranks share one (contract 2).
fn attempt(rank: i32) -> i64 {
    1_795_000_000_000 + 101 * i64::from(rank)
}

/// One rank's observation at a step, as the writer puts it in `extra_json`: `w`'s norm is `norm`.
fn observation(group: &[i32], norm: f64) -> Value {
    let params: serde_json::Map<String, Value> = (0..200)
        .map(|at| {
            let fp = if at == 0 {
                json!([norm, -1.25, 0.75])
            } else {
                json!([0.5, 0.25, -0.125])
            };
            let name = if at == 0 {
                "w".to_string()
            } else {
                format!("layers.{at}.weight")
            };
            (
                name,
                json!({"shape": [64, 64], "dtype": "bfloat16", "fp": fp, "sig": "-4611686018427387904"}),
            )
        })
        .collect();
    json!({"replica": {"v": 1, "launch": LAUNCH, "group": group, "device": "cuda", "projection": "name-v1",
        "signature": "fold64-v1", "params": params}})
}

fn batch(rank: i32, step: i64, extra: &Value) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(StringArray::from(vec![RUN])),
            Arc::new(Int32Array::from(vec![rank])),
            Arc::new(Int64Array::from(vec![attempt(rank)])),
            Arc::new(Int64Array::from(vec![step])),
            Arc::new(Float64Array::from(vec![step as f64])),
            Arc::new(StringArray::from(vec![extra.to_string()])),
        ],
    )
    .unwrap()
}

struct Put {
    rank: i32,
    sent: i64,
    messages: mpsc::UnboundedSender<Result<FlightData, FlightError>>,
    acks: BoxStream<'static, Result<PutResult, FlightError>>,
}

async fn open(server: &Server, rank: i32) -> Put {
    let channel = Channel::from_shared(server.address.clone())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightClient::new(channel);
    let (messages, outgoing) = mpsc::unbounded_channel();
    let descriptor = FlightDescriptor::new_path(vec!["runs".into(), RUN.into(), format!("rank={rank}")]);
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
    let acks = client.do_put(UnboundedReceiverStream::new(outgoing)).await.unwrap();
    Put {
        rank,
        sent: 0,
        messages,
        acks,
    }
}

impl Put {
    /// Sends one step's record and waits for its acknowledgement; returns when it arrived.
    async fn send(&mut self, step: i64, extra: &Value) -> Instant {
        self.sent += 1000;
        let metadata = format!(
            "{{\"schema_version\":1,\"run_id\":\"{RUN}\",\"rank\":{},\"min_step\":{step},\"max_step\":{step},\"end_byte_offset\":{},\"sender_attempt\":{}}}",
            self.rank,
            self.sent,
            attempt(self.rank)
        );
        let encoded: Vec<FlightData> = FlightDataEncoderBuilder::new()
            .with_max_flight_data_size(usize::MAX)
            .build(futures::stream::iter([Ok(batch(self.rank, step, extra))]))
            .try_collect()
            .await
            .unwrap();
        for mut message in encoded.into_iter().skip(1) {
            message.app_metadata = metadata.clone().into();
            self.messages.send(Ok(message)).unwrap();
        }
        let reply = tokio::time::timeout(Duration::from_secs(20), self.acks.next())
            .await
            .expect("no reply")
            .expect("the stream ended")
            .expect("refused");
        let at = Instant::now();
        let value: Value = serde_json::from_slice(&reply.app_metadata).unwrap();
        assert_eq!(value["end_byte_offset"], self.sent);
        at
    }
}

fn status(root: &Path) -> Value {
    std::fs::read(root.join("runs").join(RUN).join("watchdog.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or(Value::Null)
}

fn entry(status: &Value, step: i64) -> Option<Value> {
    status["replica"]["steps"]
        .as_array()?
        .iter()
        .find(|entry| entry["step"] == step)
        .cloned()
}

/// Looks at `watchdog.json` every 2 ms until `ready` holds of it; returns the status and when.
async fn first_status(root: &Path, what: &str, ready: impl Fn(&Value) -> bool) -> (Value, Instant) {
    let started = Instant::now();
    loop {
        let found = status(root);
        if ready(&found) {
            return (found, Instant::now());
        }
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "timed out waiting for {what}"
        );
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

/// `ranks` ranks send an observation at each of `steps` steps, the last rank late each time. Its
/// step reads `insufficient` until it arrives; once acknowledged, the verdict is in a status within
/// one interval. The last rank's `w` differs on odd steps, which raises `replica_drift`.
async fn verdict_within_one_interval(ranks: i32) -> Vec<Duration> {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    let group: Vec<i32> = (0..ranks).collect();
    let mut puts = Vec::new();
    for rank in 0..ranks {
        puts.push(open(&server, rank).await);
    }
    let mut waits = Vec::new();
    for (at, step) in (0..8).map(|at| (at, 99 + 100 * at)) {
        let drift = at % 2 == 1;
        for put in puts.iter_mut().take(ranks as usize - 1) {
            put.send(step, &observation(&group, 3.5)).await;
        }
        let (early, _) = first_status(root, "the step without its last rank", |status| {
            entry(status, step).is_some()
        })
        .await;
        let missing = vec![ranks - 1];
        assert_eq!(
            entry(&early, step).unwrap(),
            json!({"launch": LAUNCH, "group": group, "step": step, "coverage": "insufficient", "missing": missing})
        );
        // The last rank is sent at another moment of the status writer's cycle each time.
        tokio::time::sleep(INTERVAL * (at as u32 % 4) / 4 + Duration::from_millis(37 * at as u64)).await;
        let last = puts.last_mut().unwrap();
        let acknowledged = last
            .send(step, &observation(&group, if drift { 3.6 } else { 3.5 }))
            .await;
        let (judged, seen) = first_status(root, "the step's verdict", |status| {
            entry(status, step).is_some_and(|entry| entry["coverage"] == "complete")
        })
        .await;
        waits.push(seen.saturating_duration_since(acknowledged));
        let verdict = entry(&judged, step).unwrap();
        assert_eq!(
            verdict["fingerprints"],
            if drift { "drift" } else { "agree" },
            "{verdict}"
        );
        let raised: Vec<&Value> = judged["replica"]["replica_drift"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|incident| incident["step"] == step)
            .collect();
        assert_eq!(raised.len(), usize::from(drift), "{judged}");
        if drift {
            assert_eq!(
                raised[0],
                &json!({"step": step, "parameters": ["w"], "source": "replica", "launch": LAUNCH, "group": group})
            );
            // A finding about the run: the server's own run state counts it.
            let state = &judged["run_state"];
            assert_eq!(state["state"], "unhealthy", "{state}");
            assert!(state["reasons"][0]["kinds"]
                .as_array()
                .unwrap()
                .contains(&json!("replica_drift")));
        }
        assert_eq!(judged["replica"]["v"], 1);
        assert_eq!(judged["replica"]["tolerance"], 1e-6);
        assert!(judged["replica"].get("omitted").is_none(), "{judged}");
    }
    // No rank's own report says anything of it (contract 5; the per-rank rule reads only a spread).
    for rank in 0..ranks {
        let report: Value = serde_json::from_slice(
            &std::fs::read(root.join(format!("runs/{RUN}/rank={rank}/incidents.json"))).unwrap(),
        )
        .unwrap();
        assert!(report["incidents"].as_array().unwrap().is_empty(), "{report}");
        assert_eq!(report["kind_checks"]["replica_drift"]["value"], "not_configured");
    }
    server.ingest.stop(server.tasks).await.unwrap();
    waits
}

fn say(ranks: i32, waits: &[Duration]) {
    let ms: Vec<u128> = waits.iter().map(Duration::as_millis).collect();
    eprintln!(
        "lane Z3: {ranks} ranks, status interval {} ms: acknowledgement to the status with the verdict, ms: {ms:?} (max {})",
        INTERVAL.as_millis(),
        ms.iter().max().unwrap()
    );
}

/// The write of a status is a local file here; what it takes is added to the interval.
const WRITE_ALLOWANCE: Duration = Duration::from_millis(150);

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_ranks_the_verdict_is_written_within_one_status_interval() {
    let waits = verdict_within_one_interval(2).await;
    say(2, &waits);
    assert!(
        waits.iter().all(|wait| *wait <= INTERVAL + WRITE_ALLOWANCE),
        "{waits:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn three_ranks_the_verdict_is_written_within_one_status_interval() {
    let waits = verdict_within_one_interval(3).await;
    say(3, &waits);
    assert!(
        waits.iter().all(|wait| *wait <= INTERVAL + WRITE_ALLOWANCE),
        "{waits:?}"
    );
}

/// A rank that never reports leaves the step `insufficient` for good, naming it, through many
/// status writes; and a restarted server keeps every verdict written before it. A rank that reports
/// after the restart for a step left unfinished is counted, not compared.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_silent_rank_stays_missing_and_a_restart_keeps_what_was_written() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(root).await;
    let group = [0, 1, 2];
    let mut puts = Vec::new();
    for rank in 0..2 {
        puts.push(open(&server, rank).await);
    }
    // Step 99 in the world group, rank 2 silent; step 199 in a group of the two, which drifts.
    for put in puts.iter_mut() {
        put.send(99, &observation(&group, 3.5)).await;
    }
    for put in puts.iter_mut() {
        let norm = if put.rank == 1 { 3.6 } else { 3.5 };
        put.send(199, &observation(&[0, 1], norm)).await;
    }
    let (before, _) = first_status(root, "both steps", |status| {
        entry(status, 199).is_some_and(|entry| entry["coverage"] == "complete")
    })
    .await;
    // Several writes later, step 99 still names rank 2.
    tokio::time::sleep(INTERVAL * 4).await;
    let later = status(root);
    assert_eq!(entry(&later, 99).unwrap()["missing"], json!([2]));
    let written = before["replica"].clone();
    assert_eq!(written["replica_drift"].as_array().unwrap().len(), 1);
    server.ingest.stop(server.tasks).await.unwrap();
    drop(puts);

    let server = start(root).await;
    let (after, _) = first_status(root, "a status from the restarted server", |status| {
        status["server_running"] == true
    })
    .await;
    assert_eq!(after["replica"], written, "the restarted server keeps what was written");
    let mut late = open(&server, 2).await;
    late.send(99, &observation(&group, 3.5)).await;
    let (counted, _) = first_status(root, "the late row counted", |status| {
        status["replica"].get("omitted").is_some()
    })
    .await;
    assert_eq!(entry(&counted, 99).unwrap()["coverage"], "insufficient");
    assert_eq!(
        counted["replica"]["omitted"],
        json!({"frozen": [{"launch": LAUNCH, "step": 99, "why": "restart", "rows_not_compared": 1}]})
    );
    assert_eq!(counted["replica"]["replica_drift"], written["replica_drift"]);
    server.ingest.stop(server.tasks).await.unwrap();
}
