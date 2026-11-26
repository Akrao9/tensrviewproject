//! Lane AP-2 of wave AP: a run that finished cleanly must not be left with a `watchdog.json` that
//! says its state is `unknown`.
//!
//! - AO-02: a batch taken in part. `Ingest::accept` takes a batch's rows piece by piece; when a later
//!   piece cannot be taken (the quota, a write that failed, a read of the store) the error returned
//!   before the watchdog was told of the pieces already taken. The sender sends the batch again,
//!   every row taken is then not newer, and the rank's `last_step` never moves: `last_step` null,
//!   `attempt_without_steps` true and the run `unknown`, though its rows are stored and judged.
//! - AO-03: a retry pending at the stop. A report whose write failed (`report_after`) or a rebuild of
//!   the detectors that failed (`replay_after`) waits for its backoff; the stop flushed and wrote the
//!   status without trying it, so a store healthy again by then was left with no `incidents.json`
//!   and the run `unknown`. A restarted server does not take a finished run up, so nothing judges it
//!   later. A run whose rank waits for such a retry was also let go of by `evict_run`.
//!
//! Every wait here is for a condition, with a deadline; no test synchronises by sleeping.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::record_batch::RecordBatch;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::{Action, FlightClient, FlightData, FlightDescriptor, PutResult};
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt, TryStreamExt};
use object_store::local::LocalFileSystem;
use object_store::path::Path as StorePath;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore, PutMultipartOptions,
    PutOptions, PutPayload,
};
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
/// How long a stop over a store that still fails may take (it tries each pending retry once, with
/// no backoff): far less than any backoff the tests set.
const STOP_BOUND: Duration = Duration::from_secs(10);
/// Bytes of padding on a "16 kB" row.
const PAD: usize = 16_000;

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

fn padding(pad: usize) -> Value {
    if pad == 0 {
        Value::Null
    } else {
        json!("x".repeat(pad))
    }
}

/// A training row: row version 2, the record's version 1, `pad` bytes of padding.
fn step_row(run: &str, rank: i32, attempt: i64, step: i64, pad: usize) -> Value {
    json!({"schema_version": 2, "run_id": run, "rank": rank, "attempt": attempt, "step": step,
           "ts": step as f64, "grad_norm": 1.0, "record_version": 1, "extra_json": padding(pad)})
}

/// A whole scalar-only row (section 4.4 of `WAVE_AL_CONTRACT.md`): its `seq`, the record's version 2.
fn scalar_row(run: &str, rank: i32, attempt: i64, seq: i64, pad: usize) -> Value {
    json!({"schema_version": 2, "run_id": run, "rank": rank, "attempt": attempt, "step": seq,
           "ts": 1000.0 + seq as f64, "scalars": {"eval_loss": 1.0}, "seq": seq, "record_version": 2,
           "extra_json": padding(pad)})
}

fn steps(run: &str, rank: i32, attempt: i64, steps: std::ops::Range<i64>, pad: usize) -> Vec<Value> {
    steps.map(|step| step_row(run, rank, attempt, step, pad)).collect()
}

fn batch(rows: &[Value]) -> RecordBatch {
    let text: String = rows.iter().map(|row| format!("{row}\n")).collect();
    let mut reader = arrow::json::ReaderBuilder::new(schema())
        .with_batch_size(rows.len().max(1))
        .build(Cursor::new(text))
        .unwrap();
    let batch = reader.next().unwrap().unwrap();
    assert!(reader.next().is_none());
    batch
}

// --- a local store that can be told to fail --------------------------------------------------------

/// A local store that can be told to fail the reads of parts (a rebuild of the detectors reads them;
/// the step watermark reads only their names), the writes of `scalars-*` objects and the writes of
/// `incidents.json`.
#[derive(Debug)]
struct FaultStore {
    inner: Arc<dyn ObjectStore>,
    fail_part_gets: AtomicBool,
    fail_scalar_puts: AtomicBool,
    fail_incident_puts: AtomicBool,
}

impl std::fmt::Display for FaultStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FaultStore({})", self.inner)
    }
}

impl FaultStore {
    fn new(root: &Path) -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(LocalFileSystem::new_with_prefix(root).unwrap()),
            fail_part_gets: AtomicBool::new(false),
            fail_scalar_puts: AtomicBool::new(false),
            fail_incident_puts: AtomicBool::new(false),
        })
    }
}

fn failure() -> object_store::Error {
    object_store::Error::Generic {
        store: "FaultStore",
        source: "the test fails this request".into(),
    }
}

impl ObjectStore for FaultStore {
    fn put_opts<'life0, 'life1, 'async_trait>(
        &'life0 self,
        location: &'life1 StorePath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> BoxFuture<'async_trait, object_store::Result<object_store::PutResult>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        let name = location.filename().unwrap_or_default();
        if (name.starts_with("scalars-") && self.fail_scalar_puts.load(Ordering::SeqCst))
            || (name == "incidents.json" && self.fail_incident_puts.load(Ordering::SeqCst))
        {
            return futures::future::err(failure()).boxed();
        }
        self.inner.put_opts(location, payload, opts)
    }

    fn put_multipart_opts<'life0, 'life1, 'async_trait>(
        &'life0 self,
        location: &'life1 StorePath,
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
        location: &'life1 StorePath,
        options: GetOptions,
    ) -> BoxFuture<'async_trait, object_store::Result<GetResult>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        let name = location.filename().unwrap_or_default();
        if name.starts_with("part-") && self.fail_part_gets.load(Ordering::SeqCst) {
            return futures::future::err(failure()).boxed();
        }
        self.inner.get_opts(location, options)
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<StorePath>>,
    ) -> BoxStream<'static, object_store::Result<StorePath>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&StorePath>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_delimiter<'life0, 'life1, 'async_trait>(
        &'life0 self,
        prefix: Option<&'life1 StorePath>,
    ) -> BoxFuture<'async_trait, object_store::Result<ListResult>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        self.inner.list_with_delimiter(prefix)
    }

    fn copy_opts<'life0, 'life1, 'life2, 'async_trait>(
        &'life0 self,
        from: &'life1 StorePath,
        to: &'life2 StorePath,
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

// --- a server in this process ----------------------------------------------------------------------

struct Server {
    address: String,
    ingest: Arc<Ingest>,
    flusher: Option<tokio::task::JoinHandle<()>>,
    _stop: oneshot::Sender<()>,
}

/// A server over `store`. Its timed flush runs when `flusher`; the status is written only when a
/// test asks (`status`), never by a task of its own.
async fn start(store: Arc<dyn ObjectStore>, config: IngestConfig, flusher: bool) -> Server {
    let ingest = Ingest::new(Storage::new(store, "".into()), config);
    let flusher = flusher.then(|| ingest.spawn_flusher());
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
        flusher,
        _stop: stop,
    }
}

/// The reproduced AO-02 server: a quota of 1 MiB, a buffer written at 1,000 steps, at an hour, or at
/// the end of its stream.
fn quota_config() -> IngestConfig {
    IngestConfig {
        quota_bytes: 1024 * 1024,
        flush_steps: 1000,
        flush_interval: Duration::from_secs(3600),
        ..IngestConfig::default()
    }
}

/// A server whose buffers are written at each batch, and whose retries after a failed report or
/// rebuild wait an hour: nothing but the stop (or a test) tries them again.
fn retry_config() -> IngestConfig {
    IngestConfig {
        flush_steps: 1,
        flush_interval: Duration::from_secs(3600),
        replay_backoff: Duration::from_secs(3600),
        ..IngestConfig::default()
    }
}

struct Put {
    run: String,
    rank: i32,
    messages: mpsc::UnboundedSender<Result<FlightData, FlightError>>,
    acks: BoxStream<'static, Result<PutResult, FlightError>>,
}

async fn open(address: &str, run: &str, rank: i32) -> Put {
    let channel = Channel::from_shared(address.to_string())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightClient::new(channel);
    let (messages, outgoing) = mpsc::unbounded_channel();
    let descriptor = FlightDescriptor::new_path(vec!["runs".into(), run.into(), format!("rank={rank}")]);
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
    Put {
        run: run.to_string(),
        rank,
        messages,
        acks,
    }
}

impl Put {
    async fn send(&mut self, rows: &[Value], end_byte_offset: i64) {
        let metadata = json!({"schema_version": 2, "run_id": self.run, "rank": self.rank, "min_step": null,
                              "max_step": null, "end_byte_offset": end_byte_offset})
        .to_string();
        let encoded: Vec<FlightData> = FlightDataEncoderBuilder::new()
            .with_max_flight_data_size(usize::MAX)
            .build(futures::stream::iter([Ok(batch(rows))]))
            .try_collect()
            .await
            .unwrap();
        for mut message in encoded.into_iter().skip(1) {
            message.app_metadata = metadata.clone().into();
            let _ = self.messages.send(Ok(message));
        }
    }

    /// The next acknowledged offset, or the status that ended the stream.
    async fn reply(&mut self) -> Result<i64, tonic::Status> {
        match tokio::time::timeout(BOUND, self.acks.next()).await.expect("no reply") {
            Some(Ok(result)) => {
                let reply: Value = serde_json::from_slice(&result.app_metadata).unwrap();
                Ok(reply["end_byte_offset"].as_i64().unwrap())
            }
            Some(Err(FlightError::Tonic(status))) => Err(*status),
            Some(Err(other)) => panic!("{other}"),
            None => panic!("the stream ended without a reply"),
        }
    }

    /// Closes the stream as `close()` does and waits for the server to end it; every reply after
    /// the close must be an acknowledgement.
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

/// A heartbeat of `rank` that names `attempt`, as the SDK's thread sends one.
async fn beat(address: &str, run: &str, rank: i32, attempt: i64) {
    let channel = Channel::from_shared(address.to_string())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightClient::new(channel);
    let body = json!({"run_id": run, "rank": rank, "phase": "train", "interval_seconds": 1.0, "work": 0,
                      "attempt": attempt})
    .to_string();
    let mut replies = client.do_action(Action::new("heartbeat", body)).await.unwrap();
    while let Some(reply) = replies.next().await {
        reply.unwrap();
    }
}

/// Waits until `ready` holds, for at most `BOUND`.
async fn until(what: &str, mut ready: impl FnMut() -> bool) {
    let deadline = tokio::time::Instant::now() + BOUND;
    while !ready() {
        assert!(tokio::time::Instant::now() < deadline, "never: {what}");
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

// --- what the store holds --------------------------------------------------------------------------

fn rank_dir(root: &Path, run: &str, rank: i32) -> PathBuf {
    root.join("runs").join(run).join(format!("rank={rank}"))
}

fn json_file(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))).unwrap()
}

fn report(root: &Path, run: &str, rank: i32) -> Option<Value> {
    let path = rank_dir(root, run, rank).join("incidents.json");
    path.is_file().then(|| json_file(&path))
}

/// The run's `watchdog.json` as written last.
fn written(root: &Path, run: &str) -> Value {
    json_file(&root.join("runs").join(run).join("watchdog.json"))
}

/// The run's status, written now and read back.
async fn status(server: &Server, root: &Path, run: &str) -> Value {
    server.ingest.write_watchdog().await;
    written(root, run)
}

fn entry(status: &Value, rank: i32) -> Value {
    status["ranks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["rank"] == rank)
        .cloned()
        .unwrap_or_else(|| panic!("no rank {rank} in {status}"))
}

fn codes(status: &Value) -> Vec<String> {
    status["run_state"]["reasons"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|reason| reason["code"].as_str().map(str::to_string))
        .collect()
}

/// Every rank of the run is judged: no reason of the run state says a rank is not.
fn judged(status: &Value) -> bool {
    let codes = codes(status);
    !codes.iter().any(|code| code == "ranks_unknown" || code == "no_steps")
}

/// What the status says of rank 0's progress: its attempt, `last_step` and `attempt_without_steps`.
fn progress(status: &Value) -> (Value, Value, bool) {
    let rank = entry(status, 0);
    (
        rank["attempt"].clone(),
        rank["last_step"].clone(),
        rank["attempt_without_steps"] == true,
    )
}

// --- AO-02 -----------------------------------------------------------------------------------------

/// The reproduced setup: both ranks stored attempt 100's steps 0 to 9, heartbeats announced attempt
/// 200 on both, and rank 1 holds the quota with 58 rows of 16 kB of attempt 200 not yet stored. The
/// stream of rank 1 is returned open.
async fn quota_held_by_rank_1(server: &Server, run: &str) -> Put {
    for rank in [0, 1] {
        let mut put = open(&server.address, run, rank).await;
        put.send(&steps(run, rank, 100, 0..10, 0), 10).await;
        put.finish().await;
    }
    for rank in [0, 1] {
        beat(&server.address, run, rank, 200).await;
    }
    let mut holder = open(&server.address, run, 1).await;
    holder.send(&steps(run, 1, 200, 0..58, PAD), 68).await;
    let ingest = server.ingest.clone();
    until("rank 1 holds the quota", || ingest.buffered_bytes() >= 58 * PAD).await;
    holder
}

/// AO-02, the scalar path, as reproduced: rank 0's batch of steps 5 to 9 and ten scalar-only rows of
/// 16 kB meets the quota at its scalar rows, after its step rows were taken (and stored and judged by
/// the flush that tried to make room). Once the batch sent again is accepted, rank 0's `last_step`
/// is 9 and its attempt has steps, as if the batch had been taken at once, and the run is judged. A
/// later batch moves `last_step` on. On `8c77d06` the status said `last_step` null,
/// `attempt_without_steps` true and `ranks_unknown: [0]`.
#[tokio::test]
async fn ao02_a_batch_cut_at_its_scalar_rows_leaves_the_rank_at_its_last_step() {
    let run = "ap2_scalar_cut";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(FaultStore::new(root), quota_config(), true).await;
    let holder = quota_held_by_rank_1(&server, run).await;
    let mut cut: Vec<Value> = steps(run, 0, 200, 5..10, 0);
    cut.extend((0..10).map(|seq| scalar_row(run, 0, 200, seq, PAD)));
    let mut put = open(&server.address, run, 0).await;
    put.send(&cut, 25).await;
    let refused = put.reply().await.expect_err("the scalar rows do not fit");
    assert_eq!(refused.code(), tonic::Code::ResourceExhausted, "{refused:?}");
    // The step rows were taken: stored and judged by the flush that tried to make room.
    let judged_at = report(root, run, 0).expect("rank 0 judged");
    assert_eq!(
        (&judged_at["attempt"], &judged_at["last_step"]),
        (&json!(200), &json!(9))
    );
    // The watchdog was told of them when they were taken, not only when the whole batch is.
    let now = status(&server, root, run).await;
    assert_eq!(progress(&now), (json!(200), json!(9), false), "{now}");
    holder.finish().await;
    // The sender sends the batch again, on a new stream; it is accepted.
    let mut put = open(&server.address, run, 0).await;
    put.send(&cut, 25).await;
    assert_eq!(put.reply().await.expect("the batch sent again is accepted"), 25);
    let now = status(&server, root, run).await;
    assert_eq!(progress(&now), (json!(200), json!(9), false), "{now}");
    // L30's count is unchanged in meaning: the five step rows sent again were not newer.
    assert_eq!(entry(&now, 0)["rows_not_newer"], 5, "{now}");
    // A later batch, after the cut one.
    put.send(&steps(run, 0, 200, 10..13, 0), 28).await;
    put.finish().await;
    let now = status(&server, root, run).await;
    assert_eq!(progress(&now), (json!(200), json!(12), false), "{now}");
    assert_eq!(now["summary"], "finished", "{now}");
    assert!(judged(&now), "{}", now["run_state"]);
    let judged_at = report(root, run, 0).unwrap();
    assert_eq!(
        (&judged_at["attempt"], &judged_at["last_step"]),
        (&json!(200), &json!(12))
    );
    server.ingest.shutdown().await.unwrap();
}

/// AO-02, the two-attempt path (older than protocol 5): rank 0's batch holds the owner attempt 200's
/// steps 5 to 9 and then a backlog of attempt 100, steps 10 to 19 of 16 kB, which meets the quota
/// after the owner's rows were taken. Once the batch sent again is accepted, `last_step` is 9 and
/// the backlog's ten rows are counted as superseded, as if the batch had been taken at once.
#[tokio::test]
async fn ao02_a_batch_cut_at_its_second_attempt_leaves_the_rank_at_its_last_step() {
    let run = "ap2_attempt_cut";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let server = start(FaultStore::new(root), quota_config(), true).await;
    let holder = quota_held_by_rank_1(&server, run).await;
    let mut cut: Vec<Value> = steps(run, 0, 200, 5..10, 0);
    cut.extend(steps(run, 0, 100, 10..20, PAD));
    let mut put = open(&server.address, run, 0).await;
    put.send(&cut, 25).await;
    let refused = put.reply().await.expect_err("the backlog does not fit");
    assert_eq!(refused.code(), tonic::Code::ResourceExhausted, "{refused:?}");
    let now = status(&server, root, run).await;
    assert_eq!(progress(&now), (json!(200), json!(9), false), "{now}");
    holder.finish().await;
    let mut put = open(&server.address, run, 0).await;
    put.send(&cut, 25).await;
    put.finish().await;
    let now = status(&server, root, run).await;
    assert_eq!(progress(&now), (json!(200), json!(9), false), "{now}");
    assert_eq!(entry(&now, 0)["superseded_rows"], 10, "{now}");
    assert_eq!(entry(&now, 0)["rows_not_newer"], 5, "{now}");
    assert!(judged(&now), "{}", now["run_state"]);
    server.ingest.shutdown().await.unwrap();
}

/// AO-02 with a write that failed rather than the quota. A batch of one scalar-only row is refused
/// because its object cannot be written, which seals the rank's scalar rows; the next batch, steps
/// 0 to 2 and another scalar-only row, has its step rows taken, and its scalar rows must first
/// store the sealed ones: that flush stores and judges the step part and fails again at the scalars
/// object. Once the store writes again and the sender's batches sent again are accepted,
/// `last_step` is 2 and the run is judged; on `8c77d06` `last_step` stayed null.
#[tokio::test]
async fn ao02_a_batch_cut_by_a_failed_scalars_write_leaves_the_rank_at_its_last_step() {
    let run = "ap2_write_cut";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let store = FaultStore::new(root);
    let server = start(store.clone(), retry_config(), false).await;
    let first = vec![scalar_row(run, 0, 1, 0, 0)];
    let mut cut: Vec<Value> = steps(run, 0, 1, 0..3, 0);
    cut.push(scalar_row(run, 0, 1, 1, 0));
    store.fail_scalar_puts.store(true, Ordering::SeqCst);
    let mut put = open(&server.address, run, 0).await;
    put.send(&first, 1).await;
    let refused = put.reply().await.expect_err("the scalars object is not stored");
    assert_eq!(refused.code(), tonic::Code::Unavailable, "{refused:?}");
    let mut put = open(&server.address, run, 0).await;
    put.send(&cut, 5).await;
    let refused = put
        .reply()
        .await
        .expect_err("the sealed scalars object is still not stored");
    assert_eq!(refused.code(), tonic::Code::Unavailable, "{refused:?}");
    let judged_at = report(root, run, 0).expect("the step part was stored and judged");
    assert_eq!(judged_at["last_step"], 2);
    let now = status(&server, root, run).await;
    assert_eq!(progress(&now), (json!(1), json!(2), false), "{now}");
    store.fail_scalar_puts.store(false, Ordering::SeqCst);
    // The sender sends again from its offset, 0: both batches.
    let mut put = open(&server.address, run, 0).await;
    put.send(&first, 1).await;
    assert_eq!(put.reply().await.unwrap(), 1);
    put.send(&cut, 5).await;
    assert_eq!(put.reply().await.unwrap(), 5);
    put.finish().await;
    let now = status(&server, root, run).await;
    assert_eq!(progress(&now), (json!(1), json!(2), false), "{now}");
    assert!(judged(&now), "{}", now["run_state"]);
    server.ingest.shutdown().await.unwrap();
}

// --- AO-03 -----------------------------------------------------------------------------------------

/// A three-step capture of rank 0, closed cleanly, whose report could not be written: in the store
/// `runs/<run>/rank=0/incidents.json` is a directory. Returns that directory.
async fn three_steps_with_the_report_refused(server: &Server, root: &Path, run: &str) -> PathBuf {
    let blocked = rank_dir(root, run, 0).join("incidents.json");
    std::fs::create_dir_all(blocked.join("in_the_way")).unwrap();
    let mut put = open(&server.address, run, 0).await;
    put.send(&steps(run, 0, 1, 0..3, 0), 3).await;
    assert_eq!(put.reply().await.unwrap(), 3);
    put.finish().await;
    assert!(report(root, run, 0).is_none());
    blocked
}

/// AO-03 as reproduced: the store is healthy again by the stop, and the report waits for its
/// backoff (an hour). The stop tries it once at once, before the status: the report is written and
/// the run is judged. On `8c77d06` there was no `incidents.json`, `summary` `finished` and the run
/// `unknown`.
#[tokio::test]
async fn ao03_a_report_pending_at_the_stop_is_written_before_the_status() {
    let run = "ap2_report_at_stop";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let mut server = start(FaultStore::new(root), retry_config(), true).await;
    let blocked = three_steps_with_the_report_refused(&server, root, run).await;
    std::fs::remove_dir_all(&blocked).unwrap();
    let tasks: Vec<_> = server.flusher.take().into_iter().collect();
    server.ingest.stop(tasks).await.unwrap();
    let judged_at = report(root, run, 0).expect("the report is written at the stop");
    assert_eq!((&judged_at["attempt"], &judged_at["last_step"]), (&json!(1), &json!(2)));
    let last = written(root, run);
    assert_eq!(last["summary"], "finished", "{last}");
    assert_eq!(progress(&last), (json!(1), json!(2), false), "{last}");
    assert!(judged(&last), "{}", last["run_state"]);
}

/// AO-03 with a store that still fails at the stop: the stop tries the report once, returns within
/// `STOP_BOUND`, and the status says truthfully that rank 0 is not judged. Then the audit's claim
/// that was not verified: a server started again on that store, the store healthy, does not take
/// the finished run up (`restore_watchdog` takes up 0 runs), and its timed flush and status writes
/// never judge the rank. So the stop is the last chance, and a rank not judged there stays so.
#[tokio::test]
async fn ao03_a_store_still_failing_at_the_stop_ends_promptly_and_a_restart_does_not_take_the_run_up() {
    let run = "ap2_still_failing";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let mut server = start(FaultStore::new(root), retry_config(), true).await;
    let blocked = three_steps_with_the_report_refused(&server, root, run).await;
    let tasks: Vec<_> = server.flusher.take().into_iter().collect();
    tokio::time::timeout(STOP_BOUND, server.ingest.stop(tasks))
        .await
        .expect("the stop ends promptly")
        .unwrap();
    assert!(report(root, run, 0).is_none());
    let last = written(root, run);
    assert_eq!(last["summary"], "finished", "{last}");
    assert!(
        codes(&last).contains(&"ranks_unknown".to_string()),
        "{}",
        last["run_state"]
    );
    // A restart over the healthy store.
    std::fs::remove_dir_all(&blocked).unwrap();
    let again = start(FaultStore::new(root), retry_config(), false).await;
    assert_eq!(
        again.ingest.restore_watchdog().await.unwrap(),
        0,
        "a finished run is not taken up"
    );
    again.ingest.flush_due().await;
    again.ingest.write_watchdog().await;
    assert!(
        report(root, run, 0).is_none(),
        "nothing judges the rank after a restart"
    );
    assert_eq!(written(root, run), last, "the status is the stop's");
}

/// AO-03 for a rebuild: the rank's first part is stored while its parts cannot be read, so the
/// rebuild of its detectors fails and waits for its backoff (an hour). The store reads again by the
/// stop, which tries the rebuild once at once: the report is written and the run is judged.
#[tokio::test]
async fn ao03_a_rebuild_pending_at_the_stop_is_tried_before_the_status() {
    let run = "ap2_rebuild_at_stop";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let store = FaultStore::new(root);
    let mut server = start(store.clone(), retry_config(), true).await;
    store.fail_part_gets.store(true, Ordering::SeqCst);
    let mut put = open(&server.address, run, 0).await;
    put.send(&steps(run, 0, 1, 0..3, 0), 3).await;
    assert_eq!(put.reply().await.unwrap(), 3);
    put.finish().await;
    assert!(report(root, run, 0).is_none(), "the rebuild failed");
    store.fail_part_gets.store(false, Ordering::SeqCst);
    let tasks: Vec<_> = server.flusher.take().into_iter().collect();
    server.ingest.stop(tasks).await.unwrap();
    let judged_at = report(root, run, 0).expect("the rebuild is tried at the stop");
    assert_eq!((&judged_at["attempt"], &judged_at["last_step"]), (&json!(1), &json!(2)));
    let last = written(root, run);
    assert_eq!(last["summary"], "finished", "{last}");
    assert!(judged(&last), "{}", last["run_state"]);
}

/// A finished run let go of at once (`evict_after` zero), whose report retries wait 1 ms and double.
fn evict_config() -> IngestConfig {
    IngestConfig {
        flush_steps: 1,
        flush_interval: Duration::from_secs(3600),
        replay_backoff: Duration::from_millis(1),
        evict_after: Duration::ZERO,
        ..IngestConfig::default()
    }
}

/// AO-03's eviction: a finished run whose rank waits to write its report again is kept (on
/// `8c77d06` it was let go of with its detectors at the next status pass); once the report is
/// written, it is let go of.
#[tokio::test]
async fn ao03_a_run_whose_report_is_pending_is_not_let_go_of_until_it_is_written() {
    let run = "ap2_evict_pending";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let store = FaultStore::new(root);
    let server = start(store.clone(), evict_config(), false).await;
    store.fail_incident_puts.store(true, Ordering::SeqCst);
    let mut put = open(&server.address, run, 0).await;
    put.send(&steps(run, 0, 1, 0..3, 0), 3).await;
    assert_eq!(put.reply().await.unwrap(), 3);
    put.finish().await;
    // Settled at the first pass, let go of at the next.
    server.ingest.write_watchdog().await;
    server.ingest.write_watchdog().await;
    assert_eq!(written(root, run)["summary"], "finished");
    assert_eq!(server.ingest.runs_held(), 1, "kept while its report is pending");
    store.fail_incident_puts.store(false, Ordering::SeqCst);
    let deadline = tokio::time::Instant::now() + BOUND;
    while report(root, run, 0).is_none() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the report was never written again"
        );
        server.ingest.flush_due().await;
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    server.ingest.write_watchdog().await;
    server.ingest.write_watchdog().await;
    assert!(judged(&written(root, run)), "{}", written(root, run)["run_state"]);
    assert_eq!(server.ingest.runs_held(), 0, "let go of once the report is written");
}

/// What bounds that hold: a store that never heals keeps the run only until the report has failed
/// `REPLAY_BACKOFF_MAX_DOUBLINGS + 1` times in a row (its wait no longer grows); then the run is let
/// go of, its status saying truthfully that the rank is not judged.
#[tokio::test]
async fn ao03_a_store_that_never_heals_keeps_the_run_only_for_a_bounded_number_of_retries() {
    let run = "ap2_evict_bound";
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let store = FaultStore::new(root);
    let server = start(store.clone(), evict_config(), false).await;
    store.fail_incident_puts.store(true, Ordering::SeqCst);
    let mut put = open(&server.address, run, 0).await;
    put.send(&steps(run, 0, 1, 0..3, 0), 3).await;
    assert_eq!(put.reply().await.unwrap(), 3);
    put.finish().await;
    server.ingest.write_watchdog().await;
    server.ingest.write_watchdog().await;
    assert_eq!(server.ingest.runs_held(), 1, "kept while its report is pending");
    // The waits are 1, 2, 4, ... 64 ms: about 130 ms of retries in all.
    let deadline = tokio::time::Instant::now() + BOUND;
    while server.ingest.runs_held() > 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "a run that never heals is kept for ever"
        );
        server.ingest.flush_due().await;
        server.ingest.write_watchdog().await;
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert!(report(root, run, 0).is_none());
    assert!(codes(&written(root, run)).contains(&"ranks_unknown".to_string()));
}
