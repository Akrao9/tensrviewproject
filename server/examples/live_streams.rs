//! Drive a running `tensorquill-server` with many ranks at once, to time it under live streams.
//!
//!     cargo run --release --example live_streams -- <source store> <run_id> http://127.0.0.1:50051 \
//!         --store <the server's store> --repeat 5 --step-ms 10
//!
//! Each rank of `<source store>/runs/<run_id>` becomes one `DoPut` stream, all opened first
//! and started together. `--ranks` above what the source holds simulates more: rank `r` replays
//! the parts of source rank `r` modulo the source's count under its own number, so 1,024 ranks
//! can be sent from a 64-rank store:
//!
//!     cargo run --release --example live_streams -- <source store> <run_id> http://127.0.0.1:50051 \
//!         --ranks 1024 --step-ms 65
//! A rank sends its stored parts in order, one part per message, and
//! like the Python sender waits for the server's acknowledgement before the next. `--repeat`
//! plays the parts that many times with the step numbers carried on, to make a longer run
//! than the source holds. `--step-ms` paces a rank at that many milliseconds per step (0
//! sends as fast as the acknowledgements return). `--first-step`, `--batches` and `--attempt`
//! send a few batches from a given step or as a new attempt, to time what a restart costs.
//!
//! Each batch's metadata names `schema_version` 2 when the source part is in protocol 5's layout,
//! told by its columns (`layers`, `moe_stats`, `probe_maps`, `probe_histograms` or `record_version`,
//! `WAVE_AL_CONTRACT.md` section 3), and 1 otherwise, as the Python sender does; `--metadata-version`
//! names one for every batch instead (1 sends a typed source to a protocol 4 server, as the
//! contract's prototype arm was measured). The parts are sent whole, probe arrays included
//! (`Storage::read_part`).
//!
//! Timed per batch: from sending it to its acknowledgement (the server stores the part first;
//! a rank's first batch is shown apart, since later ones can wait behind the detectors),
//! and, with `--store`, from the acknowledgement until the rank's `incidents.json` covers the
//! batch. Start the server with `--flush-steps` equal to the source's rows per part, so each
//! message is one flush. The client decodes Parquet and encodes Flight on the same machine,
//! so it takes cores from the server it is timing; when paced, it prepares a batch in the
//! second before the batch is due, which is after the server has dealt with the one before.
//!
//! A rank whose stream fails stops there and the others go on: the summary counts the ranks
//! that failed, by the server's message, and the timings are of the batches acknowledged.
//! Unlike the Python sender, a failed rank does not connect again. `--connect-at-once` bounds
//! how many ranks are connecting at a time (a listener's backlog is short; 128 on macOS).

use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{Array, ArrayRef, Int32Array, Int64Array, RecordBatch};
use arrow::compute::concat_batches;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::{FlightClient, FlightData, FlightDescriptor};
use clap::Parser;
use futures::{StreamExt, TryStreamExt};
use tensorquill_server::storage::{Error, Storage, StoredPart};
use tokio::sync::{mpsc, Barrier, Mutex, Semaphore};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Channel;

#[derive(Parser, Clone)]
struct Args {
    /// A store on local disk whose parts are replayed.
    source: String,
    /// The run to replay; it is sent under the same run ID.
    run_id: String,
    /// The server, as http://host:port.
    endpoint: String,
    /// The server's store, to time how long after an acknowledgement incidents.json covers the batch.
    #[arg(long)]
    store: Option<PathBuf>,
    /// Send this many ranks, numbered from 0 (default: the source's ranks under their own
    /// numbers). Above the source's count, rank r replays source rank r modulo that count.
    #[arg(long)]
    ranks: Option<usize>,
    /// How many ranks may be connecting at once.
    #[arg(long, default_value_t = 32)]
    connect_at_once: usize,
    /// With --store, look at a rank's incidents.json this often, in milliseconds.
    #[arg(long, default_value_t = 2)]
    report_poll_ms: u64,
    /// Count the acknowledgements that took longer than this many milliseconds.
    #[arg(long, default_value_t = 250.0)]
    budget_ms: f64,
    /// Play the source's parts this many times, carrying the step numbers on.
    #[arg(long, default_value_t = 1)]
    repeat: usize,
    /// Stop after this many batches per rank.
    #[arg(long)]
    batches: Option<usize>,
    /// The step number of the first row sent.
    #[arg(long, default_value_t = 0)]
    first_step: i64,
    /// Send under this attempt instead of the source's.
    #[arg(long)]
    attempt: Option<i64>,
    /// Milliseconds of training per step; 0 sends as fast as acknowledgements return.
    #[arg(long, default_value_t = 0.0)]
    step_ms: f64,
    /// The batch metadata's `schema_version` for every batch (default: 2 for a part in protocol 5's
    /// layout, told by its columns, and 1 for any other).
    #[arg(long)]
    metadata_version: Option<i64>,
}

/// The columns only protocol 5's layout has (`WAVE_AL_CONTRACT.md` sections 1.2, 2.4 and 3).
const PROTOCOL_5_COLUMNS: [&str; 5] = [
    "layers",
    "moe_stats",
    "probe_maps",
    "probe_histograms",
    "record_version",
];

/// The batch metadata version a sender gives `batch`: 2 for protocol 5's layout, 1 for protocol 4's.
fn metadata_version(batch: &RecordBatch) -> i64 {
    let schema = batch.schema();
    if PROTOCOL_5_COLUMNS
        .iter()
        .any(|name| schema.column_with_name(name).is_some())
    {
        2
    } else {
        1
    }
}

/// One source rank's parts, read once for all the simulated ranks that replay it: they ask
/// for the same part at about the same time, and the last two read are kept.
struct Source {
    storage: Storage,
    parts: Vec<StoredPart>,
    recent: Mutex<VecDeque<(usize, RecordBatch)>>,
}

impl Source {
    async fn open(storage: &Storage, run_id: &str, rank: i32) -> Result<Self, Error> {
        let parts = storage.rank_parts(run_id, rank).await?;
        if parts.is_empty() {
            return Err(format!("rank {rank} has no parts in the source").into());
        }
        Ok(Self {
            storage: storage.clone(),
            parts,
            recent: Mutex::new(VecDeque::new()),
        })
    }

    /// The part at `index`, going round the parts again past the last.
    async fn part(&self, index: usize) -> Result<RecordBatch, Error> {
        let index = index % self.parts.len();
        let mut recent = self.recent.lock().await; // the ranks waiting here get the one read
        if let Some((_, batch)) = recent.iter().find(|(held, _)| *held == index) {
            return Ok(batch.clone());
        }
        let batches = self.storage.read_part(&self.parts[index].location).await?;
        let batch = concat_batches(&batches[0].schema(), &batches)?;
        recent.push_back((index, batch.clone()));
        if recent.len() > 2 {
            recent.pop_front();
        }
        Ok(batch)
    }
}

/// A batch with its `step` column numbered from `first_step`, as `rank`, and, if asked,
/// another attempt.
fn renumber(batch: &RecordBatch, first_step: i64, rank: i32, attempt: Option<i64>) -> Result<RecordBatch, Error> {
    let schema = batch.schema();
    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    let rows = batch.num_rows();
    let mut set = |name: &str, values: ArrayRef| -> Result<(), Error> {
        let index = schema.index_of(name)?;
        if schema.field(index).data_type() != values.data_type() {
            return Err(format!("column `{name}` is not {}", values.data_type()).into());
        }
        columns[index] = values;
        Ok(())
    };
    let steps = Int64Array::from_iter_values(first_step..first_step + rows as i64);
    set("step", Arc::new(steps))?;
    set("rank", Arc::new(Int32Array::from(vec![rank; rows])))?;
    if let Some(attempt) = attempt {
        set("attempt", Arc::new(Int64Array::from(vec![attempt; rows])))?;
    }
    Ok(RecordBatch::try_new(schema, columns)?)
}

/// How long a rank's `incidents.json` takes to reach `last_step`, from now.
async fn until_reported(path: PathBuf, last_step: i64, poll: Duration) -> Option<f64> {
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(120) {
        let covered = std::fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .and_then(|report| report["last_step"].as_i64())
            .is_some_and(|reported| reported >= last_step);
        if covered {
            return Some(started.elapsed().as_secs_f64() * 1e3);
        }
        tokio::time::sleep(poll).await;
    }
    None
}

#[derive(Default)]
struct Timings {
    first_ack: Option<f64>,
    acks: Vec<f64>,
    reports: Vec<f64>,
    unreported: usize,
    steps: usize,
    late: f64,
    seconds: f64,
    /// Why the rank stopped early, in the server's words where it gave any.
    error: Option<String>,
}

/// A rank's open stream: what it sends on, and the server's acknowledgements.
struct Stream {
    messages: mpsc::UnboundedSender<Result<FlightData, FlightError>>,
    acks: futures::stream::BoxStream<'static, Result<arrow_flight::PutResult, FlightError>>,
    first: RecordBatch,
}

/// What stopped a rank, without its number, so that ranks stopped by the same thing count together.
fn reason(error: &FlightError) -> String {
    match error {
        FlightError::Tonic(status) => format!("{:?}: {}", status.code(), status.message()),
        other => other.to_string(),
    }
}

async fn open(args: &Args, rank: i32, source: &Source, connecting: &Semaphore) -> Result<Stream, Error> {
    let first = source.part(0).await?;
    let _connecting = connecting.acquire().await?;
    let channel = Channel::from_shared(args.endpoint.clone())?.connect().await?;
    let mut client = FlightClient::new(channel);
    let (messages, outgoing) = mpsc::unbounded_channel::<Result<FlightData, FlightError>>();
    let descriptor = FlightDescriptor::new_path(vec!["runs".into(), args.run_id.clone(), format!("rank={rank}")]);
    let opening: Vec<FlightData> = FlightDataEncoderBuilder::new()
        .with_flight_descriptor(Some(descriptor))
        .with_schema(first.schema())
        .build(futures::stream::empty())
        .try_collect()
        .await?;
    for message in opening {
        messages.send(Ok(message)).map_err(|_| "the stream closed")?;
    }
    let acks = client
        .do_put(UnboundedReceiverStream::new(outgoing))
        .await
        .map_err(|error| reason(&error))?;
    Ok(Stream { messages, acks, first })
}

async fn send(args: &Args, rank: i32, source: &Source, stream: Stream, timings: &mut Timings) -> Result<(), Error> {
    let Stream {
        messages,
        mut acks,
        first: mut next,
    } = stream;
    let total = args
        .batches
        .unwrap_or(source.parts.len() * args.repeat)
        .min(source.parts.len() * args.repeat);
    let report = args.store.as_ref().map(|store| {
        store
            .join("runs")
            .join(&args.run_id)
            .join(format!("rank={rank}"))
            .join("incidents.json")
    });
    let poll = Duration::from_millis(args.report_poll_ms.max(1));
    let started = Instant::now();
    let mut waiting = Vec::new();
    let mut step = args.first_step;
    for index in 0..total {
        // A training job has the batch ready when its last step is done. Getting it ready
        // is left until a second before then, when the server is done with the batch before.
        let due = started + Duration::from_secs_f64((timings.steps + next.num_rows()) as f64 * args.step_ms / 1e3);
        if let Some(early) = due.checked_sub(Duration::from_secs(1)) {
            tokio::time::sleep_until(early.into()).await;
        }
        let batch = renumber(&next, step, rank, args.attempt)?;
        let version = args.metadata_version.unwrap_or_else(|| metadata_version(&batch));
        let rows = batch.num_rows();
        step += rows as i64;
        let encoded: Vec<FlightData> = FlightDataEncoderBuilder::new()
            .with_max_flight_data_size(usize::MAX)
            .build(futures::stream::iter([Ok(batch)]))
            .try_collect()
            .await?;
        if index + 1 < total {
            next = source.part(index + 1).await?;
        }
        timings.late = timings
            .late
            .max(Instant::now().saturating_duration_since(due).as_secs_f64());
        tokio::time::sleep_until(due.into()).await;
        let metadata = format!(
            "{{\"schema_version\":{version},\"run_id\":\"{}\",\"rank\":{rank},\"min_step\":null,\"max_step\":null,\"end_byte_offset\":{}}}",
            args.run_id,
            index + 1
        );
        let sent = Instant::now();
        // Each encoder starts with a schema message; the stream already has one.
        for mut message in encoded.into_iter().skip(1) {
            message.app_metadata = metadata.clone().into();
            messages.send(Ok(message)).map_err(|_| "the stream closed")?;
        }
        match acks.next().await {
            Some(Ok(_)) if index == 0 => timings.first_ack = Some(sent.elapsed().as_secs_f64() * 1e3),
            Some(Ok(_)) => timings.acks.push(sent.elapsed().as_secs_f64() * 1e3),
            Some(Err(error)) => return Err(reason(&error).into()),
            None => return Err("the server ended the stream".into()),
        }
        timings.steps += rows;
        timings.seconds = started.elapsed().as_secs_f64();
        if let Some(report) = &report {
            waiting.push(tokio::spawn(until_reported(report.clone(), step - 1, poll)));
        }
    }
    for wait in waiting {
        match wait.await? {
            Some(lag) => timings.reports.push(lag),
            None => timings.unreported += 1,
        }
    }
    drop(messages); // ends the stream cleanly
    while acks.next().await.is_some() {}
    Ok(())
}

/// One rank from connecting to its last acknowledgement. A rank that cannot connect still
/// waits at `start`, so the others are not left waiting for it.
async fn stream_rank(
    args: Args,
    rank: i32,
    source: Arc<Source>,
    connecting: Arc<Semaphore>,
    start: Arc<Barrier>,
) -> Timings {
    let mut timings = Timings::default();
    let stream = open(&args, rank, &source, &connecting).await;
    start.wait().await;
    let sent = match stream {
        Ok(stream) => send(&args, rank, &source, stream, &mut timings).await,
        Err(error) => Err(format!("not connected: {error}").into()),
    };
    timings.error = sent.err().map(|error| error.to_string());
    timings
}

fn spread(name: &str, values: &mut [f64], budget_ms: f64) -> serde_json::Value {
    if values.is_empty() {
        return serde_json::Value::Null;
    }
    values.sort_by(f64::total_cmp);
    let at = |quantile: f64| values[((values.len() - 1) as f64 * quantile).round() as usize];
    let over = values.iter().filter(|value| **value > budget_ms).count();
    println!(
        "  {name}: median {:.1} ms, p95 {:.1} ms, p99 {:.1} ms, worst {:.1} ms ({} batches, {over} over {budget_ms:.0} ms)",
        at(0.5),
        at(0.95),
        at(0.99),
        at(1.0),
        values.len()
    );
    serde_json::json!({"median_ms": at(0.5), "p95_ms": at(0.95), "p99_ms": at(0.99), "worst_ms": at(1.0),
        "count": values.len(), "over_budget": over})
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args = Args::parse();
    let mut stored: Vec<i32> = Vec::new();
    for entry in std::fs::read_dir(std::path::Path::new(&args.source).join("runs").join(&args.run_id))? {
        let name = entry?.file_name();
        if let Some(rank) = name
            .to_str()
            .and_then(|name| name.strip_prefix("rank="))
            .and_then(|rank| rank.parse().ok())
        {
            stored.push(rank);
        }
    }
    if stored.is_empty() {
        return Err("the source holds no rank of this run".into());
    }
    stored.sort_unstable();
    let storage = Storage::from_url(&args.source)?;
    // Without --ranks each stored rank is sent under its own number; with it, ranks 0..n.
    let ranks: Vec<(i32, i32)> = match args.ranks {
        None => stored.iter().map(|rank| (*rank, *rank)).collect(),
        Some(count) => (0..count)
            .map(|rank| (rank as i32, stored[rank % stored.len()]))
            .collect(),
    };
    let mut sources: BTreeMap<i32, Arc<Source>> = BTreeMap::new();
    for (_, from) in &ranks {
        if !sources.contains_key(from) {
            sources.insert(*from, Arc::new(Source::open(&storage, &args.run_id, *from).await?));
        }
    }
    let simulated = ranks.iter().filter(|(rank, from)| rank != from).count();
    let connecting = Arc::new(Semaphore::new(args.connect_at_once.max(1)));
    let start = Arc::new(Barrier::new(ranks.len() + 1));
    let connect_started = Instant::now();
    let tasks: Vec<_> = ranks
        .iter()
        .map(|(rank, from)| {
            tokio::spawn(stream_rank(
                args.clone(),
                *rank,
                sources[from].clone(),
                connecting.clone(),
                start.clone(),
            ))
        })
        .collect();
    start.wait().await; // every rank is connected, or has failed to, and holds its first batch
    let connect_seconds = connect_started.elapsed().as_secs_f64();
    let (mut acks, mut reports, mut steps, mut unreported, mut late, mut seconds) =
        (Vec::new(), Vec::new(), 0, 0, 0.0_f64, 0.0_f64);
    let mut first_acks = Vec::new();
    let mut failures: BTreeMap<String, (usize, usize)> = BTreeMap::new(); // reason: (ranks, batches they had acknowledged)
    for task in tasks {
        let timings = task.await?;
        if let Some(error) = timings.error {
            let failed = failures.entry(error).or_default();
            failed.0 += 1;
            failed.1 += timings.first_ack.iter().len() + timings.acks.len();
        }
        first_acks.extend(timings.first_ack);
        acks.extend(timings.acks);
        reports.extend(timings.reports);
        steps += timings.steps;
        unreported += timings.unreported;
        late = late.max(timings.late);
        seconds = seconds.max(timings.seconds); // until the last rank's last acknowledgement
    }
    let failed: usize = failures.values().map(|(ranks, _)| ranks).sum();
    println!(
        "{} ranks ({simulated} simulated from {} stored, connected in {connect_seconds:.2} s), {steps} steps in {seconds:.2} s: {:.0} steps/s ({} per rank)",
        ranks.len(),
        sources.len(),
        steps as f64 / seconds,
        steps / ranks.len().max(1)
    );
    // A rank's first batch finds no part of its own being judged; a later one waits for
    // the detectors to finish with the part before it, which after a restart is a rebuild.
    let budget = args.budget_ms;
    let first_ack = spread(
        "a rank's first batch, sent to acknowledged (stored)",
        &mut first_acks,
        budget,
    );
    let ack = spread("later batches, sent to acknowledged", &mut acks, budget);
    let report = spread("acknowledged to incidents.json covering it", &mut reports, budget);
    if unreported > 0 {
        println!("  {unreported} batches were not reported on within 120 s");
    }
    if args.step_ms > 0.0 {
        println!(
            "  the client was ready at most {:.0} ms after a batch was due",
            late * 1e3
        );
    }
    println!("  {failed} of {} ranks stopped early", ranks.len());
    for (error, (ranks, batches)) in &failures {
        println!("    {ranks} ranks, after {batches} acknowledged batches in all: {error}");
    }
    let failures: Vec<_> = failures
        .iter()
        .map(|(error, (ranks, batches))| serde_json::json!({"error": error, "ranks": ranks, "acknowledged_batches": batches}))
        .collect();
    println!(
        "{}",
        serde_json::json!({"ranks": ranks.len(), "simulated_ranks": simulated, "source_ranks": sources.len(),
            "connect_seconds": connect_seconds, "steps": steps, "seconds": seconds, "steps_per_second": steps as f64 / seconds,
            "step_ms": args.step_ms, "budget_ms": budget, "first_ack": first_ack, "ack": ack, "report": report,
            "unreported": unreported, "client_late_ms": late * 1e3, "failed_ranks": failed, "failures": failures})
    );
    Ok(())
}
