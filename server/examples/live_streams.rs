//! Drive a running `tensorquill-server` with many ranks at once, to time it under live streams.
//!
//!     cargo run --release --example live_streams -- <source store> <run_id> http://127.0.0.1:50051 \
//!         --store <the server's store> --repeat 5 --step-ms 10
//!
//! Each rank of `<source store>/runs/<run_id>` becomes one `DoPut` stream, all opened first
//! and started together. A rank sends its stored parts in order, one part per message, and
//! like the Python sender waits for the server's acknowledgement before the next. `--repeat`
//! plays the parts that many times with the step numbers carried on, to make a longer run
//! than the source holds. `--step-ms` paces a rank at that many milliseconds per step (0
//! sends as fast as the acknowledgements return). `--first-step`, `--batches` and `--attempt`
//! send a few batches from a given step or as a new attempt, to time what a restart costs.
//!
//! Timed per batch: from sending it to its acknowledgement (the server stores the part first;
//! a rank's first batch is shown apart, since later ones can wait behind the detectors),
//! and, with `--store`, from the acknowledgement until the rank's `incidents.json` covers the
//! batch. Start the server with `--flush-steps` equal to the source's rows per part, so each
//! message is one flush. The client decodes Parquet and encodes Flight on the same machine,
//! so it takes cores from the server it is timing.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{ArrayRef, Int64Array, RecordBatch};
use arrow::compute::concat_batches;
use arrow::datatypes::DataType;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::{FlightClient, FlightData, FlightDescriptor};
use clap::Parser;
use futures::{StreamExt, TryStreamExt};
use tensorquill_server::storage::{Error, Storage};
use tokio::sync::{mpsc, Barrier};
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
    /// Send only the first this many ranks.
    #[arg(long)]
    ranks: Option<usize>,
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
}

/// A batch with its `step` column numbered from `first_step` and, if asked, another attempt.
fn renumber(batch: &RecordBatch, first_step: i64, attempt: Option<i64>) -> Result<RecordBatch, Error> {
    let schema = batch.schema();
    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    let rows = batch.num_rows() as i64;
    let mut set = |name: &str, values: Int64Array| -> Result<(), Error> {
        let index = schema.index_of(name)?;
        if schema.field(index).data_type() != &DataType::Int64 {
            return Err(format!("column `{name}` is not int64").into());
        }
        columns[index] = Arc::new(values);
        Ok(())
    };
    set("step", Int64Array::from_iter_values(first_step..first_step + rows))?;
    if let Some(attempt) = attempt {
        set("attempt", Int64Array::from(vec![attempt; rows as usize]))?;
    }
    Ok(RecordBatch::try_new(schema, columns)?)
}

/// How long a rank's `incidents.json` takes to reach `last_step`, from now.
async fn until_reported(path: PathBuf, last_step: i64) -> Option<f64> {
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
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    None
}

struct Timings {
    first_ack: f64,
    acks: Vec<f64>,
    reports: Vec<f64>,
    unreported: usize,
    steps: usize,
    late: f64,
    seconds: f64,
}

async fn stream_rank(args: Args, rank: i32, start: Arc<Barrier>) -> Result<Timings, Error> {
    let source = Storage::from_url(&args.source)?;
    let parts = source.rank_parts(&args.run_id, rank).await?;
    if parts.is_empty() {
        return Err(format!("rank {rank} has no parts in the source").into());
    }
    let total = args
        .batches
        .unwrap_or(parts.len() * args.repeat)
        .min(parts.len() * args.repeat);
    let read = |index: usize| {
        let (source, location) = (source.clone(), parts[index % parts.len()].location.clone());
        async move {
            let batches = source.read_part(&location).await?;
            Ok::<_, Error>(concat_batches(&batches[0].schema(), &batches)?)
        }
    };
    let mut next = read(0).await?;

    let channel = Channel::from_shared(args.endpoint.clone())?.connect().await?;
    let mut client = FlightClient::new(channel);
    let (messages, outgoing) = mpsc::unbounded_channel::<Result<FlightData, FlightError>>();
    let descriptor = FlightDescriptor::new_path(vec!["runs".into(), args.run_id.clone(), format!("rank={rank}")]);
    let opening: Vec<FlightData> = FlightDataEncoderBuilder::new()
        .with_flight_descriptor(Some(descriptor))
        .with_schema(next.schema())
        .build(futures::stream::empty())
        .try_collect()
        .await?;
    for message in opening {
        messages.send(Ok(message)).map_err(|_| "the stream closed")?;
    }
    let mut acks = client.do_put(UnboundedReceiverStream::new(outgoing)).await?;
    let report = args.store.as_ref().map(|store| {
        store
            .join("runs")
            .join(&args.run_id)
            .join(format!("rank={rank}"))
            .join("incidents.json")
    });

    start.wait().await;
    let started = Instant::now();
    let mut timings = Timings {
        first_ack: 0.0,
        acks: Vec::new(),
        reports: Vec::new(),
        unreported: 0,
        steps: 0,
        late: 0.0,
        seconds: 0.0,
    };
    let mut waiting = Vec::new();
    let mut step = args.first_step;
    for index in 0..total {
        let batch = renumber(&next, step, args.attempt)?;
        let rows = batch.num_rows();
        step += rows as i64;
        let encoded: Vec<FlightData> = FlightDataEncoderBuilder::new()
            .with_max_flight_data_size(usize::MAX)
            .build(futures::stream::iter([Ok(batch)]))
            .try_collect()
            .await?;
        if index + 1 < total {
            next = read(index + 1).await?;
        }
        // A training job has the batch ready when its last step is done.
        let due = started + Duration::from_secs_f64((timings.steps + rows) as f64 * args.step_ms / 1e3);
        timings.late = timings
            .late
            .max(Instant::now().saturating_duration_since(due).as_secs_f64());
        tokio::time::sleep_until(due.into()).await;
        let metadata = format!(
            "{{\"schema_version\":1,\"run_id\":\"{}\",\"rank\":{rank},\"min_step\":null,\"max_step\":null,\"end_byte_offset\":{}}}",
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
            Some(Ok(_)) if index == 0 => timings.first_ack = sent.elapsed().as_secs_f64() * 1e3,
            Some(Ok(_)) => timings.acks.push(sent.elapsed().as_secs_f64() * 1e3),
            Some(Err(error)) => return Err(format!("rank {rank}: {error}").into()),
            None => return Err(format!("rank {rank}: the server ended the stream").into()),
        }
        timings.steps += rows;
        if let Some(report) = &report {
            waiting.push(tokio::spawn(until_reported(report.clone(), step - 1)));
        }
    }
    timings.seconds = started.elapsed().as_secs_f64();
    for wait in waiting {
        match wait.await? {
            Some(lag) => timings.reports.push(lag),
            None => timings.unreported += 1,
        }
    }
    drop(messages); // ends the stream cleanly
    while acks.next().await.is_some() {}
    Ok(timings)
}

fn spread(name: &str, values: &mut [f64]) -> serde_json::Value {
    if values.is_empty() {
        return serde_json::Value::Null;
    }
    values.sort_by(f64::total_cmp);
    let at = |quantile: f64| values[((values.len() - 1) as f64 * quantile).round() as usize];
    println!(
        "  {name}: median {:.1} ms, p99 {:.1} ms, worst {:.1} ms ({} batches)",
        at(0.5),
        at(0.99),
        at(1.0),
        values.len()
    );
    serde_json::json!({"median_ms": at(0.5), "p99_ms": at(0.99), "worst_ms": at(1.0), "count": values.len()})
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args = Args::parse();
    let mut ranks: Vec<i32> = Vec::new();
    for entry in std::fs::read_dir(std::path::Path::new(&args.source).join("runs").join(&args.run_id))? {
        let name = entry?.file_name();
        if let Some(rank) = name
            .to_str()
            .and_then(|name| name.strip_prefix("rank="))
            .and_then(|rank| rank.parse().ok())
        {
            ranks.push(rank);
        }
    }
    ranks.sort_unstable();
    ranks.truncate(args.ranks.unwrap_or(ranks.len()));
    let start = Arc::new(Barrier::new(ranks.len() + 1));
    let tasks: Vec<_> = ranks
        .iter()
        .map(|rank| tokio::spawn(stream_rank(args.clone(), *rank, start.clone())))
        .collect();
    start.wait().await; // every rank is connected and holds its first batch
    let (mut acks, mut reports, mut steps, mut unreported, mut late, mut seconds) =
        (Vec::new(), Vec::new(), 0, 0, 0.0_f64, 0.0_f64);
    let mut first_acks = Vec::new();
    for task in tasks {
        let timings = task.await??;
        first_acks.push(timings.first_ack);
        acks.extend(timings.acks);
        reports.extend(timings.reports);
        steps += timings.steps;
        unreported += timings.unreported;
        late = late.max(timings.late);
        seconds = seconds.max(timings.seconds); // until the last rank's last acknowledgement
    }
    println!(
        "{} ranks, {steps} steps in {seconds:.2} s: {:.0} steps/s ({} per rank)",
        ranks.len(),
        steps as f64 / seconds,
        steps / ranks.len().max(1)
    );
    // A rank's first batch finds no part of its own being judged; a later one waits for
    // the detectors to finish with the part before it, which after a restart is a rebuild.
    let first_ack = spread("a rank's first batch, sent to acknowledged (stored)", &mut first_acks);
    let ack = spread("later batches, sent to acknowledged", &mut acks);
    let report = spread("acknowledged to incidents.json covering it", &mut reports);
    if unreported > 0 {
        println!("  {unreported} batches were not reported on within 120 s");
    }
    if args.step_ms > 0.0 {
        println!(
            "  the client was ready at most {:.0} ms after a batch was due",
            late * 1e3
        );
    }
    println!(
        "{}",
        serde_json::json!({"ranks": ranks.len(), "steps": steps, "seconds": seconds, "steps_per_second": steps as f64 / seconds,
            "step_ms": args.step_ms, "first_ack": first_ack, "ack": ack, "report": report, "unreported": unreported, "client_late_ms": late * 1e3})
    );
    Ok(())
}
