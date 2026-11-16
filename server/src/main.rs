use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use tensorquill_server::detect::Thresholds;
use tensorquill_server::ingest::{Ingest, IngestConfig, DEFAULT_FLUSH_STEPS, DEFAULT_QUOTA_BYTES};
use tensorquill_server::server::{check_bind, serve, Tls};
use tensorquill_server::storage::{Error, Storage, StoreOptions};
use tokio::net::TcpListener;

/// Self-hosted TensorQuill ingest: Arrow Flight DoPut in, partitioned Parquet out.
#[derive(Parser)]
#[command(name = "tensorquill-server", version, subcommand_negates_reqs = true)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,
    /// Address to listen on. Anything but loopback needs TENSORQUILL_TOKEN or --allow-unauthenticated.
    #[arg(long, default_value = "127.0.0.1:50051")]
    listen: SocketAddr,
    /// Where Parquet parts go: file:///var/lib/tensorquill, s3://bucket/prefix, gs://bucket/prefix, or a directory.
    #[arg(long, required = true)]
    store: Option<String>,
    /// PEM certificate chain; with --tls-key, serve TLS (senders then use grpcs://).
    #[arg(long, requires = "tls_key")]
    tls_cert: Option<PathBuf>,
    /// PEM private key for --tls-cert.
    #[arg(long, requires = "tls_cert")]
    tls_key: Option<PathBuf>,
    /// Listen beyond loopback with no token check.
    #[arg(long)]
    allow_unauthenticated: bool,
    /// Senders must present this as `Authorization: Bearer <token>`.
    #[arg(long, env = "TENSORQUILL_TOKEN", hide_env_values = true)]
    token: Option<String>,
    /// Cap on buffered Arrow bytes across all streams, in MiB.
    #[arg(long, default_value_t = DEFAULT_QUOTA_BYTES / (1024 * 1024))]
    quota_mib: usize,
    /// Write a rank's Parquet part once it holds this many steps.
    #[arg(long, default_value_t = DEFAULT_FLUSH_STEPS)]
    flush_steps: usize,
    /// Write a rank's Parquet part once its oldest step has waited this many seconds.
    #[arg(long, default_value_t = 10.0)]
    flush_seconds: f64,
    /// A rank with no new step for this long (or this many times its own usual time between
    /// batches, if that is longer) is reported silent in runs/<run_id>/watchdog.json. Set it
    /// above your longest evaluation or checkpoint pause; the default is not calibrated.
    #[arg(long, default_value_t = 300.0)]
    silent_seconds: f64,
    /// A rank quiet for longer than this is reported `critical`; between --silent-seconds and this it is a `warning`.
    #[arg(long, default_value_t = 900.0)]
    critical_seconds: f64,
    /// A phase other than training (evaluation, a checkpoint) that shows no sign of work is called a suspected hang after this long.
    #[arg(long, default_value_t = 1800.0)]
    pause_seconds: f64,
    /// How many times a rank's usual time between batches it may stay quiet before it is silent.
    #[arg(long, default_value_t = 5.0)]
    silent_factor: f64,
    /// Look at each run's progress this often and rewrite runs/<run_id>/watchdog.json when its verdict
    /// changed, and every third look while the run is unfinished; 0 turns the file off.
    #[arg(long, default_value_t = 10.0)]
    watchdog_seconds: f64,
    /// Do not run the detectors over stored parts or write rank=<rank>/incidents.json.
    #[arg(long)]
    no_detect: bool,
    /// Detector thresholds as one JSON object, for the server's detectors and for `detect`. Keys:
    /// overload_threshold (0.35), drift_tolerance (1e-6), starve_window (50), starve_ratio (0.1) and
    /// top_k (null: what the run recorded; a number for every router, or an object of router name
    /// to number). A key left out keeps the default shown. Refused: an unknown key, a negative or
    /// non-finite number, a window or top_k below 1. Every report lists the values applied.
    #[arg(long, global = true, value_name = "JSON", value_parser = Thresholds::parse)]
    thresholds: Option<Thresholds>,
    /// How long a failing request to S3 or GCS is retried before the stream is told the store is unavailable
    /// (the sender keeps its rows and tries again). Also bounds how long SIGTERM waits for a failing store.
    #[arg(long, default_value_t = 30.0)]
    store_retry_seconds: f64,
    /// A run with no stream open, whose ranks have all been disconnected or finished for this many hours,
    /// is given up: its watchdog.json is no longer refreshed and a restarted server does not take it up
    /// again. 0 never gives a run up.
    #[arg(long, default_value_t = 72.0)]
    abandon_hours: f64,
}

#[derive(Subcommand)]
enum Command {
    /// Run the detectors over one rank's rows and print the report the server would write as
    /// that rank's incidents.json. No server is started and no store is read.
    Detect {
        /// The rows as an Arrow IPC stream in the schema the SDK sends, grouped by attempt
        /// (oldest first) and in recorded order; `-` or nothing reads standard input.
        input: Option<PathBuf>,
    },
}

/// `tensorquill-server detect`: the report goes to standard output as one line of JSON.
fn detect(input: Option<PathBuf>, thresholds: &Arc<Thresholds>) -> Result<(), Error> {
    use tensorquill_server::detect::detect_stream;
    let report = match input.filter(|path| path.as_os_str() != "-") {
        Some(path) => detect_stream(std::io::BufReader::new(std::fs::File::open(path)?), thresholds),
        None => detect_stream(std::io::stdin().lock(), thresholds),
    }?;
    println!("{report}");
    Ok(())
}

async fn terminated() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = signal(SignalKind::terminate()).expect("cannot listen for SIGTERM");
        tokio::select! {
            _ = term.recv() => {}
            _ = tokio::signal::ctrl_c() => {}
        }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
}

/// The store location without anything a URL can carry besides the place (a password, a query).
fn store_name(store: &str) -> String {
    match url::Url::parse(store) {
        Ok(url) if url.scheme().len() > 1 => {
            format!("{}://{}{}", url.scheme(), url.host_str().unwrap_or(""), url.path())
        }
        _ => store.to_string(),
    }
}

async fn run(args: Args) -> Result<(), Error> {
    let store = args.store.ok_or("--store is required")?;
    let token = args.token.filter(|token| !token.is_empty());
    check_bind(&args.listen, token.is_some(), args.allow_unauthenticated)?;
    let tls = match (&args.tls_cert, &args.tls_key) {
        (Some(certificate), Some(key)) => Some(Tls {
            certificate_pem: std::fs::read(certificate)?,
            key_pem: std::fs::read(key)?,
        }),
        _ => None,
    };
    let config = IngestConfig {
        quota_bytes: args.quota_mib * 1024 * 1024,
        flush_steps: args.flush_steps,
        flush_interval: Duration::from_secs_f64(args.flush_seconds),
        silent_floor: Duration::from_secs_f64(args.silent_seconds),
        silent_factor: args.silent_factor,
        critical: Duration::from_secs_f64(args.critical_seconds),
        pause_limit: Duration::from_secs_f64(args.pause_seconds),
        watchdog_interval: Duration::from_secs_f64(args.watchdog_seconds),
        status_refresh: Duration::from_secs_f64(args.watchdog_seconds * 3.0),
        detect: !args.no_detect,
        thresholds: Arc::new(args.thresholds.unwrap_or_default()),
        abandon_after: Duration::from_secs_f64(args.abandon_hours * 3600.0),
    };
    let options = StoreOptions {
        retry_timeout: Duration::from_secs_f64(args.store_retry_seconds),
    };
    let ingest = Ingest::new(Storage::open(&store, std::env::vars(), &options)?, config);
    let listener = TcpListener::bind(args.listen).await?;
    // Take up the runs the last server left unfinished before any stream is served: a rank
    // that connects first would otherwise be watched from nothing. This is also the first
    // read of the store, so a wrong bucket or bad credentials stop the server here, with the
    // store's own message (which does not hold the credentials), not at the first batch.
    let resumed = ingest
        .restore_watchdog()
        .await
        .map_err(|error| format!("cannot read the store {}: {error}", store_name(&store)))?;
    if resumed > 0 {
        eprintln!("tensorquill-server: watching {resumed} unfinished runs from the last server's status");
    }
    // Scripts and tests that ask for port 0 read the real address from this line.
    println!("tensorquill-server listening on {}", listener.local_addr()?);
    let flusher = ingest.spawn_flusher();
    let watchdog = ingest.spawn_watchdog();
    // Open streams never end on their own, so do not wait for them: on a signal, stop
    // taking batches, write what is buffered, and exit.
    tokio::select! {
        result = serve(listener, ingest.clone(), token, tls, std::future::pending()) => result?,
        _ = terminated() => {}
    }
    flusher.abort();
    watchdog.abort();
    ingest.write_watchdog_stopped().await; // each status says the server stopped, not a stale verdict
    ingest.shutdown().await?;
    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    let mut args = Args::parse();
    let outcome = match args.command.take() {
        Some(Command::Detect { input }) => detect(input, &Arc::new(args.thresholds.take().unwrap_or_default())),
        None => run(args).await,
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("tensorquill-server: {error}");
            ExitCode::FAILURE
        }
    }
}
