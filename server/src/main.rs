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
use tensorquill_server::watchdog::check_silent_factor;
use tokio::net::TcpListener;

/// The one number the Python SDK checks before it runs this binary (`tensorquill-server protocol`).
/// It goes up by one whenever what `detect` reads, takes or prints changes in a way an SDK built
/// for the old number would get wrong: a kind added or removed, a threshold or option added, a
/// report key renamed or dropped. `PROTOCOL` in `sdk/src/tensorquill/server_detect.py` is the
/// same number, and a test in `sdk/tests/test_auto_detectors.py` holds the two together.
///
/// 3 (Nov 19, 2026, lanes W2 and W3, once for both): attempts are ordered by when each was first
/// seen, not by number (contract 1: `attempts.json`, the report's `attempt_order`, rows grouped in
/// order of first appearance), and a capture that stopped has a watchdog state of its own
/// (contract 3: `capture_stopped`).
const PROTOCOL: u32 = 3;

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
    #[arg(long, default_value_t = 10.0, value_parser = seconds)]
    flush_seconds: f64,
    /// A rank with no new step for this long (or this many times its own usual time between
    /// batches, if that is longer) is reported silent in runs/<run_id>/watchdog.json. Set it
    /// above your longest evaluation or checkpoint pause; the default is not calibrated.
    #[arg(long, default_value_t = 300.0, value_parser = seconds)]
    silent_seconds: f64,
    /// A rank quiet for longer than this is reported `critical`; between --silent-seconds and this it is a `warning`.
    #[arg(long, default_value_t = 900.0, value_parser = seconds)]
    critical_seconds: f64,
    /// A phase other than training (evaluation, a checkpoint) that shows no sign of work is called a suspected hang after this long.
    #[arg(long, default_value_t = 1800.0, value_parser = seconds)]
    pause_seconds: f64,
    /// A rank that has stored no batch and shown no training step (it is compiling, loading data, in its
    /// first collective, or sends no heartbeat) is `starting` for this long since its last sign of work,
    /// then reported silent, and `critical` --critical-seconds later. Set it above your longest startup;
    /// the default is not calibrated. A rank whose heartbeats show it stepping is not limited by this.
    #[arg(long, default_value_t = 1800.0, value_parser = seconds)]
    start_seconds: f64,
    /// How many times a rank's usual time between batches it may stay quiet before it is silent:
    /// 0 for --silent-seconds alone, or a finite number above 1.
    #[arg(long, default_value_t = 5.0, value_parser = factor)]
    silent_factor: f64,
    /// Look at each run's progress this often and rewrite runs/<run_id>/watchdog.json when its verdict
    /// changed, and every third look while the run is unfinished; 0 turns the file off.
    #[arg(long, default_value_t = 10.0, value_parser = seconds)]
    watchdog_seconds: f64,
    /// Do not run the detectors over stored parts or write rank=<rank>/incidents.json.
    #[arg(long)]
    no_detect: bool,
    /// Detector thresholds as one JSON object, for the server's detectors and for `detect`. Keys:
    /// overload_threshold (0.35), drift_tolerance (1e-6), starve_window (50), starve_ratio (0.1),
    /// update_window (100) and top_k (null: what the run recorded; a number for every router, or an
    /// object of router name to number). A key left out keeps the default shown. Refused: an unknown
    /// key, a negative or non-finite number, a window or top_k below 1. Every report lists
    /// the values applied.
    #[arg(long, global = true, value_name = "JSON", value_parser = Thresholds::parse)]
    thresholds: Option<Thresholds>,
    /// How long a failing request to S3 or GCS is retried before the stream is told the store is unavailable
    /// (the sender keeps its rows and tries again). Also bounds each wait for a failing store at SIGTERM:
    /// a write in progress, then the last parts, then the stopped status.
    #[arg(long, default_value_t = 30.0, value_parser = seconds)]
    store_retry_seconds: f64,
    /// A run with no stream open, whose ranks have all been disconnected or finished for this many hours,
    /// is given up: its watchdog.json is no longer refreshed and a restarted server does not take it up
    /// again. 0 never gives a run up.
    #[arg(long, default_value_t = 72.0, value_parser = hours)]
    abandon_hours: f64,
}

#[derive(Subcommand)]
enum Command {
    /// Run the detectors over one rank's rows and print the report the server would write as
    /// that rank's incidents.json; with --run, over every rank of one run. No server is started
    /// and no store is read.
    Detect {
        /// The rows as an Arrow IPC stream in the schema the SDK sends, grouped by attempt in
        /// the order first seen (each attempt is read in step order); `-` or nothing reads
        /// standard input. With
        /// --run, any number of streams.
        input: Vec<PathBuf>,
        /// Add `samples` to the report: one entry for every step and MoE router, with its
        /// counts and the experts the rules found overloaded, dead or starved there. The
        /// server's own incidents.json never has them; for a long MoE run they are tens of MB.
        /// One rank only: not with --run.
        #[arg(long, conflicts_with = "run")]
        samples: bool,
        /// Read every rank of one run, not one rank: the rows are grouped by their rank
        /// column, however the ranks are spread over the inputs, and each rank is judged as
        /// `detect` judges one. Prints the run's document: each rank's report (`reports`), and
        /// the comparison of the ranks' batch hashes that the server makes for a run's status
        /// (`duplicate_batches`, and `ranks_compared` for the ranks that had hashes).
        #[arg(long)]
        run: bool,
    },
    /// Print the protocol number and exit: one whole number on a line of its own. The Python SDK
    /// runs a binary only when this is the number it was built for.
    Protocol,
}

/// A number of seconds the server can hold as a duration: finite, zero or more, and not more
/// than `Duration` holds. A flag that is not one is refused with its name, where it panicked
/// with exit code 101 (RS15).
fn seconds(text: &str) -> Result<f64, String> {
    let value: f64 = text
        .trim()
        .parse()
        .map_err(|_| format!("`{text}` is not a number of seconds"))?;
    duration(value).map(|_| value)
}

/// A silence factor the watchdog can use (`watchdog::check_silent_factor`), refused at startup with
/// the flag's name otherwise.
fn factor(text: &str) -> Result<f64, String> {
    let value: f64 = text.trim().parse().map_err(|_| format!("`{text}` is not a number"))?;
    check_silent_factor(value)
}

/// A number of hours that is a duration in seconds, as `seconds`.
fn hours(text: &str) -> Result<f64, String> {
    let value: f64 = text
        .trim()
        .parse()
        .map_err(|_| format!("`{text}` is not a number of hours"))?;
    duration(value * 3600.0).map(|_| value)
}

fn duration(seconds: f64) -> Result<Duration, String> {
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(format!(
            "{seconds} is not a duration: give a finite number, zero or more"
        ));
    }
    Duration::try_from_secs_f64(seconds).map_err(|_| format!("{seconds} seconds is longer than the server can hold"))
}

/// A flag's value as a duration, named in the error. The parsers above have checked each flag;
/// this also covers what is derived from one (the refresh time, three intervals).
fn flag(name: &str, seconds: f64) -> Result<Duration, String> {
    duration(seconds).map_err(|error| format!("{name}: {error}"))
}

/// `tensorquill-server detect`: the report goes to standard output as one line of JSON.
fn detect(inputs: Vec<PathBuf>, samples: bool, run: bool, thresholds: &Arc<Thresholds>) -> Result<(), Error> {
    use std::io::Write;
    use tensorquill_server::detect::{detect_run, detect_stream, detect_stream_samples, read_batches};
    let named: Vec<&PathBuf> = inputs.iter().filter(|path| path.as_os_str() != "-").collect();
    if named.len() != inputs.len() && inputs.len() > 1 {
        return Err("standard input (-) cannot be read beside other inputs".into());
    }
    if run {
        let mut batches = Vec::new();
        if named.is_empty() {
            batches = read_batches(std::io::stdin().lock())?;
        }
        for path in named {
            let file = std::fs::File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
            batches.extend(
                read_batches(std::io::BufReader::new(file)).map_err(|error| format!("{}: {error}", path.display()))?,
            );
        }
        println!("{}", detect_run(&batches, thresholds)?);
        return Ok(());
    }
    let input: Box<dyn std::io::Read> = match named[..] {
        [] => Box::new(std::io::stdin().lock()),
        [path] => Box::new(std::io::BufReader::new(std::fs::File::open(path)?)),
        _ => {
            return Err(
                "several inputs are the ranks of a run: give --run (without it, detect reads one rank's rows)".into(),
            )
        }
    };
    if samples {
        let mut out = std::io::BufWriter::new(std::io::stdout().lock());
        detect_stream_samples(input, thresholds, &mut out)?;
        writeln!(out)?;
        out.flush()?;
    } else {
        println!("{}", detect_stream(input, thresholds)?);
    }
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
        quota_bytes: args.quota_mib.checked_mul(1024 * 1024).ok_or_else(|| {
            format!(
                "--quota-mib: {} MiB is more bytes than the server can count",
                args.quota_mib
            )
        })?,
        flush_steps: args.flush_steps,
        flush_interval: flag("--flush-seconds", args.flush_seconds)?,
        silent_floor: flag("--silent-seconds", args.silent_seconds)?,
        silent_factor: args.silent_factor,
        critical: flag("--critical-seconds", args.critical_seconds)?,
        pause_limit: flag("--pause-seconds", args.pause_seconds)?,
        start_limit: flag("--start-seconds", args.start_seconds)?,
        watchdog_interval: flag("--watchdog-seconds", args.watchdog_seconds)?,
        status_refresh: flag("--watchdog-seconds (three times it)", args.watchdog_seconds * 3.0)?,
        detect: !args.no_detect,
        thresholds: Arc::new(args.thresholds.unwrap_or_default()),
        abandon_after: flag("--abandon-hours", args.abandon_hours * 3600.0)?,
        ..IngestConfig::default()
    };
    let options = StoreOptions {
        retry_timeout: flag("--store-retry-seconds", args.store_retry_seconds)?,
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
    // The store's ID (L3), read from store.json at the store root or written there now, before any
    // batch: every acknowledgement carries it.
    let store_id = ingest
        .store_id()
        .await
        .map_err(|error| format!("cannot read or write the store ID of {}: {error}", store_name(&store)))?;
    eprintln!("tensorquill-server: store ID {store_id}");
    // Scripts and tests that ask for port 0 read the real address from this line.
    println!("tensorquill-server listening on {}", listener.local_addr()?);
    let flusher = ingest.spawn_flusher();
    let watchdog = ingest.spawn_watchdog();
    // Open streams never end on their own, so do not wait for them: on a signal, stop
    // taking batches, let a flush in progress finish, write and judge what is buffered, then
    // write each status to say the server stopped (not a stale verdict), and exit.
    tokio::select! {
        result = serve(listener, ingest.clone(), token, tls, std::future::pending()) => result?,
        _ = terminated() => {}
    }
    ingest.stop([flusher, watchdog]).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    let mut args = Args::parse();
    let outcome = match args.command.take() {
        Some(Command::Detect { input, samples, run }) => detect(
            input,
            samples,
            run,
            &Arc::new(args.thresholds.take().unwrap_or_default()),
        ),
        Some(Command::Protocol) => {
            println!("{PROTOCOL}");
            Ok(())
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_duration_flag_takes_zero_and_refuses_what_is_not_a_duration() {
        assert_eq!(seconds("0"), Ok(0.0));
        assert_eq!(seconds("2.5"), Ok(2.5));
        assert_eq!(hours("0"), Ok(0.0));
        assert_eq!(hours("72"), Ok(72.0));
        for bad in ["NaN", "nan", "inf", "-inf", "-1", "-0.5", "1e20", "ten", ""] {
            assert!(seconds(bad).is_err(), "{bad}");
        }
        for bad in ["NaN", "-1", "1e300", "1e16"] {
            assert!(hours(bad).is_err(), "{bad}");
        }
        // Valid alone, too long once three times it is taken (the status refresh).
        assert_eq!(seconds("1e19"), Ok(1e19));
        assert!(flag("--watchdog-seconds", 1e19 * 3.0)
            .unwrap_err()
            .starts_with("--watchdog-seconds"));
    }

    #[test]
    fn a_silent_factor_is_zero_or_a_finite_number_above_one() {
        for good in ["0", "-0", "1.0001", "5", "1e300"] {
            assert!(factor(good).is_ok(), "{good}");
        }
        for bad in ["NaN", "inf", "-inf", "1e400", "-1", "-0.5", "0.5", "1", "five", ""] {
            assert!(factor(bad).is_err(), "{bad}");
        }
    }
}
