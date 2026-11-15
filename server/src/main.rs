use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use tensorquill_server::ingest::{Ingest, IngestConfig, DEFAULT_FLUSH_STEPS, DEFAULT_QUOTA_BYTES};
use tensorquill_server::server::{check_bind, serve, Tls};
use tensorquill_server::storage::{Error, Storage};
use tokio::net::TcpListener;

/// Self-hosted TensorQuill ingest: Arrow Flight DoPut in, partitioned Parquet out.
#[derive(Parser)]
#[command(name = "tensorquill-server", version)]
struct Args {
    /// Address to listen on. Anything but loopback needs TENSORQUILL_TOKEN or --allow-unauthenticated.
    #[arg(long, default_value = "127.0.0.1:50051")]
    listen: SocketAddr,
    /// Where Parquet parts go: file:///var/lib/tensorquill, s3://bucket/prefix, gs://bucket/prefix, or a directory.
    #[arg(long)]
    store: String,
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
    /// How many times a rank's usual time between batches it may stay quiet before it is silent.
    #[arg(long, default_value_t = 5.0)]
    silent_factor: f64,
    /// Rewrite runs/<run_id>/watchdog.json this often; 0 turns the file off.
    #[arg(long, default_value_t = 10.0)]
    watchdog_seconds: f64,
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

async fn run(args: Args) -> Result<(), Error> {
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
        watchdog_interval: Duration::from_secs_f64(args.watchdog_seconds),
    };
    let ingest = Ingest::new(Storage::from_url(&args.store)?, config);
    let listener = TcpListener::bind(args.listen).await?;
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
    ingest.write_watchdog().await; // the last word on each run before the server stops
    ingest.shutdown().await?;
    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Args::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("tensorquill-server: {error}");
            ExitCode::FAILURE
        }
    }
}
