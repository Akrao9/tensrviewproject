//! Run the server's detectors over a store on local disk, to compare with Python or to time them.
//!
//!     cargo run --release --example detect_store -- <store> <run_id> --rank 0
//!     cargo run --release --example detect_store -- <store> <run_id> --bench
//!
//! `--rank` rebuilds one rank's detectors from its stored parts, as the server does after a
//! restart, and prints the incidents as JSON. `--bench` plays the store back as the flushes
//! it was written in (one part per rank and flush): for each flush it reads that part of
//! every rank, then times the detectors over those batches on one thread, apart from the
//! reading. It also times writing the same batches as Parquet parts to a scratch store, the
//! JSON decoding alone, and rebuilding the ranks from the store.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use arrow::array::{Array, StringArray};
use arrow::record_batch::RecordBatch;
use serde::de::IgnoredAny;
use tensorquill_server::detect::{replay, RankDetectors, PORTED};
use tensorquill_server::storage::{Error, Storage};

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn ranks_of(store: &str, run_id: &str) -> Result<Vec<i32>, Error> {
    let mut ranks = Vec::new();
    for entry in std::fs::read_dir(std::path::Path::new(store).join("runs").join(run_id))? {
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
    Ok(ranks)
}

/// Decodes the two JSON columns of a batch and nothing else, to see what the decoding costs.
fn decode_json(batch: &RecordBatch) -> usize {
    let mut members = 0;
    for name in ["layers_json", "moe_stats_json"] {
        let Some(column) = batch
            .column_by_name(name)
            .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        else {
            continue;
        };
        for row in 0..column.len() {
            if column.is_valid(row) {
                let parsed: Result<BTreeMap<&str, BTreeMap<&str, IgnoredAny>>, _> =
                    serde_json::from_str(column.value(row));
                members += parsed.map_or(0, |parsed| parsed.len());
            }
        }
    }
    members
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [store, run_id, mode, rest @ ..] = args.as_slice() else {
        return Err("usage: detect_store <store> <run_id> (--rank <n> | --bench)".into());
    };
    let storage = Storage::from_url(store)?;
    if mode == "--rank" {
        let rank: i32 = rest.first().ok_or("--rank needs a number")?.parse()?;
        let detectors = replay(&storage, run_id, rank).await?;
        let document = serde_json::json!({
            "run_id": run_id,
            "rank": rank,
            "last_step": detectors.last_step(),
            "steps": detectors.rows(),
            "checked": PORTED,
            "incidents": detectors.incidents(),
        });
        println!("{}", serde_json::to_string_pretty(&document)?);
        return Ok(());
    }

    let ranks = ranks_of(store, run_id)?;
    let mut parts = Vec::new();
    for rank in &ranks {
        parts.push(storage.rank_parts(run_id, *rank).await?);
    }
    let flushes = parts.iter().map(Vec::len).min().unwrap_or(0);
    println!("{} ranks, {flushes} flushes each", ranks.len());
    let scratch_dir = tempfile::tempdir()?;
    let scratch = Storage::from_url(scratch_dir.path().to_str().ok_or("bad scratch path")?)?;

    let mut detectors: Vec<RankDetectors> = ranks.iter().map(|_| RankDetectors::new()).collect();
    let (mut detect, mut write, mut decode, mut read) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut rows, mut steps_per_flush) = (0usize, 0usize);
    for flush in 0..flushes {
        let started = Instant::now();
        let mut batches: Vec<Vec<RecordBatch>> = Vec::new();
        for rank_parts in &parts {
            batches.push(storage.read_part(&rank_parts[flush].location).await?);
        }
        read.push(ms(started.elapsed()));
        steps_per_flush = batches[0].iter().map(RecordBatch::num_rows).sum();

        let started = Instant::now();
        for (detectors, batches) in detectors.iter_mut().zip(&batches) {
            for batch in batches {
                detectors.observe(batch);
                rows += batch.num_rows();
            }
        }
        detect.push(ms(started.elapsed()));

        let started = Instant::now();
        let mut members = 0;
        for batch in batches.iter().flatten() {
            members += decode_json(batch);
        }
        std::hint::black_box(members);
        decode.push(ms(started.elapsed()));

        let started = Instant::now();
        for (rank, batches) in ranks.iter().zip(&batches) {
            scratch
                .write_part(run_id, *rank, 0, batches, flush as i64, flush as i64)
                .await?;
        }
        write.push(ms(started.elapsed()));
    }
    let incidents: usize = detectors.iter().map(|detectors| detectors.incidents().len()).sum();
    let worst = |values: &[f64]| values.iter().copied().fold(0.0, f64::max);
    println!("{rows} rows read, {steps_per_flush} steps per rank and flush, {incidents} incidents");
    println!("per flush of {} ranks, one thread, ms (median / worst):", ranks.len());
    for (name, values) in [
        ("detectors", &mut detect),
        ("  of which JSON decoding alone (measured apart)", &mut decode),
        ("writing the Parquet parts", &mut write),
        ("reading the parts back (not on the live path)", &mut read),
    ] {
        let worst = worst(values);
        println!("  {name}: {:.2} / {:.2}", median(values), worst);
    }
    let per_step = median(&mut detect) * 1e3 / (ranks.len() * steps_per_flush.max(1)) as f64;
    println!("  detectors per step: {per_step:.2} us");

    // Rebuilding from the store, as after a restart: one rank, then all of them at once.
    let started = Instant::now();
    let one = replay(&storage, run_id, ranks[0]).await?;
    println!(
        "replay of one rank ({} steps): {:.0} ms",
        one.rows(),
        ms(started.elapsed())
    );
    let started = Instant::now();
    let mut tasks = Vec::new();
    for rank in ranks.clone() {
        let (storage, run_id) = (storage.clone(), run_id.clone());
        tasks.push(tokio::spawn(async move {
            replay(&storage, &run_id, rank).await.map(|detectors| detectors.rows())
        }));
    }
    let mut replayed = 0;
    for task in tasks {
        replayed += task.await??;
    }
    println!(
        "replay of all {} ranks at once ({replayed} steps): {:.0} ms",
        ranks.len(),
        ms(started.elapsed())
    );
    Ok(())
}
