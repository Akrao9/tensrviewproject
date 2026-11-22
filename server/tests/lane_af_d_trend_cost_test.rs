//! Lane D of wave AF (Nov 21, 2026): what a live report costs, measured in-process on a release
//! build, before and after the trend memoization. Ignored: run with
//! `cargo test --release --test lane_af_d_trend_cost_test -- --ignored --nocapture`.
//! It uses only what the base (`62fce96`) has (`RankDetectors::new`, `append`, `detect::report`), so
//! the same file measures both trees. It asserts nothing about the numbers.
//!
//! The input, one run of 4 ranks, each judged as the live server judges it (`Ingest::detect`: append
//! the stored part, then make the report): 50,000 steps per rank in parts of 500 rows (the server's
//! default `--flush-steps`); a total gradient norm on every step; a probe of 12 attention modules
//! every 1,000 steps, so every other part has a probe and the other has none.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use serde_json::json;
use tensorquill_server::detect::{report, RankDetectors, Thresholds};

const RANKS: i32 = 4;
const STEPS: i64 = 50_000;
const PART: i64 = 500;
const PROBE_EVERY: i64 = 1_000;
const MODULES: usize = 12;

fn part(rank: i32, start: i64) -> RecordBatch {
    let steps: Vec<i64> = (start..start + PART).collect();
    let n = steps.len();
    let wobble = |step: i64, salt: i64| 0.9 + ((step * 37 + salt * 11) % 21) as f64 / 100.0;
    let norms: Vec<f64> = steps.iter().map(|s| 2.0 * wobble(*s, 99 + rank as i64)).collect();
    let probes: Vec<Option<String>> = steps
        .iter()
        .map(|s| {
            (s % PROBE_EVERY == 0).then(|| {
                let modules: serde_json::Map<String, serde_json::Value> = (0..MODULES)
                    .map(|m| {
                        let logit = 5.0 + *s as f64 * 1e-5 * (m as f64 + 1.0) * wobble(*s, m as i64);
                        (format!("blocks.{m}.attn"), json!({"max_logit": [logit, logit - 1.0]}))
                    })
                    .collect();
                json!({"attention": modules}).to_string()
            })
        })
        .collect();
    RecordBatch::try_from_iter_with_nullable(vec![
        ("schema_version", Arc::new(Int32Array::from(vec![1; n])) as ArrayRef, false),
        ("run_id", Arc::new(StringArray::from(vec!["cost"; n])) as ArrayRef, false),
        ("rank", Arc::new(Int32Array::from(vec![rank; n])) as ArrayRef, false),
        ("attempt", Arc::new(Int64Array::from(vec![1; n])) as ArrayRef, false),
        ("step", Arc::new(Int64Array::from(steps)) as ArrayRef, false),
        ("grad_norm", Arc::new(Float64Array::from(norms)) as ArrayRef, true),
        ("probe_json", Arc::new(StringArray::from(probes)) as ArrayRef, true),
    ])
    .unwrap()
}

#[test]
#[ignore = "a measurement: run on a release build with --ignored"]
fn the_cost_of_a_live_report_over_a_long_multi_rank_run() {
    let thresholds = Arc::new(Thresholds::default());
    // Every part built before any timing.
    let parts: Vec<Vec<RecordBatch>> = (0..RANKS)
        .map(|rank| (0..STEPS / PART).map(|index| part(rank, index * PART)).collect())
        .collect();
    let mut ranks: Vec<RankDetectors> = (0..RANKS).map(|_| RankDetectors::new(&thresholds)).collect();
    let (mut report_ns, mut append_ns) = (Vec::new(), Vec::new());
    let (mut with_probe, mut without_probe) = (Vec::new(), Vec::new());
    let mut last = Vec::new();
    let started = Instant::now();
    // The ranks' parts interleaved, as they arrive at a server.
    for index in 0..(STEPS / PART) as usize {
        for (rank, detectors) in ranks.iter_mut().enumerate() {
            let at = Instant::now();
            detectors.append(1, &parts[rank][index]);
            append_ns.push(at.elapsed().as_nanos() as u64);
            let at = Instant::now();
            let document = report("cost", rank as i32, detectors, 0);
            let took = at.elapsed().as_nanos() as u64;
            report_ns.push(took);
            if (index as i64 * PART) % PROBE_EVERY == 0 {
                with_probe.push(took);
            } else {
                without_probe.push(took);
            }
            if index + 1 == (STEPS / PART) as usize {
                last.push(document);
            }
        }
    }
    let total = started.elapsed();
    let mean = |values: &[u64]| values.iter().sum::<u64>() as f64 / values.len() as f64 / 1e6;
    let tail = |values: &[u64]| mean(&values[values.len() - 40..]);
    let digest: u64 = last
        .iter()
        .map(|document| document.to_string())
        .flat_map(|text| text.into_bytes())
        .fold(0xcbf29ce484222325, |hash, byte| (hash ^ byte as u64).wrapping_mul(0x100000001b3));
    println!(
        "{}",
        json!({
            "ranks": RANKS, "steps_per_rank": STEPS, "part_rows": PART, "probe_every": PROBE_EVERY,
            "attention_modules": MODULES, "reports": report_ns.len(),
            "report_ms_mean_all": mean(&report_ns),
            "report_ms_mean_last_40": tail(&report_ns),
            "report_ms_mean_parts_with_a_probe": mean(&with_probe),
            "report_ms_mean_parts_without_a_probe": mean(&without_probe),
            "append_ms_mean_all": mean(&append_ns),
            "total_s_append_and_report": total.as_secs_f64(),
            // The last reports of every rank, hashed (FNV-1a of their JSON): equal before and after.
            "last_reports_fnv1a": format!("{digest:016x}"),
        })
    );
}
