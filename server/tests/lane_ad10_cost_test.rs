//! Lane AD10 (Nov 21, 2026): what the measurements of `DETECTOR_RULES_CONTRACT.md` part 6 cost a
//! rank's detectors on the live path, measured, not estimated. Ignored: run it on a release build,
//!
//! ```text
//! cargo test --release --test lane_ad10_cost_test -- --ignored --nocapture --test-threads=1
//! ```
//!
//! A counting allocator (this test binary's own) gives the heap the detectors hold, and the peak
//! while a report is made. The rows go in as the live path gives them: parts of 500 rows (the
//! server's default `--flush-steps`), each appended (`RankDetectors::append`) and followed by a
//! report (`detect::report`), as `Ingest::detect` does after every stored part. It prints one JSON
//! line per run length. The cost of the two trends' series is this run against the same run built
//! with `KEEP_TREND_SERIES = false` (a copy of the tree, never a shipped build).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, Float64Builder, Int32Array, Int64Array, MapBuilder, StringArray, StringBuilder};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use tensorquill_server::detect::{report, RankDetectors, Thresholds};

struct Counting;
static NOW: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let now = NOW.fetch_add(layout.size() as isize, Ordering::Relaxed) + layout.size() as isize;
        PEAK.fetch_max(now, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        NOW.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let now = NOW.fetch_add(new_size as isize - layout.size() as isize, Ordering::Relaxed) + new_size as isize
            - layout.size() as isize;
        PEAK.fetch_max(now, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

const PARAMETERS: usize = 16;
const MODULES: usize = 12;
const EVERY: i64 = 100; // a probe and a weight check every 100 steps
const PART: i64 = 500;

fn map(rows: &[Option<Vec<(String, f64)>>]) -> ArrayRef {
    let mut map = MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
    for row in rows {
        match row {
            Some(entries) => {
                for (name, value) in entries {
                    map.keys().append_value(name);
                    map.values().append_value(*value);
                }
                map.append(true).unwrap();
            }
            None => map.append(false).unwrap(),
        }
    }
    Arc::new(map.finish())
}

/// One part of the workload: steps `from..from + PART` of one rank, each with a total gradient norm
/// and `PARAMETERS` update sizes, and every `EVERY` steps a weight check of `PARAMETERS` tensors and
/// a probe of `MODULES` attention modules of four heads.
fn part(from: i64) -> RecordBatch {
    let steps: Vec<i64> = (from..from + PART).collect();
    let wobble = |step: i64, salt: i64| 0.9 + ((step * 37 + salt * 11) % 21) as f64 / 100.0;
    let updates: Vec<_> = steps
        .iter()
        .map(|s| {
            Some(
                (0..PARAMETERS)
                    .map(|p| (format!("layers.{p}.weight"), 1e-3 * wobble(*s, p as i64)))
                    .collect(),
            )
        })
        .collect();
    let weights: Vec<_> = steps
        .iter()
        .map(|s| {
            (s % EVERY == 0).then(|| {
                (0..PARAMETERS)
                    .map(|p| {
                        (
                            format!("layers.{p}.weight"),
                            10.0 + *s as f64 * 1e-4 * wobble(*s, p as i64),
                        )
                    })
                    .collect()
            })
        })
        .collect();
    let probes: Vec<Option<String>> = steps
        .iter()
        .map(|s| {
            (s % EVERY == 0).then(|| {
                let modules: Vec<String> = (0..MODULES)
                    .map(|m| {
                        let logit = 5.0 + *s as f64 * 1e-5 * (m + 1) as f64 * wobble(*s, m as i64);
                        format!(
                            r#""blocks.{m}.attn":{{"entropy_share":[0.6,0.61,0.62,0.63],"max_logit":[{logit},{a},{b},{c}]}}"#,
                            a = logit - 1.0,
                            b = logit - 2.0,
                            c = logit - 3.0
                        )
                    })
                    .collect();
                format!(r#"{{"attention":{{{}}}}}"#, modules.join(","))
            })
        })
        .collect();
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("run_id", Arc::new(StringArray::from(vec!["cost"; steps.len()]))),
        ("rank", Arc::new(Int32Array::from(vec![0; steps.len()]))),
        ("attempt", Arc::new(Int64Array::from(vec![1; steps.len()]))),
        ("step", Arc::new(Int64Array::from(steps.clone()))),
        (
            "grad_norm",
            Arc::new(arrow::array::Float64Array::from(
                steps.iter().map(|s| 2.0 * wobble(*s, 99)).collect::<Vec<_>>(),
            )),
        ),
        ("update_rms", map(&updates)),
        ("weight_norms", map(&weights)),
        ("probe_json", Arc::new(StringArray::from(probes))),
    ];
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), *name != "run_id"))
        .collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns.into_iter().map(|(_, array)| array).collect(),
    )
    .unwrap()
}

#[test]
#[ignore]
fn what_the_live_path_holds_and_spends_at_10_000_and_100_000_steps() {
    let lengths = [10_000_i64, 100_000];
    let thresholds = Arc::new(Thresholds::default());
    let before = NOW.load(Ordering::SeqCst);
    let mut detectors = RankDetectors::new(&thresholds);
    let mut times: Vec<f64> = Vec::new();
    let mut step = 0;
    for length in lengths {
        let mut last: Vec<(f64, f64, f64)> = Vec::new(); // the last five parts: append, report, peak
        while step < length {
            let batch = part(step);
            let started = Instant::now();
            detectors.append(1, &batch);
            let appended = started.elapsed().as_secs_f64();
            drop(batch);
            let held = NOW.load(Ordering::SeqCst);
            PEAK.store(held, Ordering::SeqCst);
            let started = Instant::now();
            let document = report("cost", 0, &detectors, 0);
            let reported = started.elapsed().as_secs_f64();
            let peak = PEAK.load(Ordering::SeqCst) - held;
            assert!(document["incidents"].is_array());
            drop(document);
            times.push(appended + reported);
            last.push((appended, reported, peak as f64));
            if last.len() > 5 {
                last.remove(0);
            }
            step += PART;
        }
        let document = report("cost", 0, &detectors, 0);
        let blocks: Vec<&str> = ["gradient_trend", "attention_trends", "update_trends", "weight_jumps"]
            .into_iter()
            .filter(|b| document.get(*b).is_some())
            .collect();
        let points = document["gradient_trend"]["points"].clone();
        drop(document);
        let held = NOW.load(Ordering::SeqCst) - before;
        let mean = |f: fn(&(f64, f64, f64)) -> f64| last.iter().map(f).sum::<f64>() / last.len() as f64;
        println!(
            "{}",
            serde_json::json!({
                "steps": length, "rank_count": 1, "parameters": PARAMETERS, "attention_modules": MODULES,
                "probes": length / EVERY, "weight_checks": length / EVERY,
                "gradient_samples": points, "attention_samples": (length / EVERY) as usize * MODULES,
                "blocks": blocks,
                "held_bytes": held,
                "last_5_parts": {"append_ms": mean(|p| p.0) * 1e3, "report_ms": mean(|p| p.1) * 1e3,
                                 "report_peak_bytes_above_held": mean(|p| p.2)},
                "all_parts_append_and_report_s": times.iter().sum::<f64>(),
            })
        );
    }
}
