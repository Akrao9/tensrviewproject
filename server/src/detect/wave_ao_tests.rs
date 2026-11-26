//! Lane AO-4's tests (wave AO): two findings on the Rust detectors.
//!
//! - BUG-01: a router's counts are signed 64-bit integers (`moe_counts`, `Map<Utf8, List<Int64>>`), and
//!   Python sums them as integers that do not overflow. Summed in `i64` here, a step's counts, the
//!   starved check's window and the top-k baseline's window passed `i64::MAX` (or went below
//!   `i64::MIN`): a debug build panicked, and a release build wrapped, so a router that routed
//!   tokens read as one that routed none (`moe_routing`), and a window's shares were wrong. Every
//!   such sum is judged here as Python judges it, and `routed_tokens` is written as Python writes
//!   it, the exact sum.
//! - BUG-10: a probe whose `attention` is not an object (`null`, a list, a number, text, a bool)
//!   failed the whole probe's parse, so its `error` or `not_run` was lost and the attention kinds
//!   read `not_configured` where Python reads them `unavailable`. Such an `attention` is now no
//!   attention, as Python's `isinstance(attention, dict)` reads it, and the rest of the probe is read.
//!
//! The same cases run through the binary against Python's detectors in
//! `sdk/tests/test_wave_ao_detect.py`.

use std::sync::Arc;

use arrow::array::{
    Array, Int32Array, Int32Builder, Int64Array, Int64Builder, ListBuilder, MapBuilder, StringArray, StringBuilder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Value};

use super::{detect_stream_samples, Thresholds};

const ROUTER: &str = "router";

/// One row per step from 0, each recording `ROUTER`'s counts, with `top_k` for it on every row.
fn moe_batch(steps: &[Vec<i64>], top_k: i32) -> RecordBatch {
    let mut counts = MapBuilder::new(None, StringBuilder::new(), ListBuilder::new(Int64Builder::new()));
    let mut chosen = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
    for values in steps {
        counts.keys().append_value(ROUTER);
        counts.values().append_value(values.iter().copied().map(Some));
        counts.append(true).unwrap();
        chosen.keys().append_value(ROUTER);
        chosen.values().append_value(top_k);
        chosen.append(true).unwrap();
    }
    let (counts, chosen) = (counts.finish(), chosen.finish());
    let schema = Schema::new(vec![
        Field::new("run_id", DataType::Utf8, false),
        Field::new("rank", DataType::Int32, false),
        Field::new("attempt", DataType::Int64, false),
        Field::new("step", DataType::Int64, false),
        Field::new("moe_counts", counts.data_type().clone(), true),
        Field::new("moe_top_k", chosen.data_type().clone(), true),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from_iter_values(steps.iter().map(|_| "run"))),
            Arc::new(Int32Array::from_iter_values(steps.iter().map(|_| 0))),
            Arc::new(Int64Array::from_iter_values(steps.iter().map(|_| 1))),
            Arc::new(Int64Array::from_iter_values(0..steps.len() as i64)),
            Arc::new(counts),
            Arc::new(chosen),
        ],
    )
    .unwrap()
}

fn stream(batch: &RecordBatch) -> Vec<u8> {
    let mut out = Vec::new();
    let mut writer = arrow::ipc::writer::StreamWriter::try_new(&mut out, &batch.schema()).unwrap();
    writer.write(batch).unwrap();
    writer.finish().unwrap();
    drop(writer);
    out
}

/// What `detect --samples` writes for the batch, as text: a sum above `u64::MAX` is a float once it
/// is parsed into a `Value`, so its digits are asserted in the text.
fn samples_text(batch: &RecordBatch, thresholds: &Arc<Thresholds>) -> String {
    let mut out = Vec::new();
    detect_stream_samples(&stream(batch)[..], thresholds, &mut out).unwrap();
    String::from_utf8(out).unwrap()
}

/// The report and the samples of `samples_text`, parsed.
fn report_and_samples(batch: &RecordBatch, thresholds: &Arc<Thresholds>) -> (Value, Vec<Value>) {
    let mut report: Value = serde_json::from_str(&samples_text(batch, thresholds)).unwrap();
    let samples = report.as_object_mut().unwrap().remove("samples").unwrap();
    (report, samples.as_array().unwrap().clone())
}

fn usual() -> Arc<Thresholds> {
    Arc::default()
}

fn window(starve_window: usize) -> Arc<Thresholds> {
    Arc::new(Thresholds {
        starve_window,
        ..Thresholds::default()
    })
}

fn checks_named(report: &Value, check: &str) -> Vec<Value> {
    report["not_checked"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|entry| entry["check"] == check)
        .cloned()
        .collect()
}

fn kinds(report: &Value) -> Vec<String> {
    report["incidents"]
        .as_array()
        .unwrap()
        .iter()
        .map(|incident| incident["kind"].as_str().unwrap().to_owned())
        .collect()
}

const TWO_62: i64 = 1 << 62;

// --- BUG-01 ------------------------------------------------------------------------------------

/// `[2**62, 2**62]` routed 2**63 tokens, one more than `i64::MAX`: two experts with even shares.
/// Python: a judged sample, healthy. Before: a panic (debug), or a sum of `i64::MIN`, read as a
/// router that routed nothing (`moe_routing`), degraded (release).
#[test]
fn a_steps_counts_summing_past_i64_max_are_a_judged_sample() {
    let batch = moe_batch(&[vec![TWO_62, TWO_62]], 1);
    let text = samples_text(&batch, &usual());
    assert!(text.contains(r#""routed_tokens":9223372036854775808,"#), "{text}");
    let (report, samples) = report_and_samples(&batch, &usual());
    assert_eq!(checks_named(&report, "moe_routing"), Vec::<Value>::new());
    assert_eq!(kinds(&report), Vec::<String>::new());
    assert_eq!(report["kind_checks"]["moe_imbalance"]["value"], "checked");
    let sample = &samples[0];
    assert_eq!(sample.get("missing"), None);
    assert_eq!(sample["diversity"], json!(1.0));
    assert_eq!((&sample["overloaded"], &sample["dead"]), (&json!([]), &json!([])));
    assert_eq!(sample["overload_rule"], json!({"rule": "share", "threshold": 0.75}));
}

/// Counts summing below `i64::MIN`: Python sums them to -2**64, which routed no token, a missing
/// sample. Before: a panic (debug), or a wrap to 0 (release).
#[test]
fn a_steps_counts_summing_below_i64_min_are_a_missing_sample_with_their_exact_sum() {
    let batch = moe_batch(&[vec![i64::MIN, i64::MIN]], 1);
    let text = samples_text(&batch, &usual());
    assert!(text.contains(r#""routed_tokens":-18446744073709551616,"#), "{text}");
    let (report, samples) = report_and_samples(&batch, &usual());
    assert_eq!(samples[0]["missing"], true);
    assert_eq!(checks_named(&report, "moe_routing").len(), 1);
    // A sum of 2**64 is written in full too, where a parsed `Value` holds it only as a float.
    let three = moe_batch(&[vec![i64::MAX, i64::MAX, 2]], 1);
    assert!(samples_text(&three, &usual()).contains(r#""routed_tokens":18446744073709551616,"#));
}

/// The starved check sums the trailing window's tokens and each expert's counts. Three steps of
/// `[2**62, 1]`: each step's sum fits, the window's (3 * 2**62 + 3) and expert 1's (3 * 2**62) do
/// not. Python: expert 2 has 1 / (2**62 + 1) of the window, starved at step 2.
#[test]
fn the_starved_windows_sums_past_i64_max_are_the_exact_sums() {
    let steps = vec![vec![TWO_62, 1]; 3];
    let (report, samples) = report_and_samples(&moe_batch(&steps, 1), &window(3));
    assert_eq!(samples[2]["starved"], json!([2]));
    let share = 3.0 / (3.0 * TWO_62 as f64);
    assert_eq!(samples[2]["starved_shares"], json!({"2": share}));
    let starved: Vec<&Value> = report["incidents"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|incident| incident["kind"] == "starved_experts")
        .collect();
    assert_eq!(starved.len(), 1, "{report}");
    assert_eq!(
        starved[0]["experts"],
        json!([{"layer": 1, "router": ROUTER, "expert": 2, "step": 2, "window_share": share}])
    );
    assert_eq!(checks_named(&report, "moe_routing"), Vec::<Value>::new());
}

/// A top-k router's baseline sums the window's tokens and each expert's counts. Ten even steps of
/// eight experts at 2**59 (2**62 a step, so the baseline's tokens pass `i64::MAX` at its second
/// sample), then a step where two experts take every token: its diversity dropped to a quarter of
/// the baseline's and experts 1 and 2 rose by 1.5 over their share there. Python flags them, and
/// the six with no token as dead. Before, the wrapped baseline read as no tokens and nothing was
/// flagged (release).
#[test]
fn the_top_k_baselines_sums_past_i64_max_are_the_exact_sums() {
    let even = vec![1_i64 << 59; 8];
    let mut steps = vec![even; 10];
    steps.push(vec![1 << 61, 1 << 61, 0, 0, 0, 0, 0, 0]);
    let (report, samples) = report_and_samples(&moe_batch(&steps, 4), &usual());
    let last = &samples[10];
    assert_eq!(last["overloaded"], json!([1, 2]), "{last}");
    assert_eq!(last["dead"], json!([3, 4, 5, 6, 7, 8]));
    assert_eq!(last["overload_rule"]["rule"], "diversity_drop");
    let imbalance: Vec<&Value> = report["incidents"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|incident| incident["kind"] == "moe_imbalance")
        .collect();
    assert_eq!(imbalance.len(), 1, "{report}");
    assert_eq!(
        (&imbalance[0]["step"], &imbalance[0]["overloaded"]),
        (&json!(10), &json!([1, 2]))
    );
}

/// The diversity a step's counts give when the capture recorded none sums them too: `[2**62,
/// 2**62]` is two even experts, a diversity of 1, and `[2**62, 2**62, 2**62, 2**62]` four.
#[test]
fn the_diversity_of_counts_summing_past_i64_max_is_their_evenness() {
    assert_eq!(super::diversity_from_counts(&[TWO_62, TWO_62]), 1.0);
    assert_eq!(super::diversity_from_counts(&[TWO_62; 4]), 1.0);
    assert_eq!(super::diversity_from_counts(&[i64::MIN, i64::MIN]), 0.0);
}
