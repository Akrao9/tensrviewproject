//! Lane W3's tests (Nov 19, 2026): the detector rules this lane changed, on rows built here, so
//! `cargo test` holds them without the SDK. `sdk/tests/test_lane_w3_detector_rules.py` holds the
//! same rules through both engines.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float64Builder, Int32Array, Int64Array, ListBuilder, MapBuilder, StringArray, StringBuilder,
};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use serde_json::Value;

use super::{judge, resolve, Thresholds};

fn usual() -> Arc<Thresholds> {
    Arc::default()
}

/// One row: (attempt, step).
fn keys(rows: &[(i64, i64)]) -> Vec<(&'static str, ArrayRef)> {
    vec![
        (
            "run_id",
            Arc::new(StringArray::from_iter_values(rows.iter().map(|_| "run"))) as ArrayRef,
        ),
        ("rank", Arc::new(Int32Array::from_iter_values(rows.iter().map(|_| 0)))),
        (
            "attempt",
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|(attempt, _)| *attempt))),
        ),
        (
            "step",
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|(_, step)| *step))),
        ),
    ]
}

fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
        .collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns.into_iter().map(|(_, array)| array).collect(),
    )
    .unwrap()
}

fn floats(values: &[f64]) -> ArrayRef {
    Arc::new(arrow::array::Float64Array::from(values.to_vec()))
}

/// A `grad_norms` map per row: each row's (name, norm) pairs.
fn grad_norms(rows: &[Vec<(&str, f64)>]) -> ArrayRef {
    let mut norms = MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
    for row in rows {
        for (name, norm) in row {
            norms.keys().append_value(name);
            norms.values().append_value(*norm);
        }
        norms.append(true).unwrap();
    }
    Arc::new(norms.finish())
}

/// A `weights_unchanged` list per row; `None` is a row with no weight check.
fn unchanged(rows: &[Option<Vec<&str>>]) -> ArrayRef {
    let mut lists = ListBuilder::new(StringBuilder::new());
    for row in rows {
        match row {
            Some(names) => {
                for name in names {
                    lists.values().append_value(name);
                }
                lists.append(true);
            }
            None => lists.append(false),
        }
    }
    Arc::new(lists.finish())
}

fn lrs(values: &[f64]) -> ArrayRef {
    let mut lists = ListBuilder::new(Float64Builder::new());
    for value in values {
        lists.values().append_value(*value);
        lists.append(true);
    }
    Arc::new(lists.finish())
}

fn incidents(batches: &[RecordBatch]) -> Vec<Value> {
    judge(0, batches, &usual()).unwrap().incidents()
}

fn of_kind<'a>(found: &'a [Value], kind: &str) -> Vec<&'a Value> {
    found.iter().filter(|incident| incident["kind"] == kind).collect()
}

// --- contract 2 (E1) --------------------------------------------------------------------------

/// The weight checks of one parameter `p`: (unchanged, has a gradient).
fn checks(rows: &[(bool, bool)]) -> RecordBatch {
    let steps: Vec<(i64, i64)> = (0..rows.len() as i64).map(|step| (1, step)).collect();
    let mut columns = keys(&steps);
    let grads: Vec<Vec<(&str, f64)>> = rows
        .iter()
        .map(|(_, grad)| if *grad { vec![("p", 0.5)] } else { vec![] })
        .collect();
    let lists: Vec<Option<Vec<&str>>> = rows
        .iter()
        .map(|(same, _)| Some(if *same { vec!["p"] } else { vec![] }))
        .collect();
    columns.push(("grad_norms", grad_norms(&grads)));
    columns.push(("weights_unchanged", unchanged(&lists)));
    batch(columns)
}

#[test]
fn e1_a_change_without_a_gradient_breaks_the_streak_and_a_later_stall_still_fires() {
    // The audit's case: unchanged with a gradient at 0 and 1, changed with none at 2, unchanged
    // with a gradient at 3. Three in a row at 3 until lane W3.
    let audit = [(true, true), (true, true), (false, false), (true, true)];
    assert!(of_kind(&incidents(&[checks(&audit)]), "stalled_weights").is_empty());
    // The acceptance case: two more unchanged checks with a gradient, and the stall fires at 5.
    let mut rows = audit.to_vec();
    rows.extend([(true, true), (true, true)]);
    let found = incidents(&[checks(&rows)]);
    let stalled = of_kind(&found, "stalled_weights");
    assert_eq!(stalled.len(), 1, "{found:?}");
    assert_eq!(stalled[0]["step"], 5);
    assert_eq!(stalled[0]["onsets"][0]["trained_before"], 1, "{}", stalled[0]);
}

#[test]
fn e1_listed_without_a_gradient_holds_the_streak() {
    // Contract 2, A: not asked to move then, so neither counts nor breaks.
    let rows = [(true, true), (true, true), (true, false), (true, true)];
    let found = incidents(&[checks(&rows)]);
    assert_eq!(of_kind(&found, "stalled_weights")[0]["step"], 3);
}

// --- one step order within an attempt (E5, DET-10, RUST-05) -----------------------------------

#[test]
fn e5_the_first_fault_is_the_lowest_step_not_the_first_row() {
    let rows = [(1, 3), (1, 4), (1, 9), (1, 0), (1, 1), (1, 2)];
    let mut columns = keys(&rows);
    columns.push(("grad_norm", floats(&[1.0, 1.0, f64::NAN, 1.0, 1.0, f64::NAN])));
    let found = incidents(&[batch(columns)]);
    assert_eq!(of_kind(&found, "non_finite")[0]["step"], 2, "{found:?}");
}

#[test]
fn rust05_a_resume_is_compared_at_the_attempts_lowest_step() {
    // Attempt 2's rows come as 15, 16, 17, 12, 13, 14, with another rate at 12.
    let mut rows: Vec<(i64, i64)> = (0..21).map(|step| (1, step)).collect();
    rows.extend([15, 16, 17, 12, 13, 14].map(|step| (2, step)));
    let rates: Vec<f64> = rows
        .iter()
        .map(|(attempt, step)| if (*attempt, *step) == (2, 12) { 0.5 } else { 0.1 })
        .collect();
    let mut columns = keys(&rows);
    columns.push(("lr", lrs(&rates)));
    let found = incidents(&[batch(columns)]);
    assert_eq!(of_kind(&found, "resume_lr_change")[0]["step"], 12, "{found:?}");
}

#[test]
fn rust05_a_numbering_restart_is_seen_whatever_row_comes_first() {
    let mut rows: Vec<(i64, i64)> = (0..10).map(|step| (1, step)).collect();
    rows.extend([5, 6, 0, 1, 2].map(|step| (2, step)));
    let found = incidents(&[batch(keys(&rows))]);
    assert_eq!(of_kind(&found, "repeated_steps")[0]["step"], 0, "{found:?}");
}

// --- contract 1, the offline half ---------------------------------------------------------------

#[test]
fn contract1_attempts_are_read_in_the_order_they_first_appear() {
    // A = 900 writes 0 to 101, then B = 2 resumes at 30: B owns the rank, whatever its number.
    let mut rows: Vec<(i64, i64)> = (0..102).map(|step| (900, step)).collect();
    rows.extend((30..36).map(|step| (2, step)));
    let detectors = judge(0, &[batch(keys(&rows))], &usual()).unwrap();
    assert_eq!(
        (detectors.attempt(), detectors.last_step(), detectors.rows()),
        (2, Some(35), 36)
    );
    assert_eq!(
        resolve(&[(900, 0), (2, 30)]),
        std::collections::BTreeMap::from([(900, Some(30)), (2, None)])
    );
}
