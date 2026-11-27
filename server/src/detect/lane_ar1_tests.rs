//! Lane AR-1's tests (Nov 26, 2026): an incident's scalar context, AQ5-D4 and AQ5-D7.
//!
//! - D4 (the author: "Rust brought to Python: \"before\" leaves out the same step's earlier
//!   writing"): an incident first reported on a later writing of a step shows the scalars of the
//!   step's first writing, with the median of the records before that writing, as Python's
//!   `scalar_context` takes the first record with the step. On the raw route and on a live rank.
//! - D7 (`WAVE_AJ_CONTRACT.md` section 3, closed on the author's word of Nov 26, 2026): a refused
//!   `scalars` entry is absent and its siblings are kept. A row stored before the change has a
//!   null `scalars` column and the whole map in `extra_json`; Rust reads the entries the column
//!   would hold from there, as Python reads the record back.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float64Builder, Int32Array, Int64Array, Int64Builder, ListBuilder, MapBuilder, StringArray, StringBuilder,
};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Value};

use super::{judge, RankDetectors, Thresholds};

fn usual() -> Arc<Thresholds> {
    Arc::default()
}

/// One row: its step, its scalars (None for a null column), its MoE counts and its `extra_json`.
struct Row {
    step: i64,
    scalars: Option<Vec<(&'static str, Option<f64>)>>,
    moe: Option<Vec<i64>>,
    extra: Option<String>,
}

fn row(step: i64, scalars: &[(&'static str, f64)]) -> Row {
    Row {
        step,
        scalars: Some(scalars.iter().map(|(name, value)| (*name, Some(*value))).collect()),
        moe: None,
        extra: None,
    }
}

fn bare(step: i64) -> Row {
    Row {
        step,
        scalars: None,
        moe: None,
        extra: None,
    }
}

impl Row {
    fn imbalanced(mut self) -> Self {
        self.moe = Some(vec![1, 1, 10_000, 1]);
        self
    }

    fn extra(mut self, text: &str) -> Self {
        self.extra = Some(text.to_owned());
        self
    }
}

fn batch(rows: &[Row]) -> RecordBatch {
    let n = rows.len();
    let mut scalars = MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
    let mut moe = MapBuilder::new(None, StringBuilder::new(), ListBuilder::new(Int64Builder::new()));
    for row in rows {
        match &row.scalars {
            Some(entries) => {
                for (name, value) in entries {
                    scalars.keys().append_value(name);
                    scalars.values().append_option(*value);
                }
                scalars.append(true).unwrap();
            }
            None => scalars.append(false).unwrap(),
        }
        match &row.moe {
            Some(counts) => {
                moe.keys().append_value("r0");
                for count in counts {
                    moe.values().values().append_value(*count);
                }
                moe.values().append(true);
                moe.append(true).unwrap();
            }
            None => moe.append(false).unwrap(),
        }
    }
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("run_id", Arc::new(StringArray::from(vec!["ar1"; n]))),
        ("rank", Arc::new(Int32Array::from(vec![0; n]))),
        ("attempt", Arc::new(Int64Array::from(vec![1; n]))),
        (
            "step",
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.step))),
        ),
        ("scalars", Arc::new(scalars.finish())),
        ("moe_counts", Arc::new(moe.finish())),
        (
            "extra_json",
            Arc::new(StringArray::from(
                rows.iter().map(|r| r.extra.clone()).collect::<Vec<_>>(),
            )),
        ),
    ];
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

fn of_kind(detectors: &RankDetectors, kind: &str) -> Value {
    detectors
        .incidents()
        .into_iter()
        .find(|incident| incident["kind"] == kind)
        .unwrap_or_else(|| panic!("a {kind} incident in {:?}", detectors.incidents()))
}

/// The incident's scalar context on the raw route (`judge`) and on a live rank (`append`).
fn contexts(rows: &[Row], kind: &str) -> [Value; 2] {
    let raw = of_kind(&judge(0, &[batch(rows)], &usual()).unwrap(), kind);
    let mut live = RankDetectors::new(&usual());
    live.append(1, &batch(rows));
    let live = of_kind(&live, kind);
    [raw["scalars"].clone(), live["scalars"].clone()]
}

// --- AQ5-D4 --------------------------------------------------------------------------------

/// AQ-5's case (seed 1863): step 26 written twice with the same loss, the second writing with the
/// MoE counts that make the incident. Python: no value before the step (its first writing is not one).
#[test]
fn d4_a_moe_incident_on_a_steps_second_writing_has_no_value_before_it() {
    let rows = [row(26, &[("loss", 2.4)]), row(26, &[("loss", 2.4)]).imbalanced()];
    for got in contexts(&rows, "moe_imbalance") {
        assert_eq!(got, json!({"loss": {"value": 2.4, "median_before": null}}));
    }
}

/// The two writings log different values: the context is the first writing's value, with the median
/// of the records before that writing, as Python's `scalar_context` gives it.
#[test]
fn d4_the_context_is_the_first_writings_value_and_the_median_before_it() {
    let rows = [
        row(24, &[("loss", 1.0)]),
        row(25, &[("loss", 2.0)]),
        row(26, &[("loss", 2.4)]),
        row(26, &[("loss", 3.0)]).imbalanced(),
    ];
    for got in contexts(&rows, "moe_imbalance") {
        assert_eq!(got, json!({"loss": {"value": 2.4, "median_before": 1.5}}));
    }
}

/// Scalars on the first writing only: the incident on the second still shows them. On the second
/// only: Python's first record with the step logged nothing, so the incident shows none.
#[test]
fn d4_only_the_first_writing_logs_a_scalar() {
    let rows = [row(26, &[("loss", 2.4)]), bare(26).imbalanced()];
    for got in contexts(&rows, "moe_imbalance") {
        assert_eq!(got, json!({"loss": {"value": 2.4, "median_before": null}}));
    }
    let rows = [bare(26), row(26, &[("loss", 3.0)]).imbalanced()];
    for got in contexts(&rows, "moe_imbalance") {
        assert_eq!(got, Value::Null, "no scalars key");
    }
}

/// The step after a step written twice counts both writings among the records before it, as
/// Python's window of the previous records does.
#[test]
fn d4_a_later_step_counts_both_writings_before_it() {
    let rows = [
        row(25, &[("loss", 1.0)]),
        row(25, &[("loss", 5.0)]),
        row(26, &[("loss", 9.0)]).imbalanced(),
    ];
    for got in contexts(&rows, "moe_imbalance") {
        assert_eq!(got, json!({"loss": {"value": 9.0, "median_before": 3.0}}));
    }
}

/// A step written many times: a later writing still shows the first one's value and the median of
/// the records before it, and the next step's median is over the last `SPIKE_WINDOW` writings, as
/// Python's window of the previous records gives them.
#[test]
fn d4_a_step_written_many_times() {
    let rows = |last_imbalanced: bool| {
        let mut rows: Vec<Row> = (0..10).map(|step| row(step, &[("loss", 5.0)])).collect();
        rows.push(row(10, &[("loss", 1.0)]));
        rows.extend((2..=200).map(|at| row(10, &[("loss", f64::from(at))])));
        if last_imbalanced {
            let last = rows.pop().unwrap();
            rows.push(last.imbalanced());
        } else {
            rows.push(row(11, &[("loss", 9.0)]).imbalanced());
        }
        rows
    };
    for got in contexts(&rows(true), "moe_imbalance") {
        assert_eq!(got, json!({"loss": {"value": 1.0, "median_before": 5.0}}));
    }
    for got in contexts(&rows(false), "moe_imbalance") {
        assert_eq!(got, json!({"loss": {"value": 9.0, "median_before": 175.5}}));
    }
}

// --- AQ5-D7: a row stored before the change -------------------------------------------------

/// A row a sender built before this change: the column null, the whole map in `extra_json`. The
/// entries the column holds are read from there (a number, the sink's "nan"); a text, a bool, a
/// null and a number too large for a float are absent.
#[test]
fn d7_a_whole_map_in_extra_json_reads_as_its_held_entries() {
    let huge = format!("-1{}", "0".repeat(400));
    let text = format!(
        r#"{{"scalars":{{"acc":0.5,"flag":true,"none":null,"nan":"nan","train/loss":{huge},"words":"text"}}}}"#
    );
    let rows = [bare(115).extra(&text).imbalanced()];
    for got in contexts(&rows, "moe_imbalance") {
        assert_eq!(
            got,
            json!({"acc": {"value": 0.5, "median_before": null}, "nan": {"value": null, "median_before": null}})
        );
    }
}

/// The records before an incident read the same way: an older row's held entries count toward the
/// median before it.
#[test]
fn d7_an_older_rows_held_entries_count_before_an_incident() {
    let rows = [
        bare(24).extra(r#"{"scalars":{"loss":1.0,"words":"text"}}"#),
        bare(25).extra(r#"{"scalars":{"loss":3.0,"words":"text"}}"#),
        row(26, &[("loss", 9.0)]).imbalanced(),
    ];
    for got in contexts(&rows, "moe_imbalance") {
        assert_eq!(got, json!({"loss": {"value": 9.0, "median_before": 2.0}}));
    }
}

/// A row with a `scalars` column is read from the column alone: what `extra_json` holds under
/// `scalars` beside it (a refused entry, a bool, an integer the column holds rounded) adds nothing.
#[test]
fn d7_the_column_when_there_is_one() {
    let rows = [row(115, &[("acc", 0.5)])
        .extra(r#"{"scalars":{"words":"text","flag":true,"acc":7}}"#)
        .imbalanced()];
    for got in contexts(&rows, "moe_imbalance") {
        assert_eq!(got, json!({"acc": {"value": 0.5, "median_before": null}}));
    }
}
