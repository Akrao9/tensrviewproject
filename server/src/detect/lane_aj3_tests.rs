//! Lane AJ-3's tests (Nov 22, 2026): A8 of `OPEN_BUGS.md` section 15. The report's `moe_source`
//! says what Python's `evidence.moe_source` says for the same records: of each router, the first
//! entry that is an object, holding only its `form` and `point` (an empty object kept as it is);
//! an entry that is not an object is passed over, so a later record can still name the router.
//! `sdk/tests/test_lane_aj3_report.py` holds the same records through both engines.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Value};

use super::{judge, report, Thresholds};

/// Rows of (attempt, step, `extra_json`).
fn rows(rows: &[(i64, i64, Value)]) -> RecordBatch {
    let columns: Vec<(&str, ArrayRef)> = vec![
        (
            "run_id",
            Arc::new(StringArray::from_iter_values(rows.iter().map(|_| "run"))),
        ),
        ("rank", Arc::new(Int32Array::from_iter_values(rows.iter().map(|_| 0)))),
        (
            "attempt",
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|(attempt, ..)| *attempt))),
        ),
        (
            "step",
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|(_, step, _)| *step))),
        ),
        (
            "extra_json",
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|(.., extra)| extra.to_string()),
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

/// The raw report of one attempt's rows, each row's `moe_source` as given (steps 0, 1, ...).
fn reported(sources: &[Value]) -> Value {
    let given: Vec<(i64, i64, Value)> = sources
        .iter()
        .enumerate()
        .map(|(step, source)| (1, step as i64, json!({"moe_source": source})))
        .collect();
    report_of(&given)
}

fn report_of(given: &[(i64, i64, Value)]) -> Value {
    let detectors = judge(0, &[rows(given)], &Arc::<Thresholds>::default()).unwrap();
    report("run", 0, &detectors, 0)
}

#[test]
fn a8_only_objects_and_only_form_and_point_are_kept() {
    // The case the author named: an extra key dropped, a string dropped, an empty object kept.
    let found = reported(&[json!({"r0": {"form": "scores", "extra": 123}, "r1": "bad", "r2": {}})]);
    assert_eq!(
        found["moe_source"],
        json!({"r0": {"form": "scores"}, "r2": {}}),
        "{found}"
    );
}

#[test]
fn a8_the_first_object_wins_and_a_non_object_does_not_hold_the_router() {
    let found = reported(&[
        json!({"r1": "bad"}),
        json!({"r1": {"form": "f", "point": "dispatched", "x": 1}}),
        json!({"r1": {"form": "g", "point": "requested"}}),
    ]);
    assert_eq!(
        found["moe_source"],
        json!({"r1": {"form": "f", "point": "dispatched"}}),
        "{found}"
    );
}

#[test]
fn a8_an_empty_object_first_still_wins_over_a_later_full_one() {
    let found = reported(&[json!({"r1": {}}), json!({"r1": {"form": "f", "point": "dispatched"}})]);
    assert_eq!(found["moe_source"], json!({"r1": {}}), "{found}");
}

#[test]
fn a8_nothing_kept_leaves_the_key_out_of_the_report() {
    let found = reported(&[json!({"r1": 5, "r2": "bad", "r3": [1], "r4": null}), json!("not a map")]);
    let report = found.as_object().unwrap();
    assert!(!report.contains_key("moe_source"), "{found}");
}

#[test]
fn a8_a_superseded_row_does_not_name_the_router() {
    // Attempt 1 reaches step 3; attempt 2 resumes at step 1, so attempt 1's step 2 does not count.
    let found = report_of(&[
        (1, 0, json!({"moe_source": {"r1": "bad"}})),
        (1, 2, json!({"moe_source": {"r1": {"form": "a", "point": "requested"}}})),
        (1, 3, json!({})),
        (
            2,
            1,
            json!({"moe_source": {"r1": {"form": "b", "point": "dispatched", "y": 2}}}),
        ),
        (2, 2, json!({})),
    ]);
    assert_eq!(
        found["moe_source"],
        json!({"r1": {"form": "b", "point": "dispatched"}}),
        "{found}"
    );
}

#[test]
fn a8_a_well_formed_source_is_kept_as_written() {
    let source = json!({"router.0": {"form": "scores", "point": "unknown"}, "router.1": {"form": "indices", "point": "dispatched"}});
    let found = reported(&[
        source.clone(),
        json!({"router.0": {"form": "other", "point": "requested"}}),
    ]);
    assert_eq!(found["moe_source"], source, "{found}");
}
