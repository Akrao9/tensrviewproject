//! Lane AJ-3's tests of the cross-rank reader (Nov 22, 2026): which `replica_spread` entries a
//! gathered spread covers, as Python's `_spread_values` and `_held_spread_table` read them on lane
//! AJ-2's f9a7dff (`WAVE_AJ_CONTRACT.md` section 8). An entry is read when it is a non-empty list
//! of numbers (not bools), nulls (not finite) or exactly the sink's "nan", "inf" and "-inf"; any
//! other entry, `[]` among them, says nothing of its parameter: it is neither covered nor over, so
//! it cannot set aside a drift the observations show. A non-empty table whose every entry says
//! nothing is no spread; an empty table is a spread that covers nothing.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Builder, Int64Array, ListBuilder, MapBuilder, StringArray, StringBuilder};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Value};

use super::{record_from_json, records_from_batch, Spread};

const TOLERANCE: f64 = 1e-6;

fn names(set: &BTreeSet<String>) -> Vec<&str> {
    set.iter().map(String::as_str).collect()
}

fn read(table: Value) -> Option<Spread> {
    record_from_json(&json!({"rank": 0, "step": 0, "replica_spread": table}), TOLERANCE)?.spread
}

#[test]
fn only_a_readable_non_empty_entry_is_covered() {
    let spread = read(json!({
        "ok": [0.0, 0.0, 0.0],
        "drift": [0.01, 0.0, 0.0],
        "nan_text": ["nan", 0.0, 0.0],
        "inf_text": ["inf", 0.0, 0.0],
        "null": [null, 0.0, 0.0],
        "empty": [],
        "text": "i",
        "a_bool": [true, 0.0, 0.0],
        "other_text": ["1.5", 0.0, 0.0],
        "spelled": ["NaN", 0.0, 0.0],
        "nested": [[0.0]],
        "a_map": {"x": 1},
    }))
    .unwrap();
    assert_eq!(names(&spread.covered), ["drift", "inf_text", "nan_text", "null", "ok"]);
    assert_eq!(names(&spread.over), ["drift", "inf_text", "nan_text", "null"]);
}

#[test]
fn a_table_of_unreadable_entries_is_no_spread_and_an_empty_one_is() {
    assert!(read(json!({"w": "invalid"})).is_none());
    assert!(read(json!({"w": [true], "v": "x"})).is_none());
    let only_empty = read(json!({"w": []})).unwrap();
    assert!(only_empty.covered.is_empty() && only_empty.over.is_empty());
    let empty = read(json!({})).unwrap();
    assert!(empty.covered.is_empty() && empty.over.is_empty());
}

/// One row of rank 0, step 0: the `replica_spread` column (None: null) and `extra_json` text.
fn row(column: Option<Vec<(&str, Vec<f64>)>>, extra: Option<&str>) -> RecordBatch {
    let mut spreads = MapBuilder::new(None, StringBuilder::new(), ListBuilder::new(Float64Builder::new()));
    match column {
        Some(entries) => {
            for (name, values) in entries {
                spreads.keys().append_value(name);
                for value in values {
                    spreads.values().values().append_value(value);
                }
                spreads.values().append(true);
            }
            spreads.append(true).unwrap();
        }
        None => spreads.append(false).unwrap(),
    }
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("run_id", Arc::new(StringArray::from(vec!["run"]))),
        ("attempt", Arc::new(Int64Array::from(vec![1]))),
        ("step", Arc::new(Int64Array::from(vec![0]))),
        ("replica_spread", Arc::new(spreads.finish())),
        ("extra_json", Arc::new(StringArray::from(vec![extra]))),
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

#[test]
fn a_held_empty_entry_in_the_column_covers_nothing() {
    let records = records_from_batch(
        &row(Some(vec![("w", vec![]), ("b", vec![0.0, 0.0, 0.0])]), None),
        0,
        TOLERANCE,
    );
    let spread = records[0].spread.as_ref().unwrap();
    assert_eq!(names(&spread.covered), ["b"]);
    assert!(spread.over.is_empty());
}

#[test]
fn the_extra_json_fallback_reads_entries_as_the_column_would() {
    // The column is null (no entry held), so the table comes from `extra_json`.
    let extra = json!({"replica_spread": {"w": "invalid", "v": [0.01, 0.0, 0.0]}}).to_string();
    let records = records_from_batch(&row(None, Some(&extra)), 0, TOLERANCE);
    let spread = records[0].spread.as_ref().unwrap();
    assert_eq!(names(&spread.covered), ["v"]);
    assert_eq!(names(&spread.over), ["v"]);
    let refused = json!({"replica_spread": {"w": "invalid"}}).to_string();
    assert!(records_from_batch(&row(None, Some(&refused)), 0, TOLERANCE).is_empty());
}

#[test]
#[ignore = "open, not fixed in lane AJ-3: serde_json refuses the whole extra_json text over an integer \
            too large for a float, so the row's observation is lost with the entry (reported to the coordinator)"]
fn an_integer_too_large_for_a_float_leaves_only_its_entry_out() {
    // Python writes 10**400 in full; it is no float, so its entry should say nothing and the rest
    // of the row (here the observation beside it) still be read. Today no part of the row is read.
    let big = format!("1{}", "0".repeat(400));
    let extra = format!(
        r#"{{"replica_spread": {{"big": [{big}, 0.0, 0.0], "v": [0.01, 0.0, 0.0]}}, "replica": {{"launch": "l", "group": [0, 1], "w": "00"}}}}"#
    );
    let records = records_from_batch(&row(None, Some(&extra)), 0, TOLERANCE);
    assert_eq!(records.len(), 1, "the row was not read");
    let spread = records[0].spread.as_ref().unwrap();
    assert_eq!(names(&spread.covered), ["v"]);
    assert!(records[0].observed.is_some());
}
