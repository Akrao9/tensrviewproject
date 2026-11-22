//! Lane D of wave AF (Nov 21, 2026): the offline `detect` and `detect --run` against section 1 of
//! `WAVE_AF_CONTRACT.md` (a record's identity fields), and section 5 (the offline report against
//! the live one). The binary's own exit status and line are in `sdk/tests/test_lane_af_d_detect.py`.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Builder, Int32Array, Int64Array, MapBuilder, StringArray, StringBuilder};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;

use super::{detect_run, detect_stream, Thresholds};

fn usual() -> Arc<Thresholds> {
    Arc::default()
}

/// One row per entry of `rows` (rank, attempt, step, schema_version), every column nullable, with
/// a `train/loss` scalar.
fn batch(rows: &[(Option<i32>, Option<i64>, Option<i64>, Option<i32>)]) -> RecordBatch {
    let mut scalars = MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
    for _ in rows {
        scalars.keys().append_value("train/loss");
        scalars.values().append_value(1.0);
        scalars.append(true).unwrap();
    }
    let columns: Vec<(&str, ArrayRef)> = vec![
        (
            "schema_version",
            Arc::new(rows.iter().map(|row| row.3).collect::<Int32Array>()),
        ),
        (
            "run_id",
            Arc::new(StringArray::from_iter_values(rows.iter().map(|_| "r"))),
        ),
        ("rank", Arc::new(rows.iter().map(|row| row.0).collect::<Int32Array>())),
        (
            "attempt",
            Arc::new(rows.iter().map(|row| row.1).collect::<Int64Array>()),
        ),
        ("step", Arc::new(rows.iter().map(|row| row.2).collect::<Int64Array>())),
        ("scalars", Arc::new(scalars.finish())),
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

fn good(step: i64) -> (Option<i32>, Option<i64>, Option<i64>, Option<i32>) {
    (Some(0), Some(0), Some(step), Some(1))
}

fn stream(batches: &[RecordBatch]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut writer = arrow::ipc::writer::StreamWriter::try_new(&mut out, &batches[0].schema()).unwrap();
    for batch in batches {
        writer.write(batch).unwrap();
    }
    writer.finish().unwrap();
    drop(writer);
    out
}

/// Both offline modes' answer to `batches`: `detect` (one rank) and `detect --run`.
fn both(batches: &[RecordBatch]) -> (Result<serde_json::Value, String>, Result<serde_json::Value, String>) {
    (
        detect_stream(stream(batches).as_slice(), &usual()),
        detect_run(batches, &usual()),
    )
}

/// Section 1: a negative `rank`, `attempt` or `step`, and a `schema_version` that is not 0 or 1,
/// refuse the input in both modes with one line that names the column, the value and the row
/// (counted across the batches given). Fails on the base (`1ca26d4`): each was judged, a report of
/// one step (and for the rank, `rank: -3`).
#[test]
fn a_negative_identity_value_or_an_unsupported_version_refuses_the_input_naming_column_value_and_row() {
    for (row, name, value) in [
        ((Some(-3), Some(0), Some(1), Some(1)), "rank", "-3"),
        ((Some(0), Some(-3), Some(1), Some(1)), "attempt", "-3"),
        ((Some(0), Some(0), Some(-3), Some(1)), "step", "-3"),
        ((Some(0), Some(0), Some(1), Some(99)), "schema_version", "99"),
        ((Some(0), Some(0), Some(1), Some(-3)), "schema_version", "-3"),
        ((Some(0), Some(0), Some(1), Some(2)), "schema_version", "2"),
    ] {
        // One record alone, as the case of `cases.json` gives it.
        let (one, run) = both(&[batch(&[row])]);
        for (mode, answer) in [("detect", one), ("detect --run", run)] {
            let error = answer.expect_err(&format!("{mode} judged {name} {value}"));
            assert!(
                error.starts_with(&format!("row 1: {name} is {value}")),
                "{mode}, {name} {value}: {error}"
            );
            assert!(!error.contains('\n'), "one line: {error}");
            if name == "schema_version" {
                assert!(error.contains("versions 0 and 1"), "{error}");
            }
        }
    }
}

/// The row named is the row's place in the whole input, counted across its batches.
#[test]
fn the_row_named_is_counted_across_the_batches() {
    let batches = [
        batch(&[good(0), good(1)]),
        batch(&[(Some(0), Some(0), Some(-3), Some(1)), good(3)]),
    ];
    let (one, run) = both(&batches);
    for answer in [one, run] {
        assert!(answer.unwrap_err().starts_with("row 3: step is -3"));
    }
}

/// A null in a nullable key column names its row and column. `schema_version` is among the columns
/// checked for a null, which the base did not check (a null version was judged).
#[test]
fn a_null_identity_value_names_its_row_and_column() {
    for (row, name) in [
        ((None, Some(0), Some(1), Some(1)), "rank"),
        ((Some(0), None, Some(1), Some(1)), "attempt"),
        ((Some(0), Some(0), None, Some(1)), "step"),
        ((Some(0), Some(0), Some(1), None), "schema_version"),
    ] {
        let batches = [batch(&[good(0), row])];
        let (one, run) = both(&batches);
        for (mode, answer) in [("detect", one), ("detect --run", run)] {
            let error = answer.expect_err(&format!("{mode} judged a null {name}"));
            assert!(error.starts_with(&format!("row 2: {name} is null")), "{mode}: {error}");
        }
    }
}

/// Pins: valid values read as before. Version 0 (what `flight.py` writes for a record that names
/// none) and 1 are read; a stream with no `schema_version` column is read; rank, attempt and step
/// of 0 are read.
#[test]
fn valid_identity_values_are_judged_as_before() {
    let batches = [batch(&[(Some(0), Some(0), Some(0), Some(0)), good(1), good(2)])];
    let (one, run) = both(&batches);
    let one = one.unwrap();
    assert_eq!(
        (&one["rank"], &one["steps"]),
        (&serde_json::json!(0), &serde_json::json!(3))
    );
    assert_eq!(run.unwrap()["reports"][0]["steps"], 3);
    let without = batches[0].project(&[1, 2, 3, 4, 5]).unwrap();
    assert!(without.column_by_name("schema_version").is_none());
    let (one, run) = both(&[without]);
    assert_eq!(one.unwrap()["steps"], 3);
    assert_eq!(run.unwrap()["reports"][0]["steps"], 3);
}
