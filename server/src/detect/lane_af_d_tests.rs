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

/// A row's identity fields: (rank, attempt, step, schema_version), each of which may be null.
type Identity = (Option<i32>, Option<i64>, Option<i64>, Option<i32>);

/// One row per entry of `rows`, every column nullable, with a `train/loss` scalar.
fn batch(rows: &[Identity]) -> RecordBatch {
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

fn good(step: i64) -> Identity {
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

// --- section 6: the cross-rank state's maps -------------------------------------------------------

mod replica_maps {
    use serde_json::{json, Value};

    use super::super::replica::{record_from_json, Bounds, Comparison};

    const LAUNCH: &str = "L";
    const LISTED: usize = 8;
    const STEPS: i64 = 400;

    fn observation(rank: i64, attempt: i64, step: i64, w: f64) -> Value {
        json!({"run_id": "r1", "rank": rank, "attempt": attempt, "step": step,
            "replica": {"v": 1, "launch": LAUNCH, "group": [0, 1, 2], "device": "cuda",
                "projection": "name-v1", "signature": "fold64-v1",
                "params": {"w": {"shape": [8, 4], "dtype": "float32", "fp": [w, 1.0, 0.5],
                    "sig": if w == 3.5 { "7" } else { "8" }}}}})
    }

    /// A long run of three ranks, `STEPS` steps, with the listing bound at `LISTED` steps: rank 1
    /// drifts every 50th step; near the end an earlier attempt of rank 1 (its backlog, first in its
    /// order) arrives for the last steps and takes the launch, so the steps it has rows at are judged
    /// again. Returns the comparison after each step.
    fn long_run(mut each: impl FnMut(i64, &Comparison)) -> Comparison {
        let bounds = Bounds {
            listed_steps: LISTED,
            ..Bounds::default()
        };
        let mut comparison = Comparison::new(1e-6, bounds);
        // Rank 1's attempt 5 comes before its attempt 11.
        let place = |rank: i64, attempt: i64| Some(if rank == 1 && attempt == 11 { 1 } else { 0 });
        for step in 0..STEPS {
            for rank in 0..3 {
                let w = if rank == 1 && step % 50 == 0 { 3.6 } else { 3.5 };
                comparison.insert(
                    record_from_json(&observation(rank, 10 + rank, step, w), 1e-6).unwrap(),
                    &place,
                );
            }
            if step >= STEPS - 4 {
                comparison.insert(record_from_json(&observation(1, 5, step, 3.5), 1e-6).unwrap(), &place);
            }
            each(step, &comparison);
        }
        comparison
    }

    /// RS-05: `rows_at` ((launch, rank) to the steps it has rows at, read only to judge again the
    /// listed steps of a launch whose owner changes) is pruned by the rule the comparison's steps are
    /// pruned by: a step forgotten past the listing bound leaves it. Its size stays within the listed
    /// steps; on the base (`1ca26d4`) it held every step of the run, one entry per launch, rank and
    /// step (1200 after 400 steps of 3 ranks; the backlog's rows are at steps rank 1 already has). `owners` and `forgotten` are not
    /// pruned (no rule to follow; the lane's notes): their sizes are asserted as they are.
    #[test]
    fn rows_at_stays_within_the_listed_steps_over_a_long_run() {
        let mut largest = 0;
        let comparison = long_run(|_, comparison| largest = largest.max(comparison.map_sizes().0));
        let (rows_at, owners, forgotten) = comparison.map_sizes();
        // At most the listed steps, for each launch and rank, and the step just being taken.
        assert!(
            largest <= (LISTED + 1) * 3 + 4,
            "rows_at held {largest} entries at most"
        );
        assert!(rows_at <= LISTED * 3 + 4, "rows_at holds {rows_at}");
        // One owner per launch and rank, whatever the length.
        assert_eq!(owners, 3);
        // One entry per step forgotten: kept for the run's length, by design (a late row is counted).
        assert_eq!(forgotten, STEPS as usize - LISTED);
    }

    /// The verdicts are the base's: the result of the same rows equals what `1ca26d4` gave, kept in
    /// `lane_af_d_replica_base.json` (written by this test on the base with `LANE_AF_D_WRITE_BASE` set).
    #[test]
    fn the_result_of_a_long_run_is_the_bases() {
        let comparison = long_run(|_, _| {});
        let result = comparison.result();
        let status = comparison.status_object();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/detect/lane_af_d_replica_base.json");
        if std::env::var_os("LANE_AF_D_WRITE_BASE").is_some() {
            std::fs::write(
                &path,
                serde_json::to_string_pretty(&json!({"result": result, "status": status})).unwrap(),
            )
            .unwrap();
        }
        let base: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(result, base["result"]);
        assert_eq!(status, base["status"]);
        // The case judges something: the drifted steps still listed, and the incidents kept.
        assert!(!result["replica_drift"].as_array().unwrap().is_empty(), "{result}");
    }
}
