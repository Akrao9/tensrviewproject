//! Lane X4's tests (Nov 19, 2026): the Rust reader of `detect/data.rs` given the SDK's `v3:` batch
//! hashes (C16, open part). The reader needs no change for them (a hash's kind is the text before
//! its first `:`); these hold it to the Python reader's verdicts on the hashes the SDK now writes.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Value};

use super::{detect_run, judge, Thresholds};

/// `tensorquill.batch_hash` at lane X4's commit: `[1, 1]` (a `v2` batch the fix leaves as it
/// was, and what the `v2` writer wrote for `[1, True]`), `[1, True]` (now `v3`), and three others.
const OLD_MIXED: &str = "v2:bb153dc27acf9bbf";
const NEW_MIXED: &str = "v3:a8d6c9c56c1210a5";
const HEAD: [&str; 3] = ["v2:bb662acaf71e051a", "v2:97c1f05b13600cd6", "v2:5d4875daabc8de23"];
const KINDS_DIFFER: &str = "the batch hashes before and after the resume are of different kinds";

/// One rank's rows as (attempt, step, batch hash), each drawing one batch.
fn rows(rank: i32, rows: &[(i64, i64, &str)]) -> RecordBatch {
    let columns: Vec<(&str, ArrayRef)> = vec![
        (
            "run_id",
            Arc::new(StringArray::from_iter_values(rows.iter().map(|_| "x4"))),
        ),
        (
            "rank",
            Arc::new(Int32Array::from_iter_values(rows.iter().map(|_| rank))),
        ),
        (
            "attempt",
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0))),
        ),
        ("step", Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.1)))),
        (
            "data_json",
            Arc::new(StringArray::from_iter_values(
                rows.iter()
                    .map(|r| json!({"batches": 1, "batch_hash": r.2}).to_string()),
            )),
        ),
    ];
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
        .collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns.into_iter().map(|(_, a)| a).collect(),
    )
    .unwrap()
}

/// Attempt 0 draws `HEAD` and then `at_step_3`; attempt 1 resumes at step 3 and draws `[1, True]`
/// with this writer. A `v2` hash there (the old writer's `[1, True]`) is not compared; a `v3` one is,
/// and matches.
#[test]
fn c16_a_resume_across_the_mixed_batch_fix_is_not_compared_and_one_after_it_is() {
    for (at_step_3, compared) in [(OLD_MIXED, false), (NEW_MIXED, true)] {
        let batch = rows(
            0,
            &[
                (0, 0, HEAD[0]),
                (0, 1, HEAD[1]),
                (0, 2, HEAD[2]),
                (0, 3, at_step_3),
                (1, 3, NEW_MIXED),
            ],
        );
        let detectors = judge(0, &[batch], &Arc::<Thresholds>::default()).unwrap();
        let incidents = detectors.incidents();
        assert!(
            !incidents.iter().any(|i| i["kind"] == "resume_data_mismatch"),
            "{at_step_3}: {incidents:?}"
        );
        let skipped: Vec<Value> = detectors
            .not_checked()
            .into_iter()
            .filter(|i| i["check"] == "resume_data")
            .collect();
        if compared {
            assert!(skipped.is_empty(), "{skipped:?}");
        } else {
            assert_eq!(skipped.len(), 1, "{skipped:?}");
            assert_eq!(skipped[0]["reason"], KINDS_DIFFER);
        }
    }
}

/// Two ranks drawing the same `v3` batches are a duplicate; a `v2` rank and a `v3` rank are not
/// compared.
#[test]
fn c16_v3_hashes_are_compared_only_with_v3_hashes_across_ranks() {
    fn same(hash: &str) -> [(i64, i64, &str); 4] {
        [(0, 0, hash), (0, 1, hash), (0, 2, hash), (0, 3, hash)]
    }
    let both_v3 = [rows(0, &same(NEW_MIXED)), rows(1, &same(NEW_MIXED))];
    let run = detect_run(&both_v3, &Arc::<Thresholds>::default()).unwrap();
    assert_eq!(run["ranks_compared"], json!([0, 1]));
    assert!(run["incidents"]
        .as_array()
        .unwrap()
        .iter()
        .any(|i| i["kind"] == "duplicate_batches"));
    let two_kinds = [rows(0, &same(OLD_MIXED)), rows(1, &same(NEW_MIXED))];
    let run = detect_run(&two_kinds, &Arc::<Thresholds>::default()).unwrap();
    assert_eq!(run["ranks_compared"], json!([]));
    assert!(!run["incidents"]
        .as_array()
        .unwrap()
        .iter()
        .any(|i| i["kind"] == "duplicate_batches"));
}
