//! Lane V7's tests (Nov 19, 2026): one test per row of `OPEN_BUGS.md` this lane fixed in the
//! detectors, each failing on the lane's base commit (66b52d7).

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Builder, Int32Array, Int64Array, MapBuilder, StringArray, StringBuilder};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Value};

use super::{detect_run, detect_stream, judge, RankDetectors, Thresholds};

fn usual() -> Arc<Thresholds> {
    Arc::default()
}

/// A batch of the given columns, every one nullable.
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

fn keys(steps: &[i64]) -> Vec<(&'static str, ArrayRef)> {
    vec![
        (
            "run_id",
            Arc::new(StringArray::from_iter_values(steps.iter().map(|_| "run"))) as ArrayRef,
        ),
        ("rank", Arc::new(Int32Array::from_iter_values(steps.iter().map(|_| 0)))),
        (
            "attempt",
            Arc::new(Int64Array::from_iter_values(steps.iter().map(|_| 1))),
        ),
        ("step", Arc::new(Int64Array::from_iter_values(steps.iter().copied()))),
    ]
}

/// One `train/loss` scalar per row.
fn losses(values: &[f64]) -> ArrayRef {
    let mut scalars = MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
    for value in values {
        scalars.keys().append_value("train/loss");
        scalars.values().append_value(*value);
        scalars.append(true).unwrap();
    }
    Arc::new(scalars.finish())
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

fn repeated(detectors: &RankDetectors) -> Value {
    detectors
        .incidents()
        .into_iter()
        .find(|incident| incident["kind"] == "repeated_steps")
        .expect("a repeated_steps incident")
}

// --- RS12 / RUST-06 ------------------------------------------------------------------------

/// RS12: a step written again within one attempt shows the scalars of its first writing, as
/// Python's `scalar_context` gives them (the first record with that step, and the median of
/// the records before it).
#[test]
fn rs12_a_repeated_step_shows_what_its_first_writing_logged() {
    // The audit's case: losses 1 then 9 at the first step. Python: 1, and no median before it.
    let steps = [0, 0, 1, 2];
    let mut columns = keys(&steps);
    columns.push(("scalars", losses(&[1.0, 9.0, 3.0, 3.0])));
    let found = repeated(&judge(0, &[batch(columns)], &usual()).unwrap());
    assert_eq!(
        found["scalars"],
        json!({"train/loss": {"value": 1.0, "median_before": null}}),
        "{found}"
    );

    // T6's case (rust06.py): steps 0..10 at 2.0, step 5 again at 7.0 and a third time at 9.0.
    let mut steps: Vec<i64> = (0..10).collect();
    steps.extend([5, 5]);
    steps.extend(10..20);
    let mut values = vec![2.0; 10];
    values.extend([7.0, 9.0]);
    values.extend([2.0; 10]);
    let mut columns = keys(&steps);
    columns.push(("scalars", losses(&values)));
    let found = repeated(&judge(0, &[batch(columns)], &usual()).unwrap());
    assert_eq!(found["step"], 5);
    assert_eq!(
        found["scalars"],
        json!({"train/loss": {"value": 2.0, "median_before": 2.0}}),
        "{found}"
    );

    // A lower step repeated later: its own first writing, not the row that repeated it.
    let mut steps: Vec<i64> = (0..10).collect();
    steps.extend([5, 3]);
    let mut values: Vec<f64> = (0..10).map(|step| step as f64).collect();
    values.extend([50.0, 30.0]);
    let mut columns = keys(&steps);
    columns.push(("scalars", losses(&values)));
    let found = repeated(&judge(0, &[batch(columns)], &usual()).unwrap());
    assert_eq!(found["step"], 3);
    assert_eq!(
        found["scalars"],
        json!({"train/loss": {"value": 3.0, "median_before": 1.0}}),
        "{found}"
    );
}

/// RS12: a step an attempt restarting the numbering writes again shows that attempt's first
/// writing of it, the only one that counts, also when that attempt writes it a third time.
#[test]
fn rs12_a_restarted_step_keeps_its_first_counted_writing() {
    let first = {
        let steps: Vec<i64> = (0..5).collect();
        let mut columns = keys(&steps);
        columns.push(("scalars", losses(&[1.0; 5])));
        batch(columns)
    };
    let restart = {
        let steps = [0, 0, 1];
        let mut columns = keys(&steps);
        columns[2].1 = Arc::new(Int64Array::from(vec![2, 2, 2]));
        columns.push(("scalars", losses(&[4.0, 5.0, 6.0])));
        batch(columns)
    };
    let found = repeated(&judge(0, &[first, restart], &usual()).unwrap());
    assert_eq!(found["step"], 0);
    assert_eq!(
        found["scalars"],
        json!({"train/loss": {"value": 4.0, "median_before": null}}),
        "{found}"
    );
}

// --- RS11 ----------------------------------------------------------------------------------

/// RS11: only a `checkpoint_json` that is a JSON object counts as a checkpoint for
/// `resume_incomplete`: Python counts `isinstance(record["checkpoint"], Mapping)`.
#[test]
fn rs11_only_a_checkpoint_that_parses_to_an_object_counts() {
    let steps = [0, 1, 2, 3, 4];
    let mut columns = keys(&steps);
    columns.push((
        "checkpoint_json",
        Arc::new(StringArray::from(vec![
            Some("{ broken"),
            Some("{}"),
            Some("[1]"),
            None,
            Some("\"{ not an object\""),
        ])),
    ));
    let detectors = judge(0, &[batch(columns)], &usual()).unwrap();
    let checks = detectors.kind_checks();
    assert_eq!(
        checks["resume_incomplete"]["records"], 1,
        "{}",
        checks["resume_incomplete"]
    );
}

// --- RS16 / RUST-14 ------------------------------------------------------------------------

/// RS16: the largest int64 step is a step like any other, not "no cutoff".
#[test]
fn rs16_the_largest_step_is_read() {
    let detectors = judge(0, &[batch(keys(&[0, i64::MAX]))], &usual()).unwrap();
    assert_eq!((detectors.rows(), detectors.last_step()), (2, Some(i64::MAX)));
    let report = detect_stream(&stream(&[batch(keys(&[i64::MAX]))])[..], &usual()).unwrap();
    assert_eq!((&report["steps"], &report["last_step"]), (&json!(1), &json!(i64::MAX)));
    let run = detect_run(&[batch(keys(&[i64::MAX]))], &usual()).unwrap();
    assert_eq!(run["reports"][0]["steps"], 1);
    // An attempt that starts at the largest step cuts the one before it there, and itself nowhere.
    let earlier = batch(keys(&[0, 1]));
    let mut later = keys(&[i64::MAX]);
    later[2].1 = Arc::new(Int64Array::from(vec![2]));
    let detectors = judge(0, &[earlier, batch(later)], &usual()).unwrap();
    assert_eq!((detectors.rows(), detectors.last_step()), (3, Some(i64::MAX)));
    let mut observed = RankDetectors::new(&usual());
    observed.observe(&batch(keys(&[i64::MAX])));
    assert_eq!(observed.rows(), 1);
}

// --- RS10 / RUST-11 ------------------------------------------------------------------------

/// RS10: both offline modes refuse a null run, rank, step or attempt, and an attempt column
/// that is not int64, one field wrong at a time. An attempt column left out is attempt 0.
#[test]
fn rs10_the_offline_modes_refuse_null_or_mistyped_keys() {
    let good = || keys(&[0, 1]);
    let mut cases: Vec<(&str, Vec<(&str, ArrayRef)>)> = Vec::new();
    let mut null_run = good();
    null_run[0].1 = Arc::new(StringArray::from(vec![Some("run"), None]));
    cases.push(("run_id", null_run));
    let mut null_rank = good();
    null_rank[1].1 = Arc::new(Int32Array::from(vec![Some(0), None]));
    cases.push(("rank", null_rank));
    let mut null_attempt = good();
    null_attempt[2].1 = Arc::new(Int64Array::from(vec![Some(1), None]));
    cases.push(("attempt", null_attempt));
    let mut text_attempt = good();
    text_attempt[2].1 = Arc::new(StringArray::from(vec!["1", "1"]));
    cases.push(("attempt", text_attempt));
    let mut null_step = good();
    null_step[3].1 = Arc::new(Int64Array::from(vec![Some(0), None]));
    cases.push(("step", null_step));
    for (field, columns) in cases {
        let rows = batch(columns);
        let error = detect_stream(&stream(std::slice::from_ref(&rows))[..], &usual())
            .err()
            .unwrap_or_else(|| panic!("detect accepted a wrong {field}"));
        assert!(error.contains(field), "detect, {field}: {error}");
        let error = detect_run(std::slice::from_ref(&rows), &usual())
            .err()
            .unwrap_or_else(|| panic!("detect --run accepted a wrong {field}"));
        assert!(error.contains(field), "detect --run, {field}: {error}");
    }
    // A sender from before the attempt column: attempt 0, in both modes.
    let mut legacy = good();
    legacy.remove(2);
    let rows = batch(legacy);
    let report = detect_stream(&stream(std::slice::from_ref(&rows))[..], &usual()).unwrap();
    assert_eq!((&report["attempt"], &report["steps"]), (&json!(0), &json!(2)));
    let run = detect_run(&[rows], &usual()).unwrap();
    assert_eq!(run["reports"][0]["attempt"], 0);
}

// --- C16, the Rust reader --------------------------------------------------------------------

/// The shared fixture of lane V2 (`sdk/tests/fixtures/c16-batch-hash-versions.json`, a
/// byte-identical copy of the one at V2's commit 59222b0).
const C16_FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sdk/tests/fixtures/c16-batch-hash-versions.json"
));

/// The reason both engines give when a resume's batch hash has no earlier hash of its kind.
const C16_REASON: &str = "the batch hashes before and after the resume are of different kinds";

/// A fixture case's records as one batch per rank, in the order given (grouped by attempt).
fn c16_batches(records: &[Value]) -> std::collections::BTreeMap<i64, RecordBatch> {
    let mut by_rank: std::collections::BTreeMap<i64, Vec<&Value>> = Default::default();
    for record in records {
        by_rank
            .entry(record["rank"].as_i64().unwrap())
            .or_default()
            .push(record);
    }
    by_rank
        .into_iter()
        .map(|(rank, rows)| {
            let columns: Vec<(&str, ArrayRef)> = vec![
                (
                    "run_id",
                    Arc::new(StringArray::from_iter_values(
                        rows.iter().map(|row| row["run_id"].as_str().unwrap()),
                    )),
                ),
                (
                    "rank",
                    Arc::new(Int32Array::from_iter_values(
                        rows.iter().map(|row| row["rank"].as_i64().unwrap() as i32),
                    )),
                ),
                (
                    "attempt",
                    Arc::new(Int64Array::from_iter_values(
                        rows.iter().map(|row| row["attempt"].as_i64().unwrap()),
                    )),
                ),
                (
                    "step",
                    Arc::new(Int64Array::from_iter_values(
                        rows.iter().map(|row| row["step"].as_i64().unwrap()),
                    )),
                ),
                (
                    "data_json",
                    Arc::new(StringArray::from_iter_values(
                        rows.iter().map(|row| row["data"].to_string()),
                    )),
                ),
            ];
            (rank, batch(columns))
        })
        .collect()
}

/// C16: every case of the shared fixture gets its verdict. Hashes of two kinds are never a
/// match and never a difference; a resume between them is "not compared", in `not_checked`.
#[test]
fn c16_hashes_of_two_kinds_are_not_compared() {
    let fixture: Value = serde_json::from_str(C16_FIXTURE).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let detector = case["detector"].as_str().unwrap();
        let batches = c16_batches(case["records"].as_array().unwrap());
        let (incidents, not_checked): (Vec<Value>, Vec<Value>) = if detector == "duplicate_batches" {
            let all: Vec<RecordBatch> = batches.values().cloned().collect();
            let run = detect_run(&all, &usual()).unwrap();
            // `compared`: both ranks are listed as compared, or neither is.
            let listed = if case["compared"] == true {
                json!([0, 1])
            } else {
                json!([])
            };
            assert_eq!(run["ranks_compared"], listed, "{name}");
            (run["incidents"].as_array().unwrap().clone(), Vec::new())
        } else {
            assert_eq!(batches.len(), 1, "{name}");
            let (rank, rows) = batches.iter().next().unwrap();
            let detectors = judge(*rank as i32, std::slice::from_ref(rows), &usual()).unwrap();
            (detectors.incidents(), detectors.not_checked())
        };
        let found: Vec<&Value> = incidents.iter().filter(|i| i["kind"] == detector).collect();
        match case["incident"].as_object() {
            None => assert!(found.is_empty(), "{name}: {found:?}"),
            Some(expected) => {
                assert_eq!(found.len(), 1, "{name}: {incidents:?}");
                for (key, value) in expected {
                    let got = if key == "finding" {
                        &found[0]["changes"][0]["finding"]
                    } else {
                        &found[0][key]
                    };
                    assert_eq!(got, value, "{name}: {key} in {}", found[0]);
                }
            }
        }
        if detector == "resume_data_mismatch" {
            let skipped: Vec<&Value> = not_checked.iter().filter(|i| i["check"] == "resume_data").collect();
            if case["compared"] == false {
                assert_eq!(skipped.len(), 1, "{name}: {not_checked:?}");
                assert_eq!(skipped[0]["reason"], C16_REASON, "{name}");
                assert_eq!(skipped[0]["step"], 3, "{name}");
            } else {
                assert!(skipped.is_empty(), "{name}: {skipped:?}");
            }
        }
    }
}

/// One rank's rows as (step, batch hash or none), each drawing one batch.
fn hashed(rows: &[(i64, Option<&str>)]) -> RecordBatch {
    let steps: Vec<i64> = rows.iter().map(|(step, _)| *step).collect();
    let mut columns = keys(&steps);
    columns.push((
        "data_json",
        Arc::new(StringArray::from_iter_values(rows.iter().map(|(_, hash)| match hash {
            Some(hash) => json!({"batches": 1, "batch_hash": hash}).to_string(),
            None => json!({"batches": 1}).to_string(),
        }))),
    ));
    batch(columns)
}

/// C16, `data_stalled`: a hash of another kind than the one before it is treated exactly as a
/// missing hash, so the row after it is not compared with it either.
#[test]
fn c16_a_change_of_hash_kind_stalls_as_a_missing_hash_does() {
    let (legacy, new) = ("5330bfb822ed0aaf", "v2:cd09ce7a08834a9e");
    let stall = |at_one: Option<&str>| {
        let mut rows = vec![(0, Some(legacy)), (1, at_one)];
        rows.extend((2..7).map(|step| (step, Some(new))));
        let detectors = judge(0, &[hashed(&rows)], &usual()).unwrap();
        detectors
            .incidents()
            .into_iter()
            .find(|incident| incident["kind"] == "data_stalled")
    };
    let missing = stall(None).expect("a stall after the missing hash");
    assert_eq!((&missing["step"], &missing["through_step"]), (&json!(3), &json!(5)));
    assert_eq!(stall(Some(new)), Some(missing));
}

/// C16, `ranks_compared`: with two or more hashed ranks, a rank is listed only when another
/// rank wrote a hash of a kind it wrote too; a lone hashed rank is listed as always.
#[test]
fn c16_ranks_compared_lists_ranks_with_a_kind_in_common() {
    use std::collections::BTreeMap;
    let window = |hashes: &[&str]| -> BTreeMap<i64, (Option<i64>, String)> {
        hashes
            .iter()
            .enumerate()
            .map(|(step, hash)| (step as i64, (None, hash.to_string())))
            .collect()
    };
    let (a, b) = ("5330bfb822ed0aaf", "v2:cd09ce7a08834a9e");
    let ranks = BTreeMap::from([(0, window(&[a])), (1, window(&[b])), (2, window(&[]))]);
    assert_eq!(super::ranks_compared(&ranks), Vec::<i32>::new());
    let ranks = BTreeMap::from([(0, window(&[a])), (1, window(&[b])), (2, window(&[a, b]))]);
    assert_eq!(super::ranks_compared(&ranks), vec![0, 1, 2]);
    let ranks = BTreeMap::from([(0, window(&[b])), (1, window(&[a])), (3, window(&[b]))]);
    assert_eq!(super::ranks_compared(&ranks), vec![0, 3]);
    let ranks = BTreeMap::from([(0, window(&[b])), (1, window(&[b])), (2, window(&[a]))]);
    assert_eq!(super::ranks_compared(&ranks), vec![0, 1]);
    // One rank with hashes, beside one without: listed, as before C16.
    let ranks = BTreeMap::from([(0, window(&[a])), (1, window(&[]))]);
    assert_eq!(super::ranks_compared(&ranks), vec![0]);
    assert_eq!(super::ranks_compared(&BTreeMap::new()), Vec::<i32>::new());
}
