//! Lane AC2 of wave AC (Nov 21, 2026): the server's side of `CHECKPOINT_TENSOR_CONTRACT.md`,
//! section 4 (TrainCheck's B14). The SDK compares a checkpoint's tensors with the model's at a
//! load and writes `tensors` into the load's manifest; the server only reads it, for two findings
//! of `resume_incomplete` (`missing_tensors`, `shape_differs`) and the rank's `checkpoint_tensors`.
//! The cases are the 17 with `"group": "records"` in `sdk/tests/fixtures/checkpoint/cases.json`.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Float64Builder, Int32Array, Int64Array, ListBuilder, StringArray};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Value};

use super::data::detect_run;
use super::{judge, report, Thresholds};

fn cases() -> Vec<(String, Value)> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../sdk/tests/fixtures/checkpoint/cases.json");
    let document: Value = serde_json::from_str(&std::fs::read_to_string(path).expect("the page's cases")).unwrap();
    document["cases"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(_, body)| body["group"] == "records")
        .map(|(name, body)| (name.clone(), body.clone()))
        .collect()
}

/// One rank's records stored as the server stores them (`flight.py`'s `record_to_row`): `step`,
/// `attempt`, `grad_norm` and `lr` in their columns, `checkpoint` as JSON text in
/// `checkpoint_json`. Each record keeps its own attempt; the server splits a batch by attempt.
fn stored(records: &[Value]) -> RecordBatch {
    let n = records.len();
    let mut lr = ListBuilder::new(Float64Builder::new());
    for record in records {
        match record.get("lr").and_then(Value::as_array) {
            Some(rates) => {
                for rate in rates {
                    lr.values().append_option(rate.as_f64());
                }
                lr.append(true);
            }
            None => lr.append(false),
        }
    }
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("run_id", Arc::new(StringArray::from(vec!["ac2"; n]))),
        (
            "rank",
            Arc::new(Int32Array::from_iter_values(
                records.iter().map(|r| r["rank"].as_i64().unwrap_or(0) as i32),
            )),
        ),
        (
            "attempt",
            Arc::new(Int64Array::from_iter_values(
                records.iter().map(|r| r["attempt"].as_i64().unwrap()),
            )),
        ),
        (
            "step",
            Arc::new(Int64Array::from_iter_values(
                records.iter().map(|r| r["step"].as_i64().unwrap()),
            )),
        ),
        (
            "grad_norm",
            Arc::new(Float64Array::from(
                records
                    .iter()
                    .map(|r| r.get("grad_norm").and_then(Value::as_f64))
                    .collect::<Vec<_>>(),
            )),
        ),
        ("lr", Arc::new(lr.finish())),
        (
            "checkpoint_json",
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|r| r.get("checkpoint").map(Value::to_string))
                    .collect::<Vec<_>>(),
            )),
        ),
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

/// The rank's report on `records`, as the server writes it.
fn judged(records: &[Value]) -> Value {
    let detectors = judge(0, &[stored(records)], &Arc::new(Thresholds::default())).unwrap();
    report("ac2", 0, &detectors, 0)
}

fn resume_incomplete(report: &Value) -> Option<&Value> {
    report["incidents"]
        .as_array()
        .unwrap()
        .iter()
        .find(|incident| incident["kind"] == "resume_incomplete")
}

/// What differs between a case's `expected` and the server's report: whether `resume_incomplete`
/// is raised, the first change's findings exactly, and the block exactly (null: no block).
fn case_differs(body: &Value) -> Option<String> {
    let report = judged(body["records"].as_array().unwrap());
    let expected = &body["expected"];
    let mut wrong = Vec::new();
    let found = resume_incomplete(&report);
    if json!(found.is_some()) != expected["fires"] {
        wrong.push(format!("fires {} expected {}", found.is_some(), expected["fires"]));
    }
    if let (Some(found), Some(findings)) = (found, expected.get("findings")) {
        let got = &found["changes"][0]["findings"];
        if got != findings {
            wrong.push(format!("findings {got} expected {findings}"));
        }
    }
    let block = report.get("checkpoint_tensors").cloned().unwrap_or(Value::Null);
    if block != expected["checkpoint_tensors"] {
        wrong.push(format!(
            "checkpoint_tensors {block} expected {}",
            expected["checkpoint_tensors"]
        ));
    }
    (!wrong.is_empty()).then(|| wrong.join("\n    "))
}

#[test]
fn every_records_case_holds_on_the_server() {
    let all = cases();
    assert_eq!(all.len(), 17, "section 4 has 17 cases over records");
    let wrong: Vec<String> = all
        .iter()
        .filter_map(|(name, body)| case_differs(body).map(|why| format!("{name}:\n    {why}")))
        .collect();
    assert!(
        wrong.is_empty(),
        "{} of 17 cases differ:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The same cases through `tensorquill-server detect --run`, which writes each rank's report.
#[test]
fn every_records_case_holds_through_detect_run() {
    for (name, body) in cases() {
        let run = detect_run(
            &[stored(body["records"].as_array().unwrap())],
            &Arc::new(Thresholds::default()),
        )
        .unwrap();
        let rank = &run["reports"][0];
        let block = rank.get("checkpoint_tensors").cloned().unwrap_or(Value::Null);
        assert_eq!(block, body["expected"]["checkpoint_tensors"], "{name}: {rank}");
    }
}

// --- readings no case states -------------------------------------------------------------------

/// Ten steps of attempt 1000, then one attempt per load in `loads`, each five steps long and
/// starting with the load's manifest.
fn resumes(loads: &[Value]) -> Vec<Value> {
    let mut out: Vec<Value> = (0..10)
        .map(|step| json!({"step": step, "rank": 0, "attempt": 1000, "lr": [0.001], "grad_norm": 1.0}))
        .collect();
    for (at, load) in loads.iter().enumerate() {
        let start = 10 + 5 * at as i64;
        let attempt = 2000 + 1000 * at as i64;
        for step in start..start + 5 {
            let mut record = json!({"step": step, "rank": 0, "attempt": attempt, "lr": [0.001], "grad_norm": 1.0});
            if step == start {
                record["checkpoint"] = json!({"load": load});
            }
            out.push(record);
        }
    }
    out
}

/// Every component a resume expects, so that today's findings are none.
fn components() -> Value {
    json!({
        "model": {"tensors": 7, "keys": ["model"]},
        "optimizer": {"entries": 7, "param_groups": 1, "step": 10, "keys": ["optimizer"]},
        "rng": {"keys": ["rng"]},
        "data": {"keys": ["sampler"]},
    })
}

fn load(tensors: Value) -> Value {
    json!({"components": components(), "tensors": tensors})
}

fn findings(report: &Value) -> Vec<Value> {
    resume_incomplete(report).map_or_else(Vec::new, |incident| {
        incident["changes"][0]["findings"].as_array().unwrap().clone()
    })
}

/// A bare state dict is compared with `entry` null (section 3): the key is carried as the record
/// has it, null included.
#[test]
fn a_null_entry_is_carried_as_the_record_has_it() {
    let report = judged(&resumes(&[load(json!({
        "state": "compared", "entry": null, "by": "names", "model": 3, "trainable": 3, "checkpoint": 3, "matched": 3,
    }))]));
    assert_eq!(
        report["checkpoint_tensors"],
        json!({"resumes": 1, "entries": [{"step": 10, "rank": 0, "attempt": 2000, "state": "compared",
            "entry": null, "by": "names", "model": 3, "trainable": 3, "checkpoint": 3, "matched": 3}]})
    );
    assert!(resume_incomplete(&report).is_none());
}

/// An `unavailable` record's lists are carried in its entry, as any entry's are, and raise
/// nothing: findings come only from `compared`.
#[test]
fn an_unavailable_records_lists_are_carried_and_raise_nothing() {
    let report = judged(&resumes(&[load(json!({
        "state": "unavailable", "why": "ambiguous", "missing": {"count": 1, "names": ["a.weight"]},
        "candidates": ["a", "b"],
    }))]));
    assert!(resume_incomplete(&report).is_none(), "{report}");
    assert_eq!(
        report["checkpoint_tensors"]["entries"],
        json!([{"step": 10, "rank": 0, "attempt": 2000, "state": "unavailable", "why": "ambiguous",
            "missing": {"count": 1, "names": ["a.weight"]}}])
    );
}

/// A `state` that is neither word gives no entry and no finding; `resumes` counts the entries.
#[test]
fn a_state_that_is_neither_word_gives_no_entry() {
    let missing = json!({"count": 1, "names": ["a.weight"]});
    let report = judged(&resumes(&[
        load(json!({"state": "skipped", "missing": missing})),
        load(json!({"state": "compared", "entry": "model"})),
        load(json!({"missing": missing})),
    ]));
    assert!(resume_incomplete(&report).is_none(), "{report}");
    assert_eq!(
        report["checkpoint_tensors"],
        json!({"resumes": 1, "entries": [{"step": 15, "rank": 0, "attempt": 3000, "state": "compared", "entry": "model"}]})
    );
}

/// Seven resumes: `resumes` counts them all, `entries` lists the first five by step, and each
/// resume's tensors are its own change's findings.
#[test]
fn the_block_lists_the_first_five_by_step() {
    let loads: Vec<Value> = (0..7)
        .map(|at| load(json!({"state": "compared", "missing": {"count": 1, "names": [format!("t{at}")]}})))
        .collect();
    let report = judged(&resumes(&loads));
    let block = &report["checkpoint_tensors"];
    assert_eq!(block["resumes"], 7);
    let steps: Vec<i64> = block["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["step"].as_i64().unwrap())
        .collect();
    assert_eq!(steps, [10, 15, 20, 25, 30]);
    let incident = resume_incomplete(&report).expect("resume_incomplete");
    assert_eq!(incident["resumes"], 7);
    assert_eq!(
        incident["changes"][4]["findings"],
        json!([{"component": "model", "finding": "missing_tensors", "count": 1, "tensors": ["t4"]}])
    );
}

/// The tensors come last: after the manifest's findings and after the optimizer snapshot's.
#[test]
fn tensor_findings_come_after_a_snapshots_finding() {
    let mut records = resumes(&[json!({
        "components": {"model": {"tensors": 7}, "rng": {}, "data": {}},
        "tensors": {"state": "compared",
            "shape_differs": {"count": 1, "names": ["head.weight"]},
            "missing": {"count": 1, "names": ["lstm.weight_hh_l0"]}},
    })]);
    records[10]["checkpoint"]["optimizer"] = json!({"entries_before": 0, "entries_after": 3});
    assert_eq!(
        findings(&judged(&records)),
        [
            json!({"component": "optimizer", "finding": "missing"}),
            json!({"component": "optimizer", "finding": "empty", "entries_after": 3}),
            json!({"component": "model", "finding": "missing_tensors", "count": 1, "tensors": ["lstm.weight_hh_l0"]}),
            json!({"component": "model", "finding": "shape_differs", "count": 1, "tensors": ["head.weight"]}),
        ]
    );
}

/// A list is read only with a positive whole `count`, read as `_int` reads it: not a float, a
/// boolean, a string, zero or less. The entry is written without such a list, and with the
/// others.
#[test]
fn a_count_that_is_not_a_positive_whole_number_is_not_read() {
    let names = json!(["a.weight"]);
    let report = judged(&resumes(&[load(json!({
        "state": "compared",
        "missing": {"count": 1.0, "names": names},
        "shape_differs": {"count": true, "names": names},
        "missing_frozen": {"count": "1", "names": names},
        "missing_buffers": {"count": -1, "names": names},
        "unexpected": {"names": names},
    }))]));
    assert!(resume_incomplete(&report).is_none(), "{report}");
    assert_eq!(
        report["checkpoint_tensors"]["entries"],
        json!([{"step": 10, "rank": 0, "attempt": 2000, "state": "compared"}])
    );
}

/// From `names` the strings are kept and anything else dropped; `count` is passed on unchanged,
/// and a list that is not an object is not read.
#[test]
fn names_keep_their_strings_and_count_is_unchanged() {
    let report = judged(&resumes(&[load(json!({
        "state": "compared",
        "missing": {"count": 60, "names": ["a", 3, null, {"b": 1}, "c"]},
        "shape_differs": {"count": 2, "names": "head.weight"},
        "unexpected": ["x"],
    }))]));
    assert_eq!(
        findings(&report),
        [
            json!({"component": "model", "finding": "missing_tensors", "count": 60, "tensors": ["a", "c"]}),
            json!({"component": "model", "finding": "shape_differs", "count": 2, "tensors": []}),
        ]
    );
    assert_eq!(
        report["checkpoint_tensors"]["entries"][0],
        json!({"step": 10, "rank": 0, "attempt": 2000, "state": "compared",
            "missing": {"count": 60, "names": ["a", "c"]}, "shape_differs": {"count": 2, "names": []}})
    );
}

/// What stays in the record (`tied`, `prefix`, `shapes`, `closest`, `candidates`) and keys the
/// contract does not name are not carried.
#[test]
fn keys_the_block_does_not_carry_stay_in_the_record() {
    let report = judged(&resumes(&[load(json!({
        "state": "compared", "entry": "model", "by": "loader",
        "tied": {"count": 1, "names": ["head.weight"]}, "prefix": {"removed": "net.", "from": "checkpoint"},
        "shapes": {"head.weight": [[2, 3], [3, 2]]}, "closest": {"entry": "x"}, "candidates": ["a"], "other": 1,
    }))]));
    assert_eq!(
        report["checkpoint_tensors"]["entries"][0],
        json!({"step": 10, "rank": 0, "attempt": 2000, "state": "compared", "entry": "model", "by": "loader"})
    );
}
