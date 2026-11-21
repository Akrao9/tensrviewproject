//! Lane AD8 of wave AD (Nov 21, 2026): the server's side of `DETECTOR_RULES_CONTRACT.md`, parts 1
//! to 5 and 7, held to the shared cases in `sdk/tests/fixtures/rules/cases.json`.
//!
//! Each case's records are stored as the server stores them (`flight.py`'s `record_to_row`: the
//! keys with a column in their column, `layers` as JSON text, every other key in `extra_json`) and
//! judged by [`judge`]; the rank's [`report`] is compared with the case's `expected`. Numbers agree
//! to one part in 1e12; everything else is exact.
//!
//! What a case's `expected` holds that is a reader's, and so not the server's to write, is left out
//! by name: a score's `display` and the `moe_source` `words` (the Python reader's words), and the
//! all-ranks union of `parameters_not_judged` (`combine_rank_evidence`). For that last case each
//! rank's own list is checked here instead. The eight cases of the score's arithmetic are in
//! `lane_ad8_score_tests.rs`, and the three `capture_stopped_*` cases, which are the live status
//! and not a report, in `watchdog/lane_ad8_tests.rs`.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Float64Array, Float64Builder, Int32Array, Int32Builder, Int64Array, Int64Builder,
    ListBuilder, MapBuilder, StringArray, StringBuilder,
};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Value};

use super::data::detect_run;
use super::{judge, report, Thresholds};

fn cases() -> serde_json::Map<String, Value> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../sdk/tests/fixtures/rules/cases.json");
    let document: Value = serde_json::from_str(&std::fs::read_to_string(path).expect("the page's cases")).unwrap();
    document["cases"].as_object().unwrap().clone()
}

fn of(part: &str) -> Vec<(String, Value)> {
    cases().into_iter().filter(|(_, body)| body["part"] == part).collect()
}

/// A number as a case writes it; `"nan"` and the like as the strings the capture writes.
fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

/// The record keys with a column of their own, as `_FIELDS` in `flight.py` names them, that these
/// cases hold; and the key columns. Every other key travels in `extra_json`.
const OWN: [&str; 15] = [
    "run_id",
    "rank",
    "attempt",
    "step",
    "grad_norm",
    "clip_grad_norm",
    "step_skipped",
    "grad_norms",
    "update_rms",
    "weight_norms",
    "lr",
    "layers",
    "moe",
    "moe_top_k",
    "scalars",
];

fn float_map(records: &[Value], key: &str) -> ArrayRef {
    let mut map = MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
    for record in records {
        match record.get(key).and_then(Value::as_object) {
            Some(table) => {
                for (name, value) in table {
                    map.keys().append_value(name);
                    map.values().append_option(number(value));
                }
                map.append(true).unwrap();
            }
            None => map.append(false).unwrap(),
        }
    }
    Arc::new(map.finish())
}

/// One rank's records stored as the server stores them.
pub(super) fn stored(records: &[Value]) -> RecordBatch {
    let n = records.len();
    let float = |key: &str| -> ArrayRef {
        Arc::new(Float64Array::from(
            records.iter().map(|r| r.get(key).and_then(number)).collect::<Vec<_>>(),
        ))
    };
    let mut lr = ListBuilder::new(Float64Builder::new());
    let mut counts = MapBuilder::new(None, StringBuilder::new(), ListBuilder::new(Int64Builder::new()));
    let mut top_k = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
    let mut extra = Vec::new();
    for record in records {
        match record.get("lr").and_then(Value::as_array) {
            Some(rates) => {
                for rate in rates {
                    lr.values().append_option(number(rate));
                }
                lr.append(true);
            }
            None => lr.append(false),
        }
        match record.get("moe").and_then(Value::as_object) {
            Some(routers) => {
                for (name, experts) in routers {
                    counts.keys().append_value(name);
                    for count in experts.as_array().unwrap() {
                        counts.values().values().append_value(count.as_i64().unwrap());
                    }
                    counts.values().append(true);
                }
                counts.append(true).unwrap();
            }
            None => counts.append(false).unwrap(),
        }
        match record.get("moe_top_k").and_then(Value::as_object) {
            Some(routers) => {
                for (name, k) in routers {
                    top_k.keys().append_value(name);
                    top_k.values().append_value(k.as_i64().unwrap() as i32);
                }
                top_k.append(true).unwrap();
            }
            None => top_k.append(false).unwrap(),
        }
        let rest: serde_json::Map<String, Value> = record
            .as_object()
            .unwrap()
            .iter()
            .filter(|(key, _)| !OWN.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        extra.push((!rest.is_empty()).then(|| Value::Object(rest).to_string()));
    }
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("run_id", Arc::new(StringArray::from(vec!["ad8"; n]))),
        (
            "rank",
            Arc::new(Int32Array::from_iter_values(
                records.iter().map(|r| r["rank"].as_i64().unwrap_or(0) as i32),
            )),
        ),
        (
            "attempt",
            Arc::new(Int64Array::from_iter_values(
                records.iter().map(|r| r["attempt"].as_i64().unwrap_or(0)),
            )),
        ),
        (
            "step",
            Arc::new(Int64Array::from_iter_values(
                records.iter().map(|r| r["step"].as_i64().unwrap()),
            )),
        ),
        ("grad_norm", float("grad_norm")),
        ("clip_grad_norm", float("clip_grad_norm")),
        (
            "step_skipped",
            Arc::new(BooleanArray::from(
                records
                    .iter()
                    .map(|r| r.get("step_skipped").and_then(Value::as_bool))
                    .collect::<Vec<_>>(),
            )),
        ),
        ("grad_norms", float_map(records, "grad_norms")),
        ("update_rms", float_map(records, "update_rms")),
        ("scalars", float_map(records, "scalars")),
        ("weight_norms", float_map(records, "weight_norms")),
        ("lr", Arc::new(lr.finish())),
        (
            "layers_json",
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|r| r.get("layers").map(Value::to_string))
                    .collect::<Vec<_>>(),
            )),
        ),
        ("moe_counts", Arc::new(counts.finish())),
        ("moe_top_k", Arc::new(top_k.finish())),
        ("extra_json", Arc::new(StringArray::from(extra))),
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
    report("ad8", 0, &detectors, 0)
}

/// The same report as `tensorquill-server detect --run` writes it, in the run's `reports`.
fn through_run(records: &[Value]) -> Value {
    let run = detect_run(&[stored(records)], &Arc::new(Thresholds::default())).unwrap();
    run["reports"][0].clone()
}

/// Where `got` differs from `expected`: numbers to one part in 1e12, everything else exactly, an
/// object's keys included.
fn differs(at: &str, got: &Value, expected: &Value, out: &mut Vec<String>) {
    match (got, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            if (a - b).abs() > 1e-12 * a.abs().max(b.abs()).max(1.0) {
                out.push(format!("{at}: {a} expected {b}"));
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            for (key, y) in b {
                match a.get(key) {
                    Some(x) => differs(&format!("{at}.{key}"), x, y, out),
                    None => out.push(format!("{at}.{key}: absent, expected {y}")),
                }
            }
            for (key, x) in a {
                if !b.contains_key(key) {
                    out.push(format!("{at}.{key}: {x} not expected"));
                }
            }
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            for (index, (x, y)) in a.iter().zip(b).enumerate() {
                differs(&format!("{at}[{index}]"), x, y, out);
            }
        }
        _ if got == expected => {}
        _ => out.push(format!("{at}: {got} expected {expected}")),
    }
}

/// The report's incident of `kind`, without `class`, which every incident of a report carries, and
/// `scalars`, what was logged on its step, which a report adds to an incident of records that log
/// any (Python's `build_evidence` does the same) and a case states of none: the cases' incidents
/// are the detector functions' with the contract's keys added. Null when there is none.
fn incident(report: &Value, kind: &str) -> Value {
    report["incidents"]
        .as_array()
        .unwrap()
        .iter()
        .find(|incident| incident["kind"] == kind)
        .map_or(Value::Null, |incident| {
            let mut incident = incident.clone();
            let fields = incident.as_object_mut().unwrap();
            assert!(
                fields.remove("class").is_some(),
                "every incident has its class: {incident}"
            );
            fields.remove("scalars");
            incident
        })
}

/// The report's `not_checked`, without the two entries every server report has (the files the
/// server never receives) and no case states.
fn not_checked(report: &Value) -> Value {
    let all = report["not_checked"].as_array().unwrap();
    let standing: Vec<&Value> = all
        .iter()
        .filter(|item| item["check"] == "flight_recorder" || item["check"] == "traincheck")
        .collect();
    assert_eq!(standing.len(), 2, "{all:?}");
    Value::Array(
        all.iter()
            .filter(|item| item["check"] != "flight_recorder" && item["check"] != "traincheck")
            .cloned()
            .collect(),
    )
}

/// The keys of a `kind_checks` entry that a case states, and those it says are absent.
fn entry_differs(at: &str, got: &Value, expected: &Value, absent: &[&str], out: &mut Vec<String>) {
    for (key, value) in expected.as_object().unwrap() {
        differs(&format!("{at}.{key}"), &got[key], value, out);
    }
    for key in absent {
        if expected.get(*key).is_none() && got.get(*key).is_some() {
            out.push(format!("{at}.{key}: {} not expected", got[*key]));
        }
    }
}

fn kind(incident: &Value) -> &str {
    incident["kind"].as_str().unwrap()
}

/// What differs between a case's `expected` and the server's report on its records.
fn case_differs(name: &str, body: &Value, judged: fn(&[Value]) -> Value) -> Vec<String> {
    let expected = &body["expected"];
    let mut wrong = Vec::new();
    let part = body["part"].as_str().unwrap();
    if part == "evidence" && body.get("records_by_rank").is_some() {
        // The union across ranks is the Python reader's (`combine_rank_evidence`). Each rank's own
        // report names its own parameters, and the run report (`detect --run`) holds each rank's.
        let by_rank = &expected["kind_checks"]["stalled_weights"]["parameters_not_judged_by_rank"];
        let mut batches = Vec::new();
        for (rank, records) in body["records_by_rank"].as_object().unwrap() {
            let records: Vec<Value> = records
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    let mut r = r.clone();
                    r["rank"] = json!(rank.parse::<i64>().unwrap());
                    r
                })
                .collect();
            let report = judged(&records);
            differs(
                &format!("rank {rank} stalled_weights.parameters_not_judged"),
                &report["kind_checks"]["stalled_weights"]["parameters_not_judged"],
                &by_rank[rank],
                &mut wrong,
            );
            batches.push(stored(&records));
        }
        let run = detect_run(&batches, &Arc::new(Thresholds::default())).unwrap();
        for rank in run["reports"].as_array().unwrap() {
            let at = rank["rank"].to_string();
            differs(
                &format!("detect --run rank {at} parameters_not_judged"),
                &rank["kind_checks"]["stalled_weights"]["parameters_not_judged"],
                &by_rank[&at],
                &mut wrong,
            );
        }
        return wrong;
    }
    let report = judged(body["records"].as_array().unwrap());
    match part {
        "score_records" => {
            let want = &expected["incident"];
            differs("incident", &incident(&report, kind(want)), want, &mut wrong);
        }
        "activation_drift" => {
            differs(
                "incident",
                &incident(&report, "activation_drift"),
                &expected["incident"],
                &mut wrong,
            );
            let check = &report["kind_checks"]["activation_drift"];
            differs(
                "kind_checks.activation_drift.records_needed",
                &check["records_needed"],
                &expected["records_needed"],
                &mut wrong,
            );
            differs(
                "kind_checks.activation_drift.enough_history",
                &check["enough_history"],
                &expected["enough_history"],
                &mut wrong,
            );
        }
        "moe_imbalance" => {
            differs(
                "incident",
                &incident(&report, "moe_imbalance"),
                &expected["incident"],
                &mut wrong,
            );
            let check = &report["kind_checks"]["moe_imbalance"];
            let starting = check.get("starting_state").cloned().unwrap_or(json!([]));
            differs(
                "kind_checks.moe_imbalance.starting_state",
                &starting,
                &expected["starting_state"],
                &mut wrong,
            );
            if starting != json!([]) && check["enough_history"] != false {
                wrong.push(format!("a router in its starting state, and enough_history {check}"));
            }
            if check.get("starting_state") == Some(&json!([])) {
                wrong.push("starting_state is written only when a router is in it".to_string());
            }
        }
        "evidence" => {
            if let Some(want) = expected.get("incident") {
                differs("incident", &incident(&report, "update_spike"), want, &mut wrong);
            }
            for (name, want) in expected
                .get("kind_checks")
                .and_then(Value::as_object)
                .into_iter()
                .flatten()
            {
                entry_differs(
                    &format!("kind_checks.{name}"),
                    &report["kind_checks"][name],
                    want,
                    &["parameters_not_judged"],
                    &mut wrong,
                );
            }
        }
        "wire" if name.starts_with("moe_source") => {
            // `words` is what the Python reader says of it.
            let got = report.get("moe_source").cloned().unwrap_or(Value::Null);
            differs("moe_source", &got, &expected["moe_source"], &mut wrong);
        }
        "wire" => {
            differs(
                "not_checked",
                &not_checked(&report),
                &expected["not_checked"],
                &mut wrong,
            );
            for (name, want) in expected["kind_checks"].as_object().unwrap() {
                entry_differs(
                    &format!("kind_checks.{name}"),
                    &report["kind_checks"][name],
                    want,
                    &["modules_not_judged"],
                    &mut wrong,
                );
            }
        }
        other => panic!("{name}: no part {other} here"),
    }
    wrong
}

/// Every case of `parts` holds on the server; the names of those that do not, and why.
fn hold(parts: &[&str], count: usize, skip: &[&str], judged: fn(&[Value]) -> Value) {
    let all: Vec<(String, Value)> = parts.iter().flat_map(|part| of(part)).collect();
    let held: Vec<&(String, Value)> = all.iter().filter(|(name, _)| !skip.contains(&name.as_str())).collect();
    assert_eq!(held.len(), count, "the cases of {parts:?}");
    let wrong: Vec<String> = held
        .iter()
        .filter_map(|(name, body)| {
            let why = case_differs(name, body, judged);
            (!why.is_empty()).then(|| format!("{name}:\n    {}", why.join("\n    ")))
        })
        .collect();
    assert!(
        wrong.is_empty(),
        "{} of {count} cases differ:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn part_1_every_score_over_records_case_holds() {
    hold(&["score_records"], 4, &[], judged);
}

#[test]
fn part_2_every_activation_drift_case_holds() {
    hold(&["activation_drift"], 9, &[], judged);
}

#[test]
fn part_3_every_moe_imbalance_case_holds() {
    hold(&["moe_imbalance"], 11, &[], judged);
}

#[test]
fn parts_4_and_5_every_learning_rate_and_parameters_not_judged_case_holds() {
    hold(&["evidence"], 4, &[], judged);
}

/// The three `capture_stopped_*` cases are the live status, in `watchdog/lane_ad8_tests.rs`.
const LIVE: [&str; 3] = [
    "capture_stopped_at_step_0",
    "capture_stopped_before_any_step",
    "capture_stopped_from_an_older_sender",
];

#[test]
fn part_7_every_wire_case_over_records_holds() {
    hold(&["wire"], 7, &LIVE, judged);
}

/// Every case over records again, through `tensorquill-server detect --run`, which writes each
/// rank's report into the run's.
#[test]
fn every_case_over_records_holds_through_detect_run() {
    hold(
        &["score_records", "activation_drift", "moe_imbalance", "evidence", "wire"],
        35,
        &LIVE,
        through_run,
    );
}

/// No case states it: an incident dated before the row that makes it carries what was logged on
/// the step it is dated at, as Python's `build_evidence` reads `scalar_context` at its `step`. A
/// router flagged since step 0 is reported on its tenth sample (step 9) and dated at step 0, so its
/// scalars are step 0's, not step 9's.
#[test]
fn a_router_reported_since_its_start_carries_what_was_logged_at_its_first_step() {
    let records: Vec<Value> = (0..12)
        .map(|step| {
            json!({"step": step, "moe": {"r": [102, 22, 22, 22, 22, 22, 22, 22]}, "moe_top_k": {"r": 1},
                   "scalars": {"loss": 10.0 + step as f64}})
        })
        .collect();
    let report = judged(&records);
    let found = report["incidents"]
        .as_array()
        .unwrap()
        .iter()
        .find(|incident| incident["kind"] == "moe_imbalance")
        .expect("moe_imbalance");
    assert_eq!(
        (&found["step"], &found["reportable_step"], &found["since_start"]),
        (&json!(0), &json!(9), &json!(true))
    );
    assert_eq!(
        found["scalars"],
        json!({"loss": {"value": 10.0, "median_before": null}})
    );
}
