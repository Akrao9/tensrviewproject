//! Lane AB2 of wave AB (Nov 20, 2026): the server's side of `REPLICA_OBSERVATION_CONTRACT.md`,
//! section 4a (D1, which rank a replica drift is attributed to; cases in
//! `sdk/tests/fixtures/replica/attribution_cases.json`), and of `STEP_AND_EVIDENCE_CONTRACT.md`
//! (cases in `sdk/tests/fixtures/behaviour/cases.json`).

use serde_json::{json, Value};

use super::lane_z3_tests::{batch, first_seen, split_attribution};
use super::replica::{compare, record_from_json, records_from_batch, Bounds, Comparison};

// --- D1 ---------------------------------------------------------------------------------------

fn attribution_document() -> Value {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../sdk/tests/fixtures/replica/attribution_cases.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("section 4a's cases")).unwrap()
}

/// Every case of section 4a: (name, records, expected, tolerance).
fn attribution_cases() -> Vec<(String, Vec<Value>, Value, f64)> {
    let document = attribution_document();
    let tolerance = document["tolerance"].as_f64().unwrap();
    document["cases"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, body)| {
            assert!(body["says"].is_string(), "{name}");
            (
                name.clone(),
                body["records"].as_array().unwrap().clone(),
                body["expected"].clone(),
                tolerance,
            )
        })
        .collect()
}

/// The coverage of each step entry, apart from what was compared (`_coverage` in
/// `test_replica_attribution_contract.py`).
fn coverage(result: &Value) -> Value {
    let keys = ["step", "launch", "group", "coverage", "missing", "why", "ranks"];
    let entries: Vec<Value> = result["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| {
            let kept: serde_json::Map<String, Value> = keys
                .iter()
                .filter_map(|key| entry.get(*key).map(|value| (key.to_string(), value.clone())))
                .collect();
            Value::Object(kept)
        })
        .collect();
    json!(entries)
}

/// The three expectations of a case, each checked apart: coverage, attribution and the number of
/// incidents. None when all three hold, else what differs.
fn differs(result: &Value, expected: &Value) -> Option<String> {
    let mut wrong = Vec::new();
    if coverage(result) != expected["coverage"] {
        wrong.push(format!(
            "coverage {} expected {}",
            coverage(result),
            expected["coverage"]
        ));
    }
    let (_, attribution) = split_attribution(result.clone());
    if attribution != expected["attribution"] {
        wrong.push(format!(
            "attribution {attribution} expected {}",
            expected["attribution"]
        ));
    }
    let incidents = result["replica_drift"].as_array().unwrap().len();
    if json!(incidents) != expected["incidents"] {
        wrong.push(format!("incidents {incidents} expected {}", expected["incidents"]));
    }
    (!wrong.is_empty()).then(|| wrong.join("\n    "))
}

#[test]
fn every_attribution_case_gives_its_coverage_attribution_and_incident_count() {
    let all = attribution_cases();
    assert_eq!(all.len(), 27, "section 4a has 27 cases");
    let wrong: Vec<String> = all
        .iter()
        .filter_map(|(name, records, expected, tolerance)| {
            differs(&compare(records, *tolerance), expected).map(|why| format!("{name}:\n    {why}"))
        })
        .collect();
    assert!(
        wrong.is_empty(),
        "{} of 27 cases differ:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// A list present only when it has an entry: a result with no difference is spelled as before.
#[test]
fn a_result_with_no_difference_has_no_attribution_key() {
    for (name, records, expected, tolerance) in attribution_cases() {
        let result = compare(&records, tolerance);
        let empty = expected["attribution"].as_array().unwrap().is_empty();
        assert_eq!(result.get("attribution").is_none(), empty, "{name}: {result}");
    }
}

/// The server's rows arrive in any order across ranks; each case gives the same three answers.
#[test]
fn every_attribution_case_holds_in_any_order_of_arrival() {
    for (name, records, expected, tolerance) in attribution_cases() {
        let places = first_seen(&records);
        let place = |rank: i64, attempt: i64| places.get(&(rank, attempt)).copied();
        let n = records.len();
        let mut orders: Vec<Vec<usize>> = vec![(0..n).collect(), (0..n).rev().collect()];
        orders.extend((1..n).map(|shift| (0..n).map(|at| (at + shift) % n).collect()));
        for order in orders {
            let mut comparison = Comparison::new(tolerance, Bounds::default());
            for at in &order {
                comparison.insert(record_from_json(&records[*at], tolerance).unwrap(), &place);
            }
            let result = comparison.result();
            assert_eq!(differs(&result, &expected), None, "{name} in the order {order:?}");
            // And exactly what a reader of the finished list says.
            assert_eq!(result, compare(&records, tolerance), "{name} in the order {order:?}");
        }
    }
}

/// The same cases read from stored batches, one batch per rank and attempt.
#[test]
fn every_attribution_case_holds_from_stored_batches() {
    for (name, records, expected, tolerance) in attribution_cases() {
        let places = first_seen(&records);
        let place = |rank: i64, attempt: i64| places.get(&(rank, attempt)).copied();
        let mut comparison = Comparison::new(tolerance, Bounds::default());
        let mut parts: Vec<((i64, i64), Vec<&Value>)> = Vec::new();
        for record in &records {
            let key = (record["rank"].as_i64().unwrap(), record["attempt"].as_i64().unwrap());
            match parts.iter_mut().find(|(held, _)| *held == key) {
                Some((_, rows)) => rows.push(record),
                None => parts.push((key, vec![record])),
            }
        }
        for ((rank, _), rows) in &parts {
            for record in records_from_batch(&batch(rows), *rank, tolerance) {
                comparison.insert(record, &place);
            }
        }
        assert_eq!(differs(&comparison.result(), &expected), None, "{name}");
    }
}

/// `attribution` survives a restart: taken up from the object written, every entry stays.
#[test]
fn attribution_survives_a_restore_from_the_written_object() {
    for (name, records, _, tolerance) in attribution_cases() {
        let mut before = Comparison::new(tolerance, Bounds::default());
        let places = first_seen(&records);
        for record in &records {
            before.insert(record_from_json(record, tolerance).unwrap(), &|rank, attempt| {
                places.get(&(rank, attempt)).copied()
            });
        }
        let written = before.status_object();
        let mut after = Comparison::new(tolerance, Bounds::default());
        after.restore(&written);
        assert_eq!(after.status_object(), written, "{name}");
        // Through the text the server writes, too.
        let text: Value = serde_json::from_str(&written.to_string()).unwrap();
        let mut again = Comparison::new(tolerance, Bounds::default());
        again.restore(&text);
        assert_eq!(again.status_object(), written, "{name}");
    }
}

const LAUNCH: &str = "2f6c1a9e4b7d4c1e9a3f6b8d0e1c2a47";

fn observation(rank: i64, attempt: i64, step: i64, w: f64) -> Value {
    json!({"run_id": "r1", "rank": rank, "attempt": attempt, "step": step,
        "replica": {"v": 1, "launch": LAUNCH, "group": [0, 1, 2], "device": "cuda",
            "projection": "name-v1", "signature": "fold64-v1",
            "params": {"w": {"shape": [8, 4], "dtype": "float32", "fp": [w, 1.0, 0.5],
                "sig": if w == 3.5 { "7" } else { "8" }}}}})
}

fn rank_1_named(step: i64) -> Value {
    json!([
        {"step": step, "launch": LAUNCH, "group": [0, 1, 2], "parameter": "w", "by": "fingerprints",
            "verdict": "attributed", "outliers": [1]},
        {"step": step, "launch": LAUNCH, "group": [0, 1, 2], "parameter": "w", "by": "signatures",
            "verdict": "attributed", "outliers": [1]},
    ])
}

/// The cell's verdict is kept once its observations are let go for the byte bound; its
/// attribution is kept with it, so a step judged again (a gathered spread arriving) still names it.
#[test]
fn attribution_is_kept_when_the_observations_are_let_go() {
    let size = observation(0, 10, 99, 3.5)["replica"].to_string().len();
    let bounds = Bounds {
        held_bytes: size * 3,
        ..Bounds::default()
    };
    let mut comparison = Comparison::new(1e-6, bounds);
    let place = |_: i64, _: i64| Some(0);
    for record in [
        observation(0, 10, 99, 3.5),
        observation(1, 11, 99, 3.6),
        observation(2, 12, 99, 3.5),
    ] {
        comparison.insert(record_from_json(&record, 1e-6).unwrap(), &place);
    }
    assert_eq!(comparison.result()["attribution"], rank_1_named(99));
    // A row of another step needs the room: the judged cell's observations go first.
    comparison.insert(record_from_json(&observation(0, 10, 199, 3.5), 1e-6).unwrap(), &place);
    assert_eq!(comparison.held_bytes(), size);
    let spread = json!({"rank": 0, "attempt": 10, "step": 99, "replica_spread": {"v": [0.0]}});
    comparison.insert(record_from_json(&spread, 1e-6).unwrap(), &place);
    let result = comparison.result();
    assert_eq!(result["attribution"], rank_1_named(99), "{result}");
    assert_eq!(result.get("omitted"), None, "{result}");
}

/// A launch and step frozen because an earlier attempt's backlog arrived after the others'
/// observations were let go (`re_pair`) keeps the attribution it had, as it keeps its entries.
#[test]
fn a_frozen_launch_and_step_keeps_its_attribution() {
    let size = observation(0, 10, 99, 3.5)["replica"].to_string().len();
    let bounds = Bounds {
        held_bytes: size * 3,
        ..Bounds::default()
    };
    let mut comparison = Comparison::new(1e-6, bounds);
    // Rank 1's attempt 5 comes before its attempt 11 in its order.
    let place = |rank: i64, attempt: i64| Some(if rank == 1 && attempt == 11 { 1 } else { 0 });
    for record in [
        observation(0, 10, 99, 3.5),
        observation(1, 11, 99, 3.6),
        observation(2, 12, 99, 3.5),
        observation(0, 10, 199, 3.5),
    ] {
        comparison.insert(record_from_json(&record, 1e-6).unwrap(), &place);
    }
    comparison.insert(record_from_json(&observation(1, 5, 99, 3.5), 1e-6).unwrap(), &place);
    let result = comparison.result();
    assert_eq!(result["omitted"]["frozen"][0]["why"], "re_pair", "{result}");
    assert_eq!(result["attribution"], rank_1_named(99), "{result}");
}

/// A step past the listing bound is forgotten with its attribution: the incidents are kept, as
/// before, and `omitted` counts the step as before.
#[test]
fn a_forgotten_step_takes_its_attribution_with_it() {
    let bounds = Bounds {
        listed_steps: 1,
        ..Bounds::default()
    };
    let mut comparison = Comparison::new(1e-6, bounds);
    let place = |_: i64, _: i64| Some(0);
    for step in [99, 199] {
        for (rank, w) in [(0, 3.5), (1, 3.6), (2, 3.5)] {
            comparison.insert(
                record_from_json(&observation(rank, 10 + rank, step, w), 1e-6).unwrap(),
                &place,
            );
        }
    }
    let result = comparison.result();
    assert_eq!(result["attribution"], rank_1_named(199), "{result}");
    assert_eq!(result["replica_drift"].as_array().unwrap().len(), 2);
    assert_eq!(result["omitted"], json!({"steps": 1}));
    // Taken up after a restart, the same.
    let mut after = Comparison::new(1e-6, bounds);
    after.restore(&comparison.status_object());
    assert_eq!(after.result(), result);
}

/// The listing bound counts the attribution entries in a step's bytes.
#[test]
fn the_listed_bytes_count_the_attribution() {
    let mut plain = Comparison::new(1e-6, Bounds::default());
    let place = |_: i64, _: i64| Some(0);
    for (rank, w) in [(0, 3.5), (1, 3.6), (2, 3.5)] {
        plain.insert(
            record_from_json(&observation(rank, 10 + rank, 99, w), 1e-6).unwrap(),
            &place,
        );
    }
    let result = plain.result();
    let entries: usize = ["steps", "replica_drift"]
        .iter()
        .flat_map(|key| result[*key].as_array().unwrap().iter())
        .map(|item| item.to_string().len() + 1)
        .sum();
    // A bound that holds the entries and incidents but not the attribution beside them forgets
    // the older of two such steps.
    let bounds = Bounds {
        listed_bytes: 2 * entries + 1,
        ..Bounds::default()
    };
    let mut bounded = Comparison::new(1e-6, bounds);
    for step in [99, 199] {
        for (rank, w) in [(0, 3.5), (1, 3.6), (2, 3.5)] {
            bounded.insert(
                record_from_json(&observation(rank, 10 + rank, step, w), 1e-6).unwrap(),
                &place,
            );
        }
    }
    assert_eq!(bounded.result()["omitted"]["steps"], 1, "{}", bounded.result());
}

// --- STEP_AND_EVIDENCE_CONTRACT.md ------------------------------------------------------------

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Float64Array, Float64Builder, Int32Array, Int64Array, ListBuilder, MapBuilder, StringArray,
    StringBuilder,
};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;

use super::{judge, report, Thresholds};

fn behaviour() -> serde_json::Map<String, Value> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../sdk/tests/fixtures/behaviour/cases.json");
    let document: Value = serde_json::from_str(&std::fs::read_to_string(path).expect("the page's cases")).unwrap();
    document["cases"].as_object().unwrap().clone()
}

/// A number as the page's records write it: `"nan"`, `"inf"` and `"-inf"` as strings.
fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

/// One rank's records stored as the server stores them (`flight.py`'s `record_to_row`): the keys
/// with columns of their own in their columns, `layers` and `checkpoint` as JSON text, and every
/// other key (`step_skipped_by`, `grads_not_read`) in `extra_json`.
fn stored(records: &[Value], attempt: i64) -> RecordBatch {
    let n = records.len();
    let float = |key: &str| -> ArrayRef {
        Arc::new(Float64Array::from(
            records.iter().map(|r| r.get(key).and_then(number)).collect::<Vec<_>>(),
        ))
    };
    let mut norms = MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
    let mut unchanged = ListBuilder::new(StringBuilder::new());
    let mut extra = Vec::new();
    let own = [
        "step",
        "grad_norm",
        "clip_grad_norm",
        "grad_scale",
        "step_skipped",
        "grad_norms",
        "weights_unchanged",
        "layers",
        "checkpoint",
    ];
    for record in records {
        match record.get("grad_norms").and_then(Value::as_object) {
            Some(table) => {
                for (name, value) in table {
                    norms.keys().append_value(name);
                    norms.values().append_option(number(value));
                }
                norms.append(true).unwrap();
            }
            None => norms.append(false).unwrap(),
        }
        match record.get("weights_unchanged").and_then(Value::as_array) {
            Some(names) => {
                for name in names {
                    unchanged.values().append_value(name.as_str().unwrap());
                }
                unchanged.append(true);
            }
            None => unchanged.append(false),
        }
        let rest: serde_json::Map<String, Value> = record
            .as_object()
            .unwrap()
            .iter()
            .filter(|(key, _)| !own.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        extra.push((!rest.is_empty()).then(|| Value::Object(rest).to_string()));
    }
    let text = |key: &str| -> ArrayRef {
        Arc::new(StringArray::from(
            records
                .iter()
                .map(|r| r.get(key).map(Value::to_string))
                .collect::<Vec<_>>(),
        ))
    };
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("run_id", Arc::new(StringArray::from(vec!["ab2"; n]))),
        ("rank", Arc::new(Int32Array::from(vec![0; n]))),
        ("attempt", Arc::new(Int64Array::from(vec![attempt; n]))),
        (
            "step",
            Arc::new(Int64Array::from_iter_values(
                records.iter().map(|r| r["step"].as_i64().unwrap()),
            )),
        ),
        ("grad_norm", float("grad_norm")),
        ("clip_grad_norm", float("clip_grad_norm")),
        ("grad_scale", float("grad_scale")),
        (
            "step_skipped",
            Arc::new(BooleanArray::from(
                records
                    .iter()
                    .map(|r| r.get("step_skipped").and_then(Value::as_bool))
                    .collect::<Vec<_>>(),
            )),
        ),
        ("grad_norms", Arc::new(norms.finish())),
        ("weights_unchanged", Arc::new(unchanged.finish())),
        ("layers_json", text("layers")),
        ("checkpoint_json", text("checkpoint")),
        ("extra_json", Arc::new(StringArray::from(extra))),
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

/// The rank's report on a case's records, as the server writes it.
fn judged(records: &[Value]) -> Value {
    let detectors = judge(0, &[stored(records, 1)], &Arc::new(Thresholds::default())).unwrap();
    report("ab2", 0, &detectors, 0)
}

fn incident<'a>(report: &'a Value, kind: &str) -> Option<&'a Value> {
    report["incidents"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["kind"] == kind)
}

/// Each case of the page that holds records, judged by the server: a pin's detector and what it
/// gives, or a new case's target (the incidents present and absent, what an incident carries, the
/// kinds' values, the `not_checked` entries and the evidence's `skipped_updates`). None when it holds.
fn case_differs(name: &str, body: &Value) -> Option<String> {
    let records = body.get("records").or(body["given"].get("records"))?.as_array()?;
    let report = judged(records);
    let expected = &body["expected"];
    let mut wrong = Vec::new();
    if let Some(detector) = body["detector"].as_str() {
        let kind = match detector {
            "detect_grad_spikes" => "grad_spike",
            "detect_non_finite" => "non_finite",
            "detect_loss_scale_trouble" => "loss_scale_trouble",
            "detect_stalled_weights" => "stalled_weights",
            other => panic!("{name}: a detector this test does not know: {other}"),
        };
        let found = incident(&report, kind);
        if expected["fires"] == true {
            match found {
                None => wrong.push(format!("no {kind}")),
                Some(found) => {
                    for (key, value) in expected.as_object().unwrap().iter().filter(|(key, _)| *key != "fires") {
                        if &found[key] != value {
                            wrong.push(format!("{key}: {} expected {value}", found[key]));
                        }
                    }
                }
            }
        } else if let Some(found) = found {
            wrong.push(format!("{kind} raised: {found}"));
        }
    }
    for kind in expected["incident_kinds_present"].as_array().into_iter().flatten() {
        let kind = kind.as_str().unwrap();
        match incident(&report, kind) {
            None => wrong.push(format!("no {kind}")),
            Some(found) => {
                for (key, value) in expected["incident_carries"].as_object().into_iter().flatten() {
                    if &found[key] != value {
                        wrong.push(format!("{kind}.{key}: {} expected {value}", found[key]));
                    }
                }
            }
        }
    }
    for kind in expected["incident_kinds_absent"].as_array().into_iter().flatten() {
        if let Some(found) = incident(&report, kind.as_str().unwrap()) {
            wrong.push(format!("{kind} raised: {found}"));
        }
    }
    for (kind, value) in expected["kind_values"].as_object().into_iter().flatten() {
        if &report["kind_checks"][kind]["value"] != value {
            wrong.push(format!("{kind}: {} expected {value}", report["kind_checks"][kind]));
        }
    }
    for wanted in expected["not_checked"].as_array().into_iter().flatten() {
        let held = report["not_checked"].as_array().unwrap().iter().any(|entry| {
            wanted
                .as_object()
                .unwrap()
                .iter()
                .all(|(key, value)| &entry[key] == value)
        });
        if !held {
            wrong.push(format!("no not_checked entry {wanted}: {}", report["not_checked"]));
        }
    }
    for (key, value) in expected["evidence_carries"].as_object().into_iter().flatten() {
        if &report[key] != value {
            wrong.push(format!("{key}: {} expected {value}", report[key]));
        }
    }
    for key in expected["evidence_lacks"].as_array().into_iter().flatten() {
        if report.get(key.as_str().unwrap()).is_some() {
            wrong.push(format!("{key} is there: {}", report[key.as_str().unwrap()]));
        }
    }
    if let Some(state) = expected.get("run_state") {
        // The rank's report as the server holds it beside the status (`Ingest::write_report`).
        let names = |key: &str, of: &str| -> Vec<String> {
            report[key]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|item| item[of].as_str())
                .filter(|check| !matches!(*check, "flight_recorder" | "traincheck"))
                .map(str::to_string)
                .collect()
        };
        let held = crate::watchdog::RankReport {
            kinds: names("incidents", "kind"),
            not_checked: names("not_checked", "check"),
            attempt: report["attempt"].as_i64().unwrap(),
            last_step: report["last_step"].as_i64(),
            behind: false,
        };
        let reports = BTreeMap::from([(0, held)]);
        let status = json!({"summary": "active", "server_running": true, "incidents": [],
            "ranks": [{"rank": 0, "state": "active", "last_step": report["last_step"]}]});
        let got = crate::watchdog::run_state(&status, &reports, true);
        if &got["state"] != state {
            wrong.push(format!("run_state {got} expected {state}"));
        }
    }
    (!wrong.is_empty()).then(|| wrong.join("; "))
}

/// Sections 1 and 2: every case that holds records, the 12 pins and the new cases alike.
#[test]
fn every_case_of_sections_1_and_2_that_holds_records_holds_on_the_server() {
    let cases = behaviour();
    let mut ran = 0;
    let mut wrong = Vec::new();
    for (name, body) in &cases {
        let holds = body.get("records").or(body["given"].get("records")).is_some();
        if !holds {
            continue;
        }
        ran += 1;
        if let Some(why) = case_differs(name, body) {
            wrong.push(format!("{name}: {why}"));
        }
    }
    assert_eq!(ran, 27, "the page's cases over records");
    assert!(wrong.is_empty(), "{} cases differ:\n{}", wrong.len(), wrong.join("\n"));
}

/// The new cases over records also say what today's detector answers; it is not the target, and
/// the server no longer gives it: each such case cannot pass before its rule is built.
#[test]
fn what_today_answers_on_the_new_cases_is_no_longer_what_the_server_says() {
    for (name, body) in behaviour() {
        let Some(today) = body.get("today") else { continue };
        let records = body["given"]["records"].as_array().unwrap();
        let report = judged(records);
        let kind = today["kind"].as_str().unwrap();
        let same = incident(&report, kind).is_some_and(|found| {
            today
                .as_object()
                .unwrap()
                .iter()
                .filter(|(key, _)| !["detector", "fires", "kind"].contains(&key.as_str()))
                .all(|(key, value)| &found[key] == value)
        });
        assert!(
            !same,
            "{name}: the server still answers as today: {}",
            report["incidents"]
        );
    }
}

/// A row with `step_skipped_by` and no skip, a value this version does not know, and a skipped row
/// without the key beside rows that carry it.
#[test]
fn who_reported_a_skip_is_read_from_extra_json() {
    let records = vec![
        json!({"step": 0, "step_skipped": true}),
        json!({"step": 1, "step_skipped": true, "step_skipped_by": "someone"}),
        json!({"step": 2, "step_skipped": false, "step_skipped_by": "caller"}),
        json!({"step": 3, "step_skipped": true, "step_skipped_by": "caller", "grad_norm": "inf", "clip_grad_norm": "nan"}),
        json!({"step": 4, "step_skipped": true, "step_skipped_by": "both", "grad_norm": "nan"}),
    ];
    let report = judged(&records);
    assert_eq!(
        report["skipped_updates"],
        json!({"caller": 1, "scaler": 2, "both": 1,
            "non_finite_gradients": {"steps": 2, "first_step": 3, "gradients": ["<total>", "<total before clipping>"]}})
    );
    // The loss scale's input is every row that says whether it skipped, whoever said so.
    assert_eq!(report["kind_checks"]["loss_scale_trouble"]["records"], 5);
    assert_eq!(
        report["kind_checks"]["loss_scale_trouble"]["needs"],
        "step_skipped (scaler=, or session.step(skipped=))"
    );
}

/// The `gradients` entry of `not_checked`, in Python's shape: each parameter once, with the first
/// step and why; only `grad_spike` is made `unavailable` by it.
#[test]
fn an_unread_gradient_is_named_with_why_and_its_first_step() {
    let records = vec![
        json!({"step": 4, "grad_norm": 1.0}),
        json!({"step": 5, "grad_norm": 1.0, "grads_not_read": {"emb.weight": "a sparse gradient"}}),
        json!({"step": 6, "grad_norm": 1.0, "grads_not_read": {"emb.weight": "on meta", "a.bias": "nested"}}),
    ];
    let report = judged(&records);
    let entry = report["not_checked"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["check"] == "gradients")
        .unwrap();
    assert_eq!(
        entry,
        &json!({"check": "gradients", "kinds": ["grad_spike"], "parameters": ["a.bias", "emb.weight"],
            "reason": "gradient not read for a.bias: nested; emb.weight: a sparse gradient", "step": 5})
    );
    assert_eq!(report["kind_checks"]["grad_spike"]["value"], "unavailable");
    assert_eq!(report["kind_checks"]["grad_spike"]["check"], "gradients");
    for kind in ["non_finite", "stalled_weights"] {
        assert_ne!(report["kind_checks"][kind]["value"], "unavailable", "{kind}");
    }
    // None named: no entry.
    let report = judged(&records[..1]);
    assert!(report["not_checked"]
        .as_array()
        .unwrap()
        .iter()
        .all(|e| e["check"] != "gradients"));
}

/// Section 2, `one_rank_with_batch_hashes`: one rank with hashes was compared with none.
#[test]
fn one_rank_with_batch_hashes_is_compared_with_none() {
    let case = &behaviour()["one_rank_with_batch_hashes"];
    let window = |hashes: &[&str]| -> BTreeMap<i64, (Option<i64>, String)> {
        hashes
            .iter()
            .enumerate()
            .map(|(step, hash)| (step as i64, (None, hash.to_string())))
            .collect()
    };
    let ranks: BTreeMap<i32, _> = case["given"]["ranks_with_batch_hashes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|rank| (rank.as_i64().unwrap() as i32, window(&["5330bfb822ed0aaf"])))
        .collect();
    assert_eq!(
        json!(super::ranks_compared(&ranks)),
        case["expected"]["status"]["ranks_compared"]
    );
    // Beside a rank with no hashes, the same; two ranks with one kind in common are both listed.
    let mut two = ranks.clone();
    two.insert(1, window(&[]));
    assert_eq!(super::ranks_compared(&two), Vec::<i32>::new());
    two.insert(1, window(&["0123456789abcdef"]));
    assert_eq!(super::ranks_compared(&two), vec![0, 1]);
}

// --- section 3 ---------------------------------------------------------------------------------

mod silence {
    use std::time::{Duration, Instant};

    use serde_json::{json, Value};

    use crate::watchdog::{Watchdog, WatchdogConfig};

    fn key(rank: i32) -> (String, i32) {
        ("run".to_string(), rank)
    }

    fn at(base: Instant, seconds: u64) -> Instant {
        base + Duration::from_secs(seconds)
    }

    fn dog() -> Watchdog {
        Watchdog::new(WatchdogConfig {
            silent_floor: Duration::from_secs(60),
            silent_factor: 5.0,
            critical: Duration::from_secs(180),
            pause_limit: Duration::from_secs(600),
            start_limit: Duration::from_secs(600),
            ..WatchdogConfig::default()
        })
    }

    fn case(name: &str) -> Value {
        super::behaviour()[name].clone()
    }

    fn silence(status: &Value) -> Value {
        let incidents = status["incidents"].as_array().unwrap();
        assert_eq!(incidents.len(), 1, "{status}");
        incidents[0].clone()
    }

    /// Rank 0 trains on; rank 1 stored steps to `last`, then (when `stopped`) its capture stopped
    /// with a writer error while it went on beating, and its beats ceased at second 400.
    fn two_ranks(last: i64, stopped: bool) -> (Watchdog, Instant) {
        let t0 = Instant::now();
        let dog = dog();
        for rank in [0, 1] {
            dog.stream_opened(&key(rank), t0);
            assert!(dog.attempt_seen(&key(rank), 1, t0));
            dog.heartbeat(&key(rank), "train", 5.0, Some(0), None, t0);
        }
        dog.batch(&key(1), 1, Some(last), Some(2), at(t0, 1));
        if stopped {
            dog.capture_stopped(&key(1), "sink_error", at(t0, 2));
        }
        for second in (2..=400).step_by(5) {
            dog.heartbeat(&key(1), "train", 5.0, Some(10), None, at(t0, second));
            dog.heartbeat(&key(0), "train", 5.0, Some(second as i64), None, at(t0, second));
            dog.batch(&key(0), 1, Some(second as i64), Some(2), at(t0, second));
        }
        for second in (405..=800).step_by(5) {
            dog.heartbeat(&key(0), "train", 5.0, Some(second as i64), None, at(t0, second));
            dog.batch(&key(0), 1, Some(second as i64), Some(2), at(t0, second));
        }
        (dog, t0)
    }

    #[test]
    fn silent_after_capture_stopped() {
        let expected = &case("silent_after_capture_stopped")["expected"];
        let (dog, t0) = two_ranks(120, true);
        let status = &dog.statuses(at(t0, 800), true)["run"];
        let incident = silence(status);
        assert_eq!(json!([incident["kind"]]), expected["incident_kinds_present"]);
        assert_eq!(incident["capture_stopped"]["reason"], "sink_error");
        // How long ago, at the status's own time: the entry's value, unchanged.
        let entry = status["ranks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["rank"] == 1)
            .unwrap();
        assert_eq!(incident["capture_stopped"], entry["capture_stopped"]);
        assert_eq!(
            incident["capture_last_step"],
            expected["incident_carries"]["capture_last_step"]
        );
    }

    #[test]
    fn silent_with_capture_running() {
        let expected = &case("silent_with_capture_running")["expected"];
        let (dog, t0) = two_ranks(120, false);
        let incident = silence(&dog.statuses(at(t0, 800), true)["run"]);
        assert_eq!(json!([incident["kind"]]), expected["incident_kinds_present"]);
        for key in expected["incident_lacks"].as_array().unwrap() {
            assert!(incident.get(key.as_str().unwrap()).is_none(), "{key}: {incident}");
        }
    }

    /// Rank 1's earlier attempt reached step 100; the job was started again, the new attempt's
    /// capture stopped before its first record, and its beats ceased. Beside rank 0 training on,
    /// the silence is `rank_silent`; alone, `job_stalled`. Either way the last step is null.
    #[test]
    fn silent_after_a_restart_that_stored_nothing() {
        let body = case("silent_after_a_restart_that_stored_nothing");
        let expected = &body["expected"];
        assert_eq!(body["given"]["earlier_attempt"]["last_step"], 100);
        for beside in [true, false] {
            let t0 = Instant::now();
            let dog = dog();
            let ranks: &[i32] = if beside { &[0, 1] } else { &[1] };
            for rank in ranks {
                dog.stream_opened(&key(*rank), t0);
                assert!(dog.attempt_seen(&key(*rank), 0, t0));
                dog.heartbeat(&key(*rank), "train", 5.0, Some(0), None, t0);
                dog.batch(&key(*rank), 0, Some(100), Some(ranks.len() as i32), at(t0, 1));
            }
            // Rank 1 is started again: attempt 1, whose capture stops before its first record.
            assert!(dog.attempt_seen(&key(1), 1, at(t0, 10)));
            dog.heartbeat(&key(1), "train", 5.0, Some(0), None, at(t0, 10));
            dog.capture_stopped(&key(1), "sink_error", at(t0, 11));
            for second in (11..=400).step_by(5) {
                dog.heartbeat(&key(1), "train", 5.0, Some(0), None, at(t0, second));
                if beside {
                    dog.heartbeat(&key(0), "train", 5.0, Some(second as i64), None, at(t0, second));
                    dog.batch(&key(0), 0, Some(100 + second as i64), Some(2), at(t0, second));
                }
            }
            let before = &dog.statuses(at(t0, 400), true)["run"];
            let entry = before["ranks"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["rank"] == 1)
                .unwrap();
            assert_eq!(
                (&entry["attempt"], &entry["last_step"], &entry["attempt_without_steps"]),
                (&json!(1), &json!(null), &json!(true)),
                "{entry}"
            );
            for second in (405..=1400).step_by(5) {
                if beside {
                    dog.heartbeat(&key(0), "train", 5.0, Some(second as i64), None, at(t0, second));
                    dog.batch(&key(0), 0, Some(100 + second as i64), Some(2), at(t0, second));
                }
            }
            let status = &dog.statuses(at(t0, 1400), true)["run"];
            let incident = silence(status);
            let kind = if beside { "rank_silent" } else { "job_stalled" };
            assert_eq!(incident["kind"], kind, "{status}");
            assert!(expected["incident_kind_one_of"]
                .as_array()
                .unwrap()
                .contains(&incident["kind"]));
            assert_eq!(incident["capture_stopped"]["reason"], "sink_error");
            assert_eq!(incident["capture_last_step"], Value::Null, "{incident}");
            assert!(incident.get("capture_last_step").is_some());
        }
    }
}

// --- section 4 ---------------------------------------------------------------------------------

/// A resume from a checkpoint whose manifest holds `components`: attempt 1 to step 5, attempt 2
/// from step 3, whose first row loaded the checkpoint.
fn resumed_from(components: Value) -> Value {
    let earlier: Vec<Value> = (0..6).map(|step| json!({"step": step, "grad_norm": 1.0})).collect();
    let later = vec![
        json!({"step": 3, "grad_norm": 1.0, "checkpoint": {"load": {"components": components}}}),
        json!({"step": 4, "grad_norm": 1.0}),
    ];
    let detectors = judge(
        0,
        &[stored(&earlier, 1), stored(&later, 2)],
        &Arc::new(Thresholds::default()),
    )
    .unwrap();
    report("ab2", 0, &detectors, 0)
}

fn findings(report: &Value) -> Vec<Value> {
    incident(report, "resume_incomplete").expect("resume_incomplete")["changes"][0]["findings"]
        .as_array()
        .unwrap()
        .clone()
}

#[test]
fn a_checkpoint_with_no_identified_model_state() {
    let expected = &behaviour()["a_checkpoint_with_no_identified_model_state"]["expected"];
    let found = findings(&resumed_from(
        json!({"optimizer": {"entries": 2}, "rng": {}, "data": {}}),
    ));
    for wanted in expected["findings"].as_array().unwrap() {
        assert!(found.contains(wanted), "{found:?}");
    }
    assert!(!found.contains(&json!({"component": "model", "finding": "missing"})));
}

#[test]
fn a_checkpoint_with_no_optimizer_state() {
    let expected = &behaviour()["a_checkpoint_with_no_optimizer_state"]["expected"];
    let found = findings(&resumed_from(json!({"model": {"tensors": 2}, "rng": {}, "data": {}})));
    assert_eq!(&json!(found), &expected["findings"]);
}
