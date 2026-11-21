//! Lane AE7 of wave AE (Nov 21, 2026): the server's side of sections 1, 2 and 4 of
//! `WAVE_AE_CONTRACT.md`, held to the shared cases in `sdk/tests/fixtures/wave_ae/cases.json`.
//!
//! Each case's records are stored as the server stores them (the keys with a column in their
//! column, every other key in `extra_json`, where `clip_calls`, `step_skipped_by` and
//! `step_skipped_part` travel) and judged by [`judge`]. The case's `expected` is read from the
//! rank's [`report`], and the counters a report does not carry (`loss_scale_trouble`'s
//! `skipped_steps`, `partial_steps` and streak at the end, when nothing was raised) from the rule
//! itself. The same cases run through the built binary in `sdk/tests/test_lane_ae7_server.py`.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Float64Array, Float64Builder, Int32Array, Int64Array, ListBuilder, MapBuilder, StringArray,
    StringBuilder,
};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Value};

use super::{judge, report, RankDetectors, Thresholds};

fn cases() -> Vec<(String, Value)> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../sdk/tests/fixtures/wave_ae/cases.json");
    let document: Value = serde_json::from_str(&std::fs::read_to_string(path).expect("the page's cases")).unwrap();
    document["cases"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(_, body)| body["section"] != 5)
        .map(|(name, body)| (name.clone(), body.clone()))
        .collect()
}

fn case(name: &str) -> Value {
    cases()
        .into_iter()
        .find(|(known, _)| known == name)
        .map(|(_, body)| body)
        .unwrap_or_else(|| panic!("no case {name}"))
}

const OWN: [&str; 11] = [
    "run_id",
    "rank",
    "attempt",
    "step",
    "grad_norm",
    "clip_grad_norm",
    "grad_scale",
    "step_skipped",
    "grad_norms",
    "update_rms",
    "weights_unchanged",
];

fn float_map(records: &[Value], key: &str) -> ArrayRef {
    let mut map = MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
    for record in records {
        match record.get(key).and_then(Value::as_object) {
            Some(table) => {
                for (name, value) in table {
                    map.keys().append_value(name);
                    map.values().append_option(value.as_f64());
                }
                map.append(true).unwrap();
            }
            None => map.append(false).unwrap(),
        }
    }
    Arc::new(map.finish())
}

/// One rank's records stored as the server stores them, in the order given.
fn stored(records: &[Value]) -> RecordBatch {
    let n = records.len();
    let float = |key: &str| -> ArrayRef {
        Arc::new(Float64Array::from(
            records
                .iter()
                .map(|r| r.get(key).and_then(Value::as_f64))
                .collect::<Vec<_>>(),
        ))
    };
    let mut unchanged = ListBuilder::new(StringBuilder::new());
    let mut extra = Vec::new();
    for record in records {
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
            .filter(|(key, _)| !OWN.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        extra.push((!rest.is_empty()).then(|| Value::Object(rest).to_string()));
    }
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("run_id", Arc::new(StringArray::from(vec!["ae7"; n]))),
        ("rank", Arc::new(Int32Array::from(vec![0; n]))),
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
        ("grad_norms", float_map(records, "grad_norms")),
        ("update_rms", float_map(records, "update_rms")),
        ("weights_unchanged", Arc::new(unchanged.finish())),
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

fn thresholds(given: &Value) -> Arc<Thresholds> {
    let options = match given.get("window") {
        Some(window) => json!({"update_window": window}),
        None => json!({}),
    };
    Arc::new(serde_json::from_value(options).unwrap())
}

/// The case's records in the order the server reads them: by attempt as first seen, then step.
fn in_order(records: &[Value]) -> Vec<Value> {
    let mut attempts: Vec<i64> = Vec::new();
    for record in records {
        let attempt = record["attempt"].as_i64().unwrap_or(0);
        if !attempts.contains(&attempt) {
            attempts.push(attempt);
        }
    }
    let mut out = records.to_vec();
    out.sort_by_key(|r| {
        let attempt = r["attempt"].as_i64().unwrap_or(0);
        (
            attempts.iter().position(|known| *known == attempt).unwrap(),
            r["step"].as_i64().unwrap(),
        )
    });
    out
}

fn judged(given: &Value) -> RankDetectors {
    let records = in_order(given["records"].as_array().unwrap());
    judge(0, &[stored(&records)], &thresholds(given)).unwrap()
}

fn incident(report: &Value, kind: &str) -> Option<Value> {
    report["incidents"]
        .as_array()
        .unwrap()
        .iter()
        .find(|incident| incident["kind"] == kind)
        .cloned()
}

/// What the server answers on a case, under the keys the case's `expected` uses.
fn answer(detectors: &RankDetectors, expected: &Value) -> Value {
    let report = report("ae7", 0, detectors, 0);
    let mut out = serde_json::Map::new();
    let scale = incident(&report, "loss_scale_trouble");
    for key in expected.as_object().unwrap().keys() {
        let value = match key.as_str() {
            "update_spike_step" => incident(&report, "update_spike").map_or(Value::Null, |i| i["step"].clone()),
            "stalled_weights_step" => incident(&report, "stalled_weights").map_or(Value::Null, |i| i["step"].clone()),
            "loss_scale_trouble_step" => scale.as_ref().map_or(Value::Null, |i| i["step"].clone()),
            "reason" | "consecutive_skips" => scale.as_ref().map_or(Value::Null, |i| i[key].clone()),
            "incident_partial_steps" => scale.as_ref().map_or(Value::Null, |i| i["partial_steps"].clone()),
            "clip_groups_entry" => report["not_checked"]
                .as_array()
                .unwrap()
                .iter()
                .find(|entry| entry["check"] == "clip_groups")
                .map_or(Value::Null, |entry| {
                    // Held to the keys the page fixes; the reason's words are pinned below.
                    let fixed: serde_json::Map<String, Value> = expected[key]
                        .as_object()
                        .map(|keys| keys.keys().map(|k| (k.clone(), entry[k].clone())).collect())
                        .unwrap_or_default();
                    Value::Object(fixed)
                }),
            "skipped_updates" => report.get("skipped_updates").cloned().unwrap_or(Value::Null),
            "counters_at_end" => {
                let rule = &detectors.loss_scale;
                json!({
                    "recorded_steps": rule.recorded,
                    "skipped_steps": rule.skipped,
                    "streak": rule.streak,
                    "partial_steps": rule.partial,
                })
            }
            other => panic!("a key no case of sections 1, 2 and 4 uses: {other}"),
        };
        out.insert(key.clone(), value);
    }
    Value::Object(out)
}

#[test]
fn every_case_of_sections_1_2_and_4_gives_its_expected() {
    let all = cases();
    assert_eq!(all.len(), 29, "the 34 cases less section 5's five");
    let mut wrong = Vec::new();
    for (name, body) in &all {
        let found = answer(&judged(&body["given"]), &body["expected"]);
        if found != body["expected"] {
            wrong.push(format!("{name}: {found} is not {}", body["expected"]));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The counters a report shows only on an incident, at the end of each section 4 case: in the
/// report they are on `loss_scale_trouble`'s incident and in `kind_checks` (`recorded_steps`).
#[test]
fn the_counters_at_the_end_agree_with_what_the_report_shows() {
    for (name, body) in cases()
        .into_iter()
        .filter(|(_, b)| b["expected"].get("counters_at_end").is_some())
    {
        let detectors = judged(&body["given"]);
        let report = report("ae7", 0, &detectors, 0);
        let counters = &body["expected"]["counters_at_end"];
        assert_eq!(
            report["kind_checks"]["loss_scale_trouble"]["records"], counters["recorded_steps"],
            "{name}"
        );
        if let Some(found) = incident(&report, "loss_scale_trouble") {
            assert_eq!(found["recorded_steps"], counters["recorded_steps"], "{name}");
            assert_eq!(found["skipped_steps"], counters["skipped_steps"], "{name}");
            let partial = counters["partial_steps"].as_u64().unwrap();
            assert_eq!(
                found.get("partial_steps").cloned(),
                (partial > 0).then(|| json!(partial)),
                "{name}"
            );
        }
    }
}

#[test]
fn the_clip_groups_entry_has_the_pages_words_and_update_spike_stays_checked() {
    for (name, comparisons) in [
        ("a_split_pair_with_two_clip_calls_is_not_compared", 1),
        ("two_pairs_that_are_not_compared", 2),
    ] {
        let report = report("ae7", 0, &judged(&case(name)["given"]), 0);
        let entries: Vec<&Value> = report["not_checked"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|entry| entry["check"] == "clip_groups")
            .collect();
        assert_eq!(entries.len(), 1, "{name}: one entry");
        assert_eq!(
            entries[0],
            &json!({
                "check": "clip_groups",
                "kinds": ["update_spike"],
                "comparisons": comparisons,
                "step": 5,
                "reason": format!(
                    "clipped by more than one clip_grad_norm_ call: the pre-clipping total cannot be split by \
                     parameter, so {comparisons} comparison(s) of update_spike were not made"
                ),
            }),
            "{name}"
        );
        assert_eq!(report["kind_checks"]["update_spike"]["value"], "checked", "{name}");
        assert!(report["kind_checks"]["update_spike"].get("check").is_none(), "{name}");
    }
}

/// A record's key changes nothing where its condition does not hold: the report on the records
/// with the key is the report on the same records without it.
fn same_report_without(name: &str, key: &str) {
    let given = case(name)["given"].clone();
    let mut without = given.clone();
    for record in without["records"].as_array_mut().unwrap() {
        record.as_object_mut().unwrap().remove(key);
    }
    let with_key = report("ae7", 0, &judged(&given), 0);
    let base = report("ae7", 0, &judged(&without), 0);
    assert_eq!(with_key, base, "{name}: {key} changed the report");
}

#[test]
fn a_key_whose_condition_does_not_hold_changes_nothing() {
    // Two clip calls on pairs that need no split, and values that are not a whole number above 1.
    for name in [
        "a_pair_naming_the_same_parameters_is_compared_whatever_the_clip_calls",
        "a_pair_that_cannot_be_split_is_compared_whatever_the_clip_calls",
        "clip_calls_that_is_one_is_read_as_no_key",
        "clip_calls_that_is_text_is_read_as_no_key",
        "clip_calls_that_is_a_fraction_is_read_as_no_key",
        "clip_calls_that_is_true_is_read_as_no_key",
    ] {
        same_report_without(name, "clip_calls");
    }
    // A part key that is not a mixed outcome, or on a skip the scaler did not confirm, or on an
    // applied step.
    for name in [
        "a_part_key_that_says_every_optimizer_skipped_is_a_full_skip",
        "a_part_key_that_says_no_optimizer_skipped_is_a_full_skip",
        "a_part_key_that_is_not_two_counts_is_a_full_skip",
        "a_part_key_on_a_skip_only_the_caller_reported_says_nothing",
        "a_part_key_on_an_applied_step_says_nothing",
    ] {
        same_report_without(name, "step_skipped_part");
    }
}

/// The kinds judged by parameter read a partial step as a skipped one: `stalled_weights`,
/// `grad_spike` and `update_spike` give the same answer whether a skipped step is partial or not.
#[test]
fn the_kinds_judged_by_parameter_read_a_partial_step_as_a_skipped_one() {
    // Only `skipped_updates` moves (the step is counted under `partial`): the stall is what it was.
    let given = case("a_stall_across_a_partial_step")["given"].clone();
    let mut without = given.clone();
    for record in without["records"].as_array_mut().unwrap() {
        record.as_object_mut().unwrap().remove("step_skipped_part");
    }
    let (partial, skipped) = (
        report("ae7", 0, &judged(&given), 0),
        report("ae7", 0, &judged(&without), 0),
    );
    assert_eq!(partial["incidents"], skipped["incidents"]);
    assert_eq!(partial["kind_checks"], skipped["kind_checks"]);
    assert_eq!(
        partial["skipped_updates"],
        json!({"caller": 0, "scaler": 0, "both": 0, "partial": 1})
    );
    assert_eq!(skipped["skipped_updates"], json!({"caller": 0, "scaler": 1, "both": 0}));
    // A skipped step that would be an update spike and a gradient spike if it were judged.
    let mut records: Vec<Value> = (0..12)
        .map(|step| {
            json!({"step": step, "grad_norm": 1.0 + 0.01 * f64::from(step), "update_rms": {"w": 1e-3},
                   "grad_norms": {"w": 1.0}, "weights_unchanged": [], "step_skipped": false})
        })
        .collect();
    records[10] = json!({"step": 10, "grad_norm": 50.0, "update_rms": {"w": 1.0}, "grad_norms": {"w": 50.0},
                         "weights_unchanged": ["w"], "step_skipped": true, "step_skipped_by": "scaler",
                         "step_skipped_part": {"skipped": 1, "of": 2}});
    let partial = report("ae7", 0, &judged(&json!({"records": records.clone()})), 0);
    for record in &mut records {
        record.as_object_mut().unwrap().remove("step_skipped_part");
    }
    let skipped = report("ae7", 0, &judged(&json!({"records": records})), 0);
    for kind in ["grad_spike", "update_spike", "stalled_weights"] {
        assert_eq!(incident(&partial, kind), incident(&skipped, kind), "{kind}");
        assert_eq!(partial["kind_checks"][kind], skipped["kind_checks"][kind], "{kind}");
    }
    assert!(incident(&partial, "update_spike").is_none() && incident(&partial, "grad_spike").is_none());
}

/// The live server reads a rank's parts as they arrive ([`RankDetectors::append`]): a pair that
/// is not compared, or a partial step, on a part boundary gives what one reading of every row
/// gives. A resume is rebuilt from the store; `test_lane_ae7_server.py` sends that to a server.
#[test]
fn rows_read_part_by_part_give_what_one_reading_gives() {
    for (name, at) in [
        ("after_a_pair_that_is_not_compared_the_next_is", 5),
        ("a_split_pair_whose_earlier_record_had_two_clip_calls", 5),
        ("two_pairs_that_are_not_compared", 6),
        ("five_full_skips_one_partial_step_five_full_skips", 5),
        ("five_full_skips_one_partial_step_five_full_skips", 6),
        ("a_partial_step_then_ten_full_skips", 1),
        ("skipped_updates_counts_a_partial_step_both_reported_apart", 5),
    ] {
        let given = case(name)["given"].clone();
        let records = in_order(given["records"].as_array().unwrap());
        let whole = judged(&given);
        let mut parts = RankDetectors::new(&thresholds(&given));
        let (before, after) = records.split_at(at);
        for piece in [before, after] {
            assert!(parts.continues(0, piece[0]["step"].as_i64().unwrap()));
            parts.append(0, &stored(piece));
        }
        assert_eq!(
            report("ae7", 0, &parts, 0),
            report("ae7", 0, &whole, 0),
            "{name} at {at}"
        );
        assert_eq!(
            answer(&parts, &case(name)["expected"]),
            case(name)["expected"],
            "{name} at {at}"
        );
    }
}
