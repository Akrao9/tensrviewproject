//! Lane AD10 of wave AD (Nov 21, 2026): the server's side of `DETECTOR_RULES_CONTRACT.md`, part 6,
//! the four measurements `gradient_trend`, `attention_trends`, `update_trends` and `weight_jumps`,
//! held to the ten `measurements` cases of `sdk/tests/fixtures/rules/cases.json`.
//!
//! Each case's records are stored as the server stores them (lane AD8's [`stored`]), judged by
//! [`judge`], and the rank's [`report`] is compared with the case's `expected`. An `expected` of
//! null is a block that must be absent: a missing key, never `null`. Numbers agree to one part in
//! 1e12; everything else is exact. The tests after the cases pin what no case states, with numbers
//! worked out by hand from `make_cases.py`'s functions. `judge` and `detect_run` keep the two
//! trends' series ([`RankDetectors::whole_run`]); the live path's detectors do not, and one test
//! holds them to that.

use std::sync::Arc;

use serde_json::{json, Value};

use super::data::detect_run;
use super::lane_ad8_tests::stored;
use super::{judge, report, RankDetectors, Thresholds};

const BLOCKS: [&str; 4] = ["gradient_trend", "attention_trends", "update_trends", "weight_jumps"];

fn measurement_cases() -> Vec<(String, Value)> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../sdk/tests/fixtures/rules/cases.json");
    let document: Value = serde_json::from_str(&std::fs::read_to_string(path).expect("the page's cases")).unwrap();
    document["cases"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(_, body)| body["part"] == "measurements")
        .map(|(name, body)| (name.clone(), body.clone()))
        .collect()
}

fn judged(records: &[Value]) -> Value {
    let detectors = judge(0, &[stored(records)], &Arc::new(Thresholds::default())).unwrap();
    report("ad10", 0, &detectors, 0)
}

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
            // An integer stays an integer (counts and steps).
            if got.is_f64() != expected.is_f64() {
                out.push(format!("{at}: {got} expected {expected} (integer or not)"));
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

/// Where a report differs from what `expected` states of the block `key`: a null is a missing key.
fn block_differs(name: &str, report: &Value, key: &str, expected: &Value, out: &mut Vec<String>) {
    match (report.get(key), expected) {
        (None, Value::Null) => {}
        (Some(got), Value::Null) => out.push(format!("{name}: {key} is {got}, expected absent")),
        (None, _) => out.push(format!("{name}: {key} absent, expected {expected}")),
        (Some(got), _) => differs(&format!("{name}: {key}"), got, expected, out),
    }
}

/// The cases of one block, each held against the report `make` gives on its records.
fn hold(block: &str, make: fn(&[Value]) -> Value) -> usize {
    let mut out = Vec::new();
    let mut held = 0;
    for (name, body) in measurement_cases() {
        let Some(expected) = body["expected"].get(block) else {
            continue;
        };
        let records = body["records"].as_array().unwrap();
        block_differs(&name, &make(records), block, expected, &mut out);
        held += 1;
    }
    assert!(out.is_empty(), "{}", out.join("\n"));
    held
}

#[test]
fn every_gradient_trend_case_holds() {
    assert_eq!(hold("gradient_trend", judged), 4);
}

#[test]
fn every_attention_trends_case_holds() {
    assert_eq!(hold("attention_trends", judged), 2);
}

#[test]
fn every_update_trends_case_holds() {
    assert_eq!(hold("update_trends", judged), 2);
}

#[test]
fn every_weight_jumps_case_holds() {
    assert_eq!(hold("weight_jumps", judged), 2);
}

#[test]
fn every_measurement_case_holds_through_detect_run() {
    let held: usize = BLOCKS.iter().map(|block| hold(block, through_run)).sum();
    assert_eq!(held, 10);
}

/// Detectors that keep the trend series ([`RankDetectors::whole_run`], as `judge` makes them) and
/// read a rank one batch per row give the blocks of the whole read as one batch, and they are there.
#[test]
fn a_rank_read_row_by_row_gives_the_blocks_of_the_whole() {
    let mut out = Vec::new();
    for (name, body) in measurement_cases() {
        let records = body["records"].as_array().unwrap();
        let pieces: Vec<_> = records
            .iter()
            .map(|record| stored(std::slice::from_ref(record)))
            .collect();
        let detectors = judge(0, &pieces, &Arc::new(Thresholds::default())).unwrap();
        let piecewise = report("ad10", 0, &detectors, 0);
        let whole = judged(records);
        for block in BLOCKS {
            if piecewise.get(block) != whole.get(block) {
                out.push(format!(
                    "{name}: {block} {:?} != {:?}",
                    piecewise.get(block),
                    whole.get(block)
                ));
            }
            if let Some(expected) = body["expected"].get(block) {
                block_differs(&name, &piecewise, block, expected, &mut out);
            }
        }
    }
    assert!(out.is_empty(), "{}", out.join("\n"));
}

/// The live server's detectors ([`RankDetectors::new`], which `replay` makes and the live path
/// appends to) keep no trend series: their report has no `gradient_trend` and no
/// `attention_trends`, never a shorter trend, and the two bounded blocks as `detect` gives them.
#[test]
fn the_live_paths_detectors_keep_no_trend_series_and_give_the_bounded_blocks() {
    let mut out = Vec::new();
    let mut bounded = 0;
    for (name, body) in measurement_cases() {
        let records = body["records"].as_array().unwrap();
        let mut detectors = RankDetectors::new(&Arc::new(Thresholds::default()));
        for record in records {
            detectors.append(0, &stored(std::slice::from_ref(record)));
        }
        let live = report("ad10", 0, &detectors, 0);
        for block in ["gradient_trend", "attention_trends"] {
            if let Some(got) = live.get(block) {
                out.push(format!("{name}: the live report has {block} {got}"));
            }
        }
        for block in ["update_trends", "weight_jumps"] {
            if let Some(expected) = body["expected"].get(block) {
                block_differs(&name, &live, block, expected, &mut out);
                bounded += usize::from(!expected.is_null());
            }
            if live.get(block) != judged(records).get(block) {
                out.push(format!("{name}: {block} differs from detect's"));
            }
        }
    }
    assert_eq!(bounded, 2);
    assert!(out.is_empty(), "{}", out.join("\n"));
}

/// No rule reads a block: the incidents, `kind_checks` and `not_checked` of a run are the same with
/// and without the rows' weight checks, which only `weight_jumps` reads, and nothing outside the
/// block names it.
#[test]
fn a_block_changes_no_incident_and_no_check() {
    let body = measurement_cases()
        .into_iter()
        .find(|(name, _)| name == "weight_jumps_one_tensor_jumps")
        .unwrap()
        .1;
    let records = body["records"].as_array().unwrap();
    let with = judged(records);
    let without: Vec<Value> = records
        .iter()
        .map(|record| json!({"step": record["step"], "grad_norm": 1.0}))
        .collect();
    let with_norms: Vec<Value> = records
        .iter()
        .map(|record| json!({"step": record["step"], "grad_norm": 1.0, "weight_norms": record["weight_norms"]}))
        .collect();
    let (with_norms, without) = (judged(&with_norms), judged(&without));
    assert!(with.get("weight_jumps").is_some() && with_norms.get("weight_jumps").is_some());
    assert!(without.get("weight_jumps").is_none());
    for key in ["incidents", "kind_checks", "not_checked", "thresholds", "checked"] {
        assert_eq!(with_norms[key], without[key], "{key}");
    }
    for key in ["incidents", "kind_checks", "not_checked"] {
        let text = with[key].to_string();
        for block in BLOCKS {
            assert!(!text.contains(block), "{key} names {block}");
        }
    }
}

/// Python's `attention_trends` reads a probe's modules whether or not it says `error`; the
/// attention rules skip such a probe.
#[test]
fn a_probe_that_says_error_still_gives_its_modules_logits() {
    let records: Vec<Value> = (0..6)
        .map(|i| {
            json!({"step": 100 * i, "probe": {"error": "RuntimeError: x",
                   "attention": {"m": {"entropy_share": [0.5], "max_logit": [1.0 + i as f64, "nan", false]}}}})
        })
        .collect();
    let got = judged(&records);
    let expected = json!({"statistic": "max_logit", "modules": 1, "listed": [
        {"module": "m", "points": 6, "first_step": 0, "last_step": 500, "first": 1.5, "last": 5.5,
         "growth": 5.5 / 1.5, "per_100_steps": 100.0 * (4.0 / 400.0) / 1.5}]});
    let mut out = Vec::new();
    block_differs("error probes", &got, "attention_trends", &expected, &mut out);
    assert!(out.is_empty(), "{}", out.join("\n"));
}

/// A first third whose median is not above zero gives the trend with `growth` and
/// `per_100_steps` null, present as null; and such a module sorts after every other.
#[test]
fn a_first_third_not_above_zero_gives_null_growth_and_sorts_last() {
    let records: Vec<Value> = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        .iter()
        .enumerate()
        .map(|(i, norm)| json!({"step": i, "grad_norm": norm}))
        .collect();
    let mut out = Vec::new();
    let expected = json!({"series": "grad_norm", "points": 6, "first_step": 0, "last_step": 5,
                          "first": 0.0, "last": 3.5, "growth": null, "per_100_steps": null});
    block_differs("zero first", &judged(&records), "gradient_trend", &expected, &mut out);

    // Module "a" has negative logits (null growth), "z" a flat 2.0 (growth 1).
    let probes: Vec<Value> = (0..6)
        .map(|i| {
            json!({"step": i, "probe": {"attention": {
                "a": {"max_logit": [-1.0 - i as f64]}, "z": {"max_logit": [2.0]}}}})
        })
        .collect();
    let got = judged(&probes);
    let listed: Vec<&str> = got["attention_trends"]["listed"]
        .as_array()
        .map(|listed| listed.iter().map(|t| t["module"].as_str().unwrap()).collect())
        .unwrap_or_default();
    if listed != ["z", "a"] {
        out.push(format!("order {listed:?}, expected [\"z\", \"a\"]"));
    }
    if got["attention_trends"]["listed"][1]["growth"] != Value::Null {
        out.push(format!(
            "growth of a: {}",
            got["attention_trends"]["listed"][1]["growth"]
        ));
    }
    assert!(out.is_empty(), "{}", out.join("\n"));
}

/// `weight_jumps` lists the ten largest of twelve, ties by name, and counts all twelve.
#[test]
fn weight_jumps_lists_the_ten_largest_then_by_name() {
    // Tensor p<i> goes from 1 to 1 + (i / 2) + 1, so pairs tie: p10 and p11 the largest.
    let names: Vec<String> = (0..12).map(|i| format!("p{i:02}")).collect();
    let before: serde_json::Map<String, Value> = names.iter().map(|n| (n.clone(), json!(1.0))).collect();
    let after: serde_json::Map<String, Value> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), json!(2.0 + (i / 2) as f64)))
        .collect();
    let records = vec![
        json!({"step": 0, "weight_norms": before}),
        json!({"step": 7, "weight_norms": after}),
    ];
    let got = judged(&records);
    let block = &got["weight_jumps"];
    assert_eq!(block["checks"], 2);
    assert_eq!(block["parameters"], 12);
    let order: Vec<&str> = block["largest"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        order,
        ["p10", "p11", "p08", "p09", "p06", "p07", "p04", "p05", "p02", "p03"]
    );
    assert_eq!(
        block["largest"][0],
        json!({"name": "p10", "ratio": 7.0, "step": 7, "previous_step": 0, "before": 1.0, "after": 7.0})
    );
}

/// `weight_jumps` compares consecutive checks only: a tensor missing from the check between two
/// others is compared across neither gap, and a fall counts as a change (its inverse ratio).
#[test]
fn weight_jumps_compares_consecutive_checks_only() {
    let records = vec![
        json!({"step": 0, "weight_norms": {"a": 1.0, "b": 4.0}}),
        json!({"step": 10, "grad_norm": 1.0}),
        json!({"step": 20, "weight_norms": {"b": 2.0}}),
        json!({"step": 30, "weight_norms": {"a": 100.0, "b": 2.0}}),
    ];
    let got = judged(&records);
    assert_eq!(
        got["weight_jumps"],
        json!({"checks": 3, "parameters": 1, "largest": [
            {"name": "b", "ratio": 2.0, "step": 20, "previous_step": 0, "before": 4.0, "after": 2.0}]})
    );
}

/// `update_trends` judges a parameter from its eleventh finite value, against the median of its 50
/// finite values before (not of its whole history), and lists the ten lowest ratios, ties by name.
#[test]
fn update_trends_judges_against_the_last_fifty_and_lists_the_ten_lowest() {
    // "w": 60 values of 1.0, then 50 of 2.0, then 1.0. Against its last 50 the 1.0 is half; against
    // all 110 before it, whose median is 1.0, it would be 1.0 and not lower than the first ones.
    let mut records: Vec<Value> = (0..111)
        .map(|i| {
            let size = if i < 60 {
                1.0
            } else if i < 110 {
                2.0
            } else {
                1.0
            };
            json!({"step": i, "update_rms": {"w": size}})
        })
        .collect();
    // A NaN size and a skipped step: the NaN is no value, the skipped step counts.
    records[3]["update_rms"]["w"] = json!("nan");
    records[4]["step_skipped"] = json!(true);
    let got = judged(&records);
    assert_eq!(
        got["update_trends"],
        json!({"parameters": 1, "judged": 1, "lowest": [
            {"name": "w", "ratio": 0.5, "step": 110, "baseline": 2.0, "update_rms": 1.0}]})
    );

    // Twelve parameters, ten values of 1.0 each, then one value: only that eleventh is judged.
    let names: Vec<String> = (0..12).map(|i| format!("q{i:02}")).collect();
    let mut records: Vec<Value> = (0..10)
        .map(|i| json!({"step": i, "update_rms": names.iter().map(|n| (n.clone(), json!(1.0))).collect::<serde_json::Map<_, _>>()}))
        .collect();
    let last: serde_json::Map<String, Value> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), json!(1.0 - (i / 2) as f64 / 10.0)))
        .collect();
    records.push(json!({"step": 10, "update_rms": last}));
    let got = judged(&records);
    let block = &got["update_trends"];
    assert_eq!(block["parameters"], 12);
    assert_eq!(block["judged"], 12);
    let order: Vec<&str> = block["lowest"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        order,
        ["q10", "q11", "q08", "q09", "q06", "q07", "q04", "q05", "q02", "q03"]
    );

    // Ten values only: the parameter is counted and not judged, and the block is there.
    let got = judged(&records[..10]);
    assert_eq!(
        got["update_trends"],
        json!({"parameters": 12, "judged": 0, "lowest": []})
    );
}

/// `attention_trends` lists the ten modules of largest growth, ties by name, and counts every
/// module with a trend; a module with fewer than six probes has none and is not counted.
#[test]
fn attention_trends_lists_the_ten_of_largest_growth_then_by_name() {
    let records: Vec<Value> = (0..6)
        .map(|i| {
            let mut attention = serde_json::Map::new();
            for m in 0..12 {
                // Module m<j> grows by (j / 2) + 1 per probe from 1.0, so pairs tie.
                let rate = 1.0 + (m / 2) as f64;
                attention.insert(format!("m{m:02}"), json!({"max_logit": [1.0 + rate * i as f64]}));
            }
            if i < 5 {
                attention.insert("short".into(), json!({"max_logit": [1.0]}));
            }
            json!({"step": 10 * i, "probe": {"attention": attention}})
        })
        .collect();
    let got = judged(&records);
    let block = &got["attention_trends"];
    assert_eq!(block["modules"], 12);
    assert_eq!(block["statistic"], "max_logit");
    let order: Vec<&str> = block["listed"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["module"].as_str().unwrap())
        .collect();
    assert_eq!(
        order,
        ["m10", "m11", "m08", "m09", "m06", "m07", "m04", "m05", "m02", "m03"]
    );
}

/// `gradient_trend` reads `clip_grad_norm` once any row has it, so the rows before clipping began
/// are left out, and a norm that is not finite is left out too.
#[test]
fn gradient_trend_switches_to_the_pre_clip_norm_and_leaves_out_what_is_not_finite() {
    let mut records: Vec<Value> = (0..4).map(|i| json!({"step": i, "grad_norm": 100.0})).collect();
    records.extend((4..11).map(|i| json!({"step": i, "grad_norm": 1.0, "clip_grad_norm": i as f64})));
    records[6]["clip_grad_norm"] = json!("nan");
    let got = judged(&records);
    // Points: steps 4, 5, 7, 8, 9, 10 (6 is NaN). k = 2: steps 4.5 and 9.5, values 4.5 and 9.5.
    assert_eq!(
        got["gradient_trend"],
        json!({"series": "clip_grad_norm", "points": 6, "first_step": 4, "last_step": 10,
               "first": 4.5, "last": 9.5, "growth": 9.5 / 4.5, "per_100_steps": 100.0 * 1.0 / 4.5})
    );
}
