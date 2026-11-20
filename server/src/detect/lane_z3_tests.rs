//! Lane Z3 of the SDK-22 wave (Nov 20, 2026): the replica observation contract's cases
//! (`sdk/tests/fixtures/replica/`), read by the server's comparison (`detect/replica.rs`).

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Builder, Int64Array, ListBuilder, MapBuilder, StringArray, StringBuilder};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Value};

use super::replica::{compare, record_from_json, records_from_batch, Bounds, Comparison};

const FILES: [&str; 2] = ["no_incident_cases.json", "incident_cases.json"];

/// Every case: (file, name, records, expected, tolerance).
fn cases() -> Vec<(String, String, Vec<Value>, Value, f64)> {
    let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../sdk/tests/fixtures/replica");
    let mut out = Vec::new();
    for file in FILES {
        let text = std::fs::read_to_string(folder.join(file)).expect("the contract's cases");
        let document: Value = serde_json::from_str(&text).unwrap();
        let tolerance = document["tolerance"].as_f64().unwrap();
        for (name, body) in document["cases"].as_object().unwrap() {
            assert!(body["says"].is_string(), "{name}");
            out.push((
                file.to_string(),
                name.clone(),
                body["records"].as_array().unwrap().clone(),
                body["expected"].clone(),
                tolerance,
            ));
        }
    }
    out
}

/// An attempt's place in its rank's order: where it first appears in the case (contract 1's order).
fn first_seen(records: &[Value]) -> HashMap<(i64, i64), usize> {
    let mut places = HashMap::new();
    let mut next: HashMap<i64, usize> = HashMap::new();
    for record in records {
        let key = (record["rank"].as_i64().unwrap(), record["attempt"].as_i64().unwrap());
        if let std::collections::hash_map::Entry::Vacant(slot) = places.entry(key) {
            let at = next.entry(key.0).or_default();
            slot.insert(*at);
            *at += 1;
        }
    }
    places
}

#[test]
fn every_case_of_the_contract_gives_its_expected_result() {
    let all = cases();
    assert_eq!(all.len(), 39, "the contract has 39 cases");
    let mut wrong = Vec::new();
    for (file, name, records, expected, tolerance) in &all {
        let got = compare(records, *tolerance);
        if &got != expected {
            wrong.push(format!("{file}:{name}\n  got      {got}\n  expected {expected}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of 39 cases differ:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The server's records arrive in any order across ranks; a rank's attempts keep their place in
/// its order. Each case is fed in several orders, with the server's default bounds, and must give
/// the same result.
#[test]
fn every_case_gives_its_result_in_any_order_of_arrival() {
    for (file, name, records, expected, tolerance) in cases() {
        let places = first_seen(&records);
        let place = |rank: i64, attempt: i64| places.get(&(rank, attempt)).copied();
        let n = records.len();
        let mut orders: Vec<Vec<usize>> = vec![(0..n).collect(), (0..n).rev().collect()];
        orders.extend((1..n).map(|shift| (0..n).map(|at| (at + shift) % n).collect()));
        let mut by_rank: Vec<usize> = (0..n).collect();
        by_rank.sort_by_key(|at| std::cmp::Reverse(records[*at]["rank"].as_i64()));
        orders.push(by_rank);
        for order in orders {
            let mut comparison = Comparison::new(tolerance, Bounds::default());
            for at in &order {
                comparison.insert(record_from_json(&records[*at], tolerance).unwrap(), &place);
            }
            assert_eq!(comparison.result(), expected, "{file}:{name} in the order {order:?}");
        }
    }
}

/// One rank's records as a batch in the shape the server stores (`flight.py`'s `record_to_row`):
/// `replica_spread` in its map column, `replica` and `replica_spread_scope` in `extra_json`.
fn batch(records: &[&Value]) -> RecordBatch {
    let steps = Int64Array::from(records.iter().map(|r| r["step"].as_i64()).collect::<Vec<_>>());
    let attempts = Int64Array::from(records.iter().map(|r| r["attempt"].as_i64()).collect::<Vec<_>>());
    let mut spreads = MapBuilder::new(None, StringBuilder::new(), ListBuilder::new(Float64Builder::new()));
    let mut extra = Vec::new();
    for record in records {
        let mut extra_keys = serde_json::Map::new();
        for key in ["replica", "replica_spread_scope"] {
            if let Some(value) = record.get(key) {
                extra_keys.insert(key.into(), value.clone());
            }
        }
        match record.get("replica_spread").and_then(Value::as_object) {
            Some(table) => {
                for (name, values) in table {
                    spreads.keys().append_value(name);
                    for value in values.as_array().unwrap() {
                        spreads.values().values().append_value(value.as_f64().unwrap());
                    }
                    spreads.values().append(true);
                }
                spreads.append(true).unwrap();
            }
            None => spreads.append(false).unwrap(),
        }
        extra.push((!extra_keys.is_empty()).then(|| Value::Object(extra_keys).to_string()));
    }
    let spreads = spreads.finish();
    RecordBatch::try_from_iter(vec![
        ("step", Arc::new(steps) as ArrayRef),
        ("attempt", Arc::new(attempts) as ArrayRef),
        ("replica_spread", Arc::new(spreads) as ArrayRef),
        ("extra_json", Arc::new(StringArray::from(extra)) as ArrayRef),
    ])
    .unwrap()
}

/// The same cases read from stored batches, one batch per rank and attempt, ranks in turn.
#[test]
fn every_case_gives_its_result_from_stored_batches() {
    for (file, name, records, expected, tolerance) in cases() {
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
        assert_eq!(comparison.result(), expected, "{file}:{name}");
    }
}

fn record(rank: i64, step: i64, w: f64, group: &[i64]) -> Value {
    json!({"run_id": "r1", "rank": rank, "attempt": 1_795_000_000_000_i64 + rank, "step": step,
        "replica": {"v": 1, "launch": "2f6c1a9e4b7d4c1e9a3f6b8d0e1c2a47", "group": group, "device": "cuda",
            "projection": "name-v1", "signature": "fold64-v1",
            "params": {"w": {"shape": [8, 4], "dtype": "float32", "fp": [w, 1.0, 0.5], "sig": "7"}}}})
}

fn feed(comparison: &mut Comparison, records: &[Value]) {
    for record in records {
        comparison.insert(record_from_json(record, comparison.tolerance()).unwrap(), &|_, _| {
            Some(0)
        });
    }
}

/// Unfinished cells past the byte bound are let go: they stay `insufficient`, a rank that reports
/// for them later is counted and not compared, and nothing of them reads as `complete`.
#[test]
fn an_unfinished_step_past_the_byte_bound_stays_insufficient_and_says_so() {
    let one = record(0, 99, 3.5, &[0, 1]);
    let size = one["replica"].to_string().len();
    let bounds = Bounds {
        held_bytes: size * 2,
        ..Bounds::default()
    };
    let mut comparison = Comparison::new(1e-6, bounds);
    feed(
        &mut comparison,
        &[record(0, 99, 3.5, &[0, 1]), record(0, 199, 3.5, &[0, 1])],
    );
    assert_eq!(comparison.held_bytes(), size * 2);
    feed(&mut comparison, &[record(0, 299, 3.5, &[0, 1])]);
    assert!(comparison.held_bytes() <= size * 2);
    // Rank 1 arrives for all three; step 99 was let go, so only 199 and 299 are compared.
    feed(
        &mut comparison,
        &[
            record(1, 99, 3.6, &[0, 1]),
            record(1, 199, 3.6, &[0, 1]),
            record(1, 299, 3.5, &[0, 1]),
        ],
    );
    let result = comparison.result();
    let steps = result["steps"].as_array().unwrap();
    assert_eq!(steps[0]["step"], 99);
    assert_eq!(steps[0]["coverage"], "insufficient");
    assert_eq!(steps[0]["missing"], json!([1]));
    assert_eq!(
        (steps[1]["coverage"].clone(), steps[1]["fingerprints"].clone()),
        (json!("complete"), json!("drift"))
    );
    assert_eq!(steps[2]["fingerprints"], "agree");
    assert_eq!(
        result["omitted"]["frozen"],
        json!([{"launch": "2f6c1a9e4b7d4c1e9a3f6b8d0e1c2a47", "step": 99, "why": "held_bytes", "rows_not_compared": 1}])
    );
    let raised: Vec<i64> = result["replica_drift"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["step"].as_i64().unwrap())
        .collect();
    assert_eq!(raised, vec![199]);
    // Judged observations are let go too, once the bound needs the room; nothing is held past it.
    assert!(comparison.held_bytes() <= size * 2);
}

/// Steps past the listing bound are forgotten, least recently touched first; their incidents stay.
#[test]
fn steps_past_the_listing_bound_are_counted_and_their_incidents_kept() {
    let bounds = Bounds {
        listed_steps: 2,
        kept_incidents: 1,
        ..Bounds::default()
    };
    let mut comparison = Comparison::new(1e-6, bounds);
    for step in [99, 199, 299, 399] {
        feed(
            &mut comparison,
            &[record(0, step, 3.5, &[0, 1]), record(1, step, 3.6, &[0, 1])],
        );
    }
    let result = comparison.result();
    let listed: Vec<i64> = result["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["step"].as_i64().unwrap())
        .collect();
    assert_eq!(listed, vec![299, 399]);
    let raised: Vec<i64> = result["replica_drift"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["step"].as_i64().unwrap())
        .collect();
    assert_eq!(raised, vec![99, 299, 399]);
    assert_eq!(result["omitted"], json!({"steps": 2, "replica_drift": 1}));
}

/// A restart takes up what was written: the same result, no verdict or incident lost. A rank that
/// reports after it for a cell left `insufficient` is counted, not compared; a new launch is judged.
#[test]
fn a_restart_keeps_every_verdict_written_before_it() {
    for (file, name, records, expected, tolerance) in cases() {
        let mut written = json!({"v": 1, "tolerance": tolerance});
        written
            .as_object_mut()
            .unwrap()
            .extend(expected.as_object().unwrap().clone());
        let mut restored = Comparison::new(tolerance, Bounds::default());
        restored.restore(&written);
        assert_eq!(restored.result(), expected, "{file}:{name}");
        let _ = records;
    }
    let mut before = Comparison::new(1e-6, Bounds::default());
    feed(
        &mut before,
        &[
            record(0, 99, 3.5, &[0, 1]),
            record(0, 199, 3.5, &[0, 1]),
            record(1, 199, 3.6, &[0, 1]),
        ],
    );
    let written = before.status_object();
    let mut after = Comparison::new(1e-6, Bounds::default());
    after.restore(&written);
    assert_eq!(after.status_object(), written);
    let mut late = record(1, 99, 3.6, &[0, 1]);
    feed(&mut after, &[late.clone()]);
    let mut other = record(0, 99, 3.5, &[0, 1]);
    other["replica"]["launch"] = json!("b09d7e6f5a4c4b3a8f2e1d0c9b8a7f65");
    late["replica"]["launch"] = json!("b09d7e6f5a4c4b3a8f2e1d0c9b8a7f65");
    feed(&mut after, &[other, late]);
    let result = after.result();
    let at_99: Vec<&Value> = result["steps"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["step"] == 99)
        .collect();
    assert_eq!(at_99[0]["coverage"], "insufficient", "{result}");
    assert_eq!(at_99[1]["launch"], "b09d7e6f5a4c4b3a8f2e1d0c9b8a7f65");
    assert_eq!(at_99[1]["fingerprints"], "drift");
    assert_eq!(
        result["omitted"]["frozen"],
        json!([{"launch": "2f6c1a9e4b7d4c1e9a3f6b8d0e1c2a47", "step": 99, "why": "restart", "rows_not_compared": 1}])
    );
    assert_eq!(result["replica_drift"].as_array().unwrap().len(), 2);
}

/// A non-finite fingerprint written as a string reads as not finite; an unknown string too.
#[test]
fn a_fingerprint_written_as_a_string_is_read() {
    let mut a = record(0, 99, 3.5, &[0, 1]);
    let mut b = record(1, 99, 3.5, &[0, 1]);
    a["replica"]["params"]["w"]["fp"] = json!([3.5, "inf", 0.5]);
    b["replica"]["params"]["w"]["fp"] = json!([3.5, "1.0", 0.5]);
    let result = compare(&[a, b], 1e-6);
    assert_eq!(result["steps"][0]["fingerprints"], "drift");
}
