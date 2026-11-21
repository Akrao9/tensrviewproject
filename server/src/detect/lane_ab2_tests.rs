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
