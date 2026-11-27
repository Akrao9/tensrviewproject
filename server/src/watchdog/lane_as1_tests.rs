//! Lane AS-1 of wave AS: the run state's `ranks_unknown` reason held to the cases it shares with
//! the SDK (`sdk/tests/fixtures/ranks_unknown/cases.json`, read too by
//! `sdk/tests/test_lane_as1_ranks_unknown.py`). [`run_state`] lists the lowest [`LISTED_RANKS`]
//! unknown ranks, counts the others in `ranks_omitted`, written only when some are left out, and its
//! `detail` starts with the whole count. This is the shape lane AR-4 gave the server; the cases hold
//! Python's `run_state` to it. Tests only: the server's behaviour is unchanged.

use super::*;

fn document() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../sdk/tests/fixtures/ranks_unknown/cases.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    serde_json::from_str(&text).unwrap()
}

/// Spans `[first, last]`, both ends included, as the case file writes rank lists.
fn expand(spans: &Value) -> Vec<i64> {
    spans
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|span| span[0].as_i64().unwrap()..=span[1].as_i64().unwrap())
        .collect()
}

/// The status a case gives: one active entry of attempt 1 for each connected rank.
fn status_of(given: &Value) -> Value {
    let entries: Vec<Value> = given["connected"]
        .as_array()
        .unwrap()
        .iter()
        .map(|rank| json!({"rank": rank, "attempt": 1, "last_step": 19, "state": "active"}))
        .collect();
    let mut status = json!({
        "summary": "degraded", "world_size": given["world_size"], "ranks": entries,
        "missing_ranks": expand(&given["missing_ranks"]),
    });
    if let Some(omitted) = given.get("missing_ranks_omitted") {
        status["missing_ranks_omitted"] = omitted.clone();
    }
    status
}

/// A current report of attempt 1 for each judged rank.
fn reports_of(given: &Value) -> BTreeMap<i32, RankReport> {
    given["judged"]
        .as_array()
        .unwrap()
        .iter()
        .map(|rank| {
            let report = RankReport {
                kinds: Vec::new(),
                not_checked: Vec::new(),
                attempt: 1,
                last_step: Some(19),
                behind: false,
            };
            (i32::try_from(rank.as_i64().unwrap()).unwrap(), report)
        })
        .collect()
}

#[test]
fn the_cap_is_the_case_file_s() {
    assert_eq!(document()["cap"], json!(LISTED_RANKS));
}

#[test]
fn the_shared_cases_give_the_written_ranks_unknown() {
    let document = document();
    let cases = document["cases"].as_object().unwrap();
    assert_eq!(cases.len(), 6);
    for (name, case) in cases {
        let given = &case["given"];
        let state = run_state(&status_of(given), &reports_of(given), true);
        let reason = state["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .find(|reason| reason["code"] == "ranks_unknown")
            .unwrap_or_else(|| panic!("{name}: no ranks_unknown in {state}"));
        let expected = &case["expected"];
        assert_eq!(reason["ranks"], json!(expand(&expected["ranks"])), "{name}");
        assert_eq!(reason.get("ranks_omitted"), expected.get("ranks_omitted"), "{name}");
        let detail = reason["detail"].as_str().unwrap();
        assert!(
            detail.starts_with(expected["detail_starts"].as_str().unwrap()),
            "{name}: {detail}"
        );
    }
}
