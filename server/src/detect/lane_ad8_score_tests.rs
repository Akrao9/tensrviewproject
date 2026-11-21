//! Lane AD8 (Nov 21, 2026): the eight cases of the score's arithmetic, `DETECTOR_RULES_CONTRACT.md`
//! part 1, `"part": "score"` in `sdk/tests/fixtures/rules/cases.json`, held to [`Score::of`]. A
//! case's `display` is what a reader shows, and the server writes no words: it is not checked here.
//! The cases over records, where the score sits beside each ratio, are in `lane_ad8_tests.rs`.

use serde_json::{json, Value};

use super::Score;

fn cases() -> Vec<(String, Value)> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../sdk/tests/fixtures/rules/cases.json");
    let document: Value = serde_json::from_str(&std::fs::read_to_string(path).expect("the page's cases")).unwrap();
    document["cases"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(_, body)| body["part"] == "score")
        .map(|(name, body)| (name.clone(), body.clone()))
        .collect()
}

/// The score as a report writes it beside a ratio: `score`, and `score_note` when it is null.
fn written(score: Score) -> Value {
    let mut hit = json!({});
    score.put(&mut hit);
    hit
}

#[test]
fn every_case_of_the_scores_arithmetic_holds() {
    let all = cases();
    assert_eq!(all.len(), 8, "part 1 has eight cases of its arithmetic");
    let mut wrong = Vec::new();
    for (name, body) in &all {
        // A window entry that is not a number (`"nan"`, null) is not a finite value: NaN here.
        let window: Vec<f64> = body["window"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap_or(f64::NAN))
            .collect();
        let got = written(Score::of(body["value"].as_f64().unwrap(), window, &mut Vec::new()));
        let mut expected = body["expected"].clone();
        expected.as_object_mut().unwrap().remove("display");
        let same = match (got["score"].as_f64(), expected["score"].as_f64()) {
            (Some(a), Some(b)) => {
                (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0) && got.as_object().unwrap().len() == 1
            }
            _ => got == expected,
        };
        if !same {
            wrong.push(format!("{name}: {got} expected {expected}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The number is the reference's to the last bit where it is a number: the same order of
/// arithmetic (`_median`, the MAD as the median of the distances, `1.4826 * MAD`).
#[test]
fn a_score_is_the_references_to_the_last_bit() {
    for (name, body) in cases() {
        let Some(expected) = body["expected"]["score"].as_f64() else {
            continue;
        };
        let window: Vec<f64> = body["window"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_f64)
            .collect();
        let Score::Number(got) = Score::of(body["value"].as_f64().unwrap(), window, &mut Vec::new()) else {
            panic!("{name}: no number");
        };
        assert_eq!(got.to_bits(), expected.to_bits(), "{name}: {got} expected {expected}");
    }
}

/// The window's order does not change the score: the median selects, it does not depend on order.
#[test]
fn the_order_of_the_earlier_values_does_not_matter() {
    let window: Vec<f64> = (0..50).map(|i| 10.0 + 0.1 * ((i * 3) % 7) as f64).collect();
    let mut reversed = window.clone();
    reversed.reverse();
    let mut scratch = Vec::new();
    assert_eq!(
        Score::of(25.0, window, &mut scratch),
        Score::of(25.0, reversed, &mut scratch)
    );
}
