//! Lane AJ-3's tests (Nov 22, 2026): A8 of `OPEN_BUGS.md` section 15. The report's `moe_source`
//! says what Python's `evidence.moe_source` says for the same records: of each router, the first
//! entry that is an object, holding only its `form` and `point` (an empty object kept as it is);
//! an entry that is not an object is passed over, so a later record can still name the router.
//! `sdk/tests/test_lane_aj3_report.py` holds the same records through both engines.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float64Array, Float64Builder, Int32Array, Int64Array, ListBuilder, MapBuilder, StringArray, StringBuilder,
};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Value};

use super::{judge, report, Thresholds};

/// Rows of (attempt, step, `extra_json`).
fn rows(rows: &[(i64, i64, Value)]) -> RecordBatch {
    let columns: Vec<(&str, ArrayRef)> = vec![
        (
            "run_id",
            Arc::new(StringArray::from_iter_values(rows.iter().map(|_| "run"))),
        ),
        ("rank", Arc::new(Int32Array::from_iter_values(rows.iter().map(|_| 0)))),
        (
            "attempt",
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|(attempt, ..)| *attempt))),
        ),
        (
            "step",
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|(_, step, _)| *step))),
        ),
        (
            "extra_json",
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|(.., extra)| extra.to_string()),
            )),
        ),
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

/// The raw report of one attempt's rows, each row's `moe_source` as given (steps 0, 1, ...).
fn reported(sources: &[Value]) -> Value {
    let given: Vec<(i64, i64, Value)> = sources
        .iter()
        .enumerate()
        .map(|(step, source)| (1, step as i64, json!({"moe_source": source})))
        .collect();
    report_of(&given)
}

fn report_of(given: &[(i64, i64, Value)]) -> Value {
    let detectors = judge(0, &[rows(given)], &Arc::<Thresholds>::default()).unwrap();
    report("run", 0, &detectors, 0)
}

#[test]
fn a8_only_objects_and_only_form_and_point_are_kept() {
    // The case the author named: an extra key dropped, a string dropped, an empty object kept.
    let found = reported(&[json!({"r0": {"form": "scores", "extra": 123}, "r1": "bad", "r2": {}})]);
    assert_eq!(
        found["moe_source"],
        json!({"r0": {"form": "scores"}, "r2": {}}),
        "{found}"
    );
}

#[test]
fn a8_the_first_object_wins_and_a_non_object_does_not_hold_the_router() {
    let found = reported(&[
        json!({"r1": "bad"}),
        json!({"r1": {"form": "f", "point": "dispatched", "x": 1}}),
        json!({"r1": {"form": "g", "point": "requested"}}),
    ]);
    assert_eq!(
        found["moe_source"],
        json!({"r1": {"form": "f", "point": "dispatched"}}),
        "{found}"
    );
}

#[test]
fn a8_an_empty_object_first_still_wins_over_a_later_full_one() {
    let found = reported(&[json!({"r1": {}}), json!({"r1": {"form": "f", "point": "dispatched"}})]);
    assert_eq!(found["moe_source"], json!({"r1": {}}), "{found}");
}

#[test]
fn a8_nothing_kept_leaves_the_key_out_of_the_report() {
    let found = reported(&[json!({"r1": 5, "r2": "bad", "r3": [1], "r4": null}), json!("not a map")]);
    let report = found.as_object().unwrap();
    assert!(!report.contains_key("moe_source"), "{found}");
}

#[test]
fn a8_a_superseded_row_does_not_name_the_router() {
    // Attempt 1 reaches step 3; attempt 2 resumes at step 1, so attempt 1's step 2 does not count.
    let found = report_of(&[
        (1, 0, json!({"moe_source": {"r1": "bad"}})),
        (1, 2, json!({"moe_source": {"r1": {"form": "a", "point": "requested"}}})),
        (1, 3, json!({})),
        (
            2,
            1,
            json!({"moe_source": {"r1": {"form": "b", "point": "dispatched", "y": 2}}}),
        ),
        (2, 2, json!({})),
    ]);
    assert_eq!(
        found["moe_source"],
        json!({"r1": {"form": "b", "point": "dispatched"}}),
        "{found}"
    );
}

#[test]
fn a8_a_well_formed_source_is_kept_as_written() {
    let source = json!({"router.0": {"form": "scores", "point": "unknown"}, "router.1": {"form": "indices", "point": "dispatched"}});
    let found = reported(&[
        source.clone(),
        json!({"router.0": {"form": "other", "point": "requested"}}),
    ]);
    assert_eq!(found["moe_source"], source, "{found}");
}

// --- an incomplete `grad_norms` (the author, Nov 22, 2026) -----------------------------------
//
// `WAVE_AJ_CONTRACT.md` section 1 (lane AJ-1, 9cf73e7): `grad_norms` keeps a refused entry in the
// column as null, its raw value in `extra_json` under `grad_norms`; a name Arrow cannot encode
// cannot sit in the column even as null, and travels in `extra_json` alone. The rows here are
// built in both forms, so raw Rust is checked without the sender. The author's terms: valid
// sibling gradients stay visible to `non_finite`; when either side of an `update_spike`
// comparison had a refused gradient entry, the whole totals are compared (the existing
// fallback), so a missing entry cannot make a rise; a stalled weight's streak resets on a
// refused gradient, as Python's reading of the raw `true` does. The expected results are written
// out: the same records with the bad entry deleted are not a control here, since deleting it
// also deletes the evidence that the map was incomplete.

/// One row as the sender stores it: the typed `grad_norms` (None: the column is null; a None
/// norm: null), the refused entries in `extra_json` (None: no `extra_json`), `grad_norm`,
/// `update_rms` of `a`, and `weights_unchanged` (None: no weight check).
struct Row<'a> {
    grads: Option<Vec<(&'a str, Option<f64>)>>,
    refused: Option<Value>,
    grad_norm: Option<f64>,
    update: Option<f64>,
    unchanged: Option<Vec<&'a str>>,
}

fn row<'a>(grads: &[(&'a str, f64)], grad_norm: f64, update: f64) -> Row<'a> {
    Row {
        grads: Some(grads.iter().map(|(name, norm)| (*name, Some(*norm))).collect()),
        refused: None,
        grad_norm: Some(grad_norm),
        update: Some(update),
        unchanged: None,
    }
}

fn stored(given: &[Row]) -> RecordBatch {
    let n = given.len();
    let mut grads = MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
    let mut updates = MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
    let mut lists = ListBuilder::new(StringBuilder::new());
    for row in given {
        match &row.grads {
            Some(entries) => {
                for (name, norm) in entries {
                    grads.keys().append_value(name);
                    grads.values().append_option(*norm);
                }
                grads.append(true).unwrap();
            }
            None => grads.append(false).unwrap(),
        }
        match row.update {
            Some(size) => {
                updates.keys().append_value("a");
                updates.values().append_value(size);
                updates.append(true).unwrap();
            }
            None => updates.append(false).unwrap(),
        }
        match &row.unchanged {
            Some(names) => {
                for name in names {
                    lists.values().append_value(name);
                }
                lists.append(true);
            }
            None => lists.append(false),
        }
    }
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("run_id", Arc::new(StringArray::from_iter_values((0..n).map(|_| "run")))),
        ("rank", Arc::new(Int32Array::from_iter_values((0..n).map(|_| 0)))),
        ("attempt", Arc::new(Int64Array::from_iter_values((0..n).map(|_| 1)))),
        ("step", Arc::new(Int64Array::from_iter_values(0..n as i64))),
        (
            "grad_norm",
            Arc::new(Float64Array::from(
                given.iter().map(|row| row.grad_norm).collect::<Vec<_>>(),
            )),
        ),
        ("grad_norms", Arc::new(grads.finish())),
        ("update_rms", Arc::new(updates.finish())),
        ("weights_unchanged", Arc::new(lists.finish())),
        (
            "extra_json",
            Arc::new(StringArray::from(
                given
                    .iter()
                    .map(|row| {
                        row.refused
                            .as_ref()
                            .map(|refused| json!({"grad_norms": refused}).to_string())
                    })
                    .collect::<Vec<_>>(),
            )),
        ),
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

fn kind(given: &[Row], kind: &str) -> Vec<Value> {
    let thresholds = Arc::new(Thresholds::parse(r#"{"update_window": 3}"#).unwrap());
    let detectors = judge(0, &[stored(given)], &thresholds).unwrap();
    detectors
        .incidents()
        .into_iter()
        .filter(|incident| incident["kind"] == kind)
        .collect()
}

const ALL: [(&str, f64); 3] = [("a", 3.0), ("b", 4.0), ("c", 12.0)];
const SHARED: [(&str, f64); 2] = [("a", 3.0), ("b", 4.0)];

/// How `c`'s refused gradient is stored: as null in the column beside its raw value in
/// `extra_json`, or (a name Arrow cannot encode) in `extra_json` alone.
#[derive(Clone, Copy, Debug)]
enum Form {
    Null,
    ExtraOnly,
}

const FORMS: [Form; 2] = [Form::Null, Form::ExtraOnly];

/// A row naming a and b, with `c`'s gradient refused (raw `refused_as`) in `form`.
fn refused_c(
    form: Form,
    shared: &[(&'static str, f64)],
    refused_as: Value,
    grad_norm: f64,
    update: f64,
) -> Row<'static> {
    let mut row = row(shared, grad_norm, update);
    if let (Form::Null, Some(grads)) = (form, row.grads.as_mut()) {
        grads.push(("c", None));
    }
    row.refused = Some(json!({"c": refused_as}));
    row
}

/// Steps 0 to 3 name a, b and c (whole norm 13) with updates that change by 0.01, which fills a
/// window of 3; `last` is step 4, where the update jumps to 2.0.
fn steady_then(last: Row<'static>) -> Vec<Row<'static>> {
    let mut rows: Vec<Row> = [1.0, 1.01, 1.0, 1.01]
        .iter()
        .map(|size| row(&ALL, 13.0, *size))
        .collect();
    rows.push(last);
    rows
}

#[test]
fn incomplete_grads_on_the_later_side_compare_the_whole_totals() {
    // Step 4's whole norm is still 13 (c is 12 there too, refused as `true`). Whole totals: 13
    // against 13, no rise, so no threshold and no incident. Split over a and b alone it would be
    // 13 against 5 (a and b's share of step 3), a rise of 8 that flags the jump.
    for form in FORMS {
        let found = kind(
            &steady_then(refused_c(form, &SHARED, json!(true), 13.0, 2.0)),
            "update_spike",
        );
        assert!(found.is_empty(), "{form:?}: {found:?}");
    }
}

#[test]
fn a_parameter_left_out_without_a_refusal_is_still_split() {
    // The rule as it stands (B3, unchanged): with c simply not recorded at step 4 the totals are
    // split, 13 against 5, and the jump is flagged. This is the row above without its refusal,
    // and it reads differently on purpose.
    let found = kind(&steady_then(row(&SHARED, 13.0, 2.0)), "update_spike");
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0]["step"], 4);
    assert_eq!(found[0]["grad_norm"], 13.0);
    assert!(
        (found[0]["grad_norm_before"].as_f64().unwrap() - 5.0).abs() < 1e-12,
        "{}",
        found[0]
    );
}

#[test]
fn incomplete_grads_on_the_earlier_side_compare_the_whole_totals() {
    // Step 3 refused c; step 4 names all three with a whole norm of sqrt(2644). Whole totals:
    // sqrt(2644) against 13. Split, step 4 would read 50 (a and b alone) against 13.
    let whole = 2644f64.sqrt();
    for form in FORMS {
        let mut rows: Vec<Row> = [1.0, 1.01, 1.0].iter().map(|size| row(&ALL, 13.0, *size)).collect();
        rows.push(refused_c(form, &SHARED, json!("12"), 13.0, 1.01));
        rows.push(row(&[("a", 30.0), ("b", 40.0), ("c", 12.0)], whole, 2.0));
        let found = kind(&rows, "update_spike");
        assert_eq!(found.len(), 1, "{form:?}: {found:?}");
        assert_eq!(found[0]["step"], 4);
        assert_eq!(found[0]["grad_norm"], whole, "{form:?}");
        assert_eq!(found[0]["grad_norm_before"], 13.0, "{form:?}");
    }
}

#[test]
fn a_valid_sibling_of_a_refused_gradient_is_still_non_finite() {
    for form in FORMS {
        let mut grads = vec![("good", Some(f64::NAN))];
        if let Form::Null = form {
            grads.push(("bad", None));
        }
        let given = [Row {
            grads: Some(grads),
            refused: Some(json!({"bad": true})),
            grad_norm: Some(1.0),
            update: None,
            unchanged: None,
        }];
        let found = kind(&given, "non_finite");
        assert_eq!(found.len(), 1, "{form:?}: {found:?}");
        assert_eq!(found[0]["step"], 0);
        assert_eq!(found[0]["gradients"], json!(["good"]), "{form:?}: {}", found[0]);
    }
}

/// How `c`'s gradient is given at step 2 of four weight checks.
#[derive(Clone, Copy, Debug)]
enum AtTwo {
    Refused(Form),
    Absent,
    Null,
}

/// `c` listed unchanged at every check with a gradient of 0.5, `d` always changing; step 2 as `at_2`.
fn checks(at_2: AtTwo) -> Vec<Row<'static>> {
    (0..4)
        .map(|step| {
            let mut grads = vec![("d", Some(0.5))];
            let mut refused = None;
            match (step, at_2) {
                (2, AtTwo::Refused(form)) => {
                    if let Form::Null = form {
                        grads.push(("c", None));
                    }
                    refused = Some(json!({"c": true}));
                }
                (2, AtTwo::Absent) => {}
                (2, AtTwo::Null) => grads.push(("c", None)),
                _ => grads.push(("c", Some(0.5))),
            }
            Row {
                grads: Some(grads),
                refused,
                grad_norm: None,
                update: None,
                unchanged: Some(vec!["c"]),
            }
        })
        .collect()
}

#[test]
fn a_refused_gradient_resets_the_streak() {
    // The contract (section 1): a stalled weight's streak resets on a refused gradient, in either
    // form, as on a null one: c has only one check in a row at step 3, and no stall.
    for at_2 in [AtTwo::Refused(Form::Null), AtTwo::Refused(Form::ExtraOnly), AtTwo::Null] {
        let found = kind(&checks(at_2), "stalled_weights");
        assert!(found.is_empty(), "{at_2:?}: {found:?}");
    }
}

#[test]
fn an_absent_gradient_still_holds_the_streak() {
    // Lane W3's contract 2, unchanged: listed without a gradient (and nothing refused) holds the
    // streak, so c reaches three checks at step 3.
    let found = kind(&checks(AtTwo::Absent), "stalled_weights");
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0]["step"], 3);
    assert_eq!(found[0]["parameters"], json!(["c"]));
}
