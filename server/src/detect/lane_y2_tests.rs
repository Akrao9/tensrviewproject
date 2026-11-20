//! Lane Y2's tests (Nov 20, 2026): `update_spike` compares only the parameters both of the rows
//! it compares name (L13). The same records as `sdk/tests/test_lane_y2_detectors.py`, which also
//! holds this engine to Python's on them.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float64Array, Float64Builder, Int32Array, Int64Array, MapBuilder, StringArray, StringBuilder,
};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Value};

use super::{judge, Thresholds};

/// One row: its step, total gradient norm and each parameter's update RMS.
type Row = (i64, f64, Vec<(&'static str, f64)>);

fn batch(rows: &[Row]) -> RecordBatch {
    let mut sizes = MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
    for (_, _, own) in rows {
        for (name, size) in own {
            sizes.keys().append_value(name);
            sizes.values().append_value(*size);
        }
        sizes.append(true).unwrap();
    }
    let columns: Vec<(&str, ArrayRef)> = vec![
        (
            "run_id",
            Arc::new(StringArray::from_iter_values(rows.iter().map(|_| "y2"))),
        ),
        ("rank", Arc::new(Int32Array::from_iter_values(rows.iter().map(|_| 0)))),
        (
            "attempt",
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|_| 1))),
        ),
        ("step", Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0)))),
        (
            "grad_norm",
            Arc::new(Float64Array::from_iter_values(rows.iter().map(|r| r.1))),
        ),
        ("update_rms", Arc::new(sizes.finish())),
    ];
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
        .collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns.into_iter().map(|(_, a)| a).collect(),
    )
    .unwrap()
}

/// The largest update moves 0.001 a step; `extra` are more parameters, held still.
fn quiet(count: i64, start: i64, extra: &[(&'static str, f64)]) -> Vec<Row> {
    (start..start + count)
        .map(|at| {
            let mut own = vec![("big", 0.010 + 0.001 * (at % 2) as f64), ("small", 0.002)];
            own.extend_from_slice(extra);
            (at, 1.0, own)
        })
        .collect()
}

fn spikes(rows: &[Row], window: usize) -> Vec<Value> {
    let thresholds = Arc::new(Thresholds::parse(&json!({"update_window": window}).to_string()).unwrap());
    judge(0, &[batch(rows)], &thresholds)
        .unwrap()
        .incidents()
        .into_iter()
        .filter(|incident| incident["kind"] == "update_spike")
        .collect()
}

#[test]
fn l13_a_parameter_that_appears_raises_no_update_spike() {
    for window in [4_usize, 100] {
        let mut rows = quiet(window as i64 + 2, 0, &[]);
        rows.push((
            window as i64 + 2,
            5.0,
            vec![("big", 0.010), ("small", 0.002), ("late", 0.300)],
        ));
        assert_eq!(spikes(&rows, window), Vec::<Value>::new(), "window {window}");
    }
}

#[test]
fn l13_a_parameter_that_goes_away_raises_none_either() {
    let mut rows: Vec<Row> = (0..6)
        .map(|at| {
            let moved = 0.001 * (at % 2) as f64;
            (
                at,
                1.0,
                vec![("big", 0.010 + moved), ("small", 0.002), ("late", 0.300 + moved)],
            )
        })
        .collect();
    rows.push((6, 5.0, vec![("big", 0.010), ("small", 0.002)]));
    assert_eq!(spikes(&rows, 4), Vec::<Value>::new());
}

#[test]
fn l13_a_spike_on_a_parameter_both_rows_name_still_fires() {
    let mut rows = quiet(6, 0, &[]);
    rows.push((6, 5.0, vec![("big", 0.010), ("small", 0.300), ("late", 0.900)]));
    let found = spikes(&rows, 4);
    assert_eq!(found.len(), 1, "{found:?}");
    let hit = &found[0];
    assert_eq!(
        (&hit["step"], &hit["spike_steps"], &hit["spike_count"]),
        (&json!(6), &json!([6]), &json!(1))
    );
    // Over the two parameters both rows name: 0.3, not 0.9.
    assert_eq!((&hit["update_rms"], &hit["previous"]), (&json!(0.3), &json!(0.011)));
    assert!((hit["change"].as_f64().unwrap() - 0.289).abs() < 1e-12);
    let named: Vec<&Value> = hit["parameters"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| &p["name"])
        .collect();
    assert_eq!(named, [&json!("small"), &json!("big")]);
}

#[test]
fn l13_after_it_appears_the_new_parameter_is_compared_like_any_other() {
    let mut rows = quiet(6, 0, &[]);
    rows.push((6, 5.0, vec![("big", 0.010), ("small", 0.002), ("late", 0.300)]));
    // It moves 0.001 a step, and is then the largest on both sides.
    rows.extend((7..11).map(|at| {
        (
            at,
            1.0,
            vec![
                ("big", 0.010),
                ("small", 0.002),
                ("late", 0.300 + 0.001 * (at % 2) as f64),
            ],
        )
    }));
    rows.push((11, 5.0, vec![("big", 0.010), ("small", 0.002), ("late", 0.900)]));
    let found = spikes(&rows, 4);
    let seen: Vec<(Value, Value, Value)> = found
        .iter()
        .map(|s| (s["step"].clone(), s["update_rms"].clone(), s["previous"].clone()))
        .collect();
    assert_eq!(seen, [(json!(11), json!(0.9), json!(0.3))]);
    assert_eq!(found[0]["parameters"][0]["name"], "late");
}

#[test]
fn l13_rows_that_share_no_parameter_are_not_compared() {
    let mut rows = quiet(6, 0, &[]);
    rows.push((6, 5.0, vec![("other", 5.0)]));
    rows.push((7, 1.0, vec![("other", 5.001)]));
    rows.push((8, 5.0, vec![("other", 5.3)]));
    let found = spikes(&rows, 4);
    let seen: Vec<(Value, Value)> = found
        .iter()
        .map(|s| (s["step"].clone(), s["previous"].clone()))
        .collect();
    assert_eq!(seen, [(json!(8), json!(5.001))]);
}

/// Rows with only a `probe_json` column beside the keys.
fn probe_rows(rows: &[(i64, Option<Value>)]) -> RecordBatch {
    let columns: Vec<(&str, ArrayRef)> = vec![
        (
            "run_id",
            Arc::new(StringArray::from_iter_values(rows.iter().map(|_| "y2"))),
        ),
        ("rank", Arc::new(Int32Array::from_iter_values(rows.iter().map(|_| 0)))),
        (
            "attempt",
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|_| 1))),
        ),
        ("step", Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0)))),
        (
            "probe_json",
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| r.1.as_ref().map(Value::to_string))
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
        columns.into_iter().map(|(_, a)| a).collect(),
    )
    .unwrap()
}

/// L15: the report carries Python's `head_trends`, made from the probes' head entries.
#[test]
fn l15_the_report_carries_the_probes_head_trends() {
    let head = |norm: f64, more: Value| {
        let mut head = json!({"module": "lm_head", "norm": norm, "mean": 0.1, "gap": true});
        head.as_object_mut().unwrap().extend(more.as_object().unwrap().clone());
        json!({"head": head})
    };
    let rows = [
        (0, Some(head(4.0, json!({})))),
        (1, None),
        (2, Some(head(2.0, json!({"overflow_positions": 2})))),
        (3, Some(json!({"error": null, "head": {"norm": 100.0}}))),
        (
            4,
            Some(json!({"head": {"not_read": "no position", "overflow_positions": 5}})),
        ),
        (
            5,
            Some(head(3.0, json!({"module": "model.lm_head", "overflow_positions": 1.0}))),
        ),
    ];
    let detectors = judge(0, &[probe_rows(&rows)], &Arc::<Thresholds>::default()).unwrap();
    let report = super::report("y2", 0, &detectors, 0);
    assert_eq!(
        report["head_trends"],
        json!({
            "probes": 4, "read": 3, "module": "model.lm_head", "first_step": 0, "latest_step": 5,
            "numbers": {
                "norm": {"first": 4.0, "latest": 3.0, "lowest": 2.0, "highest": 4.0, "probes": 3},
                "mean": {"first": 0.1, "latest": 0.1, "lowest": 0.1, "highest": 0.1, "probes": 3},
            },
            "not_read": {"probes": 1, "first_step": 4, "reason": "no position"},
            "overflow_positions": {"probes": 2, "positions": 7, "most": 5, "first_step": 2, "latest_step": 4},
        })
    );
    let none = judge(
        0,
        &[probe_rows(&[(0, None), (1, Some(json!({"attention": {}})))])],
        &Arc::<Thresholds>::default(),
    )
    .unwrap();
    assert_eq!(super::report("y2", 0, &none, 0)["head_trends"], Value::Null);
}
