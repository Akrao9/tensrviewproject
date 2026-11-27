//! Lane AR-3's tests (Nov 26, 2026): AQ5-D2. An integer too large for a float inside `system_json`,
//! `data_json`, `probe_json` or `checkpoint_json` is read as wave AJ's reading of `extra_json` reads
//! it (`extra_json.rs`): the row keeps its other keys, and the value is unreadable wherever a number
//! is wanted. Before, one such integer made serde_json refuse the whole text, and the row lost the
//! whole column. Each expected value here is what Python's engine reports for the same records
//! (`sdk/tests/test_lane_ar3_json_columns.py` holds them through both engines).

use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Value};

use crate::detect::{judge, report, Thresholds};

/// The integer `10**400` as JSON text writes it.
fn big() -> String {
    format!("1{}", "0".repeat(400))
}

/// One row: its attempt, its step, and the text of each JSON column it has.
struct Row {
    attempt: i64,
    step: i64,
    columns: Vec<(&'static str, String)>,
}

fn row(step: i64, column: &'static str, text: &str) -> Row {
    Row {
        attempt: 1,
        step,
        columns: vec![(column, text.to_owned())],
    }
}

const COLUMNS: [&str; 4] = ["system_json", "data_json", "probe_json", "checkpoint_json"];

fn batch(rows: &[Row]) -> RecordBatch {
    let mut columns: Vec<(&str, ArrayRef)> = vec![
        (
            "run_id",
            Arc::new(StringArray::from_iter_values(rows.iter().map(|_| "run"))),
        ),
        ("rank", Arc::new(Int32Array::from_iter_values(rows.iter().map(|_| 0)))),
        (
            "attempt",
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.attempt))),
        ),
        (
            "step",
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.step))),
        ),
    ];
    for name in COLUMNS {
        let texts: Vec<Option<&str>> = rows
            .iter()
            .map(|row| {
                row.columns
                    .iter()
                    .find(|(column, _)| *column == name)
                    .map(|(_, text)| text.as_str())
            })
            .collect();
        columns.push((name, Arc::new(StringArray::from(texts))));
    }
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

fn report_of(rows: &[Row]) -> Value {
    let detectors = judge(0, &[batch(rows)], &Arc::<Thresholds>::default()).unwrap();
    report("run", 0, &detectors, 0)
}

/// Each kind's (value, records) in the report's `kind_checks`.
fn checks(report: &Value, kinds: &[&str]) -> Vec<(String, Value, Value)> {
    kinds
        .iter()
        .map(|kind| {
            let check = &report["kind_checks"][*kind];
            (kind.to_string(), check["value"].clone(), check["records"].clone())
        })
        .collect()
}

fn checked(kinds: &[&str], records: u64) -> Vec<(String, Value, Value)> {
    kinds
        .iter()
        .map(|kind| (kind.to_string(), json!("checked"), json!(records)))
        .collect()
}

/// The report's incidents of `kind`, each without its `scalars` and `class`.
fn incidents(report: &Value, kind: &str) -> Vec<Value> {
    report["incidents"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|incident| incident["kind"] == kind)
        .map(|incident| {
            let mut incident = incident.clone();
            let fields = incident.as_object_mut().unwrap();
            fields.remove("scalars");
            fields.remove("class");
            incident
        })
        .collect()
}

const MEMORY: [&str; 3] = ["memory_growth", "memory_pressure", "allocator_retries"];
const NVML: [&str; 2] = ["hardware_errors", "gpu_throttled"];
const DATA: [&str; 3] = ["data_stalled", "epoch_order_repeated", "resume_data_mismatch"];
const ATTENTION: [&str; 3] = ["attention_collapse", "attention_uniform", "attention_logit_growth"];

// --- system_json ----------------------------------------------------------------------------------

#[test]
fn d2_system_json_with_an_oversized_integer_is_read() {
    // AQ-5's smallest input (seeds 93, 340, 731, 750, 1490): the block is read, the count is none.
    let found = report_of(&[row(39, "system_json", &format!(r#"{{"cuda": {{"ooms": {}}}}}"#, big()))]);
    assert_eq!(checks(&found, &MEMORY), checked(&MEMORY, 1), "{found}");
    assert!(incidents(&found, "allocator_retries").is_empty(), "{found}");
}

#[test]
fn d2_system_json_keeps_the_valid_sibling_of_an_oversized_integer() {
    // Python: allocator_retries from `alloc_retries` 3, its `ooms` none (`_count` reads no integer
    // outside i64). Rust missed it.
    let found = report_of(&[row(
        39,
        "system_json",
        &format!(r#"{{"cuda": {{"ooms": {}, "alloc_retries": 3}}}}"#, big()),
    )]);
    assert_eq!(
        incidents(&found, "allocator_retries"),
        [json!({"kind": "allocator_retries", "step": 39, "alloc_retries": 3, "ooms": null})],
        "{found}"
    );
}

#[test]
fn d2_an_oversized_integer_is_not_a_throttle_reason() {
    // Python's reasons are the list's strings: the integer is left out, as `isinstance(r, str)`
    // leaves it out. Read as the unreadable text, it must not become a reason.
    let found = report_of(&[row(
        5,
        "system_json",
        &format!(
            r#"{{"nvml": {{"throttle": ["sw_thermal", {0}], "temperature_c": {0}, "power_w": 250.0}}}}"#,
            big()
        ),
    )]);
    assert_eq!(checks(&found, &NVML), checked(&NVML, 1), "{found}");
    assert_eq!(
        incidents(&found, "gpu_throttled"),
        [json!({
            "kind": "gpu_throttled", "step": 5, "reasons": ["sw_thermal"],
            "temperature_c": null, "sm_clock_mhz": null, "power_w": 250.0, "power_limit_w": null,
        })],
        "{found}"
    );
}

// --- data_json ------------------------------------------------------------------------------------

#[test]
fn d2_data_json_with_an_oversized_integer_is_read() {
    // AQ-5's smallest input (seeds 366, 729, 1352, 1413).
    let found = report_of(&[row(37, "data_json", &format!(r#"{{"sampler_epoch": {}}}"#, big()))]);
    assert_eq!(checks(&found, &DATA), checked(&DATA, 1), "{found}");
}

#[test]
fn d2_data_json_keeps_the_valid_siblings_of_an_oversized_integer() {
    // Python: the stall is found from the hash and the batch counts beside the oversized epoch.
    let text = format!(r#"{{"batches": 1, "batch_hash": "v2:aa", "sampler_epoch": {}}}"#, big());
    let found = report_of(&(0..4).map(|step| row(step, "data_json", &text)).collect::<Vec<_>>());
    assert_eq!(checks(&found, &DATA), checked(&DATA, 4), "{found}");
    assert_eq!(
        incidents(&found, "data_stalled"),
        [json!({
            "kind": "data_stalled", "step": 1, "reason": "same_batch", "through_step": 3, "batch_hash": "v2:aa",
        })],
        "{found}"
    );
}

#[test]
fn d2_an_oversized_integer_is_not_a_batch_hash() {
    // Python's hash is a non-empty string (`_hash`): an integer is no hash, so four such rows are no
    // stall. Read as the unreadable text, the four would share one hash.
    let text = format!(r#"{{"batches": 1, "batch_hash": {}}}"#, big());
    let found = report_of(&(0..4).map(|step| row(step, "data_json", &text)).collect::<Vec<_>>());
    assert_eq!(checks(&found, &DATA), checked(&DATA, 4), "{found}");
    assert!(incidents(&found, "data_stalled").is_empty(), "{found}");
}

// --- probe_json -----------------------------------------------------------------------------------

#[test]
fn d2_probe_json_with_an_oversized_integer_is_read() {
    // AQ-5's smallest input (seeds 482, 760, 1424): the list's oversized member is left out of the
    // largest logit, as Python's `_finite` leaves it out.
    let found = report_of(&[row(
        0,
        "probe_json",
        &format!(
            r#"{{"attention": {{"blk1": {{"max_logit": [3.943363681, -{}, 3.956723686]}}}}}}"#,
            big()
        ),
    )]);
    assert_eq!(checks(&found, &ATTENTION), checked(&ATTENTION, 1), "{found}");
    assert_eq!(
        found["attention_trends"]["listed"][0]["first"],
        json!(3.956723686),
        "{found}"
    );
}

#[test]
fn d2_an_attention_collapse_is_found_beside_an_oversized_logit() {
    // As at seed 1371, where Rust missed an attention_collapse Python reports: ten probes, the last
    // four collapsed, each with an oversized member in `max_logit`.
    let rows: Vec<Row> = (0..10)
        .map(|step| {
            let share = if step < 6 { "[0.9, 0.9]" } else { "[0.1, 0.1]" };
            let text = format!(
                r#"{{"attention": {{"blk1": {{"entropy_share": {share}, "max_logit": [3.9, -{}, 3.96]}}}}}}"#,
                big()
            );
            row(step, "probe_json", &text)
        })
        .collect();
    let found = report_of(&rows);
    assert_eq!(checks(&found, &ATTENTION), checked(&ATTENTION, 10), "{found}");
    let collapse = incidents(&found, "attention_collapse");
    assert_eq!(collapse.len(), 1, "{found}");
    let collapse = &collapse[0];
    assert_eq!(
        [
            &collapse["step"],
            &collapse["confirmed_step"],
            &collapse["affected_probes"],
            &collapse["probes"]
        ],
        [&json!(6), &json!(7), &json!(4), &json!(10)],
        "{collapse}"
    );
    let module = &collapse["modules"][0];
    assert_eq!(
        [
            &module["value"],
            &module["baseline"],
            &module["baseline_probes"],
            &module["heads"]
        ],
        [&json!(0.1), &json!(0.9), &json!(5), &json!(2)],
        "{collapse}"
    );
    let trend = &found["attention_trends"]["listed"][0];
    assert_eq!(
        [&trend["points"], &trend["first"], &trend["last"]],
        [&json!(10), &json!(3.96), &json!(3.96)],
        "{found}"
    );
}

#[test]
fn d2_a_head_probe_keeps_the_valid_siblings_of_an_oversized_integer() {
    // Python: each probe is read, `norm` is not a finite number in any of them, `mean` is.
    let rows: Vec<Row> = (0..3)
        .map(|step| {
            let text = format!(r#"{{"head": {{"norm": {}, "mean": {}}}}}"#, big(), 1.5 + step as f64);
            row(step, "probe_json", &text)
        })
        .collect();
    let found = report_of(&rows);
    assert_eq!(
        found["head_trends"],
        json!({
            "probes": 3, "read": 3, "module": null, "first_step": 0, "latest_step": 2,
            "numbers": {"mean": {"first": 1.5, "latest": 3.5, "lowest": 1.5, "highest": 3.5, "probes": 3}},
        }),
        "{found}"
    );
}

// --- checkpoint_json ------------------------------------------------------------------------------

#[test]
fn d2_checkpoint_json_with_an_oversized_integer_is_read() {
    // AQ-5's case found by hand: a save manifest is a checkpoint recorded.
    let found = report_of(&[row(
        0,
        "checkpoint_json",
        &format!(r#"{{"save": {{"after_step": {}}}}}"#, big()),
    )]);
    assert_eq!(
        checks(&found, &["resume_incomplete"]),
        checked(&["resume_incomplete"], 1),
        "{found}"
    );
}

#[test]
fn d2_a_resume_is_judged_from_a_load_manifest_beside_an_oversized_integer() {
    // Python: the resumed attempt's load manifest is read, and the empty optimizer is found.
    let mut rows: Vec<Row> = (0..3)
        .map(|step| Row {
            attempt: 1,
            step,
            columns: vec![],
        })
        .collect();
    rows.push(Row {
        attempt: 2,
        step: 2,
        columns: vec![(
            "checkpoint_json",
            format!(
                r#"{{"load": {{"components": {{"model": {{}}, "optimizer": {{"entries": 0}}, "rng": {{}}}}, "note": {}}}}}"#,
                big()
            ),
        )],
    });
    let found = report_of(&rows);
    assert_eq!(
        checks(&found, &["resume_incomplete"]),
        checked(&["resume_incomplete"], 1),
        "{found}"
    );
    let resume = incidents(&found, "resume_incomplete");
    assert_eq!(resume.len(), 1, "{found}");
    assert_eq!(
        resume[0]["changes"],
        json!([{
            "step": 2, "rank": 0, "attempt": 2, "source": "load",
            "findings": [{"component": "data", "finding": "missing"}, {"component": "optimizer", "finding": "empty_in_checkpoint"}],
        }]),
        "{found}"
    );
}

// --- what does not change -------------------------------------------------------------------------

#[test]
fn d2_a_text_that_parses_is_read_as_before() {
    // 2**64 is a number serde_json holds as a float, so no count; 1e308 is a finite float. AQ-5 found
    // both engines agree on these, and the reading must not touch them.
    let found = report_of(&[row(
        5,
        "system_json",
        r#"{"cuda": {"ooms": 18446744073709551616, "alloc_retries": 3}, "nvml": {"throttle": ["sw_thermal"], "power_w": 1e308}}"#,
    )]);
    assert_eq!(
        incidents(&found, "allocator_retries"),
        [json!({"kind": "allocator_retries", "step": 5, "alloc_retries": 3, "ooms": null})],
        "{found}"
    );
    assert_eq!(
        incidents(&found, "gpu_throttled")[0]["power_w"],
        json!(1e308),
        "{found}"
    );
}

#[test]
fn d2_a_text_that_is_not_json_for_another_reason_is_still_not_read() {
    // An oversized integer and a trailing comma: rewriting the integer does not make it JSON.
    for (column, kinds) in [
        ("system_json", &MEMORY[..]),
        ("data_json", &DATA[..]),
        ("probe_json", &ATTENTION[..]),
    ] {
        let text = match column {
            "system_json" => format!(r#"{{"cuda": {{"ooms": {}, "alloc_retries": 3,}}}}"#, big()),
            "data_json" => format!(r#"{{"sampler_epoch": {}, }}"#, big()),
            _ => format!(r#"{{"attention": {{"blk1": {{"max_logit": [{}, 3.9,]}}}}}}"#, big()),
        };
        let found = report_of(&[row(0, column, &text)]);
        for (kind, value, records) in checks(&found, kinds) {
            assert_eq!(
                (value, records),
                (json!("not_configured"), json!(0)),
                "{column} {kind}: {found}"
            );
        }
    }
    let found = report_of(&[row(
        0,
        "checkpoint_json",
        &format!(r#"{{"save": {{"after_step": {},}}}}"#, big()),
    )]);
    assert_eq!(
        found["kind_checks"]["resume_incomplete"]["records"],
        json!(0),
        "{found}"
    );
}
