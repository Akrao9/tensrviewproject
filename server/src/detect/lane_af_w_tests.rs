//! Wave AF-W (Nov 21, 2026): `tensorquill-server detect --follow`, one rank's detectors kept between
//! looks for the SDK's `--watch`. A first frame is judged as `detect --samples` judges it; a later
//! frame of a plain append is read as the live server reads a flush, and the answer then is what
//! `detect --samples` gives on the whole; anything else is refused and the command ends. The SDK's
//! side, and every golden record set cut at its appends, are in `sdk/tests/test_lane_af_w_watch.py`.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float64Builder, Int32Array, Int64Array, Int64Builder, ListBuilder, MapBuilder, StringArray, StringBuilder,
};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use serde_json::Value;

use super::{detect_follow, detect_stream_samples, Thresholds};

fn usual() -> Arc<Thresholds> {
    Arc::default()
}

/// Rows of attempt `attempt` at `steps`, each with a gradient norm (a spike at `spike`) and the counts
/// of the routers in `routers` (name, experts).
fn rows(attempt: i64, steps: std::ops::Range<i64>, spike: Option<i64>, routers: &[(&str, usize)]) -> RecordBatch {
    let steps: Vec<i64> = steps.collect();
    let mut moe = MapBuilder::new(None, StringBuilder::new(), ListBuilder::new(Int64Builder::new()));
    let mut scalars = MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
    for step in &steps {
        for (name, experts) in routers {
            moe.keys().append_value(name);
            for expert in 0..*experts {
                moe.values()
                    .values()
                    .append_value(3 + ((*step as usize + expert) % 4) as i64);
            }
            moe.values().append(true);
        }
        moe.append(true).unwrap();
        scalars.keys().append_value("train/loss");
        scalars.values().append_value(1.0 / (*step as f64 + 1.0));
        scalars.append(true).unwrap();
    }
    let norms: Vec<f64> = steps
        .iter()
        .map(|step| 1.0 + 0.001 * (*step % 7) as f64 + if Some(*step) == spike { 40.0 } else { 0.0 })
        .collect();
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("schema_version", Arc::new(Int32Array::from(vec![1; steps.len()]))),
        (
            "run_id",
            Arc::new(StringArray::from_iter_values(steps.iter().map(|_| "w"))),
        ),
        ("rank", Arc::new(Int32Array::from(vec![0; steps.len()]))),
        ("attempt", Arc::new(Int64Array::from(vec![attempt; steps.len()]))),
        ("step", Arc::new(Int64Array::from(steps.clone()))),
        ("grad_norm", Arc::new(arrow::array::Float64Array::from(norms))),
        ("scalars", Arc::new(scalars.finish())),
        ("moe_counts", Arc::new(moe.finish())),
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

fn stream(batches: &[RecordBatch]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut writer = arrow::ipc::writer::StreamWriter::try_new(&mut out, &batches[0].schema()).unwrap();
    for batch in batches {
        writer.write(batch).unwrap();
    }
    writer.finish().unwrap();
    drop(writer);
    out
}

/// The frames `--follow` reads: each stream with its 8-byte little-endian length.
fn frames(looks: &[Vec<RecordBatch>]) -> Vec<u8> {
    let mut input = Vec::new();
    for look in looks {
        let bytes = stream(look);
        input.extend((bytes.len() as u64).to_le_bytes());
        input.extend(bytes);
    }
    input
}

/// `--follow`'s answer lines for `looks`, parsed.
fn follow(looks: &[Vec<RecordBatch>]) -> Vec<Value> {
    let mut out = Vec::new();
    detect_follow(frames(looks).as_slice(), &usual(), true, &mut out).unwrap();
    String::from_utf8(out)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

/// `detect --samples` on all of `batches`.
fn whole(batches: &[RecordBatch]) -> Value {
    let mut out = Vec::new();
    detect_stream_samples(stream(batches).as_slice(), &usual(), &mut out).unwrap();
    serde_json::from_slice(&out).unwrap()
}

/// An answer's report, without the samples and the `follow` key.
fn report(answer: &Value) -> Value {
    let mut report = answer.clone();
    let object = report.as_object_mut().unwrap();
    object.remove("samples");
    object.remove("follow");
    report
}

const ROUTERS: &[(&str, usize)] = &[("r0", 4), ("r1", 4)];

#[test]
fn an_append_reads_only_its_rows_and_answers_what_the_whole_gives() {
    let first = vec![rows(0, 0..30, Some(20), ROUTERS)];
    let more = vec![rows(0, 30..45, None, ROUTERS), rows(0, 45..50, Some(47), ROUTERS)];
    let answers = follow(&[first.clone(), more.clone()]);
    assert_eq!(answers.len(), 2);
    let all: Vec<RecordBatch> = first.iter().chain(&more).cloned().collect();
    let expected = whole(&all);
    // The first answer is `detect --samples` on the first frame, with the follow key.
    assert_eq!(report(&answers[0]), report(&whole(&first)));
    assert_eq!(answers[0]["samples"], whole(&first)["samples"]);
    assert_eq!(
        answers[0]["follow"],
        serde_json::json!({"mode": "full", "rows": 30, "samples_from": 0})
    );
    // The append: its 20 rows read, its samples only, and the report the whole gives.
    assert_eq!(
        answers[1]["follow"],
        serde_json::json!({"mode": "append", "rows": 20, "samples_from": 60})
    );
    assert_eq!(report(&answers[1]), report(&expected));
    let mut samples = answers[0]["samples"].as_array().unwrap().clone();
    samples.extend(answers[1]["samples"].as_array().unwrap().iter().cloned());
    assert_eq!(Value::Array(samples), expected["samples"]);
}

#[test]
fn an_append_after_a_resume_answers_what_the_whole_gives() {
    // Attempt 1 resumed at step 20 over attempt 0's 0..30; then attempt 1 goes on.
    let first = vec![rows(0, 0..30, None, ROUTERS), rows(1, 20..35, Some(25), ROUTERS)];
    let more = vec![rows(1, 35..50, None, ROUTERS)];
    let answers = follow(&[first.clone(), more.clone()]);
    let all: Vec<RecordBatch> = first.iter().chain(&more).cloned().collect();
    let expected = whole(&all);
    assert_eq!(answers[1]["follow"]["mode"], "append");
    assert_eq!(report(&answers[1]), report(&expected));
    let mut samples = answers[0]["samples"].as_array().unwrap().clone();
    samples.extend(answers[1]["samples"].as_array().unwrap().iter().cloned());
    assert_eq!(Value::Array(samples), expected["samples"]);
}

#[test]
fn anything_but_a_plain_append_is_refused_and_ends_the_command() {
    let first = vec![rows(0, 0..30, None, ROUTERS)];
    for (later, says) in [
        (rows(0, 29..31, None, ROUTERS), "not above the last step read"),
        (rows(1, 31..33, None, ROUTERS), "where the newest attempt read is 0"),
        (
            rows(0, 30..32, None, &[("r0", 4), ("r1", 4), ("r2", 4)]),
            "a router not read before",
        ),
        (
            rows(0, 30..32, None, &[("r0", 4), ("r1", 6)]),
            "one more experts than before",
        ),
    ] {
        // A third frame after the refusal is never read: the command ended.
        let answers = follow(&[first.clone(), vec![later], vec![rows(0, 40..41, None, ROUTERS)]]);
        assert_eq!(answers.len(), 2, "{says}");
        let refused = answers[1]["follow"]["refused"].as_str().unwrap();
        assert!(refused.contains(says), "{refused}");
        assert!(answers[1].get("incidents").is_none());
    }
}

#[test]
fn the_end_of_the_input_or_a_length_of_zero_ends_the_command() {
    let mut out = Vec::new();
    detect_follow(&[][..], &usual(), true, &mut out).unwrap();
    assert!(out.is_empty());
    let mut input = frames(&[vec![rows(0, 0..5, None, ROUTERS)]]);
    input.extend(0_u64.to_le_bytes());
    input.extend(frames(&[vec![rows(0, 5..6, None, ROUTERS)]]));
    let mut out = Vec::new();
    detect_follow(input.as_slice(), &usual(), true, &mut out).unwrap();
    assert_eq!(String::from_utf8(out).unwrap().lines().count(), 1);
    // A frame cut short is an error, not an answer.
    let mut cut = frames(&[vec![rows(0, 0..5, None, ROUTERS)]]);
    cut.truncate(cut.len() - 3);
    assert!(detect_follow(cut.as_slice(), &usual(), true, &mut Vec::new())
        .unwrap_err()
        .contains("ended inside a frame"));
}
