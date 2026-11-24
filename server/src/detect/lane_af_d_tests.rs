//! Lane D of wave AF (Nov 21, 2026): the offline `detect` and `detect --run` against section 1 of
//! `WAVE_AF_CONTRACT.md` (a record's identity fields), and section 5 (the offline report against
//! the live one). The binary's own exit status and line are in `sdk/tests/test_lane_af_d_detect.py`.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Builder, Int32Array, Int64Array, MapBuilder, StringArray, StringBuilder};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;

use super::{detect_run, detect_stream, Thresholds};

fn usual() -> Arc<Thresholds> {
    Arc::default()
}

/// A row's identity fields: (rank, attempt, step, schema_version), each of which may be null.
type Identity = (Option<i32>, Option<i64>, Option<i64>, Option<i32>);

/// One row per entry of `rows`, every column nullable, with a `train/loss` scalar.
fn batch(rows: &[Identity]) -> RecordBatch {
    let mut scalars = MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
    for _ in rows {
        scalars.keys().append_value("train/loss");
        scalars.values().append_value(1.0);
        scalars.append(true).unwrap();
    }
    let columns: Vec<(&str, ArrayRef)> = vec![
        (
            "schema_version",
            Arc::new(rows.iter().map(|row| row.3).collect::<Int32Array>()),
        ),
        (
            "run_id",
            Arc::new(StringArray::from_iter_values(rows.iter().map(|_| "r"))),
        ),
        ("rank", Arc::new(rows.iter().map(|row| row.0).collect::<Int32Array>())),
        (
            "attempt",
            Arc::new(rows.iter().map(|row| row.1).collect::<Int64Array>()),
        ),
        ("step", Arc::new(rows.iter().map(|row| row.2).collect::<Int64Array>())),
        ("scalars", Arc::new(scalars.finish())),
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

fn good(step: i64) -> Identity {
    (Some(0), Some(0), Some(step), Some(1))
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

/// Both offline modes' answer to `batches`: `detect` (one rank) and `detect --run`.
fn both(batches: &[RecordBatch]) -> (Result<serde_json::Value, String>, Result<serde_json::Value, String>) {
    (
        detect_stream(stream(batches).as_slice(), &usual()),
        detect_run(batches, &usual()),
    )
}

/// Section 1: a negative `rank`, `attempt` or `step`, and a `schema_version` that is not 0, 1 or 2,
/// refuse the input in both modes with one line that names the column, the value and the row
/// (counted across the batches given). Fails on the base (`1ca26d4`): each was judged, a report of
/// one step (and for the rank, `rank: -3`).
#[test]
fn a_negative_identity_value_or_an_unsupported_version_refuses_the_input_naming_column_value_and_row() {
    for (row, name, value) in [
        ((Some(-3), Some(0), Some(1), Some(1)), "rank", "-3"),
        ((Some(0), Some(-3), Some(1), Some(1)), "attempt", "-3"),
        ((Some(0), Some(0), Some(-3), Some(1)), "step", "-3"),
        ((Some(0), Some(0), Some(1), Some(99)), "schema_version", "99"),
        ((Some(0), Some(0), Some(1), Some(-3)), "schema_version", "-3"),
        ((Some(0), Some(0), Some(1), Some(3)), "schema_version", "3"),
    ] {
        // One record alone, as the case of `cases.json` gives it.
        let (one, run) = both(&[batch(&[row])]);
        for (mode, answer) in [("detect", one), ("detect --run", run)] {
            let error = answer.expect_err(&format!("{mode} judged {name} {value}"));
            assert!(
                error.starts_with(&format!("row 1: {name} is {value}")),
                "{mode}, {name} {value}: {error}"
            );
            assert!(!error.contains('\n'), "one line: {error}");
            if name == "schema_version" {
                // Protocol 5's row version 2 is read since wave AL (lane AL-3).
                assert!(error.contains("versions 0, 1 and 2"), "{error}");
            }
        }
    }
}

/// The row named is the row's place in the whole input, counted across its batches.
#[test]
fn the_row_named_is_counted_across_the_batches() {
    let batches = [
        batch(&[good(0), good(1)]),
        batch(&[(Some(0), Some(0), Some(-3), Some(1)), good(3)]),
    ];
    let (one, run) = both(&batches);
    for answer in [one, run] {
        assert!(answer.unwrap_err().starts_with("row 3: step is -3"));
    }
}

/// A null in a nullable key column names its row and column. `schema_version` is among the columns
/// checked for a null, which the base did not check (a null version was judged).
#[test]
fn a_null_identity_value_names_its_row_and_column() {
    for (row, name) in [
        ((None, Some(0), Some(1), Some(1)), "rank"),
        ((Some(0), None, Some(1), Some(1)), "attempt"),
        ((Some(0), Some(0), None, Some(1)), "step"),
        ((Some(0), Some(0), Some(1), None), "schema_version"),
    ] {
        let batches = [batch(&[good(0), row])];
        let (one, run) = both(&batches);
        for (mode, answer) in [("detect", one), ("detect --run", run)] {
            let error = answer.expect_err(&format!("{mode} judged a null {name}"));
            assert!(error.starts_with(&format!("row 2: {name} is null")), "{mode}: {error}");
        }
    }
}

/// Pins: valid values read as before. Version 0 (what `flight.py` writes for a record that names
/// none) and 1 are read; a stream with no `schema_version` column is read; rank, attempt and step
/// of 0 are read.
#[test]
fn valid_identity_values_are_judged_as_before() {
    let batches = [batch(&[(Some(0), Some(0), Some(0), Some(0)), good(1), good(2)])];
    let (one, run) = both(&batches);
    let one = one.unwrap();
    assert_eq!(
        (&one["rank"], &one["steps"]),
        (&serde_json::json!(0), &serde_json::json!(3))
    );
    assert_eq!(run.unwrap()["reports"][0]["steps"], 3);
    let without = batches[0].project(&[1, 2, 3, 4, 5]).unwrap();
    assert!(without.column_by_name("schema_version").is_none());
    let (one, run) = both(&[without]);
    assert_eq!(one.unwrap()["steps"], 3);
    assert_eq!(run.unwrap()["reports"][0]["steps"], 3);
}

// --- section 6: the cross-rank state's maps -------------------------------------------------------

mod replica_maps {
    use serde_json::{json, Value};

    use super::super::replica::{record_from_json, Bounds, Comparison};

    const LAUNCH: &str = "L";
    const LISTED: usize = 8;
    const STEPS: i64 = 400;

    fn observation(rank: i64, attempt: i64, step: i64, w: f64) -> Value {
        json!({"run_id": "r1", "rank": rank, "attempt": attempt, "step": step,
            "replica": {"v": 1, "launch": LAUNCH, "group": [0, 1, 2], "device": "cuda",
                "projection": "name-v1", "signature": "fold64-v1",
                "params": {"w": {"shape": [8, 4], "dtype": "float32", "fp": [w, 1.0, 0.5],
                    "sig": if w == 3.5 { "7" } else { "8" }}}}})
    }

    /// A long run of three ranks, `STEPS` steps, with the listing bound at `LISTED` steps: rank 1
    /// drifts every 50th step; near the end an earlier attempt of rank 1 (its backlog, first in its
    /// order) arrives for the last steps and takes the launch, so the steps it has rows at are judged
    /// again. Returns the comparison after each step.
    fn long_run(mut each: impl FnMut(i64, &Comparison)) -> Comparison {
        let bounds = Bounds {
            listed_steps: LISTED,
            ..Bounds::default()
        };
        let mut comparison = Comparison::new(1e-6, bounds);
        // Rank 1's attempt 5 comes before its attempt 11.
        let place = |rank: i64, attempt: i64| Some(if rank == 1 && attempt == 11 { 1 } else { 0 });
        for step in 0..STEPS {
            for rank in 0..3 {
                let w = if rank == 1 && step % 50 == 0 { 3.6 } else { 3.5 };
                comparison.insert(
                    record_from_json(&observation(rank, 10 + rank, step, w), 1e-6).unwrap(),
                    &place,
                );
            }
            if step >= STEPS - 4 {
                comparison.insert(record_from_json(&observation(1, 5, step, 3.5), 1e-6).unwrap(), &place);
            }
            each(step, &comparison);
        }
        comparison
    }

    /// RS-05: `rows_at` ((launch, rank) to the steps it has rows at, read only to judge again the
    /// listed steps of a launch whose owner changes) is pruned by the rule the comparison's steps are
    /// pruned by: a step forgotten past the listing bound leaves it. Its size stays within the listed
    /// steps; on the base (`1ca26d4`) it held every step of the run, one entry per launch, rank and
    /// step (1200 after 400 steps of 3 ranks; the backlog's rows are at steps rank 1 already has). `owners` and `forgotten` are not
    /// pruned (no rule to follow; the lane's notes): their sizes are asserted as they are.
    #[test]
    fn rows_at_stays_within_the_listed_steps_over_a_long_run() {
        let mut largest = 0;
        let comparison = long_run(|_, comparison| largest = largest.max(comparison.map_sizes().0));
        let (rows_at, owners, forgotten) = comparison.map_sizes();
        // At most the listed steps, for each launch and rank, and the step just being taken.
        assert!(
            largest <= (LISTED + 1) * 3 + 4,
            "rows_at held {largest} entries at most"
        );
        assert!(rows_at <= LISTED * 3 + 4, "rows_at holds {rows_at}");
        // One owner per launch and rank, whatever the length.
        assert_eq!(owners, 3);
        // One entry per step forgotten: kept for the run's length, by design (a late row is counted).
        assert_eq!(forgotten, STEPS as usize - LISTED);
    }

    /// The verdicts are the base's: the result of the same rows equals what `1ca26d4` gave, kept in
    /// `lane_af_d_replica_base.json` (written by this test on the base with `LANE_AF_D_WRITE_BASE` set).
    #[test]
    fn the_result_of_a_long_run_is_the_bases() {
        let comparison = long_run(|_, _| {});
        let result = comparison.result();
        let status = comparison.status_object();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/detect/lane_af_d_replica_base.json");
        if std::env::var_os("LANE_AF_D_WRITE_BASE").is_some() {
            std::fs::write(
                &path,
                serde_json::to_string_pretty(&json!({"result": result, "status": status})).unwrap(),
            )
            .unwrap();
        }
        let base: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(result, base["result"]);
        assert_eq!(status, base["status"]);
        // The case judges something: the drifted steps still listed, and the incidents kept.
        assert!(!result["replica_drift"].as_array().unwrap().is_empty(), "{result}");
    }
}

// --- the live report's trend memoization ---------------------------------------------------------

mod trend_memo {
    use std::sync::Arc;

    use serde_json::{json, Value};

    use super::super::lane_ad8_tests::stored;
    use super::super::{judge, report, RankDetectors, Thresholds};

    const BLOCKS: [&str; 4] = ["gradient_trend", "attention_trends", "update_trends", "weight_jumps"];

    /// A rank's rows at `steps`: a total norm on each step when `norms`, update sizes on each, and a
    /// probe of three attention modules every tenth step when `probes` (module `a.2` only every
    /// fortieth), so parts differ in which series receive points.
    fn rows(steps: std::ops::Range<i64>, norms: bool, probes: bool) -> Vec<Value> {
        steps
            .map(|step| {
                let x = step as f64;
                let wobble = 1.0 + ((step * 37) % 21) as f64 / 100.0;
                let mut record = json!({"attempt": 1, "step": step, "update_rms": {"w": 1e-3 * wobble}});
                if norms {
                    record["grad_norm"] = json!((1.0 + x / 50.0) * wobble);
                }
                if probes && step % 10 == 0 {
                    let mut attention =
                        json!({"a.0": {"max_logit": [2.0 + x / 20.0]}, "a.1": {"max_logit": [3.0 * wobble]}});
                    if step % 40 == 0 {
                        attention["a.2"] = json!({"max_logit": [1.0 + x / 7.0]});
                    }
                    record["probe"] = json!({"attention": attention});
                }
                record
            })
            .collect()
    }

    /// The parts of a run, each with what its rows give: some with every series, some with norms only,
    /// some with nothing a trend reads, and the first ones too short for any trend (absent blocks).
    fn parts() -> Vec<Vec<Value>> {
        vec![
            rows(0..4, true, true),
            rows(4..8, false, false),
            rows(8..30, true, true),
            rows(30..45, false, false),
            rows(45..70, true, false),
            rows(70..120, true, true),
            rows(120..130, false, true),
            rows(130..140, false, false),
            rows(140..200, true, true),
        ]
    }

    fn usual() -> Arc<Thresholds> {
        Arc::default()
    }

    /// The memoized live report equals, at every part, the report of detectors that read every row
    /// so far afresh, which find every trend once, as the base found them at every report. That
    /// includes the parts where a block is absent (too few points). Each report finds a trend again
    /// only for the series that received a point since the last: the count grows by exactly the
    /// series that changed, and by none after a part with nothing a trend reads.
    #[test]
    fn the_memoized_report_is_the_unmemoized_one_at_every_part_and_finds_only_changed_series_again() {
        let mut live = RankDetectors::new(&usual());
        let mut so_far: Vec<Value> = Vec::new();
        let mut absent = 0;
        let mut last = live.trend_computations();
        for (index, part) in parts().into_iter().enumerate() {
            let before = |series: &str, records: &[Value]| -> usize {
                match series {
                    "gradient" => records.iter().filter(|r| r.get("grad_norm").is_some()).count(),
                    module => records
                        .iter()
                        .filter(|r| r["probe"]["attention"].get(module).is_some())
                        .count(),
                }
            };
            let changed: Vec<&str> = ["gradient", "a.0", "a.1", "a.2"]
                .into_iter()
                .filter(|series| before(series, &part) > 0)
                .collect();
            let had: Vec<&str> = ["gradient", "a.0", "a.1", "a.2"]
                .into_iter()
                .filter(|series| before(series, &so_far) > 0 || before(series, &part) > 0)
                .collect();
            live.append(1, &stored(&part));
            so_far.extend(part);
            let memoized = report("memo", 0, &live, 0);
            let fresh = report("memo", 0, &judge(0, &[stored(&so_far)], &usual()).unwrap(), 0);
            assert_eq!(memoized, fresh, "part {index}");
            absent += BLOCKS.iter().filter(|block| memoized.get(**block).is_none()).count();
            // A report finds the trend of every series the part gave a point, and of none other. The
            // first report finds every series it has.
            let expected = if index == 0 { had.len() } else { changed.len() };
            let now = live.trend_computations();
            assert_eq!(now - last, expected as u64, "part {index}: changed {changed:?}");
            // A report made again with no new row finds nothing again.
            assert_eq!(report("memo", 0, &live, 0), memoized);
            assert_eq!(live.trend_computations(), now, "part {index}, the report made again");
            last = now;
        }
        assert!(absent > 0, "some part has an absent block");
    }

    /// The gradient series is cleared when clipping begins: the trend found before is not taken for
    /// the new series, though it may have as many points.
    #[test]
    fn a_series_cleared_at_the_switch_to_clipping_is_found_again() {
        let mut live = RankDetectors::new(&usual());
        let mut so_far: Vec<Value> = rows(0..12, true, false);
        live.append(1, &stored(&so_far));
        let before = report("memo", 0, &live, 0);
        assert_eq!(before["gradient_trend"]["series"], "grad_norm");
        let clipped: Vec<Value> = (12..24)
            .map(|step| json!({"attempt": 1, "step": step, "grad_norm": 1.0, "clip_grad_norm": 5.0 + step as f64}))
            .collect();
        live.append(1, &stored(&clipped));
        so_far.extend(clipped);
        let after = report("memo", 0, &live, 0);
        assert_eq!(after["gradient_trend"]["series"], "clip_grad_norm");
        assert_eq!(
            after,
            report("memo", 0, &judge(0, &[stored(&so_far)], &usual()).unwrap(), 0)
        );
    }
}

// --- wave AG, section 8: a router recorded with no tokens -----------------------------------------

mod moe_routing {
    use std::sync::Arc;

    use serde_json::{json, Value};

    use super::super::lane_ad8_tests::stored;
    use super::super::{RankDetectors, Thresholds};

    fn record(step: i64, a: [i64; 4], b: [i64; 4]) -> Value {
        json!({"attempt": 1, "step": step, "moe": {"blocks.1.router": b, "blocks.0.router": a},
               "moe_top_k": {"blocks.0.router": 2, "blocks.1.router": 2}})
    }

    fn entry(detectors: &RankDetectors) -> Option<Value> {
        detectors
            .not_checked()
            .into_iter()
            .find(|item| item["check"] == "moe_routing")
    }

    /// On a live rank the judgement is made on what has arrived: the entry is there while a router
    /// the rows name has had only zeros, names only the routers not read, sorted by name, and goes at
    /// a router's first tokens. While no router was read the two MoE kinds are `unavailable` with the
    /// entry's check and reason; once one is, they are judged (`checked`) and the entry names the rest.
    #[test]
    fn the_entry_is_there_while_a_router_had_only_zeros_and_goes_at_its_first_tokens() {
        let thresholds = Arc::new(Thresholds::default());
        let mut live = RankDetectors::new(&thresholds);
        live.append(1, &stored(&[record(0, [0; 4], [0; 4]), record(1, [0; 4], [0; 4])]));
        let reason = "router blocks.0.router recorded no tokens on any step: its routing was not read; \
                      router blocks.1.router recorded no tokens on any step: its routing was not read";
        assert_eq!(
            entry(&live),
            Some(
                json!({"check": "moe_routing", "kinds": ["moe_imbalance", "starved_experts"],
                        "routers": ["blocks.0.router", "blocks.1.router"], "reason": reason})
            )
        );
        let kinds = live.kind_checks();
        for kind in ["moe_imbalance", "starved_experts"] {
            assert_eq!(
                (&kinds[kind]["value"], &kinds[kind]["check"], &kinds[kind]["reason"]),
                (&json!("unavailable"), &json!("moe_routing"), &json!(reason)),
                "{kind}"
            );
        }
        // blocks.1.router routes tokens: the kinds are judged, and the entry names blocks.0.router.
        live.append(1, &stored(&[record(2, [0; 4], [5, 3, 2, 6])]));
        let only = "router blocks.0.router recorded no tokens on any step: its routing was not read";
        assert_eq!(entry(&live).unwrap()["routers"], json!(["blocks.0.router"]));
        assert_eq!(entry(&live).unwrap()["reason"], only);
        let kinds = live.kind_checks();
        assert_eq!(
            (&kinds["moe_imbalance"]["value"], &kinds["starved_experts"]["value"]),
            (&json!("checked"), &json!("checked"))
        );
        // Then blocks.0.router too: no entry.
        live.append(1, &stored(&[record(3, [4, 4, 4, 4], [5, 3, 2, 6])]));
        assert_eq!(entry(&live), None);
        // Its place: right after the refused routers' entry, before `update_size` (Python's order);
        // here with neither, after the system entries.
        let mut again = RankDetectors::new(&thresholds);
        again.append(1, &stored(&[record(0, [0; 4], [0; 4])]));
        let checks: Vec<Value> = again.not_checked().iter().map(|item| item["check"].clone()).collect();
        assert_eq!(checks.last(), Some(&json!("moe_routing")), "{checks:?}");
    }
}
