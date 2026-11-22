//! Wave AH, P1-1: a rank that was already stopped when its stream dropped keeps its severity and
//! its incident while it reads `reconnecting` and after it reads `lost`; a healthy rank that
//! disconnects gets the ordinary grace. The shared cases are
//! `sdk/tests/fixtures/wave_ah/cases.json`, read by Python's `test_wave_ah_reconnecting.py` too.
//!
//! The clock: a silence floor of 60 s and critical at 180 s. Rank 1 stores step 9 at 1 s and
//! nothing after, so it is `silent` at warning from 61 s and critical from 181 s. Rank 0 stores a
//! batch every 10 s and stays active.

use super::*;

const RUN: &str = "run";

fn key(rank: i32) -> Key {
    (RUN.to_string(), rank)
}

fn at(base: Instant, seconds: u64) -> Instant {
    base + Duration::from_secs(seconds)
}

fn config() -> WatchdogConfig {
    WatchdogConfig {
        silent_floor: Duration::from_secs(60),
        silent_factor: 5.0,
        critical: Duration::from_secs(180),
        pause_limit: Duration::from_secs(600),
        start_limit: Duration::from_secs(600),
        abandon: Duration::from_secs(6000),
    }
}

/// Two ranks: rank 1 stores one batch at 1 s and then nothing; rank 0 steps until `until`.
fn two_ranks(t0: Instant, until: u64) -> Watchdog {
    let dog = Watchdog::new(config());
    for rank in 0..2 {
        dog.stream_opened(&key(rank), t0);
    }
    dog.batch(&key(1), 1, Some(9), Some(2), at(t0, 1));
    steps(&dog, t0, 1, until);
    dog
}

/// Rank 0 stores a batch every 10 s from `from` to `until`.
fn steps(dog: &Watchdog, t0: Instant, from: u64, until: u64) {
    for second in (from..=until).step_by(10) {
        dog.batch(&key(0), 1, Some(second as i64), Some(2), at(t0, second));
    }
}

fn status(dog: &Watchdog, now: Instant) -> Value {
    dog.statuses(now, true)[RUN].clone()
}

fn rank(status: &Value, rank: i64) -> Value {
    status["ranks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["rank"] == rank)
        .unwrap()
        .clone()
}

/// The rank's state, severity and `stopped_as` as the status gives them.
fn reads(status: &Value, number: i64) -> (String, Value, Value) {
    let entry = rank(status, number);
    (
        entry["state"].as_str().unwrap().to_string(),
        entry["severity"].clone(),
        entry.get("stopped_as").cloned().unwrap_or(Value::Null),
    )
}

/// The status's progress incident: kind, severity, assessment and the stopped rank numbers.
fn stall(status: &Value) -> Option<(String, String, String, Vec<i64>)> {
    let found = status["incidents"].as_array().unwrap();
    assert!(found.len() <= 1, "{found:?}");
    found.first().map(|incident| {
        let text = |key: &str| incident[key].as_str().unwrap().to_string();
        let ranks = incident["ranks"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|entry| is_stopped(entry))
            .map(|entry| entry["rank"].as_i64().unwrap())
            .collect();
        (text("kind"), text("severity"), text("assessment"), ranks)
    })
}

fn names(kind: &str, severity: &str, assessment: &str, ranks: &[i64]) -> Option<(String, String, String, Vec<i64>)> {
    Some((kind.into(), severity.into(), assessment.into(), ranks.to_vec()))
}

#[test]
fn a_healthy_rank_that_disconnects_gets_the_ordinary_grace() {
    let t0 = Instant::now();
    let dog = two_ranks(t0, 100);
    dog.stream_ended(&key(1), false, at(t0, 2));
    let grace = status(&dog, at(t0, 10));
    assert_eq!(reads(&grace, 1), ("reconnecting".into(), Value::Null, Value::Null));
    assert_eq!(grace["summary"], "active");
    assert_eq!(stall(&grace), None);
    // The whole floor after the drop, as before: nothing is held, whatever its last batch's age.
    let late = status(&dog, at(t0, 61));
    assert_eq!(reads(&late, 1), ("reconnecting".into(), Value::Null, Value::Null));
    assert_eq!(late["summary"], "active");
    // Then lost, counted from the drop.
    let lost = status(&dog, at(t0, 70));
    assert_eq!(reads(&lost, 1), ("lost".into(), json!("warning"), Value::Null));
    assert_eq!(lost["summary"], "degraded");
    assert_eq!(stall(&lost), names("rank_silent", "warning", "lost_connection", &[1]));
}

#[test]
fn a_critical_rank_that_disconnects_keeps_its_severity_and_its_incident() {
    let t0 = Instant::now();
    let dog = two_ranks(t0, 300);
    let before = status(&dog, at(t0, 200));
    assert_eq!(reads(&before, 1), ("silent".into(), json!("critical"), Value::Null));
    assert_eq!(stall(&before), names("rank_silent", "critical", "suspected_hang", &[1]));
    dog.stream_ended(&key(1), false, at(t0, 200));
    // `reconnecting` is the connection state; the stall it was in stays.
    let grace = status(&dog, at(t0, 210));
    assert_eq!(
        reads(&grace, 1),
        ("reconnecting".into(), json!("critical"), json!("silent"))
    );
    assert_eq!(rank(&grace, 1)["stream"], "dropped");
    assert_eq!(grace["summary"], "degraded");
    assert_eq!(stall(&grace), names("rank_silent", "critical", "suspected_hang", &[1]));
    assert_eq!(stall(&grace), stall(&before));
}

#[test]
fn a_warning_becomes_critical_during_the_grace_period_on_the_same_clock() {
    let t0 = Instant::now();
    let dog = two_ranks(t0, 300);
    dog.stream_ended(&key(1), false, at(t0, 150));
    let warning = status(&dog, at(t0, 160));
    assert_eq!(
        reads(&warning, 1),
        ("reconnecting".into(), json!("warning"), json!("silent"))
    );
    assert_eq!(stall(&warning), names("rank_silent", "warning", "suspected_hang", &[1]));
    // 189 s since its last step, inside the grace (which ends 60 s after the drop, at 210 s).
    let critical = status(&dog, at(t0, 190));
    assert_eq!(
        reads(&critical, 1),
        ("reconnecting".into(), json!("critical"), json!("silent"))
    );
    assert_eq!(
        stall(&critical),
        names("rank_silent", "critical", "suspected_hang", &[1])
    );
}

#[test]
fn when_the_grace_period_ends_the_rank_is_lost_and_its_severity_has_not_gone_down() {
    let t0 = Instant::now();
    let dog = two_ranks(t0, 500);
    dog.stream_ended(&key(1), false, at(t0, 200));
    // Never below critical from the drop on: not in the grace, not when it turns lost (a clock
    // started again at the drop would read warning from 261 s to 380 s).
    for second in (200..=500).step_by(5) {
        let now = status(&dog, at(t0, second));
        assert_eq!(rank(&now, 1)["severity"], "critical", "at {second} s");
        assert_eq!(now["summary"], "degraded", "at {second} s");
        assert_eq!(stall(&now).unwrap().1, "critical", "at {second} s");
    }
    let lost = status(&dog, at(t0, 270));
    assert_eq!(reads(&lost, 1), ("lost".into(), json!("critical"), json!("silent")));
    assert_eq!(stall(&lost), names("rank_silent", "critical", "lost_connection", &[1]));
}

#[test]
fn opening_a_stream_again_without_progress_clears_nothing() {
    let t0 = Instant::now();
    let dog = two_ranks(t0, 300);
    dog.stream_ended(&key(1), false, at(t0, 200));
    dog.stream_opened(&key(1), at(t0, 220));
    let back = status(&dog, at(t0, 221));
    assert_eq!(reads(&back, 1), ("silent".into(), json!("critical"), Value::Null));
    assert_eq!(stall(&back), names("rank_silent", "critical", "suspected_hang", &[1]));
    // A beat whose work counter has not moved, and a resend of rows already stored, are not progress.
    dog.heartbeat(&key(1), "train", 1.0, Some(10), None, at(t0, 222));
    dog.heartbeat(&key(1), "train", 1.0, Some(10), None, at(t0, 223));
    dog.batch(&key(1), 1, None, Some(2), at(t0, 224));
    let still = status(&dog, at(t0, 225));
    assert_eq!(reads(&still, 1), ("silent".into(), json!("critical"), Value::Null));
    assert_eq!(stall(&still), names("rank_silent", "critical", "suspected_hang", &[1]));
    // Dropping again holds it again, on the clock of its last step.
    dog.stream_ended(&key(1), false, at(t0, 230));
    let again = status(&dog, at(t0, 231));
    assert_eq!(
        reads(&again, 1),
        ("reconnecting".into(), json!("critical"), json!("silent"))
    );
    // And after the grace of the first drop, when it turned lost, a reopen is silent, critical.
    let dog = two_ranks(t0, 300);
    dog.stream_ended(&key(1), false, at(t0, 200));
    assert_eq!(reads(&status(&dog, at(t0, 270)), 1).0, "lost");
    dog.stream_opened(&key(1), at(t0, 271));
    assert_eq!(
        reads(&status(&dog, at(t0, 272)), 1),
        ("silent".into(), json!("critical"), Value::Null)
    );
}

#[test]
fn a_new_step_recovers_the_rank_by_the_rules_there_were() {
    let t0 = Instant::now();
    let dog = two_ranks(t0, 300);
    dog.stream_ended(&key(1), false, at(t0, 200));
    dog.stream_opened(&key(1), at(t0, 220));
    dog.batch(&key(1), 1, Some(19), Some(2), at(t0, 230));
    let recovered = status(&dog, at(t0, 231));
    assert_eq!(reads(&recovered, 1), ("active".into(), Value::Null, Value::Null));
    assert_eq!(recovered["summary"], "active");
    assert_eq!(stall(&recovered), None);
    // A drop now is a healthy rank's drop.
    dog.stream_ended(&key(1), false, at(t0, 232));
    let grace = status(&dog, at(t0, 240));
    assert_eq!(reads(&grace, 1), ("reconnecting".into(), Value::Null, Value::Null));
    assert_eq!(grace["summary"], "active");
}

#[test]
fn a_newer_attempt_that_takes_the_rank_clears_the_earlier_attempts_stall() {
    let t0 = Instant::now();
    let dog = two_ranks(t0, 300);
    dog.stream_ended(&key(1), false, at(t0, 200));
    assert_eq!(rank(&status(&dog, at(t0, 204)), 1)["severity"], "critical");
    assert!(dog.attempt_seen(&key(1), 2, at(t0, 205)));
    let taken = status(&dog, at(t0, 206));
    assert_eq!(reads(&taken, 1), ("reconnecting".into(), Value::Null, Value::Null));
    assert_eq!(rank(&taken, 1)["attempt"], 2);
    assert_eq!(taken["summary"], "active");
    assert_eq!(stall(&taken), None);
    // Its stream opens: a new attempt before its first step, under the start window.
    dog.stream_opened(&key(1), at(t0, 207));
    assert_eq!(
        reads(&status(&dog, at(t0, 208)), 1),
        ("starting".into(), Value::Null, Value::Null)
    );
}

#[test]
fn an_unresponsive_rank_that_disconnects_is_held_as_unresponsive() {
    let t0 = Instant::now();
    let dog = two_ranks(t0, 300);
    dog.heartbeat(&key(1), "train", 1.0, Some(10), None, at(t0, 1));
    assert_eq!(reads(&status(&dog, at(t0, 200)), 1).0, "unresponsive");
    dog.stream_ended(&key(1), false, at(t0, 200));
    let grace = status(&dog, at(t0, 210));
    assert_eq!(
        reads(&grace, 1),
        ("reconnecting".into(), json!("critical"), json!("unresponsive"))
    );
    assert_eq!(stall(&grace), names("rank_silent", "critical", "suspected_hang", &[1]));
}

#[test]
fn with_every_other_rank_finished_a_killed_critical_rank_leaves_the_job_stalled() {
    // Proof 1, phase C: rank 0 closed, rank 1 was critical and was then killed. Until wave AH the
    // run read `active` with no incident for the whole floor.
    let t0 = Instant::now();
    let dog = two_ranks(t0, 190);
    dog.stream_ended(&key(0), true, at(t0, 195));
    dog.stream_ended(&key(1), false, at(t0, 200));
    let grace = status(&dog, at(t0, 210));
    assert_eq!(reads(&grace, 0).0, "finished");
    assert_eq!(
        reads(&grace, 1),
        ("reconnecting".into(), json!("critical"), json!("silent"))
    );
    assert_eq!(grace["summary"], "stalled");
    assert_eq!(stall(&grace), names("job_stalled", "critical", "suspected_hang", &[1]));
}

#[test]
fn three_ranks_one_held_one_healthy_drop_and_one_stepping() {
    let t0 = Instant::now();
    let dog = two_ranks(t0, 300);
    dog.stream_opened(&key(2), t0);
    for second in (1..=198).step_by(10) {
        dog.batch(&key(2), 1, Some(second), Some(3), at(t0, second as u64));
    }
    dog.stream_ended(&key(1), false, at(t0, 200));
    dog.stream_ended(&key(2), false, at(t0, 200));
    let grace = status(&dog, at(t0, 210));
    assert_eq!(
        reads(&grace, 1),
        ("reconnecting".into(), json!("critical"), json!("silent"))
    );
    assert_eq!(reads(&grace, 2), ("reconnecting".into(), Value::Null, Value::Null));
    assert_eq!(grace["summary"], "degraded");
    assert_eq!(stall(&grace), names("rank_silent", "critical", "suspected_hang", &[1]));
    let others: Vec<i64> = grace["incidents"][0]["other_ranks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["rank"].as_i64().unwrap())
        .collect();
    assert_eq!(others, [0, 2]);
}

#[test]
fn a_clean_end_of_a_stopped_rank_is_finished_with_nothing_held() {
    let t0 = Instant::now();
    let dog = two_ranks(t0, 300);
    dog.stream_ended(&key(1), true, at(t0, 200));
    let done = status(&dog, at(t0, 201));
    assert_eq!(reads(&done, 1), ("finished".into(), Value::Null, Value::Null));
    assert_eq!(stall(&done), None);
}

#[test]
fn the_shared_cases_read_the_same_incident_as_python() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../sdk/tests/fixtures/wave_ah/cases.json");
    let document: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases = document["cases"].as_object().unwrap();
    assert!(cases.len() >= 9);
    for (name, case) in cases {
        let found: Vec<Value> = incidents(&case["status"])
            .iter()
            .map(|incident| {
                let numbers = |key: &str| -> Value {
                    incident[key].as_array().map_or(Value::Null, |ranks| {
                        ranks.iter().map(|entry| entry["rank"].clone()).collect()
                    })
                };
                let mut brief = json!({
                    "kind": incident["kind"], "severity": incident["severity"],
                    "assessment": incident["assessment"], "step": incident["step"], "ranks": numbers("ranks"),
                });
                if incident.get("other_ranks").is_some() {
                    brief["other_ranks"] = numbers("other_ranks");
                }
                if let Some(lost) = incident.get("lost_ranks") {
                    brief["lost_ranks"] = lost.clone();
                }
                brief
            })
            .collect();
        assert_eq!(json!(found), case["expected"], "{name}");
    }
}
