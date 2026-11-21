//! Lane AD8 (Nov 21, 2026): the step capture stopped at, on the live path
//! (`DETECTOR_RULES_CONTRACT.md` part 7). The three `capture_stopped_*` cases of
//! `sdk/tests/fixtures/rules/cases.json`: each case's heartbeat, read as the `heartbeat` action
//! reads it, gives the rank's `capture_stopped` in the status the case states (`reason`, and `step`
//! as 0, as null, or with no key), and the same after a restart from the status file. The entry's
//! `seconds`, which no case states, is the only other key.

use super::*;

fn cases() -> Vec<(String, Value)> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../sdk/tests/fixtures/rules/cases.json");
    let document: Value = serde_json::from_str(&std::fs::read_to_string(path).expect("the page's cases")).unwrap();
    document["cases"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(name, _)| name.starts_with("capture_stopped_"))
        .map(|(name, body)| (name.clone(), body.clone()))
        .collect()
}

/// The rank's `capture_stopped` without `seconds`, which says how long ago and no case states.
fn stopped(status: &Value) -> Value {
    let mut entry = status["ranks"][0]["capture_stopped"].clone();
    let seconds = entry
        .as_object_mut()
        .expect("the rank's capture_stopped")
        .remove("seconds");
    assert!(seconds.is_some_and(|seconds| seconds.is_number()), "{entry}");
    entry
}

#[test]
fn every_capture_stopped_case_holds_in_the_status_and_over_a_restart() {
    let all = cases();
    assert_eq!(all.len(), 3);
    let mut wrong = Vec::new();
    for (name, body) in &all {
        let beat = &body["heartbeat"];
        let key: Key = (
            beat["run_id"].as_str().unwrap().to_string(),
            beat["rank"].as_i64().unwrap() as i32,
        );
        let t0 = Instant::now();
        let dog = Watchdog::new(WatchdogConfig::default());
        dog.stream_opened(&key, t0);
        assert!(dog.attempt_seen(&key, beat["attempt"].as_i64().unwrap(), t0));
        dog.heartbeat(&key, "train", 5.0, Some(0), None, t0);
        dog.batch(&key, 1, Some(9), Some(1), t0 + Duration::from_secs(1));
        let (reason, step) = crate::ingest::heartbeat_stop(beat.to_string().as_bytes()).expect("a stop");
        dog.capture_stopped_at(&key, &reason, step, t0 + Duration::from_secs(2));
        let now = t0 + Duration::from_secs(10);
        let status = dog.statuses(now, true)[&key.0].clone();
        let expected = &body["expected"]["status"]["capture_stopped"];
        if stopped(&status) != *expected {
            wrong.push(format!("{name}: {} expected {expected}", stopped(&status)));
        }
        // Restarted from the status file: the same three states.
        let file: Value = serde_json::from_slice(&serde_json::to_vec_pretty(&status).unwrap()).unwrap();
        let again = Watchdog::new(WatchdogConfig::default());
        assert_eq!(again.restore(&key.0, &file, now), Ok(true), "{name}");
        let restored = again.statuses(now, true)[&key.0].clone();
        if stopped(&restored) != *expected {
            wrong.push(format!(
                "{name} after a restart: {} expected {expected}",
                stopped(&restored)
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A second beat that says a stop keeps the first one's reason and step, as the reason was kept.
#[test]
fn the_first_stop_said_keeps_its_step() {
    let key: Key = ("run".to_string(), 0);
    let t0 = Instant::now();
    let dog = Watchdog::new(WatchdogConfig::default());
    dog.stream_opened(&key, t0);
    dog.capture_stopped_at(&key, "sink_error", Some(Some(0)), t0);
    dog.capture_stopped_at(&key, "writer_behind", Some(Some(7)), t0);
    dog.capture_stopped_at(&key, "sink_error", None, t0);
    let status = dog.statuses(t0, true)["run"].clone();
    assert_eq!(stopped(&status), json!({"reason": "sink_error", "step": 0}));
}

/// What a beat's `capture_stopped_step` reads as: a whole number of at least 0 is the step, null
/// is none committed, and anything else is read as null too (the key says the sender knows it, and
/// the value names no step); no key is an older sender's, which says nothing.
#[test]
fn a_beats_step_is_read_in_three_states() {
    let stop = |more: &str| {
        crate::ingest::heartbeat_stop(
            format!(r#"{{"run_id": "run", "rank": 0, "capture_stopped": "sink_error"{more}}}"#).as_bytes(),
        )
        .expect("a stop")
        .1
    };
    assert_eq!(stop(""), None);
    assert_eq!(stop(r#", "capture_stopped_step": null"#), Some(None));
    assert_eq!(stop(r#", "capture_stopped_step": 0"#), Some(Some(0)));
    assert_eq!(stop(r#", "capture_stopped_step": 41"#), Some(Some(41)));
    for other in ["-1", "2.5", "true", r#""3""#, "[3]"] {
        assert_eq!(
            stop(&format!(r#", "capture_stopped_step": {other}"#)),
            Some(None),
            "{other}"
        );
    }
    // A beat that says no stop has no step, whatever the key holds.
    let none = crate::ingest::heartbeat_stop(br#"{"run_id": "run", "rank": 0, "capture_stopped_step": 3}"#);
    assert_eq!(none, None);
}
