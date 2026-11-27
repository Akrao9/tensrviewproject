//! Lane AR-4 of wave AR, follow-up 1: the watchdog walks no rank number above the server's limit of
//! 100,000 ranks, whether a sender names it as a world size or as its own rank, and a status lists at
//! most 1,000 missing ranks and counts the rest.
//!
//! On `ce9da4a` `run_status` took the expected ranks as the larger of the largest world size sent
//! and the highest rank seen plus one, and listed every one of them not seen in `missing_ranks`. A
//! world size of 2,147,483,647 made every status of the run walk two thousand million ranks and
//! build a list of about 8 GB; a lone rank 2,147,483,647 overflowed `highest + 1`. These tests show
//! the fault with a few hundred thousand ranks, which the base walks in well under a second: each
//! asserts what the status holds and how large it is, and only then, with a generous bound, how long
//! it took. `i32::MAX` is used once the walk is bounded.
//!
//! The clock: a silence floor of 60 s. The rank stores a step at 1 s and at 70 s, and the status is
//! written at 71 s, past the floor, so every rank expected and not seen is missing.

use super::*;

const RUN: &str = "run";

/// The most bytes the status of a one-rank run may take, whatever world size or rank it names.
const SMALL: usize = 64 * 1024;

/// A generous bound on the time a status takes to write: a second check, after the sizes, since
/// other builds share this machine.
const QUICK: Duration = Duration::from_secs(5);

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

/// A watchdog of one run whose `ranks` each stored a step at 1 s and at 70 s, naming `world_size`.
fn watched(t0: Instant, ranks: &[i32], world_size: Option<i32>) -> Watchdog {
    let dog = Watchdog::new(config());
    for rank in ranks {
        dog.stream_opened(&key(*rank), t0);
        dog.batch(&key(*rank), 1, Some(9), world_size, at(t0, 1));
        dog.batch(&key(*rank), 1, Some(19), world_size, at(t0, 70));
    }
    dog
}

/// The run's status at 71 s, with its run state as `ingest` adds it (no report: nothing judged),
/// and the time both took.
fn settled(ranks: &[i32], world_size: Option<i32>) -> (Value, Duration) {
    let t0 = Instant::now();
    let dog = watched(t0, ranks, world_size);
    let started = Instant::now();
    let mut status = dog.statuses(at(t0, 71), true).remove(RUN).unwrap();
    status["run_state"] = run_state(&status, &BTreeMap::new(), true);
    (status, started.elapsed())
}

fn size(status: &Value) -> usize {
    serde_json::to_vec(status).unwrap().len()
}

/// The `ranks_unknown` reason of a status's run state.
fn ranks_unknown(status: &Value) -> Value {
    status["run_state"]["reasons"]
        .as_array()
        .unwrap()
        .iter()
        .find(|reason| reason["code"] == "ranks_unknown")
        .cloned()
        .unwrap_or_else(|| panic!("no ranks_unknown in {}", status["run_state"]))
}

fn ranks(status: &Value) -> Vec<i64> {
    status["ranks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["rank"].as_i64().unwrap())
        .collect()
}

/// A world size over the limit is not one the server takes: the status writes none, walks no rank
/// for it, and reads the run as one with no world size. On the base, 399,999 missing ranks.
#[test]
fn a_world_size_over_the_limit_is_not_taken_and_its_ranks_are_not_walked() {
    let worlds = [400_000, 100_001];
    for world in worlds {
        let (status, took) = settled(&[0], Some(world));
        let seen = format!(
            "world size {world}: {} bytes, missing_ranks of {} entries",
            size(&status),
            status["missing_ranks"].as_array().map_or(0, Vec::len)
        );
        assert!(size(&status) < SMALL, "{seen}");
        assert_eq!(status["world_size"], Value::Null, "{seen}");
        assert_eq!(status["missing_ranks"], json!([]), "{seen}");
        assert!(status.get("missing_ranks_omitted").is_none(), "{seen}");
        assert_eq!(status["summary"], "active", "{seen}");
        assert!(ranks_unknown(&status).get("ranks_omitted").is_none(), "{seen}");
        assert!(took < QUICK, "{seen}: {took:?}");
    }
}

/// A rank over the limit, with no world size, names no rank below it as expected: the status lists
/// it and walks nothing for it. On the base, 400,000 missing ranks.
#[test]
fn a_rank_over_the_limit_with_no_world_size_is_not_walked() {
    let (status, took) = settled(&[400_000], None);
    let seen = format!(
        "{} bytes, missing_ranks of {} entries",
        size(&status),
        status["missing_ranks"].as_array().map_or(0, Vec::len)
    );
    assert!(size(&status) < SMALL, "{seen}");
    assert_eq!(ranks(&status), [400_000], "{seen}");
    assert_eq!(status["missing_ranks"], json!([]), "{seen}");
    assert_eq!(status["summary"], "active", "{seen}");
    // Beside a rank below the limit, it names no gap either: rank 0 alone is expected.
    let (status, _) = settled(&[0, 400_000], None);
    assert_eq!(status["missing_ranks"], json!([]));
    assert_eq!(status["summary"], "active");
    assert!(took < QUICK, "{seen}: {took:?}");
}

/// At the limit the world size is taken: its first 1,000 missing ranks are listed and the rest
/// counted in `missing_ranks_omitted`, the incident says the same, and every rank counts for the
/// verdict. The run state's `ranks_unknown` lists its first 1,000 ranks and counts the rest in
/// `ranks_omitted`, and its detail gives the whole count. On the base, 99,999 listed, none counted.
#[test]
fn a_world_size_at_the_limit_lists_the_first_thousand_missing_ranks_and_counts_the_rest() {
    let (status, took) = settled(&[0], Some(100_000));
    let seen = format!(
        "{} bytes, missing_ranks of {} entries",
        size(&status),
        status["missing_ranks"].as_array().map_or(0, Vec::len)
    );
    assert!(size(&status) < SMALL, "{seen}");
    assert_eq!(status["world_size"], 100_000, "{seen}");
    assert_eq!(
        status["missing_ranks"],
        json!((1..=1_000).collect::<Vec<i32>>()),
        "{seen}"
    );
    assert_eq!(status["missing_ranks_omitted"], 98_999, "{seen}");
    assert_eq!(status["summary"], "degraded", "{seen}");
    let incident = &status["incidents"][0];
    assert_eq!(incident["missing_ranks"], status["missing_ranks"], "{seen}");
    assert_eq!(incident["missing_ranks_omitted"], 98_999, "{seen}");
    let unknown = ranks_unknown(&status);
    assert_eq!(unknown["ranks"], json!((0..1_000).collect::<Vec<i32>>()), "{seen}");
    assert_eq!(unknown["ranks_omitted"], 99_000, "{seen}");
    assert!(
        unknown["detail"]
            .as_str()
            .unwrap()
            .starts_with("100000 ranks without a current report"),
        "{unknown}"
    );
    assert!(took < QUICK, "{seen}: {took:?}");
}

/// The same for a gap below the highest rank, with no world size: rank 99,999, the highest the
/// server takes, alone.
#[test]
fn a_gap_up_to_the_limit_is_listed_and_counted_the_same_way() {
    let (status, took) = settled(&[99_999], None);
    let seen = format!(
        "{} bytes, missing_ranks of {} entries",
        size(&status),
        status["missing_ranks"].as_array().map_or(0, Vec::len)
    );
    assert!(size(&status) < SMALL, "{seen}");
    assert_eq!(
        status["missing_ranks"],
        json!((0..1_000).collect::<Vec<i32>>()),
        "{seen}"
    );
    assert_eq!(status["missing_ranks_omitted"], 98_999, "{seen}");
    assert_eq!(status["summary"], "degraded", "{seen}");
    let unknown = ranks_unknown(&status);
    assert_eq!(unknown["ranks"], json!((0..1_000).collect::<Vec<i32>>()), "{seen}");
    assert_eq!(unknown["ranks_omitted"], 99_000, "{seen}");
    // A rank that connects is taken off the count, as off the list.
    let t0 = Instant::now();
    let dog = watched(t0, &[99_999, 5_000], None);
    let status = dog.statuses(at(t0, 71), true).remove(RUN).unwrap();
    assert_eq!(status["missing_ranks"], json!((0..1_000).collect::<Vec<i32>>()));
    assert_eq!(status["missing_ranks_omitted"], 98_998);
    assert!(took < QUICK, "{seen}: {took:?}");
}

/// Exactly 1,000 missing ranks, or 1,000 ranks unknown, are listed whole, with no new key: the status
/// of such a run is the same as before. One more is listed and counted.
#[test]
fn a_thousand_ranks_are_listed_whole_with_no_count() {
    // A world of 1,000 with rank 0 alone: 999 missing, and 1,000 unknown (rank 0 is not judged).
    let (status, _) = settled(&[0], Some(1_000));
    assert_eq!(status["missing_ranks"], json!((1..1_000).collect::<Vec<i32>>()));
    let unknown = ranks_unknown(&status);
    assert_eq!(unknown["ranks"], json!((0..1_000).collect::<Vec<i32>>()));
    assert!(unknown.get("ranks_omitted").is_none(), "{unknown}");
    // A world of 1,001: 1,000 missing, listed whole; 1,001 unknown, one counted.
    let (status, _) = settled(&[0], Some(1_001));
    assert_eq!(status["missing_ranks"], json!((1..=1_000).collect::<Vec<i32>>()));
    assert!(
        status.get("missing_ranks_omitted").is_none(),
        "{}",
        status["missing_ranks_omitted"]
    );
    assert!(status["incidents"][0].get("missing_ranks_omitted").is_none());
    let unknown = ranks_unknown(&status);
    assert_eq!(unknown["ranks"], json!((0..1_000).collect::<Vec<i32>>()));
    assert_eq!(unknown["ranks_omitted"], 1);
    // A world of 1,002: one missing rank counted, two unknown.
    let (status, _) = settled(&[0], Some(1_002));
    assert_eq!(status["missing_ranks"].as_array().unwrap().len(), 1_000);
    assert_eq!(status["missing_ranks_omitted"], 1);
    assert_eq!(ranks_unknown(&status)["ranks_omitted"], 2);
}

/// The status of a small run, written as before to the byte: ranks 0, 1 and 3 of a world of 6, past
/// the floor, so 2, 4 and 5 are missing and the run is degraded. Pinned on `ce9da4a`, with
/// `written_at_ms` set to 0.
#[test]
fn a_small_run_s_status_is_written_as_before_to_the_byte() {
    let (mut status, _) = settled(&[0, 1, 3], Some(6));
    status["written_at_ms"] = json!(0);
    let mut incident = status["incidents"][0].clone();
    incident["written_at_ms"] = json!(0);
    status["incidents"][0] = incident;
    assert_eq!(serde_json::to_string(&status).unwrap(), PINNED);
}

const PINNED: &str =
    "{\"abandoned\":false,\"critical_seconds\":180.0,\"incidents\":[{\"assessment\":\"ranks_never_connected\",\"class\":\"observation\",\"critical_seconds\":180.0,\"kind\":\"rank_silent\",\"missing_ranks\":[2,4,5],\"newest_step\":19,\"other_ranks\":[{\"attempt\":1,\"last_step\":19,\"rank\":0,\"state\":\"active\"},{\"attempt\":1,\"last_step\":19,\"rank\":1,\"state\":\"active\"},{\"attempt\":1,\"last_step\":19,\"rank\":3,\"state\":\"active\"}],\"ranks\":[],\"severity\":\"warning\",\"silent_floor_seconds\":60.0,\"step\":19,\"world_size\":6,\"written_at_ms\":0}],\"missing_ranks\":[2,4,5],\"pauses\":{},\"ranks\":[{\"attempt\":1,\"heartbeat_age_seconds\":null,\"last_step\":19,\"phase\":\"train\",\"phase_seconds\":71.0,\"rank\":0,\"seconds_since_progress\":1.0,\"severity\":null,\"state\":\"active\",\"stream\":\"open\",\"work\":null},{\"attempt\":1,\"heartbeat_age_seconds\":null,\"last_step\":19,\"phase\":\"train\",\"phase_seconds\":71.0,\"rank\":1,\"seconds_since_progress\":1.0,\"severity\":null,\"state\":\"active\",\"stream\":\"open\",\"work\":null},{\"attempt\":1,\"heartbeat_age_seconds\":null,\"last_step\":19,\"phase\":\"train\",\"phase_seconds\":71.0,\"rank\":3,\"seconds_since_progress\":1.0,\"severity\":null,\"state\":\"active\",\"stream\":\"open\",\"work\":null}],\"recommended_silent_seconds\":null,\"restart\":[{\"age_seconds\":71.0,\"attempt_without_steps\":false,\"dropped_rows\":{},\"gaps_seconds\":[1.0,69.0],\"heartbeat_interval_seconds\":0.0,\"pauses\":{},\"rank\":0,\"refused_batches\":{},\"since_change_seconds\":71.0,\"step_seconds\":[],\"stepped\":false,\"superseded_rows\":0,\"work\":null,\"work_still_seconds\":71.0},{\"age_seconds\":71.0,\"attempt_without_steps\":false,\"dropped_rows\":{},\"gaps_seconds\":[1.0,69.0],\"heartbeat_interval_seconds\":0.0,\"pauses\":{},\"rank\":1,\"refused_batches\":{},\"since_change_seconds\":71.0,\"step_seconds\":[],\"stepped\":false,\"superseded_rows\":0,\"work\":null,\"work_still_seconds\":71.0},{\"age_seconds\":71.0,\"attempt_without_steps\":false,\"dropped_rows\":{},\"gaps_seconds\":[1.0,69.0],\"heartbeat_interval_seconds\":0.0,\"pauses\":{},\"rank\":3,\"refused_batches\":{},\"since_change_seconds\":71.0,\"step_seconds\":[],\"stepped\":false,\"superseded_rows\":0,\"work\":null,\"work_still_seconds\":71.0}],\"run_id\":\"run\",\"run_state\":{\"reasons\":[{\"code\":\"no_steps\",\"detail\":\"no step has been judged yet\",\"state\":\"unknown\"},{\"code\":\"ranks_unknown\",\"detail\":\"6 ranks without a current report from this server: nothing is said about incidents on them\",\"ranks\":[0,1,2,3,4,5],\"state\":\"unknown\"},{\"code\":\"status_summary\",\"detail\":\"the server's status says \\\"degraded\\\"\",\"state\":\"unknown\",\"summary\":\"degraded\"},{\"code\":\"observations\",\"detail\":\"1 observation listed, not counted as failures: rank_silent\",\"kinds\":[\"rank_silent\"],\"state\":\"healthy\"}],\"state\":\"unknown\"},\"schema_version\":1,\"server_running\":true,\"silent_factor\":5.0,\"silent_floor_seconds\":60.0,\"start_seconds\":600.0,\"summary\":\"degraded\",\"world_size\":6,\"written_at_ms\":0}";

/// A status a store already holds, written by a server that took a world size and a rank over the
/// limit: a restarted server takes the run up with no world size and without that rank, and writes
/// its status small. On the base both were taken up and 399,998 ranks listed missing.
#[test]
fn a_restored_status_over_the_limit_is_taken_up_without_its_world_size_or_its_rank() {
    let t0 = Instant::now();
    let dog = watched(t0, &[0, 1], Some(2));
    let mut before = dog.statuses(at(t0, 71), true).remove(RUN).unwrap();
    over_the_limit(&mut before, 400_000, 400_000);
    let fresh = Watchdog::new(config());
    let now = Instant::now();
    assert_eq!(fresh.restore(RUN, &before, now), Ok(true));
    let started = Instant::now();
    let status = fresh.statuses(now, true).remove(RUN).unwrap();
    let took = started.elapsed();
    let seen = format!(
        "{} bytes, missing_ranks of {} entries",
        size(&status),
        status["missing_ranks"].as_array().map_or(0, Vec::len)
    );
    assert!(size(&status) < SMALL, "{seen}");
    assert_eq!(status["world_size"], Value::Null, "{seen}");
    assert_eq!(ranks(&status), [0], "{seen}");
    assert_eq!(status["missing_ranks"], json!([]), "{seen}");
    assert!(took < QUICK, "{seen}: {took:?}");
}

/// Makes `status` name `world_size` and makes its rank 1 rank `rank`, in its entries and its carries.
fn over_the_limit(status: &mut Value, world_size: i64, rank: i64) {
    status["world_size"] = json!(world_size);
    for list in ["ranks", "restart"] {
        for entry in status[list].as_array_mut().unwrap() {
            if entry["rank"] == 1 {
                entry["rank"] = json!(rank);
            }
        }
    }
}
