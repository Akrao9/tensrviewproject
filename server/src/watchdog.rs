//! A watchdog that does not depend on the training process: it notices that a rank stopped
//! making progress from what the server itself receives.
//!
//! Progress is a batch that carries a step above the last one kept for the rank. A resend
//! of old rows is not progress, so a sender stuck resending looks silent. Each rank is in
//! one of five states:
//!
//! - `active`: progress within its threshold. The threshold is the larger of a floor and a
//!   multiple of the rank's own recent time between batches, because a batch holds `every`
//!   steps and slow steps make batches far apart.
//! - `silent`: the stream is open but no progress for longer than the threshold. A hung
//!   process keeps its connection open, so this is what a hang looks like.
//! - `reconnecting`: the stream dropped without ending (a crash, or a network failure) a
//!   short while ago; the sender reconnects within seconds when its process is alive.
//! - `lost`: it dropped and did not come back for longer than the floor.
//! - `finished`: the stream ended cleanly, as `close()` does.
//!
//! The run's summary is `stalled` when no rank is active and some are silent or lost,
//! `degraded` when some are silent, lost or never connected while another is active,
//! `finished` when every rank finished, and `active` otherwise. The server writes it to
//! `runs/<run_id>/watchdog.json` every few seconds, using its own clock, so a reader sees
//! the verdict as of `written_at_ms` whenever it reads. It can only see ranks that
//! connected since the server started, plus the missing ranks implied by `world_size`
//! (sent by the SDK) or by gaps below the highest rank seen. The thresholds are not
//! calibrated: set `--silent-seconds` above the longest evaluation or checkpoint pause.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

pub const DEFAULT_SILENT_FLOOR: Duration = Duration::from_secs(300);
pub const DEFAULT_SILENT_FACTOR: f64 = 5.0;
pub const DEFAULT_WATCHDOG_INTERVAL: Duration = Duration::from_secs(10);
const GAPS_KEPT: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct WatchdogConfig {
    pub silent_floor: Duration,
    pub silent_factor: f64,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            silent_floor: DEFAULT_SILENT_FLOOR,
            silent_factor: DEFAULT_SILENT_FACTOR,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    Open,
    Finished,
    Dropped,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankState {
    Active,
    Silent,
    Reconnecting,
    Lost,
    Finished,
}

impl RankState {
    pub fn as_str(self) -> &'static str {
        match self {
            RankState::Active => "active",
            RankState::Silent => "silent",
            RankState::Reconnecting => "reconnecting",
            RankState::Lost => "lost",
            RankState::Finished => "finished",
        }
    }
}

#[derive(Debug)]
pub struct Progress {
    created: Instant,
    attempt: i64,
    last_step: Option<i64>,
    last_progress: Instant,
    last_change: Instant,
    gaps: VecDeque<f64>,
    open_streams: usize,
    stream: Stream,
    world_size: Option<i32>,
}

impl Progress {
    fn new(now: Instant) -> Self {
        Self {
            created: now,
            attempt: 0,
            last_step: None,
            last_progress: now,
            last_change: now,
            gaps: VecDeque::new(),
            open_streams: 0,
            stream: Stream::Open,
            world_size: None,
        }
    }

    /// The typical time between progress events: the median of the last few gaps.
    fn cadence(&self) -> Option<f64> {
        if self.gaps.is_empty() {
            return None;
        }
        let mut gaps: Vec<f64> = self.gaps.iter().copied().collect();
        gaps.sort_by(|a, b| a.total_cmp(b));
        Some(gaps[gaps.len() / 2])
    }

    pub fn state(&self, now: Instant, config: &WatchdogConfig) -> RankState {
        match self.stream {
            Stream::Finished => RankState::Finished,
            Stream::Dropped => {
                if now.saturating_duration_since(self.last_change) > config.silent_floor {
                    RankState::Lost
                } else {
                    RankState::Reconnecting
                }
            }
            Stream::Open => {
                let floor = config.silent_floor.as_secs_f64();
                let threshold = floor.max(self.cadence().map_or(0.0, |gap| gap * config.silent_factor));
                if now.saturating_duration_since(self.last_progress).as_secs_f64() > threshold {
                    RankState::Silent
                } else {
                    RankState::Active
                }
            }
        }
    }
}

type Key = (String, i32);

/// Progress of every rank that has connected, shared by the stream tasks and the writer.
pub struct Watchdog {
    config: WatchdogConfig,
    ranks: Mutex<HashMap<Key, Progress>>,
}

impl Watchdog {
    pub fn new(config: WatchdogConfig) -> Self {
        Self {
            config,
            ranks: Mutex::new(HashMap::new()),
        }
    }

    fn with<R>(&self, key: &Key, now: Instant, f: impl FnOnce(&mut Progress) -> R) -> R {
        let mut ranks = self.ranks.lock().unwrap();
        f(ranks.entry(key.clone()).or_insert_with(|| Progress::new(now)))
    }

    pub fn stream_opened(&self, key: &Key, now: Instant) {
        self.with(key, now, |p| {
            p.open_streams += 1;
            p.stream = Stream::Open;
            p.last_change = now;
        });
    }

    /// `clean` is true when the sender ended the stream itself, as `close()` does.
    pub fn stream_ended(&self, key: &Key, clean: bool, now: Instant) {
        self.with(key, now, |p| {
            p.open_streams = p.open_streams.saturating_sub(1);
            if p.open_streams == 0 {
                p.stream = if clean { Stream::Finished } else { Stream::Dropped };
                p.last_change = now;
            }
        });
    }

    /// A batch arrived. `new_last` is the highest step among the rows kept, `None` when every
    /// row was one the server already had.
    pub fn batch(&self, key: &Key, attempt: i64, new_last: Option<i64>, world_size: Option<i32>, now: Instant) {
        self.with(key, now, |p| {
            if world_size.is_some() {
                p.world_size = world_size;
            }
            if let Some(step) = new_last {
                let gap = now.saturating_duration_since(p.last_progress).as_secs_f64();
                if p.last_step.is_some() {
                    p.gaps.push_back(gap);
                    if p.gaps.len() > GAPS_KEPT {
                        p.gaps.pop_front();
                    }
                }
                p.attempt = attempt;
                p.last_step = Some(p.last_step.map_or(step, |last| last.max(step)));
                p.last_progress = now;
            }
        });
    }

    /// One status document per run, as of `now`. `running` is false for the last write at
    /// shutdown, which says the server stopped: the verdicts it holds would otherwise stay on
    /// disk as if they were current.
    pub fn statuses(&self, now: Instant, running: bool) -> BTreeMap<String, Value> {
        let written_at_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_millis() as i64);
        let ranks = self.ranks.lock().unwrap();
        let mut runs: BTreeMap<&str, Vec<(i32, &Progress)>> = BTreeMap::new();
        for ((run_id, rank), progress) in ranks.iter() {
            runs.entry(run_id.as_str()).or_default().push((*rank, progress));
        }
        runs.into_iter()
            .map(|(run_id, mut members)| {
                members.sort_by_key(|(rank, _)| *rank);
                (
                    run_id.to_string(),
                    run_status(run_id, &members, now, written_at_ms, &self.config, running),
                )
            })
            .collect()
    }
}

/// The status document of one run. Pure, so tests can give it any clock.
pub fn run_status(
    run_id: &str,
    members: &[(i32, &Progress)],
    now: Instant,
    written_at_ms: i64,
    config: &WatchdogConfig,
    running: bool,
) -> Value {
    let world_size = members.iter().filter_map(|(_, p)| p.world_size).max();
    let highest = members.iter().map(|(rank, _)| *rank).max().unwrap_or(-1);
    let expected = world_size.map_or(highest + 1, |size| size.max(highest + 1));
    let seen: Vec<i32> = members.iter().map(|(rank, _)| *rank).collect();
    // Ranks start a moment apart, so a rank is missing only once the run has been seen for
    // longer than the silence floor.
    let first_seen = members.iter().map(|(_, p)| p.created).min();
    let settled = first_seen.is_some_and(|first| now.saturating_duration_since(first) > config.silent_floor);
    let missing: Vec<i32> = if settled {
        (0..expected).filter(|rank| !seen.contains(rank)).collect()
    } else {
        Vec::new()
    };
    let states: Vec<RankState> = members.iter().map(|(_, p)| p.state(now, config)).collect();
    let count = |wanted: &[RankState]| states.iter().filter(|state| wanted.contains(state)).count();
    let active = count(&[RankState::Active, RankState::Reconnecting]);
    let stopped = count(&[RankState::Silent, RankState::Lost]);
    let all_finished = !members.is_empty() && count(&[RankState::Finished]) == members.len();
    let summary = if !running {
        "server_stopped"
    } else if all_finished && missing.is_empty() {
        "finished"
    } else if all_finished {
        "degraded"
    } else if active == 0 && stopped > 0 {
        "stalled"
    } else if stopped > 0 || !missing.is_empty() {
        "degraded"
    } else {
        "active"
    };
    let ranks: Vec<Value> = members
        .iter()
        .zip(&states)
        .map(|((rank, p), state)| {
            json!({
                "rank": rank,
                "attempt": p.attempt,
                "last_step": p.last_step,
                "seconds_since_progress": (now.saturating_duration_since(p.last_progress).as_secs_f64() * 10.0).round() / 10.0,
                "stream": match p.stream {
                    Stream::Open => "open",
                    Stream::Finished => "finished",
                    Stream::Dropped => "dropped",
                },
                "state": state.as_str(),
            })
        })
        .collect();
    json!({
        "schema_version": 1,
        "run_id": run_id,
        "written_at_ms": written_at_ms,
        "server_running": running,
        "silent_floor_seconds": config.silent_floor.as_secs_f64(),
        "silent_factor": config.silent_factor,
        "world_size": world_size,
        "missing_ranks": missing,
        "summary": summary,
        "ranks": ranks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(rank: i32) -> Key {
        ("run".to_string(), rank)
    }

    fn at(base: Instant, seconds: u64) -> Instant {
        base + Duration::from_secs(seconds)
    }

    fn summary(dog: &Watchdog, now: Instant) -> String {
        dog.statuses(now, true)["run"]["summary"].as_str().unwrap().to_string()
    }

    fn states(dog: &Watchdog, now: Instant) -> Vec<String> {
        dog.statuses(now, true)["run"]["ranks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["state"].as_str().unwrap().to_string())
            .collect()
    }

    fn config(floor: u64) -> WatchdogConfig {
        WatchdogConfig {
            silent_floor: Duration::from_secs(floor),
            silent_factor: 5.0,
        }
    }

    #[test]
    fn a_hung_rank_keeps_its_stream_open_and_goes_silent() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        for rank in 0..2 {
            dog.stream_opened(&key(rank), t0);
        }
        for second in [10, 20, 30] {
            for rank in 0..2 {
                dog.batch(&key(rank), 1, Some(second as i64 * 5), Some(2), at(t0, second));
            }
        }
        assert_eq!(summary(&dog, at(t0, 40)), "active");
        // Rank 1 stops; rank 0 goes on.
        for second in [40, 50, 60, 70, 80, 90, 100, 110] {
            dog.batch(&key(0), 1, Some(second as i64 * 5), Some(2), at(t0, second));
        }
        assert_eq!(states(&dog, at(t0, 110)), ["active", "silent"]);
        assert_eq!(summary(&dog, at(t0, 110)), "degraded");
    }

    #[test]
    fn when_every_rank_goes_quiet_the_job_is_stalled() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        for rank in 0..2 {
            dog.stream_opened(&key(rank), t0);
            dog.batch(&key(rank), 1, Some(9), Some(2), at(t0, 5));
        }
        assert_eq!(summary(&dog, at(t0, 30)), "active");
        assert_eq!(summary(&dog, at(t0, 90)), "stalled");
        let status = &dog.statuses(at(t0, 90), true)["run"];
        assert_eq!(status["ranks"][0]["last_step"], 9);
        assert_eq!(status["ranks"][0]["stream"], "open");
    }

    #[test]
    fn the_threshold_follows_the_rank_s_own_cadence() {
        // Batches 400 s apart (slow steps): 5 x 400 s is the threshold, not the 60 s floor.
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        for n in 1..=4u64 {
            dog.batch(&key(0), 1, Some(n as i64 * 50), Some(1), at(t0, n * 400));
        }
        assert_eq!(summary(&dog, at(t0, 1600 + 1500)), "active");
        assert_eq!(summary(&dog, at(t0, 1600 + 2100)), "stalled");
    }

    #[test]
    fn resent_rows_are_not_progress() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        dog.batch(&key(0), 1, Some(49), Some(1), at(t0, 1));
        dog.batch(&key(0), 1, None, Some(1), at(t0, 100)); // every row already stored
        assert_eq!(summary(&dog, at(t0, 100)), "stalled");
    }

    #[test]
    fn a_clean_end_is_finished_and_a_dropped_stream_becomes_lost() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        for rank in 0..2 {
            dog.stream_opened(&key(rank), t0);
            dog.batch(&key(rank), 1, Some(9), Some(2), at(t0, 1));
        }
        dog.stream_ended(&key(0), true, at(t0, 2));
        dog.stream_ended(&key(1), false, at(t0, 2));
        assert_eq!(states(&dog, at(t0, 10)), ["finished", "reconnecting"]);
        assert_eq!(summary(&dog, at(t0, 10)), "active");
        assert_eq!(states(&dog, at(t0, 100)), ["finished", "lost"]);
        assert_eq!(summary(&dog, at(t0, 100)), "stalled");
        // The sender comes back: the rank is active again.
        dog.stream_opened(&key(1), at(t0, 101));
        dog.batch(&key(1), 1, Some(19), Some(2), at(t0, 102));
        assert_eq!(states(&dog, at(t0, 103)), ["finished", "active"]);
    }

    #[test]
    fn ranks_that_never_connected_are_missing_by_world_size_or_by_gap_after_the_grace_period() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        for rank in [0, 1, 3] {
            dog.stream_opened(&key(rank), t0);
            dog.batch(&key(rank), 1, Some(9), Some(6), at(t0, 1));
        }
        // Ranks start a moment apart: inside the floor nothing is missing yet.
        let status = &dog.statuses(at(t0, 5), true)["run"];
        assert_eq!(status["missing_ranks"], json!([]));
        assert_eq!(status["summary"], "active");
        for rank in [0, 1, 3] {
            dog.batch(&key(rank), 1, Some(19), Some(6), at(t0, 70));
        }
        let status = &dog.statuses(at(t0, 71), true)["run"];
        assert_eq!(status["missing_ranks"], json!([2, 4, 5])); // world_size 6 names 4 and 5 too
        assert_eq!(status["summary"], "degraded");
        // Without a world size only gaps below the highest rank can be seen.
        let dog = Watchdog::new(config(60));
        for rank in [0, 3] {
            dog.stream_opened(&key(rank), t0);
            dog.batch(&key(rank), 1, Some(9), None, at(t0, 1));
            dog.batch(&key(rank), 1, Some(19), None, at(t0, 70));
        }
        assert_eq!(dog.statuses(at(t0, 71), true)["run"]["missing_ranks"], json!([1, 2]));
    }

    #[test]
    fn a_stopped_server_says_so_instead_of_leaving_its_last_verdict() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        dog.batch(&key(0), 1, Some(9), Some(1), at(t0, 1));
        assert_eq!(dog.statuses(at(t0, 2), true)["run"]["summary"], "active");
        let stopped = &dog.statuses(at(t0, 2), false)["run"];
        assert_eq!(stopped["summary"], "server_stopped");
        assert_eq!(stopped["server_running"], false);
    }

    #[test]
    fn a_run_is_finished_only_when_every_rank_and_the_whole_world_finished() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        for rank in 0..2 {
            dog.stream_opened(&key(rank), t0);
            dog.batch(&key(rank), 1, Some(9), Some(3), at(t0, 1));
            dog.stream_ended(&key(rank), true, at(t0, 2));
        }
        assert_eq!(summary(&dog, at(t0, 3)), "finished"); // rank 2 is not missing yet: inside the grace
        for rank in 0..2 {
            dog.stream_opened(&key(rank), at(t0, 4));
            dog.stream_ended(&key(rank), true, at(t0, 70));
        }
        assert_eq!(summary(&dog, at(t0, 71)), "degraded"); // rank 2 never connected
        let dog = Watchdog::new(config(60));
        for rank in 0..2 {
            dog.stream_opened(&key(rank), t0);
            dog.batch(&key(rank), 1, Some(9), Some(2), at(t0, 1));
            dog.stream_ended(&key(rank), true, at(t0, 2));
        }
        assert_eq!(summary(&dog, at(t0, 3)), "finished");
    }
}
