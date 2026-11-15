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
pub const DEFAULT_CRITICAL: Duration = Duration::from_secs(900);
pub const DEFAULT_PAUSE_LIMIT: Duration = Duration::from_secs(1800);
/// A rank is critical after this many times its own silence threshold, if that is longer than
/// the configured critical time, so a slow rank still gets a warning first.
const CRITICAL_THRESHOLDS: f64 = 3.0;
pub const DEFAULT_WATCHDOG_INTERVAL: Duration = Duration::from_secs(10);
/// The phase a rank is in until it says otherwise.
pub const TRAIN: &str = "train";
/// A heartbeat is stale after this many of its own intervals, and never sooner than `HEARTBEAT_MIN_STALE`.
const HEARTBEAT_STALE_INTERVALS: f64 = 3.0;
const HEARTBEAT_MIN_STALE: f64 = 30.0;
const GAPS_KEPT: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct WatchdogConfig {
    pub silent_floor: Duration,
    pub silent_factor: f64,
    /// A rank quiet for longer than this is `critical`; between the threshold and this it is a `warning`.
    pub critical: Duration,
    /// A pause (a phase other than training) with no sign of work counts as a suspected hang
    /// after this long. A pause whose work counter moves is not limited.
    pub pause_limit: Duration,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            silent_floor: DEFAULT_SILENT_FLOOR,
            silent_factor: DEFAULT_SILENT_FACTOR,
            critical: DEFAULT_CRITICAL,
            pause_limit: DEFAULT_PAUSE_LIMIT,
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
    /// The stream is open but no batch has arrived yet: the first one carries `every` steps,
    /// so a slow job takes a long time to send it. Not a stop.
    Starting,
    /// No new step, but the rank declared a phase other than training (evaluation, a
    /// checkpoint), its heartbeats continue, and its work counter moves or the pause is
    /// still within `pause_limit`.
    Paused,
    Silent,
    /// No new step and no heartbeat either: the whole process is frozen, or gone with its
    /// connection still open.
    Unresponsive,
    Reconnecting,
    Lost,
    Finished,
}

impl RankState {
    pub fn as_str(self) -> &'static str {
        match self {
            RankState::Active => "active",
            RankState::Starting => "starting",
            RankState::Paused => "paused",
            RankState::Silent => "silent",
            RankState::Unresponsive => "unresponsive",
            RankState::Reconnecting => "reconnecting",
            RankState::Lost => "lost",
            RankState::Finished => "finished",
        }
    }
}

/// How long a rank spent in a phase other than training, as it reported them.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pauses {
    pub count: u64,
    pub total: f64,
    pub max: f64,
    pub last: f64,
}

#[derive(Debug)]
pub struct Progress {
    phase: String,
    phase_since: Instant,
    last_heartbeat: Option<Instant>,
    heartbeat_interval: f64,
    /// The work counter the rank last reported (steps committed plus evaluation batches), and
    /// when it last moved.
    work: Option<i64>,
    work_moved: Instant,
    pauses: HashMap<String, Pauses>,
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
            phase: TRAIN.to_string(),
            phase_since: now,
            last_heartbeat: None,
            heartbeat_interval: 0.0,
            work: None,
            work_moved: now,
            pauses: HashMap::new(),
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

    /// The quiet time after which a rank that has been stepping is called silent: the floor, or
    /// a multiple of the rank's own time between batches if that is longer.
    pub fn threshold(&self, config: &WatchdogConfig) -> f64 {
        let floor = config.silent_floor.as_secs_f64();
        floor.max(self.cadence().map_or(0.0, |gap| gap * config.silent_factor))
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
                let heartbeat = self.heartbeat_age(now);
                let frozen = heartbeat.is_some_and(|age| age > self.stale_after(config));
                if self.last_step.is_none() {
                    // Before the first batch there is no cadence to judge by.
                    return if frozen {
                        RankState::Unresponsive
                    } else {
                        RankState::Starting
                    };
                }
                if now.saturating_duration_since(self.last_progress).as_secs_f64() <= self.threshold(config) {
                    return RankState::Active;
                }
                match heartbeat {
                    // No heartbeat was ever sent: the phase is unknown, so a quiet rank is only suspect.
                    None => RankState::Silent,
                    Some(_) if frozen => RankState::Unresponsive,
                    Some(_) if self.phase != TRAIN && self.pause_is_alive(now, config) => RankState::Paused,
                    Some(_) => RankState::Silent,
                }
            }
        }
    }

    /// A pause is alive while its work counter keeps moving, or, if it never reports one, until
    /// `pause_limit`. A counter that has stopped for a whole threshold is a stuck pause.
    fn pause_is_alive(&self, now: Instant, config: &WatchdogConfig) -> bool {
        let still = now.saturating_duration_since(self.work_moved).as_secs_f64();
        let age = now.saturating_duration_since(self.phase_since).as_secs_f64();
        if still <= self.threshold(config) {
            return true; // work moved recently
        }
        age <= config.pause_limit.as_secs_f64() && self.work_moved <= self.phase_since
    }

    fn heartbeat_age(&self, now: Instant) -> Option<f64> {
        self.last_heartbeat
            .map(|at| now.saturating_duration_since(at).as_secs_f64())
    }

    /// Three heartbeat intervals, and at least 30 seconds (or half the silence floor, if that is
    /// shorter, so a short floor is not held up by a long grace).
    fn stale_after(&self, config: &WatchdogConfig) -> f64 {
        let minimum = HEARTBEAT_MIN_STALE.min(config.silent_floor.as_secs_f64() / 2.0);
        (self.heartbeat_interval * HEARTBEAT_STALE_INTERVALS).max(minimum)
    }

    /// Seconds the rank has gone without what makes it `state`: for a stopped rank, the time
    /// since it last made progress (or, if its stream dropped, since it dropped).
    fn quiet_seconds(&self, now: Instant) -> f64 {
        let since = match self.stream {
            Stream::Dropped => self.last_change,
            _ => self.last_progress,
        };
        now.saturating_duration_since(since).as_secs_f64()
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

    /// A heartbeat from the rank's own process: it is alive, in `phase`. Leaving a phase other
    /// than training records how long it lasted.
    pub fn heartbeat(&self, key: &Key, phase: &str, interval_seconds: f64, work: Option<i64>, now: Instant) {
        self.with(key, now, |p| {
            if work.is_some() && work != p.work {
                p.work_moved = now;
            }
            if work.is_some() {
                p.work = work;
            }
            p.last_heartbeat = Some(now);
            p.heartbeat_interval = interval_seconds.clamp(0.0, 3600.0);
            if p.phase != phase {
                if p.phase != TRAIN {
                    let seconds = now.saturating_duration_since(p.phase_since).as_secs_f64();
                    let stats = p.pauses.entry(p.phase.clone()).or_default();
                    stats.count += 1;
                    stats.total += seconds;
                    stats.max = stats.max.max(seconds);
                    stats.last = seconds;
                }
                p.phase = phase.to_string();
                p.phase_since = now;
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
                // The first gap is the time to the first batch: all there is to judge a slow job by
                // until a second batch arrives (it holds the same number of steps).
                let from = if p.last_step.is_some() {
                    p.last_progress
                } else {
                    p.created
                };
                let gap = now.saturating_duration_since(from).as_secs_f64();
                p.gaps.push_back(gap);
                if p.gaps.len() > GAPS_KEPT {
                    p.gaps.pop_front();
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
    let active = count(&[
        RankState::Active,
        RankState::Reconnecting,
        RankState::Paused,
        RankState::Starting,
    ]);
    let stopped = count(&[RankState::Silent, RankState::Lost, RankState::Unresponsive]);
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
            let quiet = p.quiet_seconds(now);
            // Critical after the configured time, or three silence thresholds of this rank if that
            // is longer, so a rank that is slow by nature still gets a warning first.
            let critical = config.critical.as_secs_f64().max(CRITICAL_THRESHOLDS * p.threshold(config));
            let severity = match state {
                RankState::Silent | RankState::Lost | RankState::Unresponsive => {
                    Some(if quiet > critical { "critical" } else { "warning" })
                }
                _ => None,
            };
            json!({
                "rank": rank,
                "phase": p.phase,
                "phase_seconds": (now.saturating_duration_since(p.phase_since).as_secs_f64() * 10.0).round() / 10.0,
                "heartbeat_age_seconds": p.heartbeat_age(now).map(|age| (age * 10.0).round() / 10.0),
                "severity": severity,
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
    // Pauses the ranks reported, merged: the longest of each phase, for sizing the threshold.
    let mut pauses: BTreeMap<&str, Pauses> = BTreeMap::new();
    for (_, p) in members {
        for (phase, stats) in &p.pauses {
            let merged = pauses.entry(phase.as_str()).or_default();
            merged.count += stats.count;
            merged.total += stats.total;
            merged.max = merged.max.max(stats.max);
            merged.last = stats.last;
        }
    }
    let longest = pauses.values().map(|stats| stats.max).fold(0.0_f64, f64::max);
    // Twice the longest pause seen, never below the floor: a measurement, not a guarantee.
    let recommended = (longest > 0.0).then(|| (longest * 2.0).ceil().max(config.silent_floor.as_secs_f64()));
    let mut status = json!({
        "schema_version": 1,
        "run_id": run_id,
        "written_at_ms": written_at_ms,
        "server_running": running,
        "silent_floor_seconds": config.silent_floor.as_secs_f64(),
        "silent_factor": config.silent_factor,
        "critical_seconds": config.critical.as_secs_f64(),
        "pauses": pauses.iter().map(|(phase, stats)| (phase.to_string(), json!({
            "count": stats.count,
            "total_seconds": (stats.total * 10.0).round() / 10.0,
            "max_seconds": (stats.max * 10.0).round() / 10.0,
            "last_seconds": (stats.last * 10.0).round() / 10.0,
        }))).collect::<serde_json::Map<String, Value>>(),
        "recommended_silent_seconds": recommended,
        "world_size": world_size,
        "missing_ranks": missing,
        "summary": summary,
        "ranks": ranks,
    });
    status["incidents"] = json!(incidents(&status));
    status
}

/// A status as incidents: Python's `detect_watchdog_incidents`. `job_stalled` when no rank
/// is making progress, `rank_silent` when some are silent, lost or never connected while
/// others go on. These are observations of the stream the server receives; the cause is
/// not known, and a hang is only ever suspected from the stream alone.
pub fn incidents(status: &Value) -> Vec<Value> {
    let summary = status["summary"].as_str().unwrap_or_default();
    if summary != "stalled" && summary != "degraded" {
        return Vec::new();
    }
    let ranks: &[Value] = status["ranks"].as_array().map_or(&[], Vec::as_slice);
    let state = |rank: &Value| rank["state"].as_str().unwrap_or_default().to_owned();
    let stopped: Vec<&Value> = ranks
        .iter()
        .filter(|rank| matches!(state(rank).as_str(), "silent" | "lost" | "unresponsive"))
        .collect();
    let assessment = if stopped.iter().any(|rank| state(rank) != "lost") {
        "suspected_hang"
    } else if !stopped.is_empty() {
        "lost_connection"
    } else {
        "ranks_never_connected"
    };
    let severity = if stopped.iter().any(|rank| rank["severity"] == "critical") {
        "critical"
    } else {
        "warning"
    };
    let last_steps = |ranks: &mut dyn Iterator<Item = &Value>| -> Vec<i64> {
        ranks.filter_map(|rank| rank["last_step"].as_i64()).collect()
    };
    let newest = last_steps(&mut ranks.iter()).into_iter().max().unwrap_or(0);
    let mut incident = json!({
        "severity": severity,
        "assessment": assessment,
        "written_at_ms": status["written_at_ms"],
        "silent_floor_seconds": status["silent_floor_seconds"],
        "critical_seconds": status["critical_seconds"],
        "world_size": status["world_size"],
        "missing_ranks": status["missing_ranks"],
    });
    if summary == "stalled" {
        incident["kind"] = json!("job_stalled");
        incident["step"] = json!(newest);
        incident["ranks"] = ranks.iter().filter(|rank| state(rank) != "finished").cloned().collect();
    } else {
        incident["kind"] = json!("rank_silent");
        incident["step"] = json!(last_steps(&mut stopped.iter().copied())
            .into_iter()
            .min()
            .unwrap_or(newest));
        incident["ranks"] = json!(stopped);
        incident["newest_step"] = json!(newest);
    }
    vec![incident]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_status_is_read_as_incidents_the_way_python_reads_it() {
        let rank = |rank: i64, state: &str, severity: Value, last_step: Value| json!({"rank": rank, "state": state, "severity": severity, "last_step": last_step});
        let status = |summary: &str, ranks: Vec<Value>, missing: Value| {
            json!({"summary": summary, "ranks": ranks, "missing_ranks": missing, "written_at_ms": 7,
                   "silent_floor_seconds": 300.0, "critical_seconds": 900.0, "world_size": 3})
        };
        // Nothing to report while the run is active, finished or the server stopped.
        for summary in ["active", "finished", "server_stopped"] {
            assert!(incidents(&status(
                summary,
                vec![rank(0, "silent", json!("warning"), json!(5))],
                json!([])
            ))
            .is_empty());
        }
        // One rank lost and one silent while a third goes on: the oldest of the stopped steps.
        let found = incidents(&status(
            "degraded",
            vec![
                rank(0, "active", Value::Null, json!(90)),
                rank(1, "lost", json!("critical"), json!(40)),
                rank(2, "silent", json!("warning"), json!(60)),
            ],
            json!([]),
        ));
        assert_eq!(found.len(), 1);
        assert_eq!(
            (
                &found[0]["kind"],
                &found[0]["step"],
                &found[0]["newest_step"],
                &found[0]["assessment"],
                &found[0]["severity"]
            ),
            (
                &json!("rank_silent"),
                &json!(40),
                &json!(90),
                &json!("suspected_hang"),
                &json!("critical")
            )
        );
        assert_eq!(found[0]["ranks"].as_array().unwrap().len(), 2);
        // Every rank stopped: the newest step, and the finished ranks are left out.
        let found = incidents(&status(
            "stalled",
            vec![
                rank(0, "finished", Value::Null, json!(99)),
                rank(1, "lost", json!("warning"), json!(40)),
            ],
            json!([2]),
        ));
        assert_eq!(
            (
                &found[0]["kind"],
                &found[0]["step"],
                &found[0]["assessment"],
                &found[0]["missing_ranks"]
            ),
            (
                &json!("job_stalled"),
                &json!(99),
                &json!("lost_connection"),
                &json!([2])
            )
        );
        assert_eq!(found[0]["ranks"].as_array().unwrap().len(), 1);
        assert!(found[0].get("newest_step").is_none());
        // A rank that never connected and has no step yet.
        let found = incidents(&status(
            "degraded",
            vec![rank(0, "starting", Value::Null, Value::Null)],
            json!([1, 2]),
        ));
        assert_eq!(
            (&found[0]["step"], &found[0]["assessment"], &found[0]["severity"]),
            (&json!(0), &json!("ranks_never_connected"), &json!("warning"))
        );
    }

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
            critical: Duration::from_secs(floor * 3),
            pause_limit: Duration::from_secs(floor * 10),
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

    fn rank_field(dog: &Watchdog, now: Instant, field: &str) -> Value {
        dog.statuses(now, true)["run"]["ranks"][0][field].clone()
    }

    #[test]
    fn a_declared_pause_with_heartbeats_is_paused_and_is_measured() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        dog.batch(&key(0), 1, Some(49), Some(1), at(t0, 1));
        dog.heartbeat(&key(0), "train", 15.0, None, at(t0, 2));
        // Evaluation starts; nothing steps for four minutes but the heartbeats go on.
        dog.heartbeat(&key(0), "eval", 15.0, None, at(t0, 10));
        for second in (25..250).step_by(15) {
            dog.heartbeat(&key(0), "eval", 15.0, None, at(t0, second));
        }
        assert_eq!(states(&dog, at(t0, 250)), ["paused"]);
        assert_eq!(summary(&dog, at(t0, 250)), "active");
        assert_eq!(rank_field(&dog, at(t0, 250), "phase"), "eval");
        assert_eq!(rank_field(&dog, at(t0, 250), "severity"), Value::Null);
        // Training resumes: the pause is on record and sizes a threshold.
        dog.heartbeat(&key(0), "train", 15.0, None, at(t0, 250));
        dog.batch(&key(0), 1, Some(99), Some(1), at(t0, 251));
        let status = &dog.statuses(at(t0, 252), true)["run"];
        assert_eq!(status["pauses"]["eval"]["count"], 1);
        assert_eq!(status["pauses"]["eval"]["max_seconds"], 240.0);
        assert_eq!(status["recommended_silent_seconds"], 480.0); // twice the longest pause
        assert_eq!(status["summary"], "active");
    }

    #[test]
    fn a_rank_in_training_with_live_heartbeats_that_stops_stepping_is_silent_then_critical() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60)); // critical after 180 s
        dog.stream_opened(&key(0), t0);
        dog.batch(&key(0), 1, Some(49), Some(1), at(t0, 1));
        for second in (5..400).step_by(15) {
            dog.heartbeat(&key(0), "train", 15.0, None, at(t0, second));
        }
        // The process is alive and says it is training, yet no step: a suspected hang.
        assert_eq!(states(&dog, at(t0, 100)), ["silent"]);
        assert_eq!(rank_field(&dog, at(t0, 100), "severity"), "warning");
        assert_eq!(rank_field(&dog, at(t0, 200), "severity"), "critical");
        assert_eq!(summary(&dog, at(t0, 200)), "stalled");
    }

    #[test]
    fn without_a_heartbeat_the_phase_is_unknown_and_a_quiet_rank_is_only_suspect() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        dog.batch(&key(0), 1, Some(49), Some(1), at(t0, 1));
        assert_eq!(states(&dog, at(t0, 100)), ["silent"]);
        assert_eq!(rank_field(&dog, at(t0, 100), "heartbeat_age_seconds"), Value::Null);
    }

    #[test]
    fn heartbeats_that_stop_make_a_quiet_rank_unresponsive() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        dog.batch(&key(0), 1, Some(49), Some(1), at(t0, 1));
        dog.heartbeat(&key(0), "train", 15.0, None, at(t0, 5));
        // A frozen process: no step and, past three intervals, no heartbeat either.
        assert_eq!(states(&dog, at(t0, 70)), ["unresponsive"]);
        assert_eq!(summary(&dog, at(t0, 70)), "stalled");
        // A heartbeat missing while steps continue is not a problem for the job.
        dog.batch(&key(0), 1, Some(99), Some(1), at(t0, 71));
        assert_eq!(states(&dog, at(t0, 72)), ["active"]);
    }

    #[test]
    fn before_the_first_batch_a_rank_is_starting_not_stopped() {
        // The first batch carries `every` steps, so a slow job sends it late.
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        for second in (5..2000).step_by(15) {
            dog.heartbeat(&key(0), "train", 15.0, Some(0), at(t0, second));
        }
        assert_eq!(states(&dog, at(t0, 1500)), ["starting"]);
        assert_eq!(summary(&dog, at(t0, 1500)), "active");
        assert_eq!(rank_field(&dog, at(t0, 1500), "severity"), Value::Null);
        // A process that froze during startup is still caught by its heartbeats stopping.
        assert_eq!(states(&dog, at(t0, 2100)), ["unresponsive"]);
    }

    #[test]
    fn the_time_to_the_first_batch_sets_the_cadence_and_the_critical_point_scales() {
        // 400 s to the first batch: the rank is slow, so its threshold is 5 x 400 = 2000 s and its
        // critical point 3 x that, not the configured 180 s.
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        dog.batch(&key(0), 1, Some(49), Some(1), at(t0, 400));
        assert_eq!(states(&dog, at(t0, 400 + 1500)), ["active"]);
        assert_eq!(states(&dog, at(t0, 400 + 2100)), ["silent"]);
        assert_eq!(rank_field(&dog, at(t0, 400 + 2100), "severity"), "warning");
        assert_eq!(rank_field(&dog, at(t0, 400 + 6100), "severity"), "critical");
    }

    #[test]
    fn a_pause_is_bounded_unless_its_work_counter_moves() {
        let t0 = Instant::now();
        let new = || {
            let dog = Watchdog::new(config(60)); // pause limit 600 s
            dog.stream_opened(&key(0), t0);
            dog.batch(&key(0), 1, Some(49), Some(1), at(t0, 1));
            dog
        };
        // No work reported, or no sign of it: paused up to the limit, then a suspected hang.
        let dog = new();
        for second in (10..900).step_by(15) {
            dog.heartbeat(&key(0), "eval", 15.0, Some(7), at(t0, second));
        }
        assert_eq!(states(&dog, at(t0, 300)), ["paused"]);
        assert_eq!(states(&dog, at(t0, 500)), ["paused"]);
        assert_eq!(states(&dog, at(t0, 800)), ["silent"]);
        // Work that keeps moving (an evaluation batch per beat) is not limited.
        let dog = new();
        for (n, second) in (10..5000).step_by(15).enumerate() {
            dog.heartbeat(&key(0), "eval", 15.0, Some(7 + n as i64), at(t0, second));
        }
        assert_eq!(states(&dog, at(t0, 4900)), ["paused"]);
        // Work that moved and then stopped is a stuck evaluation, whatever the limit.
        let dog = new();
        for (n, second) in (10..400).step_by(15).enumerate() {
            let work = if second < 100 { 7 + n as i64 } else { 12 };
            dog.heartbeat(&key(0), "eval", 15.0, Some(work), at(t0, second));
        }
        assert_eq!(states(&dog, at(t0, 399)), ["silent"]);
    }
}
