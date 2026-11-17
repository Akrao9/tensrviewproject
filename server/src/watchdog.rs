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
//! Before a rank's first batch is stored there is no step to count from, and the capture holds
//! `every` steps before it writes any, so a job can hang with nothing stored. Such a rank is
//! `starting`, and that is bounded two ways. Once its heartbeats' work counter has risen while
//! it said `train`, it has taken a training step and the counter is its progress: it is
//! `silent` when the counter has stood still for longer than its threshold. Until then (it is
//! compiling, loading data, in its first collective, or it sends no heartbeat) it is `starting`
//! for `--start-seconds` since its last sign of work, then `silent`, and critical
//! `--critical-seconds` after that. A slow first batch is therefore never a stop while its
//! steps are seen, and a slow first step gets a warning first.
//!
//! The run's summary is `stalled` when no rank is active and some are silent or lost,
//! `degraded` when some are silent, lost or never connected while another is active,
//! `finished` when every rank finished, and `active` otherwise. The server writes it to
//! `runs/<run_id>/watchdog.json` every few seconds, using its own clock, so a reader sees
//! the verdict as of `written_at_ms` whenever it reads. It can only see ranks that
//! connected since the server started (or, after a restart, were in the status of a run that was
//! not finished or abandoned: see below), plus the missing ranks implied by `world_size`
//! (sent by the SDK) or by gaps below the highest rank seen. The thresholds are not
//! calibrated: set `--silent-seconds` above the longest evaluation or checkpoint pause.
//!
//! # A restart does not forget what the server saw
//!
//! Every status also carries `restart`, the clocks a restarted server needs for each rank
//! that the rank list itself leaves out (the recent gaps between batches, which give a slow
//! rank its threshold; the work counter; when the stream last changed; the pauses measured).
//! All times in the file are seconds before `written_at_ms`, so the file says how long
//! something had lasted and not when, and a restart on another host or after a long outage
//! needs no wall clock. At startup [`Watchdog::restore`] reads the file of every run that
//! was not finished or abandoned and takes its ranks up under these rules:
//!
//! - Time the server was not running does not count against a rank. It could not have
//!   delivered anything, and a rank that was active when the server stopped must not look
//!   stalled for the few seconds its sender takes to reconnect. Each clock carries on from
//!   what it read when the file was written (after a SIGKILL that is at most `status_refresh`
//!   before the death, which is forgotten; after SIGTERM it is the moment of the stop).
//! - A rank's stream is gone with the process: every rank that was not finished is restored
//!   as `dropped`, and `reconnecting` for the usual time while its sender comes back.
//! - Every rank that was not finished gets a fresh window from the restart (decided Nov 17,
//!   2026; until then a stopped rank kept the quiet time it had, and the warning's clock with
//!   it, so it could come back `critical` in the first file and fire a trigger before its
//!   sender had a chance to reconnect). No verdict clock starts before the restart: a rank that
//!   has shown no training step has `--start-seconds`, as a new rank has, and one that has
//!   stepped has its own silence threshold (`--silent-seconds`, or its slower cadence), since
//!   its process went on and has only to reconnect and send its next batch. Inside the window
//!   a rank whose sender has not come back is `reconnecting`. After it, it is `lost`, or
//!   `unresponsive` if it was that (its heartbeats had stopped: a frozen process cannot
//!   reconnect, and is not a lost connection); a rank that never stepped is critical
//!   `--critical-seconds` after the window, as one called silent before its first step is. A
//!   sender that comes back is judged from the restart: `active` or `starting` until its window
//!   has passed with no progress, then `silent`, and critical as any rank is. The clocks the
//!   status shows (`seconds_since_progress`, `phase_seconds`, the pauses measured) are kept.
//!   The cost: a hang or a killed job that spans a restart is called later by the time it had
//!   already been quiet, and a stalled run reads healthy for the window.
//! - A run is abandoned when it has no open stream and every rank has been disconnected or
//!   finished for `abandon`: it is not restored and its status is no longer refreshed.
//!
//! A status without `restart` (an older server's) is not restored, and the run is watched
//! afresh when a rank connects.
//!
//! # A job started again
//!
//! Each heartbeat names the session's attempt (from Nov 17, 2026; `attempt_seen`). A higher one
//! than the rank's is the job started again, seen before the new attempt stores a step: the
//! status shows the attempt at once and marks the rank `attempt_without_steps`, and the rank is
//! judged as one before its first step, from that beat (`starting` for `--start-seconds`, then
//! `silent`, critical `--critical-seconds` later; `reconnecting` for that window while its stream
//! is not open), not from the earlier attempt's last batch. `last_step` stays the highest stored,
//! of any attempt. The attempt's first batch ends that, and its first gap is measured from the
//! beat. A beat of the same attempt (a reconnect) changes nothing; a beat of a lower attempt (a
//! process of an earlier attempt still alive) is ignored; and an earlier attempt's rows sent late
//! raise `last_step` without being progress. A beat that names no attempt (an older sender) is
//! taken as before, and its new attempt is known from its first stored batch.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

pub const DEFAULT_SILENT_FLOOR: Duration = Duration::from_secs(300);
pub const DEFAULT_SILENT_FACTOR: f64 = 5.0;
pub const DEFAULT_CRITICAL: Duration = Duration::from_secs(900);
pub const DEFAULT_PAUSE_LIMIT: Duration = Duration::from_secs(1800);
/// How long a rank that has shown no training step may stay `starting` (see `WatchdogConfig::start_limit`).
pub const DEFAULT_START_LIMIT: Duration = Duration::from_secs(1800);
/// A run with no stream open and nothing heard for this long is given up (see `WatchdogConfig::abandon`).
pub const DEFAULT_ABANDON: Duration = Duration::from_secs(72 * 3600);
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
    /// Before its first stored batch, a rank whose work counter has shown no training step (it
    /// is compiling, loading data, in its first collective, or sends no heartbeat) is `starting`
    /// for at most this long since its last sign of work, then `silent`: a warning, and
    /// critical `critical` later. A rank whose counter shows steps is judged by the counter.
    pub start_limit: Duration,
    /// A run with no stream open, whose ranks have all been disconnected or finished for this
    /// long, is abandoned. Zero never gives a run up.
    pub abandon: Duration,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            silent_floor: DEFAULT_SILENT_FLOOR,
            silent_factor: DEFAULT_SILENT_FACTOR,
            critical: DEFAULT_CRITICAL,
            pause_limit: DEFAULT_PAUSE_LIMIT,
            start_limit: DEFAULT_START_LIMIT,
            abandon: DEFAULT_ABANDON,
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
    /// so a slow job takes a long time to send it. Not a stop, while the rank's work counter
    /// moves or, before its first training step, for `start_limit`.
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
    /// The counter rose while the rank said it was training: it has taken a training step.
    /// Until its first batch is stored, the counter is then its progress.
    stepped: bool,
    /// Seconds per unit of work at the counter's last few rises in training, before the first
    /// stored batch: the time since the counter last moved (the first time, since the rank was
    /// first seen) over how far it rose. All there is to judge a slow job by until a batch arrives.
    step_seconds: VecDeque<f64>,
    /// Where the process's threads were at its last heartbeat, when its work had stood still
    /// long enough for it to say (see `Heartbeat` in the SDK). Kept after the heartbeats stop.
    stack: Option<Value>,
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
    /// After a restart, when the server took the rank up. No verdict clock starts before it (see
    /// `from`): the rank has a fresh window, as a rank seen for the first time has. The clocks the
    /// status shows (`seconds_since_progress`, `phase_seconds`, ...) are kept as they were.
    restarted: Option<Instant>,
    /// After a restart, until the rank's stream opens again: when its fresh window ends, and the
    /// verdict it had then. Inside the window a rank with no stream is `reconnecting`; after it,
    /// one that was `unresponsive` stays so (a frozen process cannot reconnect, and is not a lost
    /// connection), and any other is judged as a dropped stream is.
    fresh: Option<(Instant, RankState)>,
    /// A new attempt of the job, seen in the process's heartbeats, that has stored no batch yet:
    /// since when. Until its first batch the rank is judged as one before its first step (the
    /// start window), not from the earlier attempt's last batch.
    attempt_since: Option<Instant>,
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
            stepped: false,
            step_seconds: VecDeque::new(),
            stack: None,
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
            restarted: None,
            fresh: None,
            attempt_since: None,
        }
    }

    /// No batch stored to judge the rank by: none at all, or none of its current attempt.
    fn no_batch(&self) -> bool {
        self.last_step.is_none() || self.attempt_since.is_some()
    }

    /// `at`, or the restart if that is later: the start of every verdict clock.
    fn from(&self, at: Instant) -> Instant {
        self.restarted.map_or(at, |restart| at.max(restart))
    }

    /// The fresh window a rank gets at a restart: `start_limit` for a rank that has shown no
    /// training step, as a new rank gets; its own silence threshold for one that has. A rank that
    /// stepped is not starting again: its process went on through the restart, and all it has to
    /// do is reconnect (the sender's backoff is capped at 30 s) and send its next batch, which its
    /// threshold already allows for.
    fn window(&self, config: &WatchdogConfig) -> f64 {
        if self.before_its_first_step() {
            config.start_limit.as_secs_f64()
        } else {
            self.threshold(config)
        }
    }

    /// The typical time between progress events: the median of the last few gaps.
    fn cadence(&self) -> Option<f64> {
        median(&self.gaps)
    }

    /// The quiet time after which a rank that has been stepping is called silent: the floor, or
    /// a multiple of the rank's own time between batches if that is longer. Before the first
    /// stored batch the time between batches is not known: the multiple is then of the rank's
    /// time per step as its work counter showed it, and never under three heartbeat intervals,
    /// because the server sees the counter once a beat.
    pub fn threshold(&self, config: &WatchdogConfig) -> f64 {
        let floor = config.silent_floor.as_secs_f64();
        if self.no_batch() {
            let step = median(&self.step_seconds).unwrap_or(0.0);
            return floor
                .max(step * config.silent_factor)
                .max(self.heartbeat_interval * HEARTBEAT_STALE_INTERVALS);
        }
        floor.max(self.cadence().map_or(0.0, |gap| gap * config.silent_factor))
    }

    /// No stored batch and a counter that has shown no training step: while its stream is open
    /// the rank is under `start_limit`, not under its threshold.
    fn before_its_first_step(&self) -> bool {
        self.no_batch() && !self.stepped
    }

    /// When the rank last showed a sign of work: its counter moved or it changed its phase
    /// (at first, when it was first seen).
    fn last_sign(&self) -> Instant {
        self.work_moved.max(self.phase_since)
    }

    pub fn state(&self, now: Instant, config: &WatchdogConfig) -> RankState {
        match self.stream {
            Stream::Finished => RankState::Finished,
            Stream::Dropped => {
                if let Some((until, was)) = self.fresh {
                    if now <= until {
                        return RankState::Reconnecting;
                    }
                    if was == RankState::Unresponsive {
                        return RankState::Unresponsive;
                    }
                }
                if now.saturating_duration_since(self.from(self.last_change)) > config.silent_floor {
                    RankState::Lost
                } else {
                    RankState::Reconnecting
                }
            }
            Stream::Open => {
                let heartbeat = self.heartbeat_age(now);
                let frozen = self.last_heartbeat.is_some_and(|at| {
                    now.saturating_duration_since(self.from(at)).as_secs_f64() > self.stale_after(config)
                });
                if self.no_batch() {
                    // Before the first batch there is no cadence of batches to judge by.
                    if frozen {
                        return RankState::Unresponsive;
                    }
                    if !self.stepped {
                        // No training step yet (or no counter): bounded, as a pause with no sign of work is.
                        let idle = now.saturating_duration_since(self.from(self.last_sign()));
                        return if idle <= config.start_limit {
                            RankState::Starting
                        } else {
                            RankState::Silent
                        };
                    }
                    // It has stepped: the counter is its progress until a batch is stored.
                    let still = now.saturating_duration_since(self.from(self.work_moved)).as_secs_f64();
                    return if still <= self.threshold(config) {
                        RankState::Starting
                    } else if self.phase != TRAIN && self.pause_is_alive(now, config) {
                        RankState::Paused
                    } else {
                        RankState::Silent
                    };
                }
                if now
                    .saturating_duration_since(self.from(self.last_progress))
                    .as_secs_f64()
                    <= self.threshold(config)
                {
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
        let still = now.saturating_duration_since(self.from(self.work_moved)).as_secs_f64();
        let age = now.saturating_duration_since(self.from(self.phase_since)).as_secs_f64();
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
    /// since it last made progress (or, if its stream dropped, since it dropped). A rank called
    /// silent before its first stored batch has made none: the time since its counter last
    /// moved, or since its last sign of work if it has not taken a training step.
    fn quiet_seconds(&self, state: RankState, now: Instant) -> f64 {
        let since = match self.stream {
            Stream::Dropped => self.last_change,
            Stream::Open if state == RankState::Silent && self.no_batch() => {
                if self.stepped {
                    self.work_moved
                } else {
                    self.last_sign()
                }
            }
            _ => self.last_progress,
        };
        now.saturating_duration_since(self.from(since)).as_secs_f64()
    }

    /// The quiet time after which a stopped rank is critical: the configured time, or three
    /// silence thresholds of this rank if that is longer, so a rank that is slow by nature still
    /// gets a warning first. A rank called silent before its first training step was given
    /// `start_limit` first, and is critical the configured time after that; so is one that had
    /// shown no step when a restart dropped its stream and that has not come back since.
    fn critical_after(&self, state: RankState, config: &WatchdogConfig) -> f64 {
        let critical = config.critical.as_secs_f64();
        let dropped_by_the_restart =
            self.stream == Stream::Dropped && self.restarted.is_some_and(|restart| self.last_change <= restart);
        if self.before_its_first_step() && (state == RankState::Silent || dropped_by_the_restart) {
            return config.start_limit.as_secs_f64() + critical;
        }
        critical.max(CRITICAL_THRESHOLDS * self.threshold(config))
    }
    /// Seconds since the stream dropped or finished, none while a stream is open.
    fn disconnected_seconds(&self, now: Instant) -> Option<f64> {
        match self.stream {
            Stream::Open => None,
            _ => Some(now.saturating_duration_since(self.last_change).as_secs_f64()),
        }
    }

    /// What a restarted server needs of this rank beyond what its entry in `ranks` says (see
    /// the module notes). Seconds are before `now`, which is the status's `written_at_ms`.
    fn carry(&self, rank: i32, now: Instant) -> Value {
        let before = |at: Instant| round3(now.saturating_duration_since(at).as_secs_f64());
        json!({
            "rank": rank,
            "age_seconds": before(self.created),
            "since_change_seconds": before(self.last_change),
            "heartbeat_interval_seconds": self.heartbeat_interval,
            "work": self.work,
            "work_still_seconds": before(self.work_moved),
            "stepped": self.stepped,
            "attempt_without_steps": self.attempt_since.is_some(),
            "step_seconds": self.step_seconds.iter().copied().map(round3).collect::<Vec<_>>(),
            "gaps_seconds": self.gaps.iter().copied().map(round3).collect::<Vec<_>>(),
            "pauses": self.pauses.iter().map(|(phase, stats)| (phase.clone(), json!({
                "count": stats.count,
                "total_seconds": round3(stats.total),
                "max_seconds": round3(stats.max),
                "last_seconds": round3(stats.last),
            }))).collect::<serde_json::Map<String, Value>>(),
        })
    }

    /// A rank as a restarted server takes it up, from its entry in a status (`entry`) and its
    /// `carry`. See the module notes for the rules. `None` when a field is missing or not a
    /// plausible number.
    fn restored(
        entry: &Value,
        carry: &Value,
        world_size: Option<i32>,
        now: Instant,
        config: &WatchdogConfig,
    ) -> Option<Self> {
        let seconds = |value: &Value| value.as_f64().filter(|seconds| seconds.is_finite() && *seconds >= 0.0);
        let stream = entry["stream"].as_str()?;
        let state = entry["state"].as_str()?;
        let since_progress = seconds(&entry["seconds_since_progress"])?;
        let since_change = seconds(&carry["since_change_seconds"])?;
        // Absent from a status written before it was kept: no step seen.
        let stepped = carry["stepped"].as_bool().unwrap_or(false);
        let work_still = seconds(&carry["work_still_seconds"])?;
        let phase_seconds = seconds(&entry["phase_seconds"])?;
        // When the stream's state last changed, as an age. A stream that had finished or dropped
        // keeps its age; one that was open is dropped now, by the restart. The age is shown and
        // gives up a run (`abandon`); the verdicts are counted from the restart (`from`).
        let change_age = if stream == "finished" || stream == "dropped" {
            since_change
        } else {
            0.0
        };
        let mut pauses = HashMap::new();
        for (phase, stats) in carry["pauses"].as_object()? {
            pauses.insert(
                phase.clone(),
                Pauses {
                    count: stats["count"].as_u64()?,
                    total: seconds(&stats["total_seconds"])?,
                    max: seconds(&stats["max_seconds"])?,
                    last: seconds(&stats["last_seconds"])?,
                },
            );
        }
        let mut gaps = VecDeque::new();
        for gap in carry["gaps_seconds"].as_array()? {
            gaps.push_back(seconds(gap)?);
        }
        while gaps.len() > GAPS_KEPT {
            gaps.pop_front();
        }
        Some(Self {
            phase: entry["phase"].as_str()?.chars().take(40).collect(),
            phase_since: ago(now, phase_seconds),
            last_heartbeat: match &entry["heartbeat_age_seconds"] {
                Value::Null => None,
                age => Some(ago(now, seconds(age)?)),
            },
            heartbeat_interval: seconds(&carry["heartbeat_interval_seconds"])?.clamp(0.0, 3600.0),
            work: carry["work"].as_i64(),
            work_moved: ago(now, work_still),
            stepped,
            step_seconds: carry["step_seconds"]
                .as_array()
                .map_or(&[][..], Vec::as_slice)
                .iter()
                .filter_map(seconds)
                .take(GAPS_KEPT)
                .collect(),
            stack: entry.get("stack").filter(|stack| stack.is_object()).cloned(),
            pauses,
            created: ago(now, seconds(&carry["age_seconds"])?),
            attempt: entry["attempt"].as_i64()?,
            last_step: entry["last_step"].as_i64(),
            last_progress: ago(now, since_progress),
            last_change: ago(now, change_age),
            gaps,
            open_streams: 0,
            stream: if stream == "finished" {
                Stream::Finished
            } else {
                Stream::Dropped
            },
            world_size,
            restarted: None,
            fresh: None,
            // Absent from a status written before it was kept: the attempt has stored steps.
            attempt_since: carry["attempt_without_steps"].as_bool().unwrap_or(false).then_some(now),
        })
        .map(|mut progress| {
            // A fresh window from the restart for every rank that was not finished (decided Nov 17,
            // 2026): its clocks start again at the restart, so it is not called stopped, nor
            // critical, before its sender has had the time a new rank gets.
            if progress.stream != Stream::Finished {
                let window = Duration::try_from_secs_f64(progress.window(config)).unwrap_or(Duration::MAX);
                let was = if state == "unresponsive" {
                    RankState::Unresponsive
                } else {
                    RankState::Lost
                };
                progress.restarted = Some(now);
                progress.fresh = Some((
                    now.checked_add(window)
                        .unwrap_or(now + Duration::from_secs(10 * 365 * 24 * 3600)),
                    was,
                ));
            }
            progress
        })
    }
}

/// The middle of a few times (the upper of two), none of none.
fn median(times: &VecDeque<f64>) -> Option<f64> {
    if times.is_empty() {
        return None;
    }
    let mut times: Vec<f64> = times.iter().copied().collect();
    times.sort_by(|a, b| a.total_cmp(b));
    Some(times[times.len() / 2])
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

/// `seconds` before `now`. A restored age can be longer than this host has been up (a run
/// restarted on a new node), and an `Instant` that far back does not exist on every platform:
/// then the oldest age the clock can show is used, never zero, so a clock is never restarted.
fn ago(now: Instant, seconds: f64) -> Instant {
    const LONGEST: Duration = Duration::from_secs(10 * 365 * 24 * 3600);
    let seconds = if seconds.is_nan() || seconds < 0.0 {
        0.0
    } else {
        seconds
    };
    let mut age = Duration::try_from_secs_f64(seconds).unwrap_or(LONGEST).min(LONGEST);
    loop {
        if let Some(at) = now.checked_sub(age) {
            return at;
        }
        age /= 2;
        if age.is_zero() {
            return now;
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

    /// Takes up the ranks of `run_id` from a status this server (or an earlier one) wrote, as
    /// described in the module notes. `Ok(true)` when the run was taken up, `Ok(false)` when
    /// there was nothing to take: a finished or abandoned run, a status without `restart`.
    /// All of a run's ranks are taken up or none. A rank that has already connected is left
    /// alone.
    pub fn restore(&self, run_id: &str, status: &Value, now: Instant) -> Result<bool, String> {
        if status["run_id"] != run_id || status["schema_version"] != 1 {
            return Err("not a version 1 status of this run".to_string());
        }
        if status["summary"] == "finished" || status["abandoned"] == true {
            return Ok(false);
        }
        let (Some(entries), Some(carried)) = (status["ranks"].as_array(), status["restart"].as_array()) else {
            return Ok(false);
        };
        if entries.is_empty() {
            return Ok(false);
        }
        let world_size = status["world_size"].as_i64().and_then(|size| i32::try_from(size).ok());
        let mut restored = Vec::new();
        for entry in entries {
            let bad = || format!("the entry of rank {} cannot be read", entry["rank"]);
            let rank = entry["rank"]
                .as_i64()
                .and_then(|rank| i32::try_from(rank).ok())
                .filter(|rank| *rank >= 0);
            let rank = rank.ok_or_else(bad)?;
            let carry = carried
                .iter()
                .find(|carry| carry["rank"].as_i64() == Some(i64::from(rank)));
            let progress = carry
                .and_then(|carry| Progress::restored(entry, carry, world_size, now, &self.config))
                .ok_or_else(bad)?;
            restored.push(((run_id.to_string(), rank), progress));
        }
        let mut ranks = self.ranks.lock().unwrap();
        for (key, progress) in restored {
            ranks.entry(key).or_insert(progress);
        }
        Ok(true)
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
            p.fresh = None;
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

    /// The attempt a heartbeat names (a sender from before Nov 17, 2026 names none). A higher
    /// attempt than the rank's is the job started again: the status shows it at once, and the
    /// rank gets the start window of a new rank from now (`start_limit`, then a warning, critical
    /// `critical` later) until the attempt stores a batch, instead of reading silent and critical
    /// from the earlier attempt's last batch. Its clocks start now (`restarted`), and a stream
    /// still dropped is `reconnecting` for that window. The same attempt is a beat as any other,
    /// whatever the stream did (a reconnect is not a new attempt). A lower one is a process of an
    /// earlier attempt still beating: false, and the beat is not taken.
    pub fn attempt_seen(&self, key: &Key, attempt: i64, now: Instant) -> bool {
        let window = self.config.start_limit;
        self.with(key, now, |p| {
            if attempt <= p.attempt {
                return attempt == p.attempt;
            }
            let known = p.attempt != 0 || p.last_step.is_some();
            p.attempt = attempt;
            if known {
                p.attempt_since = Some(now);
                p.restarted = Some(now);
                p.fresh = Some((now.checked_add(window).unwrap_or(now), RankState::Lost));
                p.stepped = false;
                p.step_seconds.clear();
                p.work = None;
                p.work_moved = now;
                p.stack = None;
            }
            true
        })
    }

    /// A heartbeat from the rank's own process: it is alive, in `phase`. Leaving a phase other
    /// than training records how long it lasted.
    pub fn heartbeat(
        &self,
        key: &Key,
        phase: &str,
        interval_seconds: f64,
        work: Option<i64>,
        stack: Option<Value>,
        now: Instant,
    ) {
        self.with(key, now, |p| {
            // The first report is the counter's value, not a movement, unless it is already above zero.
            let moved = work.is_some() && work != p.work && (p.work.is_some() || work > Some(0));
            if let (true, Some(count)) = (moved, work) {
                // A rise while the rank said `train` is a training step (a tick in another phase is
                // work, and not a step). The phase is the one it was in as the counter rose.
                let rise = count.saturating_sub(p.work.unwrap_or(0));
                let training = p.phase == TRAIN && (p.work.is_some() || phase == TRAIN);
                if rise > 0 && training {
                    if p.no_batch() {
                        let elapsed = now.saturating_duration_since(p.work_moved).as_secs_f64();
                        p.step_seconds.push_back(elapsed / rise as f64);
                        if p.step_seconds.len() > GAPS_KEPT {
                            p.step_seconds.pop_front();
                        }
                    }
                    p.stepped = true;
                }
                p.work_moved = now;
            }
            // The newest word wins: a beat with no stack says the rank is working again.
            p.stack = stack;
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
            if attempt < p.attempt {
                // An earlier attempt's rows, sent late (by the process of a later attempt that found
                // them unsent): kept, and not progress of the attempt the rank is in now.
                if let Some(step) = new_last {
                    p.last_step = Some(p.last_step.map_or(step, |last| last.max(step)));
                }
                return;
            }
            if let Some(step) = new_last {
                // The first gap is the time to the first batch: all there is to judge a slow job by
                // until a second batch arrives (it holds the same number of steps). For a new attempt,
                // the time from its first heartbeat.
                let from = if let Some(since) = p.attempt_since {
                    since
                } else if p.last_step.is_some() {
                    p.last_progress
                } else {
                    p.created
                };
                p.attempt_since = None;
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
            let quiet = p.quiet_seconds(*state, now);
            let critical = p.critical_after(*state, config);
            let severity = match state {
                RankState::Silent | RankState::Lost | RankState::Unresponsive => {
                    Some(if quiet > critical { "critical" } else { "warning" })
                }
                _ => None,
            };
            let mut entry = json!({
                "rank": rank,
                "phase": p.phase,
                "phase_seconds": (now.saturating_duration_since(p.phase_since).as_secs_f64() * 10.0).round() / 10.0,
                "heartbeat_age_seconds": p.heartbeat_age(now).map(|age| (age * 10.0).round() / 10.0),
                "severity": severity,
                "attempt": p.attempt,
                "last_step": p.last_step,
                "work": p.work,
                "seconds_since_progress": (now.saturating_duration_since(p.last_progress).as_secs_f64() * 10.0).round() / 10.0,
                "stream": match p.stream {
                    Stream::Open => "open",
                    Stream::Finished => "finished",
                    Stream::Dropped => "dropped",
                },
                "state": state.as_str(),
            });
            if p.attempt_since.is_some() {
                // `attempt` is a new one, seen in its heartbeats; `last_step` and the time since
                // progress are still the earlier attempt's.
                entry["attempt_without_steps"] = json!(true);
            }
            if let Some(stack) = &p.stack {
                entry["stack"] = stack.clone();
            }
            entry
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
        "start_seconds": config.start_limit.as_secs_f64(),
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
    status["abandoned"] = json!(
        !config.abandon.is_zero()
            && !members.is_empty()
            && members.iter().all(|(_, p)| p
                .disconnected_seconds(now)
                .is_some_and(|idle| idle >= config.abandon.as_secs_f64()))
    );
    status["restart"] = json!(members.iter().map(|(rank, p)| p.carry(*rank, now)).collect::<Vec<_>>());
    status
}

/// What this server's detectors last reported for one rank: the incident kinds in order and
/// the checks it could not make (Flight Recorder and TrainCheck left out: they are never
/// expected here). Present only while the rank's detectors are on and current.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RankReport {
    pub kinds: Vec<String>,
    pub not_checked: Vec<String>,
}

const RUN_STATES: [&str; 4] = ["healthy", "degraded", "unknown", "unhealthy"];

/// The run state of `run_state` in `evidence.py` (the reference), from this server's own view:
/// the status, and the reports its detectors keep per rank. `{"state", "reasons"}`, where each
/// reason has `state`, `code`, `detail` and, by code, `kinds`, `ranks`, `summary` or `checks`.
/// A rank this server has not judged since it started (it has not sent a part, its detectors
/// failed, or `--no-detect`) is unknown; so is a missing rank, a stopped server and any summary
/// other than `active` or `finished` (a stall among them: the stream cannot tell a hang from a
/// pause). A reader of a `watchdog.json` that is no longer being rewritten must not trust it:
/// Python calls such a status old, and the run state unknown. Each incident's class comes from
/// its kind ([`crate::detect::incident_class`]): observations are listed under `observations`,
/// a reason whose state is `healthy`, and every other kind, an unknown one among them, makes the
/// run unhealthy.
pub fn run_state(status: &Value, reports: &BTreeMap<i32, RankReport>, detect: bool) -> Value {
    let mut reasons: Vec<Value> = Vec::new();
    let ranks: Vec<i32> = status["ranks"]
        .as_array()
        .map_or(&[][..], Vec::as_slice)
        .iter()
        .filter_map(|rank| rank["rank"].as_i64().and_then(|rank| i32::try_from(rank).ok()))
        .collect();
    let stream = status["incidents"].as_array().map_or(&[][..], Vec::as_slice);
    let mut listed: Vec<String> = Vec::new();
    for rank in &ranks {
        if let Some(report) = reports.get(rank) {
            listed.extend(report.kinds.iter().cloned());
        }
    }
    listed.extend(
        stream
            .iter()
            .map(|incident| incident["kind"].as_str().unwrap_or_default().to_string()),
    );
    let (observed, counted): (Vec<String>, Vec<String>) = listed
        .into_iter()
        .partition(|kind| crate::detect::incident_class(kind) == Some("observation"));
    let distinct = |all: &[String]| {
        let mut seen = std::collections::HashSet::new();
        all.iter()
            .filter(|kind| seen.insert(kind.as_str()))
            .cloned()
            .collect::<Vec<String>>()
    };
    if !counted.is_empty() {
        let (count, kinds) = (counted.len(), distinct(&counted));
        reasons.push(json!({
            "state": "unhealthy", "code": "incidents", "kinds": kinds,
            "detail": format!("{count} incident{} listed: {}", if count == 1 { "" } else { "s" }, kinds.join(", ")),
        }));
    }
    let judged: Vec<i32> = ranks
        .iter()
        .copied()
        .filter(|rank| detect && reports.contains_key(rank))
        .collect();
    if judged.is_empty() {
        reasons.push(json!({"state": "unknown", "code": "no_steps", "detail": "no step has been judged yet"}));
    }
    let mut unknown: Vec<i32> = ranks.iter().copied().filter(|rank| !judged.contains(rank)).collect();
    unknown.extend(
        status["missing_ranks"]
            .as_array()
            .map_or(&[][..], Vec::as_slice)
            .iter()
            .filter_map(|rank| rank.as_i64().and_then(|rank| i32::try_from(rank).ok())),
    );
    // A rank below `world_size` that has not connected: the status calls it missing only after
    // the silence floor, and until then one connected rank of four must not read healthy.
    if let Some(world_size) = status["world_size"]
        .as_i64()
        .filter(|size| (1..=100_000).contains(size))
    {
        unknown.extend((0..world_size as i32).filter(|rank| !ranks.contains(rank)));
    }
    unknown.sort_unstable();
    unknown.dedup();
    if !unknown.is_empty() {
        let why = if detect {
            ""
        } else {
            " (this server runs with --no-detect)"
        };
        reasons.push(json!({
            "state": "unknown", "code": "ranks_unknown", "ranks": unknown,
            "detail": format!("{} rank{} without a current report from this server{why}: nothing is said about incidents on them",
                unknown.len(), if unknown.len() == 1 { "" } else { "s" }),
        }));
    }
    let summary = status["summary"].as_str().unwrap_or_default();
    if summary == "server_stopped" {
        reasons.push(json!({"state": "unknown", "code": "server_stopped",
            "detail": "the server has stopped: no new steps are stored and nothing watches the run for stalls"}));
    } else if !matches!(summary, "active" | "finished") {
        reasons.push(
            json!({"state": "unknown", "code": "status_summary", "summary": status["summary"],
            "detail": format!("the server's status says {summary:?}")}),
        );
    }
    let mut checks: Vec<String> = Vec::new();
    for rank in &judged {
        for check in &reports[rank].not_checked {
            if !checks.contains(check) {
                checks.push(check.clone());
            }
        }
    }
    if !checks.is_empty() {
        reasons.push(json!({"state": "degraded", "code": "not_checked", "checks": checks,
            "detail": format!("not checked: {}", checks.join(", "))}));
    }
    if !observed.is_empty() {
        let (count, kinds) = (observed.len(), distinct(&observed));
        reasons.push(json!({
            "state": "healthy", "code": "observations", "kinds": kinds,
            "detail": format!("{count} observation{} listed, not counted as failures: {}",
                if count == 1 { "" } else { "s" }, kinds.join(", ")),
        }));
    }
    let state = reasons
        .iter()
        .filter_map(|reason| RUN_STATES.iter().position(|state| reason["state"] == *state))
        .max()
        .map_or("healthy", |at| RUN_STATES[at]);
    json!({"state": state, "reasons": reasons})
}

/// A status as incidents: Python's `detect_watchdog_incidents`. `job_stalled` when no rank
/// is making progress, `rank_silent` when some are silent, lost or never connected while
/// others go on. These are observations of the stream the server receives; the cause is
/// not known, and a hang is only ever suspected from the stream alone.
/// Where a stopped rank's main thread was, from the stack its process sent: `stuck_at` in
/// `sdk/src/tensorquill/evidence.py`, which is the reference.
fn stuck_at(rank: &Value) -> Option<Value> {
    let stack = rank["stack"].as_object()?;
    let main = stack
        .get("threads")?
        .as_array()?
        .iter()
        .find(|thread| thread["main"] == true)?;
    let frames: Vec<&Value> = main["frames"]
        .as_array()?
        .iter()
        .filter(|frame| frame.is_object() && frame["file"].is_string())
        .collect();
    let brief = |frame: &Value| {
        json!({
            "function": frame["function"].as_str().unwrap_or("?"),
            "file": frame["file"],
            "line": frame["line"],
        })
    };
    let own = |frame: &Value| frame["own"] == true;
    let library = |frame: &Value| {
        let file = frame["file"].as_str().unwrap_or_default();
        file.contains("site-packages")
            || file.contains("dist-packages")
            || file.contains("/lib/python")
            || file.starts_with('<')
    };
    let mut place = brief(frames.first()?);
    place["rank"] = rank["rank"].clone();
    if let Some(caller) = frames.iter().skip(1).find(|frame| !library(frame) && !own(frame)) {
        place["caller"] = brief(caller);
    }
    if let Some(frame) = frames.iter().find(|frame| own(frame)) {
        place["in_tensorquill"] = brief(frame);
    }
    for key in ["unchanged_seconds", "still_seconds"] {
        if stack.get(key).is_some_and(Value::is_number) {
            place[key] = stack[key].clone();
        }
    }
    Some(place)
}

/// The incidents a run's status shows: `job_stalled` or `rank_silent` from its summary, and
/// `duplicate_batches` when the ranks' batch hashes say so (see `detect::duplicate_batches`),
/// whatever the summary. The same as Python's `detect_watchdog_incidents`.
pub fn incidents(status: &Value) -> Vec<Value> {
    let mut found = progress_incidents(status);
    if let Some(duplicates) = status["duplicate_batches"].as_object() {
        let mut incident = serde_json::Map::new();
        incident.insert("kind".into(), json!("duplicate_batches"));
        incident.extend(duplicates.clone());
        found.push(Value::Object(incident));
    }
    crate::detect::with_classes(&mut found);
    found
}

fn progress_incidents(status: &Value) -> Vec<Value> {
    let summary = status["summary"].as_str().unwrap_or_default();
    if summary != "stalled" && summary != "degraded" {
        return Vec::new();
    }
    let full: &[Value] = status["ranks"].as_array().map_or(&[], Vec::as_slice);
    let places: Vec<Value> = full
        .iter()
        .filter(|rank| matches!(rank["state"].as_str(), Some("silent" | "lost" | "unresponsive")))
        .filter_map(stuck_at)
        .collect();
    // The stacks stay in the status's `ranks`; an incident carries `stuck_at` instead.
    let without: Vec<Value> = full
        .iter()
        .map(|rank| {
            let mut rank = rank.clone();
            if let Some(fields) = rank.as_object_mut() {
                fields.remove("stack");
            }
            rank
        })
        .collect();
    let ranks: &[Value] = &without;
    let state = |rank: &Value| rank["state"].as_str().unwrap_or_default().to_owned();
    let stopped: Vec<&Value> = ranks
        .iter()
        .filter(|rank| matches!(state(rank).as_str(), "silent" | "lost" | "unresponsive"))
        .collect();
    // A lost rank (its stream dropped and did not come back: a killed or crashed process, or the
    // network) labels the stall when it explains it: every stopped rank is lost, or the whole job
    // stopped, whose connected ranks then wait for the one that is gone. As Python's `_progress_incidents`.
    let lost: Vec<Value> = stopped
        .iter()
        .filter(|rank| state(rank) == "lost")
        .map(|rank| rank["rank"].clone())
        .collect();
    let assessment =
        if !stopped.is_empty() && (lost.len() == stopped.len() || (!lost.is_empty() && summary == "stalled")) {
            "lost_connection"
        } else if !stopped.is_empty() {
            "suspected_hang"
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
    if !lost.is_empty() {
        incident["lost_ranks"] = json!(lost);
    }
    if !places.is_empty() {
        incident["stuck_at"] = json!(places);
    }
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
        // The ranks that are not stopped, briefly, as Python's `_progress_incidents` lists them.
        incident["other_ranks"] = ranks
            .iter()
            .filter(|rank| !matches!(state(rank).as_str(), "silent" | "lost" | "unresponsive"))
            .map(|rank| {
                json!({"rank": rank["rank"], "state": rank["state"], "last_step": rank["last_step"], "attempt": rank["attempt"]})
            })
            .collect::<Vec<_>>()
            .into();
        incident["newest_step"] = json!(newest);
    }
    vec![incident]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stopped_ranks_stack_becomes_stuck_at_the_way_python_reads_it() {
        // The same cases as `test_watchdog.py`: Python's `stuck_at` is the reference.
        let site = "/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py";
        let frame = |function: &str, file: &str, line: i64| json!({"function": function, "file": file, "line": line});
        let stack = |frames: Vec<Value>| {
            json!({"still_seconds": 140.0, "unchanged_seconds": 95.0, "threads": [
                {"name": "loader", "main": false, "frames": [frame("get", "/job/data.py", 3)]},
                {"name": "MainThread", "main": true, "frames": frames},
            ]})
        };
        let rank = |rank: i64, state: &str, stack: Value| json!({"rank": rank, "state": state, "severity": "warning", "last_step": 9, "stack": stack});
        let status =
            |ranks: Vec<Value>| json!({"summary": "stalled", "ranks": ranks, "missing_ranks": [], "written_at_ms": 7});
        let mut own = frame(
            "replica_spread",
            "/venv/lib/python3.12/site-packages/tensorquill/distributed.py",
            150,
        );
        own["own"] = json!(true);
        let hit = &incidents(&status(vec![
            rank(
                0,
                "silent",
                stack(vec![
                    frame("all_reduce", site, 2940),
                    frame("train_step", "/job/train.py", 88),
                ]),
            ),
            rank(1, "unresponsive", Value::Null),
            rank(
                2,
                "silent",
                stack(vec![
                    frame("all_gather", site, 3100),
                    own,
                    frame("loop", "/job/tensorquill/run.py", 7),
                ]),
            ),
            rank(3, "active", stack(vec![frame("step", "/job/train.py", 1)])),
        ]))[0];
        assert_eq!(
            hit["stuck_at"],
            json!([
                {"rank": 0, "function": "all_reduce", "file": site, "line": 2940, "still_seconds": 140.0, "unchanged_seconds": 95.0,
                 "caller": {"function": "train_step", "file": "/job/train.py", "line": 88}},
                {"rank": 2, "function": "all_gather", "file": site, "line": 3100, "still_seconds": 140.0, "unchanged_seconds": 95.0,
                 "caller": {"function": "loop", "file": "/job/tensorquill/run.py", "line": 7},
                 "in_tensorquill": {"function": "replica_spread", "file": "/venv/lib/python3.12/site-packages/tensorquill/distributed.py", "line": 150}},
            ])
        );
        assert!(hit["ranks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|rank| rank.get("stack").is_none()));
        // A stack that is not one gives no `stuck_at`, and nothing else changes.
        for bad in [
            json!("frames"),
            json!([]),
            json!({}),
            json!({"threads": "x"}),
            json!({"threads": [{"main": true}]}),
            json!({"threads": [{"main": true, "frames": []}]}),
            json!({"threads": [{"main": true, "frames": [{"line": 3}, 7]}]}),
            json!({"threads": [{"main": false, "frames": [{"file": "a.py"}]}]}),
            json!({"threads": [{"main": 1, "frames": [{"file": "a.py"}]}]}),
        ] {
            let hit = &incidents(&status(vec![rank(0, "silent", bad)]))[0];
            assert!(hit.get("stuck_at").is_none(), "{hit}");
        }
        // Names and numbers that are not: `?` for the function, the seconds left out.
        let odd = json!({"still_seconds": true, "unchanged_seconds": "9", "threads": [{"main": true, "frames": [{"file": "a.py", "function": 7}]}]});
        assert_eq!(
            incidents(&status(vec![rank(0, "silent", odd)]))[0]["stuck_at"],
            json!([{"rank": 0, "function": "?", "file": "a.py", "line": null}])
        );
    }

    #[test]
    fn a_lost_rank_labels_the_stall_when_it_explains_it() {
        // The same cases as `test_watchdog.py`'s `test_a_lost_rank_labels_the_stall_when_it_explains_it`.
        let rank = |rank: i64, state: &str, last_step: i64| json!({"rank": rank, "state": state, "severity": "critical", "last_step": last_step});
        let status = |summary: &str, ranks: Vec<Value>| json!({"summary": summary, "ranks": ranks, "missing_ranks": [], "written_at_ms": 7});
        let read = |found: Vec<Value>| {
            (
                found[0]["kind"].clone(),
                found[0]["assessment"].clone(),
                found[0].get("lost_ranks").cloned(),
            )
        };
        // One rank killed and the others blocked in a collective waiting for it: the job stopped,
        // and the lost rank explains it.
        assert_eq!(
            read(incidents(&status(
                "stalled",
                vec![rank(0, "silent", 40), rank(1, "lost", 40), rank(2, "unresponsive", 40)]
            ))),
            (json!("job_stalled"), json!("lost_connection"), Some(json!([1])))
        );
        // Every stopped rank lost, the rest stepping: a lost connection too.
        assert_eq!(
            read(incidents(&status(
                "degraded",
                vec![rank(0, "active", 90), rank(1, "lost", 40)]
            ))),
            (json!("rank_silent"), json!("lost_connection"), Some(json!([1])))
        );
        // A lost rank beside a silent one while another steps: the silent rank does not wait for
        // the lost one, so it is a suspected hang, and the lost rank is still named.
        assert_eq!(
            read(incidents(&status(
                "degraded",
                vec![rank(0, "active", 90), rank(1, "lost", 40), rank(2, "silent", 60)]
            ))),
            (json!("rank_silent"), json!("suspected_hang"), Some(json!([1])))
        );
        // `rank_silent` lists the ranks that are not stopped, briefly, as Python does.
        let silent = incidents(&status("degraded", vec![rank(0, "active", 90), rank(2, "silent", 60)]));
        assert_eq!(
            silent[0]["other_ranks"],
            json!([{"rank": 0, "state": "active", "last_step": 90, "attempt": null}])
        );
        // No lost rank: a suspected hang, with no `lost_ranks`.
        assert_eq!(
            read(incidents(&status("stalled", vec![rank(0, "silent", 40)]))),
            (json!("job_stalled"), json!("suspected_hang"), None)
        );
    }

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
            start_limit: Duration::from_secs(floor * 10),
            abandon: Duration::from_secs(floor * 100),
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
        dog.heartbeat(&key(0), "train", 15.0, None, None, at(t0, 2));
        // Evaluation starts; nothing steps for four minutes but the heartbeats go on.
        dog.heartbeat(&key(0), "eval", 15.0, None, None, at(t0, 10));
        for second in (25..250).step_by(15) {
            dog.heartbeat(&key(0), "eval", 15.0, None, None, at(t0, second));
        }
        assert_eq!(states(&dog, at(t0, 250)), ["paused"]);
        assert_eq!(summary(&dog, at(t0, 250)), "active");
        assert_eq!(rank_field(&dog, at(t0, 250), "phase"), "eval");
        assert_eq!(rank_field(&dog, at(t0, 250), "severity"), Value::Null);
        // Training resumes: the pause is on record and sizes a threshold.
        dog.heartbeat(&key(0), "train", 15.0, None, None, at(t0, 250));
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
            dog.heartbeat(&key(0), "train", 15.0, None, None, at(t0, second));
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
        dog.heartbeat(&key(0), "train", 15.0, None, None, at(t0, 5));
        // A frozen process: no step and, past three intervals, no heartbeat either.
        assert_eq!(states(&dog, at(t0, 70)), ["unresponsive"]);
        assert_eq!(summary(&dog, at(t0, 70)), "stalled");
        // A heartbeat missing while steps continue is not a problem for the job.
        dog.batch(&key(0), 1, Some(99), Some(1), at(t0, 71));
        assert_eq!(states(&dog, at(t0, 72)), ["active"]);
    }

    #[test]
    fn before_its_first_step_a_rank_is_starting_for_the_start_limit_then_silent() {
        // No training step yet (compiling, loading data, its first collective): starting for the
        // start limit (600 s here), then a suspected hang, critical 180 s after that.
        let t0 = Instant::now();
        let new = |beats_until: u64| {
            let dog = Watchdog::new(config(60));
            dog.stream_opened(&key(0), t0);
            for second in (5..beats_until).step_by(15) {
                dog.heartbeat(&key(0), "train", 15.0, Some(0), None, at(t0, second));
            }
            dog
        };
        let dog = new(2000);
        assert_eq!(states(&dog, at(t0, 590)), ["starting"]);
        assert_eq!(summary(&dog, at(t0, 590)), "active");
        assert_eq!(rank_field(&dog, at(t0, 590), "severity"), Value::Null);
        assert_eq!(states(&dog, at(t0, 610)), ["silent"]);
        assert_eq!(summary(&dog, at(t0, 610)), "stalled");
        assert_eq!(rank_field(&dog, at(t0, 610), "severity"), "warning");
        assert_eq!(rank_field(&dog, at(t0, 770), "severity"), "warning");
        assert_eq!(rank_field(&dog, at(t0, 790), "severity"), "critical");
        let status = &dog.statuses(at(t0, 790), true)["run"];
        assert_eq!(
            (&status["ranks"][0]["last_step"], &status["ranks"][0]["work"]),
            (&Value::Null, &json!(0))
        );
        assert_eq!(status["start_seconds"], 600.0);
        let stall = &status["incidents"][0];
        assert_eq!(
            (&stall["kind"], &stall["step"], &stall["severity"], &stall["assessment"]),
            (
                &json!("job_stalled"),
                &json!(0),
                &json!("critical"),
                &json!("suspected_hang")
            )
        );
        // The first stored batch ends it, however late.
        dog.batch(&key(0), 1, Some(49), Some(1), at(t0, 800));
        assert_eq!(states(&dog, at(t0, 801)), ["active"]);
        assert_eq!(rank_field(&dog, at(t0, 801), "severity"), Value::Null);
        // A process that froze during startup is still caught by its heartbeats stopping.
        let dog = new(100);
        assert_eq!(states(&dog, at(t0, 200)), ["unresponsive"]);
        // A rank that sends no heartbeat has no counter: the same limit.
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        assert_eq!(states(&dog, at(t0, 590)), ["starting"]);
        assert_eq!(states(&dog, at(t0, 610)), ["silent"]);
        assert_eq!(rank_field(&dog, at(t0, 610), "heartbeat_age_seconds"), Value::Null);
    }

    #[test]
    fn a_rank_that_steps_and_then_stops_before_its_first_batch_is_silent_then_critical() {
        // 40 steps, then nothing, with `every` at 50: no batch was ever stored. The heartbeats go
        // on and their counter stands at 40.
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60)); // critical after 180 s
        dog.stream_opened(&key(0), t0);
        for second in (5..1000).step_by(15) {
            let work = if second < 35 { 0 } else { 40 };
            dog.heartbeat(&key(0), "train", 15.0, Some(work), None, at(t0, second));
        }
        // The counter last moved at 35 s; the threshold is the 60 s floor.
        assert_eq!(states(&dog, at(t0, 90)), ["starting"]);
        assert_eq!(rank_field(&dog, at(t0, 90), "severity"), Value::Null);
        assert_eq!(states(&dog, at(t0, 100)), ["silent"]);
        assert_eq!(rank_field(&dog, at(t0, 100), "severity"), "warning");
        assert_eq!(summary(&dog, at(t0, 100)), "stalled");
        assert_eq!(rank_field(&dog, at(t0, 210), "severity"), "warning"); // 175 s still
        assert_eq!(rank_field(&dog, at(t0, 220), "severity"), "critical"); // 185 s
        let status = &dog.statuses(at(t0, 220), true)["run"];
        assert_eq!(
            (&status["ranks"][0]["last_step"], &status["ranks"][0]["work"]),
            (&Value::Null, &json!(40))
        );
        assert_eq!(status["incidents"][0]["kind"], "job_stalled");
        // It was not the start limit that caught it (600 s here).
        assert!(dog.statuses(at(t0, 220), true)["run"]["restart"][0]["stepped"] == true);
    }

    #[test]
    fn a_first_report_above_zero_is_a_rank_that_has_stepped() {
        // The steps were taken before the first heartbeat arrived.
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        for second in (5..400).step_by(15) {
            dog.heartbeat(&key(0), "train", 15.0, Some(40), None, at(t0, second));
        }
        assert_eq!(states(&dog, at(t0, 60)), ["starting"]);
        assert_eq!(states(&dog, at(t0, 70)), ["silent"]);
    }

    #[test]
    fn a_floor_shorter_than_the_heartbeats_does_not_call_a_stepping_rank_silent_between_them() {
        // The server sees the counter once a beat (15 s here): with a floor of 2 s the threshold
        // is three beats, or a healthy rank would be silent between any two.
        let t0 = Instant::now();
        let dog = Watchdog::new(config(2));
        dog.stream_opened(&key(0), t0);
        for second in (1..200).step_by(15) {
            dog.heartbeat(&key(0), "train", 15.0, Some(second as i64 / 2), None, at(t0, second));
            assert_eq!(states(&dog, at(t0, second + 14)), ["starting"], "at {second} s");
        }
        assert_eq!(states(&dog, at(t0, 196 + 44)), ["starting"]);
    }

    #[test]
    fn a_slow_first_batch_is_not_a_stop_while_the_counter_moves() {
        let t0 = Instant::now();
        // A step every 30 s: 50 of them take 1,500 s, far past the start limit (600 s) and the
        // critical time (180 s), and nothing is said.
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        for second in (5..1500).step_by(15) {
            dog.heartbeat(&key(0), "train", 15.0, Some(second as i64 / 30), None, at(t0, second));
            assert_eq!(states(&dog, at(t0, second + 14)), ["starting"], "at {second} s");
        }
        assert_eq!(summary(&dog, at(t0, 1500)), "active");
        dog.batch(&key(0), 1, Some(49), Some(1), at(t0, 1500));
        assert_eq!(states(&dog, at(t0, 1501)), ["active"]);
        // A step every 400 s, slower than the floor: the time to the first step sets the threshold
        // (5 x 410 s as the server saw it), so the waits between steps are not stops.
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        for second in (5..3000).step_by(15) {
            dog.heartbeat(
                &key(0),
                "train",
                15.0,
                Some((second as i64 / 400).min(3)),
                None,
                at(t0, second),
            );
            assert_eq!(states(&dog, at(t0, second + 14)), ["starting"], "at {second} s");
        }
        // It stopped at step 3, seen at 1,205 s: silent 5 x 400 s later, critical after three of those.
        for second in (3005..9000).step_by(15) {
            dog.heartbeat(&key(0), "train", 15.0, Some(3), None, at(t0, second));
        }
        assert_eq!(states(&dog, at(t0, 1205 + 1900)), ["starting"]);
        assert_eq!(states(&dog, at(t0, 1205 + 2100)), ["silent"]);
        assert_eq!(rank_field(&dog, at(t0, 1205 + 5900), "severity"), "warning");
        assert_eq!(rank_field(&dog, at(t0, 1205 + 6100), "severity"), "critical");
    }

    #[test]
    fn a_slow_first_step_is_warned_of_and_not_called_critical_early() {
        // 700 s to the first step (a long compilation), with a start limit of 600 s: a warning from
        // 600 s that the first step ends. Critical would have come at 780 s.
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        let mut said = Vec::new();
        for second in (5..1000).step_by(15) {
            let work = if second < 700 {
                0
            } else {
                (second as i64 - 700) / 15 + 1
            };
            dog.heartbeat(&key(0), "train", 15.0, Some(work), None, at(t0, second));
            let severity = rank_field(&dog, at(t0, second + 10), "severity");
            if said.last() != Some(&severity) {
                said.push(severity);
            }
        }
        assert_eq!(said, [Value::Null, json!("warning"), Value::Null]);
        assert_eq!(states(&dog, at(t0, 999)), ["starting"]);
    }

    #[test]
    fn work_in_another_phase_before_the_first_step_is_a_sign_of_work_and_not_a_step() {
        // Lightning's sanity validation: ticks in `eval` for 1,000 s, then `train` and a first step
        // that takes long. The ticks keep it starting; the limit counts from the last of them.
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60)); // start limit 600 s
        dog.stream_opened(&key(0), t0);
        for second in (5..1000).step_by(15) {
            dog.heartbeat(&key(0), "eval", 15.0, Some(second as i64), None, at(t0, second));
            assert_eq!(states(&dog, at(t0, second + 14)), ["starting"]);
        }
        for second in (1000..2000).step_by(15) {
            dog.heartbeat(&key(0), "train", 15.0, Some(990), None, at(t0, second));
        }
        assert_eq!(dog.statuses(at(t0, 1500), true)["run"]["restart"][0]["stepped"], false);
        assert_eq!(states(&dog, at(t0, 1590)), ["starting"]); // 600 s from the change of phase at 1,000 s
        assert_eq!(states(&dog, at(t0, 1610)), ["silent"]);
        assert_eq!(rank_field(&dog, at(t0, 1610), "severity"), "warning");
    }

    #[test]
    fn a_declared_pause_before_the_first_batch_is_a_pause() {
        // 30 steps, then an evaluation that does not tick: paused up to the pause limit, as later on.
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60)); // pause limit 600 s
        dog.stream_opened(&key(0), t0);
        dog.heartbeat(&key(0), "train", 15.0, Some(0), None, at(t0, 5));
        dog.heartbeat(&key(0), "train", 15.0, Some(30), None, at(t0, 20));
        for second in (30..900).step_by(15) {
            dog.heartbeat(&key(0), "eval", 15.0, Some(30), None, at(t0, second));
        }
        assert_eq!(states(&dog, at(t0, 60)), ["starting"]);
        assert_eq!(states(&dog, at(t0, 300)), ["paused"]);
        assert_eq!(rank_field(&dog, at(t0, 300), "severity"), Value::Null);
        assert_eq!(states(&dog, at(t0, 700)), ["silent"]);
    }

    #[test]
    fn a_rank_that_had_not_stored_a_batch_gets_a_fresh_window_at_a_restart() {
        // config(60): floor 60 s, critical 180 s, start limit 600 s.
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        for second in (5..200).step_by(15) {
            let work = if second < 35 { 0 } else { 40 };
            dog.heartbeat(&key(0), "train", 15.0, Some(work), None, at(t0, second));
        }
        let before = dog.statuses(at(t0, 80), true)["run"].clone();
        assert_eq!(before["ranks"][0]["state"], "starting");
        assert_eq!(before["restart"][0]["step_seconds"], json!([0.875])); // 35 s over 40 steps
        let now = Instant::now();
        let fresh = restarted(&before, 60, now);
        fresh.stream_opened(&key(0), now);
        for second in (1..400).step_by(15) {
            fresh.heartbeat(&key(0), "train", 15.0, Some(40), None, at(now, second));
        }
        // It has stepped (the counter rose in training), so its window is its own threshold (60 s)
        // from the restart, not what was left of it when the server stopped (15 s).
        assert_eq!(states(&fresh, at(now, 10)), ["starting"]);
        assert_eq!(states(&fresh, at(now, 55)), ["starting"]);
        assert_eq!(states(&fresh, at(now, 65)), ["silent"]);
        assert_eq!(rank_field(&fresh, at(now, 170), "severity"), "warning");
        assert_eq!(rank_field(&fresh, at(now, 190), "severity"), "critical");
        // A status written before these were kept is still taken up, as a rank that has not
        // stepped: the start limit from the restart.
        let mut older = before.clone();
        for field in ["stepped", "step_seconds"] {
            older["restart"][0].as_object_mut().unwrap().remove(field);
        }
        let fresh = restarted(&older, 60, now);
        fresh.stream_opened(&key(0), now);
        for second in (1..700).step_by(15) {
            fresh.heartbeat(&key(0), "train", 15.0, Some(40), None, at(now, second));
        }
        assert_eq!(states(&fresh, at(now, 20)), ["starting"]);
        assert_eq!(states(&fresh, at(now, 590)), ["starting"]);
        assert_eq!(states(&fresh, at(now, 610)), ["silent"]);
        // Silent at warning when the server stopped, 85 s after the counter last moved (critical
        // was due at 180 s, 95 s later). The first file of the new server has no stopped rank, and
        // the warning's clock starts again: silent 60 s after the restart, critical 180 s after.
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        for second in (5..180).step_by(15) {
            let work = if second < 95 { 0 } else { 40 };
            dog.heartbeat(&key(0), "train", 15.0, Some(work), None, at(t0, second));
        }
        let before = dog.statuses(at(t0, 180), true)["run"].clone();
        assert_eq!(
            (&before["ranks"][0]["state"], &before["ranks"][0]["severity"]),
            (&json!("silent"), &json!("warning"))
        );
        let fresh = restarted(&before, 60, now);
        assert_eq!(states(&fresh, at(now, 1)), ["reconnecting"]);
        assert_eq!(rank_field(&fresh, at(now, 1), "severity"), Value::Null);
        assert!(fresh.statuses(at(now, 1), true)["run"]["incidents"]
            .as_array()
            .unwrap()
            .is_empty());
        fresh.stream_opened(&key(0), at(now, 55));
        for second in (56..400).step_by(15) {
            fresh.heartbeat(&key(0), "train", 15.0, Some(40), None, at(now, second));
        }
        assert_eq!(states(&fresh, at(now, 58)), ["starting"]);
        assert_eq!(states(&fresh, at(now, 65)), ["silent"]);
        assert_eq!(rank_field(&fresh, at(now, 170), "severity"), "warning");
        assert_eq!(rank_field(&fresh, at(now, 190), "severity"), "critical");
        // A rank that never stepped: a warning from the start limit (600 s), critical at 780 s,
        // and a restart at 650 s. It gets the start limit again from the restart, whether its
        // sender comes back (silent at 600 s) or not (lost at 600 s), and is critical 180 s later.
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        for second in (5..650).step_by(15) {
            dog.heartbeat(&key(0), "train", 15.0, Some(0), None, at(t0, second));
        }
        let before = dog.statuses(at(t0, 650), true)["run"].clone();
        assert_eq!(
            (&before["ranks"][0]["state"], &before["ranks"][0]["severity"]),
            (&json!("silent"), &json!("warning"))
        );
        let away = restarted(&before, 60, now);
        let back = restarted(&before, 60, now);
        back.stream_opened(&key(0), at(now, 55));
        for second in (56..900).step_by(15) {
            back.heartbeat(&key(0), "train", 15.0, Some(0), None, at(now, second));
        }
        for (dog, after) in [(&away, "lost"), (&back, "silent")] {
            assert_eq!(rank_field(dog, at(now, 1), "severity"), Value::Null);
            assert_eq!(
                states(dog, at(now, 590)),
                [if after == "lost" { "reconnecting" } else { "starting" }]
            );
            assert_eq!(states(dog, at(now, 610)), [after]);
            assert_eq!(rank_field(dog, at(now, 610), "severity"), "warning");
            assert_eq!(rank_field(dog, at(now, 770), "severity"), "warning");
            assert_eq!(rank_field(dog, at(now, 790), "severity"), "critical");
        }
    }

    #[test]
    fn the_cost_of_the_fresh_window_with_the_default_timings() {
        // The defaults: floor 300 s, critical 900 s, start limit 1,800 s. A rank hung for 800 s
        // (a warning, 100 s short of critical) when the server restarts: its sender comes back at
        // once and nothing more arrives. Critical 900 s after the restart instead of 100 s: called
        // 800 s later, the time it had already been quiet.
        let defaults = WatchdogConfig::default();
        let t0 = Instant::now();
        let dog = Watchdog::new(defaults);
        dog.stream_opened(&key(0), t0);
        dog.batch(&key(0), 1, Some(49), Some(1), at(t0, 10));
        for second in (15..810).step_by(15) {
            dog.heartbeat(&key(0), "train", 15.0, Some(50), None, at(t0, second));
        }
        let before = dog.statuses(at(t0, 810), true)["run"].clone();
        assert_eq!(
            (&before["ranks"][0]["state"], &before["ranks"][0]["severity"]),
            (&json!("silent"), &json!("warning"))
        );
        let now = Instant::now();
        let fresh = Watchdog::new(defaults);
        assert_eq!(fresh.restore("run", &as_file(&before), now), Ok(true));
        fresh.stream_opened(&key(0), at(now, 1));
        for second in (2..1000).step_by(15) {
            fresh.heartbeat(&key(0), "train", 15.0, Some(50), None, at(now, second));
        }
        assert_eq!(states(&fresh, at(now, 100)), ["active"]); // critical here before the change
        assert_eq!(states(&fresh, at(now, 299)), ["active"]);
        assert_eq!(states(&fresh, at(now, 301)), ["silent"]);
        assert_eq!(rank_field(&fresh, at(now, 899), "severity"), "warning");
        assert_eq!(rank_field(&fresh, at(now, 901), "severity"), "critical");
        // What the status shows is kept: 800 s since its last progress at the restart.
        assert_eq!(
            fresh.statuses(now, true)["run"]["ranks"][0]["seconds_since_progress"],
            800.0
        );
        // A job killed 800 s before the restart (lost, a warning): lost again 300 s after it and
        // critical 900 s after, where it was due 100 s after; 800 s later too.
        let dog = Watchdog::new(defaults);
        dog.stream_opened(&key(0), t0);
        dog.batch(&key(0), 1, Some(49), Some(1), at(t0, 10));
        dog.stream_ended(&key(0), false, at(t0, 10));
        let before = dog.statuses(at(t0, 810), true)["run"].clone();
        assert_eq!(
            (&before["ranks"][0]["state"], &before["ranks"][0]["severity"]),
            (&json!("lost"), &json!("warning"))
        );
        let fresh = Watchdog::new(defaults);
        assert_eq!(fresh.restore("run", &as_file(&before), now), Ok(true));
        assert_eq!(states(&fresh, at(now, 299)), ["reconnecting"]);
        assert_eq!(states(&fresh, at(now, 301)), ["lost"]);
        assert_eq!(rank_field(&fresh, at(now, 899), "severity"), "warning");
        assert_eq!(rank_field(&fresh, at(now, 901), "severity"), "critical");
        // A rank that never stepped: 1,800 s from the restart, then critical 900 s after that.
        let dog = Watchdog::new(defaults);
        dog.stream_opened(&key(0), t0);
        let before = dog.statuses(at(t0, 100), true)["run"].clone();
        let fresh = Watchdog::new(defaults);
        assert_eq!(fresh.restore("run", &as_file(&before), now), Ok(true));
        assert_eq!(states(&fresh, at(now, 1799)), ["reconnecting"]);
        assert_eq!(states(&fresh, at(now, 1801)), ["lost"]);
        assert_eq!(rank_field(&fresh, at(now, 2699), "severity"), "warning");
        assert_eq!(rank_field(&fresh, at(now, 2701), "severity"), "critical");
    }

    #[test]
    fn a_job_started_again_by_a_sender_without_attempts_reads_stalled_until_it_stores_a_batch() {
        // A sender from before Nov 17, 2026 names no attempt in its heartbeats: the server learns a
        // new attempt only from its first stored batch. Until then a rank whose new process has
        // connected is judged from the old attempt's last batch, so it reads silent and critical at
        // once. The trigger does not fire for it: the status's `attempt` and `last_step` are still
        // the old attempt's, the stall episode that already fired.
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60)); // critical after 180 s
        dog.stream_opened(&key(0), t0);
        dog.batch(&key(0), 1, Some(99), Some(1), at(t0, 5));
        dog.stream_ended(&key(0), false, at(t0, 6)); // killed
        assert_eq!(rank_field(&dog, at(t0, 300), "severity"), "critical");
        // Started again: a new process connects and beats, and has stored nothing yet.
        dog.stream_opened(&key(0), at(t0, 400));
        dog.heartbeat(&key(0), "train", 15.0, Some(0), None, at(t0, 401));
        assert_eq!(states(&dog, at(t0, 402)), ["silent"]);
        assert_eq!(rank_field(&dog, at(t0, 402), "severity"), "critical");
        assert_eq!(rank_field(&dog, at(t0, 402), "attempt"), 1);
        // Its first batch, of attempt 2 and below the old highest step: active, attempt 2.
        dog.batch(&key(0), 2, Some(29), Some(1), at(t0, 450));
        assert_eq!(states(&dog, at(t0, 451)), ["active"]);
        assert_eq!(rank_field(&dog, at(t0, 451), "attempt"), 2);
        assert_eq!(rank_field(&dog, at(t0, 451), "last_step"), 99);
    }

    #[test]
    fn a_new_attempt_seen_in_its_heartbeats_gets_a_start_window_of_its_own() {
        // config(60): floor 60 s, critical 180 s, start limit 600 s.
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        dog.heartbeat(&key(0), "train", 15.0, Some(0), None, at(t0, 1));
        dog.batch(&key(0), 1, Some(99), Some(1), at(t0, 5));
        dog.stream_ended(&key(0), false, at(t0, 6)); // killed
        assert_eq!(rank_field(&dog, at(t0, 300), "severity"), "critical");
        // Started again: the new process's first beat comes before its stream is open.
        assert!(dog.attempt_seen(&key(0), 2, at(t0, 400)));
        dog.heartbeat(&key(0), "train", 15.0, Some(0), None, at(t0, 400));
        let status = &dog.statuses(at(t0, 401), true)["run"];
        assert_eq!(
            (
                &status["ranks"][0]["attempt"],
                &status["ranks"][0]["attempt_without_steps"]
            ),
            (&json!(2), &json!(true))
        );
        assert_eq!(status["ranks"][0]["state"], "reconnecting");
        assert_eq!(status["ranks"][0]["last_step"], 99); // the earlier attempt's, as stored
        assert!(status["incidents"].as_array().unwrap().is_empty());
        // Its stream opens; it beats and takes no step: starting for the start limit from its first
        // beat, a warning after it, critical 180 s later.
        dog.stream_opened(&key(0), at(t0, 402));
        for second in (415..1500).step_by(15) {
            dog.heartbeat(&key(0), "train", 15.0, Some(0), None, at(t0, second));
        }
        assert_eq!(states(&dog, at(t0, 403)), ["starting"]);
        assert_eq!(states(&dog, at(t0, 999)), ["starting"]);
        assert_eq!(states(&dog, at(t0, 1001)), ["silent"]);
        assert_eq!(rank_field(&dog, at(t0, 1001), "severity"), "warning");
        assert_eq!(rank_field(&dog, at(t0, 1179), "severity"), "warning");
        assert_eq!(rank_field(&dog, at(t0, 1181), "severity"), "critical");
        // A reconnect of the same attempt, and a beat of the earlier attempt's process still alive,
        // change nothing; the earlier attempt's rows sent late are kept and are not its progress.
        assert!(dog.attempt_seen(&key(0), 2, at(t0, 1182)));
        assert!(!dog.attempt_seen(&key(0), 1, at(t0, 1183)));
        dog.batch(&key(0), 1, Some(120), Some(1), at(t0, 1184));
        let status = &dog.statuses(at(t0, 1185), true)["run"];
        assert_eq!(
            (
                &status["ranks"][0]["attempt"],
                &status["ranks"][0]["attempt_without_steps"],
                &status["ranks"][0]["last_step"]
            ),
            (&json!(2), &json!(true), &json!(120))
        );
        assert_eq!(status["ranks"][0]["severity"], "critical");
        // Its first batch, below the old highest step: active, and judged by its batches from now.
        dog.batch(&key(0), 2, Some(29), Some(1), at(t0, 1200));
        let status = &dog.statuses(at(t0, 1201), true)["run"];
        assert_eq!(status["ranks"][0]["state"], "active");
        assert!(status["ranks"][0].get("attempt_without_steps").is_none());
        // Its first gap is from its first beat (800 s), not from the earlier attempt's last batch.
        assert_eq!(status["restart"][0]["gaps_seconds"], json!([5.0, 800.0]));
    }

    #[test]
    fn a_new_attempt_without_steps_keeps_its_window_over_a_server_restart() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        dog.batch(&key(0), 1, Some(99), Some(1), at(t0, 5));
        assert!(dog.attempt_seen(&key(0), 2, at(t0, 100)));
        for second in (100..400).step_by(15) {
            dog.heartbeat(&key(0), "train", 15.0, Some(0), None, at(t0, second));
        }
        let before = dog.statuses(at(t0, 400), true)["run"].clone();
        assert_eq!(before["restart"][0]["attempt_without_steps"], true);
        let now = Instant::now();
        let fresh = restarted(&before, 60, now);
        assert_eq!(rank_field(&fresh, now, "attempt_without_steps"), true);
        // The same process reconnects: not a new attempt, and the start limit from the restart.
        fresh.stream_opened(&key(0), now);
        assert!(fresh.attempt_seen(&key(0), 2, at(now, 1)));
        for second in (1..700).step_by(15) {
            fresh.heartbeat(&key(0), "train", 15.0, Some(0), None, at(now, second));
        }
        assert_eq!(states(&fresh, at(now, 590)), ["starting"]);
        assert_eq!(states(&fresh, at(now, 610)), ["silent"]);
        // A status written before the field was kept restores as an attempt that has stored steps.
        let mut older = before.clone();
        older["restart"][0]
            .as_object_mut()
            .unwrap()
            .remove("attempt_without_steps");
        let fresh = restarted(&older, 60, now);
        assert_eq!(rank_field(&fresh, now, "attempt_without_steps"), Value::Null);
    }

    #[test]
    fn a_new_attempt_that_hangs_before_its_first_step_is_critical_2700_s_after_its_first_beat_by_default() {
        let defaults = WatchdogConfig::default();
        let t0 = Instant::now();
        let dog = Watchdog::new(defaults);
        dog.stream_opened(&key(0), t0);
        dog.batch(&key(0), 1, Some(99), Some(1), at(t0, 5));
        dog.stream_ended(&key(0), false, at(t0, 6));
        assert!(dog.attempt_seen(&key(0), 2, at(t0, 4000)));
        dog.stream_opened(&key(0), at(t0, 4001));
        for second in (4000..7000).step_by(15) {
            dog.heartbeat(&key(0), "train", 15.0, Some(0), None, at(t0, second));
        }
        assert_eq!(states(&dog, at(t0, 4000 + 1799)), ["starting"]);
        assert_eq!(rank_field(&dog, at(t0, 4000 + 1801), "severity"), "warning");
        assert_eq!(rank_field(&dog, at(t0, 4000 + 2699), "severity"), "warning");
        assert_eq!(rank_field(&dog, at(t0, 4000 + 2701), "severity"), "critical");
    }

    #[test]
    fn a_rank_whose_heartbeats_had_stopped_stays_unresponsive_after_its_window() {
        // A frozen process cannot reconnect: after its window it is not called a lost connection.
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        dog.batch(&key(0), 1, Some(9), Some(1), at(t0, 1));
        dog.heartbeat(&key(0), "train", 15.0, Some(10), None, at(t0, 5));
        assert_eq!(states(&dog, at(t0, 200)), ["unresponsive"]);
        let before = dog.statuses(at(t0, 200), true)["run"].clone();
        let now = Instant::now();
        let fresh = restarted(&before, 60, now);
        assert_eq!(states(&fresh, at(now, 59)), ["reconnecting"]);
        assert_eq!(states(&fresh, at(now, 61)), ["unresponsive"]);
        let status = &fresh.statuses(at(now, 200), true)["run"];
        assert_eq!(status["incidents"][0]["assessment"], "suspected_hang");
        assert_eq!(status["ranks"][0]["severity"], "critical"); // 180 s after the restart
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
            dog.heartbeat(&key(0), "eval", 15.0, Some(7), None, at(t0, second));
        }
        assert_eq!(states(&dog, at(t0, 300)), ["paused"]);
        assert_eq!(states(&dog, at(t0, 500)), ["paused"]);
        assert_eq!(states(&dog, at(t0, 800)), ["silent"]);
        // Work that keeps moving (an evaluation batch per beat) is not limited.
        let dog = new();
        for (n, second) in (10..5000).step_by(15).enumerate() {
            dog.heartbeat(&key(0), "eval", 15.0, Some(7 + n as i64), None, at(t0, second));
        }
        assert_eq!(states(&dog, at(t0, 4900)), ["paused"]);
        // Work that moved and then stopped is a stuck evaluation, whatever the limit.
        let dog = new();
        for (n, second) in (10..400).step_by(15).enumerate() {
            let work = if second < 100 { 7 + n as i64 } else { 12 };
            dog.heartbeat(&key(0), "eval", 15.0, Some(work), None, at(t0, second));
        }
        assert_eq!(states(&dog, at(t0, 399)), ["silent"]);
    }

    /// The status as a file holds it: written as JSON and read back.
    fn as_file(status: &Value) -> Value {
        serde_json::from_slice(&serde_json::to_vec_pretty(status).unwrap()).unwrap()
    }

    #[test]
    fn a_stalled_ranks_stack_is_kept_until_it_works_again_and_over_a_restart() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        dog.batch(&key(0), 1, Some(9), Some(1), at(t0, 1));
        dog.heartbeat(&key(0), "train", 15.0, Some(10), None, at(t0, 5));
        assert_eq!(rank_field(&dog, at(t0, 6), "stack"), Value::Null);
        let stack = json!({"still_seconds": 61.0, "unchanged_seconds": 0.0, "threads": [
            {"name": "MainThread", "main": true, "frames": [{"file": "train.py", "line": 7, "function": "step"}]},
        ]});
        dog.heartbeat(&key(0), "train", 15.0, Some(10), Some(stack.clone()), at(t0, 70));
        // The heartbeats stop: the rank is unresponsive and still says where it was.
        assert_eq!(states(&dog, at(t0, 200)), ["unresponsive"]);
        assert_eq!(rank_field(&dog, at(t0, 200), "stack"), stack);
        let before = dog.statuses(at(t0, 200), true)["run"].clone();
        let now = Instant::now();
        let fresh = restarted(&before, 60, now);
        assert_eq!(rank_field(&fresh, now, "stack"), stack);
        // A beat with no stack is the rank at work again.
        dog.heartbeat(&key(0), "train", 15.0, Some(11), None, at(t0, 210));
        assert_eq!(rank_field(&dog, at(t0, 211), "stack"), Value::Null);
    }

    /// A new server, started `now`, takes up `status` (a file the old server wrote).
    fn restarted(status: &Value, floor: u64, now: Instant) -> Watchdog {
        let dog = Watchdog::new(config(floor));
        assert_eq!(dog.restore("run", &as_file(status), now), Ok(true));
        dog
    }

    #[test]
    fn a_stalled_run_gets_a_fresh_window_at_a_restart_and_keeps_the_quiet_time_it_shows() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60)); // critical after 180 s
        for rank in 0..2 {
            dog.stream_opened(&key(rank), t0);
            dog.batch(&key(rank), 1, Some(9), Some(2), at(t0, 5));
        }
        // Both ranks have been quiet for 200 s: silent, critical, the job stalled.
        for running in [true, false] {
            // `false` is the file a clean stop leaves (`server_stopped`, no incidents); `true` the
            // last file of a server that was killed.
            let before = dog.statuses(at(t0, 205), running)["run"].clone();
            assert_eq!(before["summary"], if running { "stalled" } else { "server_stopped" });
            let now = Instant::now();
            let fresh = restarted(&before, 60, now);
            // The first status after the restart, before any sender has reconnected: no stall and
            // no incident, while the senders have their window (the threshold, 60 s); the quiet
            // time it shows is the one it had.
            let status = &fresh.statuses(now, true)["run"];
            assert_eq!(status["summary"], "active");
            assert_eq!(states(&fresh, now), ["reconnecting", "reconnecting"]);
            assert!(status["incidents"].as_array().unwrap().is_empty());
            assert_eq!(status["ranks"][0]["seconds_since_progress"], 200.0);
            assert_eq!(status["ranks"][0]["last_step"], 9);
            // A sender that comes back with nothing new is judged from the restart: active for its
            // threshold, then silent. A new batch keeps the other active.
            fresh.stream_opened(&key(0), now);
            fresh.stream_opened(&key(1), now);
            fresh.batch(&key(1), 1, Some(19), Some(2), at(now, 2));
            assert_eq!(states(&fresh, at(now, 59)), ["active", "active"]);
            assert_eq!(states(&fresh, at(now, 61)), ["silent", "active"]);
            assert_eq!(fresh.statuses(at(now, 61), true)["run"]["summary"], "degraded");
            assert_eq!(rank_field(&fresh, at(now, 179), "severity"), "warning");
            assert_eq!(rank_field(&fresh, at(now, 181), "severity"), "critical");
            // A rank whose sender never returns is reconnecting for the window, then lost.
            let away = restarted(&before, 60, now);
            assert_eq!(states(&away, at(now, 59)), ["reconnecting", "reconnecting"]);
            assert_eq!(states(&away, at(now, 61)), ["lost", "lost"]);
            let status = &away.statuses(at(now, 61), true)["run"];
            assert_eq!(
                (
                    &status["summary"],
                    &status["incidents"][0]["assessment"],
                    &status["incidents"][0]["severity"]
                ),
                (&json!("stalled"), &json!("lost_connection"), &json!("warning"))
            );
        }
    }

    #[test]
    fn time_the_server_was_not_running_is_not_held_against_a_rank() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        dog.batch(&key(0), 1, Some(9), Some(1), at(t0, 5));
        let before = dog.statuses(at(t0, 8), true)["run"].clone(); // active, 3 s since progress
        let now = Instant::now();
        let fresh = restarted(&before, 60, now);
        // However long the outage was, the file says only how long things had lasted.
        let status = &fresh.statuses(now, true)["run"];
        assert_eq!(status["ranks"][0]["seconds_since_progress"], 3.0);
        assert_eq!(
            (status["summary"].as_str(), states(&fresh, now)),
            (Some("active"), vec!["reconnecting".to_string()])
        );
        // Its sender comes back: active at once. If nobody does, it is lost after the floor.
        assert_eq!(states(&fresh, at(now, 59)), ["reconnecting"]);
        assert_eq!(states(&fresh, at(now, 61)), ["lost"]);
        fresh.stream_opened(&key(0), at(now, 10));
        assert_eq!(states(&fresh, at(now, 11)), ["active"]);
    }

    #[test]
    fn a_slow_ranks_cadence_survives_a_restart() {
        // Batches 400 s apart: the threshold is 5 x 400 s, not the 60 s floor.
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        for n in 1..=4u64 {
            dog.batch(&key(0), 1, Some(n as i64 * 50), Some(1), at(t0, n * 400));
        }
        let before = dog.statuses(at(t0, 1600 + 1500), true)["run"].clone();
        assert_eq!(before["summary"], "active");
        let now = Instant::now();
        let fresh = restarted(&before, 60, now);
        fresh.stream_opened(&key(0), now);
        // Its threshold (5 x 400 s) is kept, and is its window from the restart: 1,600 s quiet
        // when the server stopped, active until 2,000 s after the restart, silent after that.
        assert_eq!(states(&fresh, at(now, 600)), ["active"]);
        assert_eq!(states(&fresh, at(now, 1999)), ["active"]);
        assert_eq!(states(&fresh, at(now, 2001)), ["silent"]);
    }

    #[test]
    fn a_pause_in_progress_and_the_pauses_measured_survive_a_restart() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60)); // pause limit 600 s
        dog.stream_opened(&key(0), t0);
        dog.batch(&key(0), 1, Some(49), Some(1), at(t0, 1));
        dog.heartbeat(&key(0), "checkpoint", 15.0, None, None, at(t0, 2));
        dog.heartbeat(&key(0), "train", 15.0, None, None, at(t0, 62)); // a 60 s checkpoint, on record
        dog.heartbeat(&key(0), "eval", 15.0, None, None, at(t0, 100));
        for second in (115..250).step_by(15) {
            dog.heartbeat(&key(0), "eval", 15.0, None, None, at(t0, second));
        }
        let before = dog.statuses(at(t0, 250), true)["run"].clone();
        assert_eq!(states(&dog, at(t0, 250)), ["paused"]);
        let now = Instant::now();
        let fresh = restarted(&before, 60, now);
        let status = &fresh.statuses(now, true)["run"];
        assert_eq!(status["pauses"]["checkpoint"]["max_seconds"], 60.0);
        assert_eq!(status["recommended_silent_seconds"], 120.0);
        assert_eq!(status["ranks"][0]["phase"], "eval");
        assert_eq!(status["ranks"][0]["phase_seconds"], 150.0);
        // Its sender and heartbeats come back: active for its window (60 s from the restart),
        // then paused, not silent. Training resumes and the pause is measured without the outage.
        fresh.stream_opened(&key(0), now);
        fresh.heartbeat(&key(0), "eval", 15.0, None, None, at(now, 15));
        assert_eq!(states(&fresh, at(now, 16)), ["active"]);
        assert_eq!(states(&fresh, at(now, 25)), ["active"]);
        let longer = restarted(&before, 60, now);
        longer.stream_opened(&key(0), now);
        for second in (15..200).step_by(15) {
            longer.heartbeat(&key(0), "eval", 15.0, None, None, at(now, second));
        }
        assert_eq!(states(&longer, at(now, 70)), ["paused"]);
        fresh.heartbeat(&key(0), "train", 15.0, None, None, at(now, 30));
        let status = &fresh.statuses(at(now, 31), true)["run"];
        assert_eq!(status["pauses"]["eval"]["count"], 1);
        assert_eq!(status["pauses"]["eval"]["max_seconds"], 180.0);
        assert_eq!(status["pauses"]["checkpoint"]["count"], 1);
    }

    #[test]
    fn missing_ranks_and_the_world_size_survive_a_restart() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        for rank in [0, 1] {
            dog.stream_opened(&key(rank), t0);
            dog.batch(&key(rank), 1, Some(9), Some(4), at(t0, 1));
            dog.batch(&key(rank), 1, Some(19), Some(4), at(t0, 70));
        }
        let before = dog.statuses(at(t0, 71), true)["run"].clone();
        assert_eq!(before["missing_ranks"], json!([2, 3]));
        let now = Instant::now();
        let fresh = restarted(&before, 60, now);
        // The run was seen for longer than the floor: the missing ranks are still named at once.
        let status = &fresh.statuses(now, true)["run"];
        assert_eq!(
            (&status["world_size"], &status["missing_ranks"]),
            (&json!(4), &json!([2, 3]))
        );
    }

    #[test]
    fn finished_abandoned_and_unusable_runs_are_not_taken_up() {
        let t0 = Instant::now();
        let fresh = || Watchdog::new(config(60)); // abandoned after 6,000 s
        let new_run = || {
            let dog = fresh();
            for rank in 0..2 {
                dog.stream_opened(&key(rank), t0);
                dog.batch(&key(rank), 1, Some(9), Some(2), at(t0, 1));
            }
            dog
        };
        let restore = |status: &Value| {
            let dog = fresh();
            let result = dog.restore("run", &as_file(status), t0);
            (result, dog.statuses(t0, true).len())
        };
        // Finished: nothing to watch.
        let dog = new_run();
        for rank in 0..2 {
            dog.stream_ended(&key(rank), true, at(t0, 2));
        }
        let status = dog.statuses(at(t0, 3), true)["run"].clone();
        assert_eq!(status["summary"], "finished");
        assert_eq!(restore(&status), (Ok(false), 0));
        // Both streams dropped long ago: abandoned. With one stream open and hung, never.
        let dog = new_run();
        for rank in 0..2 {
            dog.stream_ended(&key(rank), false, at(t0, 2));
        }
        let status = dog.statuses(at(t0, 2 + 6000), true)["run"].clone();
        assert_eq!(
            (&status["abandoned"], &status["summary"]),
            (&json!(true), &json!("stalled"))
        );
        assert_eq!(restore(&status), (Ok(false), 0));
        assert_eq!(dog.statuses(at(t0, 2 + 5999), true)["run"]["abandoned"], false);
        let dog = new_run();
        dog.stream_ended(&key(0), false, at(t0, 2));
        let status = dog.statuses(at(t0, 100_000), true)["run"].clone();
        assert_eq!(status["abandoned"], false); // rank 1 is hung with its stream open
        assert_eq!(restore(&status).0, Ok(true));
        // A status an older server wrote has no `restart`.
        let mut old = status.clone();
        old.as_object_mut().unwrap().remove("restart");
        assert_eq!(restore(&old), (Ok(false), 0));
        // A status that cannot be read takes up none of the run's ranks.
        let mut bad = status.clone();
        bad["restart"][1]["gaps_seconds"] = json!("x");
        let (result, ranks) = restore(&bad);
        assert!(result.is_err() && ranks == 0, "{result:?} {ranks}");
        for (field, value) in [("run_id", json!("other")), ("schema_version", json!(2))] {
            let mut wrong = status.clone();
            wrong[field] = value;
            assert!(restore(&wrong).0.is_err());
        }
        let mut negative = status.clone();
        negative["ranks"][0]["seconds_since_progress"] = json!(-1.0);
        assert!(restore(&negative).0.is_err());
    }

    #[test]
    fn a_rank_that_connected_before_the_restore_is_left_alone() {
        let t0 = Instant::now();
        let dog = Watchdog::new(config(60));
        dog.stream_opened(&key(0), t0);
        dog.batch(&key(0), 1, Some(9), Some(1), at(t0, 1));
        let before = dog.statuses(at(t0, 5), true)["run"].clone();
        let fresh = Watchdog::new(config(60));
        let now = Instant::now();
        fresh.stream_opened(&key(0), now);
        assert_eq!(fresh.restore("run", &as_file(&before), now), Ok(true));
        let status = &fresh.statuses(now, true)["run"];
        assert_eq!(status["ranks"][0]["last_step"], Value::Null);
        assert_eq!(status["ranks"][0]["stream"], "open");
    }

    #[test]
    fn an_age_longer_than_the_host_has_been_up_is_never_taken_for_zero() {
        let now = Instant::now();
        // Whatever the platform's clock reaches back to, the age is positive, never above what was
        // asked, and nothing panics for absurd values.
        for seconds in [
            1.0,
            86_400.0 * 5.0,
            86_400.0 * 365.0 * 40.0,
            f64::MAX,
            f64::INFINITY,
            f64::NAN,
            -3.0,
        ] {
            let age = now.saturating_duration_since(ago(now, seconds));
            let asked = Duration::try_from_secs_f64(seconds).unwrap_or(Duration::from_secs(10 * 365 * 86_400));
            assert!(
                age <= asked.min(Duration::from_secs(10 * 365 * 86_400)),
                "{seconds}: {age:?}"
            );
            if seconds >= 1.0 {
                assert!(age > Duration::ZERO, "{seconds}");
            }
        }
        assert_eq!(
            now.saturating_duration_since(ago(now, 1.5)),
            Duration::from_millis(1500)
        );
    }

    fn state_of(status: &Value, reports: &[(i32, &[&str], &[&str])], detect: bool) -> (String, Vec<String>) {
        let reports: BTreeMap<i32, RankReport> = reports
            .iter()
            .map(|(rank, kinds, checks)| {
                (
                    *rank,
                    RankReport {
                        kinds: kinds.iter().map(|k| k.to_string()).collect(),
                        not_checked: checks.iter().map(|k| k.to_string()).collect(),
                    },
                )
            })
            .collect();
        let found = run_state(status, &reports, detect);
        let codes = found["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["code"].as_str().unwrap().to_string())
            .collect();
        (found["state"].as_str().unwrap().to_string(), codes)
    }

    fn two_ranks(summary: &str) -> Value {
        json!({"summary": summary, "ranks": [{"rank": 0}, {"rank": 1}], "missing_ranks": [], "incidents": [], "world_size": 2})
    }

    #[test]
    fn the_run_state_is_healthy_only_when_every_rank_is_judged_and_clean() {
        let none: &[&str] = &[];
        let clean = [(0, none, none), (1, none, none)];
        assert_eq!(state_of(&two_ranks("active"), &clean, true), ("healthy".into(), vec![]));
        assert_eq!(
            state_of(&two_ranks("finished"), &clean, true),
            ("healthy".into(), vec![])
        );
        // Rank 1 not judged by this server (no part yet, detectors off or failed).
        assert_eq!(
            state_of(&two_ranks("active"), &clean[..1], true),
            ("unknown".into(), vec!["ranks_unknown".into()])
        );
        assert_eq!(
            state_of(&two_ranks("active"), &clean, false),
            ("unknown".into(), vec!["no_steps".into(), "ranks_unknown".into()])
        );
        // A rank below world_size that never connected, before the status calls it missing.
        let mut early = two_ranks("active");
        early["world_size"] = json!(3);
        assert_eq!(state_of(&early, &clean, true).1, vec!["ranks_unknown".to_string()]);
        // A rank the status calls missing, with no world_size to expect it by.
        let mut missing = two_ranks("active");
        missing["world_size"] = Value::Null;
        missing["missing_ranks"] = json!([5]);
        let found = run_state(
            &missing,
            &clean
                .iter()
                .map(|(rank, _, _)| (*rank, RankReport::default()))
                .collect(),
            true,
        );
        assert_eq!(
            (found["state"].as_str(), &found["reasons"][0]["ranks"]),
            (Some("unknown"), &json!([5]))
        );
        assert_eq!(
            state_of(&two_ranks("server_stopped"), &clean, true),
            ("unknown".into(), vec!["server_stopped".into()])
        );
        assert_eq!(state_of(&two_ranks("whatever comes next"), &clean, true).0, "unknown");
        assert_eq!(
            state_of(&two_ranks("stalled"), &clean, true),
            ("unknown".into(), vec!["status_summary".into()])
        );
        let checks: &[&str] = &["resume_lr"];
        assert_eq!(
            state_of(&two_ranks("active"), &[(0, none, checks), (1, none, none)], true),
            ("degraded".into(), vec!["not_checked".into()])
        );
    }

    #[test]
    fn an_incident_on_a_rank_or_in_the_stream_is_unhealthy() {
        let none: &[&str] = &[];
        let spike: &[&str] = &["grad_spike"];
        let (state, codes) = state_of(&two_ranks("active"), &[(0, none, none), (1, spike, none)], true);
        assert_eq!((state.as_str(), codes), ("unhealthy", vec!["incidents".to_string()]));
        // A stall is an observation of the stream (a hang, a long pause and a dead sender look
        // the same), so the stalled summary decides: unknown, never healthy.
        let mut stalled = two_ranks("stalled");
        stalled["incidents"] = json!([{"kind": "job_stalled", "step": 4}]);
        assert_eq!(
            state_of(&stalled, &[(0, none, none), (1, none, none)], true),
            ("unknown".into(), vec!["status_summary".into(), "observations".into()])
        );
        let mut silent = two_ranks("degraded");
        silent["incidents"] = json!([{"kind": "rank_silent", "step": 4}]);
        assert_eq!(
            state_of(&silent, &[(0, none, none), (1, none, none)], true).0,
            "unknown"
        );
        // Unhealthy wins over unknown, which wins over degraded; every reason is kept.
        let checks: &[&str] = &["probe"];
        let (state, codes) = state_of(&two_ranks("server_stopped"), &[(0, spike, checks)], true);
        assert_eq!(state, "unhealthy");
        assert_eq!(codes, ["incidents", "ranks_unknown", "server_stopped", "not_checked"]);
    }

    #[test]
    fn observations_are_listed_and_never_make_a_run_worse() {
        let none: &[&str] = &[];
        let observed: &[&str] = &["gpu_throttled", "resume_lr_change", "gpu_throttled"];
        let reports: &[(i32, &[&str], &[&str])] = &[(0, observed, none), (1, none, none)];
        assert_eq!(
            state_of(&two_ranks("active"), reports, true),
            ("healthy".into(), vec!["observations".into()])
        );
        let found = run_state(
            &two_ranks("active"),
            &[(
                0,
                RankReport {
                    kinds: observed.iter().map(|k| k.to_string()).collect(),
                    not_checked: vec![],
                },
            )]
            .into_iter()
            .chain([(1, RankReport::default())])
            .collect(),
            true,
        );
        assert_eq!(found["reasons"][0]["state"], "healthy");
        assert_eq!(
            found["reasons"][0]["kinds"],
            json!(["gpu_throttled", "resume_lr_change"])
        );
        assert_eq!(
            found["reasons"][0]["detail"],
            "3 observations listed, not counted as failures: gpu_throttled, resume_lr_change"
        );
        // Beside an anomaly: unhealthy, and the observations are still listed after the rest.
        let both: &[&str] = &["gpu_throttled", "grad_spike"];
        assert_eq!(
            state_of(&two_ranks("active"), &[(0, both, none), (1, none, none)], true),
            ("unhealthy".into(), vec!["incidents".into(), "observations".into()])
        );
        // Nor lifted: a rank without a report is still unknown.
        assert_eq!(
            state_of(&two_ranks("active"), &[(0, observed, none)], true),
            ("unknown".into(), vec!["ranks_unknown".into(), "observations".into()])
        );
        // A kind this version does not know is not assumed harmless.
        let new: &[&str] = &["something_new"];
        assert_eq!(
            state_of(&two_ranks("active"), &[(0, new, none), (1, none, none)], true),
            ("unhealthy".into(), vec!["incidents".into()])
        );
    }

    #[test]
    fn the_status_incidents_carry_their_class() {
        let mut status = two_ranks("stalled");
        status["ranks"] = json!([{"rank": 0, "state": "silent", "last_step": 3}]);
        status["duplicate_batches"] = json!({"step": 2, "ranks": [0, 1]});
        let classes: Vec<(String, String)> = incidents(&status)
            .iter()
            .map(|i| (i["kind"].as_str().unwrap().into(), i["class"].as_str().unwrap().into()))
            .collect();
        assert_eq!(
            classes,
            [
                ("job_stalled".into(), "observation".into()),
                ("duplicate_batches".into(), "invariant".into())
            ]
        );
    }
}
