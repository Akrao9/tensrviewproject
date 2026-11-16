//! Stateful incident detectors over step batches: `evidence.py` in Rust.
//!
//! The Python detectors (`sdk/src/tensorquill/evidence.py`) are the reference. They rerun
//! over a whole run on every look, which the 64-rank benchmark measured at 13 times the
//! plan's 250 ms budget even with state kept between flushes. This module keeps each
//! detector's state per `(run_id, rank)` and reads Arrow batches as they are, so a flush
//! costs its own rows only.
//!
//! Every kind that is read from a rank's step records is here ([`PORTED`]), each to give
//! what `build_evidence` gives on the same store. `job_stalled` and `rank_silent` come from
//! the progress status and are in `watchdog.rs`. The three kinds left to Python
//! ([`NOT_PORTED`]) read files the server never receives: PyTorch's Flight Recorder dumps
//! and TrainCheck's results.
//!
//! Python reads a finished list of records and so can look ahead; a stream cannot. The
//! rule here is that the state after any prefix of a rank's records equals what
//! `build_evidence` gives on that prefix:
//!
//! - *Clipping.* Python reads `clip_grad_norm` for every record once any record has it,
//!   and so skips the records before clipping began. The first record with it resets the
//!   gradient state here, which gives the same result.
//! - *Experts per token.* Python applies the last `moe_top_k` recorded in the run to
//!   every record; here a record is judged with the value known when it arrives. The SDK
//!   writes `moe_top_k` on every record, so the two differ only for hand-made records
//!   that change it in mid-run.
//! - *Attempts.* A resume rolls back steps that were already judged. That cannot be
//!   undone in place, so the caller rebuilds the state from the store ([`resolve`] gives
//!   the rows that count). `repeated_steps` and `resume_lr_change` read every attempt,
//!   the rolled-back rows too, and need each step's learning rate; that history is held
//!   only while the store is replayed ([`AttemptPass`]), so every new attempt is a replay.
//! - *An incident dated before the row that makes it.* One parameter's gradient spike is
//!   known only once it has lasted `SPIKE_SUSTAIN` recorded steps and is dated at the
//!   first of them. Python then looks back for what was logged on that step; here it is
//!   read when the excursion begins and kept with it. An incident, once made, is never
//!   replaced or dated again, so a prefix and the whole run agree on it.
//!
//! Three things differ from Python by design:
//!
//! - A value that does not fit its column's type travels in `extra_json` and reads as
//!   absent here (only `optimizer_ownership` and `weight_dtypes`, which have no column,
//!   are read from there, on the rows where parameters first stall).
//! - The server never stores a step twice within one attempt, so that case of
//!   `repeated_steps` cannot arise from its own store and is not looked for.
//! - NaN and Inf are `null` in the incidents, which are strict JSON.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, Float64Array, Int32Array, Int64Array, ListArray, MapArray, StringArray};
use arrow::record_batch::RecordBatch;
use serde::de::{Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::storage::{Error, Storage};

// The same values as the constants of the same names in evidence.py.
pub const OVERLOAD_THRESHOLD: f64 = 0.35;
pub const STARVE_WINDOW: usize = 50;
pub const STARVE_RATIO: f64 = 0.1;
pub const STARVE_MIN_EXPECTED: f64 = 100.0;
pub const DIVERSITY_DROP: f64 = 0.6;
pub const OVERLOAD_JUMP: f64 = 0.25;
pub const OVERLOAD_MIN_HISTORY: usize = 10;
pub const OVERLOAD_WINDOW: usize = 50;
pub const SPIKE_FACTOR: f64 = 5.0;
pub const SPIKE_WINDOW: usize = 50;
pub const SPIKE_MIN_HISTORY: usize = 10;
/// Recorded steps in a row a parameter must stay at `SPIKE_FACTOR` times its own median to be
/// reported on its own; a single such step is ordinary (see `SPIKE_SUSTAIN` in `evidence.py`).
pub const SPIKE_SUSTAIN: u32 = 3;
pub const ACT_SPIKE_FACTOR: f64 = 5.0;
pub const ACT_WINDOW: usize = 50;
pub const ACT_MIN_HISTORY: usize = 10;
pub const ACT_DRIFT_FACTOR: f64 = 3.0;
pub const ACT_RECENT: usize = 10;
pub const ACT_DEAD_RISE: f64 = 0.25;
pub const STALL_STEPS: u32 = 3;
/// Onsets listed in one `stalled_weights` incident; `onsets_omitted` counts the rest.
pub const STALL_ONSETS: usize = 20;
pub const SCALE_SKIP_RUN: u64 = 10;
pub const SCALE_FLOOR: f64 = 1.0;
pub const DRIFT_TOLERANCE: f64 = 1e-6;
pub const RESUME_LR_TOLERANCE: f64 = 1e-6;

/// The incident kinds this module can report, in the order `build_evidence` lists them.
pub const PORTED: [&str; 12] = [
    "moe_imbalance",
    "starved_experts",
    "grad_spike",
    "stalled_weights",
    "repeated_steps",
    "loss_scale_trouble",
    "resume_lr_change",
    "activation_spike",
    "activation_drift",
    "dead_activations",
    "non_finite",
    "replica_drift",
];
/// The kinds only `python -m tensorquill` reports: their sources never reach the server.
pub const NOT_PORTED: [&str; 3] = ["collective_hang", "collective_desync", "traincheck_violation"];

/// The thresholds in force, written beside the incidents so a reader knows what was applied.
pub fn thresholds() -> Value {
    json!({
        "overload_threshold": OVERLOAD_THRESHOLD,
        "starve_window": STARVE_WINDOW,
        "starve_ratio": STARVE_RATIO,
        "diversity_drop": DIVERSITY_DROP,
        "overload_jump": OVERLOAD_JUMP,
        "spike_factor": SPIKE_FACTOR,
        "spike_window": SPIKE_WINDOW,
        "spike_sustain": SPIKE_SUSTAIN,
        "activation_spike_factor": ACT_SPIKE_FACTOR,
        "activation_drift_factor": ACT_DRIFT_FACTOR,
        "activation_dead_rise": ACT_DEAD_RISE,
        "stall_steps": STALL_STEPS,
        "scale_skip_run": SCALE_SKIP_RUN,
        "scale_floor": SCALE_FLOOR,
        "drift_tolerance": DRIFT_TOLERANCE,
        "resume_lr_tolerance": RESUME_LR_TOLERANCE,
    })
}

/// Python's `_median`: the middle value, or the mean of the middle two.
fn median(values: &mut [f64]) -> f64 {
    // Selecting the middle is enough; the windows are judged on every record of every
    // router, parameter and module, so a full sort here is most of the detectors' time.
    let mid = values.len() / 2;
    let odd = values.len() % 2 == 1;
    let (below, middle, _) = values.select_nth_unstable_by(mid, f64::total_cmp);
    if odd {
        *middle
    } else {
        let low = below
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .expect("an even length is at least two");
        (low + *middle) / 2.0
    }
}

/// The lowest and highest of some values (infinities when there are none). A median lies
/// between them, which lets a check be ruled out without working the median out.
fn bounds(values: impl Iterator<Item = f64>) -> (f64, f64) {
    values.fold((f64::INFINITY, f64::NEG_INFINITY), |(lowest, highest), value| {
        (lowest.min(value), highest.max(value))
    })
}

/// States kept by name, in the order the names first appeared. A record lists its
/// parameters and modules in the same order on every step, so the position in the row
/// usually finds the state without hashing the name.
struct Named<T> {
    items: Vec<(String, T)>,
    index: HashMap<String, usize>,
}

impl<T> Default for Named<T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            index: HashMap::new(),
        }
    }
}

impl<T: Default> Named<T> {
    fn find(&self, guess: usize, name: &str) -> Option<usize> {
        if self.items.get(guess).is_some_and(|(known, _)| known == name) {
            return Some(guess);
        }
        self.index.get(name).copied()
    }

    fn slot(&mut self, guess: usize, name: &str) -> &mut T {
        let at = self.find(guess, name).unwrap_or_else(|| {
            self.index.insert(name.to_owned(), self.items.len());
            self.items.push((name.to_owned(), T::default()));
            self.items.len() - 1
        });
        &mut self.items[at].1
    }
}

/// Python's `math.fsum`: the correctly rounded sum (Shewchuk's partials, as CPython has it).
fn fsum(values: impl Iterator<Item = f64>) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for mut x in values {
        let mut kept = 0;
        for index in 0..partials.len() {
            let mut y = partials[index];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[kept] = lo;
                kept += 1;
            }
            x = hi;
        }
        partials.truncate(kept);
        partials.push(x);
    }
    let Some(mut hi) = partials.pop() else { return 0.0 };
    let mut lo = 0.0;
    while let Some(y) = partials.pop() {
        let x = hi;
        hi = x + y;
        lo = y - (hi - x);
        if lo != 0.0 {
            break;
        }
    }
    // Round half to even when the remaining partials would decide the last bit.
    if let Some(&next) = partials.last() {
        if (lo < 0.0 && next < 0.0) || (lo > 0.0 && next > 0.0) {
            let y = lo * 2.0;
            let x = hi + y;
            if y == x - hi {
                hi = x;
            }
        }
    }
    hi
}

/// Python's `_diversity_from_counts`: the effective number of experts over their number.
fn diversity_from_counts(counts: &[i64]) -> f64 {
    let total: i64 = counts.iter().sum();
    if total <= 0 || counts.is_empty() {
        return 0.0;
    }
    let entropy = -fsum(counts.iter().filter(|count| **count > 0).map(|count| {
        let share = *count as f64 / total as f64;
        share * share.ln()
    }));
    entropy.exp() / counts.len() as f64
}

fn effective_overload_threshold(experts: usize, tokens: i64) -> f64 {
    if experts <= 1 || tokens <= 0 {
        return 1.0;
    }
    let mean = 1.0 / experts as f64;
    let se = (mean * (1.0 - mean) / tokens as f64).sqrt();
    OVERLOAD_THRESHOLD.max(1.5 * mean).max(mean + 3.0 * se)
}

fn dead_experts_significant(experts: usize, tokens: i64, overloaded: bool, dead: usize, diversity: f64) -> bool {
    if experts <= 1 || tokens <= 0 || dead == 0 {
        return false;
    }
    if overloaded {
        return true;
    }
    let n = experts as f64;
    if n * (1.0 - 1.0 / n).powf(tokens as f64) >= 0.01 {
        return false;
    }
    dead as f64 / n >= 0.25 || diversity < 0.5
}

// --- reading the columns -----------------------------------------------------------------

struct MapColumn<'a, V> {
    map: &'a MapArray,
    keys: &'a StringArray,
    values: &'a V,
}

impl<'a, V: Array + 'static> MapColumn<'a, V> {
    fn new(batch: &'a RecordBatch, name: &str) -> Option<Self> {
        let map = batch.column_by_name(name)?.as_any().downcast_ref::<MapArray>()?;
        Some(Self {
            map,
            keys: map.keys().as_any().downcast_ref()?,
            values: map.values().as_any().downcast_ref()?,
        })
    }

    /// The entries of one row, as indexes into `keys` and `values`; none for a null row.
    fn row(&self, row: usize) -> std::ops::Range<usize> {
        if self.map.is_null(row) {
            return 0..0;
        }
        let offsets = self.map.value_offsets();
        offsets[row] as usize..offsets[row + 1] as usize
    }
}

/// A column of lists, with the values of every row laid end to end.
struct ListColumn<'a, V> {
    list: &'a ListArray,
    values: &'a V,
}

impl<'a, V: Array + 'static> ListColumn<'a, V> {
    fn new(batch: &'a RecordBatch, name: &str) -> Option<Self> {
        let list = batch.column_by_name(name)?.as_any().downcast_ref::<ListArray>()?;
        Some(Self {
            list,
            values: list.values().as_any().downcast_ref()?,
        })
    }

    /// The values of one row, as indexes into `values`; nothing for a null row.
    fn row(&self, row: usize) -> Option<std::ops::Range<usize>> {
        if self.list.is_null(row) {
            return None;
        }
        let offsets = self.list.value_offsets();
        Some(offsets[row] as usize..offsets[row + 1] as usize)
    }
}

fn column<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Option<&'a T> {
    batch.column_by_name(name)?.as_any().downcast_ref()
}

fn value(column: Option<&Float64Array>, row: usize) -> Option<f64> {
    column
        .filter(|column| !column.is_null(row))
        .map(|column| column.value(row))
}

fn flag(column: Option<&BooleanArray>, row: usize) -> bool {
    column.is_some_and(|column| !column.is_null(row) && column.value(row))
}

fn text(column: Option<&StringArray>, row: usize) -> Option<&str> {
    column
        .filter(|column| !column.is_null(row))
        .map(|column| column.value(row))
}

/// The columns the detectors read. A column that is missing or of another type reads as
/// absent on every row: the server stores whatever columns a sender has.
struct Columns<'a> {
    steps: &'a Int64Array,
    grad_norm: Option<&'a Float64Array>,
    clip_grad_norm: Option<&'a Float64Array>,
    step_skipped: Option<&'a BooleanArray>,
    grad_scale: Option<&'a Float64Array>,
    lr: Option<ListColumn<'a, Float64Array>>,
    weights_unchanged: Option<ListColumn<'a, StringArray>>,
    replica_spread: Option<MapColumn<'a, ListArray>>,
    replica_spread_values: Option<&'a Float64Array>,
    extra: Option<&'a StringArray>,
    grad_norms: Option<MapColumn<'a, Float64Array>>,
    scalars: Option<MapColumn<'a, Float64Array>>,
    moe_top_k: Option<MapColumn<'a, Int32Array>>,
    moe_counts: Option<MapColumn<'a, ListArray>>,
    moe_count_values: Option<&'a Int64Array>,
    moe_stats: Option<&'a StringArray>,
    layers: Option<&'a StringArray>,
}

impl<'a> Columns<'a> {
    fn new(batch: &'a RecordBatch) -> Option<Self> {
        let moe_counts: Option<MapColumn<'a, ListArray>> = MapColumn::new(batch, "moe_counts");
        let moe_count_values = moe_counts
            .as_ref()
            .and_then(|counts| counts.values.values().as_any().downcast_ref::<Int64Array>());
        let replica_spread: Option<MapColumn<'a, ListArray>> = MapColumn::new(batch, "replica_spread");
        let replica_spread_values = replica_spread
            .as_ref()
            .and_then(|spread| spread.values.values().as_any().downcast_ref::<Float64Array>());
        Some(Self {
            steps: column(batch, "step")?,
            grad_scale: column(batch, "grad_scale"),
            lr: ListColumn::new(batch, "lr"),
            weights_unchanged: ListColumn::new(batch, "weights_unchanged"),
            replica_spread: replica_spread.filter(|_| replica_spread_values.is_some()),
            replica_spread_values,
            extra: column(batch, "extra_json"),
            grad_norm: column(batch, "grad_norm"),
            clip_grad_norm: column(batch, "clip_grad_norm"),
            step_skipped: column(batch, "step_skipped"),
            grad_norms: MapColumn::new(batch, "grad_norms"),
            scalars: MapColumn::new(batch, "scalars"),
            moe_top_k: MapColumn::new(batch, "moe_top_k"),
            moe_counts: moe_counts.filter(|_| moe_count_values.is_some()),
            moe_count_values,
            moe_stats: column(batch, "moe_stats_json"),
            layers: column(batch, "layers_json"),
        })
    }
}

impl Columns<'_> {
    /// Python's `_lr_values`: the learning rate of each parameter group, or nothing when
    /// none was recorded or one of them is not a finite number.
    fn lr(&self, row: usize) -> Option<Vec<f64>> {
        let lr = self.lr.as_ref()?;
        let range = lr.row(row).filter(|range| !range.is_empty())?;
        range
            .map(|at| Some(lr.values.value(at)).filter(|value| !lr.values.is_null(at) && value.is_finite()))
            .collect()
    }
}

// --- the JSON columns --------------------------------------------------------------------

/// A JSON object's members in document order, keys borrowed from the text where they can be.
struct Members<'a, T>(Vec<(Cow<'a, str>, T)>);

struct Key<'a>(Cow<'a, str>);

impl<'de: 'a, 'a> Deserialize<'de> for Key<'a> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct KeyVisitor;
        impl<'de> Visitor<'de> for KeyVisitor {
            type Value = Key<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a string")
            }
            fn visit_borrowed_str<E>(self, text: &'de str) -> Result<Self::Value, E> {
                Ok(Key(Cow::Borrowed(text)))
            }
            fn visit_str<E>(self, text: &str) -> Result<Self::Value, E> {
                Ok(Key(Cow::Owned(text.to_owned())))
            }
        }
        deserializer.deserialize_str(KeyVisitor)
    }
}

impl<'de: 'a, 'a, T: Deserialize<'de>> Deserialize<'de> for Members<'a, T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct MembersVisitor<'a, T>(PhantomData<(&'a (), T)>);
        impl<'de: 'a, 'a, T: Deserialize<'de>> Visitor<'de> for MembersVisitor<'a, T> {
            type Value = Members<'a, T>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut members = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some(Key(key)) = map.next_key()? {
                    members.push((key, map.next_value()?));
                }
                Ok(Members(members))
            }
        }
        deserializer.deserialize_map(MembersVisitor(PhantomData))
    }
}

/// A number as Python's `_finite` reads it: a JSON number (or boolean) that is finite, and
/// nothing for any other value.
#[derive(Default)]
struct Number(Option<f64>);

impl<'de> Deserialize<'de> for Number {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct NumberVisitor;
        impl<'de> Visitor<'de> for NumberVisitor {
            type Value = Number;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any value")
            }
            fn visit_f64<E>(self, number: f64) -> Result<Number, E> {
                Ok(Number(Some(number).filter(|number| number.is_finite())))
            }
            fn visit_i64<E>(self, number: i64) -> Result<Number, E> {
                Ok(Number(Some(number as f64)))
            }
            fn visit_u64<E>(self, number: u64) -> Result<Number, E> {
                Ok(Number(Some(number as f64)))
            }
            fn visit_bool<E>(self, flag: bool) -> Result<Number, E> {
                Ok(Number(Some(f64::from(u8::from(flag)))))
            }
            fn visit_str<E>(self, _: &str) -> Result<Number, E> {
                Ok(Number(None))
            }
            fn visit_unit<E>(self) -> Result<Number, E> {
                Ok(Number(None))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Number, A::Error> {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(Number(None))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Number, A::Error> {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(Number(None))
            }
        }
        deserializer.deserialize_any(NumberVisitor)
    }
}

/// One statistic of a layer: its value as `_finite` reads it, and whether it is one of the
/// strings the capture writes for NaN and Inf (`_is_non_finite`).
struct Stat {
    number: Option<f64>,
    non_finite: bool,
}

impl Stat {
    fn of(number: f64) -> Self {
        Self {
            number: Some(number).filter(|number| number.is_finite()),
            non_finite: !number.is_finite(),
        }
    }

    const ABSENT: Self = Self {
        number: None,
        non_finite: false,
    };
}

impl<'de> Deserialize<'de> for Stat {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StatVisitor;
        impl<'de> Visitor<'de> for StatVisitor {
            type Value = Stat;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any value")
            }
            fn visit_f64<E>(self, number: f64) -> Result<Stat, E> {
                Ok(Stat::of(number))
            }
            fn visit_i64<E>(self, number: i64) -> Result<Stat, E> {
                Ok(Stat::of(number as f64))
            }
            fn visit_u64<E>(self, number: u64) -> Result<Stat, E> {
                Ok(Stat::of(number as f64))
            }
            fn visit_bool<E>(self, flag: bool) -> Result<Stat, E> {
                Ok(Stat::of(f64::from(u8::from(flag))))
            }
            fn visit_str<E>(self, text: &str) -> Result<Stat, E> {
                Ok(Stat {
                    number: None,
                    non_finite: matches!(text, "nan" | "inf" | "-inf"),
                })
            }
            fn visit_unit<E>(self) -> Result<Stat, E> {
                Ok(Stat::ABSENT)
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Stat, A::Error> {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(Stat::ABSENT)
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Stat, A::Error> {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(Stat::ABSENT)
            }
        }
        deserializer.deserialize_any(StatVisitor)
    }
}

/// The statistics of one layer that the detectors read, and whether any statistic at all
/// is NaN or Inf.
#[derive(Default)]
struct LayerStats {
    std: Option<f64>,
    abs_max: Option<f64>,
    zero_frac: Option<f64>,
    non_finite: bool,
}

impl<'de> Deserialize<'de> for LayerStats {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct LayerVisitor;
        impl<'de> Visitor<'de> for LayerVisitor {
            type Value = LayerStats;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<LayerStats, A::Error> {
                let mut stats = LayerStats::default();
                while let Some(Key(key)) = map.next_key()? {
                    let stat: Stat = map.next_value()?;
                    match key.as_ref() {
                        "std" => stats.std = stat.number,
                        "abs_max" => stats.abs_max = stat.number,
                        "zero_frac" => stats.zero_frac = stat.number,
                        _ => {}
                    }
                    stats.non_finite |= stat.non_finite;
                }
                Ok(stats)
            }
        }
        deserializer.deserialize_map(LayerVisitor)
    }
}

#[derive(Deserialize)]
struct RouterStats {
    #[serde(default)]
    diversity: Number,
}

// --- MoE load: moe_imbalance and starved_experts -----------------------------------------

#[derive(Default)]
struct Router {
    name: String,
    top_k: i64,
    /// The baseline for a top-k router: its last `OVERLOAD_WINDOW` samples that were not
    /// flagged, as (counts, assignments, diversity), with the counts summed per expert.
    window: VecDeque<(Arc<[i64]>, i64, f64)>,
    window_sums: Vec<i64>,
    window_total: i64,
    /// The trailing `STARVE_WINDOW` samples, for the starved check.
    recent: VecDeque<(Arc<[i64]>, i64)>,
    recent_sums: Vec<i64>,
    recent_tokens: i64,
    /// Where this router was last found in `moe_stats_json`.
    stats_at: usize,
}

impl Router {
    /// Python's `_jump_overloaded`: in a router whose diversity just dropped, the experts
    /// whose fraction of tokens rose by `OVERLOAD_JUMP` over their average in the baseline.
    fn jump_overloaded(&self, counts: &[i64], tokens: i64, diversity: f64, scratch: &mut Vec<f64>) -> Vec<usize> {
        if self.window.len() < OVERLOAD_MIN_HISTORY {
            return Vec::new();
        }
        // The median is at most the largest value, so most samples need no median at all.
        let (_, highest) = bounds(self.window.iter().map(|(_, _, diversity)| *diversity));
        if diversity >= DIVERSITY_DROP * highest {
            return Vec::new();
        }
        scratch.clear();
        scratch.extend(self.window.iter().map(|(_, _, diversity)| *diversity));
        if diversity >= DIVERSITY_DROP * median(scratch) {
            return Vec::new();
        }
        let k = self.top_k as f64;
        let now = tokens as f64 / k;
        let before = self.window_total as f64 / k;
        if now <= 0.0 || before <= 0.0 {
            return Vec::new();
        }
        let mut overloaded = Vec::new();
        for (index, count) in counts.iter().enumerate() {
            let base = self.window_sums.get(index).copied().unwrap_or(0) as f64 / before;
            if *count as f64 / now - base >= OVERLOAD_JUMP {
                overloaded.push(index + 1);
            }
        }
        overloaded
    }

    fn keep_in_baseline(&mut self, counts: &Arc<[i64]>, tokens: i64, diversity: f64) {
        if self.window_sums.len() < counts.len() {
            self.window_sums.resize(counts.len(), 0);
        }
        for (sum, count) in self.window_sums.iter_mut().zip(counts.iter()) {
            *sum += count;
        }
        self.window_total += tokens;
        self.window.push_back((Arc::clone(counts), tokens, diversity));
        if self.window.len() > OVERLOAD_WINDOW {
            let (old, old_tokens, _) = self.window.pop_front().expect("the window is not empty");
            for (sum, count) in self.window_sums.iter_mut().zip(old.iter()) {
                *sum -= count;
            }
            self.window_total -= old_tokens;
        }
    }

    /// Python's `_mark_starved` for one sample: the experts whose share of the trailing
    /// window is below `STARVE_RATIO` of an even share, with that share.
    fn starved(&mut self, counts: &Arc<[i64]>, tokens: i64, dead: &[usize]) -> Vec<(usize, f64)> {
        if self.recent_sums.len() != counts.len() {
            self.recent.clear();
            self.recent_sums = vec![0; counts.len()];
            self.recent_tokens = 0;
        }
        for (sum, count) in self.recent_sums.iter_mut().zip(counts.iter()) {
            *sum += count;
        }
        self.recent_tokens += tokens;
        self.recent.push_back((Arc::clone(counts), tokens));
        if self.recent.len() > STARVE_WINDOW {
            let (old, old_tokens) = self.recent.pop_front().expect("the window is not empty");
            for (sum, count) in self.recent_sums.iter_mut().zip(old.iter()) {
                *sum -= count;
            }
            self.recent_tokens -= old_tokens;
        }
        let even = 1.0 / counts.len() as f64;
        let window_tokens = self.recent_tokens as f64;
        if self.recent.len() < STARVE_WINDOW || window_tokens * even < STARVE_MIN_EXPECTED {
            return Vec::new();
        }
        self.recent_sums
            .iter()
            .enumerate()
            .filter(|(index, total)| {
                **total as f64 / window_tokens < STARVE_RATIO * even && !dead.contains(&(index + 1))
            })
            .map(|(index, total)| (index + 1, *total as f64 / window_tokens))
            .collect()
    }
}

#[derive(Default)]
struct Moe {
    index: HashMap<String, usize>,
    routers: Vec<Router>,
    /// `moe_top_k` seen for a router whose counts have not arrived yet.
    top_k_ahead: HashMap<String, i64>,
    first: Option<Value>,
    /// (layer, expert) -> (first step flagged, share of the window then).
    starved: BTreeMap<(usize, usize), (i64, f64)>,
    /// What was logged on the step the first expert was flagged.
    starved_scalars: Option<Value>,
    /// How many samples (a router on a step) had an overloaded or dead expert, and the
    /// last step one did: an incident names the first, this says whether it goes on.
    flagged_samples: u64,
    last_flagged_step: Option<i64>,
    present: Vec<Option<usize>>,
    scratch: Vec<f64>,
}

impl Moe {
    fn row(&mut self, step: i64, columns: &Columns, row: usize) {
        if let Some(top_k) = &columns.moe_top_k {
            for entry in top_k.row(row) {
                if top_k.values.is_null(entry) {
                    continue;
                }
                let (name, k) = (top_k.keys.value(entry), i64::from(top_k.values.value(entry)));
                match self.index.get(name) {
                    Some(index) => self.routers[*index].top_k = k,
                    None => {
                        self.top_k_ahead.insert(name.to_owned(), k);
                    }
                }
            }
        }
        let (Some(counts), Some(count_values)) = (&columns.moe_counts, columns.moe_count_values) else {
            return;
        };
        self.present.clear();
        self.present.resize(self.routers.len(), None);
        let entries = counts.row(row);
        let first_entry = entries.start;
        for entry in entries {
            let name = counts.keys.value(entry);
            let guess = entry - first_entry;
            let known = if self.routers.get(guess).is_some_and(|router| router.name == name) {
                Some(guess)
            } else {
                self.index.get(name).copied()
            };
            let index = match known {
                Some(index) => index,
                None => {
                    // Routers are layers 1, 2, ... in the order they first appear.
                    self.index.insert(name.to_owned(), self.routers.len());
                    self.routers.push(Router {
                        name: name.to_owned(),
                        top_k: self.top_k_ahead.remove(name).unwrap_or(1),
                        ..Router::default()
                    });
                    self.present.push(None);
                    self.routers.len() - 1
                }
            };
            if !counts.values.is_null(entry) {
                self.present[index] = Some(entry);
            }
        }
        if self.present.iter().all(Option::is_none) {
            return;
        }
        let stats: Option<Members<RouterStats>> =
            text(columns.moe_stats, row).and_then(|text| serde_json::from_str(text).ok());
        let offsets = counts.values.value_offsets();
        let mut router_counts: Vec<i64> = Vec::new();
        for index in 0..self.routers.len() {
            let Some(entry) = self.present[index] else { continue };
            router_counts.clear();
            router_counts.extend((offsets[entry] as usize..offsets[entry + 1] as usize).map(|at| {
                if count_values.is_null(at) {
                    0
                } else {
                    count_values.value(at)
                }
            }));
            let tokens: i64 = router_counts.iter().sum();
            if tokens <= 0 {
                continue; // a missing sample: nothing is judged and no window moves
            }
            let router = &mut self.routers[index];
            let experts = router_counts.len();
            let recorded = stats.as_ref().and_then(|stats| {
                // The statistics list the routers in the same order on every step.
                if stats
                    .0
                    .get(router.stats_at)
                    .is_none_or(|(name, _)| name != &router.name)
                {
                    router.stats_at = stats.0.iter().position(|(name, _)| name == &router.name)?;
                }
                stats.0[router.stats_at].1.diversity.0
            });
            let diversity = recorded.unwrap_or_else(|| diversity_from_counts(&router_counts));
            let share = |count: &i64| *count as f64 / tokens as f64;
            let mut overloaded: Vec<usize> = if router.top_k >= 3 {
                router.jump_overloaded(&router_counts, tokens, diversity, &mut self.scratch)
            } else {
                let threshold = effective_overload_threshold(experts, tokens);
                (1..=experts)
                    .filter(|expert| share(&router_counts[expert - 1]) > threshold)
                    .collect()
            };
            let zero: Vec<usize> = (1..=experts).filter(|expert| router_counts[expert - 1] == 0).collect();
            let dead = if dead_experts_significant(experts, tokens, !overloaded.is_empty(), zero.len(), diversity) {
                zero
            } else {
                Vec::new()
            };
            if router.top_k >= 3 && router.window.len() < OVERLOAD_MIN_HISTORY && !dead.is_empty() {
                // Before a top-k router has a baseline, a dead-expert collapse is attributed by share.
                let mean = 1.0 / experts as f64;
                let se = (mean * (1.0 - mean) / tokens as f64).sqrt();
                let threshold = (2.0 * mean).max(mean + 3.0 * se);
                overloaded = (1..=experts)
                    .filter(|expert| share(&router_counts[expert - 1]) > threshold)
                    .collect();
            }
            let flagged = !(overloaded.is_empty() && dead.is_empty());
            let shared: Arc<[i64]> = Arc::from(&router_counts[..]);
            if router.top_k >= 3 && !flagged {
                // A flagged sample stays out of the baseline, so a collapse that continues
                // is compared with the router as it was before.
                router.keep_in_baseline(&shared, tokens, diversity);
            }
            if flagged {
                self.flagged_samples += 1;
                self.last_flagged_step = Some(step);
            }
            if flagged && self.first.is_none() {
                self.first = Some(json!({
                    "kind": "moe_imbalance",
                    "step": step,
                    "layer": index + 1,
                    "router": router.name,
                    "diversity": diversity,
                    "overloaded": overloaded,
                    "dead": dead,
                }));
            }
            for (expert, window_share) in router.starved(&shared, tokens, &dead) {
                self.starved.entry((index + 1, expert)).or_insert((step, window_share));
            }
        }
    }

    fn incidents(&self, out: &mut Vec<Value>) {
        out.extend(self.first.clone());
        if self.starved.is_empty() {
            return;
        }
        let mut experts: Vec<_> = self.starved.iter().collect();
        experts.sort_by(|a, b| a.1 .1.total_cmp(&b.1 .1).then(a.0.cmp(b.0)));
        let mut incident = json!({
            "kind": "starved_experts",
            "step": experts.iter().map(|(_, (step, _))| *step).min(),
            "experts": experts.iter().map(|((layer, expert), (step, share))| json!({
                "layer": layer,
                "router": self.routers[layer - 1].name,
                "expert": expert,
                "step": step,
                "window_share": share,
            })).collect::<Vec<_>>(),
        });
        if let Some(scalars) = &self.starved_scalars {
            incident["scalars"] = scalars.clone();
        }
        out.push(incident);
    }
}

// --- grad_spike ----------------------------------------------------------------------------

/// Where a parameter's excursion began: what the incident reports if it lasts.
struct Start {
    step: i64,
    total: f64,
    total_baseline: Option<f64>,
    name: String,
    norm: f64,
    usual: f64,
    /// What was logged on that step. The incident is dated here and made later, when the
    /// window the medians come from has moved on, so it is read when the excursion begins.
    scalars: Option<Value>,
}

impl Start {
    fn parameter(&self) -> Value {
        json!({
            "name": self.name,
            "grad_norm": self.norm,
            "baseline": self.usual,
            "ratio": self.norm / self.usual,
            "rising_since": self.step,
        })
    }
}

/// One parameter's gradient norms: Python's `per_param` list and its `_Baseline`.
struct Parameter {
    /// The last `SPIKE_WINDOW` norms with their steps, excursions included.
    past: VecDeque<(i64, f64)>,
    /// The last `SPIKE_WINDOW` norms that were not part of an excursion, and the lowest of
    /// them: a norm below `SPIKE_FACTOR` times the lowest cannot reach that many times the
    /// median, so most steps need no median.
    usual: VecDeque<f64>,
    low: f64,
    /// Recorded steps in a row at or above `SPIKE_FACTOR` times the median, and the first.
    run: u32,
    start: Option<Start>,
}

impl Default for Parameter {
    fn default() -> Self {
        Self {
            past: VecDeque::new(),
            usual: VecDeque::new(),
            low: f64::INFINITY,
            run: 0,
            start: None,
        }
    }
}

impl Parameter {
    fn keep(&mut self, norm: f64) {
        self.usual.push_back(norm);
        if norm < self.low {
            self.low = norm;
        }
        if self.usual.len() > SPIKE_WINDOW {
            let gone = self.usual.pop_front().expect("longer than the window");
            if gone <= self.low {
                self.low = self.usual.iter().copied().fold(f64::INFINITY, f64::min);
            }
        }
    }
}

#[derive(Default)]
struct Gradients {
    /// Whether any record so far had `clip_grad_norm`; the total is read from it from then on.
    clips: bool,
    seen: usize,
    totals: VecDeque<f64>,
    parameters: Named<Parameter>,
    first: Option<Value>,
    spikes: u64,
    spike_steps: Vec<i64>,
    scratch: Vec<f64>,
}

impl Gradients {
    fn row(&mut self, step: i64, columns: &Columns, row: usize, scalars: &mut Scalars) {
        if !self.clips && value(columns.clip_grad_norm, row).is_some() {
            // Python reads the pre-clipping norm for the whole run once any record has it,
            // and the records before clipping began have none: start again from here.
            *self = Self {
                clips: true,
                ..Self::default()
            };
        }
        if columns
            .step_skipped
            .is_some_and(|skipped| !skipped.is_null(row) && skipped.value(row))
        {
            return; // a loss-scaler overflow step: its gradients are Inf or NaN by design
        }
        let total = value(
            if self.clips {
                columns.clip_grad_norm
            } else {
                columns.grad_norm
            },
            row,
        );
        let Some(total) = total.filter(|total| total.is_finite()) else {
            return;
        };
        let mut total_baseline = None;
        if self.seen >= SPIKE_MIN_HISTORY {
            self.scratch.clear();
            self.scratch.extend(self.totals.iter());
            let baseline = median(&mut self.scratch);
            total_baseline = Some(baseline);
            if baseline > 0.0 && total >= SPIKE_FACTOR * baseline {
                self.spikes += 1;
                if self.spike_steps.len() < 20 {
                    self.spike_steps.push(step);
                }
                if self.first.is_none() {
                    self.first = Some(json!({
                        "kind": "grad_spike",
                        "step": step,
                        "grad_norm": total,
                        "baseline": baseline,
                        "ratio": total / baseline,
                        "parameters": self.spiking_parameters(step, columns, row),
                    }));
                }
            }
        }
        self.seen += 1;
        self.totals.push_back(total);
        if self.totals.len() > SPIKE_WINDOW {
            self.totals.pop_front();
        }
        let Some(norms) = &columns.grad_norms else { return };
        let entries = norms.row(row);
        let first_entry = entries.start;
        // The parameters whose own norm has now stayed high for `SPIKE_SUSTAIN` recorded steps.
        let mut lasted: Vec<Start> = Vec::new();
        // This row's logged values, read once if an excursion begins here and no incident exists.
        let mut logged: Option<Option<Value>> = None;
        for entry in entries {
            if norms.values.is_null(entry) || !norms.values.value(entry).is_finite() {
                continue;
            }
            let (name, norm) = (norms.keys.value(entry), norms.values.value(entry));
            let parameter = self.parameters.slot(entry - first_entry, name);
            parameter.past.push_back((step, norm));
            if parameter.past.len() > SPIKE_WINDOW {
                parameter.past.pop_front();
            }
            // The parameter on its own: at SPIKE_FACTOR times its own median, and staying
            // there. Norms of an excursion stay out of the median it is measured against.
            if parameter.usual.len() >= SPIKE_MIN_HISTORY && norm >= SPIKE_FACTOR * parameter.low {
                self.scratch.clear();
                self.scratch.extend(parameter.usual.iter());
                let usual = median(&mut self.scratch);
                if usual > 0.0 && norm >= SPIKE_FACTOR * usual {
                    if parameter.run == 0 {
                        let scalars = if self.first.is_some() {
                            None // the incident is made already: no later excursion is reported
                        } else {
                            logged.get_or_insert_with(|| scalars.context(columns, row)).clone()
                        };
                        parameter.start = Some(Start {
                            step,
                            total,
                            total_baseline,
                            name: name.to_owned(),
                            norm,
                            usual,
                            scalars,
                        });
                    }
                    parameter.run += 1;
                    if parameter.run == SPIKE_SUSTAIN {
                        lasted.extend(parameter.start.take());
                    }
                    continue;
                }
            }
            parameter.run = 0;
            parameter.keep(norm);
        }
        if lasted.is_empty() || self.first.is_some() {
            return; // an incident, once reported, is not replaced or dated again
        }
        // Usually every excursion that lasts to this record began on the same step; if not,
        // the incident is the earliest, with the parameters that began there.
        let began = lasted.iter().map(|start| start.step).min().expect("not empty");
        lasted.retain(|start| start.step == began);
        lasted.sort_by(|a, b| {
            (b.norm / b.usual)
                .total_cmp(&(a.norm / a.usual))
                .then_with(|| a.name.cmp(&b.name))
        });
        let start = lasted.iter().find(|start| start.step == began).expect("not empty");
        let ratio = start
            .total_baseline
            .filter(|baseline| *baseline != 0.0)
            .map(|baseline| start.total / baseline);
        let mut first = json!({
            "kind": "grad_spike",
            "step": began,
            "grad_norm": start.total,
            "baseline": start.total_baseline,
            "ratio": ratio,
            "parameters": lasted.iter().take(10).map(Start::parameter).collect::<Vec<_>>(),
            "trigger": "parameter",
            "sustained": SPIKE_SUSTAIN,
        });
        if let Some(scalars) = &start.scalars {
            first["scalars"] = scalars.clone();
        }
        self.first = Some(first);
    }

    /// Python's `_spiking_parameters`: the parameters above their own median, ordered by
    /// where each started rising and then by ratio; at most ten.
    fn spiking_parameters(&mut self, step: i64, columns: &Columns, row: usize) -> Vec<Value> {
        let Some(norms) = &columns.grad_norms else {
            return Vec::new();
        };
        let mut found: Vec<(i64, f64, Value)> = Vec::new();
        let entries = norms.row(row);
        let first_entry = entries.start;
        for entry in entries {
            if norms.values.is_null(entry) {
                continue;
            }
            let (name, norm) = (norms.keys.value(entry), norms.values.value(entry));
            let Some(past) = self
                .parameters
                .find(entry - first_entry, name)
                .map(|at| &self.parameters.items[at].1.past)
                .filter(|past| past.len() >= 2)
            else {
                continue;
            };
            if !norm.is_finite() {
                continue;
            }
            self.scratch.clear();
            self.scratch.extend(past.iter().map(|(_, norm)| *norm));
            let baseline = median(&mut self.scratch);
            if baseline <= 0.0 || norm < SPIKE_FACTOR * baseline {
                continue;
            }
            let mut rising = step;
            for (past_step, past_norm) in past.iter().rev() {
                if *past_norm >= 0.5 * SPIKE_FACTOR * baseline {
                    rising = *past_step;
                } else {
                    break;
                }
            }
            let ratio = norm / baseline;
            found.push((
                rising,
                ratio,
                json!({"name": name, "grad_norm": norm, "baseline": baseline, "ratio": ratio, "rising_since": rising}),
            ));
        }
        found.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1))); // stable, as Python's sort
        found.into_iter().take(10).map(|(_, _, parameter)| parameter).collect()
    }

    fn incidents(&self, out: &mut Vec<Value>) {
        if let Some(mut first) = self.first.clone() {
            first["spike_count"] = json!(self.spikes);
            first["spike_steps"] = json!(self.spike_steps);
            out.push(first);
        }
    }
}

// --- activation_spike, activation_drift, dead_activations --------------------------------

#[derive(Default)]
struct Module {
    seen: usize,
    /// The last `ACT_WINDOW` records, as (std, abs_max, zero_frac).
    recent: VecDeque<(f64, f64, f64)>,
    /// The medians of the first `ACT_WINDOW` records: std, abs_max / std, zero_frac.
    baseline: Option<(f64, f64, f64)>,
}

/// Python's `outlier`: the median of abs_max / std over the rows whose std is positive.
fn outlier<'a>(rows: impl Iterator<Item = &'a (f64, f64, f64)>, scratch: &mut Vec<f64>) -> f64 {
    scratch.clear();
    scratch.extend(
        rows.filter(|(std, _, _)| *std > 0.0)
            .map(|(std, abs_max, _)| abs_max / std),
    );
    if scratch.is_empty() {
        return 0.0;
    }
    median(scratch)
}

#[derive(Default)]
struct Activations {
    modules: Named<Module>,
    /// The first incident of each kind, in the order the kinds first happened, with the
    /// number of steps each has happened on.
    found: Vec<(&'static str, Value, u64)>,
    scratch: Vec<f64>,
}

impl Activations {
    fn row(&mut self, step: i64, layers: &[(Cow<str>, LayerStats)]) {
        let (mut spikes, mut drifts, mut dead): (Vec<(f64, Value)>, Vec<Value>, Vec<Value>) = Default::default();
        let scratch = &mut self.scratch;
        for (position, (name, stats)) in layers.iter().enumerate() {
            let (Some(std), Some(abs_max), Some(zero_frac)) = (stats.std, stats.abs_max, stats.zero_frac) else {
                continue;
            };
            let module = self.modules.slot(position, name);
            if module.seen >= ACT_MIN_HISTORY {
                for (stat, now, pick) in [("abs_max", abs_max, 1), ("std", std, 0)] {
                    // The median is at least the lowest value: below the factor times
                    // that, nothing can be a spike and the median is not needed.
                    let (lowest, _) = bounds(module.recent.iter().map(|past| if pick == 1 { past.1 } else { past.0 }));
                    if now < ACT_SPIKE_FACTOR * lowest {
                        continue;
                    }
                    scratch.clear();
                    scratch.extend(module.recent.iter().map(|past| if pick == 1 { past.1 } else { past.0 }));
                    let base = median(scratch);
                    if base > 0.0 && now >= ACT_SPIKE_FACTOR * base {
                        let ratio = now / base;
                        spikes.push((
                            ratio,
                            json!({"module": name, "stat": stat, "value": now, "baseline": base, "ratio": ratio}),
                        ));
                        break;
                    }
                }
            }
            module.seen += 1;
            module.recent.push_back((std, abs_max, zero_frac));
            if module.recent.len() > ACT_WINDOW {
                module.recent.pop_front();
            }
            if module.seen == ACT_WINDOW {
                // The first ACT_WINDOW records never change again: summarize them once.
                let column = |pick: fn(&(f64, f64, f64)) -> f64, scratch: &mut Vec<f64>| {
                    scratch.clear();
                    scratch.extend(module.recent.iter().map(pick));
                    median(scratch)
                };
                let base_std = column(|past| past.0, scratch);
                let base_zero = column(|past| past.2, scratch);
                module.baseline = Some((base_std, outlier(module.recent.iter(), scratch), base_zero));
            }
            if module.seen < ACT_WINDOW + ACT_RECENT {
                continue;
            }
            let Some((base_std, base_outlier, base_zero)) = module.baseline else {
                continue;
            };
            let last = || module.recent.iter().skip(module.recent.len() - ACT_RECENT);
            // A recent median lies between the lowest and highest recent values. When
            // neither is ACT_DRIFT_FACTOR away from the baseline the median is not either,
            // and it is not worked out.
            for stat in ["std", "abs_max/std"] {
                let ratios = || last().filter(|past| past.0 > 0.0).map(|past| past.1 / past.0);
                let (base, (lowest, highest)) = if stat == "std" {
                    (base_std, bounds(last().map(|past| past.0)))
                } else {
                    (base_outlier, bounds(ratios()))
                };
                if base <= 0.0
                    || (lowest > 0.0 && highest / base < ACT_DRIFT_FACTOR && base / lowest < ACT_DRIFT_FACTOR)
                {
                    continue;
                }
                let now = if stat == "std" {
                    scratch.clear();
                    scratch.extend(last().map(|past| past.0));
                    median(scratch)
                } else {
                    outlier(last(), scratch)
                };
                if now > 0.0 && (now / base).max(base / now) >= ACT_DRIFT_FACTOR {
                    drifts.push(
                        json!({"module": name, "stat": stat, "baseline": base, "recent": now, "ratio": now / base}),
                    );
                    break;
                }
            }
            let (_, highest_zero) = bounds(last().map(|past| past.2));
            if highest_zero - base_zero >= ACT_DEAD_RISE {
                scratch.clear();
                scratch.extend(last().map(|past| past.2));
                let now_zero = median(scratch);
                if now_zero - base_zero >= ACT_DEAD_RISE {
                    dead.push(json!({"module": name, "baseline": base_zero, "recent": now_zero}));
                }
            }
        }
        // Largest ratio first, except that the first module in the model leads: a change in one
        // block's output shows in every later block, and the largest ratio is in the last of them.
        // The rest stay by ratio, so the largest is second and a list cut to 20 keeps both.
        let origin = spikes.first().map(|(_, hit)| hit["module"].clone());
        spikes.sort_by(|a, b| b.0.total_cmp(&a.0)); // stable, largest ratio first
        if let Some(origin) = origin {
            if let Some(at) = spikes.iter().position(|(_, hit)| hit["module"] == origin) {
                let hit = spikes.remove(at);
                spikes.insert(0, hit);
            }
        }
        self.report(
            "activation_spike",
            step,
            spikes.into_iter().map(|(_, hit)| hit).collect(),
        );
        self.report("activation_drift", step, drifts);
        self.report("dead_activations", step, dead);
    }

    fn report(&mut self, kind: &'static str, step: i64, hits: Vec<Value>) {
        if hits.is_empty() {
            return;
        }
        match self.found.iter_mut().find(|(found, _, _)| *found == kind) {
            Some((_, _, steps)) => *steps += 1,
            None => {
                let affected = hits.len();
                let modules: Vec<Value> = hits.into_iter().take(20).collect();
                self.found.push((
                    kind,
                    json!({"kind": kind, "step": step, "modules": modules, "affected": affected}),
                    1,
                ));
            }
        }
    }

    fn incidents(&self, out: &mut Vec<Value>) {
        for (_, incident, steps) in &self.found {
            let mut incident = incident.clone();
            incident["affected_steps"] = json!(steps);
            out.push(incident);
        }
    }
}

// --- stalled_weights -----------------------------------------------------------------------

/// Python's `_optimizer_evidence`: whether an optimizer that stepped holds the stalled
/// parameters, from the `optimizer_ownership` the capture wrote at this weight check.
fn optimizer_evidence(extra: Option<&Value>, stalled: &[&str]) -> Value {
    let ownership = extra
        .and_then(|extra| extra.get("optimizer_ownership"))
        .filter(|ownership| ownership.as_object().is_some_and(|ownership| !ownership.is_empty()));
    let Some(ownership) = ownership else {
        return json!({"state": "unknown", "reason": "not recorded by this capture"});
    };
    if ownership["state"] != "known" {
        let reason = ownership.get("reason").cloned().unwrap_or(json!("not established"));
        return json!({"state": "unknown", "reason": reason});
    }
    let unheld: Vec<&str> = ownership["unheld"]
        .as_array()
        .map(|unheld| unheld.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let not_held: Vec<&str> = stalled.iter().copied().filter(|name| unheld.contains(name)).collect();
    json!({
        "state": "known",
        "not_held": not_held.iter().take(20).collect::<Vec<_>>(),
        "not_held_count": not_held.len(),
        "held_count": stalled.len() - not_held.len(),
    })
}

/// One parameter's weight checks.
#[derive(Default)]
struct Weight {
    /// How many weight checks in a row it was unchanged with a gradient.
    streak: u32,
    /// Seen changing at some weight check.
    changed: bool,
    /// Already listed under an onset: a parameter is reported once.
    reported: bool,
}

#[derive(Default)]
struct Stalled {
    weights: Named<Weight>,
    first: Option<Value>,
    stalled_steps: u64,
    stalled_total: u64,
    onsets: Vec<Value>,
    onset_steps: u64,
}

impl Stalled {
    fn row(&mut self, step: i64, columns: &Columns, row: usize) {
        // No list: no weight check on this step. An empty list is a check that found
        // every parameter changed.
        let Some(unchanged) = &columns.weights_unchanged else {
            return;
        };
        let Some(listed) = unchanged.row(row) else { return };
        if value(columns.grad_norm, row).is_some_and(|norm| !norm.is_finite()) {
            return; // a loss scaler skips the optimizer step on purpose when a gradient is not finite
        }
        let unchanged: Vec<&str> = listed
            .filter(|at| !unchanged.values.is_null(*at))
            .map(|at| unchanged.values.value(at))
            .collect();
        let mut with_gradients = 0;
        if let Some(norms) = &columns.grad_norms {
            let entries = norms.row(row);
            let first_entry = entries.start;
            for entry in entries {
                let norm = norms.values.value(entry);
                let positive = !norms.values.is_null(entry) && norm.is_finite() && norm > 0.0;
                with_gradients += usize::from(positive);
                let name = norms.keys.value(entry);
                let same = unchanged.contains(&name);
                // A parameter this row does not list keeps its streak.
                let weight = self.weights.slot(entry - first_entry, name);
                weight.streak = if positive && same { weight.streak + 1 } else { 0 };
                // It cannot be newly stalled on the row that shows it changing, so marking
                // it here is the same as Python's marking after the row.
                weight.changed |= !same;
            }
        }
        let stalled: Vec<&str> = self
            .weights
            .items
            .iter()
            .filter(|(_, weight)| weight.streak >= STALL_STEPS)
            .map(|(name, _)| name.as_str())
            .collect();
        if stalled.is_empty() {
            return;
        }
        self.stalled_steps += 1;
        // Parsed only on a row that reports something: `extra_json` is not small.
        let mut extra: Option<Option<Value>> = None;
        let mut parsed = || text(columns.extra, row).and_then(|extra| serde_json::from_str::<Value>(extra).ok());
        let first_onset = self.first.is_none();
        if first_onset {
            let extra = extra.get_or_insert_with(&mut parsed);
            let mut first = json!({
                "kind": "stalled_weights",
                "step": step,
                "parameters": stalled.iter().take(20).collect::<Vec<_>>(),
                "stalled": stalled.len(),
                "with_gradients": with_gradients,
                "consecutive": STALL_STEPS,
                "optimizer": optimizer_evidence(extra.as_ref(), &stalled),
            });
            let dtypes = extra
                .as_ref()
                .and_then(|extra| extra.get("weight_dtypes"))
                .filter(|dtypes| dtypes.as_object().is_some_and(|dtypes| !dtypes.is_empty()));
            if let Some(dtypes) = dtypes {
                first["weight_dtypes"] = dtypes.clone();
            }
            self.first = Some(first);
        }
        // Each parameter on its own history: those that stall on this row for the first time,
        // the ones that had changed at an earlier check first.
        let new: Vec<(&str, bool)> = self
            .weights
            .items
            .iter()
            .filter(|(_, weight)| weight.streak >= STALL_STEPS && !weight.reported)
            .map(|(name, weight)| (name.as_str(), weight.changed))
            .collect();
        if new.is_empty() {
            return;
        }
        self.onset_steps += 1;
        self.stalled_total += new.len() as u64;
        if self.onsets.len() < STALL_ONSETS {
            let names: Vec<&str> = new.iter().map(|(name, _)| *name).collect();
            let trained = new.iter().filter(|(_, changed)| *changed).count();
            let listed: Vec<&str> = new
                .iter()
                .filter(|(_, changed)| *changed)
                .chain(new.iter().filter(|(_, changed)| !*changed))
                .map(|(name, _)| *name)
                .take(20)
                .collect();
            let extra = extra.get_or_insert_with(&mut parsed);
            self.onsets.push(json!({
                "step": step,
                "parameters": listed,
                "stalled": new.len(),
                "trained_before": trained,
                "optimizer": optimizer_evidence(extra.as_ref(), &names),
            }));
        }
        for (_, weight) in &mut self.weights.items {
            weight.reported |= weight.streak >= STALL_STEPS;
        }
    }

    fn incidents(&self, out: &mut Vec<Value>) {
        if let Some(mut first) = self.first.clone() {
            first["stalled_steps"] = json!(self.stalled_steps);
            first["stalled_total"] = json!(self.stalled_total);
            first["onsets"] = json!(self.onsets);
            first["onsets_omitted"] = json!(self.onset_steps - self.onsets.len() as u64);
            out.push(first);
        }
    }
}

// --- loss_scale_trouble --------------------------------------------------------------------

#[derive(Default)]
struct LossScale {
    streak: u64,
    skipped: u64,
    recorded: u64,
    first: Option<Value>,
}

impl LossScale {
    fn row(&mut self, step: i64, columns: &Columns, row: usize) {
        let Some(skips) = columns.step_skipped.filter(|skips| !skips.is_null(row)) else {
            return;
        };
        let skipped = skips.value(row);
        self.recorded += 1;
        self.skipped += u64::from(skipped);
        self.streak = if skipped { self.streak + 1 } else { 0 };
        let scale = value(columns.grad_scale, row).filter(|scale| scale.is_finite());
        if self.first.is_none() && (self.streak >= SCALE_SKIP_RUN || scale.is_some_and(|scale| scale < SCALE_FLOOR)) {
            self.first = Some(json!({
                "kind": "loss_scale_trouble",
                "step": step,
                "consecutive_skips": self.streak,
                "grad_scale": scale,
                "reason": if self.streak >= SCALE_SKIP_RUN { "skips" } else { "scale" },
            }));
        }
    }

    fn incidents(&self, out: &mut Vec<Value>) {
        if let Some(mut first) = self.first.clone() {
            first["skipped_steps"] = json!(self.skipped);
            first["recorded_steps"] = json!(self.recorded);
            out.push(first);
        }
    }
}

// --- non_finite ----------------------------------------------------------------------------

#[derive(Default)]
struct NonFinite {
    first: Option<Value>,
}

impl NonFinite {
    fn row(&mut self, step: i64, columns: &Columns, row: usize, layers: &[(Cow<str>, LayerStats)]) {
        if self.first.is_some() {
            return;
        }
        let bad_layers: Vec<&str> = layers
            .iter()
            .filter(|(_, stats)| stats.non_finite)
            .map(|(name, _)| name.as_ref())
            .collect();
        let mut bad_gradients: Vec<&str> = Vec::new();
        // On a step the loss scaler skipped, Inf and NaN gradients are what it expects.
        if !flag(columns.step_skipped, row) {
            if let Some(norms) = &columns.grad_norms {
                bad_gradients.extend(
                    norms
                        .row(row)
                        .filter(|entry| !norms.values.is_null(*entry) && !norms.values.value(*entry).is_finite())
                        .map(|entry| norms.keys.value(entry)),
                );
            }
            let bad = |column| value(column, row).is_some_and(|norm| !norm.is_finite());
            if bad(columns.grad_norm) && bad_gradients.is_empty() {
                bad_gradients.push("<total>");
            }
            if bad(columns.clip_grad_norm) {
                bad_gradients.push("<total before clipping>");
            }
        }
        if !(bad_layers.is_empty() && bad_gradients.is_empty()) {
            self.first = Some(json!({
                "kind": "non_finite",
                "step": step,
                "layers": bad_layers,
                "gradients": bad_gradients,
            }));
        }
    }
}

// --- replica_drift -------------------------------------------------------------------------

#[derive(Default)]
struct ReplicaDrift {
    first: Option<Value>,
}

impl ReplicaDrift {
    fn row(&mut self, step: i64, columns: &Columns, row: usize) {
        let (Some(spreads), Some(numbers)) = (&columns.replica_spread, columns.replica_spread_values) else {
            return;
        };
        if self.first.is_some() {
            return;
        }
        let offsets = spreads.values.value_offsets();
        let mut diverged = serde_json::Map::new();
        for entry in spreads.row(row) {
            if spreads.values.is_null(entry) {
                continue;
            }
            let values: Vec<f64> = (offsets[entry] as usize..offsets[entry + 1] as usize)
                .map(|at| {
                    if numbers.is_null(at) {
                        f64::NAN
                    } else {
                        numbers.value(at)
                    }
                })
                .collect();
            // A NaN or Inf weight on any replica makes the spread non-finite, which would
            // compare as no drift.
            let non_finite = !values.iter().all(|value| value.is_finite());
            if !non_finite && !values.iter().any(|value| *value > DRIFT_TOLERANCE) {
                continue;
            }
            let mut detail: serde_json::Map<String, Value> = ["norm_spread", "sum_spread", "projection_spread"]
                .iter()
                .zip(&values)
                .map(|(name, value)| (name.to_string(), json!(value)))
                .collect();
            if non_finite {
                detail.insert("non_finite".to_owned(), json!(true));
            }
            diverged.insert(spreads.keys.value(entry).to_owned(), Value::Object(detail));
        }
        if !diverged.is_empty() {
            self.first = Some(json!({
                "kind": "replica_drift",
                "step": step,
                "tolerance": DRIFT_TOLERANCE,
                "parameters": diverged,
            }));
        }
    }
}

// --- repeated_steps and resume_lr_change: what the attempts did ----------------------------

/// A step whose logged scalars an incident shows, and those scalars once its row is read.
#[derive(Default)]
struct Wanted {
    step: Option<i64>,
    scalars: Option<Value>,
}

impl Wanted {
    fn want(&mut self, step: i64) {
        if self.step != Some(step) {
            *self = Self {
                step: Some(step),
                scalars: None,
            };
        }
    }
}

/// What was found by reading every attempt. Kept with the rank; the history it was found
/// from is not ([`AttemptPass`]).
#[derive(Default)]
struct Attempts {
    /// The lowest twenty step numbers recorded more than once, and how many there are.
    repeated: Vec<i64>,
    repeated_count: u64,
    repeated_context: Wanted,
    /// When the newest attempt restarted the step numbering: the steps earlier attempts
    /// have beyond its last one, each with whether it already counts as repeated. Writing
    /// one of them again repeats it.
    ahead: VecDeque<(i64, bool)>,
    /// The resumes whose learning rate differs from the run's history, as (step, change).
    changes: Vec<(i64, Value)>,
    resume_context: Wanted,
    /// The resumes that could not be compared, in the shape of Python's `not_checked`.
    not_checked: Vec<Value>,
}

impl Attempts {
    fn add_repeated(&mut self, step: i64) {
        self.repeated_count += 1;
        let at = self.repeated.partition_point(|known| *known < step);
        if at < 20 {
            self.repeated.insert(at, step);
            self.repeated.truncate(20);
        }
        self.repeated_context.want(self.repeated[0]);
    }

    fn add_change(&mut self, step: i64, change: Value) {
        self.changes.push((step, change));
        self.changes.sort_by_key(|(step, _)| *step); // stable, as Python's sort
        self.resume_context.want(self.changes[0].0);
    }

    /// A step the newest attempt wrote after the store was last replayed.
    fn appended(&mut self, step: i64) {
        while let Some(&(ahead, known)) = self.ahead.front() {
            if ahead > step {
                break;
            }
            self.ahead.pop_front();
            if ahead == step && !known {
                self.add_repeated(step);
            }
        }
    }

    fn incidents(&self, repeated: bool, out: &mut Vec<Value>) {
        if repeated {
            if let Some(first) = self.repeated.first() {
                let mut incident = json!({
                    "kind": "repeated_steps",
                    "step": first,
                    "repeated": self.repeated_count,
                    "steps": self.repeated,
                });
                if let Some(scalars) = &self.repeated_context.scalars {
                    incident["scalars"] = scalars.clone();
                }
                out.push(incident);
            }
        } else if let Some((first, _)) = self.changes.first() {
            let mut incident = json!({
                "kind": "resume_lr_change",
                "step": first,
                "resumes": self.changes.len(),
                "changes": self.changes.iter().take(5).map(|(_, change)| change).collect::<Vec<_>>(),
                "note": "an observation: some restarts change the schedule on purpose",
            });
            if let Some(scalars) = &self.resume_context.scalars {
                incident["scalars"] = scalars.clone();
            }
            out.push(incident);
        }
    }
}

/// Python's `_compare_resume`: a change to report, nothing when the resumed learning rate
/// continues the history, or the reason the two cannot be compared.
fn compare_resume(
    history: &[(i64, i64, Vec<f64>)],
    new: Option<&[f64]>,
    resumed_at: i64,
    first_lr: Option<&[f64]>,
) -> Result<Option<Value>, &'static str> {
    let new = new.ok_or("no learning rate recorded in the resumed attempt")?;
    if history.is_empty() {
        return Err("no learning rate recorded in the earlier attempts");
    }
    let at = history.partition_point(|(step, _, _)| *step < resumed_at);
    let (basis, expected, source, slope_tolerance) = match history.get(at) {
        Some((step, attempt, lr)) if *step == resumed_at => ("same_step", lr.clone(), *attempt, 0.0),
        _ => {
            let recent = &history[at.saturating_sub(10)..at];
            let (last, source, latest) = recent
                .last()
                .ok_or("the earlier attempts recorded no learning rate before the resume step")?;
            let ahead = (resumed_at - last) as f64;
            let mut expected = latest.clone();
            let mut steepest = 0.0_f64;
            let (oldest, _, oldest_lr) = &recent[0];
            if recent.len() > 1 && oldest_lr.len() == latest.len() {
                // Continue the history's last values along their recent slope.
                let span = (last - oldest) as f64;
                for (expected, before) in expected.iter_mut().zip(oldest_lr) {
                    let slope = (*expected - before) / span;
                    *expected += slope * ahead;
                    steepest = steepest.max(slope.abs());
                }
            }
            ("extrapolated", expected, *source, 0.5 * steepest * ahead)
        }
    };
    let apart = |a: f64, b: f64, tolerance: f64| (a - b).abs() > tolerance;
    let differs = new.len() != expected.len()
        || new.iter().zip(&expected).any(|(new, expected)| {
            apart(
                *new,
                *expected,
                (RESUME_LR_TOLERANCE * expected.abs()).max(slope_tolerance),
            )
        });
    if !differs {
        return Ok(None);
    }
    let resets = first_lr.is_some_and(|first| {
        first.len() == new.len()
            && new
                .iter()
                .zip(first)
                .all(|(new, first)| !apart(*new, *first, RESUME_LR_TOLERANCE * first.abs()))
            && expected
                .iter()
                .zip(first)
                .any(|(expected, first)| apart(*expected, *first, RESUME_LR_TOLERANCE * first.abs()))
    });
    Ok(Some(json!({
        "previous_attempt": source,
        "basis": basis,
        "expected_lr": expected,
        "resumed_lr": new,
        "first_lr": first_lr,
        "equals_first_lr": resets,
    })))
}

/// Reads every row of every attempt of a rank, in the order they are stored, for
/// `repeated_steps` and `resume_lr_change` (Python's `detect_repeated_steps` and
/// `detect_resume_lr_changes`). It holds every step number and learning rate of the rank
/// while the store is replayed (under 100 bytes a step, by its layout, not measured) and is
/// dropped afterwards.
struct AttemptPass {
    rank: i32,
    first_attempt: Option<i64>,
    /// The first step of the first attempt.
    run_start: i64,
    current: Option<i64>,
    /// Whether the current attempt began at or before `run_start`: the numbering started
    /// over, and a step it shares with an earlier attempt is a repeat, not a resume.
    restarted: bool,
    current_steps: Vec<i64>,
    /// The steps of the attempts before the current one, in order and without repeats.
    earlier: Vec<i64>,
    repeated: std::collections::BTreeSet<i64>,
    /// The run as it stood before the current row: (step, attempt, learning rate) in step
    /// order, with each rollback applied.
    history: Vec<(i64, i64, Vec<f64>)>,
    first_lr: Option<Vec<f64>>,
}

impl AttemptPass {
    fn new(rank: i32, first_steps: &BTreeMap<i64, i64>) -> Self {
        let first = first_steps.iter().next();
        Self {
            rank,
            first_attempt: first.map(|(attempt, _)| *attempt),
            run_start: first.map_or(0, |(_, step)| *step),
            current: None,
            restarted: false,
            current_steps: Vec::new(),
            earlier: Vec::new(),
            repeated: Default::default(),
            history: Vec::new(),
            first_lr: None,
        }
    }

    fn row(&mut self, attempt: i64, step: i64, lr: Option<Vec<f64>>, found: &mut Attempts) {
        if self.current != Some(attempt) {
            // The first row of an attempt, which is its lowest step.
            self.earlier.append(&mut self.current_steps);
            self.earlier.sort_unstable();
            self.earlier.dedup();
            self.current = Some(attempt);
            let first = self.first_attempt == Some(attempt);
            self.restarted = !first && step <= self.run_start;
            if first {
                self.first_lr = lr.clone();
            } else if !self.restarted {
                self.resumed(attempt, step, lr.as_deref(), found);
            }
            // This attempt rolls back what earlier ones recorded from its first step on.
            self.history
                .truncate(self.history.partition_point(|(known, _, _)| *known < step));
        }
        if self.restarted && self.earlier.binary_search(&step).is_ok() && self.repeated.insert(step) {
            found.add_repeated(step);
        }
        self.current_steps.push(step);
        if let Some(lr) = lr {
            self.history.push((step, attempt, lr));
        }
    }

    fn resumed(&mut self, attempt: i64, step: i64, lr: Option<&[f64]>, found: &mut Attempts) {
        let skipped = |reason: &str| json!({"check": "resume_lr", "step": step, "rank": self.rank, "reason": reason});
        match compare_resume(&self.history, lr, step, self.first_lr.as_deref()) {
            Err(reason) => found.not_checked.push(skipped(reason)),
            Ok(None) => {}
            Ok(Some(change)) if change["basis"] == "extrapolated" && change["equals_first_lr"] == false => {
                let mut entry = skipped(
                    "the earlier attempts ended before the resume step, so the learning rate cannot be \
                     compared at the same step; a schedule that changes there looks the same",
                );
                entry["expected_lr"] = change["expected_lr"].clone();
                entry["resumed_lr"] = change["resumed_lr"].clone();
                found.not_checked.push(entry);
            }
            Ok(Some(mut change)) => {
                change["step"] = json!(step);
                change["rank"] = json!(self.rank);
                change["attempt"] = json!(attempt);
                found.add_change(step, change);
            }
        }
    }

    /// Leaves in `found` what a later row of the newest attempt can still change.
    fn finish(self, last_step: Option<i64>, found: &mut Attempts) {
        if let (true, Some(last)) = (self.restarted, last_step) {
            let beyond = self.earlier.partition_point(|step| *step <= last);
            found.ahead = self.earlier[beyond..]
                .iter()
                .map(|step| (*step, self.repeated.contains(step)))
                .collect();
        }
    }
}

// --- logged scalars, as context for an incident --------------------------------------------

/// Python's `scalar_context`: the values logged with `session.log` on an incident's step,
/// each with the median of its finite values over the previous `SPIKE_WINDOW` records.
#[derive(Default)]
struct Scalars {
    names: Named<()>,
    /// The scalars of the last `SPIKE_WINDOW` records, as (name, value).
    recent: VecDeque<Vec<(usize, f64)>>,
    scratch: Vec<f64>,
}

impl Scalars {
    fn name_at(&mut self, guess: usize, name: &str) -> usize {
        self.names.slot(guess, name);
        self.names.find(guess, name).expect("the name was just added")
    }

    fn context(&mut self, columns: &Columns, row: usize) -> Option<Value> {
        let scalars = columns.scalars.as_ref()?;
        let entries = scalars.row(row);
        let first_entry = entries.start;
        let mut context = serde_json::Map::new();
        for entry in entries {
            if scalars.values.is_null(entry) {
                continue;
            }
            let name = scalars.keys.value(entry);
            let at = self.name_at(entry - first_entry, name);
            self.scratch.clear();
            self.scratch.extend(
                self.recent
                    .iter()
                    .flatten()
                    .filter(|(known, value)| *known == at && value.is_finite())
                    .map(|(_, value)| *value),
            );
            let before = if self.scratch.is_empty() {
                Value::Null
            } else {
                json!(median(&mut self.scratch))
            };
            context.insert(
                name.to_owned(),
                json!({"value": scalars.values.value(entry), "median_before": before}),
            );
        }
        (!context.is_empty()).then_some(Value::Object(context))
    }

    fn row(&mut self, columns: &Columns, row: usize) {
        let mut logged = if self.recent.len() >= SPIKE_WINDOW {
            self.recent.pop_front().unwrap_or_default()
        } else {
            Vec::new()
        };
        logged.clear();
        if let Some(scalars) = &columns.scalars {
            let entries = scalars.row(row);
            let first_entry = entries.start;
            for entry in entries {
                if !scalars.values.is_null(entry) {
                    let at = self.name_at(entry - first_entry, scalars.keys.value(entry));
                    logged.push((at, scalars.values.value(entry)));
                }
            }
        }
        self.recent.push_back(logged);
    }
}

// --- one rank ------------------------------------------------------------------------------

/// The detectors' state for one rank of one run.
#[derive(Default)]
pub struct RankDetectors {
    moe: Moe,
    gradients: Gradients,
    stalled: Stalled,
    loss_scale: LossScale,
    activations: Activations,
    non_finite: NonFinite,
    replica_drift: ReplicaDrift,
    attempts: Attempts,
    scalars: Scalars,
    last_step: Option<i64>,
    rows: u64,
    /// The newest attempt read.
    attempt: i64,
}

impl RankDetectors {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads the rows of `batch` in order. The caller gives each row once, in step order,
    /// and only rows that count (see [`resolve`]). A batch without a `step` column is ignored.
    pub fn observe(&mut self, batch: &RecordBatch) {
        self.read(batch, i64::MAX, None);
    }

    /// Reads the rows of `batch`, which belong to `self.attempt`. Rows at or past `cutoff`
    /// were rolled back by a later attempt: only `pass`, which reads every attempt, sees them.
    fn read(&mut self, batch: &RecordBatch, cutoff: i64, mut pass: Option<&mut AttemptPass>) {
        let Some(columns) = Columns::new(batch) else { return };
        for row in 0..batch.num_rows() {
            let step = columns.steps.value(row);
            match pass.as_deref_mut() {
                Some(pass) => pass.row(self.attempt, step, columns.lr(row), &mut self.attempts),
                None => self.attempts.appended(step),
            }
            if step >= cutoff {
                continue;
            }
            let before = self.reported();
            let layers: Option<Members<LayerStats>> =
                text(columns.layers, row).and_then(|layers| serde_json::from_str(layers).ok());
            let layers = layers.as_ref().map_or(&[][..], |layers| &layers.0);
            self.moe.row(step, &columns, row);
            self.gradients.row(step, &columns, row, &mut self.scalars);
            self.stalled.row(step, &columns, row);
            self.loss_scale.row(step, &columns, row);
            self.activations.row(step, layers);
            self.non_finite.row(step, &columns, row, layers);
            self.replica_drift.row(step, &columns, row);
            let after = self.reported();
            let wanted = [&self.attempts.repeated_context, &self.attempts.resume_context].map(|w| w.step == Some(step));
            if after != before || wanted != [false, false] {
                // An incident was first reported on this step, or one names it: give it
                // what was logged here.
                let context = self.scalars.context(&columns, row);
                if wanted[0] {
                    self.attempts.repeated_context.scalars = context.clone();
                }
                if wanted[1] {
                    self.attempts.resume_context.scalars = context.clone();
                }
                if let Some(context) = context {
                    if !before.0[1] && after.0[1] {
                        self.moe.starved_scalars = Some(context.clone());
                    }
                    let firsts = [
                        self.moe.first.as_mut(),
                        None, // starved experts: several steps make up the one incident
                        self.gradients.first.as_mut(),
                        self.stalled.first.as_mut(),
                        self.loss_scale.first.as_mut(),
                        self.non_finite.first.as_mut(),
                        self.replica_drift.first.as_mut(),
                    ];
                    for (first, was) in firsts.into_iter().zip(before.0) {
                        if let (Some(first), false) = (first, was) {
                            // One parameter's spike is dated at an earlier step and carries
                            // what was logged there, not what is logged on this row.
                            if first.get("trigger").is_none() {
                                first["scalars"] = context.clone();
                            }
                        }
                    }
                    for (_, incident, _) in self.activations.found.iter_mut().skip(before.1) {
                        incident["scalars"] = context.clone();
                    }
                }
            }
            self.scalars.row(&columns, row);
            self.last_step = Some(step);
            self.rows += 1;
        }
    }

    /// Whether rows of `attempt` starting at `first_step` simply continue what was read:
    /// the same attempt going on past the last step. Anything else (a resume, an older
    /// attempt's rows) changes which earlier rows count or has to be compared with them,
    /// and the state has to be rebuilt with [`replay`].
    pub fn continues(&self, attempt: i64, first_step: i64) -> bool {
        attempt == self.attempt && self.last_step.is_none_or(|last| first_step > last)
    }

    /// Reads a batch of `attempt` that [`continues`](Self::continues) the rows read so far.
    pub fn append(&mut self, attempt: i64, batch: &RecordBatch) {
        self.attempt = attempt;
        self.observe(batch);
    }

    /// Which incidents exist: MoE imbalance, starved experts, a gradient spike, stalled
    /// weights, loss-scale trouble, a non-finite value and replica drift, and how many
    /// activation kinds.
    fn reported(&self) -> ([bool; 7], usize) {
        (
            [
                self.moe.first.is_some(),
                !self.moe.starved.is_empty(),
                self.gradients.first.is_some(),
                self.stalled.first.is_some(),
                self.loss_scale.first.is_some(),
                self.non_finite.first.is_some(),
                self.replica_drift.first.is_some(),
            ],
            self.activations.found.len(),
        )
    }

    /// How many MoE samples (a router on a step) had an overloaded or dead expert, and the
    /// last step one did. `moe_imbalance` names the first such sample; the per-step samples
    /// stay in the Parquet parts, so this is what says whether the imbalance is still there.
    pub fn moe_flagged(&self) -> Value {
        json!({"samples": self.moe.flagged_samples, "last_step": self.moe.last_flagged_step})
    }

    /// The newest attempt read.
    pub fn attempt(&self) -> i64 {
        self.attempt
    }

    /// The last step read.
    pub fn last_step(&self) -> Option<i64> {
        self.last_step
    }

    /// How many rows were read.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// The incidents so far, in the order `build_evidence` lists these kinds.
    pub fn incidents(&self) -> Vec<Value> {
        let mut out = Vec::new();
        self.moe.incidents(&mut out);
        self.gradients.incidents(&mut out);
        self.stalled.incidents(&mut out);
        self.attempts.incidents(true, &mut out);
        self.loss_scale.incidents(&mut out);
        self.attempts.incidents(false, &mut out);
        self.activations.incidents(&mut out);
        out.extend(self.non_finite.first.clone());
        out.extend(self.replica_drift.first.clone());
        out
    }

    /// What was not checked, in the shape of the `not_checked` list in Python's evidence:
    /// the kinds whose sources the server does not have, and the resumes whose learning
    /// rate could not be compared.
    pub fn not_checked(&self) -> Vec<Value> {
        let mut out = vec![
            json!({
                "check": "flight_recorder",
                "kinds": ["collective_hang", "collective_desync"],
                "reason": "PyTorch's Flight Recorder dumps stay on the training nodes; python -m tensorquill reads them",
            }),
            json!({
                "check": "traincheck",
                "kinds": ["traincheck_violation"],
                "reason": "TrainCheck's results are not sent to the server; python -m tensorquill reads them",
            }),
        ];
        out.extend(self.attempts.not_checked.iter().cloned());
        out
    }
}

/// Python's `resolve_attempts` for one rank: given each attempt's first step, the step at
/// which each attempt was cut off by a later one (`i64::MAX` for the newest). A row counts
/// when its step is below its attempt's cutoff, and read by attempt and then by step the
/// rows that count are in step order.
pub fn resolve(first_steps: &BTreeMap<i64, i64>) -> BTreeMap<i64, i64> {
    let mut cutoffs = BTreeMap::new();
    let mut cut = i64::MAX;
    for (attempt, first) in first_steps.iter().rev() {
        cutoffs.insert(*attempt, cut);
        cut = cut.min(*first);
    }
    cutoffs
}

/// Rebuilds a rank's detectors from everything stored for it: after a server restart, and
/// whenever rows arrive that are not a plain continuation (a new attempt, or an older
/// attempt's rows resent late). The whole rank is read, not its last window, and the rows a
/// later attempt rolled back are read too: a held baseline, the first incidents and the
/// counts all reach back to the start, and a resume is compared with what it replaced.
pub async fn replay(storage: &Storage, run_id: &str, rank: i32) -> Result<RankDetectors, Error> {
    let parts = storage.rank_parts(run_id, rank).await?;
    let mut first_steps: BTreeMap<i64, i64> = BTreeMap::new();
    for part in &parts {
        let first = match part.first_step {
            Some(first) => Some(first),
            // A part this server did not name: read its steps.
            None => storage
                .read_part(&part.location)
                .await?
                .iter()
                .filter_map(|batch| column::<Int64Array>(batch, "step"))
                .flat_map(|steps| steps.values().iter().copied())
                .min(),
        };
        if let Some(first) = first {
            let known = first_steps.entry(part.attempt).or_insert(first);
            *known = (*known).min(first);
        }
    }
    let cutoffs = resolve(&first_steps);
    let mut detectors = RankDetectors::new();
    let mut pass = AttemptPass::new(rank, &first_steps);
    for part in &parts {
        let Some(cutoff) = cutoffs.get(&part.attempt).copied() else {
            continue;
        };
        detectors.attempt = part.attempt;
        for batch in storage.read_part(&part.location).await? {
            detectors.read(&batch, cutoff, Some(&mut pass));
        }
    }
    pass.finish(detectors.last_step, &mut detectors.attempts);
    Ok(detectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_and_fsum_match_python() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]), 2.5);
        // math.fsum([0.1] * 10) == 1.0, where a plain sum gives 0.9999999999999999.
        assert_eq!(fsum([0.1; 10].into_iter()), 1.0);
        assert_eq!(fsum([1e100, 1.0, -1e100].into_iter()), 1.0);
        assert_eq!(fsum(std::iter::empty()), 0.0);
        // _diversity_from_counts([25] * 8, 8) == 1.0 and one expert of eight is 1/8.
        assert!((diversity_from_counts(&[25; 8]) - 1.0).abs() < 1e-15);
        assert_eq!(diversity_from_counts(&[100, 0, 0, 0, 0, 0, 0, 0]), 0.125);
    }

    #[test]
    fn the_first_module_leads_an_activation_spike_and_the_rest_follow_by_ratio() {
        let row = |abs_max: [f64; 4]| -> Vec<(Cow<str>, LayerStats)> {
            ["b0", "b1", "b2", "b3"]
                .into_iter()
                .zip(abs_max)
                .map(|(name, abs_max)| {
                    let stats = LayerStats {
                        std: Some(1.0),
                        abs_max: Some(abs_max),
                        zero_frac: Some(0.0),
                        non_finite: false,
                    };
                    (Cow::Borrowed(name), stats)
                })
                .collect()
        };
        let mut activations = Activations::default();
        for step in 0..ACT_MIN_HISTORY as i64 {
            activations.row(step, &row([3.0; 4]));
        }
        // A change in block 1 reaches every later block, largest last; block 0 is untouched.
        activations.row(20, &row([3.0, 18.0, 45.0, 90.0]));
        let mut out = Vec::new();
        activations.incidents(&mut out);
        let spike = out
            .iter()
            .find(|incident| incident["kind"] == "activation_spike")
            .unwrap();
        let modules: Vec<&str> = spike["modules"]
            .as_array()
            .unwrap()
            .iter()
            .map(|hit| hit["module"].as_str().unwrap())
            .collect();
        assert_eq!(modules, ["b1", "b3", "b2"]); // first in the model, then by ratio
        assert_eq!(spike["affected"], 3);
    }

    #[test]
    fn a_later_attempt_cuts_off_the_earlier_ones() {
        // Attempt 1 from step 0, attempt 2 resumed at 100, attempt 3 at 50.
        let cutoffs = resolve(&BTreeMap::from([(1, 0), (2, 100), (3, 50)]));
        assert_eq!(cutoffs, BTreeMap::from([(1, 50), (2, 50), (3, i64::MAX)]));
        assert_eq!(resolve(&BTreeMap::from([(7, 0)])), BTreeMap::from([(7, i64::MAX)]));
    }

    #[test]
    fn json_members_keep_their_order_and_odd_values_read_as_absent() {
        let Members(layers) = serde_json::from_str::<Members<LayerStats>>(
            r#"{"b":{"std":1,"abs_max":2.5,"zero_frac":null,"x":[1,{"y":2}]},"a\n":{}}"#,
        )
        .unwrap();
        assert_eq!(
            layers.iter().map(|(name, _)| name.as_ref()).collect::<Vec<_>>(),
            ["b", "a\n"]
        );
        assert_eq!(
            (layers[0].1.std, layers[0].1.abs_max, layers[0].1.zero_frac),
            (Some(1.0), Some(2.5), None)
        );
        assert_eq!((layers[1].1.std, layers[0].1.non_finite), (None, false));
        // The capture writes NaN and Inf as strings; any statistic counts, not only the three read.
        let Members(layers) = serde_json::from_str::<Members<LayerStats>>(
            r#"{"a":{"mean":"-inf","std":1},"b":{"std":"nan"},"c":{"std":"x"}}"#,
        )
        .unwrap();
        assert_eq!(
            layers
                .iter()
                .map(|(_, stats)| (stats.std, stats.non_finite))
                .collect::<Vec<_>>(),
            [(Some(1.0), true), (None, true), (None, false)]
        );
    }
}
