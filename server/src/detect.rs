//! Stateful incident detectors over step batches: a first slice of `evidence.py` in Rust.
//!
//! The Python detectors (`sdk/src/tensorquill/evidence.py`) are the reference. They rerun
//! over a whole run on every look, which the 64-rank benchmark measured at 13 times the
//! plan's 250 ms budget even with state kept between flushes. This module keeps each
//! detector's state per `(run_id, rank)` and reads Arrow batches as they are, so a flush
//! costs its own rows only.
//!
//! Ported so far, each to give what `build_evidence` gives on the same records:
//! `moe_imbalance`, `starved_experts`, `grad_spike`, `activation_spike`,
//! `activation_drift` and `dead_activations`. The other kinds are still Python's alone
//! ([`NOT_PORTED`]).
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
//!   the rows that count).

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, Float64Array, Int32Array, Int64Array, ListArray, MapArray, StringArray};
use arrow::compute::filter_record_batch;
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
pub const ACT_SPIKE_FACTOR: f64 = 5.0;
pub const ACT_WINDOW: usize = 50;
pub const ACT_MIN_HISTORY: usize = 10;
pub const ACT_DRIFT_FACTOR: f64 = 3.0;
pub const ACT_RECENT: usize = 10;
pub const ACT_DEAD_RISE: f64 = 0.25;

/// The incident kinds this module can report.
pub const PORTED: [&str; 6] = [
    "moe_imbalance",
    "starved_experts",
    "grad_spike",
    "activation_spike",
    "activation_drift",
    "dead_activations",
];
/// The kinds only `python -m tensorquill` reports.
pub const NOT_PORTED: [&str; 11] = [
    "stalled_weights",
    "repeated_steps",
    "resume_lr_change",
    "loss_scale_trouble",
    "non_finite",
    "replica_drift",
    "job_stalled",
    "rank_silent",
    "collective_hang",
    "collective_desync",
    "traincheck_violation",
];

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
        "activation_spike_factor": ACT_SPIKE_FACTOR,
        "activation_drift_factor": ACT_DRIFT_FACTOR,
        "activation_dead_rise": ACT_DEAD_RISE,
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

fn column<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Option<&'a T> {
    batch.column_by_name(name)?.as_any().downcast_ref()
}

fn value(column: Option<&Float64Array>, row: usize) -> Option<f64> {
    column
        .filter(|column| !column.is_null(row))
        .map(|column| column.value(row))
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
        Some(Self {
            steps: column(batch, "step")?,
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

#[derive(Deserialize)]
struct LayerStats {
    #[serde(default)]
    std: Number,
    #[serde(default)]
    abs_max: Number,
    #[serde(default)]
    zero_frac: Number,
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

#[derive(Default)]
struct Gradients {
    /// Whether any record so far had `clip_grad_norm`; the total is read from it from then on.
    clips: bool,
    seen: usize,
    totals: VecDeque<f64>,
    parameters: Named<VecDeque<(i64, f64)>>,
    first: Option<Value>,
    spikes: u64,
    spike_steps: Vec<i64>,
    scratch: Vec<f64>,
}

impl Gradients {
    fn row(&mut self, step: i64, columns: &Columns, row: usize) {
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
        if self.seen >= SPIKE_MIN_HISTORY {
            self.scratch.clear();
            self.scratch.extend(self.totals.iter());
            let baseline = median(&mut self.scratch);
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
        for entry in entries {
            if norms.values.is_null(entry) || !norms.values.value(entry).is_finite() {
                continue;
            }
            let past = self.parameters.slot(entry - first_entry, norms.keys.value(entry));
            past.push_back((step, norms.values.value(entry)));
            if past.len() > SPIKE_WINDOW {
                past.pop_front();
            }
        }
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
                .map(|at| &self.parameters.items[at].1)
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
    fn row(&mut self, step: i64, columns: &Columns, row: usize) {
        let Some(layers) = text(columns.layers, row) else {
            return;
        };
        let Ok(Members(layers)) = serde_json::from_str::<Members<LayerStats>>(layers) else {
            return;
        };
        let (mut spikes, mut drifts, mut dead): (Vec<(f64, Value)>, Vec<Value>, Vec<Value>) = Default::default();
        let scratch = &mut self.scratch;
        for (position, (name, stats)) in layers.iter().enumerate() {
            let (Some(std), Some(abs_max), Some(zero_frac)) = (stats.std.0, stats.abs_max.0, stats.zero_frac.0) else {
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
        spikes.sort_by(|a, b| b.0.total_cmp(&a.0)); // stable, largest ratio first
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
    activations: Activations,
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
        let Some(columns) = Columns::new(batch) else { return };
        for row in 0..batch.num_rows() {
            let step = columns.steps.value(row);
            let before = self.reported();
            self.moe.row(step, &columns, row);
            self.gradients.row(step, &columns, row);
            self.activations.row(step, &columns, row);
            let after = self.reported();
            if after != before {
                // An incident was first reported on this step: give it what was logged here.
                if let Some(context) = self.scalars.context(&columns, row) {
                    if !before.0 {
                        if let Some(first) = &mut self.moe.first {
                            first["scalars"] = context.clone();
                        }
                    }
                    if !before.1 && after.1 {
                        self.moe.starved_scalars = Some(context.clone());
                    }
                    if !before.2 {
                        if let Some(first) = &mut self.gradients.first {
                            first["scalars"] = context.clone();
                        }
                    }
                    for (_, incident, _) in self.activations.found.iter_mut().skip(before.3) {
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
    /// the newest attempt, or a newer one, going on past the last step. Anything else
    /// (a resume that goes back, an older attempt's rows) changes which earlier rows
    /// count, and the state has to be rebuilt with [`replay`].
    pub fn continues(&self, attempt: i64, first_step: i64) -> bool {
        attempt >= self.attempt && self.last_step.is_none_or(|last| first_step > last)
    }

    /// Reads a batch of `attempt` that [`continues`](Self::continues) the rows read so far.
    pub fn append(&mut self, attempt: i64, batch: &RecordBatch) {
        self.attempt = attempt;
        self.observe(batch);
    }

    /// Which incidents exist: MoE imbalance, starved experts, a gradient spike, and how
    /// many activation kinds.
    fn reported(&self) -> (bool, bool, bool, usize) {
        (
            self.moe.first.is_some(),
            !self.moe.starved.is_empty(),
            self.gradients.first.is_some(),
            self.activations.found.len(),
        )
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
        self.activations.incidents(&mut out);
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
/// whenever rows arrive that are not a plain continuation (a resume that rolled steps
/// back, or an older attempt's rows resent late). The whole rank is read, not its last
/// window: a held baseline, the first incidents and the counts all reach back to the start.
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
    for part in &parts {
        let Some(cutoff) = cutoffs.get(&part.attempt).copied() else {
            continue;
        };
        if part.first_step.is_some_and(|first| first >= cutoff) {
            continue; // every row was rolled back by a later attempt
        }
        for batch in storage.read_part(&part.location).await? {
            let batch = if cutoff == i64::MAX {
                batch
            } else {
                below(&batch, cutoff)?
            };
            detectors.observe(&batch);
        }
        detectors.attempt = part.attempt;
    }
    Ok(detectors)
}

/// The rows of `batch` whose step is below `cutoff`.
fn below(batch: &RecordBatch, cutoff: i64) -> Result<RecordBatch, Error> {
    let steps: &Int64Array = column(batch, "step").ok_or("a stored part has no `step` column")?;
    let keep: BooleanArray = steps.values().iter().map(|step| Some(*step < cutoff)).collect();
    Ok(filter_record_batch(batch, &keep)?)
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
            (layers[0].1.std.0, layers[0].1.abs_max.0, layers[0].1.zero_frac.0),
            (Some(1.0), Some(2.5), None)
        );
        assert_eq!(layers[1].1.std.0, None);
    }
}
