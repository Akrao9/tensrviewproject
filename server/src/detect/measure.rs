//! The four measurements of `DETECTOR_RULES_CONTRACT.md`, part 6: `gradient_trend`,
//! `attention_trends`, `update_trends` and `weight_jumps`, each a block of numbers in the rank's
//! report, as `head_trends` is.
//!
//! They are descriptions. No rule reads them, they change no run state, no incident, no
//! `kind_checks` entry and no `not_checked` entry, and no line is drawn on them. A block is in the
//! report only when the rows hold what it reads; otherwise its key is absent (never `null`, never
//! an empty object).
//!
//! The arithmetic and its order are those of `trend`, `gradient_trend`, `attention_trends`,
//! `update_trends` and `weight_jumps` in `sdk/tests/fixtures/rules/make_cases.py`, the page's rules
//! written plainly. Python sorts the records by step first; the rows come here in step order
//! (see [`RankDetectors::observe`](super::RankDetectors::observe)), rows of one step in the order
//! given, as Python's stable sort leaves them.
//!
//! What each keeps:
//!
//! - The two trends need every point of their series, since the median of the first third moves
//!   as the series grows: one step and one value per row with a total gradient norm, and per
//!   attention module one per probe that read its largest logit. Every path keeps them, the live
//!   server's detectors too (the author's decision of Nov 21, 2026), so the live report's blocks
//!   are `detect`'s on the same rows, with the same absence rules.
//! - `update_trends` keeps each parameter's last [`WINDOW`] finite update sizes and its lowest
//!   ratio so far, and `weight_jumps` the previous weight check's norms and each tensor's largest
//!   change so far. Both are bounded by the number of parameters.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, VecDeque};

use arrow::array::Array;
use serde_json::{json, Value};

use super::{flag, median, value, AttentionStats, Columns, SCORE_MIN};

/// Points a trend needs (`TREND_MIN` in `make_cases.py`).
pub const TREND_MIN: usize = 6;
/// Entries a measurement lists (`LISTED`).
pub const LISTED: usize = 10;
/// Earlier records of a parameter whose median its update size is set against (`WINDOW`).
pub const WINDOW: usize = 50;

/// Whether the two trends' series are kept. Always true in the product: no option, configuration
/// key or environment variable reaches it. It is set to false only in a copy built to measure what
/// the series cost (lane AD10's baseline), never in a build that is shipped.
const KEEP_TREND_SERIES: bool = true;

/// The robust local trend (`trend` in `make_cases.py`): a line through the median of the first
/// third of the values and the median of the last third, each placed at the median of its own
/// step numbers, which are the series' actual steps. None under [`TREND_MIN`] points, or when the
/// two thirds' step medians do not increase. `growth` and `per_100_steps` are null when the first
/// third's median is not above zero.
fn trend(steps: &[i64], values: &[f64], scratch: &mut Vec<f64>) -> Option<serde_json::Map<String, Value>> {
    let n = values.len();
    if n < TREND_MIN {
        return None;
    }
    let k = (n / 3).max(1);
    let mut middle = |of: &mut dyn Iterator<Item = f64>| {
        scratch.clear();
        scratch.extend(of);
        median(scratch)
    };
    let left_step = middle(&mut steps[..k].iter().map(|step| *step as f64));
    let left = middle(&mut values[..k].iter().copied());
    let right_step = middle(&mut steps[n - k..].iter().map(|step| *step as f64));
    let right = middle(&mut values[n - k..].iter().copied());
    if right_step - left_step <= 0.0 {
        return None;
    }
    let slope = (right - left) / (right_step - left_step);
    let positive = left > 0.0;
    let found = json!({
        "points": n,
        "first_step": steps[0],
        "last_step": steps[n - 1],
        "first": left,
        "last": right,
        "growth": positive.then(|| right / left),
        "per_100_steps": positive.then(|| 100.0 * slope / left),
    });
    match found {
        Value::Object(found) => Some(found),
        _ => unreachable!("a trend is an object"),
    }
}

/// Python's order of two keys that are never NaN: equal values (0.0 and -0.0 among them) are
/// equal, and the name then decides.
fn by(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

/// One series of steps and values.
#[derive(Default)]
struct Series {
    steps: Vec<i64>,
    values: Vec<f64>,
}

impl Series {
    fn push(&mut self, step: i64, value: f64) {
        self.steps.push(step);
        self.values.push(value);
    }

    fn clear(&mut self) {
        self.steps.clear();
        self.values.clear();
    }
}

/// A parameter's update sizes: its last [`WINDOW`] finite values, and its lowest ratio so far as
/// `(ratio, step, baseline, update_rms)`.
#[derive(Default)]
struct UpdateHistory {
    past: VecDeque<f64>,
    lowest: Option<Lowest>,
}

/// A parameter's lowest update so far: `(ratio, step, baseline, update_rms)`.
type Lowest = (f64, i64, f64, f64);

/// A tensor's largest change between two consecutive weight checks, as
/// `(ratio, step, previous_step, before, after)`.
type Jump = (f64, i64, i64, f64, f64);

/// A module's trend as `attention_trends` sorts it: its growth, its name, and the trend.
type Listed<'a> = (Option<f64>, &'a str, serde_json::Map<String, Value>);

/// The state of the four measurements over a rank's rows.
#[derive(Default)]
pub(super) struct Measurements {
    /// Set only by tests ([`RankDetectors::without_trend_series`](super::RankDetectors)): the two
    /// trends' series are not kept, and neither block is written.
    pub(super) no_series: bool,
    /// Whether any row so far had `clip_grad_norm`: the trend is read from it from then on.
    clips: bool,
    gradient: Series,
    /// Each attention module's largest logit, probe by probe.
    attention: HashMap<String, Series>,
    updates: HashMap<String, UpdateHistory>,
    /// Rows with a non-empty `weight_norms`, the last one's step and norms (a norm that is not
    /// finite is None), and each tensor's largest change.
    checks: u64,
    previous: Option<(i64, HashMap<String, Option<f64>>)>,
    jumps: HashMap<String, Jump>,
    scratch: Vec<f64>,
}

impl Measurements {
    /// Reads one row that counts (after the attempt's cutoff), every row of it but the probe,
    /// which [`probe`](Self::probe) is given.
    pub(super) fn row(&mut self, step: i64, columns: &Columns, row: usize) {
        self.gradient_row(step, columns, row);
        self.update_row(step, columns, row);
        self.weight_row(step, columns, row);
    }

    /// `gradient_trend`'s series: `clip_grad_norm` once any row has it, as Python reads it for every
    /// record when one has it (so the rows before clipping began have none and are left out),
    /// `grad_norm` before that; a skipped step and a value that is not finite are left out.
    fn gradient_row(&mut self, step: i64, columns: &Columns, row: usize) {
        if self.no_series || !KEEP_TREND_SERIES {
            return;
        }
        if !self.clips && value(columns.clip_grad_norm, row).is_some() {
            self.clips = true;
            self.gradient.clear();
        }
        if flag(columns.step_skipped, row) {
            return;
        }
        let norm = value(
            if self.clips {
                columns.clip_grad_norm
            } else {
                columns.grad_norm
            },
            row,
        );
        if let Some(norm) = norm.filter(|norm| norm.is_finite()) {
            self.gradient.push(step, norm);
        }
    }

    /// One probe's attention modules, read whether or not the probe says `error`, as Python's
    /// `attention_trends` reads them. Of a module named twice, the last entry counts, as
    /// `json.loads` keeps it.
    pub(super) fn probe(&mut self, step: i64, attention: &[(std::borrow::Cow<'_, str>, AttentionStats)]) {
        if self.no_series || !KEEP_TREND_SERIES {
            return;
        }
        let mut last: BTreeMap<&str, Option<f64>> = BTreeMap::new();
        for (name, stats) in attention {
            last.insert(name.as_ref(), stats.logit);
        }
        for (name, logit) in last {
            let Some(logit) = logit else { continue };
            match self.attention.get_mut(name) {
                Some(series) => series.push(step, logit),
                None => {
                    let mut series = Series::default();
                    series.push(step, logit);
                    self.attention.insert(name.to_owned(), series);
                }
            }
        }
    }

    /// `update_trends`: each finite `update_rms` against the median of the parameter's last
    /// [`WINDOW`] finite values, from [`SCORE_MIN`] of them on, on every row (skipped steps too).
    fn update_row(&mut self, step: i64, columns: &Columns, row: usize) {
        let Some(sizes) = &columns.update_rms else { return };
        for entry in sizes.row(row) {
            if sizes.values.is_null(entry) {
                continue;
            }
            let now = sizes.values.value(entry);
            if !now.is_finite() {
                continue;
            }
            let name = sizes.keys.value(entry);
            let history = match self.updates.get_mut(name) {
                Some(history) => history,
                None => self.updates.entry(name.to_owned()).or_default(),
            };
            if history.past.len() >= SCORE_MIN {
                self.scratch.clear();
                self.scratch.extend(history.past.iter().copied());
                let base = median(&mut self.scratch);
                if base > 0.0 && history.lowest.is_none_or(|(lowest, ..)| now / base < lowest) {
                    history.lowest = Some((now / base, step, base, now));
                }
            }
            history.past.push_back(now);
            if history.past.len() > WINDOW {
                history.past.pop_front();
            }
        }
    }

    /// `weight_jumps`: a row with a non-empty `weight_norms` is a weight check, and each tensor's
    /// positive finite norm is compared with its norm at the check before, either way.
    fn weight_row(&mut self, step: i64, columns: &Columns, row: usize) {
        let Some(norms) = &columns.weight_norms else { return };
        let entries = norms.row(row);
        if entries.is_empty() {
            return;
        }
        self.checks += 1;
        let mut current: HashMap<String, Option<f64>> = HashMap::with_capacity(entries.len());
        for entry in entries {
            let norm = Some(norms.values.value(entry)).filter(|norm| !norms.values.is_null(entry) && norm.is_finite());
            current.insert(norms.keys.value(entry).to_owned(), norm);
        }
        if let Some((previous_step, before)) = &self.previous {
            for (name, after) in &current {
                let (Some(a), Some(b)) = (before.get(name).copied().flatten(), *after) else {
                    continue;
                };
                if a <= 0.0 || b <= 0.0 {
                    continue;
                }
                let ratio = (b / a).max(a / b);
                match self.jumps.get_mut(name) {
                    Some(jump) if ratio > jump.0 => *jump = (ratio, step, *previous_step, a, b),
                    Some(_) => {}
                    None => {
                        self.jumps.insert(name.clone(), (ratio, step, *previous_step, a, b));
                    }
                }
            }
        }
        self.previous = Some((step, current));
    }

    /// The block `gradient_trend`, or None under [`TREND_MIN`] points.
    fn gradient_trend(&self, scratch: &mut Vec<f64>) -> Option<Value> {
        let mut found = trend(&self.gradient.steps, &self.gradient.values, scratch)?;
        let series = if self.clips { "clip_grad_norm" } else { "grad_norm" };
        found.insert("series".into(), json!(series));
        Some(Value::Object(found))
    }

    /// The block `attention_trends`, or None when no module has a trend: the modules with one,
    /// and the [`LISTED`] of largest `growth` (a null growth last), then by name.
    fn attention_trends(&self, scratch: &mut Vec<f64>) -> Option<Value> {
        let mut found: Vec<Listed> = self
            .attention
            .iter()
            .filter_map(|(module, series)| {
                let mut found = trend(&series.steps, &series.values, scratch)?;
                let growth = found["growth"].as_f64();
                found.insert("module".into(), json!(module));
                Some((growth, module.as_str(), found))
            })
            .collect();
        if found.is_empty() {
            return None;
        }
        let key = |growth: Option<f64>| growth.map_or(f64::INFINITY, |growth| -growth);
        found.sort_by(|a, b| by(key(a.0), key(b.0)).then_with(|| a.1.cmp(b.1)));
        let modules = found.len();
        let listed: Vec<Value> = found
            .into_iter()
            .take(LISTED)
            .map(|(_, _, found)| Value::Object(found))
            .collect();
        Some(json!({"statistic": "max_logit", "modules": modules, "listed": listed}))
    }

    /// The block `update_trends`, or None when no row had a finite update size: the parameters
    /// with one, those judged, and the [`LISTED`] lowest ratios, then by name.
    fn update_trends(&self) -> Option<Value> {
        if self.updates.is_empty() {
            return None;
        }
        let mut lowest: Vec<(&str, Lowest)> = self
            .updates
            .iter()
            .filter_map(|(name, history)| Some((name.as_str(), history.lowest?)))
            .collect();
        let judged = lowest.len();
        lowest.sort_by(|a, b| by(a.1 .0, b.1 .0).then_with(|| a.0.cmp(b.0)));
        let lowest: Vec<Value> = lowest
            .into_iter()
            .take(LISTED)
            .map(|(name, (ratio, step, baseline, update_rms))| {
                json!({"name": name, "ratio": ratio, "step": step, "baseline": baseline, "update_rms": update_rms})
            })
            .collect();
        Some(json!({"parameters": self.updates.len(), "judged": judged, "lowest": lowest}))
    }

    /// The block `weight_jumps`, or None with fewer than two weight checks or no tensor compared:
    /// the checks, the tensors compared, and the [`LISTED`] largest changes, then by name.
    fn weight_jumps(&self) -> Option<Value> {
        if self.checks < 2 || self.jumps.is_empty() {
            return None;
        }
        let mut largest: Vec<(&str, Jump)> = self.jumps.iter().map(|(name, jump)| (name.as_str(), *jump)).collect();
        largest.sort_by(|a, b| by(b.1 .0, a.1 .0).then_with(|| a.0.cmp(b.0)));
        let largest: Vec<Value> = largest
            .into_iter()
            .take(LISTED)
            .map(|(name, (ratio, step, previous_step, before, after))| {
                json!({"name": name, "ratio": ratio, "step": step, "previous_step": previous_step,
                       "before": before, "after": after})
            })
            .collect();
        Some(json!({"checks": self.checks, "parameters": self.jumps.len(), "largest": largest}))
    }

    /// The blocks the rows hold, by their report keys, in the page's order.
    pub(super) fn blocks(&self) -> Vec<(&'static str, Value)> {
        let mut scratch = Vec::new();
        [
            ("gradient_trend", self.gradient_trend(&mut scratch)),
            ("attention_trends", self.attention_trends(&mut scratch)),
            ("update_trends", self.update_trends()),
            ("weight_jumps", self.weight_jumps()),
        ]
        .into_iter()
        .filter_map(|(name, block)| Some((name, block?)))
        .collect()
    }
}
