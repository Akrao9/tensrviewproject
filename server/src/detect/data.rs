//! Data progress and checkpoint manifests: `data_checks.py` in Rust.
//!
//! The per-rank kinds are `data_stalled` and `epoch_order_repeated`, read from every row that
//! counts, and `resume_data_mismatch` and `resume_incomplete`, judged on the first row of each
//! resumed attempt while the store is replayed (as `resume_lr_change` is). Each rank also
//! keeps its last `DUPLICATE_WINDOW` batch hashes, which [`duplicate_batches`] compares across
//! the ranks of a run for the run's status.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use super::{text, Columns, Scalars, Wanted};

// The same values as the constants of the same names in data_checks.py.
pub const DATA_STALL_STEPS: u32 = 3;
pub const ORDER_MIN_BATCHES: usize = 3;
pub const ORDER_COMPARE: usize = 64;
pub const DUPLICATE_WINDOW: usize = 4096;
pub const SCHEDULER_SLACK: i64 = 1;
const CHANGES_LISTED: usize = 5;

/// Python's `_int`: a JSON integer, and nothing for any other value (a boolean included).
fn int(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) if !number.is_f64() => number.as_i64(),
        _ => None,
    }
}

/// Python's `_number`: a finite JSON number, not a boolean.
fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64().filter(|value| value.is_finite()),
        _ => None,
    }
}

/// One row's `data`, as `data_checks.py` reads it.
#[derive(Clone)]
pub(super) struct DataRow {
    batches: Option<i64>,
    hash: Option<String>,
    epoch: Option<i64>,
    sampler_epoch: Option<i64>,
    shuffle: Option<bool>,
    data_rank: Option<i64>,
}

impl DataRow {
    pub(super) fn read(columns: &Columns, row: usize) -> Option<Self> {
        let parsed: Value = serde_json::from_str(text(columns.data, row)?).ok()?;
        let fields = parsed.as_object()?;
        let get = |key: &str| fields.get(key).unwrap_or(&Value::Null);
        Some(Self {
            batches: int(get("batches")),
            hash: get("batch_hash")
                .as_str()
                .filter(|hash| !hash.is_empty())
                .map(str::to_owned),
            epoch: int(get("epoch")),
            sampler_epoch: int(get("sampler_epoch")),
            shuffle: get("shuffle").as_bool(),
            data_rank: int(get("data_rank")),
        })
    }

    fn drew(&self) -> bool {
        self.batches.is_some_and(|batches| batches > 0)
    }
}

/// Python's `detect_data_stalled`, a row at a time.
#[derive(Default)]
struct Stalled {
    run: u32,
    first: i64,
    reason: &'static str,
    previous: Option<String>,
    context: Option<Value>,
    found: Option<Value>,
}

impl Stalled {
    /// Returns whether this row started a stretch, whose step then wants its logged scalars.
    fn row(&mut self, step: i64, data: Option<&DataRow>) -> bool {
        if self.found.is_some() {
            return false;
        }
        let Some(data) = data else {
            self.run = 0;
            self.previous = None;
            return false;
        };
        let why = if data.batches == Some(0) {
            Some("no_batch")
        } else if data.drew() && data.hash.is_some() && data.hash == self.previous {
            Some("same_batch")
        } else {
            None
        };
        if data.drew() {
            self.previous = data.hash.clone();
        }
        let Some(why) = why else {
            self.run = 0;
            return false;
        };
        let started = self.run == 0;
        if started {
            self.first = step;
            self.reason = why;
            self.context = None;
        }
        self.run += 1;
        if self.run == DATA_STALL_STEPS {
            let mut incident = json!({
                "kind": "data_stalled",
                "step": self.first,
                "reason": self.reason,
                "through_step": step,
            });
            if self.reason == "same_batch" {
                incident["batch_hash"] = json!(self.previous);
            }
            if let Some(context) = &self.context {
                incident["scalars"] = context.clone();
            }
            self.found = Some(incident);
        }
        started
    }
}

/// The pass before the current one: its first hashes, its length, its number and its sampler's epoch.
struct Pass {
    hashes: Vec<Option<String>>,
    count: usize,
    epoch: i64,
    sampler_epoch: Option<i64>,
}

/// Python's `detect_epoch_order_repeated`, a row at a time.
#[derive(Default)]
struct EpochOrder {
    epoch: Option<i64>,
    current: Vec<Option<String>>,
    count: usize,
    first: i64,
    shuffle: Option<bool>,
    sampler_epoch: Option<i64>,
    previous: Option<Pass>,
    equal: bool,
    context: Option<Value>,
    found: Option<Value>,
}

impl EpochOrder {
    /// Returns whether this row began a pass, whose first step then wants its logged scalars.
    fn row(&mut self, step: i64, data: Option<&DataRow>) -> bool {
        let Some(data) = data.filter(|data| data.drew() && self.found.is_none()) else {
            return false;
        };
        let Some(number) = data.epoch else { return false };
        let mut began = false;
        if self.epoch != Some(number) {
            self.previous = match self.epoch {
                Some(epoch) if number == epoch + 1 => Some(Pass {
                    hashes: std::mem::take(&mut self.current),
                    count: self.count,
                    epoch,
                    sampler_epoch: self.sampler_epoch,
                }),
                _ => None,
            };
            self.epoch = Some(number);
            self.current.clear();
            self.count = 0;
            self.first = step;
            self.equal = true;
            self.shuffle = data.shuffle;
            self.sampler_epoch = data.sampler_epoch;
            self.context = None;
            began = true;
        }
        let position = self.count;
        self.count += 1;
        if position < ORDER_COMPARE {
            self.current.push(data.hash.clone());
        }
        let Some(previous) = &self.previous else {
            return began;
        };
        if !self.equal || self.shuffle == Some(false) || position >= ORDER_COMPARE {
            return began;
        }
        if data.hash.is_none() || previous.hashes.get(position) != Some(&data.hash) {
            self.equal = false;
            return began;
        }
        let needed = ORDER_MIN_BATCHES.min(previous.count);
        let distinct = self.current.iter().any(|hash| *hash != self.current[0]);
        if position + 1 >= needed && distinct {
            let mut incident = json!({
                "kind": "epoch_order_repeated",
                "step": self.first,
                "epoch": number,
                "previous_epoch": previous.epoch,
                "batches_compared": position + 1,
                "shuffle": self.shuffle,
                "sampler_epoch": self.sampler_epoch,
                "previous_sampler_epoch": previous.sampler_epoch,
                "through_step": step,
            });
            if let Some(context) = &self.context {
                incident["scalars"] = context.clone();
            }
            self.found = Some(incident);
        }
        began
    }
}

/// The data detectors of one rank, over the rows that count.
#[derive(Default)]
pub(super) struct DataProgress {
    stalled: Stalled,
    order: EpochOrder,
    /// The last `DUPLICATE_WINDOW` steps that drew a hashed batch: (data rank, hash).
    window: BTreeMap<i64, (Option<i64>, String)>,
}

impl DataProgress {
    pub(super) fn row(
        &mut self,
        step: i64,
        data: Option<&DataRow>,
        columns: &Columns,
        row: usize,
        scalars: &mut Scalars,
    ) {
        let stall_began = self.stalled.row(step, data);
        let pass_began = self.order.row(step, data);
        if stall_began || pass_began {
            let context = scalars.context(columns, row);
            if stall_began {
                self.stalled.context = context.clone();
            }
            if pass_began {
                self.order.context = context;
            }
        }
        if let Some(data) = data.filter(|data| data.drew()) {
            if let Some(hash) = &data.hash {
                self.window.insert(step, (data.data_rank, hash.clone()));
                if self.window.len() > DUPLICATE_WINDOW {
                    self.window.pop_first();
                }
            }
        }
    }

    pub(super) fn incidents(&self, out: &mut Vec<Value>) {
        out.extend(self.stalled.found.clone());
        out.extend(self.order.found.clone());
    }

    pub(super) fn window(&self) -> &BTreeMap<i64, (Option<i64>, String)> {
        &self.window
    }
}

/// Python's `detect_duplicate_batches` over each rank's window: the run's `duplicate_batches`,
/// without its `kind`, or nothing.
pub fn duplicate_batches(ranks: &BTreeMap<i32, BTreeMap<i64, (Option<i64>, String)>>) -> Option<Value> {
    let compared: Vec<i32> = ranks
        .iter()
        .filter(|(_, window)| !window.is_empty())
        .map(|(rank, _)| *rank)
        .collect();
    if compared.len() < 2 {
        return None;
    }
    let mut at: BTreeMap<i64, Vec<(i32, i64, &str)>> = BTreeMap::new();
    for (rank, window) in ranks {
        for (step, (share, hash)) in window {
            at.entry(*step)
                .or_default()
                .push((*rank, share.unwrap_or(i64::from(*rank)), hash.as_str()));
        }
    }
    let mut first: Option<(i64, Vec<i32>)> = None;
    let mut duplicated = 0_u64;
    for (step, rows) in &at {
        let mut groups: BTreeMap<&str, Vec<(i32, i64)>> = BTreeMap::new();
        for (rank, share, hash) in rows {
            groups.entry(hash).or_default().push((*rank, *share));
        }
        let mut involved: Vec<i32> = groups
            .values()
            .filter(|group| group.iter().any(|(_, share)| *share != group[0].1))
            .flat_map(|group| group.iter().map(|(rank, _)| *rank))
            .collect();
        involved.sort_unstable();
        involved.dedup();
        if !involved.is_empty() {
            duplicated += 1;
            if first.is_none() {
                first = Some((*step, involved));
            }
        }
    }
    let (step, involved) = first?;
    Some(json!({
        "step": step,
        "ranks": involved,
        "duplicated_steps": duplicated,
        "ranks_compared": compared,
    }))
}

// --- resumes -------------------------------------------------------------------------------

/// What a row tells the resume checks.
pub(super) struct ResumeRow {
    pub data: Option<DataRow>,
    pub scale: Option<f64>,
    pub lr: Option<Vec<f64>>,
    pub checkpoint: Option<Value>,
}

impl ResumeRow {
    pub(super) fn read(columns: &Columns, row: usize, data: Option<DataRow>) -> Self {
        let scale = super::value(columns.grad_scale, row).filter(|scale| scale.is_finite());
        let checkpoint = text(columns.checkpoint, row)
            .and_then(|text| serde_json::from_str::<Value>(text).ok())
            .filter(Value::is_object);
        Self {
            data,
            scale,
            lr: columns.lr(row),
            checkpoint,
        }
    }
}

/// What the resume checks found: kept with the rank, like `Attempts`.
#[derive(Default)]
pub(super) struct ResumeFound {
    data_changes: Vec<(i64, Value)>,
    state_changes: Vec<(i64, Value)>,
    pub data_context: Wanted,
    pub state_context: Wanted,
    data_not_checked: Vec<Value>,
    state_not_checked: Vec<Value>,
}

impl ResumeFound {
    fn add(changes: &mut Vec<(i64, Value)>, wanted: &mut Wanted, step: i64, change: Value) {
        changes.push((step, change));
        changes.sort_by_key(|(step, _)| *step); // stable, as Python's sort
        wanted.want(changes[0].0);
    }

    fn incident(kind: &str, changes: &[(i64, Value)], wanted: &Wanted, note: &str, out: &mut Vec<Value>) {
        if let Some((first, _)) = changes.first() {
            let mut incident = json!({
                "kind": kind,
                "step": first,
                "resumes": changes.len(),
                "changes": changes.iter().take(CHANGES_LISTED).map(|(_, change)| change).collect::<Vec<_>>(),
                "note": note,
            });
            if let Some(scalars) = &wanted.scalars {
                incident["scalars"] = scalars.clone();
            }
            out.push(incident);
        }
    }

    pub(super) fn incidents(&self, out: &mut Vec<Value>) {
        Self::incident(
            "resume_data_mismatch",
            &self.data_changes,
            &self.data_context,
            "an observation: a resume that starts its epoch over on purpose repeats data too",
            out,
        );
        Self::incident(
            "resume_incomplete",
            &self.state_changes,
            &self.state_context,
            "an observation: a component left out or reset on purpose looks the same",
            out,
        );
    }

    pub(super) fn not_checked(&self) -> impl Iterator<Item = &Value> {
        self.data_not_checked.iter().chain(&self.state_not_checked)
    }
}

/// The run's history for the resume checks, as `detect_resume_state` keeps it: every earlier
/// attempt with each rollback applied. Held only while the store is replayed.
#[derive(Default)]
pub(super) struct ResumePass {
    hashes: BTreeMap<i64, (Option<String>, i64)>,
    scales: BTreeMap<i64, f64>,
    /// The first learning rate seen and whether another one followed.
    lr: Option<Vec<f64>>,
    lr_varies: bool,
    /// (after step, manifest), in the order written.
    saves: Vec<(i64, Value)>,
    first_scale: Option<f64>,
    seen_data: bool,
    seen_checkpoint: bool,
}

/// The step nearest `resumed_at` (the earlier one on a tie) at which the history drew `hash`.
fn lookup(known: &BTreeMap<i64, (Option<String>, i64)>, hash: &str, resumed_at: i64) -> Option<i64> {
    let same = |(_, (known, _)): &(&i64, &(Option<String>, i64))| known.as_deref() == Some(hash);
    let before = known.range(..resumed_at).rev().find(same).map(|(step, _)| *step);
    let after = known.range(resumed_at + 1..).find(same).map(|(step, _)| *step);
    match (before, after) {
        (None, after) => after,
        (Some(before), Some(after)) if after - resumed_at < resumed_at - before => Some(after),
        (before, _) => before,
    }
}

impl ResumePass {
    /// The first row of an attempt after the first: judged against the history before it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn resumed(
        &self,
        rank: i32,
        attempt: i64,
        step: i64,
        run_start: i64,
        first: &ResumeRow,
        found: &mut ResumeFound,
    ) {
        let has_data = self.seen_data || first.data.is_some();
        let has_checkpoint = self.seen_checkpoint || first.checkpoint.is_some();
        let skipped = |check: &str, reason: &str| json!({"check": check, "step": step, "rank": rank, "reason": reason});
        match self.compare_data(first.data.as_ref().and_then(|data| data.hash.as_deref()), step) {
            Err(reason) => {
                if has_data {
                    found.data_not_checked.push(skipped("resume_data", reason));
                }
            }
            Ok(None) => {}
            Ok(Some(mut change)) => {
                let fields = change.as_object_mut().expect("an object");
                let mut ordered = Map::new();
                ordered.insert("step".into(), json!(step));
                ordered.insert("rank".into(), json!(rank));
                ordered.insert("attempt".into(), json!(attempt));
                ordered.append(fields);
                ResumeFound::add(
                    &mut found.data_changes,
                    &mut found.data_context,
                    step,
                    Value::Object(ordered),
                );
            }
        }
        let (findings, source, saved, checked) = self.state_findings(first, step, run_start);
        if !findings.is_empty() {
            let mut change = json!({
                "step": step,
                "rank": rank,
                "attempt": attempt,
                "source": source,
                "findings": findings,
            });
            if let (Some("save"), Some(after)) = (source, saved) {
                change["saved_after_step"] = json!(after);
            }
            ResumeFound::add(&mut found.state_changes, &mut found.state_context, step, change);
        } else if !checked && has_checkpoint {
            found.state_not_checked.push(skipped(
                "resume_state",
                "no checkpoint manifest, optimizer snapshot or loss scale recorded for this resume",
            ));
        }
    }

    fn compare_data(&self, new: Option<&str>, resumed_at: i64) -> Result<Option<Value>, &'static str> {
        let known: BTreeMap<i64, (Option<String>, i64)> = self
            .hashes
            .iter()
            .filter(|(_, (hash, _))| hash.is_some())
            .map(|(step, value)| (*step, value.clone()))
            .collect();
        let Some(new) = new else {
            return if known.is_empty() {
                Ok(None)
            } else {
                Err("no batch hash recorded at the first step of the resumed attempt")
            };
        };
        if known.is_empty() {
            return Err("no batch hash recorded in the earlier attempts");
        }
        let expected = known.get(&resumed_at);
        if expected.is_some_and(|(hash, _)| hash.as_deref() == Some(new)) {
            return Ok(None);
        }
        let matched = lookup(&known, new, resumed_at);
        if matched.is_none() && expected.is_none() {
            return Err(
                "the earlier attempts ended before the resume step and drew the resumed batch nowhere: a resume that \
                 repeats no batch looks the same as one that skips",
            );
        }
        let finding = match matched {
            None => "differs",
            Some(step) if step < resumed_at => "repeated",
            Some(_) => "skipped",
        };
        let source = expected
            .or_else(|| matched.and_then(|step| known.get(&step)))
            .map(|(_, attempt)| *attempt);
        Ok(Some(json!({
            "basis": if expected.is_some() { "same_step" } else { "lookup" },
            "finding": finding,
            "resumed_hash": new,
            "expected_hash": expected.and_then(|(hash, _)| hash.clone()),
            "matched_step": matched,
            "batches_apart": matched.map(|step| (resumed_at - step).abs()),
            "previous_attempt": source,
        })))
    }

    /// Python's `_state_findings`: the findings, where the components came from, the save it
    /// assumed, and whether anything was checked.
    fn state_findings(
        &self,
        first: &ResumeRow,
        resumed_at: i64,
        run_start: i64,
    ) -> (Vec<Value>, Option<&'static str>, Option<i64>, bool) {
        let empty = Value::Null;
        let checkpoint = first.checkpoint.as_ref().unwrap_or(&empty);
        let mut findings = Vec::new();
        let mut checked = false;
        // The last save written among those after the latest step before the resume.
        let saved = self
            .saves
            .iter()
            .filter(|(after, _)| *after < resumed_at)
            .map(|(after, _)| *after)
            .max()
            .and_then(|latest| self.saves.iter().rev().find(|(after, _)| *after == latest));
        let (manifest, source) = if checkpoint["load"].is_object() {
            (Some(&checkpoint["load"]), Some("load"))
        } else if let Some((_, manifest)) = saved {
            (Some(manifest), Some("save"))
        } else {
            (None, None)
        };
        if let Some(manifest) = manifest {
            checked = true;
            let components = manifest["components"].as_object();
            let has = |name: &str| components.is_some_and(|components| components.contains_key(name));
            let mut expected = vec!["model", "optimizer"];
            if self.lr_varies {
                expected.push("scheduler");
            }
            if self.first_scale.is_some() {
                expected.push("scaler");
            }
            expected.extend(["rng", "data"]);
            for name in expected {
                if !has(name) {
                    findings.push(json!({"component": name, "finding": "missing"}));
                }
            }
            let component = |name: &str| {
                components
                    .and_then(|components| components.get(name))
                    .filter(|c| c.is_object())
            };
            if component("optimizer").is_some_and(|optimizer| int(&optimizer["entries"]) == Some(0)) {
                findings.push(json!({"component": "optimizer", "finding": "empty_in_checkpoint"}));
            }
            if let Some(last_epoch) = component("scheduler").and_then(|scheduler| int(&scheduler["last_epoch"])) {
                if (last_epoch - resumed_at).abs() > SCHEDULER_SLACK {
                    findings.push(json!({"component": "scheduler", "finding": "position", "last_epoch": last_epoch}));
                }
            }
        }
        let snapshot = &checkpoint["optimizer"];
        if snapshot.is_object() {
            checked = true;
            let before = int(&snapshot["entries_before"]);
            let after = int(&snapshot["entries_after"]);
            let step_after = number(&snapshot["step_after"]);
            if before == Some(0) && after.is_some_and(|after| after > 0) {
                findings.push(json!({"component": "optimizer", "finding": "empty", "entries_after": after}));
            } else if let Some(step_after) = step_after.filter(|step| *step <= 1.0 && resumed_at - run_start > 1) {
                findings.push(json!({"component": "optimizer", "finding": "step_reset", "step_after": step_after}));
            }
        }
        let previous_scale = self.scales.range(..=resumed_at).next_back().map(|(_, scale)| *scale);
        if let (Some(scale), Some(initial), Some(previous)) = (first.scale, self.first_scale, previous_scale) {
            checked = true;
            if scale == initial && previous != initial {
                findings.push(json!({
                    "component": "scaler",
                    "finding": "initial_scale",
                    "scale": scale,
                    "expected_scale": previous,
                }));
            }
        }
        (findings, source, saved.map(|(after, _)| *after), checked)
    }

    /// An attempt begins at `step`: what earlier attempts recorded from there on is rolled back.
    pub(super) fn roll_back(&mut self, step: i64) {
        self.hashes.split_off(&step);
        self.scales.split_off(&step);
        self.saves.retain(|(after, _)| *after < step);
    }

    /// Adds a row of the attempt being read; `first_attempt` when it belongs to the run's first.
    pub(super) fn add(&mut self, attempt: i64, step: i64, row: ResumeRow, first_attempt: bool) {
        self.seen_data |= row.data.is_some();
        self.seen_checkpoint |= row.checkpoint.is_some();
        if let Some(data) = row.data.as_ref().filter(|data| data.drew()) {
            self.hashes.insert(step, (data.hash.clone(), attempt));
        }
        if let Some(scale) = row.scale {
            self.scales.insert(step, scale);
            if first_attempt && self.first_scale.is_none() {
                self.first_scale = Some(scale);
            }
        }
        if let Some(lr) = row.lr {
            match &self.lr {
                None => self.lr = Some(lr),
                Some(first) if *first != lr => self.lr_varies = true,
                Some(_) => {}
            }
        }
        if let Some(checkpoint) = row.checkpoint {
            if let Some(save) = checkpoint.get("save").filter(|save| save.is_object()) {
                let after = int(&save["after_step"]).unwrap_or(step);
                self.saves.push((after, save.clone()));
            }
        }
    }
}
