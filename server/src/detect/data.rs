//! Data progress and checkpoint manifests: `data_checks.py` in Rust.
//!
//! The per-rank kinds are `data_stalled` and `epoch_order_repeated`, read from every row that
//! counts, and `resume_data_mismatch` and `resume_incomplete`, judged on the first row of each
//! resumed attempt while the store is replayed (as `resume_lr_change` is). Each rank also
//! keeps its last `DUPLICATE_WINDOW` batch hashes, which [`duplicate_batches`] compares across
//! the ranks of a run: for the run's status in the server, and in [`detect_run`] for
//! `tensorquill-server detect --run`, which reads every rank's rows without a server.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Int32Array, StringArray};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Map, Value};

use super::{column, judge, report, text, with_classes, Columns, Scalars, Thresholds, Wanted};

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

/// A batch hash's kind: the text before its first `:` (`v2` for `v2:cd09ce7a08834a9e`), and
/// `""` for a hash with none, which the writer before C16 made (lane V2, Nov 19, 2026). Two
/// hashes are compared only when their kinds are equal. `data_stalled`,
/// `epoch_order_repeated` and `duplicate_batches` need nothing more: they look for equal
/// hashes, and two of different kinds are never equal, so a pair across the change ends a
/// stretch, breaks an order or makes no duplicate. A resume compares hashes for a difference
/// too, so [`ResumePass::compare_data`] keeps to the resumed hash's kind.
fn hash_kind(hash: &str) -> &str {
    hash.split_once(':').map_or("", |(kind, _)| kind)
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

/// The ranks [`duplicate_batches`] compares: those with a hashed batch in their window, in
/// rank order. The run's status and `detect --run` list them as `ranks_compared` whether or not
/// a duplicate is found, so "compared, none found" can be told from "not compared".
pub fn ranks_compared(ranks: &BTreeMap<i32, BTreeMap<i64, (Option<i64>, String)>>) -> Vec<i32> {
    ranks
        .iter()
        .filter(|(_, window)| !window.is_empty())
        .map(|(rank, _)| *rank)
        .collect()
}

/// Python's `detect_duplicate_batches` over each rank's window: the run's `duplicate_batches`,
/// without its `kind`, or nothing.
pub fn duplicate_batches(ranks: &BTreeMap<i32, BTreeMap<i64, (Option<i64>, String)>>) -> Option<Value> {
    let compared = ranks_compared(ranks);
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

// --- every rank of a run, without a server -------------------------------------------------

/// The record batches of one Arrow IPC stream, as `tensorquill-server detect` is given them.
pub fn read_batches(input: impl std::io::Read) -> Result<Vec<RecordBatch>, String> {
    let reader = arrow::ipc::reader::StreamReader::try_new(input, None)
        .map_err(|error| format!("the input is not an Arrow IPC stream: {error}"))?;
    reader
        .collect::<Result<Vec<RecordBatch>, _>>()
        .map_err(|error| format!("cannot read the Arrow IPC stream: {error}"))
}

/// `tensorquill-server detect --run`: the rows of every rank of one run in, and the run's
/// document out. The rows are grouped by their `rank` column, so it does not matter how the
/// ranks are spread over the batches or the inputs; within a rank they must be grouped by
/// attempt, oldest first, and in recorded order, as [`judge`] reads them. Each rank is judged
/// alone, as `detect` judges one rank, and its [`report`] is in `reports`. Then the ranks'
/// batch-hash windows are compared as the server compares them for a run's status
/// ([`duplicate_batches`]): `duplicate_batches` is that finding or null, `incidents` holds it
/// as an incident, and `ranks_compared` lists the ranks that had a hashed batch to compare,
/// found or not, so "no duplicates" can be told from "nothing to compare". Rows of a second
/// run are refused.
pub fn detect_run(batches: &[RecordBatch], thresholds: &Arc<Thresholds>) -> Result<Value, String> {
    let mut run: Option<String> = None;
    let mut by_rank: BTreeMap<i32, Vec<RecordBatch>> = BTreeMap::new();
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        super::check_keys(batch)?;
        let (Some(run_ids), Some(ranks)) = (
            column::<StringArray>(batch, "run_id"),
            column::<Int32Array>(batch, "rank"),
        ) else {
            unreachable!("check_keys found both columns");
        };
        // The batch cut where the rank changes, so every piece belongs to one rank.
        let mut start = 0;
        while start < batch.num_rows() {
            let rank = ranks.value(start);
            let mut end = start;
            while end < batch.num_rows() && ranks.value(end) == rank {
                let run_id = run_ids.value(end);
                match &run {
                    None => run = Some(run_id.to_string()),
                    Some(known) if known == run_id => {}
                    Some(known) => {
                        return Err(format!(
                            "rows of run {run_id} beside rows of run {known}: send the ranks of one run"
                        ));
                    }
                }
                end += 1;
            }
            by_rank.entry(rank).or_default().push(batch.slice(start, end - start));
            start = end;
        }
    }
    let Some(run_id) = run else {
        return Err("the inputs hold no rows".to_string());
    };
    let mut reports = Vec::new();
    let mut windows = BTreeMap::new();
    for (rank, pieces) in &by_rank {
        let detectors = judge(*rank, pieces, thresholds).map_err(|error| format!("rank {rank}: {error}"))?;
        windows.insert(*rank, detectors.data_window().clone());
        reports.push(report(&run_id, *rank, &detectors, 0));
    }
    let compared = ranks_compared(&windows);
    let found = duplicate_batches(&windows);
    // As the status lists it (`watchdog::incidents`): the finding with its kind and class.
    let mut incidents = Vec::new();
    if let Some(Value::Object(fields)) = &found {
        let mut incident = Map::new();
        incident.insert("kind".into(), json!("duplicate_batches"));
        incident.extend(fields.clone());
        incidents.push(Value::Object(incident));
    }
    with_classes(&mut incidents);
    Ok(json!({
        "schema_version": 1,
        "source": "tensorquill-server",
        "run_id": run_id,
        "ranks": by_rank.keys().collect::<Vec<_>>(),
        "ranks_compared": compared,
        "duplicate_batches": found,
        "incidents": incidents,
        "reports": reports,
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
        Self {
            data,
            scale,
            lr: columns.lr(row),
            checkpoint: checkpoint(columns, row),
        }
    }
}

/// A row's checkpoint manifest: its `checkpoint_json` when that is a JSON object, as Python's
/// `isinstance(record["checkpoint"], Mapping)`. What the resume check reads and what
/// `kind_checks` counts (RS11, lane V7, Nov 19, 2026: text that does not parse counted).
pub(super) fn checkpoint(columns: &Columns, row: usize) -> Option<Value> {
    text(columns.checkpoint, row)
        .filter(|text| text.trim_start().starts_with('{')) // an object, if anything, and no parse for the rest
        .and_then(|text| serde_json::from_str::<Value>(text).ok())
        .filter(Value::is_object)
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
        let mut known: BTreeMap<i64, (Option<String>, i64)> = self
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
        // Only hashes of the resumed one's kind are compared with it (C16): the same batch
        // hashed by two writers gives two hashes, which say neither "same" nor "different".
        let kind = hash_kind(new);
        known.retain(|_, (hash, _)| hash.as_deref().map(hash_kind) == Some(kind));
        if known.is_empty() {
            return Err("the batch hashes before and after the resume are of different kinds");
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;

    /// Rows as (rank, attempt, step, data), `data` the row's `data_json` or nothing.
    fn rows(run: &str, rows: &[(i32, i64, i64, Option<String>)]) -> RecordBatch {
        use arrow::datatypes::{DataType, Field, Schema};
        let schema = Schema::new(vec![
            Field::new("run_id", DataType::Utf8, false),
            Field::new("rank", DataType::Int32, false),
            Field::new("attempt", DataType::Int64, false),
            Field::new("step", DataType::Int64, false),
            Field::new("data_json", DataType::Utf8, true),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from_iter_values(rows.iter().map(|_| run))),
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|row| row.0))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.1))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.2))),
                Arc::new(StringArray::from(
                    rows.iter().map(|row| row.3.clone()).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    }

    fn drew(hash: &str) -> Option<String> {
        Some(json!({"batches": 1, "batch_hash": hash}).to_string())
    }

    /// One rank's steps: its own batches, but rank 0's on the steps in `shared`.
    fn rank(
        rank: i32,
        steps: std::ops::Range<i64>,
        shared: std::ops::Range<i64>,
    ) -> Vec<(i32, i64, i64, Option<String>)> {
        steps
            .map(|step| {
                let whose = if shared.contains(&step) { 0 } else { rank };
                (rank, 1, step, drew(&format!("r{whose}-{step}")))
            })
            .collect()
    }

    fn usual() -> Arc<Thresholds> {
        Arc::default()
    }

    #[test]
    fn detect_run_compares_the_ranks_however_they_are_spread_over_the_batches() {
        let (zero, one, two) = (rank(0, 0..20, 0..0), rank(1, 0..20, 5..9), rank(2, 0..20, 0..0));
        // A batch a rank, as one file a rank gives.
        let apart = [rows("run", &zero), rows("run", &one), rows("run", &two)];
        // One batch with the ranks row by row, and one with a rank's rows in two places.
        let mut mixed = Vec::new();
        for at in 0..20 {
            mixed.extend([zero[at].clone(), one[at].clone(), two[at].clone()]);
        }
        let mut split = [&one[..10], &zero[..], &two[..], &one[10..]].concat();
        let last = split.split_off(40);
        let document = detect_run(&apart, &usual()).unwrap();
        assert_eq!(document["source"], "tensorquill-server");
        assert_eq!(
            (&document["run_id"], &document["ranks"]),
            (&json!("run"), &json!([0, 1, 2]))
        );
        assert_eq!(document["ranks_compared"], json!([0, 1, 2]));
        assert_eq!(
            document["duplicate_batches"],
            json!({"step": 5, "ranks": [0, 1], "duplicated_steps": 4, "ranks_compared": [0, 1, 2]})
        );
        assert_eq!(
            document["incidents"],
            json!([{"kind": "duplicate_batches", "class": "invariant", "step": 5, "ranks": [0, 1],
                    "duplicated_steps": 4, "ranks_compared": [0, 1, 2]}])
        );
        // Each rank's report is the one `detect` gives for that rank alone.
        let reports = document["reports"].as_array().unwrap();
        assert_eq!(reports.len(), 3);
        for (number, piece) in [&zero, &one, &two].into_iter().enumerate() {
            let alone = report(
                "run",
                number as i32,
                &judge(number as i32, &[rows("run", piece)], &usual()).unwrap(),
                0,
            );
            assert_eq!(reports[number], alone);
            assert_eq!((&alone["rank"], &alone["steps"]), (&json!(number), &json!(20)));
        }
        assert_eq!(detect_run(&[rows("run", &mixed)], &usual()).unwrap(), document);
        assert_eq!(
            detect_run(&[rows("run", &split), rows("run", &last)], &usual()).unwrap(),
            document
        );
    }

    #[test]
    fn detect_run_says_which_ranks_it_could_compare_when_it_finds_nothing() {
        // Different batches on every step: compared, and clean.
        let clean = [rows("run", &rank(0, 0..5, 0..0)), rows("run", &rank(1, 0..5, 0..0))];
        let document = detect_run(&clean, &usual()).unwrap();
        assert_eq!(document["duplicate_batches"], Value::Null);
        assert_eq!(
            (&document["incidents"], &document["ranks_compared"]),
            (&json!([]), &json!([0, 1]))
        );
        // One rank, and a rank that hashed nothing: no finding, and not "compared and clean".
        let alone = detect_run(&[rows("run", &rank(0, 0..5, 0..0))], &usual()).unwrap();
        assert_eq!(
            (&alone["duplicate_batches"], &alone["ranks_compared"]),
            (&Value::Null, &json!([0]))
        );
        let unhashed: Vec<_> = (0..5)
            .map(|step| (1, 1, step, Some(json!({"batches": 1}).to_string())))
            .collect();
        let document = detect_run(&[rows("run", &rank(0, 0..5, 0..0)), rows("run", &unhashed)], &usual()).unwrap();
        assert_eq!(
            (&document["ranks"], &document["ranks_compared"]),
            (&json!([0, 1]), &json!([0]))
        );
        assert_eq!(document["duplicate_batches"], Value::Null);
    }

    #[test]
    fn detect_run_reads_a_ranks_attempts_and_leaves_out_what_a_resume_rolled_back() {
        // Rank 1 drew rank 0's batches at 10..20, then resumed at 15 with its own.
        let mut one = rank(1, 0..30, 10..20);
        one.extend((15..30).map(|step| (1, 2, step, drew(&format!("r1-{step}")))));
        let document = detect_run(&[rows("run", &rank(0, 0..30, 0..0)), rows("run", &one)], &usual()).unwrap();
        assert_eq!(
            document["duplicate_batches"],
            json!({"step": 10, "ranks": [0, 1], "duplicated_steps": 5, "ranks_compared": [0, 1]})
        );
        assert_eq!(document["reports"][1]["attempt"], 2);
    }

    #[test]
    fn detect_run_refuses_what_it_cannot_judge() {
        let error = detect_run(
            &[rows("run", &rank(0, 0..5, 0..0)), rows("other", &rank(1, 0..5, 0..0))],
            &usual(),
        )
        .err()
        .unwrap();
        assert!(error.contains("send the ranks of one run"), "{error}");
        let error = detect_run(&[rows("run", &[])], &usual()).err().unwrap();
        assert!(error.contains("no rows"), "{error}");
        // A rank's attempts out of order are refused with the rank named.
        let mut backwards: Vec<_> = (0..5).map(|step| (1, 2, step, None)).collect();
        backwards.extend((0..5).map(|step| (1, 1, step, None)));
        let error = detect_run(&[rows("run", &rank(0, 0..5, 0..0)), rows("run", &backwards)], &usual())
            .err()
            .unwrap();
        assert!(
            error.starts_with("rank 1: ") && error.contains("group the rows by attempt"),
            "{error}"
        );
        assert!(read_batches(&b"{\"step\": 0}\n"[..])
            .err()
            .unwrap()
            .contains("not an Arrow IPC stream"));
    }
}
