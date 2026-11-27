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
//! - *Experts per token.* A record is judged with the `moe_top_k` known when it arrives,
//!   here and in Python since wave AO (BUG-13), so a run that changes it in mid-run is
//!   judged step by step alike.
//! - *Attempts.* A resume rolls back steps that were already judged. That cannot be
//!   undone in place, so the caller rebuilds the state from the store ([`resolve`] gives
//!   the rows that count). `repeated_steps` and `resume_lr_change` read every attempt,
//!   the rolled-back rows too, and need each step's learning rate; that history is held
//!   only while the store is replayed ([`AttemptPass`]), so every new attempt is a replay.
//! - *An incident dated before the row that makes it.* One parameter's gradient spike is
//!   known only once it has lasted `SPIKE_SUSTAIN` recorded steps and is dated at the
//!   first of them. Python then looks back for what was logged on that step; here it is
//!   read when the excursion begins and kept with it. An incident, once made, is never
//!   replaced or dated again, so a prefix and the whole run agree on it. A router flagged
//!   since its first sample at step 0 is the same: its `moe_imbalance` is made on its tenth
//!   flagged sample and dated at the first (`DETECTOR_RULES_CONTRACT.md` part 3), with what
//!   was logged there, kept from that row.
//!
//! - *The per-step MoE samples.* Python lists a sample for every record and every router the
//!   run ever recorded, sized by the most experts that router ever had. A stream learns both
//!   only at its end, so the judged samples are kept as the rows are read and the list is
//!   completed when it is written ([`MoeSamples`]). Only `tensorquill-server detect --samples`
//!   keeps them: the server holds a rank's detectors for the whole run, and its report says
//!   how many samples were flagged ([`RankDetectors::moe_flagged`]) and no more.
//!
//! Three things differ from Python by design:
//!
//! - A value that does not fit its column's type travels in `extra_json` and reads as
//!   absent here (only `optimizer_ownership` and `weight_dtypes`, which have no column,
//!   are read from there, on the rows where parameters first stall). One exception: in
//!   protocol 5's layout, `non_finite` also reads a layer's refused fields there
//!   (`WAVE_AL_CONTRACT.md` section 1.5), as Python reads every value of a layer.
//! - The server never stores a step twice within one attempt, so that case of
//!   `repeated_steps` cannot arise from its own store. Rows given to `tensorquill-server
//!   detect` can hold one, and the pass over every attempt reports it as Python does.
//! - NaN and Inf are `null` in the incidents, which are strict JSON.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use arrow::array::{
    Array, BooleanArray, Float64Array, Int32Array, Int64Array, ListArray, MapArray, StringArray, StructArray,
};
use arrow::record_batch::RecordBatch;
use serde::de::{Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::storage::{Error, Storage};

mod data;
mod measure;
pub use data::{detect_run, duplicate_batches, ranks_compared, read_batches};
use data::{DataProgress, DataRow, ResumeFound, ResumePass, ResumeRow};
use measure::Measurements;
mod system;
use system::{System, SystemColumns};
pub(crate) mod extra_json;
#[cfg(test)]
mod lane_ab2_tests;
#[cfg(test)]
mod lane_ac2_tests;
#[cfg(test)]
mod lane_ad10_off_tests;
#[cfg(test)]
mod lane_ad10_tests;
#[cfg(test)]
mod lane_ad8_score_tests;
#[cfg(test)]
mod lane_ad8_tests;
#[cfg(test)]
mod lane_ae7_tests;
#[cfg(test)]
mod lane_af_d_tests;
#[cfg(test)]
mod lane_af_w_tests;
#[cfg(test)]
mod lane_aj3_tests;
#[cfg(test)]
mod lane_al3_tests;
#[cfg(test)]
mod lane_v7_tests;
#[cfg(test)]
mod lane_w3_tests;
#[cfg(test)]
mod lane_x4_tests;
#[cfg(test)]
mod lane_y2_tests;
#[cfg(test)]
mod lane_z3_tests;
pub mod replica;
#[cfg(test)]
mod wave_ao_tests;

// The same values as the constants of the same names in evidence.py.
pub const OVERLOAD_THRESHOLD: f64 = 0.35;
pub const STARVE_WINDOW: usize = 50;
pub const STARVE_RATIO: f64 = 0.1;
pub const STARVE_MIN_EXPECTED: f64 = 100.0;
pub const DIVERSITY_DROP: f64 = 0.6;
pub const OVERLOAD_JUMP: f64 = 0.25;
pub const OVERLOAD_MIN_HISTORY: usize = 10;
pub const OVERLOAD_WINDOW: usize = 50;
/// Samples a router may stay flagged in the state it started in before it is reported
/// (`DETECTOR_RULES_CONTRACT.md`, part 3).
pub const START_GRACE: u32 = 10;
pub const SPIKE_FACTOR: f64 = 5.0;
pub const SPIKE_WINDOW: usize = 50;
pub const SPIKE_MIN_HISTORY: usize = 10;
/// Recorded steps in a row a parameter must stay at `SPIKE_FACTOR` times its own median to be
/// reported on its own; a single such step is ordinary (see `SPIKE_SUSTAIN` in `evidence.py`).
pub const SPIKE_SUSTAIN: u32 = 3;
/// `update_spike` (Altenbernd et al., arXiv 2604.00726, eq. 13): the sensitivity, and how many
/// of the last unflagged step-to-step changes of the largest RMS update make the baseline (the
/// `update_window` option).
pub const UPDATE_ALPHA: f64 = 0.05;
pub const UPDATE_WINDOW: usize = 100;
pub const ACT_SPIKE_FACTOR: f64 = 5.0;
pub const ACT_WINDOW: usize = 50;
pub const ACT_MIN_HISTORY: usize = 10;
pub const ACT_DRIFT_FACTOR: f64 = 3.0;
pub const ACT_RECENT: usize = 10;
/// The records a module needs before `activation_drift` judges it (`DETECTOR_RULES_CONTRACT.md`,
/// part 2): its last `ACT_RECENT` against the `ACT_WINDOW` before those, which then hold none of
/// its first `ACT_WINDOW` records.
pub const ACT_DRIFT_NEEDS: usize = 2 * ACT_WINDOW + ACT_RECENT;
pub const ACT_DEAD_RISE: f64 = 0.25;
pub const STALL_STEPS: u32 = 3;
/// Onsets listed in one `stalled_weights` incident; `onsets_omitted` counts the rest.
pub const STALL_ONSETS: usize = 20;
pub const SCALE_SKIP_RUN: u64 = 10;
pub const SCALE_FLOOR: f64 = 1.0;
pub const DRIFT_TOLERANCE: f64 = 1e-6;
pub const RESUME_LR_TOLERANCE: f64 = 1e-6;
pub const ATTN_MIN_HISTORY: usize = 2;
pub const ATTN_WINDOW: usize = 5;
pub const ATTN_SUSTAIN: usize = 2;
pub const ATTN_COLLAPSE_RATIO: f64 = 0.5;
pub const ATTN_UNIFORM: f64 = 0.98;
pub const ATTN_UNIFORM_RISE: f64 = 0.2;
pub const ATTN_LOGIT_FACTOR: f64 = 2.0;
pub const ATTN_KINDS: [&str; 3] = ["attention_collapse", "attention_uniform", "attention_logit_growth"];
const ATTN_MODULES: usize = 20;

/// The incident kinds this module can report, in the order `build_evidence` lists them, but
/// that the GPU memory, NVML and data-wait kinds (`detect/system.rs`) come between
/// `non_finite` and `replica_drift`, as in [`RankDetectors::incidents`]. Python's
/// cross-rank `memory_imbalance` is not here: the detectors see one rank at a time.
pub const PORTED: [&str; 26] = [
    "moe_imbalance",
    "starved_experts",
    "grad_spike",
    "update_spike",
    "stalled_weights",
    "repeated_steps",
    "loss_scale_trouble",
    "resume_lr_change",
    "resume_data_mismatch",
    "resume_incomplete",
    "data_stalled",
    "epoch_order_repeated",
    "activation_spike",
    "activation_drift",
    "dead_activations",
    "attention_collapse",
    "attention_uniform",
    "attention_logit_growth",
    "non_finite",
    "replica_drift",
    "memory_growth",
    "memory_pressure",
    "allocator_retries",
    "hardware_errors",
    "gpu_throttled",
    "input_starvation",
];
/// The kinds only `python -m tensorquill` reports: their sources never reach the server.
pub const NOT_PORTED: [&str; 3] = ["collective_hang", "collective_desync", "traincheck_violation"];

/// What each incident kind claims, as `INCIDENT_CLASSES` in `evidence.py` (the reference, and
/// `test_incident_classes.py` holds the two identical): `invariant` (should never happen, by a
/// fixed rule), `anomaly` (judged against the run's own baseline) or `observation` (context that
/// claims no failure). Every kind either language reports, the server's or not, is here.
pub const INCIDENT_CLASSES: [(&str, &str); 33] = [
    ("moe_imbalance", "anomaly"),
    ("starved_experts", "anomaly"),
    ("grad_spike", "anomaly"),
    ("update_spike", "anomaly"),
    ("stalled_weights", "invariant"),
    ("repeated_steps", "invariant"),
    ("loss_scale_trouble", "invariant"),
    ("resume_lr_change", "observation"),
    ("resume_data_mismatch", "observation"),
    ("resume_incomplete", "observation"),
    ("data_stalled", "invariant"),
    ("epoch_order_repeated", "invariant"),
    ("duplicate_batches", "invariant"),
    ("activation_spike", "anomaly"),
    ("activation_drift", "anomaly"),
    ("dead_activations", "anomaly"),
    ("attention_collapse", "anomaly"),
    ("attention_uniform", "anomaly"),
    ("attention_logit_growth", "anomaly"),
    ("non_finite", "invariant"),
    ("replica_drift", "invariant"),
    ("memory_growth", "anomaly"),
    ("memory_pressure", "observation"),
    ("allocator_retries", "observation"),
    ("hardware_errors", "invariant"),
    ("gpu_throttled", "observation"),
    ("input_starvation", "anomaly"),
    ("memory_imbalance", "observation"),
    ("job_stalled", "observation"),
    ("rank_silent", "observation"),
    ("collective_hang", "invariant"),
    ("collective_desync", "invariant"),
    ("traincheck_violation", "invariant"),
];

/// The class of an incident kind, or None for a kind this version does not know.
pub fn incident_class(kind: &str) -> Option<&'static str> {
    INCIDENT_CLASSES
        .iter()
        .find(|(known, _)| *known == kind)
        .map(|(_, class)| *class)
}

/// Puts `class` on each incident of a known kind.
pub fn with_classes(incidents: &mut [Value]) {
    for incident in incidents {
        if let Some(class) = incident["kind"].as_str().and_then(incident_class) {
            incident["class"] = json!(class);
        }
    }
}

/// What each input count counts, as a report names it under `needs`: `INPUT_NEEDS` in
/// `evidence.py`, which is the reference.
pub const INPUT_NEEDS: [(&str, &str); 16] = [
    ("moe", "moe: a router's expert counts"),
    ("grad_total", "grad_norm or clip_grad_norm"),
    ("update", "update_rms with grad_norm or clip_grad_norm (updates=True)"),
    ("weight_check", "weights_unchanged with grad_norms (full capture)"),
    ("steps", "step records"),
    ("loss_scale", "step_skipped (scaler=)"),
    ("lr", "lr"),
    ("data", "data (session.watch_loader)"),
    ("checkpoint", "checkpoint, or grad_scale (scaler=)"),
    ("layers", "layers (full capture)"),
    ("attention", "probe with attention (probe=)"),
    ("numbers", "layers, grad_norm, clip_grad_norm or grad_norms"),
    ("replica_spread", "replica_spread (replica_group=)"),
    ("cuda", "system.cuda (system=True)"),
    ("nvml", "system.nvml (system=True)"),
    ("data_wait", "data_wait with step_time (data_wait=True)"),
];

/// The input each kind of [`PORTED`] is judged from, in that order: `KIND_INPUTS` in
/// `evidence.py` (`test_kind_checks.py` holds the two identical).
pub const KIND_INPUTS: [(&str, &str); 26] = [
    ("moe_imbalance", "moe"),
    ("starved_experts", "moe"),
    ("grad_spike", "grad_total"),
    ("update_spike", "update"),
    ("stalled_weights", "weight_check"),
    ("repeated_steps", "steps"),
    ("loss_scale_trouble", "loss_scale"),
    ("resume_lr_change", "lr"),
    ("resume_data_mismatch", "data"),
    ("resume_incomplete", "checkpoint"),
    ("data_stalled", "data"),
    ("epoch_order_repeated", "data"),
    ("activation_spike", "layers"),
    ("activation_drift", "layers"),
    ("dead_activations", "layers"),
    ("attention_collapse", "attention"),
    ("attention_uniform", "attention"),
    ("attention_logit_growth", "attention"),
    ("non_finite", "numbers"),
    ("replica_drift", "replica_spread"),
    ("memory_growth", "cuda"),
    ("memory_pressure", "cuda"),
    ("allocator_retries", "cuda"),
    ("hardware_errors", "nvml"),
    ("gpu_throttled", "nvml"),
    ("input_starvation", "data_wait"),
];

/// How many rows carrying a kind's input its rule needs: `records_needed` in a report's
/// `kind_checks`, in the order of [`PORTED`]. `RECORDS_NEEDED` in `evidence.py` is the
/// reference and gives the definition and each kind's rules: for each rule of the kind, the
/// fewest rows it can report an incident on, and the kind needs the largest of those. A kind
/// with one rule and no warm-up needs 1. `starved_experts` needs the window a run is judged
/// with and `update_spike` the baseline it is judged with ([`records_needed`]).
pub const RECORDS_NEEDED: [(&str, u64); 26] = [
    // The diversity drop of a top-k router (k >= 3): its earlier samples and the one judged.
    ("moe_imbalance", OVERLOAD_MIN_HISTORY as u64 + 1),
    ("starved_experts", STARVE_WINDOW as u64),
    // One parameter: its earlier norms, then `SPIKE_SUSTAIN` in a row (the total needs 11).
    ("grad_spike", SPIKE_MIN_HISTORY as u64 + SPIKE_SUSTAIN as u64),
    // One row to compare the first with, the changes that make the first baseline, and the one judged.
    ("update_spike", UPDATE_WINDOW as u64 + 2),
    ("stalled_weights", STALL_STEPS as u64),
    ("repeated_steps", 1),
    // Skips in a row (a scale below the floor needs 1).
    ("loss_scale_trouble", SCALE_SKIP_RUN),
    ("resume_lr_change", 1),
    ("resume_data_mismatch", 1),
    ("resume_incomplete", 1),
    // The same batch: the batch and the steps that repeat it (no batch needs 3).
    ("data_stalled", data::DATA_STALL_STEPS as u64 + 1),
    // A pass of two different batches, and the next pass matching both.
    ("epoch_order_repeated", 4),
    ("activation_spike", ACT_MIN_HISTORY as u64 + 1),
    // A module's 110th record: the trailing baseline then holds none of its first 50.
    ("activation_drift", ACT_DRIFT_NEEDS as u64),
    ("dead_activations", (ACT_WINDOW + ACT_RECENT) as u64),
    ("attention_collapse", (ATTN_MIN_HISTORY + ATTN_SUSTAIN) as u64),
    ("attention_uniform", (ATTN_MIN_HISTORY + ATTN_SUSTAIN) as u64),
    ("attention_logit_growth", (ATTN_MIN_HISTORY + ATTN_SUSTAIN) as u64),
    ("non_finite", 1),
    ("replica_drift", 1),
    ("memory_growth", system::MEMORY_GROWTH_READINGS as u64),
    ("memory_pressure", 1),
    ("allocator_retries", 1),
    // An ECC count above the attempt's first reading (an Xid needs 1).
    ("hardware_errors", 2),
    ("gpu_throttled", 1),
    // The rise: the first full window as the baseline and one more step (the share needs 20).
    ("input_starvation", system::INPUT_WINDOW as u64 + 1),
];

/// [`RECORDS_NEEDED`] for one kind as a run applies it: `starved_experts` needs the
/// `starve_window` the run is judged with, `update_spike` its `update_window` and two rows
/// more, every other kind its constant.
pub fn records_needed(kind: &str, thresholds: &Thresholds) -> u64 {
    if kind == "starved_experts" {
        return thresholds.starve_window as u64;
    }
    if kind == "update_spike" {
        return thresholds.update_window as u64 + 2;
    }
    RECORDS_NEEDED
        .iter()
        .find(|(known, _)| *known == kind)
        .map_or(1, |(_, needed)| *needed)
}

/// What a kind's clean check does not show, said in its `kind_checks` entry as `note` when the
/// kind is `checked`: `KIND_NOTES` in `evidence.py`, which is the reference.
pub const KIND_NOTES: [(&str, &str); 1] = [(
    "replica_drift",
    "equal replicas do not rule out silent data corruption: a corrupted gradient is reduced with the others \
     and the same wrong result goes back to every rank, so the replicas stay equal (SCOUT, arXiv 2608.11034, \
     pp. 1 and 3)",
)];

/// The `not_checked` entry of the `update_spike` pairs that more than one clip call kept from
/// being compared (`WAVE_AE_CONTRACT.md`, section 1). It names the kind and does not make it
/// `unavailable`.
const CLIP_GROUPS: &str = "clip_groups";

/// The kinds a `not_checked` entry stands for when it lists none itself.
const CHECK_KINDS: [(&str, &str); 3] = [
    ("resume_lr", "resume_lr_change"),
    ("resume_data", "resume_data_mismatch"),
    ("resume_state", "resume_incomplete"),
];

/// The experts-per-token override: one value for every router, or a value for each router named.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum TopK {
    Every(i64),
    Named(BTreeMap<String, i64>),
}

/// The detector options a run can set: `build_evidence`'s `overload_threshold`,
/// `drift_tolerance`, `starve_window`, `starve_ratio`, `top_k` and `update_window`. The default is the value
/// each rule has always applied. They arrive as one JSON object, on `tensorquill-server
/// --thresholds` for the server's detectors and on `tensorquill-server detect --thresholds`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Thresholds {
    /// The base assignment-share threshold for top-1 and top-2 routers.
    pub overload_threshold: f64,
    /// The relative spread between replicas above which a parameter has drifted.
    pub drift_tolerance: f64,
    /// How many samples of a router the starved check sums over.
    pub starve_window: usize,
    /// An expert is starved below this fraction of an even share of that window.
    pub starve_ratio: f64,
    /// Experts chosen per token, in place of the recorded `moe_top_k`.
    pub top_k: Option<TopK>,
    /// How many of the last unflagged step-to-step changes of the update size make the baseline
    /// of `update_spike`.
    pub update_window: usize,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            overload_threshold: OVERLOAD_THRESHOLD,
            drift_tolerance: DRIFT_TOLERANCE,
            starve_window: STARVE_WINDOW,
            starve_ratio: STARVE_RATIO,
            top_k: None,
            update_window: UPDATE_WINDOW,
        }
    }
}

impl Thresholds {
    /// Reads the JSON object given on the command line. A key left out keeps its default
    /// and `"top_k": null` means the recorded value. Refused: anything but an object, a key
    /// that is not one of the six, a value of the wrong type, a number that is negative or
    /// not finite, a window below 1 and a `top_k` below 1.
    pub fn parse(text: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(text).map_err(|error| format!("not JSON: {error}"))?;
        if !value.is_object() {
            // A derived struct would also read an array, by position.
            return Err(format!(
                "expected a JSON object such as {{\"overload_threshold\": 0.5}}, got {value}"
            ));
        }
        let thresholds: Self = serde_json::from_value(value).map_err(|error| error.to_string())?;
        for (name, number) in [
            ("overload_threshold", thresholds.overload_threshold),
            ("drift_tolerance", thresholds.drift_tolerance),
            ("starve_ratio", thresholds.starve_ratio),
        ] {
            if !number.is_finite() || number < 0.0 {
                return Err(format!("{name} must be a finite number of at least 0, got {number}"));
            }
        }
        if thresholds.starve_window < 1 {
            return Err("starve_window must be at least 1".to_string());
        }
        if thresholds.update_window < 1 {
            return Err("update_window must be at least 1".to_string());
        }
        let chosen: Vec<i64> = match &thresholds.top_k {
            None => Vec::new(),
            Some(TopK::Every(k)) => vec![*k],
            Some(TopK::Named(each)) => each.values().copied().collect(),
        };
        if let Some(k) = chosen.into_iter().find(|k| *k < 1) {
            return Err(format!("top_k must be at least 1, got {k}"));
        }
        Ok(thresholds)
    }

    /// The override for one router, if there is one.
    fn top_k_for(&self, router: &str) -> Option<i64> {
        match self.top_k.as_ref()? {
            TopK::Every(k) => Some(*k),
            TopK::Named(each) => each.get(router).copied(),
        }
    }

    /// Every threshold the detectors apply, with these options in place: the `thresholds`
    /// of a report, which a reader compares with the ones it asked for.
    pub fn applied(&self) -> Value {
        let top_k = match &self.top_k {
            None => Value::Null,
            Some(TopK::Every(k)) => json!(k),
            Some(TopK::Named(each)) => json!(each),
        };
        json!({
            "overload_threshold": self.overload_threshold,
            "starve_window": self.starve_window,
            "starve_ratio": self.starve_ratio,
            "top_k": top_k,
            "diversity_drop": DIVERSITY_DROP,
            "overload_jump": OVERLOAD_JUMP,
            "spike_factor": SPIKE_FACTOR,
            "spike_window": SPIKE_WINDOW,
            "spike_sustain": SPIKE_SUSTAIN,
            "update_window": self.update_window,
            "update_alpha": UPDATE_ALPHA,
            "activation_spike_factor": ACT_SPIKE_FACTOR,
            "activation_drift_factor": ACT_DRIFT_FACTOR,
            "activation_dead_rise": ACT_DEAD_RISE,
            "stall_steps": STALL_STEPS,
            "scale_skip_run": SCALE_SKIP_RUN,
            "scale_floor": SCALE_FLOOR,
            "drift_tolerance": self.drift_tolerance,
            "resume_lr_tolerance": RESUME_LR_TOLERANCE,
            "data_stall_steps": data::DATA_STALL_STEPS,
            "order_min_batches": data::ORDER_MIN_BATCHES,
            "order_compare": data::ORDER_COMPARE,
            "duplicate_window": data::DUPLICATE_WINDOW,
            "scheduler_slack": data::SCHEDULER_SLACK,
            "attention_min_history": ATTN_MIN_HISTORY,
            "attention_window": ATTN_WINDOW,
            "attention_sustain": ATTN_SUSTAIN,
            "attention_collapse_ratio": ATTN_COLLAPSE_RATIO,
            "attention_uniform": ATTN_UNIFORM,
            "attention_uniform_rise": ATTN_UNIFORM_RISE,
            "attention_logit_factor": ATTN_LOGIT_FACTOR,
            "memory_growth_readings": system::MEMORY_GROWTH_READINGS,
            "memory_growth_fraction": system::MEMORY_GROWTH_FRACTION,
            "memory_growth_bytes": system::MEMORY_GROWTH_BYTES,
            "memory_reserved_fraction": system::MEMORY_RESERVED_FRACTION,
            "throttle_reported": system::THROTTLE_REPORTED,
            "input_window": system::INPUT_WINDOW,
            "input_share": system::INPUT_SHARE,
            "input_rise": system::INPUT_RISE,
        })
    }
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

/// Earlier finite values a score needs (`DETECTOR_RULES_CONTRACT.md`, part 1).
pub const SCORE_MIN: usize = 10;
/// The MAD times this is the spread a score is in units of: a standard deviation, for normal data.
pub const SCORE_SCALE: f64 = 1.4826;

/// The score beside a ratio (part 1): how far `value` is from the median of the earlier values,
/// in units of `SCORE_SCALE` times their median absolute deviation. The earlier values are the
/// ones the ratio's baseline was taken over; one that is not finite does not count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Score {
    Number(f64),
    /// Fewer than `SCORE_MIN` finite earlier values.
    ShortHistory,
    /// More than half the earlier values are one value: the MAD is 0, and nothing is divided by it.
    NoDispersion,
}

impl Score {
    /// The score of `value` against `earlier`, with the arithmetic in the reference's order: the
    /// median as `_median` takes it, the MAD as the median of the absolute distances from it.
    pub(crate) fn of(value: f64, earlier: impl IntoIterator<Item = f64>, scratch: &mut Vec<f64>) -> Self {
        scratch.clear();
        scratch.extend(earlier.into_iter().filter(|value| value.is_finite()));
        if scratch.len() < SCORE_MIN {
            return Self::ShortHistory;
        }
        let centre = median(scratch);
        for value in scratch.iter_mut() {
            *value = (*value - centre).abs();
        }
        let spread = median(scratch);
        if spread == 0.0 {
            return Self::NoDispersion;
        }
        Self::Number((value - centre) / (SCORE_SCALE * spread))
    }

    /// Puts `score` on `hit`, and `score_note` when there is no number: a null score is never 0
    /// and never left out.
    pub(crate) fn put(self, hit: &mut Value) {
        match self {
            Self::Number(score) => hit["score"] = json!(score),
            Self::ShortHistory | Self::NoDispersion => {
                hit["score"] = Value::Null;
                hit["score_note"] = json!(if self == Self::ShortHistory {
                    "short_history"
                } else {
                    "no_dispersion"
                });
            }
        }
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

/// The sum of a router's counts, exact as Python's integer sum is (BUG-01): `i64` counts summed in
/// `i64` passed its range on valid rows, which panicked a debug build and wrapped in a release build.
/// Of any sum within `i64`, `as f64` is what it was.
fn counts_sum(counts: &[i64]) -> i128 {
    counts.iter().map(|count| i128::from(*count)).sum()
}

/// Python's `_diversity_from_counts`: the effective number of experts over their number.
fn diversity_from_counts(counts: &[i64]) -> f64 {
    let total = counts_sum(counts);
    if total <= 0 || counts.is_empty() {
        return 0.0;
    }
    let entropy = -fsum(counts.iter().filter(|count| **count > 0).map(|count| {
        let share = *count as f64 / total as f64;
        share * share.ln()
    }));
    entropy.exp() / counts.len() as f64
}

fn effective_overload_threshold(experts: usize, tokens: i128, base: f64) -> f64 {
    if experts <= 1 || tokens <= 0 {
        return 1.0;
    }
    let mean = 1.0 / experts as f64;
    let se = (mean * (1.0 - mean) / tokens as f64).sqrt();
    base.max(1.5 * mean).max(mean + 3.0 * se)
}

fn dead_experts_significant(experts: usize, tokens: i128, overloaded: bool, dead: usize, diversity: f64) -> bool {
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

/// Who reported a skipped update (`STEP_AND_EVIDENCE_CONTRACT.md`, section 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SkippedBy {
    /// Only the caller reported it. `applied` is the record's `scaler_applied: true`: the SDK
    /// observed the scaler apply this update (section 1, rule 2 as amended by the author). Without
    /// it the scaler's outcome is unknown.
    Caller {
        applied: bool,
    },
    Scaler,
    Both,
}

/// A row's skipped update: None when `step_skipped` is not true, else who reported it, from
/// `step_skipped_by` in `extra_json`, which is parsed only on a row that names it. A skipped row
/// without the key was written by a scaler, the only thing that could write it before the key, and
/// is read as the scaler's; so is a value this version does not know. `scaler_applied`, which
/// travels beside it, is read on a caller's skip only, and only as `true`. The second value says
/// whether the row carries the key at all (the rank's `skipped_updates` is present only then).
///
/// The third says the skip is a partial step (`WAVE_AE_CONTRACT.md`, section 4): a skip the
/// scaler confirmed (alone or with the caller) whose `step_skipped_part`, also in `extra_json`
/// and parsed only on a row that names it, is two whole numbers with 0 < skipped < of. On a skip
/// only the caller reported, and on an applied step, the key says nothing.
fn skipped_by(columns: &Columns, row: usize) -> (Option<SkippedBy>, bool, bool) {
    let extra: Option<Value> = text(columns.extra, row)
        .filter(|extra| extra.contains("\"step_skipped_by\"") || extra.contains("\"step_skipped_part\""))
        .and_then(extra_json::parse);
    let said = extra.as_ref().and_then(|extra| extra.get("step_skipped_by"));
    let carried = said.is_some();
    if !flag(columns.step_skipped, row) {
        return (None, carried, false);
    }
    let by = match said.and_then(Value::as_str) {
        Some("caller") => SkippedBy::Caller {
            applied: extra.as_ref().is_some_and(|extra| extra["scaler_applied"] == true),
        },
        Some("both") => SkippedBy::Both,
        _ => SkippedBy::Scaler,
    };
    let part = extra.as_ref().and_then(|extra| extra.get("step_skipped_part"));
    let mixed = part.is_some_and(|part| {
        let count = |key| part.get(key).and_then(Value::as_u64);
        matches!((count("skipped"), count("of")), (Some(skipped), Some(of)) if 0 < skipped && skipped < of)
    });
    let partial = mixed && matches!(by, SkippedBy::Scaler | SkippedBy::Both);
    (Some(by), carried, partial)
}

/// The gradients that are not finite on a row, named as a `non_finite` incident names them: each
/// parameter in `grad_norms`, else `<total>` for the total norm, and `<total before clipping>`.
fn non_finite_gradients<'a>(columns: &'a Columns, row: usize) -> Vec<&'a str> {
    let mut bad: Vec<&str> = Vec::new();
    if let Some(norms) = &columns.grad_norms {
        bad.extend(
            norms
                .row(row)
                .filter(|entry| !norms.values.is_null(*entry) && !norms.values.value(*entry).is_finite())
                .map(|entry| norms.keys.value(entry)),
        );
    }
    let not_finite = |column| value(column, row).is_some_and(|norm| !norm.is_finite());
    if not_finite(columns.grad_norm) && bad.is_empty() {
        bad.push("<total>");
    }
    if not_finite(columns.clip_grad_norm) {
        bad.push("<total before clipping>");
    }
    bad
}

/// The `grad_norms` entries a row's column could not hold, as they travel in `extra_json` under
/// `grad_norms` (`WAVE_AJ_CONTRACT.md` section 1), or None when it has none. `extra_json` is
/// parsed only on a row that names the key.
fn refused_gradients(columns: &Columns, row: usize) -> Option<Value> {
    let extra = text(columns.extra, row).filter(|extra| extra.contains("\"grad_norms\""))?;
    match extra_json::parse(extra)? {
        Value::Object(mut extra) => extra.remove("grad_norms"),
        _ => None,
    }
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
    /// The RMS of the optimizer's update to each parameter. A sender from before the column
    /// has none, and `update_spike` is then not configured.
    update_rms: Option<MapColumn<'a, Float64Array>>,
    /// Each tensor's norm at a weight check, read only by the measurement `weight_jumps`.
    weight_norms: Option<MapColumn<'a, Float64Array>>,
    scalars: Option<MapColumn<'a, Float64Array>>,
    moe_top_k: Option<MapColumn<'a, Int32Array>>,
    moe_counts: Option<MapColumn<'a, ListArray>>,
    moe_count_values: Option<&'a Int64Array>,
    /// The routers' statistics: protocol 5's typed map, or protocol 4's `moe_stats_json`.
    moe_stats: Option<StructMap<'a, RouterStatsColumn<'a>>>,
    /// The layers' statistics: protocol 5's typed map, or protocol 4's `layers_json`.
    layers: Option<StructMap<'a, LayersColumn<'a>>>,
    data: Option<&'a StringArray>,
    checkpoint: Option<&'a StringArray>,
    probe: Option<&'a StringArray>,
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
            update_rms: MapColumn::new(batch, "update_rms"),
            weight_norms: MapColumn::new(batch, "weight_norms"),
            scalars: MapColumn::new(batch, "scalars"),
            moe_top_k: MapColumn::new(batch, "moe_top_k"),
            moe_counts: moe_counts.filter(|_| moe_count_values.is_some()),
            moe_count_values,
            moe_stats: StructMap::new(batch, "moe_stats", "moe_stats_json", RouterStatsColumn::new),
            layers: StructMap::new(batch, "layers", "layers_json", LayersColumn::new),
            data: column(batch, "data_json"),
            checkpoint: column(batch, "checkpoint_json"),
            probe: column(batch, "probe_json"),
        })
    }
}

impl<'a> Columns<'a> {
    /// A row's layers in stored order, each with the statistics the detectors read, from whichever
    /// layout the batch has. An entry that is not an object is absent and the rest of the map is
    /// read, in both layouts: protocol 5's null struct (contract 1.5) and, since lane AL-3′ (the
    /// owner, Nov 24, S4), protocol 4's JSON entry ([`JsonLayers`]).
    fn layers(&self, row: usize) -> Vec<(Cow<'a, str>, LayerStats)> {
        match &self.layers {
            None => Vec::new(),
            Some(StructMap::Json(layers)) => text(Some(*layers), row)
                .and_then(|layers| serde_json::from_str::<JsonLayers>(layers).ok())
                .map_or_else(Vec::new, |JsonLayers(layers)| layers),
            Some(StructMap::Typed(layers)) => layers.row(row, text(self.extra, row)),
        }
    }

    /// A row's routers in stored order, each with its recorded diversity when that is a finite
    /// number, or None when the row records no statistics (or protocol 4's text does not parse).
    fn router_diversities(&self, row: usize) -> Option<Vec<(Cow<'a, str>, Option<f64>)>> {
        match self.moe_stats.as_ref()? {
            StructMap::Json(stats) => {
                let Members(stats) = serde_json::from_str::<Members<RouterStats>>(text(Some(*stats), row)?).ok()?;
                Some(
                    stats
                        .into_iter()
                        .map(|(name, stats)| (name, stats.diversity.0))
                        .collect(),
                )
            }
            StructMap::Typed(stats) => stats.row(row),
        }
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
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
                Ok(Members(read_members(map)?))
            }
        }
        deserializer.deserialize_map(MembersVisitor(PhantomData))
    }
}

/// An object's members in document order, read from its map.
fn read_members<'de, A: MapAccess<'de>, T: Deserialize<'de>>(mut map: A) -> Result<Vec<(Cow<'de, str>, T)>, A::Error> {
    let mut members = Vec::with_capacity(map.size_hint().unwrap_or(0));
    while let Some(Key(key)) = map.next_key()? {
        members.push((key, map.next_value()?));
    }
    Ok(members)
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
            // A bool is not a number (DET-16): an imported `true` is not a measurement of 1.
            fn visit_bool<E>(self, _: bool) -> Result<Number, E> {
                Ok(Number(None))
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

    /// Whether a value, as `extra_json` holds a refused field, is one of the capture's strings for
    /// NaN and Inf: what the visitor below reads as not finite.
    fn is_non_finite_text(value: &Value) -> bool {
        matches!(value, Value::String(text) if matches!(text.as_str(), "nan" | "inf" | "-inf"))
    }
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
            // A bool is not a number (DET-16), as Python's `_finite` reads it.
            fn visit_bool<E>(self, _: bool) -> Result<Stat, E> {
                Ok(Stat::ABSENT)
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
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<LayerStats, A::Error> {
                LayerStats::from_map(map)
            }
        }
        deserializer.deserialize_map(LayerVisitor)
    }
}

impl LayerStats {
    /// One layer's object, read field by field as the parser gives it.
    fn from_map<'de, A: MapAccess<'de>>(mut map: A) -> Result<Self, A::Error> {
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

/// Protocol 4's `layers_json` read entry by entry (the owner, Nov 24, S4): its layers in document
/// order, an entry that is not an object left out, so that it is absent to every detector and the
/// rest of the map is read, as `LayersColumn::row` reads protocol 5's null struct (contract 1.5).
/// The text is read once, in one pass: each entry's value is told apart by the parser's own
/// callback (`deserialize_any`), an object read as `LayerStats` and any other value skipped where
/// it stands. A text that is not JSON still reads as nothing, as before; so does one whose name
/// serde_json refuses (a lone surrogate's escape, the known difference of case
/// `layers_name_arrow_cannot_encode`). `moe_stats_json` is not read this way: it is still read
/// whole (`Members<RouterStats>`).
struct JsonLayers<'a>(Vec<(Cow<'a, str>, LayerStats)>);

impl<'de: 'a, 'a> Deserialize<'de> for JsonLayers<'a> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// One entry's value: its statistics when it is an object, else nothing.
        struct Entry(Option<LayerStats>);

        impl<'de> Deserialize<'de> for Entry {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                struct EntryVisitor;
                impl<'de> Visitor<'de> for EntryVisitor {
                    type Value = Entry;
                    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                        f.write_str("any value")
                    }
                    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Entry, A::Error> {
                        LayerStats::from_map(map).map(|stats| Entry(Some(stats)))
                    }
                    fn visit_f64<E>(self, _: f64) -> Result<Entry, E> {
                        Ok(Entry(None))
                    }
                    fn visit_i64<E>(self, _: i64) -> Result<Entry, E> {
                        Ok(Entry(None))
                    }
                    fn visit_u64<E>(self, _: u64) -> Result<Entry, E> {
                        Ok(Entry(None))
                    }
                    fn visit_bool<E>(self, _: bool) -> Result<Entry, E> {
                        Ok(Entry(None))
                    }
                    fn visit_str<E>(self, _: &str) -> Result<Entry, E> {
                        Ok(Entry(None))
                    }
                    fn visit_unit<E>(self) -> Result<Entry, E> {
                        Ok(Entry(None))
                    }
                    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Entry, A::Error> {
                        while seq.next_element::<IgnoredAny>()?.is_some() {}
                        Ok(Entry(None))
                    }
                }
                deserializer.deserialize_any(EntryVisitor)
            }
        }

        struct LayersVisitor<'a>(PhantomData<&'a ()>);
        impl<'de: 'a, 'a> Visitor<'de> for LayersVisitor<'a> {
            type Value = JsonLayers<'a>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut layers = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some(Key(key)) = map.next_key()? {
                    if let Entry(Some(stats)) = map.next_value()? {
                        layers.push((key, stats));
                    }
                }
                Ok(JsonLayers(layers))
            }
        }
        deserializer.deserialize_map(LayersVisitor(PhantomData))
    }
}

#[derive(Deserialize)]
struct RouterStats {
    #[serde(default)]
    diversity: Number,
}

// --- the typed columns (protocol 5) -------------------------------------------------------------

/// A column of struct maps in whichever layout its batch has (`WAVE_AL_CONTRACT.md` section 3): the
/// typed map of protocol 5 where the batch has it, else protocol 4's JSON text. The layout is told
/// per batch by the column's name, never per row: a null typed row is a row without the key.
enum StructMap<'a, T> {
    Typed(T),
    Json(&'a StringArray),
}

impl<'a, T> StructMap<'a, T> {
    fn new(
        batch: &'a RecordBatch,
        typed: &str,
        json: &str,
        read: impl FnOnce(TypedMap<'a>) -> Option<T>,
    ) -> Option<Self> {
        match batch.column_by_name(typed) {
            // A typed column of another shape reads as absent, as any column of another type does.
            Some(_) => TypedMap::new(batch, typed).and_then(read).map(StructMap::Typed),
            None => column(batch, json).map(StructMap::Json),
        }
    }
}

/// A `Map<Utf8, Struct<...>>` column (sections 1.2 and 1.3): its entries in stored order, each a
/// struct whose fields are looked up by name (D3), a null struct for an entry that was not an object.
struct TypedMap<'a> {
    map: &'a MapArray,
    keys: &'a StringArray,
    structs: &'a StructArray,
}

impl<'a> TypedMap<'a> {
    fn new(batch: &'a RecordBatch, name: &str) -> Option<Self> {
        let map = batch.column_by_name(name)?.as_any().downcast_ref::<MapArray>()?;
        Some(Self {
            map,
            keys: map.keys().as_any().downcast_ref()?,
            structs: map.values().as_any().downcast_ref()?,
        })
    }

    /// The entries of one row, as indexes into `keys` and `structs`; none for a null row.
    fn row(&self, row: usize) -> std::ops::Range<usize> {
        if self.map.is_null(row) {
            return 0..0;
        }
        let offsets = self.map.value_offsets();
        offsets[row] as usize..offsets[row + 1] as usize
    }

    /// The struct field `name`, when the part has it as Float64; a part without it reads as absent.
    fn float(&self, name: &str) -> Option<&'a Float64Array> {
        self.structs.column_by_name(name)?.as_any().downcast_ref()
    }

    /// A Float64 field's value at an entry whose struct is not null.
    fn value(field: Option<&Float64Array>, at: usize) -> Option<f64> {
        field.filter(|field| !field.is_null(at)).map(|field| field.value(at))
    }
}

/// Protocol 5's `layers`, with the struct fields the detectors read found once per batch.
struct LayersColumn<'a> {
    map: TypedMap<'a>,
    std: Option<&'a Float64Array>,
    abs_max: Option<&'a Float64Array>,
    zero_frac: Option<&'a Float64Array>,
    /// Every Float64 field, whatever its name: `non_finite` reads them all (contract 1.5).
    floats: Vec<&'a Float64Array>,
}

impl<'a> LayersColumn<'a> {
    fn new(map: TypedMap<'a>) -> Option<Self> {
        let floats = map
            .structs
            .columns()
            .iter()
            .filter_map(|field| field.as_any().downcast_ref::<Float64Array>())
            .collect();
        Some(Self {
            std: map.float("std"),
            abs_max: map.float("abs_max"),
            zero_frac: map.float("zero_frac"),
            floats,
            map,
        })
    }

    /// A row's layers as the JSON visitor gives them (`LayerStats`), read from the struct fields with
    /// no JSON. A null struct (an entry that was not an object) is absent to every detector, so the
    /// rest of the map is read (contract 1.5). A refused field is null in the struct and absent too,
    /// except to `non_finite`, which also reads the refused fields kept in `extra_json` under
    /// `layers`: parsed only on a row whose text names the key, as `refused_gradients` does. A layer
    /// whose name only `extra_json` holds (one Arrow cannot encode) is read after the column's
    /// entries, as `row_to_record` puts it back, and by `non_finite` alone.
    fn row(&self, row: usize, extra: Option<&str>) -> Vec<(Cow<'a, str>, LayerStats)> {
        let entries = self.map.row(row);
        let mut layers = Vec::with_capacity(entries.len());
        for at in entries {
            if self.map.structs.is_null(at) {
                continue;
            }
            let finite = |field| TypedMap::value(field, at).filter(|value| value.is_finite());
            layers.push((
                Cow::Borrowed(self.map.keys.value(at)),
                LayerStats {
                    std: finite(self.std),
                    abs_max: finite(self.abs_max),
                    zero_frac: finite(self.zero_frac),
                    non_finite: self
                        .floats
                        .iter()
                        .any(|field| TypedMap::value(Some(field), at).is_some_and(|value| !value.is_finite())),
                },
            ));
        }
        if let Some(Value::Object(mut extra)) = extra
            .filter(|extra| extra.contains("\"layers\""))
            .and_then(extra_json::parse)
        {
            if let Some(Value::Object(refused)) = extra.remove("layers") {
                for (name, fields) in refused {
                    // An entry that is not an object, refused whole, is absent (section 1.5).
                    let Value::Object(fields) = fields else { continue };
                    if !fields.values().any(Stat::is_non_finite_text) {
                        continue;
                    }
                    match layers.iter_mut().find(|(known, _)| *known == name) {
                        Some((_, stats)) => stats.non_finite = true,
                        None => layers.push((
                            Cow::Owned(name),
                            LayerStats {
                                non_finite: true,
                                ..LayerStats::default()
                            },
                        )),
                    }
                }
            }
        }
        layers
    }
}

/// Protocol 5's `moe_stats`, with its `diversity` field found once per batch: the only field a
/// detector reads (contract 1.1).
struct RouterStatsColumn<'a> {
    map: TypedMap<'a>,
    diversity: Option<&'a Float64Array>,
}

impl<'a> RouterStatsColumn<'a> {
    fn new(map: TypedMap<'a>) -> Option<Self> {
        Some(Self {
            diversity: map.float("diversity"),
            map,
        })
    }

    /// A row's routers with their recorded diversity: absent for a null struct, a null (refused or
    /// missing) field, a part without the field, and a value that is not finite, as `Number` reads
    /// protocol 4's text. None for a null row.
    fn row(&self, row: usize) -> Option<Vec<(Cow<'a, str>, Option<f64>)>> {
        if self.map.map.is_null(row) {
            return None;
        }
        Some(
            self.map
                .row(row)
                .map(|at| {
                    let diversity = Some(at)
                        .filter(|at| !self.map.structs.is_null(*at))
                        .and_then(|at| TypedMap::value(self.diversity, at))
                        .filter(|value| value.is_finite());
                    (Cow::Borrowed(self.map.keys.value(at)), diversity)
                })
                .collect(),
        )
    }
}

// --- MoE load: moe_imbalance and starved_experts -----------------------------------------

/// The `check` of the `not_checked` entry that names the routers recorded with no tokens on any row.
const MOE_ROUTING: &str = "moe_routing";

#[derive(Default)]
struct Router {
    name: String,
    top_k: i64,
    /// The baseline for a top-k router: its last `OVERLOAD_WINDOW` samples that were not
    /// flagged, as (counts, assignments, diversity), with the counts summed per expert. Every sum
    /// of counts is an `i128`, exact as Python's ([`counts_sum`]).
    window: VecDeque<(Arc<[i64]>, i128, f64)>,
    window_sums: Vec<i128>,
    window_total: i128,
    /// The trailing `starve_window` samples, for the starved check.
    recent: VecDeque<(Arc<[i64]>, i128)>,
    recent_sums: Vec<i128>,
    recent_tokens: i128,
    /// Where this router was last found in a row's `moe_stats`.
    stats_at: usize,
    /// The most experts any of its recorded counts had: the size of the sample it is given
    /// on a step that recorded nothing for it (Python's `_router_sizes`).
    largest: usize,
    /// Where it stands against the starting-state rule (part 3).
    leading: Leading,
    /// Whether any row so far gave it tokens (counts summing to more than 0). A router the rows name
    /// only with no tokens was not read (`moe_routing`, wave AG, section 8 of `WAVE_AF_CONTRACT.md`).
    read: bool,
}

/// A router against the starting-state rule (part 3). Only its judged samples (counts that sum
/// to more than 0) count.
#[derive(Default)]
enum Leading {
    /// No judged sample yet.
    #[default]
    Unknown,
    /// Its first judged sample was at step 0 and every one since was flagged: it is in the state
    /// it started in, `samples` of them so far, fewer than `START_GRACE`. `first` is the first
    /// sample as an incident names it, and `scalars` what was logged on that step: an incident
    /// made later is dated there.
    Started {
        samples: u32,
        first: Value,
        scalars: Option<Value>,
    },
    /// Judged as before the rule: its first sample was not at step 0, or the leading run ended,
    /// or it reached `START_GRACE`.
    Judged,
}

impl Router {
    /// Python's `_jump_overloaded`: in a router whose diversity just dropped, the experts
    /// whose fraction of tokens rose by `OVERLOAD_JUMP` over their average in the baseline.
    fn jump_overloaded(&self, counts: &[i64], tokens: i128, diversity: f64, scratch: &mut Vec<f64>) -> Vec<usize> {
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

    fn keep_in_baseline(&mut self, counts: &Arc<[i64]>, tokens: i128, diversity: f64) {
        if self.window_sums.len() < counts.len() {
            self.window_sums.resize(counts.len(), 0);
        }
        for (sum, count) in self.window_sums.iter_mut().zip(counts.iter()) {
            *sum += i128::from(*count);
        }
        self.window_total += tokens;
        self.window.push_back((Arc::clone(counts), tokens, diversity));
        if self.window.len() > OVERLOAD_WINDOW {
            let (old, old_tokens, _) = self.window.pop_front().expect("the window is not empty");
            for (sum, count) in self.window_sums.iter_mut().zip(old.iter()) {
                *sum -= i128::from(*count);
            }
            self.window_total -= old_tokens;
        }
    }

    /// Python's `_mark_starved` for one sample: the experts whose share of the trailing
    /// `window` samples is below `ratio` of an even share, with that share. None while the
    /// window is not full or holds too few tokens to judge (the sample then has no
    /// `starved_shares`), which is not the same as a full window with nobody starved.
    fn starved(
        &mut self,
        counts: &Arc<[i64]>,
        tokens: i128,
        dead: &[usize],
        window: usize,
        ratio: f64,
    ) -> Option<Vec<(usize, f64)>> {
        if self.recent_sums.len() != counts.len() {
            self.recent.clear();
            self.recent_sums = vec![0; counts.len()];
            self.recent_tokens = 0;
        }
        for (sum, count) in self.recent_sums.iter_mut().zip(counts.iter()) {
            *sum += i128::from(*count);
        }
        self.recent_tokens += tokens;
        self.recent.push_back((Arc::clone(counts), tokens));
        if self.recent.len() > window {
            let (old, old_tokens) = self.recent.pop_front().expect("the window is not empty");
            for (sum, count) in self.recent_sums.iter_mut().zip(old.iter()) {
                *sum -= i128::from(*count);
            }
            self.recent_tokens -= old_tokens;
        }
        let even = 1.0 / counts.len() as f64;
        let window_tokens = self.recent_tokens as f64;
        if self.recent.len() < window || window_tokens * even < STARVE_MIN_EXPECTED {
            return None;
        }
        Some(
            self.recent_sums
                .iter()
                .enumerate()
                .filter(|(index, total)| **total as f64 / window_tokens < ratio * even && !dead.contains(&(index + 1)))
                .map(|(index, total)| (index + 1, *total as f64 / window_tokens))
                .collect(),
        )
    }
}

/// The rule a sample's `overloaded` list was made by, as Python's `overload_rule`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "rule", rename_all = "snake_case")]
enum OverloadRule {
    /// Top-1 and top-2: an expert's share of the assignments above `threshold`.
    Share { threshold: f64 },
    /// Top-3 and higher: diversity below `drop` times the router's baseline, and an
    /// expert's fraction of tokens up by `jump`.
    DiversityDrop { drop: f64, jump: f64 },
    /// Top-3 and higher before there is a baseline, with dead experts: share above `threshold`.
    Collapse { threshold: f64 },
}

/// One router on one step as it was judged: what Python's sample loop in `build_evidence`
/// and `_mark_starved` put in `samples`. Kept only for `tensorquill-server detect --samples`.
struct Sample {
    router: usize,
    counts: Arc<[i64]>,
    /// The counts' exact sum, written as Python writes it ([`counts_sum`]).
    tokens: i128,
    /// None for counts that sum to zero: a missing sample, which no rule read.
    judged: Option<Judged>,
}

struct Judged {
    diversity: f64,
    overloaded: Vec<usize>,
    dead: Vec<usize>,
    rule: OverloadRule,
    /// The starved experts with their share of the window, or None while the window
    /// could not be judged.
    starved: Option<Vec<(usize, f64)>>,
}

/// The samples of a whole run, in the order the rows were read.
#[derive(Default)]
struct Kept {
    /// Each row that counts, as (step, where its samples start in `samples`), whether or
    /// not it recorded any router.
    rows: Vec<(i64, usize)>,
    /// The routers each row recorded, in layer order within the row.
    samples: Vec<Sample>,
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
    /// The per-step samples, when they were asked for. The server never asks: it keeps a
    /// rank's detectors for the whole run, and the samples grow with every step.
    kept: Option<Kept>,
    /// How many rows had a router with tokens to judge: the `moe` input count.
    judged_rows: u64,
    /// Routers the capture refused to count (`moe_refused`, said once on a record), each
    /// with the first step it was said on and why. They leave no entry in `moe_counts`.
    refused: BTreeMap<String, (i64, String)>,
}

impl Moe {
    fn row(&mut self, step: i64, columns: &Columns, row: usize, options: &Thresholds, scalars: &mut Scalars) {
        if let Some(kept) = &mut self.kept {
            kept.rows.push((step, kept.samples.len()));
        }
        // `moe_refused` travels in `extra_json`, which is parsed only on a row that names it.
        if let Some(extra) = text(columns.extra, row).filter(|extra| extra.contains("\"moe_refused\"")) {
            let extra: Option<Value> = extra_json::parse(extra);
            if let Some(refused) = extra
                .as_ref()
                .and_then(|extra| extra.get("moe_refused"))
                .and_then(Value::as_object)
            {
                for (name, why) in refused {
                    let why = why.as_str().map_or_else(|| why.to_string(), str::to_owned);
                    self.refused.entry(name.clone()).or_insert((step, why));
                }
            }
        }
        if let Some(top_k) = &columns.moe_top_k {
            for entry in top_k.row(row) {
                if top_k.values.is_null(entry) {
                    continue;
                }
                let name = top_k.keys.value(entry);
                // An override stands against every recorded value, on this row and later ones.
                let k = options
                    .top_k_for(name)
                    .unwrap_or_else(|| i64::from(top_k.values.value(entry)));
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
                        top_k: options.top_k_for(name).or(self.top_k_ahead.remove(name)).unwrap_or(1),
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
        let stats = columns.router_diversities(row);
        let offsets = counts.values.value_offsets();
        for (router, entry) in self.routers.iter_mut().zip(&self.present) {
            if let Some(entry) = entry {
                router.largest = router.largest.max((offsets[entry + 1] - offsets[*entry]) as usize);
            }
        }
        let mut router_counts: Vec<i64> = Vec::new();
        let mut judged = false;
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
            let tokens = counts_sum(&router_counts);
            if tokens <= 0 {
                // A missing sample: nothing is judged and no window moves.
                if let Some(kept) = &mut self.kept {
                    kept.samples.push(Sample {
                        router: index,
                        counts: Arc::from(&router_counts[..]),
                        tokens,
                        judged: None,
                    });
                }
                continue;
            }
            judged = true;
            let router = &mut self.routers[index];
            router.read = true;
            let experts = router_counts.len();
            let recorded = stats.as_ref().and_then(|stats| {
                // The statistics list the routers in the same order on every step.
                if stats.get(router.stats_at).is_none_or(|(name, _)| name != &router.name) {
                    router.stats_at = stats.iter().position(|(name, _)| name == &router.name)?;
                }
                stats[router.stats_at].1
            });
            let diversity = recorded.unwrap_or_else(|| diversity_from_counts(&router_counts));
            let share = |count: &i64| *count as f64 / tokens as f64;
            let mut rule = OverloadRule::DiversityDrop {
                drop: DIVERSITY_DROP,
                jump: OVERLOAD_JUMP,
            };
            let mut overloaded: Vec<usize> = if router.top_k >= 3 {
                router.jump_overloaded(&router_counts, tokens, diversity, &mut self.scratch)
            } else {
                let threshold = effective_overload_threshold(experts, tokens, options.overload_threshold);
                rule = OverloadRule::Share { threshold };
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
                let mut threshold = (2.0 * mean).max(mean + 3.0 * se);
                let most = 1.0 / router.top_k as f64;
                if threshold >= most {
                    // A top-k expert's share cannot pass 1/k, so with n_experts <= 2k that bound
                    // names nothing (DET-11): the midpoint between an even share and the largest.
                    threshold = (mean + most) / 2.0;
                }
                rule = OverloadRule::Collapse { threshold };
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
            let sample = || {
                json!({
                    "kind": "moe_imbalance",
                    "step": step,
                    "layer": index + 1,
                    "router": router.name,
                    "diversity": diversity,
                    "overloaded": overloaded,
                    "dead": dead,
                })
            };
            // The starting-state rule (part 3): a router flagged from its first sample at step 0
            // is reported once that leading run reaches `START_GRACE` samples, dated at its first
            // sample; a flagged sample after the run has ended is reported as before. Routers are
            // read in layer order, so of two that become reportable on one row the lower layer is
            // the incident, and the first row with any is the one that could be reported first.
            let reportable = match &mut router.leading {
                Leading::Unknown if flagged && step == 0 => {
                    router.leading = Leading::Started {
                        samples: 1,
                        first: sample(),
                        scalars: if self.first.is_none() {
                            scalars.context(columns, row)
                        } else {
                            None
                        },
                    };
                    None
                }
                Leading::Started { samples, .. } if flagged => {
                    *samples += 1;
                    if *samples < START_GRACE {
                        None
                    } else {
                        let Leading::Started { first, scalars, .. } = std::mem::take(&mut router.leading) else {
                            unreachable!("the router is in its starting state")
                        };
                        router.leading = Leading::Judged;
                        let mut first = first;
                        first["since_start"] = json!(true);
                        first["reportable_step"] = json!(step);
                        if let Some(scalars) = scalars {
                            first["scalars"] = scalars;
                        }
                        Some(first)
                    }
                }
                Leading::Unknown | Leading::Started { .. } => {
                    router.leading = Leading::Judged;
                    flagged.then(sample)
                }
                Leading::Judged => flagged.then(sample),
            };
            if self.first.is_none() {
                self.first = reportable;
            }
            let starved = router.starved(&shared, tokens, &dead, options.starve_window, options.starve_ratio);
            for (expert, window_share) in starved.iter().flatten() {
                self.starved
                    .entry((index + 1, *expert))
                    .or_insert((step, *window_share));
            }
            if let Some(kept) = &mut self.kept {
                kept.samples.push(Sample {
                    router: index,
                    counts: shared,
                    tokens,
                    judged: Some(Judged {
                        diversity,
                        overloaded,
                        dead,
                        rule,
                        starved,
                    }),
                });
            }
        }
        self.judged_rows += u64::from(judged);
    }

    /// The `router` entry of `not_checked`: Python's `refused_routers_not_checked`. A run
    /// whose only router was refused has no MoE incident, and that is not "MoE checked".
    /// The capture's text is printed as written: why, and the fixes for how that router was
    /// named, so the server, the Python evidence and the session's warning say the same.
    fn refused_not_checked(&self) -> Option<Value> {
        let step = self.refused.values().map(|(step, _)| *step).min()?;
        let reason: Vec<String> = self
            .refused
            .iter()
            .map(|(name, (_, why))| format!("router {name} is not counted: {why}"))
            .collect();
        Some(json!({
            "check": "router",
            "kinds": ["moe_imbalance", "starved_experts"],
            "routers": self.refused.keys().collect::<Vec<_>>(),
            "reason": reason.join("; "),
            "step": step,
        }))
    }

    /// The `moe_routing` entry of `not_checked` (wave AG, section 8 of `WAVE_AF_CONTRACT.md`): the
    /// routers the rows name whose counts are 0 on every row that names them, sorted by name, or
    /// None when every router named was read. A model trained in eval mode with reentrant
    /// checkpointing records such zeros on every step; they are not a router that routed nothing.
    /// The shape and the text are the Python readers' (lane C). [`RankDetectors::kind_checks`] takes
    /// it for the two kinds only while no router was read; otherwise they are judged on the routers
    /// that were, and the entry names the others. Over a live rank's rows it is judged on what has
    /// arrived: it appears while a named router has had only zeros, and goes at its first tokens.
    fn not_read_not_checked(&self) -> Option<Value> {
        let mut routers: Vec<&str> = self
            .routers
            .iter()
            .filter(|router| !router.read)
            .map(|router| router.name.as_str())
            .collect();
        if routers.is_empty() {
            return None;
        }
        routers.sort_unstable();
        let reason: Vec<String> = routers
            .iter()
            .map(|name| format!("router {name} recorded no tokens on any step: its routing was not read"))
            .collect();
        Some(json!({
            "check": MOE_ROUTING,
            "kinds": ["moe_imbalance", "starved_experts"],
            "routers": routers,
            "reason": reason.join("; "),
        }))
    }

    /// The routers still in the state they started in when the rows end (part 3): flagged on
    /// every judged sample since step 0, and fewer than `START_GRACE` of them. Nothing is reported
    /// for them, and the kind is not checked with enough history.
    fn starting_state(&self) -> Vec<Value> {
        self.routers
            .iter()
            .filter_map(|router| match &router.leading {
                Leading::Started { samples, .. } => Some(json!({"router": router.name, "samples": samples})),
                _ => None,
            })
            .collect()
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

/// The per-step MoE samples of a run, as `build_evidence` lists them under `samples`: every
/// row that counts, in step order, times every router the run recorded, in layer order.
/// A router that a step recorded nothing for, or only zeros, has a sample marked `missing`.
///
/// Python builds these with the whole run in hand. Here the judged samples are kept as the
/// rows are read and the rest is filled in when the run is written out: which routers
/// exist and how many experts each has is only known at the end.
pub struct MoeSamples<'a> {
    moe: &'a Moe,
    kept: &'a Kept,
    /// The first `from_rows` rows, in step order, are left out: `detect --follow` writes only the
    /// samples of the rows read since its last answer (0 everywhere else).
    from_rows: usize,
}

#[derive(Serialize)]
struct SampleOut<'a> {
    step: i64,
    layer: usize,
    router: &'a str,
    routed_tokens: i128,
    counts: &'a [i64],
    diversity: Option<f64>,
    overloaded: &'a [usize],
    dead: &'a [usize],
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    missing: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    overload_rule: Option<OverloadRule>,
    starved: Experts<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    starved_shares: Option<Shares<'a>>,
}

/// The starved experts of a sample, as a list of expert numbers.
struct Experts<'a>(&'a [(usize, f64)]);

impl Serialize for Experts<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().map(|(expert, _)| expert))
    }
}

/// The same experts as an object of expert number to its share of the window.
struct Shares<'a>(&'a [(usize, f64)]);

impl Serialize for Shares<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.iter().map(|(expert, share)| (expert.to_string(), share)))
    }
}

impl MoeSamples<'_> {
    /// How many samples there are: rows that count times routers.
    pub fn len(&self) -> usize {
        self.kept.rows.len().saturating_sub(self.from_rows) * self.moe.routers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Serialize for MoeSamples<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let (routers, rows, samples) = (&self.moe.routers, &self.kept.rows, &self.kept.samples);
        let zeros = vec![0_i64; routers.iter().map(|router| router.largest).max().unwrap_or(0)];
        // Python sorts the records by step, and a stable sort keeps a step written twice in
        // the order it was written.
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|row| rows[*row].0);
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for row in order.into_iter().skip(self.from_rows) {
            let (step, start) = rows[row];
            let end = rows.get(row + 1).map_or(samples.len(), |next| next.1);
            let mut recorded = samples[start..end].iter().peekable();
            for (index, router) in routers.iter().enumerate() {
                let sample = recorded.next_if(|sample| sample.router == index);
                let judged = sample.and_then(|sample| sample.judged.as_ref());
                let starved = judged.and_then(|judged| judged.starved.as_deref());
                seq.serialize_element(&SampleOut {
                    step,
                    layer: index + 1,
                    router: &router.name,
                    routed_tokens: sample.map_or(0, |sample| sample.tokens),
                    // A router the step did not record gets zeros for its largest size.
                    counts: sample.map_or(&zeros[..router.largest], |sample| &sample.counts),
                    diversity: judged.map(|judged| judged.diversity),
                    overloaded: judged.map_or(&[], |judged| &judged.overloaded),
                    dead: judged.map_or(&[], |judged| &judged.dead),
                    missing: judged.is_none(),
                    overload_rule: judged.map(|judged| judged.rule),
                    starved: Experts(starved.unwrap_or(&[])),
                    starved_shares: starved.map(Shares),
                })?;
            }
        }
        seq.end()
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
    /// The scores beside the parameter's ratio and the total's (part 1), taken then for the same
    /// reason, over the windows the two medians came from. None when no incident can follow.
    score: Option<Score>,
    total_score: Option<Score>,
}

impl Start {
    fn parameter(&self) -> Value {
        let mut parameter = json!({
            "name": self.name,
            "grad_norm": self.norm,
            "baseline": self.usual,
            "ratio": self.norm / self.usual,
            "rising_since": self.step,
        });
        if let Some(score) = self.score {
            score.put(&mut parameter);
        }
        parameter
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
                    let mut first = json!({
                        "kind": "grad_spike",
                        "step": step,
                        "grad_norm": total,
                        "baseline": baseline,
                        "ratio": total / baseline,
                        "parameters": self.spiking_parameters(step, columns, row),
                    });
                    Score::of(total, self.totals.iter().copied(), &mut self.scratch).put(&mut first);
                    self.first = Some(first);
                }
            }
        }
        // The total joins the window after this row's parameters are read: an excursion that
        // begins here takes its score over the totals the baseline was taken over.
        self.parameters_row(step, total, total_baseline, columns, row, scalars);
        self.seen += 1;
        self.totals.push_back(total);
        if self.totals.len() > SPIKE_WINDOW {
            self.totals.pop_front();
        }
    }

    /// One parameter on its own (Python's `_Baseline` path): the rest of [`row`](Self::row),
    /// before the row's total joins the window.
    fn parameters_row(
        &mut self,
        step: i64,
        total: f64,
        total_baseline: Option<f64>,
        columns: &Columns,
        row: usize,
        scalars: &mut Scalars,
    ) {
        let Some(norms) = &columns.grad_norms else { return };
        let entries = norms.row(row);
        let first_entry = entries.start;
        // The parameters whose own norm has now stayed high for `SPIKE_SUSTAIN` recorded steps.
        let mut lasted: Vec<Start> = Vec::new();
        // This row's logged values, read once if an excursion begins here and no incident exists,
        // and the total's score, worked out once for the same.
        let mut logged: Option<Option<Value>> = None;
        let mut total_score: Option<Score> = None;
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
                        // The incident is made already: no later excursion is reported.
                        let open = self.first.is_none();
                        let scalars = if open {
                            logged.get_or_insert_with(|| scalars.context(columns, row)).clone()
                        } else {
                            None
                        };
                        let score = open.then(|| Score::of(norm, parameter.usual.iter().copied(), &mut self.scratch));
                        let total_score = open.then(|| {
                            *total_score
                                .get_or_insert_with(|| Score::of(total, self.totals.iter().copied(), &mut self.scratch))
                        });
                        parameter.start = Some(Start {
                            step,
                            total,
                            total_baseline,
                            name: name.to_owned(),
                            norm,
                            usual,
                            scalars,
                            score,
                            total_score: total_score.filter(|_| total_baseline.is_some()),
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
        // The score is beside the ratio: there is none without one.
        if let Some(score) = start.total_score.filter(|_| ratio.is_some()) {
            score.put(&mut first);
        }
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
            let mut parameter =
                json!({"name": name, "grad_norm": norm, "baseline": baseline, "ratio": ratio, "rising_since": rising});
            Score::of(norm, past.iter().map(|(_, norm)| *norm), &mut self.scratch).put(&mut parameter);
            found.push((rising, ratio, parameter));
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

// --- update_spike --------------------------------------------------------------------------

/// The last row that could be judged: what the next one is compared with.
struct UpdateRow {
    attempt: i64,
    largest: f64,
    total: f64,
    own: Vec<(String, f64)>,
    /// Each parameter's gradient norm (`grad_norms`; a null one as NaN), or None with none.
    grads: Option<Vec<(String, f64)>>,
    /// The row's learning rates, as `_lr_values` reads them: an incident's `lr_before` (part 4).
    lr: Option<Vec<f64>>,
    /// The row's `clip_calls` is above 1: more than one `clip_grad_norm_` call clipped the step
    /// (`WAVE_AE_CONTRACT.md`, section 1), so its pre-clipping total cannot be split by parameter.
    clip_calls: bool,
}

/// A row's `clip_calls` (`WAVE_AE_CONTRACT.md`, section 1): from `extra_json`, parsed only on a row
/// that names it, as `step_skipped_by` is. True only for a whole number above 1; anything else
/// (1, text, a fraction, a boolean, `2.0`) says nothing, and an absent key means one call or a
/// count that is not known.
fn clip_calls(columns: &Columns, row: usize) -> bool {
    text(columns.extra, row)
        .filter(|extra| extra.contains("\"clip_calls\""))
        .and_then(extra_json::parse)
        .and_then(|extra| extra.get("clip_calls").and_then(Value::as_u64))
        .is_some_and(|calls| calls > 1)
}

/// Python's `detect_update_spikes` and `update_not_read_not_checked`.
#[derive(Default)]
struct Updates {
    /// Whether any record so far had `clip_grad_norm`; the total is read from it from then on.
    clips: bool,
    before: Option<UpdateRow>,
    /// The last `update_window` changes that were not flagged.
    recent: VecDeque<f64>,
    first: Option<Value>,
    spikes: u64,
    spike_steps: Vec<i64>,
    /// How many rows carried `update_rms` and a gradient total: the `update` input count.
    carried: u64,
    /// Optimizers whose update size the capture could not read (`update_not_read`, said once
    /// on a record), each with the first step it was said on and why.
    not_read: BTreeMap<String, (i64, String)>,
    /// The pairs not compared because their totals would be split while either row was clipped
    /// by more than one call, and the step of the later row of the first (section 1).
    clip_groups: u64,
    clip_groups_step: Option<i64>,
    /// The pairs that were compared, the `clip_groups` entry's `compared`: at zero, while some
    /// pair was not compared, `update_spike` reads `unavailable` (section 1, the author's
    /// second word).
    compared: u64,
}

impl Updates {
    fn row(&mut self, attempt: i64, step: i64, columns: &Columns, row: usize, window: usize) {
        // `update_not_read` travels in `extra_json`, which is parsed only on a row that names it.
        if let Some(extra) = text(columns.extra, row).filter(|extra| extra.contains("\"update_not_read\"")) {
            let extra: Option<Value> = extra_json::parse(extra);
            if let Some(said) = extra
                .as_ref()
                .and_then(|extra| extra.get("update_not_read"))
                .and_then(Value::as_object)
            {
                for (name, why) in said {
                    let why = why.as_str().map_or_else(|| why.to_string(), str::to_owned);
                    self.not_read.entry(name.clone()).or_insert((step, why));
                }
            }
        }
        if !self.clips && value(columns.clip_grad_norm, row).is_some() {
            // As for `grad_spike`: the pre-clipping norm is the total for the whole run once
            // any record has it, and the records before clipping began have none.
            self.clips = true;
            self.before = None;
            self.recent.clear();
            self.first = None;
            self.spikes = 0;
            self.spike_steps.clear();
            // Python reads every row's total from `clip_grad_norm` then: a pair judged before
            // has no total there, so it was not one that went uncompared either.
            self.clip_groups = 0;
            self.clip_groups_step = None;
            self.compared = 0;
        }
        let Some(sizes) = &columns.update_rms else { return };
        let entries = sizes.row(row);
        let either = value(columns.grad_norm, row).is_some() || value(columns.clip_grad_norm, row).is_some();
        // A step the loss scaler skipped is no update history (RUST-12), as below.
        let skipped = flag(columns.step_skipped, row);
        self.carried += u64::from(either && !entries.is_empty() && !skipped);
        if skipped {
            return; // the loss scaler skipped the step: the optimizer's state did not move
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
        let own: Vec<(String, f64)> = entries
            .filter(|entry| !sizes.values.is_null(*entry) && sizes.values.value(*entry).is_finite())
            .map(|entry| (sizes.keys.value(entry).to_owned(), sizes.values.value(entry)))
            .collect();
        if own.is_empty() {
            return;
        }
        let largest = own.iter().map(|(_, size)| *size).fold(f64::NEG_INFINITY, f64::max);
        // R on both sides is the largest update over the parameters both rows name (L13, Nov 20,
        // 2026), as in Python: a parameter recorded from a later step on, or one left out of this
        // row, is not a jump. Rows that share no parameter are not compared, and add no change to
        // the baseline. The gradient total stays the whole row's.
        let compared = self
            .before
            .as_ref()
            .filter(|before| before.attempt == attempt)
            .and_then(|before| shared_largest(&own, largest, before).map(|sides| (before, sides)));
        // A row whose `grad_norms` refused an entry has a typed map that is not the whole record's:
        // its norms are not used to split, so a comparison with it on either side takes the whole
        // totals (the author, Nov 22). A refused entry stored as null ends the split already (a
        // norm that is not finite); one only in `extra_json` (a name Arrow cannot encode) ends it here.
        let refused = refused_gradients(columns, row).is_some();
        let grads: Option<Vec<(String, f64)>> = columns.grad_norms.as_ref().filter(|_| !refused).and_then(|norms| {
            let entries = norms.row(row);
            (!entries.is_empty()).then(|| {
                entries
                    .map(|entry| {
                        let norm = if norms.values.is_null(entry) {
                            f64::NAN
                        } else {
                            norms.values.value(entry)
                        };
                        (norms.keys.value(entry).to_owned(), norm)
                    })
                    .collect()
            })
        });
        let clipped = clip_calls(columns, row);
        if let Some((before, (largest, was))) = compared {
            let split = shared_totals(total, grads.as_deref(), before.total, before.grads.as_deref());
            let (now, then) = split.unwrap_or((total, before.total));
            let (change, rise) = ((largest - was).abs(), (now - then).max(0.0));
            let suppressed = split.is_some() && (clipped || before.clip_calls);
            self.compared += u64::from(!suppressed);
            if suppressed {
                // The split assumes one call scaled every gradient by one factor (section 1): the
                // pair is not compared, nothing is flagged and its change joins no baseline. This
                // row is still the one the next is compared with.
                self.clip_groups += 1;
                self.clip_groups_step.get_or_insert(step);
            } else if self.recent.len() < window {
                self.recent.push_back(change);
            } else {
                // The mean of the last `window` unflagged changes, summed as Python's `fsum` sums.
                let baseline = fsum(self.recent.iter().copied()) / window as f64;
                let threshold = (rise > 0.0 && baseline > 0.0).then(|| baseline / (UPDATE_ALPHA * rise.sqrt()));
                if let Some(threshold) = threshold.filter(|threshold| change > *threshold) {
                    self.spikes += 1;
                    if self.spike_steps.len() < 20 {
                        self.spike_steps.push(step);
                    }
                    if self.first.is_none() {
                        let mut moved: Vec<(f64, &str, f64, f64)> = own
                            .iter()
                            .enumerate()
                            .filter_map(|(at, (name, size))| {
                                let was = match before.own.get(at) {
                                    Some((known, was)) if known == name => Some(*was),
                                    _ => before.own.iter().find(|(known, _)| known == name).map(|(_, was)| *was),
                                }?;
                                Some(((size - was).abs(), name.as_str(), *size, was))
                            })
                            .collect();
                        moved.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
                        let parameters: Vec<Value> = moved
                            .iter()
                            .take(10)
                            .map(|(change, name, size, was)| {
                                json!({"name": name, "update_rms": size, "previous": was, "change": change})
                            })
                            .collect();
                        let mut first = json!({
                            "kind": "update_spike",
                            "step": step,
                            "update_rms": largest,
                            "previous": was,
                            "change": change,
                            "baseline": baseline,
                            "threshold": threshold,
                            "grad_norm": now,
                            "grad_norm_before": then,
                            "grad_rise": rise,
                            "parameters": parameters,
                        });
                        // The learning rate at the flagged step and at the step it was compared
                        // with, each only when that record carries one (part 4). No rule reads it.
                        if let Some(lr) = columns.lr(row) {
                            first["lr"] = json!(lr);
                        }
                        if let Some(lr) = &before.lr {
                            first["lr_before"] = json!(lr);
                        }
                        self.first = Some(first);
                    }
                } else {
                    // Every change that is not flagged joins the baseline, whether or not the
                    // gradient norm rose; a flagged one never does.
                    self.recent.pop_front();
                    self.recent.push_back(change);
                }
            }
        }
        self.before = Some(UpdateRow {
            attempt,
            largest,
            total,
            own,
            grads,
            lr: columns.lr(row),
            clip_calls: clipped,
        });
    }

    fn incidents(&self, out: &mut Vec<Value>) {
        if let Some(mut first) = self.first.clone() {
            first["spike_count"] = json!(self.spikes);
            first["spike_steps"] = json!(self.spike_steps);
            out.push(first);
        }
    }

    /// The `update_size` entry of `not_checked`: Python's `update_not_read_not_checked`.
    fn not_read_not_checked(&self) -> Option<Value> {
        let step = self.not_read.values().map(|(step, _)| *step).min()?;
        let reason: Vec<String> = self
            .not_read
            .iter()
            .map(|(name, (_, why))| format!("{name}: {why}"))
            .collect();
        Some(json!({
            "check": "update_size",
            "kinds": ["update_spike"],
            "optimizers": self.not_read.keys().collect::<Vec<_>>(),
            "reason": reason.join("; "),
            "step": step,
        }))
    }

    /// The `clip_groups` entry of `not_checked` (`WAVE_AE_CONTRACT.md`, section 1): the pairs
    /// not compared because more than one clip call clipped either row, dated at the later row
    /// of the first, with `compared`, the pairs that were. While `compared` is above zero
    /// `update_spike` stays `checked`; at zero it reads `unavailable`, and the reason says that
    /// none of the comparisons was made (the author: "if every eligible comparison was
    /// suppressed, the reader must make clear that zero were judged").
    fn clip_groups_not_checked(&self) -> Option<Value> {
        let step = self.clip_groups_step?;
        let n = self.clip_groups;
        let mut reason = format!(
            "clipped by more than one clip_grad_norm_ call: the pre-clipping total cannot be split by parameter, \
             so {n} comparison(s) of update_spike were not made"
        );
        if self.compared == 0 {
            reason.push_str("; none of the comparisons of update_spike was made, so nothing was judged");
        }
        Some(json!({
            "check": CLIP_GROUPS,
            "kinds": ["update_spike"],
            "comparisons": n,
            "compared": self.compared,
            "step": step,
            "reason": reason,
        }))
    }
}

/// The largest update over the parameters both `own` and `before` name, on this row's side and
/// on the row before's, or None when they share none. `largest` is the largest over all of `own`:
/// rows that name the same parameters in the same order (every row but a few) are not searched.
fn shared_largest(own: &[(String, f64)], largest: f64, before: &UpdateRow) -> Option<(f64, f64)> {
    if own.len() == before.own.len() && own.iter().zip(&before.own).all(|((a, _), (b, _))| a == b) {
        return Some((largest, before.largest));
    }
    let mut sides: Option<(f64, f64)> = None;
    for (at, (name, size)) in own.iter().enumerate() {
        let was = match before.own.get(at) {
            Some((known, was)) if known == name => Some(*was),
            _ => before.own.iter().find(|(known, _)| known == name).map(|(_, was)| *was),
        };
        if let Some(was) = was {
            sides = Some(sides.map_or((*size, was), |(now, then)| (now.max(*size), then.max(was))));
        }
    }
    sides
}

/// The gradient totals of two rows compared by `update_spike`, over the parameters both name:
/// Python's `_shared_totals` (L13), each row's total times `sqrt(shared² / all²)` over its
/// `grad_norms`. None when the whole totals are used instead: both rows name the same
/// parameters, or they cannot be split (a row without `grad_norms`, a norm that is not finite, no
/// shared parameter, or norms that are all 0). Only a pair this splits is one that more than one
/// clip call keeps from being compared (`WAVE_AE_CONTRACT.md`, section 1). Sums as Python's
/// `fsum` sums them.
fn shared_totals(
    total: f64,
    grads: Option<&[(String, f64)]>,
    before: f64,
    grads_before: Option<&[(String, f64)]>,
) -> Option<(f64, f64)> {
    let (now, then) = (grads?, grads_before?);
    if now.len() == then.len() && now.iter().zip(then).all(|((a, _), (b, _))| a == b) {
        return None; // the usual row: the same parameters in the same order
    }
    let known: HashMap<&str, f64> = then.iter().map(|(name, norm)| (name.as_str(), *norm)).collect();
    let shared: Vec<(f64, f64)> = now
        .iter()
        .filter_map(|(name, norm)| known.get(name.as_str()).map(|was| (*norm, *was)))
        .collect();
    let named_alike = shared.len() == now.len() && known.len() == now.len();
    if named_alike || shared.is_empty() || now.iter().chain(then).any(|(_, norm)| !norm.is_finite()) {
        return None;
    }
    let (all_now, all_then) = (
        fsum(now.iter().map(|(_, v)| v * v)),
        fsum(then.iter().map(|(_, v)| v * v)),
    );
    if all_now <= 0.0 || all_then <= 0.0 {
        return None;
    }
    let part_now = fsum(shared.iter().map(|(v, _)| v * v));
    let part_then = fsum(shared.iter().map(|(_, v)| v * v));
    Some((
        total * (part_now / all_now).sqrt(),
        before * (part_then / all_then).sqrt(),
    ))
}

// --- head_trends ---------------------------------------------------------------------------

/// The numbers of the probe's language-model head entry that `head_trends` follows, in Python's
/// order (`HEAD_NUMBERS` in evidence.py).
const HEAD_NUMBERS: [&str; 5] = ["norm", "mean", "log_z", "gap", "gap_relative"];

/// One head number over the probes that read it: the first and latest value, the lowest and
/// highest, and how many probes had it finite.
struct HeadNumber {
    first: (i64, f64),
    latest: (i64, f64),
    lowest: f64,
    highest: f64,
    probes: u64,
}

/// Python's `head_logit_trends`, made row by row (L15, Nov 20, 2026): the probe's head numbers
/// over the run, for the report's `head_trends`, so evidence built from a server's report has
/// them without reading the parts. Numbers only, with no rule and no incident. Rows come in
/// step order; of rows of one step, the first is the earliest and the last the latest, as
/// Python's stable sort gives them.
#[derive(Default)]
struct HeadTrends {
    /// Probes with a `head` entry, read or not, and those that read it.
    probes: u64,
    read: u64,
    /// The latest read probe's `module`, and the first and latest read step.
    module: Value,
    first_step: Option<i64>,
    latest_step: Option<i64>,
    numbers: [Option<HeadNumber>; 5],
    /// Probes that did not read the head, with the first one's step and reason.
    not_read: u64,
    first_not_read: Option<(i64, String)>,
    /// Probes that counted overflowing positions: how many, the positions in all, the most in
    /// one, and the first and latest step of one.
    overflowed: u64,
    positions: u64,
    most: u64,
    overflow_steps: Option<(i64, i64)>,
}

impl HeadTrends {
    fn row(&mut self, step: i64, columns: &Columns, row: usize) {
        // Probes are sparse, and only one with a head entry is parsed.
        let Some(text) = text(columns.probe, row).filter(|text| text.contains("\"head\"")) else {
            return;
        };
        // An integer too large for a float is unreadable, the rest kept (AQ5-D2).
        let Some(Value::Object(probe)) = extra_json::parse(text) else {
            return;
        };
        if probe.contains_key("error") {
            return;
        }
        let Some(Value::Object(head)) = probe.get("head") else {
            return;
        };
        self.probes += 1;
        // A count from 1 to the largest `i64`, as Python's `head_logit_trends` reads one (Nov 27,
        // 2026: the author bounded Python at int64; Rust is brought to the same bound, the
        // coordinator's choice). Until then any `u64` was read here. Each count fits an `i64`, so
        // two sum exactly; the sum saturates only past `u64::MAX`, where Python's stays exact.
        if let Some(count) = head
            .get("overflow_positions")
            .and_then(Value::as_u64)
            .filter(|count| (1..=i64::MAX.unsigned_abs()).contains(count))
        {
            self.overflowed += 1;
            self.positions = self.positions.saturating_add(count);
            self.most = self.most.max(count);
            self.overflow_steps = Some(self.overflow_steps.map_or((step, step), |(first, latest)| {
                (first.min(step), if step >= latest { step } else { latest })
            }));
        }
        if let Some(why) = head.get("not_read") {
            self.not_read += 1;
            if self.first_not_read.as_ref().is_none_or(|(first, _)| step < *first) {
                self.first_not_read = Some((step, python_str(why)));
            }
            return;
        }
        self.read += 1;
        if self.first_step.is_none_or(|first| step < first) {
            self.first_step = Some(step);
        }
        if self.latest_step.is_none_or(|latest| step >= latest) {
            self.latest_step = Some(step);
            self.module = head.get("module").cloned().unwrap_or(Value::Null);
        }
        for (name, kept) in HEAD_NUMBERS.iter().zip(self.numbers.iter_mut()) {
            // A finite JSON number; a bool is not one (Python's `_finite`).
            let Some(value) = head
                .get(*name)
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite())
            else {
                continue;
            };
            match kept {
                None => {
                    *kept = Some(HeadNumber {
                        first: (step, value),
                        latest: (step, value),
                        lowest: value,
                        highest: value,
                        probes: 1,
                    })
                }
                Some(number) => {
                    if step < number.first.0 {
                        number.first = (step, value);
                    }
                    if step >= number.latest.0 {
                        number.latest = (step, value);
                    }
                    number.lowest = number.lowest.min(value);
                    number.highest = number.highest.max(value);
                    number.probes += 1;
                }
            }
        }
    }

    /// The `head_trends` of Python's evidence, or null when no probe had a head entry.
    fn value(&self) -> Value {
        if self.probes == 0 {
            return Value::Null;
        }
        let mut trends = json!({"probes": self.probes, "read": self.read});
        if self.read > 0 {
            let numbers: serde_json::Map<String, Value> = HEAD_NUMBERS
                .iter()
                .zip(&self.numbers)
                .filter_map(|(name, number)| {
                    let number = number.as_ref()?;
                    Some((
                        (*name).to_owned(),
                        json!({
                            "first": number.first.1,
                            "latest": number.latest.1,
                            "lowest": number.lowest,
                            "highest": number.highest,
                            "probes": number.probes,
                        }),
                    ))
                })
                .collect();
            trends["module"] = self.module.clone();
            trends["first_step"] = json!(self.first_step);
            trends["latest_step"] = json!(self.latest_step);
            trends["numbers"] = Value::Object(numbers);
        }
        if let Some((step, reason)) = &self.first_not_read {
            trends["not_read"] = json!({"probes": self.not_read, "first_step": step, "reason": reason});
        }
        if let Some((first, latest)) = self.overflow_steps {
            trends["overflow_positions"] = json!({
                "probes": self.overflowed,
                "positions": self.positions,
                "most": self.most,
                "first_step": first,
                "latest_step": latest,
            });
        }
        trends
    }
}

/// Python's `str()` of a JSON value, for a reason the capture wrote (a string, which it always
/// writes, is itself). Of the others, null, a bool and a number read as Python prints them, and
/// an array or object as its JSON text, where Python would print its own form.
fn python_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        other => other.to_string(),
    }
}

// --- activation_spike, activation_drift, dead_activations --------------------------------

#[derive(Default)]
struct Module {
    seen: usize,
    /// The last `ACT_WINDOW + ACT_RECENT` records, as (std, abs_max, zero_frac): the trailing
    /// baseline of `activation_drift` and the last `ACT_RECENT`. `activation_spike` reads the
    /// last `ACT_WINDOW` of them.
    recent: VecDeque<(f64, f64, f64)>,
    /// The median zero fraction of the first `ACT_WINDOW` records: `dead_activations`' baseline.
    baseline_zero: Option<f64>,
}

impl Module {
    /// The `ACT_WINDOW` records before this one, of the module: `activation_spike`'s window.
    fn spike_window(&self) -> impl Iterator<Item = &(f64, f64, f64)> {
        self.recent.iter().skip(self.recent.len().saturating_sub(ACT_WINDOW))
    }
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

/// `activation_drift` on one module from its `ACT_DRIFT_NEEDS`th record (part 2): the median of
/// its last `ACT_RECENT` records against the median of the `ACT_WINDOW` before those, for `std`
/// and then for `abs_max / std`, at `ACT_DRIFT_FACTOR` times up or down. `recent` holds exactly
/// those `ACT_WINDOW + ACT_RECENT` records, the newest last.
fn drift(name: &str, recent: &VecDeque<(f64, f64, f64)>, scratch: &mut Vec<f64>) -> Option<Value> {
    let trail = || recent.iter().take(ACT_WINDOW);
    let last = || recent.iter().skip(ACT_WINDOW);
    let ratio = |past: &(f64, f64, f64)| (past.0 > 0.0).then(|| past.1 / past.0);
    for stat in ["std", "abs_max/std"] {
        let ((low_base, high_base), (low_now, high_now)) = if stat == "std" {
            (bounds(trail().map(|past| past.0)), bounds(last().map(|past| past.0)))
        } else {
            (bounds(trail().filter_map(ratio)), bounds(last().filter_map(ratio)))
        };
        // A median lies between the lowest and highest of its values. When no recent value is
        // ACT_DRIFT_FACTOR from any baseline value the medians are not either, and are not worked
        // out. With no ratio on either side (no positive `std`) a median is 0 and nothing is judged.
        if low_base == f64::INFINITY || low_now == f64::INFINITY {
            continue;
        }
        if low_base > 0.0
            && low_now > 0.0
            && high_now / low_base < ACT_DRIFT_FACTOR
            && high_base / low_now < ACT_DRIFT_FACTOR
        {
            continue;
        }
        let (base, now) = if stat == "std" {
            scratch.clear();
            scratch.extend(trail().map(|past| past.0));
            let base = median(scratch);
            scratch.clear();
            scratch.extend(last().map(|past| past.0));
            (base, median(scratch))
        } else {
            (outlier(trail(), scratch), outlier(last(), scratch))
        };
        if base > 0.0 && now > 0.0 && (now / base).max(base / now) >= ACT_DRIFT_FACTOR {
            let mut hit = json!({"module": name, "stat": stat, "baseline": base, "recent": now, "ratio": now / base});
            if stat == "std" {
                Score::of(now, trail().map(|past| past.0), scratch).put(&mut hit);
            } else {
                Score::of(now, trail().filter_map(ratio), scratch).put(&mut hit);
            }
            return Some(hit);
        }
    }
    None
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
                    let of = |past: &(f64, f64, f64)| if pick == 1 { past.1 } else { past.0 };
                    // The median is at least the lowest value: below the factor times
                    // that, nothing can be a spike and the median is not needed.
                    let (lowest, _) = bounds(module.spike_window().map(of));
                    if now < ACT_SPIKE_FACTOR * lowest {
                        continue;
                    }
                    scratch.clear();
                    scratch.extend(module.spike_window().map(of));
                    let base = median(scratch);
                    if base > 0.0 && now >= ACT_SPIKE_FACTOR * base {
                        let ratio = now / base;
                        let mut hit =
                            json!({"module": name, "stat": stat, "value": now, "baseline": base, "ratio": ratio});
                        Score::of(now, module.spike_window().map(of), scratch).put(&mut hit);
                        spikes.push((ratio, hit));
                        break;
                    }
                }
            }
            module.seen += 1;
            module.recent.push_back((std, abs_max, zero_frac));
            if module.recent.len() > ACT_WINDOW + ACT_RECENT {
                module.recent.pop_front();
            }
            if module.seen == ACT_WINDOW {
                // The first ACT_WINDOW records never change again: summarize them once.
                scratch.clear();
                scratch.extend(module.recent.iter().map(|past| past.2));
                module.baseline_zero = Some(median(scratch));
            }
            if module.seen >= ACT_DRIFT_NEEDS {
                if let Some(hit) = drift(name, &module.recent, scratch) {
                    drifts.push(hit);
                }
            }
            if module.seen < ACT_WINDOW + ACT_RECENT {
                continue;
            }
            let Some(base_zero) = module.baseline_zero else {
                continue;
            };
            let last = || module.recent.iter().skip(module.recent.len() - ACT_RECENT);
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

// --- attention_collapse, attention_uniform, attention_logit_growth -------------------------

/// Numbers in a JSON list as `_finite` reads them; anything but a list reads as no numbers.
#[derive(Default)]
struct Numbers(Vec<Option<f64>>);

impl<'de> Deserialize<'de> for Numbers {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct NumbersVisitor;
        impl<'de> Visitor<'de> for NumbersVisitor {
            type Value = Numbers;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any value")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Numbers, A::Error> {
                let mut out = Vec::new();
                while let Some(Number(number)) = seq.next_element()? {
                    out.push(number);
                }
                Ok(Numbers(out))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Numbers, A::Error> {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(Numbers::default())
            }
            fn visit_f64<E>(self, _: f64) -> Result<Numbers, E> {
                Ok(Numbers::default())
            }
            fn visit_i64<E>(self, _: i64) -> Result<Numbers, E> {
                Ok(Numbers::default())
            }
            fn visit_u64<E>(self, _: u64) -> Result<Numbers, E> {
                Ok(Numbers::default())
            }
            fn visit_bool<E>(self, _: bool) -> Result<Numbers, E> {
                Ok(Numbers::default())
            }
            fn visit_str<E>(self, _: &str) -> Result<Numbers, E> {
                Ok(Numbers::default())
            }
            fn visit_unit<E>(self) -> Result<Numbers, E> {
                Ok(Numbers::default())
            }
        }
        deserializer.deserialize_any(NumbersVisitor)
    }
}

/// Python's `_attention_values`: one attention module's mean finite entropy share (fsum over
/// the heads in order), its largest finite logit, and how many heads had a finite share.
#[derive(Default)]
struct AttentionStats {
    share: Option<f64>,
    logit: Option<f64>,
    heads: usize,
}

impl<'de> Deserialize<'de> for AttentionStats {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StatsVisitor;
        impl<'de> Visitor<'de> for StatsVisitor {
            type Value = AttentionStats;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any value")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<AttentionStats, A::Error> {
                // A key given twice keeps its last value, as `json.loads` does.
                let (mut shares, mut logits) = (Vec::new(), Vec::new());
                while let Some(Key(key)) = map.next_key()? {
                    match key.as_ref() {
                        "entropy_share" => shares = map.next_value::<Numbers>()?.0.into_iter().flatten().collect(),
                        "max_logit" => logits = map.next_value::<Numbers>()?.0.into_iter().flatten().collect(),
                        _ => {
                            map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                Ok(AttentionStats {
                    share: (!shares.is_empty()).then(|| fsum(shares.iter().copied()) / shares.len() as f64),
                    logit: logits.iter().copied().reduce(f64::max),
                    heads: shares.len(),
                })
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<AttentionStats, A::Error> {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(AttentionStats::default())
            }
            fn visit_f64<E>(self, _: f64) -> Result<AttentionStats, E> {
                Ok(AttentionStats::default())
            }
            fn visit_i64<E>(self, _: i64) -> Result<AttentionStats, E> {
                Ok(AttentionStats::default())
            }
            fn visit_u64<E>(self, _: u64) -> Result<AttentionStats, E> {
                Ok(AttentionStats::default())
            }
            fn visit_bool<E>(self, _: bool) -> Result<AttentionStats, E> {
                Ok(AttentionStats::default())
            }
            fn visit_str<E>(self, _: &str) -> Result<AttentionStats, E> {
                Ok(AttentionStats::default())
            }
            fn visit_unit<E>(self) -> Result<AttentionStats, E> {
                Ok(AttentionStats::default())
            }
        }
        deserializer.deserialize_any(StatsVisitor)
    }
}

/// A probe's `attention`: its modules in order when it is an object, and none for any other value
/// (`null`, a list, a number, text, a bool), which is read past, as Python's `isinstance(attention,
/// dict)` reads it (BUG-10). Before, such a value failed the whole probe's parse, and its `error` and
/// `not_run` were lost with it.
struct AttentionOf<'a>(Option<Members<'a, AttentionStats>>);

impl<'de: 'a, 'a> Deserialize<'de> for AttentionOf<'a> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct AttentionVisitor<'a>(PhantomData<&'a ()>);
        impl<'de: 'a, 'a> Visitor<'de> for AttentionVisitor<'a> {
            type Value = AttentionOf<'a>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any value")
            }
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<AttentionOf<'a>, A::Error> {
                Ok(AttentionOf(Some(Members(read_members(map)?))))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<AttentionOf<'a>, A::Error> {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(AttentionOf(None))
            }
            fn visit_f64<E>(self, _: f64) -> Result<AttentionOf<'a>, E> {
                Ok(AttentionOf(None))
            }
            fn visit_i64<E>(self, _: i64) -> Result<AttentionOf<'a>, E> {
                Ok(AttentionOf(None))
            }
            fn visit_u64<E>(self, _: u64) -> Result<AttentionOf<'a>, E> {
                Ok(AttentionOf(None))
            }
            fn visit_bool<E>(self, _: bool) -> Result<AttentionOf<'a>, E> {
                Ok(AttentionOf(None))
            }
            fn visit_str<E>(self, _: &str) -> Result<AttentionOf<'a>, E> {
                Ok(AttentionOf(None))
            }
            fn visit_unit<E>(self) -> Result<AttentionOf<'a>, E> {
                Ok(AttentionOf(None))
            }
        }
        deserializer.deserialize_any(AttentionVisitor(PhantomData))
    }
}

/// What the attention rules read of a probe: its attention modules in order, or nothing when
/// the probe failed (`error`, kept as it was written) or recorded no attention. Histograms and
/// maps are skipped unread.
#[derive(Default)]
struct Probe<'a> {
    error: Option<Value>,
    /// The capture's note that `probe=` was given and is not run (a sharded model), with why.
    not_run: Option<Value>,
    attention: Option<Members<'a, AttentionStats>>,
}

impl<'de: 'a, 'a> Deserialize<'de> for Probe<'a> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ProbeVisitor<'a>(PhantomData<&'a ()>);
        impl<'de: 'a, 'a> Visitor<'de> for ProbeVisitor<'a> {
            type Value = Probe<'a>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Probe<'a>, A::Error> {
                let mut probe = Probe::default();
                while let Some(Key(key)) = map.next_key()? {
                    match key.as_ref() {
                        "error" => probe.error = Some(map.next_value()?),
                        "not_run" => probe.not_run = Some(map.next_value()?),
                        // Not an object: no attention, and the rest of the probe is read. A key
                        // given twice keeps its last value, as `json.loads` does.
                        "attention" => probe.attention = map.next_value::<AttentionOf>()?.0,
                        _ => {
                            map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                Ok(probe)
            }
        }
        deserializer.deserialize_map(ProbeVisitor(PhantomData))
    }
}

/// Where a run of probes meeting one rule began.
#[derive(Clone)]
struct Onset {
    value: f64,
    baseline: f64,
    baseline_probes: usize,
    heads: usize,
    scalars: Option<Value>,
    /// The score beside the ratio (part 1), over the probes of the baseline. They are at most
    /// `ATTN_WINDOW`, fewer than `SCORE_MIN`, so it is always `short_history`.
    score: Score,
}

/// Python's `_AttentionSeries`: a module's share or logit, its baseline of probes that met no
/// rule, and per kind the run of probes in a row meeting it (last probe index, length, onset).
#[derive(Default)]
struct AttentionSeries {
    baseline: VecDeque<f64>,
    runs: Vec<(&'static str, usize, usize, Onset)>,
}

#[derive(Default)]
struct AttentionModule {
    share: AttentionSeries,
    logit: AttentionSeries,
}

fn attention_hit(kind: &str, value: f64, base: f64) -> bool {
    match kind {
        "attention_collapse" => base > 0.0 && value <= ATTN_COLLAPSE_RATIO * base,
        "attention_uniform" => value >= ATTN_UNIFORM && value - base >= ATTN_UNIFORM_RISE,
        _ => base > 0.0 && value >= ATTN_LOGIT_FACTOR * base,
    }
}

/// Python's `detect_attention_anomalies`, a probe at a time.
#[derive(Default)]
struct Attention {
    modules: Named<AttentionModule>,
    probe_steps: Vec<i64>,
    /// The incident of each kind, by its place in `ATTN_KINDS`.
    found: [Option<Value>; 3],
    /// The probe steps that are part of a confirmed run, per kind.
    affected: [BTreeSet<i64>; 3],
    scratch: Vec<f64>,
    /// How many rows recorded a probe at all, how many of those failed, and the first
    /// failure as (step, what it said).
    asked: u64,
    failed: u64,
    first_failure: Option<(i64, String)>,
    /// The first note that the probe was given and not run, as (step, why). Not a probe asked.
    off: Option<(i64, String)>,
}

impl Attention {
    fn row(
        &mut self,
        step: i64,
        columns: &Columns,
        row: usize,
        scalars: &mut Scalars,
        measurements: &mut Measurements,
    ) {
        let Some(text) = text(columns.probe, row) else { return };
        self.asked += u64::from(text.trim_start().starts_with('{'));
        // An integer too large for a float is unreadable where a number is wanted, and the rest of
        // the probe is read (AQ5-D2): before, it lost the whole probe.
        let mut rewritten = None;
        let Some(probe) = extra_json::parse_into::<Probe>(text, &mut rewritten) else {
            return;
        };
        if let Some(why) = &probe.not_run {
            // Not a probe that was asked for and failed, as Python reads it: nothing else of it is read.
            self.asked -= 1;
            if self.off.is_none() {
                let said = why.as_str().map_or_else(|| why.to_string(), str::to_owned);
                self.off = Some((step, said));
            }
            return;
        }
        // The measurement `attention_trends` reads a probe's modules whether or not it says
        // `error`, as Python's does; the rules below do not.
        if let Some(Members(attention)) = &probe.attention {
            measurements.probe(step, attention);
        }
        if let Some(error) = &probe.error {
            self.failed += 1;
            if self.first_failure.is_none() {
                let said = error.as_str().map_or_else(|| error.to_string(), str::to_owned);
                self.first_failure = Some((step, said));
            }
            return;
        }
        let Some(Members(attention)) = probe.attention else {
            return;
        };
        if attention.is_empty() {
            return;
        }
        let index = self.probe_steps.len();
        self.probe_steps.push(step);
        let mut confirmed: [Vec<Value>; 3] = Default::default();
        let mut onset_scalars: [Option<Value>; 3] = Default::default();
        let mut context: Option<Option<Value>> = None;
        for (position, (name, stats)) in attention.iter().enumerate() {
            let module = self.modules.slot(position, name);
            for (series, value, kinds) in [
                (&mut module.share, stats.share, &ATTN_KINDS[..2]),
                (&mut module.logit, stats.logit, &ATTN_KINDS[2..]),
            ] {
                let Some(value) = value else {
                    series.runs.clear();
                    continue;
                };
                if series.baseline.len() < ATTN_MIN_HISTORY {
                    series.baseline.push_back(value);
                    continue;
                }
                self.scratch.clear();
                self.scratch.extend(series.baseline.iter().copied());
                let base = median(&mut self.scratch);
                let mut flagged = false;
                for &kind in kinds {
                    let at = ATTN_KINDS
                        .iter()
                        .position(|known| *known == kind)
                        .expect("a known kind");
                    let run = series.runs.iter().position(|(known, ..)| *known == kind);
                    if !attention_hit(kind, value, base) {
                        if let Some(run) = run {
                            series.runs.remove(run);
                        }
                        continue;
                    }
                    flagged = true;
                    let (mut length, mut onset) = match run {
                        Some(run) if series.runs[run].1 + 1 == index => {
                            (series.runs[run].2, series.runs[run].3.clone())
                        }
                        _ => (
                            0,
                            Onset {
                                value,
                                baseline: base,
                                baseline_probes: series.baseline.len(),
                                heads: stats.heads,
                                scalars: None,
                                score: Score::of(value, series.baseline.iter().copied(), &mut self.scratch),
                            },
                        ),
                    };
                    if length == 0 {
                        // What was logged on the first probe of the run, as Python reads it
                        // for the incident's step; worked out once per probe.
                        onset.scalars = context.get_or_insert_with(|| scalars.context(columns, row)).clone();
                    }
                    length += 1;
                    match run {
                        Some(run) => series.runs[run] = (kind, index, length, onset.clone()),
                        None => series.runs.push((kind, index, length, onset.clone())),
                    }
                    if length >= ATTN_SUSTAIN {
                        self.affected[at].extend(&self.probe_steps[index + 1 - length..=index]);
                    }
                    if length == ATTN_SUSTAIN {
                        let mut hit = json!({"module": name, "value": onset.value, "baseline": onset.baseline});
                        if kind == "attention_uniform" {
                            hit["rise"] = json!(onset.value - onset.baseline);
                        } else {
                            hit["ratio"] = json!(onset.value / onset.baseline);
                            onset.score.put(&mut hit);
                        }
                        hit["baseline_probes"] = json!(onset.baseline_probes);
                        hit["heads"] = json!(onset.heads);
                        // Every run confirmed on this probe began on the same probe.
                        onset_scalars[at] = onset.scalars.clone();
                        confirmed[at].push(hit);
                    }
                }
                if !flagged {
                    series.baseline.push_back(value);
                    if series.baseline.len() > ATTN_WINDOW {
                        series.baseline.pop_front();
                    }
                }
            }
        }
        for (at, (mut hits, scalars)) in confirmed.into_iter().zip(onset_scalars).enumerate() {
            if self.found[at].is_some() || hits.is_empty() {
                continue;
            }
            let affected = hits.len();
            hits.truncate(ATTN_MODULES);
            let mut incident = json!({
                "kind": ATTN_KINDS[at],
                "step": self.probe_steps[index + 1 - ATTN_SUSTAIN],
                "confirmed_step": step,
                "modules": hits,
                "affected": affected,
            });
            if let Some(scalars) = scalars {
                incident["scalars"] = scalars;
            }
            self.found[at] = Some(incident);
        }
    }

    fn incidents(&self, out: &mut Vec<Value>) {
        for (at, incident) in self.found.iter().enumerate() {
            if let Some(incident) = incident {
                let mut incident = incident.clone();
                incident["affected_probes"] = json!(self.affected[at].len());
                incident["probes"] = json!(self.probe_steps.len());
                out.push(incident);
            }
        }
    }

    /// The `probe` entry of `not_checked`: probes that failed, or, when none did, the capture's
    /// note that the probe was given and not run (a sharded model). A failed probe leaves its
    /// step unjudged, and a rule that must hold on probes in a row may have missed a change.
    /// Too few probes to judge is not an entry: the attention kinds are then `checked`
    /// without enough history ([`RankDetectors::kind_checks`]), which degrades nothing.
    fn probe_not_checked(&self) -> Option<Value> {
        let Some((step, said)) = self.first_failure.as_ref() else {
            // No probe failed: the capture's note that the probe is off, when it left one.
            let (step, why) = self.off.as_ref()?;
            return Some(json!({
                "check": "probe",
                "kinds": ATTN_KINDS,
                "reason": format!("the probe was not run: {why}"),
                "step": step,
            }));
        };
        Some(json!({
            "check": "probe",
            "kinds": ATTN_KINDS,
            "reason": format!(
                "{} of {} probe{} failed, the first at step {step}: {said}",
                self.failed,
                self.asked,
                if self.asked == 1 { "" } else { "s" },
            ),
            "step": step,
        }))
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
    /// Not shown unchanged at some weight check: changed there, or not compared (the list
    /// leaves out both, and the record does not say which).
    changed: bool,
    /// Already listed under an onset: a parameter is reported once.
    reported: bool,
}

#[derive(Default)]
struct Stalled {
    /// The attempt of the row read last: at a row of another attempt every streak starts again
    /// (`WAVE_AE_CONTRACT.md`, section 2).
    attempt: Option<i64>,
    weights: Named<Weight>,
    first: Option<Value>,
    stalled_steps: u64,
    stalled_total: u64,
    onsets: Vec<Value>,
    onset_steps: u64,
}

impl Stalled {
    /// `skipped` says the row's update was skipped, whoever reported it.
    fn row(&mut self, attempt: i64, step: i64, columns: &Columns, row: usize, skipped: bool) {
        // Over the rank's rows that count, in step order: when a row's attempt differs from that
        // of the row before it, every streak starts again before the row is read, and the first
        // check after the resume is the baseline. What was seen changing and what was already
        // reported are kept, so a stall raised before the resume stands (section 2, amending
        // contract 2 in the author's words: "Reset the streak at the attempt boundary").
        if self.attempt.is_some_and(|before| before != attempt) {
            for (_, weight) in &mut self.weights.items {
                weight.streak = 0;
            }
        }
        self.attempt = Some(attempt);
        // No list: no weight check on this step, and every streak holds. An empty list is a
        // check that showed no parameter unchanged (contract 2).
        let Some(unchanged) = &columns.weights_unchanged else {
            return;
        };
        let Some(listed) = unchanged.row(row) else { return };
        // A loss scaler skips the optimizer step on purpose when a gradient is not finite: such
        // a row adds nothing to a streak, but what it shows changing still breaks one. A skipped
        // update, whoever reported it, is read the same way: the weights were not meant to move
        // (contract section 1, signed off by the author on Nov 20, 2026): a skip cannot advance a
        // streak, a streak that exists carries across it, and the usual resets still apply.
        let evidence = !skipped && !value(columns.grad_norm, row).is_some_and(|norm| !norm.is_finite());
        let unchanged: HashSet<&str> = listed
            .filter(|at| !unchanged.values.is_null(*at))
            .map(|at| unchanged.values.value(at))
            .collect();
        // A parameter seen before and missing from the list changed or was not compared: its
        // streak starts again, with or without a gradient here.
        for (name, weight) in &mut self.weights.items {
            if !unchanged.contains(name.as_str()) {
                weight.streak = 0;
                weight.changed = true;
            }
        }
        let mut with_gradients = 0;
        if let Some(norms) = &columns.grad_norms {
            let entries = norms.row(row);
            let first_entry = entries.start;
            for entry in entries {
                let norm = norms.values.value(entry);
                let positive = !norms.values.is_null(entry) && norm.is_finite() && norm > 0.0;
                with_gradients += usize::from(positive);
                let name = norms.keys.value(entry);
                let same = unchanged.contains(name);
                // Listed with no gradient entry: not reached here, so the streak holds.
                let weight = self.weights.slot(entry - first_entry, name);
                if !(positive && same) {
                    weight.streak = 0;
                } else if evidence {
                    weight.streak += 1;
                }
                // It cannot be newly stalled on the row that shows it changing, so marking
                // it here is the same as Python's marking before the stall check.
                weight.changed |= !same;
            }
        }
        // A refused gradient (`WAVE_AJ_CONTRACT.md` section 1) is a gradient listed with no value,
        // as Python reads the raw `true`: its streak resets. One stored as null was reset above; a
        // name Arrow cannot encode is only in `extra_json`, and is reset here.
        if let Some(Value::Object(refused)) = refused_gradients(columns, row) {
            for name in refused.keys() {
                let weight = self.weights.slot(usize::MAX, name);
                weight.streak = 0;
                weight.changed |= !unchanged.contains(name.as_str());
            }
        }
        if !evidence {
            return;
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
        let mut parsed = || text(columns.extra, row).and_then(extra_json::parse);
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
        // the ones not shown unchanged at an earlier check (changed, or not compared) first.
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
    /// The observations this rule judges, `recorded_steps`, and also the kind's input count: a
    /// caller's skip of unknown scaler outcome is not one (contract section 1, rule 2).
    recorded: u64,
    /// Skips the scaler confirmed with a mixed outcome (`WAVE_AE_CONTRACT.md`, section 4).
    partial: u64,
    first: Option<Value>,
}

impl LossScale {
    /// `by` is who reported the row's skipped update, None when it was not skipped; `partial`
    /// says the scaler skipped some optimizers and applied others ([`skipped_by`]).
    fn row(&mut self, step: i64, columns: &Columns, row: usize, by: Option<SkippedBy>, partial: bool) {
        if columns.step_skipped.is_none_or(|skips| skips.is_null(row)) {
            return;
        }
        // Only a skip the scaler confirmed counts. A recorded applied update ends the streak. A
        // skip only the caller reported never adds a scaler-confirmed skip: with no
        // `scaler_applied` (the scaler's outcome unknown) it is passed over, neither adding to the
        // streak nor ending it, and is in neither `skipped_steps` nor `recorded_steps`; with
        // `scaler_applied: true` it is an applied update to this rule, which ends the streak and
        // counts in `recorded_steps` (that count is the contract session's reading, awaiting the
        // author). So `consecutive_skips: 10` reads "10 consecutive scaler-confirmed skips among
        // observations judged by this rule", which can span more training steps than it counts. A
        // measured loss scale below the floor on either kind of caller's skip still raises, with
        // reason `scale` and the counts unchanged (contract section 1: rules 1 and 3 signed off by
        // the author on Nov 20, 2026, rule 2 as the author amended it).
        // A confirmed mixed outcome counts once in `recorded_steps`, apart as `partial_steps`, and
        // ends the streak of fully skipped updates: an update occurred (section 4, amending
        // section 1 of `STEP_AND_EVIDENCE_CONTRACT.md` in the author's words).
        match by {
            Some(SkippedBy::Scaler | SkippedBy::Both) if partial => {
                self.recorded += 1;
                self.partial += 1;
                self.streak = 0;
            }
            Some(SkippedBy::Caller { applied: false }) => {}
            Some(SkippedBy::Caller { applied: true }) | None => {
                self.recorded += 1;
                self.streak = 0;
            }
            Some(SkippedBy::Scaler | SkippedBy::Both) => {
                self.recorded += 1;
                self.skipped += 1;
                self.streak += 1;
            }
        }
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
            if self.partial > 0 {
                first["partial_steps"] = json!(self.partial);
            }
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
        // On a skipped update, whoever reported it, a gradient that is not finite raises nothing
        // by itself: a scaler expects Inf and NaN there, and one the caller reported is kept in
        // the rank's `skipped_updates` instead (contract section 1). A layer statistic still raises.
        let bad_gradients: Vec<&str> = if flag(columns.step_skipped, row) {
            Vec::new()
        } else {
            non_finite_gradients(columns, row)
        };
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

// --- skipped updates -----------------------------------------------------------------------

/// The rank's `skipped_updates` (contract section 1): the skipped steps by who reported them, and
/// the skips the caller reported (alone or with the scaler) whose gradients were not finite. It
/// raises nothing and changes no kind's value.
#[derive(Default)]
struct SkippedUpdates {
    /// A row of the run carries `step_skipped_by`: only then is the object reported.
    carried: bool,
    caller: u64,
    scaler: u64,
    both: u64,
    /// Partial steps (section 4), under no other name.
    partial: u64,
    non_finite_steps: u64,
    /// The first such step, and what was not finite there.
    first: Option<(i64, Vec<String>)>,
}

impl SkippedUpdates {
    fn row(&mut self, step: i64, columns: &Columns, row: usize, by: Option<SkippedBy>, carried: bool, partial: bool) {
        self.carried |= carried;
        let Some(by) = by else { return };
        match by {
            _ if partial => self.partial += 1,
            SkippedBy::Caller { .. } => self.caller += 1,
            SkippedBy::Scaler => self.scaler += 1,
            SkippedBy::Both => self.both += 1,
        }
        if by == SkippedBy::Scaler {
            return; // as today: what a scaler skipped for is nothing to keep
        }
        let bad = non_finite_gradients(columns, row);
        if bad.is_empty() {
            return;
        }
        self.non_finite_steps += 1;
        if self.first.is_none() {
            self.first = Some((step, bad.into_iter().map(str::to_string).collect()));
        }
    }

    fn report(&self) -> Option<Value> {
        if !self.carried {
            return None;
        }
        let mut out = json!({"caller": self.caller, "scaler": self.scaler, "both": self.both});
        if self.partial > 0 {
            out["partial"] = json!(self.partial);
        }
        if let Some((step, gradients)) = &self.first {
            out["non_finite_gradients"] =
                json!({"steps": self.non_finite_steps, "first_step": step, "gradients": gradients});
        }
        Some(out)
    }
}

// --- unread gradients ----------------------------------------------------------------------

/// The kinds a gradient that could not be read leaves `unavailable` (`STEP_AND_EVIDENCE_CONTRACT.md`,
/// section 2): those judged from the total gradient norm, which leaves the unread parameter out.
/// `non_finite` and `stalled_weights` are judged from what was read and stay `checked`.
pub const GRADS_NOT_READ_KINDS: [&str; 1] = ["grad_spike"];
/// The kind that stays `checked` for the parameters read and names the others under
/// `parameters_not_judged` (`DETECTOR_RULES_CONTRACT.md`, part 5).
pub const GRADS_NOT_READ_NAMED: &str = "stalled_weights";

/// Parameters whose gradient the capture could not read (`grads_not_read`, said once on a record
/// with why): Python's `grads_not_read_not_checked`.
#[derive(Default)]
struct GradsNotRead {
    /// Parameter -> (the step that first named it, why).
    first: BTreeMap<String, (i64, String)>,
}

impl GradsNotRead {
    fn row(&mut self, step: i64, columns: &Columns, row: usize) {
        // `grads_not_read` travels in `extra_json`, which is parsed only on a row that names it.
        let Some(extra) = text(columns.extra, row).filter(|extra| extra.contains("\"grads_not_read\"")) else {
            return;
        };
        let extra: Option<Value> = extra_json::parse(extra);
        let said = extra
            .as_ref()
            .and_then(|extra| extra.get("grads_not_read"))
            .and_then(Value::as_object);
        for (name, why) in said.into_iter().flatten() {
            let why = why.as_str().map_or_else(|| why.to_string(), str::to_owned);
            self.first.entry(name.clone()).or_insert((step, why));
        }
    }

    /// The `gradients` entry of `not_checked`: each parameter, with why and the first step.
    fn not_checked(&self) -> Option<Value> {
        let step = self.first.values().map(|(step, _)| *step).min()?;
        let reason: Vec<String> = self
            .first
            .iter()
            .map(|(name, (_, why))| format!("{name}: {why}"))
            .collect();
        Some(json!({
            "check": "gradients",
            "kinds": GRADS_NOT_READ_KINDS,
            "parameters": self.first.keys().collect::<Vec<_>>(),
            "reason": format!("gradient not read for {}", reason.join("; ")),
            "step": step,
        }))
    }
}

// --- unread layers and the routing's source (part 7) ---------------------------------------

/// The kinds judged from layer statistics: what a capture that read no module leaves
/// `unavailable`, and what names the modules it did not judge when some were read.
pub const LAYER_KINDS: [&str; 3] = ["activation_spike", "activation_drift", "dead_activations"];

/// Two notes a capture says once on a record, in `extra_json` (`DETECTOR_RULES_CONTRACT.md`,
/// part 7): `layers_not_read`, the modules it cannot read at all with why, and `moe_source`, for
/// each router how the routing reached it and where it was taken.
#[derive(Default)]
struct CaptureNotes {
    /// Module -> (the step that first named it, why).
    layers_not_read: BTreeMap<String, (i64, String)>,
    /// Router -> its first `moe_source` entry that is an object, with only `form` and `point`.
    moe_source: BTreeMap<String, Value>,
}

impl CaptureNotes {
    fn row(&mut self, step: i64, columns: &Columns, row: usize) {
        // Both travel in `extra_json`, which is parsed only on a row that names one.
        let Some(extra) = text(columns.extra, row)
            .filter(|extra| extra.contains("\"layers_not_read\"") || extra.contains("\"moe_source\""))
        else {
            return;
        };
        let Some(Value::Object(extra)) = extra_json::parse(extra) else {
            return;
        };
        for (name, why) in extra
            .get("layers_not_read")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
        {
            let why = why.as_str().map_or_else(|| why.to_string(), str::to_owned);
            self.layers_not_read.entry(name.clone()).or_insert((step, why));
        }
        // As Python's `moe_source` reads it (A8, lane AJ-3): of each router the first entry that is
        // an object, holding only its `form` and `point` (an empty object kept as it is). An entry
        // that is not an object is passed over, so a later record can still name the router.
        for (router, source) in extra.get("moe_source").and_then(Value::as_object).into_iter().flatten() {
            let Some(source) = source.as_object() else { continue };
            if self.moe_source.contains_key(router) {
                continue;
            }
            let kept: serde_json::Map<String, Value> = ["form", "point"]
                .into_iter()
                .filter_map(|key| source.get(key).map(|value| (key.to_owned(), value.clone())))
                .collect();
            self.moe_source.insert(router.clone(), Value::Object(kept));
        }
    }

    /// The `layers` entry of `not_checked`. `coverage` is `partial` when some module was read on
    /// some record (`read`): the activation kinds stay checked for what was read, and name the
    /// others. It is `none` when no module was read on any record: the entry lists those kinds,
    /// which are then `unavailable`, since nothing was looked at.
    fn layers_not_checked(&self, read: bool) -> Option<Value> {
        let step = self.layers_not_read.values().map(|(step, _)| *step).min()?;
        let reason: Vec<String> = self
            .layers_not_read
            .iter()
            .map(|(name, (_, why))| format!("{name}: {why}"))
            .collect();
        let mut entry = json!({
            "check": "layers",
            "coverage": if read { "partial" } else { "none" },
            "modules": self.layers_not_read.keys().collect::<Vec<_>>(),
            "reason": format!("layer not read: {}", reason.join("; ")),
            "step": step,
        });
        if !read {
            entry["kinds"] = json!(LAYER_KINDS);
        }
        Some(entry)
    }
}

// --- replica_drift -------------------------------------------------------------------------

#[derive(Default)]
struct ReplicaDrift {
    first: Option<Value>,
}

impl ReplicaDrift {
    fn row(&mut self, step: i64, columns: &Columns, row: usize, tolerance: f64) {
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
            if !non_finite && !values.iter().any(|value| *value > tolerance) {
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
                "tolerance": tolerance,
                "parameters": diverged,
            }));
        }
    }
}

// --- repeated_steps and resume_lr_change: what the attempts did ----------------------------

/// A step whose logged scalars an incident shows, and those scalars once its row is read.
/// They are the scalars of the step's first row that counts, as Python's `scalar_context`
/// takes the first record with the step: a later writing of the step never replaces them
/// (RS12, lane V7, Nov 19, 2026; until then the last writing won).
#[derive(Default)]
struct Wanted {
    step: Option<i64>,
    scalars: Option<Value>,
    /// Whether the step's row was read, with or without scalars.
    filled: bool,
}

impl Wanted {
    fn want(&mut self, step: i64) {
        if self.step != Some(step) {
            *self = Self {
                step: Some(step),
                ..Self::default()
            };
        }
    }

    /// What was logged on a row of the wanted step, kept if it is the first such row.
    fn fill(&mut self, context: &Option<Value>) {
        if !self.filled {
            self.scalars = context.clone();
            self.filled = true;
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
    /// The steps that one attempt writes more than once, when the caller knows them in advance
    /// (`judge`, which holds every row; a store never has them), and what was logged on the
    /// first row of each that counts. By the time the second writing says the step is
    /// repeated, the first one has been read.
    written_twice: HashSet<i64>,
    first_writings: BTreeMap<i64, Option<Value>>,
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
        if let Some(context) = self.first_writings.get(&self.repeated[0]) {
            self.repeated_context.fill(context);
        }
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
    history: &BTreeMap<i64, (i64, Vec<f64>)>,
    new: Option<&[f64]>,
    resumed_at: i64,
    first_lr: Option<&[f64]>,
) -> Result<Option<Value>, &'static str> {
    let new = new.ok_or("no learning rate recorded in the resumed attempt")?;
    if history.is_empty() {
        return Err("no learning rate recorded in the earlier attempts");
    }
    let (basis, expected, source, slope_tolerance) = match history.get(&resumed_at) {
        Some((attempt, lr)) => ("same_step", lr.clone(), *attempt, 0.0),
        None => {
            // The history's last ten steps before the resume, newest first.
            let mut recent = history.range(..resumed_at).rev().take(10);
            let (last, (source, latest)) = recent
                .next()
                .ok_or("the earlier attempts recorded no learning rate before the resume step")?;
            let ahead = (resumed_at - last) as f64;
            let mut expected = latest.clone();
            let mut steepest = 0.0_f64;
            if let Some((oldest, (_, oldest_lr))) = recent.last().filter(|(_, (_, lr))| lr.len() == latest.len()) {
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
    /// The highest step of the current attempt so far. A row at or below it may be a step
    /// written twice within the attempt, which only rows from outside a store can hold.
    current_max: Option<i64>,
    /// The steps of the attempts before the current one, in order and without repeats.
    earlier: Vec<i64>,
    repeated: std::collections::BTreeSet<i64>,
    /// The run as it stood before the current row: each step's (attempt, learning rate), with
    /// each rollback applied. By step, the last writing of a step winning, as Python's dict:
    /// rows whose steps go back within an attempt (only rows from outside a store) must not
    /// leave it out of order (RUST-05).
    history: BTreeMap<i64, (i64, Vec<f64>)>,
    first_lr: Option<Vec<f64>>,
    /// The same history for the data and checkpoint checks.
    resume: ResumePass,
}

impl AttemptPass {
    /// `order` is the rank's attempts, oldest first, each with its first step (contract 1).
    fn new(rank: i32, order: &[(i64, i64)]) -> Self {
        let first = order.first();
        Self {
            rank,
            first_attempt: first.map(|(attempt, _)| *attempt),
            run_start: first.map_or(0, |(_, step)| *step),
            current: None,
            restarted: false,
            current_steps: Vec::new(),
            current_max: None,
            earlier: Vec::new(),
            repeated: Default::default(),
            history: BTreeMap::new(),
            first_lr: None,
            resume: ResumePass::default(),
        }
    }

    fn row(&mut self, attempt: i64, step: i64, resume_row: ResumeRow, found: &mut Attempts, resumes: &mut ResumeFound) {
        let lr = resume_row.lr.clone();
        let first_attempt = self.first_attempt == Some(attempt);
        if self.current != Some(attempt) {
            // The first row of an attempt, which is its lowest step: the caller reads each
            // attempt's rows in step order (`in_step_order`; a store's parts are in it).
            self.earlier.append(&mut self.current_steps);
            self.earlier.sort_unstable();
            self.earlier.dedup();
            self.current = Some(attempt);
            self.current_max = None;
            let first = self.first_attempt == Some(attempt);
            self.restarted = !first && step <= self.run_start;
            if first {
                self.first_lr = lr.clone();
            } else if !self.restarted {
                self.resumed(attempt, step, lr.as_deref(), found);
                self.resume
                    .resumed(self.rank, attempt, step, self.run_start, &resume_row, resumes);
            }
            // This attempt rolls back what earlier ones recorded from its first step on.
            self.history.split_off(&step);
            self.resume.roll_back(step);
        }
        self.resume.add(attempt, step, resume_row, first_attempt);
        let again = self.current_max.is_some_and(|max| step <= max) && self.current_steps.contains(&step);
        let shared = self.restarted && self.earlier.binary_search(&step).is_ok();
        if (again || shared) && self.repeated.insert(step) {
            found.add_repeated(step);
        }
        self.current_steps.push(step);
        self.current_max = Some(self.current_max.map_or(step, |max| max.max(step)));
        match lr {
            Some(lr) => {
                self.history.insert(step, (attempt, lr));
            }
            // A step written again without one: Python's history holds the last writing only.
            None => {
                self.history.remove(&step);
            }
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

// --- what was checked --------------------------------------------------------------------

/// How many rows carried each input that is read straight from a column: part of
/// `input_counts` in `evidence.py`. The other counts are kept by the detectors that parse
/// their input anyway (MoE counts, the probe, the system readings, the loss scaler).
#[derive(Default)]
struct Inputs {
    grad_total: u64,
    weight_check: u64,
    lr: u64,
    data: u64,
    checkpoint: u64,
    layers: u64,
    numbers: u64,
    replica_spread: u64,
}

impl Inputs {
    /// `layers` and `data` say whether the row's `layers` held any module and whether it had
    /// a `data` object, which the caller has parsed already.
    fn row(&mut self, columns: &Columns, row: usize, layers: bool, data: bool) {
        let total = value(columns.grad_norm, row).is_some() || value(columns.clip_grad_norm, row).is_some();
        let norms = columns
            .grad_norms
            .as_ref()
            .is_some_and(|norms| !norms.row(row).is_empty());
        let weights = columns
            .weights_unchanged
            .as_ref()
            .is_some_and(|unchanged| unchanged.row(row).is_some());
        let spread = columns
            .replica_spread
            .as_ref()
            .is_some_and(|spread| !spread.row(row).is_empty());
        // `resume_incomplete` judges a resume from a checkpoint manifest or from the loss scale
        // (E3): either is its input.
        let checkpoint =
            data::checkpoint(columns, row).is_some() || value(columns.grad_scale, row).is_some_and(f64::is_finite);
        self.grad_total += u64::from(total);
        self.weight_check += u64::from(weights && norms);
        self.lr += u64::from(columns.lr(row).is_some());
        self.data += u64::from(data);
        self.checkpoint += u64::from(checkpoint);
        self.layers += u64::from(layers);
        self.numbers += u64::from(layers || total || norms);
        self.replica_spread += u64::from(spread);
    }
}

// --- one rank ------------------------------------------------------------------------------

/// The detectors' state for one rank of one run.
#[derive(Default)]
pub struct RankDetectors {
    moe: Moe,
    gradients: Gradients,
    updates: Updates,
    stalled: Stalled,
    loss_scale: LossScale,
    activations: Activations,
    attention: Attention,
    head: HeadTrends,
    /// The four measurements of the rules contract, part 6: read by no rule.
    measurements: Measurements,
    non_finite: NonFinite,
    skipped_updates: SkippedUpdates,
    grads_not_read: GradsNotRead,
    notes: CaptureNotes,
    replica_drift: ReplicaDrift,
    system: System,
    attempts: Attempts,
    data: DataProgress,
    resumes: ResumeFound,
    scalars: Scalars,
    inputs: Inputs,
    last_step: Option<i64>,
    rows: u64,
    /// The newest attempt read.
    attempt: i64,
    thresholds: Arc<Thresholds>,
}

impl RankDetectors {
    /// Detectors that apply `thresholds`; `Thresholds::default()` gives each rule's usual value.
    pub fn new(thresholds: &Arc<Thresholds>) -> Self {
        Self {
            thresholds: Arc::clone(thresholds),
            ..Self::default()
        }
    }

    /// Detectors that keep no series for `gradient_trend` and `attention_trends`, whose report then
    /// has neither block: for the tests of that setting only, which no build of the server uses.
    #[cfg(test)]
    pub(crate) fn without_trend_series(mut self) -> Self {
        self.measurements.no_series = true;
        self
    }

    /// Detectors that also keep every MoE sample they judge ([`samples`](Self::samples)).
    /// For a run read once from start to end, as `tensorquill-server detect --samples`
    /// reads it: the samples take memory for every step, so the server does not keep them.
    pub fn with_samples(thresholds: &Arc<Thresholds>) -> Self {
        let mut detectors = Self::new(thresholds);
        detectors.moe.kept = Some(Kept::default());
        detectors
    }

    /// The per-step MoE samples, if these detectors keep them. They serialize as the
    /// `samples` list of `build_evidence`.
    pub fn samples(&self) -> Option<MoeSamples<'_>> {
        let kept = self.moe.kept.as_ref()?;
        Some(MoeSamples {
            moe: &self.moe,
            kept,
            from_rows: 0,
        })
    }

    /// The options these detectors apply.
    pub fn thresholds(&self) -> &Thresholds {
        &self.thresholds
    }

    /// Reads the rows of `batch` in order. The caller gives each row once, in step order,
    /// and only rows that count (see [`resolve`]). A batch without a `step` column is ignored.
    pub fn observe(&mut self, batch: &RecordBatch) {
        self.read(batch, None, None);
    }

    /// Reads the rows of `batch`, which belong to `self.attempt`. Rows at or past `cutoff`
    /// were rolled back by a later attempt: only `pass`, which reads every attempt, sees them.
    /// No cutoff (the newest attempt) is `None`, so every step counts, `i64::MAX` too (RS16).
    fn read(&mut self, batch: &RecordBatch, cutoff: Option<i64>, mut pass: Option<&mut AttemptPass>) {
        let Some(columns) = Columns::new(batch) else { return };
        let system_columns = SystemColumns::new(batch);
        for row in 0..batch.num_rows() {
            let step = columns.steps.value(row);
            let data = DataRow::read(&columns, row);
            match pass.as_deref_mut() {
                Some(pass) => {
                    let resume_row = ResumeRow::read(&columns, row, data.clone());
                    pass.row(self.attempt, step, resume_row, &mut self.attempts, &mut self.resumes)
                }
                None => self.attempts.appended(step),
            }
            if cutoff.is_some_and(|cutoff| step >= cutoff) {
                continue;
            }
            let before = self.reported();
            let layers = columns.layers(row);
            let layers = &layers[..];
            self.inputs.row(&columns, row, !layers.is_empty(), data.is_some());
            self.moe.row(step, &columns, row, &self.thresholds, &mut self.scalars);
            self.gradients.row(step, &columns, row, &mut self.scalars);
            self.updates
                .row(self.attempt, step, &columns, row, self.thresholds.update_window);
            let (skipped_by, carried, partial) = skipped_by(&columns, row);
            // A partial step is read by the kinds judged by parameter as a skipped one: the record
            // does not say which optimizer's update was applied (section 4).
            self.stalled
                .row(self.attempt, step, &columns, row, skipped_by.is_some());
            self.loss_scale.row(step, &columns, row, skipped_by, partial);
            self.skipped_updates
                .row(step, &columns, row, skipped_by, carried, partial);
            self.grads_not_read.row(step, &columns, row);
            self.notes.row(step, &columns, row);
            self.activations.row(step, layers);
            self.attention
                .row(step, &columns, row, &mut self.scalars, &mut self.measurements);
            self.measurements.row(self.attempt, step, &columns, row);
            self.head.row(step, &columns, row);
            self.non_finite.row(step, &columns, row, layers);
            self.replica_drift
                .row(step, &columns, row, self.thresholds.drift_tolerance);
            let after = self.reported();
            let wanted = [&self.attempts.repeated_context, &self.attempts.resume_context].map(|w| w.step == Some(step));
            let twice =
                self.attempts.written_twice.contains(&step) && !self.attempts.first_writings.contains_key(&step);
            if after != before || wanted != [false, false] || twice {
                // An incident was first reported on this step, or one names it: give it
                // what was logged here.
                let context = self.scalars.context(&columns, row);
                if twice {
                    self.attempts.first_writings.insert(step, context.clone());
                }
                if wanted[0] {
                    self.attempts.repeated_context.fill(&context);
                }
                if wanted[1] {
                    self.attempts.resume_context.fill(&context);
                }
                if let Some(context) = context {
                    if !before.0[1] && after.0[1] {
                        self.moe.starved_scalars = Some(context.clone());
                    }
                    let firsts = [
                        self.moe.first.as_mut(),
                        None, // starved experts: several steps make up the one incident
                        self.gradients.first.as_mut(),
                        self.updates.first.as_mut(),
                        self.stalled.first.as_mut(),
                        self.loss_scale.first.as_mut(),
                        self.non_finite.first.as_mut(),
                        self.replica_drift.first.as_mut(),
                    ];
                    for (first, was) in firsts.into_iter().zip(before.0) {
                        if let (Some(first), false) = (first, was) {
                            // One parameter's spike, and a router collapsed since its start, is
                            // dated at an earlier step and carries what was logged there, not
                            // what is logged on this row.
                            if first.get("trigger").is_none() && first.get("since_start").is_none() {
                                first["scalars"] = context.clone();
                            }
                        }
                    }
                    for (_, incident, _) in self.activations.found.iter_mut().skip(before.1) {
                        incident["scalars"] = context.clone();
                    }
                }
            }
            self.data.row(step, data.as_ref(), &columns, row, &mut self.scalars);
            let wanted = [&self.resumes.data_context, &self.resumes.state_context].map(|w| w.step == Some(step));
            if wanted != [false, false] {
                let context = self.scalars.context(&columns, row);
                if wanted[0] {
                    self.resumes.data_context.fill(&context);
                }
                if wanted[1] {
                    self.resumes.state_context.fill(&context);
                }
            }
            let new = self.system.row(self.attempt, step, &system_columns, row);
            if new != 0 {
                if let Some(context) = self.scalars.context(&columns, row) {
                    self.system.attach(new, &context);
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

    /// Which incidents exist: MoE imbalance, starved experts, a gradient spike, an update
    /// spike, stalled weights, loss-scale trouble, a non-finite value and replica drift, and
    /// how many activation kinds.
    fn reported(&self) -> ([bool; 8], usize) {
        (
            [
                self.moe.first.is_some(),
                !self.moe.starved.is_empty(),
                self.gradients.first.is_some(),
                self.updates.first.is_some(),
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
        self.updates.incidents(&mut out);
        self.stalled.incidents(&mut out);
        self.attempts.incidents(true, &mut out);
        self.loss_scale.incidents(&mut out);
        self.attempts.incidents(false, &mut out);
        self.resumes.incidents(&mut out);
        self.data.incidents(&mut out);
        self.activations.incidents(&mut out);
        self.attention.incidents(&mut out);
        out.extend(self.non_finite.first.clone());
        self.system.incidents(&mut out);
        out.extend(self.replica_drift.first.clone());
        with_classes(&mut out);
        out
    }

    /// The rank's latest GPU memory reading, or null: what Python's all-ranks view compares
    /// across ranks for `memory_imbalance`.
    pub fn memory(&self) -> Value {
        self.system.memory()
    }

    /// The probe's head numbers over the run, or null: Python's `head_trends`, which evidence
    /// built from this report carries.
    pub fn head_trends(&self) -> Value {
        self.head.value()
    }

    /// The measurements of the rules contract, part 6, that the rows hold, by their report keys:
    /// `gradient_trend`, `attention_trends`, `update_trends` and `weight_jumps`. A block the rows
    /// do not hold is left out, never written as null.
    pub fn measurements(&self) -> Vec<(&'static str, Value)> {
        self.measurements.blocks()
    }

    /// How many times a report of these detectors found a series' trend (`gradient_trend`, or one
    /// module's of `attention_trends`). A series with no new point since the last report is not
    /// found again (wave AF): the count says what each report computed, for tests and diagnosis.
    pub fn trend_computations(&self) -> u64 {
        self.measurements.trend_computations()
    }

    /// The rank's `skipped_updates` (contract section 1), present only when a row carried
    /// `step_skipped_by`: written in the report beside `not_checked`.
    pub fn skipped_updates(&self) -> Option<Value> {
        self.skipped_updates.report()
    }

    /// The capture's `moe_source` (part 7) as Python's `moe_source` gives it: each router's first
    /// object, with only `form` and `point`, or None when no record said one that is an object:
    /// written in the report after `head_trends`.
    pub fn moe_source(&self) -> Option<Value> {
        (!self.notes.moe_source.is_empty()).then(|| json!(self.notes.moe_source))
    }

    /// The rank's `checkpoint_tensors` (`CHECKPOINT_TENSOR_CONTRACT.md`, section 4), present only
    /// when a judged resume's load recorded `tensors` that was compared or unavailable: written in
    /// the report after `skipped_updates`.
    pub fn checkpoint_tensors(&self) -> Option<Value> {
        self.resumes.checkpoint_tensors()
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
        out.extend(self.resumes.not_checked().cloned());
        out.extend(self.attention.probe_not_checked());
        self.system.not_checked(&mut out);
        out.extend(self.moe.refused_not_checked());
        out.extend(self.moe.not_read_not_checked());
        out.extend(self.updates.not_read_not_checked());
        out.extend(self.updates.clip_groups_not_checked());
        out.extend(self.grads_not_read.not_checked());
        out.extend(self.notes.layers_not_checked(self.inputs.layers > 0));
        out
    }

    /// How many rows carried each input, in the order of [`INPUT_NEEDS`]: Python's
    /// `input_counts`, over the rows that count.
    fn input_counts(&self) -> [u64; 16] {
        let [cuda, nvml, data_wait] = self.system.inputs();
        [
            self.moe.judged_rows,
            self.inputs.grad_total,
            self.updates.carried,
            self.inputs.weight_check,
            self.rows,
            self.loss_scale.recorded,
            self.inputs.lr,
            self.inputs.data,
            self.inputs.checkpoint,
            self.inputs.layers,
            self.attention.probe_steps.len() as u64,
            self.inputs.numbers,
            self.inputs.replica_spread,
            cuda,
            nvml,
            data_wait,
        ]
    }

    /// What was checked: for each kind of [`PORTED`], one of three values, as Python's
    /// `kind_checks` gives them.
    ///
    /// - `unavailable`: a [`not_checked`](Self::not_checked) entry names the kind, so its
    ///   input was asked for and could not be judged. The entry's `check` and `reason` come
    ///   with it.
    /// - `checked`: otherwise, when any row carried the kind's input ([`KIND_INPUTS`]).
    /// - `not_configured`: no row did. The capture was never set up to provide it.
    ///
    /// Each entry also has `records`, the rows that carried the input, `needs`, the input by
    /// name, `records_needed` ([`records_needed`]) and `enough_history`, which is `records >=
    /// records_needed` whatever the value. A kind is `checked` from its first row, so it is
    /// `enough_history` that says whether "nothing found" is a verdict yet.
    pub fn kind_checks(&self) -> Value {
        let not_checked = self.not_checked();
        let counts = self.input_counts();
        let mut checks = serde_json::Map::new();
        for (kind, input) in KIND_INPUTS {
            let at = INPUT_NEEDS
                .iter()
                .position(|(name, _)| *name == input)
                .expect("every kind's input is listed");
            let (records, needs) = (counts[at], INPUT_NEEDS[at].1);
            let needed = records_needed(kind, &self.thresholds);
            let enough = records >= needed;
            // A `clip_groups` entry names `update_spike` and leaves it `checked` while some pair was
            // compared; with none compared it makes it `unavailable` (`WAVE_AE_CONTRACT.md`, section 1).
            // A `moe_routing` entry makes the two MoE kinds `unavailable` only while no router was
            // read (the `moe` count is 0); otherwise they are judged on the routers that were (wave AG).
            let names = |item: &&Value| {
                (item["check"] != CLIP_GROUPS || item["compared"] == 0)
                    && (item["check"] != MOE_ROUTING || self.moe.judged_rows == 0)
                    && match item["kinds"].as_array() {
                        Some(kinds) => kinds.iter().any(|named| named == kind),
                        None => CHECK_KINDS
                            .iter()
                            .any(|(check, stands_for)| item["check"] == *check && *stands_for == kind),
                    }
            };
            let mut entry = match not_checked.iter().find(names) {
                Some(item) => json!({
                    "value": "unavailable", "records": records, "records_needed": needed,
                    "enough_history": enough, "needs": needs,
                    "check": item["check"], "reason": item["reason"],
                }),
                None => json!({
                    "value": if records > 0 { "checked" } else { "not_configured" },
                    "records": records, "records_needed": needed, "enough_history": enough, "needs": needs,
                }),
            };
            if let Some((_, note)) = KIND_NOTES
                .iter()
                .find(|(noted, _)| *noted == kind && entry["value"] == "checked")
            {
                entry["note"] = json!(note);
            }
            let judged = entry["value"] != "unavailable";
            if kind == "moe_imbalance" {
                // A router still in the state it started in: nothing is reported for it, and a
                // short capture must not read as checked with enough history (part 3).
                let starting = self.moe.starting_state();
                if !starting.is_empty() {
                    entry["starting_state"] = json!(starting);
                    entry["enough_history"] = json!(false);
                }
            }
            if kind == GRADS_NOT_READ_NAMED && judged && !self.grads_not_read.first.is_empty() {
                // Each parameter on its own history: checked for those read, naming the others
                // (part 5, `STEP_AND_EVIDENCE_CONTRACT.md` section 2).
                entry["parameters_not_judged"] = json!(self.grads_not_read.first.keys().collect::<Vec<_>>());
            }
            if LAYER_KINDS.contains(&kind) && judged && self.inputs.layers > 0 && !self.notes.layers_not_read.is_empty()
            {
                // Some modules read: checked for those, naming the others (part 7).
                entry["modules_not_judged"] = json!(self.notes.layers_not_read.keys().collect::<Vec<_>>());
            }
            checks.insert(kind.to_string(), entry);
        }
        Value::Object(checks)
    }

    /// The last `DUPLICATE_WINDOW` steps that drew a hashed batch, as (step, (data rank, hash)),
    /// for the cross-rank comparison ([`duplicate_batches`]).
    pub fn data_window(&self) -> &BTreeMap<i64, (Option<i64>, String)> {
        self.data.window()
    }
}

/// Python's `resolve_attempts` for one rank: given the rank's attempts oldest first, each with
/// its first step, the step at which each attempt was cut off by a later one (none for the
/// newest, and for an attempt no later one cuts). A row counts when its step is below its
/// attempt's cutoff, and read by attempt and then by step the rows that count are in step
/// order. Until lane V7 (Nov 19, 2026) no cutoff was `i64::MAX`, and a row at that step was
/// never read (RS16). Until lane W3 the order was the attempt numbers'; it is now the order
/// the attempts were first seen in (contract 1): an attempt number is an identity, not an order.
pub fn resolve(order: &[(i64, i64)]) -> BTreeMap<i64, Option<i64>> {
    let mut cutoffs = BTreeMap::new();
    let mut cut: Option<i64> = None;
    for (attempt, first) in order.iter().rev() {
        cutoffs.insert(*attempt, cut);
        cut = Some(cut.map_or(*first, |cut| cut.min(*first)));
    }
    cutoffs
}

/// Rebuilds a rank's detectors from everything stored for it: after a server restart, and
/// whenever rows arrive that are not a plain continuation (a new attempt, or an older
/// attempt's rows resent late). The whole rank is read, not its last window, and the rows a
/// later attempt rolled back are read too: a held baseline, the first incidents and the
/// counts all reach back to the start, and a resume is compared with what it replaced.
pub async fn replay(
    storage: &Storage,
    run_id: &str,
    rank: i32,
    thresholds: &Arc<Thresholds>,
) -> Result<RankDetectors, Error> {
    let parts = storage.rank_parts(run_id, rank).await?;
    // The attempts in the order `rank_parts` lists their parts, which is the rank's attempt
    // order (contract 1), each with its first step.
    let mut order: Vec<(i64, i64)> = Vec::new();
    for part in &parts {
        let first = match part.first_step {
            Some(first) => Some(first),
            // A part this server did not name: read its steps.
            None => storage
                .read_part_for_detect(&part.location)
                .await?
                .iter()
                .filter_map(|batch| column::<Int64Array>(batch, "step"))
                .flat_map(|steps| steps.values().iter().copied())
                .min(),
        };
        if let Some(first) = first {
            first_step_of(&mut order, part.attempt, first);
        }
    }
    let cutoffs = resolve(&order);
    let mut detectors = RankDetectors::new(thresholds);
    let mut pass = AttemptPass::new(rank, &order);
    for part in &parts {
        let Some(cutoff) = cutoffs.get(&part.attempt).copied() else {
            continue;
        };
        detectors.attempt = part.attempt;
        for batch in storage.read_part_for_detect(&part.location).await? {
            detectors.read(&batch, cutoff, Some(&mut pass));
        }
    }
    pass.finish(detectors.last_step, &mut detectors.attempts);
    Ok(detectors)
}

/// Notes `first` as a step of `attempt` in `order` (the attempts oldest first, each with its
/// first step): a new attempt goes last, a known one keeps its place and its lowest step.
fn first_step_of(order: &mut Vec<(i64, i64)>, attempt: i64, first: i64) {
    match order.iter_mut().find(|(known, _)| *known == attempt) {
        Some((_, known)) => *known = (*known).min(first),
        None => order.push((attempt, first)),
    }
}

/// Each attempt's pieces as one batch with its rows in step order, rows of one step in the
/// order given (a stable sort). Pieces already in step order are passed through as they are.
fn in_step_order(pieces: Vec<(i64, RecordBatch)>) -> Result<Vec<(i64, RecordBatch)>, String> {
    let mut grouped: Vec<(i64, Vec<RecordBatch>)> = Vec::new();
    for (attempt, piece) in pieces {
        match grouped.last_mut() {
            Some((last, group)) if *last == attempt => group.push(piece),
            _ => grouped.push((attempt, vec![piece])),
        }
    }
    let mut out = Vec::new();
    for (attempt, group) in grouped {
        let steps: Vec<i64> = group
            .iter()
            .filter_map(|piece| column::<Int64Array>(piece, "step"))
            .flat_map(|steps| steps.values().iter().copied())
            .collect();
        if steps.is_sorted() {
            out.extend(group.into_iter().map(|piece| (attempt, piece)));
            continue;
        }
        let schema = group[0].schema();
        let joined = arrow::compute::concat_batches(&schema, &group).map_err(|error| error.to_string())?;
        let mut index: Vec<u32> = (0..steps.len() as u32).collect();
        index.sort_by_key(|at| steps[*at as usize]); // stable
        let index = arrow::array::UInt32Array::from(index);
        let sorted = arrow::compute::take_record_batch(&joined, &index).map_err(|error| error.to_string())?;
        out.push((attempt, sorted));
    }
    Ok(out)
}

/// The detectors over one rank's rows held in memory: [`replay`] without a store, for
/// `tensorquill-server detect`. The rows must be grouped by attempt; the attempts are taken in
/// the order they first appear (contract 1: a local file is appended across attempts, so its
/// order is arrival order), and an attempt whose rows appear again after another attempt's is
/// refused. Within an attempt the rows are read in step order, rows of one step in the order
/// given (lane W3's step-order rule: the first fault is the lowest step, and an attempt's
/// resume boundary is its lowest step, not its first row). A batch without an `attempt`
/// column is attempt 0, and one without a `step` column is ignored.
pub fn judge(rank: i32, batches: &[RecordBatch], thresholds: &Arc<Thresholds>) -> Result<RankDetectors, String> {
    judge_with(RankDetectors::new(thresholds), rank, batches)
}

/// [`judge`] with detectors made by the caller, which have read nothing yet.
fn judge_with(mut detectors: RankDetectors, rank: i32, batches: &[RecordBatch]) -> Result<RankDetectors, String> {
    // Each batch cut where the attempt changes, so every piece belongs to one attempt.
    let mut pieces: Vec<(i64, RecordBatch)> = Vec::new();
    let mut order: Vec<(i64, i64)> = Vec::new();
    for batch in batches {
        let Some(steps) = column::<Int64Array>(batch, "step") else {
            continue;
        };
        let attempts = column::<Int64Array>(batch, "attempt");
        let attempt_of = |row: usize| attempts.map_or(0, |attempts| attempts.value(row));
        let mut start = 0;
        while start < batch.num_rows() {
            let attempt = attempt_of(start);
            let mut end = start + 1;
            while end < batch.num_rows() && attempt_of(end) == attempt {
                end += 1;
            }
            if let Some((last, _)) = pieces.last() {
                if *last != attempt && order.iter().any(|(known, _)| *known == attempt) {
                    return Err(format!(
                        "rows of attempt {attempt} appear again after rows of attempt {last}: group the rows by attempt, \
                         in the order the attempts were first seen"
                    ));
                }
            }
            if let Some(first) = steps.values()[start..end].iter().copied().min() {
                first_step_of(&mut order, attempt, first);
            }
            pieces.push((attempt, batch.slice(start, end - start)));
            start = end;
        }
    }
    // The steps an attempt writes twice, whose first writing's scalars a `repeated_steps`
    // incident shows (RS12). Only rows from outside a store can hold them.
    let mut seen: HashSet<(i64, i64)> = HashSet::new();
    for (attempt, piece) in &pieces {
        if let Some(steps) = column::<Int64Array>(piece, "step") {
            for step in steps.values().iter() {
                if !seen.insert((*attempt, *step)) {
                    detectors.attempts.written_twice.insert(*step);
                }
            }
        }
    }
    drop(seen);
    let pieces = in_step_order(pieces)?;
    let cutoffs = resolve(&order);
    let mut pass = AttemptPass::new(rank, &order);
    for (attempt, piece) in &pieces {
        let Some(cutoff) = cutoffs.get(attempt).copied() else {
            continue;
        };
        detectors.attempt = *attempt;
        detectors.read(piece, cutoff, Some(&mut pass));
    }
    pass.finish(detectors.last_step, &mut detectors.attempts);
    Ok(detectors)
}

/// A rank's report: the document the server writes as `rank=<rank>/incidents.json` and
/// `tensorquill-server detect` prints. `rebuilds` is how often the server rebuilt the rank's
/// detectors from its store, and 0 for a report made outside a server.
pub fn report(run_id: &str, rank: i32, detectors: &RankDetectors, rebuilds: u64) -> Value {
    let mut report = json!({
        "schema_version": 1,
        "source": "tensorquill-server",
        "run_id": run_id,
        "rank": rank,
        "attempt": detectors.attempt(),
        "last_step": detectors.last_step(),
        "steps": detectors.rows(),
        "rebuilds": rebuilds,
        "checked": PORTED,
        "not_checked": detectors.not_checked(),
        "kind_checks": detectors.kind_checks(),
        "thresholds": detectors.thresholds().applied(),
        "moe_flagged": detectors.moe_flagged(),
        "memory": detectors.memory(),
        "head_trends": detectors.head_trends(),
        "incidents": detectors.incidents(),
    });
    if let Some(source) = detectors.moe_source() {
        report["moe_source"] = source;
    }
    // The four measurements (rules contract, part 6), each only where the rows hold what it reads.
    for (name, block) in detectors.measurements() {
        report[name] = block;
    }
    if let Some(skipped) = detectors.skipped_updates() {
        report["skipped_updates"] = skipped;
    }
    if let Some(tensors) = detectors.checkpoint_tensors() {
        report["checkpoint_tensors"] = tensors;
    }
    report
}

/// `tensorquill-server detect`: one rank's rows in, as an Arrow IPC stream in the schema the
/// SDK sends, and its [`report`] out. Rows of a second run or rank are refused, because the
/// detectors judge one rank's steps.
pub fn detect_stream(input: impl std::io::Read, thresholds: &Arc<Thresholds>) -> Result<Value, String> {
    let (run_id, rank, detectors) = detect_rows(input, RankDetectors::new(thresholds))?;
    Ok(offline_report(&run_id, rank, &detectors))
}

/// How the offline modes order a rank's attempts, as a live report says it under `attempt_order`
/// (`storage::AttemptOrder::kind`): in the order the rows first name them ([`judge`]), which is what
/// a live server whose store kept its `attempts.json` from the rank's first attempt reports,
/// `first_seen`. A live server says `numeric` (or `numeric_then_first_seen`) only for a store
/// written before that file, which offline rows never are. Until wave AF (section 5 of
/// `WAVE_AF_CONTRACT.md`, RS-04) the offline report had no `attempt_order`.
pub const OFFLINE_ATTEMPT_ORDER: &str = "first_seen";

/// A rank's [`report`] as the offline modes print it: no rebuilds, and the order of its attempts.
pub(crate) fn offline_report(run_id: &str, rank: i32, detectors: &RankDetectors) -> Value {
    let mut document = report(run_id, rank, detectors, 0);
    document["attempt_order"] = json!(OFFLINE_ATTEMPT_ORDER);
    document
}

/// `tensorquill-server detect --samples`: the same [`report`] written to `out` as one JSON
/// object with one more key, `samples`: the per-step MoE samples `build_evidence` lists
/// ([`MoeSamples`]). They are written as they are made, never held as one document: a long
/// MoE run has tens of megabytes of them.
pub fn detect_stream_samples(
    input: impl std::io::Read,
    thresholds: &Arc<Thresholds>,
    out: impl std::io::Write,
) -> Result<(), String> {
    #[derive(Serialize)]
    struct WithSamples<'a> {
        #[serde(flatten)]
        report: &'a Value,
        samples: MoeSamples<'a>,
    }
    let (run_id, rank, detectors) = detect_rows(input, RankDetectors::with_samples(thresholds))?;
    let report = offline_report(&run_id, rank, &detectors);
    let samples = detectors.samples().expect("these detectors keep their samples");
    serde_json::to_writer(
        out,
        &WithSamples {
            report: &report,
            samples,
        },
    )
    .map_err(|error| format!("cannot write the report: {error}"))
}

/// `tensorquill-server detect --follow` (wave AF-W): one rank's detectors kept between looks, for the
/// SDK's `--watch` with the Rust detectors, as the live server keeps a rank's between flushes.
///
/// Standard input is a sequence of frames, each an 8-byte little-endian length and that many bytes
/// of one complete Arrow IPC stream; a length of 0, or the end of the input, ends the command. Each
/// frame gets one line on standard output, flushed, holding what `detect` (with `--samples`, what
/// `detect --samples`) prints for the rows read so far, and one more key, `follow`:
///
/// - The first frame is every row of the rank, judged as `detect` judges them (attempts ordered and
///   resolved, every attempt read by the pass that compares a resume with what it replaced). Its
///   line is `detect`'s output with `follow: {"mode": "full", "rows": <rows read>, "samples_from": 0}`.
/// - A later frame holds only rows appended to the newest attempt, each of a higher step than any
///   read before, in step order. They are read as the live server reads a flush after a replay
///   ([`RankDetectors::observe`]): nothing read before is read again. The line has `mode` `append`,
///   `rows` the rows of this frame, and `samples` only the samples of these rows, which start at
///   `samples_from` in the whole list (the earlier ones are unchanged: a sample is judged once).
/// - A later frame that is anything else (another run, rank or attempt; a step not above the last;
///   rows that name a router not seen before or with more experts, which would change every earlier
///   sample) is refused: the line is `{"follow": {"refused": <why>}}` and the command ends. The
///   caller starts again with every row. Its earlier answers stand; none is corrected.
///
/// Rows that are not valid end the command with an error, as in `detect`.
pub fn detect_follow(
    mut input: impl std::io::Read,
    thresholds: &Arc<Thresholds>,
    samples: bool,
    mut out: impl std::io::Write,
) -> Result<(), String> {
    #[derive(Serialize)]
    struct Answer<'a> {
        #[serde(flatten)]
        report: &'a Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        samples: Option<MoeSamples<'a>>,
        follow: Value,
    }
    let write = |out: &mut dyn std::io::Write, line: &dyn erased::Line| -> Result<(), String> {
        line.write(out)
            .map_err(|error| format!("cannot write the report: {error}"))?;
        out.write_all(b"\n")
            .and_then(|()| out.flush())
            .map_err(|error| format!("cannot write the report: {error}"))
    };
    let mut state: Option<(String, i32, RankDetectors)> = None;
    loop {
        let Some(length) = frame_length(&mut input)? else {
            return Ok(());
        };
        let (from_rows, rows, mode) = match state.as_mut() {
            None => {
                let made = if samples {
                    RankDetectors::with_samples(thresholds)
                } else {
                    RankDetectors::new(thresholds)
                };
                let (run_id, rank, detectors, rows) =
                    in_frame(&mut input, length, |frame| detect_rows_counted(frame, made))?;
                state = Some((run_id, rank, detectors));
                (0, rows as u64, "full")
            }
            Some((run_id, rank, detectors)) => match in_frame(&mut input, length, |frame| {
                Ok(follow_append(run_id, *rank, detectors, frame))
            })? {
                Ok((from_rows, rows)) => (from_rows, rows, "append"),
                Err(why) => {
                    return write(&mut out, &erased::Json(json!({"follow": {"refused": why}})));
                }
            },
        };
        let (run_id, rank, detectors) = state.as_ref().expect("judged above");
        let report = offline_report(run_id, *rank, detectors);
        let kept = detectors.samples().map(|whole| MoeSamples { from_rows, ..whole });
        let from = kept.as_ref().map_or(0, |_| from_rows * detectors.moe.routers.len());
        write(
            &mut out,
            &erased::Json(Answer {
                report: &report,
                samples: kept,
                follow: json!({"mode": mode, "rows": rows, "samples_from": from}),
            }),
        )?;
    }
}

/// `serde_json::to_writer` behind a trait object, so one closure writes any answer.
mod erased {
    pub(super) trait Line {
        fn write(&self, out: &mut dyn std::io::Write) -> serde_json::Result<()>;
    }
    pub(super) struct Json<T>(pub T);
    impl<T: serde::Serialize> Line for Json<T> {
        fn write(&self, out: &mut dyn std::io::Write) -> serde_json::Result<()> {
            serde_json::to_writer(out, &self.0)
        }
    }
}

/// The length of the next frame of `detect --follow`. `None` at the end of the input or at a length
/// of 0. The frame itself is read through `take(length)`, never held as one buffer beside the rows
/// read from it (a first frame of 10,000 records is hundreds of MB of Arrow).
fn frame_length(input: &mut impl std::io::Read) -> Result<Option<u64>, String> {
    let mut length = [0_u8; 8];
    let mut got = 0;
    while got < length.len() {
        match input.read(&mut length[got..]) {
            Ok(0) if got == 0 => return Ok(None),
            Ok(0) => return Err("the input ended inside a frame's length".to_string()),
            Ok(read) => got += read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(format!("cannot read the input: {error}")),
        }
    }
    let length = u64::from_le_bytes(length);
    Ok((length != 0).then_some(length))
}

/// Reads one frame of `length` bytes from `input` with `read`, then whatever of it `read` left, so
/// the next frame starts where it should. A frame the input ends inside is said as that, whatever
/// `read` made of its first part.
fn in_frame<T>(
    input: &mut impl std::io::Read,
    length: u64,
    read: impl FnOnce(&mut dyn std::io::Read) -> Result<T, String>,
) -> Result<T, String> {
    let mut frame = std::io::Read::take(input, length);
    let found = read(&mut frame);
    std::io::copy(&mut frame, &mut std::io::sink()).map_err(|error| format!("cannot read the input: {error}"))?;
    if frame.limit() > 0 {
        return Err(format!("the input ended inside a frame of {length} bytes"));
    }
    found
}

/// Reads one later frame of `detect --follow` into `detectors`, when its rows are a plain append:
/// the same run and rank, the newest attempt, each step above the last read, and no router new or
/// grown. Every row is checked before any is read. Gives the rows that counted before the frame
/// (where its samples start) and the rows it held, or why it was refused. A refusal after the rows
/// were read (a router new or grown) leaves the detectors not to be used again; the caller ends.
fn follow_append(
    run_id: &str,
    rank: i32,
    detectors: &mut RankDetectors,
    frame: &mut dyn std::io::Read,
) -> Result<(usize, u64), String> {
    let reader = arrow::ipc::reader::StreamReader::try_new(frame, None)
        .map_err(|error| format!("the frame is not an Arrow IPC stream: {error}"))?;
    let batches = reader
        .collect::<Result<Vec<RecordBatch>, _>>()
        .map_err(|error| format!("cannot read the frame's Arrow IPC stream: {error}"))?;
    let mut last = detectors.last_step();
    let mut before = 0;
    for batch in &batches {
        if batch.num_rows() == 0 {
            continue;
        }
        check_keys(batch, before)?;
        before += batch.num_rows();
        let (Some(run_ids), Some(ranks), Some(steps)) = (
            column::<StringArray>(batch, "run_id"),
            column::<Int32Array>(batch, "rank"),
            column::<Int64Array>(batch, "step"),
        ) else {
            unreachable!("check_keys found the key columns");
        };
        let attempts = column::<Int64Array>(batch, "attempt");
        for row in 0..batch.num_rows() {
            if run_ids.value(row) != run_id || ranks.value(row) != rank {
                return Err(format!(
                    "a row of {} rank {} beside rows of {run_id} rank {rank}",
                    run_ids.value(row),
                    ranks.value(row)
                ));
            }
            let attempt = attempts.map_or(0, |attempts| attempts.value(row));
            if attempt != detectors.attempt() {
                return Err(format!(
                    "a row of attempt {attempt}, where the newest attempt read is {}",
                    detectors.attempt()
                ));
            }
            let step = steps.value(row);
            if last.is_some_and(|last| step <= last) {
                return Err(format!(
                    "a row of step {step}, not above the last step read ({})",
                    last.unwrap_or_default()
                ));
            }
            last = Some(step);
        }
    }
    let routers = |detectors: &RankDetectors| -> Vec<(String, usize)> {
        detectors
            .moe
            .routers
            .iter()
            .map(|router| (router.name.clone(), router.largest))
            .collect()
    };
    let (known, from_rows) = (
        routers(detectors),
        detectors.moe.kept.as_ref().map_or(0, |kept| kept.rows.len()),
    );
    for batch in &batches {
        detectors.observe(batch);
    }
    if routers(detectors) != known {
        return Err("the rows name a router not read before, or give one more experts than before".to_string());
    }
    Ok((from_rows, before as u64))
}

/// The key columns of rows given to the offline modes (`detect` and `detect --run`), checked
/// before anything is judged: a `run_id` string, a `rank` int32 and a `step` int64, none of
/// them null, and an `attempt` int64 without nulls when the column is there. A batch with no
/// `attempt` column is from a sender before attempts, and is attempt 0. Until lane V7 (Nov 19,
/// 2026, RS10) a null read as run "", rank 0 or step 0, and an attempt that was not int64 as
/// attempt 0. Live ingest checks its own (`ingest.rs`).
///
/// The values, too (wave AF, section 1 of `WAVE_AF_CONTRACT.md`): a `rank`, `attempt` or `step`
/// below 0, and a `schema_version` (when the column is there) that is not one of
/// [`crate::ingest::SUPPORTED_ROW_VERSIONS`] (0, 1 and 2 since protocol 5: the versions the live
/// server stores, one list for both), refuse the input with one line that names the column, the
/// value and the row. Until then they were judged: a report of rank -3, or of a row of version 99.
/// `before` is the number of rows of the input before this batch, so the row named is the row's
/// place in the whole input, 1-based, in the order the batches were given. A null names its row
/// too; a stream whose schema declares the column non-nullable is refused by Arrow first, with its
/// own text, which names the column.
pub(crate) fn check_keys(batch: &RecordBatch, before: usize) -> Result<(), String> {
    let (Some(run_ids), Some(ranks)) = (
        column::<StringArray>(batch, "run_id"),
        column::<Int32Array>(batch, "rank"),
    ) else {
        return Err("the rows need a run_id (string) and a rank (int32) column".to_string());
    };
    let Some(steps) = column::<Int64Array>(batch, "step") else {
        return Err("the rows need a step (int64) column".to_string());
    };
    let attempts = match batch.column_by_name("attempt") {
        None => None,
        Some(attempts) => Some(attempts.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
            format!(
                "the attempt column is {}, not int64: leave it out for attempt 0",
                attempts.data_type()
            )
        })?),
    };
    // A `schema_version` column of another type is read as before: no detector reads it.
    let versions = column::<Int32Array>(batch, "schema_version");
    let row_of = |at: usize| before + at + 1;
    for (name, nulls) in [
        ("run_id", run_ids.nulls()),
        ("rank", ranks.nulls()),
        ("step", steps.nulls()),
        ("attempt", attempts.and_then(Array::nulls)),
        ("schema_version", versions.and_then(Array::nulls)),
    ] {
        let Some(nulls) = nulls.filter(|nulls| nulls.null_count() > 0) else {
            continue;
        };
        let first = (0..batch.num_rows()).find(|at| nulls.is_null(*at)).expect("a null");
        return Err(format!(
            "row {}: {name} is null: every row needs its {name} ({} row(s) of this batch have none); the input is refused",
            row_of(first),
            nulls.null_count()
        ));
    }
    let negative = |name: &str, at: usize, value: i64| {
        format!(
            "row {}: {name} is {value}, not an integer of 0 or more; the input is refused",
            row_of(at)
        )
    };
    for at in 0..batch.num_rows() {
        if ranks.value(at) < 0 {
            return Err(negative("rank", at, i64::from(ranks.value(at))));
        }
        if let Some(attempts) = attempts.filter(|attempts| attempts.value(at) < 0) {
            return Err(negative("attempt", at, attempts.value(at)));
        }
        if steps.value(at) < 0 {
            return Err(negative("step", at, steps.value(at)));
        }
        if let Some(versions) =
            versions.filter(|versions| !crate::ingest::SUPPORTED_ROW_VERSIONS.contains(&versions.value(at)))
        {
            let (last, rest) = crate::ingest::SUPPORTED_ROW_VERSIONS.split_last().expect("versions");
            let rest: Vec<String> = rest.iter().map(i32::to_string).collect();
            return Err(format!(
                "row {}: schema_version is {}; this server reads versions {} and {last}; the input is refused",
                row_of(at),
                versions.value(at),
                rest.join(", ")
            ));
        }
    }
    Ok(())
}

/// Reads one rank's rows from an Arrow IPC stream into `detectors`: whose rows they are, and
/// the detectors after the last row.
fn detect_rows(input: impl std::io::Read, detectors: RankDetectors) -> Result<(String, i32, RankDetectors), String> {
    detect_rows_counted(input, detectors).map(|(run_id, rank, detectors, _)| (run_id, rank, detectors))
}

/// [`detect_rows`], with how many rows the input held.
fn detect_rows_counted(
    input: impl std::io::Read,
    detectors: RankDetectors,
) -> Result<(String, i32, RankDetectors, usize), String> {
    let reader = arrow::ipc::reader::StreamReader::try_new(input, None)
        .map_err(|error| format!("the input is not an Arrow IPC stream: {error}"))?;
    let batches = reader
        .collect::<Result<Vec<RecordBatch>, _>>()
        .map_err(|error| format!("cannot read the Arrow IPC stream: {error}"))?;
    let mut whose: Option<(String, i32)> = None;
    let mut before = 0;
    for batch in &batches {
        if batch.num_rows() == 0 {
            continue;
        }
        check_keys(batch, before)?;
        before += batch.num_rows();
        let (Some(run_ids), Some(ranks)) = (
            column::<StringArray>(batch, "run_id"),
            column::<Int32Array>(batch, "rank"),
        ) else {
            unreachable!("check_keys found both columns");
        };
        for row in 0..batch.num_rows() {
            let (run_id, rank) = (run_ids.value(row), ranks.value(row));
            match &whose {
                None => whose = Some((run_id.to_string(), rank)),
                Some((known_run, known_rank)) if known_run == run_id && *known_rank == rank => {}
                Some((known_run, known_rank)) => {
                    return Err(format!(
                        "rows of {run_id} rank {rank} beside rows of {known_run} rank {known_rank}: send one rank of one run"
                    ));
                }
            }
        }
    }
    let Some((run_id, rank)) = whose else {
        return Err("the stream holds no rows".to_string());
    };
    let detectors = judge_with(detectors, rank, &batches)?;
    Ok((run_id, rank, detectors, before))
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_kind_reported_anywhere_has_a_class_and_no_other_does() {
        let mut every: Vec<&str> = PORTED.to_vec();
        every.extend(NOT_PORTED);
        // The status's kinds (`watchdog.rs`) and Python's cross-rank `memory_imbalance`.
        every.extend(["job_stalled", "rank_silent", "duplicate_batches", "memory_imbalance"]);
        every.sort_unstable();
        let mut classed: Vec<&str> = INCIDENT_CLASSES.iter().map(|(kind, _)| *kind).collect();
        classed.sort_unstable();
        assert_eq!(classed, every);
        assert!(INCIDENT_CLASSES
            .iter()
            .all(|(_, class)| matches!(*class, "invariant" | "anomaly" | "observation")));
        assert_eq!(incident_class("gpu_throttled"), Some("observation"));
        assert_eq!(incident_class("non_finite"), Some("invariant"));
        assert_eq!(incident_class("something_new"), None);
    }

    use super::*;

    /// Rows with only the key columns, as (run, rank, attempt, step).
    fn key_rows(rows: &[(&str, i32, i64, i64)]) -> RecordBatch {
        use arrow::datatypes::{DataType, Field, Schema};
        let schema = Schema::new(vec![
            Field::new("run_id", DataType::Utf8, false),
            Field::new("rank", DataType::Int32, false),
            Field::new("attempt", DataType::Int64, false),
            Field::new("step", DataType::Int64, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from_iter_values(rows.iter().map(|row| row.0))),
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|row| row.1))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.2))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.3))),
            ],
        )
        .unwrap()
    }

    fn stream(batches: &[RecordBatch]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut writer = arrow::ipc::writer::StreamWriter::try_new(&mut out, &batches[0].schema()).unwrap();
        for batch in batches {
            writer.write(batch).unwrap();
        }
        writer.finish().unwrap();
        drop(writer);
        out
    }

    fn usual() -> Arc<Thresholds> {
        Arc::default()
    }

    #[test]
    fn thresholds_left_out_are_what_the_rules_always_applied() {
        let usual = Thresholds::parse("{}").unwrap();
        assert_eq!(usual, Thresholds::default());
        assert_eq!(
            (
                usual.overload_threshold,
                usual.drift_tolerance,
                usual.starve_window,
                usual.starve_ratio
            ),
            (0.35, 1e-6, 50, 0.1)
        );
        assert_eq!(usual.top_k, None);
        assert_eq!(usual.update_window, 100);
        let applied = usual.applied();
        assert_eq!(applied["overload_threshold"], 0.35);
        assert_eq!(applied["drift_tolerance"], 1e-6);
        assert_eq!(
            (&applied["starve_window"], &applied["starve_ratio"]),
            (&json!(50), &json!(0.1))
        );
        assert_eq!(applied["top_k"], Value::Null);
        // The rules with no option are listed as before.
        assert_eq!(
            (&applied["spike_sustain"], &applied["input_window"]),
            (&json!(3), &json!(20))
        );
        assert_eq!(
            (&applied["update_window"], &applied["update_alpha"]),
            (&json!(100), &json!(0.05))
        );
        // One key changes only itself, and a null top_k is the recorded value.
        let one = Thresholds::parse(r#"{"starve_window": 10, "top_k": null}"#).unwrap();
        let expected = Thresholds {
            starve_window: 10,
            ..Thresholds::default()
        };
        assert_eq!(one, expected);
    }

    #[test]
    fn thresholds_take_each_of_the_six_keys_and_list_what_was_applied() {
        let text = r#"{"overload_threshold": 1, "drift_tolerance": 0.001, "starve_window": 20,
                       "starve_ratio": 0.25, "top_k": 2, "update_window": 30}"#;
        let all = Thresholds::parse(text).unwrap();
        assert_eq!(all.overload_threshold, 1.0); // a whole number is a number
        assert_eq!((all.top_k_for("any"), all.top_k_for("other")), (Some(2), Some(2)));
        let applied = all.applied();
        for (name, value) in [
            ("overload_threshold", json!(1.0)),
            ("drift_tolerance", json!(0.001)),
            ("starve_window", json!(20)),
            ("starve_ratio", json!(0.25)),
            ("top_k", json!(2)),
            ("update_window", json!(30)),
        ] {
            assert_eq!(applied[name], value, "{name}");
        }
        let named = Thresholds::parse(r#"{"top_k": {"layers.3.router": 8}}"#).unwrap();
        assert_eq!(
            (named.top_k_for("layers.3.router"), named.top_k_for("layers.4.router")),
            (Some(8), None)
        );
        assert_eq!(named.applied()["top_k"], json!({"layers.3.router": 8}));
        assert_eq!(RankDetectors::new(&Arc::new(named.clone())).thresholds(), &named);
    }

    #[test]
    fn thresholds_refuse_what_they_cannot_apply() {
        for (text, reason) in [
            (r#"{"overload_treshold": 0.5}"#, "unknown field `overload_treshold`"),
            (r#"{"spike_factor": 4}"#, "unknown field `spike_factor`"), // listed in a report, but not an option
            ("[0.5]", "expected a JSON object"),
            ("null", "expected a JSON object"),
            ("0.5", "expected a JSON object"),
            ("{", "not JSON"),
            ("", "not JSON"),
            (r#"{"overload_threshold": "0.5"}"#, "invalid type"),
            (r#"{"overload_threshold": null}"#, "invalid type"),
            (
                r#"{"overload_threshold": -0.1}"#,
                "overload_threshold must be a finite number of at least 0",
            ),
            (r#"{"drift_tolerance": -1e-9}"#, "drift_tolerance must be"),
            (r#"{"starve_ratio": -1}"#, "starve_ratio must be"),
            (r#"{"starve_window": 0}"#, "starve_window must be at least 1"),
            (r#"{"starve_window": 2.5}"#, "invalid type"),
            (r#"{"starve_window": -1}"#, "invalid value"),
            (r#"{"update_window": 0}"#, "update_window must be at least 1"),
            (r#"{"update_window": 2.5}"#, "invalid type"),
            (r#"{"update_alpha": 0.1}"#, "unknown field `update_alpha`"), // listed in a report, but not an option
            (r#"{"top_k": 0}"#, "top_k must be at least 1, got 0"),
            (r#"{"top_k": {"a": 2, "b": -1}}"#, "top_k must be at least 1, got -1"),
            (r#"{"top_k": "two"}"#, "TopK"),
            (r#"{"top_k": 2.5}"#, "TopK"),
            (r#"{"top_k": [2]}"#, "TopK"),
        ] {
            let error = Thresholds::parse(text)
                .err()
                .unwrap_or_else(|| panic!("{text} was accepted"));
            assert!(error.contains(reason), "{text}: {error}");
        }
    }

    fn steps(
        run: &'static str,
        rank: i32,
        attempt: i64,
        steps: std::ops::Range<i64>,
    ) -> Vec<(&'static str, i32, i64, i64)> {
        steps.map(|step| (run, rank, attempt, step)).collect()
    }

    #[test]
    fn a_step_written_twice_within_one_attempt_is_repeated_once() {
        // Steps 0..40, then 30..50 again in the same attempt, and step 49 a third time.
        let mut rows = steps("run", 0, 1, 0..40);
        rows.extend(steps("run", 0, 1, 30..50));
        rows.push(("run", 0, 1, 49));
        let detectors = judge(0, &[key_rows(&rows)], &usual()).unwrap();
        let incidents = detectors.incidents();
        assert_eq!(incidents.len(), 1, "{incidents:?}");
        assert_eq!(incidents[0]["kind"], "repeated_steps");
        assert_eq!(incidents[0]["step"], 30);
        assert_eq!(incidents[0]["repeated"], 11);
        assert_eq!(detectors.rows(), 61);
        // Rows in step order, as a store holds them, repeat nothing.
        assert!(judge(0, &[key_rows(&steps("run", 0, 1, 0..60))], &usual())
            .unwrap()
            .incidents()
            .is_empty());
    }

    #[test]
    fn judge_reads_attempts_across_and_inside_batches_as_replay_reads_parts() {
        // A resume at 20 is not a repeat; a restart from step 0 is, for the steps it shares.
        let first = key_rows(&steps("run", 0, 1, 0..30));
        let mut rest = steps("run", 0, 2, 20..40);
        rest.extend(steps("run", 0, 3, 0..10));
        let detectors = judge(0, &[first.clone(), key_rows(&rest)], &usual()).unwrap();
        let incidents = detectors.incidents();
        let repeated: Vec<&Value> = incidents.iter().filter(|i| i["kind"] == "repeated_steps").collect();
        assert_eq!(repeated.len(), 1, "{incidents:?}");
        assert_eq!(repeated[0]["repeated"], 10);
        assert_eq!((detectors.attempt(), detectors.last_step()), (3, Some(9)));
        // Attempts are taken in the order they first appear, whatever their numbers
        // (contract 1): attempt 1 after 2 and 3 is the newest, a restart from step 0.
        let detectors = judge(0, &[key_rows(&rest), first.clone()], &usual()).unwrap();
        assert_eq!((detectors.attempt(), detectors.last_step()), (1, Some(29)));
        // An attempt whose rows appear again after another attempt's is refused.
        let mut again = steps("run", 0, 1, 0..10);
        again.extend(steps("run", 0, 2, 5..10));
        again.extend(steps("run", 0, 1, 10..20));
        let error = judge(0, &[key_rows(&again)], &usual()).err().unwrap();
        assert!(error.contains("group the rows by attempt"), "{error}");
    }

    /// One rank's rows as (attempt, step, learning rate): one group, or none for a null `lr`.
    fn lr_rows(rows: &[(i64, i64, Option<f64>)]) -> RecordBatch {
        use arrow::array::{Float64Builder, ListBuilder};
        use arrow::datatypes::{DataType, Field, Schema};
        let mut lr = ListBuilder::new(Float64Builder::new());
        for (_, _, value) in rows {
            match value {
                Some(value) => {
                    lr.values().append_value(*value);
                    lr.append(true);
                }
                None => lr.append(false),
            }
        }
        let lr = lr.finish();
        let schema = Schema::new(vec![
            Field::new("run_id", DataType::Utf8, false),
            Field::new("rank", DataType::Int32, false),
            Field::new("attempt", DataType::Int64, false),
            Field::new("step", DataType::Int64, false),
            Field::new("lr", lr.data_type().clone(), true),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from_iter_values(rows.iter().map(|_| "run"))),
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|_| 0))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.0))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.1))),
                Arc::new(lr),
            ],
        )
        .unwrap()
    }

    #[test]
    fn a_resume_is_compared_with_the_last_writing_of_each_step_when_an_attempt_went_back() {
        // RUST-05 of the review of Nov 18, 2026: the history a resume is compared with was kept
        // in the order the rows came and binary-searched, so `detect` over rows whose steps
        // went back within one attempt read it wrongly. Python's history is by step, the last
        // writing of a step winning (`detect_resume_lr_changes`); the numbers below are its
        // answers on the same rows. A store never holds such rows: only offline `detect` can.
        let lr = |step: i64| 1e-3 - 1e-5 * step as f64; // a linear decay
        let resume_lr = |detectors: &RankDetectors| {
            let incidents = detectors.incidents();
            let changes: Vec<Value> = incidents
                .iter()
                .filter(|i| i["kind"] == "resume_lr_change")
                .flat_map(|i| i["changes"].as_array().unwrap().clone())
                .collect();
            let skipped: Vec<Value> = detectors
                .not_checked()
                .into_iter()
                .filter(|n| n["check"] == "resume_lr")
                .collect();
            (changes, skipped)
        };

        // Within attempt 0 the step counter went back to 5..8 while the scheduler went on;
        // attempt 1 resumes at 7, which the history holds: compared with its second writing.
        let mut rows: Vec<(i64, i64, Option<f64>)> = (0..21).map(|s| (0, s, Some(lr(s)))).collect();
        rows.extend((5..9).map(|s| (0, s, Some(lr(s + 21)))));
        rows.extend((7..15).map(|s| (1, s, Some(lr(s + 4)))));
        let (changes, skipped) = resume_lr(&judge(0, &[lr_rows(&rows)], &usual()).unwrap());
        assert_eq!(skipped, Vec::<Value>::new());
        assert_eq!(changes.len(), 1, "{changes:?}");
        assert_eq!(
            (
                &changes[0]["basis"],
                &changes[0]["step"],
                &changes[0]["previous_attempt"]
            ),
            (&json!("same_step"), &json!(7), &json!(0))
        );
        assert_eq!(changes[0]["expected_lr"], json!([0.0007199999999999999])); // lr(28), Python's
        assert_eq!(changes[0]["resumed_lr"], json!([lr(11)]));

        // Steps 3..5 written again within attempt 0, and attempt 1 resumes past the history's
        // end: the slope is taken over the history's last ten steps, 1 to 10, as in Python.
        let mut rows: Vec<(i64, i64, Option<f64>)> = (0..11).map(|s| (0, s, Some(lr(s)))).collect();
        rows.extend([(0, 3, Some(lr(11))), (0, 4, Some(lr(12))), (0, 5, Some(lr(13)))]);
        rows.extend((12..20).map(|s| (1, s, Some(lr(s + 3)))));
        let (changes, skipped) = resume_lr(&judge(0, &[lr_rows(&rows)], &usual()).unwrap());
        assert_eq!(changes, Vec::<Value>::new());
        assert_eq!(skipped.len(), 1, "{skipped:?}");
        assert_eq!(skipped[0]["step"], 12);
        assert_eq!(skipped[0]["expected_lr"], json!([0.0008799999999999999])); // Python's
        assert_eq!(skipped[0]["resumed_lr"], json!([lr(15)]));

        // A step written again without a learning rate leaves the history without it, as
        // Python's last writing does: the resume at 7 is extrapolated from steps 0 to 6.
        let mut rows: Vec<(i64, i64, Option<f64>)> = (0..21).map(|s| (0, s, Some(lr(s)))).collect();
        rows.push((0, 7, None));
        rows.extend((7..15).map(|s| (1, s, Some(lr(0)))));
        let (changes, skipped) = resume_lr(&judge(0, &[lr_rows(&rows)], &usual()).unwrap());
        assert_eq!(skipped, Vec::<Value>::new());
        assert_eq!(changes.len(), 1, "{changes:?}");
        assert_eq!(
            (&changes[0]["basis"], &changes[0]["equals_first_lr"]),
            (&json!("extrapolated"), &json!(true))
        );

        // Rows in step order, as a store holds them, are read as before.
        let mut rows: Vec<(i64, i64, Option<f64>)> = (0..21).map(|s| (0, s, Some(lr(s)))).collect();
        rows.extend((7..15).map(|s| (1, s, Some(lr(s + 4)))));
        let (changes, _) = resume_lr(&judge(0, &[lr_rows(&rows)], &usual()).unwrap());
        assert_eq!(changes[0]["expected_lr"], json!([lr(7)]));
    }

    #[test]
    fn detect_stream_gives_the_servers_report_for_one_rank() {
        let batches = [
            key_rows(&steps("run-a", 3, 7, 0..25)),
            key_rows(&steps("run-a", 3, 7, 25..40)),
        ];
        let report = detect_stream(&stream(&batches)[..], &usual()).unwrap();
        assert_eq!(report["source"], "tensorquill-server");
        assert_eq!((&report["run_id"], &report["rank"]), (&json!("run-a"), &json!(3)));
        assert_eq!(
            (&report["attempt"], &report["last_step"], &report["steps"]),
            (&json!(7), &json!(39), &json!(40))
        );
        assert_eq!(report["rebuilds"], 0);
        assert_eq!(report["checked"], json!(PORTED));
        assert_eq!(report["thresholds"], Thresholds::default().applied());
        assert_eq!(report["incidents"], json!([]));
        let checks: Vec<&str> = report["not_checked"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|n| n["check"].as_str())
            .collect();
        assert!(
            checks.contains(&"flight_recorder") && checks.contains(&"traincheck"),
            "{checks:?}"
        );
    }

    /// A row with MoE counts: its attempt, its step, and its routers with their counts.
    type MoeRow<'a> = (i64, i64, Vec<(&'a str, Vec<i64>)>);

    /// One rank's rows with MoE counts. Every router carries `top_k` on every row, as the
    /// SDK writes it.
    fn moe_rows(rows: &[MoeRow], top_k: i32) -> RecordBatch {
        use arrow::array::{Int32Builder, Int64Builder, ListBuilder, MapBuilder, StringBuilder};
        use arrow::datatypes::{DataType, Field, Schema};
        let mut counts = MapBuilder::new(None, StringBuilder::new(), ListBuilder::new(Int64Builder::new()));
        let mut chosen = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
        for (_, _, routers) in rows {
            for (name, values) in routers {
                counts.keys().append_value(name);
                counts.values().append_value(values.iter().copied().map(Some));
                chosen.keys().append_value(name);
                chosen.values().append_value(top_k);
            }
            counts.append(true).unwrap();
            chosen.append(true).unwrap();
        }
        let (counts, chosen) = (counts.finish(), chosen.finish());
        let schema = Schema::new(vec![
            Field::new("run_id", DataType::Utf8, false),
            Field::new("rank", DataType::Int32, false),
            Field::new("attempt", DataType::Int64, false),
            Field::new("step", DataType::Int64, false),
            Field::new("moe_counts", counts.data_type().clone(), true),
            Field::new("moe_top_k", chosen.data_type().clone(), true),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from_iter_values(rows.iter().map(|_| "run"))),
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|_| 0))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.0))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.1))),
                Arc::new(counts),
                Arc::new(chosen),
            ],
        )
        .unwrap()
    }

    /// The samples `detect --samples` writes for these rows, and the rest of its report.
    fn samples_of(batches: &[RecordBatch], thresholds: &Arc<Thresholds>) -> (Vec<Value>, Value) {
        let mut out = Vec::new();
        detect_stream_samples(&stream(batches)[..], thresholds, &mut out).unwrap();
        let mut report: Value = serde_json::from_slice(&out).unwrap();
        let samples = report.as_object_mut().unwrap().remove("samples").unwrap();
        (samples.as_array().unwrap().clone(), report)
    }

    #[test]
    fn samples_list_every_row_and_router_and_mark_what_a_step_did_not_record() {
        let rows = [
            (1, 0, vec![("a", vec![9000, 1000])]),
            (1, 1, vec![]), // a step that recorded no router
            // `a` recorded with nothing routed, and `b` seen for the first time.
            (1, 2, vec![("a", vec![0, 0, 0]), ("b", vec![5000, 5000])]),
        ];
        let (samples, _) = samples_of(&[moe_rows(&rows, 1)], &usual());
        // Two experts and 10,000 tokens: 1.5 times an even share, 0.75, is the largest of the three bounds.
        let missing = |step: i64, layer: usize, router: &str, experts: usize| {
            json!({
                "step": step, "layer": layer, "router": router, "routed_tokens": 0, "counts": vec![0; experts],
                "diversity": null, "overloaded": [], "dead": [], "missing": true, "starved": [],
            })
        };
        let expected = vec![
            json!({
                "step": 0, "layer": 1, "router": "a", "routed_tokens": 10000, "counts": [9000, 1000],
                "diversity": diversity_from_counts(&[9000, 1000]), "overloaded": [1], "dead": [],
                "overload_rule": {"rule": "share", "threshold": 0.75}, "starved": [],
            }),
            // `b` exists in the run, so the steps before it appeared list it too.
            missing(0, 2, "b", 2),
            // `a` is given the most experts it ever recorded, three, where a step left it out,
            missing(1, 1, "a", 3),
            missing(1, 2, "b", 2),
            // and the counts it recorded where they sum to nothing.
            missing(2, 1, "a", 3),
            json!({
                "step": 2, "layer": 2, "router": "b", "routed_tokens": 10000, "counts": [5000, 5000],
                "diversity": diversity_from_counts(&[5000, 5000]), "overloaded": [], "dead": [],
                "overload_rule": {"rule": "share", "threshold": 0.75}, "starved": [],
            }),
        ];
        assert_eq!(samples.len(), expected.len());
        for (found, wanted) in samples.iter().zip(&expected) {
            assert_eq!(found, wanted, "step {} router {}", wanted["step"], wanted["router"]);
        }
        assert!((diversity_from_counts(&[9000, 1000]) - 0.692).abs() < 1e-3);
        // As text, the keys come in the order Python's samples have them.
        let mut out = Vec::new();
        detect_stream_samples(&stream(&[moe_rows(&rows, 1)])[..], &usual(), &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let first = r#""samples":[{"step":0,"layer":1,"router":"a","routed_tokens":10000,"counts":[9000,1000],"diversity":0.69"#;
        assert!(text.contains(first), "{text}");
        let second = r#""overloaded":[1],"dead":[],"overload_rule":{"rule":"share","threshold":0.75},"starved":[]},{"step":0,"layer":2,"router":"b","routed_tokens":0,"counts":[0,0],"diversity":null,"overloaded":[],"dead":[],"missing":true,"starved":[]}"#;
        assert!(text.contains(second), "{text}");
    }

    #[test]
    fn a_sample_says_who_is_starved_only_once_its_window_could_be_judged() {
        let thresholds = Arc::new(Thresholds::parse(r#"{"starve_window": 2}"#).unwrap());
        // Expert 4 gets 1 token in 3,001 on every step: far below a tenth of an even share.
        let rows: Vec<_> = (0..3)
            .map(|step| (1, step, vec![("a", vec![1000, 1000, 1000, 1])]))
            .collect();
        let (samples, report) = samples_of(&[moe_rows(&rows, 1)], &thresholds);
        assert_eq!(samples.len(), 3);
        // One sample is not a window of two: nothing is said, which is not "nobody is starved".
        assert_eq!(samples[0]["starved"], json!([]));
        assert!(samples[0].get("starved_shares").is_none());
        for sample in &samples[1..] {
            assert_eq!(sample["starved"], json!([4]));
            assert_eq!(sample["starved_shares"], json!({"4": 2.0 / 6002.0}));
        }
        // The incident is made from the same judgement.
        let starved = &report["incidents"][0];
        assert_eq!(
            (&starved["kind"], &starved["step"]),
            (&json!("starved_experts"), &json!(1))
        );
        assert_eq!(starved["experts"][0]["window_share"], json!(2.0 / 6002.0));

        // A full window with nobody starved has the key, empty.
        let even: Vec<_> = (0..2).map(|step| (1, step, vec![("a", vec![1000, 1000])])).collect();
        let (samples, _) = samples_of(&[moe_rows(&even, 1)], &thresholds);
        assert!(samples[0].get("starved_shares").is_none());
        assert_eq!(samples[1]["starved_shares"], json!({}));
    }

    #[test]
    fn a_sample_names_the_rule_its_overloaded_experts_were_found_by() {
        // Top-4, where no expert can be far above an even share: judged against the router's own past.
        let mut rows: Vec<_> = (0..12)
            .map(|step| (1, step, vec![("a", vec![2500, 2500, 2500, 2500])]))
            .collect();
        rows.push((1, 12, vec![("a", vec![9700, 100, 100, 100])]));
        let (samples, report) = samples_of(&[moe_rows(&rows, 4)], &usual());
        let drop = json!({"rule": "diversity_drop", "drop": DIVERSITY_DROP, "jump": OVERLOAD_JUMP});
        assert!(samples.iter().all(|sample| sample["overload_rule"] == drop));
        assert!(samples[..12].iter().all(|sample| sample["overloaded"] == json!([])));
        assert_eq!(samples[12]["overloaded"], json!([1]));
        assert_eq!(report["incidents"][0]["step"], 12);
        assert_eq!(report["moe_flagged"], json!({"samples": 1, "last_step": 12}));

        // Before there is a baseline, a top-4 router with dead experts is attributed by share:
        // above twice an even share, which here is the larger bound; a top-4 expert cannot pass
        // 1/4, so the midpoint between an even share and 1/4 is used instead (DET-11).
        let collapsed = [(1, 0, vec![("a", vec![9000, 1000, 0, 0])])];
        let (samples, _) = samples_of(&[moe_rows(&collapsed, 4)], &usual());
        assert_eq!(
            samples[0]["overload_rule"],
            json!({"rule": "collapse", "threshold": 0.25})
        );
        assert_eq!(
            (&samples[0]["overloaded"], &samples[0]["dead"]),
            (&json!([1]), &json!([3, 4]))
        );
    }

    #[test]
    fn samples_cover_the_rows_that_count_in_step_order() {
        // Attempt 2 resumes at step 2: what attempt 1 wrote there and after was rolled back.
        let token = |attempt: i64, step: i64| vec![("a", vec![1000 * attempt + step, 500])];
        let first: Vec<_> = (0..4).map(|step| (1, step, token(1, step))).collect();
        let second: Vec<_> = (2..5).map(|step| (2, step, token(2, step))).collect();
        let (samples, report) = samples_of(&[moe_rows(&first, 1), moe_rows(&second, 1)], &usual());
        let read: Vec<(i64, i64)> = samples
            .iter()
            .map(|sample| (sample["step"].as_i64().unwrap(), sample["counts"][0].as_i64().unwrap()))
            .collect();
        assert_eq!(read, [(0, 1000), (1, 1001), (2, 2002), (3, 2003), (4, 2004)]);
        assert_eq!(report["steps"], 5);

        // A step written twice in one attempt has two samples, in the order they were written.
        let twice = [
            (1, 0, token(1, 0)),
            (1, 1, token(1, 1)),
            (1, 1, token(3, 1)),
            (1, 2, token(1, 2)),
        ];
        let (samples, _) = samples_of(&[moe_rows(&twice, 1)], &usual());
        let read: Vec<i64> = samples
            .iter()
            .map(|sample| sample["counts"][0].as_i64().unwrap())
            .collect();
        assert_eq!(read, [1000, 1001, 3001, 1002]);
    }

    #[test]
    fn samples_are_kept_only_when_asked_and_the_servers_report_never_has_them() {
        let rows: Vec<_> = (0..20)
            .map(|step| (1, step, vec![("a", vec![9000, 1000]), ("b", vec![10, 10, 10])]))
            .collect();
        let batches = [moe_rows(&rows, 1)];
        // What the server keeps for a rank: no samples, however long the run.
        let server = judge(0, &batches, &usual()).unwrap();
        assert!(server.samples().is_none() && server.moe.kept.is_none());
        // As `detect` prints it: with the offline `attempt_order` (wave AF, section 5).
        let plain = offline_report("run", 0, &server);
        assert!(plain.get("samples").is_none());
        assert_eq!(detect_stream(&stream(&batches)[..], &usual()).unwrap(), plain);

        let (samples, with_samples) = samples_of(&batches, &usual());
        assert_eq!(samples.len(), 20 * 2);
        // Every other key of the report is the same, in name and value.
        assert_eq!(with_samples, plain);
        let kept = judge_with(RankDetectors::with_samples(&usual()), 0, &batches).unwrap();
        assert_eq!(kept.samples().unwrap().len(), 40);
        assert_eq!(serde_json::to_value(kept.samples().unwrap()).unwrap(), json!(samples));

        // Rows with no MoE column at all: the key is there, and empty.
        let (samples, report) = samples_of(&[key_rows(&steps("run", 0, 1, 0..5))], &usual());
        assert!(samples.is_empty());
        assert_eq!(report["steps"], 5);
    }

    /// One rank's rows of attempt 1, steps 0, 1, ..., each with the probe it recorded as JSON text.
    fn probe_rows(probes: &[Option<&str>]) -> RecordBatch {
        use arrow::datatypes::{DataType, Field, Schema};
        let schema = Schema::new(vec![
            Field::new("run_id", DataType::Utf8, false),
            Field::new("rank", DataType::Int32, false),
            Field::new("attempt", DataType::Int64, false),
            Field::new("step", DataType::Int64, false),
            Field::new("probe_json", DataType::Utf8, true),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from_iter_values(probes.iter().map(|_| "run"))),
                Arc::new(Int32Array::from_iter_values(probes.iter().map(|_| 0))),
                Arc::new(Int64Array::from_iter_values(probes.iter().map(|_| 1))),
                Arc::new(Int64Array::from_iter_values(0..probes.len() as i64)),
                Arc::new(StringArray::from(probes.to_vec())),
            ],
        )
        .unwrap()
    }

    fn values(checks: &Value, value: &str) -> Vec<String> {
        // In the order of `PORTED`, which a JSON object does not keep.
        PORTED
            .iter()
            .filter(|kind| checks[**kind]["value"] == value)
            .map(|kind| kind.to_string())
            .collect()
    }

    #[test]
    fn the_tables_of_what_each_kind_needs_cover_the_kinds_in_order() {
        let kinds: Vec<&str> = KIND_INPUTS.iter().map(|(kind, _)| *kind).collect();
        assert_eq!(kinds, PORTED);
        for (kind, input) in KIND_INPUTS {
            assert!(
                INPUT_NEEDS.iter().any(|(name, _)| *name == input),
                "{kind} needs {input}"
            );
        }
        for (check, kind) in CHECK_KINDS {
            assert!(PORTED.contains(&kind), "{check} stands for {kind}");
        }
    }

    #[test]
    fn what_each_kind_needs_is_these_numbers_and_the_window_a_run_is_judged_with() {
        let kinds: Vec<&str> = RECORDS_NEEDED.iter().map(|(kind, _)| *kind).collect();
        assert_eq!(kinds, PORTED);
        // By hand from the rules, not from the constants: evidence.py lists each kind's rules.
        // `activation_drift` needs 110 (`DETECTOR_RULES_CONTRACT.md` part 2; it needed 60).
        let by_hand: [u64; 26] = [
            11, 50, 13, 102, 3, 1, 10, 1, 1, 1, 4, 4, 11, 110, 60, 4, 4, 4, 1, 1, 6, 1, 1, 2, 1, 21,
        ];
        for ((kind, needed), expected) in RECORDS_NEEDED.iter().zip(by_hand) {
            assert_eq!(*needed, expected, "{kind}");
            assert_eq!(records_needed(kind, &usual()), expected, "{kind}");
        }
        let narrow = Arc::new(Thresholds::parse(r#"{"starve_window": 7}"#).unwrap());
        assert_eq!(records_needed("starved_experts", &narrow), 7);
        assert_eq!(records_needed("moe_imbalance", &narrow), 11);
        let short = Arc::new(Thresholds::parse(r#"{"update_window": 30}"#).unwrap());
        assert_eq!(records_needed("update_spike", &short), 32);
        assert_eq!(records_needed("starved_experts", &short), 50);
        let rows: Vec<MoeRow> = (0..7).map(|step| (1, step, vec![("a", vec![40, 40])])).collect();
        let checks = judge(0, &[moe_rows(&rows, 1)], &narrow).unwrap().kind_checks();
        assert_eq!(checks["starved_experts"]["records_needed"], 7);
        assert_eq!(checks["starved_experts"]["enough_history"], true);
        assert_eq!(checks["moe_imbalance"]["enough_history"], false);
        let checks = judge(0, &[moe_rows(&rows, 1)], &usual()).unwrap().kind_checks();
        assert_eq!(checks["starved_experts"]["records_needed"], 50);
        assert_eq!(checks["starved_experts"]["enough_history"], false);
    }

    #[test]
    fn a_kind_is_checked_from_its_first_row_and_says_when_its_rule_has_enough_history() {
        let good = r#"{"attention":{"blocks.0.attn":{"entropy_share":[0.6,0.62],"max_logit":[2.0,2.1]}}}"#;
        for (probes, enough) in [(1, false), (3, false), (4, true), (5, true)] {
            let rows: Vec<Option<&str>> = (0..probes).map(|_| Some(good)).collect();
            let detectors = judge(0, &[probe_rows(&rows)], &usual()).unwrap();
            // Too few probes is no entry of `not_checked`: only the two files the server never has.
            assert_eq!(detectors.not_checked().len(), 2, "{probes} probes");
            let checks = detectors.kind_checks();
            for kind in ATTN_KINDS {
                assert_eq!(
                    checks[kind],
                    json!({
                        "value": "checked", "records": probes, "records_needed": 4, "enough_history": enough,
                        "needs": "probe with attention (probe=)",
                    }),
                    "{kind} after {probes} probes"
                );
            }
            // Every row is the input of `repeated_steps`, which needs one.
            assert_eq!(checks["repeated_steps"]["enough_history"], true);
        }
        // A kind no row carried the input of has no history either, and is still not configured.
        let none = judge(0, &[key_rows(&steps("run", 0, 1, 0..3))], &usual())
            .unwrap()
            .kind_checks();
        assert_eq!(
            none["grad_spike"],
            json!({
                "value": "not_configured", "records": 0, "records_needed": 13, "enough_history": false,
                "needs": "grad_norm or clip_grad_norm",
            })
        );
        for kind in PORTED {
            let check = &none[kind];
            let enough = check["records"].as_u64().unwrap() >= check["records_needed"].as_u64().unwrap();
            assert_eq!(check["enough_history"], enough, "{kind}");
        }
    }

    #[test]
    fn rows_that_carry_nothing_but_their_step_check_one_kind() {
        let detectors = judge(0, &[key_rows(&steps("run", 0, 1, 0..12))], &usual()).unwrap();
        let checks = detectors.kind_checks();
        assert_eq!(checks.as_object().unwrap().len(), PORTED.len());
        assert_eq!(values(&checks, "checked"), ["repeated_steps"]);
        assert_eq!(values(&checks, "not_configured").len(), 25);
        assert_eq!(
            checks["repeated_steps"],
            json!({
                "value": "checked", "records": 12, "records_needed": 1, "enough_history": true,
                "needs": "step records",
            })
        );
        assert_eq!(
            checks["stalled_weights"],
            json!({
                "value": "not_configured", "records": 0, "records_needed": 3, "enough_history": false,
                "needs": "weights_unchanged with grad_norms (full capture)",
            })
        );
        // The report carries it, beside the two lists it always had.
        let document = report("run", 0, &detectors, 0);
        assert_eq!(document["kind_checks"], checks);
        assert_eq!(document["checked"], json!(PORTED));
        assert!(document["not_checked"].as_array().unwrap().len() == 2);
    }

    #[test]
    fn a_router_only_run_checks_the_moe_kinds_by_the_rows_that_routed_tokens() {
        let rows = [
            (1, 0, vec![("a", vec![500, 500])]),
            (1, 1, vec![("a", vec![0, 0])]), // nothing routed: nothing to judge
            (1, 2, vec![]),
            (1, 3, vec![("a", vec![0, 0]), ("b", vec![1, 0])]),
        ];
        let checks = judge(0, &[moe_rows(&rows, 1)], &usual()).unwrap().kind_checks();
        assert_eq!(
            values(&checks, "checked"),
            ["moe_imbalance", "starved_experts", "repeated_steps"]
        );
        // Two rows routed tokens: both kinds are checked, and neither rule has its history yet.
        let moe = |needed: u64| {
            json!({
                "value": "checked", "records": 2, "records_needed": needed, "enough_history": false,
                "needs": "moe: a router's expert counts",
            })
        };
        assert_eq!(
            (&checks["moe_imbalance"], &checks["starved_experts"]),
            (&moe(11), &moe(50))
        );
        assert_eq!(checks["repeated_steps"]["records"], 4);
        assert!(values(&checks, "unavailable").is_empty());
    }

    #[test]
    fn a_refused_router_makes_the_moe_kinds_unavailable_and_says_why() {
        use arrow::datatypes::{DataType, Field, Schema};
        // The capture leaves a refused router out of `moe_counts` and says why once, on the
        // first record it wrote after the refusal, under `moe_refused` in `extra_json`.
        let rows: Vec<MoeRow> = (0..4)
            .map(|step| (1, step, vec![("counted", vec![30, 30, 30, 30])]))
            .collect();
        let base = moe_rows(&rows, 2);
        let said = r#"{"moe_refused":{"refused":"its output's first integer tensor has shape (8,)"}}"#;
        let extra: Vec<Option<&str>> = vec![None, Some(said), None, Some(r#"{"moe_refused_not":1}"#)];
        let mut fields: Vec<Field> = base.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
        fields.push(Field::new("extra_json", DataType::Utf8, true));
        let mut columns = base.columns().to_vec();
        columns.push(Arc::new(StringArray::from(extra)));
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();

        let detectors = judge(0, &[batch], &usual()).unwrap();
        let reason = "router refused is not counted: its output's first integer tensor has shape (8,)";
        let entry = json!({
            "check": "router", "kinds": ["moe_imbalance", "starved_experts"], "routers": ["refused"],
            "reason": reason, "step": 1,
        });
        assert_eq!(detectors.not_checked().last(), Some(&entry));
        let checks = detectors.kind_checks();
        assert_eq!(values(&checks, "unavailable"), ["moe_imbalance", "starved_experts"]);
        // The counted router's rows still count toward the kinds' history.
        assert_eq!(checks["moe_imbalance"]["records"], 4);
        assert_eq!(
            (&checks["moe_imbalance"]["check"], &checks["moe_imbalance"]["reason"]),
            (&json!("router"), &json!(reason))
        );
        assert!(detectors
            .incidents()
            .iter()
            .all(|i| i["kind"] != "moe_imbalance" && i["kind"] != "starved_experts"));

        // Without the note, the same rows are an MoE run that was checked.
        let clean = judge(0, &[base], &usual()).unwrap();
        assert!(clean.not_checked().iter().all(|e| e["check"] != "router"));
        assert_eq!(clean.kind_checks()["moe_imbalance"]["value"], "checked");
    }

    #[test]
    fn a_failed_probe_makes_the_attention_kinds_unavailable_and_says_why() {
        let good = r#"{"attention":{"blocks.0.attn":{"entropy_share":[0.6,0.62],"max_logit":[2.0,2.1]}}}"#;
        let failed = r#"{"error":"RuntimeError: CUDA out of memory"}"#;
        let attention = &ATTN_KINDS[..];

        // Five probes that all recorded attention: checked, and nothing more is listed.
        let rows: Vec<Option<&str>> = (0..9).map(|step| (step % 2 == 0).then_some(good)).collect();
        let clean = judge(0, &[probe_rows(&rows)], &usual()).unwrap();
        assert_eq!(clean.not_checked().len(), 2);
        let checks = clean.kind_checks();
        assert_eq!(
            values(&checks, "checked"),
            [
                "repeated_steps",
                "attention_collapse",
                "attention_uniform",
                "attention_logit_growth"
            ]
        );
        assert_eq!(checks["attention_uniform"]["records"], 5);

        // One of them failed.
        let mut rows = rows;
        rows[4] = Some(failed);
        let detectors = judge(0, &[probe_rows(&rows)], &usual()).unwrap();
        let reason = "1 of 5 probes failed, the first at step 4: RuntimeError: CUDA out of memory";
        let entry = json!({"check": "probe", "kinds": attention, "reason": reason, "step": 4});
        assert_eq!(detectors.not_checked().last(), Some(&entry));
        let checks = detectors.kind_checks();
        assert_eq!(values(&checks, "unavailable"), attention);
        for kind in attention {
            assert_eq!(
                checks[*kind],
                json!({
                    "value": "unavailable", "records": 4, "records_needed": 4, "enough_history": true,
                    "needs": "probe with attention (probe=)", "check": "probe", "reason": reason,
                })
            );
        }
        assert_eq!(values(&checks, "checked"), ["repeated_steps"]);

        // Too few probes to judge and one that failed: only the failure is an entry.
        let few = judge(0, &[probe_rows(&[Some(good), Some(failed), Some(good)])], &usual()).unwrap();
        let checks: Vec<Value> = few.not_checked()[2..]
            .iter()
            .map(|item| item["check"].clone())
            .collect();
        assert_eq!(checks, [json!("probe")]);
        assert_eq!(few.kind_checks()["attention_collapse"]["check"], "probe");
        assert_eq!(few.kind_checks()["attention_collapse"]["records"], 2);
        assert_eq!(few.kind_checks()["attention_collapse"]["enough_history"], false);

        // An error that is not text is kept as the JSON it was.
        let odd = judge(0, &[probe_rows(&[Some(r#"{"error":{"code":3}}"#)])], &usual()).unwrap();
        let said = odd.not_checked().last().unwrap()["reason"].clone();
        assert_eq!(said, r#"1 of 1 probe failed, the first at step 0: {"code":3}"#);

        // The capture turned the probe off (a sharded model) and said so once: not a probe that
        // failed, and the attention kinds are unavailable with its reason, as Python reports it.
        let note = r#"{"not_run":"the model is sharded across ranks (FSDP2, fully_shard)"}"#;
        let off = judge(0, &[probe_rows(&[Some(note), None, None])], &usual()).unwrap();
        let reason = "the probe was not run: the model is sharded across ranks (FSDP2, fully_shard)";
        let entry = json!({"check": "probe", "kinds": attention, "reason": reason, "step": 0});
        assert_eq!(off.not_checked().last(), Some(&entry));
        let checks = off.kind_checks();
        assert_eq!(values(&checks, "unavailable"), attention);
        assert_eq!(checks["attention_collapse"]["reason"], reason);
        // A failed probe beside the note: the failure is the entry, counted over the probes asked.
        let both = judge(0, &[probe_rows(&[Some(note), Some(failed)])], &usual()).unwrap();
        let said = both.not_checked().last().unwrap()["reason"].clone();
        assert_eq!(
            said,
            "1 of 1 probe failed, the first at step 1: RuntimeError: CUDA out of memory"
        );
    }

    #[test]
    fn detect_stream_refuses_what_it_cannot_judge() {
        let two_ranks = [key_rows(&steps("run", 0, 1, 0..5)), key_rows(&steps("run", 1, 1, 0..5))];
        let error = detect_stream(&stream(&two_ranks)[..], &usual()).err().unwrap();
        assert!(error.contains("send one rank of one run"), "{error}");
        let two_runs = [
            key_rows(&steps("run", 0, 1, 0..5)),
            key_rows(&steps("other", 0, 1, 5..9)),
        ];
        assert!(detect_stream(&stream(&two_runs)[..], &usual()).is_err());
        let error = detect_stream(&stream(&[key_rows(&[])])[..], &usual()).err().unwrap();
        assert!(error.contains("no rows"), "{error}");
        let error = detect_stream(&b"{\"step\": 0}\n"[..], &usual()).err().unwrap();
        assert!(error.contains("not an Arrow IPC stream"), "{error}");
    }

    /// A step, its total gradient norm and each parameter's update RMS.
    type UpdateTestRow<'a> = (i64, f64, &'a [(&'a str, f64)]);

    /// Rows with a total gradient norm, each parameter's update RMS and, when given, what the
    /// capture said it could not read (`extra_json`).
    fn update_rows(rows: &[UpdateTestRow], extra: &[Option<&str>]) -> RecordBatch {
        use arrow::array::{Float64Builder, MapBuilder, StringBuilder};
        use arrow::datatypes::{DataType, Field, Schema};
        let mut sizes = MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
        for (_, _, own) in rows {
            for (name, size) in own.iter() {
                sizes.keys().append_value(name);
                sizes.values().append_value(*size);
            }
            sizes.append(!own.is_empty()).unwrap();
        }
        let sizes = sizes.finish();
        let schema = Schema::new(vec![
            Field::new("run_id", DataType::Utf8, false),
            Field::new("rank", DataType::Int32, false),
            Field::new("attempt", DataType::Int64, false),
            Field::new("step", DataType::Int64, false),
            Field::new("grad_norm", DataType::Float64, true),
            Field::new("update_rms", sizes.data_type().clone(), true),
            Field::new("extra_json", DataType::Utf8, true),
        ]);
        let said = |at: usize| extra.get(at).copied().flatten();
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from_iter_values(rows.iter().map(|_| "run"))),
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|_| 0))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|_| 1))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.0))),
                Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| row.1))),
                Arc::new(sizes),
                Arc::new(StringArray::from(
                    (0..rows.len()).map(said).collect::<Vec<Option<&str>>>(),
                )),
            ],
        )
        .unwrap()
    }

    #[test]
    fn an_update_spike_is_a_jump_past_the_baseline_on_a_step_whose_gradient_norm_rose() {
        let short = Arc::new(Thresholds::parse(r#"{"update_window": 4}"#).unwrap());
        // Six rows: one to compare the first with, four changes of 0.001 that make the first
        // baseline, and a fifth that is judged, not flagged, and takes the oldest one's place.
        let quiet: [(&str, f64); 2] = [("a", 0.010), ("b", 0.002)];
        let other: [(&str, f64); 2] = [("a", 0.011), ("b", 0.002)];
        let jumped: [(&str, f64); 2] = [("a", 0.011), ("b", 0.300)];
        let rows: Vec<UpdateTestRow> = (0..6)
            .map(|step| (step, 1.0, if step % 2 == 0 { &quiet[..] } else { &other[..] }))
            .collect();
        assert!(judge(0, &[update_rows(&rows, &[])], &short)
            .unwrap()
            .incidents()
            .is_empty());
        // The jump with a gradient norm that fell is not flagged; with one that rose it is.
        let mut fell = rows.clone();
        fell.push((6, 0.5, &jumped[..]));
        assert!(judge(0, &[update_rows(&fell, &[])], &short)
            .unwrap()
            .incidents()
            .is_empty());
        let mut rose = rows.clone();
        rose.push((6, 5.0, &jumped[..]));
        // The fall back at step 7, with no rise in the norm, is not flagged and joins the baseline:
        // it raises it, and the same jump at step 8 is then under the limit.
        rose.push((7, 1.0, &quiet[..]));
        rose.push((8, 5.0, &jumped[..]));
        let detectors = judge(0, &[update_rows(&rose, &[])], &short).unwrap();
        let incidents = detectors.incidents();
        assert_eq!(incidents.len(), 1, "{incidents:?}");
        let found = &incidents[0];
        assert_eq!(
            (&found["kind"], &found["class"], &found["step"]),
            (&json!("update_spike"), &json!("anomaly"), &json!(6))
        );
        // The limit is baseline / (alpha * sqrt(rise)) = 0.001 / (0.05 * 2) = 0.01.
        assert!((found["baseline"].as_f64().unwrap() - 0.001).abs() < 1e-12);
        assert!((found["threshold"].as_f64().unwrap() - 0.01).abs() < 1e-12);
        assert_eq!(
            (&found["grad_rise"], &found["spike_steps"], &found["spike_count"]),
            (&json!(4.0), &json!([6]), &json!(1))
        );
        assert_eq!(found["parameters"][0]["name"], "b");
        assert_eq!(found["parameters"].as_array().unwrap().len(), 2);
        let check = &detectors.kind_checks()["update_spike"];
        assert_eq!(
            (
                &check["value"],
                &check["records"],
                &check["records_needed"],
                &check["enough_history"]
            ),
            (&json!("checked"), &json!(9), &json!(6), &json!(true))
        );
    }

    #[test]
    fn an_optimizer_that_was_not_read_makes_the_update_kind_unavailable_and_says_why() {
        let none: [(&str, f64); 0] = [];
        let rows: Vec<UpdateTestRow> = (0..3).map(|step| (step, 1.0, &none[..])).collect();
        let said = r#"{"update_not_read":{"SGD":"SGD is not Adam or AdamW"}}"#;
        let detectors = judge(0, &[update_rows(&rows, &[None, Some(said), None])], &usual()).unwrap();
        let not_checked = detectors.not_checked();
        let entry = not_checked.iter().find(|item| item["check"] == "update_size").unwrap();
        assert_eq!(
            entry,
            &json!({
                "check": "update_size", "kinds": ["update_spike"], "optimizers": ["SGD"],
                "reason": "SGD: SGD is not Adam or AdamW", "step": 1,
            })
        );
        let check = &detectors.kind_checks()["update_spike"];
        assert_eq!(
            (&check["value"], &check["records"], &check["reason"]),
            (
                &json!("unavailable"),
                &json!(0),
                &json!("SGD: SGD is not Adam or AdamW")
            )
        );
        // A sender from before the column, or a run that never asked: not configured, no entry.
        let plain = judge(0, &[key_rows(&steps("run", 0, 1, 0..3))], &usual()).unwrap();
        assert_eq!(plain.kind_checks()["update_spike"]["value"], "not_configured");
        assert!(plain.not_checked().iter().all(|item| item["check"] != "update_size"));
    }

    #[test]
    fn a_checked_replica_drift_says_what_equal_replicas_do_not_rule_out() {
        let plain = judge(0, &[key_rows(&steps("run", 0, 1, 0..3))], &usual()).unwrap();
        let checks = plain.kind_checks();
        // Only a checked kind carries a note, and only the kinds of `KIND_NOTES`.
        assert!(checks
            .as_object()
            .unwrap()
            .values()
            .all(|check| check.get("note").is_none()));
        assert!(KIND_NOTES[0]
            .1
            .contains("equal replicas do not rule out silent data corruption"));
    }

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
        let cutoffs = resolve(&[(1, 0), (2, 100), (3, 50)]);
        assert_eq!(cutoffs, BTreeMap::from([(1, Some(50)), (2, Some(50)), (3, None)]));
        assert_eq!(resolve(&[(7, 0)]), BTreeMap::from([(7, None)]));
        // An attempt that starts at the largest step still cuts the one before it (RS16).
        assert_eq!(
            resolve(&[(1, 0), (2, i64::MAX)]),
            BTreeMap::from([(1, Some(i64::MAX)), (2, None)])
        );
        // The order given, not the numbers (contract 1): 900 first, then 2 resumed at 30.
        assert_eq!(
            resolve(&[(900, 0), (2, 30)]),
            BTreeMap::from([(900, Some(30)), (2, None)])
        );
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

    #[test]
    fn a_probe_reads_its_attention_in_order_and_skips_the_rest() {
        let text = r#"{"histograms":{"b":{"counts":[1]}},"attention":{"z":{"entropy_share":[0.25,"nan",0.75,true],
            "max_logit":["-inf",3,2.5],"map":[[0.1]]},"a":{"entropy_share":"x","max_logit":{}},"m":[1,2]}}"#;
        let probe: Probe = serde_json::from_str(text).unwrap();
        let Members(attention) = probe.attention.unwrap();
        assert_eq!(
            attention.iter().map(|(name, _)| name.as_ref()).collect::<Vec<_>>(),
            ["z", "a", "m"]
        );
        let z = &attention[0].1;
        // NaN and `true` left out, as Python's `_finite` reads them (a bool is not a number,
        // DET-16): (0.25 + 0.75) / 2.
        assert_eq!((z.share, z.logit, z.heads), (Some(0.5), Some(3.0), 2));
        assert_eq!(
            (attention[1].1.share, attention[1].1.logit, attention[2].1.heads),
            (None, None, 0)
        );
        let failed: Probe = serde_json::from_str(r#"{"error":"RuntimeError: x","attention":{}}"#).unwrap();
        assert_eq!(failed.error, Some(json!("RuntimeError: x")));
        // An attention that is not an object is none, and the rest of the probe is read (BUG-10).
        let listed: Probe = serde_json::from_str(r#"{"error":"x","attention":[1]}"#).unwrap();
        assert!(listed.attention.is_none());
        assert_eq!(listed.error, Some(json!("x")));
    }

    #[test]
    fn attention_rules_at_their_edges() {
        assert!(attention_hit("attention_collapse", 0.2, 0.4) && !attention_hit("attention_collapse", 0.21, 0.4));
        assert!(!attention_hit("attention_collapse", 0.0, 0.0));
        assert!(attention_hit("attention_uniform", 0.98, 0.5) && !attention_hit("attention_uniform", 0.979, 0.5));
        assert!(!attention_hit("attention_uniform", 0.99, 0.875));
        assert!(
            attention_hit("attention_logit_growth", 20.0, 10.0) && !attention_hit("attention_logit_growth", 19.9, 10.0)
        );
    }

    #[tokio::test]
    async fn replay_reads_each_parts_attempt_from_under_its_rank_whatever_the_store_root_holds() {
        // RS03 (lane U6, Nov 19, 2026). A store rooted at `.../attempt=7`: attempt 1 reaches step
        // 99 and attempt 2 resumes at step 50. The resume cuts attempt 1 at 50, so 60 rows count,
        // and the newest attempt read is 2.
        use object_store::memory::InMemory;
        let storage = crate::storage::Storage::new(Arc::new(InMemory::new()), "store/attempt=7".into());
        storage
            .write_part("prefix", 0, 1, &[key_rows(&steps("prefix", 0, 1, 0..100))], 0, 99)
            .await
            .unwrap();
        storage
            .write_part("prefix", 0, 2, &[key_rows(&steps("prefix", 0, 2, 50..60))], 50, 59)
            .await
            .unwrap();
        let detectors = replay(&storage, "prefix", 0, &usual()).await.unwrap();
        assert_eq!(
            (detectors.attempt(), detectors.last_step(), detectors.rows()),
            (2, Some(59), 60)
        );
        assert!(
            !detectors
                .incidents()
                .iter()
                .any(|incident| incident["kind"] == "repeated_steps"),
            "{:?}",
            detectors.incidents()
        );
    }
}
