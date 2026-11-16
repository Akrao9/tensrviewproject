//! The GPU memory, NVML and data-wait detectors: `evidence_system.py` in Rust.
//!
//! The readings arrive as two `f64` columns on every step (`data_wait`, `step_time`) and one
//! JSON column on the last row of each flush (`system_json`), so the JSON is parsed once per
//! flush, not per row. Every rule reads forward only and is dated at the row that meets it,
//! so the state after any prefix equals Python's evidence on that prefix. The readings
//! belong to one process: the rules start again when the attempt changes.
//!
//! `memory_imbalance` compares ranks and is Python's alone (`detect_memory_imbalance`, in
//! the all-ranks view); this module gives it each rank's latest reading
//! ([`System::memory`]), written into the rank's report as `memory`.

use std::collections::{HashSet, VecDeque};

use arrow::array::{Float64Array, StringArray};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Value};

use super::{column, median, text, value};

// The same values as the constants of the same names in evidence_system.py.
pub const MEMORY_GROWTH_READINGS: usize = 6;
pub const MEMORY_GROWTH_FRACTION: f64 = 0.05;
pub const MEMORY_GROWTH_BYTES: i64 = 64 * 1024 * 1024;
pub const MEMORY_RESERVED_FRACTION: f64 = 0.95;
pub const THROTTLE_REPORTED: [&str; 4] = ["sw_thermal", "hw_thermal", "hw_slowdown", "hw_power_brake"];
pub const INPUT_WINDOW: usize = 20;
pub const INPUT_SHARE: f64 = 0.3;
pub const INPUT_RISE: f64 = 0.2;

const MEMORY_KINDS: [&str; 3] = ["memory_growth", "memory_pressure", "allocator_retries"];
const NVML_KINDS: [&str; 2] = ["hardware_errors", "gpu_throttled"];

/// The NVML metrics a kind reads, as `_METRIC_KINDS`.
fn metric_kinds(metric: &str) -> Option<&'static [&'static str]> {
    match metric {
        "ecc_uncorrected" | "xid" => Some(&["hardware_errors"]),
        "throttle" => Some(&["gpu_throttled"]),
        _ => None,
    }
}

/// Python's `_count`: a JSON integer, never a boolean or a float.
fn count(value: Option<&Value>) -> Option<i64> {
    value.filter(|value| value.is_i64()).and_then(Value::as_i64)
}

/// Python's `_number`: a finite JSON number, a boolean as 0 or 1, nothing else.
fn number(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Bool(flag) => Some(f64::from(u8::from(*flag))),
        Value::Number(number) => number.as_f64().filter(|number| number.is_finite()),
        _ => None,
    }
}

fn or_null<T: Into<Value>>(value: Option<T>) -> Value {
    value.map_or(Value::Null, Into::into)
}

pub(super) struct SystemColumns<'a> {
    data_wait: Option<&'a Float64Array>,
    step_time: Option<&'a Float64Array>,
    system: Option<&'a StringArray>,
}

impl<'a> SystemColumns<'a> {
    pub(super) fn new(batch: &'a RecordBatch) -> Self {
        Self {
            data_wait: column(batch, "data_wait"),
            step_time: column(batch, "step_time"),
            system: column(batch, "system_json"),
        }
    }
}

#[derive(Default)]
pub(super) struct System {
    attempt: Option<i64>,
    /// Allocated bytes at the latest readings that rose one after another, as (step, bytes).
    growth: VecDeque<(i64, i64)>,
    ecc_start: Option<i64>,
    /// The latest `INPUT_WINDOW` steps' (share, wait, step time).
    shares: VecDeque<(f64, f64, f64)>,
    baseline: Option<f64>,
    found: [Option<Value>; 6],
    noted: HashSet<(&'static str, String)>,
    not_checked: Vec<Value>,
    latest: Option<Value>,
    scratch: Vec<f64>,
    /// How many rows carried a `cuda` reading, an `nvml` reading, and a usable data wait.
    inputs: [u64; 3],
}

impl System {
    /// Reads one row of attempt `attempt`. Returns which kinds were first found on it, one
    /// bit per kind in [`KINDS`] order, so the caller can give them what was logged there.
    pub(super) fn row(&mut self, attempt: i64, step: i64, columns: &SystemColumns, row: usize) -> u8 {
        if self.attempt != Some(attempt) {
            self.attempt = Some(attempt);
            self.growth.clear();
            self.ecc_start = None;
            self.shares.clear();
            self.baseline = None;
        }
        let before = self.found_mask();
        let system: Option<Value> = text(columns.system, row).and_then(|text| serde_json::from_str(text).ok());
        let part = |name: &str| {
            system
                .as_ref()
                .and_then(|system| system.get(name))
                .filter(|part| part.is_object())
        };
        let (cuda, nvml) = (part("cuda"), part("nvml"));
        self.inputs[0] += u64::from(cuda.is_some());
        self.inputs[1] += u64::from(nvml.is_some());
        self.note_unavailable(step, cuda, nvml);
        if let Some(cuda) = cuda {
            self.memory_reading(step, cuda);
        }
        if let Some(nvml) = nvml {
            self.nvml_reading(step, nvml);
        }
        let wait = value(columns.data_wait, row).filter(|wait| wait.is_finite());
        let took = value(columns.step_time, row).filter(|took| took.is_finite());
        if let (Some(wait), Some(took)) = (wait, took) {
            if took > 0.0 && wait >= 0.0 {
                self.inputs[2] += 1;
                self.input_reading(step, wait, took);
            }
        }
        self.found_mask() & !before
    }

    /// How many rows carried a `cuda` reading (one that could not be taken counts: it was
    /// asked for), an `nvml` reading, and a data wait with a step time.
    pub(super) fn inputs(&self) -> [u64; 3] {
        self.inputs
    }

    fn found_mask(&self) -> u8 {
        self.found
            .iter()
            .enumerate()
            .fold(0, |mask, (at, found)| mask | (u8::from(found.is_some()) << at))
    }

    fn memory_reading(&mut self, step: i64, cuda: &Value) {
        let [allocated, reserved, total] = ["allocated", "reserved", "total"].map(|key| count(cuda.get(key)));
        if let Some(allocated) = allocated {
            self.latest = Some(json!({
                "step": step,
                "allocated": allocated,
                "reserved": or_null(reserved),
                "peak_allocated": or_null(count(cuda.get("peak_allocated"))),
                "total": or_null(total),
            }));
            if self.growth.back().is_some_and(|&(_, last)| allocated <= last) {
                self.growth.clear();
            }
            self.growth.push_back((step, allocated));
            if self.growth.len() > MEMORY_GROWTH_READINGS {
                self.growth.pop_front();
            }
            if self.found[0].is_none() && self.growth.len() == MEMORY_GROWTH_READINGS {
                let (first_step, first) = self.growth[0];
                let rise = allocated - first;
                if rise >= MEMORY_GROWTH_BYTES && rise as f64 >= MEMORY_GROWTH_FRACTION * first as f64 {
                    let per_step = rise as f64 / (step - first_step) as f64;
                    self.found[0] = Some(json!({
                        "kind": "memory_growth",
                        "step": step,
                        "first_step": first_step,
                        "allocated_first": first,
                        "allocated": allocated,
                        "readings": MEMORY_GROWTH_READINGS,
                        "bytes_per_step": per_step,
                        "total": or_null(total),
                        "steps_to_total": or_null(total.map(|total| (total - allocated).max(0) as f64 / per_step)),
                    }));
                }
            }
        }
        if let (None, Some(reserved), Some(total)) = (&self.found[1], reserved, total) {
            if total > 0 && reserved as f64 >= MEMORY_RESERVED_FRACTION * total as f64 {
                self.found[1] = Some(json!({
                    "kind": "memory_pressure",
                    "step": step,
                    "reserved": reserved,
                    "total": total,
                    "fraction": reserved as f64 / total as f64,
                    "allocated": or_null(allocated),
                }));
            }
        }
        let (retries, ooms) = (count(cuda.get("alloc_retries")), count(cuda.get("ooms")));
        if self.found[2].is_none() && (retries.unwrap_or(0) > 0 || ooms.unwrap_or(0) > 0) {
            self.found[2] = Some(json!({
                "kind": "allocator_retries",
                "step": step,
                "alloc_retries": or_null(retries),
                "ooms": or_null(ooms),
            }));
        }
    }

    fn nvml_reading(&mut self, step: i64, nvml: &Value) {
        let ecc = count(nvml.get("ecc_uncorrected"));
        if self.ecc_start.is_none() {
            self.ecc_start = ecc;
        }
        let xid: Vec<i64> = nvml
            .get("xid")
            .and_then(Value::as_array)
            .map(|xid| xid.iter().filter_map(|x| count(Some(x))).collect())
            .unwrap_or_default();
        let rose = matches!((ecc, self.ecc_start), (Some(ecc), Some(start)) if ecc > start);
        if self.found[3].is_none() && (!xid.is_empty() || rose) {
            self.found[3] = Some(json!({
                "kind": "hardware_errors",
                "step": step,
                "xid": xid,
                "ecc_uncorrected": or_null(ecc),
                "ecc_uncorrected_at_start": or_null(self.ecc_start),
            }));
        }
        let reasons: Vec<&str> = nvml
            .get("throttle")
            .and_then(Value::as_array)
            .map(|reasons| reasons.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if self.found[4].is_none() && reasons.iter().any(|reason| THROTTLE_REPORTED.contains(reason)) {
            let mut incident = json!({"kind": "gpu_throttled", "step": step, "reasons": reasons});
            for key in ["temperature_c", "sm_clock_mhz", "power_w", "power_limit_w"] {
                incident[key] = or_null(number(nvml.get(key)));
            }
            self.found[4] = Some(incident);
        }
    }

    fn input_reading(&mut self, step: i64, wait: f64, took: f64) {
        self.shares.push_back((wait / took, wait, took));
        if self.shares.len() > INPUT_WINDOW {
            self.shares.pop_front();
        }
        if self.shares.len() < INPUT_WINDOW {
            return;
        }
        let mut medians = [0.0; 3];
        for (at, median_of) in medians.iter_mut().enumerate() {
            self.scratch.clear();
            self.scratch
                .extend(self.shares.iter().map(|&(share, wait, took)| [share, wait, took][at]));
            *median_of = median(&mut self.scratch);
        }
        let share = medians[0];
        let baseline = *self.baseline.get_or_insert(share);
        let rule = if share >= INPUT_SHARE {
            "share"
        } else if share - baseline >= INPUT_RISE {
            "rise"
        } else {
            return;
        };
        if self.found[5].is_none() {
            self.found[5] = Some(json!({
                "kind": "input_starvation",
                "step": step,
                "rule": rule,
                "median_share": share,
                "baseline_share": baseline,
                "median_data_wait": medians[1],
                "median_step_time": medians[2],
                "window": INPUT_WINDOW,
            }));
        }
    }

    fn note_unavailable(&mut self, step: i64, cuda: Option<&Value>, nvml: Option<&Value>) {
        let mut note = |check: &'static str, metric: &str, kinds: &[&str], reason: &Value| {
            if !self.noted.insert((check, metric.to_owned())) {
                return;
            }
            // Python writes `str(reason)`: a string as it is, anything else as its repr. The
            // capture writes strings; other values are told apart only by being non-strings.
            let reason = reason.as_str().map_or_else(|| reason.to_string(), str::to_owned);
            let mut entry = json!({"check": check, "kinds": kinds, "reason": reason, "step": step});
            if !metric.is_empty() {
                entry["metric"] = json!(metric);
            }
            self.not_checked.push(entry);
        };
        if let Some(reason) = cuda.and_then(|cuda| cuda.get("unavailable")) {
            note("cuda_memory", "", &MEMORY_KINDS, reason);
        }
        if let Some(unavailable) = nvml.and_then(|nvml| nvml.get("unavailable")) {
            match unavailable.as_object() {
                Some(metrics) => {
                    for (metric, reason) in metrics {
                        if let Some(kinds) = metric_kinds(metric) {
                            note("nvml", metric, kinds, reason);
                        }
                    }
                }
                None => note("nvml", "", &NVML_KINDS, unavailable),
            }
        }
    }

    /// Gives the kinds marked in `new` (see [`row`](Self::row)) what was logged on their step.
    pub(super) fn attach(&mut self, new: u8, context: &Value) {
        for (at, found) in self.found.iter_mut().enumerate() {
            if new & (1 << at) != 0 {
                if let Some(incident) = found {
                    incident["scalars"] = context.clone();
                }
            }
        }
    }

    pub(super) fn incidents(&self, out: &mut Vec<Value>) {
        out.extend(self.found.iter().flatten().cloned());
    }

    pub(super) fn not_checked(&self, out: &mut Vec<Value>) {
        out.extend(self.not_checked.iter().cloned());
    }

    /// The latest memory reading, as Python's `latest_memory`, or null.
    pub(super) fn memory(&self) -> Value {
        self.latest.clone().unwrap_or(Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    fn batch(waits: &[Option<f64>], times: &[Option<f64>], systems: &[Option<&str>]) -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("data_wait", DataType::Float64, true),
            Field::new("step_time", DataType::Float64, true),
            Field::new("system_json", DataType::Utf8, true),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float64Array::from(waits.to_vec())),
                Arc::new(Float64Array::from(times.to_vec())),
                Arc::new(StringArray::from(systems.to_vec())),
            ],
        )
        .unwrap()
    }

    fn run(system: &mut System, attempt: i64, first: i64, rows: &RecordBatch) -> Vec<u8> {
        let columns = SystemColumns::new(rows);
        (0..rows.num_rows())
            .map(|row| system.row(attempt, first + row as i64, &columns, row))
            .collect()
    }

    #[test]
    fn json_values_read_as_python_reads_them() {
        let parsed: Value = serde_json::from_str(r#"[5, 5.0, true, "5", null]"#).unwrap();
        let items: Vec<Option<&Value>> = (0..5).map(|at| parsed.get(at)).collect();
        assert_eq!(
            items.iter().map(|v| count(*v)).collect::<Vec<_>>(),
            [Some(5), None, None, None, None]
        );
        assert_eq!(
            items.iter().map(|v| number(*v)).collect::<Vec<_>>(),
            [Some(5.0), Some(5.0), Some(1.0), None, None]
        );
    }

    #[test]
    fn a_leak_is_a_rise_at_every_reading_not_a_jump() {
        let gib = 1i64 << 30;
        let reading = |allocated: i64| format!(r#"{{"cuda":{{"allocated":{allocated},"total":{}}}}}"#, 16 * gib);
        // One jump then flat: no trend.
        let flat: Vec<String> = [1, 2, 2, 2, 2, 2, 2, 2].iter().map(|g| reading(g * gib)).collect();
        let mut system = System::default();
        let rows = batch(
            &[None; 8],
            &[None; 8],
            &flat.iter().map(|s| Some(s.as_str())).collect::<Vec<_>>(),
        );
        assert!(run(&mut system, 1, 0, &rows).iter().all(|&new| new == 0));
        // Six readings each above the last: found at the sixth.
        let rising: Vec<String> = (0..6).map(|k| reading(gib + k * 100 * (1 << 20))).collect();
        let rows = batch(
            &[None; 6],
            &[None; 6],
            &rising.iter().map(|s| Some(s.as_str())).collect::<Vec<_>>(),
        );
        let mut system = System::default();
        assert_eq!(run(&mut system, 1, 0, &rows), [0, 0, 0, 0, 0, 1]);
        // A new attempt starts the trend again.
        let mut system = System::default();
        let columns = SystemColumns::new(&rows);
        for row in 0..6 {
            system.row(if row < 3 { 1 } else { 2 }, row as i64, &columns, row);
        }
        assert!(system.found[0].is_none());
    }

    #[test]
    fn the_found_slots_are_in_ported_order() {
        let mut system = System::default();
        let kinds = [
            r#"{"cuda":{"allocated":1}}"#,
            r#"{"cuda":{"reserved":95,"total":100}}"#,
            r#"{"cuda":{"ooms":1}}"#,
            r#"{"nvml":{"xid":[79]}}"#,
            r#"{"nvml":{"throttle":["hw_thermal"]}}"#,
        ];
        let rows = batch(&[None; 5], &[None; 5], &kinds.map(Some));
        assert_eq!(run(&mut system, 1, 0, &rows), [0, 1 << 1, 1 << 2, 1 << 3, 1 << 4]);
        let mut out = Vec::new();
        system.incidents(&mut out);
        let kinds: Vec<&str> = out.iter().map(|incident| incident["kind"].as_str().unwrap()).collect();
        let ported = super::super::PORTED;
        let first = ported.iter().position(|kind| *kind == "memory_growth").unwrap();
        assert_eq!(kinds, ported[first + 1..first + 5]);
    }

    #[test]
    fn starvation_needs_a_full_window() {
        let n = INPUT_WINDOW;
        let mut system = System::default();
        let rows = batch(&vec![Some(0.5); n], &vec![Some(1.0); n], &vec![None; n]);
        let new = run(&mut system, 1, 0, &rows);
        assert_eq!(new[..n - 1], vec![0; n - 1][..]);
        assert_eq!(new[n - 1], 1 << 5);
    }
}
