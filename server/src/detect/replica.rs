//! Replica observations compared across the ranks of one run: sections 2 to 7 of
//! `REPLICA_OBSERVATION_CONTRACT.md`, which is the reference, with its cases in
//! `sdk/tests/fixtures/replica/` (lane Z3 of the SDK-22 wave, Nov 20, 2026).
//!
//! One type, [`Comparison`], holds what the rules need for one run and gives the result the
//! fixtures spell (`steps`, `replica_drift`, `legacy`, and `attribution`, section 4a's D1, when it
//! has an entry). [`compare`] feeds it a finished list of
//! records, as a reader of files would; the server feeds it each part's rows as they are stored
//! (`Ingest::flush`) and writes the result into the run's `watchdog.json` under `replica`. Both
//! paths judge through the same code, so they cannot read the rules differently.
//!
//! **Which attempt keeps a launch** (contract 2): the first by contract 1's order. The server has
//! that order per rank (`attempts.json`, `AttemptOrder`); [`Comparison::insert`] asks for an
//! attempt's place in it. [`compare`] takes the order from the records themselves: a rank's
//! records are listed in contract 1's order, so an attempt's place is where it first appears.
//!
//! **What is held, and its bounds** ([`Bounds`]). Per run:
//! - Observations (tens of kilobytes each) are held only while they can still be compared: for a
//!   cell (launch, group, step) still `insufficient`, and for a cell judged `complete` until the
//!   byte bound needs the room (kept so a backlog of an earlier attempt can still be paired in its
//!   place, contract 2). Rows that can never be compared (no launch, a group fault, an attempt set
//!   aside) keep no observation. At most `held_bytes` of observation JSON is held: past it, the
//!   judged cells' observations go first, least recently touched step first; then the unfinished
//!   ones, whose launch and step are then *let go*: their entries stay as they read (`insufficient`,
//!   with the ranks then missing), and a row that arrives for them later is counted, not compared.
//! - At most `listed_steps` step numbers are listed, the most recently touched, and no more than
//!   `listed_bytes` of their entries as JSON (the object is rewritten at every status interval). A
//!   step past either is forgotten; its incidents are kept (the first `kept_incidents`) and the rest
//!   counted, and a row that arrives later for a launch it listed is counted, not listed again.
//! - What any bound dropped is said under `omitted`, a key the result has only when something was:
//!   `steps` (step numbers no longer listed), `replica_drift` (incidents no longer listed),
//!   `late_rows` (rows that arrived for a launch at a step no longer listed: counted, not listed), and
//!   `frozen` (each launch and step whose entries no longer change, with why and how many rows
//!   arrived for it that were not compared). Nothing dropped ever reads as `complete` or `agree`.
//!
//! **A restart** ([`Comparison::restore`]): the result last written is taken up again, each launch
//! and step in it frozen as written, so no verdict and no incident is lost. Observations held for
//! unfinished cells were in memory only: a rank that reports for such a cell after the restart is
//! counted under `omitted.frozen` (`why: "restart"`), and the cell keeps reading `insufficient`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int64Array, ListArray, StringArray};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Map, Value};

use super::{column, MapColumn};

/// The version of the result object (`"v"` in `watchdog.json`'s `replica`).
pub const RESULT_VERSION: u64 = 1;
/// Observation JSON held per run, judged and unfinished together.
pub const DEFAULT_HELD_BYTES: usize = 32 * 1024 * 1024;
/// Step numbers listed per run.
pub const DEFAULT_LISTED_STEPS: usize = 256;
/// JSON of the listed steps' entries, incidents and legacy entries per run: the object is rewritten
/// at every status interval. At 290 parameters a step where every one drifts lists about 49 KB.
pub const DEFAULT_LISTED_BYTES: usize = 1024 * 1024;
/// Incidents kept per run for steps no longer listed.
pub const DEFAULT_KEPT_INCIDENTS: usize = 64;
/// Entries of `omitted.frozen` listed; the rest are counted in `frozen_more`.
const FROZEN_LISTED: usize = 64;

/// What one run's comparison may hold. See the module's documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    pub held_bytes: usize,
    pub listed_steps: usize,
    pub listed_bytes: usize,
    pub kept_incidents: usize,
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            held_bytes: DEFAULT_HELD_BYTES,
            listed_steps: DEFAULT_LISTED_STEPS,
            listed_bytes: DEFAULT_LISTED_BYTES,
            kept_incidents: DEFAULT_KEPT_INCIDENTS,
        }
    }
}

impl Bounds {
    /// No bound: what a reader of a finished list of records applies.
    pub fn none() -> Self {
        Self {
            held_bytes: usize::MAX,
            listed_steps: usize::MAX,
            listed_bytes: usize::MAX,
            kept_incidents: usize::MAX,
        }
    }
}

// --- records --------------------------------------------------------------------------------

/// One parameter of an observation.
#[derive(Debug)]
struct Param {
    shape: Value,
    dtype: Value,
    fp: Vec<f64>,
    sig: Value,
}

/// What a rank observed, as the comparison reads it (contract 1).
#[derive(Debug, Default)]
pub struct Observation {
    device: Value,
    projection: Value,
    signature: Value,
    params: BTreeMap<String, Param>,
    unread: BTreeSet<String>,
}

/// A record's `replica`: its launch and group, and the observation.
#[derive(Debug)]
pub struct Observed {
    launch: Option<String>,
    /// None when it is not a list of ranks (or the observation is of another version): such a
    /// record names no group it can be compared in.
    group: Option<Vec<i64>>,
    observation: Observation,
    /// The observation's size as JSON text, for the byte bound.
    bytes: usize,
}

/// A record's `replica_spread`: what it covers (`replica_spread_scope`, contract 1), the
/// parameters it holds and those above the tolerance or not finite.
#[derive(Debug, Clone)]
pub struct Spread {
    scope: Option<(String, Vec<i64>)>,
    covered: BTreeSet<String>,
    over: BTreeSet<String>,
}

/// One step record as the comparison reads it.
#[derive(Debug)]
pub struct Record {
    pub rank: i64,
    pub attempt: i64,
    pub step: i64,
    observed: Option<Observed>,
    spread: Option<Spread>,
}

impl Record {
    /// Whether the record holds anything the comparison reads.
    pub fn is_empty(&self) -> bool {
        self.observed.is_none() && self.spread.is_none()
    }
}

/// A fingerprint: a number, or the string `"nan"`, `"inf"` or `"-inf"` (contract 1). Anything else
/// is not finite.
fn number(value: &Value) -> f64 {
    match value {
        Value::Number(number) => number.as_f64().unwrap_or(f64::NAN),
        Value::String(text) => text.trim().parse().unwrap_or(f64::NAN),
        _ => f64::NAN,
    }
}

fn ranks(value: &Value) -> Option<Vec<i64>> {
    value.as_array()?.iter().map(Value::as_i64).collect()
}

fn observed(replica: &Value) -> Observed {
    let bytes = replica.to_string().len();
    let launch = replica["launch"].as_str().map(str::to_string);
    let mut observation = Observation {
        device: replica["device"].clone(),
        projection: replica["projection"].clone(),
        signature: replica["signature"].clone(),
        ..Observation::default()
    };
    // A record of another version is not read: it names no group it can be compared in.
    let group = (replica["v"] == 1).then(|| ranks(&replica["group"])).flatten();
    for (name, _) in replica["unread"].as_object().into_iter().flatten() {
        observation.unread.insert(name.clone());
    }
    for (name, param) in replica["params"].as_object().into_iter().flatten() {
        let fp: Option<Vec<f64>> = param["fp"].as_array().map(|fp| fp.iter().map(number).collect());
        match fp.filter(|fp| fp.len() >= 2) {
            Some(fp) => {
                observation.params.insert(
                    name.clone(),
                    Param {
                        shape: param["shape"].clone(),
                        dtype: param["dtype"].clone(),
                        fp,
                        sig: param["sig"].clone(),
                    },
                );
            }
            // An entry with no readable fingerprints cannot be compared: read as unread.
            None => {
                observation.unread.insert(name.clone());
            }
        }
    }
    Observed {
        launch,
        group,
        observation,
        bytes,
    }
}

fn scope(value: Option<&Value>) -> Option<(String, Vec<i64>)> {
    let value = value?;
    Some((value["launch"].as_str()?.to_string(), ranks(&value["group"])?))
}

fn spread(
    table: impl IntoIterator<Item = (String, Vec<f64>)>,
    scope: Option<(String, Vec<i64>)>,
    tolerance: f64,
) -> Spread {
    let mut covered = BTreeSet::new();
    let mut over = BTreeSet::new();
    for (name, values) in table {
        // Section 6, as `ReplicaDrift::row` reads a spread on its own record.
        if values.iter().any(|value| !value.is_finite() || *value > tolerance) {
            over.insert(name.clone());
        }
        covered.insert(name);
    }
    Spread { scope, covered, over }
}

/// A record given as JSON (a fixture's, or a line of a rank's file); None without a rank or step.
pub fn record_from_json(record: &Value, tolerance: f64) -> Option<Record> {
    let table = record.get("replica_spread").map(|table| {
        table
            .as_object()
            .into_iter()
            .flatten()
            .map(|(name, values)| {
                let values = values.as_array().map(|values| values.iter().map(number).collect());
                (name.clone(), values.unwrap_or_default())
            })
            .collect::<Vec<_>>()
    });
    Some(Record {
        rank: record["rank"].as_i64()?,
        attempt: record["attempt"].as_i64().unwrap_or(0),
        step: record["step"].as_i64()?,
        observed: record.get("replica").map(observed),
        spread: table.map(|table| spread(table, scope(record.get("replica_spread_scope")), tolerance)),
    })
}

/// The records of one stored batch of `rank` that hold an observation or a gathered spread.
/// `replica` and `replica_spread_scope` travel in `extra_json`, which is parsed only on a row that
/// names one of them. `replica_spread` has its own column (`flight.py`), read as the per-rank rule
/// reads it; a spread that did not fit the column (a value that is not a number) travels in
/// `extra_json` and is read from there.
pub fn records_from_batch(batch: &RecordBatch, rank: i64, tolerance: f64) -> Vec<Record> {
    let (Some(steps), Some(attempts)) = (
        column::<Int64Array>(batch, "step"),
        column::<Int64Array>(batch, "attempt"),
    ) else {
        return Vec::new();
    };
    let extra: Option<&StringArray> = column(batch, "extra_json");
    let spreads: Option<MapColumn<ListArray>> = MapColumn::new(batch, "replica_spread");
    let numbers = spreads
        .as_ref()
        .and_then(|spreads| spreads.values.values().as_any().downcast_ref::<Float64Array>());
    let mut out = Vec::new();
    for row in 0..batch.num_rows() {
        if steps.is_null(row) || attempts.is_null(row) {
            continue;
        }
        let parsed: Option<Value> = extra
            .filter(|extra| !extra.is_null(row) && extra.value(row).contains("\"replica"))
            .and_then(|extra| serde_json::from_str(extra.value(row)).ok());
        let column_table = match (&spreads, numbers) {
            (Some(spreads), Some(numbers)) if !spreads.map.is_null(row) => {
                let offsets = spreads.values.value_offsets();
                let table: Vec<(String, Vec<f64>)> = spreads
                    .row(row)
                    .filter(|entry| !spreads.values.is_null(*entry))
                    .map(|entry| {
                        let values = (offsets[entry] as usize..offsets[entry + 1] as usize)
                            .map(|at| {
                                if numbers.is_null(at) {
                                    f64::NAN
                                } else {
                                    numbers.value(at)
                                }
                            })
                            .collect();
                        (spreads.keys.value(entry).to_string(), values)
                    })
                    .collect();
                Some(table)
            }
            _ => None,
        };
        let mut record = Record {
            rank,
            attempt: attempts.value(row),
            step: steps.value(row),
            observed: None,
            spread: None,
        };
        let scope = scope(parsed.as_ref().and_then(|parsed| parsed.get("replica_spread_scope")));
        if let Some(table) = column_table {
            record.spread = Some(spread(table, scope, tolerance));
        } else if let Some(table) = parsed.as_ref().and_then(|parsed| parsed.get("replica_spread")) {
            let mut json = json!({"rank": rank, "step": record.step, "replica_spread": table});
            if let Some(given) = parsed.as_ref().and_then(|parsed| parsed.get("replica_spread_scope")) {
                json["replica_spread_scope"] = given.clone();
            }
            record.spread = record_from_json(&json, tolerance).and_then(|parsed| parsed.spread);
        }
        record.observed = parsed.as_ref().and_then(|parsed| parsed.get("replica")).map(observed);
        if !record.is_empty() {
            out.push(record);
        }
    }
    out
}

// --- one cell -------------------------------------------------------------------------------

/// Python 3.12's `sum` of floats (Neumaier's compensated sum), so a spread on the edge of the
/// tolerance reads the same in both engines.
fn python_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut total, mut compensation) = (0.0_f64, 0.0_f64);
    for value in values {
        let next = total + value;
        if total.abs() >= value.abs() {
            compensation += (total - next) + value;
        } else {
            compensation += (value - next) + total;
        }
        total = next;
    }
    if compensation != 0.0 && compensation.is_finite() {
        total += compensation;
    }
    total
}

/// The median of section 4a: the middle value, or with an even number the mean of the two middle ones.
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let half = values.len() / 2;
    if values.len() % 2 == 1 {
        values[half]
    } else {
        (values[half - 1] + values[half]) / 2.0
    }
}

/// Section 4a by fingerprints, for one parameter that drifted: `rows` are the compared ranks'
/// fingerprints, each cut to the columns the drift was decided on, sorted by rank.
fn by_fingerprints(rows: &[(i64, &[f64])], tolerance: f64) -> Value {
    if rows.len() == 2 {
        return json!({"verdict": "inconclusive", "why": "two_ranks"});
    }
    let finite: Vec<&(i64, &[f64])> = rows
        .iter()
        .filter(|(_, row)| row.iter().all(|value| value.is_finite()))
        .collect();
    let mut apart: BTreeSet<i64> = rows
        .iter()
        .filter(|(_, row)| !row.iter().all(|value| value.is_finite()))
        .map(|(rank, _)| *rank)
        .collect();
    if !finite.is_empty() {
        // Measured as section 4 measures a spread: by the mean of the finite ranks' norms.
        let scale = (python_sum(finite.iter().map(|(_, row)| row[0])) / finite.len() as f64).max(1e-12);
        let width = finite.iter().map(|(_, row)| row.len()).min().unwrap_or(0);
        for at in 0..width {
            let middle = median(&mut finite.iter().map(|(_, row)| row[at]).collect::<Vec<f64>>());
            apart.extend(
                finite
                    .iter()
                    .filter(|(_, row)| (row[at] - middle).abs() / scale > tolerance)
                    .map(|(rank, _)| *rank),
            );
        }
    }
    if apart.is_empty() {
        return json!({"verdict": "inconclusive", "why": "no_outlier"});
    }
    if 2 * (rows.len() - apart.len()) <= rows.len() {
        return json!({"verdict": "inconclusive", "why": "no_majority"});
    }
    json!({"verdict": "attributed", "outliers": apart})
}

/// Section 4a by signatures, for one parameter whose signatures differ: one signature held by a
/// strict majority names the other ranks. `rows` are sorted by rank.
fn by_signatures(rows: &[(i64, &Value)]) -> Value {
    if rows.len() == 2 {
        return json!({"verdict": "inconclusive", "why": "two_ranks"});
    }
    let held = rows
        .iter()
        .map(|(_, sig)| *sig)
        .find(|sig| 2 * rows.iter().filter(|(_, other)| other == sig).count() > rows.len());
    match held {
        Some(sig) => {
            let outliers: Vec<i64> = rows
                .iter()
                .filter(|(_, other)| *other != sig)
                .map(|(rank, _)| *rank)
                .collect();
            json!({"verdict": "attributed", "outliers": outliers})
        }
        None => json!({"verdict": "inconclusive", "why": "no_majority"}),
    }
}

/// An attribution entry's place in the list (section 4a): by step, launch, group, parameter, then `by`.
fn attribution_order(entry: &Value) -> (i64, String, Vec<i64>, String, String) {
    (
        entry["step"].as_i64().unwrap_or(i64::MIN),
        entry["launch"].as_str().unwrap_or_default().to_string(),
        ranks(&entry["group"]).unwrap_or_default(),
        entry["parameter"].as_str().unwrap_or_default().to_string(),
        entry["by"].as_str().unwrap_or_default().to_string(),
    )
}

/// Sections 3 and 4 for one launch, group and step with complete coverage: `_compare` in
/// `test_replica_contract.py`. `members` are sorted by rank. Also gives section 4a's attribution
/// of each parameter that drifted or whose signatures differ, read from the same rows (`parameter`,
/// `by`, `verdict`, and `outliers` or `why`; the caller adds where): it is read here, while the
/// observations are held, and kept with the verdict.
fn compare_cell(members: &[(i64, &Observation)], tolerance: f64) -> (Map<String, Value>, Vec<Value>) {
    let mut out = Map::new();
    let mut attribution: Vec<Value> = Vec::new();
    let first = members[0].1;
    let same_device = members.iter().all(|(_, o)| o.device == first.device);
    let projection = !first.projection.is_null() && members.iter().all(|(_, o)| o.projection == first.projection);
    if !projection && members.iter().any(|(_, o)| !o.projection.is_null()) {
        out.insert("projection_compared".into(), json!(false));
    }
    let same_method = !first.signature.is_null() && members.iter().all(|(_, o)| o.signature == first.signature);
    let names: BTreeSet<&String> = members
        .iter()
        .flat_map(|(_, o)| o.params.keys().chain(o.unread.iter()))
        .collect();
    let mut not_compared = Map::new();
    let (mut drift, mut differ) = (Vec::new(), Vec::new());
    let (mut prints_compared, mut signatures_compared) = (0_usize, 0_usize);
    for name in names.into_iter().filter(|_| members.len() > 1) {
        if members.iter().any(|(_, o)| o.unread.contains(name)) {
            not_compared.insert(name.clone(), json!("unread"));
            continue;
        }
        let Some(rows) = members
            .iter()
            .map(|(_, o)| o.params.get(name))
            .collect::<Option<Vec<&Param>>>()
        else {
            not_compared.insert(name.clone(), json!("not_recorded"));
            continue;
        };
        if rows
            .iter()
            .any(|row| row.shape != rows[0].shape || row.dtype != rows[0].dtype)
        {
            not_compared.insert(name.clone(), json!("shape_dtype"));
            continue;
        }
        if same_method {
            signatures_compared += 1;
            if rows.iter().any(|row| row.sig != rows[0].sig) {
                differ.push(name.clone());
                let held: Vec<(i64, &Value)> = members
                    .iter()
                    .zip(&rows)
                    .map(|((rank, _), row)| (*rank, &row.sig))
                    .collect();
                let mut entry = json!({"parameter": name, "by": "signatures"});
                entry
                    .as_object_mut()
                    .expect("an object")
                    .extend(by_signatures(&held).as_object().cloned().unwrap_or_default());
                attribution.push(entry);
            }
        }
        if !same_device {
            continue;
        }
        prints_compared += 1;
        let columns = if projection { 3 } else { 2 };
        let values: Vec<&[f64]> = rows.iter().map(|row| &row.fp[..row.fp.len().min(columns)]).collect();
        let width = values.iter().map(|row| row.len()).min().unwrap_or(0);
        // Attribution reads the columns the drift is decided on.
        let attribute = |attribution: &mut Vec<Value>| {
            let held: Vec<(i64, &[f64])> = members
                .iter()
                .zip(&values)
                .map(|((rank, _), row)| (*rank, &row[..width]))
                .collect();
            let mut entry = json!({"parameter": name, "by": "fingerprints"});
            entry.as_object_mut().expect("an object").extend(
                by_fingerprints(&held, tolerance)
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
            );
            attribution.push(entry);
        };
        if values.iter().any(|row| row.iter().any(|value| !value.is_finite())) {
            drift.push(name.clone());
            attribute(&mut attribution);
            continue;
        }
        // `spread_from_fingerprints`: the mean of the ranks' norms, never below 1e-12.
        let scale = (python_sum(values.iter().map(|row| row[0])) / values.len() as f64).max(1e-12);
        let spread_over = (0..width).any(|at| {
            let (low, high) = values
                .iter()
                .map(|row| row[at])
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
                    (low.min(value), high.max(value))
                });
            (high - low) / scale > tolerance
        });
        if spread_over {
            drift.push(name.clone());
            attribute(&mut attribution);
        }
    }
    if !not_compared.is_empty() {
        out.insert("not_compared".into(), Value::Object(not_compared));
    }
    let one_rank = members.len() < 2;
    if prints_compared > 0 {
        out.insert(
            "fingerprints".into(),
            json!(if drift.is_empty() { "agree" } else { "drift" }),
        );
        if !drift.is_empty() {
            out.insert("drift_parameters".into(), json!(drift));
        }
    } else {
        out.insert("fingerprints".into(), json!("not_compared"));
        let why = if one_rank {
            "one_rank"
        } else if same_device {
            "no_parameters"
        } else {
            "device"
        };
        out.insert("fingerprints_why".into(), json!(why));
    }
    if signatures_compared > 0 {
        out.insert(
            "signatures".into(),
            json!(if differ.is_empty() { "agree" } else { "differ" }),
        );
        if !differ.is_empty() {
            out.insert("signature_parameters".into(), json!(differ));
        }
    } else {
        out.insert("signatures".into(), json!("not_compared"));
        let why = if one_rank {
            "one_rank"
        } else if same_method {
            "no_parameters"
        } else {
            "method"
        };
        out.insert("signatures_why".into(), json!(why));
    }
    (out, attribution)
}

// --- one run --------------------------------------------------------------------------------

/// A rank's row at a step.
#[derive(Debug)]
struct Row {
    launch: Option<String>,
    group: Option<Vec<i64>>,
    /// None once it can never be compared again, or let go for the byte bound.
    observation: Option<Arc<Observation>>,
    bytes: usize,
    /// A member of a complete cell at the last judgement: its observation is the first to go.
    judged: bool,
}

/// A spread at a step, with whether its record also held an observation.
#[derive(Debug)]
struct SpreadRow {
    rank: i64,
    spread: Spread,
    alone: bool,
}

/// A complete cell's verdict, and the rows it was judged from.
#[derive(Debug)]
struct Cell {
    members: Vec<(i64, i64)>,
    verdict: Map<String, Value>,
    /// Section 4a's entries for the cell, read when it was compared: kept with the verdict, since
    /// the observations may be let go before the cell is judged again.
    attribution: Vec<Value>,
}

/// A launch at a step whose entries no longer change.
#[derive(Debug)]
struct Frozen {
    entries: Vec<Value>,
    incidents: Vec<Value>,
    attribution: Vec<Value>,
    why: &'static str,
    rows_not_compared: u64,
}

#[derive(Debug, Default, Clone)]
struct Output {
    entries: Vec<Value>,
    incidents: Vec<Value>,
    legacy: Vec<Value>,
    /// Section 4a's entries of the step: listed with the step's entries and dropped with them.
    attribution: Vec<Value>,
}

type Scope = Option<(String, Vec<i64>)>;
type RowKey = (i64, i64, Option<String>);

#[derive(Debug, Default)]
struct StepState {
    /// By (rank, attempt, launch): one attempt can name two launches (a clock read twice).
    rows: BTreeMap<RowKey, Row>,
    spreads: Vec<SpreadRow>,
    cells: BTreeMap<(String, Vec<i64>), Cell>,
    frozen: BTreeMap<Option<String>, Frozen>,
    /// From a restart: the parameters of the gathered spreads judged before it, by scope, and the
    /// legacy entries.
    frozen_spreads: BTreeMap<Scope, BTreeSet<String>>,
    frozen_legacy: Vec<Value>,
    output: Output,
    /// The output's size as JSON, for the listing bound.
    size: usize,
    dirty: bool,
    touched: u64,
}

impl StepState {
    fn held(&self) -> usize {
        self.rows
            .values()
            .filter(|row| row.observation.is_some())
            .map(|row| row.bytes)
            .sum()
    }
}

/// An entry's place in the list (contract 4): by step, then the rows that name no launch, then by
/// launch, then by group, a step's own entry before the rows set aside from it.
fn entry_order(entry: &Value) -> (i64, bool, String, bool, Vec<i64>, bool) {
    (
        entry["step"].as_i64().unwrap_or(i64::MIN),
        !entry["launch"].is_null(),
        entry["launch"].as_str().unwrap_or_default().to_string(),
        !entry["group"].is_null(),
        ranks(&entry["group"]).unwrap_or_default(),
        entry["why"] == "attempt",
    )
}

/// An incident's place (contract 5): by step, then source, then launch and group.
fn incident_order(incident: &Value) -> (i64, String, String, Vec<i64>) {
    (
        incident["step"].as_i64().unwrap_or(i64::MIN),
        incident["source"].as_str().unwrap_or_default().to_string(),
        incident["launch"].as_str().unwrap_or_default().to_string(),
        ranks(&incident["group"]).unwrap_or_default(),
    )
}

/// What one run's records say: sections 2 to 6, kept as rows arrive. See the module's documentation.
#[derive(Debug)]
pub struct Comparison {
    tolerance: f64,
    bounds: Bounds,
    steps: BTreeMap<i64, StepState>,
    /// (launch, rank) -> the attempt that keeps the launch: the first in the rank's order.
    owners: HashMap<(String, i64), i64>,
    /// (launch, rank) -> the steps it has rows at, judged again when the owner changes.
    rows_at: HashMap<(String, i64), BTreeSet<i64>>,
    held_bytes: usize,
    clock: u64,
    /// Incidents of steps no longer listed, and how many more were not kept.
    kept: Vec<Value>,
    incidents_dropped: u64,
    steps_forgotten: u64,
    /// The launches each forgotten step listed (an index into `launches`, None for no launch): a
    /// late row for one of them is counted in `late_rows`, never listed again, so a backlog cannot
    /// bring an old step back as `insufficient` and push a recent one out. One entry per launch and
    /// forgotten step: a few bytes each for the run's whole length.
    forgotten: HashMap<i64, BTreeSet<Option<usize>>>,
    launches: Vec<String>,
    late_rows: u64,
    version: u64,
}

impl Comparison {
    pub fn new(tolerance: f64, bounds: Bounds) -> Self {
        Self {
            tolerance,
            bounds,
            steps: BTreeMap::new(),
            owners: HashMap::new(),
            rows_at: HashMap::new(),
            held_bytes: 0,
            clock: 0,
            kept: Vec::new(),
            incidents_dropped: 0,
            steps_forgotten: 0,
            forgotten: HashMap::new(),
            launches: Vec::new(),
            late_rows: 0,
            version: 0,
        }
    }

    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Changes with every record taken and every restore: a status whose `replica` is built from
    /// another version is written again.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Bytes of observation JSON held now.
    pub fn held_bytes(&self) -> usize {
        self.held_bytes
    }

    /// Takes one record. `place(rank, attempt)` is the attempt's place in the rank's order
    /// (contract 1), None when it has none, which sorts after every placed attempt.
    pub fn insert(&mut self, record: Record, place: &dyn Fn(i64, i64) -> Option<usize>) {
        if record.is_empty() {
            return;
        }
        self.clock += 1;
        self.version += 1;
        if let Some(launches) = self.forgotten.get(&record.step) {
            // A step no longer listed: a gathered spread there is not listed again, nor an
            // observation of a launch it listed. A new launch at that step number is a new cell.
            let named = record.observed.as_ref().map(|observed| {
                observed
                    .launch
                    .as_ref()
                    .map(|launch| self.launches.iter().position(|known| known == launch))
            });
            let late = match named {
                None => true,
                Some(None) => launches.contains(&None),
                Some(Some(at)) => at.is_some() && launches.contains(&at),
            };
            if late {
                self.late_rows += 1;
                return;
            }
        }
        let (clock, step_number, rank, attempt) = (self.clock, record.step, record.rank, record.attempt);
        let alone = record.observed.is_none();
        let mut owner_changed: Option<(String, i64)> = None;
        let step = self.steps.entry(step_number).or_default();
        step.touched = clock;
        step.dirty = true;
        if let Some(observed) = record.observed {
            if let Some(frozen) = step.frozen.get_mut(&observed.launch) {
                frozen.rows_not_compared += 1;
            } else if let std::collections::btree_map::Entry::Vacant(slot) =
                step.rows.entry((rank, attempt, observed.launch.clone()))
            {
                if let Some(launch) = &observed.launch {
                    let key = (launch.clone(), rank);
                    match self.owners.get(&key) {
                        None => {
                            self.owners.insert(key.clone(), attempt);
                        }
                        Some(owner) if *owner != attempt => {
                            let at = |attempt| place(rank, attempt).unwrap_or(usize::MAX);
                            if at(attempt) < at(*owner) {
                                self.owners.insert(key.clone(), attempt);
                                owner_changed = Some(key.clone());
                            }
                        }
                        Some(_) => {}
                    }
                    self.rows_at.entry(key).or_default().insert(step_number);
                }
                // A row that names no launch is never compared: nothing of it is held.
                let held = observed.launch.is_some();
                slot.insert(Row {
                    launch: observed.launch,
                    group: observed.group,
                    observation: held.then(|| Arc::new(observed.observation)),
                    bytes: if held { observed.bytes } else { 0 },
                    judged: false,
                });
                if held {
                    self.held_bytes += observed.bytes;
                }
            }
        }
        if let Some(spread) = record.spread {
            step.spreads.push(SpreadRow { rank, spread, alone });
        }
        if let Some(key) = owner_changed {
            for at in self.rows_at.get(&key).into_iter().flatten() {
                if let Some(step) = self.steps.get_mut(at) {
                    step.dirty = true;
                }
            }
        }
        self.judge_dirty();
        self.enforce_bounds();
    }

    fn judge_dirty(&mut self) {
        let dirty: Vec<i64> = self
            .steps
            .iter()
            .filter(|(_, step)| step.dirty)
            .map(|(number, _)| *number)
            .collect();
        for number in dirty {
            self.judge(number);
        }
    }

    /// Judges one step again from what it holds.
    fn judge(&mut self, number: i64) {
        let Some(mut step) = self.steps.remove(&number) else {
            return;
        };
        let before = step.held();
        loop {
            match self.judge_once(number, &mut step) {
                Ok(output) => {
                    step.size = [&output.entries, &output.incidents, &output.legacy, &output.attribution]
                        .iter()
                        .flat_map(|list| list.iter())
                        .map(|item| item.to_string().len() + 1)
                        .sum();
                    step.output = output;
                    break;
                }
                // A complete cell must be judged again with an observation already let go (a
                // backlog of an earlier attempt arrived after the byte bound took the others'):
                // its launch and step keep the entries they had (module documentation).
                Err(launch) => {
                    let key = Some(launch);
                    self.freeze(&mut step, key.clone(), "re_pair");
                    if let Some(frozen) = step.frozen.get_mut(&key) {
                        frozen.rows_not_compared = frozen.rows_not_compared.max(1);
                    }
                }
            }
        }
        step.dirty = false;
        self.held_bytes = self.held_bytes + step.held() - before;
        self.steps.insert(number, step);
    }

    /// The step's entries, incidents and legacy entries; Err(launch) when that launch must freeze.
    fn judge_once(&self, number: i64, step: &mut StepState) -> Result<Output, String> {
        let tolerance = self.tolerance;
        let mut out = Output::default();
        for frozen in step.frozen.values() {
            out.entries.extend(frozen.entries.iter().cloned());
            out.incidents.extend(frozen.incidents.iter().cloned());
            out.attribution.extend(frozen.attribution.iter().cloned());
        }
        // The gathered spreads, each under what it says it covers (contract 5).
        let mut gathered: BTreeMap<Scope, (BTreeSet<String>, BTreeSet<String>)> = BTreeMap::new();
        for row in &step.spreads {
            let held = gathered.entry(row.spread.scope.clone()).or_default();
            held.0.extend(row.spread.covered.iter().cloned());
            held.1.extend(row.spread.over.iter().cloned());
        }
        // Rows by launch, the attempts that do not keep their launch set aside (contract 2).
        let mut launches: BTreeMap<Option<String>, BTreeMap<i64, i64>> = BTreeMap::new();
        let mut aside: BTreeMap<(String, Option<Vec<i64>>), BTreeSet<i64>> = BTreeMap::new();
        for ((rank, attempt, _), row) in &step.rows {
            if step.frozen.contains_key(&row.launch) {
                continue;
            }
            match &row.launch {
                Some(launch) if self.owners.get(&(launch.clone(), *rank)) != Some(attempt) => {
                    aside
                        .entry((launch.clone(), row.group.clone()))
                        .or_default()
                        .insert(*rank);
                }
                _ => {
                    launches.entry(row.launch.clone()).or_default().insert(*rank, *attempt);
                }
            }
        }
        let mut keep: BTreeSet<RowKey> = BTreeSet::new();
        let mut judged: BTreeSet<RowKey> = BTreeSet::new();
        let mut cells: BTreeMap<(String, Vec<i64>), Cell> = BTreeMap::new();
        for (launch, members) in &launches {
            let rows: Vec<(i64, i64, &Row)> = members
                .iter()
                .map(|(rank, attempt)| (*rank, *attempt, &step.rows[&(*rank, *attempt, launch.clone())]))
                .collect();
            let groups: BTreeSet<Option<&Vec<i64>>> = rows.iter().map(|(_, _, row)| row.group.as_ref()).collect();
            let Some(launch) = launch else {
                let group = match groups.iter().collect::<Vec<_>>()[..] {
                    [Some(group)] => json!(group),
                    _ => Value::Null,
                };
                out.entries.push(json!({"launch": null, "group": group, "step": number,
                    "coverage": "not_established", "why": "launch"}));
                continue;
            };
            let named: Vec<&Vec<i64>> = groups.iter().flatten().copied().collect();
            let overlap = named
                .iter()
                .enumerate()
                .any(|(at, a)| named[at + 1..].iter().any(|b| a.iter().any(|rank| b.contains(rank))));
            let outside = rows
                .iter()
                .any(|(rank, _, row)| row.group.as_ref().is_none_or(|group| !group.contains(rank)));
            if groups.contains(&None) || overlap || outside {
                out.entries.push(json!({"launch": launch, "group": null, "step": number,
                    "coverage": "not_established", "why": "group"}));
                continue;
            }
            for group in named {
                let present: Vec<&(i64, i64, &Row)> = rows
                    .iter()
                    .filter(|(_, _, row)| row.group.as_ref() == Some(group))
                    .collect();
                let missing: Vec<i64> = group
                    .iter()
                    .copied()
                    .filter(|rank| !present.iter().any(|(r, _, _)| r == rank))
                    .collect();
                let mut entry = json!({"launch": launch, "group": group, "step": number});
                if !missing.is_empty() {
                    entry["coverage"] = json!("insufficient");
                    entry["missing"] = json!(missing);
                    keep.extend(
                        present
                            .iter()
                            .map(|(rank, attempt, _)| (*rank, *attempt, Some(launch.clone()))),
                    );
                    out.entries.push(entry);
                    continue;
                }
                let membership: Vec<(i64, i64)> = present.iter().map(|(rank, attempt, _)| (*rank, *attempt)).collect();
                let key = (launch.clone(), group.clone());
                let (verdict, attribution) = match step.cells.get(&key) {
                    Some(cell) if cell.members == membership => (cell.verdict.clone(), cell.attribution.clone()),
                    _ => {
                        let observations: Option<Vec<(i64, &Observation)>> = present
                            .iter()
                            .map(|(rank, _, row)| row.observation.as_deref().map(|o| (*rank, o)))
                            .collect();
                        let Some(observations) = observations else {
                            return Err(launch.clone());
                        };
                        let (verdict, read) = compare_cell(&observations, tolerance);
                        let attribution = read
                            .into_iter()
                            .map(|said| {
                                let mut entry = json!({"step": number, "launch": launch, "group": group});
                                entry
                                    .as_object_mut()
                                    .expect("an object")
                                    .extend(said.as_object().cloned().unwrap_or_default());
                                entry
                            })
                            .collect();
                        (verdict, attribution)
                    }
                };
                out.attribution.extend(attribution.iter().cloned());
                cells.insert(
                    key,
                    Cell {
                        members: membership.clone(),
                        verdict: verdict.clone(),
                        attribution,
                    },
                );
                judged.extend(
                    membership
                        .iter()
                        .map(|(rank, attempt)| (*rank, *attempt, Some(launch.clone()))),
                );
                let covered = gathered
                    .get(&Some((launch.clone(), group.clone())))
                    .map(|(covered, _)| covered.clone())
                    .unwrap_or_default();
                entry["coverage"] = json!("complete");
                if !covered.is_empty() {
                    entry["gathered"] = json!(covered);
                }
                let raised: Vec<Value> = verdict
                    .get("drift_parameters")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter(|name| name.as_str().is_none_or(|name| !covered.contains(name)))
                    .cloned()
                    .collect();
                entry.as_object_mut().expect("an object").extend(verdict);
                out.entries.push(entry);
                if !raised.is_empty() {
                    out.incidents
                        .push(json!({"step": number, "parameters": raised, "source": "replica",
                        "launch": launch, "group": group}));
                }
            }
        }
        for ((launch, group), set_aside) in aside {
            out.entries
                .push(json!({"launch": launch, "group": group, "step": number,
                "coverage": "not_established", "why": "attempt", "ranks": set_aside}));
        }
        // What every gathered spread raises on its own, with what it says it covers.
        let mut over: BTreeMap<Scope, BTreeSet<String>> = step.frozen_spreads.clone();
        for (scope, (_, found)) in gathered {
            over.entry(scope).or_default().extend(found);
        }
        for (scope, found) in over.into_iter().filter(|(_, found)| !found.is_empty()) {
            let mut incident = json!({"step": number, "parameters": found, "source": "replica_spread"});
            if let Some((launch, group)) = scope {
                incident["launch"] = json!(launch);
                incident["group"] = json!(group);
            }
            out.incidents.push(incident);
        }
        // Section 6: a spread on a record with no observation, judged on that record alone.
        out.legacy = step.frozen_legacy.clone();
        let mut alone: Vec<&SpreadRow> = step.spreads.iter().filter(|row| row.alone).collect();
        alone.sort_by_key(|row| row.rank);
        for row in alone {
            let mut entry = json!({"rank": row.rank, "step": number, "drift": !row.spread.over.is_empty()});
            if !row.spread.over.is_empty() {
                entry["parameters"] = json!(row.spread.over);
            }
            out.legacy.push(entry);
        }
        out.entries.sort_by_key(entry_order);
        out.incidents.sort_by_key(incident_order);
        out.attribution.sort_by_key(attribution_order);
        // Only an unfinished cell's rows and a complete cell's members keep their observations.
        for (key, row) in step.rows.iter_mut() {
            row.judged = judged.contains(key);
            if !keep.contains(key) && !row.judged {
                row.observation = None;
            }
        }
        step.cells = cells;
        Ok(out)
    }

    /// Freezes a launch at a step (None freezes every launch there that holds an observation),
    /// with the entries and incidents it last had, and drops its rows.
    fn freeze(&self, step: &mut StepState, launch: Option<String>, why: &'static str) {
        let launches: BTreeSet<Option<String>> = match launch {
            Some(launch) => [Some(launch)].into(),
            None => step
                .rows
                .values()
                .filter(|row| row.observation.is_some())
                .map(|row| row.launch.clone())
                .collect(),
        };
        for launch in launches {
            let named = |item: &&Value| item["launch"].as_str() == launch.as_deref();
            let entries: Vec<Value> = step.output.entries.iter().filter(named).cloned().collect();
            let incidents: Vec<Value> = step
                .output
                .incidents
                .iter()
                .filter(|incident| incident["source"] == "replica")
                .filter(named)
                .cloned()
                .collect();
            let attribution: Vec<Value> = step.output.attribution.iter().filter(named).cloned().collect();
            step.rows.retain(|_, row| row.launch != launch);
            step.cells.retain(|(cell, _), _| Some(cell) != launch.as_ref());
            step.frozen.entry(launch).or_insert(Frozen {
                entries,
                incidents,
                attribution,
                why,
                rows_not_compared: 0,
            });
        }
    }

    fn enforce_bounds(&mut self) {
        // Step numbers listed: the least recently touched go first; their incidents are kept.
        let mut listed: usize = self.steps.values().map(|step| step.size).sum();
        while self.steps.len() > self.bounds.listed_steps || (listed > self.bounds.listed_bytes && self.steps.len() > 1)
        {
            let Some(oldest) = self.steps.iter().min_by_key(|(_, step)| step.touched).map(|(n, _)| *n) else {
                break;
            };
            let step = self.steps.remove(&oldest).expect("listed");
            listed -= step.size;
            self.held_bytes -= step.held();
            self.steps_forgotten += 1;
            let listed: BTreeSet<Option<String>> = step
                .output
                .entries
                .iter()
                .map(|entry| entry["launch"].as_str().map(str::to_string))
                .chain(step.rows.values().map(|row| row.launch.clone()))
                .chain(step.frozen.keys().cloned())
                .collect();
            let mut indexes = BTreeSet::new();
            for launch in listed {
                indexes.insert(
                    launch.map(|launch| match self.launches.iter().position(|known| *known == launch) {
                        Some(at) => at,
                        None => {
                            self.launches.push(launch);
                            self.launches.len() - 1
                        }
                    }),
                );
            }
            self.forgotten.entry(oldest).or_default().extend(indexes);
            for incident in step.output.incidents {
                if self.kept.len() < self.bounds.kept_incidents {
                    self.kept.push(incident);
                } else {
                    self.incidents_dropped += 1;
                }
            }
        }
        if self.held_bytes <= self.bounds.held_bytes {
            return;
        }
        let mut order: Vec<(u64, i64)> = self.steps.iter().map(|(n, step)| (step.touched, *n)).collect();
        order.sort_unstable();
        // First the judged cells' observations, kept only for a late re-pairing.
        for (_, number) in &order {
            if self.held_bytes <= self.bounds.held_bytes {
                return;
            }
            let step = self.steps.get_mut(number).expect("listed");
            for row in step
                .rows
                .values_mut()
                .filter(|row| row.judged && row.observation.is_some())
            {
                row.observation = None;
                self.held_bytes -= row.bytes;
            }
        }
        // Then the unfinished ones: their launch and step are let go.
        for (_, number) in &order {
            if self.held_bytes <= self.bounds.held_bytes {
                return;
            }
            let mut step = self.steps.remove(number).expect("listed");
            let before = step.held();
            self.freeze(&mut step, None, "held_bytes");
            self.held_bytes -= before - step.held();
            step.dirty = true;
            self.steps.insert(*number, step);
            self.judge(*number);
        }
    }

    /// Takes up the result written before a restart (`watchdog.json`'s `replica`): every launch
    /// and step in it is frozen as written, and its incidents and legacy entries stay.
    pub fn restore(&mut self, written: &Value) {
        if written["v"] != RESULT_VERSION {
            return;
        }
        self.version += 1;
        let entries = written["steps"].as_array().map_or(&[][..], Vec::as_slice);
        let mut by_step: BTreeMap<i64, BTreeMap<Option<String>, Vec<Value>>> = BTreeMap::new();
        for entry in entries {
            if let Some(number) = entry["step"].as_i64() {
                by_step
                    .entry(number)
                    .or_default()
                    .entry(entry["launch"].as_str().map(str::to_string))
                    .or_default()
                    .push(entry.clone());
            }
        }
        for (number, launches) in by_step {
            self.clock += 1;
            let step = self.steps.entry(number).or_default();
            step.touched = self.clock;
            step.dirty = true;
            for (launch, entries) in launches {
                step.frozen.insert(
                    launch,
                    Frozen {
                        entries,
                        incidents: Vec::new(),
                        attribution: Vec::new(),
                        why: "restart",
                        rows_not_compared: 0,
                    },
                );
            }
        }
        for legacy in written["legacy"].as_array().into_iter().flatten() {
            if let Some(number) = legacy["step"].as_i64() {
                self.clock += 1;
                let step = self.steps.entry(number).or_default();
                step.touched = step.touched.max(self.clock);
                step.dirty = true;
                step.frozen_legacy.push(legacy.clone());
            }
        }
        for incident in written["replica_drift"].as_array().into_iter().flatten() {
            let number = incident["step"].as_i64();
            let step = number.and_then(|number| self.steps.get_mut(&number));
            let launch = incident["launch"].as_str().map(str::to_string);
            match (step, incident["source"].as_str()) {
                (Some(step), Some("replica")) if step.frozen.contains_key(&launch) => {
                    step.frozen
                        .get_mut(&launch)
                        .expect("checked")
                        .incidents
                        .push(incident.clone());
                }
                (Some(step), Some("replica_spread")) => {
                    let scope = scope(Some(incident));
                    let names = incident["parameters"].as_array().into_iter().flatten();
                    step.frozen_spreads
                        .entry(scope)
                        .or_default()
                        .extend(names.filter_map(|name| name.as_str().map(str::to_string)));
                }
                _ => self.kept.push(incident.clone()),
            }
        }
        // Section 4a's entries go with their launch and step; one whose step entry is not listed
        // has nothing to stand beside and is dropped with it, as the listing bound drops it.
        for said in written["attribution"].as_array().into_iter().flatten() {
            let frozen = said["step"]
                .as_i64()
                .and_then(|number| self.steps.get_mut(&number))
                .and_then(|step| step.frozen.get_mut(&said["launch"].as_str().map(str::to_string)));
            if let Some(frozen) = frozen {
                frozen.attribution.push(said.clone());
            }
        }
        let omitted = &written["omitted"];
        self.steps_forgotten += omitted["steps"].as_u64().unwrap_or(0);
        self.incidents_dropped += omitted["replica_drift"].as_u64().unwrap_or(0);
        self.late_rows += omitted["late_rows"].as_u64().unwrap_or(0);
        for said in omitted["frozen"].as_array().into_iter().flatten() {
            let frozen = said["step"]
                .as_i64()
                .and_then(|number| self.steps.get_mut(&number))
                .and_then(|step| step.frozen.get_mut(&said["launch"].as_str().map(str::to_string)));
            if let Some(frozen) = frozen {
                frozen.why = match said["why"].as_str() {
                    Some("held_bytes") => "held_bytes",
                    Some("re_pair") => "re_pair",
                    _ => "restart",
                };
                frozen.rows_not_compared = said["rows_not_compared"].as_u64().unwrap_or(0);
            }
        }
        self.judge_dirty();
        self.enforce_bounds();
    }

    /// The result: `steps` and `replica_drift` in the contract's order and words, `legacy` when
    /// there is one, `attribution` when it has an entry (section 4a), and `omitted` when a bound
    /// dropped something.
    pub fn result(&self) -> Value {
        let mut steps = Vec::new();
        let mut incidents = self.kept.clone();
        let mut legacy = Vec::new();
        let mut attribution = Vec::new();
        let mut frozen = Vec::new();
        for (number, step) in &self.steps {
            steps.extend(step.output.entries.iter().cloned());
            incidents.extend(step.output.incidents.iter().cloned());
            legacy.extend(step.output.legacy.iter().cloned());
            attribution.extend(step.output.attribution.iter().cloned());
            for (launch, held) in &step.frozen {
                if held.why != "restart" || held.rows_not_compared > 0 {
                    frozen.push(json!({"launch": launch, "step": number, "why": held.why,
                        "rows_not_compared": held.rows_not_compared}));
                }
            }
        }
        incidents.sort_by_key(incident_order);
        let mut out = json!({"steps": steps, "replica_drift": incidents});
        if !legacy.is_empty() {
            out["legacy"] = json!(legacy);
        }
        if !attribution.is_empty() {
            out["attribution"] = json!(attribution);
        }
        let mut omitted = Map::new();
        if self.steps_forgotten > 0 {
            omitted.insert("steps".into(), json!(self.steps_forgotten));
        }
        if self.incidents_dropped > 0 {
            omitted.insert("replica_drift".into(), json!(self.incidents_dropped));
        }
        if self.late_rows > 0 {
            omitted.insert("late_rows".into(), json!(self.late_rows));
        }
        if !frozen.is_empty() {
            if frozen.len() > FROZEN_LISTED {
                omitted.insert("frozen_more".into(), json!(frozen.len() - FROZEN_LISTED));
                frozen.truncate(FROZEN_LISTED);
            }
            omitted.insert("frozen".into(), json!(frozen));
        }
        if !omitted.is_empty() {
            out["omitted"] = Value::Object(omitted);
        }
        out
    }

    /// The object a run's `watchdog.json` carries under `replica` (contract 8).
    pub fn status_object(&self) -> Value {
        let mut out = json!({"v": RESULT_VERSION, "tolerance": self.tolerance});
        if let (Some(out), Value::Object(result)) = (out.as_object_mut(), self.result()) {
            out.extend(result);
        }
        out
    }
}

/// What a reader of a finished list of one run's records says (contract 2 to 6): the result with
/// no bound. A rank's records are listed in contract 1's order, so an attempt's place in its rank's
/// order is where it first appears.
pub fn compare(records: &[Value], tolerance: f64) -> Value {
    let mut first_seen: HashMap<i64, Vec<i64>> = HashMap::new();
    for record in records {
        if let (Some(rank), attempt) = (record["rank"].as_i64(), record["attempt"].as_i64().unwrap_or(0)) {
            let seen = first_seen.entry(rank).or_default();
            if !seen.contains(&attempt) {
                seen.push(attempt);
            }
        }
    }
    let place = |rank: i64, attempt: i64| {
        first_seen
            .get(&rank)
            .and_then(|seen| seen.iter().position(|known| *known == attempt))
    };
    let mut comparison = Comparison::new(tolerance, Bounds::none());
    for record in records {
        if let Some(record) = record_from_json(record, tolerance) {
            comparison.insert(record, &place);
        }
    }
    comparison.result()
}
