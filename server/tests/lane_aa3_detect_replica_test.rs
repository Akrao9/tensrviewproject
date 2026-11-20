//! Lane AA3 of wave AA (Nov 20, 2026): `tensorquill-server detect --run` computes the run's replica
//! result (contract 8), the object the live server writes under `replica` in a run's
//! `watchdog.json`. Until now only the live server computed it (lane Z3's known gap).
//!
//! Every case of the contract (`sdk/tests/fixtures/replica/`) is stored as the server stores it:
//! each rank's rows of each attempt as one Parquet part, through `Storage::write_part`. The parts
//! are read back, written as Arrow IPC streams in contract 1's order (a rank's attempts in the order
//! the case first names them, as the SDK's `stream_order` sends them), and given to the real binary,
//! whose `replica` must be the case's expected result with `v` and `tolerance`, as the live server
//! carries it.

use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float64Builder, Int32Array, Int64Array, ListBuilder, MapBuilder, StringArray, StringBuilder,
};
use arrow::record_batch::RecordBatch;
use object_store::local::LocalFileSystem;
use serde_json::{json, Value};
use tensorquill_server::storage::Storage;

const FILES: [&str; 2] = ["no_incident_cases.json", "incident_cases.json"];

/// Every case: (file, name, records, expected, tolerance).
fn cases() -> Vec<(String, String, Vec<Value>, Value, f64)> {
    let folder = Path::new(env!("CARGO_MANIFEST_DIR")).join("../sdk/tests/fixtures/replica");
    let mut out = Vec::new();
    for file in FILES {
        let text = std::fs::read_to_string(folder.join(file)).expect("the contract's cases");
        let document: Value = serde_json::from_str(&text).unwrap();
        let tolerance = document["tolerance"].as_f64().unwrap();
        for (name, body) in document["cases"].as_object().unwrap() {
            out.push((
                file.to_string(),
                name.clone(),
                body["records"].as_array().unwrap().clone(),
                body["expected"].clone(),
                tolerance,
            ));
        }
    }
    out
}

/// One rank's rows of one attempt in the shape the server stores (`flight.py`'s `record_to_row`):
/// the key columns, `replica_spread` in its map column, `replica` and `replica_spread_scope` in
/// `extra_json`.
fn rows(run_id: &str, rank: i32, records: &[&Value]) -> RecordBatch {
    let n = records.len();
    let steps = Int64Array::from(records.iter().map(|r| r["step"].as_i64()).collect::<Vec<_>>());
    let attempts = Int64Array::from(records.iter().map(|r| r["attempt"].as_i64()).collect::<Vec<_>>());
    let mut spreads = MapBuilder::new(None, StringBuilder::new(), ListBuilder::new(Float64Builder::new()));
    let mut extra = Vec::new();
    for record in records {
        let mut extra_keys = serde_json::Map::new();
        for key in ["replica", "replica_spread_scope"] {
            if let Some(value) = record.get(key) {
                extra_keys.insert(key.into(), value.clone());
            }
        }
        match record.get("replica_spread").and_then(Value::as_object) {
            Some(table) => {
                for (name, values) in table {
                    spreads.keys().append_value(name);
                    for value in values.as_array().unwrap() {
                        spreads.values().values().append_value(value.as_f64().unwrap());
                    }
                    spreads.values().append(true);
                }
                spreads.append(true).unwrap();
            }
            None => spreads.append(false).unwrap(),
        }
        extra.push((!extra_keys.is_empty()).then(|| Value::Object(extra_keys).to_string()));
    }
    RecordBatch::try_from_iter(vec![
        ("schema_version", Arc::new(Int32Array::from(vec![1; n])) as ArrayRef),
        ("run_id", Arc::new(StringArray::from(vec![run_id; n])) as ArrayRef),
        ("rank", Arc::new(Int32Array::from(vec![rank; n])) as ArrayRef),
        ("attempt", Arc::new(attempts) as ArrayRef),
        ("step", Arc::new(steps) as ArrayRef),
        ("replica_spread", Arc::new(spreads.finish()) as ArrayRef),
        ("extra_json", Arc::new(StringArray::from(extra)) as ArrayRef),
    ])
    .unwrap()
}

/// The case stored as parts under `root`, read back, and written as one IPC stream per part in
/// contract 1's order; the paths of the streams.
async fn stored_parts(root: &Path, records: &[Value]) -> Vec<std::path::PathBuf> {
    let run_id = records[0]["run_id"].as_str().unwrap_or("r1").to_string();
    // The case's (rank, attempt) groups, in the order the case first names them.
    let mut groups: Vec<((i32, i64), Vec<&Value>)> = Vec::new();
    for record in records {
        let key = (
            record["rank"].as_i64().unwrap() as i32,
            record["attempt"].as_i64().unwrap(),
        );
        match groups.iter_mut().find(|(held, _)| *held == key) {
            Some((_, rows)) => rows.push(record),
            None => groups.push((key, vec![record])),
        }
    }
    let store = root.join("store");
    std::fs::create_dir_all(&store).unwrap();
    let storage = Storage::new(Arc::new(LocalFileSystem::new_with_prefix(&store).unwrap()), "".into());
    for ((rank, attempt), group) in &groups {
        let batch = rows(&run_id, *rank, group);
        let steps: Vec<i64> = group.iter().map(|r| r["step"].as_i64().unwrap()).collect();
        let (first, last) = (*steps.iter().min().unwrap(), *steps.iter().max().unwrap());
        storage
            .write_part(&run_id, *rank, *attempt, &[batch], first, last)
            .await
            .unwrap();
    }
    let mut paths = Vec::new();
    for ((rank, attempt), _) in &groups {
        let parts = storage.rank_parts(&run_id, *rank).await.unwrap();
        let part = parts
            .iter()
            .find(|part| part.attempt == *attempt)
            .expect("the part just stored");
        let batches = storage.read_part(&part.location).await.unwrap();
        let path = root.join(format!("part-{}.arrows", paths.len()));
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = arrow::ipc::writer::StreamWriter::try_new(file, &batches[0].schema()).unwrap();
        for batch in &batches {
            writer.write(batch).unwrap();
        }
        writer.finish().unwrap();
        paths.push(path);
    }
    paths
}

#[tokio::test]
async fn detect_run_gives_every_contract_case_its_expected_replica_result_from_stored_parts() {
    let all = cases();
    assert_eq!(all.len(), 39, "the contract has 39 cases");
    let mut wrong = Vec::new();
    for (file, name, records, expected, tolerance) in &all {
        let dir = tempfile::tempdir().unwrap();
        let streams = stored_parts(dir.path(), records).await;
        let done = Command::new(env!("CARGO_BIN_EXE_tensorquill-server"))
            .args(["detect", "--run", "--thresholds"])
            .arg(json!({"drift_tolerance": tolerance}).to_string())
            .args(&streams)
            .output()
            .unwrap();
        assert!(
            done.status.success(),
            "{file}:{name}: {}",
            String::from_utf8_lossy(&done.stderr)
        );
        let document: Value = serde_json::from_slice(&done.stdout).unwrap();
        let mut want = json!({"v": 1, "tolerance": tolerance});
        want.as_object_mut()
            .unwrap()
            .extend(expected.as_object().unwrap().clone());
        if document["replica"] != want {
            wrong.push(format!(
                "{file}:{name}\n  got      {}\n  expected {want}",
                document["replica"]
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of 39 cases differ:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// A run with no replica observation has the empty result, as the live server writes it for every
/// run it watches with its detectors on: `steps` and `replica_drift` empty, never a missing key,
/// which a reader would take for an older server.
#[tokio::test]
async fn a_run_without_observations_has_the_empty_replica_result() {
    let dir = tempfile::tempdir().unwrap();
    let records = vec![
        json!({"run_id": "r1", "rank": 0, "attempt": 1, "step": 0}),
        json!({"run_id": "r1", "rank": 1, "attempt": 2, "step": 0}),
    ];
    let streams = stored_parts(dir.path(), &records).await;
    let done = Command::new(env!("CARGO_BIN_EXE_tensorquill-server"))
        .args(["detect", "--run"])
        .args(&streams)
        .output()
        .unwrap();
    assert!(done.status.success(), "{}", String::from_utf8_lossy(&done.stderr));
    let document: Value = serde_json::from_slice(&done.stdout).unwrap();
    assert_eq!(
        document["replica"],
        json!({"v": 1, "tolerance": 1e-6, "steps": [], "replica_drift": []})
    );
}
