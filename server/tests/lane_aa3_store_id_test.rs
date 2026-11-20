//! Lane AA3 of wave AA (Nov 20, 2026), L3's server half as the author decided it: the server writes
//! a store ID once at its store root (`store.json`) and returns it with each acknowledgement,
//! `{"end_byte_offset": N, "store_id": "..."}`. A store that has an ID keeps it across restarts; a
//! store wiped and recreated behind the same address gets a new one; two addresses that reach one
//! store return the same one. Each is shown with the real binary.

mod lane_aa3_support;

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use lane_aa3_support::{json_file, open, Binary};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("schema_version", DataType::Int32, false),
        Field::new("run_id", DataType::Utf8, false),
        Field::new("rank", DataType::Int32, false),
        Field::new("attempt", DataType::Int64, false),
        Field::new("step", DataType::Int64, false),
    ]))
}

fn batch(run_id: &str, step: i64) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(StringArray::from(vec![run_id])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(Int64Array::from(vec![step])),
        ],
    )
    .unwrap()
}

/// Sends one batch of `run_id` to `server`, ends the stream, and returns the acknowledgements.
async fn acks(server: &Binary, run_id: &str, step: i64, end: i64) -> Vec<serde_json::Value> {
    let mut put = open(&server.url(), run_id, 0, schema()).await;
    put.send(batch(run_id, step), end).await;
    put.finish().await
}

/// The one store ID the acknowledgements carry, after checking their shape: exactly
/// `end_byte_offset` and `store_id`, the offset as an older sender reads it.
fn store_id_of(acks: &[serde_json::Value], end: i64) -> String {
    assert!(!acks.is_empty(), "no acknowledgement");
    let mut ids: Vec<String> = Vec::new();
    for ack in acks {
        let keys: Vec<&String> = ack.as_object().expect("an object").keys().collect();
        assert_eq!(keys, ["end_byte_offset", "store_id"], "{ack}");
        let id = ack["store_id"].as_str().expect("a string").to_string();
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    assert_eq!(acks.last().unwrap()["end_byte_offset"], end, "{acks:?}");
    assert_eq!(ids.len(), 1, "one store, one ID: {acks:?}");
    ids.pop().unwrap()
}

/// 32 lowercase hexadecimal digits: 128 random bits.
fn well_formed(id: &str) -> bool {
    id.len() == 32 && id.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
}

fn written_id(store: &Path) -> String {
    let document = json_file(&store.join("store.json"));
    document["store_id"]
        .as_str()
        .expect("store.json names its ID")
        .to_string()
}

#[tokio::test]
async fn the_store_id_is_kept_across_restarts_shared_by_two_addresses_and_new_for_a_wiped_store() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("store");

    // A new store gets an ID, written at its root, and every acknowledgement carries it.
    let first = Binary::start(&store, "127.0.0.1:0");
    let address = first.address.clone();
    let id = store_id_of(&acks(&first, "run-a", 0, 100).await, 100);
    assert!(well_formed(&id), "{id}");
    assert_eq!(written_id(&store), id);
    first.stop();

    // The same store behind the same address after a restart: the same ID.
    let again = Binary::start(&store, &address);
    assert_eq!(store_id_of(&acks(&again, "run-a", 1, 200).await, 200), id);

    // A second address that reaches the same store: the same ID.
    let beside = Binary::start(&store, "127.0.0.1:0");
    assert_ne!(beside.address, address);
    assert_eq!(store_id_of(&acks(&beside, "run-b", 0, 300).await, 300), id);
    again.stop();
    beside.stop();

    // The store wiped and recreated behind the same address: a new ID.
    std::fs::remove_dir_all(&store).unwrap();
    std::fs::create_dir_all(&store).unwrap();
    let wiped = Binary::start(&store, &address);
    let new_id = store_id_of(&acks(&wiped, "run-a", 0, 100).await, 100);
    assert!(well_formed(&new_id), "{new_id}");
    assert_ne!(new_id, id);
    assert_eq!(written_id(&store), new_id);
}
