//! Lane AE1 of wave AE (Nov 21, 2026), RS10's open part: a batch refused for good after the server
//! itself started again. A restarted server holds no acknowledged offset for the rank, so until
//! this lane its refusal named the refused bytes from byte 0; the sender, whose range begins at the
//! offset it saved, could not confirm it and sent the batch again at every stream. A batch's
//! metadata may now claim where it begins (`start_byte_offset`), and a server that holds nothing for
//! the rank names the range from there, acknowledging only what it holds itself.

mod lane_aa3_support;

use std::sync::Arc;
use std::time::Duration;

use arrow::array::{ArrayRef, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::{FlightClient, FlightData, FlightDescriptor, PutResult};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use lane_aa3_support::{start, stored_steps, Binary};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Channel;
use tonic::Code;

const RUN: &str = "run-ae1";

fn schema(step: DataType) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("schema_version", DataType::Int32, false),
        Field::new("run_id", DataType::Utf8, false),
        Field::new("rank", DataType::Int32, false),
        Field::new("attempt", DataType::Int64, false),
        Field::new("step", step, false),
    ]))
}

/// Rows of rank 0, attempt 1, at `steps`.
fn rows(steps: &[i64]) -> RecordBatch {
    let n = steps.len();
    RecordBatch::try_new(
        schema(DataType::Int64),
        vec![
            Arc::new(Int32Array::from(vec![1; n])) as ArrayRef,
            Arc::new(StringArray::from(vec![RUN; n])),
            Arc::new(Int32Array::from(vec![0; n])),
            Arc::new(Int64Array::from(vec![1; n])),
            Arc::new(Int64Array::from(steps.to_vec())),
        ],
    )
    .unwrap()
}

/// A batch the server cannot read: its `step` is text.
fn text_steps(steps: &[i64]) -> RecordBatch {
    let n = steps.len();
    let text: Vec<String> = steps.iter().map(|step| step.to_string()).collect();
    RecordBatch::try_new(
        schema(DataType::Utf8),
        vec![
            Arc::new(Int32Array::from(vec![1; n])) as ArrayRef,
            Arc::new(StringArray::from(vec![RUN; n])),
            Arc::new(Int32Array::from(vec![0; n])),
            Arc::new(Int64Array::from(vec![1; n])),
            Arc::new(StringArray::from(text)),
        ],
    )
    .unwrap()
}

/// One `DoPut` stream whose batches carry the metadata a test gives.
struct Put {
    messages: mpsc::UnboundedSender<Result<FlightData, FlightError>>,
    acks: BoxStream<'static, Result<PutResult, FlightError>>,
}

async fn open(address: &str, schema: Arc<Schema>) -> Put {
    let channel = Channel::from_shared(address.to_string())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightClient::new(channel);
    let (messages, outgoing) = mpsc::unbounded_channel();
    let descriptor = FlightDescriptor::new_path(vec!["runs".into(), RUN.into(), "rank=0".into()]);
    let opening: Vec<FlightData> = FlightDataEncoderBuilder::new()
        .with_flight_descriptor(Some(descriptor))
        .with_schema(schema)
        .build(futures::stream::empty())
        .try_collect()
        .await
        .unwrap();
    for message in opening {
        messages.send(Ok(message)).unwrap();
    }
    let acks = match client.do_put(UnboundedReceiverStream::new(outgoing)).await {
        Ok(acks) => acks,
        Err(error) => futures::stream::once(async move { Err(error) }).boxed(),
    };
    Put { messages, acks }
}

/// The metadata of a batch that ends at `end` and, when `start` is given, claims to begin there.
fn metadata(end: i64, start: Option<i64>) -> String {
    let claim = start.map_or_else(String::new, |start| format!(",\"start_byte_offset\":{start}"));
    format!("{{\"schema_version\":1,\"run_id\":\"{RUN}\",\"rank\":0,\"min_step\":null,\"max_step\":null,\"end_byte_offset\":{end}{claim}}}")
}

impl Put {
    async fn send(&mut self, batch: RecordBatch, end: i64, start: Option<i64>) {
        // Not `FlightDataEncoder`, which sends nothing for a batch of no rows (the query below).
        let encoded = arrow_flight::utils::batches_to_flight_data(&batch.schema(), [&batch]).unwrap();
        for mut message in encoded.into_iter().skip(1) {
            message.app_metadata = metadata(end, start).into();
            let _ = self.messages.send(Ok(message));
        }
    }

    async fn reply(&mut self) -> Result<serde_json::Value, tonic::Status> {
        match tokio::time::timeout(Duration::from_secs(20), self.acks.next())
            .await
            .expect("no reply")
        {
            Some(Ok(result)) => Ok(serde_json::from_slice(&result.app_metadata).unwrap()),
            Some(Err(FlightError::Tonic(status))) => Err(*status),
            Some(Err(other)) => panic!("{other}"),
            None => panic!("the stream ended without a reply"),
        }
    }

    /// Ends the stream and returns the acknowledged offsets the server sent until it ended it.
    async fn finish(mut self) -> Vec<i64> {
        drop(self.messages);
        let mut acked = Vec::new();
        while let Some(result) = self.acks.next().await {
            let reply: serde_json::Value = serde_json::from_slice(&result.unwrap().app_metadata).unwrap();
            acked.push(reply["end_byte_offset"].as_i64().unwrap());
        }
        acked
    }

    /// The sender's question for the store: no rows, end offset 0. Its answer is offset 0.
    async fn ask(&mut self, step: DataType) {
        let empty = RecordBatch::new_empty(schema(step));
        self.send(empty, 0, None).await;
        let answer = self.reply().await.expect("the query's answer");
        assert_eq!(answer["end_byte_offset"], 0, "{answer}");
        assert!(answer.get("refused").is_none(), "{answer}");
    }
}

/// A: steps 0..3, bytes 0 to 300, stored and acknowledged by a first server (at the end of the
/// stream at the latest, for the binary, which stores a part every 10 seconds).
async fn store_a(address: &str) {
    let mut put = open(address, schema(DataType::Int64)).await;
    put.send(rows(&[0, 1, 2]), 300, Some(0)).await;
    assert_eq!(put.finish().await.last(), Some(&300));
}

/// A returning sender's stream to a server: the question, then B (bytes 300 to 500, `step` as text)
/// with the start it claims. Returns the refusal and the status that ended the stream.
async fn refuse_b(address: &str, claim: Option<i64>) -> (serde_json::Value, tonic::Status) {
    let mut put = open(address, schema(DataType::Utf8)).await;
    put.ask(DataType::Utf8).await;
    put.send(text_steps(&[3, 4]), 500, claim).await;
    let refusal = put.reply().await.expect("the refusal before the status");
    let status = put.reply().await.expect_err("the stream ends refused");
    assert_eq!(status.code(), Code::InvalidArgument);
    (refusal, status)
}

/// The run of the regression in `sdk/tests/test_wave_ad_trial.py`, with the server in this
/// process: A is stored, the server stops and a new one starts on the same store, and the returning
/// sender's B is refused. The refusal names B's bytes from the start B claimed (300), and the
/// acknowledgement it carries is 0, what the new server holds for the rank: it vouches for nothing
/// on the sender's word. Fails before lane AE1: the range began at 0.
#[tokio::test]
async fn after_a_restart_a_refusal_names_the_start_the_batch_claimed_and_acknowledges_nothing_more() {
    let dir = tempfile::tempdir().unwrap();
    {
        let first = start(dir.path()).await;
        store_a(&first.address).await;
        first.ingest.stop(std::iter::empty()).await.unwrap();
    }
    let second = start(dir.path()).await;
    let (refusal, _) = refuse_b(&second.address, Some(300)).await;
    assert_eq!(
        refusal["refused"],
        serde_json::json!({"start_byte_offset": 300, "end_byte_offset": 500, "column": "step"}),
        "{refusal}"
    );
    assert_eq!(refusal["end_byte_offset"], 0, "{refusal}");
    // Counted once however many times it comes.
    let (again, _) = refuse_b(&second.address, Some(300)).await;
    assert_eq!(again["refused"], refusal["refused"]);
    let status = second.ingest.watchdog_statuses();
    assert_eq!(status[RUN]["ranks"][0]["refused_batches"], 1, "{}", status[RUN]);
    assert_eq!(stored_steps(dir.path(), RUN, 0), vec![0, 1, 2]);
}

/// A sender older than the claim, across the same restart: the range is named from 0, as before,
/// and that sender, whose range begins at its saved offset, keeps it and loses nothing.
#[tokio::test]
async fn after_a_restart_a_batch_that_claims_no_start_is_named_from_byte_0_as_before() {
    let dir = tempfile::tempdir().unwrap();
    {
        let first = start(dir.path()).await;
        store_a(&first.address).await;
        first.ingest.stop(std::iter::empty()).await.unwrap();
    }
    let second = start(dir.path()).await;
    let (refusal, _) = refuse_b(&second.address, None).await;
    assert_eq!(
        refusal["refused"],
        serde_json::json!({"start_byte_offset": 0, "end_byte_offset": 500, "column": "step"})
    );
    assert_eq!(refusal["end_byte_offset"], 0);
}

/// A claim the server cannot take, at or past the batch's end or below 0, is left out: the range is
/// named as without it.
#[tokio::test]
async fn a_claim_that_is_not_below_the_end_or_is_negative_is_not_taken() {
    for claim in [500, 501, -1] {
        let dir = tempfile::tempdir().unwrap();
        let server = start(dir.path()).await;
        let (refusal, _) = refuse_b(&server.address, Some(claim)).await;
        assert_eq!(refusal["refused"]["start_byte_offset"], 0, "claim {claim}: {refusal}");
        assert_eq!(refusal["end_byte_offset"], 0, "claim {claim}: {refusal}");
    }
}

/// Where the server knows where the batch began, the claim changes nothing: a running server that
/// acknowledged A names B from 300, as before, whatever B claims. (A stream has one schema, so no
/// stream has a batch accepted and then one refused for a key column; that case is not built here.)
#[tokio::test]
async fn what_the_server_holds_comes_before_the_claim() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    store_a(&server.address).await;
    for claim in [Some(250), Some(300), None] {
        let (refusal, _) = refuse_b(&server.address, claim).await;
        assert_eq!(
            refusal["refused"]["start_byte_offset"], 300,
            "claim {claim:?}: {refusal}"
        );
        assert_eq!(refusal["end_byte_offset"], 300, "claim {claim:?}: {refusal}");
    }
}

/// The same run with the real binary killed (SIGKILL) between A and B: what was only in its memory is
/// gone, and the new process names B from the start it claimed.
#[tokio::test]
async fn after_the_binary_is_killed_a_refusal_names_the_start_the_batch_claimed() {
    let dir = tempfile::tempdir().unwrap();
    let first = Binary::start(dir.path(), "127.0.0.1:0");
    store_a(&first.url()).await;
    first.stop(); // SIGKILL
    let second = Binary::start(dir.path(), "127.0.0.1:0");
    let (refusal, _) = refuse_b(&second.url(), Some(300)).await;
    assert_eq!(
        refusal["refused"],
        serde_json::json!({"start_byte_offset": 300, "end_byte_offset": 500, "column": "step"}),
        "{refusal}"
    );
    assert_eq!(refusal["end_byte_offset"], 0, "{refusal}");
    second.stop();
    assert_eq!(stored_steps(dir.path(), RUN, 0), vec![0, 1, 2]);
}
