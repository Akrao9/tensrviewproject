//! Lane D, wave AG (Nov 21, 2026): section 9 of `WAVE_AF_CONTRACT.md`, several refused batches in a
//! row (D5, RS10). The server keeps, per rank, the end of the last refusal it confirmed, and names a
//! refused batch that claims to begin there from it. Until then the second refused batch in a row
//! was named from the rank's last acknowledgement, which the sender, saved at the first refusal's
//! end, could not confirm (lane D's `test_several_refused_batches_then_a_valid_batch_then_a_reconnect`).
//!
//! Each batch states the start it claims (`start_byte_offset`), as the SDK's sender does from wave AE
//! on, and its end. A refused batch has a `step` column of text.

mod lane_aa3_support;

use std::path::Path;
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
use lane_aa3_support::{start, stored_steps};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Channel;
use tonic::Code;

const RUN: &str = "run-af-d-refusals";

fn schema(step: DataType) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("schema_version", DataType::Int32, false),
        Field::new("run_id", DataType::Utf8, false),
        Field::new("rank", DataType::Int32, false),
        Field::new("attempt", DataType::Int64, false),
        Field::new("step", step, false),
    ]))
}

/// Rows of rank 0, attempt 1, at `steps`; with `text`, the `step` column is text, which the server
/// refuses for good.
fn rows(steps: &[i64], text: bool) -> RecordBatch {
    let n = steps.len();
    let step: ArrayRef = if text {
        Arc::new(StringArray::from(
            steps.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        ))
    } else {
        Arc::new(Int64Array::from(steps.to_vec()))
    };
    RecordBatch::try_new(
        schema(if text { DataType::Utf8 } else { DataType::Int64 }),
        vec![
            Arc::new(Int32Array::from(vec![1; n])) as ArrayRef,
            Arc::new(StringArray::from(vec![RUN; n])),
            Arc::new(Int32Array::from(vec![0; n])),
            Arc::new(Int64Array::from(vec![1; n])),
            step,
        ],
    )
    .unwrap()
}

/// One stream that sends one batch, claiming `claim` as its start and `end` as its end, and the
/// server's replies until the stream ends: the messages, then the status if it ended in one.
async fn put(address: &str, batch: RecordBatch, claim: i64, end: i64) -> (Vec<Value>, Option<tonic::Status>) {
    let channel = Channel::from_shared(address.to_string())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightClient::new(channel);
    let (messages, outgoing) = mpsc::unbounded_channel::<Result<FlightData, FlightError>>();
    let descriptor = FlightDescriptor::new_path(vec!["runs".into(), RUN.into(), "rank=0".into()]);
    let opening: Vec<FlightData> = FlightDataEncoderBuilder::new()
        .with_flight_descriptor(Some(descriptor))
        .with_schema(batch.schema())
        .build(futures::stream::empty())
        .try_collect()
        .await
        .unwrap();
    for message in opening {
        messages.send(Ok(message)).unwrap();
    }
    let metadata = format!(
        "{{\"schema_version\":1,\"run_id\":\"{RUN}\",\"rank\":0,\"min_step\":null,\"max_step\":null,\"end_byte_offset\":{end},\"start_byte_offset\":{claim}}}"
    );
    for mut message in arrow_flight::utils::batches_to_flight_data(&batch.schema(), [&batch])
        .unwrap()
        .into_iter()
        .skip(1)
    {
        message.app_metadata = metadata.clone().into();
        messages.send(Ok(message)).unwrap();
    }
    drop(messages);
    let mut acks: BoxStream<'static, Result<PutResult, FlightError>> =
        match client.do_put(UnboundedReceiverStream::new(outgoing)).await {
            Ok(acks) => acks,
            Err(error) => futures::stream::once(async move { Err(error) }).boxed(),
        };
    let (mut replies, mut status) = (Vec::new(), None);
    loop {
        match tokio::time::timeout(Duration::from_secs(20), acks.next())
            .await
            .expect("no reply")
        {
            Some(Ok(result)) => replies.push(serde_json::from_slice(&result.app_metadata).unwrap()),
            Some(Err(FlightError::Tonic(ended))) => {
                status = Some(*ended);
                break;
            }
            Some(Err(other)) => panic!("{other}"),
            None => break,
        }
    }
    (replies, status)
}

/// The range a refusal named, as (start, end), and the acknowledgement its message carries.
async fn refused(address: &str, steps: &[i64], claim: i64, end: i64) -> ((i64, i64), i64) {
    let (replies, status) = put(address, rows(steps, true), claim, end).await;
    assert_eq!(status.expect("the stream ends refused").code(), Code::InvalidArgument);
    let told = replies
        .iter()
        .find(|reply| reply.get("refused").is_some())
        .expect("a refusal before the status");
    assert_eq!(told["refused"]["column"], "step", "{told}");
    (
        (
            told["refused"]["start_byte_offset"].as_i64().unwrap(),
            told["refused"]["end_byte_offset"].as_i64().unwrap(),
        ),
        told["end_byte_offset"].as_i64().unwrap(),
    )
}

/// A valid batch, stored and acknowledged to `end`.
async fn stored(address: &str, steps: &[i64], claim: i64, end: i64) {
    let (replies, status) = put(address, rows(steps, false), claim, end).await;
    assert!(status.is_none(), "{status:?}");
    assert_eq!(replies.last().unwrap()["end_byte_offset"], end, "{replies:?}");
}

fn count(root: &Path) -> Vec<i64> {
    stored_steps(root, RUN, 0)
}

/// A is stored (0 to 300); then three refused batches in a row, each claiming the end of the one
/// before: each is named exactly from where it claims to begin. Then a valid batch, stored. The
/// acknowledgement each refusal carries stays at what the server stored (300): it vouches for no
/// byte it did not store. Fails on the base (`fa1aa9a`): the second was named from 300.
#[tokio::test]
async fn three_refused_batches_in_a_row_are_each_named_from_where_the_one_before_ended() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    stored(&server.address, &[0, 1, 2], 0, 300).await;
    assert_eq!(refused(&server.address, &[3, 4], 300, 500).await, ((300, 500), 300));
    assert_eq!(refused(&server.address, &[5, 6], 500, 700).await, ((500, 700), 300));
    assert_eq!(refused(&server.address, &[7, 8], 700, 900).await, ((700, 900), 300));
    stored(&server.address, &[9, 10, 11], 900, 1200).await;
    assert_eq!(count(dir.path()), vec![0, 1, 2, 9, 10, 11]);
    let status = server.ingest.watchdog_statuses();
    assert_eq!(status[RUN]["ranks"][0]["refused_batches"], 3, "{}", status[RUN]);
}

/// A refusal whose confirmation was lost moves nothing: the sender, still at 300, sends the refused
/// bytes again (from 300, and with more after them), and they are named from 300, its own offset. A
/// claim that is neither what the server acknowledged nor the end of its last refusal is not taken:
/// the range is named from what the server holds, which that sender cannot confirm. Passes on the
/// base, which never took a claim once it had acknowledged anything.
#[tokio::test]
async fn a_lost_confirmation_and_a_claim_the_server_never_named_move_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let server = start(dir.path()).await;
    stored(&server.address, &[0, 1, 2], 0, 300).await;
    assert_eq!(refused(&server.address, &[3, 4], 300, 500).await, ((300, 500), 300));
    // The confirmation did not reach the sender: it sends 300 to 700 (the refused bytes and more).
    assert_eq!(
        refused(&server.address, &[3, 4, 5, 6], 300, 700).await,
        ((300, 700), 300)
    );
    // A claim at 600, which this server never named: named from 300.
    assert_eq!(refused(&server.address, &[6], 600, 800).await, ((300, 800), 300));
    // Once a valid batch is stored past the last refusal, its end is no longer taken: 700 was the
    // end of the refusal before the store of 700 to 1000.
    stored(&server.address, &[7, 8, 9], 700, 1000).await;
    assert_eq!(refused(&server.address, &[10], 800, 1100).await, ((1000, 1100), 1000));
    assert_eq!(count(dir.path()), vec![0, 1, 2, 7, 8, 9]);
}

/// Across a restart: the kept end is in memory only, and a restarted server holds no acknowledged
/// offset for the rank either, so it takes the batch's claim by RS10's restart rule (wave AE). After
/// a valid batch is stored on the new server, refusals in a row are named as on a server that never
/// stopped. What a sender sees: every refusal names the bytes it sent, and it moves across each.
/// The first half passes on the base (the restart rule); the refusals after the store fail on it.
#[tokio::test]
async fn across_a_restart_refused_batches_in_a_row_are_still_named_from_where_each_began() {
    let dir = tempfile::tempdir().unwrap();
    {
        let first = start(dir.path()).await;
        stored(&first.address, &[0, 1, 2], 0, 300).await;
        assert_eq!(refused(&first.address, &[3, 4], 300, 500).await, ((300, 500), 300));
        first.ingest.stop(std::iter::empty()).await.unwrap();
    }
    let second = start(dir.path()).await;
    // Nothing acknowledged on this server: the claims are taken, and the acknowledgement is 0.
    assert_eq!(refused(&second.address, &[5, 6], 500, 700).await, ((500, 700), 0));
    assert_eq!(refused(&second.address, &[7, 8], 700, 900).await, ((700, 900), 0));
    stored(&second.address, &[9, 10], 900, 1100).await;
    assert_eq!(refused(&second.address, &[11], 1100, 1200).await, ((1100, 1200), 1100));
    assert_eq!(refused(&second.address, &[12], 1200, 1300).await, ((1200, 1300), 1100));
    stored(&second.address, &[13], 1300, 1400).await;
    assert_eq!(count(dir.path()), vec![0, 1, 2, 9, 10, 13]);
}
