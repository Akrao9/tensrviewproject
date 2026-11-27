//! Lane AR-7 of wave AR (item 5 of the author's list): the library functions that read an Arrow IPC
//! stream offline end with an error on a record batch Arrow's reader cannot decode, and do not panic,
//! when they are called directly and not inside `decode_guard::offline` as `main` calls them.
//!
//! The input is lane AR-4's AQ5-R1 stream (a record batch whose body is cut to half of what its header
//! names), built here as in `lane_ar4_decode_guard_test.rs`. On `266c4ee` each of these calls panicked
//! inside `arrow-buffer`; the command did not, because `main` wraps it in `offline`.

use std::io::Cursor;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Value};
use tensorquill_server::decode_guard::MALFORMED_INPUT;
use tensorquill_server::detect::{self, Thresholds};

fn schema() -> Arc<Schema> {
    let scalars = DataType::Map(
        Arc::new(Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Float64, true),
            ])),
            false,
        )),
        false,
    );
    Arc::new(Schema::new(vec![
        Field::new("schema_version", DataType::Int32, true),
        Field::new("run_id", DataType::Utf8, true),
        Field::new("rank", DataType::Int32, true),
        Field::new("attempt", DataType::Int64, true),
        Field::new("step", DataType::Int64, true),
        Field::new("ts", DataType::Float64, true),
        Field::new("grad_norm", DataType::Float64, true),
        Field::new("scalars", scalars, true),
        Field::new("seq", DataType::Int64, true),
        Field::new("record_version", DataType::Int32, true),
        Field::new("extra_json", DataType::Utf8, true),
    ]))
}

fn steps(run: &str, steps: std::ops::Range<i64>) -> Vec<Value> {
    steps
        .map(|step| {
            json!({"schema_version": 2, "run_id": run, "rank": 0, "attempt": 1, "step": step,
                   "ts": step as f64, "grad_norm": 1.0, "record_version": 1})
        })
        .collect()
}

fn batch(rows: &[Value]) -> RecordBatch {
    let text: String = rows.iter().map(|row| format!("{row}\n")).collect();
    let mut reader = arrow::json::ReaderBuilder::new(schema())
        .with_batch_size(rows.len().max(1))
        .build(Cursor::new(text))
        .unwrap();
    reader.next().unwrap().unwrap()
}

/// `rows` as one Arrow IPC stream, as `detect` reads them.
fn ipc(rows: &[Value]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut writer = arrow::ipc::writer::StreamWriter::try_new(&mut out, &schema()).unwrap();
    writer.write(&batch(rows)).unwrap();
    writer.finish().unwrap();
    drop(writer);
    out
}

/// AQ5-R1 in an Arrow IPC stream: `rows` as one stream whose record batch header says its body is
/// half as long as the buffers it names reach, and whose body is cut to that. The stream reader reads
/// the shorter body and the decode of its buffers runs past it.
fn ipc_cut_short(rows: &[Value]) -> Vec<u8> {
    let whole = ipc(rows);
    // Each message: 0xFFFFFFFF, the header's length, the header (a flatbuffer `Message`), its body.
    let message_at = |at: usize| -> (usize, usize) {
        assert_eq!(whole[at..at + 4], [0xff; 4]);
        let length = u32::from_le_bytes(whole[at + 4..at + 8].try_into().unwrap()) as usize;
        (at + 8, length)
    };
    let (schema_header, schema_length) = message_at(0);
    let (header_at, header_length) = message_at(schema_header + schema_length);
    let mut header = whole[header_at..header_at + header_length].to_vec();
    let message = arrow::ipc::root_as_message(&header).unwrap();
    assert!(message.header_as_record_batch().is_some());
    let body = message.bodyLength() as usize;
    let half = (body / 2) & !7;
    // `bodyLength`, the `Message` table's fourth field, found through the table's vtable.
    let u16_at = |at: usize| u16::from_le_bytes(header[at..at + 2].try_into().unwrap()) as usize;
    let table = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let back = i32::from_le_bytes(header[table..table + 4].try_into().unwrap()) as isize;
    let vtable = (table as isize - back) as usize;
    let at = table + u16_at(vtable + 4 + 2 * 3);
    assert_eq!(i64::from_le_bytes(header[at..at + 8].try_into().unwrap()), body as i64);
    header[at..at + 8].copy_from_slice(&(half as i64).to_le_bytes());
    assert_eq!(arrow::ipc::root_as_message(&header).unwrap().bodyLength(), half as i64);
    let body_at = header_at + header_length;
    let mut out = whole[..header_at].to_vec();
    out.extend_from_slice(&header);
    out.extend_from_slice(&whole[body_at..body_at + half]);
    out.extend_from_slice(&[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0]);
    out
}

fn thresholds() -> Arc<Thresholds> {
    Arc::new(Thresholds::default())
}

fn frame(bytes: Vec<u8>) -> Vec<u8> {
    let mut out = (bytes.len() as u64).to_le_bytes().to_vec();
    out.extend(bytes);
    out
}

/// `call`'s error, which must name a malformed record batch as `detect` does, with no panic.
fn refused_as_malformed<T>(what: &str, call: impl FnOnce() -> Result<T, String>) -> String {
    match catch_unwind(AssertUnwindSafe(call)) {
        Ok(Ok(_)) => panic!("{what}: a cut-short stream was read"),
        Ok(Err(error)) => {
            assert!(error.starts_with(MALFORMED_INPUT), "{what}: {error}");
            assert!(
                error.contains("the offset of the new Buffer cannot exceed the existing length"),
                "{what}: {error}"
            );
            error
        }
        Err(_) => panic!("{what}: panicked on a cut-short stream"),
    }
}

#[test]
fn detect_stream_called_directly_returns_the_malformed_batch_error() {
    let rows = steps("ar7_detect", 0..4);
    assert!(detect::detect_stream(Cursor::new(ipc(&rows)), &thresholds()).is_ok());
    refused_as_malformed("detect_stream", || {
        detect::detect_stream(Cursor::new(ipc_cut_short(&rows)), &thresholds())
    });
}

#[test]
fn detect_stream_samples_called_directly_returns_the_malformed_batch_error_and_writes_nothing() {
    let rows = steps("ar7_detect", 0..4);
    let mut out = Vec::new();
    refused_as_malformed("detect_stream_samples", || {
        detect::detect_stream_samples(Cursor::new(ipc_cut_short(&rows)), &thresholds(), &mut out)
    });
    assert!(out.is_empty());
}

#[test]
fn read_batches_called_directly_returns_the_malformed_batch_error() {
    let rows = steps("ar7_detect", 0..4);
    assert_eq!(detect::read_batches(Cursor::new(ipc(&rows))).unwrap().len(), 1);
    refused_as_malformed("read_batches", || {
        detect::read_batches(Cursor::new(ipc_cut_short(&rows)))
    });
}

#[test]
fn detect_follow_called_directly_on_a_cut_first_frame_returns_the_malformed_batch_error() {
    let mut out = Vec::new();
    refused_as_malformed("detect_follow, first frame", || {
        detect::detect_follow(
            Cursor::new(frame(ipc_cut_short(&steps("ar7_detect", 0..4)))),
            &thresholds(),
            false,
            &mut out,
        )
    });
    assert!(out.is_empty());
}

#[test]
fn detect_follow_called_directly_answers_the_good_frame_then_returns_the_error_at_the_cut_one() {
    // As the command does: the good frame's answer, then the error (exit 1), not a "refused" answer.
    let mut input = frame(ipc(&steps("ar7_detect", 0..4)));
    input.extend(frame(ipc_cut_short(&steps("ar7_detect", 4..8))));
    let mut out = Vec::new();
    refused_as_malformed("detect_follow, a later frame", || {
        detect::detect_follow(Cursor::new(input), &thresholds(), true, &mut out)
    });
    let text = String::from_utf8(out).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 1, "{text}");
    let answer: Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(answer["follow"]["mode"], "full", "{answer}");
}

#[test]
fn a_later_frame_that_is_not_an_arrow_stream_is_still_refused_not_an_error() {
    // The reader's own errors keep their words and their place: a later frame that is not an Arrow IPC
    // stream is refused (`{"follow": {"refused": ...}}`, exit 0), as before.
    let mut input = frame(ipc(&steps("ar7_detect", 0..4)));
    input.extend(frame(b"not arrow".to_vec()));
    let mut out = Vec::new();
    detect::detect_follow(Cursor::new(input), &thresholds(), false, &mut out).unwrap();
    let text = String::from_utf8(out).unwrap();
    let last: Value = serde_json::from_str(text.lines().last().unwrap()).unwrap();
    assert!(
        last["follow"]["refused"]
            .as_str()
            .unwrap()
            .starts_with("the frame is not an Arrow IPC stream: "),
        "{last}"
    );
}
