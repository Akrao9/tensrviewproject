//! The guard around Arrow's decode of bytes a client or a user gives (lane AR-4 of wave AR: AQ5-R1 and
//! AQ5-R2).
//!
//! `arrow-ipc` 60.1.0 panics, through `arrow-buffer`, on some malformed record batch messages: a
//! header that names buffers its body does not hold (AQ5-R1, `Buffer::slice`), or a buffer whose
//! length is not a whole number of its values (AQ5-R2, `typed_data`). Both are reached by a client's
//! `DoPut` and by `tensorquill-server detect`. Validating a message before Arrow reads it cannot be
//! shown to cover everything Arrow's reader asserts, so the decode is caught instead: [`guard`] runs
//! a call under `catch_unwind` and says whether a panic inside it was raised in an Arrow crate.
//!
//! The process-wide panic hook is chained once ([`guard`] installs it on first use): a panic raised in
//! an Arrow crate during a guarded call on the same thread is not printed, since the caller turns it
//! into an error of its own. Any other panic is printed by the hook that was there before. The flag
//! the hook reads is set only for the length of one synchronous guarded call, never across an
//! `.await`, so it cannot leak to another task that runs later on the same thread. If something else
//! replaces the hook after this one was installed, a panic in Arrow is no longer known as one
//! ([`Caught::Other`]), which callers treat as they did before this guard.

use std::any::Any;
use std::cell::Cell;
use std::panic::{self, UnwindSafe};
use std::sync::Once;

thread_local! {
    /// True while a guarded call runs on this thread.
    static GUARDED: Cell<bool> = const { Cell::new(false) };
    /// Set by the hook when a panic inside the guarded call now running was raised in an Arrow crate.
    static IN_ARROW: Cell<bool> = const { Cell::new(false) };
}

/// A panic caught by [`guard`].
pub enum Caught {
    /// Raised in an Arrow crate (`arrow-*`), with its message: the decode refused the bytes. The
    /// hook did not print it.
    Arrow(String),
    /// Raised anywhere else, as it was; the hook printed it.
    Other(Box<dyn Any + Send>),
}

/// Whether `file`, a panic's location, is in one of Arrow's crates (`arrow-buffer-60.1.0/src/...` in
/// the cargo registry, `arrow-buffer/src/...` when vendored).
fn in_arrow(file: &str) -> bool {
    file.split(['/', '\\']).any(|part| part.starts_with("arrow-"))
}

fn install_hook() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            let guarded = GUARDED.try_with(Cell::get).unwrap_or(false);
            if guarded && info.location().is_some_and(|location| in_arrow(location.file())) {
                let _ = IN_ARROW.try_with(|flag| flag.set(true));
                return;
            }
            previous(info);
        }));
    });
}

/// The message of a panic's payload, at most 200 characters.
fn text(payload: &(dyn Any + Send)) -> String {
    let text = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("a panic without a message");
    text.chars().take(200).collect()
}

/// Runs `call`, catching a panic. The caller must not use again anything `call` may have left half
/// changed (what it touched through `AssertUnwindSafe`): that is its reason for wrapping it.
pub fn guard<T>(call: impl FnOnce() -> T + UnwindSafe) -> Result<T, Caught> {
    install_hook();
    let outer = GUARDED.with(|flag| flag.replace(true));
    let outer_arrow = IN_ARROW.with(|flag| flag.replace(false));
    let result = panic::catch_unwind(call);
    let arrow = IN_ARROW.with(|flag| flag.replace(outer_arrow));
    GUARDED.with(|flag| flag.set(outer));
    result.map_err(|payload| {
        if arrow {
            Caught::Arrow(text(payload.as_ref()))
        } else {
            Caught::Other(payload)
        }
    })
}

/// The first words of the error `tensorquill-server detect` ends with when Arrow's reader panics on
/// its input.
pub const MALFORMED_INPUT: &str = "cannot read the Arrow IPC stream: a record batch in it is malformed";

/// `tensorquill-server detect`, or any offline read of an Arrow IPC stream, guarded: a panic raised in
/// Arrow becomes the error `MALFORMED_INPUT` with Arrow's message (no location: the hook printed
/// nothing), which `main` prints and ends with exit code 1, as for any input it cannot read. Any other
/// panic (a fault of the detectors) goes on unwinding, as before, with its trace and exit code 101.
///
/// Since lane AR-7 of wave AR the reads in `detect.rs` and `detect/data.rs` are guarded themselves
/// ([`collect_batches`]), so the library functions return this error and do not panic; `main`'s call
/// stays as a backstop. The call spans the detectors as well as the reader, so a panic in Arrow raised
/// by the detectors' own use of an array would still be named a malformed input here (none is known).
/// Sound to catch: the caller ends with the error and uses nothing the call held.
pub fn offline<T, E: From<String>>(call: impl FnOnce() -> Result<T, E> + UnwindSafe) -> Result<T, E> {
    match guard(call) {
        Ok(result) => result,
        Err(Caught::Arrow(text)) => Err(E::from(format!(
            "{MALFORMED_INPUT}, and the Arrow reader stopped on it ({text})"
        ))),
        Err(Caught::Other(payload)) => panic::resume_unwind(payload),
    }
}

/// The record batches an Arrow IPC stream reader gives, collected inside [`guard`]: what `detect.rs`'s
/// readers (`read_batches`, `detect_rows_counted`, `follow_append`) call in place of
/// `reader.collect()` (since lane AR-7), so a malformed batch is an error of the read itself and not a panic, and
/// [`offline`] then only stands behind them. A panic in Arrow is the error [`MALFORMED_INPUT`] with
/// Arrow's message; any other error is the reader's own error's text, as `collect` gave it (the caller adds its own words, as now). Any other panic goes on
/// unwinding.
pub fn collect_batches<R: std::io::Read>(
    reader: arrow::ipc::reader::StreamReader<R>,
) -> Result<Vec<arrow::record_batch::RecordBatch>, String> {
    // Sound to catch: the reader is consumed by the call and nothing of it is used after a panic.
    match guard(std::panic::AssertUnwindSafe(|| reader.collect::<Result<Vec<_>, _>>())) {
        Ok(read) => read.map_err(|error| error.to_string()),
        Err(Caught::Arrow(text)) => Err(format!(
            "{MALFORMED_INPUT}, and the Arrow reader stopped on it ({text})"
        )),
        Err(Caught::Other(payload)) => panic::resume_unwind(payload),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_location_is_in_arrow_by_its_crate_folder() {
        assert!(in_arrow(
            "registry/src/index.crates.io-1949cf8c6b5b557f/arrow-buffer-60.1.0/src/buffer/immutable.rs"
        ));
        assert!(in_arrow("vendor\\arrow-ipc\\src\\reader.rs"));
        assert!(!in_arrow("src/detect.rs"));
        assert!(!in_arrow("/rustc/abc/library/core/src/slice/index.rs"));
        assert!(!in_arrow("tests/arrow_like.rs"));
    }

    #[test]
    fn a_panic_outside_arrow_is_caught_as_other_and_the_flags_are_put_back() {
        let caught = guard(|| -> u8 { panic!("not arrow") });
        assert!(matches!(caught, Err(Caught::Other(_))));
        assert!(!GUARDED.with(Cell::get));
        assert!(!IN_ARROW.with(Cell::get));
        assert_eq!(guard(|| 7).ok(), Some(7));
    }

    #[test]
    fn a_panic_in_arrow_is_caught_as_arrow_with_its_message() {
        let caught = guard(|| arrow::buffer::Buffer::from(vec![0_u8; 8]).slice(64));
        match caught {
            Err(Caught::Arrow(text)) => assert!(text.contains("the offset of the new Buffer"), "{text}"),
            Err(Caught::Other(_)) => panic!("raised in arrow-buffer, so caught as Arrow"),
            Ok(_) => panic!("a slice past the end panics"),
        }
        assert!(!GUARDED.with(Cell::get));
    }

    #[test]
    fn offline_turns_a_panic_in_arrow_into_its_error_and_lets_any_other_go_on() {
        let found: Result<(), String> = offline(|| {
            let _ = arrow::buffer::Buffer::from(vec![0_u8; 8]).slice(64);
            Ok(())
        });
        assert!(found.unwrap_err().starts_with(MALFORMED_INPUT));
        let other = panic::catch_unwind(|| offline::<(), String>(|| panic!("a detector's fault")));
        assert!(other.is_err());
    }

    #[test]
    fn collect_batches_gives_the_batches_or_the_reader_s_error() {
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![Field::new("step", DataType::Int64, false)]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![1_i64, 2, 3]))],
        )
        .unwrap();
        let mut bytes = Vec::new();
        let mut writer = arrow::ipc::writer::StreamWriter::try_new(&mut bytes, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        drop(writer);
        let read = |bytes: &[u8]| {
            collect_batches(
                arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes.to_vec()), None).unwrap(),
            )
        };
        assert_eq!(read(&bytes).unwrap(), vec![batch]);
        // Cut inside the batch's body: the reader's own error, not the malformed-batch one.
        let error = read(&bytes[..bytes.len() - 12]).unwrap_err();
        assert!(!error.starts_with(MALFORMED_INPUT), "{error}");
    }
}
