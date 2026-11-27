//! The one reader of a row's `extra_json` text, shared by every detector that reads a key there
//! (`detect.rs`: `step_skipped_by` and `step_skipped_part`, `clip_calls`, refused `grad_norms`,
//! `moe_refused`, `update_not_read`, the stalled-weight onset's notes, `grads_not_read`,
//! `layers_not_read`, `moe_source`, `scalars` for a row whose column is null (a row stored before
//! AQ5-D7, `Scalars::logged`) and, in protocol 5's layout, a layer's refused fields for
//! `non_finite`; `replica.rs`: the observation, its scope and the spread's
//! fallback), and of the four other JSON columns (lane AR-3, Nov 26, 2026, AQ5-D2; the author:
//! "The Nov 22 reading of an oversized integer extended to `system_json`, `data_json`,
//! `probe_json`, `checkpoint_json` in Rust"): `system.rs`'s reading of `system_json` and
//! `data.rs`'s of `data_json` and `checkpoint_json` (each [`read_column`]), and `detect.rs`'s two
//! of `probe_json`, the head's ([`parse`]) and the attention rules' and `attention_trends`' typed
//! one ([`parse_into`]). Every route reads them there: the raw
//! file `tensorquill-server detect` reads, the live detectors, and the rows a restart reads back
//! from the store, which holds each column's text as it was sent.
//!
//! A number no `f64` can hold, such as the `10**400` a refused map entry carries (the sender keeps
//! the integer in full, so that Python reads it back exactly), made serde_json refuse the whole
//! text, and the row lost every key in it. The author (Nov 22, 2026): "treat it as unreadable
//! where a float measurement is required, and keep valid sibling fields available to Rust."
//!
//! So when the text does not parse, each number outside a string that overflows an `f64` is read
//! as the string [`UNREADABLE`], and the text is parsed again. A string is unreadable wherever the
//! detectors want a number (a spread entry is left out, a count says nothing, a fingerprint is not
//! finite as any other text is), and the other values are read as written. A row that parses is
//! read exactly as before: nothing changes for any other number, nor for what a report writes.
//! The alternative, serde_json's `arbitrary_precision`, would change how every number the server
//! parses is held and written again, so it is not used.

use serde::Deserialize;
use serde_json::Value;

#[cfg(test)]
#[path = "lane_ar3_tests.rs"]
mod lane_ar3_tests;

/// What a number too large for a float reads as.
pub(crate) const UNREADABLE: &str = "a number too large for a float";

/// The row's `extra_json` as JSON, or None when it is not JSON for any reason but a number too
/// large for a float.
pub(crate) fn parse(text: &str) -> Option<Value> {
    match serde_json::from_str(text) {
        Ok(value) => Some(value),
        Err(_) => serde_json::from_str(&without_oversized(text)?).ok(),
    }
}

/// A JSON column's text read as [`parse`] reads it, and whether a number in it was read as
/// [`UNREADABLE`].
pub(crate) struct Column {
    pub(crate) value: Value,
    /// A number too large for a float was read as [`UNREADABLE`]: a string equal to it is then
    /// that number, not text (see [`Column::text`]).
    pub(crate) oversized: bool,
}

impl Column {
    /// `value`, a part of this column, as text, as Python's `isinstance(value, str)` takes it: in a
    /// column whose oversized numbers were read as [`UNREADABLE`], that string is a number and no
    /// text. So an oversized integer is no batch hash, throttle reason or tensor name. In a text that
    /// parses, every string is text, those words included.
    pub(crate) fn text<'v>(&self, value: &'v Value) -> Option<&'v str> {
        value.as_str().filter(|text| !(self.oversized && *text == UNREADABLE))
    }
}

/// The column's text as JSON, as [`parse`] reads it, or None when it is not JSON for any reason
/// but a number too large for a float.
pub(crate) fn read_column(text: &str) -> Option<Column> {
    match serde_json::from_str(text) {
        Ok(value) => Some(Column {
            value,
            oversized: false,
        }),
        Err(_) => serde_json::from_str(&without_oversized(text)?)
            .ok()
            .map(|value| Column { value, oversized: true }),
    }
}

/// [`parse`] for a typed reader that borrows from the text (`probe_json`'s `Probe`): `T` read from
/// `text`, or, when only a number too large for a float keeps the text from parsing, from the text
/// with each such number written as [`UNREADABLE`], held in `rewritten` for `T` to borrow from. A
/// text that parses is read from itself, and nothing is held.
pub(crate) fn parse_into<'t, T: Deserialize<'t>>(text: &'t str, rewritten: &'t mut Option<String>) -> Option<T> {
    match serde_json::from_str(text) {
        Ok(value) => Some(value),
        Err(_) => {
            let rewritten: &'t str = Option::insert(rewritten, without_oversized(text)?);
            serde_json::from_str(rewritten).ok()
        }
    }
}

/// `text` with every number outside a string that overflows an `f64` written as the string
/// [`UNREADABLE`], or None when it has none (then nothing else can make it parse).
fn without_oversized(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let (mut at, mut copied, mut found, mut in_string) = (0, 0, false, false);
    while at < bytes.len() {
        let byte = bytes[at];
        if in_string {
            match byte {
                b'\\' => at += 1, // the escaped byte is skipped with it
                b'"' => in_string = false,
                _ => {}
            }
            at += 1;
            continue;
        }
        match byte {
            b'"' => {
                in_string = true;
                at += 1;
            }
            b'-' | b'0'..=b'9' => {
                let start = at;
                while at < bytes.len() && matches!(bytes[at], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                    at += 1;
                }
                // Token bounds are ASCII bytes outside a string, so the slices are whole characters.
                if text[start..at].parse::<f64>().is_ok_and(f64::is_infinite) {
                    out.push_str(&text[copied..start]);
                    out.push('"');
                    out.push_str(UNREADABLE);
                    out.push('"');
                    copied = at;
                    found = true;
                }
            }
            _ => at += 1,
        }
    }
    found.then(|| {
        out.push_str(&text[copied..]);
        out
    })
}
