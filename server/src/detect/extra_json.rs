//! The one reader of a row's `extra_json` text, shared by every detector that reads a key there
//! (`detect.rs`: `step_skipped_by` and `step_skipped_part`, `clip_calls`, refused `grad_norms`,
//! `moe_refused`, `update_not_read`, the stalled-weight onset's notes, `grads_not_read`,
//! `layers_not_read`, `moe_source` and, in protocol 5's layout, a layer's refused fields for
//! `non_finite`; `replica.rs`: the observation, its scope and the spread's
//! fallback).
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
