//! The one reader of a row's `extra_json` text, shared by every detector that reads a key there.

use serde_json::Value;

/// The row's `extra_json` as JSON, or None when it is not JSON.
pub(crate) fn parse(text: &str) -> Option<Value> {
    serde_json::from_str(text).ok()
}
