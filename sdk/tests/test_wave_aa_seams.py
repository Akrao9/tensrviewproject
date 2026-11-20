"""Where the lanes of wave AA meet, on the tree that holds all of them.

Lane AA3's server now drops a row whose `schema_version` it does not support and counts it under
`unsupported_schema_version`, beside the rows it drops for a null key. Lane Z5's reader of that
count (`runs.dropped_rows`) said every dropped row had a null key, and neither lane could change the
other's file.

Fail before (the trial tree without the change in `runs.py`): the note and the listing's line say
"null key" of rows that were dropped for their schema version.
"""

from __future__ import annotations

from tensorquill import runs


def _status(keys: dict) -> dict:
    return {"ranks": [{"rank": 0, "dropped_rows": sum(keys.values()), "dropped_row_keys": keys}]}


def test_rows_dropped_for_their_schema_version_are_not_called_null_keys():
    dropped = runs.dropped_rows(_status({"step": 2, "unsupported_schema_version": 3}))
    assert dropped["rows"] == 5 and dropped["by_key"] == {"step": 2, "unsupported_schema_version": 3}
    assert "schema version" in dropped["note"] and dropped["upper_bound"] is True
    text = runs._dropped_text(dropped)
    assert "with a null key or an unsupported schema version" in text
    assert "by reason: step 2, unsupported_schema_version 3" in text and "by null key" not in text


def test_a_run_that_dropped_rows_for_null_keys_alone_reads_as_before():
    dropped = runs.dropped_rows(_status({"step": 2}))
    assert dropped["note"] == runs.DROPPED_ROWS_NOTE
    text = runs._dropped_text(dropped)
    assert "with a null key and stored none of them" in text and "by null key: step 2" in text
    assert "schema version" not in text
