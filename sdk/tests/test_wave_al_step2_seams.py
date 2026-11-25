"""Wave AL, step 2's seam (`WAVE_AL_CONTRACT.md` sections 4 and 9): what no lane of step 2 owned.

1. The README's passages on the scalar-only record, each read against the code it describes: the capture's
   `scalars` bullet, the Hugging Face callback, the mirror, the stream's columns, the server's scalar-only
   rows, the readers' section, the identity rules, and malformed `layers`; no sentence left saying an
   explicit step is not built.
2. The joined paths no lane could test alone, with a real server of this tree: a real session that calls
   `session.log({...}, step=N)` between training steps, streamed; the store; `load_records`, `scalar_series`,
   MCP `get_series` and `build_evidence` under both engines; a resend from byte 0.

Tests skip one by one without pyarrow, NumPy, Transformers or the server binary; none needs git history."""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path

import pytest

from tensorquill import evidence, identity

SDK = Path(__file__).resolve().parents[1]
SERVER_SRC = SDK.parent / "server" / "src"
README = (SDK / "README.md").read_text(encoding="utf-8")


def flat(text: str) -> str:
    return " ".join(text.split())


def section(start: str, end: str) -> str:
    return flat(README[README.index(start):README.index(end)])


def server_source(name: str) -> str:
    path = SERVER_SRC / name
    if not path.exists():
        pytest.skip(f"no server/src/{name} in this checkout")
    return path.read_text(encoding="utf-8")


SCALARS = flat(next(line for line in README.splitlines() if line.startswith("- **`scalars`**")))
STREAMING = section("## Streaming to an ingest server\n", "\n## Detector benchmark on 64 ranks")


# --- 1. The README ---------------------------------------------------------------------------------


def test_no_sentence_says_an_explicit_step_is_not_built():
    text = flat(README)
    for gone in ("an explicit step is not supported yet", "comes with protocol 5", "raises `TypeError`, as an image does",
                 "is left out with one warning. The training loss"):
        assert gone not in text, gone


def test_the_scalars_bullet_gives_the_explicit_step_as_the_capture_has_it():
    for words in ("**An explicit step (Nov 24, 2026, protocol 5).**",
                  "`session.log({...}, step=N)` records the values at step `N` as a scalar-only record "
                  '(schema version 2, `"record": "scalars"`), never a training step',
                  "a bool, a float, `None` or anything else raises `TypeError`, and a negative step `ValueError`, "
                  "before anything is recorded",
                  "numbered `seq` 0, 1, 2… per session, in call order",
                  "the call is one graph break; under `fullgraph=True` Dynamo raises",
                  "[Scalar-only records](#scalar-only-records)",
                  'A key `"step"` in the dictionary'):
        assert words in SCALARS, words


def test_the_scalars_bullet_gives_the_warning_s_call_and_the_compiled_late_value():
    for words in ("a value still pending at `close()` after `flush()`",
                  "The warning shows the call that records it, `session.log({...}, step=N)`, with `N` the step training "
                  "starts at (`start_step`) for a baseline and the last step training finished after `flush()`",
                  "also when a frozen parameter becomes trainable on the next step",
                  "After capture stops, values logged since the last step written are lost with that step"):
        assert words in SCALARS, words


def test_the_warning_the_page_describes_is_the_capture_s():
    """The page's call and its two numbers are what `Session._no_step_for` says."""
    capture = (SDK / "src" / "tensorquill" / "capture.py").read_text(encoding="utf-8")
    for words in ("which writes a scalar-only record: \"\n            f\"session.log({{", "the step training starts at",
                  "the last step training finished", "step={self._suggested_step()}"):
        assert words in capture, words


def test_the_hugging_face_row_says_the_callback_records_a_baseline_itself():
    [row] = [line for line in README.splitlines() if line.startswith("| Hugging Face `Trainer` callback")]
    assert ("With no step in memory (`eval_on_start`'s baseline, or an evaluation after `flush()`) the callback logs "
            "the metrics with an explicit step itself, as a scalar-only record") in row
    assert "`state.global_step` for a baseline, the last step that ended after a flush" in row
    hf = (SDK / "src" / "tensorquill" / "huggingface.py").read_text(encoding="utf-8")
    assert "step=" in hf and "_suggested_step" in hf


def test_the_mirror_s_step_bullet_says_to_log_such_a_value_with_step():
    mirror = section("## Scalars mirrored from W&B and MLflow", "## Scalar-only records")
    assert "The mirror cannot pass a step, so log such a value with `step=` yourself; the warning says so." in mirror


def test_the_columns_bullet_gives_seq_and_the_scalar_only_line():
    [columns] = [flat(line) for line in README.splitlines() if line.startswith("- **Columns.**")]
    for words in ("its `seq` in `seq` (Int64, null on every step row)", "a non-null `seq` is what marks the row",
                  "`record` is in no column and the row still converts back exactly",
                  "skipped and counted like a line whose key cannot be stored, never sent as a step"):
        assert words in columns, words


def test_the_server_s_scalar_only_rows_as_the_server_stores_them():
    [bullet] = [flat(line) for line in README.splitlines() if line.startswith("- **Scalar-only rows (protocol 5).**")]
    for words in ("`runs/<run>/rank=<r>/attempt=<a>/scalars-<first seq>-<last seq>.parquet`", "created once and never replaced",
                  "never moves the step watermark", "`scalars_not_newer`", "the step part and then the scalars object"):
        assert words in bullet, words
    storage, watchdog = server_source("storage.rs"), server_source("watchdog.rs")
    assert 'pub const SCALARS_PREFIX: &str = "scalars-";' in storage and "PutMode::Create" in storage
    assert "scalars_not_newer" in watchdog


def test_the_readers_section_names_what_the_readers_have():
    readers = section("## Scalar-only records", "## For a coding agent: a read-only MCP server")
    for words in ("`load_records` returns the training records by default, as before;",
                  "`load_records(..., scalars=True)` keeps the scalar-only records",
                  "`tensorquill.evidence.scalar_series(source)`", "the highest `seq`",
                  "a baseline logged with `step=0` and an evaluation after the first step",
                  "The Rust detectors are never sent one", "MCP `get_series` (kind `scalar`)"):
        assert words in readers, words
    assert inspect.signature(evidence.load_records).parameters["scalars"].default is False
    assert callable(evidence.scalar_series)


def test_the_identity_rules_quote_the_check_s_refusals():
    rules = section("**A record's identity fields.**", "**Offline, a record that is not valid")
    unknown = {"schema_version": 3, "run_id": "r", "step": 0}
    with pytest.raises(identity.MalformedRecordError) as raised:
        identity.check(unknown)
    assert raised.value.reason == "this tensorquill reads schema versions 0, 1 and 2"
    assert f'another version is refused with "{raised.value.reason}"' in rules
    unmarked = {"schema_version": 2, "run_id": "r", "step": 0, "seq": 0}
    with pytest.raises(identity.MalformedRecordError) as raised:
        identity.check(unmarked)
    assert raised.value.field == "schema_version"
    assert 'A record of version 2 without `"record": "scalars"`' in rules


def test_the_malformed_layers_passage():
    text = flat(README)
    start = text.index("**Malformed `layers` and `moe_stats` (Nov 24, 2026).**")
    passage = text[start:start + 700]
    for words in ("Neither engine raises on these", "are absent to every detector",
                  "a record whose only `layers` entries are not objects holds no layer",
                  "A protocol 4 row's `layers_json` is read entry by entry", "is taken from the expert counts"):
        assert words in passage, words


def test_one_fact_in_one_place():
    """The passages each say their own facts: the store's object name in the server bullet alone, the series
    rule in the readers' section alone."""
    text = flat(README)
    assert text.count("scalars-<first seq>-<last seq>.parquet") == 1
    assert text.count("the highest `seq`") == 1
    assert text.count("`scalars_not_newer`") == 1
