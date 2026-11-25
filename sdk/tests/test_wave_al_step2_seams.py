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
                  # The owner's turns of lane AL-5's decisions 1, 4 and 9 (Nov 24), each beside `step=`.
                  "`step=None` is the same as leaving `step=` out",
                  "a bool, a float, a string or a negative integer raises (`TypeError`, or `ValueError` for a negative "
                  "step) before anything is recorded",
                  "numbered `seq` 0, 1, 2… per session, in call order",
                  "at an explicit `flush()` (also with no step in memory)",
                  "inside a function compiled with `torch.compile(fullgraph=True)`, `step=` raises Dynamo's `Unsupported`",
                  "Call `session.log(..., step=N)` outside the compiled function",
                  "or compile without `fullgraph`, which costs one graph break",
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


# --- 2. The joined paths, with a real server of this tree ------------------------------------------


def _binary() -> str | None:
    path = os.environ.get("TENSORQUILL_SERVER_BIN")
    return path if path and Path(path).is_file() else None


def _have(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


needs_server = pytest.mark.skipif(_binary() is None or not _have("pyarrow"),
                                  reason="needs the server binary (TENSORQUILL_SERVER_BIN) and pyarrow")


class _Server:
    """A server of this tree on a free port it picks itself (`--listen 127.0.0.1:0`), stopped with SIGTERM."""

    def __init__(self, store: Path) -> None:
        import subprocess

        env = {k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"}
        self.process = subprocess.Popen([_binary(), "--listen", "127.0.0.1:0", "--store", str(store),
                                         "--watchdog-seconds", "0.5", "--flush-seconds", "0.2"],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        line = self.process.stdout.readline()
        assert "listening on" in line, line
        self.endpoint = f"grpc://127.0.0.1:{int(line.rsplit(':', 1)[1])}"

    def close(self) -> None:
        import signal

        self.process.send_signal(signal.SIGTERM)
        self.process.wait(30)
        self.process.stdout.close()
        self.process.stderr.close()


def _lines(path: Path) -> list[dict]:
    import json

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _entry(run: Path) -> dict:
    import json

    status = json.loads((run / "watchdog.json").read_text(encoding="utf-8"))
    [entry] = status["ranks"] if isinstance(status["ranks"], list) else status["ranks"].values()
    return entry


EXPECTED_EVAL = [[0, 2.5], [5, 1.5], [8, 1.25], [11, 1.0]]


@pytest.fixture
def streamed(tmp_path, monkeypatch):
    """A real session of 12 steps that calls `session.log({...}, step=N)` before training (a baseline), between
    steps (one with a tensor value) and after the last step, streamed to a server of this tree, which then
    stops. The rank's `.jsonl`, the store's run folder, and the run ID."""
    torch = pytest.importorskip("torch")
    import tensorquill

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_TOKEN", raising=False)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    store = tmp_path / "store"
    server = _Server(store)
    try:
        session = tensorquill.init(model, run_id="s2", every=5, endpoint=server.endpoint)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        session.log({"eval/loss": 2.5}, step=0)  # a baseline, before the first step
        for step in range(12):
            optimizer.zero_grad()
            loss = model(torch.randn(16, 8)).pow(2).mean()
            loss.backward()
            session.log({"train/loss": loss})
            optimizer.step()
            if step == 5:
                session.log({"eval/loss": 1.5, "eval/acc": 0.25}, step=5)
            if step == 8:
                session.log({"eval/loss": torch.tensor(1.25)}, step=8)  # read with the next block, no sync
        session.log({"eval/loss": 1.0}, step=11)
        session.close()
        import time

        time.sleep(1.5)  # a watchdog pass writes the rank's status
    finally:
        server.close()
    return tmp_path / "tensorquill-out" / "s2.rank0.jsonl", store / "runs" / "s2", "s2"


@needs_server
def test_a_real_session_s_scalar_only_records_are_written_streamed_and_stored_apart(streamed):
    import pyarrow.parquet as pq

    path, run, _ = streamed
    written = _lines(path)
    scalar = [r for r in written if identity.is_scalar_record(r)]
    assert [(r["seq"], r["step"]) for r in scalar] == [(0, 0), (1, 5), (2, 8), (3, 11)]
    assert all(list(r) == ["schema_version", "record", "run_id", "rank", "attempt", "step", "seq", "ts", "scalars"]
               for r in scalar)
    assert [r["step"] for r in written if not identity.is_scalar_record(r)] == list(range(12))
    parts = sorted(run.rglob("part-*.parquet"))
    objects = sorted(run.rglob("scalars-*.parquet"))
    assert parts and objects
    steps = [s for p in parts for s in pq.read_table(p, partitioning=None).column("step").to_pylist()]
    assert steps == list(range(12))  # the parts hold the training steps alone (D8)
    assert {v for p in parts for v in pq.read_table(p, partitioning=None).column("seq").to_pylist()} == {None}
    seqs = []
    for obj in objects:
        found = re.fullmatch(r"scalars-(\d{8})-(\d{8})\.parquet", obj.name)
        column = pq.read_table(obj, partitioning=None).column("seq").to_pylist()
        assert found and (int(found.group(1)), int(found.group(2))) == (column[0], column[-1]), obj.name
        seqs += column
    assert seqs == [0, 1, 2, 3]
    entry = _entry(run)
    assert not {"dropped_rows", "dropped_row_keys", "scalars_not_newer"} & set(entry), entry


@needs_server
def test_load_records_gives_the_capture_s_records_back_from_the_store(streamed):
    path, run, _ = streamed
    from_store = evidence.load_records(run, scalars=True, all_attempts=True)
    from_file = evidence.load_records(path, scalars=True, all_attempts=True)
    training, scalar = evidence.set_apart_scalars(from_store)
    file_training, file_scalar = evidence.set_apart_scalars(from_file)
    assert training == file_training
    assert sorted(scalar, key=lambda r: r["seq"]) == file_scalar  # each record exactly as written
    assert evidence.load_records(run) == evidence.load_records(path) == file_training  # the default, as before


@needs_server
def test_the_merged_series_from_the_store_the_file_and_mcp(streamed):
    import json

    from tensorquill import mcp_server

    path, run, _ = streamed
    series = evidence.scalar_series(run)
    assert series == evidence.scalar_series(path)
    assert series["eval/loss"] == EXPECTED_EVAL and series["eval/acc"] == [[5, 0.25]]
    assert [s for s, _ in series["train/loss"]] == list(range(12))
    for source in (path, run):
        reply = mcp_server.call_tool(mcp_server.Source(source), 1,
                                     {"name": "get_series", "arguments": {"name": "eval/loss", "kind": "scalar"}})
        reply = reply["result"]
        assert not reply["isError"], reply
        got = json.loads(reply["content"][0]["text"])
        assert [[s, v] for s, v in zip(got["steps"], got["values"]["value"])] == EXPECTED_EVAL, source


def _judged(document: dict) -> dict:
    """The evidence without what says where and when it was made."""
    return {k: v for k, v in document.items() if k not in ("generated_at", "source", "sources")}


@needs_server
@pytest.mark.parametrize("engine", ["python", "rust"])
def test_build_evidence_on_the_store_gives_what_the_training_records_alone_give(streamed, engine):
    path, run, _ = streamed
    mixed = evidence.load_records(run, scalars=True, all_attempts=True)
    training, scalar = evidence.set_apart_scalars(mixed)
    assert scalar
    with_scalars = evidence.build_evidence(mixed, detectors=engine)
    alone = evidence.build_evidence(training, detectors=engine)
    assert with_scalars.get("detectors", engine) == engine
    assert _judged(with_scalars) == _judged(alone)
    assert "repeated_steps" not in [i["kind"] for i in with_scalars["incidents"]]


@needs_server
def test_a_resend_from_byte_0_stores_nothing_twice(streamed):
    import pyarrow.parquet as pq

    from tensorquill import flight

    path, run, run_id = streamed
    parts, objects = sorted(run.rglob("part-*.parquet")), sorted(run.rglob("scalars-*.parquet"))
    flight.offset_path(path).unlink()  # the offset is lost: the whole file is sent again
    server = _Server(run.parent.parent)
    try:
        assert flight.send_file(path, server.endpoint, run_id, 0, timeout=60) is True
        import time

        time.sleep(1.5)
    finally:
        server.close()
    assert sorted(run.rglob("part-*.parquet")) == parts and sorted(run.rglob("scalars-*.parquet")) == objects
    seqs = [s for obj in objects for s in pq.read_table(obj, partitioning=None).column("seq").to_pylist()]
    assert seqs == [0, 1, 2, 3]
    entry = _entry(run)
    assert entry.get("scalars_not_newer") == 4, entry


@needs_server
def test_hugging_face_s_eval_on_start_baseline_reaches_the_store_as_a_scalar_only_record(tmp_path, monkeypatch):
    """No download: a tiny model and a real `Trainer` on CPU, its callback streaming to a server of this tree."""
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    import sys

    import torch
    import transformers
    from torch import nn

    from tensorquill.huggingface import TensorQuillCallback

    for name in ("wandb", "mlflow", "mlflow.tracking", "mlflow.tracking.fluent", "wandb.sdk", "wandb.sdk.wandb_run"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.delenv("TENSORQUILL_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

        def forward(self, x, labels=None):
            return {"loss": (self.net(x).squeeze(-1) - labels).pow(2).mean()}

    def data(n, seed):
        torch.manual_seed(seed)
        return [{"x": torch.randn(4), "labels": torch.randn(())} for _ in range(n)]

    store = tmp_path / "store"
    server = _Server(store)
    try:
        callback = TensorQuillCallback(run_id="hf", every=4, endpoint=server.endpoint)
        args = transformers.TrainingArguments(
            output_dir=str(tmp_path / "out"), max_steps=4, per_device_train_batch_size=8, per_device_eval_batch_size=16,
            report_to="none", save_strategy="no", disable_tqdm=True, use_cpu=True, learning_rate=0.01,
            logging_steps=2, eval_strategy="steps", eval_steps=2, eval_on_start=True)
        trainer = transformers.Trainer(model=Model(), args=args, train_dataset=data(64, 0), eval_dataset=data(32, 1),
                                       callbacks=[callback])
        trainer.train()
        import time

        time.sleep(1.5)
    finally:
        server.close()
    run = store / "runs" / "hf"
    read = evidence.load_records(run, scalars=True, all_attempts=True)
    training, scalar = evidence.set_apart_scalars(read)
    assert [r["step"] for r in training] == [0, 1, 2, 3]
    [baseline] = scalar
    assert (baseline["step"], baseline["seq"]) == (0, 0)
    first = next(e for e in trainer.state.log_history if "eval_loss" in e and e["step"] == 0)
    assert baseline["scalars"]["eval/loss"] == pytest.approx(first["eval_loss"])
    assert evidence.scalar_series(run)["eval/loss"][0] == [0, pytest.approx(first["eval_loss"])]


def test_an_older_sdk_sending_a_newer_capture_s_file_loses_its_step_values_on_the_server():
    """The owner, on the cross-version run's finding (Nov 25): a protocol 4 sender given a `.jsonl` that a
    protocol 5 capture wrote sends each scalar-only record as a step row; the server drops and counts it, stores
    every training step, and the values stay in the local file. Beside the passage on an older side.

    The passage names no way to get the values to the server afterwards. It said to send the file with an SDK of
    the capture's version; the cross-version run measured that (`WAVE_AL_CONTRACT.md` section 5, row 9 c): the
    older sender was acknowledged to the end of the file, so `send_file` finds nothing owed and sends no row."""
    older = section("**An older server or an older SDK.**", "**A downgrade is not supported.**")
    for words in ("A file that a protocol 5 capture wrote, sent by such a sender, loses its `step=` values on the server",
                  "row version 2 in a batch of metadata version 1",
                  "drops and counts under `unsupported_schema_version` in the rank's entry of `watchdog.json` "
                  "(`dropped_rows`, `dropped_row_keys`), with a line on its stderr",
                  "every training step of the file is stored",
                  "The values stay in the local `.jsonl`, which a protocol 5 reader reads."):
        assert words in older, words
    assert "to get them to the server" not in older and "SDK of the capture's version" not in older
    table = section("**Protocol 5's versions.**", "**An older server or an older SDK.**")
    assert "the server stores 0, 1 and 2 (0 and 1 only in a batch of metadata version 1)" in table
    ingest = server_source("ingest.rs")
    assert 'UNSUPPORTED_SCHEMA_VERSION: &str = "unsupported_schema_version"' in ingest
    assert "dropped rows that cannot be stored" in ingest
