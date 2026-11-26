"""Wave AO's seam (lane AO-S): what no lane of wave AO could test alone, on the joined tree.

1. BUG-20's callers: the MCP series, `first_divergence` and `compare_runs` take the attempt order of
   the whole record stream, as `scalar_series` does, also where a rollback leaves out every training
   record of the older attempt (so the records left show the newer attempt first).
2. The two overflow guards no lane made (lane AO-9's list): an integer too large for a float is
   unreadable to the Python detectors, as the Rust detectors read it (`detect/extra_json.rs`), and
   not finite to `first_divergence`; nothing raises.
3. Checks across lanes, through a real server of this tree: (a) BUG-07 end to end at batch sizes 1
   and 64, the default series equal through `scalar_series` and MCP `get_series`; (b) BUG-20 through
   the store, all four readers giving the newer attempt's point; (c) a version 2 line with no
   `"record": "scalars"` beside good lines, through the sender, the store and the local readers;
   (d) a session that logs a value it cannot read, streamed; (e) lane AO-1's `run_id` rule against
   what the Lightning and Hugging Face callbacks pass (what happens today, pinned; nothing changed).
4. The pages: each sentence the seam wrote or removed, read as the page has it, beside the code it
   describes (`sdk/README.md`, `sdk/RUN_ON_YOUR_JOB.md`, the bake-off's README, `detect.rs`'s module
   note and `WAVE_AL_CONTRACT.md`).

Records are written here by hand, in the shapes `JsonlSink` writes. Tests that need pyarrow or a
`tensorquill-server` binary (`TENSORQUILL_SERVER_BIN`) skip one by one without them.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import signal
import subprocess
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorquill import evidence, server_detect

BINARY = server_detect.find_binary()
needs_pyarrow = pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow is not installed")
needs_binary = pytest.mark.skipif(BINARY is None or not os.path.isfile(BINARY),
                                  reason="tensorquill-server is not built (TENSORQUILL_SERVER_BIN)")


def _write(path: Path, records: list[dict], tail: str = "") -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + tail)
    return path


def _step(attempt: int, step: int, run_id: str = "r", **more) -> dict:
    return {"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": attempt, "step": step, **more}


def _point(attempt: int, step: int, seq: int, run_id: str = "r", **scalars) -> dict:
    return {"schema_version": 2, "record": "scalars", "run_id": run_id, "rank": 0, "attempt": attempt,
            "step": step, "seq": seq, "scalars": scalars}


# --- 1. BUG-20's callers -------------------------------------------------------------------------------
#
# Lane AO-5's shapes. LATE: attempt 100 trains steps 0 to 6, attempt 200 resumes at step 5 and logs
# `val` at step 2, then attempt 100 flushes its own late `val` at step 2; both points survive the
# rollback (2 < 5), and 200, which first appears later in the stream, stands. ROLLED_BACK_WHOLE:
# attempt 100 trained only steps 10 to 12, all rolled back by attempt 200's resume at step 5, so once
# resolved the records left begin with attempt 200's: only the stream read before the rollback says
# 100 is the older attempt.

LATE = [*[_step(100, s, scalars={"loss": 10.0}) for s in range(7)],
        _point(200, 2, 0, val=2.0),
        _step(200, 5, scalars={"loss": 1.0}),
        _point(100, 2, 0, val=999.0)]

ROLLED_BACK_WHOLE = [*[_step(100, s) for s in (10, 11, 12)], _point(200, 2, 0, val=2.0), _step(200, 5),
                     _point(100, 2, 0, val=999.0)]


@pytest.mark.parametrize("records", [LATE, ROLLED_BACK_WHOLE], ids=["late", "rolled_back_whole"])
def test_bug20_the_mcp_series_gives_the_newer_attempts_point(tmp_path, records):
    from tensorquill.mcp_server import Source, get_series

    path = _write(tmp_path / "r.rank0.jsonl", records)
    assert evidence.scalar_series(path)["val"] == [[2, 2.0]]
    assert get_series(Source(path), {"kind": "scalar", "name": "val"})["values"] == {"value": [2.0]}


def _compared_run(run_id: str, late: bool, whole: bool) -> list[dict]:
    """A run long enough to compare. Attempt 20 logs `val` at step 31 and trains steps 35 to 99;
    attempt 10 trained steps 0 to 29 (`whole=False`, kept below 35) or 40 to 45 (`whole=True`,
    rolled back whole), and with `late` its own late `val` at 31 follows, far off. The newer
    attempt's point stands, so a run with the late point trains as one without it."""
    trained = range(40, 46) if whole else range(30)
    rows = [_step(10, s, run_id, scalars={"loss": 1.0, "val": 2.0}) for s in trained]
    rows.append(_point(20, 31, 0, run_id, val=2.0))
    rows += [_step(20, s, run_id, scalars={"loss": 1.0, "val": 2.0}) for s in range(35, 100)]
    if late:
        rows.append(_point(10, 31, 0, run_id, val=999.0))
    return rows


@pytest.mark.parametrize("whole", [False, True], ids=["late", "rolled_back_whole"])
@pytest.mark.parametrize("resolved", [False, True], ids=["raw", "loaded"])
def test_bug20_first_divergence_reads_the_newer_attempts_point(whole, resolved):
    """Raw rows (`all_attempts=True`), and for the shape whose older attempt keeps its first records,
    rows `load_records` resolved too. Rows resolved by `load_records` from the shape rolled back
    whole no longer show which attempt is older: the README says to give such rows raw."""
    from tensorquill.divergence import first_divergence

    a, b = _compared_run("a", True, whole), _compared_run("b", False, whole)
    if resolved:
        if whole:
            pytest.skip("rows resolved by load_records lose the older attempt's place (README, first_divergence)")
        a, b = evidence.load_records(a, scalars=True), evidence.load_records(b, scalars=True)
    found = first_divergence(a, b)
    assert found["first_exact_difference"] is None, found["first_exact_difference"]


@pytest.mark.parametrize("whole", [False, True], ids=["late", "rolled_back_whole"])
@pytest.mark.parametrize("torn", [False, True], ids=["whole_file", "torn_last_line"])
def test_bug20_compare_runs_on_one_file_a_rank_reads_the_newer_attempts_point(tmp_path, whole, torn):
    from tensorquill.compare import compare_runs

    tail = '{"schema_version": 1, "run_id": "a", "rank": 0, "attempt": 20, "step": 10' if torn else ""
    a = _write(tmp_path / "a.rank0.jsonl", _compared_run("a", True, whole), tail)
    b = _write(tmp_path / "b.rank0.jsonl", _compared_run("b", False, whole))
    part = compare_runs(a, b)["divergence"]
    assert part["first_exact_difference"] is None, part["lines"]
    assert part["state"] == "no_divergence", part["lines"]


# --- 2. An integer too large for a float ----------------------------------------------------------------

BIG = 10**400


def _plain_run(at: int = 20, steps: int = 40, **big) -> list[dict]:
    """A healthy run whose record at `at` holds BIG in the places `big` names."""
    rows = []
    for step in range(steps):
        row = {"schema_version": 1, "run_id": "r", "rank": 0, "step": step, "grad_norm": 1.0,
               "layers": {"blk": {"mean": 0.0, "std": 1.0, "abs_max": 1.0, "zero_frac": 0.0}},
               "grad_norms": {"w": 1.0}}
        if step == at:
            if "grad_norm" in big:
                row["grad_norm"] = BIG
            if "clip_grad_norm" in big:
                row["clip_grad_norm"] = BIG
            if "layer" in big:
                row["layers"]["blk"]["abs_max"] = BIG
            if "parameter" in big:
                row["grad_norms"]["w"] = BIG
        rows.append(row)
    return rows


PLACES = ["grad_norm", "clip_grad_norm", "layer", "parameter"]


def _judged(evidence_: dict) -> tuple:
    return (evidence_["run_state"], sorted(i["kind"] for i in evidence_["incidents"]))


@pytest.mark.parametrize("place", PLACES)
def test_the_python_detectors_read_an_integer_too_large_for_a_float_as_unreadable(place):
    """`evidence._finite` gives no value and `_is_non_finite` says no: a finite number that no float
    holds is neither a measurement nor a NaN or an infinity, so no `non_finite` fires."""
    found = evidence.build_evidence(_plain_run(**{place: True}), detectors="python")
    assert _judged(found) == _judged(evidence.build_evidence(_plain_run(), detectors="python"))
    assert evidence._finite(BIG) is None and evidence._finite(-BIG) is None
    assert evidence._is_non_finite(BIG) is False and evidence._is_non_finite(-BIG) is False
    assert evidence._is_non_finite(math.inf) is True and evidence._finite(2**1023) == float(2**1023)


@needs_pyarrow
@needs_binary
@pytest.mark.parametrize("place", PLACES)
def test_both_engines_judge_a_record_holding_an_integer_too_large_for_a_float_alike(place):
    """The Rust detectors read such a number as unreadable where a float is wanted
    (`detect/extra_json.rs`, wave AJ); the Python detectors now read it so too."""
    records = _plain_run(**{place: True})
    python, rust = (evidence.build_evidence(records, detectors=engine) for engine in ("python", "rust"))
    assert _judged(python) == _judged(rust)
    assert "non_finite" not in _judged(rust)[1]


def _loss_run(run_id: str, third) -> list[dict]:
    return [{"schema_version": 1, "run_id": run_id, "rank": 0, "step": step,
             "scalars": {"loss": third if step == 2 else 1.0 + 0.01 * step}} for step in range(60)]


@pytest.mark.parametrize("sign", [1, -1])
def test_first_divergence_reads_an_integer_too_large_for_a_float_as_an_infinity(sign):
    """`divergence._number` keeps the point, as the infinity of its sign: the runs then compare as
    they do with the infinity recorded, and the difference at step 2 is not lost."""
    from tensorquill.divergence import _number, first_divergence

    assert _number(sign * BIG) == sign * math.inf
    big = first_divergence(_loss_run("a", sign * BIG), _loss_run("b", 3.0))
    infinite = first_divergence(_loss_run("a", sign * math.inf), _loss_run("b", 3.0))
    assert big == infinite
    assert big["first_exact_difference"] == {"step": 2, "series": 1}


# --- 3. Checks across lanes, through a real server ---------------------------------------------------

class _Server:
    """A server of this tree on a free port, stopped with SIGTERM by `close`, which returns its stderr."""

    def __init__(self, store: Path) -> None:
        env = {k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"}
        self.process = subprocess.Popen(
            [BINARY, "--listen", "127.0.0.1:0", "--store", str(store), "--flush-steps", "1",
             "--flush-seconds", "0.2", "--watchdog-seconds", "0.5"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        line = self.process.stdout.readline()
        assert "listening on" in line, line + self.process.stderr.read()
        self.endpoint = f"grpc://127.0.0.1:{int(line.rsplit(':', 1)[1])}"

    def close(self) -> str:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
        self.process.wait(30)
        said = self.process.stderr.read()
        self.process.stdout.close()
        self.process.stderr.close()
        return said


def _sent(tmp_path: Path, path: Path, run_id: str, batch_steps: int, attempt: int | None = None) -> Path:
    """`path` sent by one `FlightSender` cut at `batch_steps`, claiming `attempt`; the run's store folder."""
    from tensorquill import flight

    store = tmp_path / f"store-{run_id}"
    server = _Server(store)
    try:
        sender = flight.FlightSender(path, server.endpoint, run_id, 0, batch_steps=batch_steps,
                                     await_attempt=attempt is not None)
        if attempt is not None:
            sender.attempt = attempt
        assert sender.close(timeout=30.0) is True
        assert sender.acked_offset == path.stat().st_size
    finally:
        server.close()
    return store / "runs" / run_id


def _mcp_points(folder: Path, name: str) -> list[list]:
    from tensorquill.mcp_server import Source, get_series

    answer = get_series(Source(folder), {"kind": "scalar", "name": name})
    return [[step, value] for step, value in zip(answer["steps"], answer["values"]["value"])]


# (a) BUG-07: the audit's three sessions under one run ID on rank 0 (lane AO-3's scenario), with a
# training loss on B's and C's steps so the default series has points: A logs `eval_loss` at an
# explicit step 50 and closes, B trains step 0, C trains step 100 and is the sender's attempt.
A, B, C = 1_732_000_000_000, 1_732_000_100_000, 1_732_000_200_000


def _tq01(run_id: str) -> list[dict]:
    return [
        {"schema_version": 2, "record": "scalars", "run_id": run_id, "rank": 0, "attempt": A, "step": 50,
         "seq": 0, "ts": 10.0, "scalars": {"eval_loss": 99.0}},
        {"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": B, "step": 0, "ts": 20.0,
         "grad_norm": 1.0, "scalars": {"loss": 1.0}},
        {"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": C, "step": 100, "ts": 30.0,
         "grad_norm": 1.0, "scalars": {"loss": 2.0}},
    ]


@needs_pyarrow
@needs_binary
@pytest.mark.parametrize("batch_steps", [1, 64])
def test_bug07_the_default_series_from_the_store_is_the_files_through_both_readers(tmp_path, batch_steps):
    from tensorquill.mcp_server import Source, ToolError, get_series

    run_id = f"seam_a_{batch_steps}"
    path = _write(tmp_path / f"{run_id}.rank0.jsonl", _tq01(run_id))
    folder = _sent(tmp_path, path, run_id, batch_steps, attempt=C)
    order = json.loads((folder / "rank=0" / "attempts.json").read_text())["attempts"]
    assert [entry["attempt"] for entry in order] == [A, B, C]
    expected = {"loss": [[0, 1.0], [100, 2.0]]}  # B's first step (0) rolls A's point at 50 back
    assert evidence.scalar_series(path) == evidence.scalar_series(folder, rank=0) == expected
    assert _mcp_points(folder, "loss") == expected["loss"]
    with pytest.raises(ToolError, match="no scalar is named 'eval_loss'"):
        get_series(Source(folder), {"kind": "scalar", "name": "eval_loss"})
    assert evidence.scalar_series(folder, rank=0, all_attempts=True)["eval_loss"] == [[50, 99.0]]


# (b) BUG-20 through the store: the four readers on a store the joined sender and server wrote.

@needs_pyarrow
@needs_binary
@pytest.mark.parametrize("records", [LATE, ROLLED_BACK_WHOLE], ids=["late", "rolled_back_whole"])
@pytest.mark.parametrize("batch_steps", [1, 64])
def test_bug20_the_series_of_a_store_gives_the_newer_attempts_point(tmp_path, records, batch_steps):
    path = _write(tmp_path / "r.rank0.jsonl", records)
    folder = _sent(tmp_path, path, "r", batch_steps)
    assert evidence.scalar_series(folder, rank=0)["val"] == [[2, 2.0]]
    assert _mcp_points(folder, "val") == [[2, 2.0]]


@needs_pyarrow
@needs_binary
@pytest.mark.parametrize("whole", [False, True], ids=["late", "rolled_back_whole"])
def test_bug20_first_divergence_and_compare_runs_on_a_store_read_the_newer_attempts_point(tmp_path, whole):
    from tensorquill.compare import compare_runs
    from tensorquill.divergence import first_divergence

    store = tmp_path / "store"
    server = _Server(store)
    try:
        from tensorquill import flight

        for run_id, late in (("a", True), ("b", False)):
            path = _write(tmp_path / f"{run_id}.rank0.jsonl", _compared_run(run_id, late, whole))
            assert flight.send_file(path, server.endpoint, run_id, 0, timeout=60) is True
        time.sleep(0.5)
    finally:
        server.close()
    raw = [evidence.load_records(store / "runs" / run_id, rank=0, all_attempts=True, scalars=True) for run_id in "ab"]
    assert first_divergence(*raw)["first_exact_difference"] is None
    part = compare_runs("a", "b", source=store)["divergence"]
    assert part["first_exact_difference"] is None and part["state"] == "no_divergence", part["lines"]


# (c) BUG-02, both sides joined: the sender leaves the line out (lane AO-2), so the store reads back;
# the local file is refused by the readers that check every line, and listed by `list_runs`.

def _unmarked(run_id: str) -> list[dict]:
    return [{"schema_version": 2 if step == 2 else 1, "run_id": run_id, "rank": 0, "attempt": 1, "step": step,
             "ts": float(step), "grad_norm": 1.0} for step in range(5)]


@needs_pyarrow
@needs_binary
def test_bug02_a_version_2_line_without_the_mark_is_left_out_and_the_store_reads_back(tmp_path):
    from tensorquill import flight, identity, runs

    run_id = "v2u"
    local = tmp_path / "local"
    local.mkdir()
    path = _write(local / f"{run_id}.rank0.jsonl", _unmarked(run_id))
    store = tmp_path / "store"
    server = _Server(store)
    try:
        assert flight.send_file(path, server.endpoint, run_id, 0, timeout=60) is False
        sender = flight.FlightSender(path, server.endpoint, run_id, 0)  # a later sender of the same file
        assert sender.close(30.0) is False and sender.left_out_lines == 1
        time.sleep(1.0)  # a watchdog pass writes the rank's status
    finally:
        said = server.close()
    assert [r["step"] for r in evidence.load_records(store / "runs" / run_id, all_attempts=True)] == [0, 1, 3, 4]
    status = json.loads((store / "runs" / run_id / "watchdog.json").read_text())
    [entry] = status["ranks"] if isinstance(status["ranks"], list) else status["ranks"].values()
    assert "dropped_rows" not in entry and "1 with a null seq" not in said  # nothing of it reached the server
    # The local file: the readers that check every line refuse it, naming the line (wave AF's rule,
    # unchanged by wave AO); the listing, which reads no line's identity, lists the run.
    for read in (evidence.build_evidence, evidence.load_records):
        with pytest.raises(identity.MalformedRecordError, match="line 3: schema version 2 is the scalar-only record"):
            read(path)
    [listed] = runs.list_runs(local)
    assert (listed["run_id"], listed["last_step"], listed["ranks"], listed["evidence"]["state"]) == (run_id, 4, [0], "missing")


# (d) A session that logs a value it cannot read (lane AO-1, BUG-03), streamed by the joined sender.

@needs_pyarrow
@needs_binary
def test_a_streamed_session_refuses_a_meta_value_and_closes_over_a_value_whose_move_fails(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import tensorquill

    class _Unreadable(torch.Tensor):
        """A tensor whose copy off the device fails (lane AO-1's stand-in)."""

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if func is torch.stack:
                raise RuntimeError("stand-in: the copy off the device failed")
            return super().__torch_function__(func, types, args, kwargs or {})

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_TOKEN", raising=False)
    store = tmp_path / "store"
    server = _Server(store)
    try:
        model = torch.nn.Linear(4, 1)
        session = tensorquill.init(model, run_id="seam_d", every=2, endpoint=server.endpoint, heartbeat_seconds=None)
        for step in (0, None):
            with pytest.raises(ValueError, match="x: the tensor is on meta, which holds no values"):
                session.log({"x": torch.tensor(1.0, device="meta")}, step=step)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(4):
            optimizer.zero_grad()
            model(torch.randn(3, 4)).sum().backward()
            session.log({"train/loss": 1.0})
            optimizer.step()
        session.log({"eval/loss": 0.5, "eval/bad": 1.0}, step=3)
        session._scalar_records[-1]["scalars"]["eval/bad"] = torch.tensor(2.0).as_subclass(_Unreadable)
        with warnings.catch_warnings(record=True) as said:
            warnings.simplefilter("always")
            session.close()  # never raises
        sender = session._sink.sender
        time.sleep(0.5)
    finally:
        server.close()
    out = tmp_path / "tensorquill-out"
    path = out / "seam_d.rank0.jsonl"
    assert (out / "seam_d.rank0.end.json").exists()
    assert sum("left out the value 'eval/bad'" in str(w.message) for w in said) == 1
    assert (sender.acked_offset, sender.left_out_lines, sender.skipped_lines) == (path.stat().st_size, 0, 0)
    stored = evidence.load_records(store / "runs" / "seam_d", scalars=True, all_attempts=True)
    assert [(r["step"], r.get("scalars")) for r in stored] == [*[(s, {"train/loss": 1.0}) for s in range(4)],
                                                               (3, {"eval/loss": 0.5})]


# (e) Lane AO-1's `run_id` rule (BUG-06) against the callbacks: an explicit `run_id` that breaks it is
# refused at `init`, one from the environment is fitted with a warning. The Lightning callback passes
# its logger's run ID to `init` as an explicit one, so such an ID would stop the fit; MLflow's IDs
# (32 hex digits) and W&B's (letters and digits) never break it. Pinned as it is today, for the owner.

class _MLflowLoggerStandIn:
    run_id = "exp/1"  # no real MLflow run ID has a '/': a stand-in for one that breaks the rule


def _lightning_trainer(loggers: list) -> SimpleNamespace:
    return SimpleNamespace(loggers=loggers, is_global_zero=True, precision_plugin=SimpleNamespace(), global_step=0)


def test_today_a_lightning_loggers_run_id_that_breaks_the_rule_stops_the_fit_at_its_start(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("lightning")
    from tensorquill import manifest
    from tensorquill.lightning import TensorQuillCallback

    monkeypatch.chdir(tmp_path)
    callback = TensorQuillCallback(heartbeat_seconds=None)
    with pytest.raises(ValueError, match="run_id 'exp/1' cannot be used: it holds '/'"):
        callback.on_fit_start(_lightning_trainer([_MLflowLoggerStandIn()]), torch.nn.Linear(2, 2))
    assert callback.session is None and not list(tmp_path.iterdir())  # nothing opened, nothing written
    # The same ID from the environment is fitted, with a warning (lane AO-1's rule).
    monkeypatch.setenv("MLFLOW_RUN_ID", "exp/1")
    callback = TensorQuillCallback(heartbeat_seconds=None)
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        callback.on_fit_start(_lightning_trainer([]), torch.nn.Linear(2, 2))
    callback.session.close()
    assert any("MLFLOW_RUN_ID gives the run ID 'exp/1'" in str(w.message) for w in said)
    assert (tmp_path / "tensorquill-out" / "exp-1.rank0.jsonl").exists()
    for real in ("0123456789abcdef0123456789abcdef", "3k9x2m1q"):  # an MLflow and a W&B run ID
        assert manifest.run_id_problem(real) is None


def test_a_run_id_given_to_the_hugging_face_callback_is_the_users_and_refused_when_it_breaks_the_rule(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tensorquill.huggingface import TensorQuillCallback

    monkeypatch.chdir(tmp_path)
    callback = TensorQuillCallback(run_id="exp/1", heartbeat_seconds=None)
    args = SimpleNamespace(eval_on_start=False, resume_from_checkpoint=None, output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="run_id 'exp/1' cannot be used: it holds '/'"):
        callback.on_train_begin(args, SimpleNamespace(global_step=0), SimpleNamespace(), model=torch.nn.Linear(2, 2))
    assert callback.session is None and not list(tmp_path.iterdir())


# --- 4. The pages ------------------------------------------------------------------------------------

SDK = Path(__file__).resolve().parents[1]
ROOT = SDK.parent


def _flat(text: str) -> str:
    return " ".join(text.split())


def _page(relative: str) -> str:
    path = ROOT / relative
    if not path.exists():
        pytest.skip(f"no {relative} in this checkout")
    return path.read_text(encoding="utf-8")


def _section(text: str, start: str, end: str) -> str:
    return _flat(text[text.index(start):text.index(end)])


def _readme(start: str, end: str) -> str:
    return _section(_page("sdk/README.md"), start, end)


WHAT_IS_BUILT = ("## What is built\n", "\n## Support: what is captured")
STREAMING = ("## Streaming to an ingest server\n", "\n## Detector benchmark on 64 ranks")
DETECTORS = ("## Detectors in the server\n", "\n## Compact evidence and the trigger")
DATA_WAIT = ("## GPU memory, NVML and data wait\n", "\n## Projects, run names and comparing runs")
COMPARING = ("## Projects, run names and comparing runs\n", "\n## Scalars mirrored from W&B and MLflow")
MIRROR = ("## Scalars mirrored from W&B and MLflow\n", "\n## Scalar-only records")
MCP = ("## For a coding agent: a read-only MCP server\n", "\n## A score, four measurements")
PLACED = ("## Records that cannot be placed, every rank's keys", "\n## Things to know")


def test_the_scalars_bullet_says_what_log_refuses_and_what_is_left_out():
    text = _flat(next(line for line in _page("sdk/README.md").splitlines() if line.startswith("- **`scalars`**")))
    for words in ("**A value it cannot read (Nov 26, 2026).** `log` refuses a tensor on `meta` (it holds no values), a "
                  "nested tensor and one that is not dense, such as a sparse one, with `ValueError` naming the key, "
                  "before anything is recorded, with or without `step=`.",
                  "A scalar-only record's tensor whose copy off the device fails when the record is written is left out "
                  "of the record, with one warning per session naming the first; the record's other values are kept, a "
                  "record left with no value is not written, and capture goes on: `flush()` and `close()` do not raise "
                  "from it, and `close()` writes the end marker."):
        assert words in text, words


def test_log_refuses_those_values_and_one_warning_names_the_first_left_out(tmp_path):
    """The two sentences above, run: the refusals' words, and two values whose copy fails, in two records."""
    torch = pytest.importorskip("torch")
    import tensorquill
    from tensorquill.sink import JsonlSink

    class _Unreadable(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if func is torch.stack:
                raise RuntimeError("stand-in: the copy off the device failed")
            return super().__torch_function__(func, types, args, kwargs or {})

    path = tmp_path / "r.rank0.jsonl"
    session = tensorquill.init(torch.nn.Linear(2, 2), run_id="r", sink=JsonlSink(path), heartbeat_seconds=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # PyTorch's own warning that nested tensors are a prototype
        nested = torch.nested.nested_tensor([torch.ones(1), torch.ones(2)])
    for value, said in ((torch.tensor(1.0, device="meta"), "x: the tensor is on meta, which holds no values"),
                        (nested, "x: the tensor is a nested tensor"),
                        (torch.ones(1).to_sparse(), "x: the tensor is a sparse_coo tensor, not a dense one")):
        for step in (None, 3):
            with pytest.raises(ValueError, match=said):
                session.log({"x": value}, step=step)
    session.log({"a/kept": 0.5, "a/bad": 1.0}, step=1)
    session._scalar_records[-1]["scalars"]["a/bad"] = torch.tensor(2.0).as_subclass(_Unreadable)
    session.log({"b/bad": 1.0}, step=2)
    session._scalar_records[-1]["scalars"]["b/bad"] = torch.tensor(2.0).as_subclass(_Unreadable)
    with warnings.catch_warnings(record=True) as told:
        warnings.simplefilter("always")
        session.close()
    left_out = [str(w.message) for w in told if "left out the value" in str(w.message)]
    assert len(left_out) == 1 and "'a/bad'" in left_out[0]
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["step"], r["scalars"]) for r in lines if r.get("record") == "scalars"] == [(1, {"a/kept": 0.5})]
    assert path.with_name("r.rank0.end.json").exists()


def test_the_run_id_row_says_what_is_refused_and_what_is_fitted():
    from tensorquill import manifest

    text = _readme(*WHAT_IS_BUILT)
    assert ("A `run_id=` that could name a file outside the output folder, one that holds `/`, `\\`, `:` or a control "
            "character or is `.` or `..`, is refused at `init` with `ValueError` before anything is opened (Nov 26, "
            "2026); one from the environment, a tracker or a launcher is made to fit instead, each such character, or "
            "the whole `.` or `..`, becoming `-`, with a warning. The Lightning callback passes its logger's run ID as "
            "`run_id=`, so there such an ID would be refused; MLflow's and W&B's run IDs hold none of these "
            "characters.") in text
    for refused, fitted in (("a/b", "a-b"), ("a\\b", "a-b"), ("C:x", "C-x"), ("a\nb", "a-b"), ("a\x00b", "a-b"),
                            (".", "-"), ("..", "-")):
        assert manifest.run_id_problem(refused) and manifest.conform_run_id(refused) == fitted
    for taken in ("a..b", "exp 1", "résumé_7"):
        assert manifest.run_id_problem(taken) is None


def test_the_streaming_section_says_what_the_sender_leaves_out_and_returns():
    text = _readme(*STREAMING)
    for words in ("So is a row of a version 2 record whose `seq` is null in a batch of protocol 5 (a version 2 record "
                  "that is not a whole scalar-only record, which this SDK's sender does not send), counted under `seq` "
                  "(Nov 26, 2026).",
                  "`tensorquill.flight.send_file(path, endpoint, run_id, rank)` sends the rest later, and returns whether "
                  "the server confirmed the whole file and holds every line of the run in it: False while a line of the "
                  "run was left out.",
                  "A scalar-only line whose `seq` is not an integer of 0 or more is skipped and counted like a line whose "
                  "key cannot be stored, never sent as a step, and so is a line of schema version 2 with no "
                  '`"record": "scalars"` (Nov 26, 2026).',
                  "In `scalars`, an integer a float64 holds only rounded (past 2**53) is in the column as the nearest "
                  "float, which the Rust detectors read, and travels exact in `extra_json` too; a Boolean, which the "
                  "column leaves out, travels in `extra_json`. Both come back from the store as the file has them, a "
                  "Boolean after the column's entries (Nov 26, 2026)."):
        assert words in text, words


def test_the_placed_section_says_what_the_sender_leaves_out_and_what_init_refuses():
    text = _readme(*PLACED)
    for words in ("- The capture writes a `session.step(step=...)` below 0 as given; such a file is then refused, so give "
                  "steps of 0 or more. `init` refuses a `start_step` that is not an integer of 0 or more and an `every` "
                  "that is not an integer of 1 or more, before anything is written: `TypeError` for a bool, a float or "
                  "another type, `ValueError` for one too small (Nov 26, 2026).",
                  "nor one whose key the stream cannot hold, nor a line of `schema_version` 2 that is not a whole "
                  'scalar-only record (no `"record": "scalars"`).',
                  "`close()` and `send_file` then return False, and `close()` says so in one line on stderr: "
                  "`sender.left_out_lines` counts such lines of the run in the part of the file the server confirmed, "
                  "and the `.offset` file keeps the count, so sending the file again says False too. A line that is not "
                  "JSON, or is another run's, is counted in `skipped_lines` alone and changes neither (Nov 26, 2026)."):
        assert words in text, words
    assert "writes a `step=` or `start_step=` below 0 as given" not in text


def test_init_refuses_those_steps_as_the_page_says(tmp_path):
    torch = pytest.importorskip("torch")
    import tensorquill
    from tensorquill.sink import JsonlSink

    for given, raised in (({"start_step": -1}, ValueError), ({"start_step": True}, TypeError),
                          ({"start_step": 1.0}, TypeError), ({"every": 0}, ValueError), ({"every": True}, TypeError),
                          ({"every": 2.0}, TypeError)):
        path = tmp_path / "s.rank0.jsonl"
        with pytest.raises(raised):
            tensorquill.init(torch.nn.Linear(2, 2), run_id="s", sink=JsonlSink(path), heartbeat_seconds=None, **given)
        assert path.read_text() == ""


def test_no_difference_of_the_engines_on_moe_records_is_left_on_the_page():
    """Both bullets of "Two more, in hand-made records only" no longer hold: the `moe_top_k` one since
    wave AO (BUG-13, lane AO-5), the diversity one since wave AL (`fd31084`, section 1.5 of the
    contract). The run below shows the second."""
    text = _readme(*DETECTORS)
    for gone in ("Two more, in hand-made records only", "One more, in hand-made records only",
                 "Python applies the last `moe_top_k` a run recorded to every step",
                 "A recorded `moe_stats` diversity that is not a finite number is used as it is by Python"):
        assert gone not in text, gone


@needs_pyarrow
@needs_binary
@pytest.mark.parametrize("recorded", ["nan", "inf", None, True, 10**400])
def test_both_engines_take_a_diversity_that_is_not_a_finite_number_from_the_counts(recorded):
    rows = [{"schema_version": 1, "run_id": "r", "rank": 0, "step": step, "moe": {"router": [37, 1, 1, 1]},
             "moe_top_k": {"router": 1}, "moe_stats": {"router": {"diversity": recorded if step == 10 else 0.9}}}
            for step in range(30)]
    python, rust = ([s["diversity"] for s in evidence.build_evidence(rows, detectors=engine)["samples"] if s["step"] == 10]
                    for engine in ("python", "rust"))
    assert python == rust and python[0] != 0.9 and math.isfinite(python[0])


def test_the_data_wait_bullet_says_a_nested_fetch_is_counted_once():
    assert ("A fetch made inside another fetch on the same thread (a dataset or collate function that pulls from a "
            "second `DataLoader` or a `time_data` iterator) is part of the outer wait and is counted once (Nov 26, "
            "2026).") in _readme(*DATA_WAIT)


def test_the_first_divergence_bullet_says_how_attempts_are_resolved_and_which_point_stands():
    text = _readme(*COMPARING)
    assert ("Attempts are resolved where their training step ranges overlap or touch, or where resolving them leaves "
            "out only scalar-only records (Nov 26, 2026), so that rows `compare` has already resolved are read as they "
            "are; raw rows whose first attempt trained wholly above a later one, with no step in common, are left as "
            "given. Of two attempts' scalar-only points at one step the later attempt's stands, by the order the "
            "attempts first appear in the rows given: where a resumed job rolled back every training step of an earlier "
            "attempt, give the rows raw, since rows `load_records` resolved no longer show that attempt first.") in text
    assert "Attempts are resolved only where their step ranges overlap or touch" not in text


def test_rows_load_records_resolved_lose_the_place_of_an_attempt_rolled_back_whole():
    """The limit the page states, run: the shape rolled back whole, as `load_records` resolves it."""
    from tensorquill.divergence import first_divergence

    a, b = (evidence.load_records(_compared_run(run_id, late, True), scalars=True) for run_id, late in (("a", True), ("b", False)))
    assert first_divergence(a, b)["first_exact_difference"] == {"step": 31, "series": 1}


def test_the_mirror_section_states_no_limit_on_a_small_integer():
    text = _readme(*MIRROR)
    assert "One limit: if the mirror stored a key first and you then `session.log` the very same small integer" not in text
    assert "(Python shares the objects for -5 to 256)" not in text


def test_the_hugging_face_callback_keeps_the_pause_for_the_trainers_own_datasets_only():
    """The callback reads the datasets from the `Trainer` it holds (`huggingface._more_datasets_follow`),
    so a dictionary given to `evaluate()` alone is not among them: the page says so."""
    pytest.importorskip("transformers")
    from tensorquill.huggingface import _more_datasets_follow

    trainer = SimpleNamespace(eval_dataset={"a": [0], "b": [1]})
    assert _more_datasets_follow(trainer, {"eval_a_runtime": 0.1}) is True
    assert _more_datasets_follow(trainer, {"eval_b_runtime": 0.1}) is False
    assert _more_datasets_follow(SimpleNamespace(eval_dataset=[0]), {"eval_a_runtime": 0.1}) is False


def test_the_lightning_and_hugging_face_rows_say_the_loss_and_the_baseline_phase():
    text = _readme(*WHAT_IS_BUILT)
    for words in ("the back-propagated loss per optimizer step (under a `GradScaler`, as with `16-mixed`, the loss before "
                  "the scaler multiplied it; since Nov 26, 2026), and the `16-mixed` scaler",
                  "During `eval_on_start`'s evaluation the callback sets the phase `\"baseline\"`, a pause as `\"eval\"` "
                  "is, so its loader time is not step 0's `data_wait`; with several evaluation datasets given to the "
                  "`Trainer` (`eval_dataset={...}`) the phase stays `\"eval\"` until the last one ends, so no dataset's "
                  "loader time is training's; a dictionary passed to `trainer.evaluate()` itself is not covered (Nov 26, "
                  "2026)."):
        assert words in text, words


def test_the_mcp_section_says_what_bounds_an_answer_and_a_value_too_large_for_a_float():
    from tensorquill import mcp_server

    text = _readme(*MCP)
    assert ("No answer is longer than 50,000 bytes, errors included; in a batch, each answer is held to that. A long "
            "series is thinned to 200 points with the stride reported, and the minimum, maximum and median are still "
            "taken over every recorded step; a value too large for a float is shown as recorded and left out of them, "
            "counted as `too_large_for_a_float` with `first_too_large_step` (Nov 26, 2026).") in text
    assert "No reply is longer than 50 KB." not in text
    assert mcp_server.MAX_ANSWER_BYTES == 50_000


def test_the_guide_says_init_refuses_a_negative_start_step():
    text = _flat(_page("sdk/RUN_ON_YOUR_JOB.md"))
    assert ("The capture writes a `session.step(step=...)` below 0 as given, and the file is then refused this way, so "
            "give steps of 0 or more; `init` refuses a `start_step` below 0 before anything is written.") in text
    assert "writes a `step=` or `start_step=` below 0 as given" not in text


def test_the_bake_offs_page_counts_the_verifiers_four_tables():
    text = _flat(_page("experiments/moe-bakeoff/README.md"))
    assert ("The verifier recomputes the loads, changes and row order of the four layer 31 expert load tables from the "
            "counts (the status label is the evidence's flag; the first table, at step 1,750, has no earlier step to "
            "compare with, Nov 26)") in text
    assert "the three layer 31 expert load tables" not in text


def test_the_server_detectors_module_note_says_both_engines_judge_experts_per_token_alike():
    text = _flat(_page("server/src/detect.rs").replace("//!", ""))
    assert ("*Experts per token.* A record is judged with the `moe_top_k` known when it arrives, here and in Python "
            "since wave AO (BUG-13), so a run that changes it in mid-run is judged step by step alike.") in text
    assert "Python applies the last `moe_top_k` recorded in the run" not in text


def test_the_contract_turns_item_5_of_lane_al_4s_list_and_names_the_seq_drop_beside_s15():
    text = _flat(_page("WAVE_AL_CONTRACT.md"))
    assert ("which matters only for an attempt that holds scalar-only records alone. **Goes beyond.** **Accepted as built "
            "by the owner, Nov 24 (row S14).** **Turned by the owner, Nov 26 (wave AO, accepting lane AO-6's BUG-22):** "
            "`compare` on a torn local file reads its whole lines in the file's order, the scalar-only records in their "
            "place (lane AO-6, BUG-22).") in text
    assert ("The offline modes read all three (sections 3, 4.5, 5). Under metadata 2, a version 2 row whose `seq` is "
            "null, which is not a whole scalar-only record, is dropped and counted the same way under the key `seq` "
            "(lane AO-3, BUG-02; accepted by the owner, Nov 26)") in text


# --- 5. The follow-up (Nov 26) ---------------------------------------------------------------------------

def _timed_run(place: str, value) -> list[dict]:
    """A healthy run of 40 steps with `data_wait`, `step_time`, `lr`, `grad_scale` and a throttled
    GPU's NVML reading, whose record at step 20 holds `value` in `place`."""
    rows = []
    for step in range(40):
        row = {"schema_version": 1, "run_id": "r", "rank": 0, "step": step, "grad_norm": 1.0,
               "data_wait": 0.01, "step_time": 0.1, "lr": [0.001], "grad_scale": 1024.0,
               "system": {"nvml": {"throttle": ["sw_thermal"] if step >= 30 else [], "temperature_c": 70,
                                   "sm_clock_mhz": 1500, "power_w": 300.0, "power_limit_w": 400.0}}}
        if step == 20:
            if place == "lr":
                row["lr"] = [value]
            elif place in ("data_wait", "step_time", "grad_scale"):
                row[place] = value
            else:
                row["system"]["nvml"][place] = value
        rows.append(row)
    return rows


TIMED_PLACES = ["lr", "data_wait", "step_time", "grad_scale"]


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("place", TIMED_PLACES)
def test_the_python_detectors_read_such_an_integer_in_lr_data_wait_step_time_and_grad_scale(place, sign):
    found = evidence.build_evidence(_timed_run(place, sign * BIG), detectors="python")
    assert _judged(found) == _judged(evidence.build_evidence(_timed_run(place, 1.0), detectors="python"))


@needs_pyarrow
@needs_binary
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("place", TIMED_PLACES)
def test_both_engines_judge_such_an_integer_in_lr_data_wait_step_time_and_grad_scale_alike(place, sign):
    records = _timed_run(place, sign * BIG)
    python, rust = (evidence.build_evidence(records, detectors=engine) for engine in ("python", "rust"))
    assert _judged(python) == _judged(rust)


@pytest.mark.parametrize("sign", [1, -1])
def test_a_throttled_gpus_reading_too_large_for_a_float_is_no_value(sign):
    """`gpu_throttled` carries the NVML reading at its step: such an integer there is no value."""
    rows = _timed_run("lr", 0.001)
    rows[30]["system"]["nvml"]["temperature_c"] = sign * BIG
    [incident] = [i for i in evidence.build_evidence(rows, detectors="python")["incidents"] if i["kind"] == "gpu_throttled"]
    assert incident["step"] == 30 and incident["temperature_c"] is None and incident["power_w"] == 300.0


@pytest.mark.parametrize("module", ["data_checks", "evidence_system"])
def test_each_readers_number_reads_such_an_integer_as_a_value_it_cannot_read(module):
    import importlib

    number = importlib.import_module(f"tensorquill.{module}")._number
    assert number(BIG) is None and number(-BIG) is None
    assert number(2**1023) == float(2**1023) and number(3) == 3.0 and number(math.inf) is None and number(True) is None


# The sweep: ±10**400 in each of 43 fields of a full record, through every reader of records (both
# engines, `compact`, `incident_summary`, the Rerun export, `scalar_series`, `first_divergence`,
# `compare_runs`, `list_runs`, the MCP tools, `compare_replicas`). What it found beyond the two above.

def _memory_run(place: str, value) -> list[dict]:
    rows = []
    for step in range(12):
        cuda = {"allocated": 100 + step, "reserved": 200, "total": 10_000, "peak_allocated": 150,
                "alloc_retries": 0, "ooms": 0}
        if step == 6:
            cuda[place] = value
        rows.append({"schema_version": 1, "run_id": "r", "rank": 0, "step": step, "grad_norm": 1.0,
                     "system": {"cuda": cuda}})
    return rows


@pytest.mark.parametrize("value", [BIG, -BIG, 2**64], ids=["big", "minus_big", "past_int64"])
@pytest.mark.parametrize("place", ["total", "reserved", "allocated"])
def test_a_memory_count_rust_cannot_hold_is_read_as_no_count(place, value):
    """Rust's `system::count` takes a value only when it is an `i64` (`is_i64`): one outside is no
    count, and the memory rules raise nothing on it."""
    from tensorquill import evidence_system

    assert evidence_system._count(value) is None and evidence_system._count(2**63 - 1) == 2**63 - 1
    found = evidence.build_evidence(_memory_run(place, value), detectors="python")
    assert _judged(found) == _judged(evidence.build_evidence(_memory_run(place, None), detectors="python"))


@needs_pyarrow
@needs_binary
@pytest.mark.parametrize("value", [BIG, -BIG, 2**64], ids=["big", "minus_big", "past_int64"])
@pytest.mark.parametrize("place", ["total", "reserved", "allocated"])
def test_both_engines_judge_a_memory_count_rust_cannot_hold_alike(place, value):
    records = _memory_run(place, value)
    python, rust = (evidence.build_evidence(records, detectors=engine) for engine in ("python", "rust"))
    assert _judged(python) == _judged(rust)


def test_a_step_too_large_for_a_float_gives_no_trend_and_raises_nothing():
    """A step is any integer of 0 or more to the readers; past a float's range no trend can place
    it (`robust_trend`), so there is none, and the rest of the evidence is read."""
    rows = [{"schema_version": 1, "run_id": "r", "rank": 0, "step": step, "grad_norm": 1.0} for step in range(30)]
    rows.append({"schema_version": 1, "run_id": "r", "rank": 0, "step": BIG, "grad_norm": 1.0})
    assert evidence.robust_trend([*range(29), BIG], [1.0] * 30) is None
    found = evidence.build_evidence(rows, detectors="python")
    assert found["run_state"]["state"] == "healthy"


def _replica_records(fp_of_rank_1: list) -> list[dict]:
    case = json.loads((SDK / "tests" / "fixtures" / "replica" / "incident_cases.json").read_text())
    records = case["cases"]["drift_on_one_of_three"]["records"]
    records[1]["replica"]["params"]["w"]["fp"] = fp_of_rank_1
    return records


@pytest.mark.parametrize("at", [0, 1, 2])
@pytest.mark.parametrize("sign", [1, -1])
def test_a_replica_fingerprint_too_large_for_a_float_reads_as_one_not_finite(at, sign):
    """Rust reads such a number in the observation as text (`detect/extra_json.rs`) and a text
    fingerprint as NaN (`replica.rs`'s `number`); Python's `_fingerprint` now gives NaN too, so the
    comparison is the one a fingerprint of "nan" gives."""
    big, nan = [3.6, -1.2, 0.7], [3.6, -1.2, 0.7]
    big[at], nan[at] = sign * BIG, "nan"
    assert evidence._fingerprint(sign * BIG) != evidence._fingerprint(sign * BIG)  # NaN
    assert evidence.compare_replicas(_replica_records(big)) == evidence.compare_replicas(_replica_records(nan))


@needs_pyarrow
@needs_binary
@pytest.mark.parametrize("at", [0, 1, 2])
def test_both_engines_compare_a_replica_fingerprint_too_large_for_a_float_alike(at):
    from tensorquill.server_detect import server_run_report

    fp = [3.6, -1.2, 0.7]
    fp[at] = BIG
    records = _replica_records(fp)
    python, rust = evidence.compare_replicas(records), server_run_report(records, binary=BINARY)["replica"]
    for key in ("steps", "replica_drift", "attribution"):
        assert python.get(key) == rust.get(key), key


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("place", ["grad_norm", "layer"])
def test_the_rerun_export_leaves_out_a_value_too_large_for_a_float(tmp_path, monkeypatch, place, sign):
    """The export logs a finite number only (`_finite`): such an integer is left out, and nothing raises."""
    pytest.importorskip("numpy")
    from unittest import mock
    import sys

    logged: list[str] = []
    rr = mock.MagicMock(name="rerun")
    rr.escape_entity_path_part.side_effect = lambda part: part
    rr.log.side_effect = lambda entity, *_, **__: logged.append(entity)
    monkeypatch.setitem(sys.modules, "rerun", rr)
    monkeypatch.setitem(sys.modules, "rerun.blueprint", mock.MagicMock(name="rerun.blueprint"))
    rows = _plain_run(**{place: True})
    for row in rows:
        if row["step"] == 20 and sign < 0:
            row["grad_norm"] = -BIG if place == "grad_norm" else row["grad_norm"]
            row["layers"]["blk"]["abs_max"] = -BIG if place == "layer" else row["layers"]["blk"]["abs_max"]
    built = evidence.build_evidence(rows, detectors="python")
    evidence.export_rerun(built, tmp_path / "r.rrd", records=evidence.load_records(rows, scalars=True))
    entity = "/training/grad_norm" if place == "grad_norm" else "/layers/blk/abs_max"
    assert logged.count(entity) == 39  # 40 steps, the one holding the integer left out
