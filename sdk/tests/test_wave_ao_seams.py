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
