"""Lane U1: ways the capture session could still raise into, block or change a training run.

Each test fails on `main` at 1767a4a and passes with the lane's change in `capture.py`:
C01 (an unreadable gradient raised out of `optimizer.step()`), C04 (a closed session's step and
flush blocked for good), B8 (`init` traced by `torch.compile`), C10 (a rejected `init` left hooks
and files), C19 (closing a session removed someone else's `GradScaler.update`), the probe's
warning under `-W error`, and `init`'s docstring on the drift check's gather. No server binary.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import warnings
from datetime import timedelta
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim.optimizer import _global_optimizer_post_hooks, _global_optimizer_pre_hooks

import tensorquill
from tensorquill import capture, datawait
from tensorquill.distributed import FINGERPRINT_WIDTH
from tensorquill.flight import record_to_row, row_to_record

DEADLINE = 20.0  # seconds; a blocked call never returns, so a deadline is the only way to see it


# --- C01: a gradient that cannot be read ---------------------------------------------------------


class WithMeta(nn.Module):
    """A linear layer, and one trainable parameter on `meta`, as an unmaterialized layer leaves it."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 2)
        self.extra = nn.Parameter(torch.ones(4, device="meta"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def _train_with_meta(observe: bool, steps: int = 3, every: int = 2) -> tuple[torch.Tensor, list[dict]]:
    torch.manual_seed(0)
    model = WithMeta()
    opt = torch.optim.SGD(model.parameters(), lr=0.5, foreach=False)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=every, weights=1) if observe else None
    try:
        for _ in range(steps):
            opt.zero_grad()
            model(torch.ones(2, 3)).sum().backward()
            model.extra.grad = torch.ones_like(model.extra)
            opt.step()
    finally:
        if session is not None:
            session.close()
    return model.lin.weight.detach().clone(), sink.records


def test_a_gradient_on_meta_is_left_out_on_its_own_and_the_step_runs_as_without_tensorquill():
    without, _ = _train_with_meta(observe=False)
    with_tq, records = _train_with_meta(observe=True)
    assert torch.equal(without, with_tq)
    assert [r["step"] for r in records] == [0, 1, 2]
    for record in records:
        assert set(record["grad_norms"]) == {"lin.weight", "lin.bias"}
        assert record["grad_norm"] == pytest.approx((2 * 3 * 4 + 2 * 4) ** 0.5)  # |2| x 6 and |2| x 2
        assert record["grad_norms"]["lin.bias"] == pytest.approx(8 ** 0.5)
    # Said once, on the first record written after it was found, across blocks and held steps.
    said = [r.get("grads_not_read") for r in records]
    assert said == [{"extra": "its gradient is on meta, which holds no values"}, None, None]
    # The weight check leaves the same parameter out, as lane T2 made it.
    assert records[0]["weight_norms"]["extra"] != records[0]["weight_norms"]["extra"]  # NaN
    # The new field reaches the evidence and the stream's rows like any other key.
    tensorquill.build_evidence(records)
    assert row_to_record(record_to_row(records[0]))["grads_not_read"] == said[0]


def _one(make: callable) -> dict:
    """One step of a model whose second trainable parameter has the gradient `make()` returns.
    Skipped where this PyTorch cannot make such a parameter and gradient (a capability check,
    as the plan's PyTorch 2.6 entry does, not a version string)."""
    try:
        param = nn.Parameter(make().detach().clone())
        param.grad = make()
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"this PyTorch cannot make the parameter: {type(error).__name__}: {error}")
    model = nn.ParameterList([nn.Parameter(torch.ones(2, 2)), param])
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, layers=(), weights=False, auto_step=False, every=1)
    model[0].grad = torch.full((2, 2), 3.0)
    param.grad = make()
    session.step()
    session.close()
    return sink.records[0]


SPARSE = {
    "coo": lambda: torch.full((2, 2), 2.0).to_sparse(),
    "csr": lambda: torch.full((2, 2), 2.0).to_sparse_csr(),
    "csc": lambda: torch.full((2, 2), 2.0).to_sparse_csc(),
    "bsr": lambda: torch.full((2, 2), 2.0).to_sparse_bsr((1, 1)),
    "bsc": lambda: torch.full((2, 2), 2.0).to_sparse_bsc((1, 1)),
}


@pytest.mark.parametrize("layout", sorted(SPARSE))
def test_a_sparse_gradient_is_read_from_its_stored_values(layout):
    record = _one(SPARSE[layout])
    assert record["grad_norms"] == {"0": pytest.approx(6.0), "1": pytest.approx(4.0)}
    assert "grads_not_read" not in record


UNREADABLE = {
    "meta": (lambda: torch.ones(2, 2, device="meta"), "its gradient is on meta, which holds no values"),
    "coo-on-meta": (lambda: torch.ones(2, 2).to_sparse().to("meta"), "its gradient is on meta, which holds no values"),
    # The CPU has no norm for float8: found by reading the rest of its group one at a time.
    "float8": (lambda: torch.full((2, 2), 2.0).to(torch.float8_e4m3fn), "reading it raised"),
}


@pytest.mark.parametrize("kind", sorted(UNREADABLE))
def test_an_unreadable_gradient_is_named_and_the_others_keep_their_norms(kind):
    make, why = UNREADABLE[kind]
    if kind == "float8":
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("this PyTorch has no float8")
        try:
            torch.linalg.vector_norm(make())
        except Exception:  # noqa: BLE001
            pass  # the case under test: the CPU has no norm for it
        else:
            pytest.skip("this PyTorch can take a float8 norm on the CPU, so nothing raises to test")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # PyTorch's own note on the float8 copy, if any
        record = _one(make)
    assert record["grad_norms"] == {"0": pytest.approx(6.0)}
    assert record["grad_norm"] == pytest.approx(6.0)
    assert list(record["grads_not_read"]) == ["1"] and record["grads_not_read"]["1"].startswith(why)


def test_a_sparse_embedding_steps_as_before():
    torch.manual_seed(0)
    model = nn.Embedding(10, 3, sparse=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, every=1):
        model(torch.tensor([1, 2, 2])).sum().backward()
        opt.step()
    assert set(sink.records[0]["grad_norms"]) == {"weight"} and "grads_not_read" not in sink.records[0]


# --- C04: a closed session ------------------------------------------------------------------------


class Loader:
    """A minimal loader: iterating it yields two batches."""

    def __iter__(self):
        return iter([torch.ones(1), torch.ones(1)])


def _every_public_call(session: capture.Session, model: nn.Module, tmp_path: Path) -> None:
    model(torch.randn(2, 4)).sum().backward()
    session.record_grads()
    session.record_grad_norm(torch.tensor(1.0))
    session.record_updates(torch.optim.SGD(model.parameters(), lr=0.1))
    session.record_replica_spread(torch.zeros(2, 3, dtype=torch.float64))
    session.log({"loss": 1.0})
    session.data([1, 2])
    session.checkpoint({"model": {}}, event="save")
    session.checkpoint_files(tmp_path)
    session.set_phase("eval")
    session.tick()
    with session.phase("eval"):
        pass
    session.watch_loaders()
    assert list(session.watch_loader(Loader())) == [torch.ones(1)] * 2
    assert list(session.time_data([1, 2])) == [1, 2]
    session.step()
    session.flush()
    session.close()


def test_every_public_call_on_a_closed_session_returns_at_once_and_queues_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "QUEUE_STEPS", 8)  # a queue of 8 blocks at every=1, to go past quickly
    model = nn.Linear(4, 4)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, auto_step=False, updates=True,
                               heartbeat_seconds=None, stack_seconds=None)
    model(torch.randn(2, 4)).sum().backward()
    session.step()
    session.close()
    written = list(sink.records)
    assert [r["step"] for r in written] == [0]
    failed: list[BaseException] = []

    def calls() -> None:
        try:
            for _ in range(capture._queue_blocks(1) * 3):  # past the writer queue's capacity
                _every_public_call(session, model, tmp_path)
        except BaseException as error:  # noqa: BLE001
            failed.append(error)

    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        worker = threading.Thread(target=calls, daemon=True)
        worker.start()
        worker.join(DEADLINE)
    assert not worker.is_alive(), "a call on a closed session blocked"
    assert failed == []
    assert session._queue.qsize() == 0 and sink.records == written
    assert [w for w in said if "TensorQuill" in str(w.message)] == []  # nothing said on the caller's thread


def test_a_closed_session_does_not_raise_under_warnings_as_errors(caplog):
    model = nn.Linear(4, 4)
    session = tensorquill.init(model, sink=tensorquill.MemorySink(), every=1)  # auto_step: step() raised before
    session.close()
    with warnings.catch_warnings(), caplog.at_level(logging.DEBUG, logger="tensorquill"):
        warnings.simplefilter("error")
        session.step()
        session.log({"loss": 1.0})
        session.data([1])
        session.flush()
    assert caplog.records == []  # no warning and no log line: it returns and does nothing


def test_close_still_writes_the_steps_left_in_the_ring():
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=50)
    for _ in range(3):
        model(torch.randn(2, 4)).sum().backward()
        opt.step()
    session.close()
    assert [r["step"] for r in sink.records] == [0, 1, 2]


# --- B8: init traced by torch.compile -------------------------------------------------------------


def _b8_loop(sink: tensorquill.MemorySink | None, counts: list[float]) -> callable:
    def loop() -> capture.Session | None:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(2, 3), nn.Sigmoid(), nn.Linear(3, 1), nn.Sigmoid())
        params = list(model.parameters())
        opt = torch.optim.Adam(params)
        session = tensorquill.init(model, sink=sink, every=1) if sink is not None else None
        for i in range(6):
            opt.zero_grad()
            if i != 3:
                loss = model(torch.ones(3, 2)).sum()
                loss.backward()
                if session is not None:
                    session.log({"loss": loss})
            opt.step()
            counts.append(float(opt.state[params[0]]["step"]))
        return session

    return loop


def _compiled(fn: callable, *, errors: bool) -> object:
    torch._dynamo.reset()
    try:
        with warnings.catch_warnings():
            # Dynamo's notes on what it does not trace, or every warning as an error.
            warnings.simplefilter("error" if errors else "ignore", Warning if errors else UserWarning)
            return torch.compile(fn, backend="eager")()
    finally:
        torch._dynamo.reset()


def _control(fn: callable, *, errors: bool) -> None:
    """The same compiled loop without TensorQuill: where this PyTorch cannot run it, the test has
    nothing to show and is skipped (a capability check, not a version string)."""
    try:
        _compiled(fn, errors=errors)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"this PyTorch cannot run the loop compiled without TensorQuill: {type(error).__name__}")


def test_init_inside_a_compiled_function_runs_and_captures():
    """TrainCheck's PT-115607 loop (B8 in the plan), with TensorQuill started inside the function
    Dynamo traces, as `experiments/traincheck-bugs/b08_pt115607.py` starts it."""
    _control(_b8_loop(None, []), errors=False)
    sink = tensorquill.MemorySink()
    counts: list[float] = []
    session = _compiled(_b8_loop(sink, counts), errors=False)
    session.close()
    assert counts == [1.0, 2.0, 3.0, 3.0, 4.0, 5.0]
    assert [r["step"] for r in sink.records] == list(range(6))
    for i, record in enumerate(sink.records):
        trained = i != 3
        assert bool(record["layers"]) == trained and bool(record["grad_norms"]) == trained
        assert ("scalars" in record) == trained


def test_init_then_compile_the_model_still_captures():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    torch._dynamo.reset()
    with tensorquill.init(model, sink=sink, every=1, layers="all"):
        compiled = torch.compile(model, backend="eager")
        for _ in range(2):
            opt.zero_grad()
            compiled(torch.ones(3, 2)).sum().backward()
            opt.step()
    torch._dynamo.reset()
    assert [sorted(r["layers"]) for r in sink.records] == [["0", "2"], ["0", "2"]]


def test_the_optimizer_hooks_do_not_raise_when_optimizer_step_is_traced_under_warnings_as_errors():
    """`optimizer.step()` inside a compiled function: traced, the session's optimizer hooks made
    Dynamo warn on `sys._getframe`, which `-W error` raised out of the step."""
    def make() -> tuple[nn.Module, callable]:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(2, 3), nn.Sigmoid(), nn.Linear(3, 1))
        opt = torch.optim.Adam(model.parameters())
        compiled_model = torch.compile(model, backend="eager")

        def train() -> None:
            for _ in range(3):
                opt.zero_grad()
                compiled_model(torch.ones(3, 2)).sum().backward()
                opt.step()

        return model, train

    _control(make()[1], errors=True)
    model, train = make()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1)
    _compiled(train, errors=True)
    session.close()
    assert [r["step"] for r in sink.records] == [0, 1, 2]
    assert all(r["grad_norms"] and r["layers"] for r in sink.records)


# --- C10: a rejected init changes nothing ---------------------------------------------------------


def _state(model: nn.Module, scaler: torch.amp.GradScaler, root: Path) -> dict:
    import torch.nn.utils as nn_utils

    return {
        "hooks": [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()],
        # TensorQuill's own threads (writer, stream sender, heartbeat): others' come and go.
        "threads": sorted(t.name for t in threading.enumerate() if "tensorquill" in t.name.lower()),
        "optimizer_hooks": (len(_global_optimizer_pre_hooks), len(_global_optimizer_post_hooks)),
        "clip": nn_utils.clip_grad_norm_,
        "scaler": dict(vars(scaler)).get("update"),
        "loader_next": datawait._BaseDataLoaderIter.__dict__.get("__next__"),
        "watching": list(datawait._watching),
        "files": sorted(str(p.relative_to(root)) for p in root.rglob("*")),
    }


class Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), nn.ReLU()) for _ in range(2)])
        self.router = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.router(x))


REJECTED = {
    "absent probe head": {"probe": torch.ones(1, 4), "probe_head": "nope"},
    "probe head not a Linear": {"probe": torch.ones(1, 4), "probe_head": "blocks.0"},
    "unknown router": {"routers": {"nope": 4}, "capture": "full"},
    "unknown layer": {"layers": ["nope"]},
    "bad replica_params": {"replica_params": ["nope"]},
    "bad probe": {"probe": 3},
    "bad every": {"every": 0},
}


@pytest.mark.parametrize("case", sorted(REJECTED))
def test_a_rejected_init_leaves_hooks_threads_globals_and_files_as_they_were(case, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = Head()
    scaler = torch.amp.GradScaler("cpu")
    before = _state(model, scaler, tmp_path)
    with pytest.raises((ValueError, TypeError)):
        tensorquill.init(model, run_id="rejected", scaler=scaler, data_wait=True, **REJECTED[case])
    assert _state(model, scaler, tmp_path) == before
    assert not (tmp_path / "tensorquill-out").exists()


def test_a_later_failure_undoes_every_effect_made_before_it(tmp_path, monkeypatch):
    """The last effect `init` makes (the data-loader timer) fails: everything made before it is
    undone, with the default sink's file and folder and the run manifest."""
    monkeypatch.chdir(tmp_path)
    model = Head()
    scaler = torch.amp.GradScaler("cpu")
    before = _state(model, scaler, tmp_path)

    def late(session: object) -> None:
        raise RuntimeError("late")

    monkeypatch.setattr(datawait, "watch", late)
    with pytest.raises(RuntimeError, match="late"):
        tensorquill.init(model, run_id="late", scaler=scaler, data_wait=True, probe=torch.ones(1, 4),
                         probe_head="lm_head", routers={"router": 4}, capture="full")
    monkeypatch.undo()
    assert _state(model, scaler, tmp_path) == before
    assert not (tmp_path / "tensorquill-out").exists()


def test_a_later_failure_puts_an_earlier_runs_manifest_and_end_marker_back(tmp_path, monkeypatch):
    """A run started again under its run ID: its manifest and the finished run's end marker are
    left as they were when the new start fails."""
    monkeypatch.chdir(tmp_path)
    model = nn.Linear(4, 4)
    session = tensorquill.init(model, run_id="again")
    session.close()
    folder = tmp_path / "tensorquill-out"
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    assert any(name.endswith(".end.json") for name in before) and any(name.endswith(".run.json") for name in before)

    def late(session: object) -> None:
        raise RuntimeError("late")

    monkeypatch.setattr(datawait, "watch", late)
    with pytest.raises(RuntimeError, match="late"):
        tensorquill.init(model, run_id="again", data_wait=True, tags=["changed"])
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


def test_a_rejected_init_does_not_remove_a_finished_runs_end_marker(tmp_path):
    model = nn.Linear(4, 4)
    path = tmp_path / "run.rank0.jsonl"
    session = tensorquill.init(model, sink=tensorquill.JsonlSink(path), run_id="run")
    session.close()
    marker = tmp_path / "run.rank0.end.json"
    assert marker.exists()
    with pytest.raises(ValueError):
        tensorquill.init(model, sink=tensorquill.JsonlSink(path), run_id="run", layers=["nope"])
    assert marker.exists()


# --- C19: the scaler's update ---------------------------------------------------------------------


def _scaled_steps(model: nn.Module, opt: torch.optim.Optimizer, scaler: torch.amp.GradScaler, n: int = 1) -> None:
    for _ in range(n):
        opt.zero_grad()
        scaler.scale(model(torch.ones(2, 4)).sum()).backward()
        scaler.step(opt)
        scaler.update()


def test_closing_puts_back_an_update_someone_installed_before_init():
    scaler = torch.amp.GradScaler("cpu")
    original = scaler.update
    calls: list[int] = []

    def theirs(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    scaler.update = theirs
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, scaler=scaler, every=1):
        _scaled_steps(model, opt, scaler, 2)
    assert vars(scaler)["update"] is theirs and len(calls) == 2
    assert [r["step"] for r in sink.records] == [0, 1]


def test_closing_leaves_a_scaler_with_its_class_method():
    scaler = torch.amp.GradScaler("cpu")
    session = tensorquill.init(nn.Linear(4, 1), sink=tensorquill.MemorySink(), scaler=scaler)
    assert "update" in vars(scaler)
    session.close()
    assert "update" not in vars(scaler) and scaler.update.__func__ is torch.amp.GradScaler.update


@pytest.mark.parametrize("first", ["a", "b"])
def test_two_sessions_on_one_scaler_keep_counting_whichever_closes_first(first):
    scaler = torch.amp.GradScaler("cpu")
    models = {"a": nn.Linear(4, 1), "b": nn.Linear(4, 1)}
    opts = {k: torch.optim.SGD(m.parameters(), lr=0.1) for k, m in models.items()}
    sinks = {k: tensorquill.MemorySink() for k in models}
    sessions = {k: tensorquill.init(models[k], sink=sinks[k], scaler=scaler, every=1) for k in models}
    other = "b" if first == "a" else "a"
    sessions[first].close()
    _scaled_steps(models[other], opts[other], scaler, 2)
    sessions[other].close()
    assert [r["step"] for r in sinks[other].records] == [0, 1]
    assert sinks[first].records == []
    assert "update" not in vars(scaler)


def test_a_wrapper_installed_on_top_after_init_is_left_in_place():
    scaler = torch.amp.GradScaler("cpu")
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, scaler=scaler, every=1)
    ours = scaler.update

    def later(*args, **kwargs):
        return ours(*args, **kwargs)

    scaler.update = later
    _scaled_steps(model, opt, scaler)
    session.close()
    assert vars(scaler)["update"] is later
    _scaled_steps(model, opt, scaler)  # what is left under it passes the call through
    assert [r["step"] for r in sink.records] == [0]


# --- The probe's warning, and the other warnings training reaches --------------------------------


def test_a_failed_probe_does_not_raise_out_of_the_step_under_warnings_as_errors(caplog):
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()

    def probe(model: nn.Module) -> None:
        raise ValueError("probe boom")

    session = tensorquill.init(model, sink=sink, probe=probe, probe_every=1, every=1)
    with warnings.catch_warnings(), caplog.at_level(logging.WARNING, logger="tensorquill"):
        warnings.simplefilter("error")
        model(torch.randn(2, 4)).sum().backward()
        opt.step()
    session.close()
    assert sink.records[0]["probe"] == {"error": "ValueError: probe boom"}
    assert any("probe failed" in r.getMessage() for r in caplog.records)


def test_a_failed_probe_still_warns_where_warnings_are_not_errors():
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    def probe(model: nn.Module) -> None:
        raise ValueError("probe boom")

    with tensorquill.init(model, sink=tensorquill.MemorySink(), probe=probe, probe_every=1, every=1):
        with pytest.warns(RuntimeWarning, match="probe failed"):
            model(torch.randn(2, 4)).sum().backward()
            opt.step()


class NoTensor(nn.Module):
    def forward(self, x: torch.Tensor) -> str:
        return "no tensor"


class WithRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.router = NoTensor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.router(x)
        return self.lin(x)


def test_a_refused_router_and_steps_of_another_model_do_not_raise_under_warnings_as_errors(caplog):
    model = WithRouter()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    foreign = torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.1)
    session = tensorquill.init(nn.Linear(2, 2), sink=tensorquill.MemorySink(), every=1)  # steps none of these
    routed = tensorquill.init(model, sink=tensorquill.MemorySink(), routers={"router": 4}, every=1)
    with warnings.catch_warnings(), caplog.at_level(logging.WARNING, logger="tensorquill"):
        warnings.simplefilter("error")
        for _ in range(capture.FOREIGN_STEPS_WARNING + 1):
            model(torch.randn(2, 4)).sum().backward()
            opt.step()
            foreign.param_groups[0]["params"][0].grad = torch.ones(2)
            foreign.step()
    session.close()
    routed.close()
    messages = [r.getMessage() for r in caplog.records]
    assert any("not counting the router 'router'" in m for m in messages)
    assert any("has recorded no steps" in m for m in messages)


# --- The drift check's gather, in init's docstring ------------------------------------------------


def test_init_says_how_many_bytes_the_drift_check_gathers():
    """Rewritten by lane Z1 (SDK-22): the scheduled check gathers nothing, which `init` says; the
    bytes an explicit gather moves are said where the gather is, in `replica_spread`."""
    from tensorquill.distributed import replica_spread

    gathered = (FINGERPRINT_WIDTH + 1) * 8  # float64 fingerprints and the flag column (`_fingerprint_table`)
    assert gathered == 32
    doc = " ".join(tensorquill.init.__doc__.split())
    assert "no collective, no store and no wait for another rank" in doc
    assert "a collective every rank of the group must call, which can block" in doc
    assert "gathers 32 bytes" not in doc
    spread_doc = " ".join(replica_spread.__doc__.split())
    assert f"It gathers {gathered} bytes per parameter per rank (24 without the projection)" in spread_doc


def test_the_writer_thread_is_named_once_per_open_session():
    """Guards the C10 thread check: a session's writer thread exists only while it is open."""
    names = lambda: [t.name for t in threading.enumerate()].count("tensorquill-writer")  # noqa: E731
    start = names()
    session = tensorquill.init(nn.Linear(2, 2), sink=tensorquill.MemorySink())
    assert names() == start + 1
    session.close()
    assert names() == start


# --- C03, decided by the author: a failing or lagging sink stops capture, training goes on --------


class FailingSink(tensorquill.MemorySink):
    """Writes the first `good` blocks, then raises on every write, as a disk that fills up."""

    def __init__(self, good: int = 0) -> None:
        super().__init__()
        self.good = good

    def write(self, records):
        if self.good <= 0:
            raise OSError("disk gone")
        self.good -= 1
        super().write(records)


class FailingJsonl(tensorquill.JsonlSink):
    def __init__(self, path: Path, good: int) -> None:
        super().__init__(path)
        self.good = good

    def write(self, records):
        if self.good <= 0:
            raise OSError("disk gone")
        self.good -= 1
        super().write(records)


class StuckSink(tensorquill.MemorySink):
    """A sink whose `write` never returns until the test lets it go, as a hung network filesystem."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def write(self, records):
        self.release.wait()
        super().write(records)


def _stopped_by_a_failing_sink(model: nn.Module, opt: torch.optim.Optimizer, **init) -> capture.Session:
    """A session whose sink fails at its first write, stepped until capture has stopped: the writer
    keeps the error and the next step on the training thread turns it into the stop (`_commit`)."""
    session = tensorquill.init(model, sink=FailingSink(), every=1, **init)
    deadline = time.monotonic() + DEADLINE
    while not session._stopped and time.monotonic() < deadline:
        opt.zero_grad()
        model(torch.ones(2, 4)).sum().backward()
        opt.step()
        if session._error is None:
            time.sleep(0.01)  # give the writer thread its turn
    assert session._stopped
    return session


def _train_plain(sink: object | None, steps: int = 30, every: int = 1) -> tuple[torch.Tensor, object]:
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tensorquill.init(model, sink=sink, every=every) if sink is not None else None
    for _ in range(steps):
        opt.zero_grad()
        model(torch.ones(2, 4)).sum().backward()
        opt.step()
    if session is not None:
        session.close()
    return model.weight.detach().clone(), session


@pytest.mark.parametrize("errors", [False, True], ids=["plain", "warnings-as-errors"])
@pytest.mark.parametrize("every", [1, 4])
def test_a_failing_sink_stops_capture_and_every_step_runs_as_without_tensorquill(errors, every, caplog):
    without, _ = _train_plain(None)
    with warnings.catch_warnings(record=not errors) as said, caplog.at_level(logging.WARNING, logger="tensorquill"):
        warnings.simplefilter("error" if errors else "always")
        with_tq, session = _train_plain(FailingSink(good=2), every=every)
    assert torch.equal(without, with_tq)
    assert isinstance(session.writer_error, OSError) and str(session.writer_error) == "disk gone"
    told = [str(w.message) for w in said if "TensorQuill" in str(w.message)] if not errors else [
        r.getMessage() for r in caplog.records]
    assert len(told) == 1 and "stopped capturing" in told[0] and "OSError: disk gone" in told[0]


def test_the_users_own_exception_comes_out_unchanged_after_the_sink_failed():
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    mine = LookupError("the user's own training error")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(LookupError) as raised:
            with tensorquill.init(model, sink=FailingSink(), every=1) as session:
                for _ in range(3):
                    opt.zero_grad()
                    model(torch.ones(2, 4)).sum().backward()
                    opt.step()
                raise mine
    assert raised.value is mine and raised.value.__context__ is None
    assert isinstance(session.writer_error, OSError)


def test_a_sink_that_never_returns_holds_neither_training_nor_close(monkeypatch):
    monkeypatch.setattr(capture, "QUEUE_STEPS", 8)  # fill the queue in 8 steps rather than 400
    sink = StuckSink()
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    done: list[str] = []

    def run() -> None:
        session = tensorquill.init(model, sink=sink, every=1)
        for _ in range(capture._queue_blocks(1) * 4):
            opt.zero_grad()
            model(torch.ones(2, 4)).sum().backward()
            opt.step()
        done.append("trained")
        session.close()
        done.append(str(session.writer_error))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(DEADLINE)
    sink.release.set()  # let the writer thread go, so it ends with the test
    assert not worker.is_alive(), f"held: {done}"
    assert done[0] == "trained" and "behind training" in done[1]


def test_after_a_stop_no_end_marker_the_cause_kept_and_exactly_one_warning(tmp_path):
    path = tmp_path / "run.rank0.jsonl"
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        session = tensorquill.init(model, sink=FailingJsonl(path, good=3), run_id="run", every=1)
        for _ in range(10):
            opt.zero_grad()
            model(torch.ones(2, 4)).sum().backward()
            opt.step()
        session.log({"loss": 1.0})
        session.flush()
        session.close()
    assert not (tmp_path / "run.rank0.end.json").exists()
    assert isinstance(session.writer_error, OSError)
    assert len([w for w in said if "TensorQuill" in str(w.message)]) == 1
    steps = [json.loads(line)["step"] for line in path.read_text().splitlines()]
    assert steps == [0, 1, 2]  # an unbroken run of steps, then nothing


def test_after_a_stop_the_optimizer_hooks_do_nothing_and_close_takes_them_off():
    before = (len(_global_optimizer_pre_hooks), len(_global_optimizer_post_hooks))
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = _stopped_by_a_failing_sink(model, opt)
    hooks = [_global_optimizer_pre_hooks.get(h.id) for h in session._optimizer_handles]
    hooks += [_global_optimizer_post_hooks.get(h.id) for h in session._optimizer_handles]
    assert capture._noop_hook in hooks
    session.close()
    assert (len(_global_optimizer_pre_hooks), len(_global_optimizer_post_hooks)) == before


def test_what_the_heartbeat_reads_stays_live_after_a_stop():
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = _stopped_by_a_failing_sink(model, opt)
    session.set_phase("eval")
    session.tick()
    assert session._phase == "eval" and session._ticks == 1
    with session.phase("checkpoint"):
        assert session._phase == "checkpoint"
    assert session._phase == "eval"
    session.close()


def test_close_returns_within_its_bound_when_the_last_write_is_slow(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "CLOSE_SECONDS", 0.3)
    path = tmp_path / "slow.rank0.jsonl"
    release = threading.Event()

    class Slow(tensorquill.JsonlSink):
        def write(self, records):
            release.wait()
            super().write(records)

    session = tensorquill.init(nn.Linear(2, 2), sink=Slow(path), run_id="slow", every=50, auto_step=False)
    session.step()
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        started = time.monotonic()
        session.close()
        took = time.monotonic() - started
    release.set()
    assert took < 0.3 + capture.STOPPED_CLOSE_SECONDS + 1.0
    assert "had not finished" in str(session.writer_error)
    assert not (tmp_path / "slow.rank0.end.json").exists()
    assert len([w for w in said if "TensorQuill" in str(w.message)]) == 1


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_a_stop_causes_no_graph_break_and_no_recompile(toy_moe, backend):
    """The forward hooks stay as they are after a stop: nothing they read under `torch.compile`
    changes, so the compiled model is neither traced again nor broken."""
    sink = FailingSink(good=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = tensorquill.init(toy_moe, sink=sink, every=1, capture="full", routers={"router": 4})
        compiled = torch.compile(toy_moe, backend=backend)
        opt = torch.optim.SGD(toy_moe.parameters(), lr=0.01)
        torch._dynamo.utils.counters.clear()
        with torch._dynamo.config.patch(error_on_recompile=True):
            for _ in range(8):
                opt.zero_grad()
                compiled(torch.randn(16, 8)).sum().backward()
                opt.step()
        session.close()
    assert session._stopped and isinstance(session.writer_error, OSError)
    assert dict(torch._dynamo.utils.counters["graph_break"]) == {}
    assert torch._dynamo.utils.counters["frames"]["ok"] >= 1
    assert [r["step"] for r in sink.records] == [0, 1]


STOPPED_RANK_LIMIT = 120.0  # seconds from the spawn the test waits for both ranks
STOPPED_RANK_STACKS = 100.0  # seconds after its start a worker still running dumps every thread's stack


def _progress(out_dir, rank, what: str) -> None:
    """One line in `progress<rank>`: wall-clock time, pid, what the rank is about to do or has done.
    The file is opened, appended to and closed each time, so the line is on disk whatever the
    process does next (waits for good, or is terminated by the test). The test prints these when
    a rank does not finish: CI job 136014937059 timed out here once with no output from either
    rank, so nothing said where each one was."""
    import os

    with open(Path(out_dir) / f"progress{rank}", "a") as file:
        file.write(f"{time.time():.3f} pid {os.getpid()}: {what}\n")


def _stopped_rank_worker(rank, store_path, out_dir):
    import atexit
    import faulthandler
    import signal

    import torch.distributed as dist

    def say(what: str) -> None:
        _progress(out_dir, rank, what)

    say("worker started")
    # Where every thread is, written from C without the GIL (so also while blocked in gloo or the
    # store): once on its own STOPPED_RANK_STACKS s in, and again when the test, giving up, sends
    # SIGUSR1. faulthandler keeps the file open for as long as it needs it.
    stacks = open(Path(out_dir) / f"stacks{rank}", "w")
    faulthandler.dump_traceback_later(STOPPED_RANK_STACKS, file=stacks)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=stacks, all_threads=True)
    # A worker that dies of a signal says where too. In CI on Nov 20, 2026 (`73eac65`, Python 3.11,
    # PyTorch 2.15) rank 0 aborted after this function had returned ("terminate called without an
    # active exception", SIGABRT) and nothing was dumped: the two lines below are for the next time.
    faulthandler.enable(file=stacks, all_threads=True)
    atexit.register(lambda: say("atexit: the interpreter is shutting down; threads alive: "
                                + ", ".join(sorted(t.name for t in threading.enumerate()))))
    # Any gather the session enters, logged on the way in and out and passed through unchanged.
    # Since the SDK-22 wave (lane Z1) the session issues none on its own schedule, live or
    # stopped, and the agreement at init is gone: the test asserts no gather is entered at all.
    box: list = []
    real_gather = capture._spread_and_unread
    gathers = [0]

    def gather(*args, **kwargs):
        gathers[0] += 1
        n, session = gathers[0], (box[0] if box else None)
        where = "" if session is None else f" at step {session._next_step}, session stopped {session._stopped}"
        say(f"gather {n} entered{where}")
        try:
            out = real_gather(*args, **kwargs)
        except BaseException as error:
            say(f"gather {n} raised {' '.join(f'{type(error).__name__}: {error}'.split())[:300]}")
            raise
        say(f"gather {n} returned")
        return out

    capture._spread_and_unread = gather
    say("init_process_group")
    dist.init_process_group("gloo", rank=rank, world_size=2, store=dist.FileStore(store_path, 2),
                            timeout=timedelta(seconds=60))
    try:
        say("init_process_group returned")
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        path = Path(out_dir) / f"rank{rank}.jsonl"
        sink = FailingJsonl(path, good=1) if rank == 0 else tensorquill.JsonlSink(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            say("tensorquill.init")
            session = tensorquill.init(model, run_id="r1", sink=sink, every=1, replica_group=dist.group.WORLD,
                                       replica_every=2)
            box.append(session)
            say("tensorquill.init returned")
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            for i in range(8):
                model(torch.ones(2, 4)).sum().backward()
                optimizer.step()
                optimizer.zero_grad()
                say(f"step {i} done, session stopped {session._stopped}")
            say("session.close")
            session.close()
            say(f"session.close returned, writer_error {session.writer_error!r}")
        (Path(out_dir) / f"done{rank}").write_text(str(session.writer_error))
    finally:
        say("destroy_process_group")
        dist.destroy_process_group()
        say("destroy_process_group returned; the worker returns; threads alive: "
            + ", ".join(sorted(t.name for t in threading.enumerate())))


def _ranks_report(context, out_dir: Path, started: float, status: list[str]) -> str:
    """What each rank said (`_progress`), in seconds since the test spawned them, and the stacks
    its threads were in (`_stopped_rank_worker`), after the status taken by the caller."""
    parts = list(status)
    for rank in range(len(context.processes)):
        parts.append(f"--- rank {rank} progress (seconds since the spawn) ---")
        try:
            for line in (out_dir / f"progress{rank}").read_text().splitlines():
                stamp, _, rest = line.partition(" ")
                try:
                    parts.append(f"+{float(stamp) - started:7.2f}s {rest}")
                except ValueError:
                    parts.append(line)
        except OSError:
            parts.append("(no progress file: the worker function never started)")
        parts.append(f"--- rank {rank} stacks ---")
        try:
            parts.append((out_dir / f"stacks{rank}").read_text().strip() or "(none dumped)")
        except OSError:
            parts.append("(no stacks file)")
    return "\n".join(parts)


def test_a_stopped_rank_and_a_live_one_enter_no_gather_and_both_finish(tmp_path):
    """Rank 0's sink fails at its second block, so its capture stops; rank 1's goes on. Both were
    given a replica group with the check due every second step. Before lane Z1 the check was an
    `all_gather` that a stopped rank had to keep entering (`_gather_and_drop`), or rank 1 waited
    there; now neither rank enters a gather, and rank 1 writes its own observation on each
    scheduled step. Rewritten by lane Z1 from `test_a_stopped_rank_keeps_entering_the_drift_checks_gather`,
    whose premise (a scheduled gather) is gone; the diagnostics are kept."""
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available():
        pytest.skip("needs torch.distributed")
    import os
    import signal

    import torch.multiprocessing as mp

    def status(context) -> list[str]:
        return [
            f"rank {rank}: pid {p.pid}, alive {p.is_alive()}, exit code {p.exitcode}, "
            f"done file {'written' if (tmp_path / f'done{rank}').exists() else 'not written'}"
            for rank, p in enumerate(context.processes)
        ]

    started = time.time()
    context = mp.start_processes(_stopped_rank_worker, args=(str(tmp_path / "store"), str(tmp_path)), nprocs=2,
                                 join=False, start_method="spawn")
    try:
        while not context.join(timeout=0.5):
            if time.time() - started > STOPPED_RANK_LIMIT:
                break
    except Exception as error:  # a rank raised or died: its traceback, with where each rank was
        pytest.fail(f"{error}\n{_ranks_report(context, tmp_path, started, status(context))}")
    if not all(p.exitcode is not None for p in context.processes):
        seen = status(context)
        for process in context.processes:
            if process.is_alive() and hasattr(signal, "SIGUSR1"):
                try:
                    os.kill(process.pid, signal.SIGUSR1)  # each thread's stack, to its stacks file
                except OSError:
                    pass
        time.sleep(1.0)
        report = _ranks_report(context, tmp_path, started, seen)
        for process in context.processes:
            process.terminate()
        pytest.fail(f"a rank had not finished {STOPPED_RANK_LIMIT:g} s after the spawn; where each rank was:\n"
                    f"{report}")
    for rank in (0, 1):  # neither rank entered a gather, the stopped one included
        said = (tmp_path / f"progress{rank}").read_text()
        assert "gather 1 entered" not in said, said
        assert "step 7 done" in said, said
    assert "disk gone" in (tmp_path / "done0").read_text() and (tmp_path / "done1").read_text() == "None"
    records = [json.loads(line) for line in (tmp_path / "rank1.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == list(range(8))
    assert [r["step"] for r in records if "replica" in r] == [1, 3, 5, 7]
    assert not any("replica_spread" in r for r in records)
    assert {r["replica"]["group"] == [0, 1] for r in records if "replica" in r} == {True}


def _scaled_run(monkeypatch, stop: bool) -> tuple[int, int]:
    """Four iterations under a GradScaler, the second with an inf gradient (the scaler skips that
    step), on a session with a drift-check group (its gather faked): where the session's step count
    and its gathers end up. The group is a stand-in whose ranks cannot be read, so it gets no
    observation either."""
    gathers = [0]

    def fake(params, group, *, projection, device):
        gathers[0] += 1
        n = len(params)
        return torch.zeros(n, 3, dtype=torch.float64), torch.zeros(n, dtype=torch.float64), [None] * n

    monkeypatch.setattr(capture, "_spread_and_unread", fake)
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = tensorquill.init(model, sink=tensorquill.MemorySink(), every=1, scaler=scaler,
                                   replica_group=object(), replica_every=1)
        if stop:
            session._stop(RuntimeError("stopped for the test"))
        for i in range(4):
            opt.zero_grad()
            scaler.scale(model(torch.ones(2, 4)).sum()).backward()
            if i == 1:
                model.weight.grad.fill_(float("inf"))
            scaler.step(opt)
            scaler.update()
        steps = session._next_step
        session.close()
    return steps, gathers[0]


def test_under_a_scaler_neither_a_live_nor_a_stopped_rank_gathers_on_its_own(monkeypatch):
    """A step the scaler skipped ends a live session's step at `scaler.update()`, and the live
    session counts all four. Before lane Z1 a stopped rank had to count them too, or its scheduled
    gather fell a step behind the other ranks'; with no scheduled gather (SDK-22) neither enters
    one, and a stopped rank's hooks are off like any stopped session's. Rewritten by lane Z1 from
    `test_a_stopped_rank_counts_the_steps_a_scaler_ends_as_a_live_one_does`."""
    assert _scaled_run(monkeypatch, stop=False) == (4, 0)
    assert _scaled_run(monkeypatch, stop=True)[1] == 0


# --- C03's trigger, decided by the author: the queue sized in steps, about 400, floor 2 rings -----


@pytest.mark.parametrize(("every", "rings", "blocks"), [
    (1, 400, 400), (3, 134, 268), (50, 8, 16), (100, 4, 8), (200, 2, 4), (1000, 2, 4),
])
def test_the_writer_queue_holds_about_400_steps_with_a_floor_of_two_rings(every, rings, blocks):
    assert capture._queue_rings(every) == rings and capture._queue_blocks(every) == blocks
    session = tensorquill.init(nn.Linear(2, 2), sink=tensorquill.MemorySink(), every=every, auto_step=False)
    assert session._queue.maxsize == blocks
    session.close()


def test_the_stop_warning_names_the_real_capacity(monkeypatch):
    monkeypatch.setattr(capture, "QUEUE_STEPS", 6)  # 2 rings of 3 steps: 4 blocks
    sink = StuckSink()
    session = tensorquill.init(nn.Linear(2, 2), sink=sink, every=3, auto_step=False)
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        for _ in range(30):
            session.step()
    sink.release.set()
    session.close()
    assert "4 blocks (6 steps) behind" in str(session.writer_error)
    assert len([w for w in said if "TensorQuill" in str(w.message)]) == 1


def test_close_at_the_two_ring_floor_waits_for_a_slow_but_working_writer(tmp_path):
    """At every=200 the queue holds 2 rings, 4 blocks, and here it is full when `close()` flushes the
    step kept back. A writer that is slow but working gets until the close's deadline: every step is
    written and the end marker is left, as for any run that closed normally."""
    path = tmp_path / "floor.rank0.jsonl"
    release = threading.Event()

    class Slow(tensorquill.JsonlSink):
        def write(self, records):
            release.wait(DEADLINE)
            super().write(records)

    session = tensorquill.init(nn.Linear(2, 2), sink=Slow(path), run_id="floor", every=200, auto_step=False)
    for _ in range(600):  # 3 rings: one block in the write, 4 queued, the newest step kept back
        session.step()
    assert session._queue.full() and not session._stopped
    threading.Timer(0.5, release.set).start()
    session.close()
    assert session.writer_error is None
    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == list(range(600))
    assert (tmp_path / "floor.rank0.end.json").exists()


def test_an_exception_part_way_through_a_flush_stops_capture_and_later_steps_run(monkeypatch):
    """Lane U5 saw every step after one raise inside `_flush` raise `IndexError`: the ring's lists were
    left longer than the ring. A flush that raises now stops capture, which clears the ring."""
    real = capture._scalars_to_cpu
    calls = [0]

    def once(rows):
        calls[0] += 1
        if calls[0] == 3:  # in the second flush (each flush converts two lists of scalars)
            raise RuntimeError("device lost for a moment")
        return real(rows)

    monkeypatch.setattr(capture, "_scalars_to_cpu", once)
    without, _ = _train_plain(None, steps=12)
    sink = tensorquill.MemorySink()
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        with_tq, session = _train_plain(sink, steps=12, every=2)
    assert torch.equal(without, with_tq)
    assert "device lost" in str(session.writer_error)
    told = [str(w.message) for w in said if "TensorQuill" in str(w.message)]
    assert len(told) == 1 and "moving a block of steps off the device raised" in told[0]
    steps = [r["step"] for r in sink.records]
    assert steps and steps == list(range(len(steps))) and len(steps) < 12  # an unbroken run, then nothing


# --- Lane T2's item 3, decided by the author: option (b), subclasses refused under torch.compile ---


class Strict(torch.Tensor):
    """A traceable wrapper subclass that refuses a plain tensor beside it, as DTensor does (lane
    T2's `lane-t2-router-strict-subclasses.py`)."""

    @staticmethod
    def __new__(cls, elem):
        return torch.Tensor._make_wrapper_subclass(cls, elem.shape, dtype=elem.dtype, device=elem.device)

    def __init__(self, elem):
        self.elem = elem

    def __tensor_flatten__(self):
        return ["elem"], None

    @staticmethod
    def __tensor_unflatten__(inner, meta, outer_size, outer_stride):
        return Strict(inner["elem"])

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        from torch.utils._pytree import tree_flatten, tree_map_only

        flat, _ = tree_flatten((args, kwargs or {}))
        if any(type(a) is torch.Tensor for a in flat):
            raise RuntimeError(f"{func} got mixed torch.Tensor and Strict")
        unwrap = lambda x: tree_map_only(Strict, lambda t: t.elem, x)  # noqa: E731
        out = func(*unwrap(args), **unwrap(kwargs or {}))
        return tree_map_only(torch.Tensor, lambda t: Strict(t) if not isinstance(t, Strict) else t, out)


class NoEq(torch.Tensor):
    """A `__torch_function__` subclass that raises on `eq`."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func in (torch.Tensor.__eq__, torch.eq, torch.Tensor.eq):
            raise RuntimeError("NoEq refuses eq")
        return super().__torch_function__(func, types, args, kwargs or {})


WRAPS = {"strict": lambda t: Strict(t), "no_eq": lambda t: t.as_subclass(NoEq)}


class SubRouter(nn.Module):
    def __init__(self, wrap):
        super().__init__()
        self.proj = nn.Linear(8, 4)
        self.wrap = wrap

    def forward(self, x):
        return self.wrap(self.proj(x).detach())  # beside the computation: the model never mixes them


class SubBlock(nn.Module):
    def __init__(self, wrap):
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.wrap = wrap

    def forward(self, x):
        y = self.lin(x)
        return self.wrap(y.detach()), y


class SubModel(nn.Module):
    def __init__(self, wrap):
        super().__init__()
        self.router = SubRouter(wrap)
        self.blocks = nn.ModuleList([SubBlock(wrap)])
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        self.router(x)
        _, y = self.blocks[0](x)
        return self.out(y).sum()


def _sub_run(wrap: str, *, backend: str | None, fullgraph: bool = False, hooked: bool = True,
             indices: bool = False) -> list[dict]:
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = SubModel(WRAPS[wrap])
    sink = tensorquill.MemorySink()
    session = None
    if hooked:
        spec = (tensorquill.Router(4, 2, indices=lambda out: out.topk(2, dim=-1).indices) if indices
                else tensorquill.Router(4, 2))
        session = tensorquill.init(model, sink=sink, every=1, routers={"router": spec}, layers=["blocks.0"])
    forward = model if backend is None else torch.compile(model, backend=backend, fullgraph=fullgraph)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(3):
                forward(torch.randn(3, 8)).backward()
                opt.step()
                opt.zero_grad()
            if session is not None:
                session.close()
    finally:
        torch._dynamo.reset()
    return sink.records


def _sub_control(wrap: str, backend: str, fullgraph: bool) -> None:
    """The same model compiled without TensorQuill: where this PyTorch cannot trace the subclass (it
    has never run on PyTorch 2.6), there is nothing to test, and the test is skipped by that capability."""
    try:
        _sub_run(wrap, backend=backend, fullgraph=fullgraph, hooked=False)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"this PyTorch cannot compile the model without TensorQuill: {type(error).__name__}")


@pytest.mark.parametrize("fullgraph", [False, True], ids=["graph-breaks", "fullgraph"])
@pytest.mark.parametrize("indices", [False, True], ids=["default", "indices="])
def test_a_strict_subclass_is_refused_under_compile_not_raised(fullgraph, indices):
    """Lane T2: the strict wrapper as a router's output raised "aten.eq.Tensor got mixed torch.Tensor
    and Strict", and as a watched block's output "aten.maximum.default got mixed", out of training
    under backend="eager". Now both are refused: the router in `moe_refused`, the block left out."""
    _sub_control("strict", "eager", fullgraph)
    records = _sub_run("strict", backend="eager", fullgraph=fullgraph, indices=indices)
    assert [r["step"] for r in records] == [0, 1, 2]
    assert all("moe" not in r and r["layers"] == {} for r in records)
    reason = records[0]["moe_refused"]["router"]
    assert "is a Strict, a tensor subclass, which TensorQuill does not count under torch.compile" in reason


def test_a_subclass_that_raises_on_eq_is_refused_under_fullgraph():
    """Lane T2: under fullgraph=True Dynamo raised "Observed exception" from the hook's `eq`."""
    _sub_control("no_eq", "eager", True)
    records = _sub_run("no_eq", backend="eager", fullgraph=True)
    assert "is a NoEq, a tensor subclass" in records[0]["moe_refused"]["router"]
    assert all(r["layers"] == {} for r in records)


@pytest.mark.parametrize("wrap", sorted(WRAPS))
def test_eager_mode_still_reads_subclasses_through_its_try(wrap):
    """Eager mode is as before: nothing is refused for its type; what raises is caught."""
    records = _sub_run(wrap, backend=None)
    assert "counting its output raised RuntimeError" in records[0]["moe_refused"]["router"]
    assert all("blocks.0" not in r["layers"] for r in records)  # its reading raised, and was caught


class ParamRouter(nn.Module):
    """A router that returns a parameter as it is, as the author's rule keeps allowed."""

    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.tensor([[2.0, 1.0, 0.0, -1.0]] * 3))

    def forward(self, x):
        return self.table


class ParamBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.full((3, 8), 0.5))

    def forward(self, x):
        return self.weight


class ParamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = ParamRouter()
        self.blocks = nn.ModuleList([ParamBlock()])
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        logits = self.router(x)
        y = self.blocks[0](x)
        return self.out(x + y).sum() + logits.sum() * 0


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_a_module_that_returns_a_parameter_is_still_read_under_compile(backend):
    def run(hooked: bool) -> list[dict]:
        torch._dynamo.reset()
        model = ParamModel()
        sink = tensorquill.MemorySink()
        session = tensorquill.init(model, sink=sink, every=1, routers={"router": tensorquill.Router(4, 2)},
                                   layers=["blocks.0"]) if hooked else None
        compiled = torch.compile(model, backend=backend)
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        try:
            for _ in range(2):
                compiled(torch.ones(3, 8)).backward()
                opt.step()
                opt.zero_grad()
            if session is not None:
                session.close()
        finally:
            torch._dynamo.reset()
        return sink.records

    try:
        run(hooked=False)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"this PyTorch cannot compile the model without TensorQuill: {type(error).__name__}")
    records = run(hooked=True)
    assert [r["moe"]["router"] for r in records] == [[3, 3, 0, 0]] * 2  # top-2 of each of 3 tokens
    assert all("moe_refused" not in r for r in records)
    assert [r["layers"]["blocks.0"]["mean"] for r in records] == [pytest.approx(0.5)] * 2


# --- F1, decided by the author: option (a), the bool routing map is read -------------------------


def _drop(routing_probs, routing_map, topk, capacity_factor, drop_policy, pad_to_capacity):
    """Megatron's `apply_router_token_dropping` (moe_utils.py), as in the reference clone; the plan's
    F1 stand-in (`f1-megatron-capacity-drop.py`). Megatron itself is not installed here."""
    import math

    num_tokens, num_experts = routing_probs.shape
    capacity = math.ceil((num_tokens * topk / num_experts) * capacity_factor)
    if capacity > num_tokens:
        capacity_mask = torch.ones_like(routing_probs).bool()
    elif drop_policy == "probs":
        _, keep = torch.topk(routing_probs, k=capacity, dim=0, sorted=False)
        capacity_mask = torch.zeros_like(routing_probs).scatter(0, keep, 1).bool()
    else:
        _, keep = torch.topk(routing_map.int(), k=capacity, dim=0, sorted=False)
        capacity_mask = torch.zeros_like(routing_probs).scatter(0, keep, 1).bool()
    final_map = capacity_mask if pad_to_capacity else torch.logical_and(routing_map, capacity_mask)
    return routing_probs * final_map, final_map


class MegatronStyleRouter(nn.Module):
    """Megatron's `TopKRouter` routing with a capacity factor: returns `(probs, routing_map)`."""

    def __init__(self, logits, topk=2, capacity_factor=1.0, policy="probs", pad=False, extra_logits=False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(logits.shape[1], 8))
        self.fixed = logits
        self.topk, self.cf, self.policy, self.pad, self.extra = topk, capacity_factor, policy, pad, extra_logits
        self.last = None

    def forward(self, x):
        logits = x @ self.weight.t() + self.fixed
        scores = torch.softmax(logits, dim=-1)
        top_p, top_i = scores.topk(self.topk, dim=-1)
        probs = torch.zeros_like(scores).scatter(1, top_i, top_p)
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter(1, top_i, True)
        probs, routing_map = _drop(probs, routing_map, self.topk, self.cf, self.policy, self.pad)
        self.last = (probs.detach().clone(), routing_map.clone())
        return (logits, probs, routing_map) if self.extra else (probs, routing_map)


class MoELayer(nn.Module):
    def __init__(self, router):
        super().__init__()
        self.router = router
        self.out = nn.Linear(8, 8)

    def forward(self, x):
        probs = self.router(x)[-2]
        return self.out(x) * probs.sum(-1, keepdim=True)


def _routed(router: nn.Module, x: torch.Tensor, *, backend: str | None = None, hooked: bool = True) -> list[int] | None:
    torch._dynamo.reset()
    model = MoELayer(router)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, routers={"router": tensorquill.Router(4, 2)}) if hooked else None
    forward = model if backend is None else torch.compile(model, backend=backend)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    try:
        forward(x).pow(2).mean().backward()
        opt.step()
        if session is not None:
            session.close()
    finally:
        torch._dynamo.reset()
    return sink.records[0]["moe"]["router"] if hooked else None


def _f1_cases() -> dict[str, tuple[nn.Module, torch.Tensor]]:
    torch.manual_seed(0)
    cases = {
        "two-tokens-capacity-1": (
            MegatronStyleRouter(torch.tensor([[4.0, 3.0, 0.0, 0.0], [3.5, 2.5, 0.0, 0.0]])), torch.zeros(2, 8)),
        "partial-drop": (MegatronStyleRouter(torch.tensor(
            [[4.0, 3.0, 0.0, 0.0], [3.5, 0.0, 2.5, 0.0], [0.0, 0.0, 3.0, 2.0], [0.0, 3.0, 0.0, 2.5]]),
            capacity_factor=0.5), torch.zeros(4, 8)),
    }
    xs = torch.randn(64, 8)
    for policy in ("probs", "position"):
        for pad in (False, True):
            cases[f"64-{policy}{'-padded' if pad else ''}"] = (
                MegatronStyleRouter(torch.randn(64, 4) * 2, policy=policy, pad=pad), xs)
    cases["64-no-drop"] = (MegatronStyleRouter(torch.randn(64, 4) * 2, capacity_factor=100.0), xs)
    return cases


# The two cases the plan's F1 entry lists that do not depend on random numbers. The 64-token cases
# are checked against the router's own map; `f1-megatron-capacity-drop.py` (in the session notes) gives
# the plan's numbers for them: [24, 32, 32, 32], [32, 32, 28, 32], [29, 32, 32, 31] padded, no change
# with no capacity limit.
F1_EXPECTED = {"two-tokens-capacity-1": [1, 1, 0, 0], "partial-drop": [1, 1, 1, 1]}


@pytest.mark.parametrize("case", ["two-tokens-capacity-1", "partial-drop", "64-probs", "64-probs-padded",
                                  "64-position", "64-position-padded", "64-no-drop"])
def test_a_megatron_style_router_is_counted_as_its_routing_map_routes(case):
    router, x = _f1_cases()[case]
    counted = _routed(router, x)
    probs, routing_map = router.last
    assert counted == (probs != 0).sum(0).tolist()  # the map's entries whose probability is not zero
    if not case.endswith("padded"):
        assert counted == routing_map.sum(0).tolist()
    if case in F1_EXPECTED:
        assert counted == F1_EXPECTED[case]


def test_every_one_of_the_121_two_token_maps_is_counted_as_it_routes():
    """CORRECTNESS_AUDIT.md's script (`router121.py`): every map of two tokens over four experts with at
    most two assignments a token, the empty token and the token with one assignment included. Before,
    36 of 36 full maps matched and 0 of 85 with fewer assignments."""
    from collections import Counter
    from itertools import combinations, product

    class Gate(nn.Module):
        def forward(self, weights):
            return weights, weights > 0

    model = nn.Sequential(Gate())
    model.register_parameter("scale", nn.Parameter(torch.ones(())))
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, routers={"0": tensorquill.Router(4, top_k=2)}, sink=sink, auto_step=False)
    choices = [indices for size in range(3) for indices in combinations(range(4), size)]
    expected = {}
    for step, rows in enumerate(product(choices, repeat=2), start=1):
        weights = torch.zeros(2, 4)
        for token, experts in enumerate(rows):
            for expert in experts:
                weights[token, expert] = 1
        model(weights)
        session.step(step=step)
        expected[step] = [sum(expert in row for row in rows) for expert in range(4)]
    session.close()
    matched = Counter(record["moe"]["0"] == expected[record["step"]] for record in sink.records)
    assert matched == Counter({True: 121})


class MaskOverExperts(nn.Module):
    def forward(self, x):
        logits = x[:, :4]
        return logits, torch.tensor([True, False, True, False])  # one flag per expert, not per token


def test_a_mask_over_the_experts_alone_is_not_taken_for_a_routing_map():
    model = nn.Sequential(MaskOverExperts())
    model.register_parameter("scale", nn.Parameter(torch.ones(())))
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, routers={"0": tensorquill.Router(4, top_k=1)}, sink=sink, auto_step=False)
    model(torch.tensor([[0.0, 3.0, 1.0, 0.0, 9.0], [2.0, 0.0, 0.0, 1.0, 9.0], [0.0, 0.0, 0.0, 5.0, 9.0]]))
    session.step()
    session.close()
    assert sink.records[0]["moe"]["0"] == [1, 1, 0, 1]  # the top-1 of the logits, as before


def test_with_logits_before_the_probabilities_the_logits_are_paired_and_padding_is_counted():
    """The documented limit: the paired float is the first of the map's shape. Logits are never zero,
    so with `pad_to_capacity` the padding is counted, as the map holds it."""
    torch.manual_seed(0)
    router = MegatronStyleRouter(torch.randn(64, 4) * 2, pad=True, extra_logits=True)
    counted = _routed(router, torch.randn(64, 8))
    probs, routing_map = router.last
    assert counted == routing_map.sum(0).tolist() != (probs != 0).sum(0).tolist()


def test_a_routing_map_on_another_device_is_refused_as_any_output_there_is():
    class MetaMap(nn.Module):
        def forward(self, x):
            probs = torch.softmax(x[:, :4], dim=-1)
            return probs, torch.ones(x.shape[0], 4, dtype=torch.bool, device="meta")

    model = nn.Sequential(MetaMap())
    model.register_parameter("scale", nn.Parameter(torch.ones(())))
    sink = tensorquill.MemorySink()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = tensorquill.init(model, routers={"0": tensorquill.Router(4, top_k=1)}, sink=sink, auto_step=False)
        model(torch.randn(3, 8))
        session.step()
        session.close()
    assert "its routing map is on meta" in sink.records[0]["moe_refused"]["0"]


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_a_compiled_megatron_style_router_counts_as_in_eager_mode(backend):
    router, x = _f1_cases()["64-probs-padded"]
    try:
        _routed(router, x, backend=backend, hooked=False)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"this PyTorch cannot compile the model without TensorQuill: {type(error).__name__}")
    eager = _routed(router, x)
    probs, _ = router.last
    assert _routed(router, x, backend=backend) == eager == (probs != 0).sum(0).tolist()
