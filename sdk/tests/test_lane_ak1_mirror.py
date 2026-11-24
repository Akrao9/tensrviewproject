"""Lane AK-1: the passive in-process scalar mirror, `tensorquill.init(..., mirror_scalars=True)`.

The plan's Phase 1 item 2, the architecture page's section 2.5 and the decision log's entry
"Passive In-Process Scalar Mirroring" (Nov 23, 2026). Plain numeric scalars a script logs through
`wandb.log` / `wandb.Run.log` and `mlflow.log_metric` / `mlflow.log_metrics` are teed into
`session.log()` on TensorQuill's own step, once per user call, never raising, never syncing.

`wandb` and `mlflow` are not installed in the test interpreter, so these tests run against stand-in
modules shaped from the reference sources:
- wandb (`wandb/sdk/wandb_run.py`, `wandb/sdk/lib/module.py`, `wandb/sdk/lib/preinit.py`): the
  module's `log` is a pre-init stub that raises until `wandb.init()`, which rebinds it to the new
  run's bound `log` (`module.set_global(log=run.log)`); `wandb.finish()` puts a new stub back
  (`unset_globals`). `Run.log(self, data, step=None, commit=None)`; `Run.step` is a property that
  waits up to 30 s on the service and raises in shared mode, so the stand-in's raises if read.
- mlflow (`mlflow/tracking/fluent.py`, `mlflow/__init__.py`): `log_metric(key, value, step=None,
  synchronous=None, timestamp=None, run_id=None, model_id=None, dataset=None)` and
  `log_metrics(metrics, step=None, synchronous=None, run_id=None, timestamp=None, model_id=None,
  dataset=None)` are defined in `mlflow.tracking.fluent` and re-exported by name from `mlflow`.
The real packages are exercised in `test_lane_ak1_real_trackers.py`, behind `pytest.importorskip`.
"""

from __future__ import annotations

import importlib.abc
import os
import subprocess
import sys
import types
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import capture
from tensorquill.sink import JsonlSink, MemorySink

SDK = Path(__file__).resolve().parents[1]


# ---- stand-ins -------------------------------------------------------------------------------


class WandbError(Exception):
    """`wandb.Error` / `wandb.errors.UsageError` in the stand-in."""


class _StandInWandb:
    """A `wandb` module, its `Run` class and what the tracker was called with."""

    def __init__(self, *, module_log_delegates: bool = False, run_class: type | None = None) -> None:
        self.calls: list[tuple[object, object, object]] = []
        self.returned = object()
        self.raise_next: BaseException | None = None
        stand_in = self
        module = types.ModuleType("wandb")

        class Run:
            def __init__(self, run_id: str) -> None:
                self.id = run_id
                self.project = "ak1-project"
                self.name = "ak1-name"
                self.group = None
                self._is_finished = False

            @property
            def step(self) -> int:
                raise AssertionError("wandb.run.step must never be read")

            def log(self, data, step=None, commit=None):
                if stand_in.raise_next is not None:
                    error, stand_in.raise_next = stand_in.raise_next, None
                    raise error
                if self._is_finished:
                    raise WandbError(f"Run ({self.id}) is finished. The call to `log` will be ignored.")
                if not isinstance(data, Mapping):
                    raise TypeError("wandb.log must be passed a dictionary")
                stand_in.calls.append((data, step, commit))
                return stand_in.returned

        class NoopRun(Run):
            """`wandb/sdk/lib/noop_run.py`: what `wandb.init(mode="disabled")` makes, a subclass of
            `Run` whose own `log` does not call `Run.log`."""

            def log(self, data, *args, **kwargs):
                stand_in.calls.append((data, "disabled", None))
                return None

        self.Run = run_class if run_class is not None else Run
        self.NoopRun = NoopRun
        self.module = module
        self.module_log_delegates = module_log_delegates
        module.Run = self.Run
        module.Error = WandbError
        module.run = None
        module.log = self._preinit()

        class Image:
            """`wandb.Image`: media, never mirrored."""

        class Histogram:
            pass

        module.Image = Image
        module.Histogram = Histogram
        module.init = self.init
        module.finish = self.finish

    @staticmethod
    def _preinit():
        def preinit_wrapper(*args, **kwargs):
            raise WandbError("You must call wandb.init() before wandb.log()")

        preinit_wrapper.__name__ = "wandb.log"
        return preinit_wrapper

    def init(self, run_id: str = "w-ak1", mode: str = "online"):
        run = (self.NoopRun if mode == "disabled" else self.Run)(run_id)
        self.module.run = run
        if self.module_log_delegates:
            # An older wandb shape: a module function that calls the current run's `log`.
            module = self.module

            def log(data, step=None, commit=None):
                return module.run.log(data, step=step, commit=commit)

            self.module.log = log
        else:
            self.module.log = run.log  # `module.set_global(log=run.log)`
        return run

    def finish(self) -> None:
        run = self.module.run
        if run is not None:
            run._is_finished = True
        self.module.run = None
        self.module.log = self._preinit()  # `module.unset_globals()`

    def values(self) -> list[dict]:
        return [dict(data) for data, _, _ in self.calls]


def wandb_stand_in(put=None, **kwargs) -> _StandInWandb:
    stand_in = _StandInWandb(**kwargs)
    (put or sys.modules.__setitem__)("wandb", stand_in.module)
    return stand_in


class _StandInMlflow:
    def __init__(self, *, metric_via_metrics: bool = False) -> None:
        self.calls: list[tuple] = []
        self.returned = object()
        self.raise_next: BaseException | None = None
        stand_in = self
        mlflow = types.ModuleType("mlflow")
        tracking = types.ModuleType("mlflow.tracking")
        fluent = types.ModuleType("mlflow.tracking.fluent")

        def log_metrics(metrics, step=None, synchronous=None, run_id=None, timestamp=None, model_id=None,
                        dataset=None):
            if stand_in.raise_next is not None:
                error, stand_in.raise_next = stand_in.raise_next, None
                raise error
            stand_in.calls.append(("log_metrics", metrics, step))
            return stand_in.returned

        def log_metric(key, value, step=None, synchronous=None, timestamp=None, run_id=None, model_id=None,
                       dataset=None):
            if metric_via_metrics:
                # A shape where one fluent function calls the other through the module.
                return mlflow.log_metrics({key: value}, step=step)
            if stand_in.raise_next is not None:
                error, stand_in.raise_next = stand_in.raise_next, None
                raise error
            stand_in.calls.append(("log_metric", key, value, step))
            return stand_in.returned

        fluent.log_metric, fluent.log_metrics = log_metric, log_metrics
        mlflow.log_metric, mlflow.log_metrics = log_metric, log_metrics
        mlflow.active_run = lambda: None
        mlflow.tracking = tracking
        tracking.fluent = fluent
        self.module, self.tracking, self.fluent = mlflow, tracking, fluent
        self.log_metric, self.log_metrics = log_metric, log_metrics


def mlflow_stand_in(put=None, **kwargs) -> _StandInMlflow:
    stand_in = _StandInMlflow(**kwargs)
    put = put or sys.modules.__setitem__
    put("mlflow", stand_in.module)
    put("mlflow.tracking", stand_in.tracking)
    put("mlflow.tracking.fluent", stand_in.fluent)
    return stand_in


# ---- helpers ---------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_trackers(monkeypatch):
    for name in ("wandb", "mlflow", "mlflow.tracking", "mlflow.tracking.fluent", "wandb.sdk", "wandb.sdk.wandb_run"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.delenv("TENSORQUILL_ENDPOINT", raising=False)
    yield
    # A test that failed before closing its session leaves no wrapper for the next one.
    mirror = sys.modules.get("tensorquill.mirror")
    for session in list(getattr(mirror, "_sessions", [])):
        session.close()


@pytest.fixture
def wandb_(monkeypatch) -> _StandInWandb:
    return wandb_stand_in(lambda name, module: monkeypatch.setitem(sys.modules, name, module))


@pytest.fixture
def mlflow_(monkeypatch) -> _StandInMlflow:
    return mlflow_stand_in(lambda name, module: monkeypatch.setitem(sys.modules, name, module))


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(4, 1)


def _open(model: nn.Module, **kwargs) -> tuple[tensorquill.Session, MemorySink, torch.optim.Optimizer]:
    sink = MemorySink()
    options = dict(sink=sink, run_id="ak1", layers=(), grads=False, weights=False, heartbeat_seconds=None,
                   stack_seconds=None)
    options.update(kwargs)
    session = tensorquill.init(model, **options)
    return session, sink, torch.optim.SGD(model.parameters(), lr=0.01)


def _step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    loss = model(torch.ones(2, 4)).sum()
    loss.backward()
    optimizer.step()  # the TensorQuill step ends here (auto_step)
    optimizer.zero_grad()


def _scalars(sink: MemorySink) -> list[dict]:
    return [r.get("scalars", {}) for r in sorted(sink.records, key=lambda r: r["step"])]


# ---- what is mirrored --------------------------------------------------------------------------


def test_wandb_log_after_wandb_init_is_mirrored_once_on_the_current_step(wandb_):
    wandb_.module.init()
    model = _model()
    session, sink, opt = _open(model)
    wandb_.module.log({"loss": 0.5})
    _step(model, opt)
    wandb_.module.log({"loss": 0.25})
    _step(model, opt)
    session.close()
    assert _scalars(sink) == [{"loss": 0.5}, {"loss": 0.25}]
    assert wandb_.values() == [{"loss": 0.5}, {"loss": 0.25}]


def test_wandb_log_when_tensorquill_init_comes_before_wandb_init(wandb_):
    model = _model()
    session, sink, opt = _open(model)
    run = wandb_.module.init()  # rebinds wandb.log to the new run's bound log
    wandb_.module.log({"loss": 1.5})
    run.log({"eval/acc": 0.75})  # Run.log directly, as Lightning's WandbLogger and Accelerate call it
    _step(model, opt)
    session.close()
    assert _scalars(sink) == [{"loss": 1.5, "eval/acc": 0.75}]
    assert len(wandb_.calls) == 2


@pytest.mark.parametrize("tensorquill_first", [True, False])
def test_a_disabled_wandb_run_is_mirrored_in_either_order(wandb_, tensorquill_first):
    model = _model()
    if tensorquill_first:
        session, sink, opt = _open(model)
        run = wandb_.module.init(mode="disabled")  # a `Run` subclass with its own `log`
    else:
        run = wandb_.module.init(mode="disabled")
        session, sink, opt = _open(model)
    wandb_.module.log({"loss": 1.0})
    run.log({"acc": 0.5})
    _step(model, opt)
    session.close()
    assert _scalars(sink) == [{"loss": 1.0, "acc": 0.5}]
    assert [c[1] for c in wandb_.calls] == ["disabled", "disabled"]
    assert "log" in vars(wandb_.NoopRun) and vars(wandb_.NoopRun)["log"].__qualname__.endswith("NoopRun.log")


def test_a_module_log_that_delegates_to_run_log_records_once(monkeypatch):
    stand_in = wandb_stand_in(lambda name, module: monkeypatch.setitem(sys.modules, name, module),
                              module_log_delegates=True)
    stand_in.module.init()
    model = _model()
    session, sink, opt = _open(model)
    stand_in.module.log({"loss": 2.0})
    _step(model, opt)
    stand_in.module.init()  # a second run while the session is open: the module log is rebound again
    stand_in.module.log({"loss": 3.0}, commit=False)
    _step(model, opt)
    session.close()
    assert _scalars(sink) == [{"loss": 2.0}, {"loss": 3.0}]
    assert len(stand_in.calls) == 2


def test_wandb_finish_then_a_new_run_while_the_session_is_open(wandb_):
    wandb_.module.init()
    model = _model()
    session, sink, opt = _open(model)
    wandb_.module.log({"a": 1})
    old_log = wandb_.module.log
    wandb_.module.finish()
    with pytest.raises(WandbError, match="before wandb.log"):
        wandb_.module.log({"a": 9})  # the pre-init stub's own error, unchanged
    with pytest.raises(WandbError, match="is finished"):
        old_log({"a": 9})  # the finished run's error, unchanged
    wandb_.module.init()
    wandb_.module.log({"b": 2})
    _step(model, opt)
    session.close()
    assert _scalars(sink) == [{"a": 1, "b": 2}]


def test_mlflow_log_metric_and_log_metrics_ignore_mlflows_step(mlflow_):
    model = _model()
    session, sink, opt = _open(model)
    mlflow_.module.log_metric("loss", 0.5, step=0)  # MLflow's default step=0 is never trusted
    mlflow_.module.log_metrics({"eval/loss": 0.75, "eval/acc": 0.5}, step=1000)
    _step(model, opt)
    mlflow_.tracking.fluent.log_metric("loss", 0.25)  # the defining module's own name
    mlflow_.module.log_metrics(metrics={"eval/acc": 0.625}, step=0)
    _step(model, opt)
    session.close()
    assert _scalars(sink) == [{"loss": 0.5, "eval/loss": 0.75, "eval/acc": 0.5}, {"loss": 0.25, "eval/acc": 0.625}]
    assert [c[0] for c in mlflow_.calls] == ["log_metric", "log_metrics", "log_metric", "log_metrics"]


def test_mlflow_log_metric_calling_log_metrics_records_once(monkeypatch):
    stand_in = mlflow_stand_in(lambda name, module: monkeypatch.setitem(sys.modules, name, module),
                               metric_via_metrics=True)
    model = _model()
    session, sink, opt = _open(model)
    calls = []
    real_log = session.log

    def counting_log(values=None, **named):
        calls.append(dict(values or {}, **named))
        return real_log(values, **named)

    monkeypatch.setattr(session, "log", counting_log)
    stand_in.module.log_metric("loss", 4.0)
    _step(model, opt)
    session.close()
    assert calls == [{"loss": 4.0}]
    assert _scalars(sink) == [{"loss": 4.0}]


def test_wandb_step_and_commit_arguments_do_not_move_the_tensorquill_step(wandb_):
    wandb_.module.init()
    model = _model()
    session, sink, opt = _open(model)
    wandb_.module.log({"a": 1}, step=500)
    wandb_.module.log({"b": 2}, commit=True)
    _step(model, opt)
    session.close()
    assert [r["step"] for r in sink.records] == [0]
    assert _scalars(sink) == [{"a": 1, "b": 2}]


def test_values_mirrored_and_values_refused(wandb_):
    wandb_.module.init()
    model = _model()
    session, sink, opt = _open(model)
    data = {
        "int": 3,
        "float": 0.5,
        "nan": float("nan"),
        "np_float32": np.float32(0.25),
        "np_int64": np.int64(7),
        "cpu_0d": torch.tensor(1.5),
        "cpu_0d_grad": torch.tensor(2.5, requires_grad=True) * 1,
        "cpu_0d_int": torch.tensor(4),
        # refused:
        "bool": True,
        "np_bool": np.bool_(True),
        "cpu_0d_bool": torch.tensor(True),
        "cpu_0d_complex": torch.tensor(1 + 2j),
        "cpu_1elem_1d": torch.tensor([1.0]),
        "np_0d_array": np.array(1.0),
        "complex": 1 + 2j,
        "str": "0.5",
        "list": [1, 2],
        "nested": {"loss": 1.0},
        "none": None,
        "image": wandb_.module.Image(),
        "histogram": wandb_.module.Histogram(),
    }
    wandb_.module.log(data)
    _step(model, opt)
    session.close()
    (scalars,) = _scalars(sink)
    assert set(scalars) == {"int", "float", "nan", "np_float32", "np_int64", "cpu_0d", "cpu_0d_grad", "cpu_0d_int"}
    assert scalars["int"] == 3 and scalars["np_int64"] == 7 and scalars["cpu_0d_int"] == 4
    assert scalars["float"] == 0.5 and scalars["np_float32"] == 0.25
    assert scalars["cpu_0d"] == 1.5 and scalars["cpu_0d_grad"] == 2.5
    assert scalars["nan"] == "nan" or scalars["nan"] != scalars["nan"]
    assert wandb_.calls[0][0] is data  # the tracker got the caller's own dict


def _non_cpu_scalar() -> torch.Tensor:
    if torch.backends.mps.is_available():
        return torch.tensor(1.0, device="mps")
    return torch.empty((), device="meta")  # a stand-in: not on the CPU, and `.item()` cannot read it


def test_a_tensor_not_on_the_cpu_is_skipped_without_reading_it(wandb_, monkeypatch):
    wandb_.module.init()
    value = _non_cpu_scalar()
    read = []
    for name in ("item", "tolist", "cpu", "numpy", "__float__", "__int__", "__bool__"):
        original = getattr(torch.Tensor, name)

        def guarded(self, *args, _original=original, _name=name, **kwargs):
            if self.device.type != "cpu":
                read.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, guarded)
    original_to = torch.Tensor.to

    def guarded_to(self, *args, **kwargs):
        if self.device.type != "cpu":
            read.append("to")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", guarded_to)
    model = _model()
    session, sink, opt = _open(model)
    wandb_.module.log({"gpu_loss": value, "loss": 1.0})
    _step(model, opt)
    session.close()
    assert read == []
    assert _scalars(sink) == [{"loss": 1.0}]
    assert wandb_.calls[0][0]["gpu_loss"] is value


def test_several_calls_in_one_step_accumulate_and_the_last_value_of_a_key_wins(wandb_, mlflow_):
    wandb_.module.init()
    model = _model()
    session, sink, opt = _open(model)
    wandb_.module.log({"a": 1}, commit=False)
    wandb_.module.log({"b": 2}, commit=False)
    mlflow_.module.log_metric("c", 3.0)
    wandb_.module.log({"a": 10})
    _step(model, opt)
    wandb_.module.log({"d": 4})
    _step(model, opt)
    session.close()
    assert _scalars(sink) == [{"a": 10, "b": 2, "c": 3.0}, {"d": 4}]


# ---- explicit session.log() wins ---------------------------------------------------------------


@pytest.mark.parametrize("order", ["explicit_then_mirror", "mirror_then_explicit", "mirror_explicit_mirror"])
def test_an_explicit_session_log_wins_over_a_mirrored_key_on_the_same_step(wandb_, order):
    wandb_.module.init()
    model = _model()
    session, sink, opt = _open(model)
    if order == "explicit_then_mirror":
        session.log({"loss": 1.0})
        wandb_.module.log({"loss": 9.0, "other": 3.0})
    elif order == "mirror_then_explicit":
        wandb_.module.log({"loss": 9.0, "other": 3.0})
        session.log({"loss": 1.0})
    else:
        wandb_.module.log({"loss": 9.0, "other": 3.0})
        session.log({"loss": 1.0})
        wandb_.module.log({"loss": 8.0, "other": 4.0})
    _step(model, opt)
    wandb_.module.log({"loss": 7.0})  # the next step: the mirror writes the key again
    _step(model, opt)
    session.close()
    other = 4.0 if order == "mirror_explicit_mirror" else 3.0
    assert _scalars(sink) == [{"loss": 1.0, "other": other}, {"loss": 7.0}]


def test_an_adapters_explicit_loss_wins_over_the_mirrored_one(wandb_):
    """The Accelerate adapter logs its loss with `session.log` before the step ends; a tracker
    logging the same key in the same step (Accelerate's own `WandBTracker` calls `run.log`) does
    not replace it."""
    run = wandb_.module.init()
    model = _model()
    session, sink, opt = _open(model)
    loss = model(torch.ones(2, 4)).sum()
    loss.backward()
    session.log({"loss": loss.detach()})  # what the adapters do
    run.log({"loss": 123.0, "lr": 0.01})
    opt.step()
    session.close()
    (scalars,) = _scalars(sink)
    assert scalars["lr"] == 0.01 and scalars["loss"] != 123.0


# ---- the tracker's call is unchanged -------------------------------------------------------------


def test_the_trackers_call_gets_the_same_arguments_and_returns_the_same_value(wandb_, mlflow_):
    run = wandb_.module.init()
    model = _model()
    session, sink, opt = _open(model)
    data = {"loss": 1.0}
    assert wandb_.module.log(data, step=3, commit=False) is wandb_.returned
    assert run.log(data, 4, True) is wandb_.returned
    assert wandb_.calls == [(data, 3, False), (data, 4, True)]
    assert wandb_.calls[0][0] is data and wandb_.calls[1][0] is data
    assert mlflow_.module.log_metric("k", 1.0, 5, synchronous=False) is mlflow_.returned
    metrics = {"m": 2.0}
    assert mlflow_.module.log_metrics(metrics, step=6) is mlflow_.returned
    assert mlflow_.calls == [("log_metric", "k", 1.0, 5), ("log_metrics", metrics, 6)]
    assert mlflow_.calls[1][1] is metrics
    session.close()


def test_the_trackers_own_exception_reaches_the_caller_unchanged(wandb_, mlflow_):
    run = wandb_.module.init()
    model = _model()
    session, sink, opt = _open(model)
    error = WandbError("the tracker's own")
    wandb_.raise_next = error
    with pytest.raises(WandbError) as caught:
        wandb_.module.log({"loss": 1.0})
    assert caught.value is error
    with pytest.raises(TypeError, match="must be passed a dictionary"):
        run.log([("loss", 1.0)])
    error2 = KeyboardInterrupt()
    mlflow_.raise_next = error2
    with pytest.raises(KeyboardInterrupt) as caught2:
        mlflow_.module.log_metric("loss", 1.0)
    assert caught2.value is error2
    _step(model, opt)
    session.close()
    assert _scalars(sink) == [{}]  # a call the tracker refused records nothing


def test_a_call_outside_any_session_passes_through_and_records_nothing(wandb_):
    wandb_.module.init()
    model = _model()
    session, sink, opt = _open(model)
    session.close()
    assert wandb_.module.log({"x": 1}) is wandb_.returned
    assert len(wandb_.calls) == 1


# ---- fail silent -----------------------------------------------------------------------------


class _BrokenMapping(Mapping):
    """A mapping the tracker reads by key, whose `items()` warns: an error under `-W error`."""

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def items(self):
        warnings.warn("broken items()", RuntimeWarning)
        return self._data.items()


def test_broken_internals_never_raise_under_warnings_as_errors(wandb_, mlflow_, monkeypatch):
    run = wandb_.module.init()
    model = _model()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        session, sink, opt = _open(model)
        assert wandb_.module.log(_BrokenMapping({"loss": 1.0})) is wandb_.returned
        # The session's side fails too: a log that warns (an error here).
        real_log = session.log

        def warning_log(*args, **kwargs):
            warnings.warn("session.log broke", RuntimeWarning)
            return real_log(*args, **kwargs)

        monkeypatch.setattr(session, "log", warning_log)
        assert run.log({"loss": 2.0}) is wandb_.returned
        assert mlflow_.module.log_metric("loss", 3.0) is mlflow_.returned
        monkeypatch.setattr(session, "log", real_log)
        # A key the session refuses (not a string) is the tracker's business, not ours.
        assert mlflow_.module.log_metrics({1: 2.0, "": 3.0}) is mlflow_.returned
        _step(model, opt)
        session.close()
    assert len(wandb_.calls) == 2 and len(mlflow_.calls) == 2


class _Refusing(type):
    def __setattr__(cls, name, value):
        warnings.warn("no patching here", RuntimeWarning)
        super().__setattr__(name, value)


def test_a_run_class_that_refuses_the_wrapper_leaves_init_and_logging_working(monkeypatch):
    class Run(metaclass=_Refusing):
        def __init__(self, run_id):
            self.id = run_id

        def log(self, data, step=None, commit=None):
            return "ok"

    original = Run.__dict__["log"]
    stand_in = wandb_stand_in(lambda name, module: monkeypatch.setitem(sys.modules, name, module), run_class=Run)
    run = Run("r")
    stand_in.module.run = run
    stand_in.module.log = run.log
    model = _model()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        session, sink, opt = _open(model)
        assert stand_in.module.log({"loss": 1.0}) == "ok"  # the module's name still wrapped
        assert run.log({"acc": 2.0}) == "ok"  # not mirrored: the class refused the wrapper
        _step(model, opt)
        session.close()
    assert Run.__dict__["log"] is original and stand_in.module.log == run.log
    assert _scalars(sink) == [{"loss": 1.0}]


@pytest.mark.parametrize("run_attr", [42, None, "Run", types.SimpleNamespace(log=1)])
def test_a_wandb_whose_run_is_not_a_class_or_whose_log_is_not_callable_is_left_alone(monkeypatch, run_attr):
    stand_in = wandb_stand_in(lambda name, module: monkeypatch.setitem(sys.modules, name, module))
    stand_in.module.Run = run_attr
    stand_in.module.log = 7  # not callable
    model = _model()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        session, sink, opt = _open(model)
        _step(model, opt)
        session.close()
    assert stand_in.module.Run is run_attr and stand_in.module.log == 7


_W_ERROR_SCRIPT = r"""
import sys, warnings
sys.path.insert(0, {tests!r})
import torch
from torch import nn
import test_lane_ak1_mirror as t
import tensorquill
from tensorquill.sink import MemorySink

w = t.wandb_stand_in()
m = t.mlflow_stand_in()

class Refusing(type):
    def __setattr__(cls, name, value):
        warnings.warn("no patching here", RuntimeWarning)
        super().__setattr__(name, value)

class Run(metaclass=Refusing):
    def log(self, data, step=None, commit=None):
        w.calls.append((data, step, commit))
        return "ok"

w.module.Run = Run
run = Run()
w.module.log = run.log
torch.manual_seed(0)
model = nn.Linear(4, 1)
sink = MemorySink()
session = tensorquill.init(model, sink=sink, run_id="ak1", layers=(), grads=False, weights=False,
                           heartbeat_seconds=None, stack_seconds=None)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
for i in range(3):
    assert w.module.log(t._BrokenMapping({{"loss": float(i)}})) == "ok"
    assert w.module.log({{"loss": float(i), "gpu": torch.empty((), device="meta")}}) == "ok"
    m.module.log_metrics({{1: 2.0, "x": object()}})
    m.module.log_metric("lr", 0.01)
    model(torch.ones(2, 4)).sum().backward()
    opt.step()
    opt.zero_grad()
session.close()
print("trained", len(sink.records), len(w.calls), len(m.calls))
print("scalars", [r.get("scalars") for r in sink.records])
"""


def test_the_mirror_under_python_w_error_with_broken_tracker_internals():
    env = {**os.environ, "PYTHONPATH": str(SDK / "src")}
    script = _W_ERROR_SCRIPT.format(tests=str(SDK / "tests"))
    done = subprocess.run([sys.executable, "-W", "error", "-W", "ignore:Failed to initialize NumPy", "-c", script],
                          capture_output=True, text=True, env=env, timeout=300, cwd=str(SDK))
    assert done.returncode == 0, done.stderr[-3000:]
    assert "trained 3 6 6" in done.stdout
    assert "Traceback" not in done.stderr
    # The module-level `wandb.log` (a bound method) is still wrapped; `Run.log` refused its wrapper.
    assert "'loss': 2.0" in done.stdout and "'lr': 0.01" in done.stdout


# ---- installed only where imported, and taken out again -----------------------------------------


class _ImportSpy(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self.asked: list[str] = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("wandb", "mlflow"):
            self.asked.append(fullname)
        return None


def test_no_tracker_is_imported_by_init_or_close(monkeypatch):
    spy = _ImportSpy()
    monkeypatch.setattr(sys, "meta_path", [spy, *sys.meta_path])
    model = _model()
    session, sink, opt = _open(model)
    _step(model, opt)
    session.close()
    assert spy.asked == []
    assert "wandb" not in sys.modules and "mlflow" not in sys.modules


def test_no_tracker_imported_in_a_fresh_interpreter():
    script = (
        "import sys, torch, tensorquill\n"
        "from tensorquill.sink import MemorySink\n"
        "m = torch.nn.Linear(2, 1)\n"
        "s = tensorquill.init(m, sink=MemorySink(), run_id='ak1', heartbeat_seconds=None, stack_seconds=None)\n"
        "s.close()\n"
        "print(sorted(k for k in sys.modules if k.split('.')[0] in ('wandb', 'mlflow')))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SDK / "src")}
    env.pop("TENSORQUILL_ENDPOINT", None)
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=300,
                          cwd=str(SDK))
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "[]"


def _originals(wandb_: _StandInWandb, mlflow_: _StandInMlflow) -> dict:
    return {
        "Run.log": wandb_.Run.__dict__["log"],
        "NoopRun.log": wandb_.NoopRun.__dict__["log"],
        "wandb.log": wandb_.module.log,
        "mlflow.log_metric": mlflow_.module.log_metric,
        "mlflow.log_metrics": mlflow_.module.log_metrics,
        "fluent.log_metric": mlflow_.fluent.log_metric,
        "fluent.log_metrics": mlflow_.fluent.log_metrics,
    }


def _now(wandb_: _StandInWandb, mlflow_: _StandInMlflow) -> dict:
    return _originals(wandb_, mlflow_)


def test_mirror_scalars_false_installs_nothing(wandb_, mlflow_):
    wandb_.module.init()
    before = _originals(wandb_, mlflow_)
    model = _model()
    session, sink, opt = _open(model, mirror_scalars=False)
    assert _now(wandb_, mlflow_) == before
    wandb_.module.log({"loss": 1.0})
    mlflow_.module.log_metric("x", 1.0)
    _step(model, opt)
    session.close()
    assert _scalars(sink) == [{}]


def test_mirror_scalars_must_be_a_bool_and_a_refused_init_changes_nothing(wandb_, mlflow_):
    wandb_.module.init()
    before = _originals(wandb_, mlflow_)
    with pytest.raises(ValueError, match="mirror_scalars"):
        _open(_model(), mirror_scalars="yes")
    assert _now(wandb_, mlflow_) == before


def test_init_installs_and_close_restores_every_name(wandb_, mlflow_):
    wandb_.module.init()
    before = _originals(wandb_, mlflow_)
    model = _model()
    session, sink, opt = _open(model)
    now = _now(wandb_, mlflow_)
    assert all(now[name] is not before[name] for name in before)
    session.close()
    after = _now(wandb_, mlflow_)
    assert all(after[name] is before[name] for name in before)


def test_two_overlapping_sessions_share_the_wrappers_until_the_last_closes(wandb_, mlflow_):
    wandb_.module.init()
    before = _originals(wandb_, mlflow_)
    model_a, model_b = _model(), _model()
    a, sink_a, opt_a = _open(model_a)
    b, sink_b, opt_b = _open(model_b)
    installed = _now(wandb_, mlflow_)
    wandb_.module.log({"loss": 1.0})
    _step(model_a, opt_a)
    _step(model_b, opt_b)
    a.close()
    assert _now(wandb_, mlflow_) == installed  # b is still open
    wandb_.module.log({"loss": 2.0})
    mlflow_.module.log_metric("lr", 0.5)
    _step(model_b, opt_b)
    b.close()
    assert _now(wandb_, mlflow_) == before
    assert _scalars(sink_a) == [{"loss": 1.0}]
    assert _scalars(sink_b) == [{"loss": 1.0}, {"loss": 2.0, "lr": 0.5}]
    assert len(wandb_.calls) == 2


def test_a_tracker_imported_between_two_sessions_is_wrapped_by_the_second(monkeypatch):
    model_a, model_b = _model(), _model()
    a, sink_a, opt_a = _open(model_a)
    stand_in = wandb_stand_in(lambda name, module: monkeypatch.setitem(sys.modules, name, module))
    stand_in.module.init()
    original_run_log, original_log = stand_in.Run.__dict__["log"], stand_in.module.log
    b, sink_b, opt_b = _open(model_b)
    stand_in.module.log({"loss": 1.0})
    _step(model_a, opt_a)
    _step(model_b, opt_b)
    b.close()
    a.close()
    assert stand_in.Run.__dict__["log"] is original_run_log and stand_in.module.log == original_log
    assert _scalars(sink_b) == [{"loss": 1.0}]
    assert len(stand_in.calls) == 1


def _user_wrappers(wandb_: _StandInWandb, mlflow_: _StandInMlflow) -> list:
    """What a user (or another library) installs on top of TensorQuill's wrappers after `init`."""
    seen: list = []
    run_log = wandb_.Run.__dict__["log"]

    def user_run_log(self, *args, **kwargs):
        seen.append("Run.log")
        return run_log(self, *args, **kwargs)

    module_log = wandb_.module.log

    def user_log(*args, **kwargs):
        seen.append("wandb.log")
        return module_log(*args, **kwargs)

    log_metric = mlflow_.module.log_metric

    def user_log_metric(*args, **kwargs):
        seen.append("mlflow.log_metric")
        return log_metric(*args, **kwargs)

    wandb_.Run.log = user_run_log
    wandb_.module.log = user_log
    mlflow_.module.log_metric = user_log_metric
    return seen


@pytest.mark.parametrize("how", ["close", "stop"])
def test_close_and_stop_take_the_mirror_out_and_leave_a_later_user_wrapper(wandb_, mlflow_, how):
    run = wandb_.module.init()
    before = _originals(wandb_, mlflow_)
    model = _model()
    session, sink, opt = _open(model)
    wandb_.module.log({"loss": 1.0})
    seen = _user_wrappers(wandb_, mlflow_)
    users = _now(wandb_, mlflow_)
    if how == "close":
        session.close()
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session._stop(RuntimeError("the sink failed"), reason="sink_error")
    after = _now(wandb_, mlflow_)
    # The user's wrappers stay where they are; the names nobody wrapped again are the originals.
    for name in ("Run.log", "wandb.log", "mlflow.log_metric"):
        assert after[name] is users[name]
    for name in ("mlflow.log_metrics", "fluent.log_metric", "fluent.log_metrics"):
        assert after[name] is before[name]
    # Calls go through the user's wrapper and ours under it, to the tracker, once each, unrecorded.
    assert wandb_.module.log({"loss": 2.0}) is wandb_.returned
    assert run.log({"loss": 3.0}) is wandb_.returned
    assert mlflow_.module.log_metric("lr", 0.1) is mlflow_.returned
    assert seen == ["wandb.log", "Run.log", "mlflow.log_metric"]
    assert len(wandb_.calls) == 3 and len(mlflow_.calls) == 1
    if how == "stop":
        session.close()
    assert all(r.get("scalars", {}).get("loss") not in (2.0, 3.0) for r in sink.records)


def test_a_new_session_after_a_user_wrapper_records_each_call_once(wandb_):
    run = wandb_.module.init()
    model = _model()
    first, _, _ = _open(model)
    seen = _user_wrappers(wandb_, mlflow_stand_in(lambda n, m: None))
    first.close()
    model2 = _model()
    second, sink, opt = _open(model2)  # wraps the user's wrapper, which calls ours under it
    wandb_.module.log({"loss": 1.0})
    run.log({"acc": 0.5})
    _step(model2, opt)
    second.close()
    assert _scalars(sink) == [{"loss": 1.0, "acc": 0.5}]
    assert seen == ["wandb.log", "Run.log"]


def test_stop_is_called_when_the_sink_fails_and_the_mirror_comes_out(wandb_, mlflow_):
    wandb_.module.init()
    before = _originals(wandb_, mlflow_)

    class Failing(MemorySink):
        def write(self, records):
            raise OSError("disk full")

    model = _model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session, sink, opt = _open(model, sink=Failing(), every=1)
        for _ in range(4):
            wandb_.module.log({"loss": 1.0})
            _step(model, opt)
        session.flush()
        for _ in range(50):
            if session._stopped:
                break
            _step(model, opt)
            import time

            time.sleep(0.01)
    assert session._stopped
    assert _now(wandb_, mlflow_) == before
    session.close()


def test_a_failure_later_in_init_rolls_the_mirror_back(wandb_, mlflow_, monkeypatch, tmp_path):
    wandb_.module.init()
    before = _originals(wandb_, mlflow_)

    def fail(path):
        raise OSError("cannot clear the end marker")

    monkeypatch.setattr(capture, "clear_end_marker", fail)
    with pytest.raises(OSError, match="end marker"):
        tensorquill.init(_model(), sink=JsonlSink(tmp_path / "r.jsonl"), run_id="ak1", layers=(),
                         heartbeat_seconds=None, stack_seconds=None)
    assert _now(wandb_, mlflow_) == before
    assert wandb_.module.log({"loss": 1.0}) is wandb_.returned


# ---- a run with no tracker is as before ----------------------------------------------------------


def _frozen_run(folder: Path, **kwargs) -> bytes:
    import time as real_time

    clock = types.SimpleNamespace(**{k: getattr(real_time, k) for k in dir(real_time) if not k.startswith("__")})
    clock.time = lambda: 1_790_000_000.0
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(capture, "time", clock)
        patch.setattr(capture, "_next_attempt", lambda: 1_790_000_000_000)
        return _a_run(folder, **kwargs)


def _a_run(folder: Path, **kwargs) -> bytes:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "run.jsonl"
    session = tensorquill.init(model, sink=JsonlSink(path), run_id="ak1", heartbeat_seconds=None, stack_seconds=None,
                               every=3, **kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(7):
        loss = model(torch.full((2, 4), float(i))).pow(2).mean()
        loss.backward()
        session.log({"loss": loss.detach(), "i": i})
        opt.step()
        opt.zero_grad()
    session.close()
    return path.read_bytes()


def test_a_run_with_no_tracker_imported_writes_the_same_bytes_with_the_mirror_on_or_off(tmp_path):
    on = _frozen_run(tmp_path / "on")  # the default, mirror_scalars=True
    off = _frozen_run(tmp_path / "off", mirror_scalars=False)
    assert on.count(b"\n") == 7 and b'"loss"' in on
    assert on == off


# ---- the README ----------------------------------------------------------------------------------


def test_the_readme_describes_the_mirror_and_the_order_it_needs():
    readme = " ".join((SDK / "README.md").read_text(encoding="utf-8").split())
    assert "## Scalars mirrored from W&B and MLflow" in (SDK / "README.md").read_text(encoding="utf-8")
    for phrase in ("mirror_scalars=False", "import wandb", "before `tensorquill.init", "from mlflow import log_metric",
                   "wandb.run.step", "MlflowClient"):
        assert phrase in readme, phrase
