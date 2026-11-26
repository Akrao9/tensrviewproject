"""Wave AO, lane AO-1: the capture's findings on `453f1e2`.

Each finding's tests were committed failing before its fix. They run on the bare interpreters as
CI does (PyTorch 2.6 and the newest release): nothing here needs a package beyond PyTorch, and the
tests that need Apple's MPS skip where it is not there.
"""

from __future__ import annotations

import numbers
import warnings

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import capture, identity
from tensorquill.evidence import load_records
from tensorquill.sink import MemorySink

# ---- SIG-26: the version counter's calling form ---------------------------------------------------
#
# PyTorch 2.6 takes `_unsafe_set_version_counter(tensor, version)`, one tensor at a time; later
# releases take sequences. A call in the wrong form raises a `TypeError` whose text prints each
# tensor (`Tensor.__repr__` runs inside the C call), and an exception raised while it prints, such as
# a job's SIGTERM handler raising `SystemExit`, is lost there. So no step may make a call that fails.


def _counting_repr(monkeypatch):
    calls = []
    original = torch.Tensor.__repr__

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "__repr__", spy)
    return calls


def _recording_setter(monkeypatch):
    """Wrap the private setter: each call's arguments' kind and whether it raised."""
    original = torch._C._autograd._unsafe_set_version_counter
    calls = []

    def setter(*args):
        try:
            original(*args)
        except BaseException as error:
            calls.append((type(args[0]).__name__, type(error).__name__))
            raise
        calls.append((type(args[0]).__name__, None))

    monkeypatch.setattr(torch._C._autograd, "_unsafe_set_version_counter", setter)
    return calls


def test_restoring_a_version_prints_no_tensor(monkeypatch):
    tensor = torch.zeros(3)
    tensor.add_(1)
    assert tensor._version == 1
    calls = _counting_repr(monkeypatch)
    capture._restore_versions([tensor], [0])
    assert calls == []
    assert tensor._version == 0


def test_restoring_versions_makes_no_call_that_fails(monkeypatch):
    tensors = [torch.zeros(2), torch.ones(4)]
    for tensor in tensors:
        tensor.add_(1)
        tensor.add_(1)
    capture._restore_versions(tensors, [0, 1])  # the first call may choose the form; never by failing
    calls = _recording_setter(monkeypatch)
    for _ in range(3):
        for tensor in tensors:
            tensor.add_(1)
        capture._restore_versions(tensors, [0, 1])
    assert [t._version for t in tensors] == [0, 1]
    assert calls and all(error is None for _, error in calls)


def test_an_exception_raised_while_a_step_restores_versions_comes_out(monkeypatch):
    """The SIGTERM case made deterministic: whatever runs Python code during the restore (here
    `Tensor.__repr__`, as on PyTorch 2.6 before the fix) raising `SystemExit` must not be lost. With
    nothing printed, the restore runs no Python code at all, so the training's own handler is the
    only place such an exit can come from, and it comes out of the step."""
    tensor = torch.zeros(3)
    tensor.add_(1)
    raised = []

    def raising(self, *args, **kwargs):
        raised.append(1)
        raise SystemExit(143)

    monkeypatch.setattr(torch.Tensor, "__repr__", raising)
    try:
        capture._restore_versions([tensor], [0])
        came_out = False
    except SystemExit:
        came_out = True
    assert came_out or not raised, "a SystemExit raised during the restore was lost"
    assert raised == [] and tensor._version == 0


# ---- BUG-05: `start_step` and `every` are integers, as `log`'s `step=` is ------------------------
#
# A bool is an `int` in Python, so `start_step=True` was taken and step 0 was written `"step": true`,
# which every reader refuses. `init` refuses them by `log`'s rule for `step=` (`_explicit_step`): a
# bool, a float or anything not an integer raises `TypeError`, a number below the least `ValueError`,
# each naming the argument, before anything is opened.


def _init_in(tmp_path, monkeypatch, **options):
    monkeypatch.chdir(tmp_path)
    return tensorquill.init(nn.Linear(2, 2), run_id="r", heartbeat_seconds=None, **options)


@pytest.mark.parametrize("value", [True, False, 1.0, 0.0, "1", None])
def test_init_refuses_a_start_step_that_is_not_an_integer(tmp_path, monkeypatch, value):
    with pytest.raises(TypeError, match=r"start_step must be an integer of 0 or more"):
        _init_in(tmp_path, monkeypatch, start_step=value)
    assert list(tmp_path.iterdir()) == []  # nothing opened, nothing written


def test_init_refuses_a_negative_start_step(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match=r"start_step must be an integer of 0 or more, not -1"):
        _init_in(tmp_path, monkeypatch, start_step=-1)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", [True, False, 2.0, "2", None])
def test_init_refuses_an_every_that_is_not_an_integer(tmp_path, monkeypatch, value):
    with pytest.raises(TypeError, match=r"every must be an integer of 1 or more"):
        _init_in(tmp_path, monkeypatch, every=value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", [0, -3])
def test_init_refuses_an_every_below_one(tmp_path, monkeypatch, value):
    with pytest.raises(ValueError, match=r"every must be at least 1"):
        _init_in(tmp_path, monkeypatch, every=value)
    assert list(tmp_path.iterdir()) == []


class _Index:
    """An integer that is not an `int`, as a NumPy integer is: `numbers.Integral`."""

    def __init__(self, value):
        self.value = value

    def __int__(self):
        return self.value

    __index__ = __int__


numbers.Integral.register(_Index)


@pytest.mark.parametrize("start_step", [0, 3, _Index(3)])
def test_a_start_step_that_is_an_integer_is_taken_and_written_as_an_int(start_step):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = MemorySink()
    with tensorquill.init(model, sink=sink, every=_Index(2), start_step=start_step, heartbeat_seconds=None):
        for _ in range(3):
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
    first = int(start_step)
    assert [r["step"] for r in sink.records] == [first, first + 1, first + 2]
    assert all(type(r["step"]) is int for r in sink.records)
    for record in sink.records:
        identity.check(record)  # every reader takes the record: raises if it does not
    load_records(sink.records)


def test_logs_step_rule_and_its_message_are_unchanged():
    session = tensorquill.init(nn.Linear(2, 2), sink=MemorySink(), heartbeat_seconds=None)
    try:
        with pytest.raises(TypeError, match=r"^session\.log\(\.\.\., step=\.\.\.\): step must be an integer"):
            session.log({"a": 1.0}, step=True)
        with pytest.raises(ValueError, match=r"^session\.log\(\.\.\., step=\.\.\.\): step must be an integer"):
            session.log({"a": 1.0}, step=-1)
    finally:
        session.close()


# ---- BUG-26, the clipping wrapper: a parameter unfrozen before `clip_grad_norm_` ------------------
#
# A parameter frozen at `init` joins the session's trainable ones at the optimizer's step
# (`_take_unfrozen`), after `clip_grad_norm_` ran, so the wrapper did not match it on the step it was
# unfrozen and `clip_grad_norm` was missing there. It counts once it is trainable, as at that step.


def _one_step(model, optimizer, clip, *, inputs=None):
    model(torch.ones(2, 4) if inputs is None else inputs).sum().backward()
    norm = clip()
    optimizer.step()
    optimizer.zero_grad()
    return norm


def _quiet_init(model, sink, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # every parameter frozen at `init` is warned of
        return tensorquill.init(model, sink=sink, every=1, heartbeat_seconds=None, **options)


def test_a_parameter_unfrozen_just_before_clipping_has_its_norm_recorded_on_that_step():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    for p in model.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    sink = MemorySink()
    with _quiet_init(model, sink):
        for p in model.parameters():
            p.requires_grad_(True)
        norm = _one_step(model, optimizer, lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5))
        _one_step(model, optimizer, lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5))
    assert [r["step"] for r in sink.records] == [0, 1]
    assert sink.records[0].get("clip_grad_norm") == pytest.approx(float(norm), rel=1e-6)
    assert "clip_grad_norm" in sink.records[1]


def test_a_lora_like_model_clipping_only_its_newly_unfrozen_part_is_recorded():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    for p in model[1].parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    sink = MemorySink()
    with _quiet_init(model, sink):
        for p in model[1].parameters():
            p.requires_grad_(True)
        norm = _one_step(model, optimizer, lambda: torch.nn.utils.clip_grad_norm_(model[1].parameters(), 0.5))
    assert sink.records[0].get("clip_grad_norm") == pytest.approx(float(norm), rel=1e-6)


def test_a_clip_over_parameters_still_frozen_or_of_another_model_records_nothing():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    for p in model[1].parameters():
        p.requires_grad_(False)
    other = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model[0].parameters(), lr=0.01)
    sink = MemorySink()
    with _quiet_init(model, sink):
        def clip():
            torch.nn.utils.clip_grad_norm_(model[1].parameters(), 0.5)  # frozen still
            other(torch.ones(1, 4)).sum().backward()
            torch.nn.utils.clip_grad_norm_(other.parameters(), 0.5)  # not this model's
        _one_step(model, optimizer, clip)
    assert "clip_grad_norm" not in sink.records[0]
