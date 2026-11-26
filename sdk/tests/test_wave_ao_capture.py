"""Wave AO, lane AO-1: the capture's findings on `453f1e2`.

Each finding's tests were committed failing before its fix. They run on the bare interpreters as
CI does (PyTorch 2.6 and the newest release): nothing here needs a package beyond PyTorch, and the
tests that need Apple's MPS skip where it is not there.
"""

from __future__ import annotations

import torch

from tensorquill import capture

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
