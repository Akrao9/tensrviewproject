"""Wave AO, lane AO-8: the helpers (`datawait.py`, `mirror.py`, `collectives.py`, `data.py`).

Each section holds the test that showed one finding on the candidate and the tests that pin what
the fix must leave as it was. Optional packages (NumPy, wandb, mlflow, PyTorch's Flight Recorder
analyzer) are asked for test by test, so the file runs on an interpreter with PyTorch alone.
"""

from __future__ import annotations

import time

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import tensorquill
from tensorquill import capture, datawait
from tensorquill.sink import MemorySink


def _model():
    torch.manual_seed(0)
    return nn.Linear(1, 1)


def _open(model, **kwargs):
    sink = MemorySink()
    options = dict(sink=sink, run_id="ao8", layers=(), grads=False, weights=False, heartbeat_seconds=None,
                   stack_seconds=None)
    options.update(kwargs)
    return tensorquill.init(model, **options), sink


# ---- BUG-17: a DataLoader fetched inside another DataLoader's fetch ---------------------------------


@pytest.fixture(autouse=True)
def _no_session_left_watching():
    yield
    for session in list(datawait._watching):
        datawait.unwatch(session)


class _Clock:
    """`time` for `datawait` and `capture`, with `perf_counter` moved by hand; the rest is `time`'s."""

    def __init__(self) -> None:
        self.now = 1000.0

    def perf_counter(self) -> float:
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


@pytest.fixture
def clock(monkeypatch):
    stand_in = _Clock()
    monkeypatch.setattr(datawait, "time", stand_in)
    monkeypatch.setattr(capture, "time", stand_in)
    return stand_in


def _charging(clock, seconds):
    """A collate function that spends `seconds` on the stand-in clock and hands its batch over."""

    def collate(batch):
        clock.now += seconds
        return batch

    return collate


def _run_steps(session, clock, fetch, steps, *, train=4.0):
    """`steps` steps that each fetch once (`fetch()`) and then train for `train` seconds."""
    session.step()  # the clock starts at the first step's end
    for _ in range(steps):
        fetch()
        clock.now += train
        session.step()


def _waits(sink):
    return [(r["data_wait"], r["step_time"]) for r in sink.records if r.get("step_time") is not None]


def test_a_loader_fetched_inside_another_loaders_fetch_is_counted_once(clock):
    inner = iter(DataLoader(range(64), collate_fn=_charging(clock, 1.0)))
    outer = iter(DataLoader(range(64), collate_fn=lambda _: next(inner)))
    session, sink = _open(_model(), auto_step=False, data_wait=True)
    with session:
        _run_steps(session, clock, lambda: next(outer), 5)
    assert _waits(sink) == [(1.0, 5.0)] * 5


def test_nested_loaders_with_a_fifth_of_the_step_waiting_raise_no_starvation(clock):
    """The two runs spend the same simulated time: one second in the loader, four training."""
    verdicts, shares = {}, {}
    for nested in (False, True):
        inner = iter(DataLoader(range(64), collate_fn=_charging(clock, 1.0)))
        loader = iter(DataLoader(range(64), collate_fn=lambda _: next(inner))) if nested else inner
        session, sink = _open(_model(), auto_step=False, data_wait=True)
        with session:
            _run_steps(session, clock, lambda: next(loader), 24)
        shares[nested] = sorted({wait / took for wait, took in _waits(sink)})
        incidents = tensorquill.build_evidence(sink.records, detectors="python")["incidents"]
        verdicts[nested] = [i["kind"] for i in incidents if i["kind"] == "input_starvation"]
    assert verdicts == {False: [], True: []}
    assert shares == {False: [0.2], True: [0.2]}


def test_a_time_data_iterator_fetched_inside_a_loaders_fetch_is_counted_once(clock):
    session, sink = _open(_model(), auto_step=False, data_wait=True)
    with session:
        inner = iter(session.time_data(DataLoader(range(64), collate_fn=_charging(clock, 1.0))))
        outer = iter(DataLoader(range(64), collate_fn=lambda _: next(inner)))
        _run_steps(session, clock, lambda: next(outer), 3)
    assert _waits(sink) == [(1.0, 5.0)] * 3


class _Sleeping(Dataset):
    def __init__(self, seconds):
        self.seconds = seconds

    def __len__(self):
        return 64

    def __getitem__(self, i):
        time.sleep(self.seconds)
        return torch.ones(1)


def test_on_the_wall_clock_nested_loaders_never_wait_longer_than_the_step():
    inner = iter(DataLoader(_Sleeping(0.01), batch_size=None))
    outer = iter(DataLoader(range(64), batch_size=None, collate_fn=lambda _: next(inner)))
    model = _model()
    session, sink = _open(model, auto_step=False, data_wait=True)
    with session:
        session.step()
        for _ in range(6):
            model(next(outer).reshape(1, 1)).sum().backward()
            session.step()
    waits = _waits(sink)
    assert len(waits) == 6 and all(0 < wait <= took for wait, took in waits)


# Pinned beside the fix: what counted before counts as before.


def test_the_flag_is_down_after_an_epoch_ends_and_the_next_epoch_counts(clock):
    loader = DataLoader(range(2), collate_fn=_charging(clock, 1.0))
    session, sink = _open(_model(), auto_step=False, data_wait=True)
    with session:
        session.step()
        for _epoch in range(2):
            for _batch in loader:  # each epoch's last `next()` raises StopIteration inside the timer
                clock.now += 4.0
                session.step()
            assert session._fetching is False
    assert _waits(sink) == [(1.0, 5.0)] * 4


def test_the_flag_is_down_after_a_fetch_that_raises_and_the_next_fetch_counts(clock):
    calls = []

    def collate(batch):
        calls.append(batch)
        clock.now += 1.0
        if len(calls) == 1:
            raise RuntimeError("a bad batch")
        return batch

    loader = iter(DataLoader(range(8), collate_fn=collate))
    session, sink = _open(_model(), auto_step=False, data_wait=True)
    with session:
        session.step()
        with pytest.raises(RuntimeError, match="a bad batch"):
            next(loader)
        assert session._fetching is False
        next(loader)
        clock.now += 4.0
        session.step()
    assert _waits(sink) == [(2.0, 6.0)]


def test_two_loaders_one_after_the_other_in_a_step_each_count(clock):
    first = iter(DataLoader(range(8), collate_fn=_charging(clock, 1.0)))
    second = iter(DataLoader(range(8), collate_fn=_charging(clock, 2.0)))
    session, sink = _open(_model(), auto_step=False, data_wait=True)
    with session:
        _run_steps(session, clock, lambda: (next(first), next(second)), 2)
    assert _waits(sink) == [(3.0, 7.0)] * 2


def test_time_data_around_a_loader_counts_once(clock):
    session, sink = _open(_model(), auto_step=False, data_wait=True)
    with session:
        loader = iter(session.time_data(DataLoader(range(8), collate_fn=_charging(clock, 1.0))))
        _run_steps(session, clock, lambda: next(loader), 2)
    assert _waits(sink) == [(1.0, 5.0)] * 2


def test_two_sessions_on_one_thread_each_count_a_nested_fetch_once(clock):
    inner = iter(DataLoader(range(8), collate_fn=_charging(clock, 1.0)))
    outer = iter(DataLoader(range(8), collate_fn=lambda _: next(inner)))
    first, first_sink = _open(_model(), auto_step=False, data_wait=True, run_id="ao8-a")
    second, second_sink = _open(_model(), auto_step=False, data_wait=True, run_id="ao8-b")
    with first, second:
        first.step()
        second.step()
        for _ in range(2):
            next(outer)
            clock.now += 4.0
            first.step()
            second.step()
    assert _waits(first_sink) == _waits(second_sink) == [(1.0, 5.0)] * 2


def test_a_fetch_under_the_eval_phase_is_not_counted_and_leaves_the_flag_down(clock):
    inner = iter(DataLoader(range(8), collate_fn=_charging(clock, 1.0)))
    outer = iter(DataLoader(range(8), collate_fn=lambda _: next(inner)))
    session, sink = _open(_model(), auto_step=False, data_wait=True)
    with session:
        session.step()
        with session.phase("eval"):
            next(outer)
            assert session._fetching is False
        next(outer)
        clock.now += 4.0
        session.step()
    assert _waits(sink) == [(1.0, 5.0)]
