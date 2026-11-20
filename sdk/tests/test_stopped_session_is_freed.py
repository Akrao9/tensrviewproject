"""A session that stopped on a writer error is freed when its caller lets go of it, like any other.

Found on Nov 20, 2026, while looking into an abort at process exit in CI (`73eac65`; that abort
is not explained and this is not shown to be its cause). The error a stopped session keeps as
`writer_error` held its traceback, whose frame of the writer thread held the session: a reference
cycle. The session then outlived its last reference until a garbage collection, and with it the
`replica_group` it was given, after the job had called `destroy_process_group`. The error is now
kept without its traceback, as `flight.py` keeps a sender's, and a closed session lets go of its
process group.
"""

from __future__ import annotations

import gc
import warnings
import weakref

import pytest
import torch
from torch import nn

import tensorquill

from test_lane_u1_capture import FailingJsonl, FailingSink


def _train(session_of, steps: int = 8):
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = session_of(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(steps):
            model(torch.ones(2, 4)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        session.close()
    return session


@pytest.fixture
def no_collector():
    """With the collector off, only reference counts free an object: a cycle keeps it."""
    was = gc.isenabled()
    gc.collect()
    gc.disable()
    yield
    if was:
        gc.enable()


@pytest.mark.parametrize("sink", ["jsonl", "memory"])
def test_a_session_stopped_by_its_writer_is_freed_without_a_collection(tmp_path, no_collector, sink):
    def stopped(model):
        made = FailingJsonl(tmp_path / "run.jsonl", good=1) if sink == "jsonl" else FailingSink()
        return tensorquill.init(model, run_id="r1", sink=made, every=1)

    session = _train(stopped)
    assert session._stopped
    error = session.writer_error
    assert isinstance(error, OSError) and str(error) == "disk gone"
    assert error.__traceback__ is None  # fail-before: the writer thread's frames, which hold the session
    gone = weakref.ref(session)
    del session
    assert gone() is None  # fail-before: alive until a collection


def test_a_session_that_closed_normally_is_freed_as_before(tmp_path, no_collector):
    session = _train(lambda model: tensorquill.init(model, run_id="r1", sink=tensorquill.JsonlSink(tmp_path / "run.jsonl"),
                                                    every=1))
    assert not session._stopped and session.writer_error is None
    gone = weakref.ref(session)
    del session
    assert gone() is None


@pytest.mark.parametrize("failing", [False, True])
def test_a_closed_session_lets_go_of_its_process_group(tmp_path, failing):
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available():
        pytest.skip("needs torch.distributed")
    from datetime import timedelta

    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.FileStore(str(tmp_path / "store"), 1),
                            timeout=timedelta(seconds=60))
    try:
        group = dist.group.WORLD

        def with_group(model):
            sink = FailingJsonl(tmp_path / "run.jsonl", good=1) if failing else tensorquill.JsonlSink(tmp_path / "run.jsonl")
            return tensorquill.init(model, run_id="r1", sink=sink, every=1, replica_group=group, replica_every=2)

        session = _train(with_group)
        assert session._stopped is failing
        assert session._replica_group is None  # fail-before: still the group
        session.record_replica_spread()  # a closed session's call still does nothing and raises nothing
    finally:
        dist.destroy_process_group()
