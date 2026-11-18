"""Which record each piece of a step lands on: a checkpoint saved right after a full ring
(SDK-07), a step the loss scaler skipped (SDK-03), and an optimizer that steps after the one
that ended the step (SDK-06)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill.data_checks import detect_resume_state


def _steps(model: nn.Module, optimizer: torch.optim.Optimizer, n: int) -> None:
    for _ in range(n):
        optimizer.zero_grad()
        model(torch.randn(8, 4)).pow(2).mean().backward()
        optimizer.step()


def _written(session: tensorquill.Session) -> None:
    """Wait for the writer thread to write everything queued so far (it stops at the None)."""
    session._queue.put(None)
    session._worker.join()


# --- SDK-07: a save right after the ring was full -------------------------------------------


@pytest.mark.parametrize("every, steps", [(50, 50), (4, 4), (4, 8), (4, 6), (1, 3)])
def test_a_save_after_the_last_step_is_on_that_steps_record_when_the_session_closes(every, steps):
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=every)
    _steps(model, optimizer, steps)
    session.checkpoint({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, event="save")
    session.close()
    assert [r["step"] for r in sink.records] == list(range(steps))  # no step invented or written twice
    saves = [(r["step"], r["checkpoint"]["save"]["after_step"]) for r in sink.records if "checkpoint" in r]
    assert saves == [(steps - 1, steps - 1)]


def test_a_full_ring_is_written_when_the_next_step_ends():
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=4)
    _steps(model, optimizer, 4)
    # The fourth step stays in the ring until the fifth ends, so what arrives for it after its
    # step ended (a checkpoint saved then) is still written on its record.
    assert session._steps == [0, 1, 2, 3]
    _steps(model, optimizer, 1)
    assert session._steps == [4]
    _written(session)
    assert [r["step"] for r in sink.records] == [0, 1, 2, 3]


def test_an_explicit_flush_still_writes_every_step():
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=4)
    _steps(model, optimizer, 4)
    session.flush()
    _written(session)
    assert [r["step"] for r in sink.records] == [0, 1, 2, 3]


def test_a_resume_from_a_checkpoint_saved_at_a_multiple_of_every_is_compared_with_that_save():
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = tensorquill.MemorySink()
    with tensorquill.init(model, sink=first, run_id="resume") as session:  # every=50, the default
        _steps(model, optimizer, 50)
        # No rng and no data position in it: the resume check names both.
        saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        session.checkpoint(saved, event="save")  # as Trainer.on_save does, after step 50
    resumed_model = nn.Linear(4, 1)
    resumed_model.load_state_dict(saved["model"])
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    second = tensorquill.MemorySink()
    # No load manifest: the resume is judged against the save the earlier attempt recorded.
    with tensorquill.init(resumed_model, sink=second, run_id="resume", start_step=50):
        _steps(resumed_model, resumed_optimizer, 3)
    records = first.records + second.records
    assert first.records[0]["attempt"] != second.records[0]["attempt"]
    _, incomplete = detect_resume_state(records)
    assert incomplete is not None
    change = incomplete["changes"][0]
    assert (change["step"], change["source"], change["saved_after_step"]) == (50, "save", 49)
    assert {f["component"] for f in change["findings"] if f["finding"] == "missing"} == {"rng", "data"}
