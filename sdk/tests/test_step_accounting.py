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


# --- SDK-03: a step the loss scaler skipped -------------------------------------------------


def _scaled_steps(model, optimizer, scaler, n, *, overflow=(), clip=False, other=None):
    """`n` AMP iterations; on the steps in `overflow` one gradient element of the last layer is
    made Inf after backward, so the scaler skips that step."""
    for step in range(n):
        optimizer.zero_grad()
        scaler.scale(model(torch.randn(8, 4)).pow(2).mean()).backward()
        if step in overflow:
            model[1].weight.grad[0, 0] = float("inf")
        if clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        if other is not None:
            other(step)
        scaler.update()


def _two_layers() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))


def test_a_skipped_step_records_its_gradient_norms_and_names_the_parameter_that_overflowed():
    model = _two_layers()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu")
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, every=2, scaler=scaler, layers=()):
        _scaled_steps(model, optimizer, scaler, 4, overflow=(0, 2))
    records = sink.records
    assert [r["step_skipped"] for r in records] == [True, False, True, False]
    names = {"0.weight", "0.bias", "1.weight", "1.bias"}
    assert all(set(r["grad_norms"]) == names for r in records)
    for r in (records[0], records[2]):
        assert r["grad_norms"]["1.weight"] == float("inf")
        assert all(0 < r["grad_norms"][n] < float("inf") for n in names - {"1.weight"})
    # Inf gradients on a skipped step are the scaler's business: no non-finite incident.
    assert not [i for i in tensorquill.build_evidence(records)["incidents"] if i["kind"] == "non_finite"]


def test_a_skipped_first_step_gives_no_false_reason_for_the_update_size():
    model = _two_layers()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu")
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, every=1, scaler=scaler, updates=True):
        _scaled_steps(model, optimizer, scaler, 4, overflow=(0, 2), clip=True)
    records = sink.records
    assert [r["step_skipped"] for r in records] == [True, False, True, False]
    assert not [r for r in records if "update_not_read" in r]
    assert ["update_rms" in r for r in records] == [False, True, False, True]
    evidence = tensorquill.build_evidence(records)
    assert not [e for e in evidence.get("not_checked", []) if e.get("check") == "update_size"]


def test_an_optimizer_that_holds_none_of_the_parameters_is_still_named_when_the_first_step_was_skipped():
    model = _two_layers()
    copy = nn.Parameter(torch.zeros(5))
    optimizer = torch.optim.AdamW([copy], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, every=1, scaler=scaler, updates=True):
        for step in range(3):
            scaler.scale(model(torch.randn(2, 4)).sum()).backward()
            copy.grad = torch.full((5,), float("inf") if step == 0 else 1.0)
            scaler.step(optimizer)
            scaler.update()
    records = sink.records
    assert [r["step_skipped"] for r in records] == [True, False, False]
    # Not on the skipped step (no optimizer stepped at all), but on the first that was not.
    assert [list(r.get("update_not_read", {})) for r in records] == [[], ["no optimizer"], []]


def test_a_weight_check_due_on_a_skipped_step_is_made_on_the_next_step_that_was_not_skipped():
    model = _two_layers()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu")
    sink = tensorquill.MemorySink()
    # A weight check on every second step: steps 1, 3 and 5. Step 1 is skipped.
    with tensorquill.init(model, sink=sink, every=8, scaler=scaler, weights=2, layers=()):
        _scaled_steps(model, optimizer, scaler, 6, overflow=(1,))
    records = sink.records
    assert [r["step_skipped"] for r in records] == [False, True, False, False, False, False]
    assert ["weight_norms" in r for r in records] == [False, False, True, True, False, True]


def test_scaler_update_with_a_new_scale_and_no_optimizer_seen_is_not_a_step():
    model = _two_layers()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu")
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, every=4, scaler=scaler):
        scaler.scale(model(torch.randn(2, 4)).sum())  # makes the scale, as the first forward would
        scaler.update(2.0**10)  # sets the scale: no optimizer went through the scaler
        _scaled_steps(model, optimizer, scaler, 2)
        scaler.update(2.0**12)
    assert [r["step"] for r in sink.records] == [0, 1]
    assert [r["step_skipped"] for r in sink.records] == [False, False]
