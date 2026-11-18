"""Review findings SDK-02 and SDK-04.

SDK-02: `watch_accelerator(session)` wraps `session.close`, and `Session.__exit__` calls
`close(_exception=...)`; the wrapper must pass that through, or leaving `with session:` raises
`TypeError` and no end marker is written.

SDK-04: `clip_grad_norm_` is matched to a session by any of the session's trainable parameters,
not by the first parameter passed, which is frozen in LoRA and PEFT models (the Trainer clips
`model.parameters()`). Only an L2 norm is recorded, on the torch path and the Accelerate path
alike: `record_grad_norm` sums the squares of a step's norms, which means nothing for another norm.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill.sink import read_end_marker


def _frozen_first(frozen: bool = True) -> nn.Module:
    """A model whose first parameter is frozen, as a LoRA model's base weights are."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    if frozen:
        model[0].weight.requires_grad_(False)
    return model


# ---- SDK-02 -----------------------------------------------------------------------------


@pytest.mark.parametrize("raises", [False, True])
def test_with_block_closes_after_watch_accelerator(tmp_path, monkeypatch, raises):
    pytest.importorskip("accelerate")
    from accelerate import Accelerator

    from tensorquill import accelerator as watched

    monkeypatch.chdir(tmp_path)
    original = Accelerator.clip_grad_norm_
    try:
        with tensorquill.init(nn.Linear(4, 1), run_id="with-block") as session:
            watch = watch_twice = None
            watch = watched.watch_accelerator(session, data_wait=False)
            # A second watch wraps the wrapper; both layers must pass the arguments on.
            watch_twice = watched.watch_accelerator(session, data_wait=False, loss_name=None)
            assert Accelerator.clip_grad_norm_ is not original
            if raises:
                raise ValueError("boom")
    except ValueError as error:
        assert raises and str(error) == "boom"
    assert session._closed
    marker = read_end_marker(tmp_path / "tensorquill-out" / "with-block.rank0.jsonl")
    assert marker["state"] == "closed"
    if raises:
        assert (marker["outcome"], marker["exception"]) == ("raised", "ValueError")
    else:
        assert marker["outcome"] == "closed" and "exception" not in marker
    # Closing the session stopped both watches and put Accelerate back.
    assert watch not in watched._active and watch_twice not in watched._active
    assert Accelerator.clip_grad_norm_ is original


def test_close_after_watch_accelerator_still_takes_no_arguments(tmp_path, monkeypatch):
    pytest.importorskip("accelerate")
    from tensorquill.accelerator import watch_accelerator

    monkeypatch.chdir(tmp_path)
    session = tensorquill.init(nn.Linear(4, 1), run_id="plain-close")
    watch_accelerator(session, data_wait=False)
    session.close()
    session.close()  # a second close does nothing, as without the watch
    assert read_end_marker(tmp_path / "tensorquill-out" / "plain-close.rank0.jsonl")["outcome"] == "closed"


# ---- SDK-04, torch.nn.utils.clip_grad_norm_ ------------------------------------------------


def _torch_loop(model: nn.Module, clip, steps: int = 3) -> tuple[list[dict], list[float]]:
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, capture="full", every=1)
    returned = []
    for _ in range(steps):
        loss = model(torch.randn(8, 4)).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        returned.append(float(clip(model.parameters())))
        optimizer.step()
    session.close()
    return sink.records, returned


@pytest.mark.parametrize("frozen", [False, True])
def test_torch_clip_is_recorded_when_the_first_parameter_is_frozen(frozen):
    records, returned = _torch_loop(_frozen_first(frozen), lambda ps: torch.nn.utils.clip_grad_norm_(ps, 1.0))
    assert [r.get("clip_grad_norm") for r in records] == pytest.approx(returned, rel=1e-6)


def test_torch_clip_of_a_wholly_frozen_list_is_not_recorded():
    model = _frozen_first()
    frozen_only = [model[0].weight]

    def clip(_):
        return torch.nn.utils.clip_grad_norm_(frozen_only, 1.0)

    records, _ = _torch_loop(model, clip)
    assert all("clip_grad_norm" not in r for r in records)


@pytest.mark.parametrize(
    ("call", "recorded"),
    [
        (lambda ps: torch.nn.utils.clip_grad_norm_(ps, 1.0, 2), True),
        (lambda ps: torch.nn.utils.clip_grad_norm_(ps, 1.0, norm_type=2.0), True),
        (lambda ps: torch.nn.utils.clip_grad_norm_(ps, max_norm=1.0, norm_type="2"), True),
        (lambda ps: torch.nn.utils.clip_grad_norm_(ps, 1.0, math.inf), False),
        (lambda ps: torch.nn.utils.clip_grad_norm_(ps, 1.0, norm_type="inf"), False),
        (lambda ps: torch.nn.utils.clip_grad_norm_(ps, 1.0, 1), False),
    ],
)
@pytest.mark.parametrize("frozen", [False, True])
def test_torch_clip_records_an_l2_norm_only(call, recorded, frozen):
    records, returned = _torch_loop(_frozen_first(frozen), call)
    if recorded:
        assert [r.get("clip_grad_norm") for r in records] == pytest.approx(returned, rel=1e-6)
    else:
        assert all("clip_grad_norm" not in r for r in records)


# ---- SDK-04, Accelerator.clip_grad_norm_ ---------------------------------------------------


def _accelerate_loop(frozen: bool, clip) -> tuple[list[dict], list[float]]:
    pytest.importorskip("accelerate")
    from accelerate import Accelerator

    from tensorquill.accelerator import watch_accelerator

    model = _frozen_first(frozen)
    accelerator = Accelerator(cpu=True)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, capture="full", every=1)
    watch_accelerator(session, data_wait=False)
    returned = []
    for _ in range(3):
        loss = model(torch.randn(8, 4)).pow(2).mean()
        accelerator.backward(loss)
        returned.append(float(clip(accelerator, model.parameters())))
        optimizer.step()
        optimizer.zero_grad()
    session.close()
    return sink.records, returned


@pytest.mark.parametrize("frozen", [False, True])
def test_accelerate_clip_is_recorded_when_the_first_parameter_is_frozen(frozen):
    records, returned = _accelerate_loop(frozen, lambda acc, ps: acc.clip_grad_norm_(ps, 1.0))
    assert [r.get("clip_grad_norm") for r in records] == pytest.approx(returned, rel=1e-6)


@pytest.mark.parametrize(
    ("call", "recorded"),
    [
        (lambda acc, ps: acc.clip_grad_norm_(ps, 1.0, 2.0), True),
        (lambda acc, ps: acc.clip_grad_norm_(ps, 1.0, norm_type=2), True),
        (lambda acc, ps: acc.clip_grad_norm_(ps, 1.0, math.inf), False),
        (lambda acc, ps: acc.clip_grad_norm_(ps, 1.0, norm_type=1), False),
    ],
)
@pytest.mark.parametrize("frozen", [False, True])
def test_accelerate_clip_records_an_l2_norm_only(call, recorded, frozen):
    records, returned = _accelerate_loop(frozen, call)
    if recorded:
        assert [r.get("clip_grad_norm") for r in records] == pytest.approx(returned, rel=1e-6)
    else:
        assert all("clip_grad_norm" not in r for r in records)
