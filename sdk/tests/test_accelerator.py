import pytest
import torch
from torch import nn

pytest.importorskip("accelerate")

import tensorlens  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from tensorlens.accelerator import watch_accelerator  # noqa: E402


def _patched():
    return Accelerator.clip_grad_norm_, Accelerator.backward


def _loop(accelerator, model, optimizer, batches, max_norm=0.05):
    """An Accelerate loop as its documentation writes it. Returns each microbatch's loss
    and each update's gradient norm before clipping."""
    losses, norms = [], []
    for batch in batches:
        with accelerator.accumulate(model):
            loss = model(batch).pow(2).mean()
            losses.append(loss.item())
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grads = [p.grad for p in model.parameters()]
                norms.append(torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])).item())
                accelerator.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()
    return losses, norms


def _setup(accumulate=1):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=accumulate)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # A scheduler replaces `step` on the optimizer instance, as in most real loops.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    return accelerator, model, optimizer


def test_loop_records_loss_and_the_norm_before_clipping_once():
    originals = _patched()
    accelerator, model, optimizer = _setup(accumulate=2)
    sink = tensorlens.MemorySink()
    session = tensorlens.init(model, sink=sink, capture="full")
    watch_accelerator(session)
    assert _patched() != originals
    losses, norms = _loop(accelerator, model, optimizer, torch.randn(8, 8, 4))
    session.close()
    records = sink.records
    assert [r["step"] for r in records] == [0, 1, 2, 3]
    # Once, not twice: older Accelerate also goes through torch's clip_grad_norm_.
    assert [r["clip_grad_norm"] for r in records] == pytest.approx(norms, rel=1e-5)
    assert all(r["grad_norm"] == pytest.approx(0.05, rel=1e-3) for r in records)
    expected = [(a + b) / 2 for a, b in zip(losses[::2], losses[1::2])]
    assert [r["scalars"]["train/loss"] for r in records] == pytest.approx(expected, rel=1e-5)


def test_closing_the_session_or_the_watch_restores_accelerate():
    originals = _patched()
    accelerator, model, optimizer = _setup()
    session = tensorlens.init(model, sink=tensorlens.MemorySink())
    watch = watch_accelerator(session, loss_name=None)
    watch.close()
    assert _patched() == originals

    sink = tensorlens.MemorySink()
    session2 = tensorlens.init(model, sink=sink)
    watch_accelerator(session2, loss_name=None)
    _loop(accelerator, model, optimizer, torch.randn(2, 8, 4))
    session.close()
    session2.close()
    assert "scalars" not in sink.records[0] or "train/loss" not in sink.records[0]["scalars"]
    assert _patched() == originals


def test_another_models_accelerator_calls_are_not_recorded():
    accelerator, model, optimizer = _setup()
    other = nn.Linear(4, 1)
    other_optimizer = accelerator.prepare(torch.optim.SGD(other.parameters(), lr=0.01))
    sink = tensorlens.MemorySink()
    session = tensorlens.init(model, sink=sink, capture="full")
    watch_accelerator(session)
    _loop(accelerator, other, other_optimizer, torch.randn(3, 8, 4))
    _loop(accelerator, model, optimizer, torch.randn(1, 8, 4))
    session.close()
    assert len(sink.records) == 1
    assert sink.records[0]["clip_grad_norm"] > 0


def test_a_single_tensor_is_clipped_as_accelerate_does_but_not_recorded():
    # Accelerate iterates a bare tensor into rows and returns 0, where torch returns the
    # real norm. TensorLens must neither fail nor record that 0 as the step's norm.
    accelerator, model, optimizer = _setup()
    sink = tensorlens.MemorySink()
    session = tensorlens.init(model, sink=sink, capture="full")
    watch_accelerator(session, loss_name=None)
    param = next(model.parameters())
    loss = model(torch.randn(4, 4)).pow(2).mean()
    accelerator.backward(loss)
    watched = accelerator.clip_grad_norm_(param, 1e9)
    optimizer.step()
    session.close()  # restores Accelerate's own method
    assert watched == accelerator.clip_grad_norm_(param, 1e9)  # what Accelerate itself returns
    optimizer.zero_grad()
    assert len(sink.records) == 1
    assert sink.records[0].get("clip_grad_norm") is None
