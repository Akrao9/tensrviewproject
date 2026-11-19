"""Which record each piece of a step lands on: a checkpoint saved right after a full ring
(SDK-07), a step the loss scaler skipped (SDK-03), and an optimizer that steps after the one
that ended the step (SDK-06).

A full ring is written as soon as it fills but for its newest step, kept back until the next
step ends (or a flush, or close), so what arrives for that step after it ended (a save, a second
optimizer's update) is still written on its record. A run that hangs or is killed in the next
step has one step fewer on disk, never a block."""

from __future__ import annotations

import time

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


def _written(sink: tensorquill.MemorySink, count: int) -> None:
    """Wait for the writer thread to have written `count` records."""
    deadline = time.monotonic() + 10
    while len(sink.records) < count:
        assert time.monotonic() < deadline, "the writer thread did not write the records"
        time.sleep(0.005)


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


def test_a_full_ring_is_written_at_once_but_for_its_newest_step_until_the_next_step_ends():
    # With no flush and no close, as a run that hangs or is killed would leave it.
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=4)
    try:
        _steps(model, optimizer, 4)
        assert session._steps == [3]  # kept back: a save made now is still written on its record
        _written(sink, 3)
        session.checkpoint({"model": model.state_dict()}, event="save")
        # One optimizer: nothing on the device changed for the held step, so writing it reads nothing
        # off the device again; it is queued from the copy the ring was read into.
        assert session._held is not None and session._held["dirty"] == set()
        _steps(model, optimizer, 1)
        assert session._steps == [4]
        _written(sink, 4)
        assert [r["step"] for r in sink.records] == [0, 1, 2, 3]
        assert [r["checkpoint"]["save"]["after_step"] for r in sink.records if "checkpoint" in r] == [3]
        assert "checkpoint" in sink.records[3]
    finally:
        session.close()


def test_an_explicit_flush_still_writes_every_step():
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=4)
    try:
        _steps(model, optimizer, 4)
        session.flush()
        assert session._steps == []
        _written(sink, 4)
        assert [r["step"] for r in sink.records] == [0, 1, 2, 3]
    finally:
        session.close()


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


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_under_clipping_a_skipped_step_still_has_its_gradient_norms(bad):
    # Clipping by a total that is not finite multiplies every gradient by 0 (an Inf total) or by
    # NaN: an Inf still shows where it was (NaN there, 0 elsewhere), a NaN makes every norm NaN.
    model = _two_layers()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu")
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, every=2, scaler=scaler, layers=()):
        optimizer.zero_grad()
        scaler.scale(model(torch.randn(8, 4)).pow(2).mean()).backward()
        model[1].weight.grad[0, 0] = bad
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    [record] = sink.records
    assert record["step_skipped"] is True
    norms = record["grad_norms"]
    assert set(norms) == {"0.weight", "0.bias", "1.weight", "1.bias"}
    assert norms["1.weight"] != norms["1.weight"]  # NaN
    others = [norms[n] for n in ("0.weight", "0.bias", "1.bias")]
    assert others == [0.0, 0.0, 0.0] if bad == float("inf") else all(v != v for v in others)


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


# --- SDK-06: an optimizer that steps after the one that ended the step -----------------------
# Adam works on each parameter alone, so two AdamWs over two halves of a model, stepped one
# after the other, train it exactly as one AdamW over all of it does: every record of the two
# runs must be the same.


def _mlp() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))


def _optimizers(model: nn.Module, split: bool) -> list[torch.optim.Optimizer]:
    if not split:
        return [torch.optim.AdamW(model.parameters(), lr=1e-3)]
    last = list(model[4].parameters())
    rest = [p for p in model.parameters() if all(p is not q for q in last)]
    return [torch.optim.AdamW(rest, lr=1e-3), torch.optim.AdamW(last, lr=1e-3)]


def _split_run(split: bool, steps: int, fault_at: int | None = None, **init):
    model = _mlp()
    optimizers = _optimizers(model, split)
    data = torch.randn(4096, 16)
    target = data[:, :1].sin()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, run_id="split", updates=True, **init)
    seen = []
    for step in range(steps):
        i = torch.randint(0, 4096, (64,))
        loss = (model(data[i]) - target[i]).pow(2).mean()
        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        if step == fault_at:
            model[4].weight.grad.mul_(1e4)  # a gradient fault in the second optimizer's parameters
        for opt in optimizers:
            opt.step()
        seen.append(model[4].weight.detach().clone())
    session.close()
    return sink.records, seen, session


@pytest.mark.parametrize("every", [1, 3, 50])
def test_each_optimizers_update_is_in_the_record_of_its_step(every):
    one, _, _ = _split_run(False, 7, every=every, weights=1)
    two, _, _ = _split_run(True, 7, every=every, weights=1)
    assert [r["step"] for r in two] == [r["step"] for r in one] == list(range(7))
    for a, b in zip(one, two):
        assert b["update_rms"] == pytest.approx(a["update_rms"], rel=1e-6), a["step"]
        # The weight check is read after both optimizers stepped, as with one.
        assert b["weight_norms"] == pytest.approx(a["weight_norms"], rel=1e-9), a["step"]
        assert b.get("weights_unchanged") == a.get("weights_unchanged")  # none at step 0: nothing before
    # Both optimizers had stepped when step 0's check judged which parameters one holds.
    assert all(r["optimizer_ownership"] == {"state": "known", "unheld": [], "unheld_count": 0} for r in two)


# 300 is the review's step; with every=50, 299 fills the ring and 300 is the first of the next.
@pytest.mark.parametrize("fault_at", [299, 300, 310])
def test_a_gradient_fault_in_the_second_optimizers_parameters_is_an_update_spike(fault_at):
    for split in (False, True):
        records, _, _ = _split_run(split, fault_at + 100, fault_at=fault_at)
        incidents = tensorquill.build_evidence(records)["incidents"]
        spikes = [i for i in incidents if i["kind"] == "update_spike"]
        assert spikes and spikes[0]["step"] == fault_at, (split, spikes)
        assert spikes[0]["parameters"][0]["name"] == "4.weight", split


@pytest.mark.parametrize("every", [2, 50])
def test_the_probe_runs_after_every_optimizer_of_the_step(monkeypatch, every):
    from tensorquill import capture

    probed = []

    def fake_probe(model, fn, state, devices):
        probed.append(model[4].weight.detach().clone())
        return {"probed": len(probed) - 1}

    monkeypatch.setattr(capture, "run_probe", fake_probe)
    records, seen, _ = _split_run(True, 5, every=every, probe=lambda m: None, probe_every=1)
    # Each step's probe saw the last layer as its own optimizer had just left it. Step 0's ran
    # twice: when the lead ended it, before the second optimizer was known, and again after it.
    assert len(probed) == 6
    assert [r["step"] for r in records if "probe" in r] == [0, 1, 2, 3, 4]
    assert all(torch.equal(probed[r["probe"]["probed"]], seen[r["step"]]) for r in records)


def test_the_usual_amp_loop_with_two_optimizers_ends_the_step_after_both():
    def run(split):
        model = _mlp()
        optimizers = _optimizers(model, split)
        scaler = torch.amp.GradScaler("cpu", init_scale=2.0**8)
        sink = tensorquill.MemorySink()
        with tensorquill.init(model, sink=sink, every=2, scaler=scaler, updates=True):
            for _ in range(4):
                for opt in optimizers:
                    opt.zero_grad()
                scaler.scale(model(torch.randn(8, 16)).pow(2).mean()).backward()
                for opt in optimizers:
                    scaler.step(opt)
                scaler.update()
        return sink.records

    one, two = run(False), run(True)
    assert [r["step"] for r in two] == [0, 1, 2, 3]
    for a, b in zip(one, two):
        assert b["update_rms"] == pytest.approx(a["update_rms"], rel=1e-6)


@pytest.mark.parametrize("every", [50, 1])
def test_lightning_manual_optimization_with_two_optimizers_and_two_backward_passes(tmp_path, monkeypatch, every):
    L = pytest.importorskip("lightning.pytorch")
    from tensorquill.lightning import TensorQuillCallback

    class TwoLosses(L.LightningModule):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.body = nn.Linear(4, 8)
            self.head = nn.Linear(8, 1)
            self.automatic_optimization = False
            self.losses = []

        def training_step(self, batch, batch_idx):
            first, second = self.optimizers()
            loss = self.head(torch.tanh(self.body(batch))).pow(2).mean()  # trains the body
            first.zero_grad()
            self.manual_backward(loss)
            first.step()
            other = (self.head(torch.tanh(self.body(batch)).detach()) - 1).pow(2).mean()  # the head
            second.zero_grad()
            self.manual_backward(other)
            second.step()
            self.losses.append((loss.item(), other.item()))

        def configure_optimizers(self):
            return (torch.optim.AdamW(self.body.parameters(), lr=1e-2),
                    torch.optim.AdamW(self.head.parameters(), lr=1e-2))

        def train_dataloader(self):
            torch.manual_seed(0)
            return torch.utils.data.DataLoader(torch.randn(32, 4), batch_size=8)

    monkeypatch.chdir(tmp_path)
    sink = tensorquill.MemorySink()
    module = TwoLosses()
    trainer = L.Trainer(
        default_root_dir=tmp_path, max_epochs=1, callbacks=[TensorQuillCallback(sink=sink, updates=True, every=every)],
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False, logger=False,
        accelerator="cpu", devices=1,
    )
    trainer.fit(module)
    records = sink.records
    assert [r["step"] for r in records] == [0, 1, 2, 3]  # one step per batch, as without Lightning
    names = {"body.weight", "body.bias", "head.weight", "head.bias"}
    losses = [r["scalars"]["train/loss"] for r in records]
    assert all(set(r["update_rms"]) == names for r in records)
    # The loss of the step is everything back-propagated in it, the second optimizer's included.
    assert losses == pytest.approx([a + b for a, b in module.losses], rel=1e-5)


def test_two_optimizers_with_every_step_its_own_block_raise_nothing_and_write_every_step(monkeypatch):
    # Every step fills the ring at every=1: each is the held step when the second optimizer steps.
    # Weights, updates and the probe on every step, each written once.
    from tensorquill import capture

    monkeypatch.setattr(capture, "run_probe", lambda model, fn, state, devices: {"ran": True})
    records, _, session = _split_run(True, 6, every=1, weights=1, probe=lambda m: None, probe_every=1)
    assert [r["step"] for r in records] == list(range(6))
    assert all(r["probe"] == {"ran": True} for r in records)
    assert not session._probe_owed and session._held is None
    assert all({"4.weight", "4.bias"} <= set(r["update_rms"]) for r in records)


def test_an_optimizer_that_steps_before_the_lead_is_taken_for_the_step_before():
    # The open case: the hooks cannot tell an optimizer that steps after the lead in the same
    # iteration from one that steps before the lead in the next. Here the first to step (the
    # lead) is last in the loop from step 2 on, so the other's update lands one step early.
    model = _mlp()
    first, other = _optimizers(model, True)
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, every=8, updates=True, layers=()):
        for step in range(4):
            for opt in (first, other):
                opt.zero_grad()
            model(torch.randn(8, 16)).pow(2).mean().backward()
            if step >= 2:
                other.step()  # starts later, but sits before the lead in the loop
            first.step()
    last = {"4.weight", "4.bias"}
    # Its updates at steps 2 and 3 are on the records of steps 1 and 2.
    both = ["4.bias", "4.weight"]
    assert [sorted(last & set(r["update_rms"])) for r in sink.records] == [[], both, both, []]
