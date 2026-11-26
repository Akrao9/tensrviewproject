"""Wave AO, lane AO-7: the Lightning, Hugging Face and Accelerate integrations against the real frameworks.

- BUG-14: under a `GradScaler` (Lightning's `MixedPrecision` plugin, `precision="16-mixed"`) the loss
  `on_before_backward` is given has been multiplied by the scale (`MixedPrecision.pre_backward` scales it
  before it calls the callbacks), so it was recorded scaled; and a fit that raised between the backward and
  the optimizer step left its loss in the callback, added to the next fit's first step.
- BUG-15: with several evaluation datasets the callback left the "eval" phase after the first one's
  `on_evaluate`, so what the second logged went to the next training step and its loader time to that
  step's `data_wait`; `eval_on_start`'s evaluation ran in the "train" phase; and a resume given to
  `Trainer.train(resume_from_checkpoint=path)` (not in `TrainingArguments`) was not recorded.
- BUG-16: the Accelerate watch added the loss before `Accelerator.backward` ran, so a backward that raised
  left its loss on the next step; and `Watch.close()` left the loader timing it had turned on.

Each framework is imported inside each test, so an interpreter without it skips test by test.
"""

from __future__ import annotations

import shutil
import sys
import time

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill.sink import MemorySink


@pytest.fixture(autouse=True)
def no_trackers(monkeypatch):
    for name in ("wandb", "mlflow", "mlflow.tracking", "mlflow.tracking.fluent", "wandb.sdk", "wandb.sdk.wandb_run"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    yield


def _steps(sink):
    return [r for r in sink.records if r.get("record") != "scalars"]


# --- Lightning (BUG-14) -----------------------------------------------------------------------------------


def _lightning():
    return pytest.importorskip("lightning.pytorch")


def _lightning_module(L, *, target=1.0, fail=False):
    """Weights at zero and inputs of ones: the loss is `target ** 2` exactly, at every step of one fit."""

    class Module(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 1, bias=False)
            nn.init.zeros_(self.lin.weight)
            self.losses = []

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = (self.lin(x) - y * target).pow(2).mean()
            self.losses.append(loss.detach().float().cpu().item())
            return loss

        def on_after_backward(self):
            if fail:
                raise RuntimeError("the fit fails between backward and the optimizer step")

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.0)

        def train_dataloader(self):
            data = torch.utils.data.TensorDataset(torch.ones(8, 4), torch.ones(8, 1))
            return torch.utils.data.DataLoader(data, batch_size=2)

    return Module()


def _lightning_trainer(L, callback, **kwargs):
    options = dict(max_steps=1, logger=False, enable_checkpointing=False, enable_progress_bar=False,
                   enable_model_summary=False, accelerator="cpu", devices=1, callbacks=[callback])
    options.update(kwargs)
    return L.Trainer(**options)


def test_lightning_loss_under_a_grad_scaler_is_the_loss_not_the_scaled_loss(tmp_path):
    """Lightning's own `MixedPrecision` plugin with a real `GradScaler` (on the CPU, so it runs on any
    machine): the loss is 1.0 and the scale 1024, and `train/loss` is 1.0, not 1024.0."""
    L = _lightning()
    from lightning.pytorch.plugins.precision.amp import MixedPrecision

    from tensorquill.lightning import TensorQuillCallback

    sink = MemorySink()
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0, enabled=True)
    module = _lightning_module(L)
    _lightning_trainer(
        L, TensorQuillCallback(every=1, sink=sink, mirror_scalars=False), max_steps=2, default_root_dir=tmp_path,
        plugins=[MixedPrecision(precision="16-mixed", device="cpu", scaler=scaler)],
    ).fit(module)
    losses = [r["scalars"]["train/loss"] for r in _steps(sink)]
    assert losses == pytest.approx([1.0, 1.0])
    assert module.losses == pytest.approx([1.0, 1.0])


def test_lightning_loss_under_a_grad_scaler_with_accumulation_is_the_mean_of_the_microbatches(tmp_path):
    """Two microbatches a step under the scaler: each is divided by the scale it was multiplied by, and the
    sum is their mean (Lightning has divided each by `accumulate_grad_batches`), as without a scaler."""
    L = _lightning()
    from lightning.pytorch.plugins.precision.amp import MixedPrecision

    from tensorquill.lightning import TensorQuillCallback

    sink = MemorySink()
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0**16, enabled=True)
    _lightning_trainer(
        L, TensorQuillCallback(every=1, sink=sink, mirror_scalars=False), max_steps=2, default_root_dir=tmp_path,
        accumulate_grad_batches=2, plugins=[MixedPrecision(precision="16-mixed", device="cpu", scaler=scaler)],
    ).fit(_lightning_module(L, target=3.0))
    assert [r["scalars"]["train/loss"] for r in _steps(sink)] == pytest.approx([9.0, 9.0])


def test_lightning_16_mixed_on_mps_records_the_loss_not_the_scaled_loss(tmp_path):
    """`precision="16-mixed"` as a user writes it, on a device where Lightning gives it a `GradScaler`.
    On the CPU Lightning turns "16-mixed" into "bf16-mixed" with no scaler, so this needs a GPU: Apple MPS
    here (CUDA takes the same path in the plugin, and is not on this machine)."""
    L = _lightning()
    if not (torch.backends.mps.is_available()):
        pytest.skip("16-mixed with a GradScaler needs a GPU; this machine has no MPS device")
    from tensorquill.lightning import TensorQuillCallback

    sink = MemorySink()
    module = _lightning_module(L, target=2.0)
    trainer = _lightning_trainer(L, TensorQuillCallback(every=1, sink=sink, mirror_scalars=False), max_steps=2,
                                 default_root_dir=tmp_path, accelerator="mps", precision="16-mixed")
    trainer.fit(module)
    assert getattr(trainer.precision_plugin, "scaler", None) is not None
    assert [r["scalars"]["train/loss"] for r in _steps(sink)] == pytest.approx(module.losses, rel=1e-3)
    assert module.losses == pytest.approx([4.0, 4.0], rel=1e-3)


def test_lightning_a_failed_fit_leaves_no_loss_to_the_next_fit(tmp_path, monkeypatch):
    """The first fit raises from `on_after_backward` (loss 100); the second, through the same callback, has
    loss 1: its first step records 1.0, not 101.0."""
    L = _lightning()
    from tensorquill.lightning import TensorQuillCallback

    monkeypatch.chdir(tmp_path)
    sink = MemorySink()
    callback = TensorQuillCallback(every=1, sink=sink, mirror_scalars=False)
    with pytest.raises(RuntimeError, match="between backward"):
        _lightning_trainer(L, callback, default_root_dir=tmp_path).fit(_lightning_module(L, target=10.0, fail=True))
    assert _steps(sink) == []
    _lightning_trainer(L, callback, default_root_dir=tmp_path).fit(_lightning_module(L, target=1.0))
    assert [r["scalars"]["train/loss"] for r in _steps(sink)] == pytest.approx([1.0])


def test_lightning_loss_without_a_scaler_is_as_before(tmp_path):
    """Pin: full precision, no scaler: the loss Lightning back-propagates, as it was."""
    L = _lightning()
    from tensorquill.lightning import TensorQuillCallback

    sink = MemorySink()
    _lightning_trainer(L, TensorQuillCallback(every=1, sink=sink, mirror_scalars=False), max_steps=2,
                       default_root_dir=tmp_path, accumulate_grad_batches=2).fit(_lightning_module(L, target=2.0))
    assert [r["scalars"]["train/loss"] for r in _steps(sink)] == pytest.approx([4.0, 4.0])


# --- Hugging Face (BUG-15) --------------------------------------------------------------------------------


def _transformers():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    return transformers


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Linear(4, 1)

    def forward(self, x, labels=None):
        return {"loss": (self.net(x).squeeze(-1) - labels).pow(2).mean()}


class _Data(torch.utils.data.Dataset):
    """`n` rows; each one read sleeps `delay` seconds, so its loader's time shows in `data_wait`."""

    def __init__(self, n, delay=0.0, seed=0):
        generator = torch.Generator().manual_seed(seed)
        self.rows = [{"x": torch.randn(4, generator=generator), "labels": torch.randn((), generator=generator)}
                     for _ in range(n)]
        self.delay = delay

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.delay:
            time.sleep(self.delay)
        return self.rows[index]


SLOW = 0.15  # seconds a slow evaluation row takes to read; two rows, one batch: 0.3 s of loader time
FAST_ENOUGH = 0.1  # a training step's whole `data_wait` here is milliseconds


def _hf_trainer(output, model, callbacks, eval_dataset=None, **kwargs):
    transformers = _transformers()
    options = dict(
        output_dir=str(output), max_steps=3, per_device_train_batch_size=4, per_device_eval_batch_size=4,
        report_to="none", save_strategy="no", disable_tqdm=True, use_cpu=True, learning_rate=0.01,
        logging_steps=100, dataloader_num_workers=0,
    )
    options.update(kwargs)
    args = transformers.TrainingArguments(**options)
    return transformers.Trainer(model=model, args=args, train_dataset=_Data(32), eval_dataset=eval_dataset,
                                callbacks=callbacks)


def _logs_on_log(transformers, callback, prefix):
    """Logs one value at each `on_log` of an evaluation named `prefix`, through the session, as the tracker
    mirror does with what a W&B or MLflow logger logs there (the `Trainer` calls `on_log` before `on_evaluate`)."""

    class LogsOnLog(transformers.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and f"{prefix}_loss" in logs and callback.session is not None:
                callback.session.log({f"seen/{prefix}": 1.0})

    return LogsOnLog()


def test_second_evaluation_dataset_is_recorded_on_the_step_the_evaluation_followed(tmp_path):
    """`eval_dataset={"a": ..., "b": ...}` evaluated after step 1 (TensorQuill's step 1, `global_step` 2):
    what is logged during "b"'s evaluation (here at its `on_log`, as the tracker mirror logs) is on step 1,
    with "a"'s, and not on step 2; and "b"'s slow loader is not step 2's `data_wait`."""
    transformers = _transformers()
    from tensorquill.huggingface import TensorQuillCallback

    sink = MemorySink()
    callback = TensorQuillCallback(sink=sink, every=1, mirror_scalars=False)
    callbacks = [callback, _logs_on_log(transformers, callback, "eval_a"), _logs_on_log(transformers, callback, "eval_b")]
    _hf_trainer(tmp_path / "out", _Model(), callbacks, eval_dataset={"a": _Data(2, seed=1), "b": _Data(2, SLOW, seed=2)},
                eval_strategy="steps", eval_steps=2).train()
    steps = _steps(sink)
    assert [r["step"] for r in steps] == [0, 1, 2]
    after, next_step = steps[1]["scalars"], steps[2]["scalars"]
    assert {"eval/a_loss", "eval/b_loss", "seen/eval_a", "seen/eval_b"} <= set(after)
    assert not {name for name in next_step if name.startswith(("eval/", "seen/"))}
    assert steps[2]["data_wait"] < FAST_ENOUGH


def test_first_of_several_evaluation_datasets_slow_is_not_the_next_steps_data_wait(tmp_path):
    """Pin beside it: with the slow dataset first, its time was already out of training's; it stays out."""
    _transformers()
    from tensorquill.huggingface import TensorQuillCallback

    sink = MemorySink()
    _hf_trainer(tmp_path / "out", _Model(), [TensorQuillCallback(sink=sink, every=1, mirror_scalars=False)],
                eval_dataset={"a": _Data(2, SLOW, seed=1), "b": _Data(2, seed=2)},
                eval_strategy="steps", eval_steps=2).train()
    steps = _steps(sink)
    assert steps[2]["data_wait"] < FAST_ENOUGH
    assert {"eval/a_loss", "eval/b_loss"} <= set(steps[1]["scalars"])


def test_one_evaluation_dataset_then_training_counts_the_next_fetch(tmp_path):
    """Pin: after an evaluation, the next training step's own fetch is still its `data_wait` (the "eval"
    phase ends at the last `on_evaluate`, before the Trainer fetches the next step's batches)."""
    _transformers()
    from tensorquill.huggingface import TensorQuillCallback

    class SlowTrain(_Data):
        def __getitem__(self, index):
            time.sleep(0.05)
            return self.rows[index]

    sink = MemorySink()
    trainer = _hf_trainer(tmp_path / "out", _Model(), [TensorQuillCallback(sink=sink, every=1, mirror_scalars=False)],
                          eval_dataset=_Data(2, seed=1), eval_strategy="steps", eval_steps=2)
    trainer.train_dataset = SlowTrain(32)
    trainer.train()
    steps = _steps(sink)
    # Four rows a batch at 0.05 s each: every training step, the one after the evaluation too, waited 0.2 s.
    assert all(r["data_wait"] >= 0.15 for r in steps), [r["data_wait"] for r in steps]


def test_eval_on_start_is_not_training_data_wait(tmp_path):
    """`eval_on_start=True` evaluates before the first training step, with no event before it: its slow
    loader is not step 0's `data_wait`."""
    _transformers()
    from tensorquill.huggingface import TensorQuillCallback

    sink = MemorySink()
    _hf_trainer(tmp_path / "out", _Model(), [TensorQuillCallback(sink=sink, every=1, mirror_scalars=False)],
                eval_dataset=_Data(2, SLOW, seed=1), eval_strategy="steps", eval_steps=100, eval_on_start=True,
                max_steps=2).train()
    steps = _steps(sink)
    assert [r["step"] for r in steps] == [0, 1]
    assert steps[0]["data_wait"] < FAST_ENOUGH
    baseline = [r for r in sink.records if r.get("record") == "scalars"]
    assert len(baseline) == 1 and baseline[0]["step"] == 0 and "eval/loss" in baseline[0]["scalars"]


def test_eval_on_start_with_several_datasets_records_both_and_waits_for_neither(tmp_path):
    _transformers()
    from tensorquill.huggingface import TensorQuillCallback

    sink = MemorySink()
    _hf_trainer(tmp_path / "out", _Model(), [TensorQuillCallback(sink=sink, every=1, mirror_scalars=False)],
                eval_dataset={"a": _Data(2, SLOW, seed=1), "b": _Data(2, SLOW, seed=2)}, eval_strategy="steps",
                eval_steps=100, eval_on_start=True, max_steps=2).train()
    steps = _steps(sink)
    assert steps[0]["data_wait"] < FAST_ENOUGH
    scalars = {}
    for record in sink.records:
        if record.get("record") == "scalars":
            scalars.update(record["scalars"])
    assert {"eval/a_loss", "eval/b_loss"} <= set(scalars)


def _saved_checkpoint(tmp_path):
    """A real checkpoint of one step, `output/checkpoint-1`; its `model.safetensors` is what the load compares."""
    from tensorquill.huggingface import TensorQuillCallback

    output = tmp_path / "output"
    _hf_trainer(output, _Model(), [TensorQuillCallback(sink=MemorySink(), every=1, mirror_scalars=False)],
                max_steps=1, save_strategy="steps", save_steps=1).train()
    folder = output / "checkpoint-1"
    assert (folder / "trainer_state.json").is_file()
    return output, folder


def test_resume_from_a_checkpoint_given_to_train_records_the_load(tmp_path):
    """`trainer.train(resume_from_checkpoint=<a folder outside output_dir>)`: the resumed step records the
    checkpoint load of that folder, its tensors compared, as a resume given in `TrainingArguments` does."""
    _transformers()
    from tensorquill.huggingface import TensorQuillCallback

    output, usual = _saved_checkpoint(tmp_path)
    external = tmp_path / "external"
    shutil.move(str(usual), str(external))
    sink = MemorySink()
    _hf_trainer(output, _Model(), [TensorQuillCallback(sink=sink, every=1, mirror_scalars=False)],
                max_steps=2).train(resume_from_checkpoint=str(external))
    steps = _steps(sink)
    assert [r["step"] for r in steps] == [1]
    load = steps[0]["checkpoint"]["load"]
    assert {"model", "optimizer"} <= set(load["components"])
    assert "tensors" in steps[0]["checkpoint"]


def test_resume_given_to_train_is_the_folder_recorded_not_the_one_under_output_dir(tmp_path):
    """A decoy at the conventional `output/checkpoint-1` with its `optimizer.pt` removed, and the resume
    from an intact copy elsewhere: what is recorded is the folder the Trainer loaded, with its optimizer."""
    _transformers()
    from tensorquill.huggingface import TensorQuillCallback

    output, usual = _saved_checkpoint(tmp_path)
    external = tmp_path / "external"
    shutil.copytree(usual, external)
    (usual / "optimizer.pt").unlink()
    sink = MemorySink()
    _hf_trainer(output, _Model(), [TensorQuillCallback(sink=sink, every=1, mirror_scalars=False)],
                max_steps=2).train(resume_from_checkpoint=str(external))
    load = _steps(sink)[0]["checkpoint"]["load"]
    assert "optimizer" in load["components"]


def test_resume_true_records_the_newest_checkpoint_under_output_dir(tmp_path):
    """Pin: `train(resume_from_checkpoint=True)` resumes from the newest checkpoint under `output_dir`,
    which is the folder recorded."""
    _transformers()
    from tensorquill.huggingface import TensorQuillCallback

    output, _usual = _saved_checkpoint(tmp_path)
    sink = MemorySink()
    _hf_trainer(output, _Model(), [TensorQuillCallback(sink=sink, every=1, mirror_scalars=False)],
                max_steps=2).train(resume_from_checkpoint=True)
    load = _steps(sink)[0]["checkpoint"]["load"]
    assert {"model", "optimizer"} <= set(load["components"])


# --- Accelerate (BUG-16) ----------------------------------------------------------------------------------


def _accelerate():
    return pytest.importorskip("accelerate")


def _zero_linear():
    model = nn.Linear(4, 1, bias=False)
    nn.init.zeros_(model.weight)
    return model


def test_accelerate_a_backward_that_raises_adds_no_loss(tmp_path):
    """A backward that raises (a loss with no gradient) and is caught; the next microbatch's loss is 1.0,
    and that is the step's `train/loss`, not 51.0."""
    accelerate = _accelerate()
    from tensorquill.accelerator import watch_accelerator

    accelerator = accelerate.Accelerator(cpu=True)
    model = _zero_linear()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    model, optimizer = accelerator.prepare(model, optimizer)
    sink = MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, data_wait=False, mirror_scalars=False)
    try:
        watch_accelerator(session, data_wait=False)
        with pytest.raises(RuntimeError):
            accelerator.backward(torch.tensor(50.0, requires_grad=False))
        accelerator.backward((model(torch.ones(2, 4)) - torch.ones(2, 1)).pow(2).mean())
        optimizer.step()
    finally:
        session.close()
    assert [r["scalars"]["train/loss"] for r in _steps(sink)] == pytest.approx([1.0])


def test_accelerate_a_step_that_ends_inside_backward_keeps_its_loss(tmp_path, monkeypatch):
    """Pin: under DeepSpeed, `Accelerator.backward` also steps the optimizer at the end of an accumulation
    (`DeepSpeedEngineWrapper.backward` calls `engine.step()`). Stood in here by a backward that steps: each
    step's loss, the mean of its two microbatches, is on that step and not the next."""
    accelerate = _accelerate()
    from tensorquill.accelerator import watch_accelerator

    model = _zero_linear()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    calls = {"n": 0}

    def stepping_backward(self, loss, **kwargs):
        (loss / 2).backward()
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            optimizer.step()
            optimizer.zero_grad()

    monkeypatch.setattr(accelerate.Accelerator, "backward", stepping_backward)
    accelerator = accelerate.Accelerator(cpu=True, gradient_accumulation_steps=2)
    sink = MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, data_wait=False, mirror_scalars=False)
    try:
        watch_accelerator(session, data_wait=False)
        for target in (1.0, 3.0, 2.0, 4.0):
            accelerator.backward((model(torch.ones(2, 4)) - target).pow(2).mean())
    finally:
        session.close()
    assert [r["scalars"]["train/loss"] for r in _steps(sink)] == pytest.approx([5.0, 10.0])


def test_accelerate_watch_close_stops_the_loader_timing_it_started(tmp_path):
    """A session opened without `data_wait`; a default watch (which times loaders) opened and closed; then a
    fetch from a `DataLoader` that sleeps: the step has no `data_wait`, as before the watch."""
    _accelerate()
    from tensorquill.accelerator import watch_accelerator

    sink = MemorySink()
    with tensorquill.init(nn.Linear(1, 1), sink=sink, auto_step=False, mirror_scalars=False) as session:
        watch_accelerator(session).close()

        def slow(batch):
            time.sleep(0.12)
            return batch

        next(iter(torch.utils.data.DataLoader([1], collate_fn=slow)))
        session.step()
    assert sink.records[0].get("data_wait") is None


def test_accelerate_watch_close_leaves_loader_timing_the_session_had(tmp_path):
    """Pin: a session that times loaders itself (`data_wait=True` at `init`) still does after the watch closes."""
    _accelerate()
    from tensorquill.accelerator import watch_accelerator

    sink = MemorySink()
    with tensorquill.init(nn.Linear(1, 1), sink=sink, auto_step=False, data_wait=True, mirror_scalars=False) as session:
        watch_accelerator(session).close()

        def slow(batch):
            time.sleep(0.12)
            return batch

        next(iter(torch.utils.data.DataLoader([1], collate_fn=slow)))
        session.step()
    assert sink.records[0]["data_wait"] >= 0.1


def test_accelerate_open_watch_times_loaders_until_closed(tmp_path):
    """Pin: while the default watch is open, the loader is timed; after its close, it is not."""
    _accelerate()
    from tensorquill.accelerator import watch_accelerator

    def slow(batch):
        time.sleep(0.12)
        return batch

    sink = MemorySink()
    with tensorquill.init(nn.Linear(1, 1), sink=sink, auto_step=False, mirror_scalars=False) as session:
        watch = watch_accelerator(session)
        next(iter(torch.utils.data.DataLoader([1], collate_fn=slow)))
        session.step()
        watch.close()
        watch.close()  # twice: nothing more
        next(iter(torch.utils.data.DataLoader([1], collate_fn=slow)))
        session.step()
    assert sink.records[0]["data_wait"] >= 0.1
    assert sink.records[1].get("data_wait") is None
