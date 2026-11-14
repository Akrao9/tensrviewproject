import sys

import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")

import tensorlens  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from tensorlens.huggingface import TensorLensCallback  # noqa: E402


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        self.losses = []

    def forward(self, x, labels=None):
        loss = (self.net(x).squeeze(-1) - labels).pow(2).mean()
        if not torch.compiler.is_compiling():
            self.losses.append(loss.item())
        return {"loss": loss}


def _data(n=64):
    torch.manual_seed(0)
    return [{"x": torch.randn(4), "labels": torch.randn(())} for _ in range(n)]


def _trainer(tmp_path, model, callback, **kwargs):
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path), max_steps=4, per_device_train_batch_size=8, report_to="none",
        save_strategy="no", disable_tqdm=True, use_cpu=True, learning_rate=0.01, logging_steps=2, **kwargs,
    )
    return transformers.Trainer(model=model, args=args, train_dataset=_data(), callbacks=[callback])


def test_callback_records_steps_clipping_and_loss(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_RUN_ID", "mlflow-run-9")
    sink = tensorlens.MemorySink()
    model = Model()
    callback = TensorLensCallback(sink=sink, capture="full")
    trainer = _trainer(tmp_path, model, callback, max_grad_norm=0.05, gradient_accumulation_steps=2)
    originals = (Accelerator.clip_grad_norm_, Accelerator.backward)
    trainer.train()
    assert (Accelerator.clip_grad_norm_, Accelerator.backward) == originals and callback.session is None
    records = sink.records
    assert [r["step"] for r in records] == list(range(trainer.state.global_step)) == [0, 1, 2, 3]
    assert all(r["run_id"] == "mlflow-run-9" for r in records)
    # The norm before clipping, recorded once: it is the norm Trainer logs (every 2 steps),
    # while the captured gradients are read after clipping.
    logged = [entry["grad_norm"] for entry in trainer.state.log_history if "grad_norm" in entry]
    assert [records[1]["clip_grad_norm"], records[3]["clip_grad_norm"]] == pytest.approx(logged, rel=1e-4)
    assert all(r["grad_norm"] == pytest.approx(0.05, rel=1e-3) for r in records)
    # Each step's loss is the mean over its two accumulated microbatches.
    expected = [(a + b) / 2 for a, b in zip(model.losses[::2], model.losses[1::2])]
    assert [r["scalars"]["train/loss"] for r in records] == pytest.approx(expected, rel=1e-5)


def test_callback_under_trainer_torch_compile(tmp_path):
    sink = tensorlens.MemorySink()
    model = Model()
    trainer = _trainer(
        tmp_path, model, TensorLensCallback(sink=sink, capture="full"),
        torch_compile=True, torch_compile_backend="aot_eager",
    )
    trainer.train()
    assert [r["step"] for r in sink.records] == [0, 1, 2, 3]
    assert all(r["layers"] and r["grad_norm"] > 0 and "train/loss" in r["scalars"] for r in sink.records)


def test_second_train_call_and_resumed_step_numbers(tmp_path):
    sink = tensorlens.MemorySink()
    trainer = _trainer(tmp_path, Model(), TensorLensCallback(sink=sink))
    trainer.train()
    trainer.state.global_step = 0  # train() again from scratch: a second session
    trainer.train()
    assert [r["step"] for r in sink.records] == [0, 1, 2, 3] * 2

    callback = TensorLensCallback(sink=tensorlens.MemorySink())
    state = transformers.TrainerState(global_step=40)
    callback.on_train_begin(None, state, None, model=Model(), optimizer=None)
    assert callback.session._next_step == 40
    callback.close()


def test_a_failed_train_can_be_closed_and_the_next_one_starts_clean(tmp_path):
    class Broken(Model):
        def forward(self, x, labels=None):
            if len(self.losses) == 2:
                raise RuntimeError("boom")
            return super().forward(x, labels)

    sink = tensorlens.MemorySink()
    callback = TensorLensCallback(sink=sink)
    with pytest.raises(RuntimeError, match="boom"):
        _trainer(tmp_path, Broken(), callback).train()
    assert callback.session is not None
    callback.close()
    callback.close()
    assert callback.session is None and [r["step"] for r in sink.records] == [0, 1]


def test_tracker_run_on_rank_zero_reaches_every_rank(monkeypatch):
    import torch.distributed as dist

    from tensorlens import huggingface

    for name in ("MLFLOW_RUN_ID",):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)  # rank 1 has no W&B run
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(dist, "broadcast_object_list", lambda box, src=0: box.__setitem__(0, "wandb-run-3"))
    assert huggingface._shared_tracker_run_id() == "wandb-run-3"
