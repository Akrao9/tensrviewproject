"""Checkpoint manifests from the adapters: Lightning's `on_save_checkpoint` / `on_load_checkpoint`,
the `Trainer`'s checkpoint folders, and Accelerate's `save_state` / `load_state`. Each test saves,
resumes from the save, and reads both attempts with `build_evidence`."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import tensorquill


def _resume_findings(records):
    evidence = tensorquill.build_evidence(records)
    [hit] = [i for i in evidence["incidents"] if i["kind"] == "resume_incomplete"] or [None]
    return evidence, hit


def test_lightning_records_its_checkpoints_and_the_resume_that_loads_one(tmp_path):
    L = pytest.importorskip("lightning.pytorch")
    from tensorquill.lightning import TensorQuillCallback

    class Module(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4, 1)

        def training_step(self, batch, batch_idx):
            return self.net(batch).pow(2).mean()

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.01)

        def train_dataloader(self):
            torch.manual_seed(0)
            return torch.utils.data.DataLoader(torch.randn(64, 4), batch_size=8)

    def fit(steps, sink, ckpt_path=None):
        saver = L.callbacks.ModelCheckpoint(dirpath=tmp_path / "ckpt", every_n_train_steps=2, save_top_k=-1)
        trainer = L.Trainer(default_root_dir=tmp_path, max_steps=steps, callbacks=[TensorQuillCallback(sink=sink, run_id="resumed"), saver],
                            enable_progress_bar=False, enable_model_summary=False, accelerator="cpu", devices=1,
                            logger=False)
        trainer.fit(Module(), ckpt_path=ckpt_path)

    first, second = tensorquill.MemorySink(), tensorquill.MemorySink()
    fit(4, first)
    saves = [r["checkpoint"]["save"] for r in first.records if "checkpoint" in r]
    assert [s["after_step"] for s in saves] == [1, 3]
    assert list(saves[0]["components"]) == ["model", "optimizer", "data"]  # data: Lightning's `loops`
    assert saves[0]["components"]["optimizer"]["entries"] == 2
    fit(6, second, ckpt_path=str(tmp_path / "ckpt" / "epoch=0-step=4.ckpt"))
    resumed = second.records[0]
    assert resumed["step"] == 4 and list(resumed["checkpoint"]["load"]["components"]) == ["model", "optimizer", "data"]
    assert resumed["checkpoint"]["optimizer"]["entries_before"] == 2  # Lightning restored the optimizer
    _, hit = _resume_findings(first.records + second.records)
    # Lightning saves no RNG state by default, and that is all the resume left out.
    assert hit["changes"][0]["findings"] == [{"component": "rng", "finding": "missing"}]


def test_the_trainer_records_its_checkpoint_folders_and_a_complete_resume_finds_nothing(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    from tensorquill.huggingface import TensorQuillCallback

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4, 1)

        def forward(self, x, labels=None):
            return {"loss": (self.net(x).squeeze(-1) - labels).pow(2).mean()}

    torch.manual_seed(0)
    data = [{"x": torch.randn(4), "labels": torch.randn(())} for _ in range(64)]

    def train(steps, sink, resume=None):
        args = transformers.TrainingArguments(
            output_dir=str(tmp_path), max_steps=steps, per_device_train_batch_size=8, report_to="none",
            save_strategy="steps", save_steps=2, disable_tqdm=True, use_cpu=True, learning_rate=0.01)
        trainer = transformers.Trainer(model=Model(), args=args, train_dataset=data, callbacks=[TensorQuillCallback(sink=sink, run_id="resumed")])
        trainer.train(resume_from_checkpoint=resume)

    first, second = tensorquill.MemorySink(), tensorquill.MemorySink()
    train(4, first)
    saves = [r["checkpoint"]["save"] for r in first.records if "checkpoint" in r]
    assert [s["after_step"] for s in saves] == [1, 3]
    assert {"model", "optimizer", "scheduler", "rng", "data"} == set(saves[-1]["components"])
    train(6, second, resume=str(tmp_path / "checkpoint-4"))
    resumed = second.records[0]
    assert resumed["step"] == 4 and resumed["checkpoint"]["load"]["components"].keys() == saves[-1]["components"].keys()
    assert resumed["checkpoint"]["optimizer"]["entries_before"] == 2
    evidence, hit = _resume_findings(first.records + second.records)
    assert hit is None and not [n for n in evidence.get("not_checked", []) if n["check"] == "resume_state"]


def test_accelerate_save_and_load_state_are_recorded(tmp_path):
    pytest.importorskip("accelerate")
    from accelerate import Accelerator
    from tensorquill.accelerator import watch_accelerator

    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    accelerator = Accelerator(cpu=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    originals = (Accelerator.save_state, Accelerator.load_state)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, heartbeat_seconds=None, every=4)
    watch_accelerator(session)
    assert (Accelerator.save_state, Accelerator.load_state) != originals
    for step in range(3):
        accelerator.backward(model(torch.randn(2, 4)).pow(2).mean())
        optimizer.step()
        optimizer.zero_grad()
    accelerator.save_state(str(tmp_path / "state"))
    accelerator.load_state(str(tmp_path / "state"))
    accelerator.backward(model(torch.randn(2, 4)).pow(2).mean())
    optimizer.step()
    session.close()
    assert (Accelerator.save_state, Accelerator.load_state) == originals  # restored with the session
    save = sink.records[2]["checkpoint"]["save"]
    assert save["after_step"] == 2 and {"model", "optimizer", "rng"} <= set(save["components"])
    assert set(sink.records[3]["checkpoint"]["load"]["components"]) == set(save["components"])
