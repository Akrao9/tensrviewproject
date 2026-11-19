import sys

import pytest
import torch
from torch import nn

L = pytest.importorskip("lightning.pytorch")

import tensorquill  # noqa: E402
from tensorquill.lightning import TensorQuillCallback  # noqa: E402


class Module(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        self.losses = []

    def training_step(self, batch, batch_idx):
        loss = self.net(batch).pow(2).mean()
        self.losses.append(loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def train_dataloader(self):
        torch.manual_seed(0)
        return torch.utils.data.DataLoader(torch.randn(64, 4), batch_size=8)


class RunLogger(L.loggers.Logger):
    """Stands in for MLFlowLogger: a `run_id` on rank 0."""

    name, version = "run-logger", "0"

    @property
    def run_id(self):
        # Only rank 0 knows the run, as with MLFlowLogger; other ranks must get it by broadcast.
        dist = torch.distributed
        on_rank_zero = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
        return "mlflow-run-7" if on_rank_zero else None

    def log_metrics(self, metrics, step=None):
        pass

    def log_hyperparams(self, params, *args, **kwargs):
        pass


class GivenOnRankZero(TensorQuillCallback):
    """A `run_id=` on rank 0 only, as `run_id=wandb.run.id if wandb.run else None` gives."""

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            self.init_kwargs = {**self.init_kwargs, "run_id": "given-0"}
        super().on_fit_start(trainer, pl_module)


def _trainer(tmp_path, callback, **kwargs):
    return L.Trainer(
        default_root_dir=tmp_path, max_steps=4, callbacks=[callback], enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, accelerator="cpu", devices=1, **kwargs,
    )


def test_callback_records_steps_clipping_and_loss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sink = tensorquill.MemorySink()
    module = Module()
    trainer = _trainer(
        tmp_path, TensorQuillCallback(sink=sink, capture="full"), logger=RunLogger(),
        gradient_clip_val=1.0, accumulate_grad_batches=2,
    )
    trainer.fit(module)
    records = sink.records
    assert [r["step"] for r in records] == list(range(trainer.global_step)) == [0, 1, 2, 3]
    assert all(r["run_id"] == "mlflow-run-7" for r in records)
    assert all(r["clip_grad_norm"] > 0 and r["grad_norm"] > 0 for r in records)
    # Each step's loss is the mean over its two accumulated microbatches.
    expected = [(a + b) / 2 for a, b in zip(module.losses[::2], module.losses[1::2])]
    assert [r["scalars"]["train/loss"] for r in records] == pytest.approx(expected, rel=1e-5)


def test_callback_without_a_tracker_writes_a_generated_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("MLFLOW_RUN_ID", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    trainer = _trainer(tmp_path, TensorQuillCallback(), logger=False)
    trainer.fit(Module())
    files = list((tmp_path / "tensorquill-out").glob("run-*.rank0.jsonl"))
    assert len(files) == 1 and len(tensorquill.load_records(files[0])) == 4


def test_callback_closes_the_session_when_fitting_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class Broken(Module):
        def training_step(self, batch, batch_idx):
            if batch_idx == 2:
                raise RuntimeError("boom")
            return super().training_step(batch, batch_idx)

    callback = TensorQuillCallback(sink=tensorquill.MemorySink())
    with pytest.raises(RuntimeError, match="boom"):
        _trainer(tmp_path, callback, logger=False).fit(Broken())
    assert callback.session is None
    # With a file, the end marker says the fit raised, not that it closed normally (row S4).
    from tensorquill.sink import read_end_marker

    path = tmp_path / "broken.rank0.jsonl"
    callback = TensorQuillCallback(sink=tensorquill.JsonlSink(path))
    with pytest.raises(RuntimeError, match="boom"):
        _trainer(tmp_path, callback, logger=False).fit(Broken())
    end = read_end_marker(path)
    assert (end["state"], end["outcome"], end["exception"]) == ("closed", "raised", "RuntimeError")


@pytest.mark.skipif(not torch.distributed.is_available(), reason="needs torch.distributed")
def test_rank_zero_run_id_reaches_every_rank_under_ddp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    trainer = L.Trainer(
        default_root_dir=tmp_path, max_steps=3, callbacks=[TensorQuillCallback()], logger=RunLogger(),
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        accelerator="cpu", devices=2, strategy="ddp_spawn",
    )
    trainer.fit(Module())
    out = tmp_path / "tensorquill-out"
    assert sorted(p.name for p in out.glob("*.jsonl")) == ["mlflow-run-7.rank0.jsonl", "mlflow-run-7.rank1.jsonl"]
    for path in out.glob("*.jsonl"):
        assert [r["step"] for r in tensorquill.load_records(path)] == [0, 1, 2]


@pytest.mark.skipif(not torch.distributed.is_available(), reason="needs torch.distributed")
def test_a_run_id_given_on_rank_zero_only_reaches_every_rank(tmp_path, monkeypatch):
    # Every rank takes the same path whatever it was given: before lane T1 the callback
    # broadcast only on ranks with no run_id, so rank 1 waited in a broadcast rank 0 skipped.
    from datetime import timedelta

    from lightning.pytorch.strategies import DDPStrategy

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    trainer = L.Trainer(
        default_root_dir=tmp_path, max_steps=3, callbacks=[GivenOnRankZero()], logger=RunLogger(),
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        accelerator="cpu", devices=2, strategy=DDPStrategy(start_method="spawn", timeout=timedelta(seconds=20)),
    )
    trainer.fit(Module())
    out = tmp_path / "tensorquill-out"
    assert sorted(p.name for p in out.glob("*.jsonl")) == ["given-0.rank0.jsonl", "given-0.rank1.jsonl"]
    for path in out.glob("*.jsonl"):
        assert [r["step"] for r in tensorquill.load_records(path)] == [0, 1, 2]
