"""Lane AK-5: the mirror's precedence on the newest step, against the real `wandb` and `mlflow`.

Skipped where the trackers are not installed (the project's `.venv`, CI); the owner's
`.venv-trackers` has both. Offline only: W&B in offline mode in a temporary `WANDB_DIR`, MLflow on a
SQLite store in a temporary folder (MLflow 3.18 refuses a `file:` store).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill.sink import MemorySink


def _open(every: int):
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    session = tensorquill.init(model, sink=MemorySink(), run_id="ak5-real", layers=(), grads=False, weights=False,
                               heartbeat_seconds=None, stack_seconds=None, every=every)
    return session, model, torch.optim.SGD(model.parameters(), lr=0.01)


def _step(session, model, opt, loss: float) -> None:
    session.log({"train/loss": loss})
    model(torch.ones(2, 4)).sum().backward()
    opt.step()
    opt.zero_grad()


@pytest.mark.parametrize("every", [4, 2])
@pytest.mark.parametrize("explicit_first", [True, False])
def test_real_wandb_late_values(tmp_path, monkeypatch, every, explicit_first):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    run = wandb.init(project="ak5", mode="offline")
    session, model, opt = _open(every)
    try:
        for i in range(2):
            _step(session, model, opt, float(i))
        with session.phase("eval"):
            if explicit_first:
                session.log({"eval/loss": 1.0})
            wandb.log({"eval/loss": 9.0, "train/loss": 42.0, "eval/acc": 0.5}, step=500)
            if not explicit_first:
                session.log({"eval/loss": 1.0})
    finally:
        run.finish()
        session.close()
    scalars = {r["step"]: r.get("scalars", {}) for r in session._sink.records}
    assert scalars[1] == {"train/loss": 1.0, "eval/loss": 1.0, "eval/acc": 0.5}


@pytest.mark.parametrize("explicit_first", [True, False])
def test_real_mlflow_late_values(tmp_path, monkeypatch, explicit_first):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    session, model, opt = _open(2)
    try:
        with mlflow.start_run():
            for i in range(2):
                _step(session, model, opt, float(i))
            with session.phase("eval"):
                if explicit_first:
                    session.log({"eval/loss": 1.0})
                mlflow.log_metrics({"eval/loss": 9.0, "train/loss": 42.0}, step=500)
                mlflow.log_metric("eval/acc", 0.5)
                if not explicit_first:
                    session.log({"eval/loss": 1.0})
    finally:
        session.close()
    scalars = {r["step"]: r.get("scalars", {}) for r in session._sink.records}
    assert scalars[1] == {"train/loss": 1.0, "eval/loss": 1.0, "eval/acc": 0.5}
