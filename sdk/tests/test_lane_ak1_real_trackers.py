"""Lane AK-1: the scalar mirror against the real `wandb` and `mlflow` packages, where installed.

Neither is installed in the project's test interpreter, so both tests skip there; the stand-ins in
`test_lane_ak1_mirror.py` cover the same behaviour. These run W&B offline and MLflow on a local
file store: no network, no API key.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill.sink import MemorySink


def _open(model):
    return tensorquill.init(model, sink=MemorySink(), run_id="ak1-real", layers=(), grads=False, weights=False,
                            heartbeat_seconds=None, stack_seconds=None)


def _step(model, opt):
    model(torch.ones(2, 4)).sum().backward()
    opt.step()
    opt.zero_grad()


@pytest.mark.parametrize("tensorquill_first", [False, True])
def test_real_wandb_offline_in_both_orders(tmp_path, monkeypatch, tensorquill_first):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    original = wandb.Run.__dict__["log"]
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    if tensorquill_first:
        session = _open(model)
        run = wandb.init(project="ak1", mode="offline")
    else:
        run = wandb.init(project="ak1", mode="offline")
        session = _open(model)
    try:
        assert wandb.log({"loss": 0.5, "flag": True}) is None
        run.log({"eval/acc": 0.25}, commit=False)
        _step(model, opt)
    finally:
        run.finish()
        session.close()
    scalars = [r.get("scalars", {}) for r in session._sink.records]
    assert scalars == [{"loss": 0.5, "eval/acc": 0.25}]
    assert wandb.Run.__dict__["log"] is original


def test_real_mlflow_on_a_local_file_store(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    original = (mlflow.log_metric, mlflow.log_metrics)
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    session = _open(model)
    try:
        with mlflow.start_run():
            mlflow.log_metric("loss", 0.5)  # MLflow's own step 0 is not TensorQuill's
            mlflow.log_metrics({"eval/acc": 0.25}, step=1000)
            _step(model, opt)
    finally:
        session.close()
    scalars = [r.get("scalars", {}) for r in session._sink.records]
    assert scalars == [{"loss": 0.5, "eval/acc": 0.25}]
    assert (mlflow.log_metric, mlflow.log_metrics) == original
