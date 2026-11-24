"""Lane AK-1: the scalar mirror against the real `wandb` and `mlflow` packages, where installed.

Neither is installed in the project's `.venv` (or in CI), so these skip there and the stand-ins in
`test_lane_ak1_mirror.py` cover the same behaviour; the owner's `.venv-trackers` has both. W&B runs
offline (and disabled) in a temporary `WANDB_DIR`, MLflow on a SQLite store in a temporary folder
(MLflow 3.18 refuses a `file:` store): no network, no API key.
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


@pytest.mark.parametrize("mode", ["offline", "disabled"])
@pytest.mark.parametrize("tensorquill_first", [False, True])
def test_real_wandb_in_both_orders(tmp_path, monkeypatch, tensorquill_first, mode):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", mode)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    original = wandb.Run.__dict__["log"]
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    if tensorquill_first:
        session = _open(model)
        run = wandb.init(project="ak1", mode=mode)
    else:
        run = wandb.init(project="ak1", mode=mode)
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


def test_real_wandb_finished_and_a_new_run_while_the_session_is_open(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    before_init = wandb.log
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    first = wandb.init(project="ak1", mode="offline")
    session = _open(model)
    second = None
    try:
        wandb.log({"a": 1.0})
        first.finish()
        with pytest.raises(wandb.Error):
            wandb.log({"a": 9.0})  # W&B's own pre-init error, unchanged
        second = wandb.init(project="ak1", mode="offline")
        assert wandb.log is not before_init
        wandb.log({"b": 2.0})
        _step(model, opt)
    finally:
        if second is not None:
            second.finish()
        session.close()
    assert [r.get("scalars", {}) for r in session._sink.records] == [{"a": 1.0, "b": 2.0}]


def test_real_mlflow_on_a_local_sqlite_store(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    original = (mlflow.log_metric, mlflow.log_metrics)
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    session = _open(model)
    try:
        with mlflow.start_run():
            mlflow.log_metric("loss", 0.5)  # MLflow's own step 0 is not TensorQuill's
            mlflow.log_metrics({"eval/acc": 0.25}, step=1000)
            mlflow.tracking.fluent.log_metric("lr", 0.01)  # the defining module's own name
            _step(model, opt)
    finally:
        session.close()
    scalars = [r.get("scalars", {}) for r in session._sink.records]
    assert scalars == [{"loss": 0.5, "eval/acc": 0.25, "lr": 0.01}]
    assert (mlflow.log_metric, mlflow.log_metrics) == original
