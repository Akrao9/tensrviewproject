import sys
import types

import tensorlens
from tensorlens.tracker import resolve_run


def clear(monkeypatch):
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)


def test_explicit_run_id_wins(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv("MLFLOW_RUN_ID", "from-env")
    assert resolve_run("mine") == tensorlens.RunInfo("mine", "explicit", 0)


def test_env_then_active_mlflow_run_then_wandb(monkeypatch):
    clear(monkeypatch)
    assert resolve_run().source == "none"

    run = types.SimpleNamespace(id="w-1")
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(run=run))
    assert (resolve_run().run_id, resolve_run().source) == ("w-1", "wandb")

    active = types.SimpleNamespace(info=types.SimpleNamespace(run_id="m-1"))
    monkeypatch.setitem(sys.modules, "mlflow", types.SimpleNamespace(active_run=lambda: active))
    assert (resolve_run().run_id, resolve_run().source) == ("m-1", "mlflow")

    monkeypatch.setenv("MLFLOW_RUN_ID", "e-1")
    assert (resolve_run().run_id, resolve_run().source) == ("e-1", "env")


def test_no_active_mlflow_run_falls_through(monkeypatch):
    # What PyTorch Lightning's MLFlowLogger looks like: mlflow imported, no active run.
    clear(monkeypatch)
    monkeypatch.setitem(sys.modules, "mlflow", types.SimpleNamespace(active_run=lambda: None))
    assert resolve_run().source == "none"


def test_wandb_step_is_never_read(monkeypatch):
    clear(monkeypatch)

    class Run:
        id = "w-2"

        @property
        def step(self):
            raise AssertionError("wandb.run.step must not be read")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(run=Run()))
    assert resolve_run().run_id == "w-2"
