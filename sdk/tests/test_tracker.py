import sys
import types

import tensorquill
from tensorquill.tracker import resolve_run


def clear(monkeypatch):
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    for name in ("TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "SLURM_STEP_ID"):
        monkeypatch.delenv(name, raising=False)


def test_explicit_run_id_wins(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv("MLFLOW_RUN_ID", "from-env")
    assert resolve_run("mine") == tensorquill.RunInfo("mine", "explicit", 0)


def test_env_then_active_mlflow_run_then_wandb(monkeypatch):
    clear(monkeypatch)
    assert resolve_run().source == "generated"

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
    assert resolve_run().source == "generated"


def test_wandb_step_is_never_read(monkeypatch):
    clear(monkeypatch)

    class Run:
        id = "w-2"

        @property
        def step(self):
            raise AssertionError("wandb.run.step must not be read")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(run=Run()))
    assert resolve_run().run_id == "w-2"


def test_untracked_runs_get_distinct_ids_and_files(monkeypatch, tmp_path):
    import re

    from torch import nn

    clear(monkeypatch)
    first, second = resolve_run(), resolve_run()
    assert re.fullmatch(r"run-\d{8}-\d{6}-[0-9a-f]{6}", first.run_id)
    assert first.run_id != second.run_id
    monkeypatch.chdir(tmp_path)
    for _ in range(2):
        tensorquill.init(nn.Linear(2, 2)).close()
    assert len(list((tmp_path / "tensorquill-out").glob("run-*.rank0.jsonl"))) == 2


def test_launcher_job_id_is_shared_by_every_rank(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "none")  # torchrun's default: not a job ID
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_STEP_ID", "1")
    assert resolve_run() == tensorquill.RunInfo("run-slurm4242.1", "launcher", 0)
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "5c1e")
    assert resolve_run().run_id == "run-5c1e"


def test_generated_id_is_broadcast_from_rank_zero(monkeypatch):
    import torch.distributed as dist

    clear(monkeypatch)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 1)

    def broadcast(box, src=0):
        box[0] = "run-from-rank-0"

    monkeypatch.setattr(dist, "broadcast_object_list", broadcast)
    assert resolve_run() == tensorquill.RunInfo("run-from-rank-0", "generated", 1)


def test_ranks_with_no_shared_id_are_warned(monkeypatch, recwarn):
    import pytest

    clear(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "none")
    with pytest.warns(RuntimeWarning, match="rank 1 of 2 found no run ID the ranks share"):
        run = resolve_run()
    assert run.source == "generated" and run.rank == 1
    # An ID the ranks can share, or one passed in, is not a problem.
    recwarn.clear()
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "5c1e")
    resolve_run()
    resolve_run("mine")
    monkeypatch.delenv("WORLD_SIZE")
    monkeypatch.delenv("TORCHELASTIC_RUN_ID")
    resolve_run()  # one process
    assert not [w for w in recwarn if "run ID the ranks share" in str(w.message)]
