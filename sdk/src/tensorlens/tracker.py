"""Find the tracker run that the tensor stream belongs to.

TensorLens never imports a tracker itself. It only looks at trackers the training
script has already imported, so it adds no import cost and no dependency.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class RunInfo:
    run_id: str | None
    source: str  # "explicit", "env", "mlflow", "wandb" or "none"
    rank: int


def _rank() -> int:
    dist = sys.modules.get("torch.distributed")
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    for name in ("RANK", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is not None and value.isdigit():
            return int(value)
    return 0


def resolve_run(run_id: str | None = None) -> RunInfo:
    """Resolve the run ID: explicit, then MLFLOW_RUN_ID, then MLflow, then W&B.

    `mlflow.active_run()` is thread-local and is None when the run was created through
    `MlflowClient` (PyTorch Lightning's `MLFlowLogger` does that, and only on rank 0), so
    those setups must pass `run_id` explicitly.
    """
    rank = _rank()
    if run_id:
        return RunInfo(run_id, "explicit", rank)
    env = os.environ.get("MLFLOW_RUN_ID")
    if env:
        return RunInfo(env, "env", rank)
    mlflow = sys.modules.get("mlflow")
    if mlflow is not None:
        active = mlflow.active_run()
        if active is not None:
            return RunInfo(active.info.run_id, "mlflow", rank)
    wandb = sys.modules.get("wandb")
    run = getattr(wandb, "run", None)
    if run is not None:
        # Read the ID once. Never read `wandb.run.step`: it is a round trip to the W&B
        # background process and raises in shared-mode runs.
        return RunInfo(run.id, "wandb", rank)
    return RunInfo(None, "none", rank)
