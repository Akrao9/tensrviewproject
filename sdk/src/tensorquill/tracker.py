"""Find the tracker run that the tensor stream belongs to.

TensorQuill never imports a tracker itself. It only looks at trackers the training
script has already imported, so it adds no import cost and no dependency.
"""

from __future__ import annotations

import os
import secrets
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class RunInfo:
    run_id: str | None
    source: str  # "explicit", "env", "mlflow", "wandb", "launcher" or "generated"
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
    """Resolve the run ID: explicit, then MLFLOW_RUN_ID, then MLflow, then W&B, then a
    launcher's job ID (torchrun, SLURM), and otherwise a generated unique ID, so separate
    runs never append to the same file.

    `mlflow.active_run()` is thread-local and is None when the run was created through
    `MlflowClient` (PyTorch Lightning's `MLFlowLogger` does that, and only on rank 0), so
    those setups must pass `run_id` explicitly.
    """
    rank = _rank()
    if run_id:
        return RunInfo(run_id, "explicit", rank)
    found = tracker_run_id()
    if found is not None:
        return RunInfo(found[0], found[1], rank)
    shared = _launcher_run_id()
    if shared is not None:
        return RunInfo(shared, "launcher", rank)
    return RunInfo(_shared_generated_id(), "generated", rank)


def tracker_run_id() -> tuple[str, str] | None:
    """The tracker run this process can see, as (run ID, source), with no collective call.

    A tracker's run often exists on rank 0 only, so a caller with several ranks should
    read it there and broadcast it.
    """
    env = os.environ.get("MLFLOW_RUN_ID")
    if env:
        return env, "env"
    mlflow = sys.modules.get("mlflow")
    if mlflow is not None:
        active = mlflow.active_run()
        if active is not None:
            return active.info.run_id, "mlflow"
    wandb = sys.modules.get("wandb")
    run = getattr(wandb, "run", None)
    if run is not None:
        # Read the ID once. Never read `wandb.run.step`: it is a round trip to the W&B
        # background process and raises in shared-mode runs.
        return run.id, "wandb"
    return None


def _launcher_run_id() -> str | None:
    """A job ID every rank of the job sees: torchrun's rendezvous ID (a fresh UUID with
    `--standalone`, "none" unless set otherwise), or SLURM's job and step IDs."""
    elastic = os.environ.get("TORCHELASTIC_RUN_ID")
    if elastic and elastic != "none":
        return f"run-{elastic}"
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        step = os.environ.get("SLURM_STEP_ID")
        return f"run-slurm{job}" + (f".{step}" if step else "")
    return None


def generate_run_id() -> str:
    """`run-<UTC time>-<6 hex digits>`, sortable by start time, as W&B generates run IDs
    when none is given (`wandb/sdk/lib/runid.py`)."""
    return f"run-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def _world_size() -> int:
    dist = sys.modules.get("torch.distributed")
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    value = os.environ.get("WORLD_SIZE")
    return int(value) if value is not None and value.isdigit() and int(value) > 0 else 1


def _shared_generated_id() -> str:
    # Every rank must use the same ID, or one run's files split under different names.
    # With torch.distributed initialized, rank 0's ID is broadcast: a collective, so it
    # needs every rank to call tensorquill.init, which per-rank capture does anyway.
    run_id = generate_run_id()
    dist = sys.modules.get("torch.distributed")
    if dist is not None and dist.is_available() and dist.is_initialized():
        if dist.get_world_size() > 1:
            box = [run_id]
            dist.broadcast_object_list(box, src=0)
            run_id = box[0]
    elif _world_size() > 1:
        # Several ranks, nothing to share an ID through yet: each rank picks its own, so the
        # job is written as one run per rank, each missing the others, and nothing says so.
        warnings.warn(
            f"tensorquill.init on rank {_rank()} of {_world_size()} found no run ID the ranks share "
            "(no tracker run, TORCHELASTIC_RUN_ID or SLURM_JOB_ID) and torch.distributed is not "
            f"initialized, so this rank's run is {run_id!r} and the other ranks' runs have other IDs. "
            "Pass the same run_id= on every rank, or call tensorquill.init after init_process_group.",
            RuntimeWarning,
            stacklevel=4,
        )
    return run_id
