"""How a process knows its rank and world size when no process group is initialised.

A process group, when there is one, decides. Without one, the rank is the first of `RANK`,
`SLURM_PROCID` (inside an `srun` step only), `OMPI_COMM_WORLD_RANK` and `PMI_RANK`, and the world
size is read from the same launcher: `WORLD_SIZE`, `SLURM_STEP_NUM_TASKS`, `OMPI_COMM_WORLD_SIZE`,
`PMI_SIZE`. The SLURM world size is the step's task count, never the job's (`SLURM_NTASKS`): one
plain `python` started in an allocation of several tasks is one rank of a world of one.
`LOCAL_RANK` alone is read as before, with one warning at `init` when the world size is above 1
or not known. A rank that is not below the world size is said once at `init`.

Each case sets the variables a launcher sets (srun(1), sbatch(1), Open MPI's "Environment variables
set for MPI applications") and calls `tensorquill.init` in a temporary folder.
"""

from __future__ import annotations

import json
import os
import warnings

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import tracker
from tensorquill.sink import MemorySink

LAUNCHER_PREFIXES = ("SLURM_", "OMPI_", "PMI_", "TORCHELASTIC_", "MLFLOW_", "WANDB_", "TENSORQUILL_")
LAUNCHER_NAMES = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
                  "GROUP_WORLD_SIZE", "MV2_COMM_WORLD_RANK", "MV2_COMM_WORLD_SIZE")


@pytest.fixture
def clean(monkeypatch, tmp_path):
    """No launcher variable from the machine running the tests, and a folder of its own."""
    for name in list(os.environ):
        if name in LAUNCHER_NAMES or name.startswith(LAUNCHER_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    assert not (torch.distributed.is_available() and torch.distributed.is_initialized())
    return monkeypatch


def _train(model, steps: int) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(steps):
        model(torch.randn(3, 2)).pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad()


def _init(env: dict[str, str], monkeypatch, model=None, **options):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    model = nn.Linear(2, 1) if model is None else model
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, every=1, heartbeat_seconds=None, stack_seconds=None, **options)
    said = [str(w.message) for w in caught if "tensorquill" in str(w.message).lower()]
    return session, said


# torchrun alone; torchrun under srun (`RANK` is the global rank, `SLURM_PROCID` the node's task);
# nothing set; srun alone with two tasks; Open MPI; PMI; mpirun started inside a batch script, which
# sees the batch script's `SLURM_PROCID` and no step's task count.
CASES = {
    "torchrun": ({"RANK": "2", "LOCAL_RANK": "2", "WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4",
                  "TORCHELASTIC_RUN_ID": "abc"}, 2, 4),
    "torchrun_under_srun": ({"RANK": "5", "LOCAL_RANK": "1", "WORLD_SIZE": "8", "TORCHELASTIC_RUN_ID": "abc",
                             "SLURM_JOB_ID": "7", "SLURM_STEP_ID": "0", "SLURM_PROCID": "1", "SLURM_LOCALID": "0",
                             "SLURM_NTASKS": "2", "SLURM_STEP_NUM_TASKS": "2"}, 5, 8),
    "nothing": ({}, 0, 1),
    "srun_two_tasks": ({"SLURM_JOB_ID": "7", "SLURM_STEP_ID": "0", "SLURM_PROCID": "1", "SLURM_LOCALID": "1",
                        "SLURM_NTASKS": "2", "SLURM_NPROCS": "2", "SLURM_STEP_NUM_TASKS": "2"}, 1, 2),
    "open_mpi": ({"OMPI_COMM_WORLD_RANK": "3", "OMPI_COMM_WORLD_SIZE": "4", "OMPI_COMM_WORLD_LOCAL_RANK": "3"}, 3, 4),
    "pmi": ({"PMI_RANK": "1", "PMI_SIZE": "2"}, 1, 2),
    "mpirun_in_a_batch_script": ({"SLURM_JOB_ID": "7", "SLURM_PROCID": "0", "SLURM_NTASKS": "4",
                                  "OMPI_COMM_WORLD_RANK": "1", "OMPI_COMM_WORLD_SIZE": "4"}, 1, 4),
    "an_empty_rank_is_passed_over": ({"RANK": "", "SLURM_PROCID": "1", "SLURM_STEP_NUM_TASKS": "2"}, 1, 2),
    "a_digit_that_is_not_ascii_is_passed_over": ({"RANK": "٣", "PMI_RANK": "1", "PMI_SIZE": "2"}, 1, 2),
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_rank_and_world_size_come_from_the_launcher(clean, case):
    env, rank, world = CASES[case]
    session, said = _init(env, clean, run_id="r", sink=MemorySink())
    try:
        assert (session.run.rank, tracker._rank(), tracker._world_size()) == (rank, rank, world)
        assert said == []
    finally:
        session.close()


def test_one_plain_python_in_an_allocation_of_several_tasks_is_one_rank_of_one(clean):
    """What a batch script (or `salloc`'s shell) sees, with no `srun`: the job's task count and
    no step's. The process is rank 0 of 1, its manifest says world size 1, and the listing implies
    no missing rank."""
    from tensorquill.runs import list_runs

    env = {"SLURM_JOB_ID": "7", "SLURM_NTASKS": "4", "SLURM_NPROCS": "4", "SLURM_PROCID": "0", "SLURM_LOCALID": "0",
           "SLURM_TASKS_PER_NODE": "4", "SLURM_JOB_NUM_NODES": "1", "SLURM_NNODES": "1"}
    model = nn.Linear(2, 1)
    session, said = _init(env, clean, model)
    _train(model, 3)
    session.close()
    assert (session.run.rank, session.run.run_id, tracker._world_size()) == (0, "run-slurm7", 1)
    assert said == []
    manifest = json.loads((session._sink.path.parent / "run-slurm7.run.json").read_text())
    assert manifest["world_size"] == 1
    rows = list_runs("tensorquill-out")
    assert [(row["run_id"], row["ranks"], row["rank_count"]) for row in rows] == [("run-slurm7", [0], 1)]
    assert rows[0]["end"]["state"] == "closed"
    assert not rows[0]["end"].get("missing")


@pytest.mark.parametrize("env, rank, world", [
    ({"LOCAL_RANK": "1"}, 1, 1),  # no world size: not known
    ({"LOCAL_RANK": "0", "WORLD_SIZE": "2"}, 0, 2),
    ({"LOCAL_RANK": "3", "WORLD_SIZE": "2"}, 3, 2),  # and not below the world size: still one warning
])
def test_local_rank_alone_is_read_with_one_warning_that_it_is_node_local(clean, env, rank, world):
    session, said = _init(env, clean, run_id="r", sink=MemorySink())
    try:
        assert (session.run.rank, tracker._world_size()) == (rank, world)
        assert len(said) == 1, said
        assert "LOCAL_RANK" in said[0] and "node" in said[0]
    finally:
        session.close()


def test_local_rank_in_a_world_of_one_says_nothing(clean):
    session, said = _init({"LOCAL_RANK": "0", "WORLD_SIZE": "1"}, clean, run_id="r", sink=MemorySink())
    try:
        assert (session.run.rank, tracker._world_size(), said) == (0, 1, [])
    finally:
        session.close()


@pytest.mark.parametrize("env, rank, world, names", [
    ({"RANK": "3", "WORLD_SIZE": "2"}, 3, 2, ("RANK", "WORLD_SIZE")),
    ({"RANK": "1"}, 1, 1, ("RANK", "WORLD_SIZE")),
    ({"SLURM_PROCID": "2", "SLURM_STEP_NUM_TASKS": "2"}, 2, 2, ("SLURM_PROCID", "SLURM_STEP_NUM_TASKS")),
    ({"PMI_RANK": "4"}, 4, 1, ("PMI_RANK", "PMI_SIZE")),
])
def test_a_rank_not_below_the_world_size_is_said_once(clean, env, rank, world, names):
    session, said = _init(env, clean, run_id="r", sink=MemorySink())
    try:
        assert (session.run.rank, tracker._world_size()) == (rank, world)
        assert len(said) == 1, said
        assert all(name in said[0] for name in names), said[0]
    finally:
        session.close()


def test_the_warning_is_said_at_each_init_and_once_there(clean):
    clean.setenv("LOCAL_RANK", "1")
    for _ in range(2):
        session, said = _init({}, clean, run_id="r", sink=MemorySink())
        session.close()
        assert len(said) == 1, said


def test_a_process_group_decides_over_the_launchers_variables(clean, tmp_path):
    """Under a group (gloo, world of one here) the group's rank and size are read, whatever the
    variables say, and nothing is said about them."""
    import torch.distributed as dist

    for name, value in {"SLURM_PROCID": "1", "SLURM_STEP_NUM_TASKS": "2", "LOCAL_RANK": "3",
                        "OMPI_COMM_WORLD_RANK": "5", "OMPI_COMM_WORLD_SIZE": "8"}.items():
        clean.setenv(name, value)
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1)
    try:
        session, said = _init({}, clean, run_id="r", sink=MemorySink())
        session.close()
        assert (session.run.rank, tracker._rank(), tracker._world_size(), said) == (0, 0, 1, [])
    finally:
        dist.destroy_process_group()


def test_the_slurm_world_size_is_the_steps_task_count():
    """The one place the SLURM world size is read: `SLURM_STEP_NUM_TASKS`, which srun sets for
    each task of a step and sbatch and salloc do not set (srun(1), sbatch(1), salloc(1))."""
    assert tracker._LAUNCHERS[1] == ("SLURM_PROCID", "SLURM_STEP_NUM_TASKS")
