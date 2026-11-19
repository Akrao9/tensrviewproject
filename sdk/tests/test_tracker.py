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


def _ranks_sharing_a_store(monkeypatch, store=None):
    """Fake a process group of two ranks whose default store is a real in-process HashStore,
    and make every collective that could carry the run ID raise: under NCCL each one puts
    tensors on `torch.cuda.current_device()`. `as_rank(r)` makes this process rank r and starts
    its count of exchanges again, as a process of its own would."""
    import itertools

    import torch.distributed as dist
    from torch.distributed import distributed_c10d

    from tensorquill import tracker

    clear(monkeypatch)
    store = dist.HashStore() if store is None else store
    rank = {"now": 0}
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: rank["now"])
    monkeypatch.setattr(distributed_c10d, "_get_default_store", lambda: store)

    def forbidden(*args, **kwargs):
        raise AssertionError("the run ID was shared by a collective")

    for name in ("broadcast_object_list", "all_gather_object", "broadcast", "all_gather", "barrier"):
        monkeypatch.setattr(dist, name, forbidden)

    def as_rank(r):
        rank["now"] = r
        monkeypatch.setattr(tracker, "_exchanges", itertools.count(), raising=False)

    return as_rank


def test_rank_zeros_run_id_and_source_are_shared_through_the_store(monkeypatch, recwarn):
    # Under torch.distributed rank 0 resolves the whole order and puts (ID, source) in the
    # process group's store; every other rank takes it from there, whatever it sees itself.
    # Two real processes run in test_run_id_ranks.py; this checks what each rank keeps.
    as_rank = _ranks_sharing_a_store(monkeypatch)
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "5c1e")
    # Rank 0 takes its W&B run over the launcher's ID, and shares it.
    as_rank(0)
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(run=types.SimpleNamespace(id="w-0")))
    assert resolve_run() == tensorquill.RunInfo("w-0", "wandb", 0)
    # Rank 1 sees no W&B run and a launcher ID, and still takes rank 0's.
    as_rank(1)
    monkeypatch.delitem(sys.modules, "wandb")
    assert resolve_run() == tensorquill.RunInfo("w-0", "wandb", 1)
    assert not [w for w in recwarn if "run ID" in str(w.message)]


def test_a_run_id_on_some_ranks_takes_the_same_path(monkeypatch, recwarn):
    # A run_id passed on rank 0 only (`run_id=wandb.run.id if wandb.run else None`) used to
    # leave the other ranks waiting in the broadcast alone. Now every rank exchanges, and an
    # explicit ID is rank 0's resolution like any other: rank 0's wins.
    import pytest

    as_rank = _ranks_sharing_a_store(monkeypatch)
    as_rank(0)
    assert resolve_run("mine") == tensorquill.RunInfo("mine", "explicit", 0)
    as_rank(1)
    assert resolve_run() == tensorquill.RunInfo("mine", "explicit", 1)
    assert not [w for w in recwarn if "run ID" in str(w.message)]
    # The same ID on every rank: no warning.
    as_rank(1)
    assert resolve_run("mine") == tensorquill.RunInfo("mine", "explicit", 1)
    assert not [w for w in recwarn if "run ID" in str(w.message)]
    # Another ID on rank 1: rank 0's wins, and rank 1 is told.
    as_rank(1)
    with pytest.warns(RuntimeWarning, match="rank 1 was given run_id='theirs', and rank 0's run ID is 'mine'"):
        assert resolve_run("theirs") == tensorquill.RunInfo("mine", "explicit", 1)


def test_a_rank_that_never_hears_from_rank_zero_falls_back(monkeypatch):
    # Rank 0 never shares an ID (it never calls init). Another rank waits a bounded time, then
    # writes under an ID of its own (its explicit one, else the launcher's, else a generated
    # one), with one warning, and raises nothing.
    import time

    import pytest

    from tensorquill import tracker

    as_rank = _ranks_sharing_a_store(monkeypatch)
    monkeypatch.setattr(tracker, "RUN_ID_WAIT_SECONDS", 0.3, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "9")
    as_rank(1)
    started = time.monotonic()
    fell_back = r"rank 1 of 2 did not get rank 0's run ID \(.*; it waits at most 0.3 s\)"
    with pytest.warns(RuntimeWarning, match=fell_back) as caught:
        assert resolve_run() == tensorquill.RunInfo("run-slurm9", "launcher", 1)
    assert 0.25 < time.monotonic() - started < 5
    assert len([w for w in caught if "run ID" in str(w.message)]) == 1
    as_rank(1)
    with pytest.warns(RuntimeWarning, match="did not get rank 0's run ID"):
        assert resolve_run("mine") == tensorquill.RunInfo("mine", "explicit", 1)


def test_a_store_that_fails_never_raises(monkeypatch):
    # No usable store (MPI's default group has a placeholder one): every rank keeps an ID of
    # its own at once, with one warning, and nothing raises.
    import pytest
    import torch.distributed as dist

    from tensorquill import tracker

    monkeypatch.setattr(tracker, "RUN_ID_WAIT_SECONDS", 60, raising=False)
    as_rank = _ranks_sharing_a_store(monkeypatch, store=dist.Store())
    monkeypatch.setenv("SLURM_JOB_ID", "9")
    for rank, match in ((0, "rank 0 of 2 could not share its run ID"), (1, "rank 1 of 2 did not get rank 0's run ID")):
        as_rank(rank)
        with pytest.warns(RuntimeWarning, match=match) as caught:
            assert resolve_run() == tensorquill.RunInfo("run-slurm9", "launcher", rank)
        assert len([w for w in caught if "run ID" in str(w.message)]) == 1

    from torch.distributed import distributed_c10d

    def no_store():
        raise ValueError("Default process group has not been initialized")

    monkeypatch.setattr(distributed_c10d, "_get_default_store", no_store)
    as_rank(1)
    with pytest.warns(RuntimeWarning, match="did not get rank 0's run ID"):
        assert resolve_run() == tensorquill.RunInfo("run-slurm9", "launcher", 1)


def test_rank_zero_passes_over_a_tracker_that_raises(monkeypatch):
    import pytest

    as_rank = _ranks_sharing_a_store(monkeypatch)
    as_rank(0)

    def active_run():
        raise RuntimeError("tracking server unreachable")

    monkeypatch.setitem(sys.modules, "mlflow", types.SimpleNamespace(active_run=active_run))
    monkeypatch.setenv("SLURM_JOB_ID", "9")
    with pytest.warns(RuntimeWarning, match="could not read the tracker's run"):
        assert resolve_run() == tensorquill.RunInfo("run-slurm9", "launcher", 0)
    as_rank(1)
    assert resolve_run() == tensorquill.RunInfo("run-slurm9", "launcher", 1)


def test_tracker_run_without_a_process_group_is_warned(monkeypatch, recwarn):
    # Several ranks and no process group: rank 0's W&B run cannot be shared, so a rank that
    # does not see it (here, one with the launcher's ID) writes under another ID. Say so.
    import pytest

    clear(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "5c1e")
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(run=types.SimpleNamespace(id="w-3")))
    with pytest.warns(RuntimeWarning, match="rank 0 of 2 took the run ID 'w-3' from the W&B run"):
        assert resolve_run() == tensorquill.RunInfo("w-3", "wandb", 0)
    # MLFLOW_RUN_ID is the environment's, which every rank of a job is given: no warning.
    recwarn.clear()
    monkeypatch.setenv("MLFLOW_RUN_ID", "e-3")
    assert resolve_run().source == "env"
    assert not [w for w in recwarn if "cannot be shared" in str(w.message)]


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
