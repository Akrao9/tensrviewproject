"""Lane U5: a broken tracker or logger must not fail `init`, and a drift-check gather that cannot
run must not raise into training.

- C05 (outside audit): on one process with no `run_id`, `mlflow.active_run()` raising came out of
  `resolve_run` and so out of `init`; reading a W&B run's ID, or `wandb.run` itself, could raise
  the same way. Rank 0 of a distributed run already passed a failing tracker over. Now every
  path takes the same guarded lookup, and its warning cannot raise under `-W error`.
- SDK-22: the session's drift check gathered with `dist.all_gather` on a group that could not
  carry it, and the gather's error came out of `optimizer.step()`. Real cases on CPU: a group
  with no backend for the table's device (as NCCL, which serves CUDA only, given CPU tensors),
  and a rank outside the replica group. The check is now skipped where every rank of the group
  would skip it alike, and issues no collective there, so no peer is left waiting. A gather that
  raises at run time on one rank only is still not caught: a catch there was measured to leave
  the peers waiting a whole group timeout and the next training collective failing (lane U5's
  notes, `lane-u5-catch-one-rank.py`).
- `lightning.py`: a logger property that raised aborted rank 0's `on_fit_start`, before the
  other ranks heard its run ID.

The multi-rank tests start Gloo CPU processes with a 15 s group timeout and a 90 s limit on the
processes, so a hang fails the test rather than hanging the suite. None needs the server binary.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import distributed as D
from tensorquill import tracker

# Whatever the parent's environment says about a run, a launcher or ranks must not reach a case.
SCRUBBED = (
    "MLFLOW_RUN_ID", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "SLURM_STEP_ID", "RANK", "LOCAL_RANK", "WORLD_SIZE",
    "MASTER_ADDR", "MASTER_PORT", "TENSORQUILL_ENDPOINT", "TENSORQUILL_PROJECT", "TENSORQUILL_RUN_NAME",
    "MLFLOW_EXPERIMENT_NAME",
)


@pytest.fixture
def clean(monkeypatch):
    for name in SCRUBBED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    return monkeypatch


def _init(**kwargs):
    return tensorquill.init(
        nn.Linear(2, 2), sink=tensorquill.MemorySink(), heartbeat_seconds=None, stack_seconds=None, **kwargs
    )


def _mlflow_raising():
    def active_run():
        raise RuntimeError("tracker temporarily down")

    return types.SimpleNamespace(active_run=active_run)


def _mlflow_run_id_raising():
    class Info:
        run_name = "name-m"

        @property
        def run_id(self):
            raise RuntimeError("run_id unreadable")

    return types.SimpleNamespace(active_run=lambda: types.SimpleNamespace(info=Info()))


class _WandbRunIdRaises:
    project, name, group = "proj-w", "name-w", "group-w"

    @property
    def id(self):
        raise RuntimeError("wandb run id unreadable")


class _WandbModuleRunRaises(types.ModuleType):
    def __getattr__(self, name):
        if name == "run":
            raise RuntimeError("wandb.run unreadable")
        raise AttributeError(name)


# --------------------------------------------------------------------------------------------
# C05: one process, no run_id, a tracker that raises.


@pytest.mark.parametrize(
    "tracker_module",
    [
        pytest.param(("mlflow", _mlflow_raising()), id="mlflow-active_run-raises"),
        pytest.param(("mlflow", _mlflow_run_id_raising()), id="mlflow-run_id-raises"),
        pytest.param(("wandb", types.SimpleNamespace(run=_WandbRunIdRaises())), id="wandb-run.id-raises"),
        pytest.param(("wandb", _WandbModuleRunRaises("wandb")), id="wandb.run-raises"),
    ],
)
def test_c05_a_tracker_that_raises_gives_a_generated_run_id_and_one_warning(clean, tracker_module):
    name, module = tracker_module
    clean.setitem(sys.modules, name, module)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = _init()
    session.close()
    assert session.run.source == "generated"
    assert session.run.run_id.startswith("run-")
    said = [str(w.message) for w in caught if "could not read the tracker's run" in str(w.message)]
    assert len(said) == 1, [str(w.message) for w in caught]
    assert "on rank 0" in said[0] and "RuntimeError" in said[0]


def test_c05_a_tracker_that_raises_falls_through_to_the_launchers_id(clean):
    clean.setitem(sys.modules, "mlflow", _mlflow_raising())
    clean.setenv("TORCHELASTIC_RUN_ID", "abc123")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = tracker.resolve_run()
    assert (info.run_id, info.source) == ("run-abc123", "launcher")


def test_c05_a_project_or_name_that_raises_is_left_out(clean):
    class Run:
        id, group = "wandb-abc", "group-w"

        @property
        def project(self):
            raise RuntimeError("project unreadable")

        @property
        def name(self):
            raise RuntimeError("name unreadable")

    clean.setitem(sys.modules, "wandb", types.SimpleNamespace(run=Run()))
    session = _init()
    session.close()
    assert (session.run.run_id, session.run.source) == ("wandb-abc", "wandb")
    assert tracker.tracker_labels() == {"group": "group-w"}


def test_c05_tracker_labels_never_raises(clean):
    clean.setitem(sys.modules, "wandb", _WandbModuleRunRaises("wandb"))
    clean.setitem(sys.modules, "mlflow", _mlflow_raising())
    assert tracker.tracker_labels() == {}


def test_c05_under_warnings_as_errors_init_still_returns(clean, caplog):
    # `python -W error`: the warning that names the failing tracker must not become the raise;
    # it is logged on the `tensorquill` logger instead.
    clean.setitem(sys.modules, "mlflow", _mlflow_raising())
    with caplog.at_level("WARNING", logger="tensorquill"), warnings.catch_warnings():
        warnings.simplefilter("error")
        session = _init()
    session.close()
    assert session.run.source == "generated"
    assert "could not read the tracker's run" in caplog.text


def test_c05_the_several_ranks_warnings_are_kept(clean):
    # Several ranks and no process group: a rank's own ID cannot be shared, and init says so.
    clean.setenv("WORLD_SIZE", "2")
    clean.setenv("RANK", "0")
    clean.setitem(sys.modules, "mlflow", _mlflow_raising())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        info = tracker.resolve_run()
    messages = [str(w.message) for w in caught]
    assert info.source == "generated"
    assert any("could not read the tracker's run" in m for m in messages), messages
    assert any("found no run ID the ranks share" in m for m in messages), messages

    clean.setitem(sys.modules, "wandb", types.SimpleNamespace(run=types.SimpleNamespace(id="wandb-abc")))
    clean.delitem(sys.modules, "mlflow")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        info = tracker.resolve_run()
    assert (info.run_id, info.source) == ("wandb-abc", "wandb")
    assert any("cannot be shared" in str(w.message) for w in caught), [str(w.message) for w in caught]


def test_c05_the_single_process_path_takes_the_guarded_lookup(clean):
    # The one process path and rank 0's resolve through the same function.
    seen = []
    real = tracker._resolve_here

    def spy(*args, **kwargs):
        seen.append(args)
        return real(*args, **kwargs)

    clean.setattr(tracker, "_resolve_here", spy)
    tracker.resolve_run()
    tracker.resolve_run("given")
    assert len(seen) == 2


# --------------------------------------------------------------------------------------------
# Helpers for the tests of several processes.

_HEADER = """
import datetime, json, sys, types, warnings
import torch, torch.distributed as dist
from torch import nn

rank, world, store_path, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
plan = json.loads(sys.argv[5])
timeout = datetime.timedelta(seconds=15)
dist.init_process_group(plan.get("backend", "gloo"), store=dist.FileStore(store_path, world), rank=rank,
                        world_size=world, timeout=timeout)
found = {"rank": rank}
"""

_FOOTER = """
with open(out, "w") as file:
    json.dump(found, file)
dist.destroy_process_group()
"""


def _ranks(tmp_path: Path, world: int, body: str, plan: dict) -> list[dict]:
    script = tmp_path / "worker.py"
    script.write_text(_HEADER + textwrap.dedent(body) + _FOOTER)
    env = {k: v for k, v in os.environ.items() if k not in SCRUBBED}
    env["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    # The ranks import the tensorquill this test imported, not an installed copy.
    source = str(Path(tensorquill.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([source, *filter(None, [os.environ.get("PYTHONPATH")])])
    work = tmp_path / "work"
    work.mkdir()
    store = str(tmp_path / "store")
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(rank), str(world), store, str(tmp_path / f"rank{rank}.json"),
             json.dumps(plan)],
            cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for rank in range(world)
    ]
    logs = []
    try:
        for proc in procs:
            logs.append(proc.communicate(timeout=90)[0])
    except subprocess.TimeoutExpired:
        for proc in procs:
            proc.kill()
        pytest.fail(f"a rank did not finish within 90 s: one was left waiting ({plan})")
    found = []
    for rank in range(world):
        result = tmp_path / f"rank{rank}.json"
        assert result.exists(), f"rank {rank} wrote nothing; its output:\n{logs[rank][-3000:]}"
        found.append(json.loads(result.read_text()))
    return found


# --------------------------------------------------------------------------------------------
# C05 on two ranks: warnings as errors, a tracker that raises on rank 0.

_TRACKER_BODY = """
if rank == 0:
    def active_run():
        raise RuntimeError("tracking server unreachable")
    sys.modules["mlflow"] = types.SimpleNamespace(active_run=active_run)
import tensorquill
try:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        session = tensorquill.init(nn.Linear(2, 2), sink=tensorquill.MemorySink(), heartbeat_seconds=None,
                                   stack_seconds=None)
    found["run"] = [session.run.run_id, session.run.source]
    session.close()
except BaseException as error:
    found["raised"] = f"{type(error).__name__}: {error}"[:300]
"""


def test_c05_two_ranks_under_warnings_as_errors_share_rank_zeros_generated_id(tmp_path):
    # Before: rank 0's warning about the tracker raised under `-W error` before it shared its ID,
    # so rank 0's init failed and rank 1 waited out the group's timeout.
    found = _ranks(tmp_path, 2, _TRACKER_BODY, {})
    assert all("raised" not in f for f in found), found
    assert found[0]["run"] == found[1]["run"]
    assert found[0]["run"][1] == "generated"


# --------------------------------------------------------------------------------------------
# SDK-22: the drift check's gather on a group that cannot carry it.

_GATHER_BODY = """
group = dist.new_group(plan["members"]) if plan.get("members") else dist.group.WORLD
import tensorquill
torch.manual_seed(0)  # the same weights on every rank: replicas that agree
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
sink = tensorquill.MemorySink()
found["steps"] = 0
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        session = tensorquill.init(model, run_id="u5", sink=sink, every=4, replica_group=group, replica_every=2,
                                   heartbeat_seconds=None, stack_seconds=None)
        for step in range(8):
            opt.zero_grad()
            model(torch.ones(2, 4)).sum().backward()
            if plan.get("all_reduce"):
                # Stands in for DDP's gradient all_reduce: the training run's own collective on
                # every rank and every step, which a rank that skipped or added one would break.
                for p in model.parameters():
                    dist.all_reduce(p.grad)
            opt.step()
            found["steps"] += 1
        session.close()
    except BaseException as error:
        found["raised"] = f"{type(error).__name__}: {error}"[:300]
found["spread_steps"] = [r["step"] for r in sink.records if "replica_spread" in r]
found["spreads"] = [r["replica_spread"] for r in sink.records if "replica_spread" in r]
found["warnings"] = [str(w.message) for w in caught if "across replicas" in str(w.message)]
found["torch_warnings"] = [str(w.message) for w in caught if "does not belong to the given group" in str(w.message)]
"""


@pytest.mark.skipif(not torch.distributed.is_available(), reason="needs torch.distributed")
def test_sdk22_a_group_with_no_backend_for_the_tables_device_skips_on_every_rank(tmp_path):
    # Gloo registered for CUDA tensors only: like NCCL, which serves CUDA only, given CPU tensors.
    # Before: "No backend type associated with device type cpu" came out of optimizer.step() on
    # both ranks at the first scheduled check.
    found = _ranks(tmp_path, 2, _GATHER_BODY, {"backend": "cuda:gloo"})
    for f in found:
        assert "raised" not in f, f
        assert f["steps"] == 8
        assert f["spread_steps"] == []  # no record claims a comparison
        assert f["warnings"], f
        assert all("no backend for cpu tensors" in w for w in f["warnings"]), f["warnings"]
    # Named once per parameter, the same on every rank.
    assert sorted(found[0]["warnings"]) == sorted(found[1]["warnings"])
    assert len(found[0]["warnings"]) == 2


@pytest.mark.skipif(not torch.distributed.is_available(), reason="needs torch.distributed")
def test_sdk22_a_rank_outside_the_replica_group_skips_and_its_peers_are_not_left_waiting(tmp_path):
    # Three ranks; the replica group is ranks 0 and 1. Before: rank 2's gather on a group it is not
    # in returned nothing, and `torch.stack([])` came out of its optimizer.step(); rank 2 then
    # stopped issuing the training all_reduce, and ranks 0 and 1 waited in it.
    found = _ranks(tmp_path, 3, _GATHER_BODY, {"members": [0, 1], "all_reduce": True})
    for f in found:
        assert "raised" not in f, f
        assert f["steps"] == 8
        assert f["torch_warnings"] == []  # no all_gather was issued on a group the rank is not in
    for f in found[:2]:
        assert f["spread_steps"] == [1, 3, 5, 7]
        assert all(v == [0.0, 0.0, 0.0] for spread in f["spreads"] for v in spread.values())
        assert f["warnings"] == []
    assert found[2]["spread_steps"] == []
    assert found[2]["warnings"] and all("not a member" in w for w in found[2]["warnings"]), found[2]


def test_sdk22_the_check_is_decided_without_a_collective(monkeypatch):
    # A rank outside the group: no all_gather, every row flagged with the same reason.
    import torch.distributed as dist

    def forbidden(*args, **kwargs):
        raise AssertionError("the drift check issued a collective")

    monkeypatch.setattr(dist, "all_gather", forbidden)
    monkeypatch.setattr(dist, "get_world_size", forbidden)
    params = [torch.ones(3), torch.ones(2, 2)]
    spread, unread, reasons = D._spread_and_unread(params, dist.GroupMember.NON_GROUP_MEMBER, projection=True)
    assert spread.shape == (2, 3) and spread.dtype == torch.float64 and not spread.any()
    assert unread.tolist() == [float(D._NOT_GATHERED)] * 2
    assert len(set(reasons)) == 1 and "not a member" in reasons[0]
    assert D._unread_reason(D._NOT_GATHERED).startswith("the drift check's gather cannot run")


def test_sdk22_existing_codes_keep_their_numbers():
    # The codes travel between ranks: every rank must read the same number the same way.
    assert D._UNREAD[0] == "it is on meta, which holds no values"
    assert D._unread_reason(D._RAISED) == "reading it raised an error"
    assert D._RAISED == 9 and D._NOT_GATHERED == 10
    assert D._unread_reason(99) == "reading it raised an error"


def test_sdk22_a_stand_in_group_is_still_gathered(monkeypatch):
    # Tests and tools fake `all_gather` with a group that is not a ProcessGroup: it is gathered.
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_world_size", lambda g=None: 2)
    monkeypatch.setattr(dist, "all_gather", lambda out, t, group=None: [o.copy_(t) for o in out])
    spread, unread, reasons = D._spread_and_unread([torch.ones(3)], object(), projection=True)
    assert unread.tolist() == [0.0] and reasons == [None]


# --------------------------------------------------------------------------------------------
# Lightning: a logger property that raises.

try:
    import lightning.pytorch as L
except ImportError:  # the core SDK does not need Lightning: only these tests are skipped
    L = None

needs_lightning = pytest.mark.skipif(L is None, reason="needs lightning")

if L is not None:
    from tensorquill.lightning import TensorQuillCallback, _logger_run_id

    class Module(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4, 1)

        def training_step(self, batch, batch_idx):
            return self.net(batch).pow(2).mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

        def train_dataloader(self):
            torch.manual_seed(0)
            return torch.utils.data.DataLoader(torch.randn(32, 4), batch_size=8)


    class BrokenLogger(L.loggers.Logger):
        """A logger whose `run_id` raises, as `MLFlowLogger.run_id` can when the tracking server is
        unreachable (it creates the run on first use)."""

        name, version = "broken", "0"

        @property
        def run_id(self):
            raise RuntimeError("tracking server unreachable")

        def log_metrics(self, metrics, step=None):
            pass

        def log_hyperparams(self, params, *args, **kwargs):
            pass


    class GoodLogger(BrokenLogger):
        name = "good"

        @property
        def run_id(self):
            return "mlflow-run-9"


def _fit(tmp_path, loggers, **kwargs):
    trainer = L.Trainer(
        default_root_dir=tmp_path, max_steps=2, callbacks=[TensorQuillCallback()], logger=loggers,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False, accelerator="cpu",
        **kwargs,
    )
    trainer.fit(Module())
    return sorted(p.name for p in (tmp_path / "tensorquill-out").glob("*.jsonl"))


@needs_lightning
def test_lightning_a_logger_that_raises_does_not_abort_the_fit(tmp_path, clean):
    clean.chdir(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        files = _fit(tmp_path, [BrokenLogger()], devices=1)
    assert len(files) == 1 and files[0].startswith("run-") and files[0].endswith(".rank0.jsonl")
    said = [str(w.message) for w in caught if "could not read the run ID of the trainer's logger" in str(w.message)]
    assert len(said) == 1 and "BrokenLogger" in said[0] and "RuntimeError" in said[0]


@needs_lightning
def test_lightning_a_logger_that_raises_is_passed_over_for_the_next(tmp_path, clean):
    clean.chdir(tmp_path)
    with warnings.catch_warnings():
        # The callback's warning cannot become the raise (Lightning's own warnings are left alone).
        warnings.filterwarnings("error", message="TensorQuillCallback")
        files = _fit(tmp_path, [BrokenLogger(), GoodLogger()], devices=1)
    assert files == ["mlflow-run-9.rank0.jsonl"]


@needs_lightning
def test_lightning_loggers_that_cannot_be_listed_or_read_give_none():
    class Trainer:
        @property
        def loggers(self):
            raise RuntimeError("no loggers")

    class WandbLogger:
        @property
        def version(self):
            raise RuntimeError("wandb.init failed")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _logger_run_id(Trainer()) is None
        assert _logger_run_id(types.SimpleNamespace(loggers=[WandbLogger()])) is None


@needs_lightning
@pytest.mark.skipif(not torch.distributed.is_available(), reason="needs torch.distributed")
def test_lightning_a_logger_that_raises_on_rank_zero_still_gives_every_rank_its_id(tmp_path, clean):
    # Before: rank 0's on_fit_start raised before it shared a run ID, and the fit failed.
    from datetime import timedelta

    from lightning.pytorch.strategies import DDPStrategy

    clean.chdir(tmp_path)
    clean.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    files = _fit(
        tmp_path, [BrokenLogger()], devices=2,
        strategy=DDPStrategy(start_method="spawn", timeout=timedelta(seconds=20)),
    )
    assert len(files) == 2, files
    run_ids = {name.rsplit(".rank", 1)[0] for name in files}
    assert len(run_ids) == 1 and next(iter(run_ids)).startswith("run-"), files
