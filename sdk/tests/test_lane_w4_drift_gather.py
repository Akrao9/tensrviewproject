"""Lane W4: the drift check's gather, decided alike on every rank at `init`; revised by lane Z1.

- SDK-22, criterion 1 (lane W4's): every rank skipped a check known at `init` to be incompatible,
  through an agreement at `init` (`_agree_on_gather`, the members' verdicts exchanged through the
  default store). Lane Z1 (Nov 20, 2026, SDK-22 "Yes, approve") took the scheduled gather out of
  the training step and the agreement with it. Removed with it, since what they tested no longer
  exists: the exchange's unit tests (`_exchange_verdicts`, `_agree_on_gather`), the explicit-level
  L5 test (ranks on different device types, which without the agreement leaves one rank of an
  explicit `replica_spread` in the gather alone: the README now says each rank decides alone), and
  the test of tables of different shapes (a gather Gloo would now be given). The session-level
  test is rewritten: the ranks gather nothing on their own.
- SDK-22, the public `replica_spread` on a `meta` or sparse parameter: it raised before the gather
  (so a rank that raised left its peers waiting in it). It now enters the same gather as every
  other rank and gives NaN for a parameter some rank could not read, with a warning that says why.

The tests of several processes start Gloo CPU processes with a 15 s group timeout and a 90 s
limit, so a rank left waiting fails the test rather than hanging the suite. None needs the server
binary. No GPU, NCCL or MPI here: a real mixed CPU and CUDA job is not run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest
import torch

import tensorquill
from tensorquill import distributed as D

needs_dist = pytest.mark.skipif(not torch.distributed.is_available(), reason="needs torch.distributed")

SCRUBBED = (
    "MLFLOW_RUN_ID", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "SLURM_STEP_ID", "RANK", "LOCAL_RANK", "WORLD_SIZE",
    "MASTER_ADDR", "MASTER_PORT", "TENSORQUILL_ENDPOINT", "TENSORQUILL_PROJECT", "TENSORQUILL_RUN_NAME",
    "MLFLOW_EXPERIMENT_NAME",
)

_HEADER = """
import datetime, json, sys, types, warnings
import torch, torch.distributed as dist
from torch import nn

rank, world, store_path, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
plan = json.loads(sys.argv[5])
timeout = datetime.timedelta(seconds=15)
if plan.get("port"):  # a TCPStore, the store `env://` and torchrun give real jobs
    dist.init_process_group(plan.get("backend", "gloo"), init_method=f"tcp://127.0.0.1:{plan['port']}", rank=rank,
                            world_size=world, timeout=timeout)
else:
    dist.init_process_group(plan.get("backend", "gloo"), store=dist.FileStore(store_path, world), rank=rank,
                            world_size=world, timeout=timeout)
found = {"rank": rank}
from tensorquill import distributed as D
"""


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

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
# The explicit gather at the level of `distributed.py`, between steps of a training all_reduce
# over both ranks, so a rank that skipped the gather while the other entered it breaks the next
# one. (Lane W4 also ran it with rank 1's table on `meta`, covered by the agreement lane Z1
# removed; see the module docstring.)

_DEVICES_BODY = """
devices = plan["devices"]
device = torch.device(devices[rank])
params = [torch.ones(3), torch.ones(2, 2)]
agree = getattr(D, "_agree_on_gather", None)  # absent on the base: each rank decides alone
found["raised"] = None
found["codes"] = []
found["reasons"] = []
try:
    if agree is not None:
        agree(dist.group.WORLD, device, params)
    grad = torch.ones(4)
    for step in range(4):
        dist.all_reduce(grad)  # the training run's own collective, every step, every rank
        if step % 2 == 1:
            spread, unread, reasons = D._spread_and_unread(params, dist.group.WORLD, projection=True, device=device)
            found["codes"].append(unread.cpu().tolist() if unread.device.type != "meta" else ["meta"])
            found["reasons"].append(reasons)
    found["steps"] = step + 1
except BaseException as error:
    found["raised"] = f"{type(error).__name__}: {error}"[:300]
"""


@needs_dist
def test_sdk22_ranks_that_agree_still_gather(tmp_path):
    # Ranks that can gather do, in the explicit call (`agree` is absent since lane Z1).
    found = _ranks(tmp_path, 2, _DEVICES_BODY, {"devices": ["cpu", "cpu"]})
    for f in found:
        assert f["raised"] is None, f
        assert f["codes"] == [[0.0, 0.0]] * 2
        assert f["reasons"] == [[None, None]] * 2


# --------------------------------------------------------------------------------------------
# SDK-22 end to end, through `tensorquill.init`. Lane W4's agreement at `init` (`_agree_on_gather`)
# made ranks that would decide differently all skip the scheduled gather. Lane Z1 removed the
# scheduled gather (SDK-22, "Yes, approve"), and the agreement with it: rewritten from
# `test_sdk22_a_session_whose_ranks_decide_differently_skips_on_every_rank`, the same ranks now
# gather nothing on their own, and each writes its own observation.

_SESSION_BODY = """
if rank == 1:
    # Stands in for a session on another device type: lane U5's check says skip on rank 1 only.
    D._gather_problem = lambda group, device: "this rank's table is on a device the group cannot carry"
import tensorquill
torch.manual_seed(0)
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
sink = tensorquill.MemorySink()
found["steps"] = 0
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        session = tensorquill.init(model, run_id="w4", sink=sink, every=4, replica_group=dist.group.WORLD,
                                   replica_every=2, heartbeat_seconds=None, stack_seconds=None)
        for step in range(8):
            opt.zero_grad()
            model(torch.ones(2, 4)).sum().backward()
            for p in model.parameters():
                dist.all_reduce(p.grad)
            opt.step()
            found["steps"] += 1
        session.close()
    except BaseException as error:
        found["raised"] = f"{type(error).__name__}: {error}"[:300]
found["spread_steps"] = [r["step"] for r in sink.records if "replica_spread" in r]
found["observed"] = [r["step"] for r in sink.records if "replica" in r]
found["warnings"] = [str(w.message) for w in caught if "across replicas" in str(w.message)]
"""


@needs_dist
def test_sdk22_a_session_whose_ranks_would_decide_differently_gathers_nothing_on_its_own(tmp_path):
    found = _ranks(tmp_path, 2, _SESSION_BODY, {})
    for f in found:
        assert "raised" not in f, f
        assert f["steps"] == 8
        assert f["spread_steps"] == [] and f["warnings"] == []
        assert f["observed"] == [1, 3, 5, 7]


# --------------------------------------------------------------------------------------------
# The public `replica_spread` on a parameter some rank cannot read.


def _stand_in(monkeypatch, world=2, other=None):
    """A stand-in group of `world` ranks: rank 0's table is this one; `other(table)` gives the other
    ranks' (by default the same). Counts the gathers."""
    import torch.distributed as dist

    calls = []

    def all_gather(out, table, group=None):
        calls.append(tuple(table.shape))
        out[0].copy_(table)
        for o in out[1:]:
            o.copy_(table if other is None else other(table))

    monkeypatch.setattr(dist, "get_world_size", lambda g=None: world)
    monkeypatch.setattr(dist, "all_gather", all_gather)
    return calls


@pytest.mark.parametrize(
    "odd",
    [
        pytest.param(lambda: torch.empty(3, device="meta"), id="meta"),
        pytest.param(lambda: torch.eye(3).to_sparse(), id="sparse_coo"),
        pytest.param(lambda: torch.eye(3).to_sparse_csr(), id="sparse_csr"),
    ],
)
def test_sdk22_public_replica_spread_on_a_parameter_it_cannot_read_gives_nan_and_says_why(monkeypatch, odd):
    # Before: parameter_fingerprints raised before the gather, out of the caller's step, and a
    # rank that raised there left its peers waiting in the gather.
    calls = _stand_in(monkeypatch)
    params = [torch.ones(3), odd(), torch.ones(2, 2)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spread = D.replica_spread(params, object())
    assert calls == [(3, 4)]  # one gather, the same shape whatever was read
    assert spread.shape == (3, 3) and spread.dtype == torch.float64
    assert spread[0].eq(0).all() and spread[2].eq(0).all()
    assert spread[1].isnan().all()
    said = [str(w.message) for w in caught if "replica_spread" in str(w.message)]
    assert len(said) == 1, said
    assert "parameter 1 of 3" in said[0] and D._UNREAD[D._unread_code(params[1]) - 1] in said[0], said


def test_sdk22_public_replica_spread_a_parameter_another_rank_cannot_read_is_nan_on_this_one(monkeypatch):
    def other(table):
        t = table.clone()
        t[0, :3] = 0
        t[0, 3] = 1.0  # rank 1 has it on meta
        return t

    _stand_in(monkeypatch, other=other)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spread = D.replica_spread([torch.ones(3), torch.ones(2)], object())
    assert spread[0].isnan().all() and spread[1].eq(0).all()
    said = [str(w.message) for w in caught if "replica_spread" in str(w.message)]
    assert len(said) == 1 and "another replica" in said[0] and "meta" in said[0]


def test_sdk22_public_replica_spread_still_finds_drift_and_keeps_its_columns(monkeypatch):
    def other(table):
        t = table.clone()
        t[1, 0] += 0.5  # rank 1's second parameter has drifted
        return t

    _stand_in(monkeypatch, other=other)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing to say: every parameter was read
        three = D.replica_spread([torch.ones(3), torch.ones(4)], object())
        two = D.replica_spread([torch.ones(3), torch.ones(4)], object(), projection=False)
    assert three.shape == (2, 3) and two.shape == (2, 2)
    assert three[0].eq(0).all() and three[1, 0] > 0


def test_sdk22_public_replica_spread_on_a_rank_outside_the_group_gathers_nothing(monkeypatch):
    import torch.distributed as dist

    def forbidden(*a, **k):
        raise AssertionError("a collective was issued")

    monkeypatch.setattr(dist, "all_gather", forbidden)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spread = D.replica_spread([torch.ones(3)], dist.GroupMember.NON_GROUP_MEMBER)
    assert spread.shape == (1, 3) and spread.isnan().all()
    assert any("not a member" in str(w.message) for w in caught)


def test_sdk22_public_replica_spread_warning_cannot_raise(monkeypatch):
    _stand_in(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spread = D.replica_spread([torch.empty(2, device="meta")], object())
    assert spread.isnan().all()


def test_sdk22_public_replica_spread_keeps_its_caller_errors():
    # Raised before any collective, alike on every rank that makes the same call.
    with pytest.raises(ValueError):
        D.replica_spread([torch.ones(2)], None)
    with pytest.raises(ValueError):
        D.replica_spread([], object())
