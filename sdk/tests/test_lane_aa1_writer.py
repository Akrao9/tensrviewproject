"""Lane AA1 (wave AA): the writer. Each section names its item and how its tests failed before.

1. The explicit gather calls (`distributed.replica_spread`, `Session.record_replica_spread(group)`)
   get their protection back inside the call (the author's decision 1 on L26): a fixed-size
   exchange first, then every rank gathers or every rank skips, and says why. Brought back in this
   form from lane W4's tests as they stood at `633c9da`: the two L5 tests (ranks on different
   device types), the test of unequal parameter counts, the session-level test, and the
   agreement's unit tests as tests of the decision.

The tests of several processes start Gloo CPU processes with a 15 s group timeout and a 90 s
limit, so a rank left waiting fails the test rather than hanging the suite. No GPU, NCCL or MPI:
a real mix of CPU and CUDA sessions is not run here.
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
    "MLFLOW_EXPERIMENT_NAME", "TORCHELASTIC_RESTART_COUNT", "GROUP_WORLD_SIZE", "LOCAL_WORLD_SIZE",
)

_HEADER = """
import datetime, json, sys, types, warnings
import torch, torch.distributed as dist
from torch import nn

rank, world, store_path, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
plan = json.loads(sys.argv[5])
timeout = datetime.timedelta(seconds=15)
if plan.get("port"):  # a TCPStore, the store `env://` and torchrun give real jobs
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{plan['port']}", rank=rank,
                            world_size=world, timeout=timeout)
else:
    dist.init_process_group("gloo", store=dist.FileStore(store_path, world), rank=rank,
                            world_size=world, timeout=timeout)
found = {"rank": rank}
from tensorquill import distributed as D
"""

_FOOTER = """
with open(out, "w") as file:
    json.dump(found, file)
dist.destroy_process_group()
"""


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ranks(tmp_path: Path, world: int, body: str, plan: dict | None = None) -> list[dict]:
    plan = plan or {}
    script = tmp_path / "worker.py"
    script.write_text(_HEADER + textwrap.dedent(body) + _FOOTER)
    env = {k: v for k, v in os.environ.items() if k not in SCRUBBED}
    env["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    source = str(Path(tensorquill.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([source, *filter(None, [os.environ.get("PYTHONPATH")])])
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
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
        assert result.exists(), f"rank {rank} wrote nothing (exit {procs[rank].returncode}); its output:\n" + logs[rank][-3000:]
        found.append(json.loads(result.read_text()))
    return found


# =============================================================================================
# Item 1. The explicit gather, protected inside the call.
#
# Every body below runs the training run's own all_reduce over every rank after the explicit call,
# so a rank that skipped the gather while another entered it breaks that all_reduce (or the 90 s
# limit) rather than passing quietly.

_DEVICES_BODY = """
device = torch.device(plan["devices"][rank])
params = [torch.ones(3), torch.ones(2, 2)]
found["raised"] = None
found["codes"] = []
found["reasons"] = []
try:
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
@pytest.mark.parametrize("store", ["file", "tcp"])
def test_l5_ranks_on_different_device_types_all_skip_the_gather(tmp_path, store):
    # Lane W4's test at `633c9da`, without the agreement at `init`. Rank 1's table is on `meta`,
    # for which gloo has no backend, so rank 1 cannot gather. Before (on `2e2f4dc`): rank 1 skipped
    # alone and rank 0 entered the gather, which failed after the group's 15 s timeout.
    plan = {"devices": ["cpu", "meta"], **({"port": _free_port()} if store == "tcp" else {})}
    found = _ranks(tmp_path, 2, _DEVICES_BODY, plan)
    for f in found:
        assert f["raised"] is None, f
        assert f["steps"] == 4
        assert len(f["reasons"]) == 2
        for reasons in f["reasons"]:
            assert all(r and "gather cannot run" in r for r in reasons), reasons
    # Rank 0 is told why: rank 1's verdict, which it learned in the exchange.
    assert "rank 1" in found[0]["reasons"][0][0] and "meta" in found[0]["reasons"][0][0], found[0]["reasons"]
    assert found[0]["codes"] == [[float(D._NOT_GATHERED)] * 2] * 2


_COUNTS_BODY = """
params = [torch.ones(3)] * (2 if rank == 0 else 3)
found["raised"] = None
try:
    spread, unread, reasons = D._spread_and_unread(params, dist.group.WORLD, projection=True,
                                                   device=torch.device("cpu"))
    found["codes"] = unread.tolist()
    found["reasons"] = reasons
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        public = D.replica_spread(params, dist.group.WORLD)
    found["public_nan"] = bool(public.isnan().all())
    found["public_said"] = [str(w.message) for w in caught]
    dist.all_reduce(torch.ones(2))
except BaseException as error:
    found["raised"] = f"{type(error).__name__}: {error}"[:300]
"""


@needs_dist
def test_unequal_parameter_counts_every_rank_skips_and_says_why(tmp_path):
    # Lane W4's test of tables of different shapes, brought back. Before (on `2e2f4dc`): the ranks
    # entered a Gloo all_gather of tables of different shapes, which does not complete.
    found = _ranks(tmp_path, 2, _COUNTS_BODY)
    for f in found:
        assert f["raised"] is None, f
        assert set(f["codes"]) == {float(D._NOT_GATHERED)}
        assert all("2 on rank 0" in r and "3 on rank 1" in r for r in f["reasons"]), f["reasons"]
        assert f["public_nan"]
        assert any("2 on rank 0" in s and "3 on rank 1" in s for s in f["public_said"]), f["public_said"]


@needs_dist
def test_ranks_that_agree_still_gather_and_find_drift(tmp_path):
    # The exchange must not switch a working check off, and the gather after it still compares.
    body = """
    params = [torch.ones(3), torch.ones(4) * (1.0 + 0.5 * rank)]
    found["raised"] = None
    try:
        spread = D.replica_spread(params, dist.group.WORLD)
        found["spread"] = spread.tolist()
        dist.all_reduce(torch.ones(2))
    except BaseException as error:
        found["raised"] = f"{type(error).__name__}: {error}"[:300]
    """
    found = _ranks(tmp_path, 3, body)
    for f in found:
        assert f["raised"] is None, f
        assert f["spread"][0] == [0.0, 0.0, 0.0]
        assert f["spread"][1][0] > 0.1


@needs_dist
def test_mixed_projection_settings_every_rank_skips(tmp_path):
    # A third column on one rank only: tables of different widths, the same failure as unequal counts.
    body = """
    params = [torch.ones(3), torch.ones(2)]
    found["raised"] = None
    try:
        spread, unread, reasons = D._spread_and_unread(params, dist.group.WORLD, projection=rank == 0,
                                                       device=torch.device("cpu"))
        found["codes"] = unread.tolist()
        found["reasons"] = reasons
        dist.all_reduce(torch.ones(2))
    except BaseException as error:
        found["raised"] = f"{type(error).__name__}: {error}"[:300]
    """
    found = _ranks(tmp_path, 2, body)
    for f in found:
        assert f["raised"] is None, f
        assert set(f["codes"]) == {float(D._NOT_GATHERED)}
        assert all("projection" in r for r in f["reasons"]), f["reasons"]


_SESSION_BODY = """
mode = plan["mode"]
if mode == "verdict" and rank == 1:
    # Stands in for a session whose rank 1 cannot gather (another device type, say): the check that
    # rank 1 runs on its own says skip, and rank 0's says gather.
    D._gather_problem = lambda group, device: "this rank's table is on a device the group cannot carry"
import tensorquill
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
opt = torch.optim.SGD(model.parameters(), lr=0.01)
sink = tensorquill.MemorySink()
found["steps"] = 0
replica_params = None
if mode == "counts" and rank == 1:
    replica_params = lambda name, p: name.startswith("0.")  # 2 parameters on rank 1, 4 on rank 0
if mode == "none" and rank == 1:
    replica_params = lambda name, p: False  # compares nothing: still takes part
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        kwargs = {} if replica_params is None else {"replica_params": replica_params}
        session = tensorquill.init(model, run_id="aa1", sink=sink, every=4, replica_group=dist.group.WORLD,
                                   replica_every=1000, heartbeat_seconds=None, stack_seconds=None, **kwargs)
        for step in range(8):
            opt.zero_grad()
            model(torch.ones(2, 4)).sum().backward()
            for p in model.parameters():
                dist.all_reduce(p.grad)
            if step in (2, 5):
                session.record_replica_spread(dist.group.WORLD)
            opt.step()
            found["steps"] += 1
        session.close()
    except BaseException as error:
        found["raised"] = f"{type(error).__name__}: {error}"[:300]
found["spread_steps"] = [r["step"] for r in sink.records if "replica_spread" in r]
found["warnings"] = [str(w.message) for w in caught if "across replicas" in str(w.message)]
"""


@needs_dist
def test_a_session_whose_ranks_decide_differently_skips_on_every_rank(tmp_path):
    # Lane W4's session-level test, on the explicit call. Before (on `2e2f4dc`): rank 1 skipped,
    # rank 0 waited in the gather and raised after the group's timeout.
    found = _ranks(tmp_path, 2, _SESSION_BODY, {"mode": "verdict"})
    for f in found:
        assert "raised" not in f, f
        assert f["steps"] == 8
        assert f["spread_steps"] == []
        assert f["warnings"] and all("cannot carry" in w for w in f["warnings"]), f


@needs_dist
def test_a_session_whose_ranks_compare_different_numbers_of_parameters_skips_on_every_rank(tmp_path):
    found = _ranks(tmp_path, 2, _SESSION_BODY, {"mode": "counts"})
    for f in found:
        assert "raised" not in f, f
        assert f["steps"] == 8 and f["spread_steps"] == []
        assert f["warnings"] and all("4 on rank 0" in w and "2 on rank 1" in w for w in f["warnings"]), f


@needs_dist
def test_a_session_that_compares_no_parameter_still_takes_part(tmp_path):
    # A rank whose `replica_params` choose nothing returned before the gather, leaving its peers in it.
    found = _ranks(tmp_path, 2, _SESSION_BODY, {"mode": "none"})
    for f in found:
        assert "raised" not in f, f
        assert f["steps"] == 8 and f["spread_steps"] == []
    assert found[0]["warnings"] and all("rank 1" in w for w in found[0]["warnings"]), found[0]


@needs_dist
def test_a_session_whose_ranks_agree_records_the_spread(tmp_path):
    found = _ranks(tmp_path, 2, _SESSION_BODY, {"mode": "agree"})
    for f in found:
        assert "raised" not in f, f
        assert f["spread_steps"] == [2, 5] and f["warnings"] == []


@needs_dist
def test_the_public_call_with_no_parameter_on_one_rank_takes_part_then_raises(tmp_path):
    body = """
    params = [torch.ones(3)] if rank == 0 else []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            found["spread"] = D.replica_spread(params, dist.group.WORLD).isnan().all().item()
        found["said"] = [str(w.message) for w in caught]
    except ValueError as error:
        found["value_error"] = str(error)
    dist.all_reduce(torch.ones(2))
    found["done"] = True
    """
    rank0, rank1 = _ranks(tmp_path, 2, body)
    assert rank0["done"] and rank1["done"]
    assert rank0["spread"] is True and any("rank 1" in s for s in rank0["said"]), rank0
    assert "no replicated parameters" in rank1["value_error"]


@needs_dist
def test_a_sub_group_with_unequal_counts_skips_on_its_members_and_names_global_ranks(tmp_path):
    # The docs steer users to a replication sub-group: the exchange runs on it, the decision names
    # global ranks, and a rank outside the sub-group returns at once.
    body = """
    sub = dist.new_group([1, 2])
    params = [torch.ones(3)] * (rank + 1)  # 2 on rank 1, 3 on rank 2
    found["raised"] = None
    try:
        spread, unread, reasons = D._spread_and_unread(params, sub, projection=True, device=torch.device("cpu"))
        found["codes"] = unread.tolist()
        found["reasons"] = reasons
        dist.all_reduce(torch.ones(2))  # WORLD, after it
    except BaseException as error:
        found["raised"] = f"{type(error).__name__}: {error}"[:300]
    """
    rank0, rank1, rank2 = _ranks(tmp_path, 3, body)
    for f in (rank0, rank1, rank2):
        assert f["raised"] is None, f
    assert all("not a member" in r for r in rank0["reasons"])
    for f in (rank1, rank2):
        assert set(f["codes"]) == {float(D._NOT_GATHERED)}
        assert all("2 on rank 1" in r and "3 on rank 2" in r for r in f["reasons"]), f["reasons"]


# The decision, from the exchanged rows alone: the same on every rank, since every rank sees them all.


def _row(count=2, width=4, device="cpu", problem=None):
    return D._exchange_row(count, width, torch.device(device), problem)


def test_every_rank_ready_gives_one_yes():
    assert D._decide_gather([_row(), _row()], [0, 1]) is None


def test_one_rank_that_cannot_gather_turns_the_check_off_for_all():
    why = D._decide_gather([_row(), _row(device="meta", problem="the group has no backend for meta tensors")], [0, 1])
    assert why is not None and "rank 1" in why and "meta" in why and "no backend" in why


def test_l5_different_device_types_turn_the_check_off():
    # Lane W4's test of the decision, with the rows of the exchange in place of the store's verdicts.
    why = D._decide_gather([_row(device="cpu"), _row(device="cuda"), _row(device="cuda")], [0, 1, 2])
    assert why is not None and "different device types" in why
    assert "cpu on rank 0" in why and "cuda on ranks 1, 2" in why


def test_the_decision_names_global_ranks_and_counts():
    why = D._decide_gather([_row(count=2), _row(count=3)], [4, 6])
    assert "2 on rank 4" in why and "3 on rank 6" in why


def test_a_row_from_another_exchange_version_turns_the_check_off():
    odd = _row().clone()
    odd[0] = 99
    why = D._decide_gather([_row(), odd], [0, 1])
    assert why is not None and "rank 1" in why


def test_the_row_has_one_fixed_shape_whatever_the_inputs():
    shapes = {tuple(_row(count=c, width=w, device=d, problem=p).shape)
              for c in (0, 1, 10**6) for w in (3, 4) for d in ("cpu", "meta") for p in (None, "x" * 1000)}
    assert shapes == {(D._ROW_WIDTH,)}


def test_a_stand_in_group_takes_part_in_no_exchange(monkeypatch):
    # Tests and tools pass a group that is not a ProcessGroup with their own all_gather: one gather.
    import torch.distributed as dist

    calls = []

    def all_gather(out, table, group=None):
        calls.append(tuple(table.shape))
        for o in out:
            o.copy_(table)

    monkeypatch.setattr(dist, "get_world_size", lambda g=None: 2)
    monkeypatch.setattr(dist, "all_gather", all_gather)
    spread = D.replica_spread([torch.ones(3), torch.ones(2)], object())
    assert calls == [(2, 4)] and spread.eq(0).all()


def test_the_checks_before_the_exchange_never_raise(monkeypatch):
    # An API this PyTorch does not have decides nothing: the gather goes on as before.
    import torch.distributed as dist

    def boom(*a, **k):
        raise RuntimeError("anything")

    calls = []

    def all_gather(out, table, group=None):
        calls.append(tuple(table.shape))
        for o in out:
            o.copy_(table)

    monkeypatch.setattr(dist.distributed_c10d, "_rank_not_in_group", boom)
    monkeypatch.setattr(dist, "get_world_size", lambda g=None: 2)
    monkeypatch.setattr(dist, "all_gather", all_gather)
    assert D._exchange_device(object()) is None
    assert D.replica_spread([torch.ones(3)], object()).eq(0).all() and calls == [(1, 4)]


def test_a_long_reason_is_cut_to_the_rows_text_field():
    why = D._decide_gather([_row(), _row(problem="é" * 400)], [0, 1])
    assert why.startswith("rank 1 of the replica group cannot take part: é") and len(why) < 300
