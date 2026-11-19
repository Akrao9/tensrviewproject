"""Every rank of a `torch.distributed` job resolves the same run ID (review finding SDK-01, lane T1).

A tracker's run usually exists on rank 0 only. Each case starts two Gloo CPU processes, gives
rank 0 alone a stand-in W&B run when the case has one, and calls `tensorquill.init` (or the
Hugging Face callback) on both.

- SDK-01: before lane R3, with no launcher ID, rank 0 took the W&B run's ID and entered no
  collective while rank 1 waited in `broadcast_object_list` for it; with `TORCHELASTIC_RUN_ID`
  set, rank 1 took the launcher's ID instead of the W&B run's.
- Lane T1: the ID is shared through the default process group's store, never by a collective,
  so under NCCL nothing lands on a GPU. Every collective that could carry it is made to raise in
  the worker while TensorQuill runs. A `run_id=` given on some ranks only takes the same path
  as none (rank 0's wins), and a rank that never hears from rank 0 falls back to an ID of its
  own after a bounded wait, with one warning.

A short group timeout and a timeout on the processes make a hang fail the test rather than
hang the suite.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import tensorquill

WORKER = textwrap.dedent(
    """
    import datetime, json, sys, time, types, warnings
    import torch, torch.distributed as dist

    rank, store_arg, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    plan = json.loads(sys.argv[4])
    timeout = datetime.timedelta(seconds=15)
    if store_arg.startswith("tcp:"):  # the store torchrun's ranks use
        store = dist.TCPStore("127.0.0.1", int(store_arg[4:]), 2, is_master=rank == 0, timeout=timeout)
    else:
        store = dist.FileStore(store_arg, 2)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=2, timeout=timeout)
    if plan.get("hf"):
        # Imported before the stand-in: Transformers looks up wandb's import spec.
        from tensorquill.huggingface import TensorQuillCallback
    if plan.get("wandb") and rank == 0:
        run = types.SimpleNamespace(id="wandb-abc", project="proj-w", name="name-w", group="group-w")
        sys.modules["wandb"] = types.SimpleNamespace(run=run)
    if plan.get("mlflow_raises") and rank == 0:
        def active_run():
            raise RuntimeError("tracking server unreachable")
        sys.modules["mlflow"] = types.SimpleNamespace(active_run=active_run)
    import tensorquill
    import tensorquill.tracker

    if plan.get("wait_seconds") is not None:
        tensorquill.tracker.RUN_ID_WAIT_SECONDS = plan["wait_seconds"]

    # Sharing the run ID must make no collective call: under NCCL each one puts tensors on
    # torch.cuda.current_device(), GPU 0 on every rank that never called set_device.
    FORBIDDEN = ("broadcast_object_list", "all_gather_object", "gather_object", "scatter_object_list",
                 "broadcast", "all_gather", "all_reduce", "barrier")
    originals = {name: getattr(dist, name) for name in FORBIDDEN}

    def forbid(name):
        def call(*args, **kwargs):
            raise AssertionError(f"tensorquill made the collective call {name}")
        return call

    for name in FORBIDDEN:
        setattr(dist, name, forbid(name))
    found = {"rank": rank, "runs": []}
    given = plan.get("run_id", {}).get(str(rank))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(plan.get("inits", 1)):
                if rank == 0 and plan.get("rank0_skips"):
                    break
                if rank == 0:
                    time.sleep(plan.get("rank0_delay", 0))
                started = time.monotonic()
                kwargs = {} if given is None else {"run_id": given}
                if plan.get("hf"):
                    callback = TensorQuillCallback(**kwargs)
                    callback.on_train_begin(types.SimpleNamespace(output_dir="."), types.SimpleNamespace(global_step=0),
                                            None, model=torch.nn.Linear(4, 4))
                    session = callback.session
                else:
                    session = tensorquill.init(torch.nn.Linear(4, 4), **kwargs)
                found["runs"].append({"run_id": session.run.run_id, "source": session.run.source,
                                      "seconds": time.monotonic() - started})
                if plan.get("hf"):
                    callback.close()
                else:
                    session.close()
        found["warnings"] = [str(w.message) for w in caught if "run ID" in str(w.message) or "run_id" in str(w.message)]
    except BaseException as error:
        found = {"rank": rank, "raised": f"{type(error).__name__}: {error}"[:300]}
    for name, call in originals.items():
        setattr(dist, name, call)
    # The store still works after a wait that timed out (TCPStore cancels it on the server).
    shared = dist.distributed_c10d._get_default_store()
    if rank == 1:
        shared.set("worker/done", "1")
    else:
        shared.wait(["worker/done"], timeout)
    with open(out, "w") as file:
        json.dump(found, file)
    dist.destroy_process_group()
    """
)

# Whatever the parent's environment says about a run or a launcher must not reach the ranks.
SCRUBBED = (
    "MLFLOW_RUN_ID", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "SLURM_STEP_ID", "RANK", "LOCAL_RANK", "WORLD_SIZE",
    "MASTER_ADDR", "MASTER_PORT", "TENSORQUILL_ENDPOINT", "TENSORQUILL_PROJECT", "TENSORQUILL_RUN_NAME",
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _two_ranks(tmp_path: Path, plan: dict, launcher: dict[str, str]) -> list[dict]:
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    env = {k: v for k, v in os.environ.items() if k not in SCRUBBED}
    env.update(launcher)
    # The ranks import the tensorquill this test imported, not an installed copy.
    source = str(Path(tensorquill.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([source, *filter(None, [os.environ.get("PYTHONPATH")])])
    work = tmp_path / "work"
    work.mkdir()
    store = f"tcp:{_free_port()}" if plan.get("tcp") else str(tmp_path / "store")
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(rank), store, str(tmp_path / f"rank{rank}.json"), json.dumps(plan)],
            cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for rank in (0, 1)
    ]
    logs = []
    try:
        for proc in procs:
            logs.append(proc.communicate(timeout=60)[0])
    except subprocess.TimeoutExpired:
        for proc in procs:
            proc.kill()
        pytest.fail(f"tensorquill.init did not return on both ranks within 60 s ({plan}, {launcher})")
    found = []
    for rank in (0, 1):
        result = tmp_path / f"rank{rank}.json"
        assert result.exists(), f"rank {rank} wrote nothing; its output:\n{logs[rank][-2000:]}"
        found.append(json.loads(result.read_text()))
    assert all("raised" not in f for f in found), found
    return found


@pytest.mark.parametrize(
    ("plan", "launcher", "run_id", "source"),
    [
        # The finding's two cases: W&B on rank 0 only, without and with a launcher's job ID.
        ({"wandb": True}, {}, "wandb-abc", "wandb"),
        ({"wandb": True}, {"TORCHELASTIC_RUN_ID": "abc123"}, "wandb-abc", "wandb"),
        ({"wandb": True}, {"SLURM_JOB_ID": "77", "TORCHELASTIC_RUN_ID": "none"}, "wandb-abc", "wandb"),
        # The Hugging Face callback leaves the ID to init, so the source is the tracker's.
        ({"wandb": True, "hf": True}, {"TORCHELASTIC_RUN_ID": "abc123"}, "wandb-abc", "wandb"),
        ({}, {"TORCHELASTIC_RUN_ID": "abc123"}, "run-abc123", "launcher"),
        ({}, {}, None, "generated"),
        # A tracker that raises on rank 0 must not leave rank 1 waiting: rank 0 falls through.
        ({"mlflow_raises": True}, {"TORCHELASTIC_RUN_ID": "abc123"}, "run-abc123", "launcher"),
        # A run_id on rank 0 only (`run_id=wandb.run.id if wandb.run else None`) is rank 0's
        # resolution like any other: before lane T1 rank 1 waited in the broadcast alone.
        ({"run_id": {"0": "given-0"}}, {"TORCHELASTIC_RUN_ID": "abc123"}, "given-0", "explicit"),
        ({"run_id": {"0": "given-0"}, "hf": True}, {}, "given-0", "explicit"),
        ({"run_id": {"0": "same", "1": "same"}}, {}, "same", "explicit"),
        # Rank 0 reaching init late, inside the bound, through either store.
        ({"rank0_delay": 3}, {}, None, "generated"),
        ({"rank0_delay": 3, "tcp": True}, {}, None, "generated"),
    ],
)
def test_every_rank_gets_rank_zeros_run_id(tmp_path, plan, launcher, run_id, source):
    if plan.get("hf"):
        pytest.importorskip("transformers")
        pytest.importorskip("accelerate")
    found = _two_ranks(tmp_path, plan, launcher)
    runs = [f["runs"][0] for f in found]
    assert runs[0]["run_id"] == runs[1]["run_id"], found
    assert runs[0]["source"] == runs[1]["source"] == source, found
    assert found[1]["warnings"] == [], found
    if plan.get("mlflow_raises"):  # rank 0 says it passed over its tracker, once
        [warning] = found[0]["warnings"]
        assert "rank 0 could not read the tracker's run (RuntimeError: tracking server unreachable)" in warning
    else:
        assert found[0]["warnings"] == [], found
    if run_id is not None:
        assert runs[0]["run_id"] == run_id
    out = tmp_path / "work" / "tensorquill-out"
    shared = runs[0]["run_id"]
    assert sorted(p.name for p in out.glob("*.jsonl")) == [f"{shared}.rank0.jsonl", f"{shared}.rank1.jsonl"]
    # One manifest, written by rank 0, under the shared ID. tracker_labels is read on rank 0
    # only, so the labels come from the same tracker run as the ID.
    manifests = list(out.glob("*.run.json"))
    assert [p.name for p in manifests] == [f"{shared}.run.json"]
    manifest = json.loads(manifests[0].read_text())
    assert manifest["run_id"] == shared and manifest["run_id_source"] == source and manifest["world_size"] == 2
    if plan.get("wandb"):
        assert (manifest["project"], manifest["name"], manifest["group"]) == ("proj-w", "name-w", "group-w")


def test_a_run_id_given_on_another_rank_than_zero_gives_way_to_rank_zeros(tmp_path):
    found = _two_ranks(tmp_path, {"run_id": {"1": "given-1"}}, {"TORCHELASTIC_RUN_ID": "abc123"})
    assert [f["runs"][0]["run_id"] for f in found] == ["run-abc123", "run-abc123"], found
    assert [f["runs"][0]["source"] for f in found] == ["launcher", "launcher"], found
    assert found[0]["warnings"] == [], found
    [warning] = found[1]["warnings"]
    assert "run_id='given-1'" in warning and "'run-abc123'" in warning, warning


def test_each_init_shares_its_own_run_id(tmp_path):
    # A second init (a second train(), a restarted session) in the same process group reads
    # the ID rank 0 resolved for it, never the first call's.
    found = _two_ranks(tmp_path, {"inits": 2}, {})
    first, second = ([f["runs"][i]["run_id"] for f in found] for i in (0, 1))
    assert first[0] == first[1] and second[0] == second[1] and first[0] != second[0], found
    assert found[0]["warnings"] == found[1]["warnings"] == [], found


@pytest.mark.parametrize("tcp", [False, True], ids=["file-store", "tcp-store"])
def test_a_rank_that_never_hears_from_rank_zero_falls_back_with_one_warning(tmp_path, tcp):
    # Rank 0 never calls init. Rank 1 waits the bound (3 s here), then writes under an ID of
    # its own, with one warning, and raises nothing; the store still works afterwards.
    plan = {"rank0_skips": True, "wait_seconds": 3, "tcp": tcp}
    found = _two_ranks(tmp_path, plan, {"TORCHELASTIC_RUN_ID": "abc123"})
    assert found[0]["runs"] == [] and found[0]["warnings"] == [], found
    [run] = found[1]["runs"]
    assert (run["run_id"], run["source"]) == ("run-abc123", "launcher"), found
    assert 2.5 < run["seconds"] < 13, run
    [warning] = found[1]["warnings"]
    assert "rank 1 of 2 did not get rank 0's run ID" in warning and "it waits at most 3 s" in warning, warning
    assert "writes under 'run-abc123' (launcher)" in warning, warning
