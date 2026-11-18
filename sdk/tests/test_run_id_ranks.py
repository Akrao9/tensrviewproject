"""Every rank of a `torch.distributed` job resolves the same run ID (review finding SDK-01).

A tracker's run usually exists on rank 0 only. Each case starts two Gloo CPU processes, gives
rank 0 alone a stand-in W&B run when the case has one, and calls `tensorquill.init` on both.
Before the fix, with no launcher ID, rank 0 took the W&B run's ID and entered no collective while
rank 1 waited in `broadcast_object_list` for it; with `TORCHELASTIC_RUN_ID` set, rank 1 took the
launcher's ID instead of the W&B run's. A short group timeout and a timeout on the processes make
a hang fail the test rather than hang the suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import tensorquill

WORKER = textwrap.dedent(
    """
    import datetime, json, sys, types
    import torch, torch.distributed as dist

    rank, store, case, out = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    dist.init_process_group("gloo", store=dist.FileStore(store, 2), rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=15))
    if case == "wandb-hf":
        # Imported before the stand-in: Transformers looks up wandb's import spec.
        from tensorquill.huggingface import _shared_tracker_run_id
    if case.startswith("wandb") and rank == 0:
        run = types.SimpleNamespace(id="wandb-abc", project="proj-w", name="name-w", group="group-w")
        sys.modules["wandb"] = types.SimpleNamespace(run=run)
    if case == "mlflow-raises" and rank == 0:
        def active_run():
            raise RuntimeError("tracking server unreachable")
        sys.modules["mlflow"] = types.SimpleNamespace(active_run=active_run)
    import tensorquill

    try:
        found = {"rank": rank}
        if case == "wandb-hf":
            # The Hugging Face callback's own sharing, before init's: two broadcasts, in the
            # same order on every rank.
            found["hf_run_id"] = _shared_tracker_run_id()
        session = tensorquill.init(torch.nn.Linear(4, 4))
        found.update(run_id=session.run.run_id, source=session.run.source)
        session.close()
    except Exception as error:
        found = {"rank": rank, "raised": f"{type(error).__name__}: {error}"[:300]}
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


def _two_ranks(tmp_path: Path, case: str, launcher: dict[str, str]) -> list[dict]:
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    env = {k: v for k, v in os.environ.items() if k not in SCRUBBED}
    env.update(launcher)
    # The ranks import the tensorquill this test imported, not an installed copy.
    source = str(Path(tensorquill.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([source, *filter(None, [os.environ.get("PYTHONPATH")])])
    work = tmp_path / "work"
    work.mkdir()
    store = tmp_path / "store"
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(rank), str(store), case, str(tmp_path / f"rank{rank}.json")],
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
        pytest.fail(f"tensorquill.init did not return on both ranks within 60 s (case {case}, {launcher})")
    found = []
    for rank in (0, 1):
        result = tmp_path / f"rank{rank}.json"
        assert result.exists(), f"rank {rank} wrote nothing; its output:\n{logs[rank][-2000:]}"
        found.append(json.loads(result.read_text()))
    return found


@pytest.mark.parametrize(
    ("case", "launcher", "run_id", "source"),
    [
        # The finding's two cases: W&B on rank 0 only, without and with a launcher's job ID.
        ("wandb", {}, "wandb-abc", "wandb"),
        ("wandb", {"TORCHELASTIC_RUN_ID": "abc123"}, "wandb-abc", "wandb"),
        ("wandb", {"SLURM_JOB_ID": "77", "TORCHELASTIC_RUN_ID": "none"}, "wandb-abc", "wandb"),
        # The Hugging Face callback shares the tracker's ID itself; both ways agree.
        ("wandb-hf", {"TORCHELASTIC_RUN_ID": "abc123"}, "wandb-abc", "wandb"),
        # The paths that worked before still do.
        ("none", {"TORCHELASTIC_RUN_ID": "abc123"}, "run-abc123", "launcher"),
        ("none", {}, None, "generated"),
        # A tracker that raises on rank 0 must not leave rank 1 waiting: rank 0 falls through.
        ("mlflow-raises", {"TORCHELASTIC_RUN_ID": "abc123"}, "run-abc123", "launcher"),
    ],
)
def test_every_rank_gets_rank_zeros_run_id(tmp_path, case, launcher, run_id, source):
    if case == "wandb-hf":
        pytest.importorskip("transformers")
    found = _two_ranks(tmp_path, case, launcher)
    assert all("raised" not in f for f in found), found
    if case == "wandb-hf":
        assert [f["hf_run_id"] for f in found] == [f["run_id"] for f in found], found
    assert found[0]["run_id"] == found[1]["run_id"], found
    assert found[0]["source"] == found[1]["source"] == source, found
    if run_id is not None:
        assert found[0]["run_id"] == run_id
    out = tmp_path / "work" / "tensorquill-out"
    shared = found[0]["run_id"]
    assert sorted(p.name for p in out.glob("*.jsonl")) == [f"{shared}.rank0.jsonl", f"{shared}.rank1.jsonl"]
    # One manifest, written by rank 0, under the shared ID. tracker_labels is read on rank 0
    # only, so the labels come from the same tracker run as the ID.
    manifests = list(out.glob("*.run.json"))
    assert [p.name for p in manifests] == [f"{shared}.run.json"]
    manifest = json.loads(manifests[0].read_text())
    assert manifest["run_id"] == shared and manifest["run_id_source"] == source and manifest["world_size"] == 2
    if case.startswith("wandb"):
        assert (manifest["project"], manifest["name"], manifest["group"]) == ("proj-w", "name-w", "group-w")
