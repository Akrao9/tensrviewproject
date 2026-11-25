"""Ranks started without `torchrun`: two processes of one SLURM job and no process group.

`srun` alone sets `SLURM_PROCID` and `SLURM_NTASKS` for each task and not `RANK` or `WORLD_SIZE`,
which `torchrun` sets. A script that calls `tensorquill.init` before `init_process_group`, or never
starts a group, runs here as two real processes with the SLURM variables set, `RANK`, `LOCAL_RANK`
and `WORLD_SIZE` removed, and a temporary folder as the working directory, each training a few
steps into the default output folder.

How a rank started this way should be known is not decided, so nothing here holds what a process
calls its rank, its world size or its run ID, what the file holds, or what the readers say of it;
the lane's report records each. One thing is held, as a strict expected failure: two processes of
one job that run at the same time must not write one records file with no warning. Two separate
tests check that the children ran to their end, so a broken child cannot pass as that failure.

Only torch and pytest are needed. Each child has a time limit and is killed if it is still running.
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

STEPS = 6
LIMIT = 120  # seconds for each child, and for the barrier inside it

CHILD = textwrap.dedent(
    """
    import json, logging, os, sys, time, warnings
    from pathlib import Path

    steps, ready_dir, peers, limit = int(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    me = os.environ["SLURM_PROCID"]
    said = []

    class Keep(logging.Handler):
        def emit(self, record):
            said.append(record.getMessage())

    logging.getLogger("tensorquill").addHandler(Keep())
    import torch
    from torch import nn
    import tensorquill
    from tensorquill import tracker

    torch.manual_seed(int(me))
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with warnings.catch_warnings(record=True) as at_init:
        warnings.simplefilter("always")
        session = tensorquill.init(model, every=1, heartbeat_seconds=None, stack_seconds=None)
    init_said = [str(w.message) for w in at_init] + list(said)
    logged_at_init = len(said)
    # Both processes have their file open before either writes a step.
    (ready_dir / f"ready-{me}").write_text("")
    deadline = time.monotonic() + limit
    while len(list(ready_dir.glob("ready-*"))) < peers:
        if time.monotonic() > deadline:
            sys.exit("the other process never got ready")
        time.sleep(0.01)
    done = 0
    with warnings.catch_warnings(record=True) as later:
        warnings.simplefilter("always")
        for _ in range(steps):
            model(torch.randn(4, 4)).pow(2).mean().backward()
            optimizer.step()
            optimizer.zero_grad()
            done += 1
            time.sleep(0.01)
        path = getattr(session._sink, "path", None)
        session.close()
    print(json.dumps({
        "procid": me, "rank": session.run.rank, "world_size": tracker._world_size(), "run_id": session.run.run_id,
        "path": None if path is None else str(path), "steps_done": done,
        "init_said": init_said, "later_said": [str(w.message) for w in later] + said[logged_at_init:],
    }))
    """
)

# Each arrangement: the two tasks' LOCAL_RANK (None leaves it out), their SLURM_NODEID, and whether
# they run at the same time.
ARRANGEMENTS = {
    "slurm_together": (None, (0, 0), True),
    "slurm_one_after_the_other": (None, (0, 0), False),
    "two_nodes_local_rank": ((0, 0), (0, 1), True),  # one task on each of two nodes
    "one_node_local_ranks": ((0, 1), (0, 0), True),  # two tasks on one node
}
SHARED_AT_ONCE = ["slurm_together", "two_nodes_local_rank"]
LEFT_OUT = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "MASTER_ADDR",
            "MASTER_PORT", "PYTHONWARNINGS")
LEFT_OUT_PREFIXES = ("SLURM_", "TORCHELASTIC_", "MLFLOW_", "WANDB_", "TENSORQUILL_", "OMPI_", "PMI_")


def _environment(procid: int, local_rank: int | None, node: int) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in LEFT_OUT and not k.startswith(LEFT_OUT_PREFIXES)}
    # pytest's `pythonpath` setting does not reach a child: name the package this run imports.
    src = str(Path(tensorquill.__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(SLURM_JOB_ID="4242", SLURM_STEP_ID="0", SLURM_NTASKS="2", SLURM_PROCID=str(procid),
               SLURM_NODEID=str(node))
    if local_rank is not None:
        env["LOCAL_RANK"] = str(local_rank)
    return env


def _start(script: Path, work: Path, ready: Path, procid: int, arrangement: str, peers: int):
    local_ranks, nodes, _ = ARRANGEMENTS[arrangement]
    local_rank = None if local_ranks is None else local_ranks[procid]
    return subprocess.Popen(
        [sys.executable, str(script), str(STEPS), str(ready), str(peers), str(LIMIT)],
        cwd=work, env=_environment(procid, local_rank, nodes[procid]), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _finish(process) -> tuple[int, str, str]:
    try:
        out, err = process.communicate(timeout=LIMIT)
    except subprocess.TimeoutExpired:
        process.kill()
        out, err = process.communicate()
        return -9, out, err + "\n(killed at the time limit)"
    return process.returncode, out, err


def _launch(work: Path, arrangement: str) -> dict:
    together = ARRANGEMENTS[arrangement][2]
    script = work.parent / f"{work.name}-child.py"
    script.write_text(CHILD)
    work.mkdir()
    children = []
    processes = []
    try:
        if together:
            ready = work.parent / f"{work.name}-ready"
            ready.mkdir()
            processes = [_start(script, work, ready, i, arrangement, 2) for i in (0, 1)]
            children = [_finish(p) for p in processes]
        else:
            for i in (0, 1):
                ready = work.parent / f"{work.name}-ready-{i}"
                ready.mkdir()
                process = _start(script, work, ready, i, arrangement, 1)
                processes.append(process)
                children.append(_finish(process))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    reports = []
    for code, out, err in children:
        try:
            report = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            report = None
        reports.append({"code": code, "report": report, "stderr": err[-2000:]})
    out_dir = work / "tensorquill-out"
    files = {}
    for path in sorted(out_dir.glob("*.jsonl")) if out_dir.exists() else []:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = []
        for line in lines:
            try:
                row = json.loads(line)
                rows.append({k: row.get(k) for k in ("step", "rank", "attempt", "run_id")})
            except ValueError:
                rows.append({"cut": line[:80]})
        files[path.name] = rows
    others = sorted(p.name for p in out_dir.iterdir() if p.suffix != ".jsonl") if out_dir.exists() else []
    return {"children": reports, "files": files, "others": others, "out_dir": out_dir}


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory):
    found: dict[str, dict] = {}

    def outcome(arrangement: str) -> dict:
        if arrangement not in found:
            found[arrangement] = _launch(tmp_path_factory.mktemp("ranks") / arrangement, arrangement)
            print(f"\n{arrangement}: {json.dumps({k: v for k, v in found[arrangement].items() if k != 'out_dir'})}")
        return found[arrangement]

    return outcome


@pytest.mark.parametrize("arrangement", list(ARRANGEMENTS))
def test_each_process_ran_to_its_end(outcomes, arrangement):
    """Each child exited 0, said what it did and took every step. Nothing about what it called
    itself or where it wrote is held."""
    result = outcomes(arrangement)
    for child in result["children"]:
        assert child["code"] == 0, child["stderr"]
        assert child["report"] is not None, child["stderr"]
        assert child["report"]["steps_done"] == STEPS
        assert child["report"]["path"] is not None
    rows = [row for rows in result["files"].values() for row in rows]
    assert len(rows) == 2 * STEPS, "every step of both processes is somewhere in the output folder"


@pytest.mark.parametrize("arrangement", SHARED_AT_ONCE)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Without RANK and a process group, both processes read rank 0 and world size 1, take the "
    "same SLURM run ID and append to the same tensorquill-out/<run_id>.rank0.jsonl at the same "
    "time, and tensorquill.init gives no warning (ledger row OG4).",
)
def test_two_processes_of_one_job_do_not_share_a_file_without_a_warning(outcomes, arrangement):
    result = outcomes(arrangement)
    reports = [child["report"] for child in result["children"]]
    assert all(r is not None for r in reports), [c["stderr"] for c in result["children"]]
    paths = {r["path"] for r in reports}
    warned = any(r["init_said"] for r in reports)
    assert len(paths) == 2 or warned, f"one file, {sorted(paths)}, and no warning at init"
