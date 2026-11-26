"""Ranks started without `torchrun`: two processes of one SLURM job and no process group.

`srun` alone sets `SLURM_PROCID`, `SLURM_NTASKS` and `SLURM_STEP_NUM_TASKS` for each task and not
`RANK` or `WORLD_SIZE`, which `torchrun` sets. A script that calls `tensorquill.init` before
`init_process_group`, or never starts a group, runs here as two real processes with the SLURM
variables set, `RANK`, `LOCAL_RANK` and `WORLD_SIZE` removed, and a temporary folder as the working
directory, each training a few steps into the default output folder.

With no process group, the rank is read from the launchers' variables (`tracker._rank`): `RANK`,
then `SLURM_PROCID` inside an `srun` step, then Open MPI's and PMI's, and the world size from the
same launcher (`SLURM_STEP_NUM_TASKS` for `srun`). So the two tasks are ranks 0 and 1 of a world
of two, under one run ID, each in a file of its own. A separate test checks that the children ran
to their end, so a broken child cannot pass for a result.

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
            said.append("tensorquill logger: " + record.getMessage())

    logging.getLogger("tensorquill").addHandler(Keep())  # what `_say` logs instead of warning
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
    # What `srun --ntasks 2` sets for each task (srun(1), OUTPUT ENVIRONMENT VARIABLES).
    env.update(SLURM_JOB_ID="4242", SLURM_STEP_ID="0", SLURM_NTASKS="2", SLURM_STEP_NUM_TASKS="2",
               SLURM_PROCID=str(procid), SLURM_NODEID=str(node))
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


@pytest.mark.parametrize("arrangement", list(ARRANGEMENTS))
def test_the_two_tasks_are_ranks_0_and_1_of_two_each_in_a_file_of_its_own(outcomes, arrangement):
    """`SLURM_PROCID` names the rank and `SLURM_STEP_NUM_TASKS` the world size: tasks 0 and 1 are
    ranks 0 and 1 of 2, under the job's run ID, each writing `<run_id>.rank<N>.jsonl` with its own
    steps once, whether `LOCAL_RANK` is there or not, and nothing is said at `init`."""
    result = outcomes(arrangement)
    reports = [child["report"] for child in result["children"]]
    assert all(r is not None for r in reports), [c["stderr"] for c in result["children"]]
    assert [(r["rank"], r["world_size"], r["run_id"]) for r in reports] == [
        (0, 2, "run-slurm4242.0"), (1, 2, "run-slurm4242.0")]
    assert sorted(Path(r["path"]).name for r in reports) == [
        "run-slurm4242.0.rank0.jsonl", "run-slurm4242.0.rank1.jsonl"]
    assert sorted(result["files"]) == ["run-slurm4242.0.rank0.jsonl", "run-slurm4242.0.rank1.jsonl"]
    for rank in (0, 1):
        rows = result["files"][f"run-slurm4242.0.rank{rank}.jsonl"]
        assert [row.get("step") for row in rows] == list(range(STEPS))
        assert {row.get("rank") for row in rows} == {rank}
    said = [text for r in reports for text in r["init_said"] if "tensorquill" in text.lower()]
    assert said == []


@pytest.mark.parametrize("arrangement", SHARED_AT_ONCE)
def test_two_processes_of_one_job_do_not_share_a_file(outcomes, arrangement):
    """Run at the same time, the two tasks write two files (they were one, interleaved, before)."""
    result = outcomes(arrangement)
    reports = [child["report"] for child in result["children"]]
    assert all(r is not None for r in reports), [c["stderr"] for c in result["children"]]
    assert len({r["path"] for r in reports}) == 2


@pytest.mark.parametrize("arrangement", ["slurm_together"])
def test_the_listing_reads_the_job_as_two_closed_ranks(outcomes, arrangement):
    """`list_runs` reads one run of two ranks, both closed, and none missing: the manifest says
    world size 2 and both files are there."""
    from tensorquill.runs import list_runs

    rows = list_runs(outcomes(arrangement)["out_dir"])
    assert [(row["run_id"], row["ranks"], row["rank_count"], row["last_step"]) for row in rows] == [
        ("run-slurm4242.0", [0, 1], 2, STEPS - 1)]
    assert rows[0]["end"]["state"] == "closed"
