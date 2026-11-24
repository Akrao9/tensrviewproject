"""The probe is off on a sharded model (FSDP1 and FSDP2), so one rank that skips it cannot leave
the job mismatched.

Under sharding the model's forward is a collective (each sharded unit gathers its parameters
from every rank), so the probe's forward is one too. A rank whose session had stopped skipped
its probe while its peers ran theirs, and a probe that raised part-way left its forward early:
the peers' collectives then met that rank's next training collectives. Reproduced on two Gloo
ranks on the CPU on Nov 24, 2026 (the bug list's section 19): one rank's process aborted in
Gloo and the other waited.

Each case here is two real processes over Gloo on the CPU, with `probe=` set on both and a sink
on rank 1 that fails from step 5, under a deadline that ends every process:

- sharded, FSDP2 (`fully_shard`) and FSDP1 with nested wrappers: the probe callback is never
  entered (it leaves a marker and raises if it is), both ranks train all 16 steps, the healthy
  rank's capture is whole, each rank says once that the probe is off, with `RuntimeWarning`
  made an error, and the evidence calls the probe's kinds unavailable with the reason;
- not sharded: the probes still run on both ranks.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from tensorquill.evidence import ATTN_KINDS, detect_attention_anomalies

WORKER = Path(__file__).with_name("fsdp_probe_worker.py")
STEPS, PROBE_EVERY = 16, 4
DEADLINE = 60.0  # seconds; the cases that end take a few
SAID = "the probe is off"


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _run(flavor: str, out: Path) -> dict:
    """Both ranks of one case: whether each ended inside the deadline, its exit status, its trace,
    the records its sink was handed and its output. Every process is ended before this returns."""
    out.mkdir()
    port = _free_port()
    source = str(Path(__file__).resolve().parents[1] / "src")
    procs, logs = [], []
    for rank in range(2):
        env = dict(
            os.environ, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port), OMP_NUM_THREADS="1",
            PYTHONPATH=os.pathsep.join(filter(None, [source, os.environ.get("PYTHONPATH")])),
        )
        log = (out / f"rank{rank}.log").open("w")
        logs.append(log)
        procs.append(subprocess.Popen(
            # A RuntimeWarning is an error here: what the session says must not raise out of `init`.
            [sys.executable, "-W", "error::RuntimeWarning", str(WORKER), "--flavor", flavor, "--out", str(out)],
            env=env, stdout=log, stderr=subprocess.STDOUT,
        ))
    start = time.monotonic()
    try:
        while time.monotonic() - start < DEADLINE and any(p.poll() is None for p in procs):
            time.sleep(0.1)
        in_time = [p.poll() is not None for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        codes = [p.wait(timeout=30) for p in procs]
        for log in logs:
            log.close()
    assert all(p.poll() is not None for p in procs), "a rank's process is still running"
    ranks = []
    for rank in range(2):
        trace = out / f"rank{rank}.trace"
        lines = [line.split(" ", 1)[1] for line in trace.read_text().splitlines()] if trace.exists() else []
        jsonl = out / f"rank{rank}.jsonl"
        ranks.append({
            "in_time": in_time[rank], "code": codes[rank], "trace": lines,
            "steps": sum(line.endswith(" done") for line in lines),
            "records": [json.loads(line) for line in jsonl.read_text().splitlines()] if jsonl.exists() else [],
            "entered": (out / f"probe-entered.rank{rank}").exists(),
            "log": (out / f"rank{rank}.log").read_text(),
        })
    if any(r["code"] == 77 for r in ranks):
        pytest.skip(f"PyTorch {torch.__version__} cannot shard this model on the CPU with {flavor}: "
                    + " | ".join(r["log"].strip().splitlines()[-1] for r in ranks if r["code"] == 77))
    return {"ranks": ranks, "seconds": time.monotonic() - start}


def _where(run: dict) -> str:
    return "; ".join(
        f"rank {i}: {'ended' if r['in_time'] else 'still running at the deadline'}, exit {r['code']}, {r['steps']} of {STEPS} steps,"
        f" last '{r['trace'][-1] if r['trace'] else 'no trace'}', log tail: {' / '.join(r['log'].strip().splitlines()[-3:])[-400:]}"
        for i, r in enumerate(run["ranks"])
    )


@pytest.mark.parametrize("flavor", ["fsdp2", "fsdp1"])
def test_a_sharded_model_trains_to_the_end_with_the_probe_off_when_one_rank_stops(flavor: str, tmp_path: Path) -> None:
    run = _run(flavor, tmp_path / flavor)
    healthy, stopped = run["ranks"]

    # The probe callback was never entered, on either rank, in any step.
    assert not healthy["entered"] and not stopped["entered"], f"the probe callback was entered on a sharded model: {_where(run)}"
    # Both ranks finish every step inside the deadline, and neither process aborted.
    assert all(r["in_time"] for r in run["ranks"]), f"a rank did not finish in {DEADLINE:.0f} s: {_where(run)}"
    assert [r["code"] for r in run["ranks"]] == [0, 0], _where(run)
    assert [r["steps"] for r in run["ranks"]] == [STEPS, STEPS], _where(run)

    # Ordinary capture goes on: the healthy rank wrote every step, with its gradient norms.
    assert [r["step"] for r in healthy["records"]] == list(range(STEPS))
    assert all(r.get("grad_norm") is not None for r in healthy["records"])
    # The rank whose sink failed stopped capturing, as before, and trained on.
    assert "stopped capturing" in stopped["log"] and len(stopped["records"]) < STEPS

    # Said once on each rank, under `-W error::RuntimeWarning`, and `init` returned.
    for rank, r in enumerate(run["ranks"]):
        assert r["log"].count(SAID) == 1, f"rank {rank} said it {r['log'].count(SAID)} times:\n{r['log']}"
        assert "session started" in r["trace"]

    # The capture says so once, on its first record, and no record holds a probe's results or error.
    notes = [r["probe"] for r in healthy["records"] if "probe" in r]
    assert len(notes) == 1 and "probe" in healthy["records"][0], notes
    assert set(notes[0]) == {"not_run"} and "shard" in notes[0]["not_run"], notes
    # The evidence: the probe's kinds are unavailable, with that reason, and nothing was found.
    not_checked: list[dict] = []
    assert detect_attention_anomalies(healthy["records"], not_checked=not_checked) == []
    assert len(not_checked) == 1, not_checked
    entry = not_checked[0]
    assert entry["check"] == "probe" and entry["kinds"] == list(ATTN_KINDS)
    assert notes[0]["not_run"] in entry["reason"] and "failed" not in entry["reason"], entry


def test_without_sharding_the_probe_still_runs_on_both_ranks(tmp_path: Path) -> None:
    run = _run("none", tmp_path / "none")
    healthy, stopped = run["ranks"]
    assert all(r["in_time"] for r in run["ranks"]) and [r["code"] for r in run["ranks"]] == [0, 0], _where(run)
    assert [r["steps"] for r in run["ranks"]] == [STEPS, STEPS], _where(run)
    assert healthy["entered"] and stopped["entered"], _where(run)
    probes = [(r["step"], sorted(r["probe"])) for r in healthy["records"] if "probe" in r]
    assert [step for step, _ in probes] == list(range(PROBE_EVERY - 1, STEPS, PROBE_EVERY)), probes
    assert all("error" not in keys and "not_run" not in keys for _, keys in probes), probes
    assert all(SAID not in r["log"] for r in run["ranks"])
