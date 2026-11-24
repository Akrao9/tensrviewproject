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

- sharded, FSDP2 (`fully_shard`), FSDP1 with nested wrappers, and FSDP1 with one wrapper outside
  the module `init` is given (Lightning's FSDP strategy with no wrap policy): the probe callback is never
  entered (it leaves a marker and raises if it is), both ranks train all 16 steps, the healthy
  rank's capture is whole, each rank says once that the probe is off, with `RuntimeWarning`
  made an error, and the evidence calls the probe's kinds unavailable with the reason;
- not sharded: the probes still run on both ranks.
"""

from __future__ import annotations

import json
import os
import shutil
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


PORT_TAKEN = "failed to listen on any local network address"  # PyTorch's TCPStore, when the port is in use


def _port_taken(out: Path) -> bool:
    log = out / "rank0.log"
    return log.exists() and PORT_TAKEN in log.read_text(errors="replace")


def _run(flavor: str, out: Path) -> dict:
    """One case, with the port chosen again (three times at most) when another process took it between
    `_free_port` and rank 0's listen: that is this harness's race, not what the case is about."""
    for last in (False, False, True):
        run = _run_once(flavor, out)
        if last or not _port_taken(out):
            break
        shutil.rmtree(out)
    return run


def _run_once(flavor: str, out: Path) -> dict:
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
            if procs[0].poll() is not None and _port_taken(out):
                break  # rank 1 would wait for a store that never came up: ended below, chosen again in `_run`
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
        pytest.skip(f"PyTorch {torch.__version__} cannot train this sharded model on the CPU with {flavor}: "
                    + " | ".join(r["log"].strip().splitlines()[-1] for r in ranks if r["code"] == 77))
    return {"ranks": ranks, "seconds": time.monotonic() - start}


def _where(run: dict) -> str:
    return "; ".join(
        f"rank {i}: {'ended' if r['in_time'] else 'still running at the deadline'}, exit {r['code']}, {r['steps']} of {STEPS} steps,"
        f" last '{r['trace'][-1] if r['trace'] else 'no trace'}', log tail: {' / '.join(r['log'].strip().splitlines()[-3:])[-400:]}"
        for i, r in enumerate(run["ranks"])
    )


def test_a_port_taken_between_its_choice_and_the_listen_is_chosen_again(tmp_path: Path, monkeypatch) -> None:
    """The port is chosen here and listened on by rank 0 a moment later, and another process can take it in
    between: on CI (Python 3.11, PyTorch 2.14, run 45494662452) rank 0 then ended with PyTorch's "The server
    socket has failed to listen on any local network address" and the case failed at the deadline. The
    harness chooses again; the case is not about ports."""
    # Taken on every address, as the port was on CI: PyTorch's store listens on all of them, and on macOS a
    # port taken on 127.0.0.1 alone does not stop it.
    try:
        taken = socket.socket(socket.AF_INET6)
        taken.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        taken.bind(("::", 0))
    except OSError:  # no IPv6 here
        taken = socket.socket()
        taken.bind(("0.0.0.0", 0))
    with taken:
        taken.listen(1)
        choices = iter([taken.getsockname()[1]])
        free = _free_port
        monkeypatch.setattr(sys.modules[__name__], "_free_port", lambda: next(choices, None) or free())
        run = _run("fsdp2", tmp_path / "fsdp2")
    assert [(r["in_time"], r["code"]) for r in run["ranks"]] == [(True, 0), (True, 0)], _where(run)
    assert run["seconds"] < DEADLINE, "the taken port was waited out to the deadline"


@pytest.mark.parametrize("flavor", ["fsdp2", "fsdp1", "fsdp1-inner"])
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


def test_a_note_that_the_probe_is_off_is_not_a_failed_probe_and_a_failure_beside_it_is_the_entry() -> None:
    note = {"step": 0, "probe": {"not_run": "the model is sharded across ranks (FSDP2, fully_shard)"}}
    plain = [{"step": step} for step in range(1, 4)]
    not_checked: list[dict] = []
    assert detect_attention_anomalies([note, *plain], not_checked=not_checked) == []
    assert not_checked == [{
        "check": "probe", "kinds": list(ATTN_KINDS), "step": 0,
        "reason": "the probe was not run: the model is sharded across ranks (FSDP2, fully_shard)",
    }]
    failed = {"step": 1, "probe": {"error": "RuntimeError: CUDA out of memory"}}
    not_checked = []
    detect_attention_anomalies([note, failed], not_checked=not_checked)
    assert [entry["reason"] for entry in not_checked] == ["1 of 1 probe failed, the first at step 1: RuntimeError: CUDA out of memory"]


def test_a_model_that_is_not_sharded_keeps_its_probe_and_detection_never_raises() -> None:
    from torch import nn

    import tensorquill
    from tensorquill.capture import _sharded
    from tensorquill.sink import MemorySink

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    assert _sharded(model) is None

    class Odd(nn.Module):
        def modules(self):  # noqa: ANN202
            raise RuntimeError("no modules to list")

    assert _sharded(Odd()) is None  # what cannot be read is not called sharded, and `init` goes on

    calls = []
    fixed = torch.randn(2, 4)

    def probe(m):  # noqa: ANN001, ANN202
        calls.append(1)
        return m(fixed)

    sink = MemorySink()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with tensorquill.init(model, sink=sink, every=2, probe=probe, probe_every=2, heartbeat_seconds=None, stack_seconds=None):
        for _ in range(4):
            model(torch.randn(3, 4)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
    assert len(calls) == 2
    probes = [r["probe"] for r in sink.records if "probe" in r]
    assert len(probes) == 2 and all("not_run" not in p and "error" not in p for p in probes)
