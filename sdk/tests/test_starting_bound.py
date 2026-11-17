"""A real training process that stops before its first batch is stored, beside a real server.

The capture keeps `every` steps (50 by default) on the device before it writes any, so a job
that hangs in its first steps, or before its first step, has stored nothing. Its stream is open
and its heartbeats go on. The server calls such a rank `starting`, within two bounds (see
`Progress::state` in `server/src/watchdog.rs`): once its heartbeats' work counter shows it
stepping, the counter is its progress; before its first training step it has `--start-seconds`.

Needs `pyarrow` and the server binary, as `test_two_ranks_e2e.py` does. `every` is left at its
default in every rank here.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")

import tensorquill  # noqa: E402
from tensorquill.evidence import detect_watchdog_incidents, read_watchdog  # noqa: E402
from tensorquill.evidence import _describe_stream  # noqa: E402
from test_two_ranks_e2e import BINARY, Server  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")

SRC = Path(tensorquill.__file__).resolve().parents[1]
RUN_ID = "start"

# One rank as a launcher starts it. It waits `TQ_BEFORE` seconds before its first step (a
# compilation, a data load), takes `TQ_STEPS` steps `TQ_STEP_SECONDS` apart, and then stops
# stepping without closing: its process, its stream and its heartbeats go on, as those of a job
# hung in training do. `every` is not passed: 50 steps are held before any is written.
_RANK = """
import os, time
import torch, tensorquill
from torch import nn

model = nn.Linear(8, 1)
session = tensorquill.init(model, run_id=os.environ["TQ_RUN_ID"], heartbeat_seconds=0.5, stack_seconds=1.0)
session._sink.sender._batch_seconds, session._sink.sender._backoff_cap = 0.25, 0.5
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
time.sleep(float(os.environ["TQ_BEFORE"]))
for _ in range(int(os.environ["TQ_STEPS"])):
    optimizer.zero_grad()
    loss = model(torch.randn(16, 8)).pow(2).mean()
    loss.backward()
    session.log({"train/loss": loss})
    optimizer.step()
    time.sleep(float(os.environ["TQ_STEP_SECONDS"]))
print("stopped stepping", flush=True)

def stuck_here():
    while True:
        time.sleep(0.2)

stuck_here()
"""
# The status is looked at every 0.2 s and an unchanged one is written again every 0.6 s.
_QUICK = ("--silent-factor", "0", "--watchdog-seconds", "0.2", "--flush-seconds", "0.3")


class _Job:
    """A server and one rank, and what the server's status said of the rank, in order."""

    def __init__(self, tmp_path: Path, flags: tuple[str, ...], *, before: float, steps: int, step_seconds: float) -> None:
        self.store = tmp_path / "store"
        self.status_file = self.store / "runs" / RUN_ID / "watchdog.json"
        work = tmp_path / "work"
        work.mkdir()
        clean = {k: v for k, v in os.environ.items()
                 if k not in ("TENSORQUILL_TOKEN", "TENSORQUILL_TLS_CA", "MLFLOW_RUN_ID", "TORCHELASTIC_RUN_ID")}
        self.server = Server(self.store, 0, *_QUICK, *flags)
        self.log = (tmp_path / "rank0.log").open("w")
        self.rank = subprocess.Popen(
            [sys.executable, "-c", _RANK], cwd=work, stdout=self.log, stderr=subprocess.STDOUT,
            env={**clean, "PYTHONPATH": str(SRC), "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
                 "TENSORQUILL_ENDPOINT": self.server.endpoint, "TQ_RUN_ID": RUN_ID, "TQ_BEFORE": str(before),
                 "TQ_STEPS": str(steps), "TQ_STEP_SECONDS": str(step_seconds), "PYTHONUNBUFFERED": "1"},
        )
        self.said: list[tuple[str, str | None]] = []  # (state, severity) each time it changed

    def status(self) -> dict:
        return read_watchdog(self.status_file) or {}

    def entry(self) -> dict:
        """The rank's entry in the status, noting its state and severity when they changed."""
        ranks = self.status().get("ranks") or [{}]
        now = (ranks[0].get("state"), ranks[0].get("severity"))
        if now[0] is not None and (not self.said or self.said[-1] != now):
            self.said.append(now)
        return ranks[0]

    def until(self, ready, what: str, seconds: float = 90.0):
        deadline = time.monotonic() + seconds
        while True:
            found = ready(self.entry())
            if found:
                return found
            assert self.rank.poll() is None, f"the rank ended while waiting for {what}"
            assert time.monotonic() < deadline, f"timed out waiting for {what}: {self.said}"
            time.sleep(0.05)

    def close(self) -> None:
        for process in (self.rank, self.server.process):
            if process.poll() is None:
                process.kill()
                process.wait(20)
        if not self.server.process.stdout.closed:
            self.server.process.stdout.close()
        self.log.close()


def test_a_real_rank_that_stops_after_40_steps_with_nothing_stored_is_a_warning_and_then_critical(tmp_path):
    """Forty steps and then none, `every` at 50: no batch reaches the server. The heartbeats'
    counter stands at 40, and the server calls the rank silent 2 s later and critical after 8 s.
    `--start-seconds` is at its default of 1800, so it is the counter that tells."""
    job = _Job(tmp_path, ("--silent-seconds", "2", "--critical-seconds", "8"), before=0.0, steps=40, step_seconds=0.01)
    try:
        warned = job.until(lambda rank: rank.get("severity") == "warning" and rank, "the rank at warning")
        assert (warned["state"], warned["last_step"], warned["work"], warned["stream"]) == ("silent", None, 40, "open")
        assert warned["heartbeat_age_seconds"] is not None and warned["heartbeat_age_seconds"] < 2.0
        assert job.status()["summary"] == "stalled"
        critical = job.until(lambda rank: rank.get("severity") == "critical" and rank, "the rank at critical")
        assert (critical["state"], critical["last_step"], critical["work"]) == ("silent", None, 40)
        status = job.status()
        assert job.rank.poll() is None  # its process lives, and it never stored a step
        assert not list((job.store / "runs" / RUN_ID).rglob("*.parquet"))
    finally:
        job.close()
    assert job.said == [("starting", None), ("silent", "warning"), ("silent", "critical")]
    assert status["start_seconds"] == 1800.0
    # The server's own incident, and Python's reading of the same status.
    (stall,) = status["incidents"]
    assert (stall["kind"], stall["step"], stall["severity"], stall["assessment"]) == ("job_stalled", 0, "critical", "suspected_hang")
    (read,) = detect_watchdog_incidents(status)
    assert (read["kind"], read["step"], read["severity"]) == ("job_stalled", 0, "critical")
    assert read["stuck_at"][0]["function"] == "stuck_here"
    said = _describe_stream(read)
    assert said.startswith("Job stalled before its first batch was stored: none of 1 rank has sent a step in ")
    assert "the heartbeats' work counter stands at 40" in said and "step None" not in said and "step 0" not in said


def test_a_real_rank_that_never_takes_a_step_has_the_start_limit_and_is_critical_after_it(tmp_path):
    """A rank stuck before its first step (a loader that never yields, its first collective): its
    counter stays at 0. It is `starting` for `--start-seconds` (3 s here), then a warning, and
    critical `--critical-seconds` (4 s) after that. The silence floor of 1 s does not apply to it."""
    job = _Job(tmp_path, ("--silent-seconds", "1", "--start-seconds", "3", "--critical-seconds", "4"),
               before=3600.0, steps=0, step_seconds=0.0)
    try:
        opened = time.monotonic()
        job.until(lambda rank: rank.get("state") == "starting", "the rank's stream")
        warned = job.until(lambda rank: rank.get("severity") == "warning" and rank, "the rank at warning")
        assert time.monotonic() - opened > 2.5  # it was given the start limit, not the 1 s floor
        assert (warned["state"], warned["last_step"], warned["work"]) == ("silent", None, 0)
        critical = job.until(lambda rank: rank.get("severity") == "critical" and rank, "the rank at critical")
        assert critical["seconds_since_progress"] > 6.5
        status = job.status()
    finally:
        job.close()
    assert job.said == [("starting", None), ("silent", "warning"), ("silent", "critical")]
    said = _describe_stream(detect_watchdog_incidents(status)[0])
    assert said.startswith("Job stalled before its first batch was stored") and "work counter" not in said


def test_a_real_rank_that_is_slow_to_its_first_step_and_its_first_batch_is_never_called_critical(tmp_path):
    """Six seconds before the first step with a start limit of 3 s: a warning, which the first
    step ends well before it would be critical (11 s). Then 50 steps 0.3 s apart: 15 s to the
    first stored batch, past the silence floor (2 s), the start limit and the critical time (8 s),
    and nothing is said, because the counter moves."""
    job = _Job(tmp_path, ("--silent-seconds", "2", "--start-seconds", "3", "--critical-seconds", "8"),
               before=6.0, steps=60, step_seconds=0.3)
    try:
        job.until(lambda rank: rank.get("severity") == "warning", "the warning before the first step")
        job.until(lambda rank: rank.get("state") == "starting" and (rank.get("work") or 0) > 0, "the first steps seen")
        stepping = time.monotonic()
        stored = job.until(lambda rank: rank.get("last_step") is not None and rank, "the first stored batch")
        assert time.monotonic() - stepping > 10.0  # longer than the floor, the limit and the critical time
        assert (stored["state"], stored["severity"]) == ("active", None)
    finally:
        job.close()
    assert job.said == [("starting", None), ("silent", "warning"), ("starting", None), ("active", None)]
