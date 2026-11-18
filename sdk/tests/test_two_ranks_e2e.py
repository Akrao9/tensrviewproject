"""End to end with two real ranks: two training processes stream at the same time, under one
run ID, to one real `tensorquill-server`, and what the server stored and reported is read back.

Each rank is its own Python interpreter, started as a launcher starts it: `RANK`,
`LOCAL_RANK` and `WORLD_SIZE=2` in its environment, `TENSORQUILL_ENDPOINT` pointing at the
server, and a plain `tensorquill.init(model, run_id=...)` in the script. Needs the server
binary (`cargo build --manifest-path server/Cargo.toml`, or `TENSORQUILL_SERVER_BIN`).
Skipped when there is none.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")
import pyarrow.parquet as pq  # noqa: E402

import tensorquill  # noqa: E402
from tensorquill.flight import offset_path, read_offset, row_to_record  # noqa: E402
from test_flight_e2e import _without_status  # noqa: E402


def _server_binary() -> Path | None:
    named = os.environ.get("TENSORQUILL_SERVER_BIN")
    if named:
        return Path(named)
    target = Path(__file__).resolve().parents[2] / "server" / "target"
    built = [p for p in (target / "debug" / "tensorquill-server", target / "release" / "tensorquill-server") if p.exists()]
    return max(built, key=lambda p: p.stat().st_mtime) if built else None


BINARY = _server_binary()
pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")

RUN_ID = "two-ranks"
# The server's watchdog flags: a rank quiet for 2 s is silent, one that dropped for 2 s is lost.
WATCHDOG = ("--silent-seconds", "2", "--silent-factor", "0", "--watchdog-seconds", "0.2", "--flush-seconds", "0.3")


class Server:
    def __init__(self, store: Path, port: int = 0, *extra: str) -> None:
        env = {k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"}
        self.process = subprocess.Popen(
            [str(BINARY), "--listen", f"127.0.0.1:{port}", "--store", str(store), *extra],
            stdout=subprocess.PIPE, text=True, env=env,
        )
        line = self.process.stdout.readline()  # "tensorquill-server listening on 127.0.0.1:PORT"
        self.port = int(line.rsplit(":", 1)[1])
        self.endpoint = f"grpc://127.0.0.1:{self.port}"

    def terminate(self) -> int:
        """SIGTERM: the server writes what it has buffered, then exits."""
        self.process.send_signal(signal.SIGTERM)
        code = self.process.wait(20)
        self.process.stdout.close()
        return code

    def kill(self) -> None:
        """SIGKILL: whatever was only in the server's memory is gone."""
        self.process.kill()
        self.process.wait(20)
        self.process.stdout.close()


# What every rank runs. A small mixture-of-experts model seeded by its rank, so the two ranks
# write different numbers and a mix-up between them shows. `TQ_STEPS=0` means: step until the
# file named by `TQ_STOP` exists.
_RANK_JOB = """
import os, time
import torch, tensorquill
from torch import nn

class Router(nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        self.proj = nn.Linear(dim, n)
    def forward(self, h):
        logits = self.proj(h)
        return logits, logits.argmax(dim=-1)

class ToyMoE(nn.Module):
    def __init__(self, dim=8, n=4):
        super().__init__()
        self.inp = nn.Linear(dim, dim)
        self.router = Router(dim, n)
        self.experts = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n))
        self.out = nn.Linear(dim, 1)
    def forward(self, x):
        h = torch.relu(self.inp(x))
        _, choice = self.router(h)
        stacked = torch.stack([expert(h) for expert in self.experts], dim=1)
        return self.out(stacked.gather(1, choice[:, None, None].expand(-1, 1, h.shape[-1])).squeeze(1))

rank = int(os.environ["RANK"])
torch.manual_seed(rank)
model = ToyMoE()
session = tensorquill.init(model, run_id=os.environ["TQ_RUN_ID"], every=10, routers={"router": 4})
# Batches leave every 0.25 s, so a server that calls a rank silent after 2 s hears it in time.
session._sink.sender._batch_seconds, session._sink.sender._backoff_cap = 0.25, 0.5
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
steps, stop, pause = int(os.environ["TQ_STEPS"]), os.environ["TQ_STOP"], float(os.environ["TQ_PAUSE"])
done = 0
while done < steps if steps else not os.path.exists(stop):
    optimizer.zero_grad()
    loss = model(torch.randn(16, 8)).pow(2).mean()
    loss.backward()
    session.log({"train/loss": loss})
    optimizer.step()
    done += 1
    time.sleep(pause)
session.close()
print("closed after", done, "steps", flush=True)
"""


class Rank:
    """One training process: its own interpreter, with the environment a launcher gives it."""

    def __init__(self, work: Path, server: Server, rank: int, *, steps: int = 0, pause: float = 0.01,
                 world_size: int = 2) -> None:
        env = {k: v for k, v in os.environ.items()
               if k not in ("TENSORQUILL_TOKEN", "TENSORQUILL_TLS_CA", "MLFLOW_RUN_ID", "TORCHELASTIC_RUN_ID")}
        self.rank = rank
        self.stop_file = work / f"stop-rank{rank}"
        env.update({
            "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world_size),
            "TENSORQUILL_ENDPOINT": server.endpoint, "TQ_RUN_ID": RUN_ID,
            "TQ_STEPS": str(steps), "TQ_PAUSE": str(pause), "TQ_STOP": str(self.stop_file),
            "PYTHONUNBUFFERED": "1",
        })
        self.log_path = work / f"rank{rank}.log"
        self._log = self.log_path.open("w")
        self.process = subprocess.Popen(
            [sys.executable, "-c", _RANK_JOB], cwd=work, env=env, stdout=self._log, stderr=subprocess.STDOUT
        )
        self.local_file = work / "tensorquill-out" / f"{RUN_ID}.rank{rank}.jsonl"

    def output(self) -> str:
        self._log.flush()
        return self.log_path.read_text()

    def finish(self, timeout: float = 60) -> None:
        """Ask a stepping rank to stop, and wait for it to close and exit cleanly."""
        self.stop_file.touch()
        self.wait(timeout)

    def wait(self, timeout: float = 60) -> None:
        code = self.process.wait(timeout)
        assert code == 0, f"rank {self.rank} exited {code}:\n{self.output()}"

    def kill(self) -> None:
        """SIGKILL, as the OOM killer does: no close(), no atexit, the stream is never ended."""
        self.process.kill()
        assert self.process.wait(20) == -signal.SIGKILL

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(20)
        self._log.close()


class Cluster:
    """One server and the ranks started against it, all cleaned up at the end."""

    def __init__(self, root: Path) -> None:
        self.work = root / "work"
        self.work.mkdir()
        self.store = root / "store"
        self.run_dir = self.store / "runs" / RUN_ID
        self.server: Server | None = None
        self.ranks: list[Rank] = []

    def start_server(self, *flags: str, port: int = 0) -> Server:
        self.server = Server(self.store, port, *flags)
        return self.server

    def start_rank(self, rank: int, **kwargs) -> Rank:
        started = Rank(self.work, self.server, rank, **kwargs)
        self.ranks.append(started)
        return started

    def status(self) -> dict | None:
        return tensorquill.evidence.read_watchdog(self.run_dir / "watchdog.json")

    def wait_for(self, ready, seconds: float = 30) -> dict:
        deadline = time.monotonic() + seconds
        status = None
        while time.monotonic() < deadline:
            status = self.status()
            if status and ready(status):
                return status
            for started in self.ranks:
                assert started.process.poll() in (None, 0, -signal.SIGKILL), (
                    f"rank {started.rank} died:\n{started.output()}")
            time.sleep(0.1)
        raise AssertionError(
            f"timed out; last status {json.dumps(status, indent=1)}\n"
            + "\n".join(f"rank {r.rank}:\n{r.output()}" for r in self.ranks)
        )

    def stored(self, rank: int) -> list[dict]:
        parts = sorted((self.run_dir / f"rank={rank}").glob("attempt=*/part-*.parquet"))
        return [row_to_record(row) for part in parts for row in pq.read_table(part, partitioning=None).to_pylist()]

    def close(self) -> None:
        for started in self.ranks:
            started.stop()
        if self.server is not None:
            assert self.server.terminate() == 0


@pytest.fixture
def cluster(tmp_path):
    started = Cluster(tmp_path)
    yield started
    started.close()


def _state(status: dict, rank: int) -> dict:
    [found] = [r for r in status["ranks"] if r["rank"] == rank]
    return found


def _local_prefix(path: Path) -> list[dict]:
    """The complete lines of a rank's file; a process killed mid-write may leave a torn last one."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            break
    return records


# --- Both ranks run to close() ---------------------------------------------------------------


@pytest.fixture(scope="module")
def clean_run(tmp_path_factory):
    """Two ranks stream at once, 120 steps and 95 steps, and both close(); the server's
    verdict is read once it has seen both streams end."""
    cluster = Cluster(tmp_path_factory.mktemp("two-ranks-clean"))
    try:
        cluster.start_server("--watchdog-seconds", "0.2", "--flush-seconds", "0.3")
        ranks = [cluster.start_rank(0, steps=120), cluster.start_rank(1, steps=95)]
        for started in ranks:
            started.wait()
        cluster.final_status = cluster.wait_for(lambda s: s["summary"] == "finished")
        yield cluster
    finally:
        cluster.close()


def test_both_ranks_reach_parquet_unchanged_and_the_run_is_finished(clean_run):
    cluster = clean_run
    for rank, steps in ((0, 120), (1, 95)):
        local = tensorquill.load_records(cluster.work / "tensorquill-out" / f"{RUN_ID}.rank{rank}.jsonl")
        stored = cluster.stored(rank)
        assert [r["step"] for r in local] == list(range(steps))
        assert [r["step"] for r in stored] == list(range(steps))  # none lost, none twice, in order
        assert stored == local  # and every column is the rank's own
        assert {r["rank"] for r in stored} == {rank}
        sent = cluster.work / "tensorquill-out" / f"{RUN_ID}.rank{rank}.jsonl"
        assert read_offset(offset_path(sent)) == sent.stat().st_size  # close() saw it all stored
    assert cluster.stored(0) != cluster.stored(1)  # not the same data stored twice
    assert sorted(p.name for p in (cluster.run_dir).glob("rank=*")) == ["rank=0", "rank=1"]

    status = cluster.final_status
    assert status["world_size"] == 2 and status["missing_ranks"] == []
    assert [(r["rank"], r["state"], r["stream"], r["last_step"]) for r in status["ranks"]] == [
        (0, "finished", "finished", 119), (1, "finished", "finished", 94)]
    assert status["summary"] == "finished"
    assert tensorquill.build_evidence(cluster.run_dir, rank=0)["watchdog"]["summary"] == "finished"


def _cli(cluster: Cluster, *args: str) -> tuple[str, dict]:
    out = cluster.work / "evidence.json"
    out.unlink(missing_ok=True)
    done = subprocess.run(
        # `--detectors python`: these reads are compared with `build_evidence` on each rank's own file,
        # which is the Python detectors'. Until Nov 17, 2026 they were the default for one read; `auto`
        # would now take the Rust ones wherever the binary is found and add `"detectors": "rust"`.
        [sys.executable, "-m", "tensorquill", str(cluster.run_dir), *args, "-o", str(out), "--detectors", "python"],
        cwd=cluster.work, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout, json.loads(out.read_text(encoding="utf-8"))


def _from_file(cluster: Cluster, rank: int) -> dict:
    """What the detectors say about one rank's own local file, as it reads back from JSON."""
    path = cluster.work / "tensorquill-out" / f"{RUN_ID}.rank{rank}.jsonl"
    return json.loads(json.dumps(tensorquill.build_evidence(path)))


def test_the_command_reads_the_rank_it_is_asked_for_and_rank_0_by_default(clean_run):
    cluster = clean_run
    mine = {rank: _from_file(cluster, rank) for rank in (0, 1)}
    assert mine[0] != mine[1]  # the two ranks differ, so reading the wrong one would show

    for rank in (1, 0):
        out, evidence = _cli(cluster, "--rank", str(rank))
        assert "several ranks here" not in out
        assert evidence["watchdog"]["summary"] == "finished"
        assert _without_status(evidence) == _without_status(mine[rank])

    out, evidence = _cli(cluster)  # no --rank: the command's own default
    assert "several ranks here; reading rank 0" in out
    assert evidence["watchdog"]["summary"] == "finished"
    assert _without_status(evidence) == _without_status(mine[0])


# --- One rank is killed while the other keeps stepping ----------------------------------------


def test_a_killed_rank_is_lost_and_named_while_the_other_stays_active(cluster):
    cluster.start_server(*WATCHDOG)
    survivor, victim = cluster.start_rank(0), cluster.start_rank(1)
    both = cluster.wait_for(lambda s: len(s["ranks"]) == 2 and all(
        r["state"] == "active" and (r["last_step"] or 0) >= 30 for r in s["ranks"]))
    assert both["world_size"] == 2 and both["summary"] == "active" and both["missing_ranks"] == []

    victim.kill()  # mid-run: the stream is never ended
    status = cluster.wait_for(lambda s: _state(s, 1)["state"] == "lost")
    assert status["summary"] == "degraded"
    assert _state(status, 1)["stream"] == "dropped" and _state(status, 1)["severity"] == "warning"
    assert _state(status, 0)["state"] == "active" and _state(status, 0)["stream"] == "open"
    assert status["missing_ranks"] == []  # it connected; it is not missing, it is lost
    lost_at = _state(status, 1)["last_step"]
    assert lost_at >= 30

    # The survivor stays active, and keeps advancing, through several silence floors.
    before = _state(status, 0)["last_step"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        later = cluster.status()
        assert later["summary"] == "degraded" and _state(later, 0)["state"] == "active", later
        assert _state(later, 1)["state"] == "lost" and _state(later, 1)["last_step"] == lost_at
        time.sleep(0.25)
    assert _state(later, 0)["last_step"] > before

    evidence = tensorquill.build_evidence(cluster.run_dir, rank=0)
    assert not [i for i in evidence["incidents"] if i["kind"] == "job_stalled"]
    [incident] = [i for i in evidence["incidents"] if i["kind"] == "rank_silent"]
    assert [r["rank"] for r in incident["ranks"]] == [1] and incident["ranks"][0]["state"] == "lost"
    assert incident["step"] == lost_at and incident["newest_step"] > lost_at
    assert incident["missing_ranks"] == [] and incident["assessment"] == "lost_connection"
    # The server writes the same incident into the status itself.
    assert later["incidents"] == tensorquill.evidence.detect_watchdog_incidents(later)
    assert [(i["kind"], i["step"], i["assessment"]) for i in later["incidents"]] == [("rank_silent", lost_at, "lost_connection")]
    assert "rank 1 sent no new step since step" in tensorquill.incident_summary(evidence)

    # What the killed rank had delivered is in Parquet, whole and in order, a prefix of its file.
    delivered = cluster.stored(1)
    on_disk = _local_prefix(victim.local_file)
    assert [r["step"] for r in delivered] == list(range(len(delivered)))
    assert len(delivered) >= lost_at + 1 and delivered == on_disk[: len(delivered)]

    # The survivor goes on to finish cleanly; the killed rank stays lost.
    survivor.finish()
    final = cluster.wait_for(lambda s: _state(s, 0)["state"] == "finished")
    assert _state(final, 1)["state"] == "lost" and final["summary"] != "finished"
    steps = [r["step"] for r in cluster.stored(0)]
    assert steps == list(range(len(steps))) == [r["step"] for r in _local_prefix(survivor.local_file)]


# --- One rank never starts ---------------------------------------------------------------------


@pytest.mark.parametrize("missing", [1, 0])
def test_a_rank_that_never_starts_is_named_by_world_size_while_the_other_runs(cluster, missing):
    running = 1 - missing
    cluster.start_server(*WATCHDOG)
    only = cluster.start_rank(running)  # WORLD_SIZE is 2, and the other rank is never launched
    status = cluster.wait_for(lambda s: s["missing_ranks"] == [missing] and _state(s, running)["state"] == "active")
    assert status["world_size"] == 2 and status["summary"] == "degraded"
    assert [r["rank"] for r in status["ranks"]] == [running]
    time.sleep(2.5)  # well past --silent-seconds: stepping keeps the rank active, and the other stays missing
    later = cluster.status()
    assert later["missing_ranks"] == [missing] and later["summary"] == "degraded"
    assert _state(later, running)["state"] == "active"

    evidence = tensorquill.build_evidence(cluster.run_dir, rank=running)
    assert not [i for i in evidence["incidents"] if i["kind"] == "job_stalled"]
    [incident] = [i for i in evidence["incidents"] if i["kind"] == "rank_silent"]
    assert incident["missing_ranks"] == [missing] and incident["ranks"] == []
    assert incident["assessment"] == "ranks_never_connected" and incident["world_size"] == 2
    assert incident["step"] == incident["newest_step"] > 0
    assert later["incidents"] == tensorquill.evidence.detect_watchdog_incidents(later)
    assert [(i["kind"], i["assessment"]) for i in later["incidents"]] == [("rank_silent", "ranks_never_connected")]
    assert f"{missing} never connected" in tensorquill.incident_summary(evidence)

    # The rank that ran closes cleanly; with the other never seen, the run is not finished.
    only.finish()
    final = cluster.wait_for(lambda s: _state(s, running)["state"] == "finished")
    assert final["missing_ranks"] == [missing] and final["summary"] == "degraded"
    steps = [r["step"] for r in cluster.stored(running)]
    assert steps == list(range(len(steps))) == [r["step"] for r in tensorquill.load_records(only.local_file)]
    assert not (cluster.run_dir / f"rank={missing}").exists()


def test_a_rank_that_starts_after_being_reported_missing_stops_being_missing(cluster):
    cluster.start_server(*WATCHDOG)
    first = cluster.start_rank(0)
    cluster.wait_for(lambda s: s["missing_ranks"] == [1] and s["summary"] == "degraded")
    late = cluster.start_rank(1)  # the launcher was slow, not the rank dead
    status = cluster.wait_for(lambda s: s["missing_ranks"] == [] and len(s["ranks"]) == 2
                              and all(r["state"] == "active" for r in s["ranks"]))
    assert status["summary"] == "active" and status["world_size"] == 2
    assert not [i for i in tensorquill.build_evidence(cluster.run_dir, rank=0)["incidents"] if i["kind"] == "rank_silent"]
    first.finish()
    late.finish()
    assert cluster.wait_for(lambda s: s["summary"] == "finished")["missing_ranks"] == []


# --- The server is killed and restarted under both ranks --------------------------------------


def test_a_server_killed_and_restarted_under_two_streaming_ranks_loses_and_repeats_nothing(cluster):
    # A long flush interval and no step threshold: when the server is killed it holds steps only
    # in memory, steps the ranks have sent but that were never confirmed.
    flags = ("--flush-seconds", "30", "--flush-steps", "1000000", "--watchdog-seconds", "0.2")
    server = cluster.start_server(*flags)
    port = server.port
    ranks = [cluster.start_rank(0), cluster.start_rank(1)]
    cluster.wait_for(lambda s: len(s["ranks"]) == 2 and all((r["last_step"] or 0) >= 30 for r in s["ranks"]))
    assert cluster.stored(0) == [] and cluster.stored(1) == []  # sent, but not stored
    server.kill()
    time.sleep(2)  # both ranks keep stepping with no server
    restarted_ms = int(time.time() * 1000)
    cluster.start_server(*flags, port=port)
    # The dead server left its last watchdog.json behind, both ranks active; only a status the
    # new server wrote, after both ranks reconnected to it, counts.
    status = cluster.wait_for(lambda s: s["written_at_ms"] > restarted_ms and s["server_running"]
                              and len(s["ranks"]) == 2 and all(r["state"] == "active" for r in s["ranks"]))
    assert status["world_size"] == 2 and status["summary"] == "active"
    for started in ranks:
        started.finish()
    cluster.wait_for(lambda s: s["summary"] == "finished")
    for started in ranks:
        local = tensorquill.load_records(started.local_file)
        stored = cluster.stored(started.rank)
        assert len(local) > 60
        assert [r["step"] for r in stored] == list(range(len(local)))  # none lost, none twice, in order
        assert stored == local
        assert read_offset(offset_path(started.local_file)) == started.local_file.stat().st_size
