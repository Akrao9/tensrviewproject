"""End to end: a real capture streamed to the real `tensorlens-server`, read back from the
Parquet parts it wrote.

Needs the server binary: `cargo build --manifest-path server/Cargo.toml`, or point
`TENSORLENS_SERVER_BIN` at one. Skipped when there is none.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

import tensorlens  # noqa: E402
from tensorlens.flight import offset_path, read_offset, row_to_record  # noqa: E402


def _server_binary() -> Path | None:
    named = os.environ.get("TENSORLENS_SERVER_BIN")
    if named:
        return Path(named)
    target = Path(__file__).resolve().parents[2] / "server" / "target"
    built = [p for p in (target / "debug" / "tensorlens-server", target / "release" / "tensorlens-server") if p.exists()]
    return max(built, key=lambda p: p.stat().st_mtime) if built else None


BINARY = _server_binary()
pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorlens-server is not built")


class Server:
    def __init__(self, store: Path, port: int = 0, *extra: str) -> None:
        env = {k: v for k, v in os.environ.items() if k != "TENSORLENS_TOKEN"}
        self.process = subprocess.Popen(
            [str(BINARY), "--listen", f"127.0.0.1:{port}", "--store", str(store), *extra],
            stdout=subprocess.PIPE, text=True, env=env,
        )
        line = self.process.stdout.readline()  # "tensorlens-server listening on 127.0.0.1:PORT"
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


def stored_records(store: Path, run_id: str, rank: int = 0) -> list[dict]:
    parts = sorted((store / "runs" / run_id / f"rank={rank}").glob("part-*.parquet"))
    return [row_to_record(row) for part in parts for row in pq.read_table(part).to_pylist()]


def train(model: torch.nn.Module, session, steps: int, pause: float = 0.0) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for step in range(steps):
        optimizer.zero_grad()
        loss = model(torch.randn(16, 8)).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        session.log({"train/loss": loss, "nan_now_and_then": float("nan") if step % 7 == 0 else 1.0})
        optimizer.step()
        if pause:
            time.sleep(pause)


def test_a_capture_file_reaches_parquet_unchanged(tmp_path, toy_moe, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORLENS_TOKEN", raising=False)
    store = tmp_path / "store"
    server = Server(store)
    try:
        # Full capture: block statistics, gradient and weight norms, router load, a probe
        # pass and logged scalars, so every kind of column is exercised.
        session = tensorlens.init(
            toy_moe, run_id="e2e", every=10, capture="full", routers={"router": 4}, weights=5,
            probe=torch.randn(4, 8), probe_every=20, endpoint=server.endpoint,
        )
        train(toy_moe, session, 130)
        session.close()

        path = tmp_path / "tensorlens-out" / "e2e.rank0.jsonl"
        written = tensorlens.load_records(path)
        assert [r["step"] for r in written] == list(range(130))
        assert all(r["schema_version"] == 1 for r in written)
        # close() returned only after the server confirmed the whole file as stored.
        assert read_offset(offset_path(path)) == path.stat().st_size
        stored = stored_records(store, "e2e")
        assert stored == written
        assert any("probe" in r for r in stored) and any("weight_norms" in r for r in stored)
        assert any(r["scalars"]["nan_now_and_then"] == "nan" for r in stored)
        # The detectors read the same thing from either side.
        assert tensorlens.build_evidence(stored) == tensorlens.build_evidence(written)
    finally:
        assert server.terminate() == 0


def test_server_killed_for_200_steps_loses_and_repeats_nothing(tmp_path, toy_moe, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORLENS_TOKEN", raising=False)
    store = tmp_path / "store"
    # A 30-second flush interval: when the server is killed it holds steps only in memory,
    # steps the sender has sent but that were never confirmed.
    server = Server(store, 0, "--flush-seconds", "30")
    port = server.port
    session = tensorlens.init(
        toy_moe, run_id="outage", every=10, routers={"router": 4}, endpoint=server.endpoint
    )
    sender = session._sink.sender
    sender._batch_seconds, sender._poll_seconds, sender._backoff_cap = 0.05, 0.02, 0.2
    path = tmp_path / "tensorlens-out" / "outage.rank0.jsonl"
    try:
        train(toy_moe, session, 100, pause=0.002)
        session.flush()
        time.sleep(0.5)  # the sender has sent these steps; the server has not stored them
        assert stored_records(store, "outage") == []
        assert read_offset(offset_path(path)) == 0
        server.kill()

        train(toy_moe, session, 200, pause=0.002)  # 200 steps with no server
        session.flush()
        assert stored_records(store, "outage") == []

        server = Server(store, port, "--flush-seconds", "30")
        train(toy_moe, session, 50)
        session.close()
    finally:
        code = server.terminate()
    assert code == 0
    written = tensorlens.load_records(path)
    stored = stored_records(store, "outage")
    assert [r["step"] for r in stored] == list(range(350))  # none lost, none twice, in order
    assert stored == written
    assert read_offset(offset_path(path)) == path.stat().st_size
