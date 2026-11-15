"""End to end: a real capture streamed to the real `tensorquill-server`, read back from the
Parquet parts it wrote.

Needs the server binary: `cargo build --manifest-path server/Cargo.toml`, or point
`TENSORQUILL_SERVER_BIN` at one. Skipped when there is none.
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

import tensorquill  # noqa: E402
from tensorquill.flight import offset_path, read_offset, row_to_record  # noqa: E402


def _server_binary() -> Path | None:
    named = os.environ.get("TENSORQUILL_SERVER_BIN")
    if named:
        return Path(named)
    target = Path(__file__).resolve().parents[2] / "server" / "target"
    built = [p for p in (target / "debug" / "tensorquill-server", target / "release" / "tensorquill-server") if p.exists()]
    return max(built, key=lambda p: p.stat().st_mtime) if built else None


BINARY = _server_binary()
pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")


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


def stored_records(store: Path, run_id: str, rank: int = 0) -> list[dict]:
    parts = sorted((store / "runs" / run_id / f"rank={rank}").glob("attempt=*/part-*.parquet"))
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
    monkeypatch.delenv("TENSORQUILL_TOKEN", raising=False)
    store = tmp_path / "store"
    server = Server(store)
    try:
        # Full capture: block statistics, gradient and weight norms, router load, a probe
        # pass and logged scalars, so every kind of column is exercised.
        session = tensorquill.init(
            toy_moe, run_id="e2e", every=10, capture="full", routers={"router": 4}, weights=5,
            probe=torch.randn(4, 8), probe_every=20, endpoint=server.endpoint,
        )
        train(toy_moe, session, 130)
        session.close()

        path = tmp_path / "tensorquill-out" / "e2e.rank0.jsonl"
        written = tensorquill.load_records(path)
        assert [r["step"] for r in written] == list(range(130))
        assert all(r["schema_version"] == 1 for r in written)
        # close() returned only after the server confirmed the whole file as stored.
        assert read_offset(offset_path(path)) == path.stat().st_size
        stored = stored_records(store, "e2e")
        assert stored == written
        assert any("probe" in r for r in stored) and any("weight_norms" in r for r in stored)
        assert any(r["scalars"]["nan_now_and_then"] == "nan" for r in stored)
        # The detectors read the same thing from either side.
        assert tensorquill.build_evidence(stored) == tensorquill.build_evidence(written)
        assert tensorquill.load_records(store / "runs" / "e2e") == written
        assert tensorquill.build_evidence(store / "runs" / "e2e") == tensorquill.build_evidence(path)
    finally:
        assert server.terminate() == 0


def test_server_killed_for_200_steps_loses_and_repeats_nothing(tmp_path, toy_moe, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_TOKEN", raising=False)
    store = tmp_path / "store"
    # A 30-second flush interval: when the server is killed it holds steps only in memory,
    # steps the sender has sent but that were never confirmed.
    server = Server(store, 0, "--flush-seconds", "30")
    port = server.port
    session = tensorquill.init(
        toy_moe, run_id="outage", every=10, routers={"router": 4}, endpoint=server.endpoint
    )
    sender = session._sink.sender
    sender._batch_seconds, sender._poll_seconds, sender._backoff_cap = 0.05, 0.02, 0.2
    path = tmp_path / "tensorquill-out" / "outage.rank0.jsonl"
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
    written = tensorquill.load_records(path)
    stored = stored_records(store, "outage")
    assert [r["step"] for r in stored] == list(range(350))  # none lost, none twice, in order
    assert stored == written
    assert read_offset(offset_path(path)) == path.stat().st_size


def _steps(path: Path, run_id: str, rank: int, steps: range) -> None:
    import json

    with open(path, "a", encoding="utf-8") as file:
        for step in steps:
            file.write(json.dumps({"schema_version": 1, "run_id": run_id, "rank": rank, "step": step,
                                   "ts": 1.0, "grad_norm": 0.5}) + "\n")


def test_the_python_sender_presents_its_token_and_a_non_zero_rank(tmp_path, monkeypatch, capsys):
    from tensorquill.flight import FlightSender

    store = tmp_path / "store"
    server = Server(store, 0, "--token", "s3cret")
    try:
        path = tmp_path / "tok.rank1.jsonl"
        _steps(path, "tok", 1, range(120))
        monkeypatch.setenv("TENSORQUILL_TOKEN", "s3cret")
        assert FlightSender(path, server.endpoint, "tok", 1).close(20) is True

        wrong = tmp_path / "bad.rank0.jsonl"
        _steps(wrong, "bad", 0, range(5))
        monkeypatch.setenv("TENSORQUILL_TOKEN", "not-it")
        sender = FlightSender(wrong, server.endpoint, "bad", 0)
        assert sender.close(2) is False
        assert "Bearer token" in str(sender.last_error) and "Bearer token" in capsys.readouterr().err
    finally:
        server.terminate()
    assert [r["step"] for r in stored_records(store, "tok", rank=1)] == list(range(120))
    assert not (store / "runs" / "bad").exists()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_forked_dataloader_workers_do_not_disturb_the_sender(tmp_path, monkeypatch):
    # Linux's default: DataLoader workers are forked while the sender's gRPC thread runs.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_TOKEN", raising=False)
    store = tmp_path / "store"
    server = Server(store)
    try:
        model = torch.nn.Linear(8, 1)
        session = tensorquill.init(model, run_id="forked", capture="full", every=5, endpoint=server.endpoint)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        data = torch.utils.data.TensorDataset(torch.randn(128, 8))
        for _ in range(3):
            loader = torch.utils.data.DataLoader(data, batch_size=8, num_workers=2, multiprocessing_context="fork")
            for (x,) in loader:
                optimizer.zero_grad()
                model(x).pow(2).mean().backward()
                optimizer.step()
        session.close()
    finally:
        server.terminate()
    assert [r["step"] for r in stored_records(store, "forked")] == list(range(48))


def test_a_resumed_run_keeps_its_repeated_steps_and_watch_reads_them(tmp_path, toy_moe, monkeypatch):
    from tensorquill.evidence import watch

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_TOKEN", raising=False)
    store = tmp_path / "store"
    server = Server(store)
    path = tmp_path / "tensorquill-out" / "resumed.rank0.jsonl"
    run_dir = store / "runs" / "resumed"
    try:
        session = tensorquill.init(toy_moe, run_id="resumed", every=10, routers={"router": 4}, endpoint=server.endpoint)
        train(toy_moe, session, 120)
        session.close()
        live = watch(run_dir, tmp_path / "live.json", interval=0.0, polls=1)
        assert live is not None and len(tensorquill.load_records(run_dir)) == 120

        # The job crashed and resumes from the checkpoint at step 80, under the same run ID.
        session = tensorquill.init(
            toy_moe, run_id="resumed", every=10, routers={"router": 4}, endpoint=server.endpoint, start_step=80
        )
        train(toy_moe, session, 60)
        session.close()

        # The offset file is lost: the whole file, both attempts, is sent again.
        offset_path(path).unlink()
        from tensorquill.flight import send_file

        assert send_file(path, server.endpoint, "resumed") is True
    finally:
        assert server.terminate() == 0
    written = tensorquill.load_records(path, all_attempts=True)
    assert len(written) == 180 and len({r["attempt"] for r in written}) == 2
    assert len(list(run_dir.glob("rank=0/attempt=*"))) == 2
    assert tensorquill.load_records(run_dir, all_attempts=True) == written  # none dropped, none twice
    resolved = tensorquill.load_records(run_dir)
    assert [r["step"] for r in resolved] == list(range(140))
    assert {r["attempt"] for r in resolved if r["step"] >= 80} == {max(r["attempt"] for r in written)}
    evidence = tensorquill.build_evidence(run_dir)
    assert evidence == tensorquill.build_evidence(path)
    assert not [i for i in evidence["incidents"] if i["kind"] == "repeated_steps"]


def test_grpcs_streams_with_custom_ca_and_rejects_untrusted_server(tmp_path, toy_moe, monkeypatch):
    import shutil

    from tensorquill.flight import FlightSender

    if shutil.which("openssl") is None:
        pytest.skip("needs openssl")

    cert, key, conf = tmp_path / "cert.pem", tmp_path / "key.pem", tmp_path / "openssl.cnf"
    conf.write_text(
        "[req]\ndistinguished_name=dn\nx509_extensions=san\nprompt=no\n"
        "[dn]\nCN=localhost\n"
        "[san]\nsubjectAltName=DNS:localhost,IP:127.0.0.1\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(key), "-out", str(cert), "-days", "1", "-config", str(conf)],
        check=True, capture_output=True,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_TOKEN", raising=False)
    monkeypatch.delenv("TENSORQUILL_TLS_CA", raising=False)
    store = tmp_path / "store"
    server = Server(store, 0, "--tls-cert", str(cert), "--tls-key", str(key))
    endpoint = f"grpcs://127.0.0.1:{server.port}"
    try:
        # Without TENSORQUILL_TLS_CA the self-signed certificate is rejected.
        untrusted = tmp_path / "untrusted.rank0.jsonl"
        _steps(untrusted, "untrusted", 0, range(3))
        bad_sender = FlightSender(untrusted, endpoint, "untrusted", 0)
        assert bad_sender.close(timeout=0.5) is False
        assert "certificate verify failed" in str(bad_sender.last_error)  # not some other failure
        assert not (store / "runs" / "untrusted").exists()

        # With TENSORQUILL_TLS_CA pointing at the server certificate, streaming succeeds.
        monkeypatch.setenv("TENSORQUILL_TLS_CA", str(cert))
        session = tensorquill.init(
            toy_moe, run_id="tls", every=5, routers={"router": 4}, endpoint=endpoint
        )
        train(toy_moe, session, 20)
        session.close()
    finally:
        assert server.terminate() == 0

    written = tensorquill.load_records(tmp_path / "tensorquill-out" / "tls.rank0.jsonl")
    assert len(written) == 20
    assert stored_records(store, "tls") == written
