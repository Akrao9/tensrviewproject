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
    return [row_to_record(row) for part in parts for row in pq.read_table(part, partitioning=None).to_pylist()]


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
        # Optimizer ownership travels in extra_json and must survive the server.
        assert any(r.get("optimizer_ownership", {}).get("state") == "known" for r in stored)
        assert any(r["scalars"]["nan_now_and_then"] == "nan" for r in stored)
        # The detectors read the same thing from either side.
        assert tensorquill.build_evidence(stored) == tensorquill.build_evidence(written)
        assert tensorquill.load_records(store / "runs" / "e2e") == written
        # (This server writes no status. A store is expected to have one, a local file is not.)
        assert tensorquill.build_evidence(store / "runs" / "e2e") == tensorquill.build_evidence(
            path, watchdog=store / "runs" / "e2e" / "watchdog.json"
        )
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
        # Heartbeats every 50 ms, so beats and forks overlap.
        session = tensorquill.init(model, run_id="forked", capture="full", every=5, endpoint=server.endpoint,
                                   heartbeat_seconds=0.05)
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
    # The server's own progress status is all the directory has and the file has not.
    assert evidence["watchdog"]["summary"] == "finished"  # close() ended every stream cleanly
    from_file = tensorquill.build_evidence(path)
    # A local file no server was watching has no status to miss.
    assert "no_status" not in [r["code"] for r in from_file["run_state"]["reasons"]]
    assert from_file["kind_checks"]["job_stalled"]["value"] == "not_configured"
    assert evidence["kind_checks"]["job_stalled"]["value"] == "checked"
    assert _without_status(evidence) == _without_status(from_file)
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


def test_the_server_calls_a_job_that_goes_quiet_stalled_and_evidence_says_so(tmp_path, toy_moe, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_TOKEN", raising=False)
    store = tmp_path / "store"
    server = Server(store, 0, "--silent-seconds", "1", "--silent-factor", "0", "--watchdog-seconds", "0.2", "--flush-seconds", "0.3")
    run_dir = store / "runs" / "hung"
    try:
        session = tensorquill.init(
            toy_moe, run_id="hung", every=10, routers={"router": 4}, endpoint=server.endpoint,
        )
        train(toy_moe, session, 40)  # then the job hangs: the session stays open and sends nothing
        deadline = time.monotonic() + 15
        status = None
        while time.monotonic() < deadline:
            status = tensorquill.evidence.read_watchdog(run_dir / "watchdog.json")
            if status and status["summary"] == "stalled" and list(run_dir.glob("rank=0/attempt=*/part-*.parquet")):
                break
            time.sleep(0.2)
        assert status and status["summary"] == "stalled", status
        assert status["world_size"] == 1 and status["ranks"][0]["state"] == "silent"
        assert status["ranks"][0]["stream"] == "open" and status["ranks"][0]["last_step"] == 39
        evidence = tensorquill.build_evidence(run_dir)
        [incident] = [i for i in evidence["incidents"] if i["kind"] == "job_stalled"]
        assert incident["step"] == 39
        assert [i["kind"] for i in same_incidents_as_python(status)] == ["job_stalled"]
        # Training resumes and finishes: the next status says so.
        session.close()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = tensorquill.evidence.read_watchdog(run_dir / "watchdog.json")
            if status["summary"] == "finished":
                break
            time.sleep(0.2)
        assert status["summary"] == "finished" and same_incidents_as_python(status) == []
    finally:
        assert server.terminate() == 0


_DIES = """
import os, signal, sys, time
import torch, tensorquill
model = torch.nn.Linear(4, 2)
session = tensorquill.init(model, run_id="dies", every=10, endpoint=sys.argv[1])
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for _ in range(60):
    optimizer.zero_grad()
    model(torch.randn(8, 4)).sum().backward()
    optimizer.step()
time.sleep(1.5)  # let the sender deliver what was written
if sys.argv[2] == "kill":
    os.kill(os.getpid(), signal.SIGKILL)  # as the OOM killer does: no close(), no atexit
os._exit(0)  # leaves without close(): the connection closes, the stream is never ended
"""


@pytest.mark.parametrize("how", ["kill", "exit"])
def test_a_job_that_dies_without_closing_is_lost_not_finished(tmp_path, how):
    server = Server(tmp_path / "store", 0, "--silent-seconds", "1", "--silent-factor", "0", "--watchdog-seconds", "0.2", "--flush-seconds", "0.3")
    run_dir = tmp_path / "store" / "runs" / "dies"
    try:
        import sys

        child = subprocess.run(
            [sys.executable, "-c", _DIES, server.endpoint, how], cwd=tmp_path, timeout=60,
            capture_output=True, text=True,
        )
        assert child.returncode == (-signal.SIGKILL if how == "kill" else 0), child.stderr
        deadline = time.monotonic() + 15
        status = None
        while time.monotonic() < deadline:
            status = tensorquill.evidence.read_watchdog(run_dir / "watchdog.json")
            if status and status["ranks"] and status["ranks"][0]["state"] == "lost":
                break
            time.sleep(0.2)
        assert status and status["ranks"][0]["state"] == "lost", status
        assert status["ranks"][0]["stream"] == "dropped"  # never "finished": nobody called close()
        assert status["summary"] == "stalled" and status["ranks"][0]["last_step"] >= 49
        [incident] = same_incidents_as_python(status)
        assert (incident["kind"], incident["assessment"]) == ("job_stalled", "lost_connection")
    finally:
        assert server.terminate() == 0


def _status_of(run_dir):
    return tensorquill.evidence.read_watchdog(run_dir / "watchdog.json")


def same_incidents_as_python(status):
    """The server writes the status as incidents too; Python's reading of the same status
    is the reference."""
    assert status["incidents"] == tensorquill.evidence.detect_watchdog_incidents(status), status
    return status["incidents"]


def _wait_for(run_dir, ready, seconds=20):
    deadline = time.monotonic() + seconds
    status = None
    while time.monotonic() < deadline:
        status = _status_of(run_dir)
        if status and ready(status):
            return status
        time.sleep(0.1)
    raise AssertionError(f"timed out; last status {status}")


def test_evaluation_is_paused_and_measured_but_a_quiet_training_phase_is_a_suspected_hang(tmp_path, toy_moe, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_TOKEN", raising=False)
    server = Server(tmp_path / "store", 0, "--silent-seconds", "1", "--silent-factor", "0", "--critical-seconds", "3",
                    "--watchdog-seconds", "0.1", "--flush-seconds", "0.2")
    run_dir = tmp_path / "store" / "runs" / "phases"
    try:
        session = tensorquill.init(toy_moe, run_id="phases", every=10, routers={"router": 4},
                                   endpoint=server.endpoint, heartbeat_seconds=0.2)
        train(toy_moe, session, 30)
        # Evaluation: no steps for longer than the silence floor, heartbeats saying so.
        with session.phase("eval"):
            status = _wait_for(run_dir, lambda s: s["ranks"][0]["phase"] == "eval"
                               and s["ranks"][0]["state"] == "paused")
            assert status["summary"] == "active" and status["ranks"][0]["severity"] is None
            assert tensorquill.build_evidence(run_dir)["incidents"] == [] or all(
                i["kind"] not in ("job_stalled", "rank_silent") for i in tensorquill.build_evidence(run_dir)["incidents"])
        status = _wait_for(run_dir, lambda s: s["pauses"].get("eval", {}).get("count") == 1)
        longest = status["pauses"]["eval"]["max_seconds"]
        assert longest > 0.2  # the pause was measured by the server from the phase changes
        assert status["recommended_silent_seconds"] == max(1.0, float(-(-2 * longest // 1)))  # twice, rounded up, not below the floor
        # Back in training, the rank goes quiet: its heartbeats continue and say `train`.
        status = _wait_for(run_dir, lambda s: s["ranks"][0]["state"] == "silent")
        assert status["ranks"][0]["phase"] == "train" and status["ranks"][0]["severity"] == "warning"
        assert status["ranks"][0]["heartbeat_age_seconds"] is not None
        status = _wait_for(run_dir, lambda s: s["ranks"][0]["severity"] == "critical")
        assert status["summary"] == "stalled"
        evidence = tensorquill.build_evidence(run_dir)
        [incident] = [i for i in evidence["incidents"] if i["kind"] == "job_stalled"]
        assert (incident["severity"], incident["assessment"]) == ("critical", "suspected_hang")
        status = _status_of(run_dir)
        if status["summary"] == "stalled":  # read a moment later than the evidence was
            [written] = same_incidents_as_python(status)
            assert (written["severity"], written["assessment"]) == ("critical", "suspected_hang")
        assert "Pauses the job reported: eval up to" in tensorquill.incident_summary(evidence)
        session.close()
        _wait_for(run_dir, lambda s: s["summary"] == "finished")
    finally:
        assert server.terminate() == 0


_FREEZES = """
import os, signal, sys, time
import torch, tensorquill
model = torch.nn.Linear(4, 2)
session = tensorquill.init(model, run_id="frozen", every=10, endpoint=sys.argv[1], heartbeat_seconds=0.2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for _ in range(30):
    optimizer.zero_grad()
    model(torch.randn(8, 4)).sum().backward()
    optimizer.step()
time.sleep(1.0)
os.kill(os.getpid(), signal.SIGSTOP)  # every thread stops, the connection stays open
"""


def test_a_stopped_process_whose_heartbeats_cease_is_unresponsive(tmp_path):
    import sys

    server = Server(tmp_path / "store", 0, "--silent-seconds", "1", "--silent-factor", "0", "--critical-seconds", "3",
                    "--watchdog-seconds", "0.1", "--flush-seconds", "0.2")
    run_dir = tmp_path / "store" / "runs" / "frozen"
    child = subprocess.Popen([sys.executable, "-c", _FREEZES, server.endpoint], cwd=tmp_path,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        status = _wait_for(run_dir, lambda s: s["ranks"][0]["state"] == "unresponsive", 40)
        assert status["ranks"][0]["stream"] == "open" and status["summary"] == "stalled"
        evidence = tensorquill.build_evidence(run_dir)
        assert "heartbeats stopped too" in tensorquill.incident_summary(evidence)
    finally:
        child.kill()
        child.wait(20)
        assert server.terminate() == 0


_WAITS = """
import sys, threading, time
import torch, tensorquill
model = torch.nn.Linear(4, 2)
session = tensorquill.init(model, run_id="waits", every=10, endpoint=sys.argv[1], heartbeat_seconds=0.2,
                           stack_seconds=0.6)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def wait_for_the_other_ranks():
    threading.Event().wait(120)  # a call that never returns, with the interpreter lock released

def train_step(n):
    optimizer.zero_grad()
    model(torch.randn(8, 4)).sum().backward()
    optimizer.step()
    if n == 29:
        wait_for_the_other_ranks()

for n in range(30):
    train_step(n)
"""


def test_a_rank_stuck_in_a_call_is_reported_with_the_call_it_is_stuck_in(tmp_path):
    import sys

    server = Server(tmp_path / "store", 0, "--silent-seconds", "1", "--silent-factor", "0", "--critical-seconds", "30",
                    "--watchdog-seconds", "0.1", "--flush-seconds", "0.2")
    run_dir = tmp_path / "store" / "runs" / "waits"
    script = tmp_path / "job.py"
    script.write_text(_WAITS)
    child = subprocess.Popen([sys.executable, str(script), server.endpoint], cwd=tmp_path,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        status = _wait_for(run_dir, lambda s: s["ranks"][0]["state"] == "silent"
                           and (s["ranks"][0].get("stack") or {}).get("unchanged_seconds", 0) > 0.3, 40)
        stack = status["ranks"][0]["stack"]
        [main] = [t for t in stack["threads"] if t["main"]]
        names = [f["function"] for f in main["frames"]]
        # Innermost first: the standard library's wait, then the job's own functions.
        assert names[0] == "wait" and names[-3:] == ["wait_for_the_other_ranks", "train_step", "<module>"]
        assert not any(t["name"].startswith("tensorquill-") for t in stack["threads"])
        assert stack["still_seconds"] >= 0.6
        evidence = tensorquill.build_evidence(run_dir)
        [incident] = [i for i in evidence["incidents"] if i["kind"] == "job_stalled"]
        # The server's own incidents say the same, and neither carries the whole stack.
        [written] = same_incidents_as_python(status)
        assert written["stuck_at"][0]["caller"]["function"] == "wait_for_the_other_ranks"
        assert "stack" not in written["ranks"][0] and "stack" not in incident["ranks"][0]
        [place] = incident["stuck_at"]
        assert place["function"] == "wait" and place["caller"]["function"] == "wait_for_the_other_ranks"
        assert place["caller"]["file"].endswith("job.py") and "in_tensorquill" not in place
        assert incident["assessment"] == "suspected_hang"
        text = tensorquill.incident_summary(evidence)
        assert "Last stack: rank 0: main thread in `wait` (threading.py:" in text
        assert "called from `wait_for_the_other_ranks` (job.py:" in text
    finally:
        child.kill()
        child.wait(20)
        assert server.terminate() == 0


def test_a_pause_with_no_sign_of_work_is_limited_and_one_that_ticks_is_not(tmp_path, toy_moe, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_TOKEN", raising=False)
    server = Server(tmp_path / "store", 0, "--silent-seconds", "1", "--silent-factor", "0", "--critical-seconds", "30", "--pause-seconds", "6",
                    "--watchdog-seconds", "0.1", "--flush-seconds", "0.2")
    run_dir = tmp_path / "store" / "runs" / "bounded"
    try:
        session = tensorquill.init(toy_moe, run_id="bounded", every=10, routers={"router": 4},
                                   endpoint=server.endpoint, heartbeat_seconds=0.2)
        train(toy_moe, session, 30)
        with session.phase("eval"):
            # No sign of work: paused for a while, then a suspected hang, phase still `eval`.
            _wait_for(run_dir, lambda s: s["ranks"][0]["state"] == "paused")
            status = _wait_for(run_dir, lambda s: s["ranks"][0]["state"] == "silent", 30)
            assert status["ranks"][0]["phase"] == "eval" and status["summary"] == "stalled"
        train(toy_moe, session, 10)  # steps end the pause: back to active
        _wait_for(run_dir, lambda s: s["ranks"][0]["state"] == "active" and s["ranks"][0]["phase"] == "train")
        with session.phase("eval"):
            deadline = time.monotonic() + 8.0  # longer than the limit, but evaluation batches keep ticking
            while time.monotonic() < deadline:
                session.tick()
                time.sleep(0.1)
            assert _status_of(run_dir)["ranks"][0]["state"] == "paused"
        session.close()
    finally:
        assert server.terminate() == 0


def _without_status(evidence: dict) -> dict:
    """The evidence less what only a run with the server's status has: the status, whether the stall
    kinds were checked, and the reason the run state gives when a status was expected and is absent
    (`no_status`)."""
    out = {k: v for k, v in evidence.items() if k not in ("watchdog", "run_state", "kind_checks")}
    reasons = [r for r in evidence["run_state"]["reasons"] if r["code"] != "no_status"]
    out["run_state_reasons"] = reasons
    out["kind_checks"] = {k: v for k, v in evidence["kind_checks"].items() if k not in ("job_stalled", "rank_silent")}
    return out
