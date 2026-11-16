"""End to end: the real `tensorquill-server` against an S3-compatible store.

The Python sender streams runs to `s3://<bucket>/<prefix>` and the tests read the objects back
with boto3: Parquet parts, `incidents.json` and `watchdog.json`, across a SIGKILL and a restart
of the server. What this covers: part writes, the footer-only (suffix range) watermark read,
per-rank listings (rank 1 and rank 10 share a name prefix), `PUT` replacing a JSON document,
the delimiter listing that finds runs after a restart, and the watchdog's state surviving a
restart. What it does not: real AWS, TLS to S3, IAM, latency, or GCS (there is no GCS test).

Needs the server binary (`cargo build --release --manifest-path server/Cargo.toml`, or
`TENSORQUILL_SERVER_BIN`), `pyarrow` (the `flight` extra), `boto3`, and an S3 emulator:
- `TENSORQUILL_S3_ENDPOINT=http://127.0.0.1:PORT` names one that is already running (MinIO,
  `moto_server`, ...; the tests create their own buckets in it), or
- `pip install 'moto[server]'` lets the tests start moto in a thread.
Skipped when any of that is missing. Keys are dummies; nothing here reaches a real account.
"""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")
boto3 = pytest.importorskip("boto3")
import pyarrow.parquet as pq  # noqa: E402

from tensorquill.flight import FlightSender, offset_path, row_to_record  # noqa: E402

from test_flight_e2e import BINARY  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")

KEY_ID = "tensorquill-test-key-id"
SECRET = "tensorquill-test-secret-never-logged"  # the server must not print this
REGION = "us-east-1"
FAST = ("--flush-steps", "50", "--flush-seconds", "0.2", "--watchdog-seconds", "0.2")


@pytest.fixture(scope="module")
def endpoint():
    named = os.environ.get("TENSORQUILL_S3_ENDPOINT")
    if named:
        yield named
        return
    try:
        from moto.server import ThreadedMotoServer
    except ImportError:
        pytest.skip("no S3 emulator: set TENSORQUILL_S3_ENDPOINT or pip install 'moto[server]'")
    emulator = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    emulator.start()
    host, port = emulator.get_host_and_port()
    try:
        yield f"http://{host}:{port}"
    finally:
        emulator.stop()


class Bucket:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.name = f"tensorquill-e2e-{uuid.uuid4().hex[:12]}"
        self.client = boto3.client(
            "s3", endpoint_url=endpoint, aws_access_key_id=KEY_ID, aws_secret_access_key=SECRET, region_name=REGION
        )
        self.client.create_bucket(Bucket=self.name)

    def keys(self, prefix: str = "") -> list[str]:
        found, token = [], None
        while True:
            page = self.client.list_objects_v2(
                Bucket=self.name, Prefix=prefix, **({"ContinuationToken": token} if token else {})
            )
            found += [item["Key"] for item in page.get("Contents", [])]
            token = page.get("NextContinuationToken")
            if not token:
                return sorted(found)

    def read(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.name, Key=key)["Body"].read()

    def json(self, key: str):
        return json.loads(self.read(key))

    def maybe_json(self, key: str):
        try:
            return self.json(key)
        except self.client.exceptions.NoSuchKey:
            return None

    def records(self, prefix: str, run: str, rank: int) -> list[dict]:
        """Every stored record of a rank, in part order."""
        parts = [k for k in self.keys(f"{prefix}/runs/{run}/rank={rank}/") if k.endswith(".parquet")]
        return [row_to_record(row) for key in parts for row in pq.read_table(io.BytesIO(self.read(key))).to_pylist()]


@pytest.fixture
def bucket(endpoint):
    return Bucket(endpoint)


class Server:
    """The binary with an explicit environment: nothing inherited, so no real AWS variable can reach it."""

    def __init__(self, tmp_path: Path, bucket: Bucket, prefix: str, *extra: str, port: int = 0) -> None:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "AWS_ACCESS_KEY_ID": KEY_ID,
            "AWS_SECRET_ACCESS_KEY": SECRET,
            "AWS_REGION": REGION,
            "AWS_ENDPOINT_URL": bucket.endpoint,
            "AWS_ALLOW_HTTP": "true",
        }
        self.stderr_path = tmp_path / f"server-{uuid.uuid4().hex[:6]}.stderr"
        self.stderr = open(self.stderr_path, "wb")
        self.process = subprocess.Popen(
            [str(BINARY), "--listen", f"127.0.0.1:{port}", "--store", f"s3://{bucket.name}/{prefix}", *extra],
            stdout=subprocess.PIPE, stderr=self.stderr, text=True, env=env,
        )
        line = self.process.stdout.readline()  # "tensorquill-server listening on 127.0.0.1:PORT"
        if "listening" not in line:
            code = self.process.wait(20)
            self.close_files()
            raise RuntimeError(f"the server exited with {code}: {self.stderr_path.read_text()}")
        self.port = int(line.rsplit(":", 1)[1])
        self.grpc = f"grpc://127.0.0.1:{self.port}"
        self.stdout_rest = ""

    def close_files(self) -> None:
        self.stdout_rest = self.process.stdout.read() or ""
        self.process.stdout.close()
        self.stderr.close()

    def output(self) -> str:
        return self.stdout_rest + self.stderr_path.read_text()

    def terminate(self) -> int:
        self.process.send_signal(signal.SIGTERM)
        code = self.process.wait(30)
        self.close_files()
        return code

    def kill(self) -> None:
        self.process.kill()
        self.process.wait(20)
        self.close_files()


def write_steps(path: Path, run: str, rank: int, steps: range, *, spike_at: int | None = None, blob: int = 0) -> None:
    with open(path, "a", encoding="utf-8") as file:
        for step in steps:
            norm = 80.0 if step == spike_at else 0.5 + 0.01 * (step % 7)
            record = {"schema_version": 1, "run_id": run, "rank": rank, "step": step, "ts": 1000.0 + step, "grad_norm": norm}
            if blob:
                record["blob"] = os.urandom(blob // 2).hex()  # incompressible: the part grows past 64 KiB
            file.write(json.dumps(record) + "\n")


def wait_for(what: str, ready, timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = ready()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def test_runs_stream_to_s3_and_a_killed_server_resumes_without_losing_or_repeating_a_step(tmp_path, bucket):
    prefix = "team/prod"  # a prefix, not the bucket root
    run = "s3run"
    server = Server(tmp_path, bucket, prefix, *FAST)
    files = {}
    try:
        # Rank 1 and rank 10 share a name prefix; rank 10's part is over 64 KiB, rank 1's under it,
        # so the watermark read after a restart is a suffix read of a whole small part and of
        # the tail of a big one.
        for rank, blob in ((1, 0), (10, 8000), (2, 0)):
            path = tmp_path / f"{run}.rank{rank}.jsonl"
            write_steps(path, run, rank, range(130), spike_at=100, blob=blob)
            files[rank] = path
            assert FlightSender(path, server.grpc, run, rank).close(30) is True
        keys = bucket.keys(prefix + "/")
        for rank in (1, 2, 10):
            assert any(k.startswith(f"{prefix}/runs/{run}/rank={rank}/attempt=") and k.endswith(".parquet") for k in keys)
            assert f"{prefix}/runs/{run}/rank={rank}/incidents.json" in keys
        sizes = {
            k: bucket.client.head_object(Bucket=bucket.name, Key=k)["ContentLength"]
            for k in keys
            if k.endswith(".parquet") and ("/rank=10/" in k or "/rank=1/" in k)
        }
        assert all(size > 64 * 1024 for k, size in sizes.items() if "/rank=10/" in k)  # a footer read is a tail, not the part
        assert all(size < 64 * 1024 for k, size in sizes.items() if "/rank=1/" in k)  # the whole part is the suffix
        for rank in (1, 2, 10):
            stored = bucket.records(prefix, run, rank)
            assert [r["step"] for r in stored] == list(range(130))
            assert all(r["rank"] == rank for r in stored)
            incidents = bucket.json(f"{prefix}/runs/{run}/rank={rank}/incidents.json")
            assert incidents["last_step"] == 129 and incidents["rank"] == rank
            assert any(i["kind"] == "grad_spike" and i["step"] == 100 for i in incidents["incidents"])
        status_key = f"{prefix}/runs/{run}/watchdog.json"
        status = wait_for(  # written by the watchdog's next look
            "the status of all three ranks",
            lambda: (lambda s: s if s and {r["rank"] for r in s["ranks"]} == {1, 2, 10} else None)(bucket.maybe_json(status_key)),
        )
        assert status["server_running"] is True and status["restart"] and "abandoned" in status

        # SIGKILL with nothing left to flush; a new server knows nothing in memory. Rank 1's sender
        # lost its offset file (it resends the whole file) and writes 20 more steps meanwhile.
        server.kill()
        server = Server(tmp_path, bucket, prefix, *FAST)
        offset_path(files[1]).unlink()
        write_steps(files[1], run, 1, range(130, 150))
        assert FlightSender(files[1], server.grpc, run, 1).close(30) is True
        assert [r["step"] for r in bucket.records(prefix, run, 1)] == list(range(150))  # none lost, none twice
        assert bucket.json(f"{prefix}/runs/{run}/rank=1/incidents.json")["last_step"] == 149
        # Rank 10 resends everything: all of it is already stored, so nothing is written again.
        before = bucket.keys(f"{prefix}/runs/{run}/rank=10/")
        offset_path(files[10]).unlink()
        assert FlightSender(files[10], server.grpc, run, 10).close(30) is True
        assert bucket.keys(f"{prefix}/runs/{run}/rank=10/") == before
        assert [r["step"] for r in bucket.records(prefix, run, 10)] == list(range(130))
        # A prefix is not the bucket: nothing outside it was written.
        assert all(k.startswith(prefix + "/") for k in bucket.keys())
    finally:
        assert server.terminate() == 0
    # Credentials are handed to the store and never printed.
    assert SECRET not in server.output() and KEY_ID not in server.output()


CHILD = """
import sys, time
from tensorquill.flight import FlightSender
sender = FlightSender(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
print("started", flush=True)
time.sleep(300)
"""


def test_the_watchdog_state_in_s3_survives_a_killed_server(tmp_path, bucket):
    prefix = "stall"
    run = "stalled"
    key = f"{prefix}/runs/{run}/watchdog.json"
    watching = (*FAST, "--silent-seconds", "1", "--critical-seconds", "2")
    server = Server(tmp_path, bucket, prefix, *watching)
    path = tmp_path / f"{run}.rank0.jsonl"
    write_steps(path, run, 0, range(60))
    child = subprocess.Popen([sys.executable, "-c", CHILD, str(path), server.grpc, run, "0"], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "started"
        wait_for("the steps stored", lambda: len(bucket.records(prefix, run, 0)) == 60)
        child.send_signal(signal.SIGKILL)  # the training process dies; its stream is cut, not closed
        child.wait(20)
        # Quiet for longer than the floor: lost, the job stalled. The status in S3 says so.
        before = wait_for("a stalled status", lambda: (lambda s: s if s["summary"] == "stalled" else None)(bucket.json(key)))
        assert before["incidents"][0]["kind"] == "job_stalled" and before["ranks"][0]["last_step"] == 59
        quiet_before = before["ranks"][0]["seconds_since_progress"]
        server.kill()
        time.sleep(2.5)  # an outage longer than the floor

        server = Server(tmp_path, bucket, prefix, *watching)
        # Nobody has reconnected. The first status the new server writes already has the incident, with
        # the old quiet time: not reset, and not grown by the outage.
        after = wait_for(
            "the new server's first status", lambda: (lambda s: s if s["written_at_ms"] > before["written_at_ms"] else None)(bucket.json(key))
        )
        assert after["server_running"] is True and after["summary"] == "stalled"
        assert after["incidents"][0]["kind"] == "job_stalled" and after["ranks"][0]["last_step"] == 59
        assert quiet_before <= after["ranks"][0]["seconds_since_progress"] < quiet_before + 5
        # A sender comes back and the run is active again.
        write_steps(path, run, 0, range(60, 120))
        sender = FlightSender(path, server.grpc, run, 0)
        wait_for("the run active", lambda: bucket.json(key)["summary"] == "active")
        assert sender.close(30) is True
        assert [r["step"] for r in bucket.records(prefix, run, 0)] == list(range(120))
    finally:
        if child.poll() is None:
            child.kill()
        if server.process.poll() is None:
            server.terminate()
    assert SECRET not in server.output()


def test_a_wrong_bucket_stops_the_server_at_startup_without_printing_the_secret(tmp_path, bucket):
    missing = Bucket.__new__(Bucket)
    missing.endpoint, missing.name = bucket.endpoint, "tensorquill-e2e-no-such-bucket"
    with pytest.raises(RuntimeError) as error:
        Server(tmp_path, missing, "p")
    message = str(error.value)
    assert "cannot read the store" in message and "NoSuchBucket" in message
    assert SECRET not in message and KEY_ID not in message
