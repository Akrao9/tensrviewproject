"""What the read-api scripts share: the server under `TENSORQUILL_SERVER_BIN`, a DoPut stream that
waits for each acknowledgement as the SDK's sender does, the benchmark's synthetic records, and the
summaries. No product code; this tree's `sdk/src` is put first on `sys.path`."""

from __future__ import annotations

import json
import os
import platform
import selectors
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "sdk" / "examples"))
sys.path.insert(0, str(ROOT / "sdk" / "src"))


def binary() -> str:
    path = os.environ.get("TENSORQUILL_SERVER_BIN")
    if not path or not Path(path).is_file():
        raise SystemExit("set TENSORQUILL_SERVER_BIN to the tensorquill-server binary of the tree being measured")
    return path


def load() -> float:
    return round(os.getloadavg()[0], 2)


def spread(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    p99 = ordered[min(len(ordered) - 1, max(0, round(0.99 * (len(ordered) - 1))))]
    return {"median": statistics.median(values), "p99": p99, "min": ordered[0], "max": ordered[-1], "n": len(values)}


def machine() -> dict:
    return {"platform": platform.platform(), "python": platform.python_version(), "cpus": os.cpu_count(),
            "binary": binary(), "binary_bytes": Path(binary()).stat().st_size}


class Server:
    """`tensorquill-server` on `store`, Flight on a free port; `http=True` also asks for the read API
    (`--http-listen`, WAVE_AN_CONTRACT.md section 2.1), which a server before wave AN refuses."""

    def __init__(self, store: Path, *extra: str, http: bool = False, token: str | None = None) -> None:
        env = {k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"}
        if token:
            env["TENSORQUILL_TOKEN"] = token
        args = [binary(), "--listen", "127.0.0.1:0", "--store", str(store), *extra]
        if http:
            args += ["--http-listen", "127.0.0.1:0"]
        self.args = args
        self.process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        self.flight = self.http = None
        for line in self._lines(2 if http else 1):
            if line.startswith("tensorquill-server listening on "):
                self.flight = "grpc://127.0.0.1:" + line.rsplit(":", 1)[1].strip()
            if line.startswith("tensorquill-server http on "):
                self.http = "http://127.0.0.1:" + line.rsplit(":", 1)[1].strip()
        if self.flight is None or (http and self.http is None):
            err = self.close()
            raise RuntimeError(f"the server did not start as asked: exit {self.process.returncode}: {err[-400:]}")

    def _lines(self, wanted: int, deadline: float = 120.0) -> list[str]:
        lines: list[str] = []
        with selectors.DefaultSelector() as chosen:
            chosen.register(self.process.stdout, selectors.EVENT_READ)
            end = time.monotonic() + deadline
            while len(lines) < wanted and time.monotonic() < end:
                if not chosen.select(timeout=max(0.0, end - time.monotonic())):
                    break
                line = self.process.stdout.readline()
                if not line:
                    break
                lines.append(line.strip())
        return lines

    def close(self) -> str:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(120)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(10)
        err = self.process.stderr.read() if self.process.stderr else ""
        for stream in (self.process.stdout, self.process.stderr):
            if stream:
                stream.close()
        return err


def put(endpoint: str, run_id: str, rank: int, batches: list, start: int = 0) -> list[dict]:
    """One DoPut stream; per batch, the seconds from writing it to its acknowledgement."""
    import pyarrow as pa
    from pyarrow import flight

    client = flight.FlightClient(endpoint)
    descriptor = flight.FlightDescriptor.for_path("runs", run_id, f"rank={rank}")
    writer, reader = client.do_put(descriptor, batches[0].schema)
    out, end = [], start
    try:
        for batch in batches:
            steps = batch.column("step").to_pylist()
            metadata = {"schema_version": 2, "run_id": run_id, "rank": rank, "min_step": min(steps),
                        "max_step": max(steps), "start_byte_offset": end, "end_byte_offset": end + 1000}
            end += 1000
            began = time.perf_counter()
            writer.write_with_metadata(batch, pa.py_buffer(json.dumps(metadata).encode()))
            ack = reader.read()
            out.append({"seconds": time.perf_counter() - began, "acked": time.time(),
                        "ack": json.loads(ack.to_pybytes()) if ack is not None else None})
        writer.done_writing()
    except Exception as error:  # a measurement records the failure and goes on
        out.append({"error": f"{type(error).__name__}: {error}"})
    finally:
        try:
            writer.close()
        except Exception:
            pass
        client.close()
    return out


def report_reaches(rank_folder: Path, last_step: int, deadline: float = 600.0) -> float | None:
    """The wall-clock time `incidents.json` first covers `last_step` (its `last_step` at or above it: a
    later batch may be judged before the poll sees this one), polled every 10 ms; None if never."""
    path = rank_folder / "incidents.json"
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        try:
            reached = json.loads(path.read_bytes()).get("last_step")
            if isinstance(reached, int) and reached >= last_step:
                return time.time()
        except (OSError, ValueError):
            pass
        time.sleep(0.01)
    return None


def template(shape: dict) -> dict:
    """The benchmark's record template (`sdk/examples/benchmark_detectors.py`), from a real capture of
    its toy model, cached beside this script's output when a store holds one."""
    import benchmark_detectors as bd

    return bd.capture_template(**shape)


def batches_for(template_: dict, rank: int, first: int, count: int, every: int = 50) -> list:
    import benchmark_detectors as bd
    from tensorquill import flight

    out = []
    for low in range(first, first + count * every, every):
        records = [bd.synth_record(template_, rank % bd.VARIANTS, rank, step) for step in range(low, low + every)]
        out.append(flight.records_to_batch(records))
    return out


def clone_tree(source: Path, dest: Path) -> None:
    """A copy of a store: a clone on APFS (`cp -c`), a reflink where the filesystem has one, else bytes."""
    if dest.exists():
        raise SystemExit(f"{dest} exists; nothing was deleted: remove it yourself or give another --work")
    if sys.platform == "darwin":
        subprocess.run(["cp", "-cR", str(source), str(dest)], check=True)
    else:
        subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(dest)], check=True)


def bytes_under(folder: Path, pattern: str) -> int:
    return sum(p.stat().st_size for p in folder.rglob(pattern) if p.is_file())
