"""Lane AP-2 of wave AP, through the real binary and its `SIGTERM`: AO-03, a report pending at the stop.

A store in which `runs/<run>/rank=0/incidents.json` is a directory, so the server's first write of
the rank's report fails and waits for its backoff; a three-step capture sent by the SDK's own
`FlightSender` and closed with True; the directory removed, so the store is healthy again; then
`SIGTERM`. The stop tries the pending report once, at once, before it writes the last status: the
rank's `incidents.json` covers step 2 and the run is judged. On `8c77d06` there was no
`incidents.json`, `summary` `finished` and the run state `unknown` (`ranks_unknown: [0]`).

`server/tests/wave_ap_ingest_test.rs` holds the same case on `Ingest::stop` in-process, with the
store still failing at the stop, a pending rebuild and the eviction hold; this file adds the signal
path of the real binary and the SDK's sender.

Needs pyarrow and the server binary (`TENSORQUILL_SERVER_BIN`, or one beside the interpreter); the
test is skipped without them.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from tensorquill.server_detect import find_binary

try:
    import pyarrow  # noqa: F401  (the sender needs it)
except ImportError:
    pyarrow = None

BINARY = find_binary()
needs_pyarrow = pytest.mark.skipif(pyarrow is None, reason="pyarrow is not installed")
needs_binary = pytest.mark.skipif(not BINARY or not os.path.isfile(BINARY), reason="tensorquill-server is not built")

ATTEMPT = 1_732_000_000_000
# The server's first wait before it writes a failed report again (`DEFAULT_REPLAY_BACKOFF`).
RETRY_WAIT_SECONDS = 10.0


@needs_pyarrow
@needs_binary
def test_ao03_a_report_pending_at_sigterm_is_written_before_the_last_status(tmp_path):
    from tensorquill import flight

    run_id = "ap2_sig"
    store = tmp_path / "store"
    blocked = store / "runs" / run_id / "rank=0" / "incidents.json"
    (blocked / "in_the_way").mkdir(parents=True)
    capture = tmp_path / "capture.jsonl"
    capture.write_text("".join(
        json.dumps({"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": ATTEMPT, "step": step,
                    "ts": float(step), "grad_norm": 1.0}) + "\n"
        for step in range(3)))
    env = {k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"}
    process = subprocess.Popen(
        [BINARY, "--listen", "127.0.0.1:0", "--store", str(store), "--flush-steps", "1",
         "--watchdog-seconds", "0.2"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        line = process.stdout.readline()
        assert "listening on" in line, line + process.stderr.read()
        endpoint = f"grpc://127.0.0.1:{int(line.rsplit(':', 1)[1])}"
        sender = flight.FlightSender(capture, endpoint, run_id, 0)
        assert sender.close(timeout=20.0) is True
        refused_at = time.monotonic()
        assert sender.acked_offset == capture.stat().st_size
        for child in blocked.iterdir():
            child.rmdir()
        blocked.rmdir()
        # The report's retry waits the server's backoff, 10 s by default: the stop comes well before
        # it, so only the stop can have written the report.
        assert time.monotonic() - refused_at < RETRY_WAIT_SECONDS / 2
        process.send_signal(signal.SIGTERM)
        assert process.wait(30) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(10)
        err = process.stderr.read()
        process.stdout.close()
        process.stderr.close()
    assert "cannot write incidents.json" in err, err
    report = json.loads(blocked.read_text())
    assert (report["attempt"], report["last_step"]) == (ATTEMPT, 2), report
    status = json.loads((store / "runs" / run_id / "watchdog.json").read_text())
    assert status["summary"] == "finished", status
    codes = [reason["code"] for reason in status["run_state"]["reasons"]]
    assert "ranks_unknown" not in codes and "no_steps" not in codes, status["run_state"]
    rank = next(entry for entry in status["ranks"] if entry["rank"] == 0)
    assert rank["last_step"] == 2, rank
