"""Lane U7 (Nov 19, 2026): the readers of `evidence.py`, `evidence_system.py`, `mcp_server.py`
and `trigger.py` where the audit of the code found no report, a wrong report, or a report an
agent cannot parse. Each test failed on `main` at 1767a4a and passes with the lane's fix.

Only the tests marked with `RUST` need the server binary (`TENSORQUILL_SERVER_BIN`); the
others run without it. Nothing here sends to a webhook or starts any command but a shell
`printf` into a file in the test's own temporary folder.
"""

from __future__ import annotations

import json
import os
import pickle
import threading
import time
from pathlib import Path

import pytest

import tensorquill
from tensorquill import compact, mcp_server
from tensorquill import evidence as ev
from tensorquill.trigger import Trigger

_BINARY = os.environ.get("TENSORQUILL_SERVER_BIN")
RUST = pytest.mark.skipif(
    not (_BINARY and Path(_BINARY).is_file()), reason="needs the server binary: set TENSORQUILL_SERVER_BIN"
)


def _strict_loads(text: str):
    """`json.loads` as a strict JSON reader takes it: no NaN, Infinity or -Infinity."""

    def refuse(constant: str):
        raise ValueError(f"not JSON: {constant}")

    return json.loads(text, parse_constant=refuse)


def _capture(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


# --- 1. A duplicate-batch incident beside a stuck collective ------------------------------------


def _flight_recorder_hang(directory: Path) -> Path:
    """The two-rank dump of `test_collectives.py`: rank 1 never entered all_reduce #2."""

    def entry(record_id, seq, state="completed"):
        return {"profiling_name": "nccl:all_reduce", "state": state, "process_group": ("0", "default_pg"),
                "input_sizes": [[4]], "output_sizes": [[4]], "input_dtypes": "float32", "output_dtypes": "float32",
                "collective_seq_id": str(seq), "p2p_seq_id": "0", "time_created_ns": 0, "frames": [],
                "record_id": record_id, "is_p2p": False}

    directory.mkdir()
    for rank, entries in ((0, [entry(0, 1), entry(1, 2, "started")]), (1, [entry(0, 1)])):
        (directory / f"nccl_trace_rank_{rank}").write_bytes(pickle.dumps({
            "entries": entries, "version": "2.8", "pg_config": {"0": {"name": "0", "desc": "default_pg", "ranks": "[0, 1]"}}}))
    return directory


def test_a_duplicate_batch_finding_beside_a_stuck_collective_leaves_both_in_the_report(tmp_path):
    pytest.importorskip("torch.distributed.flight_recorder.components.builder")  # PyTorch 2.10 or later
    dumps = _flight_recorder_hang(tmp_path / "fr")
    now = time.time()
    records = [{"run_id": "r", "step": s, "grad_norm": 1.0, "ts": now - 100 + s} for s in range(5)]
    duplicates = {"step": 3, "ranks": [0, 1], "duplicated_steps": 3, "ranks_compared": [0, 1]}
    silent = {"rank": 0, "attempt": 1, "last_step": 4, "seconds_since_progress": 900.0, "stream": "open",
              "state": "silent", "phase": "train", "heartbeat_age_seconds": 2.0, "severity": "critical"}
    status = {"schema_version": 1, "run_id": "r", "written_at_ms": 1_000, "silent_floor_seconds": 300.0,
              "summary": "stalled", "ranks": [silent], "missing_ranks": [], "duplicate_batches": duplicates}
    evidence = tensorquill.build_evidence(records, watchdog=status, flight_recorder=dumps)  # KeyError on main
    kinds = {incident["kind"]: incident for incident in evidence["incidents"]}
    assert {"duplicate_batches", "collective_hang", "job_stalled"} <= set(kinds)
    # Only the stall is corroborated by the stuck collective; the duplicate batches are as reported.
    assert kinds["job_stalled"]["assessment"] == "confirmed_hang"
    assert kinds["job_stalled"]["corroborated_by"] == "collective_hang"
    assert "assessment" not in kinds["duplicate_batches"] and "corroborated_by" not in kinds["duplicate_batches"]
    # With no stall at all, the duplicate batches and the collective are both reported.
    quiet = {**status, "summary": "active", "ranks": [{**silent, "state": "active", "severity": None}]}
    evidence = tensorquill.build_evidence(records, watchdog=quiet, flight_recorder=dumps)
    assert {"duplicate_batches", "collective_hang"} <= {i["kind"] for i in evidence["incidents"]}
    assert "Duplicate" in tensorquill.incident_summary(evidence) or "duplicate" in tensorquill.incident_summary(evidence)
