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


# --- 3. One run's evidence never shows another run's records -------------------------------------


TWO_RUNS = [{"run_id": "X", "rank": 0, "step": 0, "grad_norm": 1.0},
            {"run_id": "Y", "rank": 0, "step": 1, "grad_norm": float("nan")}]


@pytest.mark.parametrize("rows", [
    TWO_RUNS,
    # A row with no run ID beside a named run's: of no named run, so not of X's stream either.
    [TWO_RUNS[0], {"rank": 0, "step": 1, "grad_norm": float("nan")}],
], ids=["two runs", "a named run and a row with none"])
def test_records_of_two_runs_are_refused_before_any_rule_runs(tmp_path, rows):
    # On main: evidence for X, showing Y's non_finite gradient at step 1.
    with pytest.raises(ValueError, match="more than one run"):
        tensorquill.build_evidence(rows)
    # Two jobs appending to one custom-named file: the same refusal from the file.
    shared = _capture(tmp_path / "shared.jsonl", [{**r, "grad_norm": "nan"} if r["step"] else r for r in rows])
    with pytest.raises(ValueError, match="more than one run"):
        tensorquill.build_evidence(shared)


@RUST
def test_the_rust_detectors_refuse_the_same_rows():
    # The Rust side's own refusal (detect.rs `detect_rows`), unchanged here: both engines refuse.
    with pytest.raises(ValueError, match="more than one run"):
        tensorquill.build_evidence(TWO_RUNS, detectors="rust", server_binary=_BINARY)
    from tensorquill.server_detect import server_report

    with pytest.raises(RuntimeError, match="send one rank of one run"):
        server_report(TWO_RUNS, binary=_BINARY)


def test_a_later_attempt_of_another_run_does_not_roll_this_one_back():
    x = [{"run_id": "X", "rank": 0, "attempt": 1, "step": s, "grad_norm": 1.0} for s in range(5)]
    y = [{"run_id": "Y", "rank": 0, "attempt": 2, "step": s, "grad_norm": 1.0} for s in range(2)]
    assert len(ev.resolve_attempts(x + y)) == 7  # main: 2, X's steps 0 to 4 cut by Y's start at 0
    # Within one run a resume still replaces what it rolled back.
    again = [{**r, "run_id": "X"} for r in y]
    assert [(r["attempt"], r["step"]) for r in ev.resolve_attempts(x + again)] == [(2, 0), (2, 1)]


def test_rows_with_no_run_id_still_make_one_stream():
    # Files written before run_id existed: every row has none, and they are read as before.
    old = [{"rank": 0, "step": 0, "grad_norm": 1.0}, {"rank": 0, "step": 1, "grad_norm": float("nan")}]
    evidence = tensorquill.build_evidence(old)
    assert evidence["run_id"] is None and [(i["kind"], i["step"]) for i in evidence["incidents"]] == [("non_finite", 1)]


# --- 2. MCP answers a rank from that rank's records ---------------------------------------------

ROUTER = "blocks.0.router"


def _two_rank_capture(path: Path, steps: int = 3) -> Path:
    """Expert 1 gets every one of 100 assignments on rank 0 and none on rank 1."""
    rows = []
    for step in range(steps):
        rows.append({"run_id": "m", "rank": 0, "step": step, "moe": {ROUTER: [100, 0, 0, 0]}, "moe_top_k": {ROUTER: 1}})
        rows.append({"run_id": "m", "rank": 1, "step": step, "moe": {ROUTER: [0, 50, 25, 25]}, "moe_top_k": {ROUTER: 1}})
    return _capture(path, rows)


def test_every_evidence_file_says_which_rank_it_judged(tmp_path):
    capture = _two_rank_capture(tmp_path / "x.jsonl")
    assert tensorquill.build_evidence(capture, rank=1)["rank"] == 1
    assert tensorquill.build_evidence([{"step": 0, "grad_norm": 1.0}])["rank"] == 0
    assert tensorquill.build_evidence([], rank=3)["rank"] == 3
    # The compact file keeps it.
    written = compact.write_compact(tensorquill.build_evidence(capture, rank=1), tmp_path / "y.evidence.json")
    assert json.loads(written.read_text())["rank"] == 1


def test_rank_1_is_never_answered_with_rank_0s_saved_samples(tmp_path):
    capture = _two_rank_capture(tmp_path / "x.jsonl")
    rank0 = tensorquill.build_evidence(capture, rank=0)
    own = mcp_server.get_series(mcp_server.Source(capture), {"name": ROUTER, "rank": 1, "expert": 1})
    assert own["rank"] == 1 and own["values"]["expert_share"] == [0.0, 0.0, 0.0]

    full = ev.write_evidence(rank0, tmp_path / "x.evidence.json")
    beside = mcp_server.get_series(mcp_server.Source(full), {"name": ROUTER, "rank": 1, "expert": 1})
    # main: [1.0, 1.0, 1.0], labelled rank 1, from rank 0's samples in the file.
    assert beside["rank"] == 1 and beside["values"]["expert_share"] == [0.0, 0.0, 0.0]
    assert beside["values"] == own["values"] and "judged rank 0" in beside["evidence_file"]
    # Rank 0, the rank the file judged: its own samples, as before, with no note.
    zero = mcp_server.get_series(mcp_server.Source(full), {"name": ROUTER, "rank": 0, "expert": 1})
    assert zero["values"]["expert_share"] == [1.0, 1.0, 1.0] and "evidence_file" not in zero
    # Asked for no rank: the rank the file judged, not the lowest in the capture.
    rank1_file = ev.write_evidence(tensorquill.build_evidence(capture, rank=1), tmp_path / "x.evidence.json")
    unasked = mcp_server.get_series(mcp_server.Source(rank1_file), {"name": ROUTER, "expert": 1})
    assert unasked["rank"] == 1 and unasked["values"] == own["values"] and "evidence_file" not in unasked


def test_a_compact_files_flags_are_used_only_for_the_rank_it_judged(tmp_path):
    capture = _two_rank_capture(tmp_path / "x.jsonl", steps=6)
    rank0 = tensorquill.build_evidence(capture, rank=0)
    handed = compact.write_compact(rank0, capture.with_suffix(".evidence.json"), window=1)
    assert compact._omits(json.loads(handed.read_text())) is not None  # it left samples out
    own = mcp_server.get_series(mcp_server.Source(capture), {"name": ROUTER, "rank": 1})
    beside = mcp_server.get_series(mcp_server.Source(handed), {"name": ROUTER, "rank": 1})
    # main: rank 1's routing with rank 0's overloaded and dead counts at the steps the file kept.
    assert beside["values"] == own["values"] and "compact" not in beside
    assert "judged rank 0" in beside["evidence_file"]


def test_an_evidence_file_that_does_not_say_its_rank_is_not_taken_on_trust(tmp_path):
    capture = _two_rank_capture(tmp_path / "x.jsonl")
    older = tensorquill.build_evidence(capture, rank=0)
    del older["rank"]  # written before the key
    full = ev.write_evidence(older, tmp_path / "x.evidence.json")
    own = mcp_server.get_series(mcp_server.Source(capture), {"name": ROUTER, "rank": 1, "expert": 1})
    beside = mcp_server.get_series(mcp_server.Source(full), {"name": ROUTER, "rank": 1, "expert": 1})
    assert beside["values"] == own["values"] and "a rank it does not name" in beside["evidence_file"]
    # And another run's evidence for the same rank is not this run's either.
    other = {**tensorquill.build_evidence(capture, rank=1), "run_id": "another"}
    full = ev.write_evidence(other, tmp_path / "x.evidence.json")
    beside = mcp_server.get_series(mcp_server.Source(full), {"name": ROUTER, "rank": 1, "expert": 1})
    assert beside["values"] == own["values"] and "run 'another'" in beside["evidence_file"]
