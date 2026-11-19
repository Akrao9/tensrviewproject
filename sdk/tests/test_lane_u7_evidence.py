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


# --- 4. Evidence a strict JSON reader takes ------------------------------------------------------


def _nonfinite_records() -> list[dict]:
    """A replica fingerprint the capture wrote as "nan" and as "inf", and a NaN loss."""
    rows = [{"run_id": "r", "rank": 0, "step": s, "grad_norm": 1.0, "scalars": {"loss": 2.0}} for s in range(4)]
    rows[1]["replica_spread"] = {"w": ["nan", 0.0, 0.0], "v": ["inf", "-inf", 0.0]}
    rows[2]["scalars"] = {"loss": float("nan")}
    rows[3]["grad_norm"] = float("inf")
    return rows


def test_a_nonfinite_reading_is_written_as_the_capture_writes_it_and_read_back_whole(tmp_path):
    evidence = tensorquill.build_evidence(_nonfinite_records())
    [drift] = [i for i in evidence["incidents"] if i["kind"] == "replica_drift"]
    assert drift["parameters"]["w"]["non_finite"] is True  # the finding itself is kept
    path = ev.write_evidence(evidence, tmp_path / "r.rank0.evidence.json")
    text = path.read_text()
    written = _strict_loads(text)  # main: "not JSON: NaN"
    assert "NaN" not in text and "Infinity" not in text
    parameters = [i for i in written["incidents"] if i["kind"] == "replica_drift"][0]["parameters"]
    assert parameters["w"] == {"norm_spread": "nan", "sum_spread": 0.0, "projection_spread": 0.0, "non_finite": True}
    assert parameters["v"]["norm_spread"] == "inf" and parameters["v"]["sum_spread"] == "-inf"
    # In memory the evidence keeps its floats: only what is written changes.
    assert isinstance(drift["parameters"]["w"]["norm_spread"], float)
    # And the file read back serves every reader: the summary, MCP and the trigger.
    assert "Replicas diverged at step 1" in tensorquill.incident_summary(written)
    listed = mcp_server.list_incidents(mcp_server.Source(path), {})
    assert {"replica_drift", "non_finite"} <= {entry["kind"] for entry in listed["incidents"]}
    for entry in listed["incidents"]:
        detail = mcp_server.get_incident(mcp_server.Source(path), {"id": entry["id"]})
        mcp_server._line({"result": detail})  # the reply itself is strict JSON
    sink = tmp_path / "payload.json"
    trigger = Trigger(f"cat > '{sink}'", state_path=tmp_path / "fired.json", log=lambda _: None)
    trigger.look(written, path, rank=0)
    trigger.close(wait=10)
    assert _strict_loads(sink.read_text())["kind"] in ("replica_drift", "non_finite")
    _strict_loads((tmp_path / "fired.json").read_text())


def test_the_compact_file_and_the_triggers_payload_are_strict_too(tmp_path):
    evidence = tensorquill.build_evidence(_nonfinite_records())
    handed = compact.write_compact(evidence, tmp_path / "r.evidence.json")
    assert _strict_loads(handed.read_text())["incidents"]
    sink = tmp_path / "payload.json"
    trigger = Trigger(f"cat > '{sink}'", log=lambda _: None)
    trigger.look(evidence, ev.write_evidence(evidence, tmp_path / "r.evidence.json"), rank=0)  # the dict in memory
    trigger.close(wait=10)
    body = _strict_loads(sink.read_text())
    assert body["kind"] == "non_finite"  # the first that fires
    assert body["incident"]["step"] == 3
    drift = [i for i in evidence["incidents"] if i["kind"] == "replica_drift"][0]
    payload = trigger._payload({"id": "x", "ranks": [0], "command": {"attempts": 1}},
                               {"ranks": [0], "kind": "replica_drift", "class": "invariant", "step": 1,
                                "assessment": None, "summary": "", "incident": drift}, "command")
    assert payload["incident"]["parameters"]["v"]["sum_spread"] == "-inf"  # main: None


# --- 5. A second run at the same path fires its own incidents ------------------------------------


def _sink_command(sink: Path, fail_run: str | None = None) -> str:
    """A command that only appends the run and the firing's id to a file in the test's folder,
    and exits 1 for `fail_run`."""
    command = f'printf "%s %s\\n" "$TENSORQUILL_RUN_ID" "$TENSORQUILL_INCIDENT_ID" >> \'{sink}\''
    return command + (f' && test "$TENSORQUILL_RUN_ID" != "{fail_run}"' if fail_run else "")


def _fired(sink: Path) -> list[str]:
    return sink.read_text().splitlines() if sink.exists() else []


def _look(run: str, out: Path, sink: Path, **kwargs) -> Trigger:
    evidence = tensorquill.build_evidence([{"run_id": run, "rank": 0, "step": 0, "grad_norm": float("nan")}])
    trigger = Trigger(_sink_command(sink), log=lambda _: None, **kwargs)
    trigger.look(evidence, ev.write_evidence(evidence, out), rank=0)
    trigger.close(wait=10)
    return trigger


def test_a_fresh_run_at_the_same_path_fires_and_a_restarted_watcher_does_not(tmp_path):
    out, sink = tmp_path / "run.evidence.json", tmp_path / "sink.txt"
    _look("A", out, sink)
    _look("A", out, sink)  # the watcher of run A restarted: nothing fires again
    assert _fired(sink) == ["A A/rank0/non_finite/0"]
    _look("B", out, sink)  # another run whose evidence goes to the same path (main: held back)
    assert _fired(sink) == ["A A/rank0/non_finite/0", "B B/rank0/non_finite/0"]
    saved = json.loads((tmp_path / "run.evidence.fired.json").read_text())
    assert [(d["run_id"], d["id"]) for d in saved["deliveries"]] == [("A", "A/rank0/non_finite/0"), ("B", "B/rank0/non_finite/0")]
    _look("B", out, sink)  # and B's restarted watcher fires nothing either
    assert len(_fired(sink)) == 2


def test_a_state_file_in_the_old_format_is_still_read_by_run(tmp_path):
    out, sink = tmp_path / "run.evidence.json", tmp_path / "sink.txt"
    state = tmp_path / "run.evidence.fired.json"
    # As main wrote it: no run_id in a firing, one episode, the run only in each id.
    state.write_text(json.dumps({"version": 1, "evidence": str(out), "episode": None, "deliveries": [
        {"id": "A/rank0/non_finite/0", "kind": "non_finite", "step": 0, "ranks": [0], "first_seen_ms": 1,
         "command": {"status": "done", "attempts": 1, "exit_status": 0}},
        {"id": "with/a/slash/rank0/non_finite/0", "kind": "non_finite", "step": 0, "ranks": [0], "first_seen_ms": 1,
         "command": {"status": "done", "attempts": 1, "exit_status": 0}},
    ]}))
    _look("A", out, sink)  # a true restart of run A: its firing is in the file
    _look("with/a/slash", out, sink)
    assert _fired(sink) == []
    _look("B", out, sink)  # a fresh run: fires
    assert _fired(sink) == ["B B/rank0/non_finite/0"]


def _stall(run: str, step: int) -> dict:
    stopped = {"rank": 0, "state": "silent", "last_step": step, "attempt": 1, "severity": "critical"}
    return {"run_id": run, "incidents": [{"kind": "job_stalled", "step": step, "severity": "critical",
                                          "assessment": "suspected_hang", "ranks": [stopped]}]}


def test_stall_episodes_are_kept_apart_by_run(tmp_path):
    out, sink = tmp_path / "run.evidence.json", tmp_path / "sink.txt"
    for run in ("A", "A", "B"):  # A fires, A again is its open episode, B is a run of its own
        trigger = Trigger(_sink_command(sink), log=lambda _: None)
        trigger.look(_stall(run, 5), out, rank=None)
        trigger.close(wait=10)
    assert _fired(sink) == ["A A/run/job_stalled/5", "B B/run/job_stalled/5"]  # main: B's held back by A's episode
    saved = json.loads((tmp_path / "run.evidence.fired.json").read_text())
    assert sorted(saved["episodes"]) == ["A", "B"] and saved["episode"]["delivery"] == "B/run/job_stalled/5"
    # A file with one episode, as main wrote it: the episode is its firing's run's.
    saved["episode"] = saved.pop("episodes")["A"]
    for delivery in saved["deliveries"]:
        delivery.pop("run_id")
    (tmp_path / "run.evidence.fired.json").write_text(json.dumps(saved))
    for run in ("A", "C"):
        trigger = Trigger(_sink_command(sink), log=lambda _: None)
        trigger.look(_stall(run, 5), out, rank=None)
        trigger.close(wait=10)
    assert _fired(sink)[2:] == ["C C/run/job_stalled/5"]


def test_an_earlier_runs_failed_firing_is_never_sent_with_this_runs_incident(tmp_path):
    out, sink = tmp_path / "run.evidence.json", tmp_path / "sink.txt"
    trigger = Trigger(_sink_command(sink, fail_run="A"), retry_seconds=0.0, log=lambda _: None)
    for run in ("A", "B"):
        evidence = tensorquill.build_evidence([{"run_id": run, "rank": 0, "step": 0, "grad_norm": float("nan")}])
        trigger.look(evidence, ev.write_evidence(evidence, out), rank=0)
        trigger.close(wait=10)
        trigger.tick()
        trigger.close(wait=10)
    # main: B's incident was held back as fired, and A's failed firing was tried again with B's
    # incident under A's id ("B A/rank0/non_finite/0").
    assert [line for line in _fired(sink) if line.startswith("B")] == ["B B/rank0/non_finite/0"]


# --- 6. A torn line in the middle of a capture ---------------------------------------------------


def _torn_capture(path: Path) -> Path:
    """Step 0, the fragment a killed writer left (on a line of its own once the restarted writer
    appended after it), then steps 1 and 2, step 2 with a nonfinite gradient."""
    whole = [json.dumps({"run_id": "r", "rank": 0, "step": s, "grad_norm": "nan" if s == 2 else 1.0}) for s in range(3)]
    path.write_text(f'{whole[0]}\n{{"run_id": "r", "rank": 0, "st\n{whole[1]}\n{whole[2]}\n', encoding="utf-8")
    return path


def test_a_torn_line_in_the_middle_is_skipped_counted_and_said(tmp_path, monkeypatch, capsys):
    capture = _torn_capture(tmp_path / "r.rank0.jsonl")
    records, left_out = ev.read_jsonl(capture)
    assert [r["step"] for r in records] == [0, 1, 2] and left_out == 1
    evidence = tensorquill.build_evidence(capture)  # main: JSONDecodeError, no evidence
    assert [(i["kind"], i["step"]) for i in evidence["incidents"]] == [("non_finite", 2)]
    assert evidence["lines_left_out"] == 1
    assert "Lines left out: 1 line of the capture is not a whole JSON record" in tensorquill.incident_summary(evidence)

    # The command line's one read.
    out = tmp_path / "one.evidence.json"
    monkeypatch.setattr("sys.argv", ["tensorquill", str(capture), "-o", str(out), "--detectors", "python"])
    ev.main()
    assert json.loads(out.read_text())["lines_left_out"] == 1 and "Lines left out" in capsys.readouterr().out

    # The watcher's tail, which skipped it without a word.
    watched = tmp_path / "watched.evidence.json"
    ev.watch(capture, watched, polls=1, interval=0.0, server_reports=False, detectors="python")
    written = json.loads(watched.read_text())
    assert written["lines_left_out"] == 1 and [(i["kind"], i["step"]) for i in written["incidents"]] == [("non_finite", 2)]

    # MCP already read whole lines and said how many it left out.
    listed = mcp_server.list_incidents(mcp_server.Source(capture), {})
    assert listed["source"]["lines_left_out"] == 1 and [i["kind"] for i in listed["incidents"]] == ["non_finite"]


def test_a_last_line_being_written_is_counted_and_a_whole_one_without_newline_is_read(tmp_path):
    whole = json.dumps({"run_id": "r", "step": 0, "grad_norm": 1.0})
    finished = tmp_path / "finished.jsonl"
    finished.write_text(whole, encoding="utf-8")  # no newline, as a hand-made file may end
    assert ev.read_jsonl(finished) == ([json.loads(whole)], 0)
    live = tmp_path / "live.jsonl"
    live.write_text(whole + '\n{"run_id": "r", "st', encoding="utf-8")
    assert ev.read_jsonl(live) == ([json.loads(whole)], 1)
    assert tensorquill.build_evidence(live)["lines_left_out"] == 1


# --- 7. Three small ones -------------------------------------------------------------------------


def _held_at_the_rename(monkeypatch, writers: int = 2) -> None:
    """Every writer waits for the others just before its rename, as two watchers can."""
    barrier, real = threading.Barrier(writers), os.replace

    def replace(source, destination):
        barrier.wait(timeout=10)
        return real(source, destination)

    monkeypatch.setattr(os, "replace", replace)


def _together(*calls) -> list[BaseException]:
    failed: list[BaseException] = []

    def run(call):
        try:
            call()
        except BaseException as error:  # noqa: BLE001 - the test counts every failure
            failed.append(error)

    threads = [threading.Thread(target=run, args=(call,)) for call in calls]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return failed


def test_two_writers_of_one_evidence_file_do_not_take_each_others_temporary_file(tmp_path, monkeypatch):
    out = tmp_path / "run.evidence.json"
    _held_at_the_rename(monkeypatch)
    failed = _together(lambda: ev.write_evidence({"incidents": [], "n": 1}, out),
                       lambda: ev.write_evidence({"incidents": [], "n": 2}, out))
    assert failed == []  # main: FileNotFoundError from the second rename of `<path>.tmp`
    assert json.loads(out.read_text())["n"] in (1, 2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.evidence.json"]  # no temporary file left
    assert (out.stat().st_mode & 0o777) == (0o666 & ~_umask())  # the usual mode, not a private one


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def test_two_triggers_saving_one_state_file_do_not_fail(tmp_path, monkeypatch):
    out = tmp_path / "run.evidence.json"
    triggers = [Trigger("true", state_path=tmp_path / "fired.json", log=lambda _: None) for _ in range(2)]
    for trigger in triggers:
        trigger._evidence_path = out
    logs: list[str] = []
    for trigger in triggers:
        trigger._log = logs.append
        trigger._deliveries = [{"id": f"r/rank0/x/{id(trigger)}", "kind": "x", "step": 0, "ranks": [0]}]
    _held_at_the_rename(monkeypatch)
    failed = _together(*(trigger._save for trigger in triggers))
    assert failed == [] and not [line for line in logs if "could not be written" in line]  # main: logged a failed save
    assert json.loads((tmp_path / "fired.json").read_text())["version"] == 1


def test_the_weight_trend_median_of_two_parameters_is_their_mean():
    rows = [{"run_id": "r", "step": 0, "weight_norms": {"a.weight": 1.0, "b.weight": 1.0, "c.bias": 1.0}},
            {"run_id": "r", "step": 1, "weight_norms": {"a.weight": 1.0, "b.weight": 5.0, "c.bias": 2.0}}]
    trends = ev.weight_norm_trends(rows)
    assert trends["kinds"]["matrix"]["median_ratio"] == 3.0  # main: 5.0, the upper middle value
    assert trends["kinds"]["matrix"]["highest"]["ratio"] == 5.0


def _growing_at_one_step() -> list[dict]:
    """Six allocations at one repeated step, 32 MiB apart: growth, with no steps between."""
    return [{"run_id": "r", "rank": 0, "step": 5,
             "system": {"cuda": {"allocated": 2**30 + i * 32 * 2**20, "total": 2**34}}} for i in range(6)]


def test_growing_memory_at_one_repeated_step_reports_the_repeat_and_no_rate():
    evidence = tensorquill.build_evidence(_growing_at_one_step())  # main: ZeroDivisionError
    kinds = {i["kind"]: i for i in evidence["incidents"]}
    assert {"repeated_steps", "memory_growth"} <= set(kinds)
    assert kinds["memory_growth"]["bytes_per_step"] is None and kinds["memory_growth"]["steps_to_total"] is None
    assert "every reading at step 5, so no rate per step" in tensorquill.incident_summary(evidence)
    # Across steps the rate is as before.
    across = [{**row, "step": i} for i, row in enumerate(_growing_at_one_step())]
    growth = [i for i in tensorquill.build_evidence(across)["incidents"] if i["kind"] == "memory_growth"][0]
    assert growth["bytes_per_step"] == 32 * 2**20 and growth["steps_to_total"] == (2**34 - 2**30 - 5 * 32 * 2**20) / (32 * 2**20)


@RUST
def test_the_rust_detectors_on_the_same_rows_and_the_summary_of_their_answer():
    evidence = tensorquill.build_evidence(_growing_at_one_step(), detectors="rust", server_binary=_BINARY)
    kinds = {i["kind"]: i for i in evidence["incidents"]}
    assert {"repeated_steps", "memory_growth"} <= set(kinds)
    assert kinds["memory_growth"]["bytes_per_step"] is None  # Rust's infinite rate, written null
    text = tensorquill.incident_summary(evidence)
    assert "no rate per step" in text and "full in about" not in text
