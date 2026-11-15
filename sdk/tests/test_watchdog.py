import json
import time

import pytest

import tensorquill
from tensorquill.evidence import detect_watchdog_incidents, read_watchdog


def _status(summary, ranks, missing=(), world_size=None):
    return {
        "schema_version": 1, "run_id": "r", "written_at_ms": 1_000, "silent_floor_seconds": 300.0,
        "silent_factor": 5.0, "world_size": world_size, "missing_ranks": list(missing),
        "summary": summary, "ranks": ranks,
    }


def _rank(rank, state, last_step, quiet=0.0, stream="open", phase="train", beat=None, severity=None):
    return {"rank": rank, "attempt": 1, "last_step": last_step, "seconds_since_progress": quiet,
            "stream": stream, "state": state, "phase": phase, "heartbeat_age_seconds": beat, "severity": severity}


def test_a_stalled_job_and_a_silent_rank_become_incidents():
    stalled = _status("stalled", [_rank(0, "silent", 149, 640.0), _rank(1, "lost", 149, 700.0, "dropped")])
    [hit] = detect_watchdog_incidents(stalled)
    assert hit["kind"] == "job_stalled" and hit["step"] == 149 and len(hit["ranks"]) == 2
    summary = tensorquill.incident_summary({"incidents": [hit], "samples": [], "layer_count": 0,
                                             "expert_count": 0, "router_names": []})
    assert "Job stalled after step 149" in summary and "its connection dropped and did not return" in summary
    assert "Flight Recorder" in summary
    assert hit["severity"] == "warning" and hit["assessment"] == "suspected_hang"

    degraded = _status("degraded", [_rank(0, "active", 400), _rank(1, "silent", 349, 900.0)], missing=[2], world_size=3)
    [hit] = detect_watchdog_incidents(degraded)
    assert hit["kind"] == "rank_silent" and hit["step"] == 349 and hit["newest_step"] == 400
    assert hit["missing_ranks"] == [2]
    text = tensorquill.incident_summary({"incidents": [hit], "samples": [], "layer_count": 0,
                                          "expert_count": 0, "router_names": []})
    assert "rank 1 sent no new step since step 349" in text and "ranks 2 never connected" in text
    assert "no heartbeat, so the phase is unknown" in text

    # Only missing ranks, everyone else fine: still reported, at the newest step.
    [hit] = detect_watchdog_incidents(_status("degraded", [_rank(0, "active", 50)], missing=[1], world_size=2))
    assert hit["kind"] == "rank_silent" and hit["step"] == 50 and hit["ranks"] == []


def test_healthy_or_finished_runs_and_unreadable_status_give_nothing(tmp_path):
    for summary in ("active", "finished"):
        assert detect_watchdog_incidents(_status(summary, [_rank(0, "active", 9)])) == []
    assert detect_watchdog_incidents(None) == []
    path = tmp_path / "watchdog.json"
    assert read_watchdog(path) is None  # missing
    path.write_text('{"schema_version": 1, "ranks": [')  # half-written, as seen mid-replace
    assert read_watchdog(path) is None
    path.write_text(json.dumps({"schema_version": 2, "ranks": []}))  # not a version this reader knows
    assert read_watchdog(path) is None
    path.write_text(json.dumps(_status("active", [])))
    assert read_watchdog(path)["summary"] == "active"


def test_build_evidence_reads_the_status_beside_the_parts(tmp_path):
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    records = [{"run_id": "r", "rank": 0, "step": s, "grad_norm": 1.0} for s in range(5)]
    stalled = _status("stalled", [_rank(0, "silent", 4, 800.0)])
    (run / "watchdog.json").write_text(json.dumps(stalled))
    # Records come from a list here; the status is named. A directory finds its own.
    evidence = tensorquill.build_evidence(records, watchdog=run / "watchdog.json")
    assert [i["kind"] for i in evidence["incidents"]] == ["job_stalled"]
    assert evidence["watchdog"]["summary"] == "stalled"
    assert "watchdog" not in tensorquill.build_evidence(records)


def test_watch_rebuilds_when_only_the_status_changes(tmp_path):
    # A hung job writes no records, so the status file is the only thing that moves.
    path = tmp_path / "run.rank0.jsonl"
    path.write_text("".join(json.dumps({"run_id": "r", "rank": 0, "step": s, "grad_norm": 1.0}) + "\n" for s in range(3)))
    status = tmp_path / "watchdog.json"
    status.write_text(json.dumps(_status("active", [_rank(0, "active", 2)])))
    seen = []

    def on_update(evidence, written):
        seen.append([i["kind"] for i in evidence["incidents"]])
        if len(seen) == 1:  # the server calls the job stalled between two looks
            time.sleep(0.01)
            status.write_text(json.dumps(_status("stalled", [_rank(0, "silent", 2, 900.0)])))

    tensorquill.evidence.watch(path, tmp_path / "out.json", interval=0.01, polls=4, on_update=on_update, watchdog=status)
    assert seen == [[], ["job_stalled"]]


def test_the_command_reads_the_status_beside_a_run_directory(tmp_path, capsys, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    from tensorquill.evidence import main
    from tensorquill.flight import records_to_batch

    run = tmp_path / "runs" / "r"
    part = run / "rank=0" / "attempt=1"
    part.mkdir(parents=True)
    records = [{"run_id": "r", "rank": 0, "attempt": 1, "step": s, "schema_version": 1, "grad_norm": 1.0} for s in range(5)]
    pq.write_table(__import__("pyarrow").Table.from_batches([records_to_batch(records)]), part / "part-00000000-00000004.parquet")
    (run / "watchdog.json").write_text(json.dumps(_status("stalled", [_rank(0, "silent", 4, 900.0)])))
    monkeypatch.setattr("sys.argv", ["tensorquill", str(run), "-o", str(tmp_path / "out.json")])
    main()
    assert "Job stalled after step 4" in capsys.readouterr().out


def test_the_rerun_export_handles_every_incident_kind_from_outside_tools(tmp_path):
    rr = pytest.importorskip("rerun")
    del rr
    records = [{"run_id": "r", "rank": 0, "step": s, "grad_norm": 1.0, "lr": [1e-3]} for s in range(30)]
    records += [{"run_id": "r", "rank": 0, "attempt": 2, "step": s, "grad_norm": 1.0, "lr": [5e-5]} for s in range(20, 30)]
    for r in records[:30]:
        r["attempt"] = 1
    traincheck = tmp_path / "violations_summary.json"
    traincheck.write_text(json.dumps({"first_violation_step": 4, "distinct_invariants_violated": 1,
                                      "violations": [{"display_name": "x", "relation_type": "R", "first_step": 4,
                                                      "last_step": 9, "occurrences": 2}]}))
    kwargs = {"traincheck": traincheck, "watchdog": _status("degraded", [_rank(0, "active", 29), _rank(1, "silent", 20, 600.0)], missing=[2], world_size=3)}
    pytest.importorskip("torch.distributed.flight_recorder.components.builder")
    import pickle

    def entry(i, seq, state="completed"):
        return {"profiling_name": "nccl:all_reduce", "state": state, "process_group": ("0", "default_pg"),
                "input_sizes": [[4]], "output_sizes": [[4]], "input_dtypes": "float32", "output_dtypes": "float32",
                "collective_seq_id": str(seq), "p2p_seq_id": "0", "time_created_ns": 0, "frames": [], "record_id": i, "is_p2p": False}

    dumps = tmp_path / "fr"
    dumps.mkdir()
    for rank, entries in ((0, [entry(0, 1), entry(1, 2, "started")]), (1, [entry(0, 1)])):
        (dumps / f"nccl_trace_rank_{rank}").write_bytes(pickle.dumps({
            "entries": entries, "version": "2.8",
            "pg_config": {"0": {"name": "0", "desc": "default_pg", "ranks": "[0, 1]"}}}))
    evidence = tensorquill.build_evidence(records, flight_recorder=dumps, **kwargs)
    kinds = {i["kind"] for i in evidence["incidents"]}
    assert {"rank_silent", "collective_hang", "traincheck_violation", "resume_lr_change"} <= kinds
    out = tmp_path / "x.rrd"
    tensorquill.export_rerun(evidence, out, records=tensorquill.load_records(records))
    assert out.stat().st_size > 0
    assert tensorquill.incident_summary(evidence).count("\n") >= 3


def _text(incidents, extra=None):
    return tensorquill.incident_summary({"incidents": incidents, "samples": [], "layer_count": 0,
                                          "expert_count": 0, "router_names": [], **(extra or {})})


def test_severity_and_what_the_heartbeats_say_are_in_the_incident():
    stalled = _status("stalled", [_rank(0, "silent", 149, 1000.0, phase="train", beat=4.0, severity="critical")])
    stalled["critical_seconds"] = 900.0
    [hit] = detect_watchdog_incidents(stalled)
    assert hit["severity"] == "critical" and hit["assessment"] == "suspected_hang"
    text = _text([hit])
    assert "its heartbeats continue and it says it is in phase `train`" in text
    assert "[critical, suspected hang]" in text

    frozen = _status("stalled", [_rank(0, "unresponsive", 149, 400.0, phase="train", beat=120.0, severity="warning")])
    [hit] = detect_watchdog_incidents(frozen)
    assert "its heartbeats stopped too, so the whole process is frozen or gone" in _text([hit])

    # A rank that is evaluating is paused, not stopped: no incident while its heartbeats go on.
    paused = _status("active", [_rank(0, "paused", 149, 600.0, phase="eval", beat=3.0)])
    assert detect_watchdog_incidents(paused) == []
    # Only lost ranks are a lost connection, not a suspected hang.
    lost = _status("stalled", [_rank(0, "lost", 149, 400.0, "dropped", severity="warning")])
    assert detect_watchdog_incidents(lost)[0]["assessment"] == "lost_connection"


def test_a_stuck_collective_in_pytorch_s_dump_confirms_the_hang_the_stream_suggested(tmp_path):
    pytest.importorskip("torch.distributed.flight_recorder.components.builder")
    import pickle

    def entry(i, seq, state="completed"):
        return {"profiling_name": "nccl:all_reduce", "state": state, "process_group": ("0", "default_pg"),
                "input_sizes": [[4]], "output_sizes": [[4]], "input_dtypes": "float32", "output_dtypes": "float32",
                "collective_seq_id": str(seq), "p2p_seq_id": "0", "time_created_ns": 0, "frames": [], "record_id": i, "is_p2p": False}

    dumps = tmp_path / "fr"
    dumps.mkdir()
    for rank, entries in ((0, [entry(0, 1), entry(1, 2, "started")]), (1, [entry(0, 1)])):
        (dumps / f"nccl_trace_rank_{rank}").write_bytes(pickle.dumps({
            "entries": entries, "version": "2.8",
            "pg_config": {"0": {"name": "0", "desc": "default_pg", "ranks": "[0, 1]"}}}))
    import os
    import time

    now = time.time()
    records = [{"run_id": "r", "step": s, "grad_norm": 1.0, "ts": now - 100 + s} for s in range(5)]
    stalled = _status("stalled", [_rank(0, "silent", 4, 900.0, beat=2.0, severity="critical")])
    alone = tensorquill.build_evidence(records, watchdog=stalled)
    assert alone["incidents"][0]["assessment"] == "suspected_hang"
    both = tensorquill.build_evidence(records, watchdog=stalled, flight_recorder=dumps)
    [stream] = [i for i in both["incidents"] if i["kind"] == "job_stalled"]
    assert stream["assessment"] == "confirmed_hang" and stream["corroborated_by"] == "collective_hang"
    assert "corroborated by collective_hang" in tensorquill.incident_summary(both)

    # Dumps from before the last recorded step prove nothing about now.
    for path in dumps.iterdir():
        os.utime(path, (now - 1000, now - 1000))
    old = tensorquill.build_evidence(records, watchdog=stalled, flight_recorder=dumps)
    assert [i["assessment"] for i in old["incidents"] if i["kind"] == "job_stalled"] == ["suspected_hang"]
    assert any(i["kind"] == "collective_hang" for i in old["incidents"])  # still reported, just not confirming
    for path in dumps.iterdir():
        os.utime(path, (now, now))
    # A dump from only some of the ranks does not confirm either.
    (dumps / "nccl_trace_rank_1").unlink()
    partial = tensorquill.build_evidence(records, watchdog=stalled, flight_recorder=dumps)
    assert [i["assessment"] for i in partial["incidents"] if i["kind"] == "job_stalled"] == ["suspected_hang"]


def test_measured_pauses_come_with_a_recommended_threshold():
    status = _status("active", [_rank(0, "active", 9)])
    status.update({"pauses": {"eval": {"count": 3, "total_seconds": 900.0, "max_seconds": 410.0, "last_seconds": 300.0},
                              "checkpoint": {"count": 1, "total_seconds": 95.0, "max_seconds": 95.0, "last_seconds": 95.0}},
                   "recommended_silent_seconds": 820.0, "critical_seconds": 900.0})
    records = [{"run_id": "r", "step": s, "grad_norm": 1.0} for s in range(3)]
    evidence = tensorquill.build_evidence(records, watchdog=status)
    assert evidence["watchdog"]["recommended_silent_seconds"] == 820.0
    text = tensorquill.incident_summary(evidence)
    assert "eval up to 410 s (3x)" in text and "checkpoint up to 95 s (1x)" in text and "820 s" in text


def test_a_rank_that_never_connected_is_not_called_a_lost_connection():
    only_missing = _status("degraded", [_rank(0, "active", 400)], missing=[2], world_size=3)
    [hit] = detect_watchdog_incidents(only_missing)
    assert hit["assessment"] == "ranks_never_connected"


def test_watch_sees_a_warning_become_critical_with_no_new_records(tmp_path):
    path = tmp_path / "run.rank0.jsonl"
    path.write_text("".join(json.dumps({"run_id": "r", "rank": 0, "step": s, "grad_norm": 1.0}) + "\n" for s in range(3)))
    status = tmp_path / "watchdog.json"
    status.write_text(json.dumps(_status("stalled", [_rank(0, "silent", 2, 400.0, beat=1.0, severity="warning")])))
    seen = []

    def on_update(evidence, written):
        seen.append(evidence["incidents"][0]["severity"])
        if len(seen) == 1:  # the server's clock moves on: the same rank is now critical
            status.write_text(json.dumps(_status("stalled", [_rank(0, "silent", 2, 1000.0, beat=1.0, severity="critical")])))

    tensorquill.evidence.watch(path, tmp_path / "out.json", interval=0.01, polls=4, on_update=on_update, watchdog=status)
    assert seen == ["warning", "critical"]
