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


def _rank(rank, state, last_step, quiet=0.0, stream="open"):
    return {"rank": rank, "attempt": 1, "last_step": last_step, "seconds_since_progress": quiet,
            "stream": stream, "state": state}


def test_a_stalled_job_and_a_silent_rank_become_incidents():
    stalled = _status("stalled", [_rank(0, "silent", 149, 640.0), _rank(1, "lost", 149, 700.0, "dropped")])
    [hit] = detect_watchdog_incidents(stalled)
    assert hit["kind"] == "job_stalled" and hit["step"] == 149 and len(hit["ranks"]) == 2
    summary = tensorquill.incident_summary({"incidents": [hit], "samples": [], "layer_count": 0,
                                             "expert_count": 0, "router_names": []})
    assert "Job stalled after step 149" in summary and "1 lost its connection" in summary
    assert "Flight Recorder" in summary

    degraded = _status("degraded", [_rank(0, "active", 400), _rank(1, "silent", 349, 900.0)], missing=[2], world_size=3)
    [hit] = detect_watchdog_incidents(degraded)
    assert hit["kind"] == "rank_silent" and hit["step"] == 349 and hit["newest_step"] == 400
    assert hit["missing_ranks"] == [2]
    text = tensorquill.incident_summary({"incidents": [hit], "samples": [], "layer_count": 0,
                                          "expert_count": 0, "router_names": []})
    assert "rank 1 sent no new step since step 349" in text and "ranks 2 never connected" in text

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
