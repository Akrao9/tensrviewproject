import json
import time
from pathlib import Path

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
    assert "Flight Recorder shows a stuck collective" not in summary
    # The job stopped as a whole and rank 1 is gone: read as a lost connection (Nov 17, 2026).
    assert hit["severity"] == "warning" and hit["assessment"] == "lost_connection" and hit["lost_ranks"] == [1]
    assert "read as a lost connection, not a hang" in summary and "still connected are most likely waiting" in summary

    degraded = _status("degraded", [_rank(0, "active", 400), _rank(1, "silent", 349, 900.0)], missing=[2], world_size=3)
    [hit] = detect_watchdog_incidents(degraded)
    assert hit["kind"] == "rank_silent" and hit["step"] == 349 and hit["newest_step"] == 400
    assert hit["missing_ranks"] == [2]
    text = tensorquill.incident_summary({"incidents": [hit], "samples": [], "layer_count": 0,
                                          "expert_count": 0, "router_names": []})
    assert "rank 1 sent no new step since step 349" in text and "ranks 2 never connected" in text
    assert "no heartbeat, so the phase is unknown" in text
    assert "Flight Recorder shows a stuck collective" not in text

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


@pytest.mark.parametrize("rank_only", [False, True])
def test_the_command_reads_the_run_status(tmp_path, capsys, monkeypatch, rank_only):
    pq = pytest.importorskip("pyarrow.parquet")
    from tensorquill.evidence import main
    from tensorquill.flight import records_to_batch

    run = tmp_path / "runs" / "r"
    part = run / "rank=0" / "attempt=1"
    part.mkdir(parents=True)
    records = [{"run_id": "r", "rank": 0, "attempt": 1, "step": s, "schema_version": 1, "grad_norm": 1.0} for s in range(5)]
    pq.write_table(__import__("pyarrow").Table.from_batches([records_to_batch(records)]), part / "part-00000000-00000004.parquet")
    (run / "watchdog.json").write_text(json.dumps(_status("stalled", [_rank(0, "silent", 4, 900.0)])))
    source = part.parent if rank_only else run
    assert tensorquill.build_evidence(source)["watchdog"]["summary"] == "stalled"
    out = tmp_path / "out.json"
    tensorquill.evidence.watch(source, out, polls=1)
    assert json.loads(out.read_text())["watchdog"]["summary"] == "stalled"
    monkeypatch.setattr("sys.argv", ["tensorquill", str(source), "-o", str(out)])
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


def _stack(frames, unchanged=95.0, others=()):
    threads = [{"name": "MainThread", "main": True,
                "frames": [{"file": f, "line": n, "function": fn, **({"own": True} if own else {})}
                           for fn, f, n, *own in frames]}]
    threads += [{"name": name, "main": False, "frames": []} for name in others]
    return {"still_seconds": 140.0, "unchanged_seconds": unchanged, "threads": threads}


def test_a_stopped_rank_that_sent_its_stack_is_reported_with_where_it_was():
    site = "/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py"
    stuck = _stack([("all_reduce", site, 2940), ("wrapper", "/venv/lib/python3.12/site-packages/torch/x.py", 3),
                    ("train_step", "/job/train.py", 88), ("<module>", "/job/train.py", 120)])
    ranks = [{**_rank(0, "silent", 149, 640.0, beat=4.0, severity="warning"), "stack": stuck},
             {**_rank(1, "silent", 149, 640.0, beat=4.0, severity="warning"), "stack": _stack(
                 [("all_reduce", site, 2940), ("train_step", "/job/train.py", 88)], unchanged=80.0)},
             _rank(2, "unresponsive", 149, 640.0, beat=300.0, severity="warning")]
    [hit] = detect_watchdog_incidents(_status("stalled", ranks))
    # The stack narrows the cause; what the stream alone can say is unchanged.
    assert hit["assessment"] == "suspected_hang"
    assert [p["rank"] for p in hit["stuck_at"]] == [0, 1]  # rank 2 sent none
    assert not any("stack" in r for r in hit["ranks"])  # the stacks stay in the status file
    first = hit["stuck_at"][0]
    assert (first["function"], first["line"], first["unchanged_seconds"]) == ("all_reduce", 2940, 95.0)
    assert first["caller"] == {"function": "train_step", "file": "/job/train.py", "line": 88}
    assert "in_tensorquill" not in first
    text = _text([hit])
    assert ("Last stack: ranks 0, 1: main thread in `all_reduce` (distributed_c10d.py:2940),"
            " called from `train_step` (train.py:88), unchanged for 80 s.") in text

    # A call made by TensorQuill itself is named as that, and the caller is the job's own line.
    own = _stack([("all_gather", site, 3100),
                  ("replica_spread", "/venv/lib/python3.12/site-packages/tensorquill/distributed.py", 150, True),
                  ("loop", "/job/tensorquill/run.py", 7)])  # a folder of the job's with that name is the job's
    [hit] = detect_watchdog_incidents(_status("degraded", [
        {**_rank(0, "silent", 100, 400.0, beat=2.0, severity="warning"), "stack": own}, _rank(1, "active", 149)]))
    assert hit["kind"] == "rank_silent"
    [place] = hit["stuck_at"]
    assert place["in_tensorquill"]["function"] == "replica_spread" and place["caller"]["function"] == "loop"
    assert "reached through TensorQuill's `replica_spread` (distributed.py:150)" in _text([hit])


def test_names_and_numbers_that_are_not_are_left_out_of_where_a_rank_was():
    odd = {"still_seconds": True, "unchanged_seconds": "9",
           "threads": [{"main": True, "frames": [{"file": "a.py", "function": 7}]}]}
    [hit] = detect_watchdog_incidents(_status("stalled", [
        {**_rank(0, "silent", 149, 640.0, beat=4.0, severity="warning"), "stack": odd}]))
    assert hit["stuck_at"] == [{"rank": 0, "function": "?", "file": "a.py", "line": None}]
    assert "main thread in `?` (a.py:None)." in _text([hit])


@pytest.mark.parametrize("stack", [
    None, "frames", [], {}, {"threads": "x"}, {"threads": [{"main": True}]}, {"threads": [{"main": True, "frames": []}]},
    {"threads": [{"main": True, "frames": [{"line": 3}, 7]}]}, {"threads": [{"main": False, "frames": [{"file": "a.py"}]}]},
    {"threads": [{"main": 1, "frames": [{"file": "a.py"}]}]},
])
def test_a_stack_that_is_not_one_is_left_out_and_nothing_else_changes(stack):
    plain = _rank(0, "silent", 149, 640.0, beat=4.0, severity="warning")
    [expected] = detect_watchdog_incidents(_status("stalled", [plain]))
    [hit] = detect_watchdog_incidents(_status("stalled", [{**plain, "stack": stack}]))
    assert "stuck_at" not in hit and "Last stack" not in _text([hit])
    assert _text([hit]) == _text([expected])


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
    assert "Flight Recorder shows a stuck collective" in _text([stream])

    degraded = _status("degraded", [_rank(0, "active", 4), _rank(1, "silent", 4, 900.0)])
    both = tensorquill.build_evidence(records, watchdog=degraded, flight_recorder=dumps)
    [stream] = [i for i in both["incidents"] if i["kind"] == "rank_silent"]
    assert "Flight Recorder shows a stuck collective" in _text([stream])

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


def _write_rank_part(run: Path, rank: int, steps: range, *, extra: dict | None = None) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa
    from tensorquill.flight import records_to_batch

    attempt = run / f"rank={rank}" / "attempt=1"
    attempt.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "run_id": run.name,
            "rank": rank,
            "attempt": 1,
            "step": s,
            "schema_version": 1,
            "grad_norm": 1.0,
            "moe": {"r": [10, 10]},
            **(extra or {}),
        }
        for s in steps
    ]
    pq.write_table(
        pa.Table.from_batches([records_to_batch(records)]),
        attempt / f"part-{steps.start:08d}-{steps.stop - 1:08d}.parquet",
    )


def _write_current_report(run: Path, rank: int, last_step: int, steps: int) -> None:
    from tensorquill import evidence as ev

    folder = run / f"rank={rank}"
    report = {
        "schema_version": 1,
        "source": "tensorquill-server",
        "run_id": run.name,
        "rank": rank,
        "attempt": 1,
        "last_step": last_step,
        "steps": steps,
        "rebuilds": 0,
        "thresholds": {
            "overload_threshold": ev.OVERLOAD_THRESHOLD,
            "drift_tolerance": ev.DRIFT_TOLERANCE,
            "starve_window": ev.STARVE_WINDOW,
            "starve_ratio": ev.STARVE_RATIO,
            "spike_sustain": ev.SPIKE_SUSTAIN,  # a report without it is an older server's
        },
        "checked": list(ev.SERVER_KINDS),
        "not_checked": [
            {"check": "flight_recorder", "reason": "not given"},
            {"check": "traincheck", "reason": "not given"},
        ],
        "incidents": [],
    }
    (folder / "incidents.json").write_text(json.dumps(report))


def _write_stuck_fr_dumps(dumps: Path) -> bool:
    try:
        __import__("torch.distributed.flight_recorder.components.builder")
    except Exception:
        return False
    import pickle

    def entry(i, seq, state="completed"):
        return {
            "profiling_name": "nccl:all_reduce",
            "state": state,
            "process_group": ("0", "default_pg"),
            "input_sizes": [[4]],
            "output_sizes": [[4]],
            "input_dtypes": "float32",
            "output_dtypes": "float32",
            "collective_seq_id": str(seq),
            "p2p_seq_id": "0",
            "time_created_ns": 0,
            "frames": [],
            "record_id": i,
            "is_p2p": False,
        }

    dumps.mkdir(parents=True, exist_ok=True)
    for rank, entries in ((0, [entry(0, 1), entry(1, 2, "started")]), (1, [entry(0, 1)])):
        (dumps / f"nccl_trace_rank_{rank}").write_bytes(pickle.dumps({
            "entries": entries,
            "version": "2.8",
            "pg_config": {"0": {"name": "0", "desc": "default_pg", "ranks": "[0, 1]"}},
        }))
    return True


def test_watch_surfaces_watchdog_and_external_sources_during_report_grace(tmp_path, monkeypatch):
    from tensorquill import evidence as ev

    now = time.time()
    run = tmp_path / "runs" / "r"
    _write_rank_part(run, 0, range(0, 5), extra={"ts": now - 10})
    _write_current_report(run, 0, 4, 5)
    status = run / "watchdog.json"
    status.write_text(json.dumps(_status("active", [_rank(0, "active", 4)])))
    tc_file = tmp_path / "violations_summary.json"
    fr_dir = tmp_path / "fr_dumps"
    fr_dir.mkdir()
    has_fr = False

    seen: list[tuple[str, list[str]]] = []
    poll_count = 0
    real_read_watchdog = ev.read_watchdog

    def step_hooks(path):
        nonlocal poll_count, has_fr
        poll_count += 1
        if poll_count == 3:
            # Server still hasn't caught up on incidents.json (e.g. restarted with --no-detect),
            # but the job stalls -> must emit immediately without waiting out report_grace.
            status.write_text(json.dumps(_status("stalled", [_rank(0, "silent", 4, 900.0, beat=2.0, severity="critical")])))
        elif poll_count == 4:
            # Later-written TrainCheck summary with no incidents.json or watchdog.json change.
            tc_file.write_text(json.dumps({
                "distinct_invariants_violated": 1,
                "first_violation_step": 3,
                "violations": [{"display_name": "grad_norm_bounds", "first_step": 3, "last_step": 3, "occurrences": 1}],
            }))
        elif poll_count == 5:
            # Later-written Flight Recorder dump triggers a fresh emit via seen signature too.
            has_fr = _write_stuck_fr_dumps(fr_dir)
            if not has_fr:
                (fr_dir / "dump_rank_0").write_bytes(b"new-dump")
        return real_read_watchdog(path)

    monkeypatch.setattr(ev, "read_watchdog", step_hooks)

    def on_update(evidence, _out):
        seen.append((evidence["detection"]["state"], [i["kind"] for i in evidence["incidents"]]))
        if len(seen) == 1:
            # Make incidents.json "behind" while inside the long 3600 s grace window; poll 2 changes nothing else.
            _write_rank_part(run, 0, range(5, 10), extra={"ts": now - 5})

    ev.watch(
        run,
        tmp_path / "out.json",
        interval=0.0,
        polls=5,
        report_grace=3600.0,
        traincheck=tc_file,
        flight_recorder=fr_dir,
        on_update=on_update,
    )
    expected_poll5 = ["job_stalled", "traincheck_violation", "collective_hang"] if has_fr else ["job_stalled", "traincheck_violation"]
    assert seen == [
        ("current", []),
        ("behind", ["job_stalled"]),
        ("behind", ["job_stalled", "traincheck_violation"]),
        ("behind", expected_poll5),
    ]
    latest = json.loads((tmp_path / "out.json").read_text())
    if has_fr:
        [stall] = [i for i in latest["incidents"] if i["kind"] == "job_stalled"]
        assert stall["assessment"] == "confirmed_hang" and stall["corroborated_by"] == "collective_hang"
    else:
        assert any(item.get("check") == "flight_recorder" for item in latest.get("not_checked", []))

    # Same on the Python tail path: once stalled and emitted, later Flight Recorder dumps trigger a new emit.
    monkeypatch.setattr(ev, "read_watchdog", real_read_watchdog)
    jsonl = tmp_path / "run.rank0.jsonl"
    jsonl.write_text("".join(json.dumps({"run_id": "r", "rank": 0, "step": s, "grad_norm": 1.0, "ts": now - 10 + s}) + "\n" for s in range(5)))
    py_fr = tmp_path / "py_fr"
    py_fr.mkdir()
    py_status = tmp_path / "py_watchdog.json"
    py_status.write_text(json.dumps(_status("stalled", [_rank(0, "silent", 4, 900.0, beat=2.0, severity="critical")])))
    py_seen: list[list[str]] = []

    def py_update(evidence, _out):
        py_seen.append([i["kind"] for i in evidence["incidents"]])
        if len(py_seen) == 1:
            if not _write_stuck_fr_dumps(py_fr):
                (py_fr / "dump_rank_0").write_bytes(b"new-dump")

    ev.watch(jsonl, tmp_path / "py_out.json", interval=0.0, polls=3, watchdog=py_status, flight_recorder=py_fr, on_update=py_update)
    assert len(py_seen) == 2
    if has_fr:
        assert py_seen == [["job_stalled"], ["job_stalled", "collective_hang"]]


def test_watch_at_t0_handles_late_ranks_without_latching_or_mixing(tmp_path, monkeypatch, capsys):
    import functools
    import sys
    from tensorquill import evidence as ev

    # 1) CLI --watch starts before rank=0 flushes (only rank=1 exists), then rank=0 arrives and falls back to Python.
    run = tmp_path / "runs" / "r"
    _write_rank_part(run, 1, range(0, 5))
    poll = 0
    real_stamp = ev._other_sources_stamp

    def advance_ranks(**kw):
        nonlocal poll
        poll += 1
        if poll == 2:
            _write_rank_part(run, 0, range(0, 3))
        elif poll == 3:
            _write_rank_part(run, 0, range(3, 10))
        return real_stamp(**kw)

    monkeypatch.setattr(ev, "_other_sources_stamp", advance_ranks)
    real_watch = ev.watch
    monkeypatch.setattr(ev, "watch", functools.partial(real_watch, polls=3))
    out = tmp_path / "cli.evidence.json"
    monkeypatch.setattr(sys, "argv", ["tensorquill", str(run), "--watch", "--interval", "0", "-o", str(out)])
    ev.main()
    evidence = json.loads(out.read_text())
    assert [s["step"] for s in evidence["samples"]] == list(range(10))
    assert "several ranks here; reading rank 0" in capsys.readouterr().out

    # Undo test patches before Subtests 2 & 3.
    monkeypatch.setattr(ev, "watch", real_watch)
    monkeypatch.setattr(ev, "_other_sources_stamp", real_stamp)

    # 2) rank=0 flushes first with a current report; rank=1 arrives on poll 2; poll 3 goes stale -> Tail reads rank=0 only.
    run2 = tmp_path / "runs" / "r2"
    _write_rank_part(run2, 0, range(0, 4))
    _write_current_report(run2, 0, 3, 4)
    poll2 = 0
    real_read_report = ev.read_server_report

    def spawn_rank1_and_stale(folder, **kw):
        nonlocal poll2
        poll2 += 1
        if poll2 == 2:
            _write_rank_part(run2, 1, range(0, 4))
        elif poll2 == 3:
            _write_rank_part(run2, 0, range(4, 8))
        return real_read_report(folder, **kw)

    monkeypatch.setattr(ev, "read_server_report", spawn_rank1_and_stale)
    out2 = tmp_path / "cli2.evidence.json"
    real_watch(run2, out2, interval=0.0, polls=3, report_grace=0.0, default_rank=0)
    ev2 = json.loads(out2.read_text())
    assert ev2["detection"]["source"] == "python"
    assert [s["step"] for s in ev2["samples"]] == list(range(8))

    # 3) Direct watch(run, rank=None) without default_rank does NOT stay latched on rank=1 when rank=0 appears.
    monkeypatch.setattr(ev, "read_server_report", real_read_report)
    run3 = tmp_path / "runs" / "r3"
    _write_rank_part(run3, 1, range(0, 4))
    _write_current_report(run3, 1, 3, 4)

    def add_rank0_after_first(_ev, _out):
        _write_rank_part(run3, 0, range(0, 4))

    with pytest.raises(ValueError, match="pass rank="):
        real_watch(run3, tmp_path / "direct.json", interval=0.0, polls=2, on_update=add_rank0_after_first)


def test_watch_status_clock_skew_uses_monotonic_time(tmp_path, monkeypatch):
    from tensorquill import evidence as ev

    path = tmp_path / "run.rank0.jsonl"
    path.write_text("".join(json.dumps({"run_id": "r", "rank": 0, "step": s, "grad_norm": 1.0}) + "\n" for s in range(3)))
    status_file = tmp_path / "watchdog.json"

    # Server clock 1 h behind watcher clock: poll 1 looks stale by wall clock, then a refresh clears staleness.
    base_ms = int((time.time() - 3600.0) * 1000)
    status_file.write_text(json.dumps({
        **_status("active", [_rank(0, "active", 2)]),
        "server_running": True,
        "refresh_seconds": 5.0,
        "written_at_ms": base_ms,
    }))
    ages: list[float | None] = []

    def on_update_behind(evidence, _out):
        ages.append(evidence["watchdog"].get("not_written_for_seconds"))
        if len(ages) == 1:
            status_file.write_text(json.dumps({
                **_status("active", [_rank(0, "active", 2)]),
                "server_running": True,
                "refresh_seconds": 5.0,
                "written_at_ms": base_ms + 5000,
            }))

    ev.watch(path, tmp_path / "out_behind.json", interval=0.0, polls=2, on_update=on_update_behind, watchdog=status_file)
    assert ages[0] is not None and ages[0] > 3000.0
    assert ages[1] is None

    # Server clock 1 h ahead of watcher clock (negative wall age): monotonic elapsed catches hang anyway.
    status_file.write_text(json.dumps({
        **_status("active", [_rank(0, "active", 2)]),
        "server_running": True,
        "refresh_seconds": 1.0,
        "written_at_ms": int((time.time() + 3600.0) * 1000),
    }))
    mono_times = iter([100.0, 105.0, 105.0])
    monkeypatch.setattr(ev.time, "monotonic", lambda: next(mono_times, 105.0))
    ages.clear()
    ev.watch(
        path,
        tmp_path / "out_ahead.json",
        interval=0.0,
        polls=2,
        on_update=lambda e, _: ages.append(e["watchdog"].get("not_written_for_seconds")),
        watchdog=status_file,
    )
    assert ages == [None, 5.0]


def test_a_stall_before_the_first_stored_batch_is_said_as_that():
    """The server calls a rank silent that has an open stream and no batch stored (its `last_step`
    is null): the summary must not say "after step 0" or "since step None"."""
    def said(hit):
        return tensorquill.incident_summary({"incidents": [hit], "samples": [], "layer_count": 0,
                                             "expert_count": 0, "router_names": []})

    hung = {**_rank(0, "silent", None, 410.0, beat=3.0, severity="warning"), "work": 40}
    waiting = {**_rank(1, "silent", None, 410.0, beat=3.0, severity="warning"), "work": 0}
    [hit] = detect_watchdog_incidents(_status("stalled", [hung, waiting]))
    assert (hit["kind"], hit["step"], hit["severity"]) == ("job_stalled", 0, "warning")
    text = said(hit)
    assert ("Job stalled before its first batch was stored: none of 2 ranks has sent a step in 410 s, and the"
            " heartbeats' work counter stands at 40 (its heartbeats continue and it says it is in phase `train`).") in text
    assert "a long startup (compilation, a data load)" in text and "after step" not in text
    # No counter moved, or an older server that writes none: nothing is said of it.
    for ranks in ([waiting], [_rank(0, "silent", None, 1900.0)]):
        text = said(detect_watchdog_incidents(_status("stalled", ranks))[0])
        assert "before its first batch was stored: none of 1 rank has sent a step in" in text and "work counter" not in text
    # One rank stuck before its first step while another goes on.
    [hit] = detect_watchdog_incidents(_status("degraded", [_rank(0, "active", 400), hung | {"rank": 1}]))
    assert (hit["kind"], hit["step"], hit["newest_step"]) == ("rank_silent", 400, 400)
    text = said(hit)
    assert "rank 1 has had no step stored yet (its heartbeats continue" in text and "None" not in text
    # A job with one rank past its first batch is a stall after that step, as before.
    [hit] = detect_watchdog_incidents(_status("stalled", [_rank(0, "silent", 49, 700.0), hung | {"rank": 1}]))
    assert "Job stalled after step 49" in said(hit)


def _rule(summary, *ranks):
    [hit] = detect_watchdog_incidents(_status(summary, [_rank(r, state, step, severity="critical") for r, state, step in ranks]))
    return hit["kind"], hit["assessment"], hit.get("lost_ranks")


def test_a_lost_rank_labels_the_stall_when_it_explains_it():
    """The same cases as `a_lost_rank_labels_the_stall_when_it_explains_it` in `watchdog.rs`."""
    # One rank OOM-killed, the others blocked in a collective waiting for it: the job stopped.
    assert _rule("stalled", (0, "silent", 40), (1, "lost", 40), (2, "unresponsive", 40)) == (
        "job_stalled", "lost_connection", [1])
    # Every stopped rank lost while the rest step.
    assert _rule("degraded", (0, "active", 90), (1, "lost", 40)) == ("rank_silent", "lost_connection", [1])
    # A lost rank beside an unrelated silent one while another steps: the silent one is not
    # waiting for the lost one, so it is a suspected hang, and the lost rank is still named.
    assert _rule("degraded", (0, "active", 90), (1, "lost", 40), (2, "silent", 60)) == (
        "rank_silent", "suspected_hang", [1])
    assert _rule("stalled", (0, "silent", 40)) == ("job_stalled", "suspected_hang", None)
    # The summary says it is not a hang, and names the lost rank; a suspected hang keeps its sentence.
    text = _text(detect_watchdog_incidents(_status("degraded", [_rank(0, "active", 90), _rank(1, "lost", 40, stream="dropped", severity="critical")])))
    assert "The connection of rank 1 dropped and did not come back: read as a lost connection, not a hang." in text
    assert "out-of-memory kill" in text and "[critical, lost connection]" in text
    assert "look the same to the server" not in text and "waiting for it" not in text
    hang = _text(detect_watchdog_incidents(_status("stalled", [_rank(0, "silent", 40, severity="critical")])))
    assert "can look the same from here" in hang and "lost connection" not in hang
