import json
import pickle
import sys

import pytest

import tensorquill

pytest.importorskip("torch.distributed.flight_recorder.components.builder")  # PyTorch 2.10 or later

from tensorquill.collectives import analyze_flight_recorder, flight_recorder_incident  # noqa: E402


def _entry(record_id, name, seq, state="completed", sizes=((4, 4),), pg=("0", "default_pg")):
    shapes = [list(s) for s in sizes]
    return {
        "profiling_name": f"nccl:{name}", "state": state, "process_group": pg, "input_sizes": shapes,
        "output_sizes": shapes, "input_dtypes": "float32", "output_dtypes": "float32",
        "collective_seq_id": str(seq), "p2p_seq_id": "0", "time_created_ns": 0, "frames": [],
        "record_id": record_id, "is_p2p": False,
    }


def _dump(directory, rank, entries, world=(0, 1)):
    document = {
        "entries": entries, "version": "2.8", "pg_config": {"0": {"name": "0", "desc": "default_pg", "ranks": str(list(world))}},
    }
    (directory / f"nccl_trace_rank_{rank}").write_bytes(pickle.dumps(document))


def _write(tmp_path, rank0, rank1, **kwargs):
    directory = tmp_path / "fr"
    directory.mkdir()
    _dump(directory, 0, rank0, **kwargs)
    if rank1 is not None:
        _dump(directory, 1, rank1, **kwargs)
    return directory


def test_a_rank_that_never_entered_a_collective_is_the_hang_s_culprit(tmp_path, capsys):
    both = _entry(0, "all_reduce", 1)
    directory = _write(tmp_path, [both, _entry(1, "all_reduce", 2, "started")], [both])
    analysis = analyze_flight_recorder(directory)
    assert capsys.readouterr().out == ""  # the analyzer's console output is taken, not printed
    assert analysis["dumped_ranks"] == [0, 1] and analysis["problem_count"] == 1
    problem = analysis["problem"]
    assert problem["type"] == "hang" and problem["missing_ranks"] == [1] and problem["all_ranks_entered"] is False
    assert (problem["collective"], problem["collective_seq_id"], problem["state"]) == ("all_reduce", 2, "started")
    incident = flight_recorder_incident(analysis, step=149)
    assert incident["kind"] == "collective_hang" and incident["step"] == 149
    text = tensorquill.incident_summary({"incidents": [incident], "samples": [], "layer_count": 0,
                                          "expert_count": 0, "router_names": []})
    assert "rank 1 never entered it" in text and "all_reduce #2 in group" in text and "default_pg" in text


def test_ranks_that_issued_different_collectives_are_a_desync(tmp_path):
    ok = _entry(0, "all_reduce", 1)
    directory = _write(tmp_path, [ok, _entry(1, "all_reduce", 2, "scheduled")], [ok, _entry(1, "all_gather", 2, "scheduled")])
    problem = analyze_flight_recorder(directory)["problem"]
    assert problem["type"] == "desync" and "COLLECTIVE_TYPE_MISMATCH" in problem["mismatch"]
    incident = flight_recorder_incident(analyze_flight_recorder(directory), step=7)
    assert incident["kind"] == "collective_desync"
    text = tensorquill.incident_summary({"incidents": [incident], "samples": [], "layer_count": 0,
                                          "expert_count": 0, "router_names": []})
    assert "Ranks issued different collectives at all_reduce #2" in text and "Culprit rank 1" in text


def test_every_rank_entering_but_not_finishing_points_at_the_network_not_one_rank(tmp_path):
    ok = _entry(0, "all_reduce", 1)
    stuck = _entry(1, "all_reduce", 2, "started")
    directory = _write(tmp_path, [ok, stuck], [ok, dict(stuck)])
    incident = flight_recorder_incident(analyze_flight_recorder(directory), step=3)
    assert incident["kind"] == "collective_hang" and incident["all_ranks_entered"] is True
    text = tensorquill.incident_summary({"incidents": [incident], "samples": [], "layer_count": 0,
                                          "expert_count": 0, "router_names": []})
    assert "every rank entered it but it did not complete" in text and "network, NCCL or a GPU" in text


def test_a_healthy_dump_and_a_partial_dump(tmp_path):
    ok = _entry(0, "all_reduce", 1)
    healthy = _write(tmp_path, [ok], [dict(ok)])
    analysis = analyze_flight_recorder(healthy)
    assert analysis["problem"] is None and flight_recorder_incident(analysis, step=0) is None
    partial = tmp_path / "partial"
    partial.mkdir()
    _dump(partial, 0, [ok])
    analysis = analyze_flight_recorder(partial)
    assert analysis["dumped_ranks"] == [0] and analysis["incomplete_dumps"] is True


def test_build_evidence_adds_the_incident_and_lists_what_it_could_not_check(tmp_path, monkeypatch):
    ok = _entry(0, "all_reduce", 1)
    directory = _write(tmp_path, [ok, _entry(1, "all_reduce", 2, "started")], [ok])
    records = [{"run_id": "r", "step": s, "grad_norm": 1.0} for s in range(12)]
    evidence = tensorquill.build_evidence(records, flight_recorder=directory)
    [incident] = [i for i in evidence["incidents"] if i["kind"] == "collective_hang"]
    assert incident["step"] == 11 and evidence["flight_recorder"]["problem_count"] == 1

    empty = tmp_path / "empty"
    empty.mkdir()
    evidence = tensorquill.build_evidence(records, flight_recorder=empty)
    assert evidence["incidents"] == []
    assert evidence["not_checked"][0]["check"] == "flight_recorder" and "no readable" in evidence["not_checked"][0]["reason"]
    assert "Not checked (flight_recorder)" in tensorquill.incident_summary(evidence)

    # A PyTorch without the analyzer (before 2.10): not checked, with the reason, never a crash.
    for name in list(sys.modules):
        if name.startswith("torch.distributed.flight_recorder"):
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "torch.distributed.flight_recorder", None)
    evidence = tensorquill.build_evidence(records, flight_recorder=directory)
    assert "no Flight Recorder analyzer" in evidence["not_checked"][0]["reason"]


def test_the_command_reads_the_dumps_and_the_servers_status(tmp_path, capsys):
    ok = _entry(0, "all_reduce", 1)
    directory = _write(tmp_path, [ok, _entry(1, "all_reduce", 2, "started")], [ok])
    run = tmp_path / "run.rank0.jsonl"
    run.write_text("".join(json.dumps({"run_id": "r", "rank": 0, "step": s, "grad_norm": 1.0}) + "\n" for s in range(4)))
    monkey_argv = ["tensorquill", str(run), "-o", str(tmp_path / "out.json"), "--flight-recorder", str(directory)]
    from tensorquill.evidence import main

    old = sys.argv
    sys.argv = monkey_argv
    try:
        main()
    finally:
        sys.argv = old
    assert "Collective hang at all_reduce #2" in capsys.readouterr().out
    assert any(i["kind"] == "collective_hang" for i in json.loads((tmp_path / "out.json").read_text())["incidents"])
