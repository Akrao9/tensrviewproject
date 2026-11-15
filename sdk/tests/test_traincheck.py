import json
import sys

import pytest

import tensorquill
from tensorquill.invariants import read_traincheck, traincheck_incident


def _summary(violations, first=None):
    return {"first_violation_step": first, "distinct_invariants_violated": len(violations), "violations": violations}


def _violation(name, first, last, relation="VarPeriodicChangeRelation", occurrences=3):
    return {"display_name": name, "relation_type": relation, "first_step": first, "last_step": last, "occurrences": occurrences}


def test_violations_become_one_incident_with_traincheck_s_own_step(tmp_path):
    path = tmp_path / "violations_summary.json"
    path.write_text(json.dumps(_summary([
        _violation("optimizer.step() should change model parameters", 7, 12),
        _violation("zero_grad() resets .grad", 3, 5, "APIContainRelation"),
    ], first=3)))
    summary = read_traincheck(path)
    assert [v["first_step"] for v in summary["violations"]] == [3, 7]  # earliest first
    incident = traincheck_incident(summary, step=99)
    assert incident["kind"] == "traincheck_violation" and incident["step"] == 3
    assert incident["distinct_invariants_violated"] == 2
    text = tensorquill.incident_summary({"incidents": [incident], "samples": [], "layer_count": 0,
                                          "expert_count": 0, "router_names": []})
    assert "TrainCheck reports 2 violated invariants, the first at its step 3" in text
    assert "zero_grad() resets .grad (steps 3-5)" in text and "own iterations" in text


def test_a_results_directory_with_several_traces_is_merged_and_no_violation_is_no_incident(tmp_path):
    for name, violations, first in (("trace_a", [_violation("a", 10, 10)], 10), ("trace_b", [_violation("b", 4, 8)], 4)):
        (tmp_path / "check_results" / name).mkdir(parents=True)
        (tmp_path / "check_results" / name / "violations_summary.json").write_text(json.dumps(_summary(violations, first)))
    merged = read_traincheck(tmp_path / "check_results")
    assert merged["distinct_invariants_violated"] == 2 and merged["first_violation_step"] == 4
    assert len(merged["files"]) == 2

    clean = tmp_path / "clean.json"
    clean.write_text(json.dumps(_summary([])))
    assert traincheck_incident(read_traincheck(clean), step=5) is None
    # A violation with no step information falls back to the last step TensorQuill recorded.
    nostep = tmp_path / "nostep.json"
    nostep.write_text(json.dumps(_summary([_violation("x", None, None)])))
    assert traincheck_incident(read_traincheck(nostep), step=41)["step"] == 41


def test_build_evidence_reads_it_and_lists_what_it_could_not(tmp_path, monkeypatch, capsys):
    good = tmp_path / "violations_summary.json"
    good.write_text(json.dumps(_summary([_violation("optimizer.step() changes params", 2, 6)], first=2)))
    records = [{"run_id": "r", "step": s, "grad_norm": 1.0} for s in range(9)]
    evidence = tensorquill.build_evidence(records, traincheck=good)
    assert [i["kind"] for i in evidence["incidents"]] == ["traincheck_violation"]
    assert evidence["traincheck"] == {"distinct_invariants_violated": 1, "first_violation_step": 2}

    (tmp_path / "empty").mkdir()
    for bad, reason in ((tmp_path / "missing.json", "does not exist"), (tmp_path / "empty", "no violations_summary.json")):
        evidence = tensorquill.build_evidence(records, traincheck=bad)
        assert evidence["incidents"] == [] and reason in evidence["not_checked"][0]["reason"]
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"something": "else"}))
    assert "not a TrainCheck violations summary" in tensorquill.build_evidence(records, traincheck=wrong)["not_checked"][0]["reason"]
    wrong.write_text("not json")
    assert "not readable JSON" in tensorquill.build_evidence(records, traincheck=wrong)["not_checked"][0]["reason"]

    run = tmp_path / "run.rank0.jsonl"
    run.write_text("".join(json.dumps(r) + "\n" for r in records))
    from tensorquill.evidence import main

    monkeypatch.setattr(sys, "argv", ["tensorquill", str(run), "-o", str(tmp_path / "out.json"), "--traincheck", str(good)])
    main()
    assert "TrainCheck reports 1 violated invariant" in capsys.readouterr().out
