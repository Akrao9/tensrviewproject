"""Lane AI-3 (Nov 22, 2026): the distributed diagnosis harnesses, on their CPU paths.

`flight_recorder_hang_check.py`: the bounded driver is run for real (a Gloo job that truly hangs,
ended by the driver, every process gone). On torch 2.15 the Gloo dumps cannot be read by PyTorch's own
analyzer, so that run is NOT PROVEN; its check of TensorQuill's answer against the plan is tested
here on dumps CONSTRUCTED in the analyzer's test format (as `test_collectives.py` builds them).

`fault_injection_check.py`: the wrapper is run under torchrun on a STUB campaign module
(`fault_injection_check_stub.py`, not lm-resiliency's code), so the captures are real TensorQuill
captures; the ground truth and its negative variants are CONSTRUCTED from the documented format of
`injection.json`.
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import socket
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SRC = Path(__file__).resolve().parents[1] / "src"
SIBLING = Path("/Users/ak/Documents/tensrview/lm-resiliency")
sys.path.insert(0, str(EXAMPLES))

import fault_injection_check as fic  # noqa: E402
import fault_injection_check_entry as entry  # noqa: E402
import fault_injection_check_stub as stub  # noqa: E402
import flight_recorder_hang_check as frc  # noqa: E402


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])}


def _alive(pid: int) -> bool:
    return frc._alive(pid)


# --- the bounded Flight Recorder hang ------------------------------------------------------------------


def test_the_hang_is_real_bounded_and_every_process_is_gone(tmp_path):
    out = tmp_path / "fr"
    done = subprocess.run([sys.executable, str(EXAMPLES / "flight_recorder_hang_check.py"), "--cpu", "--out", str(out),
                           "--master-port", str(_port())], env=_env(), capture_output=True, text=True, timeout=240)
    result = json.loads((out / "result.json").read_text())
    assert result["backend"] == "gloo" and result["dump"] == "on-request"
    assert result["hang_established"] is True, done.stdout
    assert result["left_running"] == [] and len(result["worker_pids"]) == 3
    assert not any(_alive(pid) for pid in result["worker_pids"])
    assert result["dump_files"] == [f"fr_trace_rank_{r}" for r in range(3)]  # written by PyTorch's own writer
    assert result["plan"] == {"skip_rank": 1, "skip_step": 5, "collective": "all_reduce", "collective_seq_id": 6,
                              "waiting_ranks": [0, 2], "last_recorded_step": 3, "nproc": 3}
    # The capture of a waiting rank has steps 0..3 on disk: step 4 ran, and is held back until a next step ends.
    capture = next((out / result["run_id"] / "captures").glob("*.rank0.jsonl"))
    steps = [json.loads(line)["step"] for line in capture.read_text().splitlines() if line.strip()]
    assert steps == [0, 1, 2, 3]
    if done.returncode == frc.NOT_PROVEN:
        # torch 2.15: PyTorch's own analyzer cannot read Gloo's dumps (pg_config under an empty name).
        assert result["pytorch_analyzer"]["returncode"] != 0
        assert "PyTorch's own Flight Recorder analyzer cannot read these dumps" in result["not_proven"][0]
        assert "NOT PROVEN" in done.stdout and result["failures"] == []
    else:  # a torch whose Gloo dumps its analyzer reads: the check against the plan decides
        assert done.returncode == frc.PASS, done.stdout
        assert result["incident"]["missing_ranks"] == [1]


def _entry(record_id, seq, state="completed"):
    return {"profiling_name": "nccl:all_reduce", "state": state, "process_group": ("0", "default_pg"),
            "input_sizes": [[4, 4]], "output_sizes": [[4, 4]], "input_dtypes": "float32", "output_dtypes": "float32",
            "collective_seq_id": str(seq), "p2p_seq_id": "0", "time_created_ns": 0, "frames": [],
            "record_id": record_id, "is_p2p": False}


def _constructed_dumps(directory: Path, skip_rank: int, skip_step: int, world: int = 3) -> None:
    """CONSTRUCTED dumps in the analyzer's test format: one all_reduce per step, rank `skip_rank` never
    entered step `skip_step`'s, the others started it and never finished."""
    directory.mkdir()
    for rank in range(world):
        entries = [_entry(i, i + 1) for i in range(skip_step)]
        if rank != skip_rank:
            entries.append(_entry(skip_step, skip_step + 1, "started"))
        document = {"entries": entries, "version": "2.8",
                    "pg_config": {"0": {"name": "0", "desc": "default_pg", "ranks": str(list(range(world)))}}}
        (directory / f"{frc.PREFIX}{rank}").write_bytes(pickle.dumps(document))


def _plan(**changes):
    plan = {"skip_rank": 1, "skip_step": 5, "collective": "all_reduce", "collective_seq_id": 6,
            "waiting_ranks": [0, 2], "last_recorded_step": 3, "nproc": 3}
    return {**plan, **changes}


def test_the_check_holds_tensorquill_s_answer_against_the_plan(tmp_path):
    pytest.importorskip("torch.distributed.flight_recorder.components.builder")
    from tensorquill.collectives import analyze_flight_recorder, flight_recorder_incident

    _constructed_dumps(tmp_path / "dumps", skip_rank=1, skip_step=5)
    incident = flight_recorder_incident(analyze_flight_recorder(tmp_path / "dumps", prefix=frc.PREFIX), step=3)
    evidence = {"incidents": [incident]}
    failures, hit = frc._check(_plan(), evidence)
    assert failures == [] and hit["kind"] == "collective_hang"
    # Each part of the plan is checked on its own.
    assert any("missing_ranks" in f for f in frc._check(_plan(skip_rank=2, waiting_ranks=[0, 1]), evidence)[0])
    assert any("collective_seq_id" in f for f in frc._check(_plan(collective_seq_id=5), evidence)[0])
    assert any("step" in f for f in frc._check(_plan(last_recorded_step=4), evidence)[0])
    assert any("collective 'all_reduce'" in f for f in frc._check(_plan(collective="broadcast"), evidence)[0])
    assert frc._check(_plan(), {"incidents": []})[0] == ["no collective_hang in the evidence (incidents: none)"]


def test_the_hang_driver_refuses_a_plan_it_cannot_judge(tmp_path):
    assert frc.main(["--cpu", "--out", str(tmp_path), "--skip-step", "1"]) == frc.USAGE
    assert frc.main(["--cpu", "--out", str(tmp_path), "--skip-rank", "3"]) == frc.USAGE


# --- the fault-injection campaign ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def stub_run(tmp_path_factory):
    """The wrapper under torchrun on the STUB campaign: real captures, constructed ground truth."""
    root = tmp_path_factory.mktemp("stub")
    stub.write_stub_tree(root)
    artifacts = tmp_path_factory.mktemp("artifacts")
    done = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc-per-node=3", "--master-addr",
                           "127.0.0.1", "--master-port", str(_port()), str(EXAMPLES / "fault_injection_check_entry.py"),
                           "--lm-resiliency", str(root), "--cpu-stub", "--artifact-dir", str(artifacts)],
                          env=_env(), capture_output=True, text=True, timeout=240)
    assert done.returncode == 0, done.stdout + done.stderr
    return artifacts


def _variant(stub_run: Path, tmp_path: Path, change) -> Path:
    """A copy of the stub run with a CONSTRUCTED change to its injection.json."""
    folder = tmp_path / "variant"
    folder.mkdir()
    (folder / "tensorquill").symlink_to(stub_run / "tensorquill")
    injection = json.loads((stub_run / "injection.json").read_text())
    change(injection)
    (folder / "injection.json").write_text(json.dumps(injection))
    return folder


def _compared(folder: Path, *extra: str) -> tuple[int, dict]:
    status = fic.main(["--artifact-dir", str(folder), *extra])
    return status, json.loads((folder / "tensorquill_comparison.json").read_text())


def test_the_wrapper_attaches_a_session_on_every_rank_beside_the_ground_truth(stub_run):
    captures = stub_run / "tensorquill"
    attach = json.loads((captures / "attach.json").read_text())
    assert attach["start_step"] == 1 and attach["world_size"] == 3 and attach["replica_every"] == 1
    assert sorted(fic.capture_files(captures)) == [0, 1, 2]
    for path in fic.capture_files(captures).values():
        steps = [record["step"] for record in fic._records(path)]
        assert steps == list(range(1, 9))  # TensorQuill's step N is the iteration N; closed before teardown
        assert path.with_name(path.name.replace(".jsonl", ".end.json")).exists()


def test_tensorquill_names_the_planted_rank_and_step(stub_run):
    status, result = _compared(stub_run)
    assert result["summary"]["found"] == 1 and result["missed"] == []
    (occurrence,) = result["occurrences"]
    assert occurrence["ranks"] == [2] and occurrence["iteration"] == 4
    assert {(m["rank"], m["step"]) for m in occurrence["actions"][0]["matched_by"]} == {(2, 4)}
    assert status == (fic.PASS if not result["false_reports"] else fic.FAIL)


def test_the_right_step_on_the_wrong_rank_is_not_found(stub_run, tmp_path):
    def wrong_rank(injection):
        injection["injections"][0]["execution_rank"] = 1
    status, result = _compared(_variant(stub_run, tmp_path, wrong_rank))
    assert result["summary"]["found"] == 0 and status == fic.FAIL
    assert (2, 4) in {(r["rank"], r["step"]) for r in result["false_reports"]}


def test_the_right_rank_at_the_wrong_step_is_not_found(stub_run, tmp_path):
    def wrong_step(injection):
        injection["injections"][0]["iteration"] = 6
    status, result = _compared(_variant(stub_run, tmp_path, wrong_step))
    assert result["summary"]["missed"] == 1 and status == fic.FAIL


def test_a_failed_injection_is_not_scored_and_a_report_at_it_is_still_false(stub_run, tmp_path):
    def failed(injection):
        injection["injections"][0].update(status="failed", verified=False, injection_succeeded=False)
    status, result = _compared(_variant(stub_run, tmp_path, failed))
    assert result["summary"]["occurrences"] == 0 and result["summary"]["not_injected"] == 1
    at = [r for r in result["false_reports"] if (r["rank"], r["step"]) == (2, 4)]
    assert at and all(r["matches_record_not_injected"] == ["stub-weight-noise@4"] for r in at)
    assert status == fic.FAIL
    assert fic.main(["--artifact-dir", str(tmp_path / "variant"), "--measure"]) == fic.PASS


def test_a_report_with_no_injection_is_a_false_report(stub_run, tmp_path):
    def none(injection):
        injection["injections"] = []
    status, result = _compared(_variant(stub_run, tmp_path, none))
    assert (2, 4) in {(r["rank"], r["step"]) for r in result["false_reports"]} and status == fic.FAIL


def test_a_skipped_occurrence_is_left_out(stub_run, tmp_path):
    def skipped(injection):
        injection["injections"][0].update(status="skipped_probability", verified=False, injection_succeeded=False)
    _, result = _compared(_variant(stub_run, tmp_path, skipped))
    assert result["summary"]["occurrences"] == 0 and result["summary"]["not_injected"] == 0


def test_every_action_of_a_correlated_occurrence_must_be_named_within_its_window():
    actions = [
        {"injection_id": "a", "occurrence_id": "o", "incident_id": "i", "fault_id": "f", "failure_type": "t",
         "expected_kind": "sdc", "surface": "weight", "rank": 1, "iteration": 10, "window": [10, 12], "status": "completed"},
        {"injection_id": "b", "occurrence_id": "o", "incident_id": "i", "fault_id": "g", "failure_type": "t",
         "expected_kind": "sdc", "surface": "weight", "rank": 3, "iteration": 10, "window": [10, 12], "status": "completed"},
    ]
    one = [{"source": "rank_evidence", "kind": "grad_spike", "rank": 1, "step": 12}]
    assert fic.compare(actions, [], one)["occurrences"][0]["result"] == "partial"
    both = one + [{"source": "rank_evidence", "kind": "grad_spike", "rank": 3, "step": 11}]
    assert fic.compare(actions, [], both)["occurrences"][0]["result"] == "found"
    late = [{"source": "rank_evidence", "kind": "grad_spike", "rank": 1, "step": 13}]  # one past the window
    result = fic.compare(actions, [], late)
    assert result["occurrences"][0]["result"] == "missed" and len(result["false_reports"]) == 1
    # start_step shifts TensorQuill's steps onto the campaign's iterations.
    shifted = [{"source": "rank_evidence", "kind": "x", "rank": 1, "step": 9}, {"source": "rank_evidence", "kind": "x", "rank": 3, "step": 9}]
    assert fic.compare(actions, [], shifted, start_step=0)["occurrences"][0]["result"] == "found"


def test_what_cannot_be_ground_truth_is_not_proven(tmp_path, stub_run):
    assert fic.main(["--artifact-dir", str(tmp_path)]) == fic.NOT_PROVEN  # no injection.json
    folder = _variant(stub_run, tmp_path, lambda i: i["injections"][0].update(execution_rank="2"))
    assert fic.main(["--artifact-dir", str(folder)]) == fic.NOT_PROVEN  # not an integer
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "injection.json").write_text((stub_run / "injection.json").read_text())
    assert fic.main(["--artifact-dir", str(empty)]) == fic.NOT_PROVEN  # no capture


def test_the_entry_stops_with_one_line_without_the_campaign_s_requirements(tmp_path, capsys):
    assert entry.main(["--lm-resiliency", str(tmp_path), "--artifact-dir", str(tmp_path / "a")]) == entry.USAGE
    assert "is not an lm-resiliency checkout" in capsys.readouterr().out
    if SIBLING.is_dir():
        assert entry.main(["--lm-resiliency", str(SIBLING), "--artifact-dir", str(tmp_path / "a")]) == entry.USAGE
        line = capsys.readouterr().out.strip()
        assert line.count("\n") == 0 and ("lm_resiliency is not installed" in line or "PyTorch 2.10 to 2.13" in line
                                          or "eight ranks" in line)


@pytest.mark.skipif(not SIBLING.is_dir(), reason="the lm-resiliency checkout is not beside this one")
def test_the_campaign_still_defines_the_names_the_wrapper_replaces():
    text = (SIBLING / "tests/validation/fault_injection/pytorch.py").read_text()
    assert "\ndef main(" in text and "\ndef _teardown(" in text
    assert "enable_fault_injection,\n" in text and "faults = enable_fault_injection(" in text
    assert "_teardown(\n" in text and "resiliency = enable_resiliency(" in text
    assert text.index("resiliency = enable_resiliency(") < text.index("faults = enable_fault_injection(")
    record = (SIBLING / "lm_resiliency/fault_injection/reports.py").read_text()
    for field in ("execution_rank", "iteration", "occurrence_id", "injection_succeeded", "status"):
        assert f'"{field}": self.' in record
