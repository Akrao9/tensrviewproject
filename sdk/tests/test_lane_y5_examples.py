"""Lane Y5 (wave Y): the seeded-failures example and CI. Rows XC1, XC8 and XC10 of `OPEN_BUGS.md`.

- XC1: the healthy verdict of `examples/seeded_failures.py` passed on an empty incident list alone,
  so a run that recorded nothing passed. It now needs every expected step, the two seeded kinds
  checked with enough history, and a capture that closed. The judge is tested on made-up records
  (empty, cut short, checks not run, not closed); the toy runs through `main` need `transformers`.
- XC8: the grad_spike judge's note formatted a missing ratio (the total's median before the seed
  was 0) and raised `TypeError`.
- XC10: CI's server job ran a fixed list of test files, so the server-dependent tests of later
  lanes were skipped on every Linux job. The job now runs the whole suite with the binary and fails
  if any test is skipped for want of it.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

import pytest

import tensorquill

SDK = Path(__file__).resolve().parents[1]
REPO = SDK.parent
sys.path.insert(0, str(SDK / "examples"))
import seeded_failures as sf  # noqa: E402

PARAM = "model.layers.2.self_attn.k_proj.weight"


# ---- XC1: the healthy verdict ----------------------------------------------------------------


def checked(records: int = 300, needed: int = 3) -> dict:
    return {"value": "checked", "records": records, "records_needed": needed, "enough_history": True}


def healthy_evidence(**kind_checks) -> dict:
    checks = {kind: checked() for kind in sf.CASES}
    checks.update(kind_checks)
    return {"incidents": [], "kind_checks": checks}


def rows(steps) -> list[dict]:
    return [{"step": step, "loss": 1.0} for step in steps]


CLOSED = {"state": "closed", "outcome": "closed"}


def test_xc1_a_whole_closed_capture_with_both_kinds_checked_and_no_incident_passes():
    verdict = sf.judge_healthy(healthy_evidence(), rows(range(350)), 350, CLOSED)
    assert verdict.passed, verdict
    assert "steps 0 to 349" in verdict.summary and "grad_spike" in verdict.summary and "stalled_weights" in verdict.summary


def test_xc1_an_empty_capture_fails():
    verdict = sf.judge_healthy(healthy_evidence(), [], 350, CLOSED)
    assert not verdict.passed
    assert "no rows" in verdict.summary


def test_xc1_a_capture_cut_short_fails_and_says_where_it_stopped():
    verdict = sf.judge_healthy(healthy_evidence(), rows(range(120)), 350, CLOSED)
    assert not verdict.passed
    assert "230 of the 350 expected steps are missing" in verdict.summary and "from step 120" in verdict.summary


def test_xc1_a_gap_in_the_middle_or_a_step_beyond_the_run_fails():
    gap = sf.judge_healthy(healthy_evidence(), rows([*range(100), *range(101, 350)]), 350, CLOSED)
    assert not gap.passed and "1 of the 350 expected steps are missing" in gap.summary and "from step 100" in gap.summary
    extra = sf.judge_healthy(healthy_evidence(), rows(range(351)), 350, CLOSED)
    assert not extra.passed and "outside 0 to 349" in extra.summary


@pytest.mark.parametrize("run_end", [
    {"state": "not_closed"},
    {"state": "closed", "outcome": "raised", "exception": "RuntimeError"},
    {"state": "written_after_close"},
    None,
])
def test_xc1_a_capture_not_known_to_have_closed_fails(run_end):
    verdict = sf.judge_healthy(healthy_evidence(), rows(range(350)), 350, run_end)
    assert not verdict.passed
    assert "not known to have closed cleanly" in verdict.summary


def test_xc1_a_capture_that_stopped_fails_and_names_the_error():
    verdict = sf.judge_healthy(healthy_evidence(), rows(range(350)), 350, CLOSED, capture_error=OSError("disk full"))
    assert not verdict.passed
    assert "capture stopped" in verdict.summary and "disk full" in verdict.summary


@pytest.mark.parametrize("kind", sf.CASES)
@pytest.mark.parametrize("check", [
    {"value": "not_configured", "records": 0, "records_needed": 3, "enough_history": False},
    {"value": "checked", "records": 2, "records_needed": 3, "enough_history": False},
    {"value": "unavailable", "records": 0, "records_needed": 3, "reason": "the detector failed"},
    None,
])
def test_xc1_a_seeded_kind_that_was_not_checked_fails(kind, check):
    evidence = healthy_evidence(**{kind: check})
    if check is None:
        del evidence["kind_checks"][kind]
    verdict = sf.judge_healthy(evidence, rows(range(350)), 350, CLOSED)
    assert not verdict.passed
    assert f"{kind} was not checked" in verdict.summary


def test_xc1_an_incident_still_fails_a_whole_capture():
    evidence = healthy_evidence()
    evidence["incidents"] = [{"kind": "nan_inf", "step": 3}]
    verdict = sf.judge_healthy(evidence, rows(range(350)), 350, CLOSED)
    assert not verdict.passed and "lists nan_inf" in verdict.summary


def test_xc1_every_problem_is_said_not_only_the_first():
    evidence = healthy_evidence(stalled_weights={"value": "not_configured", "records": 0, "records_needed": 3})
    verdict = sf.judge_healthy(evidence, rows(range(10)), 350, {"state": "not_closed"})
    assert not verdict.passed
    for part in ("expected steps are missing", "not known to have closed", "stalled_weights was not checked"):
        assert part in verdict.summary, verdict.summary


def test_xc1_a_healthy_run_shorter_than_its_checks_need_is_refused_before_it_trains(tmp_path):
    with pytest.raises(SystemExit) as stopped, contextlib.redirect_stderr(io.StringIO()) as said:
        sf.main(["--toy", "--case", "healthy", "--steps", str(sf.HEALTHY_MIN_STEPS - 1), "--out", str(tmp_path / "o")])
    assert stopped.value.code == 2
    assert f"needs at least {sf.HEALTHY_MIN_STEPS}" in said.getvalue()
    assert not (tmp_path / "o").exists()


# The toy Llama through the real capture: needs transformers.


def run_main(argv) -> tuple[int, str]:
    printed = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(printed):
        try:
            sf.main(argv)
        except SystemExit as stopped:
            code = stopped.code
    return code, printed.getvalue()


def test_xc1_a_healthy_toy_run_whose_sink_fails_is_a_fail_not_a_pass(tmp_path, monkeypatch):
    """The cross-check's reproduction, at the healthy case's own step count: every write fails."""
    pytest.importorskip("transformers")

    def write(self, records):
        raise OSError("injected sink write failure")

    monkeypatch.setattr(tensorquill.JsonlSink, "write", write)
    code, printed = run_main(["--toy", "--case", "healthy", "--out", str(tmp_path / "out")])
    assert code == 1, printed
    assert "FAIL: " in printed and "injected sink write failure" in printed and "PASS: " not in printed
    assert (tmp_path / "out" / "healthy.jsonl").read_text() == ""


def test_xc1_a_healthy_toy_run_whose_sink_fails_part_way_is_a_fail_that_says_where_it_stopped(tmp_path, monkeypatch):
    """The cut-short control: the first write lands, every later one fails, so the file holds a prefix."""
    pytest.importorskip("transformers")
    real = tensorquill.JsonlSink.write
    calls = []

    def write(self, records):
        calls.append(len(records))
        if len(calls) > 1:
            raise OSError("injected sink write failure after the first write")
        real(self, records)

    monkeypatch.setattr(tensorquill.JsonlSink, "write", write)
    code, printed = run_main(["--toy", "--case", "healthy", "--out", str(tmp_path / "out")])
    written = tensorquill.load_records(tmp_path / "out" / "healthy.jsonl")
    assert 0 < len(written) < 350, len(written)
    assert code == 1, printed
    assert f"expected steps are missing, from step {len(written)}" in printed, printed
    assert "the capture stopped" in printed and "not known to have closed cleanly (not_closed)" in printed


def test_xc1_a_healthy_toy_run_at_the_shortest_allowed_length_passes_with_both_kinds_checked(tmp_path):
    """`HEALTHY_MIN_STEPS` is enough for the detectors' own `kind_checks`, so the parser's refusal
    above and the verdict agree."""
    pytest.importorskip("transformers")
    code, printed = run_main(["--toy", "--case", "healthy", "--steps", str(sf.HEALTHY_MIN_STEPS), "--out", str(tmp_path / "out")])
    assert code == 0, printed
    assert f"PASS: no incident in steps 0 to {sf.HEALTHY_MIN_STEPS - 1}" in printed


def test_xc1_one_step_short_of_the_minimum_the_verdict_itself_fails(tmp_path):
    """The run the parser refuses, through `run_case` directly: the verdict does not lean on the parser."""
    pytest.importorskip("transformers")
    import torch

    args = sf.build_parser().parse_args(["--toy", "--case", "healthy", "--steps", str(sf.HEALTHY_MIN_STEPS - 1),
                                         "--out", str(tmp_path / "out")])
    device = torch.device("cpu")
    with contextlib.redirect_stdout(io.StringIO()):
        verdict = sf.run_case(sf.HEALTHY, args, device, sf.make_batches(args, device, args.steps), args.out)
    assert not verdict.passed
    assert "stalled_weights was not checked" in verdict.summary


# ---- XC8: a zero median before the seed --------------------------------------------------------


def zero_median_records() -> list[dict]:
    # The cross-check's totals: 0, 0, 1 before the seed (median 0) and 20 at it.
    return [{"step": step, "grad_norm": total, "grad_norms": {PARAM: total}}
            for step, total in zip(range(247, 251), (0.0, 0.0, 1.0, 20.0))]


def test_xc8_a_zero_median_and_no_incident_is_a_fail_that_says_the_ratio_is_undefined():
    verdict = sf.judge_grad_spike({"incidents": []}, zero_median_records(), PARAM, 250, 100.0)
    assert not verdict.passed
    assert verdict.summary.startswith("no grad_spike incident")
    note = verdict.notes[0]
    assert "median over the 3 recorded steps before the seed is 0" in note and "undefined" in note, note


def test_xc8_a_zero_median_with_an_incident_at_the_seed_passes():
    incident = {"kind": "grad_spike", "step": 250, "ratio": None, "trigger": "total",
                "parameters": [{"name": PARAM, "ratio": 20.0}], "spike_count": 1, "spike_steps": [250]}
    verdict = sf.judge_grad_spike({"incidents": [incident]}, zero_median_records(), PARAM, 250, 100.0)
    assert verdict.passed, verdict
    assert "total ratio" not in verdict.summary or "unknown" in verdict.summary
    assert "undefined" in verdict.notes[0]


def test_xc8_a_zero_median_and_no_share_at_all_still_gives_a_verdict():
    records = [{"step": step, "grad_norm": 0.0, "grad_norms": {PARAM: 0.0}} for step in range(240, 250)]
    records.append({"step": 250, "grad_norm": 20.0, "grad_norms": {PARAM: 20.0}})
    verdict = sf.judge_grad_spike({"incidents": []}, records, PARAM, 250, 100.0)
    assert not verdict.passed and verdict.summary.startswith("no grad_spike incident")
    assert any("undefined" in note for note in verdict.notes), verdict.notes


# ---- XC10: the server job runs every server-dependent test ------------------------------------

WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
# The skip reasons a test gives when it has no binary, or no pyarrow to hand it records, as written today:
# a `skipif` of its own, or `server_detect.server_binary()`'s error (lane W3's tests skip with that text).
SERVER_SKIPS = (
    "tensorquill-server is not built",
    "tensorquill-server is not built (TENSORQUILL_SERVER_BIN)",
    "TENSORQUILL_SERVER_BIN names no server binary",
    "needs the server binary and pyarrow",
    "needs the server binary: set TENSORQUILL_SERVER_BIN",
    "the Rust detectors need the tensorquill-server binary and none was found: pip install tensorquill-server",
    "/x/tensorquill-server speaks protocol 3 and this tensorquill needs protocol 4: install a tensorquill and a "
    "tensorquill-server that match",
    "pyarrow is not installed",
    "could not import 'pyarrow': No module named 'pyarrow'",
)


def server_job() -> str:
    if not WORKFLOW.exists():
        pytest.skip("no workflow in this checkout")
    text = WORKFLOW.read_text(encoding="utf-8")
    return text.split("\n  server:\n", 1)[1].split("\n  package:\n", 1)[0]


def server_pytest_step() -> str:
    job = server_job()
    step = job.split("- name: Sender and end-to-end tests", 1)[1]
    return step.split("\n      - name:", 1)[0]


def selected_paths(step: str) -> list[str]:
    """The paths the step's pytest line names (none: pytest's own `testpaths`, the whole suite)."""
    line = next(line for line in step.splitlines() if re.search(r"\bpytest\b", line) and "grep" not in line)
    words = line.split("pytest", 1)[1].split("|", 1)[0].split()
    return [w for w in words if not w.startswith("-") and not w.startswith(("2>", ">"))]


def test_xc10_the_server_job_runs_the_whole_suite_not_a_list_of_files():
    """A list misses every later lane's file (V7, W2 and W3 were missed) and every file that reaches
    the binary through a helper, so the job names no file: pytest's `testpaths`, all of `tests/`."""
    paths = selected_paths(server_pytest_step())
    assert paths in ([], ["tests"], ["tests/"]), paths
    for name in ("test_lane_v7_server_detect.py", "test_lane_w2_attempt_order.py", "test_lane_w3_detector_rules.py"):
        assert (SDK / "tests" / name).exists(), name  # the files the fixed list left out are in the folder it runs


def test_xc10_the_server_job_names_the_binary_and_fails_on_a_skip_for_want_of_it():
    step = server_pytest_step()
    assert "TENSORQUILL_SERVER_BIN: ${{ github.workspace }}/server/target/debug/tensorquill-server" in step
    assert 'test -x "$TENSORQUILL_SERVER_BIN"' in step
    assert "set -o pipefail" in step  # a failing pytest piped into tee still fails the step
    assert re.search(r"pytest -ra\b[^\n]*\| tee ", step)
    guard = step.split("tee", 1)[1]
    assert "grep" in guard and "exit 1" in guard
    pattern = guard_pattern(guard)
    for reason in SERVER_SKIPS:
        assert re.search(pattern, f"SKIPPED [1] tests/test_x.py:10: {reason}"), reason
    for other in ("needs lightning", "could not import 'transformers': No module named 'transformers'",
                  "needs torch.distributed", "the recorded GPU capture is not on this machine"):
        assert not re.search(pattern, f"SKIPPED [1] tests/test_x.py:10: {other}"), other
    assert not re.search(pattern, "PASSED tests/test_flight_e2e.py::test_tensorquill_server_binary_runs")


def guard_pattern(guard: str) -> str:
    return re.search(r"grep -E '([^']+)'", guard).group(1)


def test_xc10_every_job_but_the_server_one_and_the_triggers_are_as_they_were():
    if not WORKFLOW.exists():
        pytest.skip("no workflow in this checkout")
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.split("\njobs:\n", 1)[0].startswith("name: CI\n\non:\n  push:\n    branches: [main]\n  pull_request:\n")
    assert [m.group(1) for m in re.finditer(r"^  (\w+):\n    (?:#.*\n    )*name:", text, re.M)] == [
        "cpu", "extras", "server", "package", "gpu"]
