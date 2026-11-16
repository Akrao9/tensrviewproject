"""`examples/data_checkpoint_check.py` on CPU: a forgotten `set_epoch` and a bad resume are caught
through the real capture (`watch_loader`, `checkpoint`, the resumed optimizer's snapshot) and
`build_evidence`, and a run that does both right reports none of the data or checkpoint kinds."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import data_checkpoint_check  # noqa: E402

import tensorquill  # noqa: E402


def test_a_forgotten_set_epoch_and_a_bad_resume_are_caught(tmp_path):
    evidence = data_checkpoint_check.run("bad", tmp_path)
    ok, detail = data_checkpoint_check.check("bad", evidence)
    assert ok, detail
    by_kind = {i["kind"]: i for i in evidence["incidents"]}
    assert by_kind["epoch_order_repeated"]["sampler_epoch"] == 0
    assert by_kind["resume_incomplete"]["changes"][0]["source"] == "load"
    summary = tensorquill.incident_summary(evidence)
    assert "set_epoch" in summary and "Resume at step 14" in summary and "repeats the batch drawn at step 12" in summary


def test_a_run_that_does_both_right_reports_nothing_of_them(tmp_path):
    evidence = data_checkpoint_check.run("good", tmp_path)
    ok, detail = data_checkpoint_check.check("good", evidence)
    assert ok, detail
    assert evidence.get("not_checked", []) == []


def test_the_script_exits_zero_when_both_pass(tmp_path, capsys):
    assert data_checkpoint_check.main(["--case", "both", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "PASS bad" in out and "PASS good" in out
