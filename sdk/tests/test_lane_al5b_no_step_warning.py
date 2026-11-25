"""Lane AL-5, change 3: a value with no step in memory and no `step=` (`WAVE_AL_CONTRACT.md`, section
Q5).

It is left out, with one warning per session whose text is the contract's: what case it is, and the
explicit call that records it, `session.log({<first>: value}, step=N)`, with `N` from section Q5's
table (`start_step` before the first step, the last step training finished after `flush()`). A value
the mirror brought gets one more sentence (the mirror passes no step, Q7). The six
`scalar_no_step_*` cases are driven from `fixtures/wave_al/cases.json`, each by its `setup`.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill.sink import MemorySink

from test_lane_ak1_mirror import wandb_stand_in  # noqa: E402

CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_al" / "cases.json").read_text())["cases"]
NO_STEP = [c for c in CASES if c["name"].startswith("scalar_no_step_")]
LEFT_OUT = "TensorQuill left out the scalar"
MIRROR_SENTENCE = ("It came from your tracker's log call through mirror_scalars, which cannot pass a step (the "
                   "tracker's own step is ignored): call session.log(..., step=...) yourself for it.")


@pytest.fixture(autouse=True)
def no_trackers(monkeypatch):
    for name in ("wandb", "mlflow", "mlflow.tracking", "mlflow.tracking.fluent", "wandb.sdk", "wandb.sdk.wandb_run"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.delenv("TENSORQUILL_ENDPOINT", raising=False)
    yield
    mirror = sys.modules.get("tensorquill.mirror")
    for session in list(getattr(mirror, "_sessions", [])):
        session.close()


def _open(start_step: int = 0):
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    sink = MemorySink()
    session = tensorquill.init(model, sink=sink, run_id="al5_q5", every=50, heartbeat_seconds=None,
                               stack_seconds=None, start_step=start_step)
    return session, sink, model, torch.optim.SGD(model.parameters(), lr=0.01)


def _step(session, model, opt) -> None:
    session.log({"train/loss": 1.0})
    model(torch.ones(2, 4)).sum().backward()
    opt.step()
    opt.zero_grad()


def _run(setup: dict, value_name: str, monkeypatch) -> tuple[list[dict], list[str]]:
    """One case's `setup`: steps, an optional flush, then the value under its phase, logged by the
    session or brought by a stand-in `wandb.log`, then `close()`. Returns the records and the
    left-out warnings."""
    wandb = None
    if setup["mirrored"]:
        wandb = wandb_stand_in(lambda name, module: monkeypatch.setitem(sys.modules, name, module))
        wandb.module.init()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session, sink, model, opt = _open(setup["start_step"])

        def log(value):
            if wandb is not None:
                wandb.module.log({value_name: value})
            else:
                session.log({value_name: value})

        if setup.get("baseline_first"):
            with session.phase("eval"):
                session.log({"eval/baseline": 2.5})
        for _ in range(setup["steps_before"]):
            _step(session, model, opt)
        if setup["flush"]:
            session.flush()
        if setup["phase"] == "eval":
            with session.phase("eval"):
                log(1.9)
        else:
            log(1.9)  # pending: left out at close() if no step follows
        session.close()
    said = [str(w.message) for w in caught if LEFT_OUT in str(w.message)]
    return sink.records, said


@pytest.mark.parametrize("case", NO_STEP, ids=[c["name"] for c in NO_STEP])
def test_each_no_step_case_is_left_out_with_the_contracts_warning(case, monkeypatch):
    expect = case["expect"]
    records, said = _run(expect["setup"], "eval/loss", monkeypatch)
    assert [r for r in records if r.get("record") == "scalars"] == []  # records_written: 0
    assert len([r for r in records if r.get("record") == "scalars"]) == expect["records_written"]
    assert all("eval/loss" not in r.get("scalars", {}) and "eval/baseline" not in r.get("scalars", {})
               for r in records)
    assert len(said) == expect["warnings"]
    for text in expect["warning_contains"]:
        assert text in said[0], (text, said[0])
    for text in expect.get("warning_lacks", ()):
        assert text not in said[0], (text, said[0])
    assert said[0].endswith("Said once per session.")
    assert "protocol 5" not in said[0]  # wave AK's text named the record to come; now it shows the call


def test_the_baseline_text_in_full():
    """Section Q5's text for a baseline evaluation, word for word."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session, sink, model, opt = _open(0)
        with session.phase("eval"):
            session.log({"eval/loss": 2.5})
        session.close()
    (said,) = [str(w.message) for w in caught if LEFT_OUT in str(w.message)]
    assert said == (
        "TensorQuill left out the scalar 'eval/loss': it was logged before the first training step (a baseline "
        "evaluation), when no step is in memory. To record it, log it with an explicit step, which writes a "
        "scalar-only record: session.log({'eval/loss': value}, step=0), the step training starts at. Said once "
        "per session."
    )


def test_the_after_flush_text_in_full():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session, sink, model, opt = _open(10)
        for _ in range(3):
            _step(session, model, opt)
        session.flush()
        with session.phase("eval"):
            session.log({"eval/b": 1.0, "eval/a": 2.0})
        session.close()
    (said,) = [str(w.message) for w in caught if LEFT_OUT in str(w.message)]
    assert said == (
        "TensorQuill left out the scalars 'eval/a', 'eval/b': it was logged after flush() with no training step "
        "since (an evaluation after flush()), when no step is in memory. To record it, log it with an explicit "
        "step, which writes a scalar-only record: session.log({'eval/a': value}, step=12), the last step training "
        "finished. Said once per session."
    )


def test_the_pending_at_close_text_in_full():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session, sink, model, opt = _open(0)
        for _ in range(5):
            _step(session, model, opt)
        session.flush()
        session.log({"eval/loss": 1.9})
        session.close()
    (said,) = [str(w.message) for w in caught if LEFT_OUT in str(w.message)]
    assert said == (
        "TensorQuill left out the scalar 'eval/loss': it was still pending at close() after flush() with no "
        "training step since, when no step is in memory. To record it, log it with an explicit step, which "
        "writes a scalar-only record: session.log({'eval/loss': value}, step=4), the last step training "
        "finished. Said once per session."
    )


def test_a_value_pending_at_close_in_a_session_that_never_trained_gets_the_baseline_text():
    """Section Q5: "A value pending at close() with no step at all (a session that never trained) is the
    first case"."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session, sink, model, opt = _open(7)
        session.log({"eval/loss": 1.9})
        session.close()
    (said,) = [str(w.message) for w in caught if LEFT_OUT in str(w.message)]
    assert "(a baseline evaluation)" in said and "step=7), the step training starts at" in said


def test_more_than_five_names_are_shown_as_today():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session, sink, model, opt = _open(0)
        with session.phase("eval"):
            session.log({f"eval/m{i}": float(i) for i in range(7)})
        session.close()
    (said,) = [str(w.message) for w in caught if LEFT_OUT in str(w.message)]
    assert said.startswith("TensorQuill left out the scalars 'eval/m0', 'eval/m1', 'eval/m2', 'eval/m3', "
                           "'eval/m4' and 2 more: ")
    assert "session.log({'eval/m0': value}, step=0)" in said


def test_the_mirrors_sentence_is_said_once_and_only_for_its_value(monkeypatch):
    wandb = wandb_stand_in(lambda name, module: monkeypatch.setitem(sys.modules, name, module))
    wandb.module.init()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session, sink, model, opt = _open(0)
        with session.phase("eval"):
            wandb.module.log({"eval/acc": 0.5})
            session.log({"eval/own": 1.0})  # left out silently: said once per session
        session.close()
    (said,) = [str(w.message) for w in caught if LEFT_OUT in str(w.message)]
    assert said.count(MIRROR_SENTENCE) == 1
    assert said.endswith(MIRROR_SENTENCE + " Said once per session.")
    assert "'eval/acc'" in said
    assert wandb.values() == [{"eval/acc": 0.5}]  # the tracker still gets it


def test_a_value_the_session_logs_itself_gets_no_mirror_sentence(monkeypatch):
    wandb = wandb_stand_in(lambda name, module: monkeypatch.setitem(sys.modules, name, module))
    wandb.module.init()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session, sink, model, opt = _open(0)
        with session.phase("eval"):
            session.log({"eval/own": 1.0})
            wandb.module.log({"eval/acc": 0.5})
        session.close()
    (said,) = [str(w.message) for w in caught if LEFT_OUT in str(w.message)]
    assert "mirror_scalars" not in said and "'eval/own'" in said


def test_a_mirrored_value_pending_at_close_after_a_flush_gets_the_mirror_sentence(monkeypatch):
    wandb = wandb_stand_in(lambda name, module: monkeypatch.setitem(sys.modules, name, module))
    wandb.module.init()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session, sink, model, opt = _open(0)
        for _ in range(2):
            _step(session, model, opt)
        session.flush()
        wandb.module.log({"train/epoch": 1.0})  # outside "eval": pending, and no step follows
        session.close()
    (said,) = [str(w.message) for w in caught if LEFT_OUT in str(w.message)]
    assert "it was still pending at close() after flush()" in said and MIRROR_SENTENCE in said
    assert "step=1), the last step training finished" in said


def test_a_value_left_out_is_never_written_and_the_explicit_call_records_it(tmp_path):
    """What the warning shows, done: the same value with the suggested `step=` is a scalar-only record."""
    sink_path = tmp_path / "shown.jsonl"
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    session = tensorquill.init(model, sink=tensorquill.JsonlSink(sink_path), run_id="al5_q5", every=50,
                               heartbeat_seconds=None, stack_seconds=None)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with session.phase("eval"):
            session.log({"eval/loss": 2.5})  # left out
            session.log({"eval/loss": 2.5}, step=0)  # the call the warning shows
    session.close()
    lines = [json.loads(line) for line in sink_path.read_text().splitlines()]
    assert [(r["record"], r["step"], r["seq"], r["scalars"]) for r in lines] == [("scalars", 0, 0, {"eval/loss": 2.5})]
