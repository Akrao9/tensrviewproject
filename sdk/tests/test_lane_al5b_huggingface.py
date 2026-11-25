"""Lane AL-5, change 4: the Hugging Face callback passes `step=` itself when no step is in memory
(`WAVE_AL_CONTRACT.md` section Q5, "The addition").

`on_evaluate` with no step in memory (`eval_on_start`'s baseline, or an evaluation after a `flush()`
with no step since) logs the evaluation metrics with `step=N`, a scalar-only record, so nothing is
left out and nothing is warned. `N` is the session's own number (`Session._suggested_step`); the
tests show it is `state.global_step` for a baseline and `state.global_step - 1` after a flush, on a
real `Trainer`, with gradient accumulation too. Cases `scalar_hf_eval_on_start_baseline` and
`scalar_hf_eval_on_start_on_a_resume`.

Transformers and Accelerate are imported inside each test, so an interpreter without them skips
test by test.
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

CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_al" / "cases.json").read_text())["cases"]
LEFT_OUT = "TensorQuill left out the scalar"


def _case(name: str) -> dict:
    return next(c for c in CASES if c["name"] == name)


def _transformers():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    return transformers


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x, labels=None):
        return {"loss": (self.net(x).squeeze(-1) - labels).pow(2).mean()}


def _data(n=64, seed=0):
    torch.manual_seed(seed)
    return [{"x": torch.randn(4), "labels": torch.randn(())} for _ in range(n)]


def _trainer(tmp_path, model, callbacks, **kwargs):
    transformers = _transformers()
    options = dict(
        output_dir=str(tmp_path), max_steps=4, per_device_train_batch_size=8, per_device_eval_batch_size=16,
        report_to="none", save_strategy="no", disable_tqdm=True, use_cpu=True, learning_rate=0.01,
        logging_steps=2, eval_strategy="steps", eval_steps=2,
    )
    options.update(kwargs)
    args = transformers.TrainingArguments(**options)
    return transformers.Trainer(model=model, args=args, train_dataset=_data(), eval_dataset=_data(32, seed=1),
                                callbacks=callbacks)


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _scalar_records(path: Path) -> list[dict]:
    return [r for r in _lines(path) if r.get("record") == "scalars"]


def _step_records(path: Path) -> list[dict]:
    return [r for r in _lines(path) if r.get("record") != "scalars"]


@pytest.fixture(autouse=True)
def no_trackers(monkeypatch):
    for name in ("wandb", "mlflow", "mlflow.tracking", "mlflow.tracking.fluent", "wandb.sdk", "wandb.sdk.wandb_run"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    yield


def _seen(transformers, tq, seen: list):
    """A callback after TensorQuill's that notes, at each `on_evaluate`, the `Trainer`'s `global_step`,
    whether TensorQuill had a step in memory, and the step it would suggest."""

    class Seen(transformers.TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):
            seen.append(state.global_step)

    class Noted(transformers.TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):
            session = tq.session
            seen.append((session._newest_scalars() is None, session._steps_committed, session._suggested_step()))

    return Noted(), Seen()


def test_eval_on_start_baseline_is_a_scalar_only_record_and_nothing_is_warned(tmp_path):
    """`scalar_hf_eval_on_start_baseline`: on a fresh run the baseline's metrics are one scalar-only
    record at step 0 (`state.global_step`), `seq` 0, names under `eval/`, and no warning."""
    from tensorquill.huggingface import TensorQuillCallback

    expect = _case("scalar_hf_eval_on_start_baseline")["expect"]
    path = tmp_path / "hf.jsonl"
    tq = TensorQuillCallback(sink=tensorquill.JsonlSink(path), every=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer = _trainer(tmp_path / "out", Model(), [tq], eval_on_start=True)
        trainer.train()
    assert [str(w.message) for w in caught if LEFT_OUT in str(w.message)] == []
    records = _scalar_records(path)
    assert len(records) == expect["records_written"]
    (record,) = records
    assert record["step"] == expect["record"]["step"] and record["seq"] == expect["record"]["seq"]
    assert record["scalars"] and all(name.startswith(expect["record"]["names_start_with"]) for name in record["scalars"])
    baseline = next(e for e in trainer.state.log_history if "eval_loss" in e and e["step"] == 0)
    assert record["scalars"]["eval/loss"] == pytest.approx(baseline["eval_loss"])
    assert [r["step"] for r in _step_records(path)] == [0, 1, 2, 3]
    # The evaluations after steps 1 and 3 had a step in memory: on those steps' own records, as before.
    by_step = {r["step"]: r.get("scalars", {}) for r in _step_records(path)}
    assert "eval/loss" in by_step[1] and "eval/loss" in by_step[3] and "eval/loss" not in by_step[0]


def test_eval_on_start_on_a_resume_is_recorded_at_the_resumed_step(tmp_path):
    """`scalar_hf_eval_on_start_on_a_resume`: resumed from `checkpoint-500`, `start_step` and
    `global_step` are 500, so the baseline's record is at step 500.

    The checkpoint is made by a real `Trainer` at step 2, then named `checkpoint-500` with its
    `trainer_state.json` saying step 500 (the Trainer reads `global_step` from there), and the data is
    not skipped ahead (`ignore_data_skip`), so 500 steps need not be trained to resume at 500."""
    from tensorquill.huggingface import TensorQuillCallback

    expect = _case("scalar_hf_eval_on_start_on_a_resume")["expect"]
    first = _trainer(tmp_path / "out", Model(), [], max_steps=2, save_strategy="steps", save_steps=2,
                     eval_strategy="no")
    first.train()
    made = tmp_path / "out" / "checkpoint-2"
    folder = tmp_path / "out" / "checkpoint-500"
    made.rename(folder)
    state = json.loads((folder / "trainer_state.json").read_text())
    state["global_step"] = expect["setup"]["resume_global_step"]
    (folder / "trainer_state.json").write_text(json.dumps(state))

    path = tmp_path / "hf.jsonl"
    tq = TensorQuillCallback(sink=tensorquill.JsonlSink(path), every=4)
    seen: list = []
    before, after = _seen(sys.modules["transformers"], tq, seen)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer = _trainer(tmp_path / "out", Model(), [tq, before, after], max_steps=502, eval_on_start=True,
                           ignore_data_skip=True)
        trainer.train(resume_from_checkpoint=str(folder))
    assert [str(w.message) for w in caught if LEFT_OUT in str(w.message)] == []
    (record,) = _scalar_records(path)
    assert record["step"] == expect["record"]["step"] == 500 and record["seq"] == expect["record"]["seq"]
    assert all(name.startswith("eval/") for name in record["scalars"])
    assert [r["step"] for r in _step_records(path)] == [500, 501]
    assert seen[:2] == [(True, 0, 500), 500]  # the session's number is global_step for the baseline


@pytest.mark.parametrize("accumulation", [1, 2])
def test_an_evaluation_after_a_flush_is_recorded_at_the_last_step_that_ended(tmp_path, accumulation):
    """With no step in memory after a `flush()`, `N = state.global_step - 1`, the session's
    `_next_step - 1`: shown to agree on a real `Trainer`, with gradient accumulation too."""
    from tensorquill.huggingface import TensorQuillCallback

    transformers = _transformers()
    path = tmp_path / "hf.jsonl"
    tq = TensorQuillCallback(sink=tensorquill.JsonlSink(path), every=50)

    class FlushBeforeEvaluating(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if control.should_evaluate:
                tq.session.flush()  # the user's own flush: no step in memory when the evaluation ends

    seen: list = []
    before, after = _seen(transformers, tq, seen)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer = _trainer(tmp_path / "out", Model(), [FlushBeforeEvaluating(), tq, before, after],
                           gradient_accumulation_steps=accumulation)
        trainer.train()
    assert [str(w.message) for w in caught if LEFT_OUT in str(w.message)] == []
    records = _scalar_records(path)
    evaluations = [e for e in trainer.state.log_history if "eval_loss" in e]
    assert [e["step"] for e in evaluations] == [2, 4]  # global_step at each evaluation
    assert [(r["step"], r["seq"]) for r in records] == [(1, 0), (3, 1)]
    assert [r["scalars"]["eval/loss"] for r in records] == pytest.approx([e["eval_loss"] for e in evaluations])
    # At each evaluation: no step in memory, a step committed, and the session's number global_step - 1.
    pairs = list(zip(seen[0::2], seen[1::2]))
    assert pairs == [((True, 2, 1), 2), ((True, 4, 3), 4)]
    assert all(suggested == global_step - 1 for (_, _, suggested), global_step in pairs)
    assert all("eval/loss" not in r.get("scalars", {}) for r in _step_records(path))


def test_with_a_step_in_memory_it_logs_as_before(tmp_path):
    from tensorquill.huggingface import TensorQuillCallback

    path = tmp_path / "hf.jsonl"
    tq = TensorQuillCallback(sink=tensorquill.JsonlSink(path), every=4)
    trainer = _trainer(tmp_path / "out", Model(), [tq], max_steps=6)
    trainer.train()
    assert _scalar_records(path) == []
    by_step = {r["step"]: r.get("scalars", {}) for r in _step_records(path)}
    assert all("eval/loss" in by_step[s] for s in (1, 3, 5))


def test_with_a_wandb_logger_the_baseline_is_on_step_0_and_in_its_own_record(tmp_path, monkeypatch):
    """Section Q5, "What shares a step": the mirror brings the `Trainer`'s logged baseline outside
    "eval", before `on_evaluate`, so it is pending and lands on step 0's own record; the callback's
    scalar-only record holds the same values under the same names. Nothing is warned."""
    from tensorquill.huggingface import TensorQuillCallback

    from test_lane_ak1_mirror import wandb_stand_in

    transformers = _transformers()
    wandb = wandb_stand_in(lambda name, module: monkeypatch.setitem(sys.modules, name, module))
    wandb.module.init()

    class FakeWandbLogger(transformers.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            from transformers.integrations.integration_utils import rewrite_logs

            wandb.module.log({**rewrite_logs(dict(logs)), "train/global_step": state.global_step})

    path = tmp_path / "hf.jsonl"
    tq = TensorQuillCallback(sink=tensorquill.JsonlSink(path), every=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer = _trainer(tmp_path / "out", Model(), [FakeWandbLogger(), tq], eval_on_start=True)
        trainer.train()
    assert [str(w.message) for w in caught if LEFT_OUT in str(w.message)] == []
    (record,) = _scalar_records(path)
    step_0 = next(r for r in _step_records(path) if r["step"] == 0)["scalars"]
    assert record["step"] == 0
    assert step_0["eval/loss"] == pytest.approx(record["scalars"]["eval/loss"])


def test_on_evaluate_with_no_state_still_never_raises():
    """`on_evaluate` reads the session's own number, not `state`: called with no state (as wave AK's
    test calls it), the baseline is still recorded."""
    from tensorquill.huggingface import TensorQuillCallback

    import types

    _transformers()
    model = nn.Linear(4, 1)
    callback = TensorQuillCallback(sink=tensorquill.MemorySink(), every=4)
    sink = tensorquill.MemorySink()
    callback.session = tensorquill.init(model, sink=sink, run_id="al5-hf", heartbeat_seconds=None,
                                        stack_seconds=None, start_step=3)
    control = types.SimpleNamespace(should_save=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        callback.on_evaluate(None, None, control, metrics={"eval_loss": 1.5, "epoch": 0.0})
    callback.session.close()
    assert [(r["record"], r["step"], r["scalars"]) for r in sink.records] == [("scalars", 3, {"eval/loss": 1.5})]
