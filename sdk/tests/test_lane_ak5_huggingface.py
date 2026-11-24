"""Lane AK-5: the Hugging Face callback records the `Trainer`'s evaluation metrics.

Read in Transformers (`trainer.py`, `integrations/integration_utils.py`): `Trainer.evaluate` logs
its metrics through `Trainer.log` (`on_log`, where `epoch` is added) and then hands the same
dictionary to `on_evaluate(..., metrics=...)`. Its keys carry the prefix (`eval_loss`,
`eval_<name>` from `compute_metrics`, and with several evaluation datasets `eval_<dataset>_loss`),
already plain numbers, beside bookkeeping (`epoch`, `eval_runtime`, `eval_samples_per_second`,
`eval_steps_per_second`). Its own loggers rename them (`rewrite_logs`: `eval_x` to `eval/x`,
`test_x` to `test/x`, anything else to `train/x`). The callback passes the numeric evaluation
metrics, without the bookkeeping, under those names while the phase is `"eval"`, so they land on
the step the evaluation followed; a value the mirror also brings from a W&B logger is recorded once.
"""

from __future__ import annotations

import sys
import types
import warnings

import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")

import tensorquill  # noqa: E402
from tensorquill.huggingface import TensorQuillCallback  # noqa: E402

from test_lane_ak1_mirror import wandb_stand_in  # noqa: E402


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        self.train_losses = []

    def forward(self, x, labels=None):
        loss = (self.net(x).squeeze(-1) - labels).pow(2).mean()
        if self.training:
            self.train_losses.append(loss.item())
        return {"loss": loss}


def _data(n=64, seed=0):
    torch.manual_seed(seed)
    return [{"x": torch.randn(4), "labels": torch.randn(())} for _ in range(n)]


def _trainer(tmp_path, model, callbacks, **kwargs):
    options = dict(
        output_dir=str(tmp_path), max_steps=6, per_device_train_batch_size=8, per_device_eval_batch_size=16,
        report_to="none", save_strategy="no", disable_tqdm=True, use_cpu=True, learning_rate=0.01,
        logging_steps=2, eval_strategy="steps", eval_steps=2,
    )
    options.update(kwargs)
    eval_dataset = options.pop("eval_dataset", _data(32, seed=1))
    args = transformers.TrainingArguments(**options)
    return transformers.Trainer(model=model, args=args, train_dataset=_data(), eval_dataset=eval_dataset,
                                callbacks=callbacks)


def _evaluations(trainer) -> dict[int, dict]:
    """The `Trainer`'s own evaluation entries by TensorQuill step (its `global_step` less one)."""
    return {entry["step"] - 1: entry for entry in trainer.state.log_history if "eval_loss" in entry}


@pytest.fixture(autouse=True)
def no_trackers(monkeypatch):
    for name in ("wandb", "mlflow", "mlflow.tracking", "mlflow.tracking.fluent", "wandb.sdk", "wandb.sdk.wandb_run"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    yield


def test_each_evaluation_lands_on_the_step_it_followed_final_one_included(tmp_path):
    sink = tensorquill.MemorySink()
    model = Model()
    trainer = _trainer(tmp_path, model, [TensorQuillCallback(sink=sink, every=4)])
    trainer.train()
    records = {r["step"]: r["scalars"] for r in sink.records}
    evaluations = _evaluations(trainer)
    # every=4: step 1 is in the ring, step 3 kept back when the ring filled, step 5 the last step.
    assert sorted(evaluations) == [1, 3, 5]
    for step, scalars in records.items():
        if step in evaluations:
            assert scalars["eval/loss"] == pytest.approx(evaluations[step]["eval_loss"])
        else:
            assert "eval/loss" not in scalars
        # Bookkeeping is not an evaluation metric; the training loss is the step's own.
        assert not {"epoch", "eval/epoch", "train/epoch", "eval/runtime", "eval/samples_per_second",
                    "eval/steps_per_second"} & set(scalars)
    assert [records[s]["train/loss"] for s in range(6)] == pytest.approx(model.train_losses, rel=1e-5)


def test_compute_metrics_and_several_evaluation_datasets(tmp_path):
    sink = tensorquill.MemorySink()

    def compute_metrics(prediction):
        return {"accuracy": 0.5, "note": "text is not a scalar", "flag": True}

    trainer = _trainer(tmp_path, Model(), [TensorQuillCallback(sink=sink, every=4)], max_steps=2,
                       eval_dataset={"a": _data(16, seed=1), "b": _data(16, seed=2)})
    trainer.compute_metrics = compute_metrics
    trainer.train()
    scalars = {r["step"]: r["scalars"] for r in sink.records}[1]
    assert {"eval/a_loss", "eval/b_loss", "eval/a_accuracy", "eval/b_accuracy"} <= set(scalars)
    assert scalars["eval/a_accuracy"] == 0.5
    assert not {"eval/a_note", "eval/a_flag", "eval/b_note", "eval/b_flag"} & set(scalars)


class _FakeWandbLogger(transformers.TrainerCallback):
    """What `WandbCallback.on_log` does (`integration_utils.py`), on the stand-in `wandb` module."""

    def __init__(self, wandb):
        self.wandb = wandb

    def on_log(self, args, state, control, logs=None, **kwargs):
        from transformers.integrations.integration_utils import rewrite_logs

        single = ["train_runtime", "train_samples_per_second", "train_steps_per_second", "train_loss", "total_flos"]
        self.wandb.log({**rewrite_logs({k: v for k, v in logs.items() if k not in single}),
                        "train/global_step": state.global_step})


def test_with_a_wandb_logger_each_value_is_recorded_once_and_the_explicit_one_wins(tmp_path, monkeypatch):
    wandb = wandb_stand_in(lambda name, module: monkeypatch.setitem(sys.modules, name, module))
    wandb.module.init()
    sink = tensorquill.MemorySink()
    model = Model()
    # The reporting callbacks come before the user's (`Trainer.__init__`: default + reporting + given).
    trainer = _trainer(tmp_path, model, [_FakeWandbLogger(wandb.module), TensorQuillCallback(sink=sink, every=4)])
    trainer.train()
    records = {r["step"]: r["scalars"] for r in sink.records}
    evaluations = _evaluations(trainer)
    for step in evaluations:
        assert records[step]["eval/loss"] == pytest.approx(evaluations[step]["eval_loss"])
        assert records[step]["train/global_step"] == step + 1  # mirrored onto the evaluated step
    # W&B's `train/loss` is the mean over `logging_steps`, logged in the evaluation's phase: it does
    # not replace the per-step loss recorded before the step ended.
    assert [records[s]["train/loss"] for s in range(6)] == pytest.approx(model.train_losses, rel=1e-5)
    assert any(entry.get("eval/loss") is not None for entry in wandb.values())


def test_a_baseline_evaluation_is_left_out_with_one_warning(tmp_path):
    sink = tensorquill.MemorySink()
    trainer = _trainer(tmp_path, Model(), [TensorQuillCallback(sink=sink, every=4)], max_steps=2,
                       eval_on_start=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer.train()
    said = [w for w in caught if "protocol 5" in str(w.message)]
    assert len(said) == 1
    records = {r["step"]: r["scalars"] for r in sink.records}
    assert "eval/loss" not in records[0] and "eval/loss" in records[1]


def test_a_phase_left_behind_is_set_back_when_the_next_step_begins(tmp_path):
    class LeavesEval(transformers.TrainerCallback):
        def __init__(self):
            self.tq = None

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 2:
                self.tq.session.set_phase("eval")  # announced, and no evaluation follows

    sink = tensorquill.MemorySink()
    model = Model()
    tq = TensorQuillCallback(sink=sink, every=4)
    leaves = LeavesEval()
    leaves.tq = tq
    trainer = _trainer(tmp_path, model, [tq, leaves], eval_strategy="no")
    trainer.train()
    records = {r["step"]: r["scalars"] for r in sink.records}
    assert [records[s]["train/loss"] for s in range(6)] == pytest.approx(model.train_losses, rel=1e-5)


def test_on_evaluate_never_raises_into_the_trainer():
    model = nn.Linear(4, 1)
    callback = TensorQuillCallback(sink=tensorquill.MemorySink(), every=4)
    callback.session = tensorquill.init(model, sink=tensorquill.MemorySink(), run_id="ak5-hf", heartbeat_seconds=None,
                                        stack_seconds=None)
    control = types.SimpleNamespace(should_save=False)
    odd = {"eval_ok": 1.0, "eval_text": "x", "eval_list": [1, 2], "eval_none": None, 3: 1.0, "": 2.0,
           "eval_nan": float("nan"), "eval_big": 10**400, "eval_tensor": torch.ones(3)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        callback.on_evaluate(None, None, control, metrics=odd)
        callback.on_evaluate(None, None, control, metrics="not a mapping")
        callback.on_evaluate(None, None, control)
    assert callback.session._phase == "train"
    callback.session.close()
