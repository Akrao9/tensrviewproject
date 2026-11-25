"""Lane AK-5 and the scalar mirror: on the newest step, an explicit late value still wins.

Lane AK-1's rule (`mirror._log_into`): a key the session got from `session.log` itself is never
replaced by a mirrored one, and a key mirrored twice keeps its last value. Under `phase("eval")` a
value now goes to the newest step still in memory (lane AK-5), so the same must hold there, in
either order of arrival, whether that step is the ring's last row or the one kept back when the
ring filled (`every=1` and `every=2`), and a key the step already held from its own training is not
replaced by a mirrored one either. Run against the stand-in trackers of `test_lane_ak1_mirror.py`.
"""

from __future__ import annotations

import sys
import warnings

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill.sink import MemorySink

from test_lane_ak1_mirror import mlflow_stand_in, wandb_stand_in  # noqa: E402


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


@pytest.fixture
def wandb_(monkeypatch):
    stand_in = wandb_stand_in(lambda name, module: monkeypatch.setitem(sys.modules, name, module))
    stand_in.module.init()
    return stand_in


@pytest.fixture
def mlflow_(monkeypatch):
    return mlflow_stand_in(lambda name, module: monkeypatch.setitem(sys.modules, name, module))


def _open(every: int):
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    session = tensorquill.init(model, sink=MemorySink(), run_id="ak5", layers=(), grads=False, weights=False,
                               heartbeat_seconds=None, stack_seconds=None, every=every)
    return session, model, torch.optim.SGD(model.parameters(), lr=0.01)


def _step(session, model, opt, loss: float) -> None:
    session.log({"train/loss": loss})
    model(torch.ones(2, 4)).sum().backward()
    opt.step()
    opt.zero_grad()


def _scalars(session) -> dict[int, dict]:
    return {r["step"]: r.get("scalars", {}) for r in session._sink.records}


# every=4 with two steps: the newest step is the ring's last row. every=2: it is kept back (held).
# every=1: kept back at once.
SHAPES = [(4, 2), (2, 2), (1, 2)]


@pytest.mark.parametrize("every,steps", SHAPES)
@pytest.mark.parametrize("order", ["explicit_first", "mirrored_first"])
def test_an_explicit_late_value_wins_over_a_mirrored_one_in_either_order(wandb_, every, steps, order):
    session, model, opt = _open(every)
    for i in range(steps):
        _step(session, model, opt, float(i))
    with session.phase("eval"):
        if order == "explicit_first":
            session.log({"eval/loss": 1.0})
            wandb_.module.log({"eval/loss": 9.0, "eval/acc": 0.5})
        else:
            wandb_.module.log({"eval/loss": 9.0, "eval/acc": 0.5})
            session.log({"eval/loss": 1.0})
        wandb_.module.log({"eval/acc": 0.75})  # mirrored twice: the last value
    session.close()
    newest = _scalars(session)[steps - 1]
    assert newest["eval/loss"] == 1.0 and newest["eval/acc"] == 0.75
    assert newest["train/loss"] == float(steps - 1)
    assert all("eval/loss" not in s for step, s in _scalars(session).items() if step != steps - 1)


@pytest.mark.parametrize("every,steps", SHAPES)
def test_a_mirrored_late_value_never_replaces_the_steps_own_value(wandb_, mlflow_, every, steps):
    """The Hugging Face `Trainer` logs its averaged training loss through `on_log` after the
    callback has set `"eval"` for an evaluation on the same step: W&B's `train/loss` must not
    replace the per-step loss the session recorded before the step ended."""
    session, model, opt = _open(every)
    for i in range(steps):
        _step(session, model, opt, float(i))
    with session.phase("eval"):
        wandb_.module.log({"train/loss": 42.0, "train/learning_rate": 0.01})
        mlflow_.module.log_metric("train/loss", 43.0)
    session.close()
    newest = _scalars(session)[steps - 1]
    assert newest["train/loss"] == float(steps - 1) and newest["train/learning_rate"] == 0.01


def test_a_mirrored_key_after_an_evaluation_still_counts_as_mirrored_on_the_next_step(wandb_):
    """The mirror's record of what it stored is kept per step: an evaluation in between does not make
    a value it mirrored into the step in progress look like the caller's."""
    session, model, opt = _open(4)
    _step(session, model, opt, 0.0)
    wandb_.module.log({"lr": 0.1})  # after step 0, in "train": step 1's
    with session.phase("eval"):
        wandb_.module.log({"eval/loss": 2.0})
    wandb_.module.log({"lr": 0.2})  # mirrored twice on step 1: the last value
    _step(session, model, opt, 1.0)
    session.close()
    scalars = _scalars(session)
    assert scalars[0]["eval/loss"] == 2.0 and scalars[1]["lr"] == 0.2


@pytest.mark.parametrize("every", [4, 2])
def test_a_key_mirrored_into_a_step_and_mirrored_again_after_it_keeps_the_last_value(wandb_, every):
    """A Hugging Face W&B logger logs `train/global_step` after every logged step: one into the step in
    progress, the next under the evaluation's phase onto that same step once it ended. Both are the
    mirror's own, so the last wins, as for a key mirrored twice in a step."""
    session, model, opt = _open(every)
    _step(session, model, opt, 0.0)
    wandb_.module.log({"lr": 0.1})  # step 1's, in "train"
    session.log({"tokens": 8})  # the caller's own: kept
    _step(session, model, opt, 1.0)
    with session.phase("eval"):
        wandb_.module.log({"lr": 0.2, "tokens": 99})
    session.close()
    assert _scalars(session)[1]["lr"] == 0.2 and _scalars(session)[1]["tokens"] == 8


def test_the_mirror_never_trips_the_step_refusal(wandb_, mlflow_):
    session, model, opt = _open(4)
    _step(session, model, opt, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert wandb_.module.log({"step": 5, "a": 1.0}, step=7) is wandb_.returned
        assert mlflow_.module.log_metrics({"b": 2.0}, step=9) is mlflow_.returned
        with session.phase("eval"):
            wandb_.module.log({"eval/acc": 0.5}, step=11, commit=True)
            mlflow_.module.log_metric("eval/b", 3.0, step=12)
    _step(session, model, opt, 1.0)
    session.close()
    scalars = _scalars(session)
    assert scalars[0]["eval/acc"] == 0.5 and scalars[0]["eval/b"] == 3.0
    assert scalars[1]["step"] == 5 and scalars[1]["a"] == 1.0 and scalars[1]["b"] == 2.0


def test_a_mirrored_value_with_no_step_in_memory_is_left_out_and_the_tracker_still_gets_it(wandb_):
    session, model, opt = _open(4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with session.phase("eval"):
            assert wandb_.module.log({"eval/baseline": 2.5}) is wandb_.returned
        _step(session, model, opt, 0.0)
        session.close()
    assert wandb_.values() == [{"eval/baseline": 2.5}]
    assert "eval/baseline" not in _scalars(session)[0]
    # Wave AL (section Q5): the warning shows the explicit call, and says the mirror cannot pass a step.
    said = [str(w.message) for w in caught if "left out the scalar" in str(w.message)]
    assert len(said) == 1
    assert "(a baseline evaluation)" in said[0] and "through mirror_scalars, which cannot pass a step" in said[0]
