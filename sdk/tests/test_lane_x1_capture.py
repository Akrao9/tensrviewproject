"""Lane X1 (wave X): capture rows L9, C06 (its open part) and C12 with TrainCheck B1.

Written before each fix, against the branch's base. Each test says whether it fails there
(fail-before) or guards behaviour the fix must keep (guard).

- L9: the lead optimizer was the first that ever stepped, for as long as that object lived. A new
  optimizer on the same model (an LR finder's, or a rebuilt one while the old is kept) never led,
  and its steps were not counted, silently. Now an optimizer that holds parameters the lead holds
  takes the lead when it steps in a later iteration than the lead last did, `step(closure)` included.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import tensorquill


def _steps(records: list[dict]) -> list[int]:
    return [r["step"] for r in records if "step" in r]


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(8, 4)
    return x, x[:, :1] * 2


def _iterate(model: nn.Module, opt: torch.optim.Optimizer, n: int, *, set_to_none: bool = True) -> None:
    for _ in range(n):
        x, y = _batch()
        opt.zero_grad(set_to_none=set_to_none)
        (model(x) - y).pow(2).mean().backward()
        opt.step()


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))


# --- L9: a replaced lead optimizer ----------------------------------------------------------------


@pytest.mark.parametrize("set_to_none", [True, False])
def test_l9_a_rebuilt_optimizer_whose_predecessor_stays_alive_ends_steps(set_to_none):
    # fail-before: only the first optimizer's 3 steps were recorded.
    model = _model()
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, every=4):
        first = torch.optim.SGD(model.parameters(), lr=0.1)
        _iterate(model, first, 3, set_to_none=set_to_none)
        second = torch.optim.SGD(model.parameters(), lr=0.01)  # `first` is still referenced
        _iterate(model, second, 4, set_to_none=set_to_none)
    assert _steps(sink.records) == list(range(7))
    assert [r["lr"] for r in sink.records] == [[0.1]] * 3 + [[0.01]] * 4
    assert first is not None


def test_l9_an_lr_finder_then_the_real_optimizer_records_every_step_with_its_updates():
    # fail-before: the real optimizer's steps were folded into the finder's last record.
    model = _model()
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, every=3, updates=True, weights=1):
        finder = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for i in range(5):
            for group in finder.param_groups:
                group["lr"] = 1e-4 * 10**i
            _iterate(model, finder, 1)
        real = torch.optim.AdamW(model.parameters(), lr=1e-3)
        _iterate(model, real, 6)
    records = sink.records
    assert _steps(records) == list(range(11))
    assert [r["lr"] for r in records[5:]] == [[1e-3]] * 6
    names = {name for name, _ in model.named_parameters()}
    assert all(set(r["update_rms"]) == names for r in records)
    assert finder is not None and real is not None


def test_l9_under_a_scaler_the_new_lead_gives_the_learning_rate():
    # fail-before: `scaler.update()` ended each step, but the lr was read from the old optimizer.
    model = _model()
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0**8)
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, every=4, scaler=scaler):
        first = torch.optim.SGD(model.parameters(), lr=0.1)
        second = torch.optim.SGD(model.parameters(), lr=0.01)
        for i in range(5):
            opt = first if i < 2 else second
            x, y = _batch()
            opt.zero_grad()
            scaler.scale((model(x) - y).pow(2).mean()).backward()
            scaler.step(opt)
            scaler.update()
    assert _steps(sink.records) == list(range(5))
    assert [r["lr"] for r in sink.records] == [[0.1]] * 2 + [[0.01]] * 3


def test_l9_two_optimizers_used_in_turn_by_phase_each_lead_in_their_phase():
    # fail-before: the second phase's steps were not counted.
    model = _model()
    sink = tensorquill.MemorySink()
    a = torch.optim.SGD(model.parameters(), lr=0.1)
    b = torch.optim.SGD(model.parameters(), lr=0.01)
    with tensorquill.init(model, sink=sink, every=5):
        _iterate(model, a, 2)
        _iterate(model, b, 2)
        _iterate(model, a, 2)
    assert _steps(sink.records) == list(range(6))
    assert [r["lr"] for r in sink.records] == [[0.1]] * 2 + [[0.01]] * 2 + [[0.1]] * 2


def test_l9_a_rebuilt_optimizer_stepped_with_a_closure_ends_steps_from_its_first():
    # fail-before (3 steps then). With `step(closure)` the backward runs inside the step, so whether
    # the new optimizer replaces the lead is judged again after its step.
    model = _model()
    sink = tensorquill.MemorySink()

    def run(opt: torch.optim.Optimizer, n: int) -> None:
        for _ in range(n):
            x, y = _batch()

            def closure() -> torch.Tensor:
                opt.zero_grad()
                loss = (model(x) - y).pow(2).mean()
                loss.backward()
                return loss

            opt.step(closure)

    with tensorquill.init(model, sink=sink, every=4):
        first = torch.optim.SGD(model.parameters(), lr=0.1)
        run(first, 3)
        second = torch.optim.SGD(model.parameters(), lr=0.01)
        run(second, 4)
    assert _steps(sink.records) == list(range(7))
    assert [r["lr"] for r in sink.records] == [[0.1]] * 3 + [[0.01]] * 4
    assert first is not None


def test_l9_guard_two_optimizers_on_the_same_parameters_in_one_iteration_stay_one_step():
    # guard: no backward between the two steps, so the second follows the first, as before.
    model = _model()
    a = torch.optim.SGD(model.parameters(), lr=0.1)
    b = torch.optim.SGD(model.parameters(), lr=0.01)
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, every=3, updates=False):
        for _ in range(5):
            x, y = _batch()
            a.zero_grad()
            (model(x) - y).pow(2).mean().backward()
            a.step()
            b.step()
    assert _steps(sink.records) == list(range(5))
    assert all(r["lr"] == [0.1] for r in sink.records)


def test_l9_guard_a_follower_on_other_parameters_that_steps_more_often_never_leads():
    # guard: a critic stepped several times per generator step, the generator stepping first. They
    # share no parameter, so the critic follows, as before: one step per generator step.
    torch.manual_seed(0)
    model = nn.ModuleDict({"gen": nn.Linear(4, 4), "critic": nn.Linear(4, 1)})
    gen = torch.optim.SGD(model["gen"].parameters(), lr=0.1)
    critic = torch.optim.SGD(model["critic"].parameters(), lr=0.01)
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, every=4, layers=()):
        for _ in range(4):
            gen.zero_grad()
            (-model["critic"](model["gen"](torch.randn(8, 4)))).mean().backward()
            gen.step()
            for _ in range(3):
                critic.zero_grad()
                model["critic"](model["gen"](torch.randn(8, 4)).detach()).mean().backward()
                critic.step()
    assert _steps(sink.records) == list(range(4))
    assert all(r["lr"] == [0.1] for r in sink.records)


def test_l9_guard_nothing_raises_under_warnings_as_errors():
    model = _model()
    sink = tensorquill.MemorySink()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with tensorquill.init(model, sink=sink, every=2, updates=True):
            first = torch.optim.AdamW(model.parameters(), lr=1e-3)
            _iterate(model, first, 2)
            second = torch.optim.AdamW(model.parameters(), lr=1e-4)
            _iterate(model, second, 3)
    assert _steps(sink.records) == list(range(5))
    assert first is not None
