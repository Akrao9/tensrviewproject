"""Lane X1 (wave X): capture rows L9, C06 (its open part) and C12 with TrainCheck B1.

Written before each fix, against the branch's base. Each test says whether it fails there
(fail-before) or guards behaviour the fix must keep (guard).

- L9: the lead optimizer was the first that ever stepped, for as long as that object lived. A new
  optimizer on the same model (an LR finder's, or a rebuilt one while the old is kept) never led,
  and its steps were not counted, silently. Now an optimizer that holds parameters the lead holds
  takes the lead when it steps in a later iteration than the lead last did, `step(closure)` included.
- C06, its open part: the length check on a router's expert ids took any whole multiple of the
  tokens, whatever `top_k` said. Now the ids must number the tokens times the declared `top_k`;
  any other count refuses the router with a reason that names both, on the default integer path
  and the `indices=` path alike. Never a raise, and the same under `torch.compile`. One exception,
  kept because `test_router_guard.py` (`int-top_k-above`) guards it: on the default integer path a
  declared `top_k` above `n_experts` is only written down, and the router is counted as before.
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


# --- C06: expert ids counted against the tokens and the declared top_k ----------------------------

EXPERTS = 4


class IdsRouter(nn.Module):
    """A router whose output is (logits, ids), with the ids the test sets (one row per token)."""

    def __init__(self, ids: list[list[int]]) -> None:
        super().__init__()
        self.register_buffer("ids", torch.tensor(ids, dtype=torch.int64))
        self.proj = nn.Linear(4, EXPERTS)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.proj(h), self.ids.clone()


class OneRouter(nn.Module):
    def __init__(self, router: nn.Module) -> None:
        super().__init__()
        self.inp = nn.Linear(4, 4)
        self.router = router

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        return h.sum() + self.router(h)[0].sum()


def _spec(route: str, top_k: int) -> tensorquill.Router:
    if route == "indices":
        return tensorquill.Router(EXPERTS, top_k, indices=lambda out: out[1])
    return tensorquill.Router(EXPERTS, top_k)


def _route(ids: list[list[int]], spec: tensorquill.Router, *, backend: str | None = None) -> list[dict]:
    torch.manual_seed(0)
    model = OneRouter(IdsRouter(ids))
    sink = tensorquill.MemorySink()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = tensorquill.init(model, sink=sink, every=1, routers={"router": spec})
        run = torch.compile(model, backend=backend, fullgraph=True) if backend else model
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(2):
            opt.zero_grad()
            run(torch.ones(len(ids), 4)).backward()
            opt.step()
        session.close()
    assert session.writer_error is None
    return sink.records


def _refused(records: list[dict]) -> str | None:
    found = [r["moe_refused"]["router"] for r in records if "router" in r.get("moe_refused", {})]
    return found[0] if found else None


TWO_PER_TOKEN = [[0, 1], [2, 3], [1, 2]]  # three tokens, two ids each


@pytest.mark.parametrize("route", ["default", "indices"])
def test_c06_two_ids_per_token_under_a_declared_top_1_refuse_the_router(route):
    # fail-before: counted as six top-1 assignments of three tokens, with moe_top_k 1.
    records = _route(TWO_PER_TOKEN, _spec(route, 1))
    why = _refused(records)
    assert why is not None, [r.get("moe") for r in records]
    assert "6 expert ids for the 3 tokens" in why and "top_k is 1" in why, why
    assert "; " not in why
    assert "tensorquill.Router(4, 2" in why, why  # the fix names the top_k the ids show
    assert not any("router" in r.get("moe", {}) for r in records)


@pytest.mark.parametrize("route", ["default", "indices"])
@pytest.mark.parametrize("top_k", [3, 4])
def test_c06_fewer_ids_per_token_than_the_declared_top_k_refuse_the_router(route, top_k):
    # fail-before: two ids a token, counted under a declared top_k of 3 or 4.
    why = _refused(_route(TWO_PER_TOKEN, _spec(route, top_k)))
    assert why is not None
    assert f"top_k is {top_k}" in why and "2 for each" in why, why


@pytest.mark.parametrize("route", ["default", "indices"])
def test_c06_one_id_per_token_under_a_declared_top_2_refuses_the_router(route):
    # fail-before: the share detectors would read top-2 limits over top-1 counts.
    why = _refused(_route([[0], [1], [3], [2]], _spec(route, 2)))
    assert why is not None and "top_k is 2" in why and "1 for each" in why, why


@pytest.mark.parametrize("route", ["default", "indices"])
@pytest.mark.parametrize("ids,top_k,counts", [
    ([[0], [1], [1], [3]], 1, [1, 2, 0, 1]),
    ([[0, 1], [2, 3], [1, 2]], 2, [1, 2, 2, 1]),
])
def test_c06_guard_ids_that_match_the_declared_top_k_are_counted(route, ids, top_k, counts):
    records = _route(ids, _spec(route, top_k))
    assert _refused(records) is None
    assert [r["moe"]["router"] for r in records] == [counts, counts]
    assert all(r["moe_top_k"] == {"router": top_k} for r in records)


def test_c06_guard_a_count_that_is_no_multiple_of_the_tokens_keeps_its_reason():
    # guard: three ids for two tokens is no whole multiple; the reason stays as it was worded.
    why = _refused(_route_flat([0, 1, 2], tokens=2))
    assert why is not None and "which is not expert indices for the 2 tokens" in why, why


class FlatIdsRouter(nn.Module):
    def __init__(self, ids: list[int]) -> None:
        super().__init__()
        self.register_buffer("ids", torch.tensor(ids, dtype=torch.int64))
        self.proj = nn.Linear(4, EXPERTS)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.proj(h), self.ids.clone()


def _route_flat(ids: list[int], tokens: int) -> list[dict]:
    torch.manual_seed(0)
    model = OneRouter(FlatIdsRouter(ids))
    sink = tensorquill.MemorySink()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with tensorquill.init(model, sink=sink, every=1, routers={"router": EXPERTS}):
            model(torch.ones(tokens, 4)).backward()
            torch.optim.SGD(model.parameters(), lr=0.01).step()
    return sink.records


def test_c06_a_mismatch_raises_nothing_under_warnings_as_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = OneRouter(IdsRouter(TWO_PER_TOKEN))
        sink = tensorquill.MemorySink()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        with tensorquill.init(model, sink=sink, every=1, routers={"router": EXPERTS}):
            for _ in range(2):
                opt.zero_grad()
                model(torch.ones(3, 4)).backward()
                opt.step()
    assert _refused(sink.records) is not None


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_c06_compiled_refuses_a_mismatch_as_eager_does(backend):
    # fail-before. Shapes only, so `fullgraph=True` holds: a graph break in the hook fails here.
    torch._dynamo.reset()
    try:
        torch.compile(OneRouter(IdsRouter(TWO_PER_TOKEN)), backend=backend, fullgraph=True)(torch.ones(3, 4))
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"this PyTorch cannot compile the model without TensorQuill: {type(error).__name__}")
    torch._dynamo.reset()
    eager = _refused(_route(TWO_PER_TOKEN, _spec("default", 1)))
    torch._dynamo.reset()
    compiled = _refused(_route(TWO_PER_TOKEN, _spec("default", 1), backend=backend))
    torch._dynamo.reset()
    counted = _route(TWO_PER_TOKEN, _spec("default", 2), backend=backend)
    torch._dynamo.reset()
    assert compiled == eager and compiled is not None
    assert _refused(counted) is None and counted[0]["moe"]["router"] == [1, 2, 2, 1]


def test_c06_guard_a_default_integer_router_under_a_top_k_above_its_experts_is_counted_as_before():
    # guard: the exception `test_router_guard.py`'s `int-top_k-above` case keeps.
    records = _route(TWO_PER_TOKEN, tensorquill.Router(EXPERTS, EXPERTS + 1))
    assert _refused(records) is None
    assert [r["moe"]["router"] for r in records] == [[1, 2, 2, 1]] * 2


def test_c06_an_indices_router_under_a_top_k_above_its_experts_is_refused():
    # fail-before: the `indices=` path has no exception.
    why = _refused(_route(TWO_PER_TOKEN, _spec("indices", EXPERTS + 1)))
    assert why is not None and f"top_k is {EXPERTS + 1}" in why, why


def test_c06_more_ids_a_token_than_experts_get_the_usual_fix_not_a_top_k_above_the_experts():
    # fail-before. Five ids a token for four experts: no Router(4, 5) would be counted on the
    # `indices=` path, so the fix offered is the usual one.
    ids = [[0, 1, 2, 3, 0], [1, 2, 3, 0, 1]]
    why = _refused(_route(ids, _spec("indices", 1)))
    assert why is not None and "5 for each" in why, why
    assert "Router(4, 5" not in why and "change its indices= function" in why, why
