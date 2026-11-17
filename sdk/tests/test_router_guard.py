"""A router whose output's first integer tensor cannot be expert indices is not counted.

By default the capture takes the first integer tensor in a router's output as the chosen expert
indices. nanoMoE's `Router` returns `(used_capacity, weights, mask)`, and `used_capacity` is an
integer tensor of per-expert token counts: read as indices it gives wrong loads and no error.
The shapes show it (eight numbers for hundreds of tokens), so the hook leaves that router
uncounted and the session says so once.
"""

from __future__ import annotations

import json
import warnings

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill.capture import Router

EXPERTS, TOP_K = 8, 2


class CountsFirstRouter(nn.Module):
    """As nanoMoE's: per-expert counts, then weights, then a mask. No indices in the output."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.w_g = nn.Linear(width, EXPERTS, bias=False)

    def forward(self, x):
        logits = self.w_g(x)
        chosen = logits.topk(TOP_K, dim=-1).indices
        mask = torch.nn.functional.one_hot(chosen, EXPERTS).sum(-2)  # [B, T, EXPERTS]
        used_capacity = mask.sum(dim=(0, 1))  # [EXPERTS], integer counts
        return used_capacity, logits.softmax(-1) * mask, mask.bool()


class IndexRouter(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.w_g = nn.Linear(width, EXPERTS, bias=False)

    def forward(self, x):
        return self.w_g(x).topk(TOP_K, dim=-1).indices  # [B, T, TOP_K]


class Model(nn.Module):
    def __init__(self, router: nn.Module) -> None:
        super().__init__()
        self.embed = nn.Linear(4, 16)
        self.router = router
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        hidden = self.embed(x)
        self.router(hidden)
        return self.head(hidden).pow(2).mean()


def _run(tmp_path, router, routers, steps=4, compile_backend=None):
    torch.manual_seed(0)
    model = Model(router)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "guard.jsonl"
    session = tensorquill.init(model, run_id="guard", sink=tensorquill.JsonlSink(path), routers=routers, every=steps)
    stepped = torch.compile(model, backend=compile_backend) if compile_backend else model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad()
        stepped(torch.randn(6, 20, 4)).backward()  # 120 tokens
        optimizer.step()
    session.close()
    return [json.loads(line) for line in path.read_text().splitlines()]


def _routed(records, name="router"):
    return [r["moe_stats"][name]["routed_tokens"] for r in records if "moe_stats" in r]


def test_a_router_that_returns_counts_first_is_not_counted_and_is_said_once(tmp_path):
    with pytest.warns(RuntimeWarning, match="not counting the router 'router'") as caught:
        records = _run(tmp_path, CountsFirstRouter(16), {"router": Router(EXPERTS, TOP_K)})
    said = [w for w in caught if "not counting the router" in str(w.message)]
    assert len(said) == 1  # once, not at every step
    assert "shape (8,)" in str(said[0].message) and "120 tokens" in str(said[0].message)
    # Nothing was counted: not the eight counts read as indices.
    assert _routed(records) and set(_routed(records)) == {0}
    assert not [i for i in tensorquill.build_evidence(records)["incidents"] if i["kind"].startswith(("moe", "starved"))]


def test_the_same_router_is_counted_through_its_inner_logits_or_an_indices_function(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        inner = _run(tmp_path / "a", CountsFirstRouter(16), {"router.w_g": Router(EXPERTS, TOP_K)})
        given = _run(tmp_path / "b", CountsFirstRouter(16),
                     {"router": Router(EXPERTS, TOP_K, indices=lambda out: out[2].nonzero()[:, -1])})
    assert set(_routed(inner, "router.w_g")) == {120 * TOP_K}
    assert set(_routed(given)) == {120 * TOP_K}


def test_a_router_that_returns_indices_is_counted_as_before(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        records = _run(tmp_path, IndexRouter(16), {"router": Router(EXPERTS, TOP_K)})
    assert set(_routed(records)) == {120 * TOP_K}


def test_under_torch_compile_the_guard_says_the_same_and_counts_nothing(tmp_path):
    with pytest.warns(RuntimeWarning, match="not counting the router 'router'"):
        records = _run(tmp_path, CountsFirstRouter(16), {"router": Router(EXPERTS, TOP_K)}, compile_backend="eager")
    assert _routed(records) and set(_routed(records)) == {0}
