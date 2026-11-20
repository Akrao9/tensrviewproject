"""Lane Y1 (capture): C06's last case and SDK-23.

C06: on the default path, where a router's first integer tensor is taken as the chosen expert
indices, a declared `top_k` above `n_experts` was counted (only written down as `moe_top_k`). No
router chooses more distinct experts per token than it has, and the share detectors judge an
expert's share against `1 / top_k`, so the router is now refused with the reason the logits path
already gives, and training goes on untouched. A valid `top_k` counts as before.

SDK-23: the default blocks were the children of the one `ModuleList` with the most parameters whose
composite children share one class. A stack that mixes two block classes (a hybrid of attention and
state-space layers) was skipped for a smaller list inside one of its blocks, and an encoder-decoder
kept only its larger stack. Now every outermost `ModuleList` of composite blocks is watched, each
block once; a list inside a block already watched is not, and a plain single-class stack is
watched as before.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn
from test_router_guard import COMPILED, EXPERTS, TOP_K, Emits, _refused, _routed, _run

import tensorquill
from tensorquill.capture import Router, _block_layers

ABOVE_FIX = (
    "To count it, pass routers={'router': tensorquill.Router(n_experts, top_k)} with its number of experts "
    "and a top_k of at most that number"
)


# --- C06: a declared top_k above the experts on the default integer path ---------------------------


@COMPILED
@pytest.mark.parametrize(
    "make",
    [
        lambda logits: logits.topk(TOP_K, dim=-1).indices,
        # As many ids a token as the declared top_k, so the count check alone would pass them.
        lambda logits: torch.cat([logits.topk(EXPERTS, dim=-1).indices, logits.topk(4, dim=-1).indices], dim=-1),
    ],
    ids=["two-ids-a-token", "twelve-ids-a-token"],
)
def test_c06_an_integer_router_whose_top_k_is_above_its_experts_is_refused_with_the_reason(
    tmp_path, make, compile_backend
):
    why = _refused(tmp_path, Emits(make), "router", Router(EXPERTS, 12), compile_backend)
    assert why == f"its top_k is 12 but it has {EXPERTS} experts. {ABOVE_FIX}"


def _train(tmp_path, routers, steps=4, watch=True):
    """The final weights and the losses of four steps of `_run`'s model, watched or not."""
    from test_router_guard import Model

    torch.manual_seed(0)
    model = Model(Emits(lambda logits: logits.topk(TOP_K, dim=-1).indices))
    session = None
    if watch:
        session = tensorquill.init(
            model, run_id="untouched", sink=tensorquill.JsonlSink(tmp_path / "u.jsonl"), routers=routers, every=steps
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model(torch.randn(6, 20, 4))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if session is not None:
        session.close()
    return [p.detach().clone() for p in model.parameters()], losses


def test_c06_the_refusal_leaves_training_untouched_and_never_raises(tmp_path):
    with warnings.catch_warnings():
        # As under `python -W error`: the session's warning must not raise into training.
        warnings.simplefilter("error", RuntimeWarning)
        watched, watched_losses = _train(tmp_path, {"router": Router(EXPERTS, 12)})
    plain, plain_losses = _train(tmp_path, {}, watch=False)
    assert watched_losses == plain_losses
    assert all(torch.equal(a, b) for a, b in zip(watched, plain))


@COMPILED
@pytest.mark.parametrize(
    ("make", "spec", "routed"),
    [
        (lambda logits: logits.topk(TOP_K, dim=-1).indices, Router(EXPERTS, TOP_K), 120 * TOP_K),
        (lambda logits: logits.argmax(dim=-1), Router(EXPERTS), 120),
        # top_k equal to the experts is a valid top_k: every expert for every token.
        (lambda logits: logits.topk(EXPERTS, dim=-1).indices, Router(EXPERTS, EXPERTS), 120 * EXPERTS),
    ],
    ids=["top-2", "top-1", "top_k-equal"],
)
def test_c06_a_valid_top_k_is_counted_as_before(tmp_path, make, spec, routed, compile_backend):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        records = _run(tmp_path, Emits(make), {"router": spec}, compile_backend=compile_backend)
    assert _routed(records) == [routed] * 4 and not any("moe_refused" in r for r in records)
    assert all(r["moe_top_k"]["router"] == spec.top_k for r in records)
    assert all(sum(r["moe"]["router"]) == routed for r in records)
