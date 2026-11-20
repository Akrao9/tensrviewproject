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


# --- SDK-23: which blocks are watched by default -------------------------------------------------


class Expert(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.up, self.down = nn.Linear(width, 2 * width), nn.Linear(2 * width, width)

    def forward(self, x):
        return self.down(torch.relu(self.up(x)))


class Attn(nn.Module):
    def __init__(self, experts: int = 4) -> None:
        super().__init__()
        self.qkv = nn.Linear(8, 8)
        self.experts = nn.ModuleList([Expert() for _ in range(experts)])

    def forward(self, x):
        h = self.qkv(x)
        return x + sum(e(h) for e in self.experts)


class Mamba(nn.Module):
    def __init__(self, experts: int = 4) -> None:
        super().__init__()
        self.inp = nn.Linear(8, 8)
        self.experts = nn.ModuleList([Expert() for _ in range(experts)])

    def forward(self, x):
        h = torch.tanh(self.inp(x))
        return x + sum(e(h) for e in self.experts)


class Hybrid(nn.Module):
    """Lane T6's case: attention and state-space layers in one `ModuleList`, each with experts."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(8, 8)
        self.layers = nn.ModuleList([Attn(), Mamba(), Attn(), Mamba()])
        self.head = nn.Linear(8, 1)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h).pow(2).mean()


class EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn, self.mlp = nn.Linear(8, 8), nn.Linear(8, 8)

    def forward(self, x):
        return x + self.mlp(torch.relu(self.attn(x)))


class DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn, self.cross, self.mlp = nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8)

    def forward(self, x, memory):
        return x + self.mlp(torch.relu(self.attn(x) + self.cross(memory)))


class Stack(nn.Module):
    def __init__(self, layer: type[nn.Module], n: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([layer() for _ in range(n)])


class EncoderDecoder(nn.Module):
    """Two stacks of different layer classes; the decoder's is the larger."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(8, 8)
        self.encoder = Stack(EncoderLayer, 2)
        self.decoder = Stack(DecoderLayer, 3)
        self.head = nn.Linear(8, 1)

    def forward(self, x):
        memory = self.embed(x)
        for layer in self.encoder.layers:
            memory = layer(memory)
        h = self.embed(x)
        for layer in self.decoder.layers:
            h = layer(h, memory)
        return self.head(h).pow(2).mean()


class Plain(nn.Module):
    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.embed = nn.Linear(8, 8)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n)])
        self.heads = nn.ModuleList([nn.Linear(8, 1) for _ in range(2)])  # leaf modules: not blocks
        self.out = nn.Linear(8, 1)


class MoEStack(nn.Module):
    """One layer class, each layer holding a list of experts: the layers are the blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Attn(experts=6) for _ in range(3)])


class Stage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([EncoderLayer() for _ in range(2)])


class Staged(nn.Module):
    """Swin-like: a list of stages, each a list of blocks. The stages are the outermost list."""

    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.ModuleList([Stage() for _ in range(3)])


def _names(model):
    return [name for name, _ in _block_layers(model)]


def test_sdk23_a_stack_of_two_block_classes_is_watched_block_by_block():
    assert _names(Hybrid()) == ["layers.0", "layers.1", "layers.2", "layers.3"]


def test_sdk23_both_stacks_of_an_encoder_decoder_are_watched():
    assert _names(EncoderDecoder()) == [
        "encoder.layers.0", "encoder.layers.1", "decoder.layers.0", "decoder.layers.1", "decoder.layers.2",
    ]


def test_sdk23_a_block_in_two_lists_is_watched_once():
    class Shared(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Stack(EncoderLayer, 2)
            self.decoder = nn.ModuleList([self.encoder.layers[0], DecoderLayer(), DecoderLayer()])

    model = Shared()
    blocks = _block_layers(model)
    assert [name for name, _ in blocks] == ["encoder.layers.0", "encoder.layers.1", "decoder.1", "decoder.2"]
    assert len({id(module) for _, module in blocks}) == len(blocks)


def test_sdk23_a_list_inside_a_watched_block_is_not_watched_as_well():
    # Without the expert lists left out, every expert of every layer would be watched beside it.
    assert _names(MoEStack()) == ["layers.0", "layers.1", "layers.2"]
    assert _names(Staged()) == ["stages.0", "stages.1", "stages.2"]


@pytest.mark.parametrize(
    ("make", "names"),
    [
        (Plain, ["layers.0", "layers.1", "layers.2"]),
        (MoEStack, ["layers.0", "layers.1", "layers.2"]),
        (Staged, ["stages.0", "stages.1", "stages.2"]),
        (lambda: nn.ModuleList([EncoderLayer() for _ in range(3)]), ["0", "1", "2"]),
        (lambda: Stack(EncoderLayer, 1), []),  # one block is no repeated block
        (lambda: nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1)), []),
    ],
    ids=["plain", "experts-inside", "stages", "root-list", "one-block", "no-list"],
)
def test_sdk23_a_single_class_stack_is_watched_as_before(make, names):
    assert _names(make()) == names


def test_sdk23_a_list_mixing_blocks_and_leaf_modules_is_not_taken_for_blocks():
    class Mixed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([EncoderLayer(), nn.LayerNorm(8), EncoderLayer()])

    assert _names(Mixed()) == []


@pytest.mark.parametrize(
    ("make", "names"),
    [
        (Hybrid, ["layers.0", "layers.1", "layers.2", "layers.3"]),
        (EncoderDecoder, ["encoder.layers.0", "encoder.layers.1", "decoder.layers.0", "decoder.layers.1", "decoder.layers.2"]),
    ],
    ids=["hybrid", "encoder-decoder"],
)
def test_sdk23_a_session_records_each_block_once(tmp_path, make, names):
    torch.manual_seed(0)
    model = make()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, run_id="blocks", sink=sink, every=1)
    assert session.layer_names == names
    # One forward hook a block, and nothing hooked on a module inside one.
    hooked = [name for name, module in model.named_modules() if module._forward_hooks]
    assert hooked == names
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.randn(4, 8)).backward()
        optimizer.step()
    session.close()
    assert sink.records and all(list(r["layers"]) == names for r in sink.records)
    assert all(r["layers"][name]["std"] > 0 for r in sink.records for name in names)
