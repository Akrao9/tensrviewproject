"""Lane SR2: `routers="auto"`, router auto-detection from a registry, opt-in.

`init(model, routers="auto")` looks the model's class up, by its module and qualified name exactly,
in `tensorquill.router_registry`, and from the matched entry and the model's configuration
works out the router modules, `n_experts` and `top_k` before any hook is registered. There is
no scan that guesses: a class not in the registry finds no router and says so (once, and under
`moe_refused`, the key that already names a router not counted); an entry that does not fit the
model in hand is refused by name under `moe_refused`, never counted as zeros. A run that does not
pass `"auto"` never reaches the registry.

The tests on stand-in classes (plain modules given a registered class's module and name, in the
layouts transformers 4.46 to 4.57 and 5.0 to 5.12 have) need no `transformers` and run in the
plain CI jobs. The tests on tiny Hugging Face models with random weights need `transformers` and
are skipped without it.
"""

from __future__ import annotations

import logging
import sys
import types
import warnings

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import router_registry
from tensorquill.evidence import MOE_KINDS, build_evidence

H, E, K, LAYERS = 16, 8, 2, 3
BATCH, SEQ = 2, 6
TOKENS = BATCH * SEQ


def _named(cls: type, qualified: str) -> type:
    """`cls` given the module and qualified name of a class in the registry, as the registry
    reads a class: by those two names only."""
    module, _, name = qualified.rpartition(".")
    cls.__module__, cls.__qualname__, cls.__name__ = module, name, name
    return cls


MIXTRAL = "transformers.models.mixtral.modeling_mixtral"
DEEPSEEK_V3 = "transformers.models.deepseek_v3.modeling_deepseek_v3"


class _Block4x(nn.Module):
    """The routing of transformers 4.46 to 4.57's `MixtralSparseMoeBlock`, as read at those tags: an
    `nn.Linear` gate whose logits the block takes the softmax of, then the top-k. The experts are one
    linear layer each. `chosen` keeps what the block itself chose, for the test to compare."""

    def __init__(self, top_k: int = K) -> None:
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(H, E, bias=False)
        self.experts = nn.ModuleList(nn.Linear(H, H) for _ in range(E))
        self.chosen: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, H)
        logits = self.gate(flat)
        weights, selected = torch.topk(torch.softmax(logits, dim=-1, dtype=torch.float), self.top_k, dim=-1)
        if not torch.compiler.is_compiling():
            self.chosen.append(selected.detach().clone())
        # Every expert on every token, weighted by the routing (zero where not chosen): the same output
        # as a dispatch, with no data-dependent shape, so the block also compiles with fullgraph.
        dense = torch.zeros(flat.shape[0], logits.shape[-1], dtype=weights.dtype).scatter_add(1, selected, weights)
        out = sum(expert(flat) * dense[:, e : e + 1].to(flat.dtype) for e, expert in enumerate(self.experts))
        return out.reshape(x.shape)


_named(_Block4x, f"{MIXTRAL}.MixtralSparseMoeBlock")


class _Layer4x(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_sparse_moe = _Block4x()

    def forward(self, x):
        return x + self.block_sparse_moe(x)


class _Base(nn.Module):
    def __init__(self, layer: type, layers: int = LAYERS) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layer() for _ in range(layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _ForCausalLM(nn.Module):
    def __init__(self, layer: type, config: types.SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.model = _Base(layer, config.num_hidden_layers)
        self.head = nn.Linear(H, 1)

    def forward(self, x):
        return self.head(self.model(x)).pow(2).mean()


def _mixtral_config(**changes):
    values = dict(num_local_experts=E, num_experts_per_tok=K, num_hidden_layers=LAYERS)
    values.update(changes)
    return types.SimpleNamespace(**{k: v for k, v in values.items() if v is not None})


def _mixtral_4x(**changes) -> nn.Module:
    """A stand-in for transformers 4.46 to 4.57's `MixtralForCausalLM` (its layout and routing)."""
    cls = _named(type("MixtralForCausalLM", (_ForCausalLM,), {}), f"{MIXTRAL}.MixtralForCausalLM")
    torch.manual_seed(0)
    return cls(_Layer4x, _mixtral_config(**changes))


def _train(model: nn.Module, steps: int = 3, **kwargs) -> tuple[tensorquill.MemorySink, list[str]]:
    """`steps` optimizer steps under `init(model, **kwargs)`; the records and the warnings said."""
    sink = tensorquill.MemorySink()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    torch.manual_seed(1)
    batches = [torch.randn(BATCH, SEQ, H) for _ in range(steps)]
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        session = tensorquill.init(model, run_id="sr2", sink=sink, every=2, **kwargs)
        try:
            for x in batches:
                optimizer.zero_grad()
                model(x).backward()
                optimizer.step()
        finally:
            session.close()
    return sink, [str(w.message) for w in said]


def _expected_counts(block: _Block4x, steps: int) -> list[list[int]]:
    return [torch.bincount(chosen.reshape(-1), minlength=E).tolist() for chosen in block.chosen[:steps]]


# --- matching, on stand-in classes (no transformers) ---------------------------------------------


def test_an_exact_class_gives_the_entrys_routers_with_n_experts_and_top_k_from_the_configuration(caplog):
    model = _mixtral_4x()
    caplog.set_level(logging.INFO, logger="tensorquill")
    sink, said = _train(model, routers="auto")
    names = [f"model.layers.{i}.block_sparse_moe.gate" for i in range(LAYERS)]
    records = sink.records
    assert len(records) == 3
    for record in records:
        assert list(record["moe"]) == names
        assert record["moe_top_k"] == {name: K for name in names}
        assert all(len(record["moe"][name]) == E for name in names)
        # Every token's top_k choices, and exactly what the block chose (a plain top-k of the gate's logits).
        assert all(sum(record["moe"][name]) == TOKENS * K for name in names)
    for i, name in enumerate(names):
        block = model.model.layers[i].block_sparse_moe
        assert [r["moe"][name] for r in records] == _expected_counts(block, 3)
    assert not any("moe_refused" in r for r in records)
    assert not any("router" in text for text in said)
    # What was chosen is logged once: the class, the entry, the modules, n_experts and top_k.
    chosen = [r.getMessage() for r in caplog.records if 'routers="auto"' in r.getMessage()]
    assert len(chosen) == 1
    line = chosen[0]
    assert f"{MIXTRAL}.MixtralForCausalLM" in line and "num_local_experts" in line and "num_experts_per_tok" in line
    assert f"n_experts={E}" in line and f"top_k={K}" in line
    assert all(name in line for name in names)


def test_auto_records_equal_the_records_of_the_same_routers_given_by_hand():
    def strip(records):
        return [{k: v for k, v in r.items() if k not in ("ts", "attempt")} for r in records]

    auto, _ = _train(_mixtral_4x(), routers="auto")
    names = [f"model.layers.{i}.block_sparse_moe.gate" for i in range(LAYERS)]
    by_hand, _ = _train(_mixtral_4x(), routers={name: tensorquill.Router(E, K) for name in names})
    assert strip(auto.records) == strip(by_hand.records)


def test_a_subclass_of_a_registered_class_is_not_in_the_registry():
    base = type(_mixtral_4x())
    sub = type("MyMixtral", (base,), {})
    sub.__module__ = "my_project.models"
    assert router_registry.lookup(sub) is None
    assert router_registry.lookup(base) is not None


def test_the_same_class_name_in_another_module_is_not_in_the_registry():
    cls = _named(type("MixtralForCausalLM", (_ForCausalLM,), {}), "my_project.modeling.MixtralForCausalLM")
    assert router_registry.lookup(cls) is None


def test_a_class_not_in_the_registry_finds_no_router_and_says_so_once_under_moe_refused():
    class Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(H, 1)

        def forward(self, x):
            return self.net(x).pow(2).mean()

    torch.manual_seed(0)
    model = Dense()
    hooks_before = len(model.net._forward_hooks)
    sink, said = _train(model, routers="auto")
    key = f"{Dense.__module__}.{Dense.__qualname__}"
    notes = [text for text in said if "not in TensorQuill's router registry" in text]
    assert len(notes) == 1 and key in notes[0]
    records = sink.records
    assert records[0]["moe_refused"] == {key: records[0]["moe_refused"][key]}
    assert "not in TensorQuill's router registry" in records[0]["moe_refused"][key]
    assert not any("moe_refused" in r for r in records[1:])
    assert not any("moe" in r or "moe_top_k" in r for r in records)
    # With no router found the model is captured in the full mode, as with no routers= at all (the
    # owner, Nov 27, 2026): its layer is hooked, and the refusal above is still written.
    assert len(model.net._forward_hooks) == hooks_before + 1
    evidence = build_evidence(records)
    router = [entry for entry in evidence["not_checked"] if entry.get("check") == "router"]
    assert router and router[0]["routers"] == [key] and router[0]["kinds"] == list(MOE_KINDS)


def test_auto_adds_no_key_to_the_record():
    """The owner's first condition: the keys of an auto run's records are keys a hand-given run writes."""
    auto, _ = _train(_mixtral_4x(), routers="auto")
    names = [f"model.layers.{i}.block_sparse_moe.gate" for i in range(LAYERS)]
    by_hand, _ = _train(_mixtral_4x(), routers={name: tensorquill.Router(E, K) for name in names})
    assert [set(r) for r in auto.records] == [set(r) for r in by_hand.records]
    refused, _ = _train(_mixtral_4x(num_experts_per_tok=None), routers="auto")
    hand_refused, _ = _train(_mixtral_4x(), routers={names[0]: tensorquill.Router(E, E + 1)})
    assert set(refused.records[0]) <= set(hand_refused.records[0])


def test_auto_takes_the_router_only_mode_as_a_hand_given_map_does():
    names = [f"model.layers.{i}.block_sparse_moe.gate" for i in range(LAYERS)]
    watched = []
    for routers in ("auto", {name: tensorquill.Router(E, K) for name in names}, None):
        session = tensorquill.init(_mixtral_4x(), run_id="mode", sink=tensorquill.MemorySink(), routers=routers)
        watched.append(list(session.layer_names))
        session.close()
    assert watched[0] == watched[1] == [] != watched[2]


def test_auto_that_finds_no_router_takes_the_full_mode_as_no_routers_does():
    """The owner's answer of Nov 27, 2026: "auto" takes the router-only mode only when a router is found.
    A class not in the registry, and a registered class none of whose routers fits, are captured as a
    run with no routers= is; `capture="routers"` given by hand still holds."""
    def watched(model, **kwargs):
        session = tensorquill.init(model, run_id="mode", sink=tensorquill.MemorySink(), **kwargs)
        try:
            return list(session.layer_names)
        finally:
            session.close()

    dense = lambda: nn.Sequential(nn.Linear(H, H), nn.Linear(H, 1))  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert watched(dense(), routers="auto") == watched(dense()) != []
        unfit = lambda: _mixtral_4x(num_experts_per_tok=None)  # noqa: E731  # every router refused
        assert watched(unfit(), routers="auto") == watched(unfit()) != []
        assert watched(dense(), routers="auto", capture="routers") == []


class _Router5(nn.Module):
    """transformers 5.x's `MixtralTopKRouter`, as read at those tags: it returns (probabilities or
    logits, weights, indices), the indices chosen by a top-k of its softmax."""

    def __init__(self) -> None:
        super().__init__()
        self.top_k = K
        self.weight = nn.Parameter(torch.randn(E, H))

    def forward(self, x):
        logits = nn.functional.linear(x.reshape(-1, H), self.weight)
        weights, indices = torch.topk(torch.softmax(logits.float(), dim=-1), self.top_k, dim=-1)
        return logits, weights, indices


_named(_Router5, f"{MIXTRAL}.MixtralTopKRouter")


class _Block5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = _Router5()
        self.experts = nn.Linear(H, H)

    def forward(self, x):
        _, weights, indices = self.gate(x)
        scale = weights.sum(-1, keepdim=True) * (indices.float().mean(-1, keepdim=True) + 1) / E
        return x + self.experts(x) * scale.reshape(*x.shape[:-1], 1)


class _Layer5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _Block5()

    def forward(self, x):
        return self.mlp(x)


@pytest.mark.parametrize("layout", ["4.x", "5.x"])
def test_under_torch_compile_with_fullgraph_the_auto_routers_count_as_in_eager_mode(layout):
    """The registry's `indices=` (5.x) and the top-k of logits (4.x) trace with no graph break
    (`fullgraph=True` raises on one), and count what eager mode counts."""
    cls = _named(type("MixtralForCausalLM", (_ForCausalLM,), {}), f"{MIXTRAL}.MixtralForCausalLM")

    def run(compiled: bool):
        torch.manual_seed(0)
        model = cls(_Layer4x if layout == "4.x" else _Layer5, _mixtral_config())
        sink = tensorquill.MemorySink()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        session = tensorquill.init(model, run_id="compiled", sink=sink, every=2, routers="auto")
        runner = torch.compile(model, backend="eager", fullgraph=True) if compiled else model
        torch.manual_seed(1)
        for _ in range(3):
            optimizer.zero_grad()
            runner(torch.randn(BATCH, SEQ, H)).backward()
            optimizer.step()
        session.close()
        return [r["moe"] for r in sink.records]

    torch._dynamo.reset()
    eager = run(False)
    assert eager and all(len(step) == LAYERS for step in eager)
    assert run(True) == eager


# --- an entry that does not fit is refused by name, never counted as zeros ------------------------


def test_a_router_module_missing_where_the_configuration_puts_one_is_refused_by_name():
    model = _mixtral_4x()
    del model.model.layers[1].block_sparse_moe.gate
    model.model.layers[1].block_sparse_moe.gate = lambda x: torch.zeros(x.shape[0], E)  # not a module
    sink, said = _train(model, routers="auto")
    missing = "model.layers.1.block_sparse_moe.gate"
    first = sink.records[0]
    assert list(first["moe_refused"]) == [missing]
    assert "no module" in first["moe_refused"][missing]
    assert all(missing not in r.get("moe", {}) for r in sink.records)
    assert list(first["moe"]) == ["model.layers.0.block_sparse_moe.gate", "model.layers.2.block_sparse_moe.gate"]
    assert sum("not counting the router 'model.layers.1.block_sparse_moe.gate'" in t for t in said) == 1
    # The fix does not tell the user to name a module that does not exist.
    assert "with the name of the module that routes it" in first["moe_refused"][missing]
    assert f"routers={{'{missing}'" not in first["moe_refused"][missing]


def test_the_first_layer_not_fitting_is_refused_by_its_own_name_and_the_others_counted():
    model = _mixtral_4x()
    model.model.layers[0].block_sparse_moe.gate = nn.Sequential(nn.Linear(H, E, bias=False))
    sink, _ = _train(model, routers="auto")
    first = sink.records[0]
    assert list(first["moe_refused"]) == ["model.layers.0.block_sparse_moe.gate"]
    assert list(first["moe"]) == ["model.layers.1.block_sparse_moe.gate", "model.layers.2.block_sparse_moe.gate"]


def test_a_configuration_value_absent_refuses_every_router_of_the_entry_by_name():
    sink, said = _train(_mixtral_4x(num_experts_per_tok=None), routers="auto")
    names = [f"model.layers.{i}.block_sparse_moe.gate" for i in range(LAYERS)]
    first = sink.records[0]
    assert list(first["moe_refused"]) == names
    assert all("num_experts_per_tok" in first["moe_refused"][name] for name in names)
    assert not any("moe" in r for r in sink.records)


def test_a_gate_of_another_class_or_width_is_refused_by_name():
    model = _mixtral_4x()
    model.model.layers[0].block_sparse_moe.gate = nn.Linear(H, E + 2, bias=False)  # 10 rows, config says 8
    model.model.layers[2].block_sparse_moe.gate = nn.Sequential(nn.Linear(H, E, bias=False))
    sink, _ = _train(model, routers="auto")
    first = sink.records[0]
    assert sorted(first["moe_refused"]) == ["model.layers.0.block_sparse_moe.gate", "model.layers.2.block_sparse_moe.gate"]
    assert f"{E + 2}" in first["moe_refused"]["model.layers.0.block_sparse_moe.gate"]
    assert "Sequential" in first["moe_refused"]["model.layers.2.block_sparse_moe.gate"]
    assert list(first["moe"]) == ["model.layers.1.block_sparse_moe.gate"]


def test_a_block_whose_top_k_is_not_the_configurations_is_refused_by_name():
    model = _mixtral_4x()
    model.model.layers[1].block_sparse_moe.top_k = 3
    sink, _ = _train(model, routers="auto")
    first = sink.records[0]
    assert list(first["moe_refused"]) == ["model.layers.1.block_sparse_moe.gate"]
    why = first["moe_refused"]["model.layers.1.block_sparse_moe.gate"]
    assert "top_k" in why
    # The fix names no numbers: the configuration's top_k (2) given by hand would count a top-3 block wrongly.
    assert f"Router({E}, {K})" not in why and "the top_k it routes with" in why


class _LogitsOnlyRouter(nn.Module):
    """transformers 5.0 to 5.12's `DeepseekV3TopkRouter`, as read at those tags: it returns its logits
    only, and the expert choice (sigmoid, a correction bias, group-limited top-k) is made by its parent.
    A plain top-k of these logits passes every shape check and counts the wrong experts."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(E, H))

    def forward(self, x):
        return nn.functional.linear(x.reshape(-1, H).float(), self.weight.float())


_named(_LogitsOnlyRouter, f"{DEEPSEEK_V3}.DeepseekV3TopkRouter")


class _DeepseekMoe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = _LogitsOnlyRouter()
        self.bias = torch.linspace(-1, 1, E)

    def forward(self, x):
        scores = self.gate(x).sigmoid()
        chosen = torch.topk(scores + self.bias, K, dim=-1).indices
        return x + scores.gather(1, chosen).sum(-1).reshape(x.shape[:-1]).unsqueeze(-1) * x


class _DeepseekLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _DeepseekMoe()

    def forward(self, x):
        return self.mlp(x)


def test_a_deepseek_v3_router_that_returns_logits_only_is_refused_never_counted_as_its_top_k():
    cls = _named(type("DeepseekV3ForCausalLM", (_ForCausalLM,), {}), f"{DEEPSEEK_V3}.DeepseekV3ForCausalLM")
    config = types.SimpleNamespace(n_routed_experts=E, num_experts_per_tok=K, num_hidden_layers=LAYERS,
                                   first_k_dense_replace=0)
    torch.manual_seed(0)
    model = cls(_DeepseekLayer, config)
    sink, said = _train(model, routers="auto")
    names = [f"model.layers.{i}.mlp.gate" for i in range(LAYERS)]
    first = sink.records[0]
    assert sorted(first["moe_refused"]) == names
    why = first["moe_refused"][names[0]]
    assert "float32" in why and 'routers="auto"' in why
    assert not any("moe" in r for r in sink.records)
    assert sum("not counting the router" in t for t in said) == LAYERS


def test_a_router_of_the_entrys_class_in_a_layer_the_configuration_gives_none_is_refused_by_name():
    cls = _named(type("DeepseekV3ForCausalLM", (_ForCausalLM,), {}), f"{DEEPSEEK_V3}.DeepseekV3ForCausalLM")
    config = types.SimpleNamespace(n_routed_experts=E, num_experts_per_tok=K, num_hidden_layers=LAYERS,
                                   first_k_dense_replace=1)
    torch.manual_seed(0)
    sink, _ = _train(cls(_DeepseekLayer, config), routers="auto")
    refused = sink.records[0]["moe_refused"]
    assert "model.layers.0.mlp.gate" in refused
    assert "gives no router" in refused["model.layers.0.mlp.gate"]


@pytest.mark.parametrize("case", ["missing", "logits_only", "unknown"])
def test_the_pinned_flush_writes_the_same_auto_refusals_as_the_blocking_path(monkeypatch, case):
    """`_settle_refusals` (the pinned path) holds the registry's refusals as `_router_refusals` does."""
    from test_pinned_flush import FakeCopies  # the CPU stand-in for the CUDA copier

    from tensorquill import capture

    def model():
        if case == "missing":
            m = _mixtral_4x()
            del m.model.layers[1].block_sparse_moe.gate
            m.model.layers[1].block_sparse_moe.gate = lambda x: torch.zeros(x.shape[0], E)
            return m
        if case == "unknown":
            torch.manual_seed(0)
            return nn.Sequential(nn.Linear(H, 1))
        cls = _named(type("DeepseekV3ForCausalLM", (_ForCausalLM,), {}), f"{DEEPSEEK_V3}.DeepseekV3ForCausalLM")
        torch.manual_seed(0)
        return cls(_DeepseekLayer, types.SimpleNamespace(n_routed_experts=E, num_experts_per_tok=K,
                                                         num_hidden_layers=LAYERS, first_k_dense_replace=0))

    def strip(records):
        return [{k: v for k, v in r.items() if k not in ("ts", "attempt")} for r in records]

    if case == "unknown":
        # A plain stack: its forward returns a tensor, so the loss is taken here.
        def train(**kwargs):
            m = model()
            sink = tensorquill.MemorySink()
            opt = torch.optim.SGD(m.parameters(), lr=0.01)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Full capture, so that the block holds tensors for the pinned copy to move.
                session = tensorquill.init(m, run_id="pinned", sink=sink, every=2, capture="full", **kwargs)
                for _ in range(3):
                    opt.zero_grad()
                    m(torch.ones(BATCH, H)).pow(2).mean().backward()
                    opt.step()
                session.close()
            return sink
    else:
        def train(**kwargs):
            return _train(model(), **kwargs)[0]

    blocking = train(routers="auto")
    copies = FakeCopies("late")
    monkeypatch.setattr(capture, "_pinned_backend", lambda device: copies)
    pinned = train(routers="auto", pinned_flush=True)
    assert copies.copies > 0  # the pinned path ran
    assert "moe_refused" in blocking.records[0]
    assert strip(pinned.records) == strip(blocking.records)


# --- without the opt-in, the registry is never reached --------------------------------------------


@pytest.mark.parametrize("routers", [None, "hand"])
def test_without_auto_the_registry_is_never_read(monkeypatch, routers):
    def fail(*args, **kwargs):
        raise AssertionError("the registry was read without routers='auto'")

    monkeypatch.setattr(router_registry, "resolve", fail)
    monkeypatch.setattr(router_registry, "lookup", fail)
    names = [f"model.layers.{i}.block_sparse_moe.gate" for i in range(LAYERS)]
    given = None if routers is None else {name: tensorquill.Router(E, K) for name in names}
    sink, _ = _train(_mixtral_4x(), routers=given)
    assert len(sink.records) == 3


def test_auto_is_resolved_before_any_hook_and_the_hooks_are_those_of_the_hand_given_map():
    names = [f"model.layers.{i}.block_sparse_moe.gate" for i in range(LAYERS)]
    auto, by_hand = _mixtral_4x(), _mixtral_4x()
    a = tensorquill.init(auto, run_id="a", sink=tensorquill.MemorySink(), routers="auto")
    b = tensorquill.init(by_hand, run_id="b", sink=tensorquill.MemorySink(),
                         routers={name: tensorquill.Router(E, K) for name in names})
    try:
        hooked = lambda m: {n for n, mod in m.named_modules() if mod._forward_hooks}  # noqa: E731
        assert hooked(auto) == hooked(by_hand) == set(names)
    finally:
        a.close()
        b.close()


def test_the_registry_imports_no_transformers():
    """The owner's second condition, read from the source: neither module names the package in an import."""
    import inspect
    import re

    from tensorquill import capture

    for module in (capture, router_registry):
        source = inspect.getsource(module)
        assert not re.search(r"^\s*(import|from)\s+transformers\b", source, re.M), module.__name__


# --- tiny Hugging Face models with random weights (need transformers) -----------------------------


def _transformers():
    return pytest.importorskip("transformers")


def _major() -> int:
    return int(_transformers().__version__.split(".")[0])


def _deepseek_logits_only() -> bool:
    """transformers 5.0 to 5.12, whose `DeepseekV3TopkRouter` returns its logits only (read at those tags)."""
    major, minor = (int(part) for part in _transformers().__version__.split(".")[:2])
    return major == 5 and minor < 13


def _tiny(family: str):
    """A tiny configuration of `family`, its causal-LM class, the routers the configuration places (by
    the module names of the installed transformers' layout) and their n_experts and top_k."""
    tf = _transformers()
    v5 = _major() >= 5
    common = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                  num_key_value_heads=4, max_position_embeddings=64)
    if family == "mixtral":
        config = tf.MixtralConfig(num_local_experts=8, num_experts_per_tok=2, **common)
        path = "mlp.gate" if v5 else "block_sparse_moe.gate"
        return tf.MixtralForCausalLM, config, [f"model.layers.{i}.{path}" for i in range(3)], 8, 2
    if family == "olmoe":
        config = tf.OlmoeConfig(num_experts=8, num_experts_per_tok=2, **common)
        return tf.OlmoeForCausalLM, config, [f"model.layers.{i}.mlp.gate" for i in range(3)], 8, 2
    if family == "qwen2_moe":
        # Six layers: decoder_sparse_step=2 gives layers 1, 3 and 5 a router, and mlp_only_layers takes 1 away.
        config = tf.Qwen2MoeConfig(num_experts=6, num_experts_per_tok=3, moe_intermediate_size=16,
                                   shared_expert_intermediate_size=32, mlp_only_layers=[1], decoder_sparse_step=2,
                                   **dict(common, num_hidden_layers=6))
        return tf.Qwen2MoeForCausalLM, config, ["model.layers.3.mlp.gate", "model.layers.5.mlp.gate"], 6, 3
    if family == "qwen3_moe":
        if not hasattr(tf, "Qwen3MoeConfig"):
            pytest.skip(f"transformers {tf.__version__} has no Qwen3-MoE (added in 4.51)")
        config = tf.Qwen3MoeConfig(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
                                   decoder_sparse_step=2, **dict(common, num_hidden_layers=4))
        return tf.Qwen3MoeForCausalLM, config, ["model.layers.1.mlp.gate", "model.layers.3.mlp.gate"], 8, 2
    if family == "deepseek_v3":
        if not hasattr(tf, "DeepseekV3Config"):
            pytest.skip(f"transformers {tf.__version__} has no DeepSeek-V3 (added in 4.51)")
        # Group-limited routing (4 groups, the best 2 kept) and, set by the test, a correction bias that is
        # not zero: a plain top-k of the logits would choose other experts.
        config = tf.DeepseekV3Config(n_routed_experts=8, num_experts_per_tok=2, n_group=4, topk_group=2,
                                     first_k_dense_replace=1, n_shared_experts=1, moe_intermediate_size=16,
                                     kv_lora_rank=16, q_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8,
                                     v_head_dim=8, **common)
        return tf.DeepseekV3ForCausalLM, config, ["model.layers.1.mlp.gate", "model.layers.2.mlp.gate"], 8, 2
    if family == "granitemoe":
        config = tf.GraniteMoeConfig(num_local_experts=8, num_experts_per_tok=2, **common)
        modern = hasattr(sys.modules[tf.GraniteMoeForCausalLM.__module__], "GraniteMoeTopKRouter")
        path = "block_sparse_moe.router" if modern else "block_sparse_moe.router.layer"
        return tf.GraniteMoeForCausalLM, config, [f"model.layers.{i}.{path}" for i in range(3)], 8, 2
    raise AssertionError(family)


FAMILIES = ["mixtral", "olmoe", "qwen2_moe", "qwen3_moe", "deepseek_v3", "granitemoe"]


def _model_counts(model: nn.Module, names: list[str], experts: int, top_k: int) -> tuple[dict[str, list], list]:
    """Hooks of the test's own that read, for each router, the experts the model itself routed each
    forward to: with transformers 5, the ids its block hands its experts module (a pre-hook on the
    sibling `experts`); with 4.x, Granite's `expert_size` (its gating's own per-expert counts) or, for
    the others, the block's own rule (the top-k of the softmax of the router logits it returns)."""
    counts: dict[str, list] = {name: [] for name in names}
    handles = []
    for name in names:
        parent_name = name.rpartition(".")[0]
        parent = model.get_submodule(parent_name)
        if hasattr(parent, "experts") and not isinstance(parent.experts, nn.ModuleList):
            def keep(module, args, name=name):
                counts[name].append(torch.bincount(args[1].reshape(-1), minlength=experts).tolist())
            handles.append(parent.experts.register_forward_pre_hook(keep))
        elif callable(getattr(parent, "moe", None)):
            # DeepSeek-V3 on 4.51 to 4.57: `DeepseekV3MoE.moe(hidden_states, topk_indices, topk_weights)`, a
            # method, is handed the ids; wrapped on this instance for the test.
            def wrapped(hidden, ids, weights, name=name, inner=parent.moe):
                counts[name].append(torch.bincount(ids.reshape(-1), minlength=experts).tolist())
                return inner(hidden, ids, weights)
            parent.moe = wrapped
        elif name.endswith(".router.layer"):
            def sizes(module, args, out, name=name):
                counts[name].append([int(v) for v in out[3]])
            handles.append(parent.register_forward_hook(sizes))
        else:
            def chosen(module, args, out, name=name):
                ids = torch.softmax(out[1].detach().float(), dim=-1).topk(top_k, dim=-1).indices
                counts[name].append(torch.bincount(ids.reshape(-1), minlength=experts).tolist())
            handles.append(parent.register_forward_hook(chosen))
    return counts, handles


def _hf_train(model, steps: int = 3, watch: tuple = (), **kwargs):
    sink = tensorquill.MemorySink()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    torch.manual_seed(2)
    batches = [torch.randint(0, 64, (BATCH, SEQ)) for _ in range(steps)]
    counts, handles = _model_counts(model, *watch) if watch else ({}, [])
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        session = tensorquill.init(model, run_id="sr2-hf", sink=sink, every=2, **kwargs)
        try:
            for ids in batches:
                optimizer.zero_grad()
                model(input_ids=ids, labels=ids).loss.backward()
                optimizer.step()
        finally:
            session.close()
            for handle in handles:
                handle.remove()
    return sink, counts, [str(w.message) for w in said]


def _build(family: str):
    cls, config, names, experts, top_k = _tiny(family)
    torch.manual_seed(0)
    model = cls(config)
    model.train()
    if family == "deepseek_v3":
        for name in names:
            model.get_submodule(name).e_score_correction_bias.copy_(torch.linspace(-0.5, 0.5, experts))
    return model, names, experts, top_k


@pytest.mark.parametrize("family", FAMILIES)
def test_a_tiny_hugging_face_model_has_its_routers_found_and_counted_as_the_model_routes(family, caplog):
    model, names, experts, top_k = _build(family)
    caplog.set_level(logging.INFO, logger="tensorquill")
    sink, routed, said = _hf_train(model, watch=(names, experts, top_k), routers="auto")
    records = sink.records
    assert len(records) == 3
    if family == "deepseek_v3" and _deepseek_logits_only():
        # Its router returns logits only: refused at the first forward, never counted as their top-k.
        assert sorted(records[0]["moe_refused"]) == names
        assert all("found a float32 tensor" in records[0]["moe_refused"][name] for name in names)
        assert not any("moe" in r for r in records)
        return
    assert not any("moe_refused" in r for r in records), records[0].get("moe_refused")
    assert not any("router" in text for text in said)
    for record in records:
        assert list(record["moe"]) == names
        assert record["moe_top_k"] == {name: top_k for name in names}
        assert all(len(record["moe"][name]) == experts for name in names)
        assert all(sum(record["moe"][name]) == TOKENS * top_k for name in names)
    # The counts are what the model routed, step by step: not only the right total.
    for name in names:
        assert [r["moe"][name] for r in records] == routed[name], name
    line = [r.getMessage() for r in caplog.records if 'routers="auto"' in r.getMessage()]
    cls = type(model)
    assert len(line) == 1 and f"{cls.__module__}.{cls.__qualname__}" in line[0]
    assert f"n_experts={experts}" in line[0] and f"top_k={top_k}" in line[0]


def test_a_tiny_deepseek_v3_would_be_counted_wrongly_as_a_plain_top_k_of_its_logits():
    """The test above discriminates: with this configuration and bias the model's choice is not the
    top-k of its router's logits, so counting that top-k would not match what the model routed."""
    model, names, experts, top_k = _build("deepseek_v3")
    if _deepseek_logits_only():
        pytest.skip("on transformers 5.0 to 5.12 the router is refused, so there is nothing to compare")
    logits: list[torch.Tensor] = []

    def keep(module, args, out):
        logits.append(nn.functional.linear(args[0].reshape(-1, args[0].shape[-1]).float(), module.weight.float()))

    handle = model.get_submodule(names[0]).register_forward_hook(keep)
    _, routed, _ = _hf_train(model, steps=1, watch=(names, experts, top_k))
    handle.remove()
    plain = torch.bincount(logits[0].topk(top_k, dim=-1).indices.reshape(-1), minlength=experts)
    assert plain.tolist() != routed[names[0]][0]


def test_a_tiny_hugging_face_auto_run_writes_the_records_of_the_same_map_given_by_hand():
    def strip(records):
        return [{k: v for k, v in r.items() if k not in ("ts", "attempt")} for r in records]

    model, names, experts, top_k = _build("mixtral")
    auto, _, _ = _hf_train(model, routers="auto")
    model, _, _, _ = _build("mixtral")
    by_hand, _, _ = _hf_train(model, routers={name: tensorquill.Router(experts, top_k) for name in names})
    assert strip(auto.records) == strip(by_hand.records)


def test_a_tiny_hugging_face_model_of_a_class_not_in_the_registry_finds_no_router():
    tf = _transformers()
    config = tf.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    torch.manual_seed(0)
    model = tf.LlamaForCausalLM(config)
    sink, _, said = _hf_train(model, routers="auto")
    key = "transformers.models.llama.modeling_llama.LlamaForCausalLM"
    assert list(sink.records[0]["moe_refused"]) == [key]
    assert sum("not in TensorQuill's router registry" in t for t in said) == 1


@pytest.mark.parametrize("family", FAMILIES)
def test_a_tiny_hugging_face_base_model_resolves_to_the_same_routers_without_the_heads_prefix(family):
    cls, config, names, experts, top_k = _tiny(family)
    base = getattr(_transformers(), cls.__name__.replace("ForCausalLM", "Model"))
    torch.manual_seed(0)
    found = router_registry.resolve(base(config))
    assert not found.refused
    assert found.routers == {name.removeprefix("model."): (experts, top_k, found.layout.form) for name in names}


def test_a_tiny_hugging_face_router_of_another_width_is_refused_by_name():
    model, names, experts, top_k = _build("olmoe")
    # One expert fewer than the configuration says: the model still runs (on transformers 4.x and 5.x).
    model.get_submodule(names[1]).weight = nn.Parameter(torch.zeros(experts - 1, 32))
    sink, _, _ = _hf_train(model, routers="auto")
    first = sink.records[0]
    assert list(first["moe_refused"]) == [names[1]]
    assert f"{experts - 1} rows" in first["moe_refused"][names[1]]
    assert list(first["moe"]) == [names[0], names[2]]
