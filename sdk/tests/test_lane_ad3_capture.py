"""Lane AD3 (wave AD): the capture and the probe.

- C22: two of the three overflow cases still counted and not read are read now (a squared norm
  past float32's range on MPS; logits near float32's largest value with both signs in one row),
  and the third (a BF16 logit past BF16's range) stays counted.
- Router counting allocates in proportion to the assignments, not experts by assignments.
- The ring allocates no gradient rows in the router-only mode, and two columns without the weight check.
- A session made before `to_empty` says why it counts no step at the first step.
- Clipping inside a compiled optimizer step no longer recompiles at every step.

Every test here fails on `eb5a71b` and passes with the lane's change.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import probe as P

MPS = torch.backends.mps.is_available()
DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not MPS, reason="no MPS"))]


def _numbers(hidden, weight, bias=None):
    sums = P._head_logit_sums(hidden, weight, bias)
    return P._head_numbers([{**sums, "input_dtype": "torch.float32", "weight_dtype": "torch.float32"}])


# --- C22 (a): a squared norm past float32's range on MPS ------------------------------------------


@pytest.mark.skipif(not MPS, reason="no MPS")
def test_c22_a_norm_whose_square_passes_float32_is_read_on_mps():
    # Every logit 1e20 over 1,000 entries: a norm of 3.16e21, whose square (1e43) is past float32's
    # range. Before, MPS added the squares in float32 and counted the position as overflow.
    numbers = _numbers(torch.ones(1, 4, device="mps"), torch.full((1000, 4), 2.5e19, device="mps"))
    assert numbers["positions"] == 1 and "overflow_positions" not in numbers and "not_read" not in numbers
    assert numbers["norm"] == pytest.approx(1e20 * 1000 ** 0.5, rel=1e-5)
    assert numbers["mean"] == pytest.approx(1e20, rel=1e-5)
    assert numbers["log_z"] == pytest.approx(1e20, rel=1e-5)


@pytest.mark.skipif(not MPS, reason="no MPS")
def test_c22_the_spread_and_gap_of_large_logits_are_read_on_mps():
    # A spread of 1e20 about a mean of 1e20: the squared deviations and the BF16 gap's squares are
    # past float32's range, and each was counted as overflow on MPS before.
    weight = torch.tensor([[2e20], [0.0]] * 500)
    numbers = _numbers(torch.ones(1, 1, device="mps"), weight.to("mps"))
    cpu = _numbers(torch.ones(1, 1), weight)
    assert numbers["positions"] == 1 and "overflow_positions" not in numbers
    for key in ("norm", "mean", "log_z", "gap", "gap_relative"):
        assert numbers[key] == pytest.approx(cpu[key], rel=1e-5), key
    assert numbers["gap"] / numbers["gap_relative"] == pytest.approx(1e20, rel=1e-5)


# --- C22 (c): both signs near float32's largest value ---------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_c22_logits_near_float32_max_with_both_signs_are_read(device):
    # 3e38, 3e38 and -3e38: a mean of 1e38, and -3e38 - 1e38 overflows float32.
    weight = torch.tensor([[3e38], [3e38], [-3e38]])
    numbers = _numbers(torch.ones(1, 1, device=device), weight.to(device))
    z = weight.double().reshape(-1)
    assert numbers["positions"] == 1 and numbers["nonfinite_positions"] == 0
    assert "overflow_positions" not in numbers and "not_read" not in numbers
    assert numbers["mean"] == pytest.approx(float(z.mean()), rel=1e-6)
    assert numbers["norm"] == pytest.approx(float(z.norm()), rel=1e-6)
    assert numbers["log_z"] == pytest.approx(float(torch.logsumexp(z, 0)), rel=1e-6)
    spread = float(z.std(unbiased=False))
    assert numbers["gap"] / numbers["gap_relative"] == pytest.approx(spread, rel=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_c22_both_signs_are_read_across_chunks(monkeypatch, device):
    monkeypatch.setattr(P, "CHUNK_ELEMENTS", 3)  # one position: three logits a chunk, four chunks
    weight = torch.tensor([[3e38], [1e38], [3e38], [-3e38], [-2e38], [3e38], [3e38], [-3e38], [2e38], [3e38], [0.0], [1.0]])
    numbers = _numbers(torch.ones(1, 1, device=device), weight.to(device))
    z = weight.double().reshape(-1)
    assert numbers["positions"] == 1 and "overflow_positions" not in numbers
    assert numbers["mean"] == pytest.approx(float(z.mean()), rel=1e-6)
    assert numbers["norm"] == pytest.approx(float(z.norm()), rel=1e-6)
    assert numbers["gap"] / numbers["gap_relative"] == pytest.approx(float(z.std(unbiased=False)), rel=1e-5)


def test_c22_only_the_broken_row_is_taken_again_and_its_neighbours_are_unchanged():
    # Three positions: an ordinary one, one with both signs near float32's largest value, and one
    # whose BF16 logit is past BF16's range, which is still counted as overflow, alone.
    # The third position's FP32 logit is 2 x 1.7e38 = 3.4e38, finite; in BF16 it rounds past 3.39e38.
    hidden = torch.tensor([[1e-38, 0.0], [1.0, 0.0], [0.0, 2.0]])
    weight = torch.tensor([[3e38, 1.7e38], [3e38, 0.0], [-3e38, 0.0]])
    numbers = _numbers(hidden, weight)
    assert numbers["positions"] == 2 and numbers["overflow_positions"] == 1 and numbers["nonfinite_positions"] == 0
    z = torch.nn.functional.linear(hidden.double(), weight.double())[:2]
    assert numbers["mean"] == pytest.approx(float(z.mean()), rel=1e-6)
    assert numbers["norm"] == pytest.approx(float(z.norm(dim=-1).mean()), rel=1e-6)


class _Head(nn.Module):
    """A model whose head gives every position the logits `bias`."""

    def __init__(self, bias: list[float]) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.head = nn.Linear(4, len(bias))
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.copy_(torch.tensor(bias))

    def forward(self, ids):
        return self.head(self.embed(ids))


def test_c22_both_signs_are_read_in_the_record():
    model = _Head([3e38, 3e38, -3e38, 0.0])
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, probe=torch.arange(6).reshape(2, 3), probe_every=1,
                               probe_head="head")
    opt = torch.optim.SGD(model.embed.parameters(), lr=0.01)
    for _ in range(2):
        model.embed(torch.arange(4)).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    session.close()
    heads = [r["probe"]["head"] for r in sink.records if "probe" in r]
    assert heads and all(h["positions"] == 6 and "overflow_positions" not in h for h in heads)
    assert heads[-1]["mean"] == pytest.approx(float(torch.tensor([3e38, 3e38, -3e38, 0.0]).double().mean()), rel=1e-6)
    assert math.isfinite(heads[-1]["gap_relative"])


# --- Router counting in proportion to the assignments ----------------------------------------------

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from tensorquill.capture import Router  # noqa: E402

TOKENS, E, K = 120, 64, 4  # experts by assignments is 30,720 entries; the logits are 7,680


class _Largest(TorchDispatchMode):
    """The most elements any op that is not a view made while it was on, and how many outputs of
    `size` elements there were."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.largest, self.size, self.of_size = 0, size, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if not func.is_view:
            for t in out if isinstance(out, (tuple, list)) else (out,):
                if isinstance(t, torch.Tensor):
                    self.largest = max(self.largest, t.numel())
                    self.of_size += t.numel() == self.size
        return out


class _Gate(nn.Module):
    def __init__(self, make) -> None:
        super().__init__()
        self.w = nn.Linear(16, E, bias=False)
        self.make = make

    def forward(self, x):
        return self.make(self.w(x))


class _Net(nn.Module):
    def __init__(self, make) -> None:
        super().__init__()
        self.embed = nn.Linear(4, 16)
        self.router = _Gate(make)
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        hidden = self.embed(x)
        self.router(hidden)
        return self.head(hidden).pow(2).mean()


def _ids_with_none(logits):
    ids = logits.topk(K, dim=-1).indices
    return torch.where(logits[..., :K] > 1.0, torch.full_like(ids, -1), ids)  # some "no route" (-1)


def _routing_map(logits):
    probs = logits.softmax(-1)
    top = probs.topk(K, dim=-1)
    weights = torch.zeros_like(probs).scatter(-1, top.indices, top.values)
    weights = torch.where(weights < 0.02, torch.zeros_like(weights), weights)  # some routed with weight 0
    return weights, torch.zeros_like(probs, dtype=torch.bool).scatter(-1, top.indices, True)


PATHS = {
    "logits": (lambda logits: logits, None),
    "integer ids": (_ids_with_none, None),
    "indices=": (lambda logits: (logits, logits.topk(K, dim=-1).indices.to(torch.int32)),
                 lambda out: out[1]),
    "routing map": (_routing_map, None),
}


def _expected(outputs, path):
    """The counts the hook made before lane AD3, from the outputs it saw: experts by assignments."""
    total, unassigned, left_out = torch.zeros(E, dtype=torch.int64), 0, torch.zeros(E, dtype=torch.int64)
    for out in outputs:
        if path == "routing map":
            weights, routing = out
            total += (routing & (weights != 0)).reshape(-1, E).sum(0)
            left_out += (routing & (weights == 0)).reshape(-1, E).sum(0)
            continue
        ids = out.topk(K, dim=-1).indices if path == "logits" else out[1] if path == "indices=" else out
        flat = ids.reshape(-1, 1).long()
        total += (flat == torch.arange(E)).sum(0)
        unassigned += int((flat < 0).sum())
    return total.tolist(), unassigned, left_out.tolist()


def _count(path, device, backend, steps=3):
    make, indices = PATHS[path]
    torch.manual_seed(0)
    model = _Net(make).to(device)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=steps, capture="routers",
                               routers={"router": Router(E, K, indices=indices)})
    seen: list = []
    largest = _Largest(TOKENS * E)
    nodes: list[int] = []

    def keep(module, args, output):
        seen.append(tuple(t.detach().cpu() for t in output) if isinstance(output, tuple) else output.detach().cpu())

    handles = []
    if backend is None:
        def enter(*_):
            largest.__enter__()  # returns None, so the output is left as it is

        def leave(*_):
            largest.__exit__(None, None, None)

        handles.append(model.router.register_forward_hook(enter, prepend=True))
        handles.append(model.router.register_forward_hook(leave))
        handles.append(model.router.register_forward_hook(keep))  # after `leave`: its copy is not counted
        stepped = model
    else:
        def record(gm, example_inputs):
            for node in gm.graph.nodes:
                value = node.meta.get("example_value")
                if node.op == "call_function" or node.op == "call_method":
                    for t in value if isinstance(value, (tuple, list)) else (value,):
                        if isinstance(t, torch.Tensor):
                            nodes.append(t.numel())
            return gm.forward

        handles.append(model.router.register_forward_hook(keep))
        stepped = torch.compile(model, backend=record if backend == "graph" else backend)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad()
        stepped(torch.randn(6, 20, 4, device=device)).backward()
        optimizer.step()
    for h in handles:
        h.remove()
    session.close()
    if path == "indices=":
        seen = [(None, out) for out in [s[1] for s in seen]]
    return sink.records, seen, largest, nodes


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("path", list(PATHS))
def test_router_counts_are_as_before_and_no_experts_by_assignments_matrix_is_made(path, device):
    records, seen, largest, _ = _count(path, device, None)
    assert len(records) == 3 and all("moe_refused" not in r for r in records)
    per_step = len(seen) // 3
    for i, record in enumerate(records):
        counts, unassigned, left_out = _expected(seen[i * per_step:(i + 1) * per_step], path)
        assert record["moe"]["router"] == counts
        if path in ("integer ids", "indices="):
            assert record["moe_unassigned"]["router"] == unassigned
        if path == "routing map":
            assert record["moe_left_out"]["router"] == left_out
    if path == "routing map":
        # The map is tokens x experts already: the hook makes two bool masks of that size, not four.
        assert largest.of_size == 2 * 3, largest.of_size
    else:
        # Before, a bool matrix of assignments by experts (TOKENS * K * E = 30,720 entries).
        assert largest.largest <= TOKENS * K * 2, largest.largest


@pytest.mark.parametrize("path", ["logits", "integer ids", "indices="])
def test_compiled_router_counts_are_as_before_and_the_graph_holds_no_experts_by_assignments_matrix(path):
    eager, _, _, _ = _count(path, "cpu", None)
    records, _, _, nodes = _count(path, "cpu", "graph")
    assert [r["moe"]["router"] for r in records] == [r["moe"]["router"] for r in eager]
    assert nodes and max(nodes) < TOKENS * K * E, max(nodes)  # the logits (7,680) are the largest now


def _cost_holds(path, largest):
    if path == "routing map":
        return largest.of_size == 2 * 3  # two map-sized masks a forward, not four
    return largest.largest <= TOKENS * K * 2  # not a matrix of assignments by experts


def _same_counts(records, eager):
    for key in ("moe", "moe_unassigned", "moe_left_out"):
        assert [r.get(key, {}).get("router") for r in records] == [r.get(key, {}).get("router") for r in eager], key


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("path", list(PATHS))
def test_router_counts_under_aot_eager_match_eager_on_every_path(path, device):
    eager, seen, largest, _ = _count(path, device, None)
    records, _, _, _ = _count(path, device, "aot_eager")
    assert len(records) == 3 and all("moe_refused" not in r for r in records)
    _same_counts(records, eager)
    per_step = len(seen) // 3
    assert [r["moe"]["router"] for r in records] == [
        _expected(seen[i * per_step:(i + 1) * per_step], path)[0] for i in range(3)]
    assert _cost_holds(path, largest), (largest.largest, largest.of_size)


def test_router_counts_under_inductor_match_eager_and_make_no_matrix(tmp_path, monkeypatch):
    # A cache of its own: other processes on this machine compile into the shared one at the same time.
    # Its precompiled header is shared too, and another process rebuilding it fails this compile.
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    import torch._inductor.config as inductor_config

    if hasattr(inductor_config, "cpp_cache_precompile_headers"):  # absent before PyTorch 2.14: nothing shared to turn off
        monkeypatch.setattr(inductor_config, "cpp_cache_precompile_headers", False)
    torch._dynamo.reset()
    eager, _, largest, _ = _count("integer ids", "cpu", None)
    records, _, _, _ = _count("integer ids", "cpu", "inductor")
    _same_counts(records, eager)
    assert _cost_holds("integer ids", largest)


# --- The ring allocates only the signals that are on -----------------------------------------------


class _Two(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), nn.ReLU()) for _ in range(2)])
        self.out = nn.Linear(4, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.out(x).pow(2).mean()


def _steps(model, n, optimizers):
    for _ in range(n):
        model(torch.randn(3, 4)).backward()
        for opt in optimizers:
            opt.step()
        for opt in optimizers:
            opt.zero_grad()


def _router_net():
    torch.manual_seed(0)
    return _Net(lambda logits: logits)


def _router_steps(model, n):
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(n):
        model(torch.randn(6, 20, 4)).backward()
        opt.step()
        opt.zero_grad()


ROUTER = {"router": Router(E, K)}


def test_the_router_only_mode_makes_no_gradient_rows_in_the_ring():
    model = _router_net()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=8, capture="routers", routers=ROUTER)
    assert session._ring_grad is None  # before: 8 rows of four floats for each of 5 parameters
    _router_steps(model, 3)
    assert session._ring_grad is None
    session.close()
    assert [r["step"] for r in sink.records] == [0, 1, 2]
    assert all(r["grad_norms"] == {} and r["grad_norm"] is None and "weight_norms" not in r for r in sink.records)
    assert all(sum(r["moe"]["router"]) == TOKENS * K for r in sink.records)


def test_gradients_read_by_hand_in_the_router_only_mode_are_still_recorded():
    model = _router_net()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=4, capture="routers", routers=ROUTER, auto_step=False)
    assert session._ring_grad is None
    for step in range(3):
        model(torch.randn(6, 20, 4)).backward()
        if step == 1:
            session.record_grads()
        session.step()
        model.zero_grad()
    assert session._ring_grad is not None and session._ring_grad.shape == (4, 5, 2)
    session.close()
    records = sink.records
    assert [bool(r["grad_norms"]) for r in records] == [False, True, False]
    expected = {"embed.weight", "embed.bias", "head.weight", "head.bias"}  # the router's output is not in the loss
    assert set(records[1]["grad_norms"]) == expected and records[1]["grad_norm"] > 0


def test_gradients_without_the_weight_check_take_two_columns_and_read_as_before():
    model = _Two()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=3, grads=True, weights=False, layers="blocks")
    assert session._ring_grad.shape == (3, 6, 2) and session._grad.shape == (6, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    norms = []
    for _ in range(5):
        model(torch.randn(3, 4)).backward()
        norms.append({n: float(p.grad.norm()) for n, p in model.named_parameters()})
        opt.step()
        opt.zero_grad()
    session.close()
    assert [r["step"] for r in sink.records] == list(range(5))
    for record, expected in zip(sink.records, norms):
        assert record["grad_norms"] == pytest.approx(expected, rel=1e-6)
        assert "weight_norms" not in record and set(record["layers"]) == {"blocks.0", "blocks.1"}


# --- Item 5: a session made before to_empty ---------------------------------------------------------


def test_a_session_made_before_to_empty_says_why_at_the_first_step():
    # `to_empty` replaces the meta parameters with new objects; the optimizer, made after, steps those,
    # so the session counts no step. Before, the first word was a warning after ten steps naming
    # DeepSpeed and Megatron; now the first step says the parameters were replaced.
    with torch.device("meta"):
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=2, weights=1)
    model.to_empty(device="cpu")
    for p in model.parameters():
        nn.init.normal_(p, std=0.1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.warns(RuntimeWarning, match="replaced the ones it had when tensorquill.init ran") as caught:
        model(torch.randn(3, 4)).pow(2).mean().backward()
        opt.step()
    assert sum("TensorQuill" in str(w.message) for w in caught) == 1
    for _ in range(11):  # no second warning, and the old one with another cause is not given
        model(torch.randn(3, 4)).pow(2).mean().backward()
        opt.step()
    session.close()
    assert sink.records == []  # nothing is written for the old parameters, no zero among it


def test_an_optimizer_of_parameters_assigned_after_init_is_warned_of_once_and_the_lead_still_records():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=2)
    main = torch.optim.SGD(model[0].parameters(), lr=0.1)
    model[2] = nn.Linear(4, 1)  # a head assigned after init, stepped by an optimizer of its own
    head = torch.optim.SGD(model[2].parameters(), lr=0.1)
    with pytest.warns(RuntimeWarning, match="replaced the ones it had"):
        model(torch.randn(3, 4)).pow(2).mean().backward()
        head.step()  # before the main optimizer has ever stepped
    main.step()
    for _ in range(2):
        model(torch.randn(3, 4)).pow(2).mean().backward()
        main.step()
        head.step()
    session.close()
    assert [r["step"] for r in sink.records] == [0, 1, 2]


# --- Item 5: clipping inside a compiled optimizer step ---------------------------------------------


def test_clipping_inside_a_compiled_step_does_not_recompile_at_every_step():
    # Before, `record_grad_norm` read and wrote its call count while traced, Dynamo guarded on the
    # count, and the step recompiled at steps 1 to 7, then hit the recompile limit and ran eagerly.
    # The control without TensorQuill compiles once; so does the watched step now.
    from torch._dynamo.utils import counters

    def run(watched):
        torch._dynamo.reset()
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1))
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sink = tensorquill.MemorySink()
        session = tensorquill.init(model, sink=sink, every=5) if watched else None

        @torch.compile(backend="eager")
        def update():
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        counters.clear()
        graphs, truth = [], []
        for _ in range(10):
            model(torch.randn(4, 8)).pow(2).mean().backward()
            truth.append(math.sqrt(sum(float(p.grad.norm()) ** 2 for p in model.parameters())))
            before = counters["stats"]["unique_graphs"]
            update()
            graphs.append(counters["stats"]["unique_graphs"] - before)
            opt.zero_grad()
        if session is not None:
            session.close()
        torch._dynamo.reset()
        return graphs, [r.get("clip_grad_norm") for r in sink.records], truth

    control, _, _ = run(False)
    graphs, clips, truth = run(True)
    assert sum(graphs[1:]) == 0 and sum(control[1:]) == 0, (graphs, control)  # nothing compiles after the first step
    assert clips == pytest.approx(truth, rel=1e-5)  # and the recorded pre-clipping norm is the true one

    # The first step against the control. What was observed, on CPU with the `eager` backend (Nov 23, 2026):
    # - PyTorch 2.15.0: `clip_grad_norm_` compiled alone makes a graph, and the watched step compiles as
    #   many graphs as the control.
    # - PyTorch 2.6.0: `clip_grad_norm_` compiled alone makes no graph (Dynamo breaks on
    #   `_group_tensors_by_device_and_dtype`), and the watched step compiles one graph more than the
    #   control: TensorQuill's own reading of the pre-clipping norm, once (accepted by the author as a
    #   documented difference).
    # The clip alone was measured on these two versions only, so the rule follows what this PyTorch does
    # with the clip alone, not its version: where the clip compiles, exact parity with the control; where it compiles
    # nothing, the same count or one more, never more than one.
    def clip_alone() -> int:
        torch._dynamo.reset()
        plain = nn.Linear(4, 1)
        plain(torch.randn(2, 4)).sum().backward()
        counters.clear()
        torch.compile(lambda: torch.nn.utils.clip_grad_norm_(plain.parameters(), 1.0), backend="eager")()
        made = counters["stats"]["unique_graphs"]
        torch._dynamo.reset()
        return made

    clip_graphs = clip_alone()
    difference = graphs[0] - control[0]
    if clip_graphs >= 1:
        assert difference == 0, (graphs, control, clip_graphs)
    else:
        assert difference in (0, 1), (graphs, control, clip_graphs)


def test_a_fullgraph_compiled_clip_no_longer_raises_into_training_at_the_recompile_limit():
    # Before, each step recompiled on the call count, and under fullgraph=True the eighth step raised
    # FailOnRecompileLimitHit out of the user's training step. Now it compiles once.
    from torch._dynamo.utils import counters

    # PyTorch's own limit, checked with no TensorQuill session open: on PyTorch 2.6, Dynamo cannot trace
    # `clip_grad_norm_` under fullgraph=True at all (`_group_tensors_by_device_and_dtype` returns a dict).
    torch._dynamo.reset()
    plain = nn.Linear(4, 1)
    plain(torch.randn(2, 4)).sum().backward()
    try:
        torch.compile(lambda: torch.nn.utils.clip_grad_norm_(plain.parameters(), 1.0), backend="eager", fullgraph=True)()
    except Exception as error:  # noqa: BLE001  whatever Dynamo raises there
        torch._dynamo.reset()
        pytest.skip(f"this PyTorch cannot compile clip_grad_norm_ under fullgraph=True without TensorQuill either: {type(error).__name__}")

    torch._dynamo.reset()
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=5)

    @torch.compile(backend="eager", fullgraph=True)
    def clip():
        return torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    counters.clear()
    try:
        for _ in range(10):
            model(torch.randn(4, 8)).pow(2).mean().backward()
            clip()
            opt.step()
            opt.zero_grad()
    finally:
        session.close()
        torch._dynamo.reset()
    assert counters["stats"]["unique_graphs"] == 1
    assert len(sink.records) == 10 and all("clip_grad_norm" in r for r in sink.records)


# --- Item 7 (contract part 7): `layers_not_read` and `moe_source` on the record ---------------------

import json  # noqa: E402

from tensorquill import flight  # noqa: E402


class _Out(nn.Module):
    """A block whose output is unreadable (`bad(step)` true: a sparse tensor) or a plain tensor."""

    def __init__(self, bad) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.bad = bad
        self.calls = 0

    def forward(self, x):
        y = self.lin(x)
        bad = self.bad(self.calls)
        self.calls += 1
        return y.to_sparse() if bad else y


class _Blocks(nn.Module):
    def __init__(self, bad0, bad1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Out(bad0), _Out(bad1)])

    def forward(self, x):
        total = 0
        for block in self.blocks:
            out = block(x)
            total = total + (out.to_dense() if out.layout != torch.strided else out).sum()
        return total


def _layer_records(bad0, bad1, steps=4, every=2):
    model = _Blocks(bad0, bad1)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=every, layers=["blocks.0", "blocks.1"])
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(steps):
        model(torch.randn(3, 4)).backward()
        opt.step()
        opt.zero_grad()
    session.close()
    return sink.records


NEVER, ALWAYS = (lambda step: False), (lambda step: True)
SPARSE = "its output is a sparse_coo tensor, not a dense one"


def test_layers_not_read_names_the_module_that_is_never_read_once_on_the_first_record():
    records = _layer_records(NEVER, ALWAYS)
    assert records[0]["layers_not_read"] == {"blocks.1": SPARSE}
    assert all("layers_not_read" not in r for r in records[1:])
    assert all(set(r["layers"]) == {"blocks.0"} for r in records)


def test_layers_not_read_names_every_module_when_none_is_read():
    records = _layer_records(ALWAYS, ALWAYS)
    assert records[0]["layers_not_read"] == {"blocks.0": SPARSE, "blocks.1": SPARSE}
    assert all(r["layers"] == {} for r in records)
    assert all("layers_not_read" not in r for r in records[1:])


def test_a_module_unread_on_some_steps_and_read_on_others_is_not_named():
    # Unreadable at the first step of the block and read on the others: a gap, not a limitation of
    # the capture. Its records lack it on that step only, as v1 states.
    # Beside it, a module never read, which is named.
    records = _layer_records(ALWAYS, lambda step: step == 0, steps=4, every=4)
    assert records[0]["layers_not_read"] == {"blocks.0": SPARSE}
    assert all("layers_not_read" not in r for r in records[1:])
    assert "blocks.1" not in records[0]["layers"] and all("blocks.1" in r["layers"] for r in records[1:])


def test_layers_not_read_travels_in_extra_json_as_moe_refused_does():
    records = _layer_records(NEVER, ALWAYS)
    row = flight.record_to_row(records[0])
    assert json.loads(row["extra_json"])["layers_not_read"] == {"blocks.1": SPARSE}
    assert flight.row_to_record(row)["layers_not_read"] == {"blocks.1": SPARSE}


SOURCES = {
    "logits": {"form": "scores", "point": "requested"},
    "integer ids": {"form": "indices", "point": "unknown"},
    "indices=": {"form": "indices", "point": "unknown"},
    "routing map": {"form": "map", "point": "unknown"},
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", [None, "aot_eager"], ids=["eager", "compiled"])
@pytest.mark.parametrize("path", list(PATHS))
def test_moe_source_says_the_form_and_the_point_once_on_the_first_record(path, backend, device):
    records, _, _, _ = _count(path, device, backend)
    assert records[0]["moe_source"] == {"router": SOURCES[path]}
    assert all("moe_source" not in r for r in records[1:])
    # Counts are as before: the two slots added for the form are not in `moe`.
    assert all(len(r["moe"]["router"]) == E for r in records)


def test_moe_source_never_says_dispatched_and_travels_in_extra_json():
    for path in PATHS:
        records, _, _, _ = _count(path, "cpu", None)
        assert records[0]["moe_source"]["router"]["point"] != "dispatched"
        row = flight.record_to_row(records[0])
        assert flight.row_to_record(row)["moe_source"] == {"router": SOURCES[path]}


def test_a_refused_router_has_no_moe_source_and_the_counted_one_beside_it_has():
    torch.manual_seed(0)
    model = _Net(lambda logits: logits)
    model.other = _Gate(lambda logits: logits[..., :3])  # logits of another width: refused
    forward = model.forward
    model.forward = lambda x: (model.other(model.embed(x)), forward(x))[1]
    sink = tensorquill.MemorySink()
    with pytest.warns(RuntimeWarning, match="not counting the router 'other'"):
        session = tensorquill.init(model, sink=sink, every=2, capture="routers",
                                   routers={"router": Router(E, K), "other": Router(E, K)})
        _router_steps(model, 2)
        session.close()
    assert "other" in sink.records[0]["moe_refused"]
    assert sink.records[0]["moe_source"] == {"router": SOURCES["logits"]}


def test_moe_source_waits_for_the_first_step_that_counted_the_router():
    torch.manual_seed(0)
    model = _Net(lambda logits: logits)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, capture="routers", routers={"router": Router(E, K)})
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for step in range(3):
        if step == 0:
            model.head(model.embed(torch.randn(6, 20, 4))).pow(2).mean().backward()  # the router not called
        else:
            model(torch.randn(6, 20, 4)).backward()
        opt.step()
        opt.zero_grad()
    session.close()
    assert [("moe_source" in r) for r in sink.records] == [False, True, False]
    assert sink.records[1]["moe_source"] == {"router": SOURCES["logits"]}
