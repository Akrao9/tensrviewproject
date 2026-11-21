"""Lane AD3 (wave AD): the capture and the probe.

- C22: two of the three overflow cases still counted and not read are read now (a squared norm
  past float32's range on MPS; logits near float32's largest value with both signs in one row),
  and the third (a BF16 logit past BF16's range) stays counted.
- Router counting allocates in proportion to the assignments, not experts by assignments.
- The ring allocates no gradient rows in the router-only mode.

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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_router_counts_under_aot_and_inductor_match_eager_and_make_no_matrix(device, backend, tmp_path, monkeypatch):
    if device == "mps" and backend == "inductor":
        pytest.skip("Inductor on MPS is not what this lane measures")
    # A cache of its own: other processes on this machine compile into the shared one at the same time.
    # Its precompiled header is shared too, and another process rebuilding it fails this compile.
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    import torch._inductor.config as inductor_config

    monkeypatch.setattr(inductor_config, "cpp_cache_precompile_headers", False, raising=False)
    torch._dynamo.reset()
    eager, _, largest, _ = _count("integer ids", device, None)
    records, _, _, _ = _count("integer ids", device, backend)
    assert [r["moe"]["router"] for r in records] == [r["moe"]["router"] for r in eager]
    assert [r["moe_unassigned"]["router"] for r in records] == [r["moe_unassigned"]["router"] for r in eager]
    assert largest.largest <= TOKENS * K * 2


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
