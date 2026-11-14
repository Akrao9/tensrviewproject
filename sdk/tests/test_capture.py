import json
import math

import pytest
import torch
from torch import nn

import tensorlens


def train(model, session_steps, opt=None, batch=16, dim=8):
    opt = opt or torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(session_steps):
        opt.zero_grad()
        model(torch.randn(batch, dim)).sum().backward()
        opt.step()
    return opt


def test_layer_statistics_match_a_direct_calculation():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 512))
    sink = tensorlens.MemorySink()
    session = tensorlens.init(model, sink=sink, every=1, auto_step=False)
    x = torch.randn(5, 4)
    expected0 = model[0](x).detach()
    expected2 = model(x).detach()
    model(x).sum().backward()
    session.step()
    session.close()

    (record,) = sink.records
    assert list(record["layers"]) == ["0", "2"]  # leaf modules with parameters
    for name, expected in (("0", expected0), ("2", expected2)):
        stats = record["layers"][name]
        assert stats["mean"] == pytest.approx(expected.mean().item(), rel=1e-6)
        assert stats["std"] == pytest.approx(expected.std().item(), rel=1e-6)
        assert stats["abs_max"] == pytest.approx(expected.abs().max().item(), rel=1e-6)
        assert stats["zero_frac"] == pytest.approx((expected == 0).float().mean().item(), rel=1e-6)


def test_gradient_norms_match_and_total_is_their_root_sum_square():
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    sink = tensorlens.MemorySink()
    session = tensorlens.init(model, sink=sink, every=1, auto_step=False)
    model(torch.randn(6, 4)).sum().backward()
    expected = {name: p.grad.norm().item() for name, p in model.named_parameters()}
    session.step()
    session.close()

    record = sink.records[0]
    assert record["grad_norms"] == pytest.approx(expected, rel=1e-6)
    assert record["grad_norm"] == pytest.approx(math.sqrt(sum(v * v for v in expected.values())), rel=1e-6)


def test_optimizer_hooks_count_steps_and_flush_every_n(toy_moe):
    sink = tensorlens.MemorySink()
    session = tensorlens.init(toy_moe, sink=sink, every=4)
    train(toy_moe, 10)
    session.close()
    assert [record["step"] for record in sink.records] == list(range(10))
    assert all(record["grad_norm"] > 0 for record in sink.records)


def test_other_optimizers_are_ignored(toy_moe):
    sink = tensorlens.MemorySink()
    session = tensorlens.init(toy_moe, sink=sink, every=1)
    other = nn.Linear(3, 3)
    other_opt = torch.optim.SGD(other.parameters(), lr=0.1)
    other(torch.randn(2, 3)).sum().backward()
    other_opt.step()
    session.close()
    assert sink.records == []


def test_expert_counts_sum_to_tokens_and_match_the_router(toy_moe):
    x = torch.randn(32, 8)
    _, choice = toy_moe.router(torch.relu(toy_moe.inp(x)))
    sink = tensorlens.MemorySink()
    session = tensorlens.init(toy_moe, sink=sink, every=1, routers={"router": 4}, auto_step=False)
    toy_moe(x).sum().backward()
    session.step()
    session.close()
    counts = sink.records[0]["moe"]["router"]
    assert sum(counts) == 32
    assert counts == torch.bincount(choice, minlength=4).tolist()


def test_router_counts_accumulate_across_microbatches_and_reset_on_step(toy_moe):
    x1 = torch.randn(16, 8)
    x2 = torch.randn(16, 8)
    _, c1 = toy_moe.router(torch.relu(toy_moe.inp(x1)))
    _, c2 = toy_moe.router(torch.relu(toy_moe.inp(x2)))
    expected = (torch.bincount(c1, minlength=4) + torch.bincount(c2, minlength=4)).tolist()

    sink = tensorlens.MemorySink()
    session = tensorlens.init(toy_moe, sink=sink, every=1, routers={"router": 4}, auto_step=False)
    toy_moe(x1).sum().backward()
    toy_moe(x2).sum().backward()
    session.step(0)
    # Step 1: router skipped entirely
    session.step(1)
    session.close()

    assert sink.records[0]["moe"]["router"] == expected
    assert sink.records[0]["moe_stats"]["router"]["routed_tokens"] == 32
    assert sink.records[1]["moe"]["router"] == [0, 0, 0, 0]
    assert sink.records[1]["moe_stats"]["router"]["routed_tokens"] == 0
    assert sink.records[1]["moe_stats"]["router"]["dead_experts"] == 0


def test_a_layer_that_did_not_run_is_left_out_and_a_nan_layer_is_kept():
    class Branchy(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Linear(2, 2)
            self.unused = nn.Linear(2, 2)

        def forward(self, x):
            return self.used(x)

    model = Branchy()
    sink = tensorlens.MemorySink()
    session = tensorlens.init(model, sink=sink, every=1, auto_step=False)
    with torch.no_grad():
        model.used.weight.fill_(float("nan"))
    model(torch.ones(1, 2)).sum().backward()
    session.step()
    session.close()
    layers = sink.records[0]["layers"]
    assert "unused" not in layers
    assert math.isnan(layers["used"]["mean"])
    assert "unused.weight" not in sink.records[0]["grad_norms"]


def test_explicit_step_numbers_and_auto_step_guard(toy_moe):
    sink = tensorlens.MemorySink()
    session = tensorlens.init(toy_moe, sink=sink, every=8, auto_step=False)
    for step in (100, 101, 150):
        toy_moe(torch.randn(4, 8)).sum().backward()
        session.step(step)
    session.close()
    assert [record["step"] for record in sink.records] == [100, 101, 150]

    auto = tensorlens.init(nn.Linear(2, 2), sink=tensorlens.MemorySink())
    with pytest.raises(RuntimeError):
        auto.step()
    auto.close()


def test_init_rejects_a_compiled_model_and_unknown_names(toy_moe):
    with pytest.raises(ValueError, match="before torch.compile"):
        tensorlens.init(torch.compile(toy_moe, backend="eager"), sink=tensorlens.MemorySink())
    with pytest.raises(ValueError, match="no such modules"):
        tensorlens.init(toy_moe, layers=["nope"], sink=tensorlens.MemorySink())
    with pytest.raises(ValueError, match="no such router"):
        tensorlens.init(toy_moe, routers={"nope": 4}, sink=tensorlens.MemorySink())


def test_jsonl_sink_writes_standard_json_with_non_finite_values(tmp_path):
    path = tmp_path / "out" / "run.jsonl"
    sink = tensorlens.JsonlSink(path)
    sink.write([{"step": 0, "layers": {"a": {"mean": float("nan"), "abs_max": float("inf")}}}])
    sink.close()
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"step": 0, "layers": {"a": {"mean": "nan", "abs_max": "inf"}}}


def test_a_failing_sink_is_reported_on_close(toy_moe):
    class Broken:
        def write(self, records):
            raise OSError("disk full")

        def close(self):
            pass

    session = tensorlens.init(toy_moe, sink=Broken(), every=1)
    train(toy_moe, 1)
    with pytest.raises(RuntimeError, match="writer thread failed"):
        session.close()


def test_single_element_layer_output_has_zero_std_not_nan():
    container = nn.Sequential(nn.Linear(4, 1))
    sink = tensorlens.MemorySink()
    session = tensorlens.init(container, sink=sink, every=1, auto_step=False)
    container(torch.ones(1, 4)).sum().backward()
    session.step()
    session.close()
    assert sink.records[0]["layers"]["0"]["std"] == 0.0


def test_float_logit_router_and_moe_stats():
    class GateMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.gate(x)

    model = GateMoE()
    with torch.no_grad():
        model.gate.weight.zero_()
        model.gate.weight[0, 0] = 10.0
        model.gate.weight[2, 0] = 5.0

    sink = tensorlens.MemorySink()
    session = tensorlens.init(
        model,
        sink=sink,
        every=1,
        routers={"gate": tensorlens.Router(4, top_k=2)},
        auto_step=False,
    )
    model(torch.ones(8, 4)).sum().backward()
    session.step()
    session.close()

    record = sink.records[0]
    assert record["moe"]["gate"] == [8, 0, 8, 0]
    stats = record["moe_stats"]["gate"]
    assert stats["routed_tokens"] == 16
    assert stats["dead_experts"] == 2
    assert stats["dead_frac"] == pytest.approx(0.5)
    assert stats["entropy"] == pytest.approx(math.log(2.0))
    assert stats["diversity"] == pytest.approx(2.0 / 4.0)


def test_cpu_replica_spread_does_not_alias_ring_buffer_when_writer_is_paused():
    import threading

    model = nn.Linear(4, 2, bias=False)
    sink = tensorlens.MemorySink()
    session = tensorlens.init(model, sink=sink, every=1, auto_step=False)

    # Pause the writer thread by blocking it on an internal gate item before queuing two flushes.
    gate = threading.Event()
    entered = threading.Event()
    orig_decode = session._decode

    def gated_decode(*args, **kwargs):
        entered.set()
        gate.wait(timeout=5.0)
        return orig_decode(*args, **kwargs)

    session._decode = gated_decode  # type: ignore[method-assign]

    model(torch.ones(2, 4)).sum().backward()
    session.record_replica_spread(torch.tensor([[0.125, 0.125]], dtype=torch.float64))
    session.step(0)
    entered.wait(timeout=5.0)

    # While step 0 is waiting in the writer, step 1 overwrites row 0 of `_ring_spread`.
    model(torch.ones(2, 4)).sum().backward()
    session.record_replica_spread(torch.tensor([[0.5, 0.5]], dtype=torch.float64))
    session.step(1)

    gate.set()
    session.close()

    assert sink.records[0]["replica_spread"]["weight"] == [0.125, 0.125]
    assert sink.records[1]["replica_spread"]["weight"] == [0.5, 0.5]


def test_session_setup_allocates_no_float64_buffers_by_default_and_keeps_mps_spread_on_cpu(
    monkeypatch,
):
    orig_zeros = torch.zeros
    f64_calls = []

    def spy_zeros(*args, **kwargs):
        if kwargs.get("dtype") == torch.float64:
            f64_calls.append(kwargs.get("device"))
        return orig_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", spy_zeros)
    model = nn.Linear(4, 2)
    session = tensorlens.init(model, sink=tensorlens.MemorySink(), every=2, auto_step=False)
    assert f64_calls == []
    assert session._spread is None and session._ring_spread is None

    # Simulate an MPS session device and verify lazy replica buffers allocate on CPU, not MPS.
    session.device = torch.device("mps")
    spread, ring_spread = session._ensure_spread_buffers()
    assert spread.device.type == "cpu"
    assert ring_spread.device.type == "cpu"
    session.close()


def test_gradient_gating_and_router_only_capture(toy_moe, monkeypatch):
    import tensorlens.capture as cap

    norm_calls = 0
    orig_foreach_norm = cap._foreach_norm

    def spy_foreach_norm(*args, **kwargs):
        nonlocal norm_calls
        norm_calls += 1
        return orig_foreach_norm(*args, **kwargs)

    monkeypatch.setattr(cap, "_foreach_norm", spy_foreach_norm)

    # grads=3 gates gradient norm computation before running _foreach_norm. The default
    # weight check (every 50 steps) does not run in 6 steps.
    sink = tensorlens.MemorySink()
    with tensorlens.init(toy_moe, sink=sink, every=2, grads=3, routers={"router": 4}):
        train(toy_moe, 6)
    assert norm_calls == 2
    assert not any("weight_norms" in r for r in sink.records)
    assert [r["grad_norm"] is not None for r in sink.records] == [
        False,
        False,
        True,
        False,
        False,
        True,
    ]

    # Router-only (layers=(), grads=False) computes no layer or gradient reductions
    norm_calls = 0
    router_sink = tensorlens.MemorySink()
    with tensorlens.init(
        toy_moe, sink=router_sink, every=2, layers=(), grads=False, routers={"router": 4}
    ):
        train(toy_moe, 4)
    assert norm_calls == 0
    assert all(r["layers"] == {} and r["grad_norm"] is None for r in router_sink.records)
    assert all(sum(r["moe"]["router"]) == 16 for r in router_sink.records)


@pytest.mark.parametrize("layers, grads", [(None, False), ((), True)])
def test_cpu_flush_owns_single_stats_buffer(layers, grads):
    import threading

    model = nn.Sequential(nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(1)
    gate = threading.Event()
    entered = threading.Event()
    sink = tensorlens.MemorySink()
    with tensorlens.init(model, sink=sink, layers=layers, grads=grads, every=1, auto_step=False) as session:
        decode = session._decode

        def gated_decode(*args):
            entered.set()
            gate.wait(timeout=5)
            return decode(*args)

        session._decode = gated_decode
        try:
            for sample in (1.0, 3.0):
                model.zero_grad(set_to_none=True)
                model(torch.tensor([[sample]])).sum().backward()
                session.step()
                assert entered.wait(timeout=5)
        finally:
            gate.set()

    if grads:
        assert [record["grad_norm"] for record in sink.records] == [1.0, 3.0]
    else:
        assert [record["layers"]["0"]["mean"] for record in sink.records] == [1.0, 3.0]


def test_manual_gradient_capture_preserves_automatic_cadence():
    model = nn.Linear(1, 1, bias=False)
    sink = tensorlens.MemorySink()
    with tensorlens.init(model, sink=sink, auto_step=False, grads=3) as session:
        for i in range(4):
            model.zero_grad(set_to_none=True)
            model(torch.ones(1, 1)).sum().backward()
            if i == 0:
                session.record_grads()
            session.step()

    assert [record["grad_norm"] for record in sink.records] == [1.0, None, 1.0, None]


def test_weight_norms_learning_rate_start_step_and_top_k(toy_moe):
    sink = tensorlens.MemorySink()
    opt = torch.optim.SGD(toy_moe.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    session = tensorlens.init(
        toy_moe, sink=sink, every=2, start_step=100, weights=1,
        routers={"router": tensorlens.Router(4, top_k=1)},
    )
    for _ in range(3):
        opt.zero_grad()
        toy_moe(torch.randn(16, 8)).sum().backward()
        opt.step()
        sched.step()
    session.close()

    assert [r["step"] for r in sink.records] == [100, 101, 102]
    assert [r["lr"] for r in sink.records] == [[0.1], [0.05], [0.025]]
    assert all(r["moe_top_k"] == {"router": 1} for r in sink.records)
    last = sink.records[-1]["weight_norms"]
    for name, param in toy_moe.named_parameters():
        assert last[name] == pytest.approx(param.detach().norm().item(), rel=1e-5)


def test_weights_false_and_manual_step_learning_rate(toy_moe):
    sink = tensorlens.MemorySink()
    session = tensorlens.init(toy_moe, sink=sink, every=1, weights=False, auto_step=False)
    toy_moe(torch.randn(16, 8)).sum().backward()
    session.step(lr=3e-4)
    session.close()
    assert "weight_norms" not in sink.records[0]
    assert sink.records[0]["lr"] == [3e-4]


def test_weight_check_cadence_turns_on_gradients_for_its_steps(toy_moe):
    sink = tensorlens.MemorySink()
    with tensorlens.init(toy_moe, sink=sink, every=3, grads=False, weights=2):
        train(toy_moe, 6)
    # Checks on the 2nd, 4th and 6th steps, each with that step's gradient norms; the first
    # check has nothing to compare with.
    assert ["weight_norms" in r for r in sink.records] == [False, True, False, True, False, True]
    assert [r["grad_norm"] is not None for r in sink.records] == [False, True, False, True, False, True]
    assert ["weights_unchanged" in r for r in sink.records] == [False, False, False, True, False, True]
    with tensorlens.init(toy_moe, sink=tensorlens.MemorySink(), weights=False) as off:
        assert off.weight_every == 0
    with pytest.raises(ValueError):
        tensorlens.init(toy_moe, sink=tensorlens.MemorySink(), weights=-1)


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(dim, dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, h):
        return h + self.mlp(torch.relu(self.attn(h)))


class _Stack(nn.Module):
    def __init__(self, dim: int = 8, n: int = 3) -> None:
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.layers = nn.ModuleList(_Block(dim) for _ in range(n))
        self.heads = nn.ModuleList(nn.Linear(dim, 1) for _ in range(2))  # leaf modules
        self.out = nn.Linear(dim, 1)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.out(h) + sum(head(h) for head in self.heads)


def test_default_layers_are_the_repeated_blocks_and_all_keeps_every_leaf(toy_moe):
    model = _Stack()
    with tensorlens.init(model, sink=tensorlens.MemorySink()) as session:
        assert session.layer_names == ["layers.0", "layers.1", "layers.2"]
    with tensorlens.init(model, sink=tensorlens.MemorySink(), layers="all") as session:
        assert len(session.layer_names) == 1 + 3 * 2 + 2 + 1
    # A model with no list of composite blocks falls back to every leaf module.
    with tensorlens.init(toy_moe, sink=tensorlens.MemorySink()) as session:
        with tensorlens.init(toy_moe, sink=tensorlens.MemorySink(), layers="all") as every:
            assert session.layer_names == every.layer_names
    with pytest.raises(ValueError):
        tensorlens.init(model, sink=tensorlens.MemorySink(), layers="leaves")


def test_block_statistics_match_and_compile_cleanly():
    torch.manual_seed(0)
    model = _Stack()
    sink = tensorlens.MemorySink()
    session = tensorlens.init(model, sink=sink, every=2, auto_step=False)
    compiled = torch.compile(model, backend="aot_eager")
    x = torch.randn(16, 8)
    torch._dynamo.utils.counters.clear()
    with torch._dynamo.config.patch(error_on_recompile=True):
        for _ in range(2):
            compiled(x).sum().backward()
            session.step()
    session.close()
    assert dict(torch._dynamo.utils.counters["graph_break"]) == {}
    with torch.no_grad():
        h = model.embed(x)
        for layer in model.layers:
            h = layer(h)
        expected = {"mean": h.mean().item(), "std": h.std().item(), "abs_max": h.abs().max().item()}
    got = sink.records[-1]["layers"]["layers.2"]
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, rel=1e-5)


def test_capture_modes(toy_moe):
    # MoE models default to router-only capture; dense models to full capture.
    with tensorlens.init(toy_moe, sink=tensorlens.MemorySink(), routers={"router": 4}) as s:
        assert (s.layer_names, s.grad_every, s.weight_every) == ([], 0, 0)
    with tensorlens.init(_Stack(), sink=tensorlens.MemorySink()) as s:
        assert (s.layer_names, s.grad_every, s.weight_every) == (
            ["layers.0", "layers.1", "layers.2"], 1, 50
        )
    with tensorlens.init(toy_moe, sink=tensorlens.MemorySink(), capture="full", routers={"router": 4}) as s:
        assert s.grad_every == 1 and s.weight_every == 50 and s.layer_names
    # Explicit settings override the mode.
    with tensorlens.init(toy_moe, sink=tensorlens.MemorySink(), routers={"router": 4}, grads=10) as s:
        assert (s.layer_names, s.grad_every) == ([], 10)
    with pytest.raises(ValueError):
        tensorlens.init(_Stack(), sink=tensorlens.MemorySink(), capture="routers")
    with pytest.raises(ValueError):
        tensorlens.init(_Stack(), sink=tensorlens.MemorySink(), capture="everything")


def test_clip_grad_norm_is_recorded_in_router_only_mode(toy_moe):
    original = torch.nn.utils.clip_grad_norm_
    other = nn.Linear(4, 1)
    sink = tensorlens.MemorySink()
    session = tensorlens.init(toy_moe, sink=sink, every=2, routers={"router": 4})
    assert torch.nn.utils.clip_grad_norm_ is not original
    opt = torch.optim.SGD(toy_moe.parameters(), lr=0.01)
    returned = []
    for _ in range(3):
        opt.zero_grad()
        toy_moe(torch.randn(16, 8)).sum().backward()
        returned.append(torch.nn.utils.clip_grad_norm_(toy_moe.parameters(), 0.5).item())
        other(torch.randn(2, 4)).sum().backward()
        torch.nn.utils.clip_grad_norm_(other.parameters(), 0.5)  # another model: ignored
        opt.step()
    session.close()
    assert torch.nn.utils.clip_grad_norm_ is original  # restored on close
    assert [r["grad_norm"] for r in sink.records] == [None, None, None]
    assert [r["clip_grad_norm"] for r in sink.records] == pytest.approx(returned, rel=1e-6)


def test_record_grad_norm_by_hand_and_spikes_without_per_parameter_norms(toy_moe):
    sink = tensorlens.MemorySink()
    session = tensorlens.init(
        toy_moe, sink=sink, every=8, routers={"router": 4}, clip_norm=False, auto_step=False
    )
    assert torch.nn.utils.clip_grad_norm_.__module__ == "torch.nn.utils.clip_grad"
    for step in range(30):
        session.record_grad_norm(40.0 if step == 25 else 1.0 + 0.01 * (step % 3))  # Megatron-style float
        session.step()
    session.close()
    assert sink.records[25]["clip_grad_norm"] == pytest.approx(40.0)
    hit = tensorlens.detect_grad_spikes(sink.records)
    assert hit["step"] == 25 and hit["parameters"] == []


def test_lightning_gradient_clipping_is_recorded(toy_moe):
    precision = pytest.importorskip("lightning.pytorch.plugins.precision")
    sink = tensorlens.MemorySink()
    session = tensorlens.init(toy_moe, sink=sink, every=1, routers={"router": 4})
    opt = torch.optim.SGD(toy_moe.parameters(), lr=0.01)
    opt.zero_grad()
    toy_moe(torch.randn(16, 8)).sum().backward()
    expected = torch.nn.utils.get_total_norm([p.grad for p in toy_moe.parameters() if p.grad is not None])
    precision.Precision().clip_grad_by_norm(opt, 0.5)  # what Trainer(gradient_clip_val=0.5) calls
    opt.step()
    session.close()
    assert sink.records[0]["clip_grad_norm"] == pytest.approx(expected.item(), rel=1e-6)


def test_activation_spike_is_found_through_real_capture():
    torch.manual_seed(0)
    model = _Stack(dim=16)
    sink = tensorlens.MemorySink()
    session = tensorlens.init(model, sink=sink, every=10)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    for step in range(80):
        if step == 70:
            with torch.no_grad():
                model.layers[1].mlp.weight.mul_(50.0)  # the planted failure
        opt.zero_grad()
        model(torch.randn(32, 16)).pow(2).mean().backward()
        opt.step()
    session.close()
    hits = {h["kind"]: h for h in tensorlens.detect_activation_anomalies(sink.records)}
    spike = hits["activation_spike"]
    assert spike["step"] == 70
    assert spike["modules"][0]["module"] in ("layers.1", "layers.2")


def test_log_records_scalars_for_the_step_in_progress():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorlens.MemorySink()
    session = tensorlens.init(model, sink=sink, every=2)
    peaks = torch.arange(1.0, 7.0)
    losses = []
    for step in range(3):
        loss = model(torch.ones(2, 4)).pow(2).mean()
        loss.backward()
        session.log({"train/loss": loss}, tokens=8, peak=peaks[step])
        session.log(tokens=16)  # the last value for a name wins
        losses.append(loss.item())
        opt.step()
        opt.zero_grad()
        if step == 0:
            peaks.zero_()  # a logged view is copied, so later changes do not reach it
    session.log(after_last_step=1.0)  # belongs to a step that never ends: not written
    session.close()

    scalars = [record["scalars"] for record in sink.records]
    assert [s["tokens"] for s in scalars] == [16, 16, 16]
    assert [s["peak"] for s in scalars] == [1.0, 0.0, 0.0]
    assert [s["train/loss"] for s in scalars] == pytest.approx(losses)
    assert all("after_last_step" not in s for s in scalars)
    with pytest.raises(RuntimeError, match="closed"):
        session.log(x=1.0)


def test_log_rejects_non_scalars():
    session = tensorlens.init(nn.Linear(2, 2), sink=tensorlens.MemorySink())
    with pytest.raises(ValueError, match="one value"):
        session.log(x=torch.ones(3))
    with pytest.raises(TypeError, match="number"):
        session.log(image=object())
    with pytest.raises(TypeError, match="names"):
        session.log({"": 1.0})
    session.close()
