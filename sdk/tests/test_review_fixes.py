"""Reproductions of review findings from Nov 14, 2026, each fixed."""

import math

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import tensorquill
from tensorquill import probe as probe_mod


class _Routed(nn.Module):
    def __init__(self, use_reentrant):
        super().__init__()
        self.inner = nn.Linear(8, 8)
        self.router = nn.Linear(8, 4)
        self.use_reentrant = use_reentrant

    def _block(self, x):
        h = self.inner(x)
        weights = self.router(h).softmax(-1).topk(2, dim=-1).values.sum(-1, keepdim=True)
        return h * weights

    def forward(self, x):
        return checkpoint(self._block, x, use_reentrant=self.use_reentrant)


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_activation_checkpointing_does_not_double_count_routing(use_reentrant):
    model = _Routed(use_reentrant)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, routers={"router": tensorquill.Router(4, top_k=2)})
    x = torch.randn(16, 8, requires_grad=use_reentrant)
    model(x).sum().backward()
    opt.step()
    session.close()
    assert sum(sink.records[0]["moe"]["router"]) == 16 * 2


def _checkpointed_records(backend, use_reentrant, hooks):
    """Records of three steps of `_Routed`, uncompiled (backend None) or compiled. Each step runs
    two forwards before one backward, so a backward that replayed a hook's write would leave the
    first forward's statistics, or count its tokens again."""
    torch.manual_seed(0)
    model = _Routed(use_reentrant)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    if hooks == "full":
        kwargs = {"capture": "full", "layers": ["inner", "router"]}
    else:
        indices = (lambda out: out.detach().topk(2, dim=-1).indices) if hooks == "indices" else None
        kwargs = {"routers": {"router": tensorquill.Router(4, top_k=2, indices=indices)}}
    session = tensorquill.init(model, sink=sink, **kwargs)
    runner = model if backend is None else torch.compile(model, backend=backend)
    torch._dynamo.utils.counters.clear()
    for _ in range(3):
        a, b = (torch.randn(16, 8, requires_grad=use_reentrant) for _ in range(2))
        (runner(a).sum() + runner(b).sum()).backward()
        opt.step()
        opt.zero_grad()
    session.close()
    assert dict(torch._dynamo.utils.counters["graph_break"]) == {}
    return [{key: r.get(key) for key in ("step", "moe", "layers")} for r in sink.records]


@pytest.mark.parametrize("hooks", ["routers", "indices", "full"])
@pytest.mark.parametrize("use_reentrant", [True, False])
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_checkpointed_hooks_under_compile_record_what_eager_records(backend, use_reentrant, hooks):
    # Under backend="eager" a checkpointed block holding a hooked module made backward() raise
    # "modified by an inplace operation": the block's saved inputs include the hook's buffer.
    compiled = _checkpointed_records(backend, use_reentrant, hooks)
    eager = _checkpointed_records(None, use_reentrant, hooks)
    assert [r["moe"] for r in compiled] == [r["moe"] for r in eager]
    if hooks == "routers":
        assert all(sum(r["moe"]["router"]) == 2 * 16 * 2 for r in compiled)
    for got, want in zip(compiled, eager):
        if want["layers"]:
            for name, stats in want["layers"].items():
                assert got["layers"][name] == pytest.approx(stats, rel=1e-5, abs=1e-7)


def _two_part():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    return model, [torch.optim.SGD(model[0].parameters(), lr=0.1), torch.optim.AdamW(model[1].parameters())]


def test_optimizers_that_clear_their_own_gradients_still_count_one_step_per_iteration():
    model, opts = _two_part()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, capture="full", every=2)
    for _ in range(4):
        model(torch.ones(2, 4)).sum().backward()
        for opt in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
    session.close()
    assert [r["step"] for r in sink.records] == [0, 1, 2, 3]
    assert all(len(r["grad_norms"]) == 4 for r in sink.records)  # both optimizers' parameters


def test_a_step_that_raised_does_not_stop_later_steps_from_counting():
    model = nn.Linear(4, 1)

    class Flaky(torch.optim.SGD):
        calls = 0

        def step(self, closure=None):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RuntimeError("transient")
            return super().step(closure)

    opt = Flaky(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink)
    model(torch.ones(1, 4)).sum().backward()
    with pytest.raises(RuntimeError, match="transient"):
        opt.step()
    for _ in range(2):
        opt.step()
    session.close()
    assert [r["step"] for r in sink.records] == [0, 1]


@pytest.mark.parametrize("kwargs", [{"layers": ()}, {}])
def test_probe_on_a_model_without_capture_buffers(kwargs):
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, probe=torch.ones(1, 4), probe_every=1, capture="full", **kwargs)
    model(torch.ones(1, 4)).sum().backward()
    opt.step()
    session.close()
    assert "error" not in sink.records[0]["probe"]


def test_closure_steps_record_gradients_after_the_closure_runs():
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, capture="full")

    def closure():
        opt.zero_grad()
        loss = model(torch.ones(2, 4)).sum()
        loss.backward()
        return loss

    for _ in range(2):
        opt.step(closure)
    session.close()
    assert all(r["grad_norm"] and r["grad_norm"] > 0 for r in sink.records)
    assert len(sink.records) == 2


def test_replica_with_nan_weights_is_reported():
    spreads = [{"run_id": "r", "step": 0, "replica_spread": {"w": [0.0, 0.0]}},
               {"run_id": "r", "step": 1, "replica_spread": {"w": ["nan", "nan"]}}]
    hit = tensorquill.detect_replica_drift(spreads)
    assert hit["step"] == 1 and hit["parameters"]["w"]["non_finite"]
    evidence = tensorquill.build_evidence(spreads)
    assert "NaN or Inf weights" in tensorquill.incident_summary(evidence)


def test_clipped_spike_is_found_from_the_pre_clipping_norm_and_clip_inf_is_non_finite():
    records = [
        {"run_id": "c", "step": s, "grad_norm": 1.0, "grad_norms": {"w": 1.0},
         "clip_grad_norm": 30.0 if s == 20 else 3.0}
        for s in range(25)
    ]
    hit = tensorquill.detect_grad_spikes(records)
    assert hit is not None and hit["step"] == 20 and hit["ratio"] == pytest.approx(10.0)
    records[22]["clip_grad_norm"] = "inf"
    bad = tensorquill.detect_non_finite(records)
    assert bad["step"] == 22 and "<total before clipping>" in bad["gradients"]


def test_probe_counts_rows_over_a_broadcast_batch():
    q, k = torch.randn(1, 2, 3, 4), torch.randn(3, 2, 3, 4)
    stats = probe_mod._attention_stats(q, k, None, False, None, map_size=3)
    assert stats["rows"] == 3 * 2 * 3


def test_old_probe_records_keep_their_entropy_curve_on_export(tmp_path, monkeypatch):
    rr = pytest.importorskip("rerun")
    old = {"histograms": {}, "attention": {"blocks.0": {
        "max_logit": [1.0], "entropy": [math.log(4) / 2], "max_entropy": math.log(4),
        "query_len": 4, "key_len": 4, "map": [[0.25] * 4] * 4}}}
    records = [{"run_id": "o", "step": 0, "probe": old}]
    logged = []
    real = rr.log
    monkeypatch.setattr(rr, "log", lambda p, *a, **kw: (logged.append((p, a)), real(p, *a, **kw)))
    tensorquill.export_rerun(tensorquill.build_evidence(records), tmp_path / "o.rrd", records=records)
    assert any(p == "/probe/entropy/blocks.0" for p, _ in logged)


def test_warns_when_only_another_optimizer_steps():
    # DeepSpeed ZeRO and Megatron step copies of the weights: the stepping optimizer
    # holds none of the model's parameters, so automatic counting would record nothing.
    model = nn.Linear(4, 1)
    copy = nn.Parameter(torch.zeros(5))
    optimizer = torch.optim.SGD([copy], lr=0.1)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, capture="full")
    with pytest.warns(RuntimeWarning, match="auto_step=False") as caught:
        for _ in range(25):
            model(torch.randn(2, 4)).sum().backward()
            copy.grad = torch.ones(5)
            optimizer.step()
    session.close()
    assert len(caught) == 1 and sink.records == []


def test_no_warning_when_the_models_optimizer_steps_too():
    import warnings

    model = nn.Linear(4, 1)
    other = torch.optim.SGD([nn.Parameter(torch.zeros(5, requires_grad=True))], lr=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tensorquill.init(model, sink=tensorquill.MemorySink(), capture="full")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(25):
            model(torch.randn(2, 4)).sum().backward()
            optimizer.step()
            other.param_groups[0]["params"][0].grad = torch.ones(5)
            other.step()
    session.close()
