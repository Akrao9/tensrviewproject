import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import tensorquill
from tensorquill import probe as probe_mod


class Block(nn.Module):
    def __init__(self, width=16, heads=4, kv_heads=2, experts=4):
        super().__init__()
        self.heads, self.kv_heads, self.dim = heads, kv_heads, width // heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * kv_heads * self.dim)
        self.drop = nn.Dropout(0.5)
        self.router = nn.Linear(width, experts)
        self.mlp = nn.Linear(width, width)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.heads, self.dim).transpose(1, 2)
        k, v = self.kv(x).view(b, t, 2, self.kv_heads, self.dim).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + self.drop(a.transpose(1, 2).reshape(b, t, -1))
        gate = self.router(x).softmax(-1).topk(2, dim=-1).values.sum(-1, keepdim=True)
        return x + self.mlp(x) * gate


class Toy(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.blocks = nn.ModuleList(Block() for _ in range(layers))
        self.frozen = nn.Linear(16, 16)
        self.frozen.eval()  # kept in eval mode by the user; the probe must not unfreeze it
        self.head = nn.Linear(16, 32)

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.frozen(x))


def _reference(q, k, causal=True):
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        t = q.shape[-2]
        scores = scores.masked_fill(~torch.ones(t, t, dtype=torch.bool).tril(), -math.inf)
    p = scores.softmax(-1)
    ent = -(p * p.clamp_min(1e-30).log()).sum(-1)  # [batch, heads, rows]
    # Rows with one visible key (the first query under a causal mask) carry no choice.
    visible = torch.arange(1, q.shape[-2] + 1).float() if causal else torch.full((q.shape[-2],), float(k.shape[-2]))
    rich = visible >= 2
    share = (ent / visible.clamp_min(2).log())[..., rich].mean(dim=(0, 2))
    return scores.amax(dim=(0, 2, 3)), ent[..., rich].mean(dim=(0, 2)), share, p[0].mean(0)


def _train(model, session, steps, batch):
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(steps):
        model(batch).float().pow(2).mean().backward()
        opt.step()
        opt.zero_grad()


def test_probe_runs_on_cadence_and_records_histograms_and_attention():
    torch.manual_seed(0)
    model = Toy()
    fixed = torch.randint(0, 32, (2, 12))
    routers = {f"blocks.{i}.router": tensorquill.Router(4, top_k=2) for i in range(2)}
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=3, routers=routers, probe=fixed, probe_every=3)
    batch = torch.randint(0, 32, (2, 12))
    _train(model, session, 6, batch)
    session.close()

    assert [("probe" in r) for r in sink.records] == [False, False, True] * 2
    # The probe's forward pass does not leak into the step's router counts.
    assert all(sum(r["moe"]["blocks.0.router"]) == 2 * 12 * 2 for r in sink.records)
    assert model.training and not model.frozen.training and model.blocks[0].drop.training

    probe = sink.records[2]["probe"]
    assert set(probe["histograms"]) == {"blocks.0", "blocks.1"}
    hist = probe["histograms"]["blocks.0"]
    assert len(hist["counts"]) == 64 and sum(hist["counts"]) == 2 * 12 * 16 and hist["lo"] < hist["hi"]
    assert set(probe["attention"]) == {"blocks.0", "blocks.1"}
    att = probe["attention"]["blocks.1"]
    assert len(att["max_logit"]) == 4 and len(att["entropy"]) == 4
    assert att["query_len"] == att["key_len"] == 12 and len(att["map"]) == 12
    assert all(abs(sum(row) - 1) < 1e-5 for row in att["map"])  # pooled rows still sum to 1
    assert "attention_note" not in probe


def test_attention_stats_match_explicit_softmax_with_chunking(monkeypatch):
    torch.manual_seed(1)
    q, k = torch.randn(2, 4, 10, 8), torch.randn(2, 2, 10, 8)
    full = probe_mod._attention_stats(q, k, None, True, None, map_size=10)
    max_logit, entropy, share, attn = _reference(q, k)
    assert full["max_logit"] == pytest.approx(max_logit.tolist(), rel=1e-5)
    assert full["entropy"] == pytest.approx(entropy.tolist(), rel=1e-5)
    assert full["entropy_share"] == pytest.approx(share.tolist(), rel=1e-5)
    assert full["masked_rows"] == full["invalid_rows"] == 0 and full["rows"] == 2 * 4 * 10
    assert torch.allclose(torch.tensor(full["map"]), attn, atol=1e-6)

    monkeypatch.setattr(probe_mod, "CHUNK_ELEMENTS", 2 * 4 * 10 * 3)  # 3 query rows per chunk
    chunked = probe_mod._attention_stats(q, k, None, True, None, map_size=4)
    assert chunked["max_logit"] == pytest.approx(full["max_logit"], rel=1e-6)
    assert chunked["entropy"] == pytest.approx(full["entropy"], rel=1e-6)
    assert chunked["entropy_share"] == pytest.approx(full["entropy_share"], rel=1e-6)
    pooled = torch.tensor(chunked["map"])
    assert pooled.shape == (4, 4) and torch.allclose(pooled.sum(-1), torch.ones(4), atol=1e-5)

    # A boolean mask that hides the last key gives the same result as attending to 9 keys.
    mask = torch.ones(10, 10, dtype=torch.bool)
    mask[:, -1] = False
    masked = probe_mod._attention_stats(q, k, mask, False, None, map_size=10)
    ref = probe_mod._attention_stats(q, k[..., :9, :], None, False, None, map_size=10)
    assert masked["entropy"] == pytest.approx(ref["entropy"], rel=1e-5)


def test_probe_leaves_rng_and_flags_untouched_and_switches_flash_attention():
    model = Toy()
    model.config = type("Config", (), {"_attn_implementation": "flash_attention_2"})()
    seen = []
    original = model.forward

    def forward(ids):
        seen.append(model.config._attn_implementation)
        return original(ids)

    model.forward = forward
    session = tensorquill.init(model, sink=tensorquill.MemorySink(), probe=lambda m: m(torch.zeros(1, 4, dtype=torch.long)))
    torch.manual_seed(3)
    state = torch.get_rng_state()
    result = session._probe()
    assert torch.equal(torch.get_rng_state(), state)
    assert seen == ["sdpa"] and model.config._attn_implementation == "flash_attention_2"
    assert result["attention_note"] == "FlashAttention switched to SDPA for the probe"
    assert model.training and not model.frozen.training
    assert F.scaled_dot_product_attention is torch._C._nn.scaled_dot_product_attention
    session.close()


def test_histogram_handles_constant_and_non_finite_values():
    hist = probe_mod._histogram(torch.tensor([2.0, 2.0, float("nan"), float("inf")]), bins=4)
    assert hist["lo"] == 1.5 and hist["hi"] == 2.5 and sum(hist["counts"]) == 2 and hist["nonfinite"] == 2
    empty = probe_mod._histogram(torch.tensor([float("nan")]), bins=4)
    assert empty["counts"] == [0, 0, 0, 0] and empty["nonfinite"] == 1


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_probe_causes_no_graph_breaks_or_recompiles(backend):
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    model = Toy()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=2, probe=torch.zeros(2, 8, dtype=torch.long), probe_every=2)
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    batch = torch.randint(0, 32, (2, 8))
    with torch._dynamo.config.patch(error_on_recompile=True):
        _train(compiled, session, 6, batch)
    session.close()
    assert dict(torch._dynamo.utils.counters["graph_break"]) == {}
    assert sum("probe" in r for r in sink.records) == 3


def test_probe_input_forms():
    with pytest.raises(TypeError, match="probe must be"):
        probe_mod.probe_callable(3)
    calls = []
    fake = lambda *a, **kw: calls.append((a, kw))  # noqa: E731
    probe_mod.probe_callable({"input_ids": 1})(fake)
    probe_mod.probe_callable(torch.ones(1))(fake)
    assert calls[0] == ((), {"input_ids": 1}) and calls[1][0][0].shape == (1,)
    with pytest.raises(ValueError, match="probe_every"):
        tensorquill.init(nn.Linear(2, 2), sink=tensorquill.MemorySink(), probe=torch.ones(1, 2), probe_every=0)


def test_probe_is_written_to_jsonl_and_exported_to_rerun(tmp_path, monkeypatch):
    rr = pytest.importorskip("rerun")
    torch.manual_seed(0)
    model = Toy()
    path = tmp_path / "run.jsonl"
    session = tensorquill.init(
        model, sink=tensorquill.JsonlSink(path), every=2, probe=torch.zeros(1, 6, dtype=torch.long), probe_every=2
    )
    _train(model, session, 4, torch.randint(0, 32, (2, 6)))
    session.close()
    records = tensorquill.load_records(path)
    assert sum("probe" in r for r in records) == 2

    logged = []
    real_log = rr.log
    monkeypatch.setattr(rr, "log", lambda p, *a, **kw: (logged.append(p), real_log(p, *a, **kw)))
    evidence = tensorquill.build_evidence(records)
    tensorquill.export_rerun(evidence, tmp_path / "run.rrd", records=records)
    for layer in ("blocks.0", "blocks.1"):
        assert logged.count(f"/probe/histograms/{layer}") == 2
        assert logged.count(f"/probe/attention/{layer}") == 2
        assert f"/probe/max_logit/{layer}" in logged and f"/probe/entropy/{layer}" in logged


def test_uniform_attention_reads_as_full_entropy_despite_causal_mask_and_padding():
    q, k = torch.zeros(1, 2, 4, 8), torch.zeros(1, 2, 4, 8)  # every score equal: uniform
    causal = probe_mod._attention_stats(q, k, None, True, None, map_size=4)
    assert causal["entropy_share"] == pytest.approx([1.0, 1.0])
    padded_mask = torch.ones(4, 4, dtype=torch.bool)
    padded_mask[2:] = False  # the last two queries are padding: every key masked
    padded = probe_mod._attention_stats(q, k, padded_mask, False, None, map_size=4)
    assert padded["entropy_share"] == pytest.approx([1.0, 1.0])
    assert padded["masked_rows"] == 2 * 2 and padded["invalid_rows"] == 0
    assert padded["map"][2] == [0.0] * 4 and sum(padded["map"][0]) == pytest.approx(1.0)


def test_nan_scores_are_reported_as_invalid_not_as_concentrated_attention():
    torch.manual_seed(2)
    q, k = torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)
    q[0, :, 1] = float("nan")  # query 1 is broken in both heads
    stats = probe_mod._attention_stats(q, k, None, False, None, map_size=4)
    assert stats["invalid_rows"] == 2 and stats["masked_rows"] == 0
    assert all(math.isfinite(v) for v in stats["max_logit"])
    assert all(0 < v <= 1 for v in stats["entropy_share"])  # from the three healthy rows
    assert stats["map"][1] == [0.0] * 4
    broken = probe_mod._attention_stats(torch.full_like(q, float("nan")), k, None, False, None, map_size=4)
    assert broken["invalid_rows"] == 8 and all(math.isnan(v) for v in broken["entropy"])


def test_a_failing_probe_keeps_the_step_and_records_the_error():
    model = Toy()
    calls = []

    def flaky(m):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("probe broke")
        m(torch.zeros(1, 4, dtype=torch.long))

    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=4, probe=flaky, probe_every=1)
    with pytest.warns(RuntimeWarning, match="probe broke"):
        _train(model, session, 3, torch.randint(0, 32, (2, 6)))
    session.close()
    assert [r["step"] for r in sink.records] == [0, 1, 2]
    assert sink.records[1]["probe"] == {"error": "RuntimeError: probe broke"}
    assert "histograms" in sink.records[2]["probe"]
    assert model.training and not model.frozen.training
