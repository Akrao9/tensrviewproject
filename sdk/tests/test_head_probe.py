"""The probe's language-model head numbers (row K4): norm, signed mean, log Z, and the gap
between BF16 and FP32 logits on the same hidden states. Numbers only, never an incident."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import tensorquill
from tensorquill import probe as probe_mod
from test_probe import Toy, _train


def _hidden(model, ids):
    """The head's input on the probe batch, computed the way the toy's forward does."""
    with torch.no_grad():
        x = model.embed(ids)
        for block in model.blocks:
            x = block(x)
        return model.frozen(x)


def _reference(h, weight, bias):
    weight, bias = weight.detach(), bias.detach()
    zf = F.linear(h.float(), weight.float(), bias.float())
    zb = F.linear(h.bfloat16(), weight.bfloat16(), bias.bfloat16()).float()
    gap = (zb - zf).pow(2).mean().sqrt()
    return {
        "norm": float(zf.norm(dim=-1).mean()),
        "mean": float(zf.mean()),
        "log_z": float(torch.logsumexp(zf, -1).mean()),
        "gap": float(gap),
        "gap_relative": float(gap / zf.var(-1, unbiased=False).mean().sqrt()),
    }


def _probe_once(model, ids, **kw):
    session = tensorquill.init(model, sink=tensorquill.MemorySink(), probe=ids, **kw)
    result = session._probe()
    session.close()
    return result


def test_head_numbers_are_in_the_probe_record_and_in_the_evidence():
    torch.manual_seed(0)
    model = Toy()
    fixed = torch.randint(0, 32, (2, 12))
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, probe=fixed, probe_every=2, probe_head="head")
    _train(model, session, 4, torch.randint(0, 32, (2, 12)))
    session.close()

    heads = [r["probe"]["head"] for r in sink.records if "probe" in r]
    assert len(heads) == 2
    head = heads[-1]
    assert head["module"] == "head" and head["found_by"] == "probe_head"
    assert head["positions"] == 24 and head["nonfinite_positions"] == 0 and head["vocab"] == 32
    assert head["calls"] == 1 and head["input_dtype"] == head["weight_dtype"] == "float32"
    model.eval()
    expected = _reference(_hidden(model, fixed), model.head.weight, model.head.bias)
    model.train()
    for name, value in expected.items():
        assert head[name] == pytest.approx(value, rel=1e-5), name

    evidence = tensorquill.build_evidence(sink.records)
    trends = evidence["head_trends"]
    assert trends == tensorquill.head_logit_trends(sink.records)
    assert trends["probes"] == trends["read"] == 2 and trends["module"] == "head"
    assert trends["first_step"] == 1 and trends["latest_step"] == 3
    assert set(trends["numbers"]) == {"norm", "mean", "log_z", "gap", "gap_relative"}
    assert trends["numbers"]["mean"]["first"] == heads[0]["mean"]
    assert trends["numbers"]["mean"]["latest"] == heads[1]["mean"]
    assert "not_read" not in trends
    # Numbers only: the head never makes an incident or a `not_checked` entry.
    assert not any("head" in str(i.get("kind")) or "logit" in str(i.get("kind")) for i in evidence["incidents"])
    assert not any(item.get("check") == "head" for item in evidence.get("not_checked", ()))


def test_chunked_vocabulary_and_a_large_common_shift_give_the_same_numbers(monkeypatch):
    torch.manual_seed(1)
    h, weight, bias = torch.randn(3, 5, 16), torch.randn(40, 16), torch.randn(40) + 3e4
    whole = probe_mod._head_numbers([{**probe_mod._head_logit_sums(h, weight, bias),
                                      "input_dtype": "float32", "weight_dtype": "float32"}])
    monkeypatch.setattr(probe_mod, "CHUNK_ELEMENTS", 15 * 7)  # 7 vocabulary rows per chunk
    chunked = probe_mod._head_numbers([{**probe_mod._head_logit_sums(h, weight, bias),
                                        "input_dtype": "float32", "weight_dtype": "float32"}])
    zf = F.linear(h, weight, bias).double()
    for name in ("norm", "mean", "log_z", "gap", "gap_relative"):
        assert chunked[name] == pytest.approx(whole[name], rel=1e-6), name
    # The spread survives a shift of 30,000 (the variance is summed about each position's mean).
    spread = zf.var(-1, unbiased=False).mean().sqrt()
    assert whole["gap"] / whole["gap_relative"] == pytest.approx(float(spread), rel=1e-4)


def test_the_gap_grows_with_a_scaled_head_and_stays_small_on_a_healthy_toy(capsys):
    torch.manual_seed(0)
    fixed = torch.randint(0, 32, (4, 16))
    results = {}
    for case in ("healthy", "weight x1000", "bias +1000"):
        torch.manual_seed(0)
        model = Toy()
        with torch.no_grad():
            if case == "weight x1000":
                model.head.weight.mul_(1000)
                model.head.bias.mul_(1000)
            elif case == "bias +1000":
                model.head.bias.add_(1000)  # a common shift of the logits, as a drifting mean
        results[case] = _probe_once(model, fixed, probe_head="head")["head"]
    with capsys.disabled():
        for case, head in results.items():
            print(f"\n  {case:>13}: norm {head['norm']:.4g}  mean {head['mean']:+.4g}  log Z {head['log_z']:.4g}"
                  f"  gap {head['gap']:.3g}  gap_relative {head['gap_relative']:.3g}")
    healthy, scaled, shifted = results["healthy"], results["weight x1000"], results["bias +1000"]
    assert healthy["gap"] < 0.01 and healthy["gap_relative"] < 0.01
    assert scaled["gap"] > 100 * healthy["gap"]  # the absolute gap scales with the logits
    assert shifted["gap"] > 100 * healthy["gap"] and shifted["gap_relative"] > 100 * healthy["gap_relative"]
    assert shifted["mean"] > 999 and healthy["mean"] == pytest.approx(shifted["mean"] - 1000, abs=1e-2)


class _HFLike(Toy):
    def __init__(self, returns="head"):
        super().__init__()
        self.returns = returns

    def get_output_embeddings(self):
        return {"head": self.head, "none": None, "embed": self.embed}[self.returns]


class _OwnForward(nn.Linear):
    def forward(self, x):
        return super().forward(x).clamp(-30, 30)


def test_the_head_is_found_only_when_named_or_reported_and_otherwise_not_read():
    fixed = torch.zeros(1, 6, dtype=torch.long)
    found = _probe_once(_HFLike("head"), fixed)["head"]
    assert found["module"] == "head" and found["found_by"] == "get_output_embeddings" and found["positions"] == 6

    # A module called `head` is not taken for the head unless the user names it.
    assert _probe_once(Toy(), fixed)["head"] == {
        "not_read": "the model names no output head; pass probe_head=<module name> to read one"
    }
    assert "no language-model head" in _probe_once(_HFLike("none"), fixed)["head"]["not_read"]
    embed = _probe_once(_HFLike("embed"), fixed)["head"]
    assert set(embed) == {"not_read"} and "is a Embedding, not an nn.Linear" in embed["not_read"]

    # Named but never called in the probe's forward.
    model = Toy()
    session = tensorquill.init(model, sink=tensorquill.MemorySink(), probe=lambda m: m.embed(fixed), probe_head="head")
    assert session._probe()["head"] == {
        "module": "head", "found_by": "probe_head", "not_read": "the head was not called in the probe's forward"
    }
    session.close()

    with pytest.raises(ValueError, match="no module named 'lm_head'"):
        tensorquill.init(Toy(), sink=tensorquill.MemorySink(), probe=fixed, probe_head="lm_head")
    with pytest.raises(ValueError, match="ModuleList, not an nn.Linear"):
        tensorquill.init(Toy(), sink=tensorquill.MemorySink(), probe=fixed, probe_head="blocks")
    own = Toy()
    own.head = _OwnForward(16, 32)
    with pytest.raises(ValueError, match="subclass of nn.Linear with its own forward"):
        tensorquill.init(own, sink=tensorquill.MemorySink(), probe=fixed, probe_head="head")

    # Off: no `head` key, so the record is as before and the evidence has no trend.
    assert "head" not in _probe_once(_HFLike("head"), fixed, probe_head=False)


def test_a_run_whose_head_is_not_read_says_so_in_the_evidence():
    model = Toy()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, probe=torch.zeros(1, 6, dtype=torch.long), probe_every=2)
    _train(model, session, 4, torch.randint(0, 32, (2, 6)))
    session.close()
    trends = tensorquill.build_evidence(sink.records)["head_trends"]
    assert trends == {
        "probes": 2, "read": 0,
        "not_read": {"probes": 2, "first_step": 1,
                     "reason": "the model names no output head; pass probe_head=<module name> to read one"},
    }
    old = [{**r, "probe": {k: v for k, v in r["probe"].items() if k != "head"}} if "probe" in r else r
           for r in sink.records]
    assert "head_trends" not in tensorquill.build_evidence(old)  # records from before K4: no key


def _seeded_run(probe_head):
    torch.manual_seed(0)
    model = Toy()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    session = None
    if probe_head is not None:
        session = tensorquill.init(model, sink=tensorquill.MemorySink(), every=1,
                                   probe=torch.zeros(2, 8, dtype=torch.long), probe_every=1, probe_head=probe_head)
    losses, grads, rng = [], [], []
    for _ in range(4):
        batch = torch.randint(0, 32, (2, 8))  # drawn from the global generator, as dropout is
        out = model(batch)
        loss = out.float().pow(2).mean()
        loss.backward()
        opt.step()  # with a session, the probe runs here, after the update and before zero_grad
        losses.append(out.detach().clone())
        grads.append([p.grad.clone() for p in model.parameters()])
        rng.append(torch.get_rng_state())
        opt.zero_grad()
    if session is not None:
        records = session._sink.records
        session.close()
        assert sum("probe" in r for r in records) == 4
        assert all(r["probe"]["head"]["positions"] == 16 for r in records if "probe" in r)
    return losses, grads, rng, [p.detach().clone() for p in model.parameters()]


def test_reading_the_head_changes_no_weight_gradient_random_state_or_output():
    plain = _seeded_run(None)
    probed = _seeded_run("head")
    for a, b in zip(plain[0], probed[0]):
        assert torch.equal(a, b)  # training outputs, dropout included
    for a, b in zip(plain[1], probed[1]):
        assert all(torch.equal(x, y) for x, y in zip(a, b))  # gradients at the probe
    for a, b in zip(plain[2], probed[2]):
        assert torch.equal(a, b)  # the CPU generator
    assert all(torch.equal(a, b) for a, b in zip(plain[3], probed[3]))  # final weights


def test_a_probe_run_under_bf16_autocast_still_compares_bf16_with_fp32():
    torch.manual_seed(0)
    model = Toy()
    ids = torch.randint(0, 32, (2, 12))

    def under_autocast(m):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            m(ids)

    session = tensorquill.init(model, sink=tensorquill.MemorySink(), probe=under_autocast, probe_head="head")
    head = session._probe()["head"]
    session.close()
    assert head["input_dtype"] == "bfloat16"  # the hidden states autocast made, read as they are
    model.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        hidden = _hidden(model, ids)
    model.train()
    expected = _reference(hidden, model.head.weight, model.head.bias)
    assert head["gap"] > 0  # autocast would make the FP32 pass BF16 too, and the gap 0
    for name, value in expected.items():
        assert head[name] == pytest.approx(value, rel=1e-5), name


def test_positions_with_non_finite_logits_are_left_out_and_counted():
    torch.manual_seed(0)
    weight, bias = torch.randn(8, 4), torch.randn(8)
    hidden = torch.randn(3, 4)
    hidden[1, 2] = float("nan")
    calls = [{**probe_mod._head_logit_sums(hidden, weight, bias), "input_dtype": "float32", "weight_dtype": "float32"}]
    numbers = probe_mod._head_numbers(calls)
    assert numbers["positions"] == 2 and numbers["nonfinite_positions"] == 1
    expected = _reference(hidden[[0, 2]], weight, bias)
    for name, value in expected.items():
        assert numbers[name] == pytest.approx(value, rel=1e-5), name
    hidden[:, 0] = float("inf")
    calls = [{**probe_mod._head_logit_sums(hidden, weight, bias), "input_dtype": "float32", "weight_dtype": "float32"}]
    gone = probe_mod._head_numbers(calls)
    assert gone["positions"] == 0 and gone["nonfinite_positions"] == 3
    assert gone["not_read"] == "every position's FP32 logits hold a NaN or an Inf"


def test_head_numbers_survive_the_jsonl_round_trip(tmp_path):
    model = Toy()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()  # every logit 0: no spread, so `gap_relative` is None
    path = tmp_path / "run.jsonl"
    session = tensorquill.init(model, sink=tensorquill.JsonlSink(path), every=1,
                               probe=torch.zeros(1, 6, dtype=torch.long), probe_every=2, probe_head="head")
    _train(model, session, 4, torch.randint(0, 32, (2, 6)))
    session.close()
    records = tensorquill.load_records(path)
    heads = [r["probe"]["head"] for r in records if "probe" in r]
    assert len(heads) == 2 and heads[0]["gap"] == 0.0 and heads[0]["gap_relative"] is None
    trends = tensorquill.build_evidence(records)["head_trends"]
    assert trends["read"] == 2 and "gap_relative" not in trends["numbers"]
    assert trends["numbers"]["log_z"]["first"] == heads[0]["log_z"]
