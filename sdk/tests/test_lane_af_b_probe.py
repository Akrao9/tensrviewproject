"""Lane AF-B, SDK-05: the head probe's gap between its BF16 and FP32 passes.

`zb - zf` overflows float32 when both logits are finite and of opposite signs above about
1.7e38 (2e38 minus -2e38 is `inf` in float32, finite by halves). Such a position was counted
under `overflow_positions` and left out of all four numbers, though both passes are finite. It
is now read from halves; a position whose BF16 pass is itself not finite is still counted, and
the reason given says it is the BF16 pass, not the FP32 logits' sums.

No real head was found that reaches opposite-sign BF16 and FP32 logits that large (BF16 rounds
a logit by under 1%), so the path is driven by a stand-in `F.linear` in the probe's module.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorquill import probe


def test_the_float32_arithmetic():
    a, b = torch.tensor([2e38]), torch.tensor([-2e38])
    assert torch.isinf(a - b).all()
    assert torch.isfinite(a * 0.5 - b * 0.5).all()


def _stand_in(fp32_rows, bf16_rows):
    """`F.linear` returning fixed logits by the weight's dtype: one list of values per position."""
    def linear(x, w, b=None):
        rows = fp32_rows if w.dtype == torch.float32 else bf16_rows
        if x.shape[0] != len(rows):  # the rows taken again: here, those with a huge value and no NaN
            rows = [r for r in rows if not any(math.isnan(v) for v in r) and any(abs(v) >= 1e30 for v in r)]
        out = torch.tensor(rows, dtype=torch.float32)[:, : w.shape[0]]
        return out.to(w.dtype)
    return linear


def _sums(monkeypatch, fp32_rows, bf16_rows):
    monkeypatch.setattr(probe.F, "linear", _stand_in(fp32_rows, bf16_rows))
    hidden = torch.ones(len(fp32_rows), 3)
    weight = torch.ones(len(fp32_rows[0]), 3)
    sums = probe._head_logit_sums(hidden, weight, None)
    return sums, {**sums, "input_dtype": "torch.float32", "weight_dtype": "torch.float32"}


def test_sdk05_finite_logits_of_opposite_signs_are_read(monkeypatch):
    fp32 = [[2e38, 1.0], [1.0, 2.0]]
    bf16 = [[-2e38, 1.0], [1.0, 2.0]]
    sums, call = _sums(monkeypatch, fp32, bf16)
    assert bool(sums["logits_finite"].all())
    assert torch.isfinite(sums["gap_sq"]).all(), sums["gap_sq"]
    zb = torch.tensor(bf16).to(torch.bfloat16).double()
    zf = torch.tensor(fp32).double()
    assert sums["gap_sq"][0].item() == pytest.approx(((zb[0] - zf[0]) ** 2).sum().item(), rel=1e-6)
    assert sums["gap_sq"][1].item() == ((zb[1] - zf[1]) ** 2).sum().item()
    numbers = probe._head_numbers([call])
    assert numbers["positions"] == 2 and "overflow_positions" not in numbers and "not_read" not in numbers
    expected = math.sqrt(((zb - zf) ** 2).sum().item() / 4)
    assert numbers["gap"] == pytest.approx(expected, rel=1e-6)


def test_sdk05_a_bf16_pass_that_is_not_finite_is_still_counted_and_named(monkeypatch):
    fp32 = [[3.0e38, 1.0]]
    bf16 = [[math.inf, 1.0]]
    _, call = _sums(monkeypatch, fp32, bf16)
    numbers = probe._head_numbers([call])
    assert numbers["positions"] == 0 and numbers["overflow_positions"] == 1
    assert "BF16" in numbers["not_read"] and "sums overflow" not in numbers["not_read"]


def test_sdk05_mixed_reason_names_the_bf16_pass(monkeypatch):
    fp32 = [[3.0e38, 1.0], [math.nan, 1.0]]
    bf16 = [[math.inf, 1.0], [math.nan, 1.0]]
    _, call = _sums(monkeypatch, fp32, bf16)
    numbers = probe._head_numbers([call])
    assert numbers["positions"] == 0 and numbers["nonfinite_positions"] == 1 and numbers["overflow_positions"] == 1
    assert "BF16" in numbers["not_read"] and "sums overflow" not in numbers["not_read"]


def test_an_ordinary_head_gives_the_same_numbers_as_float64():
    torch.manual_seed(0)
    hidden, weight, bias = torch.randn(5, 16), torch.randn(40, 16), torch.randn(40)
    sums = probe._head_logit_sums(hidden, weight, bias)
    zf = hidden.double() @ weight.double().T + bias.double()
    assert torch.allclose(sums["norm"], zf.norm(dim=-1), rtol=1e-5)
    zb = torch.nn.functional.linear(hidden.bfloat16(), weight.bfloat16(), bias.bfloat16()).double()
    assert torch.allclose(sums["gap_sq"], ((zb - (hidden @ weight.T + bias).double()) ** 2).sum(-1), rtol=1e-4)
