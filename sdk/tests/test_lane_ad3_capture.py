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
