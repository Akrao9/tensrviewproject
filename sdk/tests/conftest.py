import pytest
import torch
from torch import nn


class ToyMoE(nn.Module):
    """A router that picks one of `n_experts` linear experts per token."""

    def __init__(self, dim: int = 8, n_experts: int = 4) -> None:
        super().__init__()
        self.inp = nn.Linear(dim, dim)
        self.router = Router(dim, n_experts)
        self.experts = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_experts))
        self.out = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.inp(x))
        _, choice = self.router(h)
        stacked = torch.stack([expert(h) for expert in self.experts], dim=1)  # [B, E, D]
        picked = stacked.gather(1, choice[:, None, None].expand(-1, 1, h.shape[-1])).squeeze(1)
        return self.out(picked)


class Router(nn.Module):
    def __init__(self, dim: int, n_experts: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, n_experts)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.proj(h)
        return logits, logits.argmax(dim=-1)


@pytest.fixture
def toy_moe() -> ToyMoE:
    torch.manual_seed(0)
    return ToyMoE()


@pytest.fixture(autouse=True)
def _reset_dynamo():
    yield
    torch._dynamo.reset()


def pytest_addoption(parser):
    parser.addoption(
        "--write-golden", action="store_true",
        help="rewrite tests/golden/ from the Python detectors before test_golden.py checks it",
    )
