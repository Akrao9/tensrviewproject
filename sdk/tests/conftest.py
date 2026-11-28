import os
from pathlib import Path

import pytest
import torch
from torch import nn

SERVER_BIN_ENV = "TENSORQUILL_SERVER_BIN"
_ROOT = Path(__file__).resolve().parents[2]
_TARGET = _ROOT / "server" / "target"
CONTRACT_PAGES_ENV = "TENSORQUILL_CONTRACT_PAGES"


def built_server_binary(target: Path = _TARGET) -> Path | None:
    """The newer of `debug/` and `release/` `tensorquill-server` under `target`, or None. The choice
    `_server_binary()` in test_flight_e2e.py makes when the variable is not set."""
    built = [p for p in (target / "debug" / "tensorquill-server", target / "release" / "tensorquill-server")
             if p.exists()]
    return max(built, key=lambda p: p.stat().st_mtime) if built else None


def name_built_server(environ=os.environ, target: Path = _TARGET) -> Path | None:
    """Make the SDK's own lookup (`server_detect.find_binary`: the variable, then beside the interpreter,
    then PATH) find the binary the tests find under `server/target`. A variable that is set, even to a
    wrong path, is left exactly as it is, and when nothing is built nothing is set. Returns what it set."""
    if environ.get(SERVER_BIN_ENV):
        return None
    chosen = built_server_binary(target)
    if chosen is not None:
        environ[SERVER_BIN_ENV] = str(chosen)
    return chosen


# Before any test module is imported, so the tests' skip decision and the Rust detectors agree for the run.
name_built_server()


def contract_page(name: str) -> str:
    """The text of the contract page `name` (a `*_CONTRACT.md`), for a test that holds the page to the code.

    The contract pages are kept outside this repository since Nov 27, 2026, so a checkout holds none of
    them. When the variable TENSORQUILL_CONTRACT_PAGES names a folder, the page is read from it, and a page
    missing there is an error, never a skip: a run that was told where the pages are must read them. With
    the variable unset, a page at the repository's root is read, and with none there the calling test is
    skipped. Call it inside the test, after the checks that need no page, so that those still run."""
    folder = os.environ.get(CONTRACT_PAGES_ENV)
    if folder:
        return (Path(folder) / name).read_text(encoding="utf-8")
    page = _ROOT / name
    if page.is_file():
        return page.read_text(encoding="utf-8")
    pytest.skip(f"the contract page {name} is kept outside this repository; {CONTRACT_PAGES_ENV} names its folder")


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
