"""A block's four numbers over a step's gradient-accumulation microbatches, as today's behaviour.

The capture keeps a block's `abs_max` as the largest over every forward until the optimizer step,
and writes `mean`, `std` and `zero_frac` from the last forward before it (the block hook in
`capture.py`). With three microbatches a step, the first made the largest on purpose, a record's
`abs_max` is that microbatch's and its other three numbers are the last microbatch's. The router
counts, which add every microbatch, are held in `test_capture.py`
(`test_router_counts_accumulate_across_microbatches_and_reset_on_step`) and
`test_eval_forwards.py` (`test_gradient_accumulation_still_adds_its_microbatches`).

Eager mode only. Only torch and pytest are needed.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import tensorquill

QUIET = dict(heartbeat_seconds=None, stack_seconds=None)
MICROBATCHES = 3
LOUD = 50.0


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))  # zeros, so `zero_frac` differs between microbatches


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(2))
        self.out = nn.Linear(6, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.out(x)


def test_abs_max_is_over_every_microbatch_and_the_other_three_are_the_last_microbatchs():
    torch.manual_seed(0)
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    seen: dict[str, list[dict]] = {"blocks.0": [], "blocks.1": []}

    def keep(name: str):
        def hook(module, args, output):
            x = output.detach().float()
            std, mean = torch.std_mean(x)
            seen[name].append({"mean": float(mean), "std": float(std), "abs_max": float(x.abs().amax()),
                               "zero_frac": float((x == 0).float().mean())})
        return hook

    for index, block in enumerate(model.blocks):
        block.register_forward_hook(keep(f"blocks.{index}"))
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, run_id="microbatches", capture="full", **QUIET)
    data = torch.Generator().manual_seed(1)
    steps = 3
    for _ in range(steps):
        optimizer.zero_grad()
        for micro in range(MICROBATCHES):
            x = torch.randn(8, 6, generator=data) * (LOUD if micro == 0 else 1.0)
            (model(x).pow(2).mean() / MICROBATCHES).backward()
        optimizer.step()
    session.close()

    records = sink.records
    assert [r["step"] for r in records] == list(range(steps))
    for step, record in enumerate(records):
        for name, forwards in seen.items():
            micro = forwards[step * MICROBATCHES:(step + 1) * MICROBATCHES]
            block = record["layers"][name]
            assert micro[0]["abs_max"] > 5 * max(f["abs_max"] for f in micro[1:])
            assert block["abs_max"] == pytest.approx(micro[0]["abs_max"], rel=1e-6)
            last = micro[-1]
            for key in ("mean", "std", "zero_frac"):
                assert block[key] == pytest.approx(last[key], rel=1e-6, abs=1e-12), (step, name, key)
            # And not the first microbatch's, which differs, so the test can tell the two apart.
            assert block["mean"] != pytest.approx(micro[0]["mean"], rel=1e-3)
            assert block["zero_frac"] != pytest.approx(micro[0]["zero_frac"], rel=1e-3)
