"""A STUB of lm-resiliency's campaign module, for the CPU dry run of `fault_injection_check_entry.py`.

Lane AI-3, Nov 22, 2026. lm-resiliency is not installed here and needs PyTorch 2.10 to 2.13 and eight
GPUs, so its campaign cannot run on this machine. This file writes a module tree shaped like the
part of the checkout the wrapper imports (`tests/validation/fault_injection/pytorch.py`, with `main`,
`enable_fault_injection` and `_teardown` at module level, as in the real file) whose `main` trains a
small DDP model on Gloo and plants one weight corruption of its own: noise added to one parameter on
one rank before one iteration's forward pass, the clean weights broadcast back from rank 0 after
that iteration's optimizer step (as the campaign's `EvaluationStateReset` restores them). Its rank 0
writes `injection.json` in the documented format with that one record.

Nothing here is lm-resiliency's code, and a result on this stub says nothing about TensorQuill on the
campaign. It shows only that the wrapper attaches, that a session is opened on every rank and closed
before teardown, that the captures land beside `injection.json`, and that the comparison runs end to
end on real TensorQuill captures. The ground truth it writes is constructed, and labelled so.

    python sdk/examples/fault_injection_check_stub.py --write <stub-root>
    torchrun --nproc-per-node=3 --master-addr 127.0.0.1 --master-port 29651 \\
        sdk/examples/fault_injection_check_entry.py --lm-resiliency <stub-root> --cpu-stub --artifact-dir <dir>
    python sdk/examples/fault_injection_check.py --artifact-dir <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

STUB_MODULE = r'''"""STUB, written by sdk/examples/fault_injection_check_stub.py: not lm-resiliency's campaign."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

STUB = True
CAMPAIGN = "stub-campaign (constructed, not lm-resiliency's)"


class _Faults:
    """The stub's injector: one planted weight corruption."""

    def __init__(self, model, rank, plant_rank, plant_iteration):
        self.model, self.rank, self.plant_rank, self.plant_iteration = model, rank, plant_rank, plant_iteration
        self.records = []

    def arm(self, iteration):
        if self.rank == self.plant_rank and iteration == self.plant_iteration:
            weight = self.model.module[0].weight
            with torch.no_grad():
                weight.add_(torch.randn(weight.shape, generator=torch.Generator().manual_seed(7)) * 0.5)
        if iteration == self.plant_iteration:
            self.records.append({
                "injection_id": f"stub-weight-noise@{iteration}", "occurrence_id": f"stub-weight-noise#{iteration}",
                "incident_id": "stub-weight-noise", "fault_id": f"rank-{self.plant_rank}-weight-noise",
                "iteration": iteration, "attempt": 1, "temporal_behavior": "transient",
                "failure_type": "tensor_corruption", "expected_kind": "sdc", "safety": "local",
                "framework": "pytorch", "executor": "stub", "execution_rank": self.plant_rank,
                "target": {"component": "linear", "index": 0, "model_part": 0, "rank": self.plant_rank,
                           "surface": "weight"},
                "parameters": {"operation": "noise", "scale": 0.5}, "status": "completed", "verified": True,
                "injection_succeeded": True, "activated_at_ns": 0, "completed_at_ns": 0, "evidence": {},
                "error": None,
            })

    def close(self):
        pass


def enable_fault_injection(model, optimizer, *, campaign):
    return _Faults(model, dist.get_rank(), campaign["plant_rank"], campaign["plant_iteration"])


def _teardown(faults, state_reset, resiliency, *, active_error, destroy_process_group):
    if faults is not None:
        faults.close()
    destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--plant-rank", type=int, default=2)
    parser.add_argument("--plant-iteration", type=int, default=4)
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    dist.init_process_group("gloo")
    faults = None
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.manual_seed(123)
        model = DistributedDataParallel(torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        faults = enable_fault_injection(model, optimizer, campaign={"plant_rank": args.plant_rank,
                                                                    "plant_iteration": args.plant_iteration})
        generator = torch.Generator().manual_seed(1000 + rank)
        for iteration in range(1, args.steps + 1):
            faults.arm(iteration)
            inputs = torch.randn(4, 8, generator=generator)
            optimizer.zero_grad(set_to_none=True)
            model(inputs).pow(2).mean().backward()
            optimizer.step()
            if iteration == args.plant_iteration:
                for p in model.module.parameters():  # the clean state back, as the campaign's reset does
                    dist.broadcast(p.data, src=0)
        if rank == 0:
            manifest = {"name": CAMPAIGN, "incidents": [{"incident_id": "stub-weight-noise", "lifetime": {"iterations": 1}}]}
            (args.artifact_dir / "injection.json").write_text(json.dumps({
                "schema_version": 1, "campaign": CAMPAIGN, "manifest_identity": "constructed",
                "manifest": manifest, "framework": "pytorch", "world_size": world,
                "completed_iterations": args.steps, "state_reset_iterations": [args.plant_iteration],
                "injections": faults.records, "constructed": True,
            }, indent=2) + "\n", encoding="utf-8")
    finally:
        _teardown(faults, None, None, active_error=None, destroy_process_group=dist.destroy_process_group)
'''


def write_stub_tree(root: Path) -> Path:
    """Write the stub tree under `root`; returns the module's path."""
    package = root / "tests" / "validation" / "fault_injection"
    package.mkdir(parents=True, exist_ok=True)
    for folder in (root / "tests", root / "tests" / "validation", package):
        (folder / "__init__.py").write_text('"""STUB package (fault_injection_check_stub.py)."""\n', encoding="utf-8")
    module = package / "pytorch.py"
    module.write_text(STUB_MODULE, encoding="utf-8")
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", type=Path, required=True, help="the folder to write the stub tree into")
    args = parser.parse_args(argv)
    print(f"fault_injection_check_stub: wrote the STUB campaign module {write_stub_tree(args.write)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
