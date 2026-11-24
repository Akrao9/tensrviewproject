"""One rank of `test_fsdp_probe_guard.py`: a small model, sharded or not, trained 16 steps over
Gloo on the CPU with `probe=` set, and on rank 1 a sink that fails from step 5.

    RANK=0 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=... \
        python fsdp_probe_worker.py --flavor fsdp2|fsdp1|none --out DIR

It leaves, in DIR: `rank<r>.trace` (one line per stage, so the test can say where a rank was),
`rank<r>.jsonl` (the records its sink was handed) and `probe-entered.rank<r>` (written by the
probe callback before anything else, each time it is entered). Under sharding the callback then
raises: a probe entered there is the failure the test is about, whatever the SDK does with it.
Exit status 77 with one line on stderr when this PyTorch cannot shard the model, or run one
forward and backward of it, on the CPU (tried before TensorQuill is started).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

import tensorquill

STEPS, PROBE_EVERY, STOP_AT, EVERY = 16, 4, 5, 2


def shard(model: nn.Module, flavor: str) -> nn.Module:
    if flavor == "fsdp2":
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh("cpu", (dist.get_world_size(),))  # the default mesh picks MPS on a Mac
        for layer in model:
            if isinstance(layer, nn.Linear):
                fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)
        return model
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    # A unit per layer inside the root's: nested FSDP wrappers, as Lightning's wrap of the blocks makes.
    return FSDP(model, device_id=torch.device("cpu"), use_orig_params=True, auto_wrap_policy=ModuleWrapPolicy({nn.Linear}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flavor", choices=["fsdp2", "fsdp1", "none"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    trace = (args.out / f"rank{rank}.trace").open("a", buffering=1)

    def say(text: str) -> None:
        trace.write(f"{time.time():.3f} {text}\n")

    dist.init_process_group("gloo")
    torch.manual_seed(0)
    model: nn.Module = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 4))
    if args.flavor != "none":
        try:
            model = shard(model, args.flavor)
            # One training forward and backward before TensorQuill is there, the same on every rank:
            # PyTorch 2.6's FSDP2 raises in its first forward on the CPU (its CPU stream has no
            # `record_event`), which is PyTorch's and not what this test is about.
            model(torch.zeros(2, 16)).sum().backward()
            model.zero_grad()
        except Exception as error:  # noqa: BLE001
            print(f"cannot train a sharded model on the CPU with {args.flavor} on PyTorch {torch.__version__}: {error!r}", file=sys.stderr)
            return 77
    say(f"model ready ({args.flavor})")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    class Sink:
        """Writes what it is handed; on rank 1 it raises once a record of step STOP_AT or later comes."""

        def __init__(self) -> None:
            self.file = (args.out / f"rank{rank}.jsonl").open("w", buffering=1)

        def write(self, records: list[dict]) -> None:
            if rank == 1 and any(r.get("step", -1) >= STOP_AT for r in records):
                raise OSError("planted: the sink fails on rank 1")
            for record in records:
                self.file.write(json.dumps(record, default=str) + "\n")

        def close(self) -> None:
            self.file.close()

    fixed = torch.randn(4, 16, generator=torch.Generator().manual_seed(1))

    def probe(m: nn.Module) -> torch.Tensor:
        with (args.out / f"probe-entered.rank{rank}").open("a") as marker:
            marker.write("entered\n")
        if args.flavor != "none":
            raise AssertionError("the probe callback was entered on a sharded model")
        return m(fixed)

    session = tensorquill.init(
        model, sink=Sink(), every=EVERY, run_id=f"fsdp-probe-guard-{args.flavor}", probe=probe, probe_every=PROBE_EVERY,
        heartbeat_seconds=None, stack_seconds=None,
    )
    say("session started")
    data = torch.Generator().manual_seed(100 + rank)
    for step in range(STEPS):
        x, y = torch.randn(8, 16, generator=data), torch.randn(8, 4, generator=data)
        say(f"step {step} forward")
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        say(f"step {step} optimizer")
        optimizer.step()
        optimizer.zero_grad()
        say(f"step {step} done")
    say("training finished")
    session.close()
    dist.destroy_process_group()
    say("exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
