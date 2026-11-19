"""B9, PT-104336: `Module.to` does not keep DDP's gradient-sync hooks, so DDP replicas drift apart.

Re-created here from the synopsis ("Module.to not preserve the gradient syncing hooks ...,
causing DDP model weights to go out of sync"; TrainCheck's `bugs.md`: "`nn.Module.to` does
not preserve DDP gradient sync hooks, leading to grad and data out of sync"). Neither the
issue's text nor TrainCheck's script is on disk, so the trigger here is one that takes the
same code path on PyTorch 2.15: with
`torch.__future__.set_overwrite_module_params_on_conversion(True)`, `Module._apply` puts a
new `Parameter` in place of each one instead of swapping its data, as it does when a
conversion's result cannot share the old tensor's type (moving to another device type).
DDP's reducer keeps hooks on the old parameters' gradient accumulators; the new ones have
none, so backward reduces nothing and each rank steps alone. Nothing raises.

Two CPU processes with gloo; the same MLP, data and AdamW as `b10_x_ddp.py`. TensorQuill is
started after the `.to()` call, where the guide says to start it.

Buggy: the flag set, then `ddp_model.to(torch.float32)` after wrapping. Healthy: the same
`.to()` call without the flag (the data is swapped in place and the hooks stay).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import harness
from b10_x_ddp import mlp

BUG = "B09-PT-104336"
WORLD = 2


def worker(rank: int, world_size: int, init_file: str, variant: str, run_id: str, steps: int,
           weights: int | None, replica_every: int | None) -> None:
    harness.join_group(rank, world_size, init_file)
    torch.manual_seed(0)
    model = mlp()
    ddp_model = DDP(model)
    before = [id(p) for p in model.parameters()]
    if variant == "buggy":
        torch.__future__.set_overwrite_module_params_on_conversion(True)
    ddp_model.to(torch.float32)
    torch.__future__.set_overwrite_module_params_on_conversion(False)
    replaced = sum(1 for a, p in zip(before, model.parameters()) if a != id(p))
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)
    session = harness.start(model, run_id, rank=rank, weights=weights, replica_group=dist.group.WORLD,
                            **({'replica_every': replica_every} if replica_every else {}))

    data = harness.blobs(32 * world_size * 50, (64,), 10, seed=1, noise=3.0)
    sampler = torch.utils.data.distributed.DistributedSampler(data, num_replicas=world_size, rank=rank, seed=0)
    loader = torch.utils.data.DataLoader(data, batch_size=32, sampler=sampler)
    losses: list[float] = []
    epoch = 0
    while len(losses) < steps:
        sampler.set_epoch(epoch)
        for x, y in session.watch_loader(loader, epoch=epoch):
            loss = F.cross_entropy(ddp_model(x), y)
            optimizer.zero_grad()
            loss.backward()
            session.log({"train/loss": loss})
            optimizer.step()
            losses.append(loss.item())
            if len(losses) == steps:
                break
        epoch += 1

    diff = harness.replica_max_diff(list(model.parameters()))
    session.close()
    dist.barrier()
    symptom = {"steps": steps, "parameters_replaced_by_to": replaced, "replica_max_abs_diff": diff,
               "bug_present": diff > 0, "rank0_loss_first_last": [round(losses[0], 4), round(losses[-1], 4)]}
    settings = {"world_size": world_size, "steps": steps, "optimizer": "AdamW(lr=1e-3)",
                "replica_every": replica_every or "default (50)", "weights": weights or "default (50)",
                "source": "re-created from synopsis; trigger: overwrite_module_params_on_conversion"}
    harness.finish(None, run_id, bug=BUG, variant=variant, symptom=symptom, settings=settings, rank=rank,
                   world_size=world_size, relevant=("replica_drift", "update_spike", "grad_spike"))
    dist.destroy_process_group()


def main() -> None:
    p = harness.parser(__doc__)
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()
    run_id = harness.run_id_for(BUG, args)
    harness.launch(worker, WORLD, run_id, args.variant, run_id, args.steps, args.weights, args.replica_every)


if __name__ == "__main__":
    main()
