"""B10, X-DDP (x-jxmnop-ddp-out-of-sync): forward called on the module inside DDP, not on DDP itself.

Re-created here from the synopsis and TrainCheck's blog post (`docs/blog/posts/
traincheck-launch.md`: "The code called the raw `nn.Module` instead of the DDP wrapper.
Gradients did not synchronize, so each GPU learned its own solution"). TrainCheck's script
is not in the clone. Two CPU processes with gloo stand in for the GPUs; a small MLP
classifier on a seeded synthetic set stands in for the language model; AdamW, so
`updates=True` reads the update sizes.

Buggy: `output = ddp_model.module(x)`. DDP's reducer is only armed by DDP's own forward, so
backward runs its hooks without reducing anything and each rank steps on its own gradient.
Healthy: `output = ddp_model(x)`.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import harness

BUG = "B10-X-DDP"
WORLD = 2


def mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))


def worker(rank: int, world_size: int, init_file: str, variant: str, run_id: str, steps: int,
           weights: int | None, replica_every: int | None) -> None:
    harness.join_group(rank, world_size, init_file)
    torch.manual_seed(0)
    model = mlp()
    ddp_model = DDP(model)
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
            output = ddp_model.module(x) if variant == "buggy" else ddp_model(x)
            loss = F.cross_entropy(output, y)
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
    symptom = {"steps": steps, "replica_max_abs_diff": diff, "bug_present": diff > 0,
               "rank0_loss_first_last": [round(losses[0], 4), round(losses[-1], 4)]}
    settings = {"world_size": world_size, "steps": steps, "optimizer": "AdamW(lr=1e-3)",
                "replica_every": replica_every or "default (50)", "weights": weights or "default (50)",
                "source": "re-created from synopsis and TrainCheck's blog post"}
    harness.finish(None, run_id, bug=BUG, variant=variant, symptom=symptom, settings=settings, rank=rank,
                   world_size=world_size, relevant=("replica_drift", "update_spike", "grad_spike"))
    dist.destroy_process_group()


def main() -> None:
    p = harness.parser(__doc__)
    p.add_argument("--steps", type=int, default=200, help="the X post saw the loss turn near step 150")
    args = p.parse_args()
    run_id = harness.run_id_for(BUG, args)
    harness.launch(worker, WORLD, run_id, args.variant, run_id, args.steps, args.weights, args.replica_every)


if __name__ == "__main__":
    main()
