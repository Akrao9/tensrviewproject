"""B2 (PT-84803), B17 (PT-96600), B18 (BC-86): data lost in GPU peer-to-peer copies. A simulated effect only.

These three are faults of the I/O setup or the driver on multi-GPU machines and cannot
happen on a CPU. TrainCheck does not reproduce them either: its AE guide reuses the
PT-104336 trace for all three, where DDP replicas go out of sync. What is simulated here is
that effect, and it is not counted as running any of the three bugs.

Two CPU processes with gloo train the MLP of `b10_x_ddp.py` correctly under DDP. Buggy: at
step `--corrupt-at` rank 1's copy of the first layer's weight loses a tenth of its elements
(set to zero), as a lossy device-to-device copy of a module would leave it; DDP keeps
averaging the gradients from then on, so the two replicas stay different. Healthy: nothing
is corrupted.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import harness
from b10_x_ddp import mlp

BUG = "HW-P2P-SIM"
WORLD = 2


def worker(rank: int, world_size: int, init_file: str, variant: str, run_id: str, steps: int,
           weights: int | None, replica_every: int | None, corrupt_at: int) -> None:
    harness.join_group(rank, world_size, init_file)
    torch.manual_seed(0)
    model = mlp()
    ddp_model = DDP(model)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)
    session = harness.start(model, run_id, rank=rank, weights=weights, replica_group=dist.group.WORLD,
                            **({"replica_every": replica_every} if replica_every else {}))
    data = harness.blobs(32 * world_size * 50, (64,), 10, seed=1, noise=3.0)
    sampler = torch.utils.data.distributed.DistributedSampler(data, num_replicas=world_size, rank=rank, seed=0)
    loader = torch.utils.data.DataLoader(data, batch_size=32, sampler=sampler)
    losses: list[float] = []
    epoch = 0
    while len(losses) < steps:
        sampler.set_epoch(epoch)
        for x, y in session.watch_loader(loader, epoch=epoch):
            if variant == "buggy" and rank == 1 and len(losses) == corrupt_at:
                with torch.no_grad():
                    w = model[0].weight.view(-1)
                    w[: w.numel() // 10] = 0.0
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
    symptom = {"steps": steps, "corrupted_at_step": corrupt_at if variant == "buggy" else None,
               "replica_max_abs_diff": diff, "bug_present": diff > 0,
               "rank0_loss_first_last": [round(losses[0], 4), round(losses[-1], 4)]}
    settings = {"world_size": world_size, "steps": steps, "optimizer": "AdamW(lr=1e-3)",
                "replica_every": replica_every or "default (50)", "weights": weights or "default (50)",
                "source": "simulated effect (TrainCheck reuses the PT-104336 trace for these three)"}
    harness.finish(None, run_id, bug=BUG, variant=variant, symptom=symptom, settings=settings, rank=rank,
                   world_size=world_size, relevant=("replica_drift", "activation_spike", "grad_spike"))
    dist.destroy_process_group()


def main() -> None:
    p = harness.parser(__doc__)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--corrupt-at", type=int, default=75)
    args = p.parse_args()
    run_id = harness.run_id_for(BUG, args)
    harness.launch(worker, WORLD, run_id, args.variant, run_id, args.steps, args.weights, args.replica_every,
                   args.corrupt_at)


if __name__ == "__main__":
    main()
