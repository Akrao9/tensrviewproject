"""B3, DS-1801: gradient clipping in DeepSpeed's BF16 optimizer leaves LayerNorm weights out of sync across tensor-parallel ranks.

Simulated here: DeepSpeed and Megatron-DeepSpeed are not installed, and TrainCheck's clone
holds only a config that points at an external Megatron-DeepSpeed checkout
(`example_pipelines/DS-1801/config.json`, deleted in 7de7430). What is kept is the mechanism
as the BLOOM-176B account and the v3 plan state it: gradient clipping was applied on only one
worker in each tensor-parallel group, so the LayerNorm weights every rank of the group holds
in full were updated differently and drifted apart while the loss looked healthy.

Two CPU processes with gloo form one tensor-parallel group of two. Each rank holds the same
LayerNorm and output bias (replicated, as Megatron does) and its own shard of a
column-parallel and a row-parallel linear layer, joined by Megatron's two collectives (copy
to the group: identity forward, all-reduce backward; reduce from the group: all-reduce
forward, identity backward). Both ranks read the same batches. The global gradient norm is
worked out as Megatron does (replicated parameters once, sharded ones summed over the group)
and the gradients scaled by `max_norm / norm` when it is above `max_norm`. AdamW, float32 (the
BF16 optimizer's master weights are float32).

Buggy: only tensor-parallel rank 0 scales its gradients. Healthy: both do.
TensorQuill: `replica_group` is the tensor-parallel group and `replica_params` the replicated
parameters, as the guide says for tensor parallelism.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import harness

BUG = "B03-DS-1801"
WORLD = 2
REPLICATED = ("ln.weight", "ln.bias", "out_bias")


class CopyToGroup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):  # type: ignore[override]
        return x

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        grad = grad.clone()
        dist.all_reduce(grad)
        return grad


class ReduceFromGroup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):  # type: ignore[override]
        x = x.clone()
        dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        return grad


class ShardedBlock(nn.Module):
    def __init__(self, rank: int, dim: int = 64, hidden: int = 128, classes: int = 10, world: int = WORLD) -> None:
        super().__init__()
        torch.manual_seed(0)  # replicated parameters: the same on every rank
        self.ln = nn.LayerNorm(dim)
        self.out_bias = nn.Parameter(torch.zeros(classes))
        torch.manual_seed(100 + rank)  # this rank's shards
        self.fc1 = nn.Linear(dim, hidden // world)
        self.fc2 = nn.Linear(hidden // world, classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = CopyToGroup.apply(self.ln(x))
        y = self.fc2(F.gelu(self.fc1(h)))
        return ReduceFromGroup.apply(y) + self.out_bias


def global_grad_norm(model: nn.Module) -> torch.Tensor:
    replicated = torch.zeros((), dtype=torch.float64)
    sharded = torch.zeros((), dtype=torch.float64)
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        sq = (p.grad.double() ** 2).sum()
        if name in REPLICATED:
            replicated += sq
        else:
            sharded += sq
    dist.all_reduce(sharded)
    return torch.sqrt(replicated + sharded)


def worker(rank: int, world_size: int, init_file: str, variant: str, run_id: str, steps: int,
           weights: int | None, replica_every: int | None, max_norm: float) -> None:
    harness.join_group(rank, world_size, init_file)
    model = ShardedBlock(rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    session = harness.start(model, run_id, rank=rank, weights=weights, replica_group=dist.group.WORLD,
                            **({'replica_every': replica_every} if replica_every else {}),
                            replica_params=lambda name, p: name in REPLICATED)
    data = harness.blobs(32 * 100, (64,), 10, seed=1, noise=3.0)
    # One data share for the whole tensor-parallel group: a sampler of one replica on both ranks.
    sampler = torch.utils.data.distributed.DistributedSampler(data, num_replicas=1, rank=0, seed=0)
    loader = torch.utils.data.DataLoader(data, batch_size=32, sampler=sampler)
    clipped = 0
    losses: list[float] = []
    epoch = 0
    while len(losses) < steps:
        sampler.set_epoch(epoch)
        for x, y in session.watch_loader(loader, epoch=epoch):
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            session.log({"train/loss": loss})
            norm = global_grad_norm(model)
            session.record_grad_norm(norm)
            coef = max_norm / (float(norm) + 1e-6)
            if coef < 1.0:
                clipped += 1
                if variant == "healthy" or rank == 0:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(coef)
            optimizer.step()
            losses.append(loss.item())
            if len(losses) == steps:
                break
        epoch += 1

    replicated = [p for n, p in model.named_parameters() if n in REPLICATED]
    diff = harness.replica_max_diff(replicated)
    ln_diff = harness.replica_max_diff([model.ln.weight])
    session.close()
    dist.barrier()
    symptom = {"steps": steps, "steps_clipped": clipped, "replicated_max_abs_diff": diff,
               "layernorm_weight_max_abs_diff": ln_diff, "bug_present": diff > 0,
               "rank0_loss_first_last": [round(losses[0], 4), round(losses[-1], 4)]}
    settings = {"world_size": world_size, "steps": steps, "optimizer": "AdamW(lr=1e-3)", "max_norm": max_norm,
                "replica_every": replica_every or "default (50)", "weights": weights or "default (50)",
                "source": "simulated mechanism (BLOOM-176B account); DeepSpeed not installed"}
    harness.finish(None, run_id, bug=BUG, variant=variant, symptom=symptom, settings=settings, rank=rank,
                   world_size=world_size, relevant=("replica_drift", "grad_spike", "update_spike"))
    dist.destroy_process_group()


def main() -> None:
    p = harness.parser(__doc__)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--max-norm", type=float, default=1.0)
    args = p.parse_args()
    run_id = harness.run_id_for(BUG, args)
    harness.launch(worker, WORLD, run_id, args.variant, run_id, args.steps, args.weights, args.replica_every, args.max_norm)


if __name__ == "__main__":
    main()
