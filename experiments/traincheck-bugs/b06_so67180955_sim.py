"""B6, SO-67180955: data-loader workers share one random seed, so they produce identical random data.

Simulated here. TrainCheck's script is not in the clone, and its AE guide leaves this bug
out because it "requires older Python version no longer supported". The reading taken here:
before PyTorch 1.9 every `DataLoader` worker inherited the parent's NumPy random state, so
a dataset that draws its augmentation from `np.random` handed every worker the same
sequence; PyTorch 1.9 seeds NumPy in each worker and the bug cannot happen on 2.15 as
written. The old behaviour is put back with a `worker_init_fn` that seeds NumPy with the same
value in every worker.

A map-style dataset adds NumPy noise to each sample (the augmentation), four workers, a small
MLP, Adam. Buggy: the `worker_init_fn` above. Healthy: PyTorch's own per-worker seeding.
TensorQuill watches the loader (`session.watch_loader`), which hashes each batch's sample
indices.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness

BUG = "B06-SO-67180955"
WORKERS = 4


class NoisyBlobs(torch.utils.data.Dataset):
    def __init__(self, n: int) -> None:
        self.base = harness.blobs(n, (64,), 10, seed=1, noise=1.0)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):  # type: ignore[no-untyped-def]
        x, y = self.base[i]
        noise = np.random.normal(0.0, 1.0, size=x.shape).astype(np.float32)  # the augmentation
        return x + torch.from_numpy(noise), y, torch.tensor(float(noise[0]))


def same_seed_in_every_worker(worker_id: int) -> None:
    np.random.seed(1234)  # what every worker inherited before PyTorch 1.9


def main() -> None:
    p = harness.parser(__doc__)
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()
    run_id = harness.run_id_for(BUG, args)
    harness.folder_for(run_id, clean=True)

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    session = harness.start(model, run_id, weights=args.weights)
    loader = torch.utils.data.DataLoader(
        NoisyBlobs(32 * args.steps), batch_size=32, shuffle=True, num_workers=WORKERS,
        generator=torch.Generator().manual_seed(3),
        worker_init_fn=same_seed_in_every_worker if args.variant == "buggy" else None,
    )
    marks: list[float] = []
    losses: list[float] = []
    for x, y, mark in session.watch_loader(loader):
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        session.log({"train/loss": loss})
        optimizer.step()
        losses.append(loss.item())
        marks.extend(mark.tolist())
    distinct = len(set(marks))
    symptom = {
        "steps": len(losses),
        "samples": len(marks),
        "distinct_augmentations": distinct,
        "repeated_augmentation_share": round(1 - distinct / len(marks), 3),
        "train_loss_first_last": [round(losses[0], 4), round(losses[-1], 4)],
        "bug_present": distinct < len(marks) // 2,
    }
    settings = {"steps": args.steps, "workers": WORKERS, "optimizer": "Adam(lr=1e-3)",
                "weights": args.weights or "default (50)",
                "source": "simulated pre-1.9 worker seeding through worker_init_fn"}
    harness.finish(session, run_id, bug=BUG, variant=args.variant, symptom=symptom, settings=settings,
                   relevant=("data_stalled", "epoch_order_repeated", "duplicate_batches"))


if __name__ == "__main__":
    main()
