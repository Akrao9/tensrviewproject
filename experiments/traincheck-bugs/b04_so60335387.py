"""B4, SO-60335387: a missed `optimizer.zero_grad()`, so gradients accumulate and training is unstable.

Re-created here from the synopsis: TrainCheck's script for this bug is not in the clone. The
pipeline is TrainCheck's own healthy reference, `docs/assets/code/mnist.py` (the PyTorch MNIST
example: its `Net`, Adadelta at lr 1.0, StepLR with gamma 0.7, batch 64, 51 batches an epoch
as its `batch_idx == 50` break leaves, 2 epochs), which is what TrainCheck's `bugs.md` names
as the input its invariants for this bug were inferred from. MNIST through torchvision is
replaced by a seeded synthetic set of 1x28x28 images, 10 classes.

Buggy: the `optimizer.zero_grad()` line is removed. Healthy: the script as written.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import harness

BUG = "B04-SO-60335387"


class Net(nn.Module):
    """As in TrainCheck's mnist.py (the PyTorch MNIST example)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return F.log_softmax(self.fc2(x), dim=1)


def accuracy(model: nn.Module, data: torch.utils.data.TensorDataset) -> float:
    model.eval()
    with torch.no_grad():
        x, y = data.tensors
        return float((model(x).argmax(1) == y).float().mean())


def main() -> None:
    p = harness.parser(__doc__)
    p.add_argument("--epochs", type=int, default=2, help="mnist.py's default in TrainCheck's copy")
    p.add_argument("--batches", type=int, default=51, help="batches an epoch (mnist.py breaks at batch_idx == 50)")
    args = p.parse_args()
    run_id = harness.run_id_for(BUG, args)
    harness.folder_for(run_id, clean=True)

    torch.manual_seed(1)
    train_set = harness.blobs(64 * args.batches, (1, 28, 28), 10, seed=1, noise=2.0)
    test_set = harness.blobs(1000, (1, 28, 28), 10, seed=2, noise=2.0)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True,
                                               generator=torch.Generator().manual_seed(3))
    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    session = harness.start(model, run_id, weights=args.weights)

    losses, grad_norms = [], []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(session.watch_loader(train_loader, epoch=epoch - 1)):
            if args.variant == "healthy":
                optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            session.log({"train/loss": loss})
            grad_norms.append(float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in model.parameters()))))
            optimizer.step()
            losses.append(loss.item())
            if batch_idx == args.batches - 1:
                break
        print(f"epoch {epoch}: loss {losses[-1]:.4f} accuracy {accuracy(model, test_set):.3f}", flush=True)
        scheduler.step()

    # What one batch's gradient is at the end, against what the optimizer was given.
    model.train()
    optimizer.zero_grad()
    data, target = next(iter(train_loader))
    F.nll_loss(model(data), target).backward()
    one_batch = float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in model.parameters())))
    n = len(losses)
    tenth = max(1, n // 10)
    symptom = {
        "steps": n,
        "grad_norm_first_last": [round(grad_norms[0], 4), round(grad_norms[-1], 4)],
        "grad_norm_last_over_one_batch": round(grad_norms[-1] / one_batch, 2),
        "loss_first_tenth_mean": round(sum(losses[:tenth]) / tenth, 4),
        "loss_last_tenth_mean": round(sum(losses[-tenth:]) / tenth, 4),
        "test_accuracy": round(accuracy(model, test_set), 3),
        "bug_present": args.variant == "buggy",
    }
    settings = {"epochs": args.epochs, "batches_per_epoch": args.batches, "optimizer": "Adadelta(lr=1.0), StepLR(gamma=0.7)",
                "weights": args.weights or "default (50)", "source": "re-created on TrainCheck mnist.py"}
    harness.finish(session, run_id, bug=BUG, variant=args.variant, symptom=symptom, settings=settings,
                   relevant=("grad_spike", "update_spike", "non_finite", "activation_spike"))


if __name__ == "__main__":
    main()
