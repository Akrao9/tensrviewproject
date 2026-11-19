"""B1, PT-84911: the forum post where only BatchNorm layers are trainable and training does not progress.

Source: TrainCheck's own reproduction, `docs/assets/code/84911.py` in the TrainCheck clone
(7a62cbb). Its training and validation loops, its two freezing loops and its optimizer line
are kept as written. Two things are swapped because they need downloads or packages that
are not installed: the pretrained EfficientNet-b0 (`efficientnet_pytorch` plus weights) is
replaced by a small randomly initialized CNN whose parameters carry the same names
(`_conv_stem`, `_bn0`, `_blocks.N._bn1`, `_fc`, ...), so the `"bn" not in name` test picks
the same kinds of parameters; CIFAR-100 through torchvision is replaced by a seeded
synthetic image set (10 classes, 3x32x32). The `traincheck` stage annotations are dropped.

The bug: every parameter whose name lacks "bn" is frozen, then `_fc` is frozen too, and the
optimizer is given only `_fc`'s parameters. Nothing the optimizer holds has a gradient, the
BatchNorm parameters get gradients but no optimizer holds them, so no weight ever changes;
only BatchNorm's running statistics move.

Healthy variant (written here, not TrainCheck's `mnist.py`, which is another pipeline): the
same script without the two freezing loops and with the optimizer over all parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

import harness

BUG = "B01-PT-84911"


class MBConvBlock(nn.Module):
    """A small stand-in for EfficientNet's MBConvBlock, called as `block(x)` as there."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self._depthwise_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self._bn1 = nn.BatchNorm2d(channels)
        self._project_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self._bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.silu(self._bn1(self._depthwise_conv(x)))
        return x + self._bn2(self._project_conv(y))


class TinyEfficientNet(nn.Module):
    """Stands in for EfficientNet-b0 with the same parameter-name pattern."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self._conv_stem = nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False)
        self._bn0 = nn.BatchNorm2d(16)
        self._blocks = nn.ModuleList([MBConvBlock(16) for _ in range(2)])
        self._conv_head = nn.Conv2d(16, 32, 1, bias=False)
        self._bn1 = nn.BatchNorm2d(32)
        self._fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.silu(self._bn0(self._conv_stem(x)))
        for block in self._blocks:
            x = block(x)
        x = torch.nn.functional.silu(self._bn1(self._conv_head(x)))
        return self._fc(x.mean(dim=(2, 3)))


def main() -> None:
    p = harness.parser(__doc__)
    p.add_argument("--epochs", type=int, default=2, help="84911.py trains 2 epochs")
    p.add_argument("--batches", type=int, default=11, help="training batches per epoch (84911.py breaks at batch_idx == 10)")
    p.add_argument("--init-at", choices=("to", "loop"), default="to",
                   help="where tensorquill.init is called: right after model.to(device), as the guide says, "
                        "which in 84911.py comes before the freezing loops; or just before the training loop")
    args = p.parse_args()
    run_id = harness.run_id_for(BUG, args)
    harness.folder_for(run_id, clean=True)

    seed = 786  # as in 84911.py
    torch.manual_seed(seed)
    train_set = harness.blobs(64 * args.batches, (3, 32, 32), 10, seed=1)
    valid_set = harness.blobs(64, (3, 32, 32), 10, seed=2)
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1, num_workers=0, shuffle=False)
    loaders = {"train": train_loader, "valid": valid_loader}

    model_transfer = TinyEfficientNet()
    n_inputs = model_transfer._fc.in_features
    num_classes = 10  # 100 in 84911.py (CIFAR-100); the synthetic set has 10
    model_transfer._fc = nn.Linear(n_inputs, num_classes)
    device = torch.device("cpu")
    criterion_transfer = nn.CrossEntropyLoss()

    if args.variant == "buggy":
        # --- verbatim from 84911.py -------------------------------------------------------
        optimizer_transfer = optim.Adadelta(model_transfer._fc.parameters(), lr=1)
        model_transfer.to(device)
        session = harness.start(model_transfer, run_id, weights=args.weights) if args.init_at == "to" else None
        for name, param in model_transfer.named_parameters():
            if "bn" not in name:
                param.requires_grad = False
        for param in model_transfer._fc.parameters():
            param.requires_grad = False
        # ---------------------------------------------------------------------------------
    else:
        optimizer_transfer = optim.Adadelta(model_transfer.parameters(), lr=1)
        model_transfer.to(device)
        session = harness.start(model_transfer, run_id, weights=args.weights) if args.init_at == "to" else None

    if session is None:
        session = harness.start(model_transfer, run_id, weights=args.weights)
    before = harness.param_snapshot(model_transfer)
    losses = []

    model = model_transfer
    optimizer = optimizer_transfer
    criterion = criterion_transfer
    for epoch in range(1, args.epochs + 1):
        train_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(session.watch_loader(loaders["train"], epoch=epoch - 1)):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            session.log({"train/loss": loss})
            optimizer.step()
            train_loss += (1 / (batch_idx + 1)) * (loss.item() - train_loss)
            losses.append(loss.item())
            if batch_idx == args.batches - 1:
                break
        model.eval()
        valid_loss = 0.0
        for batch_idx, (data, target) in enumerate(loaders["valid"]):
            with torch.no_grad():
                output = model(data)
            loss = criterion(output, target)
            valid_loss += (1 / (batch_idx + 1)) * (float(loss) - valid_loss)
            if batch_idx == 5:
                break
        print(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}", flush=True)

    changed = harness.changed_params(model, before)
    held = {id(p) for g in optimizer.param_groups for p in g["params"]}
    with_grad = [n for n, p in model.named_parameters() if p.requires_grad]
    symptom = {
        "steps": len(losses),
        "parameters": len(before),
        "parameters_changed": len(changed),
        "parameters_with_requires_grad": len(with_grad),
        "of_those_held_by_optimizer": sum(1 for n, p in model.named_parameters() if p.requires_grad and id(p) in held),
        "loss_first_last": [round(losses[0], 4), round(losses[-1], 4)],
        "bug_present": len(changed) == 0,
    }
    settings = {"epochs": args.epochs, "batches_per_epoch": args.batches, "optimizer": "Adadelta(lr=1)",
                "weights": args.weights or "default (50)", "init_at": args.init_at, "source": "TrainCheck 84911.py, model and data swapped"}
    harness.finish(session, run_id, bug=BUG, variant=args.variant, symptom=symptom, settings=settings,
                   relevant=("stalled_weights", "update_spike"))


if __name__ == "__main__":
    main()
