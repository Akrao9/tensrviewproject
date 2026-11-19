"""B19, PT-51800: SpectralNorm layers need a forward in train mode before evaluation; evaluating first gives misleading results.

Re-created here from the synopsis and TrainCheck's `bugs.md` ("before `eval`, forward should
be called under `train`"); TrainCheck's script is not in the clone. `torch.nn.utils.
spectral_norm` (the original API, still in PyTorch 2.15) starts its power-iteration vectors
`u` and `v` at random and only iterates in train mode, so an evaluation before any training
forward divides each weight by `u . W v` for random vectors instead of its largest singular
value. (`torch.nn.utils.parametrizations.spectral_norm` iterates 15 times when it is
registered and does not have the problem.)

A small MLP with two spectral-normalized linear layers on a seeded synthetic set, Adam.
Buggy: the script evaluates the validation set right after building the model, then
trains. Healthy: one forward in train mode, without gradients, before that evaluation.
TensorQuill is started right after the model is built, before either.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

import harness

BUG = "B19-PT-51800"


def sigma_used_over_true(layer: nn.Linear) -> float:
    w = layer.weight_orig.detach()
    used = float(torch.dot(layer.weight_u, torch.mv(w, layer.weight_v)))
    return used / float(torch.linalg.matrix_norm(w, 2))


def main() -> None:
    p = harness.parser(__doc__)
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()
    run_id = harness.run_id_for(BUG, args)
    harness.folder_for(run_id, clean=True)

    torch.manual_seed(0)
    model = nn.Sequential(spectral_norm(nn.Linear(64, 128)), nn.ReLU(), spectral_norm(nn.Linear(128, 10)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    session = harness.start(model, run_id, weights=args.weights)
    train = harness.blobs(32 * args.steps, (64,), 10, seed=1, noise=3.0)
    valid = harness.blobs(512, (64,), 10, seed=2, noise=3.0)
    vx, vy = valid.tensors

    if args.variant == "healthy":
        model.train()
        with torch.no_grad():
            model(vx[:32])
    model.eval()
    with torch.no_grad():
        loss_before = F.cross_entropy(model(vx), vy).item()
        acc_before = float((model(vx).argmax(1) == vy).float().mean())
    ratios_at_eval = [round(sigma_used_over_true(model[0]), 4), round(sigma_used_over_true(model[2]), 4)]

    model.train()
    loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(3))
    losses = []
    for x, y in session.watch_loader(loader):
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        session.log({"train/loss": loss})
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        loss_after = F.cross_entropy(model(vx), vy).item()

    symptom = {
        "steps": len(losses),
        "sigma_used_over_true_at_first_eval": ratios_at_eval,
        "valid_loss_before_training": round(loss_before, 4),
        "valid_accuracy_before_training": round(acc_before, 3),
        "valid_loss_after_training": round(loss_after, 4),
        "train_loss_first_last": [round(losses[0], 4), round(losses[-1], 4)],
        "bug_present": args.variant == "buggy" and any(abs(r - 1) > 0.05 for r in ratios_at_eval),
    }
    settings = {"steps": args.steps, "optimizer": "Adam(lr=1e-3)", "weights": args.weights or "default (50)",
                "source": "re-created from synopsis"}
    harness.finish(session, run_id, bug=BUG, variant=args.variant, symptom=symptom, settings=settings,
                   relevant=("activation_spike", "grad_spike", "update_spike"))


if __name__ == "__main__":
    main()
