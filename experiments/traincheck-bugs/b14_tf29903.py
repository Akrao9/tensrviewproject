"""B14, TF-29903: a checkpointing function leaves some LSTM weights out of the saved state dict.

Re-created here from the synopsis ("Certain LSTM weights are not saved due to logic bug in
the checkpointing function"; the paper adds that the corrupted state dict is built inside a
`safe_checkpoint` function). TrainCheck's script is not in the clone, and Transformers 5.23,
the installed version, is not the one with the bug.

A small LSTM classifier on seeded synthetic sequences, AdamW. The run trains 100 steps,
saves a checkpoint (model, optimizer, step), and resumes from it in a second session, as a
restarted job would: a new model, `load_state_dict(strict=False)` (Transformers loads with
missing keys and initializes them afresh), the optimizer state, `start_step=100`, 100 more
steps. TensorQuill records the save and the load (`session.checkpoint`), as its guide says.

Buggy: the checkpoint function drops the LSTM's hidden-to-hidden weight and bias. Healthy:
it saves the whole state dict. `--checkpoint minimal` saves only model, optimizer and step
(no RNG state, no data position).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import harness

BUG = "B14-TF-29903"
DROPPED = ("lstm.weight_hh_l0", "lstm.bias_hh_l0")


class SeqClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(8, 32, batch_first=True)
        self.head = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def sequences(n: int, seed: int) -> torch.utils.data.TensorDataset:
    flat = harness.blobs(n, (16 * 8,), 10, seed=seed, noise=2.0)
    x, y = flat.tensors
    return torch.utils.data.TensorDataset(x.reshape(n, 16, 8), y)


def train(session, model, optimizer, loader, steps, losses):  # type: ignore[no-untyped-def]
    done = 0
    while done < steps:
        for x, y in loader:
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            session.log({"train/loss": loss})
            optimizer.step()
            losses.append(loss.item())
            done += 1
            if done == steps:
                return


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int, buggy: bool, full: bool) -> dict:
    model_state = {k: v.clone() for k, v in model.state_dict().items() if not (buggy and k in DROPPED)}
    state = {"model": model_state, "optimizer": optimizer.state_dict(), "step": step}
    if full:  # what a Trainer checkpoint also holds: the RNG states and where the data stands
        state["rng"] = torch.get_rng_state()
        state["data_position"] = {"step": step}
    return state


def main() -> None:
    p = harness.parser(__doc__)
    p.add_argument("--steps", type=int, default=100, help="steps before and after the resume")
    p.add_argument("--checkpoint", choices=("full", "minimal"), default="full",
                   help="full: model, optimizer, step, RNG state and data position; minimal: the first three")
    args = p.parse_args()
    run_id = harness.run_id_for(BUG, args)
    harness.folder_for(run_id, clean=True)
    data = sequences(32 * 50, seed=1)
    losses: list[float] = []

    torch.manual_seed(0)
    model = SeqClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    session = harness.start(model, run_id, weights=args.weights)
    loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(3))
    train(session, model, optimizer, loader, args.steps, losses)
    state = save_checkpoint(model, optimizer, args.steps, args.variant == "buggy", args.checkpoint == "full")
    session.checkpoint(state, event="save")
    saved_hh = model.lstm.weight_hh_l0.detach().clone()
    session.close()

    # The restarted job.
    torch.manual_seed(1)
    model = SeqClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    missing = model.load_state_dict(state["model"], strict=False).missing_keys
    optimizer.load_state_dict(state["optimizer"])
    session = harness.start(model, run_id, weights=args.weights, start_step=state["step"])
    session.checkpoint(state, event="load")
    loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(4))
    resumed_hh_diff = float((model.lstm.weight_hh_l0.detach() - saved_hh).abs().max())
    before_resume = losses[-1]
    train(session, model, optimizer, loader, args.steps, losses)

    symptom = {
        "steps": len(losses),
        "missing_keys_at_load": missing,
        "lstm_weight_hh_max_abs_change_across_resume": round(resumed_hh_diff, 4),
        "loss_last_before_first_after_resume": [round(before_resume, 4), round(losses[args.steps], 4)],
        "loss_last": round(losses[-1], 4),
        "bug_present": bool(missing),
    }
    settings = {"steps_per_attempt": args.steps, "checkpoint": args.checkpoint, "optimizer": "AdamW(lr=1e-3)", "weights": args.weights or "default (50)",
                "source": "re-created from synopsis"}
    harness.finish(session, run_id, bug=BUG, variant=args.variant, symptom=symptom, settings=settings,
                   relevant=("resume_incomplete", "resume_lr_change", "grad_spike", "update_spike",
                             "activation_spike", "repeated_steps"))


if __name__ == "__main__":
    main()
