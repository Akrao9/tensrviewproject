"""A forgotten `set_epoch` and a bad resume, caught on CPU, beside a run that does both right.

Trains a small linear model on 36 samples through a `DistributedSampler` (6 batches an epoch)
with `session.watch_loader`, saves a checkpoint with `session.checkpoint`, "crashes" a few steps
later and resumes in a second `tensorquill.init` from that checkpoint, the way a job restarted by
its scheduler does. Then `build_evidence` reads both attempts.

- `--case bad`: `set_epoch` is never called, so every epoch draws the batches in the first
  epoch's order (`epoch_order_repeated`). The checkpoint is taken in the middle of epoch 2 and
  holds the model and optimizer only; the resume forgets `optimizer.load_state_dict` and starts
  the epoch over, so it draws batches the run already trained on (`resume_data_mismatch`) and
  its optimizer starts empty, with no RNG state or data position in the checkpoint
  (`resume_incomplete`).
- `--case good`: `set_epoch` each epoch; the checkpoint is taken at the end of an epoch with the
  model, optimizer, RNG state and the sampler's epoch; the resume restores all of it. None of
  these kinds is reported.

Prints a PASS or FAIL line per case and exits 1 if any fails.

    python examples/data_checkpoint_check.py --case both --out runs/data-check
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import tensorquill

KINDS = ("epoch_order_repeated", "resume_data_mismatch", "resume_incomplete", "data_stalled")
EPOCH = 6  # batches an epoch


def _attempt(case: str, out: Path, *, start: int, epochs: range, state: dict[str, Any] | None,
             save_after: int | None, stop_after: int) -> dict[str, Any] | None:
    """One process's life: train from `start`, save after step `save_after`, stop after `stop_after`."""
    torch.manual_seed(0)
    data = TensorDataset(torch.randn(36, 4), torch.randn(36, 1))
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    sampler = DistributedSampler(data, num_replicas=1, rank=0, shuffle=True, seed=1)
    session = tensorquill.init(model, run_id=f"data-{case}", every=4, start_step=start, heartbeat_seconds=None,
                               sink=tensorquill.JsonlSink(out / f"data-{case}.rank0.jsonl"))
    if state is not None:
        model.load_state_dict(state["model"])
        if case == "good":
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng"])
        session.checkpoint(state, event="load")
    loader = session.watch_loader(DataLoader(data, batch_size=6, sampler=sampler), epoch=epochs.start)
    saved = None
    step = start
    for epoch in epochs:
        if case == "good":
            sampler.set_epoch(epoch)
        for x, y in loader:
            loss = ((model(x) - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            session.log(loss=loss)
            if step == save_after:
                saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
                if case == "good":
                    saved.update(rng=torch.get_rng_state(), sampler={"epoch": epoch + 1})
                saved = copy.deepcopy(saved)
                session.checkpoint(saved)
            if step == stop_after:
                session.close()
                return saved
            step += 1
    session.close()
    return saved


def run(case: str, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"data-{case}.rank0.jsonl").unlink(missing_ok=True)
    if case == "bad":
        # Saved after step 13 (the middle of epoch 2), killed after 16; resumed at 14 from the
        # start of epoch 2.
        state = _attempt(case, out, start=0, epochs=range(4), state=None, save_after=13, stop_after=16)
        _attempt(case, out, start=14, epochs=range(2, 4), state=state, save_after=None, stop_after=23)
    else:
        # Saved after step 11 (the end of epoch 1), killed after 14; resumed at 12 in epoch 2.
        state = _attempt(case, out, start=0, epochs=range(4), state=None, save_after=11, stop_after=14)
        _attempt(case, out, start=12, epochs=range(2, 4), state=state, save_after=None, stop_after=23)
    return tensorquill.build_evidence(out / f"data-{case}.rank0.jsonl")


def check(case: str, evidence: dict[str, Any]) -> tuple[bool, str]:
    found = {i["kind"]: i for i in evidence["incidents"] if i["kind"] in KINDS}
    if case == "good":
        return not found, f"data and checkpoint incidents: {sorted(found) or 'none'}"
    wanted = {"epoch_order_repeated", "resume_data_mismatch", "resume_incomplete"}
    ok = set(found) == wanted
    if ok:
        change = found["resume_data_mismatch"]["changes"][0]
        findings = {(f["component"], f["finding"]) for f in found["resume_incomplete"]["changes"][0]["findings"]}
        ok = (found["epoch_order_repeated"]["step"] == EPOCH and change["finding"] == "repeated"
              and change["matched_step"] == 2 * EPOCH
              and {("optimizer", "empty"), ("rng", "missing"), ("data", "missing")} <= findings)
    return ok, f"found {sorted(found)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=("bad", "good", "both"), default="both")
    parser.add_argument("--out", type=Path, default=Path("tensorquill-out/data-check"))
    args = parser.parse_args(argv)
    failed = False
    for case in ("bad", "good") if args.case == "both" else (args.case,):
        evidence = run(case, args.out)
        ok, detail = check(case, evidence)
        failed |= not ok
        print(f"{'PASS' if ok else 'FAIL'} {case}: {detail}")
        if case == "bad":
            print(tensorquill.incident_summary(evidence))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
