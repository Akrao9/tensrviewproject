"""Hugging Face `Trainer` callback: `Trainer(..., callbacks=[TensorQuillCallback(...)])`.

Transformers and Accelerate are imported only by this module, so the core SDK keeps no
dependency on them. `SFTTrainer` and the other TRL trainers subclass `Trainer` and take
the same callback.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import TrainerCallback

from .accelerator import Watch, watch_accelerator
from .capture import Session, init
from .tracker import tracker_run_id


class TensorQuillCallback(TrainerCallback):
    """Start TensorQuill when training starts and stop it when training ends.

    - **Model.** `Trainer` compiles the model (`torch_compile=True`) before any callback
      runs, but the compiled forward is traced on its first call, so hooks added in
      `on_train_begin` are part of it.
    - **Run ID.** `Trainer` starts the W&B or MLflow run on the main process only. The
      callback reads it there and broadcasts it, so every rank writes under one run.
    - **Steps** start at `state.global_step`, so a resumed run continues the numbering,
      and are counted from optimizer steps, as `global_step` is.
    - **Mixed precision.** Accelerate's `GradScaler`, when it has one, is passed as
      `scaler=`, so steps the scaler skips are counted and marked.
    - **Clipping and loss.** `Trainer` clips and back-propagates through its
      `Accelerator`, and gives callbacks neither value. While a session is open the
      callback watches the `Accelerator` (`tensorquill.accelerator.watch_accelerator`) to
      record the gradient norm before clipping and the loss, as `loss_name`, for each
      optimizer step (the sum over gradient-accumulation microbatches of the loss
      `Trainer` back-propagates, which is their mean). Both stay tensors on the device,
      with no sync. `loss_name=None` turns the loss off.

    `Trainer` has no event for a failed run. If `trainer.train()` raises, call
    `callback.close()` to write what was captured.

    Other keyword arguments go to `tensorquill.init`, for example `routers=`, `capture=`
    or `probe=`.
    """

    def __init__(self, *, loss_name: str | None = "train/loss", **init_kwargs: Any) -> None:
        self.loss_name = loss_name
        self.init_kwargs = init_kwargs
        self.session: Session | None = None
        self._watch: Watch | None = None

    def on_train_begin(self, args: Any, state: Any, control: Any, model: Any = None, optimizer: Any = None, **_: Any) -> None:
        self.close()  # a previous train() that raised leaves its session open
        kwargs = dict(self.init_kwargs)
        run_id = kwargs.pop("run_id", None)
        if run_id is None:
            run_id = _shared_tracker_run_id()
        # AcceleratedOptimizer keeps the scaler it steps and updates.
        scaler = getattr(optimizer, "scaler", None)
        if scaler is not None:
            kwargs.setdefault("scaler", scaler)
        kwargs.setdefault("start_step", state.global_step)
        self.session = init(model, run_id=run_id, **kwargs)
        self._watch = watch_accelerator(self.session, loss_name=self.loss_name)

    # `Trainer` has no event before an evaluation or a save, only after. But its flow callback
    # sets `control.should_evaluate` / `should_save` at the end of the step or epoch, just
    # before it does them, so that is when the stretch with no training steps begins. This
    # relies on the flow callback running before this one, as it does by default.
    def _expect_pause(self, control: Any) -> None:
        if self.session is None:
            return
        if getattr(control, "should_evaluate", False):
            self.session.set_phase("eval")
        elif getattr(control, "should_save", False):
            self.session.set_phase("checkpoint")

    def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        self._expect_pause(control)

    def on_epoch_end(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        self._expect_pause(control)

    def on_prediction_step(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        if self.session is not None:
            self.session.tick()  # one evaluation batch: the pause is working, not stuck

    def on_evaluate(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        if self.session is not None:
            self.session.set_phase("checkpoint" if getattr(control, "should_save", False) else "train")

    def on_save(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        if self.session is not None:
            self.session.set_phase("train")

    def on_train_end(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Write what was captured and stop. Safe to call twice."""
        if self._watch is not None:
            watch, self._watch = self._watch, None
            watch.close()
        if self.session is not None:
            session, self.session = self.session, None
            session.close()


def _shared_tracker_run_id() -> str | None:
    """The tracker run rank 0 sees, on every rank. None when there is none, in which case
    `init` falls back to the launcher's job ID or a generated one, shared the same way."""
    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        found = tracker_run_id()
        return found[0] if found else None
    found = tracker_run_id() if dist.get_rank() == 0 else None
    box = [found[0] if found else None]
    dist.broadcast_object_list(box, src=0)
    return box[0]
