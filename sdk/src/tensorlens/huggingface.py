"""Hugging Face `Trainer` callback: `Trainer(..., callbacks=[TensorLensCallback(...)])`.

Transformers and Accelerate are imported only by this module, so the core SDK keeps no
dependency on them. `SFTTrainer` and the other TRL trainers subclass `Trainer` and take
the same callback.
"""

from __future__ import annotations

from typing import Any

import torch
from accelerate import Accelerator
from transformers import TrainerCallback

from .capture import Session, init
from .tracker import tracker_run_id

_active: list["TensorLensCallback"] = []
_originals: tuple[Any, Any] | None = None


class TensorLensCallback(TrainerCallback):
    """Start TensorLens when training starts and stop it when training ends.

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
      callback wraps `Accelerator.clip_grad_norm_` and `Accelerator.backward` to record
      the gradient norm before clipping and the loss, as `loss_name`, for each optimizer
      step (the sum over gradient-accumulation microbatches of the loss `Trainer`
      back-propagates, which is their mean). Both stay tensors on the device, with no
      sync. `loss_name=None` turns the loss off.

    `Trainer` has no event for a failed run. If `trainer.train()` raises, call
    `callback.close()` to write what was captured.

    Other keyword arguments go to `tensorlens.init`, for example `routers=`, `capture=`
    or `probe=`.
    """

    def __init__(self, *, loss_name: str | None = "train/loss", **init_kwargs: Any) -> None:
        self.loss_name = loss_name
        self.init_kwargs = init_kwargs
        self.session: Session | None = None
        self._loss: Any = None

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
        self._loss = None
        self.session = init(model, run_id=run_id, **kwargs)
        _watch_accelerator(self)

    def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        # After the last microbatch's backward and before the optimizer's update, so the
        # loss belongs to the step TensorLens is about to record.
        if self.session is not None and self._loss is not None:
            self.session.log({self.loss_name: self._loss})
            self._loss = None

    def on_train_end(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Write what was captured and stop. Safe to call twice."""
        _unwatch_accelerator(self)
        if self.session is not None:
            session, self.session = self.session, None
            session.close()

    def _add_loss(self, loss: Any) -> None:
        if self.loss_name is not None and isinstance(loss, torch.Tensor) and loss.numel() == 1:
            loss = loss.detach()
            self._loss = loss if self._loss is None else self._loss + loss


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


def _clip_grad_norm(self: Any, parameters: Any, *args: Any, **kwargs: Any) -> Any:
    parameters = list(parameters)
    sessions = [
        c.session for c in _active
        if c.session is not None and c.session._clip_watched and parameters and id(parameters[0]) in c.session._param_ids
    ]
    before = [s._clip_calls for s in sessions]
    norm = _originals[0](self, parameters, *args, **kwargs)  # type: ignore[index]
    norm_type = args[1] if len(args) > 1 else kwargs.get("norm_type", 2)
    if norm is not None and norm_type == 2:
        for session, calls in zip(sessions, before):
            # Older Accelerate versions clip through `torch.nn.utils.clip_grad_norm_`,
            # which TensorLens already wraps; record only when that did not happen.
            if session._clip_calls == calls:
                session.record_grad_norm(norm)
    return norm


def _backward(self: Any, loss: Any, **kwargs: Any) -> Any:
    for callback in _active:
        callback._add_loss(loss)
    return _originals[1](self, loss, **kwargs)  # type: ignore[index]


def _watch_accelerator(callback: TensorLensCallback) -> None:
    global _originals
    if not _active:
        _originals = (Accelerator.clip_grad_norm_, Accelerator.backward)
        Accelerator.clip_grad_norm_ = _clip_grad_norm  # type: ignore[method-assign]
        Accelerator.backward = _backward  # type: ignore[method-assign]
    _active.append(callback)


def _unwatch_accelerator(callback: TensorLensCallback) -> None:
    if callback in _active:
        _active.remove(callback)
        if not _active and _originals is not None:
            Accelerator.clip_grad_norm_, Accelerator.backward = _originals  # type: ignore[method-assign]
