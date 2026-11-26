"""Hugging Face Accelerate: record the loss and the clipping norm of a custom loop.

    session = tensorquill.init(model, scaler=accelerator.scaler)
    tensorquill.accelerator.watch_accelerator(session)

Accelerate is imported only by this module, so the core SDK keeps no dependency on it.
"""

from __future__ import annotations

from typing import Any

import torch
from accelerate import Accelerator

from . import datawait
from .capture import Session, _clip_is_l2, _clips_session

_active: list["Watch"] = []
_originals: tuple[Any, Any, Any, Any] | None = None


class Watch:
    """What `watch_accelerator` returns. `close()` stops watching; closing the session
    does too."""

    def __init__(self, session: Session, loss_name: str | None, checkpoints: bool = True) -> None:
        self.session = session
        self.loss_name = loss_name
        self.checkpoints = checkpoints
        self._loss: Any = None
        self._loss_step = -1
        # Whether this watch turned on the session's `DataLoader` timer (`data_wait=True` on a
        # session that was not watching its loaders), and so turns it off at `close()`; and whether
        # the session recorded `data_wait` before (`time_data`), which it then still does.
        self._timed_loaders = False
        self._timing_before = False

    def close(self) -> None:
        if self in _active:
            _active.remove(self)
        if self._timed_loaders:
            self._timed_loaders = False
            _stop_loader_timing(self.session, keep_timing=self._timing_before)
        _restore_if_unused()

    def _add_loss(self, loss: Any, accumulation_steps: int, step: int) -> None:
        """Count `loss` once its `Accelerator.backward` has returned; `step` is the session's step
        when that backward began."""
        if self.loss_name is None or not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            return
        if self.session._closed:
            return
        loss = loss.detach()
        if accumulation_steps != 1:
            loss = loss / accumulation_steps  # as `Accelerator.backward` scales it
        # Logged on every backward as the sum so far, and the last value logged in a step
        # is the one kept, so no hook on the optimizer is needed to know when a step ends.
        # (Wrapping `AcceleratedOptimizer.step` would not do: a learning-rate scheduler
        # replaces `step` on the instance.) The sum starts again once the session has
        # moved to a new step, which also holds when a GradScaler skipped the update.
        first = self._loss is None or step != self._loss_step
        self._loss, self._loss_step = (loss, step) if first else (self._loss + loss, step)
        if self.session._next_step == step:
            self.session.log({self.loss_name: self._loss})
        elif self.session._newest_scalars() is not None:
            # The step ended inside the backward: under DeepSpeed `Accelerator.backward` also
            # steps the optimizer at the end of an accumulation (`DeepSpeedEngineWrapper.backward`
            # calls `engine.step()`). The loss is that step's, the newest: added to what its
            # earlier microbatches logged there.
            self.session._add_to_newest(self.loss_name, loss)


def watch_accelerator(
    session: Session, *, loss_name: str | None = "train/loss", checkpoints: bool = True, data_wait: bool = True
) -> Watch:
    """Record what goes through any `Accelerator` for `session`'s model, until the session
    or the returned `Watch` is closed.

    - **Clipping.** The norm `accelerator.clip_grad_norm_` returns, before clipping, is
      recorded as `clip_grad_norm` when the parameters passed include any of the session's
      trainable ones (a LoRA model's first parameter is frozen) and the norm is L2
      (`norm_type` 2, the default; another norm type records nothing). Accelerate 1.16
      stopped clipping through `torch.nn.utils.clip_grad_norm_`, which is what TensorQuill
      wraps by default; this records the norm once on either version.
    - **Loss.** The loss passed to `accelerator.backward` is logged as `loss_name` for
      each optimizer step: the mean over gradient-accumulation microbatches. It stays a
      tensor on the device, with no sync. `loss_name=None` turns this off. A loss counts
      once its `backward` returns: one whose `backward` raises is not added to the step.
      Every loss passed to `backward` counts, so with two models trained through
      Accelerate in one process, turn it off and call `session.log` yourself.
    - **Checkpoints.** After `accelerator.save_state()` and `accelerator.load_state(folder)`, what
      the folder holds is recorded from its file names (`session.checkpoint_files`), so a resume
      that left a component out is reported (`resume_incomplete`); at a load the model's tensors
      are also compared, from the folder's `.safetensors` headers alone (no tensor is read, and
      a folder with no such file is not compared). `checkpoints=False` turns
      this off (the `Trainer` callback records its own checkpoint folders).
    - **Data wait.** The host time spent fetching batches from the `DataLoader` inside
      each prepared `DataLoaderShard` is recorded as `data_wait`
      (`session.watch_loaders()`, see `tensorquill.datawait`). The timer stops before
      Accelerate moves the batch to the device. `data_wait=False` turns it off. Closing
      the watch turns it off again, unless the session was already watching its loaders.

    It wraps `Accelerator.clip_grad_norm_`, `backward`, `save_state` and `load_state` for the whole
    process while something is watched, and restores them afterwards. Call it before or after `accelerator.prepare`.
    """
    global _originals
    watch = Watch(session, loss_name, checkpoints)
    if _originals is None:
        _originals = (Accelerator.clip_grad_norm_, Accelerator.backward, Accelerator.save_state, Accelerator.load_state)
        Accelerator.clip_grad_norm_ = _clip_grad_norm  # type: ignore[method-assign]
        Accelerator.backward = _backward  # type: ignore[method-assign]
        Accelerator.save_state = _save_state  # type: ignore[method-assign]
        Accelerator.load_state = _load_state  # type: ignore[method-assign]
    _active.append(watch)
    if data_wait and not session._closed and session not in datawait._watching:
        # Turned on here, so turned off at `Watch.close()`; a session already watching its loaders
        # (`init(..., data_wait=True)`, `watch_loaders()`) keeps them watched.
        watch._timing_before = session._data_timing
        session.watch_loaders()
        watch._timed_loaders = True
    close = session.close

    def close_and_unwatch(*args: Any, **kwargs: Any) -> None:
        # Whatever `close` takes is passed on: `Session.__exit__` calls
        # `close(_exception=...)`, which the end marker records.
        try:
            close(*args, **kwargs)
        finally:
            watch.close()

    session.close = close_and_unwatch  # type: ignore[method-assign]
    return watch


def _live() -> list[Watch]:
    _active[:] = [w for w in _active if not w.session._closed]
    return _active


def _restore_if_unused() -> None:
    global _originals
    if not _live() and _originals is not None:
        (
            Accelerator.clip_grad_norm_, Accelerator.backward, Accelerator.save_state, Accelerator.load_state,
        ) = _originals  # type: ignore[method-assign]
        _originals = None


def _clip_grad_norm(self: Any, parameters: Any, *args: Any, **kwargs: Any) -> Any:
    original = _originals[0]  # type: ignore[index]
    if isinstance(parameters, torch.Tensor):
        # Unlike torch's, Accelerate's clip_grad_norm_ (1.15 and 1.16.0.dev0) iterates a
        # single tensor into its rows, which have no gradients, and returns 0. Recording
        # that would be a false norm, so a bare tensor is passed through unrecorded.
        norm = original(self, parameters, *args, **kwargs)
        _restore_if_unused()
        return norm
    parameters = list(parameters)
    # The same match and the same norm rule as torch's wrapper (`capture._clips_session`,
    # `capture._clip_is_l2`): any of the session's trainable parameters, an L2 norm only.
    sessions = [w.session for w in _live() if w.session._clip_watched and _clips_session(w.session, parameters)]
    before = [s._clip_calls for s in sessions]
    norm = original(self, parameters, *args, **kwargs)
    if norm is not None and _clip_is_l2(args, kwargs):
        for session, calls in zip(sessions, before):
            # Accelerate up to 1.15 clips through `torch.nn.utils.clip_grad_norm_`, which
            # TensorQuill already wraps; record only when that did not happen.
            if session._clip_calls == calls:
                session.record_grad_norm(norm)
    _restore_if_unused()
    return norm


def _backward(self: Any, loss: Any, **kwargs: Any) -> Any:
    original = _originals[1]  # type: ignore[index]
    # A loss counts once its backward has returned: one whose backward raises (a loss with no
    # gradient, an out-of-memory error a loop catches) is not added to the step. The step each
    # session was at is taken before, for a backward that ends a step itself (DeepSpeed).
    watched = [(watch, watch.session._next_step) for watch in _live()]
    try:
        result = original(self, loss, **kwargs)
    except BaseException:
        _restore_if_unused()
        raise
    for watch, step in watched:
        watch._add_loss(loss, self.gradient_accumulation_steps, step)
    _restore_if_unused()
    return result


def _stop_loader_timing(session: Session, *, keep_timing: bool) -> None:
    """Undo the `session.watch_loaders()` a watch made: the `DataLoader` timer stops counting for
    the session, and unless it recorded `data_wait` before the watch (`keep_timing`, from
    `time_data`), its steps are recorded with no `data_wait` again, the step under way too. A
    closed session has already stopped it."""
    if session._closed:
        return
    datawait.unwatch(session)
    if not keep_timing:
        session._data_timing = False
        session._pending_data_wait = 0.0



def _note_folder(folder: Any, event: str) -> None:
    for watch in _live():
        if watch.checkpoints and folder is not None:
            try:
                watch.session.checkpoint_files(folder, event=event)
            except Exception:  # noqa: BLE001 - a manifest never fails a save or a load
                pass


def _save_state(self: Any, output_dir: Any = None, *args: Any, **kwargs: Any) -> Any:
    original = _originals[2]  # type: ignore[index]
    location = original(self, output_dir, *args, **kwargs)
    _note_folder(location if location is not None else output_dir, "save")
    _restore_if_unused()
    return location


def _load_state(self: Any, input_dir: Any = None, *args: Any, **kwargs: Any) -> Any:
    original = _originals[3]  # type: ignore[index]
    result = original(self, input_dir, *args, **kwargs)
    # Without a folder Accelerate picks the newest under its project folder, which is not
    # known here: nothing is recorded.
    _note_folder(input_dir, "load")
    _restore_if_unused()
    return result
