"""Hugging Face `Trainer` callback: `Trainer(..., callbacks=[TensorQuillCallback(...)])`.

Transformers and Accelerate are imported only by this module, so the core SDK keeps no
dependency on them. `SFTTrainer` and the other TRL trainers subclass `Trainer` and take
the same callback.
"""

from __future__ import annotations

import atexit
import inspect
import os
import sys
from pathlib import Path
from typing import Any

from transformers import TrainerCallback
from transformers.trainer_callback import CallbackHandler

from .accelerator import Watch, watch_accelerator
from .capture import Session, init


class TensorQuillCallback(TrainerCallback):
    """Start TensorQuill when training starts and stop it when training ends.

    - **Model.** `Trainer` compiles the model (`torch_compile=True`) before any callback
      runs, but the compiled forward is traced on its first call, so hooks added in
      `on_train_begin` are part of it.
    - **Run ID.** `Trainer` starts the W&B or MLflow run on the main process only. `init`
      reads it there and shares it, with where it came from, with every rank through the
      process group's store (no collective, nothing on a GPU), so every rank writes under one
      run. A `run_id=` given to the callback goes to `init` as it is: with several ranks,
      rank 0's wins.
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
    - **Data wait.** The host time spent fetching each step's batches from the `DataLoader`
      under Accelerate's `DataLoaderShard`, before the batch is moved to the device, is
      recorded as `data_wait` (`tensorquill.datawait`). `data_wait=False` turns it off.

    - **A `train()` that raises.** `Trainer` has no event for it (it calls `on_train_end`
      only at the end of a finished loop), so the session stays open after the raise: the
      newest step is still held and the file has no end marker. It is closed by the first of
      `callback.close()`, the next `train()` and the end of the process (an `atexit` hook that
      is there only while a session is open). Each of these closes it as raised, since
      `on_train_end` was not reached: the end marker's `outcome` is `"raised"`, with the
      exception's type when it can be told for this `train()` (one being handled where
      `close()` is called, as in `except:` or `finally:`, or the uncaught one the process ends
      with) and `TrainEndNotReached` otherwise. Nothing is raised into the program and the
      exception is not touched; in a process that goes on (a notebook, a caught
      `KeyboardInterrupt`), call `callback.close()` to close it at once.

    Other keyword arguments go to `tensorquill.init`, for example `routers=`, `capture=`
    or `probe=`.
    """

    def __init__(self, *, loss_name: str | None = "train/loss", **init_kwargs: Any) -> None:
        self.loss_name = loss_name
        self.init_kwargs = init_kwargs
        self.session: Session | None = None
        self._watch: Watch | None = None
        # While a session is open and its train() has not reached on_train_end: the frame that
        # called on_train_begin, as (id, code), to tell an exception that came out of it.
        self._training: tuple[int, Any] | None = None
        self._pid: int | None = None  # the process that registered the atexit hook

    def on_train_begin(self, args: Any, state: Any, control: Any, model: Any = None, optimizer: Any = None, **_: Any) -> None:
        self.close()  # a previous train() that raised leaves its session open: closed as raised
        # The run ID, a `run_id=` given here included, is init's to resolve: rank 0's goes to
        # every rank through the process group's store, so every rank takes the same path
        # whatever it was given.
        kwargs = dict(self.init_kwargs)
        # AcceleratedOptimizer keeps the scaler it steps and updates.
        scaler = getattr(optimizer, "scaler", None)
        if scaler is not None:
            kwargs.setdefault("scaler", scaler)
        kwargs.setdefault("start_step", state.global_step)
        kwargs.setdefault("data_wait", True)
        self.session = init(model, **kwargs)
        self._watch = watch_accelerator(self.session, loss_name=self.loss_name, checkpoints=False, data_wait=False)
        self._training = _training_frame(self)
        self._pid = os.getpid()
        atexit.register(self._at_exit)
        if state.global_step > 0:
            # Resumed: the Trainer loaded `checkpoint-<global_step>` (or the folder it was given).
            folder = _checkpoint_folder(args, state, getattr(args, "resume_from_checkpoint", None))
            if folder is not None:
                self.session.checkpoint_files(folder, event="load")

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
            # The Trainer has just written `checkpoint-<global_step>` under the output folder.
            folder = _checkpoint_folder(args, state)
            if folder is not None:
                self.session.checkpoint_files(folder, event="save")

    def on_train_end(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        self._training = None  # the loop finished: a clean close
        self.close()

    def close(self) -> None:
        """Write what was captured and stop. Safe to call twice; never raises.

        Called while a `train()` is open (it began and did not reach `on_train_end`, so it
        raised), it closes the session as raised: the end marker names the exception being
        handled where `close()` is called, or the uncaught one the process ends with, when that
        exception came out of this `train()`, and `TrainEndNotReached` otherwise."""
        exception = self._stopped_by() if self.session is not None and self._training is not None else None
        self._training = None
        if self._pid is not None:
            self._pid = None
            try:
                atexit.unregister(self._at_exit)
            except Exception:  # noqa: BLE001  nothing here may stop the program
                pass
        if self._watch is not None:
            watch, self._watch = self._watch, None
            watch.close()
        if self.session is not None:
            session, self.session = self.session, None
            session.close(_exception=exception)

    def _at_exit(self) -> None:
        """The process ends with a session open: its `train()` raised and nothing closed it."""
        if self._pid != os.getpid():
            return  # a forked child (a DataLoader worker) that runs atexit hooks: not its session
        try:
            self.close()
        except BaseException:  # noqa: BLE001  a second Ctrl-C during the close: the exit goes on
            pass

    def _stopped_by(self) -> BaseException:
        """The exception that ended this `train()`, if one can be told, else `TrainEndNotReached`.

        One is named only if its traceback passes through the frame that called
        `on_train_begin` (`_training`): the one being handled where `close()` is called
        (`sys.exc_info()`, set in `except:` and `finally:`), then the last uncaught one
        (`sys.last_exc`, `sys.last_value` before Python 3.12: set by the interpreter for the
        exception a script ends with, and by IPython for the last one it showed). With each, the
        exceptions it was raised from or during are looked at too. A `train()` called again
        after a caught raise has a frame of its own, so an older exception is not named."""
        try:
            seen: set[int] = set()
            for candidate in (sys.exc_info()[1], getattr(sys, "last_exc", None), getattr(sys, "last_value", None)):
                while isinstance(candidate, BaseException) and id(candidate) not in seen:
                    seen.add(id(candidate))
                    if _passed_through(candidate, self._training):
                        return candidate
                    candidate = candidate.__cause__ or candidate.__context__
        except Exception:  # noqa: BLE001  nothing here may stop the program
            pass
        return TrainEndNotReached("train() did not reach on_train_end, and what stopped it is not known here")


class TrainEndNotReached(Exception):
    """The type an end marker names when a `TensorQuillCallback` session was closed after its
    `train()` stopped before `on_train_end`, and the exception that stopped it could not be
    told (it was caught and is no longer being handled, or the process ended by `sys.exit()`).
    It is never raised: it says the run ended by a raise of an unknown type, not a clean end."""


# A train() whose frame could not be read: open all the same, and no exception is named for it.
_UNKNOWN_FRAME: tuple[int, Any] = (0, None)


def _training_frame(callback: TensorQuillCallback) -> tuple[int, Any]:
    """The frame that called `on_train_begin` through the `Trainer`'s `CallbackHandler`
    (`Trainer._inner_training_loop`, or what called the callback directly), as (id, code),
    or `_UNKNOWN_FRAME`. The frame itself is not kept, so its locals are not held after a raise."""
    handler = CallbackHandler.call_event.__code__.co_filename
    frame = inspect.currentframe()
    try:
        while frame is not None and (
            frame.f_code.co_filename == handler or frame.f_locals.get("self") is callback
            or frame.f_code is _training_frame.__code__
        ):
            frame = frame.f_back
        return _UNKNOWN_FRAME if frame is None else (id(frame), frame.f_code)
    except Exception:  # noqa: BLE001  without it, an exception is not named: TrainEndNotReached
        return _UNKNOWN_FRAME
    finally:
        del frame


def _passed_through(exception: BaseException, training: tuple[int, Any] | None) -> bool:
    """Whether `exception`'s traceback passes through the frame `training` names."""
    if training is None or training[1] is None:
        return False
    tb = exception.__traceback__
    while tb is not None:
        if id(tb.tb_frame) == training[0] and tb.tb_frame.f_code is training[1]:
            return True
        tb = tb.tb_next
    return False


def _checkpoint_folder(args: Any, state: Any, given: Any = None) -> Path | None:
    """The folder the `Trainer` saved to or resumed from: `given`, else `<output_dir>/checkpoint-<global_step>`."""
    if isinstance(given, (str, os.PathLike)) and Path(given).is_dir():
        return Path(given)
    output = getattr(args, "output_dir", None)
    step = getattr(state, "global_step", None)
    if output is None or step is None:
        return None
    folder = Path(output) / f"checkpoint-{step}"
    return folder if folder.is_dir() else None
