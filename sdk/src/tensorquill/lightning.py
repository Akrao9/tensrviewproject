"""PyTorch Lightning callback: `Trainer(callbacks=[TensorQuillCallback(...)])`.

Lightning is imported only by this module, so the core SDK keeps no dependency on it.
"""

from __future__ import annotations

from typing import Any

import lightning.pytorch as L
import torch

from .capture import Session, init
from .tracker import _brief, _say


class TensorQuillCallback(L.Callback):
    """Start TensorQuill when fitting starts and stop it when fitting ends.

    - **Run ID.** `MLFlowLogger` creates its run through `MlflowClient` on rank 0 only, so
      `mlflow.active_run()` is empty and the other ranks know nothing of it. The callback
      reads the run ID from the trainer's logger (`MLFlowLogger.run_id`, `WandbLogger`'s run)
      on rank 0 and passes it to `init`, which shares rank 0's run ID with every rank through
      the process group's store, so every rank writes under the tracker's run. A `run_id=`
      given to the callback wins over the logger's; with several ranks, rank 0's wins.
    - **Steps** start at `trainer.global_step`, so a resumed fit continues the numbering.
      They are counted from optimizer steps, as Lightning's `global_step` is.
    - **Mixed precision.** With `precision="16-mixed"`, the plugin's `GradScaler` is passed
      as `scaler=`, so steps the scaler skips are counted and marked. Lightning gives
      `on_before_backward` the loss after the scaler multiplied it; the loss recorded is
      divided by that scale again, so it is the loss, as without a scaler.
    - **Loss.** The loss Lightning back-propagates is logged as `loss_name` for each
      optimizer step, summed over gradient-accumulation microbatches (Lightning has
      already divided each by `accumulate_grad_batches`, so the sum is their mean). It
      stays a tensor on the device, with no sync. `loss_name=None` turns this off. A fit
      that raises after a backward and before its optimizer step leaves that loss out; the
      next fit through the same callback starts with none. With
      several optimizers (manual optimization; Lightning allows no other), TensorQuill
      counts one step per batch, ended by the first optimizer, and the losses
      back-propagated for the others in the same batch are added to that step's.
    - **Data wait.** The host time spent in `next()` on the training `DataLoader`, which
      Lightning calls before it moves the batch to the device, is recorded as `data_wait`
      (`tensorquill.datawait`). `data_wait=False` turns it off.

    Other keyword arguments go to `tensorquill.init`, for example `routers=`, `capture=`
    or `probe=`.
    """

    def __init__(self, *, loss_name: str | None = "train/loss", **init_kwargs: Any) -> None:
        self.loss_name = loss_name
        self.init_kwargs = init_kwargs
        self.session: Session | None = None
        self._loss: Any = None
        self._loaded: dict[str, Any] | None = None
        self._loaded_step = 0

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        kwargs = dict(self.init_kwargs)
        if kwargs.get("run_id") is None and trainer.is_global_zero:
            # The logger's run is known on rank 0 only. `init` gives every rank rank 0's run ID
            # through the process group's store, so the other ranks pass what they were given:
            # every rank takes the same path, and there is no collective here.
            kwargs["run_id"] = _logger_run_id(trainer)
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        if scaler is not None:
            kwargs.setdefault("scaler", scaler)
        # Lightning restores the loops' progress after this hook: a resumed fit still reads 0
        # here, and the step comes from the checkpoint `on_load_checkpoint` was given.
        kwargs.setdefault("start_step", max(trainer.global_step, self._loaded_step))
        kwargs.setdefault("data_wait", True)
        self._loss = None  # a loss left by a fit that stopped before its optimizer step is not this fit's
        self.session = init(pl_module, **kwargs)
        if self._loaded is not None:
            loaded, self._loaded, self._loaded_step = self._loaded, None, 0
            self.session._record_manifest(loaded, "load")

    def on_before_backward(self, trainer: L.Trainer, pl_module: L.LightningModule, loss: Any) -> None:
        if self.session is not None and self.loss_name is not None:
            loss = _unscaled(trainer, loss.detach())
            self._loss = loss if self._loss is None else self._loss + loss

    def on_before_optimizer_step(self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: Any) -> None:
        # Runs after the closure's backward and before the optimizer's update, so the loss
        # belongs to the step TensorQuill is about to record. With several optimizers (manual
        # optimization), one that steps after the first has ended the TensorQuill step: the loss
        # back-propagated for it since is added to that step's, which is still the newest.
        if self.session is not None and self._loss is not None:
            if self.session._follows(optimizer):
                self.session._add_to_newest(self.loss_name, self._loss)
            else:
                self.session.log({self.loss_name: self._loss})
            self._loss = None

    # Validation is a stretch with no training steps: say so, so the server does not call it a hang.
    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.session is not None:
            self._outer_phase = self.session._phase
            self.session.set_phase("eval")

    def on_validation_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, *args: Any, **kwargs: Any) -> None:
        if self.session is not None:
            self.session.tick()  # evaluation is working, not stuck

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.session is not None:
            self.session.set_phase(getattr(self, "_outer_phase", "train"))

    def on_save_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict[str, Any]) -> None:
        """What the checkpoint holds (`session.checkpoint`): Lightning's `state_dict`,
        `optimizer_states`, `lr_schedulers`, the precision plugin's scaler state and `loops`
        (where the data stands)."""
        if self.session is not None:
            self.session.checkpoint(checkpoint, event="save")

    def on_load_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict[str, Any]) -> None:
        # Lightning restores before `on_fit_start`, where the session begins: kept until then.
        # Compared here with the module `init` is given there, and Lightning loads the checkpoint's
        # `state_dict` into it, so that is the loader's entry (CHECKPOINT_TENSOR_CONTRACT.md).
        from .data import load_manifest

        self._loaded = load_manifest(checkpoint, pl_module, model_state=checkpoint.get("state_dict"))
        step = checkpoint.get("global_step")
        self._loaded_step = step if isinstance(step, int) and not isinstance(step, bool) else 0

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._close()

    def on_exception(self, trainer: L.Trainer, pl_module: L.LightningModule, exception: BaseException) -> None:
        # Write what was captured and stop the writer thread; the end marker says the fit raised.
        self._close(exception)

    def _close(self, exception: BaseException | None = None) -> None:
        # A fit that raised between a backward and its optimizer step leaves that loss here: it is
        # the stopped fit's, never the next one's.
        self._loss = None
        if self.session is not None:
            session, self.session = self.session, None
            session.close(_exception=exception)


def _unscaled(trainer: L.Trainer, loss: Any) -> Any:
    """The loss before the precision plugin's `GradScaler` multiplied it.

    Under `precision="16-mixed"` Lightning's `MixedPrecision.pre_backward` scales the loss and then
    calls `on_before_backward` with the scaled one. The scaler's scale changes only in its `update()`,
    after the optimizer step, so the scale this loss was multiplied by is the scaler's now. It is
    read as `scaler.scale(1.0)`, a tensor on the loss's device, so nothing is synced; a disabled
    scaler returns the 1.0 itself. The default scale is a power of two, so the division is exact.
    With no scaler (full precision, `"bf16-mixed"`, a plugin that scales inside its own backward)
    the loss is as given."""
    scaler = getattr(trainer.precision_plugin, "scaler", None)
    if scaler is None or not isinstance(loss, torch.Tensor):
        return loss
    try:
        scale = scaler.scale(torch.ones((), dtype=torch.float32, device=loss.device))
        return (loss / scale).reshape(loss.shape)
    except Exception:  # noqa: BLE001  nothing here may stop the fit: the loss as given
        return loss


def _logger_run_id(trainer: L.Trainer) -> str | None:
    """The tracker run of the trainer's first logger that has one: MLflow or W&B.

    Never raises. It runs on rank 0 before `init` shares rank 0's run ID, so a raise here would
    leave the other ranks without it: a logger whose `run_id` or `version` raises (`MLFlowLogger`
    creates its run on first use, `WandbLogger.version` may start W&B) is passed over with one
    warning, and the next logger is read. With none left, `init` resolves the ID by its order
    (a tracker's run, the launcher's job ID or a generated one) and shares it."""
    try:
        loggers = list(trainer.loggers)
    except Exception as error:  # noqa: BLE001  nothing here may stop the fit
        _say(
            f"TensorQuillCallback could not list the trainer's loggers ({_brief(error)}), {_PASSED_OVER}",
            stacklevel=3,
        )
        return None
    for logger in loggers:
        try:
            run_id = getattr(logger, "run_id", None)  # MLFlowLogger
            if run_id:
                return str(run_id)
            if type(logger).__name__ == "WandbLogger" and logger.version:
                return str(logger.version)  # the W&B run ID
        except Exception as error:  # noqa: BLE001  nothing here may stop the fit
            _say(
                f"TensorQuillCallback could not read the run ID of the trainer's logger {type(logger).__name__} "
                f"({_brief(error)}), {_PASSED_OVER}",
                stacklevel=3,
            )
    return None


_PASSED_OVER = (
    "so it is passed over: the run ID is the next logger's, or else what tensorquill.init resolves (a tracker's "
    "run, the launcher's job ID or a generated one), shared with every rank. Pass run_id= to name the run."
)
