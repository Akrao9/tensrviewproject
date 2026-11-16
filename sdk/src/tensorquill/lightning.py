"""PyTorch Lightning callback: `Trainer(callbacks=[TensorQuillCallback(...)])`.

Lightning is imported only by this module, so the core SDK keeps no dependency on it.
"""

from __future__ import annotations

from typing import Any

import lightning.pytorch as L

from .capture import Session, init


class TensorQuillCallback(L.Callback):
    """Start TensorQuill when fitting starts and stop it when fitting ends.

    - **Run ID.** `MLFlowLogger` creates its run through `MlflowClient` on rank 0 only, so
      `mlflow.active_run()` is empty and the other ranks know nothing of it. The callback
      reads the run ID from the trainer's logger (`MLFlowLogger.run_id`, `WandbLogger`'s run)
      on rank 0 and broadcasts it, so every rank writes under the tracker's run.
    - **Steps** start at `trainer.global_step`, so a resumed fit continues the numbering.
      They are counted from optimizer steps, as Lightning's `global_step` is.
    - **Mixed precision.** With `precision="16-mixed"`, the plugin's `GradScaler` is passed
      as `scaler=`, so steps the scaler skips are counted and marked.
    - **Loss.** The loss Lightning back-propagates is logged as `loss_name` for each
      optimizer step, summed over gradient-accumulation microbatches (Lightning has
      already divided each by `accumulate_grad_batches`, so the sum is their mean). It
      stays a tensor on the device, with no sync. `loss_name=None` turns this off.

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
        run_id = kwargs.pop("run_id", None)
        if run_id is None:
            run_id = _logger_run_id(trainer) if trainer.is_global_zero else None
            run_id = trainer.strategy.broadcast(run_id, src=0)
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        if scaler is not None:
            kwargs.setdefault("scaler", scaler)
        # Lightning restores the loops' progress after this hook: a resumed fit still reads 0
        # here, and the step comes from the checkpoint `on_load_checkpoint` was given.
        kwargs.setdefault("start_step", max(trainer.global_step, self._loaded_step))
        self.session = init(pl_module, run_id=run_id, **kwargs)
        if self._loaded is not None:
            loaded, self._loaded, self._loaded_step = self._loaded, None, 0
            self.session._record_manifest(loaded, "load")

    def on_before_backward(self, trainer: L.Trainer, pl_module: L.LightningModule, loss: Any) -> None:
        if self.session is not None and self.loss_name is not None:
            loss = loss.detach()
            self._loss = loss if self._loss is None else self._loss + loss

    def on_before_optimizer_step(self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: Any) -> None:
        # Runs after the closure's backward and before the optimizer's update, so the loss
        # belongs to the step TensorQuill is about to record.
        if self.session is not None and self._loss is not None:
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
        from .data import checkpoint_manifest

        self._loaded = checkpoint_manifest(checkpoint)
        step = checkpoint.get("global_step")
        self._loaded_step = step if isinstance(step, int) and not isinstance(step, bool) else 0

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._close()

    def on_exception(self, trainer: L.Trainer, pl_module: L.LightningModule, exception: BaseException) -> None:
        self._close()  # write what was captured, and stop the writer thread

    def _close(self) -> None:
        if self.session is not None:
            session, self.session = self.session, None
            session.close()


def _logger_run_id(trainer: L.Trainer) -> str | None:
    """The tracker run of the trainer's first logger that has one: MLflow or W&B."""
    for logger in trainer.loggers:
        run_id = getattr(logger, "run_id", None)  # MLFlowLogger
        if run_id:
            return str(run_id)
        if type(logger).__name__ == "WandbLogger" and logger.version:
            return str(logger.version)  # the W&B run ID
    return None
