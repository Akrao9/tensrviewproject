"""Tier 0 and Tier 1 capture: per-layer activation statistics, gradient and weight norms,
learning rates and MoE load.

Rules this module follows (see the Capture SDK section of the v3 plan):

- Forward hooks are registered once, before `torch.compile`, and never changed.
- A hook only computes fixed-size statistics and writes them in place into a buffer that
  was allocated once on the model's device. No `.cpu()`, no I/O, no Python-side state.
- After each optimizer step the buffers are copied into one row of an on-device ring.
  Every `every` steps the ring moves to the CPU and a worker thread hands it to the sink.
"""

from __future__ import annotations

import math
import numbers
import queue
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn
from torch.optim.optimizer import (
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)

from .distributed import replica_spread, unwrap_local
from .probe import ProbeState, install_probe_hooks, probe_callable, run_probe
from .sink import JsonlSink, Sink
from .tracker import RunInfo, resolve_run

STATS = ("mean", "std", "abs_max", "zero_frac")
_SEEN = len(STATS)  # last column: 1.0 when the hook ran this step
_LOW_PRECISION = (torch.float16, torch.bfloat16)
# Steps between weight checks when `weights=True`. A check reads every weight once, which
# no per-step method can make cheap (about 1.5 ms for Granite's 2.6 GB of bf16 weights on
# an RTX PRO 6000), so it is sampled, as wandb.watch (every 1,000 batches) and TrainCheck
# (`sampling_interval`) also do.
WEIGHT_EVERY = 50
# Probe passes are rare, like wandb.watch's histograms (every 1,000 batches by default).
PROBE_EVERY = 1000
# Settings per capture mode. Router load is the always-on signal; layer statistics and
# gradient and weight norms are a debugging mode, as in Composer's ActivationMonitor,
# wandb.watch and Lightning's grad_norm.
CAPTURE_MODES: dict[str, dict[str, Any]] = {
    "routers": {"layers": (), "grads": False, "weights": False},
    "full": {"layers": "blocks", "grads": True, "weights": True},
}

LayerFilter = Callable[[str, nn.Module], bool]


@dataclass(frozen=True)
class Router:
    """An MoE router to watch.

    `indices` picks the tensor of chosen expert indices out of the module's output. By
    default it is the first integer tensor in the output, or the top-`top_k` indices of
    the first floating-point tensor whose last dimension equals `n_experts`.
    """

    n_experts: int
    top_k: int = 1
    indices: Callable[[Any], torch.Tensor] | None = None
    # `top_k` is also written into every record as `moe_top_k`, because the detectors
    # need it to judge expert shares: in a top-8 router no expert can exceed 1/8 of the
    # assignments. Set it when `indices` returns top-k indices.


def _first_tensor(output: Any, *, integer: bool) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        candidates = (output,)
    elif isinstance(output, (tuple, list)):
        candidates = output
    elif isinstance(output, dict):
        candidates = output.values()
    else:
        return None
    for item in candidates:
        if not isinstance(item, torch.Tensor) or item.is_complex():
            continue
        if item.is_floating_point() != integer and item.dtype != torch.bool:
            return item
    return None


def _stats_hook(buf: torch.Tensor, stat_idx: torch.Tensor, row: int) -> Callable[..., None]:
    def hook(module: nn.Module, args: Any, output: Any) -> None:
        x = _first_tensor(output, integer=False)
        if x is None or x.numel() == 0:
            return
        x = x.detach()
        xf = x.float()
        if x.numel() > 1:
            std, mean = torch.std_mean(xf)
        else:
            mean = xf.reshape(())
            std = torch.zeros_like(mean)
        # Retain the largest magnitude across forwards until the optimizer step.
        # maximum propagates NaN, so a later clean microbatch cannot hide it.
        amax = torch.maximum(xf.abs().amax(), buf[row, 2])
        zfrac = (x == 0).float().mean()
        buf[row] = torch.where(
            stat_idx == 0,
            mean,
            torch.where(
                stat_idx == 1,
                std,
                torch.where(
                    stat_idx == 2,
                    amax,
                    torch.where(stat_idx == 3, zfrac, 1.0),
                ),
            ),
        )

    return hook


def _router_hook(
    counts: torch.Tensor, expert_ids: torch.Tensor, spec: Router
) -> Callable[..., None]:
    def hook(module: nn.Module, args: Any, output: Any) -> None:
        if spec.indices is not None:
            idx = spec.indices(output)
        else:
            idx = _first_tensor(output, integer=True)
            if idx is None:
                logits = _first_tensor(output, integer=False)
                if logits is not None and logits.ndim >= 1 and logits.shape[-1] == spec.n_experts:
                    idx = (
                        logits.argmax(dim=-1)
                        if spec.top_k <= 1
                        else logits.topk(spec.top_k, dim=-1).indices
                    )
        if idx is None:
            return
        # Fixed output shape: `bincount` would make the shape depend on the data.
        # Accumulate across gradient-accumulation microbatches; `_commit` zeroes `counts`.
        counts.add_((idx.detach().reshape(-1, 1) == expert_ids).sum(0))

    return hook


def _mark_static(buf: torch.Tensor) -> None:
    # Tells Inductor the address never changes, so in-place hook writes do not turn CUDA
    # graphs off under mode="reduce-overhead". Private API, so it is best effort.
    try:
        torch._dynamo.mark_static_address(buf)
    except Exception:  # noqa: BLE001
        pass


def _cpu_copy(tensor: torch.Tensor) -> torch.Tensor:
    # `.cpu()` on a CPU tensor returns a view that aliases the caller's ring buffer.
    return tensor.detach().clone() if tensor.device.type == "cpu" else tensor.detach().cpu()


def _leaf_layers(named: list[tuple[str, nn.Module]]) -> list[tuple[str, nn.Module]]:
    return [
        (name, module)
        for name, module in named
        if not any(module.children()) and any(True for _ in module.parameters(recurse=False))
    ]


def _block_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """The model's repeated blocks: the children of its largest `ModuleList` of composite
    modules that share one class, such as a transformer's decoder layers.

    This is the unit PyTorch's FSDP `transformer_auto_wrap_policy` wraps. One hook per
    block locates a problem to a block at a fraction of the cost of one per leaf module.
    """
    best: list[tuple[str, nn.Module]] = []
    best_params = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) < 2:
            continue
        children = list(module)
        if len({type(c) for c in children}) != 1 or not any(children[0].children()):
            continue
        count = sum(p.numel() for p in module.parameters())
        if count > best_params:
            best_params = count
            best = [(f"{name}.{i}" if name else str(i), child) for i, child in enumerate(children)]
    return best


def _select_layers(
    model: nn.Module, layers: LayerFilter | Sequence[str] | str | None
) -> list[tuple[str, nn.Module]]:
    named = [(name, module) for name, module in model.named_modules() if name]
    if layers is None or layers == "blocks":
        return _block_layers(model) or _leaf_layers(named)
    if layers == "all":
        return _leaf_layers(named)
    if isinstance(layers, str):
        raise ValueError('layers must be "blocks", "all", a list of module names or a filter')
    if callable(layers):
        return [(name, module) for name, module in named if layers(name, module)]
    wanted = list(layers)
    by_name = dict(named)
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise ValueError(f"no such modules: {missing}")
    return [(name, by_name[name]) for name in wanted]


_clip_sessions: list["Session"] = []
_clip_original: Callable[..., Any] | None = None


def _clip_grad_norm_wrapper(parameters: Any, *args: Any, **kwargs: Any) -> Any:
    # `clip_grad_norm_` turns a generator into a list itself; do it first so the
    # parameters can also be matched against each session's model.
    if not isinstance(parameters, torch.Tensor):
        parameters = list(parameters)
    total_norm = _clip_original(parameters, *args, **kwargs)  # type: ignore[misc]
    first = parameters if isinstance(parameters, torch.Tensor) else next(iter(parameters), None)
    for session in list(_clip_sessions):
        if first is not None and id(first) in session._param_ids:
            session.record_grad_norm(total_norm)
    return total_norm


def _watch_clipping(session: "Session") -> None:
    """Record the norm `torch.nn.utils.clip_grad_norm_` returns while `session` is open.

    The norm is computed anyway when gradients are clipped, so recording it is free.
    Lightning calls `torch.nn.utils.clip_grad_norm_` and discards the result, so this
    also covers Lightning's `gradient_clip_val`. Callers that bound the function before
    `init` (`from torch.nn.utils import clip_grad_norm_`) bypass it; they, and Megatron,
    whose optimizer returns the norm from `step()`, can call `record_grad_norm` instead.
    """
    global _clip_original
    import torch.nn.utils as nn_utils
    import torch.nn.utils.clip_grad as clip_module

    if not _clip_sessions:
        _clip_original = nn_utils.clip_grad_norm_
        nn_utils.clip_grad_norm_ = _clip_grad_norm_wrapper  # type: ignore[assignment]
        clip_module.clip_grad_norm_ = _clip_grad_norm_wrapper  # type: ignore[assignment]
    _clip_sessions.append(session)


def _unwatch_clipping(session: "Session") -> None:
    global _clip_original
    import torch.nn.utils as nn_utils
    import torch.nn.utils.clip_grad as clip_module

    if session in _clip_sessions:
        _clip_sessions.remove(session)
        if not _clip_sessions and _clip_original is not None:
            nn_utils.clip_grad_norm_ = _clip_original  # type: ignore[assignment]
            clip_module.clip_grad_norm_ = _clip_original  # type: ignore[assignment]
            _clip_original = None


class Session:
    """One model's capture. Create it with `tensorlens.init`."""

    def __init__(
        self,
        model: nn.Module,
        *,
        run: RunInfo,
        every: int,
        sink: Sink,
        layers: LayerFilter | Sequence[str] | str | None,
        routers: Mapping[str, Router | int] | None,
        grads: bool | int = True,
        weights: bool | int = True,
        auto_step: bool,
        replica_group: Any = None,
        start_step: int = 0,
        clip_norm: bool = True,
        replica_every: int = WEIGHT_EVERY,
        probe: Any = None,
        probe_every: int = PROBE_EVERY,
    ) -> None:
        if hasattr(model, "_orig_mod"):
            raise ValueError("call tensorlens.init(model) before torch.compile(model)")
        if every < 1:
            raise ValueError("every must be at least 1")
        if isinstance(grads, bool):
            grad_every = 1 if grads else 0
        elif isinstance(grads, int) and grads >= 0:
            grad_every = grads
        else:
            raise ValueError("grads must be a bool or a non-negative integer")
        self.run = run
        self.every = every
        self.grad_every = grad_every
        if isinstance(weights, bool):
            weight_every = WEIGHT_EVERY if weights else 0
        elif isinstance(weights, int) and weights >= 0:
            weight_every = weights
        else:
            raise ValueError("weights must be a bool or a non-negative integer")
        self.weight_every = weight_every
        self.weights = weight_every > 0
        if isinstance(replica_every, bool) or not isinstance(replica_every, int) or replica_every < 1:
            raise ValueError("replica_every must be a positive integer")
        self.replica_every = replica_every
        if isinstance(probe_every, bool) or not isinstance(probe_every, int) or probe_every < 1:
            raise ValueError("probe_every must be a positive integer")
        self.probe_every = probe_every
        self._probe_fn = probe_callable(probe) if probe is not None else None
        self._model = model
        self.auto_step = auto_step
        self._sink = sink
        self._replica_group = replica_group

        params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        self._params = [p for _, p in params]
        self._param_ids = {id(p) for p in self._params}
        self.param_names = [name for name, _ in params]
        self.device = self._params[0].device if self._params else torch.device("cpu")

        selected = _select_layers(model, layers)
        self.layer_names = [name for name, _ in selected]
        modules = dict(model.named_modules())
        self._routers = {
            name: spec if isinstance(spec, Router) else Router(int(spec))
            for name, spec in (routers or {}).items()
        }
        unknown = [name for name in self._routers if name not in modules]
        if unknown:
            raise ValueError(f"no such router modules: {unknown}")

        n_layers, n_params = len(self.layer_names), len(self._params)
        f32 = {"dtype": torch.float32, "device": self.device}
        self._act = torch.zeros(n_layers, _SEEN + 1, **f32)
        self._stat_idx = torch.arange(_SEEN + 1, dtype=torch.int64, device=self.device)
        # gradient norm, seen, weight norm, weight unchanged since the previous record
        # (1 = exactly unchanged, 0 = changed, -1 = not known)
        self._grad = torch.zeros(n_params, 4, **f32)
        # Float64 weight norms of the previous record, to tell exactly whether a step changed
        # each weight. A float32 norm misses the few-element changes typical of bf16 weights
        # under small updates. Apple MPS has no float64, so there only float32 norms are kept.
        self._exact_weights = self.device.type != "mps"
        self._prev_weight: torch.Tensor | None = None
        self._spread: torch.Tensor | None = None
        self._counts = {
            name: torch.zeros(spec.n_experts, dtype=torch.int64, device=self.device)
            for name, spec in self._routers.items()
        }
        self._expert_ids = {
            name: torch.arange(spec.n_experts, dtype=torch.int64, device=self.device)
            for name, spec in self._routers.items()
        }
        self._ring_act = torch.zeros(every, n_layers, _SEEN + 1, **f32)
        self._ring_grad = torch.zeros(every, *self._grad.shape, **f32)
        self._ring_grad_seen: list[bool] = []
        self._ring_weight_seen: list[bool] = []
        # Squared total gradient norm reported by gradient clipping this step (summed over
        # calls, for models clipped in several groups), and its ring.
        self._clip = torch.zeros((), **f32)
        self._ring_clip = torch.zeros(every, **f32)
        self._clip_recorded = False
        self._ring_clip_seen: list[bool] = []
        self._ring_spread: torch.Tensor | None = None
        self._ring_spread_seen: list[bool] = []
        if replica_group is not None and n_params > 0:
            self._ensure_spread_buffers()
        self._ring_counts = {
            name: torch.zeros(every, *counts.shape, dtype=torch.int64, device=self.device)
            for name, counts in self._counts.items()
        }
        self._counts_list = list(self._counts.values())
        self._ring_counts_list = list(self._ring_counts.values())
        self._grad_idx_cache: dict[tuple[int, ...], torch.Tensor] = {}
        for buf in (self._act, self._stat_idx, *self._counts_list, *self._expert_ids.values()):
            _mark_static(buf)

        self._handles = [
            module.register_forward_hook(_stats_hook(self._act, self._stat_idx, row))
            for row, (_, module) in enumerate(selected)
        ]
        self._handles += [
            modules[name].register_forward_hook(
                _router_hook(self._counts[name], self._expert_ids[name], spec)
            )
            for name, spec in self._routers.items()
        ]
        self._probe_state = ProbeState()
        if self._probe_fn is not None:
            self._handles += install_probe_hooks(_select_layers(model, "blocks"), self._probe_state)

        self._steps: list[int] = []
        self._ring_lr: list[list[float] | None] = []
        self._pending_lr: list[float] | None = None
        self._ring_scalars: list[dict[str, Any]] = []
        self._pending_scalars: dict[str, Any] = {}
        self._ring_probe: list[dict[str, Any] | None] = []
        self._next_step = start_step
        self._steps_committed = 0
        self._grads_recorded = False
        self._spread_recorded = False
        self._owned: weakref.WeakKeyDictionary[Any, bool] = weakref.WeakKeyDictionary()
        self._closed = False
        self._error: BaseException | None = None
        self._queue: queue.Queue[Any] = queue.Queue()
        self._worker = threading.Thread(target=self._drain, name="tensorlens-writer", daemon=True)
        self._worker.start()

        self._clip_watched = clip_norm
        if clip_norm:
            _watch_clipping(self)

        self._optimizer_handles = []
        if auto_step:
            self._optimizer_handles = [
                register_optimizer_step_pre_hook(self._before_optimizer_step),
                register_optimizer_step_post_hook(self._after_optimizer_step),
            ]

    # ---- training-loop side -------------------------------------------------------

    def _owns(self, optimizer: Any) -> bool:
        # The global hooks fire for every optimizer in the process.
        owned = self._owned.get(optimizer)
        if owned is None:
            owned = any(
                id(p) in self._param_ids for group in optimizer.param_groups for p in group["params"]
            )
            self._owned[optimizer] = owned
        return owned

    def _should_record_grads(self) -> bool:
        # Weight checks need the step's gradient norms, so they also turn gradients on.
        return self._cadence(self.grad_every) or self._should_check_weights()

    def _should_check_weights(self) -> bool:
        return self._cadence(self.weight_every)

    def _cadence(self, every: int) -> bool:
        # The every-th, 2*every-th, ... step, counting from 1 (as Lightning and wandb do).
        return every > 0 and (self._steps_committed + 1) % every == 0

    def _before_optimizer_step(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        if (
            not self._closed
            and self._owns(optimizer)
            and not self._grads_recorded
            and self._should_record_grads()
        ):
            self.record_grads()

    def _after_optimizer_step(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        if not self._closed and self._owns(optimizer):
            self._pending_lr = _learning_rates(optimizer)
            self._commit(None)

    @torch.no_grad()
    def record_grads(self) -> None:
        """Record per-parameter gradient norms now.

        With `auto_step` this runs just before `optimizer.step()`, which is after any
        gradient clipping. Call it yourself before clipping to record unclipped norms.
        """
        n_params = len(self._params)
        if self._grads_recorded:
            self._grad.zero_()
        if n_params == 0:
            self._grads_recorded = True
            return

        groups: dict[tuple[torch.device, torch.dtype], list[tuple[int, torch.Tensor]]] = {}
        for index, param in enumerate(self._params):
            grad = param.grad
            if grad is None:
                continue
            if type(grad) is not torch.Tensor:
                grad = unwrap_local(grad)
            if grad.is_sparse:
                grad = grad.coalesce().values()
            if grad.numel() == 0:
                continue
            groups.setdefault((grad.device, grad.dtype), []).append((index, grad))

        for (_, dtype), items in groups.items():
            grads = [grad for _, grad in items]
            norms = _foreach_norm(grads, torch.float32 if dtype in _LOW_PRECISION else None)
            stacked = torch.stack(norms).to(self._grad)
            if len(items) == n_params:
                self._grad[:, 0].copy_(stacked)
                self._grad[:, 1].fill_(1.0)
            else:
                idx = self._index(tuple(index for index, _ in items))
                self._grad[:, :2].index_copy_(
                    0, idx, torch.stack((stacked, torch.ones_like(stacked)), dim=1)
                )
        self._grads_recorded = True

    def _index(self, key: tuple[int, ...]) -> torch.Tensor:
        idx = self._grad_idx_cache.get(key)
        if idx is None:
            idx = torch.tensor(key, dtype=torch.int64, device=self._grad.device)
            self._grad_idx_cache[key] = idx
        return idx

    @torch.no_grad()
    def _record_weights(self) -> None:
        # The norm of each parameter after the optimizer step, on the steps that record
        # gradient norms, and whether it is exactly what it was at the previous record.
        # One read of the weights, no copy of them.
        n_params = len(self._params)
        dtype = torch.float64 if self._exact_weights else torch.float32
        current = torch.zeros(n_params, dtype=dtype, device=self._grad.device)
        groups: dict[tuple[torch.device, torch.dtype], list[tuple[int, torch.Tensor]]] = {}
        for index, param in enumerate(self._params):
            data = param.detach()
            if type(data) is not torch.Tensor:
                data = unwrap_local(data)
            if data.numel() == 0:
                continue
            groups.setdefault((data.device, data.dtype), []).append((index, data))
        for (_, param_dtype), items in groups.items():
            out = dtype if self._exact_weights else (
                torch.float32 if param_dtype in _LOW_PRECISION else None
            )
            norms = torch.stack(_foreach_norm([data for _, data in items], out)).to(current)
            if len(items) == n_params:
                current.copy_(norms)
            else:
                current.index_copy_(0, self._index(tuple(i for i, _ in items)), norms)
        self._grad[:, 2].copy_(current)
        if not self._exact_weights or self._prev_weight is None:
            self._grad[:, 3].fill_(-1.0)
        else:
            self._grad[:, 3].copy_(current == self._prev_weight)
        if self._exact_weights:
            if self._prev_weight is None:
                self._prev_weight = current
            else:
                self._prev_weight.copy_(current)

    @torch.no_grad()
    def record_grad_norm(self, norm: torch.Tensor | float) -> None:
        """Record the total gradient norm (L2, before clipping) computed this step.

        Called automatically with the value `torch.nn.utils.clip_grad_norm_` returns.
        Call it yourself with a norm computed elsewhere, such as the `grad_norm` that
        Megatron's `optimizer.step()` returns. A tensor stays on its device; no sync.
        """
        if isinstance(norm, torch.Tensor):
            value = norm.detach()
            if type(value) is not torch.Tensor:
                value = unwrap_local(value)
            if value.numel() != 1:
                return
            self._clip.add_(value.reshape(()).to(self._clip).square())
        else:
            self._clip.add_(float(norm) ** 2)
        self._clip_recorded = True

    def log(self, values: Mapping[str, Any] | None = None, **named: Any) -> None:
        """Record scalars such as the loss for the step in progress, without a GPU sync.

        `session.log({"train/loss": loss})` or `session.log(loss=loss)`. Values are
        Python numbers or one-element tensors. A tensor is copied on its device and moved
        to the CPU with the rest of the step data every `every` steps, so logging a CUDA
        loss does not stall training. Call it before `optimizer.step()`: with automatic
        step counting, the step ends there, and a value logged afterwards belongs to the
        next step. A name logged twice in one step keeps the last value. Images, tables
        and other non-scalar values raise `TypeError`; pass the tracker the same values
        it already logs, not a replacement for it.
        """
        if self._closed:
            raise RuntimeError("the session is closed")
        items = dict(values or {}, **named)
        for name, value in items.items():
            if not isinstance(name, str) or not name:
                raise TypeError(f"scalar names must be non-empty strings, got {name!r}")
            if isinstance(value, torch.Tensor):
                tensor = value.detach()
                if type(tensor) is not torch.Tensor:
                    tensor = unwrap_local(tensor)
                if tensor.numel() != 1:
                    raise ValueError(f"{name}: expected one value, got shape {tuple(tensor.shape)}")
                # A copy, so the value neither keeps a larger tensor alive nor changes later.
                dtype = torch.float32 if tensor.device.type == "mps" else torch.float64
                self._pending_scalars[name] = tensor.reshape(()).to(dtype, copy=True)
            elif isinstance(value, numbers.Real):
                self._pending_scalars[name] = (
                    int(value) if isinstance(value, numbers.Integral) else float(value)
                )
            else:
                raise TypeError(f"{name}: expected a number or one-element tensor, got {type(value).__name__}")

    def _ensure_spread_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._spread is None or self._ring_spread is None:
            # Apple MPS rejects float64 allocations; keep float64 replica buffers on CPU there.
            spread_device = torch.device("cpu") if self.device.type == "mps" else self.device
            n_params = len(self._params)
            self._spread = torch.zeros(n_params, 2, dtype=torch.float64, device=spread_device)
            self._ring_spread = torch.zeros(
                self.every, n_params, 2, dtype=torch.float64, device=spread_device
            )
        return self._spread, self._ring_spread

    @torch.no_grad()
    def record_replica_spread(self, group_or_spread: Any = None) -> None:
        """Record cross-GPU parameter relative spread `(norm_spread, sum_spread)` for this step.

        Pass a replication `ProcessGroup` (or rely on `replica_group` passed to `init`), or
        pass a precomputed `[n_params, 2]` tensor from `spread_from_fingerprints`.
        """
        if not self._params:
            return
        target = self._replica_group if group_or_spread is None else group_or_spread
        if target is None:
            raise ValueError("pass a replication process group or precomputed spread tensor")
        spread_buf, _ = self._ensure_spread_buffers()
        if isinstance(target, torch.Tensor):
            spread = target.to(spread_buf)
        else:
            spread = replica_spread(self._params, target).to(spread_buf)
        spread_buf.copy_(spread)
        self._spread_recorded = True

    def step(self, step: int | None = None, lr: float | Sequence[float] | None = None) -> None:
        """Mark the end of one optimizer step. Only for `auto_step=False`.

        Use this when the optimizer hooks would miscount, for example with
        `torch.amp.GradScaler`, which skips `optimizer.step()` on overflow. Pass `lr` (one
        value per parameter group, or a single value) to record the learning rate.
        """
        if self.auto_step:
            raise RuntimeError("step() is called automatically; pass auto_step=False to call it yourself")
        if not self._grads_recorded and self._should_record_grads():
            self.record_grads()
        if lr is not None:
            self._pending_lr = [float(lr)] if isinstance(lr, (int, float)) else [float(v) for v in lr]
        self._commit(step)

    @torch.no_grad()
    def _commit(self, step: int | None) -> None:
        if (
            not self._spread_recorded
            and self._replica_group is not None
            and self._params
            and self._cadence(self.replica_every)
        ):
            # Like Megatron's check_weight_hash_across_dp_replicas_interval: a float64 pass
            # over every weight plus an all_gather is too costly for every step.
            self.record_replica_spread(self._replica_group)
        weights_checked = self._grads_recorded and self._should_check_weights() and bool(self._params)
        if weights_checked:
            self._record_weights()
        row = len(self._steps)
        has_act = self._act.numel() > 0
        if has_act and self._grads_recorded:
            torch._foreach_copy_((self._ring_act[row], self._ring_grad[row]), (self._act, self._grad))
            torch._foreach_zero_((self._act, self._grad))
        elif has_act:
            self._ring_act[row].copy_(self._act)
            self._act.zero_()
        elif self._grads_recorded:
            self._ring_grad[row].copy_(self._grad)
            self._grad.zero_()
        self._ring_grad_seen.append(self._grads_recorded)
        self._ring_weight_seen.append(weights_checked)
        if self._clip_recorded:
            self._ring_clip[row].copy_(self._clip)
            self._clip.zero_()
        self._ring_clip_seen.append(self._clip_recorded)
        self._clip_recorded = False
        if self._spread_recorded and self._spread is not None and self._ring_spread is not None:
            self._ring_spread[row].copy_(self._spread)
            self._spread.zero_()
        self._ring_spread_seen.append(self._spread_recorded)
        if self._counts_list:
            torch._foreach_copy_(
                [ring[row] for ring in self._ring_counts_list], self._counts_list
            )
            torch._foreach_zero_(self._counts_list)
        self._grads_recorded = False
        self._spread_recorded = False
        self._ring_probe.append(self._probe() if self._cadence_probe() else None)
        current = self._next_step if step is None else step
        self._steps.append(current)
        self._ring_lr.append(self._pending_lr)
        self._pending_lr = None
        self._ring_scalars.append(self._pending_scalars)
        self._pending_scalars = {}
        self._next_step = current + 1
        self._steps_committed += 1
        if len(self._steps) == self.every:
            self.flush()

    def _cadence_probe(self) -> bool:
        return self._probe_fn is not None and self._cadence(self.probe_every)

    def _probe(self) -> dict[str, Any]:
        # Runs after this step's statistics are in the ring; the capture hooks fire during
        # the probe's forward pass too, so their buffers are cleared again afterwards.
        devices = sorted({p.device for p in self._params}, key=str) or [self.device]
        try:
            return run_probe(self._model, self._probe_fn, self._probe_state, devices)
        finally:
            buffers = [self._act, *self._counts_list]
            torch._foreach_zero_([b for b in buffers if b.numel() > 0])

    def flush(self) -> None:
        """Move the ring to the CPU and queue it for the sink."""
        if self._error is not None:
            self._steps = []
            self._ring_lr = []
            self._ring_scalars = []
            self._ring_probe = []
            self._ring_grad_seen = []
            self._ring_weight_seen = []
            self._ring_clip_seen = []
            self._ring_spread_seen = []
            self._raise_worker_error()
        n = len(self._steps)
        if n == 0:
            return
        # One transfer for the float statistics, one for the integer expert counts.
        has_act = self._act.numel() > 0
        has_grads = any(self._ring_grad_seen) and self._grad.numel() > 0
        if has_act and has_grads:
            floats: torch.Tensor | None = torch.cat(
                [self._ring_act[:n].reshape(-1), self._ring_grad[:n].reshape(-1)]
            ).cpu()
        elif has_act:
            floats = _cpu_copy(self._ring_act[:n].reshape(-1))
        elif has_grads:
            floats = _cpu_copy(self._ring_grad[:n].reshape(-1))
        else:
            floats = None
        counts = (
            torch.cat([ring[:n].reshape(-1) for ring in self._ring_counts.values()]).cpu()
            if self._ring_counts
            else None
        )
        spreads = (
            _cpu_copy(self._ring_spread[:n])
            if self._ring_spread is not None and any(self._ring_spread_seen)
            else None
        )
        clips = _cpu_copy(self._ring_clip[:n]) if any(self._ring_clip_seen) else None
        clip_seen = self._ring_clip_seen
        grad_seen = self._ring_grad_seen
        weight_seen = self._ring_weight_seen
        spread_seen = self._ring_spread_seen
        scalars = _scalars_to_cpu(self._ring_scalars)
        probes = self._ring_probe
        steps, lrs = self._steps, self._ring_lr
        self._steps = []
        self._ring_lr = []
        self._ring_scalars = []
        self._ring_probe = []
        self._ring_grad_seen = []
        self._ring_weight_seen = []
        self._ring_clip_seen = []
        self._ring_spread_seen = []
        self._queue.put(
            (
                steps, lrs, floats, counts, spreads, clips, clip_seen, grad_seen, weight_seen,
                spread_seen, scalars, probes,
            )
        )

    def close(self) -> None:
        """Flush what is left and stop the writer. Forward hooks stay on the model, because
        removing a hook from a compiled module triggers a recompile."""
        if self._closed:
            return
        self._closed = True
        for handle in self._optimizer_handles:
            handle.remove()
        _unwatch_clipping(self)
        try:
            self.flush()
        finally:
            self._queue.put(None)
            self._worker.join()
            self._sink.close()
        self._raise_worker_error()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- writer thread ------------------------------------------------------------

    def _raise_worker_error(self) -> None:
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError("the TensorLens writer thread failed") from error

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if self._error is not None:
                continue
            try:
                self._sink.write(self._decode(*item))
            except BaseException as error:  # noqa: BLE001
                self._error = error

    def _decode(
        self,
        steps: list[int],
        lrs: list[list[float] | None],
        floats: torch.Tensor | None,
        counts: torch.Tensor | None,
        spreads: torch.Tensor | None,
        clips: torch.Tensor | None,
        clip_seen: list[bool],
        grad_seen: list[bool],
        weight_seen: list[bool],
        spread_seen: list[bool],
        scalars: list[dict[str, Any]] | None = None,
        probes: list[dict[str, Any] | None] | None = None,
    ) -> list[dict[str, Any]]:
        n, n_layers, n_params = len(steps), len(self.layer_names), len(self.param_names)
        split = n * n_layers * (_SEEN + 1) if n_layers > 0 and floats is not None else 0
        act = (
            floats[:split].reshape(n, n_layers, _SEEN + 1).tolist()
            if n_layers > 0 and floats is not None
            else None
        )
        grad = (
            floats[split:].reshape(n, n_params, 4).tolist()
            if any(grad_seen) and n_params > 0 and floats is not None
            else None
        )
        spread_rows = spreads.reshape(n, n_params, 2).tolist() if spreads is not None else None
        moe: dict[str, list[list[int]]] = {}
        offset = 0
        for name, spec in self._routers.items():
            size = n * spec.n_experts
            moe[name] = counts[offset : offset + size].reshape(n, spec.n_experts).tolist()
            offset += size

        records = []
        for i, step in enumerate(steps):
            grad_norms = (
                {name: norm for name, (norm, seen, _, _) in zip(self.param_names, grad[i]) if seen}
                if grad is not None and grad_seen[i]
                else {}
            )
            record: dict[str, Any] = {
                "run_id": self.run.run_id,
                "rank": self.run.rank,
                "step": step,
                "layers": (
                    {
                        name: dict(zip(STATS, row[:_SEEN]))
                        for name, row in zip(self.layer_names, act[i])
                        if row[_SEEN]
                    }
                    if act is not None
                    else {}
                ),
                "grad_norm": math.sqrt(sum(v * v for v in grad_norms.values())) if grad_norms else None,
                "grad_norms": grad_norms,
            }
            if grad is not None and weight_seen[i]:
                record["weight_norms"] = {
                    name: row[2] for name, row in zip(self.param_names, grad[i])
                }
                if self.param_names and grad[i][0][3] >= 0:
                    record["weights_unchanged"] = [
                        name for name, row in zip(self.param_names, grad[i]) if row[3] > 0
                    ]
            if clips is not None and clip_seen[i]:
                record["clip_grad_norm"] = math.sqrt(float(clips[i]))
            if lrs[i] is not None:
                record["lr"] = lrs[i]
            if scalars is not None and scalars[i]:
                record["scalars"] = scalars[i]
            if probes is not None and probes[i] is not None:
                record["probe"] = probes[i]
            if spread_rows is not None and spread_seen[i]:
                record["replica_spread"] = {
                    name: [float(norm_s), float(sum_s)]
                    for name, (norm_s, sum_s) in zip(self.param_names, spread_rows[i])
                }
            if moe:
                step_moe = {name: rows[i] for name, rows in moe.items()}
                record["moe"] = step_moe
                record["moe_top_k"] = {name: spec.top_k for name, spec in self._routers.items()}
                record["moe_stats"] = {
                    name: _router_stats(row_counts) for name, row_counts in step_moe.items()
                }
            records.append(record)
        return records


def _scalars_to_cpu(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace logged tensors with Python floats: one transfer per device for the block."""
    by_device: dict[torch.device, list[tuple[int, str, torch.Tensor]]] = {}
    for i, row in enumerate(rows):
        for name, value in row.items():
            if isinstance(value, torch.Tensor):
                by_device.setdefault(value.device, []).append((i, name, value))
    out = [dict(row) for row in rows]
    for entries in by_device.values():
        values = torch.stack([t.to(entries[0][2].dtype) for _, _, t in entries]).cpu().tolist()
        for (i, name, _), value in zip(entries, values):
            out[i][name] = value
    return out


def _router_stats(counts: list[int]) -> dict[str, Any]:
    n_experts = len(counts)
    total = sum(counts)
    if total > 0 and n_experts > 0:
        dead = sum(1 for c in counts if c == 0)
        shares = [c / total for c in counts if c > 0]
        entropy = -math.fsum(s * math.log(s) for s in shares)
        diversity = math.exp(entropy) / n_experts
        dead_frac = dead / n_experts
    else:
        dead = 0
        entropy = 0.0
        diversity = 0.0
        dead_frac = 0.0
    return {
        "routed_tokens": total,
        "entropy": entropy,
        "diversity": diversity,
        "dead_experts": dead,
        "dead_frac": dead_frac,
    }


def _learning_rates(optimizer: Any) -> list[float] | None:
    # A tensor learning rate on the GPU (capturable optimizers) would need a device sync
    # to read, so it is left out rather than stalling the step.
    rates = []
    for group in optimizer.param_groups:
        lr = group.get("lr")
        if isinstance(lr, torch.Tensor):
            if lr.device.type != "cpu":
                return None
            lr = lr.item()
        if lr is None:
            return None
        rates.append(float(lr))
    return rates


def _foreach_norm(
    tensors: list[torch.Tensor], dtype: torch.dtype | None
) -> tuple[torch.Tensor, ...] | list[torch.Tensor]:
    # `torch._foreach_norm` is private (clip_grad_norm_ uses it). One list must share a
    # device and dtype and be contiguous to stay on the fused path.
    foreach = getattr(torch, "_foreach_norm", None)
    if foreach is not None:
        try:
            return foreach(tensors, 2, dtype=dtype)
        except (RuntimeError, TypeError):
            pass
    return [torch.linalg.vector_norm(t, 2, dtype=dtype) for t in tensors]


def init(
    model: nn.Module,
    *,
    run_id: str | None = None,
    every: int = 50,
    sink: Sink | None = None,
    capture: str | None = None,
    layers: LayerFilter | Sequence[str] | str | None = None,
    routers: Mapping[str, Router | int] | None = None,
    grads: bool | int | None = None,
    weights: bool | int | None = None,
    auto_step: bool = True,
    replica_group: Any = None,
    start_step: int = 0,
    clip_norm: bool = True,
    replica_every: int = WEIGHT_EVERY,
    probe: Any = None,
    probe_every: int = PROBE_EVERY,
) -> Session:
    """Start capturing `model`. Call after `model.to(device)` and before `torch.compile`.

    Args:
        run_id: the tracker's run ID. Defaults to MLFLOW_RUN_ID, the active MLflow run or
            the active W&B run, in that order.
        every: steps kept on the device between transfers to the CPU.
        capture: `"routers"` records MoE router load only: cheap enough to leave on
            (+0.02% to +0.78% on Granite on an RTX PRO 6000). `"full"` adds block
            activation statistics, gradient norms every step and a weight check every 50
            steps, for debugging (+4.15% on the same run). Defaults to `"routers"` when
            `routers` are given and `"full"` otherwise. `layers`, `grads` and `weights`
            override the mode's settings.
        sink: where records go. Defaults to a JSON Lines file under `tensorlens-out/`.
        layers: which modules' outputs to summarize. In full capture, `"blocks"`: each repeated
            block, such as each transformer layer, or every leaf module with parameters if
            the model has no repeated blocks. `"all"`: every leaf module with parameters,
            which costs several times more. Or module names, or a `(name, module) -> bool`
            filter; `()` for none.
        routers: MoE router module names mapped to their expert count or a `Router`.
        grads: True to record gradient norms every step, False or 0 to disable automatic
            gradient norm capture, or a positive integer N to record every Nth captured step.
        weights: check each parameter's norm after the optimizer step every N steps
            (True: every 50; an integer: every N; False or 0: never), so the detectors can
            tell when weights stopped changing. A check reads every weight once and also
            records that step's gradient norms; a parameter is reported after 3 checks in
            a row find it exactly unchanged.
        auto_step: count optimizer steps with PyTorch's global optimizer hooks. Pass False
            and call `session.step()` yourself when using `torch.amp.GradScaler`.
        replica_group: optional replication `ProcessGroup` (or precomputed spread tensor)
            for cross-GPU parameter drift checks.
        replica_every: run the cross-GPU drift check every N steps (default 50). Each
            check reads every weight in float64 and gathers 16 bytes per parameter tensor
            from each rank. `session.record_replica_spread()` records one on demand.
        probe: a fixed batch to run through the model every `probe_every` steps, for
            activation histograms per block and attention statistics: a tensor (called as
            `model(batch)`), a mapping of keyword inputs (`model(**batch)`), or a callable
            `fn(model)`. It runs without the compiler, in eval mode and without gradients,
            right after the step's statistics are recorded, and restores each module's
            training flag and the random number generator state afterwards.
        probe_every: steps between probe passes (default 1000).
        start_step: the first step number. Pass the step you resume from, so a resumed
            run does not repeat step numbers already written.
        clip_norm: record the total gradient norm that `torch.nn.utils.clip_grad_norm_`
            returns for this model's parameters, as `clip_grad_norm`, in any capture mode.
            Free when you clip anyway, and enough for gradient-spike detection.
    """
    mode = capture if capture is not None else ("routers" if routers else "full")
    if mode not in CAPTURE_MODES:
        raise ValueError(f"capture must be one of {sorted(CAPTURE_MODES)}, got {capture!r}")
    if mode == "routers" and not routers:
        raise ValueError('capture="routers" needs routers=; use capture="full" for a dense model')
    defaults = CAPTURE_MODES[mode]
    layers = defaults["layers"] if layers is None else layers
    grads = defaults["grads"] if grads is None else grads
    weights = defaults["weights"] if weights is None else weights
    run = resolve_run(run_id)
    if sink is None:
        name = run.run_id or "run"
        sink = JsonlSink(Path("tensorlens-out") / f"{name}.rank{run.rank}.jsonl")
    return Session(
        model,
        run=run,
        every=every,
        sink=sink,
        layers=layers,
        routers=routers,
        grads=grads,
        weights=weights,
        auto_step=auto_step,
        replica_group=replica_group,
        start_step=start_step,
        clip_norm=clip_norm,
        replica_every=replica_every,
        probe=probe,
        probe_every=probe_every,
    )
