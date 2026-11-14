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
from .sink import JsonlSink, Sink
from .tracker import RunInfo, resolve_run

STATS = ("mean", "std", "abs_max", "zero_frac")
_SEEN = len(STATS)  # last column: 1.0 when the hook ran this step
_LOW_PRECISION = (torch.float16, torch.bfloat16)

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
        amax = xf.abs().amax()
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


def _select_layers(
    model: nn.Module, layers: LayerFilter | Sequence[str] | None
) -> list[tuple[str, nn.Module]]:
    named = [(name, module) for name, module in model.named_modules() if name]
    if layers is None:
        return [
            (name, module)
            for name, module in named
            if not any(module.children()) and any(True for _ in module.parameters(recurse=False))
        ]
    if callable(layers):
        return [(name, module) for name, module in named if layers(name, module)]
    wanted = list(layers)
    by_name = dict(named)
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise ValueError(f"no such modules: {missing}")
    return [(name, by_name[name]) for name in wanted]


class Session:
    """One model's capture. Create it with `tensorlens.init`."""

    def __init__(
        self,
        model: nn.Module,
        *,
        run: RunInfo,
        every: int,
        sink: Sink,
        layers: LayerFilter | Sequence[str] | None,
        routers: Mapping[str, Router | int] | None,
        grads: bool | int = True,
        weights: bool = True,
        auto_step: bool,
        replica_group: Any = None,
        start_step: int = 0,
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
        self.weights = weights
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

        self._steps: list[int] = []
        self._ring_lr: list[list[float] | None] = []
        self._pending_lr: list[float] | None = None
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
        return self.grad_every > 0 and (self._steps_committed + 1) % self.grad_every == 0

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
        if not self._spread_recorded and self._replica_group is not None and self._params:
            self.record_replica_spread(self._replica_group)
        if self._grads_recorded and self.weights and self._params:
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
        current = self._next_step if step is None else step
        self._steps.append(current)
        self._ring_lr.append(self._pending_lr)
        self._pending_lr = None
        self._next_step = current + 1
        self._steps_committed += 1
        if len(self._steps) == self.every:
            self.flush()

    def flush(self) -> None:
        """Move the ring to the CPU and queue it for the sink."""
        if self._error is not None:
            self._steps = []
            self._ring_lr = []
            self._ring_grad_seen = []
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
        grad_seen = self._ring_grad_seen
        spread_seen = self._ring_spread_seen
        steps, lrs = self._steps, self._ring_lr
        self._steps = []
        self._ring_lr = []
        self._ring_grad_seen = []
        self._ring_spread_seen = []
        self._queue.put((steps, lrs, floats, counts, spreads, grad_seen, spread_seen))

    def close(self) -> None:
        """Flush what is left and stop the writer. Forward hooks stay on the model, because
        removing a hook from a compiled module triggers a recompile."""
        if self._closed:
            return
        self._closed = True
        for handle in self._optimizer_handles:
            handle.remove()
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
        grad_seen: list[bool],
        spread_seen: list[bool],
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
            if self.weights and grad is not None and grad_seen[i]:
                record["weight_norms"] = {
                    name: row[2] for name, row in zip(self.param_names, grad[i])
                }
                if self.param_names and grad[i][0][3] >= 0:
                    record["weights_unchanged"] = [
                        name for name, row in zip(self.param_names, grad[i]) if row[3] > 0
                    ]
            if lrs[i] is not None:
                record["lr"] = lrs[i]
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
    layers: LayerFilter | Sequence[str] | None = None,
    routers: Mapping[str, Router | int] | None = None,
    grads: bool | int = True,
    weights: bool = True,
    auto_step: bool = True,
    replica_group: Any = None,
    start_step: int = 0,
) -> Session:
    """Start capturing `model`. Call after `model.to(device)` and before `torch.compile`.

    Args:
        run_id: the tracker's run ID. Defaults to MLFLOW_RUN_ID, the active MLflow run or
            the active W&B run, in that order.
        every: steps kept on the device between transfers to the CPU.
        sink: where records go. Defaults to a JSON Lines file under `tensorlens-out/`.
        layers: module names, or a `(name, module) -> bool` filter. Defaults to every leaf
            module that has parameters.
        routers: MoE router module names mapped to their expert count or a `Router`.
        grads: True to record gradient norms every step, False or 0 to disable automatic
            gradient norm capture, or a positive integer N to record every Nth captured step.
        weights: record each parameter's norm after the optimizer step, on the same steps
            as gradient norms, so the detectors can tell when a step left the weights
            unchanged. Costs one more read of the weights on those steps.
        auto_step: count optimizer steps with PyTorch's global optimizer hooks. Pass False
            and call `session.step()` yourself when using `torch.amp.GradScaler`.
        replica_group: optional replication `ProcessGroup` (or precomputed spread tensor)
            for cross-GPU parameter drift checks.
        start_step: the first step number. Pass the step you resume from, so a resumed
            run does not repeat step numbers already written.
    """
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
    )
