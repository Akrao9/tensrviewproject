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

import contextlib
import math
import numbers
import queue
import sys
import threading
import time
import warnings
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
from torch import nn
from torch.optim.optimizer import (
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)

from .data import (
    DataStep,
    WatchedLoader,
    checkpoint_manifest,
    data_fields,
    files_manifest,
    optimizer_entries,
    optimizer_step_value,
)
from . import datawait
from .distributed import FINGERPRINT_WIDTH, replica_spread, unwrap_local
from .probe import ProbeState, install_probe_hooks, probe_callable, run_probe
from .sink import JsonlSink, Sink, clear_end_marker, write_end_marker
from .system import SystemPart, SystemReadings
from . import manifest as _manifest
from .tracker import RunInfo, _world_size, resolve_run, tracker_labels

STATS = ("mean", "std", "abs_max", "zero_frac")
_SEEN = len(STATS)  # last column: 1.0 when the hook ran this step
_LOW_PRECISION = (torch.float16, torch.bfloat16)
# Steps between weight checks when `weights=True`. A check reads every weight once, which
# no per-step method can make cheap (about 1.5 ms for Granite's 2.6 GB of bf16 weights on
# an RTX PRO 6000), so it is sampled, as wandb.watch (every 1,000 batches) and TrainCheck
# (`sampling_interval`) also do.
WEIGHT_EVERY = 50
# Version of the step-record format; readers treat records without it as version 0.
SCHEMA_VERSION = 1
# Probe passes are rare, like wandb.watch's histograms (every 1,000 batches by default).
PROBE_EVERY = 1000
# Blocks of `every` steps waiting for the writer thread before training waits for it.
WRITER_QUEUE = 8
FOREIGN_STEPS_WARNING = 10  # optimizer steps on other parameters before warning that none was counted
# Why the update size was not read (`updates=True`), by what it is said about: an optimizer's
# class name, or one of these two.
UPDATES_HOOKS_OFF = "auto_step=False"
UPDATES_NO_OPTIMIZER = "no optimizer"
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

    A function given as `indices` is checked as the default is, and more: it must return an
    integer tensor on the counts' device, of a size that is a whole multiple of the tokens
    the router was called on, with values in `0..n_experts-1`. One that raises, or returns
    anything else, has its router refused for the session (`_router_hook`), never an error
    in training.
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


def _stats_hook(
    buf: torch.Tensor, stat_idx: torch.Tensor, row: int, phase: _Evaluating | None = None
) -> Callable[..., None]:
    phase = phase or _Evaluating()

    def hook(module: nn.Module, args: Any, output: Any) -> None:
        if _recomputing() or _evaluating(module, phase):
            return
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
        peak = xf.abs().amax()
        kept = _row_max_compiled(buf, row) if torch.compiler.is_compiling() else buf[row, 2]
        amax = torch.maximum(peak, kept)
        zfrac = (x == 0).float().mean()
        stats = torch.where(
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
        if torch.compiler.is_compiling():
            _set_row_compiled(buf, row, stats)
        else:
            buf[row] = stats

    return hook


def _tokens(args: Any) -> int | None:
    """How many tokens a router was called on: every dimension of its first tensor argument
    but the last. None when that cannot be read from the shapes alone."""
    x = args[0] if isinstance(args, (tuple, list)) and args else args
    if not isinstance(x, torch.Tensor) or x.ndim < 2:
        return None
    width = x.shape[-1]
    if type(width) is not int or width <= 0:
        return None
    total = x.numel()
    return total // width if type(total) is int else None


def _refusal(name: str, spec: Router, why: str, fix: str | None = None, device: Any = None) -> str:
    """Why a router is not counted and the two ways to have it counted, as one text: the
    session's warning and the value of `moe_refused` (so the evidence and the server print
    the same words). The fixes depend on how the router was named, which only the capture
    knows, and for two reasons on the reason itself (`fix`, set by `_router_hook`): output on
    another device than the counts (`device`), where the inner module's output would be too,
    and a `top_k` above the number of experts. No "; ": the evidence joins several routers'
    reasons with it."""
    inner = "name the module inside it whose output is the router's logits"
    if fix == "device":
        fix = (
            f"pass routers={{{name!r}: tensorquill.Router({spec.n_experts}, {spec.top_k}, indices=...)}} with a "
            f"function that returns the chosen expert indices from its output on {device}, where the counts are"
        )
    elif fix == "top_k":
        fix = (
            f"pass routers={{{name!r}: tensorquill.Router(n_experts, top_k)}} with its number of experts and a "
            f"top_k of at most that number"
        )
    elif spec.indices is None:
        fix = (
            f"pass routers={{{name!r}: tensorquill.Router({spec.n_experts}, {spec.top_k}, indices=...)}} with a "
            f"function that returns the chosen expert indices from its output, or {inner}"
        )
    else:
        fix = (
            f"change its indices= function to return the chosen expert indices, integers from 0 to "
            f"{spec.n_experts - 1}, one or {spec.top_k} for each token, or drop indices= and {inner}"
        )
    return f"{why}. To count it, {fix}"


def _short(error: BaseException) -> str:
    # One bounded line: it is written into a record and printed in the evidence, which joins
    # several routers' reasons with "; ".
    text = " ".join(f"{type(error).__name__}: {error}".split()).replace("; ", ", ")
    return text if len(text) <= 200 else text[:199] + "…"


def _not_indices(idx: Any, device: torch.device) -> str | None:
    """What makes the `indices=` function's result unusable as expert indices, from its type
    alone; None when it is an integer tensor on the counts' device."""
    if not isinstance(idx, torch.Tensor):
        return f"its indices= function returned {type(idx).__name__}, not a tensor"
    if idx.is_floating_point() or idx.is_complex() or idx.dtype == torch.bool:
        return f"its indices= function returned a {str(idx.dtype).removeprefix('torch.')} tensor, not integer expert indices"
    if idx.device != device:
        return f"its indices= function returned a tensor on {idx.device}, not on {device} where the counts are"
    wrong = _unusable(idx, device, "the tensor its indices= function returned")
    return None if wrong is None else wrong[0]


try:  # DTensor (tensor parallelism); absent from builds of PyTorch without torch.distributed
    from torch.distributed.tensor import DTensor as _DTensor
except Exception:  # noqa: BLE001
    _DTensor = None
# Integer types `==` cannot compare with the int64 expert numbers: it raises, eager and compiled.
_UNCOMPARABLE = tuple(getattr(torch, name) for name in ("uint16", "uint32", "uint64") if hasattr(torch, name))


def _unusable(t: torch.Tensor, device: torch.device, what: str) -> tuple[str, str | None] | None:
    """What makes a tensor of indices or logits raise in the counting (`_router_hook`), from
    its type, layout, device and dtype alone: no data, so no sync, and the same under
    `torch.compile`, where an op that raises fails the trace before any `except` could see
    it. The reason, and the fix `_refusal` names for it; None when the counting can read the
    tensor. Only what raises today is refused: nothing counted before is."""
    if _DTensor is not None and isinstance(t, _DTensor):
        return f"{what} is a DTensor, not a plain tensor", None
    if t.is_nested:
        return f"{what} is a nested tensor", None
    if t.layout != torch.strided:
        return f"{what} is a {str(t.layout).removeprefix('torch.')} tensor, not a dense one", None
    if t.device != device:
        return f"{what} is on {t.device}, not on {device} where the counts are", "device"
    if t.dtype in _UNCOMPARABLE or (t.is_floating_point() and t.dtype.itemsize == 1):
        # uint16/32/64 cannot be compared with int64; PyTorch has no argmax or topk for float8.
        return f"{what} is {str(t.dtype).removeprefix('torch.')}, which TensorQuill cannot count", None
    return None


def _count_width(spec: Router) -> int:
    """Slots in a router's counts buffer: one per expert, and for an `indices=` router one more
    for the indices that match no expert (`_router_hook`, judged in `flush`)."""
    return spec.n_experts + (spec.indices is not None)


def _router_hook(
    counts: torch.Tensor,
    expert_ids: torch.Tensor,
    spec: Router,
    problems: list[tuple[str, str | None]] | None = None,
    phase: _Evaluating | None = None,
) -> Callable[..., None]:
    """The forward hook that adds a router's chosen experts to `counts`. Nothing in it raises
    into training: what cannot be counted refuses the router for the session (`problems`,
    said once by the session and written under `moe_refused`), first from types, layouts,
    devices and shapes alone, so the same under `torch.compile`, and in eager mode from any
    exception left, which is caught. A forward that is not training (`_evaluating`) is left
    out before anything is read, so an evaluation batch is neither counted nor refused."""
    phase = phase or _Evaluating()

    def refuse(why: Callable[[], str], fix: str | None = None) -> None:
        # Built only the first time: a refused router's hook still runs on every forward, and
        # the text may format a user's exception or a shape. `fix` picks the fix `_refusal` names.
        if problems is not None and not problems:
            problems.append((why(), fix))

    def count(args: Any, output: Any) -> None:
        if spec.indices is not None:
            # The user's function: one that raises, or returns what cannot be expert indices,
            # refuses this router and never raises into training. Checked from types and
            # shapes only; its values are judged in `flush`, from the last slot of `counts`, after the
            # copy to the CPU that is made anyway.
            try:
                idx = spec.indices(output)
            except Exception as error:  # noqa: BLE001
                refuse(lambda: f"its indices= function raised {_short(error)}")
                return
            wrong = _not_indices(idx, counts.device)
            if wrong is not None:
                refuse(lambda: wrong)
                return
            tokens, chosen = _tokens(args), idx.numel()
            if tokens and type(chosen) is int and (chosen < tokens or chosen % tokens):
                refuse(lambda: (
                    f"its indices= function returned shape {tuple(idx.shape)}, which is not expert indices for "
                    f"the {tokens} tokens it was called on"
                ))
                return
        else:
            idx = _first_tensor(output, integer=True)
            if idx is not None:
                # An integer tensor in the output is taken as the chosen expert indices, one or
                # `top_k` for each token. One that cannot be that (a router that returns
                # per-expert token counts first, as nanoMoE's does) would be counted as indices
                # and give wrong loads with no error. The shapes alone show it, so nothing here
                # depends on the data; the session warns once and this router is not counted.
                tokens, chosen = _tokens(args), idx.numel()
                if tokens and type(chosen) is int and (chosen < tokens or chosen % tokens):
                    refuse(lambda: (
                        f"its output's first integer tensor has shape {tuple(idx.shape)}, which is not expert "
                        f"indices for the {tokens} tokens it was called on"
                    ))
                    return
                # One the counting below would raise on (another device than the counts, say).
                wrong = _unusable(idx, counts.device, "its output's first integer tensor")
                if wrong is not None:
                    refuse(lambda: wrong[0], wrong[1])
                    return
            else:
                logits = _first_tensor(output, integer=False)
                if logits is None:
                    # No tensor to read: nothing is counted and nothing refused, as before.
                    return
                # Logits of another width are not this router's: counting nothing would write
                # zeros that read as a router that routed no token.
                if logits.ndim < 1 or logits.shape[-1] != spec.n_experts:
                    refuse(lambda: (
                        f"its output has no integer tensor and its first floating-point tensor has shape "
                        f"{tuple(logits.shape)}, which is not logits for its {spec.n_experts} experts"
                    ))
                    return
                if spec.n_experts < max(spec.top_k, 1):
                    # `topk` would raise (and `argmax` over no experts).
                    refuse(lambda: f"its top_k is {spec.top_k} but it has {spec.n_experts} experts", "top_k")
                    return
                wrong = _unusable(logits, counts.device, "its output's first floating-point tensor")
                if wrong is not None:
                    refuse(lambda: wrong[0], wrong[1])
                    return
                # Detached, so the hook saves nothing for backward: non-reentrant
                # checkpointing checks that recomputation saves the same tensors.
                logits = logits.detach()
                idx = (
                    logits.argmax(dim=-1)
                    if spec.top_k <= 1
                    else logits.topk(spec.top_k, dim=-1).indices
                )
        # Fixed output shape: `bincount` would make the shape depend on the data.
        # Accumulate across gradient-accumulation microbatches; `_commit` zeroes `counts`.
        hits = (idx.detach().reshape(-1, 1) == expert_ids).sum(0)
        if spec.indices is not None:
            # An `indices=` router's buffer has a last slot for the indices outside
            # 0..n_experts-1, which match no expert (`_count_width`): device work only, read in
            # `flush` with the counts (no sync here). Default routers keep the work they had.
            hits = torch.cat([hits, (idx.numel() - hits.sum()).reshape(1)])
        if torch.compiler.is_compiling():
            _add_compiled(counts, hits)
        else:
            counts.add_(hits)

    def hook(module: nn.Module, args: Any, output: Any) -> None:
        if _recomputing() or _evaluating(module, phase):
            return
        if torch.compiler.is_compiling():
            # No `try` in a traced hook: the checks above are what keep it from raising there.
            count(args, output)
            return
        try:
            count(args, output)
        except Exception as error:  # noqa: BLE001
            # What the checks above do not foresee (an unusual tensor type, say) refuses the
            # router as they do, and training goes on.
            refuse(lambda: f"counting its output raised {_short(error)}")

    return hook


# The two writes below are what a hook does to its buffer when it is compiled. Dynamo puts each
# call into the graph as one opaque node instead of tracing its body.
#
# Under a backend that runs Dynamo's graph as it is (`backend="eager"`, or a custom backend that
# returns `gm.forward`), the body runs at execution time on the real buffer. A block wrapped in
# `torch.utils.checkpoint.checkpoint` (reentrant or not: Dynamo runs both as non-reentrant) then
# runs as one region whose tensor inputs include the buffer, lifted from the hook's closure, and
# non-reentrant checkpointing saves every input with its version. Bumping the buffer's version in
# the forward made `backward()` raise "modified by an inplace operation". The region's forward is
# also replayed in eager mode during backward, from the graph, where `_recomputing()` was decided
# at trace time; a replayed write would count the same tokens twice.
#
# So on a real buffer the write keeps the buffer's version (nothing differentiable reads it) and
# is skipped while the autograd engine runs, as `_recomputing()` does in eager mode. `_commit`
# keeps the versions too when it zeroes the buffers, for an optimizer step taken between a
# forward and its backward. Under AOTAutograd (aot_eager, Inductor) the body is traced on fake
# tensors instead and adds exactly the operations the hook added before, so those graphs are
# unchanged. There the hook's read and write of its buffer are traced outside the checkpointed
# block's recompute set (`_NotRecomputed`), which PyTorch 2.6 needs and later versions do not.


def _real(buf: torch.Tensor) -> bool:
    # The buffers are plain tensors; tracing passes fake or functional tensor subclasses.
    return type(buf) is torch.Tensor


def _restore_versions(tensors: Sequence[torch.Tensor], versions: Sequence[int]) -> None:
    """Set each tensor's autograd version back, what `torch.autograd._unsafe_preserve_version_counter`
    does on exit. Private API, so best effort: without it the version stays bumped, as before."""
    try:
        torch._C._autograd._unsafe_set_version_counter(tuple(tensors), tuple(versions))
    except Exception:  # noqa: BLE001
        try:  # older releases take one tensor at a time
            for tensor, version in zip(tensors, versions):
                torch._C._autograd._unsafe_set_version_counter(tensor, version)
        except Exception:  # noqa: BLE001
            pass


class _NotRecomputed:
    """While AOTAutograd traces a hook's write: leave the write out of a checkpointed block's
    recompute set.

    Dynamo marks every node of a block wrapped in `torch.utils.checkpoint.checkpoint` as "must
    recompute", the hook's write among them, and the mark reaches the nodes AOTAutograd traces
    through `torch.fx.traceback`'s current metadata. PyTorch 2.6's partitioner then saves every
    graph input such a node reads for the backward pass, so it saved the hook's buffer, though
    nothing in backward reads it. The next write to the buffer (a second forward before the
    backward, or the zeroing at a step) then made `backward()` raise "modified by an inplace
    operation". PyTorch 2.9 and later no longer save an input backward does not use. The write
    needs no recomputing: it has no gradient and backward never reads the buffer.

    Private API, so best effort: where the metadata cannot be reached the write is traced as it
    was."""

    __slots__ = ("_meta", "_tag")

    def __enter__(self) -> None:
        self._meta = None
        try:
            meta = torch.fx.traceback.get_current_meta()
            if "recompute" in meta:
                self._tag = meta.pop("recompute")
                self._meta = meta
        except Exception:  # noqa: BLE001
            self._meta = None

    def __exit__(self, *exc: object) -> None:
        if self._meta is not None:
            self._meta["recompute"] = self._tag


@torch.compiler.allow_in_graph
def _add_compiled(counts: torch.Tensor, hits: torch.Tensor) -> None:
    if not _real(counts):
        with _NotRecomputed():
            counts.add_(hits)
    elif torch._C._current_graph_task_id() == -1:
        version = counts._version
        counts.add_(hits)
        _restore_versions((counts,), (version,))


@torch.compiler.allow_in_graph
def _row_max_compiled(buf: torch.Tensor, row: int) -> torch.Tensor:
    """The largest magnitude a layer's row holds so far: the hook's one read of its buffer."""
    if not _real(buf):
        with _NotRecomputed():
            return buf[row, 2]
    return buf[row, 2]


@torch.compiler.allow_in_graph
def _set_row_compiled(buf: torch.Tensor, row: int, stats: torch.Tensor) -> None:
    if not _real(buf):
        with _NotRecomputed():
            buf[row] = stats
    elif torch._C._current_graph_task_id() == -1:
        version = buf._version
        buf[row] = stats
        _restore_versions((buf,), (version,))


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


def _on_stack(frame: Any, start: Any, limit: int = 64) -> bool:
    while start is not None and limit > 0:
        if start is frame:
            return True
        start, limit = start.f_back, limit - 1
    return False


_last_attempt = 0


def _next_attempt() -> int:
    """This moment in Unix milliseconds, and never the same value twice in one process."""
    global _last_attempt
    _last_attempt = max(time.time_ns() // 1_000_000, _last_attempt + 1)
    return _last_attempt


def _recomputing() -> bool:
    """True while activation checkpointing recomputes a forward pass during backward.

    Recomputation runs eagerly inside the autograd engine (PyTorch disables Dynamo for
    it), so a hook firing then would count the same tokens and activations twice. Under
    compilation the check is skipped here, since it would be decided once at trace time;
    a compiled block that is checkpointed can still be replayed when the backend runs
    Dynamo's graph as it is, and `_add_compiled` and `_set_row_compiled` check for that
    when they run.
    """
    return not torch.compiler.is_compiling() and torch._C._current_graph_task_id() != -1


class _Evaluating:
    """Whether the session's phase is `"eval"` (`Session.set_phase`), shared with its hooks.
    An attribute of an object of its own, so a hook that reads it guards on one bool. Set
    on the instance from the start: one that moved from the class to the instance at the
    first `set_phase` would fail the guard and compile the training graph again."""

    def __init__(self) -> None:
        self.on = False


def _evaluating(module: nn.Module, phase: _Evaluating) -> bool:
    """True for a forward that is not training: one whose statistics belong to no optimizer
    step, so they must not reach the next step's record.

    Lightning and the Hugging Face `Trainer` evaluate with the model in eval mode and without
    gradients (`torch.no_grad` or `torch.inference_mode`), as most hand-written loops do, so
    the two together mark an evaluation without anything from the user. Neither alone does:
    reentrant activation checkpointing runs a training forward without gradients, and some
    training keeps modules in eval mode (frozen blocks, dropout turned off). Under
    `session.phase("eval")` (which the Lightning and Hugging Face callbacks set), either one
    is enough. The phase alone never is: a training forward in train mode with gradients
    counts even under a phase left behind by a callback that missed its end.

    What is read is ordered for `torch.compile`, where each value read becomes a guard.
    Dynamo always guards on grad mode, but on a module's `training` only when something
    reads it. With gradients on, the phase flag is read first and `training` only under the
    eval phase, so `model.eval()` alone compiles nothing that the model without the hooks
    would not; the training graph gains one guard, on the flag. With gradients off (a new
    graph anyway), `training` is read first and the flag only in train mode.
    """
    if torch.is_grad_enabled():
        return phase.on and not module.training
    return not module.training or phase.on


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
        if _clip_original is None:
            _clip_original = nn_utils.clip_grad_norm_
        nn_utils.clip_grad_norm_ = _clip_grad_norm_wrapper  # type: ignore[assignment]
        clip_module.clip_grad_norm_ = _clip_grad_norm_wrapper  # type: ignore[assignment]
    _clip_sessions.append(session)


def _unwatch_clipping(session: "Session") -> None:
    import torch.nn.utils as nn_utils
    import torch.nn.utils.clip_grad as clip_module

    if session in _clip_sessions:
        _clip_sessions.remove(session)
        if not _clip_sessions and _clip_original is not None:
            # `_clip_original` stays set: code that kept a reference to the wrapper (for
            # example `from torch.nn.utils import clip_grad_norm_` while a session was
            # open) still calls it after the last session closes.
            nn_utils.clip_grad_norm_ = _clip_original  # type: ignore[assignment]
            clip_module.clip_grad_norm_ = _clip_original  # type: ignore[assignment]


class Session:
    """One model's capture. Create it with `tensorquill.init`."""

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
        replica_projection: bool = True,
        probe: Any = None,
        probe_every: int = PROBE_EVERY,
        scaler: Any = None,
        replica_params: Any = None,
        system: bool = False,
        data_wait: bool = False,
        updates: bool = False,
    ) -> None:
        if hasattr(model, "_orig_mod"):
            raise ValueError("call tensorquill.init(model) before torch.compile(model)")
        if every < 1:
            raise ValueError("every must be at least 1")
        if isinstance(grads, bool):
            grad_every = 1 if grads else 0
        elif isinstance(grads, int) and grads >= 0:
            grad_every = grads
        else:
            raise ValueError("grads must be a bool or a non-negative integer")
        self.run = run
        # When this session started, in Unix milliseconds. A job that resumes from an
        # earlier checkpoint under the same run ID writes the same step numbers again;
        # the attempt tells those records from the ones they replace.
        self.attempt = _next_attempt()
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
        self._replica_projection = replica_projection
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
        self._param_index = {id(p): index for index, p in enumerate(self._params)}
        self.param_names = [name for name, _ in params]
        # The parameters the cross-GPU drift check compares: every trainable parameter by
        # default (data parallelism), or only the replicated ones under tensor parallelism.
        replicas = _select_replica_params(params, replica_params)
        self._replica_params = [p for _, p in replicas]
        self.replica_names = [name for name, _ in replicas]
        self.device = self._params[0].device if self._params else torch.device("cpu")

        selected = _select_layers(model, layers)
        self.layer_names = [name for name, _ in selected]
        modules = dict(model.named_modules())
        self._routers = {
            name: spec if isinstance(spec, Router) else Router(int(spec))
            for name, spec in (routers or {}).items()
        }
        # What a router's hook found wrong with its output, said once at the next step (`_commit`).
        # A router with a problem is refused for the rest of the session: its counts leave no
        # entry in any record written after that (`_decode`), and the first such record says
        # why under `moe_refused`. Not per step: the guard reads shapes, so a short batch
        # can let through a step it would refuse at full size, and that step's counts would
        # be the same misread.
        # Each problem is (why, which fix `_refusal` names, None for the usual two).
        self._router_problems: dict[str, list[tuple[str, str | None]]] = {name: [] for name in self._routers}
        self._router_problems_said: set[str] = set()
        self._router_refusals_written: set[str] = set()
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
        self._weight_dtypes: dict[str, int] = {}
        self._spread: torch.Tensor | None = None
        self._counts = {
            name: torch.zeros(_count_width(spec), dtype=torch.int64, device=self.device)
            for name, spec in self._routers.items()
        }
        self._expert_ids = {
            name: torch.arange(spec.n_experts, dtype=torch.int64, device=self.device)
            for name, spec in self._routers.items()
        }
        self._ring_act = torch.zeros(every, n_layers, _SEEN + 1, **f32)
        self._ring_grad = torch.zeros(every, *self._grad.shape, **f32)
        self._ring_grad_seen: list[bool] = []
        self._ring_weight_seen: list[Any] = []  # False, or the weight check's optimizer ownership and dtypes
        # Squared total gradient norm reported by gradient clipping this step (summed over
        # calls, for models clipped in several groups), and its ring.
        self._clip = torch.zeros((), **f32)
        self._ring_clip = torch.zeros(every, **f32)
        self._clip_recorded = False
        self._foreign_steps = 0
        self._clip_calls = 0  # lets an outer wrapper see that an inner one already recorded
        self._ring_clip_seen: list[bool] = []
        # The size of the optimizer's update to each parameter (`updates=True`): its RMS and
        # whether it was read this step, and the ring. Empty when the option is off.
        self.updates = bool(updates)
        self._update = torch.zeros(n_params if self.updates else 0, 2, **f32)
        self._ring_update = torch.zeros(every, *self._update.shape, **f32)
        self._update_recorded = False
        self._ring_update_seen: list[bool] = []
        # Why an optimizer's update is not read, said once on a record (`update_not_read`), and
        # what each optimizer seen is: None for one that is read.
        self._update_why: dict[str, str] = {}
        self._update_why_written: set[str] = set()
        self._update_readable: weakref.WeakKeyDictionary[Any, str | None] = weakref.WeakKeyDictionary()
        self._ring_spread: torch.Tensor | None = None
        self._ring_spread_seen: list[int] = []  # columns recorded at each step; 0 if none
        if replica_group is not None and self._replica_params:
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

        # Forwards that are not training (evaluation, the probe) are left out by the hooks
        # (`_evaluating`); this says whether the session's phase is "eval" (`set_phase`).
        self._evaluating = _Evaluating()
        self._handles = [
            module.register_forward_hook(_stats_hook(self._act, self._stat_idx, row, self._evaluating))
            for row, (_, module) in enumerate(selected)
        ]
        self._handles += [
            modules[name].register_forward_hook(
                _router_hook(
                    self._counts[name], self._expert_ids[name], spec, self._router_problems[name], self._evaluating
                )
            )
            for name, spec in self._routers.items()
        ]
        self._probe_state = ProbeState()
        if self._probe_fn is not None:
            self._handles += install_probe_hooks(_select_layers(model, "blocks"), self._probe_state)

        self._steps: list[int] = []
        self._ring_lr: list[list[float] | None] = []
        self._ring_ts: list[float] = []
        self._pending_lr: list[float] | None = None
        self._ring_scalars: list[dict[str, Any]] = []
        self._ring_amp: list[dict[str, Any]] = []
        self._pending_amp: dict[str, Any] = {}
        self._pending_scalars: dict[str, Any] = {}
        self._ring_probe: list[dict[str, Any] | None] = []
        # Data progress (`data`): None until the first batch is noted, then every step has one.
        self._data_step: DataStep | None = None
        self._ring_data: list[dict[str, Any] | None] = []
        # Checkpoint manifests and the resumed optimizer's state (`checkpoint`), for the next step.
        self._pending_checkpoint: dict[str, Any] = {}
        self._ring_checkpoint: list[dict[str, Any] | None] = []
        # A session that starts after step 0 resumed: its first optimizer step says whether the
        # optimizer came back with its state (`checkpoint.optimizer`). Once, so free.
        self._snapshot_due = start_step > 0
        self._snapshot_before: int | None = None
        self._next_step = start_step
        self._steps_committed = 0
        self._grads_recorded = False
        self._spread_recorded = 0
        self._owned: weakref.WeakKeyDictionary[Any, bool] = weakref.WeakKeyDictionary()
        self._lead: weakref.ref[Any] | None = None
        self._leading = True
        self._grads_after_step = False
        self._step_depth = 0
        self._outer_frame: Any = None
        self._closed = False
        self._phase = "train"
        self._ticks = 0
        beat = getattr(self._sink, "heartbeat", None) if hasattr(self, "_sink") else None
        if beat is not None:
            beat.counter = lambda: self._steps_committed + self._ticks
        self._error: BaseException | None = None
        # GPU memory and NVML readings once per flush (system.py), and the host time spent
        # waiting for batches (datawait.py). Both off unless asked for.
        self._system = SystemReadings(self.device) if system else None
        self._data_timing = False
        self._data_thread = threading.get_ident()
        self._fetching = False
        self._pending_data_wait = 0.0
        self._last_step_clock: float | None = None
        self._ring_wait: list[tuple[float, float | None] | None] = []
        # Bounded, as TensorBoard's SummaryWriter is (`max_queue=10`): if the sink falls
        # behind, for example on a stalled network filesystem, training waits for it
        # instead of holding every unwritten step in memory. No step is dropped.
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=WRITER_QUEUE)
        self._warned_backpressure = False
        self._worker = threading.Thread(target=self._drain, name="tensorquill-writer", daemon=True)
        self._worker.start()
        if getattr(sink, "path", None) is not None:
            clear_end_marker(sink.path)  # the file is being written again (see `sink.write_end_marker`)

        self._clip_watched = clip_norm
        if clip_norm:
            _watch_clipping(self)

        self._scaler = scaler if scaler is not None and scaler.is_enabled() else None
        # With a GradScaler, `scaler.update()` runs once per iteration whether or not the
        # optimizer stepped, so it ends the TensorQuill step; optimizer hooks still read
        # gradients. Without it, a step the scaler skipped would never be counted.
        self._commit_on_update = self._scaler is not None and auto_step
        if self._scaler is not None:
            self._watch_scaler(self._scaler)

        self._optimizer_handles = []
        if auto_step:
            self._optimizer_handles = [
                register_optimizer_step_pre_hook(self._before_optimizer_step),
                register_optimizer_step_post_hook(self._after_optimizer_step),
            ]
        if data_wait:
            self.watch_loaders()

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

    def _optimizer_ownership(self) -> dict[str, Any]:
        """Whether an optimizer that has stepped holds each trainable parameter, read at a
        weight check. Ownership can only be established from optimizers seen stepping, so it
        is reported as unknown, never as "not held", when that is not the case: the optimizer
        hooks are off (`auto_step=False`), or no optimizer that stepped holds any of this
        model's parameters (DeepSpeed ZeRO, Megatron and FSDP1 step flattened or copied
        parameters). A parameter left out on purpose also shows as not held; the detector
        only uses this to explain a stalled norm, never as an alarm of its own.
        """
        if not self.auto_step:
            return {"state": "unknown", "reason": "optimizer hooks are off (auto_step=False)"}
        held: set[int] = set()
        seen = False
        for optimizer in list(self._owned):
            ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
            if ids & self._param_ids:
                seen = True
                held |= ids
        if not seen:
            return {"state": "unknown", "reason": "no optimizer that has stepped holds this model's parameters"}
        unheld = [name for name, p in zip(self.param_names, self._params) if id(p) not in held]
        return {"state": "known", "unheld": unheld, "unheld_count": len(unheld)}

    def _should_record_grads(self) -> bool:
        # Weight checks need the step's gradient norms, so they also turn gradients on. So does
        # the update size: its rule reads the rise of the total gradient norm before clipping,
        # which the record has already when clipping was seen this step (`clip_grad_norm`).
        return (
            self._cadence(self.grad_every) or self._should_check_weights()
            or (self.updates and not self._clip_recorded)
        )

    def _should_check_weights(self) -> bool:
        return self._cadence(self.weight_every)

    def _cadence(self, every: int) -> bool:
        # The every-th, 2*every-th, ... step, counting from 1 (as Lightning and wandb do).
        return every > 0 and (self._steps_committed + 1) % every == 0

    def _note_foreign_step(self) -> None:
        # DeepSpeed's ZeRO optimizers and Megatron's mixed-precision optimizer step copies
        # of the weights, so the optimizer that steps holds none of the model's
        # parameters and no step would ever be counted. Say so instead of writing nothing.
        if self._lead is not None or self._foreign_steps < 0:
            return
        self._foreign_steps += 1
        if self._foreign_steps == FOREIGN_STEPS_WARNING:
            self._foreign_steps = -1  # warn once
            warnings.warn(
                f"TensorQuill has recorded no steps: {FOREIGN_STEPS_WARNING} optimizer steps ran, none on this "
                "model's parameters. If your framework steps copies of the weights (DeepSpeed ZeRO, "
                "Megatron-LM), pass auto_step=False and call session.step() after each update.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _before_optimizer_step(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        if self._closed:
            return
        if not self._owns(optimizer):
            self._note_foreign_step()
            return
        # PyTorch wraps each optimizer class's own `step` with the global hooks, so an
        # optimizer that subclasses another and calls `super().step()` fires them again,
        # nested inside the outer call. A call is nested only while the outer call's
        # wrapper frame is still running; after an outer step raised, it is not, and the
        # next call starts afresh instead of being ignored forever.
        wrapper = sys._getframe(1)
        if self._step_depth and _on_stack(self._outer_frame, wrapper.f_back):
            self._step_depth += 1
            return
        self._step_depth, self._outer_frame = 1, wrapper
        # One TensorQuill step per training iteration. With several optimizers on one
        # model, the first one that ever stepped leads: its steps end TensorQuill steps,
        # and the others' steps add none and record nothing twice. Gradients are read
        # before the lead steps, after backward, so they include every optimizer's
        # parameters.
        lead = self._lead() if self._lead is not None else None
        if lead is None:
            self._lead, lead = weakref.ref(optimizer), optimizer
        self._leading = optimizer is lead
        if self._leading and self._snapshot_due:
            self._snapshot_before = optimizer_entries(optimizer)
        # With `step(closure)`, the closure runs forward and backward inside the step, so
        # gradients are read after it instead (Lightning steps this way).
        closure = kwargs.get("closure", args[1] if len(args) > 1 else None)
        self._grads_after_step = callable(closure)
        if self._leading and not self._grads_after_step and not self._grads_recorded and self._should_record_grads():
            self.record_grads()

    def _after_optimizer_step(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        if self._closed or not self._owns(optimizer):
            return
        if self._step_depth > 1:
            self._step_depth -= 1
            return
        self._step_depth, self._outer_frame = 0, None
        if self.updates:
            # Every optimizer that holds some of this model's parameters, not only the lead: one
            # that steps after the lead has ended the step is read into the next record.
            self._read_updates(optimizer)
        if not self._leading:
            return
        if self._grads_after_step and not self._grads_recorded and self._should_record_grads():
            self.record_grads()
        if self._snapshot_due and self._snapshot_before is not None:
            self._snapshot_due = False
            self._pending_checkpoint["optimizer"] = {
                "entries_before": self._snapshot_before,
                "entries_after": optimizer_entries(optimizer),
                "step_after": optimizer_step_value(optimizer),
            }
        self._pending_lr = _learning_rates(optimizer)
        if not self._commit_on_update:
            self._commit(None)

    def _watch_scaler(self, scaler: Any) -> None:
        original = scaler.update

        def update(*args: Any, **kwargs: Any) -> Any:
            if not self._closed:
                self._record_scaler(scaler)
            out = original(*args, **kwargs)
            if self._commit_on_update and not self._closed:
                if self._pending_lr is None and self._lead is not None and self._lead() is not None:
                    self._pending_lr = _learning_rates(self._lead())
                self._commit(None)
            return out

        scaler.update = update  # an instance attribute; `close()` removes it
        self._scaler_original = original

    def _record_scaler(self, scaler: Any) -> None:
        # Read before `update()` resets them: whether any optimizer's gradients had an Inf
        # or NaN this iteration (the scaler then skipped its step), and the scale used.
        # Both stay tensors on the device until the next flush, so no sync is added.
        scale = getattr(scaler, "_scale", None)
        if scale is None:
            return
        found = [
            f for state in scaler._per_optimizer_states.values()
            for f in state["found_inf_per_device"].values()
        ]
        skipped = (
            torch.stack([f.to(scale.device).reshape(()) for f in found]).sum() > 0
            if found else torch.ones((), dtype=torch.bool, device=scale.device)
        )
        self._pending_amp = {"step_skipped": skipped.float(), "grad_scale": scale.detach().clone()}

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
        dtypes: dict[str, int] = {}
        for index, param in enumerate(self._params):
            data = param.detach()
            if type(data) is not torch.Tensor:
                data = unwrap_local(data)
            name = str(data.dtype).removeprefix("torch.")
            dtypes[name] = dtypes.get(name, 0) + 1
            if data.numel() == 0:
                continue
            groups.setdefault((data.device, data.dtype), []).append((index, data))
        # How many trainable parameters have each dtype: a weight that never changes means
        # something else in bf16 or fp16, where a small update rounds away, than in float32.
        self._weight_dtypes = dtypes
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

    def record_updates(self, optimizer: Any) -> None:
        """Read the size of the update `optimizer` has just applied to each of this model's
        parameters, for this step's record (`update_rms`). Needs `init(..., updates=True)`.

        Called automatically after every `optimizer.step()` with `auto_step`. Call it yourself
        after the step with `auto_step=False`. An optimizer that is not Adam or AdamW is not
        read, and the record says why once (`update_not_read`). It never raises.
        """
        if not self.updates:
            raise RuntimeError("pass updates=True to tensorquill.init to record update sizes")
        if not self._closed:
            self._read_updates(optimizer)

    @torch.no_grad()
    def _read_updates(self, optimizer: Any) -> None:
        # Adam's update to a parameter, without weight decay, is lr * m_hat / (sqrt(v_hat) + eps)
        # (torch/optim/adam.py: the single-tensor, foreach and fused paths all apply
        # `step_size * exp_avg / (sqrt(exp_avg_sq) / sqrt(1 - beta2**t) + eps)` with
        # `step_size = lr / (1 - beta1**t)`, and amsgrad reads `max_exp_avg_sq` for the second
        # moment). Both moments are the optimizer's own state after the step, so nothing is kept
        # of the weights. One parameter at a time: the only temporary is one tensor of that
        # parameter's size. Every value stays on its device, so no sync is added.
        try:
            known = optimizer in self._update_readable
        except TypeError:  # not hashable or not weakly referenceable: judged each time
            known = False
        why = self._update_readable[optimizer] if known else _not_adam(optimizer)
        name = type(optimizer).__name__
        if not known:
            try:
                self._update_readable[optimizer] = why
            except TypeError:
                pass
        if why is not None:
            self._update_why.setdefault(name, why)
            return
        try:
            found: list[tuple[int, torch.Tensor]] = []
            corrections: dict[tuple[float, float, float], tuple[float, float]] = {}
            for group in optimizer.param_groups:
                lr, (beta1, beta2), eps = group["lr"], group["betas"], group["eps"]
                second = "max_exp_avg_sq" if group.get("amsgrad") else "exp_avg_sq"
                for param in group["params"]:
                    index = self._param_index.get(id(param))
                    state = optimizer.state.get(param) if index is not None else None
                    # Adam leaves a parameter without a gradient alone, so its state is the last
                    # step's that had one: not this step's update.
                    if not state or param.grad is None:
                        continue
                    first, moment = state["exp_avg"], state[second]
                    if first.numel() == 0:
                        continue
                    if first.is_complex():
                        first, moment = torch.view_as_real(first), torch.view_as_real(moment)
                    if first.dtype in _LOW_PRECISION:
                        first, moment = first.float(), moment.float()
                    steps = state["step"]
                    update = torch.sqrt(moment)  # the one temporary
                    on_device = any(isinstance(v, torch.Tensor) and v.device.type != "cpu" for v in (steps, beta1, beta2))
                    if not on_device:
                        # The usual case off the fused and capturable paths: the step count is on the
                        # CPU, so the bias corrections are plain numbers, worked out once for each count.
                        count = max(float(steps), 1.0)
                        key = (count, float(beta1), float(beta2))
                        if key not in corrections:
                            corrections[key] = (math.sqrt(1 - key[2] ** count), 1 / (1 - key[1] ** count))
                        root, first_correction = corrections[key]
                        # m / (sqrt(v) / root + eps) is root * m / (sqrt(v) + eps * root): one pass fewer.
                        update.add_(eps * root)
                        torch.div(first, update, out=update)
                        rms = torch.linalg.vector_norm(update) * (lr * (root * first_correction / math.sqrt(update.numel())))
                    else:
                        # On the device (fused, capturable): tensor arithmetic, so nothing is synced.
                        if not isinstance(steps, torch.Tensor):
                            steps = torch.tensor(float(steps))
                        # At least 1: under a loss scaler the fused step takes its count back when it
                        # skips, and 1 - beta**0 would divide by zero.
                        steps = steps.clamp(min=1)
                        if steps.device.type != "mps":  # float64, as PyTorch works these out; MPS has none
                            steps = steps.double()
                        update.div_(((1 - beta2**steps) ** 0.5).to(update.dtype)).add_(eps)
                        torch.div(first, update, out=update)
                        norm = torch.linalg.vector_norm(update)
                        step_size = torch.as_tensor(lr / (1 - beta1**steps)).to(norm.dtype)
                        rms = norm * step_size / math.sqrt(update.numel())
                    found.append((index, rms.reshape(())))
            if not found:
                return
            by_device: dict[torch.device, list[tuple[int, torch.Tensor]]] = {}
            for index, rms in found:
                by_device.setdefault(rms.device, []).append((index, rms))
            for items in by_device.values():
                stacked = torch.stack([rms for _, rms in items]).to(self._update).abs_()
                self._update.index_copy_(
                    0, self._index(tuple(index for index, _ in items)),
                    torch.stack((stacked, torch.ones_like(stacked)), dim=1),
                )
            self._update_recorded = True
        except Exception as error:  # noqa: BLE001
            # Nothing here may end a training step: a state this cannot read (or no memory for
            # the temporary) is said once, and that optimizer is not read again.
            self._update_why.setdefault(name, f"reading its state failed ({_short(error)})")
            try:
                self._update_readable[optimizer] = self._update_why[name]
            except TypeError:
                pass

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
        self._clip_calls += 1

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

    def data(
        self,
        indices: Any = None,
        *,
        epoch: int | None = None,
        sampler_epoch: int | None = None,
        shuffle: bool | None = None,
        data_rank: int | None = None,
        shard: Any = None,
        offset: int | None = None,
    ) -> None:
        """Note one batch drawn for the step in progress: its sample indices (or ids), which
        are hashed and never stored, and where the data stands.

        `epoch` is the pass over the data; `sampler_epoch` the sampler's own epoch (what
        `DistributedSampler.set_epoch` set); `shuffle` whether the order is meant to change
        from pass to pass (None: unknown); `data_rank` which share of the data this process
        reads (a `DistributedSampler`'s rank; ranks that are meant to read the same data, such
        as a tensor-parallel group, give the same value); `shard` and `offset` where a
        streaming dataset stands. Call it once per batch, micro-batches included; every step
        after the first call records how many batches it drew. Host-side and cheap: one
        BLAKE2b over the indices. `session.watch_loader(loader)` calls it for you.
        """
        if self._closed:
            raise RuntimeError("the session is closed")
        if self._data_step is None:
            self._data_step = DataStep()
        self._data_step.add(
            indices,
            data_fields(epoch=epoch, sampler_epoch=sampler_epoch, shuffle=shuffle, data_rank=data_rank,
                        shard=shard, offset=offset),
        )

    def watch_loader(self, loader: Any, *, ids: Callable[[Any], Any] | None = None, epoch: int = 0) -> WatchedLoader:
        """Iterate `loader` through this instead, and each batch is noted with `session.data`.

        For a map-style dataset the sample indices are the ones the loader's batch sampler
        yields, read as it yields them (in order, with or without worker processes). An
        `IterableDataset` has no indices: pass `ids`, a callable that takes a batch and returns
        its sample ids (CPU values). Each `iter()` is one pass, counted from `epoch`; the
        sampler's own epoch, whether it shuffles and its rank are read from a
        `DistributedSampler`, `RandomSampler` or `SequentialSampler`.
        """
        return WatchedLoader(self, loader, ids=ids, epoch=epoch)

    def checkpoint(self, state: Mapping[str, Any], *, event: str = "save") -> None:
        """Record what a checkpoint holds: call it with the dict passed to `torch.save`
        (`event="save"`) or the one loaded to resume (`event="load"`, before the first step).

        The manifest names the components found (model, optimizer, scheduler, scaler, rng,
        data position) with cheap identities: counts, the optimizer's step, the scheduler's
        `last_epoch`, the loss scale. A save is recorded with the last step taken (as
        `after_step`); a load with the next step, the first of the resumed run. The detectors
        then say what a resume did not restore.
        """
        self._record_manifest(checkpoint_manifest(state), event)

    def checkpoint_files(self, folder: str | Path, *, event: str = "save") -> None:
        """`checkpoint` for a checkpoint folder, as the Hugging Face `Trainer` and Accelerate's
        `save_state` write: the components are read from the file names (`optimizer.pt`,
        `scheduler.pt`, `scaler.pt`, `rng_state*.pth` or `random_states_*.pkl`, the model's
        weights, `trainer_state.json` or `sampler*.bin` for the data position). No file is opened."""
        try:
            names = [entry.name for entry in Path(folder).iterdir()]
        except OSError:
            return
        self._record_manifest(files_manifest(names), event)

    def _record_manifest(self, manifest: dict[str, Any], event: str) -> None:
        if self._closed:
            raise RuntimeError("the session is closed")
        if event not in ("save", "load"):
            raise ValueError('event must be "save" or "load"')
        last = self._next_step - 1 if self._steps_committed else None
        if event == "save":
            manifest["after_step"] = last
            if self._steps:  # the step it follows is still in the ring: record it there
                row = self._ring_checkpoint[-1] or {}
                row["save"] = manifest
                self._ring_checkpoint[-1] = row
                return
        else:
            self._snapshot_due = self._snapshot_due or self._snapshot_before is None
        self._pending_checkpoint[event] = manifest

    def _ensure_spread_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._spread is None or self._ring_spread is None:
            # Apple MPS rejects float64 allocations; keep float64 replica buffers on CPU there.
            spread_device = torch.device("cpu") if self.device.type == "mps" else self.device
            n_params = len(self._replica_params)
            self._spread = torch.zeros(n_params, FINGERPRINT_WIDTH, dtype=torch.float64, device=spread_device)
            self._ring_spread = torch.zeros(
                self.every, n_params, FINGERPRINT_WIDTH, dtype=torch.float64, device=spread_device
            )
        return self._spread, self._ring_spread

    @torch.no_grad()
    def record_replica_spread(self, group_or_spread: Any = None) -> None:
        """Record cross-GPU parameter relative spread `(norm, sum, projection)` for this step.

        Pass a replication `ProcessGroup` (or rely on `replica_group` passed to `init`), or
        pass a precomputed `[n_params, 3]` (or older `[n_params, 2]`) tensor from
        `spread_from_fingerprints`.
        """
        if not self._replica_params:
            return
        target = self._replica_group if group_or_spread is None else group_or_spread
        if target is None:
            raise ValueError("pass a replication process group or precomputed spread tensor")
        spread_buf, _ = self._ensure_spread_buffers()
        if isinstance(target, torch.Tensor):
            spread = target.to(spread_buf)
        else:
            spread = replica_spread(
                self._replica_params, target, projection=self._replica_projection
            ).to(spread_buf)
        width = spread.shape[-1]
        spread_buf.zero_()
        spread_buf[:, :width].copy_(spread)
        self._spread_recorded = width

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

    def _router_refusals(self) -> dict[str, str]:
        """Each refused router's reason: why, and the fixes for how it was named (`_refusal`)."""
        return {
            name: _refusal(name, self._routers[name], *found[0], device=self._counts[name].device)
            for name, found in self._router_problems.items()
            if found
        }

    def _say_router_problems(self, stacklevel: int) -> None:
        for name, reason in self._router_refusals().items():
            if name not in self._router_problems_said:
                self._router_problems_said.add(name)
                warnings.warn(
                    f"TensorQuill is not counting the router {name!r}: {reason}. Its expert loads are not recorded.",
                    RuntimeWarning, stacklevel=stacklevel,
                )

    def _judge_index_ranges(self, counts: torch.Tensor, n: int) -> None:
        """Refuse an `indices=` router whose function returned indices outside 0..n_experts-1
        on any step of this block. Read from the counts' last slot, already on the CPU: the
        check adds no transfer and no sync. Default routers are not judged on their values:
        their indices come from `topk`/`argmax` of logits, or from an integer tensor that the
        shape guard alone judges, as before."""
        offset = 0
        for name, spec in self._routers.items():
            width = _count_width(spec)
            size = n * width
            if spec.indices is not None and not self._router_problems[name]:
                outside = counts[offset : offset + size].reshape(n, width)[:, -1].tolist()
                bad = next((i for i, v in enumerate(outside) if v), None)
                if bad is not None:
                    self._router_problems[name].append((
                        f"its indices= function returned {outside[bad]} value{'s' if outside[bad] != 1 else ''} "
                        f"outside 0 to {spec.n_experts - 1} at step {self._steps[bad]}, which match no expert",
                        None,
                    ))
            offset += size

    @torch.no_grad()
    def _commit(self, step: int | None) -> None:
        current = self._next_step if step is None else step
        self._say_router_problems(stacklevel=4)
        if (
            not self._spread_recorded
            and self._replica_group is not None
            and self._replica_params
            and (current + 1) % self.replica_every == 0
        ):
            # Like Megatron's check_weight_hash_across_dp_replicas_interval: a float64 pass
            # over every weight plus an all_gather is too costly for every step. The
            # all_gather is a collective, so every rank must call it on the same steps:
            # it is scheduled from the step number, which the ranks share, not from each
            # rank's own count of recorded steps, which can differ (a rank that started
            # its session later, or skipped a step), and would leave the others waiting.
            self.record_replica_spread(self._replica_group)
        weights_checked = self._grads_recorded and self._should_check_weights() and bool(self._params)
        if weights_checked:
            self._record_weights()
        row = len(self._steps)
        has_act = self._act.numel() > 0
        # The buffers the forward hooks write; their versions are kept over the zeroing below.
        hooked = ([self._act] if has_act else []) + self._counts_list
        hooked_versions = [buf._version for buf in hooked]
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
        self._ring_weight_seen.append(
            {"ownership": self._optimizer_ownership(), "dtypes": self._weight_dtypes} if weights_checked else False
        )
        if self._clip_recorded:
            self._ring_clip[row].copy_(self._clip)
            self._clip.zero_()
        self._ring_clip_seen.append(self._clip_recorded)
        self._clip_recorded = False
        if self.updates:
            if self._update_recorded:
                self._ring_update[row].copy_(self._update)
                self._update.zero_()
            elif not self.auto_step:
                self._update_why.setdefault(
                    UPDATES_HOOKS_OFF,
                    "the optimizer hooks are off, so call session.record_updates(optimizer) after each optimizer.step()",
                )
            elif self._lead is None:
                # A step ended by `scaler.update()` with no optimizer of this model seen stepping.
                self._update_why.setdefault(
                    UPDATES_NO_OPTIMIZER,
                    "no optimizer that has stepped holds this model's parameters (DeepSpeed ZeRO, Megatron-LM and "
                    "FSDP1 step flattened or copied parameters), so there is no Adam state to read for them",
                )
            self._ring_update_seen.append(self._update_recorded)
            self._update_recorded = False
        if self._spread_recorded and self._spread is not None and self._ring_spread is not None:
            self._ring_spread[row].copy_(self._spread)
            self._spread.zero_()
        self._ring_spread_seen.append(self._spread_recorded)
        if self._counts_list:
            torch._foreach_copy_(
                [ring[row] for ring in self._ring_counts_list], self._counts_list
            )
            torch._foreach_zero_(self._counts_list)
        if hooked:
            _restore_versions(hooked, hooked_versions)
        self._grads_recorded = False
        self._spread_recorded = 0
        probe_due = self._cadence_probe()
        self._steps.append(current)
        self._ring_probe.append(None)
        self._ring_lr.append(self._pending_lr)
        self._ring_ts.append(time.time())  # when the step ended, Unix seconds
        self._pending_lr = None
        self._ring_scalars.append(self._pending_scalars)
        self._ring_amp.append(self._pending_amp)
        if self._data_timing:
            clock = time.perf_counter()
            last, self._last_step_clock = self._last_step_clock, clock
            self._ring_wait.append((self._pending_data_wait, None if last is None else clock - last))
            self._pending_data_wait = 0.0
        else:
            self._ring_wait.append(None)
        self._pending_amp = {}
        self._pending_scalars = {}
        if self._data_step is not None:
            self._ring_data.append(self._data_step.record())
            self._data_step = DataStep()
        else:
            self._ring_data.append(None)
        self._ring_checkpoint.append(self._pending_checkpoint or None)
        self._pending_checkpoint = {}
        self._next_step = current + 1
        self._steps_committed += 1
        if self._phase != "train":
            # A committed training step is training stepping: whatever pause was announced is over.
            # This also clears a phase left behind by a callback that missed its end.
            self.set_phase("train")
        if probe_due:
            # The step is complete before the probe runs, so a failing probe loses only
            # its own result, recorded against the step, and never the step.
            self._ring_probe[-1] = self._probe()
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
        except Exception as error:  # noqa: BLE001
            warnings.warn(f"TensorQuill probe failed and was skipped: {error!r}", RuntimeWarning, stacklevel=3)
            return {"error": f"{type(error).__name__}: {error}"}
        finally:
            buffers = [b for b in (self._act, *self._counts_list) if b.numel() > 0]
            if buffers:
                torch._foreach_zero_(buffers)

    def flush(self) -> None:
        """Move the ring to the CPU and queue it for the sink."""
        if self._error is not None:
            self._steps = []
            self._ring_lr = []
            self._ring_ts = []
            self._ring_scalars = []
            self._ring_amp = []
            self._ring_probe = []
            self._ring_grad_seen = []
            self._ring_weight_seen = []
            self._ring_clip_seen = []
            self._ring_update_seen = []
            self._ring_spread_seen = []
            self._ring_data = []
            self._ring_checkpoint = []
            self._ring_wait = []
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
        if counts is not None:
            self._judge_index_ranges(counts, n)
            self._say_router_problems(stacklevel=4)
        spreads = (
            _cpu_copy(self._ring_spread[:n])
            if self._ring_spread is not None and any(self._ring_spread_seen)
            else None
        )
        clips = _cpu_copy(self._ring_clip[:n]) if any(self._ring_clip_seen) else None
        clip_seen = self._ring_clip_seen
        update_seen, self._ring_update_seen = self._ring_update_seen, []
        updates = _cpu_copy(self._ring_update[:n]) if any(update_seen) else None
        update_why = {name: why for name, why in self._update_why.items() if name not in self._update_why_written}
        self._update_why_written.update(update_why)
        grad_seen = self._ring_grad_seen
        weight_seen = self._ring_weight_seen
        spread_seen = self._ring_spread_seen
        scalars = _scalars_to_cpu(self._ring_scalars)
        amps = _scalars_to_cpu(self._ring_amp)
        self._ring_amp = []
        probes = self._ring_probe
        data_rows, checkpoints = self._ring_data, _checkpoints_to_cpu(self._ring_checkpoint)
        self._ring_data = []
        self._ring_checkpoint = []
        steps, lrs, stamps = self._steps, self._ring_lr, self._ring_ts
        waits, self._ring_wait = self._ring_wait, []
        # Host-side allocator counters: no GPU work, no sync (system.py).
        system = self._system.flush() if self._system is not None else None
        # Read here, on the training thread, so the block is decided as it is queued.
        refused = self._router_refusals()
        announce = {name: why for name, why in refused.items() if name not in self._router_refusals_written}
        self._router_refusals_written.update(announce)
        self._ring_ts = []
        self._steps = []
        self._ring_lr = []
        self._ring_scalars = []
        self._ring_probe = []
        self._ring_grad_seen = []
        self._ring_weight_seen = []
        self._ring_clip_seen = []
        self._ring_spread_seen = []
        block = (
            steps, lrs, floats, counts, spreads, clips, clip_seen, grad_seen, weight_seen,
            spread_seen, scalars, probes, stamps, amps, data_rows, checkpoints, waits, system,
            refused, announce, updates, update_seen, update_why,
        )
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            if not self._warned_backpressure:
                self._warned_backpressure = True
                warnings.warn(
                    f"TensorQuill's writer is {WRITER_QUEUE * self.every} steps behind; training "
                    "now waits for it. The sink is slower than training, for example a stalled "
                    "network filesystem; pass sink=tensorquill.JsonlSink(path) on local disk.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._queue.put(block)

    def set_phase(self, phase: str = "train") -> None:
        """Say what the process is doing: `"train"` (the default), `"eval"`, `"checkpoint"`, or
        any short word. While it is not `"train"` and its heartbeats go on, the server does
        not call the missing steps a hang. Under `"eval"`, a forward in eval mode or without
        gradients is also left out of the records (`_evaluating`); without it, only one with
        both. Otherwise a no-op unless streaming with `endpoint=`."""
        self._phase = phase
        self._evaluating.on = phase == "eval"
        heartbeat = getattr(self._sink, "heartbeat", None)
        if heartbeat is not None:
            heartbeat.set_phase(phase)

    def tick(self) -> None:
        """Show the server that work is going on during a pause, for example once per
        evaluation batch. A pause whose work stops moving is reported as a suspected hang;
        one that keeps ticking is not limited. The Lightning and Hugging Face callbacks tick
        for you. Cheap: it only adds one to a counter."""
        self._ticks += 1

    def watch_loaders(self) -> None:
        """Record the host time this thread spends in `next()` on any PyTorch `DataLoader`
        during training, as `data_wait` on each step, until the session closes. See
        `tensorquill.datawait`. `init(..., data_wait=True)` calls it."""
        if self._closed:
            raise RuntimeError("the session is closed")
        self._start_data_timing()
        datawait.watch(self)

    def time_data(self, iterable: Any) -> "datawait.TimedData":
        """`for batch in session.time_data(loader):` records the host time spent fetching each
        batch, as `data_wait` on each step. Move the batch to the device after this, outside
        the timer. For loops whose loader is not a PyTorch `DataLoader`; with one,
        `watch_loaders()` does the same without changing the loop."""
        return datawait.TimedData(iterable, self)

    def _start_data_timing(self) -> None:
        if not self._data_timing:
            self._data_timing = True
            self._pending_data_wait = 0.0
            self._last_step_clock = None

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """`with session.phase("eval"): ...` marks a stretch with no training steps, and goes
        back to the phase it interrupted afterwards, even if the block raises."""
        previous = self._phase
        self.set_phase(name)
        try:
            yield
        finally:
            self.set_phase(previous)

    def close(self, *, _exception: BaseException | None = None) -> None:
        """Flush what is left and stop the writer. Forward hooks stay on the model, because
        removing a hook from a compiled module triggers a recompile. Once the records file is
        closed and complete, `<name>.end.json` is left beside it (`sink.write_end_marker`), so a
        reader can tell this run from one that was killed; none when the writer failed."""
        if self._closed:
            return
        self._closed = True
        datawait.unwatch(self)
        for handle in self._optimizer_handles:
            handle.remove()
        _unwatch_clipping(self)
        if self._scaler is not None and "update" in vars(self._scaler):
            del self._scaler.update
        try:
            self.flush()
        finally:
            self._queue.put(None)
            self._worker.join()
            self._sink.close()
        self._raise_worker_error()
        if getattr(self._sink, "path", None) is not None:
            write_end_marker(self._sink.path, run_id=self.run.run_id, rank=self.run.rank, attempt=self.attempt,
                             exception=_exception)

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close(_exception=exc[1])

    # ---- writer thread ------------------------------------------------------------

    def _raise_worker_error(self) -> None:
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError("the TensorQuill writer thread failed") from error

    def _drain(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                if self._error is not None:
                    continue
                if self._system is not None:
                    # NVML is read here, on the writer thread. A failure is written as a note
                    # in the record (system.py) and never stops the writer, which would stop training.
                    # Found by its type, not its place, so other parts of the block can come after it.
                    try:
                        item = tuple(self._system.write(x) if isinstance(x, SystemPart) else x for x in item)
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    self._sink.write(self._decode(*item))
                except BaseException as error:  # noqa: BLE001
                    self._error = error
        finally:
            if self._system is not None:
                self._system.close()

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
        weight_seen: list[Any],
        spread_seen: list[bool],
        scalars: list[dict[str, Any]] | None = None,
        probes: list[dict[str, Any] | None] | None = None,
        stamps: list[float] | None = None,
        amps: list[dict[str, Any]] | None = None,
        data_rows: list[dict[str, Any] | None] | None = None,
        checkpoints: list[dict[str, Any] | None] | None = None,
        waits: list[tuple[float, float | None] | None] | None = None,
        system: dict[str, Any] | None = None,
        refused: Mapping[str, str] | None = None,
        announce: Mapping[str, str] | None = None,
        updates: torch.Tensor | None = None,
        update_seen: list[bool] | None = None,
        update_why: Mapping[str, str] | None = None,
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
        spread_rows = (
            spreads.reshape(n, len(self.replica_names), FINGERPRINT_WIDTH).tolist() if spreads is not None else None
        )
        update_rows = updates.reshape(n, n_params, 2).tolist() if updates is not None and n_params > 0 else None
        moe: dict[str, list[list[int]]] = {}
        offset = 0
        for name, spec in self._routers.items():
            width = _count_width(spec)
            size = n * width
            if not refused or name not in refused:
                moe[name] = counts[offset : offset + size].reshape(n, width)[:, : spec.n_experts].tolist()
            offset += size

        records = []
        for i, step in enumerate(steps):
            grad_norms = (
                {name: norm for name, (norm, seen, _, _) in zip(self.param_names, grad[i]) if seen}
                if grad is not None and grad_seen[i]
                else {}
            )
            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run.run_id,
                "rank": self.run.rank,
                "attempt": self.attempt,
                "step": step,
                "ts": stamps[i] if stamps is not None else None,
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
                if isinstance(weight_seen[i], dict):
                    record["optimizer_ownership"] = weight_seen[i]["ownership"]
                    record["weight_dtypes"] = weight_seen[i]["dtypes"]
                if self.param_names and grad[i][0][3] >= 0:
                    record["weights_unchanged"] = [
                        name for name, row in zip(self.param_names, grad[i]) if row[3] > 0
                    ]
            if clips is not None and clip_seen[i]:
                record["clip_grad_norm"] = math.sqrt(float(clips[i]))
            if update_rows is not None and update_seen and update_seen[i]:
                # The RMS of the optimizer's update to each parameter it stepped (weight decay
                # left out): the L2 norm of the update over the square root of its size.
                record["update_rms"] = {name: rms for name, (rms, seen) in zip(self.param_names, update_rows[i]) if seen}
            if lrs[i] is not None:
                record["lr"] = lrs[i]
            if amps is not None and amps[i]:
                record["step_skipped"] = bool(amps[i]["step_skipped"])
                record["grad_scale"] = amps[i]["grad_scale"]
            if scalars is not None and scalars[i]:
                record["scalars"] = scalars[i]
            if waits is not None and waits[i] is not None:
                record["data_wait"] = waits[i][0]
                if waits[i][1] is not None:
                    record["step_time"] = waits[i][1]
            if probes is not None and probes[i] is not None:
                record["probe"] = probes[i]
            if data_rows is not None and data_rows[i] is not None:
                record["data"] = data_rows[i]
            if checkpoints is not None and checkpoints[i] is not None:
                record["checkpoint"] = checkpoints[i]
            if spread_rows is not None and spread_seen[i]:
                width = spread_seen[i]
                record["replica_spread"] = {
                    name: [float(v) for v in row[:width]]
                    for name, row in zip(self.replica_names, spread_rows[i])
                }
            if moe:
                step_moe = {name: rows[i] for name, rows in moe.items()}
                record["moe"] = step_moe
                record["moe_top_k"] = {name: spec.top_k for name, spec in self._routers.items() if name in moe}
                record["moe_stats"] = {
                    name: _router_stats(row_counts) for name, row_counts in step_moe.items()
                }
            records.append(record)
        if announce and records:
            # Once per router and session: the reason it is not counted, so a reader of the
            # records (the evidence, the server) can say MoE was not checked for it.
            records[0]["moe_refused"] = dict(announce)
        if update_why and records:
            # Once for each: why an optimizer's update size is not in these records, so a reader
            # can say the update size was not checked, and not that nothing was found.
            records[0]["update_not_read"] = dict(update_why)
        if system and records:
            records[-1]["system"] = system
        return records


def _checkpoints_to_cpu(rows: list[dict[str, Any] | None]) -> list[dict[str, Any] | None]:
    """The resumed optimizer's `step` as a number. A tensor on the device is read here, once
    per session, with the rest of the block."""
    for row in rows:
        snapshot = row.get("optimizer") if row else None
        if snapshot is not None and isinstance(snapshot.get("step_after"), torch.Tensor):
            value = snapshot["step_after"]
            snapshot["step_after"] = value.item() if value.numel() == 1 else None
    return rows


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


def _select_replica_params(
    params: list[tuple[str, torch.Tensor]], selection: Any
) -> list[tuple[str, torch.Tensor]]:
    """`None`: every trainable parameter. A callable: those for which `fn(name, param)` is
    true. Otherwise an iterable of parameters or parameter names."""
    if selection is None:
        return list(params)
    if callable(selection):
        return [(name, p) for name, p in params if selection(name, p)]
    wanted = list(selection)
    names = {name for name in wanted if isinstance(name, str)}
    ids = {id(p) for p in wanted if isinstance(p, torch.Tensor)}
    if len(names) + len(ids) != len(wanted):
        raise TypeError("replica_params must be a callable, or parameters or parameter names")
    chosen = [(name, p) for name, p in params if name in names or id(p) in ids]
    missing = names - {name for name, _ in chosen}
    if missing or len(ids) != sum(1 for _, p in chosen if id(p) in ids):
        raise ValueError(f"replica_params names no trainable parameter of this model: {sorted(missing) or 'a tensor'}")
    return chosen


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


def _not_adam(optimizer: Any) -> str | None:
    """Why `optimizer`'s update size cannot be read from its state, or None for Adam and AdamW.

    Only `torch.optim.Adam` and `torch.optim.AdamW` are read, and a subclass of either that
    leaves the step alone: the update is worked out from the two moments by Adam's formula,
    so an optimizer that keeps other state, or steps another way, would be misread.
    """
    name = type(optimizer).__name__
    # Walk from the optimizer's own class towards PyTorch's: reaching Adam or AdamW before any
    # class that defines its own step means the step is PyTorch's. AdamW is named beside Adam
    # because it is a class of its own, with its own step, in PyTorch before 2.7.
    own = (torch.optim.Adam, torch.optim.AdamW)
    for cls in type(optimizer).__mro__:
        if cls in own:
            return None
        if any(method in vars(cls) for method in ("step", "_init_group")):
            break
    if isinstance(optimizer, own):
        return f"{name} changes Adam's step, so its update is not known to be Adam's"
    return (
        f"{name} is not Adam or AdamW: the update size is worked out from the two moments Adam keeps "
        "(exp_avg and exp_avg_sq), which this optimizer does not have"
    )


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
    project: str | None = None,
    name: str | None = None,
    group: str | None = None,
    tags: Sequence[str] | None = None,
    config: Mapping[str, Any] | None = None,
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
    replica_projection: bool = True,
    heartbeat_seconds: float | None = 15.0,
    stack_seconds: float | None = 60.0,
    probe: Any = None,
    probe_every: int = PROBE_EVERY,
    scaler: Any = None,
    replica_params: Any = None,
    endpoint: str | None = None,
    system: bool = False,
    data_wait: bool = False,
    updates: bool = False,
) -> Session:
    """Start capturing `model`. Call after `model.to(device)` and before `torch.compile`.

    Args:
        run_id: the tracker's run ID. Defaults to MLFLOW_RUN_ID, the active MLflow run or
            the active W&B run, in that order. It is the run's identity: every file and
            every folder in a server's store is named by it, whatever the project.
        project: the project the run is listed and compared under. Defaults to
            `TENSORQUILL_PROJECT`, then the tracker's (W&B's project, or
            `MLFLOW_EXPERIMENT_NAME`), then `uncategorized`. W&B's rule applies: at most 128
            characters and none of `/ \\ # ? % :`.
        name: a label for the run, which may repeat within a project. Defaults to
            `TENSORQUILL_RUN_NAME`, then the tracker's run name, then the run ID.
        group: an optional label shared by related runs. Defaults to W&B's group.
        tags: optional labels.
        config: an optional flat dictionary of hyperparameters, compared between runs by
            `python -m tensorquill compare`. Nothing is read from the tracker's config.
            These five go into the run manifest, `<run_id>.run.json` beside the `.jsonl`
            file, which rank 0 writes here (see `manifest.py`); other ranks write nothing.
        every: steps kept on the device between transfers to the CPU.
        capture: `"routers"` records MoE router load only: cheap enough to leave on
            (+0.02% to +0.78% on Granite on an RTX PRO 6000). `"full"` adds block
            activation statistics, gradient norms every step and a weight check every 50
            steps, for debugging (+4.15% on the same run). Defaults to `"routers"` when
            `routers` are given and `"full"` otherwise. `layers`, `grads` and `weights`
            override the mode's settings.
        sink: where records go. Defaults to a JSON Lines file under `tensorquill-out/`.
        endpoint: also stream the local file to a `tensorquill-server`, as `grpc://host:port`
            or `grpcs://host:port`. Defaults to `TENSORQUILL_ENDPOINT`; when neither is set
            no socket is opened. For custom/self-signed server certificates under `grpcs://`,
            set `TENSORQUILL_TLS_CA` to the PEM CA path. Needs `pip install pyarrow`.
            See `flight.py`.
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
            check reads every weight in float64 and gathers 24 bytes per parameter tensor
            from each rank: norm, sum and a seeded random projection, which catches
            changes that keep the other two, such as swapped elements. It is a screen,
            not proof of equality. `session.record_replica_spread()` records one on demand.
        replica_projection: set False to skip the projection (two values per tensor). On
            CPU it takes about seven times the norm and sum; GPU cost is not yet measured.
        heartbeat_seconds: with `endpoint=`, a thread of its own tells the server every this
            many seconds that the process is alive and in which phase (see
            `session.phase`), whether or not it is stepping, so the server can tell
            evaluation or a checkpoint from a hang. `None` turns it off.
        stack_seconds: when no step and no `session.tick()` has come for this many seconds,
            each heartbeat also carries where the process's Python threads are (file, line
            and function names; no source text and no values), so a rank reported silent
            comes with the call it is stuck in. `None` turns it off.
        probe: a fixed batch to run through the model every `probe_every` steps, for
            activation histograms per block and attention statistics: a tensor (called as
            `model(batch)`), a mapping of keyword inputs (`model(**batch)`), or a callable
            `fn(model)`. It runs without the compiler, in eval mode and without gradients,
            right after the step's statistics are recorded, and restores each module's
            training flag and the random number generator state afterwards.
        probe_every: steps between probe passes (default 1000).
        scaler: the run's `torch.amp.GradScaler`. Each record then says whether the scaler
            skipped the optimizer step on Inf or NaN gradients (`step_skipped`) and the loss
            scale used (`grad_scale`), and with `auto_step` the step ends at
            `scaler.update()`, so skipped steps are counted too.
        replica_params: which parameters the cross-GPU drift check compares across
            `replica_group`: every trainable parameter by default, as under data
            parallelism. Under tensor parallelism only some weights, such as LayerNorm, are
            the same on every rank of the group; pass a `(name, param) -> bool` filter, or
            the parameters or their names. With Megatron, which marks sharded weights with
            `tensor_model_parallel`, that is
            `lambda name, p: not getattr(p, "tensor_model_parallel", False)`.
        start_step: the first step number. Pass the step you resume from, so a resumed
            run does not repeat step numbers already written.
        clip_norm: record the total gradient norm that `torch.nn.utils.clip_grad_norm_`
            returns for this model's parameters, as `clip_grad_norm`, in any capture mode.
            Free when you clip anyway, and enough for gradient-spike detection.
        system: at each flush, add the CUDA caching allocator's counters (allocated,
            reserved and peak bytes, allocation retries and OOMs; host-side, no sync) and,
            on the writer thread, NVML's temperature, SM clock, power, ECC counters,
            throttle reasons and Xid events (`pip install nvidia-ml-py`) to the block's
            last record as `system`. A reading that cannot be taken is written once as
            unavailable. Off by default until its cost is measured on a GPU. See `system.py`.
        data_wait: record the host time this thread spends in `next()` on PyTorch
            `DataLoader`s during training as `data_wait`, with `step_time`, on each step
            (`session.watch_loaders()`). The Hugging Face and Lightning callbacks turn it
            on. For other loaders wrap them: `session.time_data(loader)`.
        updates: record the size of the optimizer's update to each parameter on every step,
            as `update_rms`: the RMS of `lr * m_hat / (sqrt(v_hat) + eps)`, read from Adam's or
            AdamW's own state right after the step, with no copy of the weights (weight decay
            is not part of it). The `update_spike` rule reads it. Off by default: it reads
            every parameter's two moment tensors on every step, and that cost is not measured
            on a GPU. It also records the total gradient norm on every step, which the rule
            needs (per-parameter gradient norms, unless clipping was seen that step). Any
            other optimizer is not read, and the evidence says so (`not_checked`).
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
    # The explicit labels are checked before anything is opened; a bad one raises here.
    if project is not None:
        _manifest.resolve_project(project)
    if group is not None and not isinstance(group, str):
        raise ValueError(f"group must be a string, not {type(group).__name__}")
    _manifest.resolve_name(name, None)
    tags, config = _manifest.clean_tags(tags), _manifest.clean_config(config)
    run = resolve_run(run_id)
    if sink is None:
        sink = JsonlSink(Path("tensorquill-out") / f"{run.run_id}.rank{run.rank}.jsonl")
    # The project and the name are facts of the run, and a tracker's run exists on rank 0
    # only, so rank 0 alone resolves and writes them: no collective call is needed.
    run_manifest = None
    if run.rank == 0:
        labels = tracker_labels()
        run_manifest = _manifest.build_manifest(
            run_id=run.run_id, run_id_source=run.source,
            project=_manifest.resolve_project(project, labels.get("project")),
            name=_manifest.resolve_name(name, run.run_id, labels.get("name")),
            group=group or labels.get("group"), tags=tags, config=config, world_size=_world_size(),
        )
        records_file = getattr(sink, "path", None)
        if records_file is not None and run.run_id:
            manifest_file = _manifest.local_manifest_path(records_file, run.run_id)
            _manifest.write_manifest(manifest_file, run_manifest)
            # A run started again under its run ID keeps the first start's time.
            run_manifest = _manifest.read_manifest(manifest_file) or run_manifest
    from .flight import attach  # a no-op unless endpoint= or TENSORQUILL_ENDPOINT is set

    sink = attach(
        sink, run.run_id, run.rank, endpoint, world_size=_world_size(), heartbeat_seconds=heartbeat_seconds,
        stack_seconds=stack_seconds, manifest=run_manifest,
    )
    session = Session(
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
        replica_projection=replica_projection,
        probe=probe,
        probe_every=probe_every,
        scaler=scaler,
        replica_params=replica_params,
        system=system,
        data_wait=data_wait,
        updates=updates,
    )
    session.manifest = run_manifest  # the run manifest rank 0 wrote; None on every other rank
    return session
