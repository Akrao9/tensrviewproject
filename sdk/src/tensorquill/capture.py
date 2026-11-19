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
import logging
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
from .distributed import FINGERPRINT_WIDTH, _spread_and_unread, _unread_reason, unwrap_local
from .probe import ProbeState, find_head, install_head_hook, install_probe_hooks, probe_callable, run_probe
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
# How far the writer thread may fall behind before capture stops for the session (`Session._put`),
# the author's decision: about QUEUE_STEPS steps at any `every`, and never fewer than
# QUEUE_MIN_RINGS rings (`_queue_rings`). Training never waits for the sink.
QUEUE_STEPS = 400
QUEUE_MIN_RINGS = 2
# How long `close()` waits for the writer to take and write what is left, at most, in seconds; then
# capture is taken as stopped (no end marker) and `close()` returns. After a stop it waits
# `STOPPED_CLOSE_SECONDS` at most for a writer that may be stuck in the sink.
CLOSE_SECONDS = 60.0
STOPPED_CLOSE_SECONDS = 1.0
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
    default it is the first integer tensor in the output; else a bool routing map
    `[..., n_experts]` with one row per token, as Megatron's `TopKRouter` returns beside its
    probabilities, counted per expert as the map routes, leaving out entries whose paired float
    (the first floating-point tensor of the map's shape) is exactly zero (`_routing_map`); else
    the top-`top_k` indices of the first floating-point tensor whose last dimension equals
    `n_experts`.

    A function given as `indices` is checked as the default is, and more: it must return an
    integer tensor on the counts' device, of a size that is a whole multiple of the tokens
    the router was called on, with values in `0..n_experts-1`. One that raises, or returns
    anything else, has its router refused for the session (`_router_hook`), never an error
    in training.

    The function need not detach: it is called with grad off, on a detached view of the
    output, so it changes nothing in the model's autograd graph (activation checkpointing
    included). A router whose output holds no integer or floating-point tensor is refused.
    """

    n_experts: int
    top_k: int = 1
    indices: Callable[[Any], torch.Tensor] | None = None
    # `top_k` is also written into every record as `moe_top_k`, because the detectors
    # need it to judge expert shares: in a top-8 router no expert can exceed 1/8 of the
    # assignments. Set it when `indices` returns top-k indices.


def _detached(output: Any) -> Any:
    """The output as an `indices=` function is given it (`_router_hook`): a tensor, or a plain
    tuple, list or dict of what it holds, detached; any other output as it is. Only the top
    level, as `_first_tensor` reads it, and only the exact container types, so a named tuple or
    a model's own output class reaches the function unchanged."""
    if isinstance(output, torch.Tensor):
        return output.detach()
    if type(output) in (tuple, list):
        return type(output)(x.detach() if isinstance(x, torch.Tensor) else x for x in output)
    if type(output) is dict:
        return {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
    return output


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


# The tensor types a hook reads under `torch.compile` (the author's decision on lane T2's item 3,
# option (b)): a plain tensor, or a parameter a module returns as it is. Any other subclass is
# refused there by its type alone: under tracing an op it refuses fails the trace before any
# `except` could see it, and its type does not tell such a subclass from one that would be read.
# In eager mode the hooks' `try` still catches what raises, so nothing is refused for its type.
_PLAIN_TYPES = (torch.Tensor, nn.Parameter)


def _compiled_subclass(t: torch.Tensor) -> bool:
    """Whether `t` is a tensor subclass the hooks do not read, which only matters under `torch.compile`."""
    return torch.compiler.is_compiling() and type(t) not in _PLAIN_TYPES


def _routing_map(output: Any, n_experts: int, tokens: int | None) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """A router's bool routing map, and the float tensor paired with it, from its output's top level
    (as `_first_tensor` reads it); None when there is no map. The author's decision on F1.

    The map is the first bool tensor whose last dimension is `n_experts` and whose other dimensions
    multiply to the `tokens` the router was called on (not checked where the count is symbolic, as
    the integer branch of `_router_hook` does not), so a mask over the experts alone is not one.
    Megatron's `TopKRouter` returns `(probs, routing_map)`, both `[tokens, n_experts]`.

    Only an output that also holds a floating-point tensor is read this way, so that only an output
    of a float tensor and a bool map is counted differently from before: a bool tensor alone is
    refused, as it was. The paired float is the first floating-point tensor of exactly the map's
    shape; with none of that shape, every true entry of the map is counted. Where a router
    returns more than one (logits, then probabilities, then the map), the first is paired: logits
    are never exactly zero, so padding in the map (`pad_to_capacity`) would then be counted. Judged
    from types and shapes alone, so it traces the same under `torch.compile`."""
    if isinstance(output, torch.Tensor):
        candidates: Any = (output,)
    elif isinstance(output, (tuple, list)):
        candidates = output
    elif isinstance(output, dict):
        candidates = list(output.values())
    else:
        return None
    found = None
    for item in candidates:
        if isinstance(item, torch.Tensor) and item.dtype == torch.bool and item.ndim >= 1 and item.shape[-1] == n_experts:
            rows = item.numel() // n_experts if n_experts > 0 else None
            if tokens and type(rows) is int and rows != tokens:
                continue
            found = item
            break
    if found is None or not any(isinstance(item, torch.Tensor) and item.is_floating_point() for item in candidates):
        return None
    paired = next(
        (item for item in candidates
         if isinstance(item, torch.Tensor) and item.is_floating_point() and item.shape == found.shape),
        None,
    )
    return found, paired


def _output_unread(x: torch.Tensor, device: torch.device) -> str | None:
    """Why a watched block's output is not read into the layer statistics kept on `device`, from
    its type, layout and device alone: no data, so no sync, and the same under `torch.compile`,
    where an op that raises fails the trace before any `except` could see it. None when it is read.

    Another device is read when the four numbers can be moved to `device` without a wait the hook
    did not already make: between two GPUs, a device-to-device copy that PyTorch orders with
    stream events (`copy_device_to_device` in `aten/src/ATen/native/cuda/Copy.cu`, read at v2.6.0,
    not run here: no GPU), so the host does not wait; from the CPU, one copy to the device, where
    the hook before this made three (`torch.where` copied each CPU number to the device). An output
    on a device whose numbers would have to be read back to the host, an MPS or CUDA output with
    the statistics on the CPU (a model split between CPU and MPS; by reading, not run, also FSDP's
    CPU offload, whose first parameter is on the CPU), is not read: that copy waits for the device
    at every forward."""
    if x.is_nested:
        return "its output is a nested tensor"
    if x.layout != torch.strided:
        return f"its output is a {str(x.layout).removeprefix('torch.')} tensor, not a dense one"
    if x.device == device or x.device.type == "cpu" or x.device.type == device.type == "cuda":
        return None
    if x.device.type == "meta":
        return "its output is on meta, which holds no values"
    return (
        f"its output is on {x.device}, not on {device} where the layer statistics are kept, and moving its "
        "statistics there would make the host wait for a device at every forward"
    )


def _stats_hook(
    buf: torch.Tensor,
    stat_idx: torch.Tensor,
    row: int,
    phase: _Evaluating | None = None,
    problems: list[str] | None = None,
) -> Callable[..., None]:
    """The forward hook that writes a watched block's output statistics into row `row` of `buf`.
    Nothing in it raises into training. An output it cannot read leaves no entry for the block
    in that step's `layers` and is said once in `problems` (the hook's `problems` attribute when
    none is given): judged first from type, layout and device alone (`_output_unread`), so the
    same under `torch.compile`, and in eager mode from any exception left, which is caught. Under
    `torch.compile` an output that is a tensor subclass other than `nn.Parameter` is not read
    (`_compiled_subclass`); a DTensor is unwrapped to its local tensor first.

    A DTensor output (tensor parallelism with `use_local_output=False`) is read on its local
    tensor: under Replicate that is the output, under Shard it is this rank's part, read as
    `grad_norms` and `weight_norms` already are for FSDP2's parameters. Under Partial each rank
    holds an addend of the output, so no statistic of it would be true, and it is not read.

    Not wired to the records yet: saying `problems` in a record needs a field of its own
    (`layers_not_read`, as `moe_refused` is for routers), which is a schema change left for a
    decision."""
    phase = phase or _Evaluating()
    problems = [] if problems is None else problems

    def refuse(why: Callable[[], str]) -> None:
        # Built only the first time; the hook still runs on every forward.
        if not problems:
            problems.append(why())

    def read(output: Any) -> None:
        x = _first_tensor(output, integer=False)
        if x is None:
            return
        if _DTensor is not None and isinstance(x, _DTensor):
            if any(placement.is_partial() for placement in x.placements):
                refuse(lambda: (
                    "its output is a DTensor with a Partial placement: each rank holds an addend of the output, "
                    "not its values"
                ))
                return
            x = x.detach().to_local()
        if _compiled_subclass(x):
            refuse(lambda: (
                f"its output is a {type(x).__name__}, a tensor subclass, which TensorQuill does not read under "
                "torch.compile"
            ))
            return
        wrong = _output_unread(x, buf.device)
        if wrong is not None:
            refuse(lambda: wrong)
            return
        if x.numel() == 0:
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
        # float8 has no `==` on every device and PyTorch release; its float32 copy holds the same zeros.
        zfrac = ((xf if x.dtype.itemsize == 1 else x) == 0).float().mean()
        if x.device != buf.device:
            # One copy of the four numbers to the buffer's device (see `_output_unread`).
            mean, std, peak, zfrac = torch.stack((mean, std, peak, zfrac)).to(buf.device).unbind()
        kept = _row_max_compiled(buf, row) if torch.compiler.is_compiling() else buf[row, 2]
        amax = torch.maximum(peak, kept)
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

    def hook(module: nn.Module, args: Any, output: Any) -> None:
        if _recomputing() or _evaluating(module, phase):
            return
        if torch.compiler.is_compiling():
            # No `try` in a traced hook: the checks in `read` are what keep it from raising there.
            read(output)
            return
        try:
            read(output)
        except Exception as error:  # noqa: BLE001
            # What the checks do not foresee leaves the block out as they do, and training goes on.
            refuse(lambda: f"reading its output raised {_short(error)}")

    hook.problems = problems  # type: ignore[attr-defined]
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
    elif fix == "subclass":
        fix = (
            f"pass routers={{{name!r}: tensorquill.Router({spec.n_experts}, {spec.top_k}, indices=...)}} with a "
            f"function that returns the chosen expert indices as a plain torch.Tensor, or {inner}"
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


_log = logging.getLogger("tensorquill")


def _warn(message: str, stacklevel: int = 2) -> None:
    """`warnings.warn(message, RuntimeWarning)` for code the training thread reaches, which must
    never raise into training. Under `python -W error` (or any filter that makes a
    `RuntimeWarning` an error) `warnings.warn` raises the warning instead: it is then logged
    on the `tensorquill` logger, which prints it to stderr when no handler is set, and the
    caller goes on. A filter that ignores the warning still silences it."""
    try:
        warnings.warn(message, RuntimeWarning, stacklevel=stacklevel + 1)
    except Exception:  # noqa: BLE001
        try:
            _log.warning("%s", message)
        except Exception:  # noqa: BLE001
            pass


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
    tensor. In eager mode only what raises is refused (and the hook's `try` catches the rest).
    Under `torch.compile` a tensor subclass other than `nn.Parameter` is refused by its type alone
    (`_compiled_subclass`), the well-behaved ones that AOTAutograd would have counted included:
    the author's decision on lane T2's item 3, since a raising op fails the trace there."""
    if _DTensor is not None and isinstance(t, _DTensor):
        return f"{what} is a DTensor, not a plain tensor", None
    if _compiled_subclass(t):
        return f"{what} is a {type(t).__name__}, a tensor subclass, which TensorQuill does not count under torch.compile", "subclass"
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
            # Read with grad off, on a detached view of the output, so the function builds no
            # autograd graph and saves nothing for backward, whether or not it detaches: under
            # non-reentrant checkpointing the recomputation, where this hook does not run, must
            # save the same tensors as the forward, or `backward()` raises. The model's own
            # output keeps its graph.
            try:
                with torch.no_grad():
                    idx = spec.indices(_detached(output))
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
            elif (found := _routing_map(output, spec.n_experts, _tokens(args))) is not None:
                # A bool routing map (Megatron's `routing_map`): each expert's count is the tokens the
                # map routes to it, so an assignment dropped at capacity is not counted, and a token
                # with fewer than top_k assignments adds only what it has. Entries whose paired
                # float is exactly zero are left out: the padding `pad_to_capacity` adds to the map,
                # and with it a real assignment whose weight is exactly zero (an underflow, say).
                routing_map, paired = found
                for tensor, what in ((routing_map, "its routing map"), (paired, "the float tensor paired with its routing map")):
                    wrong = None if tensor is None else _unusable(tensor, counts.device, what)
                    if wrong is not None:
                        refuse(lambda: wrong[0], wrong[1])
                        return
                kept = routing_map.detach()
                if paired is not None:
                    kept = kept & (paired.detach() != 0)
                hits = kept.reshape(-1, spec.n_experts).sum(0)
            else:
                logits = _first_tensor(output, integer=False)
                if logits is None:
                    # Nothing to read: counting nothing would write zeros that read as a router
                    # that routed no token. Only the top level is looked in (`_first_tensor`).
                    refuse(lambda: (
                        f"its output is a {str(output.dtype).removeprefix('torch.')} tensor, not expert indices or logits"
                        if isinstance(output, torch.Tensor)
                        else f"its output, a {type(output).__name__}, holds no integer or floating-point tensor "
                        f"at its top level"
                    ))
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
        if idx is not None:
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


# Where `torch.compiler.disable`'s wrapper runs (`torch/_dynamo/eval_frame.py`): its frames sit
# between a hook that runs eagerly under it and the code that called the hook (`_caller_frame`).
_DYNAMO_DIR = str(Path(torch._dynamo.__file__).parent)


def _queue_rings(every: int) -> int:
    """Rings of `every` steps the writer may fall behind: `max(QUEUE_MIN_RINGS, ceil(QUEUE_STEPS /
    every))`, so 400 at every=1, 8 at the default 50, 4 at 100 and 2 from every=200 up."""
    return max(QUEUE_MIN_RINGS, math.ceil(QUEUE_STEPS / every))


def _queue_blocks(every: int) -> int:
    """The writer queue's size in blocks: `_queue_rings` rings. A full ring is queued as two blocks,
    all but its newest step and then that step (`_hold_newest`), one when it is one step."""
    return _queue_rings(every) * (1 if every == 1 else 2)


def _noop_hook(*args: Any, **kwargs: Any) -> None:
    """What a stopped session's optimizer hooks become until `close()` removes them (`_stop`)."""
    return None


def _caller_frame(depth: int) -> Any:
    """The frame `depth` levels above the caller, not counting `torch.compiler.disable`'s own."""
    frame = sys._getframe(depth + 1)
    while frame is not None and frame.f_code.co_filename.startswith(_DYNAMO_DIR):
        frame = frame.f_back
    return frame


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
    if _clip_is_l2(args, kwargs):
        for session in list(_clip_sessions):
            if _clips_session(session, parameters):
                session.record_grad_norm(total_norm)
    return total_norm


def _clip_is_l2(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Whether a `clip_grad_norm_(parameters, max_norm, norm_type=2.0, ...)` call computes an
    L2 norm, the only kind recorded. Torch's and Accelerate's both take `norm_type` after
    `max_norm`, and torch reads it with `float()`, so `2`, `2.0` and `"2"` are L2. Another
    norm is not recorded: `record_grad_norm` adds up the squares of a step's norms into one
    L2 total, which means nothing for an L1 or max norm."""
    norm_type = args[1] if len(args) > 1 else kwargs.get("norm_type", 2.0)
    try:
        return float(norm_type) == 2.0
    except (TypeError, ValueError):
        return False


def _clips_session(session: "Session", parameters: Any) -> bool:
    """Whether the parameters given to `clip_grad_norm_` (a list, or one tensor) include any
    of `session`'s trainable ones. Not the first alone: in a LoRA or PEFT model the first is a
    frozen base weight, and the Hugging Face `Trainer` clips `model.parameters()`. The walk
    stops at the first trainable parameter, a few in for such a model."""
    if isinstance(parameters, torch.Tensor):
        return id(parameters) in session._param_ids
    return not session._param_ids.isdisjoint(map(id, parameters))


def _watch_clipping(session: "Session") -> None:
    """Record the norm `torch.nn.utils.clip_grad_norm_` returns while `session` is open.

    The norm is computed anyway when gradients are clipped, so recording it is free. A call
    counts for the session when any of its parameters is one of the session's trainable
    ones, so a model whose first parameter is frozen (LoRA, PEFT) is covered. Only an L2 norm
    (`norm_type` 2, the default) is recorded: a call with another norm type records nothing
    as `clip_grad_norm`, and a run with no recorded `clip_grad_norm` has its spikes judged on
    `grad_norm` where that is recorded, which is read after clipping, so a spike that
    clipping flattened can go unseen.
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


@dataclass(frozen=True)
class _Plan:
    """What `Session` checks and works out from its options before it changes anything
    (`_plan_session`)."""

    grad_every: int
    weight_every: int
    probe_fn: Callable[[nn.Module], Any] | None
    params: list[tuple[str, torch.Tensor]]
    replicas: list[tuple[str, torch.Tensor]]
    selected: list[tuple[str, nn.Module]]
    routers: dict[str, Router]
    probe_blocks: list[tuple[str, nn.Module]]
    head: tuple[str | None, str | None, str | None]
    scaler: Any


def _plan_session(
    model: nn.Module,
    *,
    every: int,
    layers: LayerFilter | Sequence[str] | str | None,
    routers: Mapping[str, Router | int] | None,
    grads: bool | int,
    weights: bool | int,
    replica_every: int,
    probe: Any,
    probe_every: int,
    probe_head: str | bool | None,
    replica_params: Any,
    scaler: Any,
) -> _Plan:
    """Every check `init` and `Session` make of the model and the options, and what they work
    out from them, with nothing changed: no hook, thread, file or global. `init` runs it before
    it resolves the run or opens anything, so an option it rejects leaves everything as it was.
    A layer filter, a `replica_params` filter and `get_output_embeddings` are each called once."""
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
    if isinstance(weights, bool):
        weight_every = WEIGHT_EVERY if weights else 0
    elif isinstance(weights, int) and weights >= 0:
        weight_every = weights
    else:
        raise ValueError("weights must be a bool or a non-negative integer")
    if isinstance(replica_every, bool) or not isinstance(replica_every, int) or replica_every < 1:
        raise ValueError("replica_every must be a positive integer")
    if isinstance(probe_every, bool) or not isinstance(probe_every, int) or probe_every < 1:
        raise ValueError("probe_every must be a positive integer")
    probe_fn = probe_callable(probe) if probe is not None else None
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    # The parameters the cross-GPU drift check compares: every trainable parameter by
    # default (data parallelism), or only the replicated ones under tensor parallelism.
    replicas = _select_replica_params(params, replica_params)
    selected = _select_layers(model, layers)
    modules = dict(model.named_modules())
    planned_routers = {
        name: spec if isinstance(spec, Router) else Router(int(spec))
        for name, spec in (routers or {}).items()
    }
    unknown = [name for name in planned_routers if name not in modules]
    if unknown:
        raise ValueError(f"no such router modules: {unknown}")
    probe_blocks: list[tuple[str, nn.Module]] = []
    head: tuple[str | None, str | None, str | None] = (None, None, None)
    if probe_fn is not None:
        probe_blocks = _select_layers(model, "blocks")
        head = find_head(model, probe_head)
    return _Plan(
        grad_every=grad_every, weight_every=weight_every, probe_fn=probe_fn, params=params, replicas=replicas,
        selected=selected, routers=planned_routers, probe_blocks=probe_blocks, head=head,
        scaler=scaler if scaler is not None and scaler.is_enabled() else None,
    )


def _undo(undo: list[Callable[[], Any]]) -> None:
    """Run the undo steps registered so far, newest first. Each is tried on its own, so one that
    fails leaves the others to run; the error that started the undoing is the one raised."""
    while undo:
        try:
            undo.pop()()
        except Exception:  # noqa: BLE001
            pass


class _ScalerWatch:
    """The `update` wrapper TensorQuill installs on one `GradScaler`, and its state: the sessions
    that watch the scaler, in the order they began, and what `scaler.update` was before (an
    instance attribute someone else installed, or the class's method)."""

    __slots__ = ("sessions", "original", "had_attribute", "wrapper", "__weakref__")

    def __init__(self, original: Callable[..., Any], had_attribute: bool) -> None:
        self.sessions: list[Session] = []
        self.original = original
        self.had_attribute = had_attribute
        self.wrapper: Callable[..., Any] | None = None


def _scaler_watch(scaler: Any) -> _ScalerWatch | None:
    """The TensorQuill wrapper that is `scaler.update` now, if one is."""
    watch = getattr(vars(scaler).get("update"), "_tensorquill_watch", None)
    return watch if isinstance(watch, _ScalerWatch) else None


def _install_scaler_watch(scaler: Any) -> _ScalerWatch:
    had_attribute = "update" in vars(scaler)
    watch = _ScalerWatch(scaler.update, had_attribute)

    def update(*args: Any, **kwargs: Any) -> Any:
        sessions = list(watch.sessions)
        ends = [session._scaler_before_update(scaler) for session in sessions]
        out = watch.original(*args, **kwargs)
        for session, end in zip(sessions, ends):
            session._scaler_after_update(end)
        return out

    update = torch.compiler.disable(update)  # eagerly, as the optimizer hooks run (`Session._install`)
    update._tensorquill_watch = watch  # type: ignore[attr-defined]
    watch.wrapper = update
    scaler.update = update  # an instance attribute; the last session to close takes it out
    return watch


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
        probe_head: str | bool | None = None,
        _plan: _Plan | None = None,
    ) -> None:
        # Every check first, with nothing changed (`_plan_session`); `init` has made them already.
        plan = _plan if _plan is not None else _plan_session(
            model, every=every, layers=layers, routers=routers, grads=grads, weights=weights,
            replica_every=replica_every, probe=probe, probe_every=probe_every, probe_head=probe_head,
            replica_params=replica_params, scaler=scaler,
        )
        grad_every, weight_every = plan.grad_every, plan.weight_every
        self.run = run
        # When this session started, in Unix milliseconds. A job that resumes from an
        # earlier checkpoint under the same run ID writes the same step numbers again;
        # the attempt tells those records from the ones they replace.
        self.attempt = _next_attempt()
        self.every = every
        self.grad_every = grad_every
        self.weight_every = weight_every
        self.weights = weight_every > 0
        self.replica_every = replica_every
        self._replica_projection = replica_projection
        self.probe_every = probe_every
        self._probe_fn = plan.probe_fn
        self._model = model
        self.auto_step = auto_step
        self._sink = sink
        self._replica_group = replica_group

        params = plan.params
        self._params = [p for _, p in params]
        self._param_ids = {id(p) for p in self._params}
        self._param_index = {id(p): index for index, p in enumerate(self._params)}
        self.param_names = [name for name, _ in params]
        # The parameters the cross-GPU drift check compares (`_plan_session`).
        replicas = plan.replicas
        self._replica_params = [p for _, p in replicas]
        self.replica_names = [name for name, _ in replicas]
        self.device = _buffers_device(self._params)

        selected = plan.selected
        self.layer_names = [name for name, _ in selected]
        modules = dict(model.named_modules())
        self._routers = dict(plan.routers)
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
        # Replicated parameters the cross-GPU check could not read on this rank, with why
        # (`record_replica_spread`), and those already named in a warning (`_say_replica_unread`).
        self._replica_unread_here: dict[str, str] = {}
        self._replica_unread_said: set[str] = set()
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
        # Parameters whose gradient could not be read (`record_grads`), with why, and those already
        # said under `grads_not_read` on a record.
        self._grad_why: dict[str, str] = {}
        self._grad_why_written: set[str] = set()
        for buf in (self._act, self._stat_idx, *self._counts_list, *self._expert_ids.values()):
            _mark_static(buf)

        # Forwards that are not training (evaluation, the probe) are left out by the hooks
        # (`_evaluating`); this says whether the session's phase is "eval" (`set_phase`).
        self._evaluating = _Evaluating()
        self._handles: list[Any] = []  # the forward hooks, registered by `_install`
        self._probe_state = ProbeState()
        if self._probe_fn is not None:
            state = self._probe_state
            state.head_name, state.head_found_by, state.head_note = plan.head
            state.head_off = probe_head is False

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
        # Step accounting under a loss scaler (`_commit`, `_watch_scaler`): whether an optimizer
        # that holds this model's parameters, or one that holds none of them, stepped since the
        # last step ended (on a step the scaler skipped, neither did); a weight check that fell
        # on a skipped step, owed to the next step that was not; and for each row of the ring,
        # whether its step ended with no optimizer of this model seen stepping yet, which
        # `flush` says as `update_not_read` only for a row the scaler did not skip.
        self._stepped = False
        self._foreign_stepped = False
        self._weight_check_owed = False
        self._ring_no_optimizer: list[bool] = []
        # Several optimizers on one model, stepped one after the other (`_after_optimizer_step`):
        # whether one has ever stepped after the lead ended the step (a follower); whether the
        # newest step had a weight check, and the weights that check compared with, so that a
        # follower reads it again after its own step; and the newest step's probe, owed until
        # every optimizer of its iteration has stepped.
        self._followers = False
        self._recheck = False
        self._weights_before: torch.Tensor | None = None
        self._not_exact_before: Any = None
        self._probe_owed = False
        # The newest step of a full ring, kept back when the others were written (`_commit`,
        # `_hold_newest`) until the next step ends or the ring is flushed (`_release_held`): its
        # rows already on the CPU, and which of its device rows changed since (a follower's
        # update, its weight check read again, a follower's loss).
        self._held: dict[str, Any] | None = None
        self._step_depth = 0
        self._outer_frame: Any = None
        self._closed = False
        self._phase = "train"
        self._ticks = 0
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
        # Bounded, as TensorBoard's SummaryWriter is (`max_queue=10`): if the sink falls a full
        # queue behind, for example on a stalled network filesystem, capture stops for the
        # session (`_stop`) instead of holding every unwritten step in memory or making training
        # wait (the author's decision on C03).
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=_queue_blocks(every))
        # Capture stopped for the rest of the session (`_stop`), and why: the sink's exception, or
        # the writer falling a full queue behind. Training goes on either way.
        self._stopped = False
        self.writer_error: BaseException | None = None
        # While `close()` flushes, the time by which `_put` may wait for room (`close`); None while
        # training, when `_put` never waits.
        self._close_by: float | None = None
        self._worker = threading.Thread(target=self._drain, name="tensorquill-writer", daemon=True)
        self._clip_watched = clip_norm
        self._scaler = plan.scaler
        # With a GradScaler, `scaler.update()` runs once per iteration whether or not the
        # optimizer stepped, so it ends the TensorQuill step; optimizer hooks still read
        # gradients. Without it, a step the scaler skipped would never be counted.
        self._commit_on_update = self._scaler is not None and auto_step
        self._scaler_watch_of: _ScalerWatch | None = None
        self._optimizer_handles: list[Any] = []

        # Everything above changed nothing outside this object. What follows does, and is undone
        # if a later part of it fails, so a session that is not made leaves nothing behind.
        undo: list[Callable[[], Any]] = []
        try:
            self._install(undo, plan=plan, modules=modules, clip_norm=clip_norm, auto_step=auto_step,
                          data_wait=data_wait)
        except BaseException:
            _undo(undo)
            raise

    def _install(
        self, undo: list[Callable[[], Any]], *, plan: _Plan, modules: Mapping[str, nn.Module], clip_norm: bool,
        auto_step: bool, data_wait: bool,
    ) -> None:
        """What the session changes outside itself, each registered in `undo` as it is made: the
        forward hooks, the heartbeat's step counter, the writer thread, the clipping, scaler and
        optimizer hooks and the data-loader timer; last, the end marker of an earlier run is
        removed, the one effect that is not undone, so it happens only once all else has."""
        undo.append(lambda: [handle.remove() for handle in self._handles])
        self._handles += [
            module.register_forward_hook(_stats_hook(self._act, self._stat_idx, row, self._evaluating))
            for row, (_, module) in enumerate(plan.selected)
        ]
        self._handles += [
            modules[name].register_forward_hook(
                _router_hook(
                    self._counts[name], self._expert_ids[name], spec, self._router_problems[name], self._evaluating
                )
            )
            for name, spec in self._routers.items()
        ]
        if self._probe_fn is not None:
            self._handles += install_probe_hooks(plan.probe_blocks, self._probe_state)
            head = plan.head[0]
            if head is not None:
                self._handles += install_head_hook(modules[head], self._probe_state)

        beat = getattr(self._sink, "heartbeat", None)
        if beat is not None:
            had, before = "counter" in vars(beat), vars(beat).get("counter")
            beat.counter = lambda: self._steps_committed + self._ticks
            undo.append(lambda: setattr(beat, "counter", before) if had else vars(beat).pop("counter", None))

        def stop_writer() -> None:
            self._queue.put(None)
            self._worker.join()

        self._worker.start()
        undo.append(stop_writer)
        if clip_norm:
            _watch_clipping(self)
            undo.append(lambda: _unwatch_clipping(self))
        if self._scaler is not None:
            self._watch_scaler(self._scaler)
            undo.append(self._unwatch_scaler)
        if auto_step:
            # Run eagerly where `optimizer.step()` is called inside a function `torch.compile` traces
            # (`init` there too, say): traced, these hooks would make Dynamo warn on what it cannot
            # trace, which `python -W error` raises into the step. Outside tracing the wrapper only
            # adds a call.
            for register, hook in (
                (register_optimizer_step_pre_hook, torch.compiler.disable(self._before_optimizer_step)),
                (register_optimizer_step_post_hook, torch.compiler.disable(self._after_optimizer_step)),
            ):
                handle = register(hook)
                self._optimizer_handles.append(handle)
                undo.append(handle.remove)
        if data_wait:
            self.watch_loaders()
            undo.append(lambda: datawait.unwatch(self))
        path = getattr(self._sink, "path", None)
        if path is not None:
            clear_end_marker(path)  # the file is being written again (see `sink.write_end_marker`)

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

    def _grads_wanted(self) -> bool:
        # A weight check owed by a step the scaler skipped needs this step's gradient norms too.
        return self._should_record_grads() or self._weight_check_owed

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
            _warn(
                f"TensorQuill has recorded no steps: {FOREIGN_STEPS_WARNING} optimizer steps ran, none on this "
                "model's parameters. If your framework steps copies of the weights (DeepSpeed ZeRO, "
                "Megatron-LM), pass auto_step=False and call session.step() after each update.",
                stacklevel=2,
            )

    def _before_optimizer_step(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        if self._closed:
            return
        if not self._owns(optimizer):
            self._foreign_stepped = True
            if not self._stopped:
                self._note_foreign_step()
            return
        # PyTorch wraps each optimizer class's own `step` with the global hooks, so an
        # optimizer that subclasses another and calls `super().step()` fires them again,
        # nested inside the outer call. A call is nested only while the outer call's
        # wrapper frame is still running; after an outer step raised, it is not, and the
        # next call starts afresh instead of being ignored forever.
        wrapper = _caller_frame(1)
        if self._step_depth and _on_stack(self._outer_frame, wrapper.f_back):
            self._step_depth += 1
            return
        self._step_depth, self._outer_frame = 1, wrapper
        # One TensorQuill step per training iteration. With several optimizers on one
        # model, the first one that ever stepped leads: its steps end TensorQuill steps,
        # and the others' steps add none and record nothing twice; what they change is
        # written on the step the lead ended (`_after_optimizer_step`), which assumes they
        # step after it in each iteration. Gradients are read before the lead steps, after
        # backward, so they include every optimizer's parameters.
        lead = self._lead() if self._lead is not None else None
        if lead is None:
            self._lead, lead = weakref.ref(optimizer), optimizer
        self._leading = optimizer is lead
        if self._stopped:
            return  # only the step count goes on, for the drift check's schedule (`_count_stopped`)
        if self._leading and self._probe_owed:
            # Every optimizer of the previous iteration has stepped: its probe runs now.
            self._run_owed_probe()
        if self._leading and self._snapshot_due:
            self._snapshot_before = optimizer_entries(optimizer)
        # With `step(closure)`, the closure runs forward and backward inside the step, so
        # gradients are read after it instead (Lightning steps this way).
        closure = kwargs.get("closure", args[1] if len(args) > 1 else None)
        self._grads_after_step = callable(closure)
        if self._leading and not self._grads_after_step and not self._grads_recorded and self._grads_wanted():
            self.record_grads()

    def _after_optimizer_step(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        if self._closed or not self._owns(optimizer):
            return
        if self._step_depth > 1:
            self._step_depth -= 1
            return
        self._step_depth, self._outer_frame = 0, None
        self._stepped = True
        if self._stopped:
            if self._leading and not self._commit_on_update:
                self._count_stopped(None)
            return
        # An optimizer that steps after the lead ended the step (two optimizers on one model,
        # `opt1.step(); opt2.step()`: Muon and AdamW, a GAN's two halves in one module) belongs to
        # that step, which is still the newest in the ring (when it filled the ring, `_commit` kept
        # it back): what it changes is written on that step's record, not the next one's. Under a
        # loss scaler, `scaler.update()` ends the step after every optimizer has stepped, so
        # nothing follows there.
        follows = not self._leading and not self._commit_on_update and bool(self._steps)
        if self.updates:
            # Every optimizer that holds some of this model's parameters, not only the lead.
            if follows and not self._update_recorded:
                self._read_late_updates(optimizer)
            else:
                self._read_updates(optimizer)
        if follows:
            self._follow()
        if not self._leading:
            return
        if self._grads_after_step and not self._grads_recorded and self._grads_wanted():
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

    @torch.no_grad()
    def _read_late_updates(self, optimizer: Any) -> None:
        """`_read_updates` for an optimizer that stepped after the lead ended the step: merged
        into the newest row of the ring parameter by parameter, so the lead's entries stay."""
        self._read_updates(optimizer)
        if not self._update_recorded:
            return
        target = self._ring_update[len(self._steps) - 1]
        if not self._ring_update_seen[-1]:
            target.zero_()  # nothing was read into this row yet: it still holds an older step's values
        target.copy_(torch.where(self._update[:, 1:] > 0, self._update, target))
        self._update.zero_()
        self._update_recorded = False
        self._ring_update_seen[-1] = True
        self._mark_held("update")

    @torch.no_grad()
    def _follow(self) -> None:
        """After an optimizer that stepped later than the lead in the same iteration."""
        first = not self._followers
        self._followers = True
        if self._recheck:
            # The newest step's weight check is read again now that this optimizer has moved its
            # parameters, against the same earlier check; and which parameters an optimizer that
            # has stepped holds is judged again, now that this one has (at the first step it had not).
            # One more read of the weights per follower, on weight-check steps only.
            self._prev_weight = None if self._weights_before is None else self._weights_before.clone()
            if self._not_exact_before is not None:
                self._weights_not_exact = self._not_exact_before
            self._record_weights()
            self._ring_grad[len(self._steps) - 1, :, 2:].copy_(self._grad[:, 2:])
            self._grad[:, 2:].zero_()
            self._mark_held("grad")
            self._ring_weight_seen[-1] = {"ownership": self._optimizer_ownership(), "dtypes": self._weight_dtypes}
        if first and self._ring_probe[-1] is not None:
            # The probe ran when the lead ended the step, before any follower was known: it runs
            # again once every optimizer of this iteration has stepped. Later probes wait (`_commit`).
            self._probe_owed = True

    def _run_owed_probe(self) -> None:
        """The newest step's probe, held until every optimizer of its iteration stepped: run at the
        lead's next step, or when the next step ends, the ring is flushed or the session closes."""
        self._probe_owed = False
        if not self._steps:
            return
        # The next iteration's forward may have begun filling the buffers that the probe clears
        # after it runs: they are put back as they were, versions included (see `_commit`).
        buffers = [b for b in (self._act, *self._counts_list) if b.numel() > 0]
        versions = [b._version for b in buffers]
        kept = [b.clone() for b in buffers]
        try:
            self._ring_probe[-1] = self._probe()
        finally:
            with torch.no_grad():
                for buf, value in zip(buffers, kept):
                    buf.copy_(value)
            _restore_versions(buffers, versions)

    def _follows(self, optimizer: Any) -> bool:
        """Whether `optimizer` steps after the lead ended the step, so what it brings belongs to
        the newest step in the ring (for the Lightning callback's loss)."""
        lead = self._lead() if self._lead is not None else None
        return (
            lead is not None and optimizer is not lead and self.auto_step and not self._commit_on_update
            and bool(self._steps) and not self._closed and not self._stopped and self._owns(optimizer)
        )

    def _add_to_newest(self, name: str, value: Any) -> None:
        """Add `value` to the scalar `name` of the newest step in the ring (or set it there): the
        loss back-propagated for an optimizer that steps after the lead ended the step."""
        pending, self._pending_scalars = self._pending_scalars, {}
        try:
            self.log({name: value})  # the same checks and the same copy on the device as `log`
            new = self._pending_scalars[name]
        finally:
            self._pending_scalars = pending
        row = dict(self._ring_scalars[-1])
        row[name] = new if row.get(name) is None else row[name] + new
        self._ring_scalars[-1] = row
        self._mark_held("scalars")

    def _watch_scaler(self, scaler: Any) -> None:
        """Have `scaler.update()` end this session's steps (`_scaler_update`), by an instance
        attribute on the scaler: one wrapper per scaler, shared by every session that watches it,
        installed by the first and taken out when the last closes (`_unwatch_scaler`)."""
        watch = _scaler_watch(scaler)
        if watch is None:
            watch = _install_scaler_watch(scaler)
        watch.sessions.append(self)
        self._scaler_watch_of = watch

    def _unwatch_scaler(self) -> None:
        """Leave the scaler's shared wrapper, and when no session is left on it put the scaler back
        as it was before the first one: the instance attribute someone else had installed, or
        none, so the class's own method runs again. A wrapper someone installed on top of ours
        after `init` is left alone, with ours under it passing every call through."""
        scaler = self._scaler
        watch = self._scaler_watch_of
        self._scaler_watch_of = None
        if watch is None:
            return
        if self in watch.sessions:
            watch.sessions.remove(self)
        if watch.sessions:
            return
        if vars(scaler).get("update") is watch.wrapper:
            if watch.had_attribute:
                scaler.update = watch.original
            else:
                del scaler.update

    def _scaler_before_update(self, scaler: Any) -> bool:
        """Before the scaler's own `update()`: whether it ends this session's step."""
        if self._closed:
            return False
        if self._stopped:
            # Ends the step as a live session's does (an optimizer went through the scaler, a step it
            # skipped included, or one of this model's stepped), so a stopped rank's drift-check
            # gather stays on the other ranks' steps (`_count_stopped`); read without device work.
            return self._stepped or any(
                state.get("found_inf_per_device") for state in getattr(scaler, "_per_optimizer_states", {}).values()
            )
        # It ends a step only after an optimizer went through the scaler (or one of this
        # model's stepped): `update(new_scale)` called on its own only sets the scale.
        ends = self._record_scaler(scaler) or self._stepped
        if ends and self._commit_on_update and not self._grads_recorded and self._grads_wanted():
            # A step the scaler skipped never reaches the optimizer hooks. Its gradients are
            # still there, unscaled by `scaler.step`, Inf or NaN where they overflowed:
            # read them here, so the record names the parameters that did.
            self.record_grads()
        return ends

    def _scaler_after_update(self, ends: bool) -> None:
        if ends and self._commit_on_update and not self._closed and self._stopped:
            self._count_stopped(None)
            return
        if ends and self._commit_on_update and not self._closed:
            if self._pending_lr is None and self._lead is not None and self._lead() is not None:
                self._pending_lr = _learning_rates(self._lead())
            self._commit(None)

    def _record_scaler(self, scaler: Any) -> bool:
        # Read before `update()` resets them: whether any optimizer's gradients had an Inf
        # or NaN this iteration (the scaler then skipped its step), and the scale used.
        # Both stay tensors on the device until the next flush, so no sync is added.
        # Returns whether any optimizer went through the scaler since its last update; with
        # none, there is nothing to say of a step and nothing is recorded.
        scale = getattr(scaler, "_scale", None)
        if scale is None:
            return False
        found = [
            f for state in scaler._per_optimizer_states.values()
            for f in state["found_inf_per_device"].values()
        ]
        if not found:
            return False
        skipped = torch.stack([f.to(scale.device).reshape(()) for f in found]).sum() > 0
        self._pending_amp = {"step_skipped": skipped.float(), "grad_scale": scale.detach().clone()}
        return True

    @torch.no_grad()
    def record_grads(self) -> None:
        """Record per-parameter gradient norms now.

        With `auto_step` this runs just before `optimizer.step()`, which is after any
        gradient clipping. Call it yourself before clipping to record unclipped norms.

        Nothing here raises into training. A gradient that cannot be read is left out on its
        own, with no norm in `grad_norms` (so none in `grad_norm` either), and the others keep
        theirs: one on `meta`, which holds no values, or nested, judged from its device and
        layout alone; or one whose norm raised (float8 on the CPU, say), found by reading the
        rest of its group one at a time. Which parameters, and why, is said once per parameter
        under `grads_not_read` on the first record written after it was found. A sparse
        gradient is read from its stored values, as the weight check reads a sparse weight.
        """
        if self._closed_call("record_grads()"):
            return
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
            try:
                if type(grad) is not torch.Tensor:
                    grad = unwrap_local(grad)
                why = _grad_unread(grad)
                if why is not None:
                    self._grad_not_read(index, why)
                    continue
                if grad.layout == torch.sparse_coo:
                    grad = grad.coalesce().values()
                elif grad.layout != torch.strided:
                    grad = grad.values()  # CSR, CSC, BSR, BSC: the stored values are all of them
            except Exception as error:  # noqa: BLE001
                self._grad_not_read(index, f"reading it raised {_short(error)}")
                continue
            if grad.numel() == 0:
                continue
            groups.setdefault((grad.device, grad.dtype), []).append((index, grad))

        for (_, dtype), items in groups.items():
            out = torch.float32 if dtype in _LOW_PRECISION else None
            try:
                stacked = torch.stack(_foreach_norm([grad for _, grad in items], out)).to(self._grad)
            except Exception:  # noqa: BLE001
                # One at a time, so only a gradient whose norm raises is left out.
                read: list[tuple[int, torch.Tensor]] = []
                for index, grad in items:
                    try:
                        read.append((index, torch.linalg.vector_norm(grad, 2, dtype=out).reshape(()).to(self._grad)))
                    except Exception as error:  # noqa: BLE001
                        self._grad_not_read(index, f"reading it raised {_short(error)}")
                if not read:
                    continue
                items, stacked = read, torch.stack([norm for _, norm in read])
            if len(items) == n_params:
                self._grad[:, 0].copy_(stacked)
                self._grad[:, 1].fill_(1.0)
            else:
                idx = self._index(tuple(index for index, _ in items))
                self._grad[:, :2].index_copy_(
                    0, idx, torch.stack((stacked, torch.ones_like(stacked)), dim=1)
                )
        self._grads_recorded = True

    def _grad_not_read(self, index: int, why: str) -> None:
        """Keep why parameter `index`'s gradient was not read, for `grads_not_read` (`_flush`)."""
        self._grad_why.setdefault(self.param_names[index], why)

    def _new_grad_why(self) -> dict[str, str]:
        """The parameters whose gradients were found unreadable since the last record that said so."""
        found = {name: why for name, why in self._grad_why.items() if name not in self._grad_why_written}
        self._grad_why_written.update(found)
        return found

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
        # One read of the weights, no copy of them. A complex parameter is read as its real
        # pairs (`torch.view_as_real`, a view), whose norm is |z|'s, as `_read_updates` reads
        # Adam's moments and `distributed.py` its fingerprints. Nothing here raises into training,
        # and no norm is made up: a parameter with no values (on `meta`), or whose norm raised, gets
        # NaN and "not known" for unchanged; one read only in float32 where float64 was wanted
        # (on MPS beside buffers elsewhere) gets its norm and "not known".
        n_params = len(self._params)
        dtype = torch.float64 if self._exact_weights else torch.float32
        current = torch.zeros(n_params, dtype=dtype, device=self._grad.device)
        groups: dict[tuple[torch.device, torch.dtype], list[tuple[int, torch.Tensor]]] = {}
        dtypes: dict[str, int] = {}
        unread: list[int] = []  # no norm: NaN
        inexact: list[int] = []  # a norm, but not one to compare exactly
        for index, param in enumerate(self._params):
            data = param.detach()
            if type(data) is not torch.Tensor:
                data = unwrap_local(data)
            name = str(data.dtype).removeprefix("torch.")
            dtypes[name] = dtypes.get(name, 0) + 1
            if data.device.type == "meta":
                unread.append(index)
                continue
            if data.layout == torch.sparse_coo:
                data = data.coalesce().values()
            elif data.layout != torch.strided and not data.is_nested:
                data = data.values()  # CSR, CSC, BSR, BSC: the stored values are all of them
            if data.numel() == 0:
                continue
            if data.is_complex():
                data = torch.view_as_real(data.resolve_conj())
            groups.setdefault((data.device, data.dtype), []).append((index, data))
        # How many trainable parameters have each dtype: a weight that never changes means
        # something else in bf16 or fp16, where a small update rounds away, than in float32.
        # A complex parameter is counted under its own dtype.
        self._weight_dtypes = dtypes
        for (device, param_dtype), items in groups.items():
            exact = self._exact_weights and device.type != "mps"  # MPS has no float64
            out = dtype if exact else (torch.float32 if param_dtype in _LOW_PRECISION else None)
            indices = tuple(i for i, _ in items)
            try:
                norms = torch.stack(_foreach_norm([data for _, data in items], out))
                if norms.device != current.device:
                    # Move, then widen: MPS cannot hold the float64 `current` may be.
                    norms = norms.to(current.device)
                norms = norms.to(current)
            except Exception:  # noqa: BLE001
                unread.extend(indices)
                continue
            if not exact:
                inexact.extend(indices)
            if len(items) == n_params:
                current.copy_(norms)
            else:
                current.index_copy_(0, self._index(indices), norms)
        if unread:
            current.index_fill_(0, self._index(tuple(sorted(unread))), math.nan)
        self._grad[:, 2].copy_(current)
        # Not known where this check or the one before did not read a parameter exactly.
        blind = set(unread) | set(inexact)
        before, self._weights_not_exact = getattr(self, "_weights_not_exact", frozenset()), frozenset(blind)
        if not self._exact_weights or self._prev_weight is None:
            self._grad[:, 3].fill_(-1.0)
        else:
            self._grad[:, 3].copy_(current == self._prev_weight)
            if blind or before:
                self._grad[:, 3].index_fill_(0, self._index(tuple(sorted(blind | before))), -1.0)
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
        if self._closed_call("record_updates()"):
            return
        if not self.updates:
            raise RuntimeError("pass updates=True to tensorquill.init to record update sizes")
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
        #
        # Under FSDP2 the moments are DTensors and each rank reads its own shard (`unwrap_local`),
        # as `record_grads` and `_record_weights` do: a rank's `update_rms` for a parameter is the
        # RMS of the update over the rows that rank holds, not over the whole parameter (weighting
        # each rank's square by its element count gives the whole one's). The update-size rule
        # compares a rank's records only with that rank's own, against its total gradient norm,
        # which `grad_norm` also takes over its shards (what `clip_grad_norm` holds under FSDP2 was
        # not checked); a rank whose shard of a parameter is empty reads nothing for it.
        # A complex moment is read as its real pairs, as Adam steps it.
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
                    if type(first) is not torch.Tensor:
                        first = unwrap_local(first)
                    if type(moment) is not torch.Tensor:
                        moment = unwrap_local(moment)
                    if first.numel() == 0:
                        continue
                    if first.is_complex():
                        first = torch.view_as_real(first.resolve_conj())
                        moment = torch.view_as_real(moment.resolve_conj())
                    if first.dtype in _LOW_PRECISION:
                        first, moment = first.float(), moment.float()
                    steps = state["step"]
                    if isinstance(steps, torch.Tensor) and type(steps) is not torch.Tensor:
                        steps = unwrap_local(steps)
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
        if self._closed_call("record_grad_norm()"):
            return
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
        if self._closed_call("log()"):
            return
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
        if self._closed_call("data()"):
            return
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
        `DistributedSampler`, `RandomSampler` or `SequentialSampler`. On a closed session it
        returns `loader` itself, untouched.
        """
        if self._closed_call("watch_loader()"):
            return loader
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
        if self._closed_call("checkpoint()"):
            return
        self._record_manifest(checkpoint_manifest(state), event)

    def checkpoint_files(self, folder: str | Path, *, event: str = "save") -> None:
        """`checkpoint` for a checkpoint folder, as the Hugging Face `Trainer` and Accelerate's
        `save_state` write: the components are read from the file names (`optimizer.pt`,
        `scheduler.pt`, `scaler.pt`, `rng_state*.pth` or `random_states_*.pkl`, the model's
        weights, `trainer_state.json` or `sampler*.bin` for the data position). No file is opened."""
        if self._closed_call("checkpoint_files()"):
            return
        try:
            names = [entry.name for entry in Path(folder).iterdir()]
        except OSError:
            return
        self._record_manifest(files_manifest(names), event)

    def _record_manifest(self, manifest: dict[str, Any], event: str) -> None:
        if event not in ("save", "load"):
            raise ValueError('event must be "save" or "load"')
        last = self._next_step - 1 if self._steps_committed else None
        if event == "save":
            manifest["after_step"] = last
            # The step it follows is still in the ring, also when it filled the ring (`_commit`
            # keeps the newest step back until the next one ends): record it there. The ring is
            # empty only before the first step, or after an explicit `flush()` with no step since;
            # the save then waits for the next step's record (its `after_step` still names the
            # step it followed), and is not written if the session closes first: no record is
            # invented.
            if self._steps:
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
            # Each parameter's spread, then 0.0 where every replica read it, or the code of why
            # one did not (`distributed._UNREAD`), which leaves it out of the record (`_decode`).
            self._spread = torch.zeros(n_params, FINGERPRINT_WIDTH + 1, dtype=torch.float64, device=spread_device)
            self._ring_spread = torch.zeros(
                self.every, n_params, FINGERPRINT_WIDTH + 1, dtype=torch.float64, device=spread_device
            )
        return self._spread, self._ring_spread

    @torch.no_grad()
    def record_replica_spread(self, group_or_spread: Any = None) -> None:
        """Record cross-GPU parameter relative spread `(norm, sum, projection)` for this step.

        Pass a replication `ProcessGroup` (or rely on `replica_group` passed to `init`), or
        pass a precomputed `[n_params, 3]` (or older `[n_params, 2]`) tensor from
        `spread_from_fingerprints`.

        With a group, a replicated parameter that some replica cannot read (on `meta`, nested or
        sparse, or one whose reading raised) is still gathered, so every rank enters the same
        collective, and is then left out of `replica_spread` on every rank, never written as a
        spread: a warning names it once, at the next flush. A NaN weight that was read still
        reads as drift.
        """
        if self._closed or not self._replica_params:
            return
        target = self._replica_group if group_or_spread is None else group_or_spread
        if target is None:
            raise ValueError("pass a replication process group or precomputed spread tensor")
        if self._stopped:
            if not isinstance(target, torch.Tensor):
                self._gather_and_drop(target)  # the other ranks are entering it too
                self._spread_recorded = 1
            return
        spread_buf, _ = self._ensure_spread_buffers()
        unread: torch.Tensor | None = None
        if isinstance(target, torch.Tensor):
            spread = target.to(spread_buf)
        else:
            spread, unread, reasons = _spread_and_unread(
                self._replica_params, target, projection=self._replica_projection, device=spread_buf.device
            )
            for name, why in zip(self.replica_names, reasons):
                if why is not None:
                    self._replica_unread_here.setdefault(name, why)
        width = spread.shape[-1]
        spread_buf.zero_()
        spread_buf[:, :width].copy_(spread)
        if unread is not None:
            spread_buf[:, -1].copy_(unread)
        self._spread_recorded = width

    def _say_replica_unread(self, spreads: torch.Tensor, seen: Sequence[int]) -> None:
        """Name once each replicated parameter left out of the cross-GPU check, from the block's
        spreads, already on the CPU: no transfer and no sync is added. The same on every rank,
        since every rank leaves out the same rows; the reason is this rank's own when it could
        not read the parameter, else what another replica said."""
        codes = spreads[..., -1]
        for row, width in enumerate(seen):
            if not width or not bool(codes[row].any()):
                continue
            for name, code in zip(self.replica_names, codes[row].tolist()):
                if not code or name in self._replica_unread_said:
                    continue
                self._replica_unread_said.add(name)
                why = self._replica_unread_here.get(name) or f"on another replica in the group, {_unread_reason(code)}"
                _warn(
                    f"TensorQuill is not comparing the parameter {name!r} across replicas: {why}. It is left out "
                    "of replica_spread, so the cross-GPU drift check does not cover it.",
                    stacklevel=4,
                )

    def step(self, step: int | None = None, lr: float | Sequence[float] | None = None) -> None:
        """Mark the end of one optimizer step. Only for `auto_step=False`.

        Use this when the optimizer hooks would miscount, for example with
        `torch.amp.GradScaler`, which skips `optimizer.step()` on overflow. Pass `lr` (one
        value per parameter group, or a single value) to record the learning rate.
        """
        if self._closed_call("step()"):
            if self._stopped and not self._closed and not self.auto_step:
                self._count_stopped(step)  # the drift check's schedule goes on (`_count_stopped`)
            return
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
                _warn(
                    f"TensorQuill is not counting the router {name!r}: {reason}. Its expert loads are not recorded.",
                    stacklevel=stacklevel,
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
        if self._error is not None and not self._stopped:
            self._stop(self._error)  # the sink failed on the writer thread: said here, on this one
        if self._stopped:
            self._count_stopped(step)
            return
        current = self._next_step if step is None else step
        if self._probe_owed:
            self._run_owed_probe()  # the previous step's, before this one becomes the newest
        if self._held is not None:
            self._release_held()  # the step kept back when the ring last filled: this one ends it
            if self._stopped:  # the writer was a full queue behind (`_put`)
                self._count_stopped(step)
                return
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
        # A step `scaler.update()` ended with no optimizer stepping is one the scaler skipped: its
        # weights are the previous step's. Told on the host, with no sync. (The record's own
        # `step_skipped` is the scaler's verdict, read at the flush.)
        skipped = self._commit_on_update and not self._stepped and not self._foreign_stepped
        self._stepped = self._foreign_stepped = False
        # A weight check is not made on a skipped step, so that an unchanged weight always means
        # an optimizer step that left it alone: one due there is made on the next step instead.
        check_due = self._should_check_weights() or self._weight_check_owed
        weights_checked = check_due and self._grads_recorded and bool(self._params) and not skipped
        if weights_checked:
            self._weight_check_owed = False
            # What this check compares with, kept for an optimizer that steps after the lead to
            # read the check again (`_follow`). A copy of one number per parameter. Lane R1's
            # `_record_weights` also keeps which parameters the previous check did not read
            # exactly (`_weights_not_exact`); that is kept the same way where it exists.
            self._weights_before = None if self._prev_weight is None else self._prev_weight.clone()
            self._not_exact_before = getattr(self, "_weights_not_exact", None)
            self._record_weights()
        elif check_due and skipped:
            self._weight_check_owed = True
        self._recheck = weights_checked
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
            no_optimizer = False
            if self._update_recorded:
                self._ring_update[row].copy_(self._update)
                self._update.zero_()
            elif not self.auto_step:
                self._update_why.setdefault(
                    UPDATES_HOOKS_OFF,
                    "the optimizer hooks are off, so call session.record_updates(optimizer) after each optimizer.step()",
                )
            elif self._lead is None:
                # A step ended by `scaler.update()` with no optimizer of this model seen stepping:
                # one that steps copies of the weights, or a step the scaler skipped, where no
                # optimizer stepped at all. Which, `flush` tells from the record's `step_skipped`.
                no_optimizer = True
            self._ring_no_optimizer.append(no_optimizer)
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
            if self._followers:
                # Other optimizers step after the lead in this run: the probe waits until they
                # have (`_run_owed_probe`), so it reads the model as the whole step left it.
                self._probe_owed = True
            else:
                self._ring_probe[-1] = self._probe()
        if len(self._steps) == self.every:
            # A full ring is written at once but for its newest step, kept back until the next
            # step ends (or a flush, or close): what arrives for that step after it ended, a save
            # right after it (`_record_manifest`) or the update of an optimizer that steps after
            # the lead (`_after_optimizer_step`), is written on its own record. A run that hangs
            # or is killed in the next step has one step fewer on disk, never a block.
            self._flush(hold_newest=True)

    def _cadence_probe(self) -> bool:
        return self._probe_fn is not None and self._cadence(self.probe_every)

    def _probe(self) -> dict[str, Any]:
        # Runs after this step's statistics are in the ring; the capture hooks fire during
        # the probe's forward pass too, so their buffers are cleared again afterwards.
        devices = sorted({p.device for p in self._params}, key=str) or [self.device]
        try:
            return run_probe(self._model, self._probe_fn, self._probe_state, devices)
        except Exception as error:  # noqa: BLE001
            # The failure is in the step's record (`probe.error`); the warning only points at it, and
            # cannot raise out of `optimizer.step()` under `python -W error` (`_warn`).
            _warn(f"TensorQuill probe failed and was skipped: {error!r}", stacklevel=3)
            return {"error": f"{type(error).__name__}: {error}"}
        finally:
            buffers = [b for b in (self._act, *self._counts_list) if b.numel() > 0]
            if buffers:
                torch._foreach_zero_(buffers)

    def flush(self) -> None:
        """Move the ring to the CPU and queue it for the sink."""
        if self._closed_call("flush()"):
            return
        self._flush(hold_newest=False)

    def _clear_ring_lists(self) -> None:
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
        self._ring_no_optimizer = []
        self._ring_spread_seen = []
        self._ring_data = []
        self._ring_checkpoint = []
        self._ring_wait = []

    def _flush(self, *, hold_newest: bool) -> None:
        """Move the ring to the CPU and queue it (`_flush_ring`). An exception part way would leave
        the ring's lists longer than the ring and every later step raising (lane U5 saw an
        `IndexError` at each step after one raise here), so it stops capture instead (`_stop`),
        which clears the ring: nothing raises into training from here."""
        if self._error is not None and not self._stopped:
            self._stop(self._error)
        if self._stopped:
            return
        try:
            self._flush_ring(hold_newest=hold_newest)
        except Exception as error:  # noqa: BLE001
            self._stop(error, f"moving a block of steps off the device raised {_short(error)}")

    def _flush_ring(self, *, hold_newest: bool) -> None:
        if self._probe_owed and not hold_newest:
            self._run_owed_probe()  # the newest step's, before it leaves the ring (a held one stays)
        if self._held is not None:
            # The ring holds only the step kept back when it last filled.
            self._release_held()
            return
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
        if spreads is not None:
            self._say_replica_unread(spreads, self._ring_spread_seen)
        clips = _cpu_copy(self._ring_clip[:n]) if any(self._ring_clip_seen) else None
        clip_seen = self._ring_clip_seen
        update_seen, self._ring_update_seen = self._ring_update_seen, []
        updates = _cpu_copy(self._ring_update[:n]) if any(update_seen) else None
        amps = _scalars_to_cpu(self._ring_amp)
        self._ring_amp = []
        no_optimizer, self._ring_no_optimizer = self._ring_no_optimizer, []
        if any(flag and not (amp and amp.get("step_skipped")) for flag, amp in zip(no_optimizer, amps)):
            # A step that was not skipped, yet no optimizer of this model stepped: the one that
            # did holds none of its parameters. A skipped step says nothing of that.
            self._update_why.setdefault(
                UPDATES_NO_OPTIMIZER,
                "no optimizer that has stepped holds this model's parameters (DeepSpeed ZeRO, Megatron-LM and "
                "FSDP1 step flattened or copied parameters), so there is no Adam state to read for them",
            )
        update_why = {name: why for name, why in self._update_why.items() if name not in self._update_why_written}
        self._update_why_written.update(update_why)
        grad_why = self._new_grad_why()
        grad_seen = self._ring_grad_seen
        weight_seen = self._ring_weight_seen
        spread_seen = self._ring_spread_seen
        scalars = _scalars_to_cpu(self._ring_scalars)
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
            refused, announce, updates, update_seen, update_why, grad_why,
        )
        if hold_newest:
            first = self._hold_newest(dict(zip(_BLOCK_FIELDS, block)), act_size=self._act.numel() if has_act else 0,
                                      originals=(steps, lrs, stamps, data_rows, waits, no_optimizer))
            if first is None:
                return
            block = tuple(first[key] for key in _BLOCK_FIELDS)
        self._put(block)

    def _hold_newest(self, block: dict[str, Any], *, act_size: int, originals: tuple[Any, ...]) -> dict[str, Any] | None:
        """Split a full ring's block, already on the CPU in one transfer: the steps but the newest,
        returned to be queued now, and the newest, kept as `self._held` (`_release_held`). Its
        rows stay in the ring too, as its only row, so what arrives for it after it ended is
        recorded there as for any step in the ring; only a device row changed since is read
        again when it is written. None when the ring held one step (`every=1`)."""
        n = len(block["steps"])
        first: dict[str, Any] = {}
        held: dict[str, Any] = {}
        for key in ("steps", "lrs", "clip_seen", "grad_seen", "weight_seen", "spread_seen", "scalars", "probes",
                    "stamps", "amps", "data_rows", "checkpoints", "waits", "update_seen"):
            first[key], held[key] = block[key][:-1], block[key][-1:]
        for key in ("spreads", "clips", "updates"):
            rows = block[key]
            first[key], held[key] = (None, None) if rows is None else (rows[:-1], rows[-1:].clone())
        floats = block["floats"]
        if floats is None:
            first["floats"] = held["floats"] = None
        else:
            act = floats[: n * act_size].view(n, act_size)
            grad = floats[n * act_size :].view(n, (floats.numel() - n * act_size) // n)
            first["floats"] = torch.cat([act[:-1].reshape(-1), grad[:-1].reshape(-1)])
            held["floats"] = torch.cat([act[-1], grad[-1]])
        counts = block["counts"]
        if counts is None:
            first["counts"] = held["counts"] = None
        else:
            parts, offset = [], 0
            for ring in self._ring_counts.values():
                width = ring[0].numel()
                parts.append(counts[offset : offset + n * width].view(n, width))
                offset += n * width
            first["counts"] = torch.cat([part[:-1].reshape(-1) for part in parts])
            held["counts"] = torch.cat([part[-1] for part in parts])
        # What is said once for the block goes on its first record, written now (with one step, on
        # that step); the system readings on its last, as for any block.
        last = n == 1
        for key in ("announce", "update_why", "grad_why"):
            first[key], held[key] = (None, block[key]) if last else (block[key], {})
        first["system"], held["system"] = None, block["system"]
        first["refused"] = held["refused"] = block["refused"]
        self._held = {"block": held, "act_size": act_size, "dirty": set()}
        # The ring keeps the newest step as its only row, at row 0: its Python entries as they were
        # (logged tensors still on the device), and its update row on the device for a follower's.
        steps, lrs, stamps, data_rows, waits, no_optimizer = originals
        self._steps, self._ring_lr, self._ring_ts = steps[-1:], lrs[-1:], stamps[-1:]
        self._ring_data, self._ring_wait, self._ring_no_optimizer = data_rows[-1:], waits[-1:], no_optimizer[-1:]
        self._ring_scalars, self._ring_amp = block["scalars"][-1:], block["amps"][-1:]  # on the CPU now
        self._ring_probe, self._ring_checkpoint = block["probes"][-1:], block["checkpoints"][-1:]
        self._ring_grad_seen, self._ring_weight_seen = block["grad_seen"][-1:], block["weight_seen"][-1:]
        self._ring_clip_seen, self._ring_spread_seen = block["clip_seen"][-1:], block["spread_seen"][-1:]
        self._ring_update_seen = block["update_seen"][-1:]
        if self.updates and n > 1:
            self._ring_update[0].copy_(self._ring_update[n - 1])
        return None if last else first

    def _release_held(self) -> None:
        """Queue the step kept back when the ring last filled, with what arrived for it since:
        a save, a probe, a follower's update, weight check or loss. Only what changed on the
        device is read again (with a follower, once a block); otherwise no transfer."""
        held, self._held = self._held, None
        if held is None:
            return
        block, dirty = held["block"], held["dirty"]
        block["checkpoints"] = _checkpoints_to_cpu(self._ring_checkpoint)
        block["probes"] = self._ring_probe
        block["weight_seen"] = self._ring_weight_seen
        block["update_seen"] = self._ring_update_seen
        if "scalars" in dirty:
            block["scalars"] = _scalars_to_cpu(self._ring_scalars)
        if "update" in dirty:
            block["updates"] = _cpu_copy(self._ring_update[:1])
        if "grad" in dirty and block["floats"] is not None:
            weights = block["floats"][held["act_size"] :].view(-1, 4)
            weights[:, 2:] = self._ring_grad[0, :, 2:].cpu()
        update_why = {name: why for name, why in self._update_why.items() if name not in self._update_why_written}
        self._update_why_written.update(update_why)
        block["update_why"] = {**block["update_why"], **update_why}
        block["grad_why"] = {**block["grad_why"], **self._new_grad_why()}
        self._clear_ring_lists()
        self._put(tuple(block[key] for key in _BLOCK_FIELDS))

    def _mark_held(self, part: str) -> None:
        """The held step's device row `part` changed after it was read (`_release_held`)."""
        if self._held is not None:
            self._held["dirty"].add(part)

    def _put(self, block: tuple[Any, ...]) -> None:
        """Hand a block to the writer thread, never waiting for it. A writer a full queue behind
        stops capture for the session (`_stop`) and this block is dropped: the steps already
        queued are still written, so the records stay one unbroken run of steps."""
        if self._stopped:
            return
        try:
            if self._close_by is None:
                self._queue.put_nowait(block)
            else:
                # In `close()` no training step waits, and a writer that is slow but working gets
                # until the close's deadline to make room for the last block.
                self._queue.put(block, timeout=max(self._close_by - time.monotonic(), 0.0))
        except queue.Full:
            blocks = self._queue.maxsize
            if self._close_by is not None:
                self._stop(RuntimeError(f"the writer had not made room for the last block within {CLOSE_SECONDS:g} s "
                                        "at close"))
                return
            cause = ("for example on a stalled network filesystem (pass sink=tensorquill.JsonlSink(path) on "
                     "local disk)")
            if self._system is not None:
                # With system readings on, the writer may instead be in its bounded wait for an NVML
                # reading (system.py), after which NVML is off for the session.
                cause += ", or while the writer waited for an NVML reading that was slow to come back"
            self._stop(RuntimeError(
                f"the writer fell {blocks} blocks ({_queue_rings(self.every) * self.every} steps) behind training, "
                f"{cause}"
            ))

    def _stop(self, cause: BaseException, what: str | None = None) -> None:
        """Stop capture for the rest of the session, on the training thread, and say so once.

        The author's decision on C03: a sink that fails, or a writer a full queue behind, stops
        capture; nothing is raised into training and nothing waits. The cause is kept as
        `writer_error`. Every public entry point then returns at once (`_closed_call`), except
        what the heartbeat reads (`set_phase`, `tick`, `phase`). The forward hooks stay: a flag
        they read under `torch.compile` would be guarded, and flipping it would recompile; their
        buffers are never read again. The clipping and loader hooks come off. The optimizer hooks
        become no-ops (taking them out now could be inside PyTorch's loop over them, which would
        raise) and `close()` removes them; with a `replica_group` they instead keep counting steps,
        so this rank keeps entering the drift check's gather on the same steps as the others
        (`_count_stopped`). `close()` then writes no end marker, so the file reads as cut off."""
        if self._stopped:
            return
        self._stopped = True
        self.writer_error = cause
        self._clear_ring_lists()
        self._held = None
        self._probe_owed = False
        try:  # an idle writer ends now; a busy one ends once the queue is empty (`_drain`)
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        for undo in (lambda: _unwatch_clipping(self), lambda: datawait.unwatch(self)):
            try:
                undo()
            except Exception:  # noqa: BLE001
                pass
        if not self._keeps_schedule():
            for handle in self._optimizer_handles:
                try:
                    hooks = handle.hooks_dict_ref()
                    if hooks is not None and handle.id in hooks:
                        hooks[handle.id] = _noop_hook  # same key: safe while PyTorch iterates them
                except Exception:  # noqa: BLE001
                    pass
            try:
                self._unwatch_scaler()
            except Exception:  # noqa: BLE001
                pass
        if what is None:
            what = f"its sink raised {_short(cause)}" if cause is self._error else str(cause)
        _warn(
            f"TensorQuill stopped capturing for this session: {what}. Training goes on and does not wait; "
            "the steps already handed to the writer are written, no later step is, and the records file gets "
            "no end marker, so it reads as cut off. The cause is session.writer_error.",
            stacklevel=4,
        )

    def _keeps_schedule(self) -> bool:
        """Whether a stopped session must keep counting steps: it takes part in the drift check's
        `all_gather`, which every rank of the group must enter on the same steps."""
        return (
            self._replica_group is not None and not isinstance(self._replica_group, torch.Tensor)
            and bool(self._replica_params)
        )

    def _count_stopped(self, step: int | None) -> None:
        """A step ended on a stopped session: only the drift check's gather goes on, on the steps it
        was due (as `_commit` schedules it), and its result is dropped."""
        current = self._next_step if step is None else step
        self._stepped = self._foreign_stepped = False
        if self._keeps_schedule() and not self._spread_recorded and (current + 1) % self.replica_every == 0:
            self._gather_and_drop(self._replica_group)
        self._spread_recorded = 0
        self._next_step = current + 1

    def _gather_and_drop(self, group: Any) -> None:
        try:
            spread_buf, _ = self._ensure_spread_buffers()
            _spread_and_unread(self._replica_params, group, projection=self._replica_projection,
                               device=spread_buf.device)
        except Exception:  # noqa: BLE001
            pass  # nothing is recorded any more; a failed gather is as it would be without TensorQuill

    def set_phase(self, phase: str = "train") -> None:
        """Say what the process is doing: `"train"` (the default), `"eval"`, `"checkpoint"`, or
        any short word. While it is not `"train"` and its heartbeats go on, the server does
        not call the missing steps a hang. Under `"eval"`, a forward in eval mode or without
        gradients is also left out of the records (`_evaluating`); without it, only one with
        both. Otherwise a no-op unless streaming with `endpoint=`."""
        if self._closed:  # live after a stop: the heartbeat still says what the process is doing
            return
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
        if self._closed:  # live after a stop, as `set_phase`
            return
        self._ticks += 1

    def watch_loaders(self) -> None:
        """Record the host time this thread spends in `next()` on any PyTorch `DataLoader`
        during training, as `data_wait` on each step, until the session closes. See
        `tensorquill.datawait`. `init(..., data_wait=True)` calls it."""
        if self._closed_call("watch_loaders()"):
            return
        self._start_data_timing()
        datawait.watch(self)

    def time_data(self, iterable: Any) -> "datawait.TimedData":
        """`for batch in session.time_data(loader):` records the host time spent fetching each
        batch, as `data_wait` on each step. Move the batch to the device after this, outside
        the timer. For loops whose loader is not a PyTorch `DataLoader`; with one,
        `watch_loaders()` does the same without changing the loop. On a closed session it
        returns `iterable` itself, untimed."""
        if self._closed_call("time_data()"):
            return iterable
        return datawait.TimedData(iterable, self)

    def _start_data_timing(self) -> None:
        if not self._data_timing:
            self._data_timing = True
            self._pending_data_wait = 0.0
            self._last_step_clock = None

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """`with session.phase("eval"): ...` marks a stretch with no training steps, and goes
        back to the phase it interrupted afterwards, even if the block raises. On a closed
        session, or one closed inside the block, it marks nothing."""
        if self._closed:  # live after a stop, as `set_phase`
            yield
            return
        previous = self._phase
        self.set_phase(name)
        try:
            yield
        finally:
            if not self._closed:
                self.set_phase(previous)

    def _closed_call(self, what: str) -> bool:
        """Whether the session is closed, so the public call `what` must do nothing. A closed
        session's writer thread has stopped: a step or flush queued for it would fill the writer's
        bounded queue and then wait for good (`_put`). So every public entry point returns at once
        on a closed session: no raise, nothing queued, and no warning, since a warning on the
        caller's thread is an exception under `python -W error`. `what` names the call for a
        reader of this code; nothing is said. A session whose capture stopped (`_stop`) is the same
        state for these calls: its one warning was given when it stopped."""
        return self._closed or self._stopped

    def close(self, *, _exception: BaseException | None = None) -> None:
        """Flush what is left and stop the writer. Forward hooks stay on the model, because
        removing a hook from a compiled module triggers a recompile. Once the records file is
        closed and complete, `<name>.end.json` is left beside it (`sink.write_end_marker`), so a
        reader can tell this run from one that was killed.

        It never raises and never waits without bound. A sink that fails here, or a writer that
        has not finished within `CLOSE_SECONDS`, stops capture (`_stop`), as during training: one
        warning, `writer_error` set, and no end marker, so the file reads as cut off. After a stop
        the writer, which may be stuck in the sink, is waited for `STOPPED_CLOSE_SECONDS` at most;
        a sink still in a write is not closed under it (only a stream's heartbeat and sender are
        stopped), and the writer, a daemon thread, does not hold the process. Inside `with
        session:` the training's own exception comes out unchanged."""
        if self._closed:
            return
        self._closed = True
        for undo in (
            lambda: datawait.unwatch(self),
            lambda: [handle.remove() for handle in self._optimizer_handles],
            lambda: _unwatch_clipping(self),
            self._unwatch_scaler,
        ):
            try:
                undo()
            except Exception:  # noqa: BLE001
                pass
        deadline = time.monotonic() + CLOSE_SECONDS
        if not self._stopped:
            self._close_by = deadline  # the last block may wait for room, up to the deadline (`_put`)
            self._flush(hold_newest=False)  # the public `flush()` does nothing once closed
        if not self._stopped:
            try:
                self._queue.put(None, timeout=max(deadline - time.monotonic(), 0.0))
            except queue.Full:
                self._stop(RuntimeError(f"the writer had not taken what was left within {CLOSE_SECONDS:g} s at close"))
        wait = STOPPED_CLOSE_SECONDS if self._stopped else max(deadline - time.monotonic(), 0.0)
        self._worker.join(wait)
        if self._worker.is_alive():
            self._stop(RuntimeError(f"the writer had not finished within {wait:g} s at close"))
            try:
                _close_stream(self._sink)  # the stream's threads, not the sink the writer is still in
            except Exception:  # noqa: BLE001
                pass
        else:
            if self._error is not None:
                self._stop(self._error)  # the last write failed
            try:
                self._sink.close()
            except Exception as error:  # noqa: BLE001
                self._stop(error)
        if self._stopped or getattr(self._sink, "path", None) is None:
            return
        try:
            write_end_marker(self._sink.path, run_id=self.run.run_id, rank=self.run.rank, attempt=self.attempt,
                             exception=_exception)
        except Exception:  # noqa: BLE001
            pass  # no marker: the file reads as cut off, which is less than it is, never more

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close(_exception=exc[1])

    # ---- writer thread ------------------------------------------------------------

    def _drain(self) -> None:
        """The writer thread. It never raises and never stops training: a sink that raises is kept
        as `_error`, which the training thread turns into a stop at its next step (`_commit`); the
        blocks after it are passed over. Once capture has stopped it writes what was queued before
        the stop and ends when the queue is empty, or at the `None` that `_stop` or `close` put."""
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                if self._error is None:
                    if self._system is not None:
                        # NVML is asked for here, on the writer thread, and read on a thread of its own,
                        # waited for a bounded time (system.py). A failure is written as a note in the
                        # record and never stops the writer.
                        # Found by its type, not its place, so other parts of the block can come after it.
                        try:
                            item = tuple(self._system.write(x) if isinstance(x, SystemPart) else x for x in item)
                        except Exception:  # noqa: BLE001
                            pass
                    try:
                        self._sink.write(self._decode(*item))
                    except BaseException as error:  # noqa: BLE001
                        self._error = error
                if self._stopped and self._queue.empty():
                    return
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
        grad_why: Mapping[str, str] | None = None,
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
            spreads.reshape(n, len(self.replica_names), FINGERPRINT_WIDTH + 1).tolist() if spreads is not None else None
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
                # Written when the check knew any parameter (1 or 0, not -1); one it did not read
                # (`_record_weights`), the first included, is left out of the list, never the list.
                if any(row[3] >= 0 for row in grad[i]):
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
                # A parameter some replica could not read (a code in the last column) is left out,
                # whatever its spread holds; with none left the record has no `replica_spread`, so
                # the readers count no comparison in it (`replica_drift` is then `not_configured`
                # where no other record compared any, never `checked`).
                compared = {
                    name: [float(v) for v in row[:width]]
                    for name, row in zip(self.replica_names, spread_rows[i])
                    if not row[-1]
                }
                if compared:
                    record["replica_spread"] = compared
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
        if grad_why and records:
            # Once for each parameter: why its gradient has no norm in `grad_norms` (`record_grads`).
            records[0]["grads_not_read"] = dict(grad_why)
        if system and records:
            records[-1]["system"] = system
        return records


# The parts of a block queued for the writer, in `Session._decode`'s order.
_BLOCK_FIELDS = (
    "steps", "lrs", "floats", "counts", "spreads", "clips", "clip_seen", "grad_seen", "weight_seen",
    "spread_seen", "scalars", "probes", "stamps", "amps", "data_rows", "checkpoints", "waits", "system",
    "refused", "announce", "updates", "update_seen", "update_why", "grad_why",
)


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


def _grad_unread(grad: torch.Tensor) -> str | None:
    """Why a gradient is not read into `grad_norms`, from its device and layout alone (no values
    read, so no sync); None when it is read. As the weight check (`_record_weights`) and the drift
    check (`distributed._unread_reason`) leave out a parameter on `meta` or a nested one."""
    if grad.device.type == "meta":
        return "its gradient is on meta, which holds no values"
    if grad.is_nested:
        return "its gradient is a nested tensor"
    return None


def _buffers_device(params: Sequence[torch.Tensor]) -> torch.device:
    """Where a session keeps its buffers: the device of the first trainable parameter that holds
    values, never `meta` (a parameter registered on the root module comes first in
    `named_parameters`, and one left on `meta` there would put every buffer on `meta`, which no
    flush can read). The CPU when none does: no trainable parameter, or all on `meta`. A model
    without a parameter on `meta` gets its first parameter's device, as before.

    Not covered (by reading, not run: no GPU): under FSDP's CPU offload the parameters are on the
    CPU while the blocks run on the GPU, so the buffers are on the CPU and every block's output is
    left out (`_output_unread`)."""
    return next((p.device for p in params if p.device.type != "meta"), torch.device("cpu"))


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


def _open_default_sink(path: Path, undo: list[Callable[[], Any]]) -> JsonlSink:
    """`init`'s default `JsonlSink`, with what undoes it: closed, and the file and the folders made
    for it removed when they did not exist before and are still empty."""
    made = [folder for folder in (path.parent, *path.parent.parents) if not folder.exists()]
    existed = path.exists()
    sink = JsonlSink(path)

    def remove() -> None:
        sink.close()
        if not existed and path.exists() and path.stat().st_size == 0:
            path.unlink()
        for folder in made:  # innermost first
            try:
                folder.rmdir()
            except OSError:
                break

    undo.append(remove)
    return sink


def _restore_file(path: Path) -> Callable[[], None]:
    """An undo step that puts `path` back as it is now: its bytes, or no file."""
    try:
        before: bytes | None = path.read_bytes()
    except OSError:
        before = None

    def restore() -> None:
        if before is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(before)

    return restore


def _close_stream(sink: Any) -> None:
    """Stop what `flight.attach` started (heartbeat and sender) without closing the sink it wraps,
    which is `init`'s to close (its default) or the caller's."""
    heartbeat = getattr(sink, "heartbeat", None)
    if heartbeat is not None:
        heartbeat.close()
    sender = getattr(sink, "sender", None)
    if sender is not None:
        sender.close()


@torch.compiler.disable
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
    probe_head: str | bool | None = None,
    scaler: Any = None,
    replica_params: Any = None,
    endpoint: str | None = None,
    system: bool = False,
    data_wait: bool = False,
    updates: bool = False,
) -> Session:
    """Start capturing `model`. Call after `model.to(device)` and before `torch.compile`.

    Called inside a function that `torch.compile` traces, it runs eagerly
    (`torch.compiler.disable`), and the model's calls after it are traced with its hooks.
    Nothing is changed when it raises: every option is checked first, and what was opened before
    a later step failed is closed again.

    Args:
        run_id: the tracker's run ID. Defaults to MLFLOW_RUN_ID, the active MLflow run or
            the active W&B run, in that order, then torchrun's or SLURM's job ID, then a
            generated one. It is the run's identity: every file and every folder in a
            server's store is named by it, whatever the project. Under `torch.distributed`
            with several ranks, rank 0 resolves it, this argument included, and every rank
            takes rank 0's ID and where it came from through the process group's store (no
            collective, nothing on a GPU): a `run_id` given on rank 0 only is every rank's,
            and a rank given another one is warned and writes under rank 0's. So call `init`
            on every rank: a rank that has not heard from rank 0 within
            `tensorquill.tracker.RUN_ID_WAIT_SECONDS` (300 s, or the group's timeout if
            shorter) writes under an ID of its own, with a warning.
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
            check reads every weight in float64 and gathers 32 bytes per parameter tensor
            from each rank: norm, sum and a seeded random projection, which catches
            changes that keep the other two, such as swapped elements, and a flag for a
            parameter some replica could not read. It is a screen, not proof of equality.
            `session.record_replica_spread()` records one on demand.
        replica_projection: set False to skip the projection (two values per tensor, 24 bytes
            with the flag). On
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
        probe_head: the language-model head the probe reads output-logit numbers from
            (`head` in the probe record: norm, signed mean, log Z, and the gap between
            BF16 and FP32 logits on the same hidden states; numbers only, no incident).
            A module name, which must be a plain `nn.Linear`; by default the module
            `model.get_output_embeddings()` returns, if the model has that method, and
            otherwise the head is recorded as not read, with the reason. `False` turns it off.
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
            A call counts when any parameter it is given is one of the model's trainable
            ones. Only an L2 norm (`norm_type` 2, the default) is recorded: a call with
            another norm type records nothing. Free when you clip anyway, and enough for
            gradient-spike detection.
        system: at each flush, add the CUDA caching allocator's counters (allocated,
            reserved and peak bytes, allocation retries and OOMs; host-side, no sync) and
            NVML's temperature, SM clock, power, ECC counters, throttle reasons and Xid
            events (`pip install nvidia-ml-py`), read on a thread of their own that the
            writer waits for a bounded time, to the block's last record as `system`. A
            reading that cannot be taken is written once as unavailable. Off by default
            until its cost is measured on a GPU. See `system.py`.
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
    # Every check of the model and the options too, before the run is resolved (which tells other
    # ranks its ID) or anything is opened: an `init` that is rejected changes nothing.
    plan = _plan_session(
        model, every=every, layers=layers, routers=routers, grads=grads, weights=weights,
        replica_every=replica_every, probe=probe, probe_every=probe_every, probe_head=probe_head,
        replica_params=replica_params, scaler=scaler,
    )
    # What `init` opens is undone if a later step fails (`_undo`): the default sink and its file,
    # the run manifest, the stream to a server. A sink the caller passed in is never closed here.
    undo: list[Callable[[], Any]] = []
    try:
        run = resolve_run(run_id)
        if sink is None:
            sink = _open_default_sink(Path("tensorquill-out") / f"{run.run_id}.rank{run.rank}.jsonl", undo)
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
                undo.append(_restore_file(manifest_file))
                _manifest.write_manifest(manifest_file, run_manifest)
                # A run started again under its run ID keeps the first start's time.
                run_manifest = _manifest.read_manifest(manifest_file) or run_manifest
        from .flight import attach  # a no-op unless endpoint= or TENSORQUILL_ENDPOINT is set

        given = sink
        sink = attach(
            sink, run.run_id, run.rank, endpoint, world_size=_world_size(), heartbeat_seconds=heartbeat_seconds,
            stack_seconds=stack_seconds, manifest=run_manifest,
        )
        if sink is not given:
            undo.append(lambda: _close_stream(sink))
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
            probe_head=probe_head,
            scaler=scaler,
            replica_params=replica_params,
            system=system,
            data_wait=data_wait,
            updates=updates,
            _plan=plan,
        )
    except BaseException:
        _undo(undo)
        raise
    session.manifest = run_manifest  # the run manifest rank 0 wrote; None on every other rank
    return session
