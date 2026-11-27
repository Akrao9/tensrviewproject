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

import collections
import contextlib
import logging
import math
import numbers
import os
import queue
import sys
import threading
import time
import warnings
import weakref
from dataclasses import dataclass, replace as _replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
from torch import nn
from torch.optim.optimizer import (
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)

from .data import (
    FOLDER_TENSORS,
    DataStep,
    WatchedLoader,
    checkpoint_manifest,
    data_fields,
    files_manifest,
    load_manifest,
    optimizer_entries,
    optimizer_step_value,
)
from . import datawait
from .distributed import (
    FINGERPRINT_WIDTH,
    PROJECTION_NAME,
    SIGNATURE_NAME,
    _observation_table,
    _spread_and_unread,
    _unread_reason,
    fold64_finish,
    group_ranks,
    name_v1_seed,
    unwrap_local,
)
from .flight import _without_traceback
from .probe import ProbeState, find_head, install_head_hook, install_probe_hooks, probe_callable, run_probe
from .sink import JsonlSink, Sink, clear_end_marker, records_file_name, write_end_marker
from .system import SystemPart, SystemReadings
from . import manifest as _manifest
from .tracker import LaunchRead, RunInfo, _world_size, resolve_run, tracker_labels

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
# Version of the scalar-only record `Session.log(..., step=N)` writes (`WAVE_AL_CONTRACT.md` section
# 4.2, Q1 a): only what changed is version 2, so a step record stays version 1.
SCALARS_SCHEMA_VERSION = 2
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
# Index tensors kept on the session's device for sets of parameters read together (`Session._index`),
# the one used least recently dropped first. Routed experts with no gradient when unused
# make a new set at almost every step, so the cache is bounded rather than one tensor per set seen.
INDEX_CACHE_SIZE = 32
# Why the update size was not read (`updates=True`), by what it is said about: an optimizer's
# class name, or one of these two.
UPDATES_HOOKS_OFF = "auto_step=False"
UPDATES_NO_OPTIMIZER = "no optimizer"
# The key under which `step(skipped=...)` keeps the caller's verdict from `_commit` to `_decode`;
# never written to a record, which says `step_skipped`, `step_skipped_by` and `scaler_applied` instead.
_CALLER_SKIPPED = "caller_skipped"
# Kept beside a scaler verdict that `scaler.update()` gave after the `session.step()` of its update
# had already ended that step (`step`, `_record_scaler`): it is not this record's update, so it never
# says the scaler applied a caller's skip (`scaler_applied`). Never written to a record either.
_SCALER_LATE = "scaler_verdict_late"
# Kept in the same per-step dict from `_commit` to `_decode` (wave AE): the eager clipping calls of a
# step that had more than one (written as `clip_calls`); and, with two or more optimizers through the
# scaler, how many it skipped (a device scalar, read at the flush with `step_skipped`) and of how many
# (a host count), written as `step_skipped_part` on a mixed outcome only.
_CLIP_CALLS = "clip_calls"
_SKIPPED_OPTIMIZERS = "scaler_skipped_optimizers"
_SCALED_OPTIMIZERS = "scaler_optimizers"
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
    integer tensor on the counts' device. One that raises, or returns anything else, has its router
    refused for the session (`_router_hook`), never an error in training.

    Expert ids, from `indices` or the default integer tensor, must number `top_k` for each token
    the router was called on (the open part of C06): any other count refuses the router with a
    reason naming the count, the tokens and the declared `top_k`, since `moe_top_k` tells the
    detectors how large an expert's share can be. Judged from shapes alone, so the same under
    `torch.compile`; not judged when the tokens cannot be read from the router's first argument.
    A `top_k` above `n_experts` refuses the router on every path (the default integer tensor, a
    function given as `indices`, a routing map and the top-k of logits), with the reason "its
    top_k is K but it has E experts".

    Expert ids, from `indices` or the default integer tensor, are judged on their values (the
    author's decision on C06): an id below 0 is no assignment (Megatron's -1 for no route), not
    counted, and how many there were in a step is recorded as `moe_unassigned`; an id at or above
    `n_experts` matches no expert and refuses the router from the block in which it appears. A
    routing map's entries left out of the counts (their paired float is exactly zero) are recorded
    per expert as `moe_left_out`, so `moe[e] + moe_left_out[e]` is the map's own count. Both are
    written with explicit zeros whenever their check ran, and left out when it did not apply.

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
    # A plain loop, not `next(generator, None)`: Dynamo on PyTorch 2.6 cannot trace that call, and
    # this runs inside the compiled router hook.
    paired = None
    for item in candidates:
        if isinstance(item, torch.Tensor) and item.is_floating_point() and item.shape == found.shape:
            paired = item
            break
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
    held: _Held | None = None,
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

    The session says `problems` under `layers_not_read` (contract, part 7; lane AD3), once, for a
    block whose output was never read on any step before it (`Session._layers_not_read`)."""
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
        if torch.compiler.is_compiling():
            # No `try` in a traced hook: the checks in `read` are what keep it from raising there.
            # Nothing is held here: Dynamo runs a reentrant checkpoint as a non-reentrant one.
            if _evaluating(module, phase):
                return
            read(output)
            return
        if _skip_forward(module, phase, held):
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


def _index_range_problem(spec: Router, counts: torch.Tensor, n: int, steps: Sequence[int]) -> tuple[str, None] | None:
    """The refusal of a router whose expert ids reached `n_experts` on a step of a block, from its counts
    on the CPU (`_judge_index_ranges`, and the pinned path's `_settle_refusals`); None when none did."""
    width = _count_width(spec)
    outside = counts.reshape(n, width)[:, spec.n_experts + 1].tolist()
    bad = next((i for i, v in enumerate(outside) if v), None)
    if bad is None:
        return None
    source = "its indices= function returned" if spec.indices is not None else "its output's first integer tensor held"
    return (
        f"{source} {outside[bad]} value{'s' if outside[bad] != 1 else ''} "
        f"outside 0 to {spec.n_experts - 1} at step {steps[bad]}, which match no expert",
        None,
    )


def _refusal(name: str, spec: Router, why: str, fix: str | None = None, device: Any = None) -> str:
    """Why a router is not counted and the two ways to have it counted, as one text: the
    session's warning and the value of `moe_refused` (so the evidence and the server print
    the same words). The fixes depend on how the router was named, which only the capture
    knows, and for two reasons on the reason itself (`fix`, set by `_router_hook`): output on
    another device than the counts (`device`), where the inner module's output would be too,
    and a `top_k` above the number of experts. No "; ": the evidence joins several routers'
    reasons with it."""
    inner = "name the module inside it whose output is the router's logits"
    if fix is not None and fix.startswith("top_k=") and int(fix.removeprefix("top_k=")) > spec.n_experts:
        fix = None  # more ids a token than experts: no top_k would count them, so the usual two fixes
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
    elif fix is not None and fix.startswith("top_k="):
        # The ids numbered a whole multiple of the tokens, but not `top_k` for each (`_ids_per_token`).
        shown = fix.removeprefix("top_k=")
        if spec.indices is None:
            fix = (
                f"pass routers={{{name!r}: tensorquill.Router({spec.n_experts}, {shown})}} if it chooses {shown} "
                f"experts for each token, or tensorquill.Router({spec.n_experts}, {spec.top_k}, indices=...) with a "
                f"function that returns {spec.top_k} expert ids for each token from its output"
            )
        else:
            fix = (
                f"pass routers={{{name!r}: tensorquill.Router({spec.n_experts}, {shown}, indices=...)}} if it "
                f"chooses {shown} experts for each token, or change its indices= function to return {spec.top_k} "
                f"expert ids for each token"
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


def _sharded(model: nn.Module) -> str | None:
    """How `model` is sharded across ranks, or None: FSDP1 (a `FullyShardedDataParallel` wrapper,
    at the root or nested, or a module inside one: FSDP1 marks every module it manages with
    `_is_fsdp_managed_module` and every original parameter it flattened with `_fsdp_flattened`),
    FSDP2 (a module `fully_shard` was applied to, or a parameter that is a DTensor with a `Shard`
    placement, which is what a module inside one holds).

    Under either, a forward is a collective: each sharded unit gathers its parameters from every
    rank. Read from types and attributes on this rank alone, so asking adds no collective, and
    anything unexpected reads as not sharded: this is asked at `init`, which it must never fail.
    Not seen: a model sharded after `init`."""
    try:
        fsdp = sys.modules.get("torch.distributed.fsdp")  # not imported: nothing here is FSDP's
        v1 = getattr(fsdp, "FullyShardedDataParallel", None)
        v2 = getattr(fsdp, "FSDPModule", None)
        for module in model.modules():
            if isinstance(v1, type) and isinstance(module, v1):
                return "FSDP1, FullyShardedDataParallel"
            if isinstance(v2, type) and isinstance(module, v2):
                return "FSDP2, fully_shard"
            # Inside a wrapper that is not part of `model` (Lightning's FSDP strategy with no wrap
            # policy wraps the LightningModule, and its callback passes the LightningModule).
            if getattr(module, "_is_fsdp_managed_module", False) is True:
                return "FSDP1, inside a FullyShardedDataParallel wrapper"
        for parameter in model.parameters():
            if getattr(parameter, "_fsdp_flattened", False) is True:
                return "FSDP1, inside a FullyShardedDataParallel wrapper"
        if _DTensor is not None:
            for parameter in model.parameters():
                if isinstance(parameter, _DTensor) and any(
                    type(placement).__name__ == "Shard" for placement in parameter.placements
                ):
                    return "FSDP2, sharded DTensor parameters"
    except Exception:  # noqa: BLE001
        return None
    return None


def _explicit_step(step: Any, what: str = "session.log(..., step=...): step", least: int = 0) -> int:
    """`Session.log`'s `step=` as an `int`, or the refusal of contract section 4.2: a bool, a float and
    anything not an integer raise `TypeError`, a negative integer `ValueError`. Any `numbers.Integral`
    but a bool is an integer, as `identity.check` takes a step. `None` never reaches here from `log`:
    it is no step, the same as leaving `step=` out (the owner's decision, Nov 24).

    The same rule for `init`'s `start_step` and `every` (BUG-05: a bool is an `int`, so
    `start_step=True` was written as `"step": true`, which every reader refuses): `what` names the
    argument in the message, and `least` is the smallest value taken."""
    if isinstance(step, bool) or not isinstance(step, numbers.Integral):
        raise TypeError(f"{what} must be an integer of {least} or more, not {type(step).__name__} {step!r}")
    value = int(step)
    if value < least:
        raise ValueError(f"{what} must be an integer of {least} or more, not {value}")
    return value


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


def _finite_or_text(value: float) -> float | str:
    """A fingerprint as the observation writes it (contract 1): a finite number as it is, and
    otherwise `"nan"`, `"inf"` or `"-inf"`, in the record itself, since not every path a record
    takes (a `MemorySink`, the stream's row) goes through the JSONL sink's own conversion."""
    if math.isfinite(value):
        return float(value)
    return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")


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


# The author's decision on C06 and F1: `moe_unassigned` and `moe_left_out` are written with explicit
# zeros whenever the router was counted and the check applied, and left out when it did not apply.
# Readers take a missing value, every older record's included, as unknown.


def _ids_per_token(chosen: Any, tokens: int | None, spec: Router) -> int | None:
    """How many expert ids for each token `chosen` ids, a whole multiple of `tokens`, are, when that
    is not the router's `top_k`, so the router is refused (`_ids_per_token_why`, and the fix
    `_refusal` names for it); None when they number `top_k` for each token, or when either count is
    not a plain int (no tokens read from the first argument; a symbolic size under `torch.compile`).
    Before, any whole multiple was counted whatever `top_k` said (C06): two ids a token under a
    declared top-1 were counted as top-1 assignments, and the share detectors, which read
    `moe_top_k`, judged them against the wrong limit."""
    if not tokens or type(tokens) is not int or type(chosen) is not int:
        return None
    if chosen == tokens * max(spec.top_k, 1):
        return None
    return chosen // tokens


def _ids_per_token_why(what: str, chosen: int, tokens: int, spec: Router) -> str:
    """The refusal's reason for `_ids_per_token`, built once (`refuse`)."""
    return (
        f"{what} {chosen} expert ids for the {tokens} tokens it was called on, {chosen // tokens} for each, "
        f"but its top_k is {spec.top_k}"
    )


def _count_width(spec: Router) -> int:
    """Slots in a router's counts buffer, filled by `_router_hook` and read in `flush` with the copy to
    the CPU made anyway, so the values are judged without a sync and the hook's shapes depend on the
    `Router` alone (the same under `torch.compile`). For `E = n_experts`:

    - `[0, E)`: assignments per expert (`moe`);
    - `E`: ids below 0, which are no assignment (Megatron's -1 for no route): `moe_unassigned`;
    - `E + 1`: ids at or above `E`, which match no expert: the router is refused (`_judge_index_ranges`);
    - `[E + 2, 2E + 2)`: true routing-map entries left out because their paired float is exactly 0,
      per expert, so `moe[e] + moe_left_out[e]` is Megatron's own `routing_map.sum(dim=0)`: `moe_left_out`;
    - `2E + 2`: forwards whose expert ids were checked (the integer and `indices=` paths);
    - `2E + 3`: forwards whose routing map was checked against a paired float;
    - `2E + 4`: forwards counted from a routing map, paired or not;
    - `2E + 5`: forwards counted from the top-k of scores (logits);
    - `2E + 6`: forwards left out as evaluations and held (`_skip_forward`, eager only): on a step
      where one was held and none of `2E + 2`, `2E + 4` and `2E + 5` counts a forward, nothing of the
      step's forward was read, and the router is left out of that record (AUD-1, wave AG).

    The last four tell `moe_source`'s form (`_moe_forms`) from the counts the flush copies anyway, so
    the hooks keep no Python state of their own, which under `torch.compile` would be guarded on."""
    return 2 * spec.n_experts + 7


def _router_hook(
    counts: torch.Tensor,
    spec: Router,
    problems: list[tuple[str, str | None]] | None = None,
    phase: _Evaluating | None = None,
    held: _Held | None = None,
) -> Callable[..., None]:
    """The forward hook that adds a router's chosen experts to `counts`. Nothing in it raises
    into training: what cannot be counted refuses the router for the session (`problems`,
    said once by the session and written under `moe_refused`), first from types, layouts,
    devices and shapes alone, so the same under `torch.compile`, and in eager mode from any
    exception left, which is caught. A forward that is not training (`_evaluating`) is left
    out before anything is read, so an evaluation batch is neither counted nor refused."""
    phase = phase or _Evaluating()
    experts = spec.n_experts
    # What one forward adds to `counts` before its assignments (`_count_width`), made once on the
    # counts' device: for expert ids, the slot that says they were checked; for the top-k of
    # logits, nothing. The count goes by index (lane AD3), so no tensor of expert ids is kept: one
    # per router on the device that nothing read was made at `init` until wave AF.
    ids_start = torch.zeros(_count_width(spec), dtype=counts.dtype, device=counts.device)
    ids_start[2 * experts + 2] = 1  # the ids were checked; no routing map
    scores_start = torch.zeros_like(ids_start)
    scores_start[2 * experts + 5] = 1  # counted from the top-k of scores
    # The same two as int32, which is what one forward is counted into (see `count`).
    ids_start32, scores_start32 = ids_start.to(torch.int32), scores_start.to(torch.int32)
    no_ids = torch.zeros(2, dtype=counts.dtype, device=counts.device)
    map_checked = torch.tensor([0, 1, 1, 0, 0], dtype=counts.dtype, device=counts.device)
    unchecked_map = torch.zeros(experts + 7, dtype=counts.dtype, device=counts.device)
    unchecked_map[experts + 4] = 1  # a routing map with no paired float: counted from a map, nothing left out

    def refuse(why: Callable[[], str], fix: str | None = None) -> None:
        # Built only the first time: a refused router's hook still runs on every forward, and
        # the text may format a user's exception or a shape. `fix` picks the fix `_refusal` names.
        if problems is not None and not problems:
            problems.append((why(), fix))

    def count(args: Any, output: Any) -> None:
        logits = None  # set only on the top-k-of-logits path
        if spec.indices is not None:
            # The user's function: one that raises, or returns what cannot be expert indices,
            # refuses this router and never raises into training. Checked from types and
            # shapes only; its values are counted into fixed slots of `counts` (`_count_width`) and
            # judged in `flush`, after the copy to the CPU that is made anyway.
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
            each = _ids_per_token(chosen, tokens, spec)
            if each is not None:
                refuse(lambda: _ids_per_token_why("its indices= function returned", chosen, tokens, spec), f"top_k={each}")
                return
            if spec.top_k > spec.n_experts:
                # As on the default integer path (C06): ids that do number `top_k` for each token are
                # still judged against 1 / top_k by the share detectors, so with more ids a token than
                # experts they are refused, not counted (L19). From the `Router` alone, after the
                # count checks above, which name their own reason.
                refuse(lambda: f"its top_k is {spec.top_k} but it has {spec.n_experts} experts", "top_k")
                return
        else:
            idx = _first_tensor(output, integer=True)
            if idx is not None:
                # An integer tensor in the output is taken as the chosen expert indices, `top_k`
                # for each token (`_ids_per_token`). One that cannot be that (a router that returns
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
                if spec.top_k > spec.n_experts:
                    # No router chooses more distinct experts for a token than it has, and the share
                    # detectors judge an expert's share against 1 / top_k (`moe_top_k`): counted, these
                    # ids would be judged against the wrong limit (C06). Before, such a top_k was only
                    # written down. From the `Router` alone, so the same under `torch.compile`.
                    refuse(lambda: f"its top_k is {spec.top_k} but it has {spec.n_experts} experts", "top_k")
                    return
                each = _ids_per_token(chosen, tokens, spec)
                if each is not None:
                    refuse(lambda: _ids_per_token_why("its output's first integer tensor holds", chosen, tokens, spec), f"top_k={each}")
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
                if spec.top_k > spec.n_experts:
                    # A map routes a token to each expert at most once, so it cannot hold `top_k`
                    # assignments a token, and `moe_top_k` would judge its shares against the wrong
                    # limit (L19, C06's refusal). From the `Router` alone.
                    refuse(lambda: f"its top_k is {spec.top_k} but it has {spec.n_experts} experts", "top_k")
                    return
                for tensor, what in ((routing_map, "its routing map"), (paired, "the float tensor paired with its routing map")):
                    wrong = None if tensor is None else _unusable(tensor, counts.device, what)
                    if wrong is not None:
                        refuse(lambda: wrong[0], wrong[1])
                        return
                kept = routing_map.detach()
                if paired is None:
                    # No float of the map's shape: every true entry is counted, none can be left out,
                    # so that check did not apply (no `moe_left_out` for this forward).
                    hits = torch.cat([kept.reshape(-1, experts).sum(0), unchecked_map])
                else:
                    # The map is tokens x experts already, so its own size is the floor here; the
                    # routed count is the map's sum less what is left out, with no second mask.
                    left = (kept & (paired.detach() == 0)).reshape(-1, experts).sum(0)
                    routed = kept.reshape(-1, experts).sum(0) - left
                    hits = torch.cat([routed, no_ids, left, map_checked])
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
        # Fixed output shape: `bincount` would make the shape depend on the data. Each assignment
        # adds one at its slot (`index_add_`), so what is made is in proportion to the assignments
        # and the counts' width: before lane AD3 every id was compared with every expert, a bool
        # matrix of assignments by experts (16,384 tokens at top-8 over 256 experts: 33.5 million
        # entries a forward). Integer adds, so the counts are exact in any order, and PyTorch takes
        # a deterministic path for `index_add_` on CUDA when deterministic algorithms are on.
        # One forward is counted into int32 and widened into the int64 `counts` after: on a GPU
        # Inductor lowers an accumulating index write itself except into int64 or bool, where it
        # calls PyTorch's `index_put_` (a sort of the indices and a bounds check per router).
        # Measured on an A100 with PyTorch 2.14.1 at Granite's shape (24 routers, 1,024 tokens,
        # top-8 of 32): 6.6 ms a step less than into int64, with equal counts. A forward of 2**31
        # assignments or more (a shape, so no sync and no data-dependent branch) keeps int64.
        # Accumulate across gradient-accumulation microbatches; `_commit` zeroes `counts`.
        if idx is not None:
            flat = idx.detach().reshape(-1).to(torch.int64)
            narrow = flat.numel() < 2**31
            if spec.indices is not None or logits is None:
                # Ids the router or its `indices=` function gave: those below 0 are no assignment
                # (counted at slot E, not refused), those at or above n_experts at slot E + 1, which
                # refuses the router at the flush (`_judge_index_ranges`). Every slot is in range, so
                # no device assert. Device work only; no sync here.
                slot = torch.where(flat < 0, experts, torch.where(flat >= experts, experts + 1, flat))
                hits = (ids_start32 if narrow else ids_start).clone()
            else:
                slot = flat  # top-k of logits: always in range, nothing to check
                hits = (scores_start32 if narrow else scores_start).clone()
            hits.index_add_(0, slot, torch.ones_like(slot, dtype=hits.dtype))
            hits = hits.to(counts.dtype)
        if torch.compiler.is_compiling():
            _add_compiled(counts, hits)
        else:
            counts.add_(hits)

    def hook(module: nn.Module, args: Any, output: Any) -> None:
        if torch.compiler.is_compiling():
            # No `try` in a traced hook: the checks above are what keep it from raising there.
            # Nothing is held here: Dynamo runs a reentrant checkpoint as a non-reentrant one.
            if _evaluating(module, phase):
                return
            count(args, output)
            return
        before = held.n if held is not None else 0
        if _skip_forward(module, phase, held):
            if held is not None and held.n > before:
                try:  # said in the counts, so the writer can leave the router out (`_count_width`)
                    counts.narrow(0, 2 * experts + 6, 1).add_(1)
                except Exception:  # noqa: BLE001
                    pass
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


def _version_setter_takes_sequences() -> bool | None:
    """Whether `torch._C._autograd._unsafe_set_version_counter` takes sequences of tensors and versions
    (later releases) or one tensor and one version (PyTorch 2.6), asked once with two empty tuples:
    a release that takes sequences sets nothing, one that does not raises a `TypeError` whose text
    names only the two empty tuples. None when the function is not there."""
    setter = getattr(getattr(torch._C, "_autograd", None), "_unsafe_set_version_counter", None)
    if setter is None:
        return None
    try:
        setter((), ())
    except Exception:  # noqa: BLE001
        return False
    return True


# Chosen once, at import, with no tensor: a call in the wrong form raises a `TypeError` whose text
# prints each tensor it was given (`Tensor.__repr__` runs inside the C call), and an exception raised
# while it prints, such as the `SystemExit` of a job's SIGTERM handler, is lost there (SIG-26: on
# PyTorch 2.6 the tuple form was tried first at every step with hooked buffers).
_VERSION_SETTER_TAKES_SEQUENCES = _version_setter_takes_sequences()


def _restore_versions(tensors: Sequence[torch.Tensor], versions: Sequence[int]) -> None:
    """Set each tensor's autograd version back, what `torch.autograd._unsafe_preserve_version_counter`
    does on exit, in the form this PyTorch takes (`_VERSION_SETTER_TAKES_SEQUENCES`), so no call fails.
    Private API, so best effort: without it the version stays bumped, as before."""
    form = _VERSION_SETTER_TAKES_SEQUENCES
    if form is None:
        return
    try:
        setter = torch._C._autograd._unsafe_set_version_counter
        if form:
            setter(tuple(tensors), tuple(versions))
        else:  # PyTorch 2.6 takes one tensor at a time
            for tensor, version in zip(tensors, versions):
                setter(tensor, version)
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
    """The model's repeated blocks, such as a transformer's decoder layers: the children of every
    outermost `ModuleList` of two or more composite modules, in the model's order.

    A list counts when its children share one class and the first is composite (as before), or,
    for a list that mixes classes, when every child is composite: a hybrid stack of attention and
    state-space layers is one stack (SDK-23). Every such list is taken, not only the largest, so
    an encoder-decoder's two stacks are both watched. A list inside a block already taken (an MoE
    layer's experts, a stage's blocks) is not, and a module reachable from two lists is watched
    once, under its first name: no block is watched twice, so no block's hooks run inside
    another's. A list whose items are whole sub-models (`ModuleList([encoder, decoder])`) is taken
    as two blocks, not their layers.

    This is the unit PyTorch's FSDP `transformer_auto_wrap_policy` wraps. One hook per
    block locates a problem to a block at a fraction of the cost of one per leaf module.
    """
    blocks: list[tuple[str, nn.Module]] = []
    inside: set[int] = set()  # every module in a block taken so far, the blocks themselves included
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) < 2 or id(module) in inside:
            continue
        children = list(module)
        if len({type(c) for c in children}) == 1:
            if not any(children[0].children()):
                continue
        elif not all(any(c.children()) for c in children):
            continue
        for i, child in enumerate(children):
            if id(child) in inside:
                continue
            blocks.append((f"{name}.{i}" if name else str(i), child))
            inside.update(id(m) for m in child.modules())
    return blocks


def _root_leaf(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """A model that is itself one leaf module with parameters (an `nn.Linear` alone), named by its
    class (`_get_name()`, as sagemaker-debugger names the root module), since `named_modules` names
    it `""`. Empty for any other model: one with children has its leaves observed instead."""
    if not any(model.children()) and any(True for _ in model.parameters(recurse=False)):
        return [(model._get_name(), model)]
    return []


def _select_layers(
    model: nn.Module, layers: LayerFilter | Sequence[str] | str | None, *, root: bool = False
) -> list[tuple[str, nn.Module]]:
    """The modules whose outputs are summarized. `root`: a model that is one leaf module is its own
    layer (`_root_leaf`) under `"blocks"` and `"all"`, for the statistics; module names and filters
    never see the root."""
    named = [(name, module) for name, module in model.named_modules() if name]
    if layers is None or layers == "blocks":
        return _block_layers(model) or _leaf_layers(named) or (_root_leaf(model) if root else [])
    if layers == "all":
        return _leaf_layers(named) or (_root_leaf(model) if root else [])
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


# The attempt of the session handing its work counter to a heartbeat on this thread, while it does
# (`Session._install`), read by the heartbeat as the counter is set (`flight.Heartbeat._take_attempt`).
# Per thread, so a session started on another thread in between cannot give its attempt to this one's
# heartbeat, as `_last_attempt` alone could.
_handing_over = threading.local()


def _attempt_handed_over() -> int | None:
    return getattr(_handing_over, "attempt", None)


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


class _Held:
    """One hook's count of forwards it left out as evaluations that may be the first forward of
    reentrant activation checkpointing in a model trained in eval mode (AUD-1, `_skip_forward`).
    Python state read in eager mode only; `Session._commit` sets it back to 0 at each step."""

    __slots__ = ("n",)

    def __init__(self) -> None:
        self.n = 0


def _skip_forward(module: nn.Module, phase: _Evaluating, held: _Held | None) -> bool:
    """Whether an eager forward hook leaves this forward out: a recomputation inside backward
    (`_recomputing`) or an evaluation (`_evaluating`), with one exception (AUD-1, wave AG).

    Reentrant activation checkpointing runs a training forward without gradients and runs it again
    inside backward. In train mode the first is counted and the second left out, as before. With
    the modules in eval mode the first reads as an evaluation, so nothing of the step was counted
    and the record carried zeros. What tells the two apart is that only a training forward is run
    again in backward: a forward without gradients in eval mode, outside the `"eval"` phase, is held
    (`held`), and a recomputation inside backward is counted when it finds one held, one for one.
    It is the same forward on the same inputs (the checkpoint restores the random state), so it
    gives the counts and statistics the first would have.

    A forward counted in the usual way (gradients on, or train mode) lets go of what is held, so a
    recomputation after it is left out as before; so does the end of each step. A non-reentrant
    checkpoint's first forward has gradients on and is counted, and its recomputation is left out.
    Not told apart: an evaluation forward without gradients made in the same step before a
    training forward that is itself held, and a forward run inside backward that is not a
    checkpoint's recomputation, which would claim it; both need an evaluation inside a step."""
    if _recomputing():
        if held is not None and held.n > 0:
            held.n -= 1
            return False
        return True
    if _evaluating(module, phase):
        if held is not None and not phase.on and not module.training and not torch.is_grad_enabled():
            held.n += 1
        return True
    if held is not None:
        held.n = 0
    return False


def _add_square(acc: torch.Tensor, value: torch.Tensor | float) -> None:
    """Add the square of a norm to `acc`, in place, on `acc`'s device and with no sync. `acc` is
    `[s, e]`, standing for the sum of squares `s * 4**e`: `e` is the largest binary exponent of the
    norms added this step and `s` their squares scaled by it, so `s` stays near 1 whatever their size.

    BUG-04: the squares were added in float32, so a finite norm above about 1.8e19 (a float64 model's,
    or one passed by hand) became `inf` and a false `non_finite` incident fired, and a Python float
    above about 1.3e154 raised `OverflowError` from `float(norm) ** 2`. The norm is split into its
    mantissa and exponent (`frexp`) in its own dtype and on its own device first, so one of any float64
    size fits, also in a session on Apple MPS, which has no float64. In float32's range the sum is the
    one float32 squares added up gave, bit for bit: scaling by a power of two is exact. A NaN or an
    infinite norm stays one. Called once per clipping call: a dozen element-wise operations on two
    numbers, traced like any other under `torch.compile`."""
    # Each kind of value as `m2 * 4**e`, `m2` its squared mantissa in `acc`'s dtype, rounded as the
    # float32 buffer rounded it: a tensor moved to float32 and squared there, a Python float squared in
    # float64 and then rounded.
    if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
        mantissa, exponent = torch.frexp(value)  # on its own device: float64 is never on MPS
        m, e = mantissa.to(acc), exponent.to(acc)
        m2 = m * m
    elif isinstance(value, torch.Tensor):
        # float32 or narrower (or an integer) fits float32's range, and its square needs only one split:
        # above 2**60, scaled by 2**-66 (exact), with exponent 66; else as it is. No `frexp`, which
        # PyTorch 2.6 does not have on MPS, and no log.
        v = value.to(acc)
        big = v.abs() > _SQUARE_SPLIT
        m = torch.where(big, v * _SQUARE_SHRINK, v)
        e = big.to(acc) * _SQUARE_SHIFT
        m2 = m * m
    else:
        mantissa_host, exponent_host = math.frexp(value)
        m2 = torch.full((), mantissa_host * mantissa_host, dtype=acc.dtype, device=acc.device)
        e = torch.full((), exponent_host, dtype=acc.dtype, device=acc.device)
    s0, e0 = acc[0], acc[1]
    # The larger exponent of the two, a zero having none; the other term is scaled down to it. A zero or
    # a value that is not finite is left as it is (scaling it could make `0 * inf`, a NaN).
    top = torch.where(s0 == 0, e, torch.where(m2 == 0, e0, torch.maximum(e0, e)))
    plain0 = (s0 == 0) | ~torch.isfinite(s0)
    plain = (m2 == 0) | ~torch.isfinite(m2)
    a = torch.where(plain0, s0, torch.ldexp(s0, 2 * (e0 - top)))
    b2 = torch.where(plain, m2, torch.ldexp(m2, 2 * (e - top)))
    acc.copy_(torch.stack((a + b2, top)))


_SQUARE_SPLIT, _SQUARE_SHIFT = 2.0**60, 66.0
_SQUARE_SHRINK = 2.0 ** -_SQUARE_SHIFT


def _norm_of_squares(s: float, e: float) -> float:
    """The norm `_add_square`'s `[s, e]` stands for, `sqrt(s) * 2**e`, read on the host in float64."""
    root = math.sqrt(s)
    if root == 0 or not math.isfinite(root) or not math.isfinite(e):
        return root
    try:
        return math.ldexp(root, int(e))
    except OverflowError:  # past float64's range: the norms are finite, their total is not
        return math.inf


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


class _ClipCall:
    """What `record_grad_norm` writes as `Session._clip_calls` under `torch.compile`: a new object at each
    call, unequal to any count, so a wrapper that compares the count before and after sees the call."""

    __slots__ = ()


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
    stops at the first trainable parameter, a few in for such a model.

    A parameter frozen at `init` that is trainable now counts too (BUG-26): it joins the session's
    trainable ones only at the optimizer's step (`_take_unfrozen`), after the clipping, so on the
    step it was unfrozen it was missed and that step had no `clip_grad_norm`. Looked for only when
    no parameter matched otherwise, through the session's weak references, so an id another tensor
    took after a frozen parameter was freed never matches."""
    if isinstance(parameters, torch.Tensor):
        parameters = (parameters,)
    if not session._param_ids.isdisjoint(map(id, parameters)):
        return True
    return bool(session._frozen) and _clips_unfrozen(session, parameters)


def _clips_unfrozen(session: "Session", parameters: Any) -> bool:
    """Whether `parameters` holds one of `session`'s parameters frozen at `init` that is trainable now."""
    trainable = {}
    for _, ref in session._frozen:
        param = ref()
        if param is not None and param.requires_grad:
            trainable[id(param)] = param
    return bool(trainable) and any(trainable.get(id(p)) is p for p in parameters)


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
    probe_off: str | None  # why `probe=` was given and is not run (`_sharded`), said once
    params: list[tuple[str, torch.Tensor]]
    replicas: list[tuple[str, torch.Tensor]]
    selected: list[tuple[str, nn.Module]]
    routers: dict[str, Router]
    probe_blocks: list[tuple[str, nn.Module]]
    head: tuple[str | None, str | None, str | None]
    scaler: Any
    mirror_scalars: bool = False  # wrap the trackers' logging calls (`mirror.py`)


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
    ends_step: Any = None,
    auto_step: bool = True,
    mirror_scalars: bool = False,
    start_step: Any = 0,
) -> _Plan:
    """Every check `init` and `Session` make of the model and the options, and what they work
    out from them, with nothing changed: no hook, thread, file or global. `init` runs it before
    it resolves the run or opens anything, so an option it rejects leaves everything as it was.
    A layer filter, a `replica_params` filter and `get_output_embeddings` are each called once."""
    if hasattr(model, "_orig_mod"):
        raise ValueError("call tensorquill.init(model) before torch.compile(model)")
    # `log`'s rule for `step=` (BUG-05): a bool, a float or anything not an integer raises `TypeError`.
    if not isinstance(every, bool) and isinstance(every, numbers.Integral) and int(every) < 1:
        raise ValueError("every must be at least 1")  # the message it had
    _explicit_step(every, "every", least=1)
    _explicit_step(start_step, "start_step")
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
    if not isinstance(mirror_scalars, bool):
        raise ValueError(f"mirror_scalars must be True or False, not {type(mirror_scalars).__name__}")
    probe_fn = probe_callable(probe) if probe is not None else None
    probe_off = None
    if probe_fn is not None:
        # The author, Nov 24, 2026: on a sharded model the probe is off. Its forward would be a
        # collective, and a rank that skipped it (a stopped session) or left it part-way (a probe
        # that raised) leaves its peers' collectives to meet its next training ones: on two Gloo
        # ranks one process aborted and the other waited. Not an error: the rest is captured.
        how = _sharded(model)
        if how is not None:
            probe_fn = None
            probe_off = (
                f"the model is sharded across ranks ({how}), so a probe's forward would be a collective"
                " that every rank must enter"
            )
    if ends_step is not None:
        # Wave AE, section 3, this page's reading: refused here, never inside a step.
        if not isinstance(ends_step, torch.optim.Optimizer):
            raise ValueError(f"ends_step must be a torch.optim.Optimizer, not {type(ends_step).__name__}")
        model_ids = {id(p) for p in model.parameters()}
        if not any(id(p) in model_ids for group in ends_step.param_groups for p in group["params"]):
            raise ValueError("ends_step must be an optimizer that holds at least one of the model's parameters")
        if not auto_step:
            raise ValueError("ends_step cannot be combined with auto_step=False: session.step() ends each step there")
        if scaler is not None and scaler.is_enabled():
            raise ValueError("ends_step cannot be combined with scaler=: scaler.update() ends each step there")
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    # The parameters the cross-GPU drift check compares: every trainable parameter by
    # default (data parallelism), or only the replicated ones under tensor parallelism.
    replicas = _select_replica_params(params, replica_params)
    selected = _select_layers(model, layers, root=True)
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
        grad_every=grad_every, weight_every=weight_every, probe_fn=probe_fn, probe_off=probe_off, params=params,
        replicas=replicas,
        selected=selected, routers=planned_routers, probe_blocks=probe_blocks, head=head,
        scaler=scaler if scaler is not None and scaler.is_enabled() else None,
        mirror_scalars=mirror_scalars,
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
        ends_step: Any = None,
        mirror_scalars: bool = False,
        pinned_flush: bool = False,
        _plan: _Plan | None = None,
    ) -> None:
        # Every check first, with nothing changed (`_plan_session`); `init` has made them already.
        plan = _plan if _plan is not None else _plan_session(
            model, every=every, layers=layers, routers=routers, grads=grads, weights=weights,
            replica_every=replica_every, probe=probe, probe_every=probe_every, probe_head=probe_head,
            replica_params=replica_params, scaler=scaler, ends_step=ends_step, auto_step=auto_step,
            mirror_scalars=mirror_scalars, start_step=start_step,
        )
        # Checked by `_plan_session` (BUG-05); an integer that is not an `int` (NumPy's) is written as one.
        every, start_step = int(every), int(start_step)
        # The optimizer whose step ends each step (`init(ends_step=...)`, wave AE section 3), weakly
        # held, or None for the default lead; and the steps of this model's other optimizers before it
        # first stepped, for the one warning that it has not (`_before_optimizer_step`).
        self._ends_step: weakref.ref[Any] | None = weakref.ref(ends_step) if ends_step is not None else None
        self._ends_step_waiting = 0
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
        # `probe=` was given and is not run: written once, in place of a probe's results, on the
        # first step's record (`_commit`), and said once below.
        self._probe_off = plan.probe_off
        if self._probe_off is not None:
            said = (
                f"TensorQuill: the probe is off for this session: {self._probe_off}. A rank that skipped it or"
                " left it part-way would leave the ranks' collectives mismatched. Everything else is captured as"
                " usual; the record of the first step says so under `probe`, and the checks that read the probe"
                " are reported as unavailable."
            )
            _warn(said, stacklevel=4)  # never raises, whatever the warning filters (`_warn`)
        self._model = model
        self.auto_step = auto_step
        self._sink = sink
        self._replica_group = replica_group

        params = plan.params
        self._params = [p for _, p in params]
        self._param_ids = {id(p) for p in self._params}
        self._param_index = {id(p): index for index, p in enumerate(self._params)}
        self.param_names = [name for name, _ in params]
        # Every parameter of the model at `init`, frozen ones included: an optimizer that holds any
        # of them is this model's (`_owns`), so one that holds only frozen parameters still ends
        # steps (TrainCheck B1, and a model whose every parameter is frozen at `init`).
        everything = list(model.named_parameters())
        self._model_param_ids = {id(p) for _, p in everything}
        # The parameters frozen at `init`, recorded from the step on which they are trainable
        # (C12, `_take_unfrozen`). Weakly held: one the model drops is not kept alive.
        recorded = {id(p) for p in self._params}
        self._frozen: list[tuple[str, weakref.ref[torch.Tensor]]] = [
            (name, weakref.ref(p)) for name, p in everything if id(p) not in recorded
        ]
        self._frozen_ids = {id(p) for _, p in everything if id(p) not in recorded}  # kept whole: `_say_new_trainable`
        # The parameters the cross-GPU drift check compares (`_plan_session`).
        replicas = plan.replicas
        self._replica_params = [p for _, p in replicas]
        self.replica_names = [name for name, _ in replicas]
        # With no trainable parameter at `init`, the model's own: one unfrozen later then has its
        # norms read where it is, not moved to the CPU at every step.
        self.device = _buffers_device(self._params or [p for _, p in everything])

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
        self._said_lock = threading.Lock()  # the said set is read and added to by the writer too (pinned path)
        # On the pinned path only: each router's first reason, kept by the writer, which judges the index
        # ranges there (`_settle`); the hook's own list stays the training thread's.
        self._router_first: dict[str, tuple[str, str | None]] = {}
        self._router_refusals_written: set[str] = set()
        # Each router's `moe_source` once it is known and written (`_moe_forms`, contract part 7).
        self._moe_source_written: set[str] = set()

        n_layers, n_params = len(self.layer_names), len(self._params)
        # What each watched block's hook found it could not read (`_stats_hook`), the blocks read on any
        # step so far, and those already named under `layers_not_read` (`_layers_not_read`).
        self._layer_problems: list[list[str]] = [[] for _ in range(n_layers)]
        self._layers_ever_read: set[int] = set()
        self._layers_not_read_written: set[str] = set()
        f32 = {"dtype": torch.float32, "device": self.device}
        self._act = torch.zeros(n_layers, _SEEN + 1, **f32)
        self._stat_idx = torch.arange(_SEEN + 1, dtype=torch.int64, device=self.device)
        # gradient norm, seen, and with the weight check on, weight norm, weight unchanged since the
        # previous record (1 = exactly unchanged, 0 = changed, -1 = not known). Only the columns of the
        # signals that are on (lane AD3): without the weight check its two columns are not made, so
        # `_decode` reads the width from the block, never from this session.
        self._grad = torch.zeros(n_params, 4 if weight_every > 0 else 2, **f32)
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
        self._spread_shape_said = False  # a precomputed table of the wrong shape was warned of
        self._counts = {
            name: torch.zeros(_count_width(spec), dtype=torch.int64, device=self.device)
            for name, spec in self._routers.items()
        }
        self._ring_act = torch.zeros(every, n_layers, _SEEN + 1, **f32)
        # The ring's gradient rows. In the router-only mode, which reads no gradient on its own, none
        # are made until a step whose gradients were read by hand (`record_grads`) is committed
        # (`_grad_ring`); before lane AD3 they were `every` rows of four floats a parameter whatever was on.
        self._ring_grad: torch.Tensor | None = (
            torch.zeros(every, *self._grad.shape, **f32) if grad_every > 0 or weight_every > 0 else None
        )
        self._ring_grad_seen: list[bool] = []
        self._ring_weight_seen: list[Any] = []  # False, or the weight check's optimizer ownership and dtypes
        # Squared total gradient norm reported by gradient clipping this step (summed over
        # calls, for models clipped in several groups), and its ring: `[s, e]` for `s * 4**e`
        # (`_add_square`), so a finite norm of any float64 size stays finite (BUG-04).
        self._clip = torch.zeros(2, **f32)
        self._ring_clip = torch.zeros(every, 2, **f32)
        self._clip_recorded = False
        self._foreign_steps = 0
        # Lets an outer wrapper see that an inner one already recorded: a count, or a `_ClipCall` written
        # under `torch.compile`; compared for equality only.
        self._clip_calls: int | _ClipCall = 0
        # The eager calls of `record_grad_norm` since the last step ended (`clip_calls`, `_commit`).
        self._clip_step_calls = 0
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
        # The replica observation (contract, section 1): written on each scheduled step by a session
        # given a replica group, from this rank's own parameters, with no collective and no store.
        # The group's global ranks are read here, on the host; a group whose ranks cannot be read
        # (a stand-in object, a rank outside it) gets no observation, said once.
        self._obs_group: list[int] | None = None
        self._obs_seeds = [name_v1_seed(name) for name in self.replica_names]
        self._ring_obs: torch.Tensor | None = None
        self._ring_obs_sig: torch.Tensor | None = None
        self._ring_obs_seen: list[Any] = []  # None, or what the row's observation needs on the host
        self._obs_unread_said: set[str] = set()
        self._obs_failed_said = False
        if replica_group is not None and not isinstance(replica_group, torch.Tensor) and self._replica_params:
            self._obs_group, why = group_ranks(replica_group)
            if self._obs_group is None:
                _warn(
                    f"TensorQuill writes no replica observation for this session: {why}, so the records "
                    "could not say which ranks it is to be compared with. The explicit calls are unchanged.",
                    stacklevel=4,
                )
        # What an explicit gather covers (contract 1, `replica_spread_scope`): the group's global
        # ranks for this step, kept per step; the launch is added when the record is written.
        self._spread_scope: list[int] | None = None
        self._ring_scope: list[list[int] | None] = []
        # The launch the records carry (contract 2). Rank 0's own, at once. On a rank other than 0,
        # what `init` read is provisional: the second read (`LaunchRead`) starts at the end of the
        # first step, on a thread of its own, and until it returns the records carry null, never
        # what `init` read (`_launch_for_records`).
        self.launch: str | None = run.launch if run.rank == 0 or run.launch_source is None else None
        self._launch_read: LaunchRead | None = None
        self._launch_parked = False
        if run.rank != 0 and run.launch_source is not None:
            # The reader's thread starts here, parked, so the end of the first step only lets it go.
            # It reads the session's run when it goes, through a weak reference: it keeps no session alive.
            this = weakref.ref(self)
            try:
                self._launch_read = LaunchRead(read_run=lambda: getattr(this(), "run", None))
                self._launch_parked = True
            except Exception:  # noqa: BLE001
                self._launch_read = None
        self._launch_confirmed = False
        self.launch_read_seconds: float | None = None
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
        for buf in (self._act, self._stat_idx, *self._counts_list):
            _mark_static(buf)

        # Forwards that are not training (evaluation, the probe) are left out by the hooks
        # (`_evaluating`); this says whether the session's phase is "eval" (`set_phase`).
        self._evaluating = _Evaluating()
        # Each watched block's and router's held forwards (`_skip_forward`), set back at each step.
        self._held_forwards = [_Held() for _ in range(len(plan.selected) + len(self._routers))]
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
        # Under `auto_step=False`: `session.step()` ended a step while the scaler still held an
        # optimizer's state that its `update()` had not read, so the verdict `update()` gives next
        # is of an update already ended (`_record_scaler`, `_SCALER_LATE`).
        self._scaler_verdict_late = False
        self._pending_scalars: dict[str, Any] = {}
        self._late_said = False  # a scalar left out for want of a step in memory was said (`_no_step_for`)
        # The mirror's mark (`mirror._log_into`, section Q5): set while it logs a value it brought, and
        # the names it left pending, with the count of steps ended when it did.
        self._mirroring = False
        self._mirrored_pending: tuple[int, set[str]] = (0, set())
        self._late_traced: dict[str, Any] = {}  # logged late inside a compiled function (`_log_late`)
        # Scalar-only records (`log(..., step=N)`) not yet handed to the writer, their values as
        # logged (a tensor still on its device), and the next one's `seq`, counted from 0 per session.
        self._scalar_records: list[dict[str, Any]] = []
        self._scalars_unread_said = False  # a scalar-only record's value left out (`_scalar_rows_one_by_one`)
        self._scalar_seq = 0
        self._ring_probe: list[dict[str, Any] | None] = []
        self._probe_off_written = False
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
        # Each optimizer seen stepping: whether it holds any of this model's parameters, and the size
        # of each of its parameter groups when that was judged (`_owns`).
        self._owned: weakref.WeakKeyDictionary[Any, tuple[bool, tuple[int, ...]]] = weakref.WeakKeyDictionary()
        # Parameters found trainable after `init` and already named in a warning (`_say_new_trainable`).
        self._new_trainable_said: set[str] = set()
        self._trainable_scan_at = 0  # the committed step count from which the next scan is due
        self._lead: weakref.ref[Any] | None = None
        self._leading = True
        # Telling a replaced lead (L9, `_takes_lead`): the optimizers that are not the lead and have
        # stepped since it last did; one parameter of the lead, its gradient and that gradient's
        # version at the lead's last step, and which lead chose it; whether an optimizer holds any
        # parameter the lead holds, with the sizes that was judged at.
        self._since_lead: weakref.WeakSet[Any] = weakref.WeakSet()
        self._lead_mark: tuple[Any, Any, int | None] | None = None
        self._lead_mark_of: Callable[[], Any] = lambda: None
        self._shares_lead: weakref.WeakKeyDictionary[Any, tuple[Any, bool]] = weakref.WeakKeyDictionary()
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
        # The process that made the session (SDK-07): a child forked from it inherits the session
        # but not its writer, and the records file and its end marker are the parent's (`close`).
        self._pid = os.getpid()
        self._phase = "train"
        self._ticks = 0
        self._error: BaseException | None = None
        # What the stop says when `_error` came from the pinned path's settle, not the sink: (what, reason).
        self._error_said: tuple[str, str] | None = None
        # The pinned-memory flush (`init(pinned_flush=True)`, lane AQ-1; see `_CudaCopies`). The mode is the
        # session's, fixed here: with it, every block is settled on the writer, in queue order, even after
        # the copy itself has fallen back to a blocking read (`pinned_flush` turns False then, said once).
        self._pinned_mode = False
        self.pinned_flush = False
        self._copier: Any = None
        self._pool: _PinnedPool | None = None
        self._update_why_late: dict[str, str] = {}  # the writer's `UPDATES_NO_OPTIMIZER` note (pinned path)
        self._said_later: collections.deque[str] = collections.deque()  # warnings the writer found
        self._pinned_off_said = False
        if pinned_flush:
            try:
                self._copier = _pinned_backend(self.device)
            except Exception as error:  # noqa: BLE001  never into training: the blocking path, said once
                self._say_pinned_off(error, now=True)
            else:
                self._pinned_mode = self.pinned_flush = True
        # GPU memory and NVML readings once per flush (system.py), and the host time spent
        # waiting for batches (datawait.py). Both off unless asked for.
        self._system = SystemReadings(self.device) if system else None
        self._data_timing = False
        self._data_thread = threading.get_ident()
        self._fetching = False
        self._pending_data_wait = 0.0
        self._last_step_clock: float | None = None
        # Time spent in a phase other than "train" since the last step ended (`set_phase`), left out
        # of `step_time`, and when the pause now under way began (None outside one).
        self._paused_seconds = 0.0
        self._pause_began: float | None = None
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
        forward hooks, the heartbeat's step counter, the writer thread, the clipping and scaler
        hooks, the scalar mirror, the optimizer hooks and the data-loader timer; last, the end
        marker of an earlier run is removed, the one effect that is not undone, so it happens only
        once all else has."""
        undo.append(lambda: [handle.remove() for handle in self._handles])
        self._handles += [
            module.register_forward_hook(
                _stats_hook(self._act, self._stat_idx, row, self._evaluating, self._layer_problems[row],
                            self._held_forwards[row])
            )
            for row, (_, module) in enumerate(plan.selected)
        ]
        self._handles += [
            modules[name].register_forward_hook(
                _router_hook(
                    self._counts[name], spec, self._router_problems[name], self._evaluating,
                    self._held_forwards[len(plan.selected) + i]
                )
            )
            for i, (name, spec) in enumerate(self._routers.items())
        ]
        if self._probe_fn is not None:
            self._handles += install_probe_hooks(plan.probe_blocks, self._probe_state)
            head = plan.head[0]
            if head is not None:
                self._handles += install_head_hook(modules[head], self._probe_state)

        sender = getattr(self._sink, "sender", None)
        if sender is not None and hasattr(type(sender), "attempt"):
            # Contract 1: every batch the sender sends names this session's attempt (`sender_attempt`).
            # Named before the heartbeat's hand-over, which releases the stream's first connection.
            sender_attempt_before = sender.attempt
            sender.attempt = self.attempt
            undo.append(lambda: setattr(sender, "attempt", sender_attempt_before))
        beat = getattr(self._sink, "heartbeat", None)
        if beat is not None:
            had, before = hasattr(beat, "counter"), getattr(beat, "counter", None)
            had_attempt, attempt_before = hasattr(beat, "attempt"), getattr(beat, "attempt", None)

            def give_back() -> None:
                if had:
                    beat.counter = before
                else:
                    vars(beat).pop("counter", None)
                if had_attempt:
                    beat.attempt = attempt_before

            # The counter and this session's own attempt go over together (`_attempt_handed_over`).
            _handing_over.attempt = self.attempt
            try:
                beat.counter = lambda: self._steps_committed + self._ticks
            finally:
                _handing_over.attempt = None
            undo.append(give_back)

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
        if plan.mirror_scalars:
            from . import mirror  # the trackers' `log` wrappers, shared by every session that mirrors

            mirror.watch(self)
            undo.append(lambda: mirror.unwatch(self))
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
        # The global hooks fire for every optimizer in the process. What each holds is kept with the
        # size of each of its parameter groups, and judged again when those change: an optimizer
        # given this model's parameters by `add_param_group` after its first step ends steps then.
        # Any of the model's parameters at `init` counts, frozen ones included (TrainCheck B1).
        groups = optimizer.param_groups
        sizes = tuple(len(group["params"]) for group in groups)
        found = self._owned.get(optimizer)
        if found is not None and found[1] == sizes:
            return found[0]
        owned = any(id(p) in self._model_param_ids for group in groups for p in group["params"])
        self._owned[optimizer] = (owned, sizes)
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
            if ids & self._model_param_ids:
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

    def _note_foreign_step(self, optimizer: Any = None) -> None:
        # DeepSpeed's ZeRO optimizers and Megatron's mixed-precision optimizer step copies
        # of the weights, so the optimizer that steps holds none of the model's
        # parameters and no step would ever be counted. Say so instead of writing nothing.
        if self._lead is not None or self._foreign_steps < 0:
            return
        if self._foreign_steps == 0 and optimizer is not None and self._replaced_parameters(optimizer):
            # Lane AD3: a session made before `to_empty` (a model built on meta) holds the meta
            # parameters, which `to_empty` replaces with new ones; the optimizer, made after, steps the
            # new ones. Said at the first step, with the cause, instead of after ten with another.
            self._foreign_steps = -1
            _warn(
                "TensorQuill is not counting this optimizer's steps: it holds parameters the model has now "
                "that replaced the ones it had when tensorquill.init ran (to_empty on a model made on meta, or "
                "a parameter assigned after init), and the session reads the old ones, so no step is recorded "
                "until an optimizer of those steps. Call tensorquill.init after to_empty, once the parameters "
                "are materialised.",
                stacklevel=3,
            )
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

    def _replaced_parameters(self, optimizer: Any) -> bool:
        """Whether `optimizer` holds a parameter the model has now that it did not have at `init`: one
        pass over the model's parameters, on the host, at the first step of an optimizer that holds
        none of the session's, once a session. Never raises."""
        try:
            now = {id(p) for p in self._model.parameters()} - self._model_param_ids
            return bool(now) and any(id(p) in now for group in optimizer.param_groups for p in group["params"])
        except Exception:  # noqa: BLE001
            return False

    def _before_optimizer_step(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        if self._closed:
            return
        if not self._owns(optimizer):
            self._foreign_stepped = True
            if not self._stopped:
                self._note_foreign_step(optimizer)
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
        # The lead is replaced (L9) by an optimizer that holds some of the parameters it holds and
        # steps in a later iteration than the lead last did: an LR finder's optimizer followed by
        # the real one, or an optimizer rebuilt while the old one is still referenced. Before, the
        # first lead kept leading while it lived, and the new one's steps were never counted.
        lead = self._lead() if self._lead is not None else None
        if self._ends_step is not None:
            # `init(ends_step=...)`: that optimizer leads, from its first step, and no other ever does;
            # nothing follows it (`_after_optimizer_step`). Another optimizer's step belongs to the step
            # the named one ends next, so its update waits in the pending buffer until then.
            ender = self._ends_step()
            if ender is not None and optimizer is ender:
                if lead is not ender:
                    self._lead, lead = weakref.ref(ender), ender
            elif lead is None and self._ends_step_waiting >= 0 and not self._stopped:
                self._ends_step_waiting += 1
                if self._ends_step_waiting == FOREIGN_STEPS_WARNING:
                    self._ends_step_waiting = -1  # warn once
                    _warn(
                        f"TensorQuill has recorded no steps: {FOREIGN_STEPS_WARNING} steps of optimizers that hold "
                        "this model's parameters ran, and the optimizer passed as ends_step= has not stepped. A "
                        "step ends only when that optimizer's step returns; nothing is recorded until it does.",
                        stacklevel=2,
                    )
            self._leading = ender is not None and optimizer is ender
        elif lead is None:
            self._lead, lead = weakref.ref(optimizer), optimizer
        elif optimizer is not lead and self._takes_lead(optimizer, lead):
            self._lead, lead = weakref.ref(optimizer), optimizer
            self._followers = False  # the old lead's followers; a follower of the new one is found again
        if self._ends_step is None:
            self._leading = optimizer is lead
        if self._leading:
            self._since_lead.clear()
        else:
            try:
                self._since_lead.add(optimizer)
            except TypeError:  # not weakly referenceable: only the backward signal can tell
                pass
        if self._stopped:
            return  # only the step count goes on, for the drift check's schedule (`_count_stopped`)
        if self._leading and self._frozen and (self.updates or self._grads_wanted()):
            self._take_unfrozen()  # before this step's gradients are read; on steps that read none, nothing to record
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
        if (
            not self._leading and self._ends_step is None
            and callable(kwargs.get("closure", args[1] if len(args) > 1 else None))
        ):
            # Under `step(closure)` the backward runs inside the step, after `_before_optimizer_step`
            # judged whether this optimizer replaces the lead (L9): judged again now that it has run.
            lead = self._lead() if self._lead is not None else None
            if lead is not None and self._backward_since_lead() and self._shares_with_lead(optimizer, lead):
                self._lead, self._leading, self._followers = weakref.ref(optimizer), True, False
                self._since_lead.clear()
        if self._leading:
            self._mark_lead_grad(optimizer)
        if self._stopped:
            if self._leading and not self._commit_on_update:
                self._count_stopped(None)
            return
        # An optimizer that steps after the lead ended the step (two optimizers on one model,
        # `opt1.step(); opt2.step()`: Muon and AdamW, a GAN's two halves in one module) belongs to
        # that step, which is still the newest in the ring (when it filled the ring, `_commit` kept
        # it back): what it changes is written on that step's record, not the next one's. Under a
        # loss scaler, `scaler.update()` ends the step after every optimizer has stepped, so
        # nothing follows there; nor under `ends_step`, where another optimizer's step belongs to the
        # step that ends next, so what it brings stays pending until then.
        follows = not self._leading and not self._commit_on_update and bool(self._steps) and self._ends_step is None
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

    def _takes_lead(self, optimizer: Any, lead: Any) -> bool:
        """Whether `optimizer`, which is not the lead, replaces it from this step on (L9): it holds a
        parameter the lead holds, and it steps in a later iteration than the lead last did. That is
        known when a backward has run since the lead's last step (`_backward_since_lead`), or when
        this optimizer already stepped since then. Two optimizers stepped one after the other in one
        iteration (a follower, `_after_optimizer_step`) meet neither, and ones that hold none of each
        other's parameters (two halves of a model, a GAN's two players) never replace each other, so
        both stay as they were.

        Under `step(closure)` the backward runs inside the step, after this is judged, so it is
        judged again after the step (`_after_optimizer_step`), before the step is ended. On the
        host, no device work: a check of one gradient's identity and version on every step of an
        optimizer that is not the lead, and a pass over two optimizers' parameter lists when their
        groups change."""
        try:
            again = optimizer in self._since_lead
        except TypeError:
            again = False
        if not (again or self._backward_since_lead()):
            return False
        return self._shares_with_lead(optimizer, lead)

    def _mark_lead_grad(self, optimizer: Any) -> None:
        """After the lead's step: one of the model's parameters it holds that has a gradient, that
        gradient and its version, for `_backward_since_lead`. The parameter is kept while it has a
        gradient at the lead's step, and chosen again when it has none (a frozen one never does,
        and a fine-tuned model's first parameters are often frozen); with none, nothing is marked."""
        try:
            chosen = self._lead_mark[0]() if self._lead_mark is not None else None
            if chosen is None or chosen.grad is None or self._lead_mark_of() is not optimizer:
                chosen = next(
                    (
                        p for group in optimizer.param_groups for p in group["params"]
                        if id(p) in self._model_param_ids and p.grad is not None
                    ),
                    None,
                )
                if chosen is None:
                    self._lead_mark = None
                    return
                self._lead_mark_of = weakref.ref(optimizer)
            grad = chosen.grad
            self._lead_mark = (weakref.ref(chosen), weakref.ref(grad), grad._version)
        except Exception:  # noqa: BLE001
            self._lead_mark = None

    def _backward_since_lead(self) -> bool:
        """Whether the gradient marked at the lead's last step (`_mark_lead_grad`) has changed since:
        a new gradient (`zero_grad()` sets it to None, a backward makes another), or the same one
        written in place (`zero_grad(set_to_none=False)`, or a backward that accumulates into it)."""
        mark = self._lead_mark
        if mark is None:
            return False
        try:
            param, was, version = mark[0](), mark[1], mark[2]
            if param is None:
                return False
            grad = param.grad
            if was is None:
                return grad is not None
            if grad is None or grad is not was():
                return True
            return grad._version != version
        except Exception:  # noqa: BLE001
            return False

    def _shares_with_lead(self, optimizer: Any, lead: Any) -> bool:
        """Whether `optimizer` holds a parameter `lead` holds. Kept for the pair and the sizes of
        both optimizers' parameter groups, and judged again when those change (as `_owns` is)."""
        key = (
            id(lead),
            tuple(len(group["params"]) for group in lead.param_groups),
            tuple(len(group["params"]) for group in optimizer.param_groups),
        )
        try:
            found = self._shares_lead.get(optimizer)
        except TypeError:
            found = None
        if found is not None and found[0] == key:
            return found[1]
        held = {id(p) for group in lead.param_groups for p in group["params"]}
        shared = any(id(p) in held for group in optimizer.param_groups for p in group["params"])
        try:
            self._shares_lead[optimizer] = (key, shared)
        except TypeError:
            pass
        return shared

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
            # Through `_grad_ring`: in the router-only mode the ring is made only once it is needed
            # (SDK-01's second half; indexed directly, a ring not made yet would raise here).
            self._grad_ring()[len(self._steps) - 1, :, 2:].copy_(self._grad[:, 2:])
            self._grad[:, 2:].zero_()
            self._mark_held("grad")
            self._ring_weight_seen[-1] = {"ownership": self._optimizer_ownership(), "dtypes": self._weight_dtypes}
        if first and self._probe_fn is not None and self._ring_probe[-1] is not None:
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
            and self._ends_step is None and bool(self._steps) and not self._closed and not self._stopped and self._owns(optimizer)
        )

    def _add_to_newest(self, name: str, value: Any) -> None:
        """Add `value` to the scalar `name` of the newest step in the ring (or set it there): the
        loss back-propagated for an optimizer that steps after the lead ended the step."""
        new = self._scalar_values({name: value})[name]  # the same checks and the same copy on the device as `log`
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
        verdicts = [
            list(state["found_inf_per_device"].values()) for state in scaler._per_optimizer_states.values()
            if state["found_inf_per_device"]
        ]
        if len(verdicts) > 1:
            # Several optimizers through this scaler (wave AE, section 4): how many of them it skipped,
            # each judged over all its devices, as `scaler.step` judges it. On the device until the flush,
            # with `step_skipped`: no sync. One optimizer adds nothing, so its records are as before.
            per_optimizer = torch.stack([
                torch.stack([f.to(scale.device).reshape(()) for f in fs]).sum() > 0 for fs in verdicts
            ])
            self._pending_amp[_SKIPPED_OPTIMIZERS] = per_optimizer.sum().float()
            self._pending_amp[_SCALED_OPTIMIZERS] = len(verdicts)
        if self._scaler_verdict_late:
            self._scaler_verdict_late = False
            self._pending_amp[_SCALER_LATE] = True
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

        The parameters read are those trainable at `init`, and, unless the session is in the
        router-only mode, one frozen at `init` from its first trainable step. In the router-only
        mode a parameter unfrozen after `init` is not read here, and nothing says so (SDK-01).
        """
        if self._closed_call("record_grads()"):
            return
        if self._frozen:
            self._take_unfrozen()  # called by hand (`auto_step=False`), or by a scaler's update
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

    def _grad_ring(self) -> torch.Tensor:
        """The ring's gradient rows, made the first time a step with gradients read is committed in
        the router-only mode (`record_grads` called by hand); made at `init` in every other mode."""
        if self._ring_grad is None:
            self._ring_grad = torch.zeros(self.every, *self._grad.shape, dtype=self._grad.dtype, device=self._grad.device)
        return self._ring_grad

    def _grad_not_read(self, index: int, why: str) -> None:
        """Keep why parameter `index`'s gradient was not read, for `grads_not_read` (`_flush`)."""
        self._grad_why.setdefault(self.param_names[index], why)

    def _new_grad_why(self) -> dict[str, str]:
        """The parameters whose gradients were found unreadable since the last record that said so."""
        found = {name: why for name, why in self._grad_why.items() if name not in self._grad_why_written}
        self._grad_why_written.update(found)
        return found

    def _index(self, key: tuple[int, ...]) -> torch.Tensor:
        """The positions `key` as an index tensor on the session's device. The `INDEX_CACHE_SIZE`
        sets used most recently are kept; one not kept is made again, a copy of a few integers."""
        cache = self._grad_idx_cache
        idx = cache.pop(key, None)
        if idx is None:
            idx = torch.tensor(key, dtype=torch.int64, device=self._grad.device)
            if len(cache) >= INDEX_CACHE_SIZE:
                del cache[next(iter(cache))]  # the one used least recently
        cache[key] = idx
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
        # (on MPS beside buffers elsewhere) gets its norm and "not known", and so does one with no
        # elements (an empty local shard), whose norm is 0.
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
                # No values here (an empty local shard under FSDP): its norm is 0, and whether it
                # changed is not known, so it is never named unchanged.
                inexact.append(index)
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
        # Not known where this check or the one before did not read a parameter exactly, nor where
        # either norm is not finite: an infinite norm equal to an infinite one compares nothing, and
        # a parameter is flagged 1 only where its norms were compared (contract 2; `weights` in
        # `init` says what the flag asserts). On the device, over one number per parameter.
        blind = set(unread) | set(inexact)
        before, self._weights_not_exact = getattr(self, "_weights_not_exact", frozenset()), frozenset(blind)
        if not self._exact_weights or self._prev_weight is None:
            self._grad[:, 3].fill_(-1.0)
        else:
            compared = torch.isfinite(current) & torch.isfinite(self._prev_weight)
            same = (current == self._prev_weight).to(self._grad.dtype)
            self._grad[:, 3].copy_(torch.where(compared, same, torch.full_like(same, -1.0)))
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
            _add_square(self._clip, value.reshape(()))
        else:
            _add_square(self._clip, float(norm))
        self._clip_recorded = True
        if torch.compiler.is_compiling():
            # Written, never read, while traced (lane AD3): `+= 1` read the count, and Dynamo guarded
            # on its value, so a compiled step that clips recompiled at every step until its
            # recompile limit and then ran eagerly. A new marker still tells a wrapper that compares
            # the count before and after a call (`accelerator._clip_grad_norm`) that this call recorded.
            self._clip_calls = _ClipCall()
        else:
            calls = self._clip_calls
            self._clip_calls = calls + 1 if type(calls) is int else 1
            # This step's own count, for `clip_calls` on its record (wave AE, section 1). Eager calls
            # only: under `torch.compile` nothing new is written, so a compiled step counts none and
            # its record has no key, as the count is not known there.
            self._clip_step_calls += 1

    def log(self, values: Mapping[str, Any] | None = None, *, step: Any = None, **named: Any) -> None:
        """Record scalars such as the loss for the step in progress, without a GPU sync.

        `session.log({"train/loss": loss})` or `session.log(loss=loss)`. Values are
        Python numbers or one-element tensors. A tensor that holds no values to read, judged from
        its device and layout alone (on `meta`, nested, sparse), raises `ValueError` before anything
        is recorded. A tensor is copied on its device and moved
        to the CPU with the rest of the step data every `every` steps, so logging a CUDA
        loss does not stall training. Call it before `optimizer.step()`: with automatic
        step counting, the step ends there, and a value logged afterwards belongs to the
        next step. A name logged twice in one step keeps the last value. Images, tables
        and other non-scalar values raise `TypeError`; pass the tracker the same values
        it already logs, not a replacement for it.

        Under `session.phase("eval")` (or `set_phase("eval")`, which the Lightning and Hugging
        Face callbacks set) a value is recorded on the newest step, the one the evaluation
        followed, while that step is still in memory: it is the ring's last row, or the step
        kept back when the ring last filled, until the next step ends, a `flush()` or `close()`.
        A value still pending at `close()` goes there too, unless that step already holds its
        name. With no step in memory (before the first step, or after `flush()` with no step
        since) it is left out, said once per session in a warning that shows the call below
        (`_no_step_for`). A record already written is never changed.

        `step=N` records the values at step `N` of their own: a scalar-only record (contract
        section 4.2), never a training step, whatever the phase and whether or not a step is in
        memory. `N` is an integer of 0 or more. `step=None` is the same as leaving `step=` out.
        A bool (`True`, `False`), a float (`1.0` too), a string or anything else that is not an
        integer raises `TypeError`, and a negative integer `ValueError`, before anything is
        recorded. A tensor value is copied on its device and read with the block that writes the
        record, so the call adds no sync. The record is written with the next block of step
        records, at an explicit `flush()` (also with no step in memory), or at `close()`; the
        writer thread writes it, as it writes step records. Records of one session are numbered
        `seq` 0, 1, 2... in the order of these calls, and a reader takes the highest `seq` where
        two meet at a step. A key `"step"` in the dictionary is an ordinary scalar.

        The record is made eagerly. Inside a function compiled with `torch.compile(fullgraph=True)`,
        `step=N` therefore raises Dynamo's `torch._dynamo.exc.Unsupported` before anything is
        recorded. Call it outside the compiled function, which is where a value logged outside a
        step belongs, or compile without `fullgraph`: the call is then one graph break, compiled
        once. A call without `step=` (or with `step=None`) is traced as before.
        """
        if self._closed_call("log()"):
            return
        if step is not None:  # `None` is no step (the owner, Nov 24); traced as a constant under compile
            self._log_at(step, dict(values or {}, **named))
            return
        converted = self._scalar_values(dict(values or {}, **named))
        # The phase as the hooks read it (`_Evaluating`): a compiled step that logs guards on that
        # one flag, which its hooks already read, and not on the phase's name.
        if self._evaluating.on:
            self._log_late(converted)
        else:
            self._pending_scalars.update(converted)

    @torch.compiler.disable
    def _log_at(self, step: Any, values: Mapping[str, Any]) -> None:
        """`log(values, step=step)`: one scalar-only record, its `seq` and `ts` taken at the call, after
        every check. A call with no values writes nothing and takes no `seq`.

        Always run eagerly: inside a compiled function it is one graph break, and neither the step, nor
        `seq`, nor the pending list is a guard, so a new step on every call compiles nothing again
        (without it each call recompiled, until Dynamo's limit). Under `fullgraph=True` it cannot be
        traced, a stated limit: Dynamo raises its `Unsupported` there, before anything is recorded."""
        step = _explicit_step(step)
        converted = self._scalar_values(values)
        if not converted:
            return
        seq, self._scalar_seq = self._scalar_seq, self._scalar_seq + 1
        self._scalar_records.append({
            "schema_version": SCALARS_SCHEMA_VERSION,
            "record": "scalars",
            "run_id": self.run.run_id,
            "rank": self.run.rank,
            "attempt": self.attempt,
            "step": step,
            "seq": seq,
            "ts": time.time(),
            "scalars": converted,
        })

    @staticmethod
    def _scalar_values(items: Mapping[Any, Any]) -> dict[str, Any]:
        """`log`'s checks and copies: Python numbers as `int` or `float`, a one-element tensor copied
        on its own device (never read). Raises before anything is recorded."""
        out: dict[str, Any] = {}
        for name, value in items.items():
            if not isinstance(name, str) or not name:
                raise TypeError(f"scalar names must be non-empty strings, got {name!r}")
            if isinstance(value, torch.Tensor):
                tensor = value.detach()
                if type(tensor) is not torch.Tensor:
                    tensor = unwrap_local(tensor)
                unread = _value_unread(tensor)
                if unread is not None:
                    raise ValueError(f"{name}: {unread}")
                if tensor.numel() != 1:
                    raise ValueError(f"{name}: expected one value, got shape {tuple(tensor.shape)}")
                # A copy, so the value neither keeps a larger tensor alive nor changes later.
                dtype = torch.float32 if tensor.device.type == "mps" else torch.float64
                out[name] = tensor.reshape(()).to(dtype, copy=True)
            elif isinstance(value, numbers.Real):
                out[name] = int(value) if isinstance(value, numbers.Integral) else float(value)
            else:
                raise TypeError(f"{name}: expected a number or one-element tensor, got {type(value).__name__}")
        return out

    def _newest_scalars(self) -> dict[str, Any] | None:
        """The scalars of the newest step still in memory (the ring's last row, or the step kept back
        when the ring filled, which is then its only row), or None when no step is."""
        return self._ring_scalars[-1] if self._steps else None

    def _scalars_for_log(self) -> tuple[tuple[str, int], dict[str, Any] | None]:
        """Where `log` records a value now, for the mirror's explicit-wins rule (`mirror._log_into`):
        a key naming those scalars, unchanged until a step ends, and the dictionary itself, or None
        when the value would be left out."""
        if self._evaluating.on:
            self._take_traced_late()
            return ("newest", self._steps_committed), self._newest_scalars()
        return ("pending", self._steps_committed), self._pending_scalars

    def _log_late(self, values: dict[str, Any]) -> None:
        """Values logged after the newest step ended (under the "eval" phase): on that step while it is
        in memory, a name logged again keeping the last value. A held step's row is replaced, not
        changed in place, and marked (`_mark_held`), so its scalars are read again, in one transfer,
        only when it is written; a tensor is never read here."""
        if not values:
            return
        if torch.compiler.is_compiling():
            # Traced: the ring is not read (its length, its steps and the held step would each be a
            # guard that changes at every step, and the function would compile again). Kept here and
            # put on the newest step by the next eager call (`_take_traced_late`).
            self._late_traced.update(values)
            return
        self._take_traced_late()
        row = self._newest_scalars()
        if row is None:
            self._no_step_for(values, "log")
            return
        self._ring_scalars[-1] = {**row, **values}
        self._mark_held("scalars")

    def _take_traced_late(self) -> None:
        """Put on the newest step what a compiled function logged late (`_log_late`), eagerly: at the
        next eager `log` under "eval", the mirror's next value, the next step's end, `flush()` and
        `close()`. Emptied in place, so the traced function's guard on it holds."""
        traced = self._late_traced
        if traced:
            values = dict(traced)
            traced.clear()
            self._log_late(values)

    def _attach_pending(self) -> None:
        """At `close()`: what was logged since the last step ended goes on the newest step in memory,
        but for a name that step already holds (a loop that logs its loss after `optimizer.step()`
        has every loss one step late, and the last one pending: it does not replace the step's own)."""
        pending, self._pending_scalars = self._pending_scalars, {}
        if not pending:
            return
        row = self._newest_scalars()
        if row is None:
            self._no_step_for(pending, "close")
            return
        added = {name: value for name, value in pending.items() if name not in row}
        if added:
            self._ring_scalars[-1] = {**row, **added}
            self._mark_held("scalars")

    def _log_mirrored(self, values: dict[str, Any]) -> None:
        """`log(values)` for a value the mirror brought from a tracker's call (`mirror._log_into`): the
        mark `_no_step_for` reads, so its warning says the mirror cannot pass a step (section Q5, Q7).
        Through `self.log` itself, so what replaces `log` on the session sees the mirror's call too."""
        self._mirroring = True
        try:
            self.log(values)
        finally:
            self._mirroring = False
        if not self._evaluating.on:  # pending: left out at `close()` if no step takes it
            committed, names = self._mirrored_pending
            if committed != self._steps_committed:
                names = set()
            self._mirrored_pending = (self._steps_committed, names | set(values))

    def _suggested_step(self) -> int:
        """The step a value with no step in memory is shown to be logged at (section Q5's table): the
        step training starts at (`start_step`, unless a step was committed), else the last step training
        finished. Host integers the session keeps, the ones `_record_manifest` uses: no device read."""
        return self._next_step if self._steps_committed == 0 else self._next_step - 1

    def _no_step_for(self, values: Mapping[str, Any], at: str) -> None:
        """A scalar with no step in memory to be recorded on, and no `step=`: left out, said once per
        session (`WAVE_AL_CONTRACT.md` section Q5, the owner's decision of Nov 24).

        A record already written is never changed, and the value is not put on a later step. The
        warning names the case and shows the explicit call that records it, `log({...}, step=N)`,
        which writes a scalar-only record, with `N` from `_suggested_step`. `at` is "log" for a value
        logged under "eval", "close" for one pending at `close()`. A value the mirror brought (its
        mark, `_log_mirrored`) gets one more sentence: the mirror passes no step (Q7). The text is
        the case of the first value left out; later ones are left out silently. This is the one
        place that decides it."""
        if self._late_said:
            return
        self._late_said = True
        names = sorted(values)
        shown = ", ".join(repr(n) for n in names[:5]) + (f" and {len(names) - 5} more" if len(names) > 5 else "")
        if self._steps_committed == 0:
            # No step committed: a baseline evaluation, or a value pending at close() in a session that
            # never trained, which section Q5 counts as the same case.
            case = "it was logged before the first training step (a baseline evaluation)"
            which = "the step training starts at"
        elif at == "close":
            case = "it was still pending at close() after flush() with no training step since"
            which = "the last step training finished"
        else:
            case = "it was logged after flush() with no training step since (an evaluation after flush())"
            which = "the last step training finished"
        committed, pending = self._mirrored_pending
        mirrored = self._mirroring or (at == "close" and committed == self._steps_committed
                                       and any(name in pending for name in names))
        mirror = (" It came from your tracker's log call through mirror_scalars, which cannot pass a step (the "
                  "tracker's own step is ignored): call session.log(..., step=...) yourself for it.") if mirrored else ""
        _warn(
            f"TensorQuill left out the scalar{'s' if len(names) != 1 else ''} {shown}: {case}, when no step is in "
            "memory. To record it, log it with an explicit step, which writes a scalar-only record: "
            f"session.log({{{names[0]!r}: value}}, step={self._suggested_step()}), {which}.{mirror} Said once per "
            "session.",
            stacklevel=4,
        )

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

    def checkpoint(self, state: Mapping[str, Any], *, event: str = "save", model_state: Any = None) -> None:
        """Record what a checkpoint holds: call it with the dict passed to `torch.save`
        (`event="save"`) or the one loaded to resume (`event="load"`, before the first step).

        The manifest names the components found (model, optimizer, scheduler, scaler, rng,
        data position) with cheap identities: counts, the optimizer's step, the scheduler's
        `last_epoch`, the loss scale. A save is recorded with the last step taken (as
        `after_step`); a load with the next step, the first of the resumed run. The detectors
        then say what a resume did not restore.

        At a load the checkpoint's tensors are also compared with the model given to `init`, by
        name and shape (`tensors` on the manifest; CHECKPOINT_TENSOR_CONTRACT.md): the model is
        read from its modules' own parameter and buffer tables, never through `state_dict()`, and
        no tensor's values are read. `model_state=` is the mapping passed to `load_state_dict`
        (`model_state=state["model"]`), which says which entry is the model's; without it the
        entry is found by its key and its names. A save is not compared, and `model_state` is
        not read there.
        """
        if self._closed_call("checkpoint()"):
            return
        if event == "load":
            self._record_manifest(load_manifest(state, self._model, model_state=model_state), event)
        else:
            self._record_manifest(checkpoint_manifest(state), event)

    def checkpoint_files(self, folder: str | Path, *, event: str = "save") -> None:
        """`checkpoint` for a checkpoint folder, as the Hugging Face `Trainer` and Accelerate's
        `save_state` write: the components are read from the file names (`optimizer.pt`,
        `scheduler.pt`, `scaler.pt`, `rng_state*.pth` or `random_states_*.pkl`, the model's
        weights, `trainer_state.json` or `sampler*.bin` for the data position).

        At a load the model's weights in the folder are also compared with the model given to
        `init`, by name and shape, as `checkpoint` compares a state dict (`tensors` on the
        manifest; CHECKPOINT_TENSOR_CONTRACT.md, the folder section): `model.safetensors.index.json`
        and its shards, or the `.safetensors` files, of each of which only the 8-byte header length
        and the JSON header are read, never a tensor's bytes. A folder with only pickled weights
        (`.bin`, `.pt`) is not opened, and a header that cannot be trusted is refused: both read
        `{"state": "unavailable", "why": "folder"}`, a refusal with its reason. A save opens no
        file."""
        if self._closed_call("checkpoint_files()"):
            return
        try:
            names = [entry.name for entry in Path(folder).iterdir()]
        except OSError:
            return
        manifest = files_manifest(names)
        if event == "load":
            from .data import folder_tensor_coverage  # here, not in the import block other lanes edit

            try:
                manifest["tensors"] = folder_tensor_coverage(folder, self._model, names)
            except Exception:  # noqa: BLE001  nothing here may stop the training
                manifest["tensors"] = {**FOLDER_TENSORS, "why": "error"}
        self._record_manifest(manifest, event)

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

        A table of another shape (rows other than one per compared parameter, or other than 2 or 3
        columns) is not recorded and is said once in a warning; what this step recorded before it
        stays as it was. Nothing is broadcast to parameters the table has no row for.

        With a group this is a collective: every rank of the group must call it on the same step, and
        it waits for them, as any collective does. It first exchanges one fixed-size row per rank
        (`distributed._exchange`): how many parameters each compares, its table's device type,
        whether it can gather it, and its projection setting. Where any rank cannot gather, or the
        ranks compare different numbers of parameters or are on different device types, no rank
        gathers and each names every parameter once in a warning with the same reason; otherwise
        every rank gathers. A rank that compares no parameter, or whose session has stopped, still
        takes part. It is the only way the session gathers: the scheduled check is each rank's own observation (`replica`), with no
        collective. Where the group's global ranks can be read and the rank knows its launch, the
        record also says what the spread covers (`replica_spread_scope`); a precomputed table has
        no scope.
        """
        if self._closed:
            return
        target = self._replica_group if group_or_spread is None else group_or_spread
        if not self._replica_params:
            if target is not None and not isinstance(target, torch.Tensor):
                # Nothing to compare, but a peer that compares parameters is entering the exchange:
                # this rank says it compares none there, and every rank skips the gather.
                self._gather_and_drop(target)
            return
        if target is None:
            raise ValueError("pass a replication process group or precomputed spread tensor")
        if self._stopped:
            if not isinstance(target, torch.Tensor):
                self._gather_and_drop(target)  # the other ranks are entering it too
                self._spread_recorded = 1
            return
        if isinstance(target, torch.Tensor) and not self._spread_table_fits(target):
            return
        spread_buf, _ = self._ensure_spread_buffers()
        unread: torch.Tensor | None = None
        if isinstance(target, torch.Tensor):
            spread = target.to(spread_buf)
            self._spread_scope = None
        else:
            self._spread_scope = group_ranks(target)[0]
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

    def _spread_table_fits(self, table: torch.Tensor) -> bool:
        """Whether a precomputed spread table is `[n_params, 2]` or `[n_params, 3]` for this
        session's compared parameters; if not, said once, never raised (from its shape alone)."""
        n = len(self._replica_params)
        if table.ndim == 2 and table.shape[0] == n and table.shape[1] in (2, FINGERPRINT_WIDTH):
            return True
        if not self._spread_shape_said:
            self._spread_shape_said = True
            _warn(
                f"TensorQuill did not record the replica spread table passed to record_replica_spread: its shape is "
                f"{list(table.shape)}, and this session compares {n} parameter{'s' if n != 1 else ''}, so it must "
                f"be [{n}, 2] or [{n}, {FINGERPRINT_WIDTH}] (see spread_from_fingerprints).",
                stacklevel=4,
            )
        return False

    def _say_replica_unread(self, spreads: torch.Tensor, seen: Sequence[int],
                            say: Callable[[str], None] | None = None) -> None:
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
                message = (f"TensorQuill is not comparing the parameter {name!r} across replicas: {why}. It is left "
                           "out of replica_spread, so the cross-GPU drift check does not cover it.")
                if say is None:
                    _warn(message, stacklevel=4)
                else:
                    say(message)

    def _observe_replicas(self, row: int) -> dict[str, Any] | None:
        """This step's replica observation (contract 1) into row `row` of the observation ring:
        each compared parameter's fingerprints (norm, sum and, unless `replica_projection=False`,
        the `name-v1` projection) and its `fold64-v1` sum, all left on the device, so the step adds
        no transfer and no synchronisation; the block's flush moves them with the rest. No
        collective and no store: nothing here waits for another rank. Returns what the record needs
        from the host (the columns, the device type, each parameter's shape and dtype as read, and
        why one was not read), or None when it could not be made, which is said once. Never
        raises."""
        try:
            device = torch.device("cpu") if self.device.type == "mps" else self.device
            n = len(self._replica_params)
            if self._ring_obs is None or self._ring_obs_sig is None:
                self._ring_obs = torch.zeros(self.every, n, FINGERPRINT_WIDTH + 1, dtype=torch.float64, device=device)
                self._ring_obs_sig = torch.zeros(self.every, n, dtype=torch.int64, device=device)
            table, sums, reasons, metas = _observation_table(
                self._replica_params, self._obs_seeds, projection=self._replica_projection, device=device
            )
            width = table.shape[1] - 1
            self._ring_obs[row].zero_()
            self._ring_obs[row, :, :width].copy_(table[:, :width])
            self._ring_obs[row, :, FINGERPRINT_WIDTH].copy_(table[:, width])
            self._ring_obs_sig[row].copy_(sums)
        except Exception as error:  # noqa: BLE001  nothing here may stop a run
            if not self._obs_failed_said:
                self._obs_failed_said = True
                _warn(f"TensorQuill could not make this step's replica observation ({_short(error)}); a step "
                      "without one reads as not covered for this rank.", stacklevel=5)
            return None
        unread = {name: why for name, why in zip(self.replica_names, reasons) if why is not None}
        for name, why in unread.items():
            if name not in self._obs_unread_said:
                self._obs_unread_said.add(name)
                _warn(
                    f"TensorQuill is not observing the parameter {name!r} for the cross-GPU drift check: {why}. "
                    "It is listed under the observation's unread, and a reader leaves it out of the comparison.",
                    stacklevel=5,
                )
        return {"width": width, "device": device.type, "metas": metas, "unread": unread}

    def _start_launch_read(self) -> LaunchRead:
        """Begin the second read of the launch (`tracker.LaunchRead`), once: at the end of the first
        step, so this rank has finished a step since `init` (contract 2), and rank 0 of its start has
        had that long to set its launch. On a rank other than 0 the reader's thread was started at
        `init`, parked; here it is let go, an `Event.set` (no thread is started on the training
        thread: starting one there took up to 90 ms under a load average of 31). The store is read on
        that thread, never on the training thread or the writer's. Rank 0 and a job of one rank have
        no thread. A run replaced after `init` (tests do it) is read from a thread started here.
        Never raises: a read that cannot start leaves the launch null."""
        read = self._launch_read
        if read is None:
            try:
                read = LaunchRead(self.run)
            except Exception:  # noqa: BLE001
                read = LaunchRead(None)
            self._launch_read = read
        elif self._launch_parked:
            self._launch_parked = False
            read.go()
        return read

    def _launch_for_records(self) -> str | None:
        """The launch the records carry, on the writer thread, which never waits for it: what the
        second read gave once it has returned (kept, with how long it took, `launch_read_seconds`),
        and null while it is still out, so a record never carries the launch `init` read where the
        second read would have replaced it. A store that hangs leaves every record's launch null; one
        that fails, null; one that answers late, null until it does and its answer after."""
        if not self._launch_confirmed:
            done, launch = self._start_launch_read().result()
            if not done:
                return None
            self._launch_confirmed = True
            self.launch = launch
            self.launch_read_seconds = self._launch_read.seconds
            self._drop_launch_source()
        return self.launch

    def _drop_launch_source(self) -> None:
        """Let go of the store the second read used, once it is made (or the session closed), so a
        session the caller still holds does not keep the process group's store alive."""
        try:
            if self.run.launch_source is not None:
                self.run = _replace(self.run, launch_source=None)
        except Exception:  # noqa: BLE001
            pass

    def _observation_record(self, fp: list[list[float]], sums: list[int], seen: dict[str, Any],
                            launch: str | None) -> dict[str, Any]:
        """The `replica` key of one record (contract 1), from the row's values on the host."""
        width = seen["width"]
        params: dict[str, Any] = {}
        for name, row, total, (shape, dtype) in zip(self.replica_names, fp, sums, seen["metas"]):
            if name in seen["unread"]:
                continue
            params[name] = {
                "shape": shape,
                "dtype": dtype,
                "fp": [_finite_or_text(v) for v in row[:width]],
                "sig": str(fold64_finish(total, dtype, shape)),
            }
        return {
            "v": 1,
            "launch": launch,
            "group": list(self._obs_group or []),
            "device": seen["device"],
            "projection": PROJECTION_NAME if width == FINGERPRINT_WIDTH else None,
            "signature": SIGNATURE_NAME,
            "params": params,
            "unread": dict(seen["unread"]),
        }

    def step(
        self,
        step: int | None = None,
        lr: float | Sequence[float] | None = None,
        skipped: bool | None = None,
    ) -> None:
        """Mark the end of one optimizer step. Only for `auto_step=False`.

        Use this when the optimizer hooks cannot see the step, as when a framework steps copies
        of the weights (DeepSpeed ZeRO, Megatron-LM). A `torch.amp.GradScaler`, which skips
        `optimizer.step()` on overflow, needs neither: pass it to `init` as `scaler=` and the
        step ends at `scaler.update()`. Pass `lr` (one value per parameter group, or a single
        value) to record the learning rate.

        Call it once per optimizer update that was due, not at an accumulation microstep: a
        microstep is not a step. Pass `skipped=True` when the update was due and did not happen
        (Megatron-LM's `skipped=not update_successful`), `skipped=False` when it was applied.
        The record's `step_skipped` is then true when either you or the scaler given to `init`
        says so, and `step_skipped_by` says who: `"caller"`, `"scaler"` or `"both"`. A caller's skip
        on an update the scaler was seen to apply (an optimizer went through `scaler.step` and
        `scaler.update()` ran before this call, finding no Inf or NaN) also carries
        `scaler_applied: true`; with no such verdict for this update there is no key. Left out,
        the record says what the scaler said, as before, and nothing without a scaler. `skipped`
        takes only `True`, `False` or `None`; anything else raises `TypeError` here, before
        anything of the step is recorded, on the sessions a wrong `lr` raises on: a closed session,
        or one whose capture has stopped, returns without raising for either.
        """
        if self._closed_call("step()"):
            if self._stopped and not self._closed and not self.auto_step:
                self._count_stopped(step)  # the drift check's schedule goes on (`_count_stopped`)
            return
        if self.auto_step:
            raise RuntimeError("step() is called automatically; pass auto_step=False to call it yourself")
        if skipped is not None and not isinstance(skipped, bool):
            # Refused on the sessions a wrong `lr` is refused on: a closed one, or one whose capture
            # stopped, has returned above with no raise, as for `lr`. Here it comes before anything of
            # the step is recorded. A bool only: not 1 or 0, and not a tensor, whose truth would be a
            # device sync (and so is its repr: the message names the type alone).
            raise TypeError(f"skipped must be True, False or None, not {type(skipped).__name__}")
        if not self._grads_recorded and self._should_record_grads():
            self.record_grads()
        if lr is not None:
            self._pending_lr = [float(lr)] if isinstance(lr, (int, float)) else [float(v) for v in lr]
        if self._scaler is not None:
            # An optimizer went through `scaler.step` (or `unscale_`) and `scaler.update()` has not
            # run yet: its verdict, when it comes, is of the update this call ends. Read on the
            # host, from the scaler's own dict, as a stopped session reads it (`_scaler_before_update`).
            self._scaler_verdict_late = any(
                state.get("found_inf_per_device")
                for state in getattr(self._scaler, "_per_optimizer_states", {}).values()
            )
        self._commit(step, caller_skipped=skipped)

    def _router_refusals(self) -> dict[str, str]:
        """Each refused router's reason: why, and the fixes for how it was named (`_refusal`). On the
        pinned path a router's first reason may be one the writer found (`_router_first`)."""
        out: dict[str, str] = {}
        for name, found in self._router_problems.items():
            first = self._router_first.get(name) or (found[0] if found else None)
            if first:
                out[name] = _refusal(name, self._routers[name], *first, device=self._counts[name].device)
        return out

    def _moe_forms(self, counts: torch.Tensor, n: int, refused: Mapping[str, str]) -> dict[str, dict[str, str]]:
        """`moe_source` for each router counted in this block and not said before (contract part 7):
        `form` from the slots `_count_width` keeps for it, read from `counts`, already on the CPU, on
        the first step of the block that counted the router; `point` from the form.

        Scores are the router's requests: TensorQuill took their top-k itself, so nothing the model
        does after (a capacity limit, dropping, padding) is in them. Indices and maps are what the
        model gave at the hooked module, which may come before or after its capacity handling, and a
        forward hook on that module cannot tell which: `"unknown"`. No path writes `"dispatched"`:
        the hook runs when the hooked module returns, and nothing at that point establishes that the
        model has finished assigning (its dispatcher runs after the router). On the host, no sync."""
        out: dict[str, dict[str, str]] = {}
        offset = 0
        for name, spec in self._routers.items():
            width, e = _count_width(spec), spec.n_experts
            size = n * width
            if name not in self._moe_source_written and name not in refused:
                rows = counts[offset : offset + size].reshape(n, width)[:, 2 * e + 2 : 2 * e + 6].tolist()
                for ids, _, maps, scores in rows:
                    form = "indices" if ids else "map" if maps else "scores" if scores else None
                    if form is not None:
                        out[name] = {"form": form, "point": "requested" if form == "scores" else "unknown"}
                        self._moe_source_written.add(name)
                        break
            offset += size
        return out

    def _layers_not_read(self, floats: torch.Tensor | None, n: int,
                         firsts: Sequence[str | None] | None = None) -> dict[str, str]:
        """`layers_not_read` for this block (contract part 7, the author's word: a once-per-run note of
        a persistent limitation): each watched block whose hook found it could not read its output and
        that was read on no step so far, this block's included, with the hook's reason. Read from the
        `seen` column of the block's statistics, already on the CPU. A block read on any step is never
        named, so one that fails on some steps and is read on others is left out, as v1 states. A block
        named here and read on a later step stays named: the note is said once. On the host, no sync."""
        try:
            if floats is not None and n > 0 and self.layer_names:
                size = n * len(self.layer_names) * (_SEEN + 1)
                seen = floats[:size].reshape(n, len(self.layer_names), _SEEN + 1)[:, :, _SEEN]
                self._layers_ever_read.update(int(row) for row in seen.amax(0).nonzero().reshape(-1).tolist())
            if firsts is None:  # the hooks' first problems as they are now; the pinned path passes its snapshot
                firsts = [problems[0] if problems else None for problems in self._layer_problems]
            found = {
                name: first
                for row, (name, first) in enumerate(zip(self.layer_names, firsts))
                if first and row not in self._layers_ever_read and name not in self._layers_not_read_written
            }
        except Exception:  # noqa: BLE001  # the note is lost, never the block
            return {}
        self._layers_not_read_written.update(found)
        return found

    def _say_router_problems(self, stacklevel: int, refusals: Mapping[str, str] | None = None,
                             say: Callable[[str], None] | None = None) -> None:
        for name, reason in (self._router_refusals() if refusals is None else refusals).items():
            with self._said_lock:  # the writer says them too on the pinned path (`_settle`)
                if name in self._router_problems_said:
                    continue
                self._router_problems_said.add(name)
            message = f"TensorQuill is not counting the router {name!r}: {reason}. Its expert loads are not recorded."
            if say is None:
                _warn(message, stacklevel=stacklevel)
            else:
                say(message)

    def _judge_index_ranges(self, counts: torch.Tensor, n: int) -> None:
        """Refuse a router whose expert ids reached n_experts on any step of this block: from its
        `indices=` function, or from the integer tensor in its output (the author's decision on C06).
        Ids below 0 are no assignment and are only counted (`moe_unassigned`). Read from the counts'
        slot for them (`_count_width`), already on the CPU: the check adds no transfer and no sync.
        Ids from `topk`/`argmax` of logits are always in range."""
        offset = 0
        for name, spec in self._routers.items():
            width = _count_width(spec)
            size = n * width
            if not self._router_problems[name]:
                problem = _index_range_problem(spec, counts[offset : offset + size], n, self._steps)
                if problem is not None:
                    self._router_problems[name].append(problem)
            offset += size

    @torch.no_grad()
    def _commit(self, step: int | None, caller_skipped: bool | None = None) -> None:
        if self._said_later:
            self._say_later()  # the pinned path's warnings, found on the writer
        if self._error is not None and not self._stopped:
            self._stop(self._error)  # the sink failed on the writer thread: said here, on this one
        for held in self._held_forwards:  # a forward held in this step and not claimed is not the next step's
            held.n = 0
        if self._stopped:
            self._count_stopped(step)
            return
        self._take_traced_late()  # on the step it followed, before that step can be written below
        if self._launch_read is None or self._launch_parked:
            self._start_launch_read()  # the end of the first step: the launch's second read (contract 2)
        current = self._next_step if step is None else step
        if self._probe_owed:
            self._run_owed_probe()  # the previous step's, before this one becomes the newest
        if self._held is not None:
            self._release_held()  # the step kept back when the ring last filled: this one ends it
            if self._stopped:  # the writer was a full queue behind (`_put`)
                self._count_stopped(step)
                return
        self._say_router_problems(stacklevel=4)
        observed = None
        if self._replica_group is not None and self._replica_params and (current + 1) % self.replica_every == 0:
            # Like Megatron's check_weight_hash_across_dp_replicas_interval: a float64 pass over
            # every weight is too costly for every step. Scheduled from the step number, which
            # the ranks share, so every rank observes the same steps and a reader can pair them.
            if isinstance(self._replica_group, torch.Tensor):
                if not self._spread_recorded:
                    self.record_replica_spread(self._replica_group)  # a precomputed table: no collective
            elif self._obs_group is not None:
                # This rank's own observation (contract 1), whether or not an explicit gather ran
                # on this step: no collective, no store, nothing moved to the host here.
                observed = self._observe_replicas(len(self._steps))
        # A step `scaler.update()` ended with no optimizer stepping is one the scaler skipped: its
        # weights are the previous step's. Told on the host, with no sync. (The record's own
        # `step_skipped` is the scaler's verdict, read at the flush, or the caller's, `step()`.)
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
            torch._foreach_copy_((self._ring_act[row], self._grad_ring()[row]), (self._act, self._grad))
            torch._foreach_zero_((self._act, self._grad))
        elif has_act:
            self._ring_act[row].copy_(self._act)
            self._act.zero_()
        elif self._grads_recorded:
            self._grad_ring()[row].copy_(self._grad)
            self._grad.zero_()
        self._ring_grad_seen.append(self._grads_recorded)
        self._ring_weight_seen.append(
            {"ownership": self._optimizer_ownership(), "dtypes": self._weight_dtypes} if weights_checked else False
        )
        if self._clip_recorded:
            self._ring_clip[row].copy_(self._clip)
            self._clip.zero_()
        self._ring_clip_seen.append(self._clip_recorded)
        if self._clip_step_calls > 1 and self._clip_recorded:
            # Several clipping calls covered this model in the step (wave AE, section 1): the total
            # in `clip_grad_norm` is right, but it was not made by one clip factor, so a reader must
            # not split it by parameter. With one call, or none, or under compile, no key.
            self._pending_amp = {**self._pending_amp, _CLIP_CALLS: self._clip_step_calls}
        self._clip_step_calls = 0
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
        self._ring_scope.append(self._spread_scope if self._spread_recorded else None)
        self._spread_scope = None
        self._ring_obs_seen.append(observed)
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
        if self._probe_off is not None and not self._probe_off_written:
            self._probe_off_written = True
            self._ring_probe[-1] = {"not_run": self._probe_off}
        self._ring_lr.append(self._pending_lr)
        self._ring_ts.append(time.time())  # when the step ended, Unix seconds
        self._pending_lr = None
        self._ring_scalars.append(self._pending_scalars)
        if caller_skipped is not None:
            # What `step(skipped=...)` said, a Python bool beside the scaler's verdict (still on the
            # device): the two are combined when the record is written (`_decode`).
            self._pending_amp = {**self._pending_amp, _CALLER_SKIPPED: caller_skipped}
        self._ring_amp.append(self._pending_amp)
        if self._data_timing:
            clock = time.perf_counter()
            last, self._last_step_clock = self._last_step_clock, clock
            # A pause that ended before this step did is left out (`set_phase`). One still under way
            # (a phase a callback left behind) ran this step's training, so it counts, and the
            # `set_phase("train")` below does not end it as a pause.
            paused, self._paused_seconds, self._pause_began = self._paused_seconds, 0.0, None
            self._ring_wait.append((self._pending_data_wait, None if last is None else max(clock - last - paused, 0.0)))
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
        """Move the ring to the CPU and queue it for the sink, with the scalar-only records logged
        since the last block (`log(..., step=N)`). With no step in the ring they are queued alone,
        so an explicit `flush()` hands every pending record to the writer, as `close()` does. The
        writer thread writes them as it writes step records: `flush()` does not wait for the file."""
        if self._closed_call("flush()"):
            return
        self._take_traced_late()
        self._flush(hold_newest=False)
        self._put_scalar_records()

    def _put_scalar_records(self) -> None:
        """Queue the scalar-only records no block of steps took, in a block with no steps (`_decode`).
        Nothing when none is pending: one test of a list. Never raises (BUG-03: it ran outside the
        guard `_flush` keeps, in `flush()` and `close()`): a value that fails as it is moved is left out
        by `_put`, and anything else that raises here stops capture, as a flush that raises does."""
        if self._scalar_records and not self._stopped:
            try:
                if self._pinned_mode:
                    self._put_scalar_records_pinned()
                    return
                self._put(([],))
            except Exception as error:  # noqa: BLE001
                self._stop(error, f"queuing the scalar-only records raised {_short(error)}", reason="flush_error")

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
        self._ring_scope = []
        self._ring_obs_seen = []
        self._ring_data = []
        self._ring_checkpoint = []
        self._ring_wait = []

    def _flush(self, *, hold_newest: bool) -> None:
        """Move the ring to the CPU and queue it (`_flush_ring`). An exception part way would leave
        the ring's lists longer than the ring and every later step raising (lane U5 saw an
        `IndexError` at each step after one raise here), so it stops capture instead (`_stop`),
        which clears the ring: nothing raises into training from here.

        What a compiled function logged late (`_late_traced`) is put on the newest step first, so
        every path that hands the newest step to the writer drains it (`WAVE_AL_CONTRACT.md` section
        4.8): `_commit`, `flush()`, `close()`, and `_take_unfrozen`, which writes the ring from the
        optimizer's pre-step hook before the next step ends. So a value in that buffer always belongs
        to a step not yet handed to the writer. Eager code only: nothing a traced function reads
        changes. After a stop nothing is drained: the value is lost with its step (a stated limit)."""
        if self._error is not None and not self._stopped:
            self._stop(self._error)
        if self._stopped:
            return
        self._take_traced_late()
        self._say_new_trainable()
        try:
            self._flush_ring(hold_newest=hold_newest)
        except Exception as error:  # noqa: BLE001
            self._stop(error, f"moving a block of steps off the device raised {_short(error)}", reason="flush_error")

    def _take_unfrozen(self) -> None:
        """Start recording the parameters frozen at `init` that are trainable now (C12): from this
        step on, each has a gradient norm, a weight check and an update size, under its own name;
        the records before do not name it. The ring is written first (`_flush`), so every step in a
        block has the same parameters; the new ones go at the end of the buffers, so what is kept
        by position (the index cache, the parameters a weight check did not read exactly) stays
        right, and their first weight check compares with nothing. Said once in a warning.

        On the host, no read of a tensor: one pass over the parameters still frozen, on the lead's
        steps that read gradients or update sizes and in `record_grads`. Measured at 0.49 µs a
        parameter on a CPU under a load average of 157, so about 0.5 ms a step for 1,000 frozen
        tensors there; unloaded it was not measured. Nothing for a model with none frozen. The drift
        check across GPUs keeps the parameters chosen at `init`: each rank's observation names the
        same parameters on every step, and an explicit gather exchanges their count first.

        Not in the router-only mode (`capture="routers"`, no gradients, weight checks or update
        sizes on), a stated limit (SDK-01): there a hand `record_grads()` reads the parameters
        trainable at `init` only, and one frozen then and unfrozen later is never recorded and
        not warned of. To record such a parameter, use `capture="full"` or `grads=`."""
        if not (self.grad_every or self.weight_every or self.updates) or self._stopped:
            return
        try:
            if not any((param := ref()) is None or param.requires_grad for _, ref in self._frozen):
                return  # the usual step: nothing unfrozen
            recorded = set(self.param_names)
            new: list[tuple[str, torch.Tensor]] = []
            still: list[tuple[str, weakref.ref[torch.Tensor]]] = []
            for name, ref in self._frozen:
                param = ref()
                if param is None:
                    continue  # dropped by the model
                if param.requires_grad and name not in recorded:
                    new.append((name, param))
                else:
                    still.append((name, ref))
        except Exception:  # noqa: BLE001
            return
        self._frozen = still
        if not new:
            return
        self._flush(hold_newest=False)
        if self._stopped:
            return
        try:
            self._add_params(new)
        except Exception as error:  # noqa: BLE001
            self._stop(error, f"adding the parameters unfrozen after init raised {_short(error)}", reason="flush_error")
            return
        self._new_trainable_said.update(name for name, _ in new)
        shown = ", ".join(repr(name) for name, _ in new[:5]) + (f" and {len(new) - 5} more" if len(new) > 5 else "")
        _warn(
            f"TensorQuill is recording the parameter{'s' if len(new) != 1 else ''} {shown} from step "
            f"{self._next_step}: trainable now, but frozen when the session started (unfrozen after "
            "tensorquill.init). The records before that step do not name it, and the drift check across "
            "GPUs compares only the parameters trainable at init.",
            stacklevel=4,
        )

    @torch.no_grad()
    def _add_params(self, new: list[tuple[str, torch.Tensor]]) -> None:
        """Append `new` to the recorded parameters and widen the buffers kept per parameter, with the
        ring empty (`_take_unfrozen`). The lists are replaced, not changed in place: a block already
        queued keeps the names it was made with (`_decode`)."""
        k = len(new)
        self._params = self._params + [p for _, p in new]
        self.param_names = self.param_names + [name for name, _ in new]
        self._param_ids = {id(p) for p in self._params}
        self._param_index = {id(p): index for index, p in enumerate(self._params)}
        self._grad = torch.cat([self._grad, self._grad.new_zeros(k, self._grad.shape[1])])
        if self._ring_grad is not None:
            self._ring_grad = torch.cat(
                [self._ring_grad, self._ring_grad.new_zeros(self._ring_grad.shape[0], k, self._ring_grad.shape[2])],
                dim=1,
            )
        if self.updates:
            self._update = torch.cat([self._update, self._update.new_zeros(k, self._update.shape[1])])
            self._ring_update = torch.cat(
                [self._ring_update, self._ring_update.new_zeros(self._ring_update.shape[0], k, self._ring_update.shape[2])],
                dim=1,
            )
        # No earlier norm to compare with: the first check of a new parameter is "not known".
        if self._prev_weight is not None:
            self._prev_weight = torch.cat([self._prev_weight, self._prev_weight.new_full((k,), math.nan)])
        if self._weights_before is not None:
            self._weights_before = torch.cat([self._weights_before, self._weights_before.new_full((k,), math.nan)])
        self._owned = weakref.WeakKeyDictionary()  # what each optimizer holds of the recorded ones, judged again

    def _say_new_trainable(self) -> None:
        """Name once, in a warning, each parameter of the model that is trainable now and was not part
        of it when the session started (added or replaced since `init`): such a parameter has no
        gradient norm and no weight check. One frozen at `init` and unfrozen later is recorded from
        its first trainable step instead (`_take_unfrozen`), and said there. At a flush, at most once
        per `WEIGHT_EVERY` steps, and at close; on the host:
        one pass over the model's parameters, no read of a tensor (about 0.45 ms for 600 parameter
        tensors on a CPU, so under 10 µs a step). Not in the router-only mode, which records
        parameters only by a hand `record_grads()`, and those trainable at `init` only (SDK-01)."""
        if not (self.grad_every or self.weight_every):
            return
        if not self._closed and self._steps_committed < self._trainable_scan_at:
            return
        self._trainable_scan_at = self._steps_committed + WEIGHT_EVERY
        try:
            new = [
                name for name, p in self._model.named_parameters()
                if p.requires_grad and id(p) not in self._param_ids and id(p) not in self._frozen_ids
                and name not in self._new_trainable_said
            ]
        except Exception:  # noqa: BLE001
            return
        if not new:
            return
        self._new_trainable_said.update(new)
        shown = ", ".join(repr(name) for name in new[:5]) + (f" and {len(new) - 5} more" if len(new) > 5 else "")
        _warn(
            f"TensorQuill is not recording the parameter{'s' if len(new) != 1 else ''} {shown}: trainable now, "
            "but not part of the model when the session started (added or replaced after tensorquill.init). A "
            "session records the parameters the model had at init; to record these, close it and call "
            "tensorquill.init again.",
            stacklevel=4,
        )

    def _flush_ring(self, *, hold_newest: bool) -> None:
        if self._pinned_mode:
            self._flush_ring_pinned(hold_newest=hold_newest)
            return
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
        has_grads = any(self._ring_grad_seen) and self._grad.numel() > 0 and self._ring_grad is not None
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
        obs_seen, scopes = self._ring_obs_seen, self._ring_scope
        self._ring_obs_seen, self._ring_scope = [], []
        observed = any(obs_seen) and self._ring_obs is not None and self._ring_obs_sig is not None
        obs = _cpu_copy(self._ring_obs[:n]) if observed else None
        obs_sig = _cpu_copy(self._ring_obs_sig[:n]) if observed else None
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
            self._update_why.setdefault(UPDATES_NO_OPTIMIZER, _NO_OPTIMIZER_WHY)
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
        sources = self._moe_forms(counts, n, refused) if counts is not None else {}
        layers_why = self._layers_not_read(floats if has_act else None, n)
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
            refused, announce, updates, update_seen, update_why, grad_why, obs, obs_sig, obs_seen, scopes,
            self.param_names, sources, layers_why,
        )
        if hold_newest:
            first = self._hold_newest(dict(zip(_BLOCK_FIELDS, block)), act_size=self._act.numel() if has_act else 0,
                                      originals=(steps, lrs, stamps, data_rows, waits, no_optimizer))
            if first is None:
                return
            block = tuple(first[key] for key in _BLOCK_FIELDS)
        self._put(block)

    # ---- the pinned path (`pinned_flush=True`): gather here, settle on the writer -------------------------

    def _say_pinned_off(self, why: str, *, now: bool) -> None:
        """Turn the pinned copy off for the rest of the session (every later flush reads with a blocking
        copy; blocks are still settled on the writer, in order) and say so once: now, on the training
        thread, or from the writer through `_said_later`."""
        self.pinned_flush = False
        if self._pinned_off_said:
            return
        self._pinned_off_said = True
        message = (f"TensorQuill's pinned flush (pinned_flush=True) is off for the rest of this session: {why}. "
                   "Blocks of steps are moved off the device with a blocking copy instead, as with "
                   "pinned_flush=False; the records are the same.")
        if now:
            _warn(message, stacklevel=5)
        else:
            self._said_later.append(message)

    def _say_later(self) -> None:
        """Say, on the training thread, what the writer found to warn of on the pinned path."""
        while self._said_later:
            try:
                message = self._said_later.popleft()
            except IndexError:
                return
            _warn(message, stacklevel=4)

    def _pinned_wait_failed(self, error: BaseException) -> Callable[[torch.Tensor], torch.Tensor]:
        """On the writer: a copy's event could not be waited for. The path goes off, and the block is read
        from its staging tensor, which the session still holds, with a blocking copy on the writer."""
        self._say_pinned_off(f"waiting for a copy to pinned host memory raised {_short(error)}", now=False)
        return getattr(self._copier, "read", None) or (lambda staging: staging.cpu())

    def _launch(self, moves: _Moves) -> _Transfer:
        """Start moving `moves` to the host, on the training thread, with no host read: each part on the
        session's device packed into one fresh staging tensor (on the current stream, so after whatever
        wrote the ring), then one copy into a pinned buffer (`_CudaCopies.copy`). A part elsewhere is read
        at once (a CPU part is cloned). With the copy off, or in a forked process, every part is read now
        with a blocking copy, as `_flush_ring` reads it; a failure here turns it off, said once."""
        parts = moves.parts
        copier = self._copier
        if not (self.pinned_flush and copier is not None and os.getpid() == self._pid):
            return _Transfer([_cpu_copy(part) for part in parts])
        layout: list[Any] = []
        packed: list[tuple[int, torch.Tensor]] = []
        size = 0
        for part in parts:
            if part.device == self.device:
                nbytes = part.numel() * part.element_size()
                layout.append((size, nbytes, part.dtype, tuple(part.shape)))
                packed.append((size, part))
                size += -(-nbytes // _PINNED_ALIGN) * _PINNED_ALIGN
            else:
                layout.append(_cpu_copy(part))  # on another device: read now (a stated limit)
        if not packed:
            return _Transfer(layout)
        try:
            staging = torch.empty(size, dtype=torch.uint8, device=self.device)
            for start, part in packed:
                flat = part.detach().reshape(-1)
                if flat.numel():
                    staging[start : start + flat.numel() * flat.element_size()].copy_(flat.view(torch.uint8))
            if self._pool is None:
                self._pool = _PinnedPool(copier)
            buffer, kept = self._pool.take(size)
            try:
                event = copier.copy(staging, buffer)
            except BaseException:
                self._pool.give(buffer, kept)
                raise
        except Exception as error:  # noqa: BLE001  never into training: this flush and the rest read now
            self._say_pinned_off(f"a copy to pinned host memory could not be made ({_short(error)})", now=True)
            return _Transfer([_cpu_copy(part) for part in parts])
        return _Transfer(layout, event=event, buffer=buffer, kept=kept, staging=staging, pool=self._pool)

    def _gather_scalar_records(self, moves: _Moves) -> tuple[list[dict[str, Any]], Any] | None:
        """`_put`'s scalar-only records, their tensors added to `moves` (None when none is pending)."""
        if not self._scalar_records:
            return None
        pending, self._scalar_records = self._scalar_records, []
        rows = [record["scalars"] for record in pending]
        if not any(isinstance(value, torch.Tensor) for row in rows for value in row.values()):
            return pending, (rows, [])
        try:
            return pending, _scalars_gather(rows, moves)
        except Exception:  # noqa: BLE001  as `_put`: each value on its own, one that fails left out (BUG-03)
            return pending, (self._scalar_rows_one_by_one(pending), [])

    def _enqueue_pending(self, item: _Pending) -> None:
        self._enqueue(item)
        item.queued = not self._stopped

    def _flush_ring_pinned(self, *, hold_newest: bool) -> None:
        """`_flush_ring` on the pinned path, its training-thread half: the ring's lists taken and reset, what
        the hooks and the training thread have found so far kept as it is now, and every device tensor the
        block needs handed to `_launch`, then the item queued. What needs the values on the host (`_settle`)
        is left to the writer. The held step (`_hold_newest`) stays in the ring as there."""
        if self._probe_owed and not hold_newest:
            self._run_owed_probe()  # the newest step's, before it leaves the ring (a held one stays)
        if self._held is not None:
            self._release_held_pinned()
            return
        n = len(self._steps)
        if n == 0:
            return
        moves = _Moves()
        has_act = self._act.numel() > 0
        has_grads = any(self._ring_grad_seen) and self._grad.numel() > 0 and self._ring_grad is not None
        floats: int | None = None
        if has_act and has_grads:
            floats = moves.add(torch.cat([self._ring_act[:n].reshape(-1), self._ring_grad[:n].reshape(-1)]))  # type: ignore[index]
        elif has_act:
            floats = moves.add(self._ring_act[:n].reshape(-1))
        elif has_grads:
            floats = moves.add(self._ring_grad[:n].reshape(-1))  # type: ignore[index]
        counts = (moves.add(torch.cat([ring[:n].reshape(-1) for ring in self._ring_counts.values()]))
                  if self._ring_counts else None)
        spread_seen = self._ring_spread_seen
        spreads = moves.add(self._ring_spread[:n]) if self._ring_spread is not None and any(spread_seen) else None
        obs_seen, scopes = self._ring_obs_seen, self._ring_scope
        observed = any(obs_seen) and self._ring_obs is not None and self._ring_obs_sig is not None
        obs = moves.add(self._ring_obs[:n]) if observed else None  # type: ignore[index]
        obs_sig = moves.add(self._ring_obs_sig[:n]) if observed else None  # type: ignore[index]
        clip_seen = self._ring_clip_seen
        clips = moves.add(self._ring_clip[:n]) if any(clip_seen) else None
        update_seen = self._ring_update_seen
        updates = moves.add(self._ring_update[:n]) if any(update_seen) else None
        amp_rows, scalar_rows, checkpoint_rows = self._ring_amp, self._ring_scalars, self._ring_checkpoint
        amps = _scalars_gather(amp_rows, moves)
        scalars = _scalars_gather(scalar_rows, moves)
        checkpoints = _checkpoints_gather(checkpoint_rows, moves)
        # A one-step ring that is held (`every=1`) puts no block now: the scalar-only records wait for the
        # release, as they do on the blocking path.
        records = None if hold_newest and n == 1 else self._gather_scalar_records(moves)
        transfer = self._launch(moves)  # before the ring changes below (`_ring_update[0]`)
        no_optimizer, grad_seen, weight_seen = self._ring_no_optimizer, self._ring_grad_seen, self._ring_weight_seen
        steps, lrs, stamps, data_rows, waits, probes = (self._steps, self._ring_lr, self._ring_ts, self._ring_data,
                                                        self._ring_wait, self._ring_probe)
        act_size = self._act.numel() if has_act else 0
        data = {
            "n": n, "has_act": has_act, "act_size": act_size, "hold": hold_newest, "steps": steps, "lrs": lrs,
            "floats": floats, "counts": counts, "spreads": spreads, "clips": clips, "clip_seen": clip_seen,
            "grad_seen": grad_seen, "weight_seen": weight_seen, "spread_seen": spread_seen, "scalars": scalars,
            "probes": probes, "stamps": stamps, "amps": amps, "data_rows": data_rows, "checkpoints": checkpoints,
            "waits": waits, "updates": updates, "update_seen": update_seen, "obs": obs, "obs_sig": obs_sig,
            "obs_seen": obs_seen, "scopes": scopes, "param_names": self.param_names, "no_optimizer": no_optimizer,
            "records": records,
            # What the training thread and the hooks have produced so far, as it is now (`_settle`).
            "update_why": dict(self._update_why),
            "grad_why": self._new_grad_why(),
            "system": self._system.flush() if self._system is not None else None,
            "hook_refusals": {name: found[0] for name, found in self._router_problems.items() if found},
            "layer_firsts": [problems[0] if problems else None for problems in self._layer_problems],
        }
        item = _Pending("block", data, transfer)
        self._clear_ring_lists()
        self._ring_no_optimizer = []
        self._ring_update_seen = []
        self._ring_obs_seen, self._ring_scope = [], []
        self._ring_wait = []
        if hold_newest:
            # As `_hold_newest`: the newest step stays in the ring as its only row, its Python entries as
            # they were (here also its logged tensors, which are read again only if they change).
            self._held = {"block": None, "act_size": act_size, "dirty": set(), "source": item}
            self._steps, self._ring_lr, self._ring_ts = steps[-1:], lrs[-1:], stamps[-1:]
            self._ring_data, self._ring_wait, self._ring_no_optimizer = data_rows[-1:], waits[-1:], no_optimizer[-1:]
            self._ring_scalars, self._ring_amp = scalar_rows[-1:], amp_rows[-1:]
            self._ring_probe, self._ring_checkpoint = probes[-1:], checkpoint_rows[-1:]
            self._ring_grad_seen, self._ring_weight_seen = grad_seen[-1:], weight_seen[-1:]
            self._ring_clip_seen, self._ring_spread_seen = clip_seen[-1:], spread_seen[-1:]
            self._ring_update_seen = update_seen[-1:]
            self._ring_obs_seen, self._ring_scope = obs_seen[-1:], scopes[-1:]
            if self.updates and n > 1:
                self._ring_update[0].copy_(self._ring_update[n - 1])
            if n == 1:
                return  # queued with its release (`_settle_release`), so the queue counts one block a step
        self._enqueue_pending(item)

    def _release_held_pinned(self) -> None:
        """`_release_held` on the pinned path: what arrived for the held step since its flush (a save, a
        probe, a follower's update, weight check or loss), its device rows handed to `_launch`, and the
        release queued after the flush it belongs to; the writer puts the two together."""
        held, self._held = self._held, None
        if held is None:
            return
        dirty = held["dirty"]
        moves = _Moves()
        data = {
            "act_size": held["act_size"],
            "checkpoints": _checkpoints_gather(self._ring_checkpoint, moves),
            "probes": self._ring_probe,
            "weight_seen": self._ring_weight_seen,
            "update_seen": self._ring_update_seen,
            "scalars": _scalars_gather(self._ring_scalars, moves) if "scalars" in dirty else None,
            "updates": moves.add(self._ring_update[:1]) if "update" in dirty else None,
            # Marked only by a weight check read again (`_follow`), so the rows are four wide.
            "grad": moves.add(self._ring_grad[0, :, 2:]) if "grad" in dirty and self._ring_grad is not None else None,
            "update_why": dict(self._update_why),
            "grad_why": self._new_grad_why(),
        }
        data["records"] = self._gather_scalar_records(moves)
        transfer = self._launch(moves)
        self._clear_ring_lists()
        self._enqueue_pending(_Pending("release", data, transfer, source=held["source"]))

    def _put_scalar_records_pinned(self) -> None:
        """`_put_scalar_records` on the pinned path: a block with no steps, its tensors moved as a flush's."""
        moves = _Moves()
        records = self._gather_scalar_records(moves)
        if records is not None:
            self._enqueue_pending(_Pending("records", {"records": records}, self._launch(moves)))

    def _new_update_why(self, snapshot: Mapping[str, str]) -> dict[str, str]:
        """The update notes not yet written: the training thread's as gathered, then the writer's own."""
        combined = dict(snapshot)
        for name, why in self._update_why_late.items():
            combined.setdefault(name, why)
        found = {name: why for name, why in combined.items() if name not in self._update_why_written}
        self._update_why_written.update(found)
        return found

    def _settle_refusals(self, counts: torch.Tensor | None, n: int, steps: Sequence[int],
                         hook_refusals: Mapping[str, tuple[str, str | None]]) -> dict[str, str]:
        """Each router's refusal for this block, as `_judge_index_ranges` and `_router_refusals` make it on
        the blocking path: its first problem ever; the index ranges judged when the hooks had found none."""
        refused: dict[str, str] = {}
        offset = 0
        for name, spec in self._routers.items():
            size = n * _count_width(spec)
            first = self._router_first.get(name)
            if first is None:
                first = hook_refusals.get(name)
                if first is None and counts is not None:
                    first = _index_range_problem(spec, counts[offset : offset + size], n, steps)
                if first is not None:
                    self._router_first[name] = first
            if first is not None:
                refused[name] = _refusal(name, spec, *first, device=self._counts[name].device)
            offset += size
        return refused

    def _with_records(self, block: tuple[Any, ...], records: Any, values: list[torch.Tensor]) -> tuple[Any, ...]:
        if records is None:
            return block
        pending, gathered = records
        rows = _scalars_settle(gathered, values)
        ready = [{**record, "scalars": row} for record, row in zip(pending, rows) if row]
        return tuple(block) + (None,) * (len(_BLOCK_FIELDS) - len(block)) + (ready,)

    def _settle(self, item: _Pending) -> list[tuple[Any, ...]]:
        """The writer's half of a pinned flush, in queue order: wait for the copy (`_Transfer.take`), then
        what `_flush_ring` does with the values on the host, from what was gathered and from the state this
        half owns (`_router_first`, `_router_refusals_written`, `_moe_source_written`, `_layers_ever_read`,
        `_update_why_written`). Returns the blocks to write, as `_decode` takes them. Warnings are queued
        for the training thread (`_say_later`)."""
        if item.kind == "release":
            return self._settle_release(item)
        item.settled = True
        values = item.transfer.take(self._pinned_wait_failed)
        d = item.data
        if item.kind == "records":
            return [self._with_records(([],), d["records"], values)]

        def get(key: str) -> torch.Tensor | None:
            return None if d[key] is None else values[d[key]]

        n = d["n"]
        floats, counts, spreads, clips, updates, obs, obs_sig = (
            get(key) for key in ("floats", "counts", "spreads", "clips", "updates", "obs", "obs_sig")
        )
        amps = _scalars_settle(d["amps"], values)
        scalars = _scalars_settle(d["scalars"], values)
        checkpoints = _checkpoints_settle(d["checkpoints"], values)
        say = self._said_later.append
        refused = self._settle_refusals(counts, n, d["steps"], d["hook_refusals"])
        if counts is not None:
            self._say_router_problems(0, refusals=refused, say=say)
        if spreads is not None:
            self._say_replica_unread(spreads, d["spread_seen"], say=say)
        if any(flag and not (amp and amp.get("step_skipped")) for flag, amp in zip(d["no_optimizer"], amps)):
            self._update_why_late.setdefault(UPDATES_NO_OPTIMIZER, _NO_OPTIMIZER_WHY)
        update_why = self._new_update_why(d["update_why"])
        announce = {name: why for name, why in refused.items() if name not in self._router_refusals_written}
        self._router_refusals_written.update(announce)
        sources = self._moe_forms(counts, n, refused) if counts is not None else {}
        layers_why = self._layers_not_read(floats if d["has_act"] else None, n, firsts=d["layer_firsts"])
        block = (
            d["steps"], d["lrs"], floats, counts, spreads, clips, d["clip_seen"], d["grad_seen"], d["weight_seen"],
            d["spread_seen"], scalars, d["probes"], d["stamps"], amps, d["data_rows"], checkpoints, d["waits"],
            d["system"], refused, announce, updates, d["update_seen"], update_why, d["grad_why"], obs, obs_sig,
            d["obs_seen"], d["scopes"], d["param_names"], sources, layers_why,
        )
        if d["hold"]:
            first, item.held = self._split_newest(dict(zip(_BLOCK_FIELDS, block)), d["act_size"])
            if n == 1:
                return []
            block = tuple(first[key] for key in _BLOCK_FIELDS)
        return [self._with_records(block, d["records"], values)]

    def _settle_release(self, item: _Pending) -> list[tuple[Any, ...]]:
        """The held step's block: the half its flush kept (`item.source.held`), with what arrived since."""
        source = item.source
        if source is not None and not source.settled:
            self._settle(source)  # a one-step ring's flush, queued with its release (`_flush_ring_pinned`)
        values = item.transfer.take(self._pinned_wait_failed)
        held = None if source is None else source.held
        if held is None:
            return []
        source.held = None  # type: ignore[union-attr]
        d = item.data
        held["checkpoints"] = _checkpoints_settle(d["checkpoints"], values)
        held["probes"] = d["probes"]
        held["weight_seen"] = d["weight_seen"]
        held["update_seen"] = d["update_seen"]
        if d["scalars"] is not None:
            held["scalars"] = _scalars_settle(d["scalars"], values)
        if d["updates"] is not None:
            held["updates"] = values[d["updates"]]
        if d["grad"] is not None and held["floats"] is not None:
            weights = held["floats"][d["act_size"] :].view(-1, 4)
            weights[:, 2:] = values[d["grad"]]
        held["update_why"] = {**held["update_why"], **self._new_update_why(d["update_why"])}
        held["grad_why"] = {**held["grad_why"], **d["grad_why"]}
        return [self._with_records(tuple(held[key] for key in _BLOCK_FIELDS), d["records"], values)]

    def _hold_newest(self, block: dict[str, Any], *, act_size: int, originals: tuple[Any, ...]) -> dict[str, Any] | None:
        """Split a full ring's block, already on the CPU in one transfer: the steps but the newest,
        returned to be queued now, and the newest, kept as `self._held` (`_release_held`). Its
        rows stay in the ring too, as its only row, so what arrives for it after it ended is
        recorded there as for any step in the ring; only a device row changed since is read
        again when it is written. None when the ring held one step (`every=1`)."""
        n = len(block["steps"])
        last = n == 1
        first, held = self._split_newest(block, act_size)
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
        self._ring_obs_seen, self._ring_scope = block["obs_seen"][-1:], block["scopes"][-1:]
        if self.updates and n > 1:
            self._ring_update[0].copy_(self._ring_update[n - 1])
        return None if last else first

    def _split_newest(self, block: dict[str, Any], act_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """`_hold_newest`'s split of a block already on the CPU: the steps but the newest, and the newest.
        Host work only; on the pinned path it runs on the writer (`_settle`)."""
        n = len(block["steps"])
        first: dict[str, Any] = {}
        held: dict[str, Any] = {}
        for key in ("steps", "lrs", "clip_seen", "grad_seen", "weight_seen", "spread_seen", "scalars", "probes",
                    "stamps", "amps", "data_rows", "checkpoints", "waits", "update_seen", "obs_seen", "scopes"):
            first[key], held[key] = block[key][:-1], block[key][-1:]
        for key in ("spreads", "clips", "updates", "obs", "obs_sig"):
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
        for key in ("announce", "update_why", "grad_why", "moe_source", "layers_not_read"):
            first[key], held[key] = (None, block[key]) if last else (block[key], {})
        first["system"], held["system"] = None, block["system"]
        first["refused"] = held["refused"] = block["refused"]
        first["param_names"] = held["param_names"] = block["param_names"]
        return first, held

    def _release_held(self) -> None:
        """Queue the step kept back when the ring last filled, with what arrived for it since:
        a save, a probe, a follower's update, weight check or loss. Only what changed on the
        device is read again (with a follower, once a block); otherwise no transfer."""
        if self._pinned_mode:
            self._release_held_pinned()
            return
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
        if "grad" in dirty and block["floats"] is not None and self._ring_grad is not None:
            # Marked only by a weight check read again (`_follow`), so the rows are four wide here.
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
        queued are still written, so the records stay one unbroken run of steps.

        The scalar-only records logged since the last block (`log(..., step=N)`) go with this one,
        written after its step records (contract section 4.2): their tensors are moved to the CPU
        here, with the block, in one transfer per device, and only when one holds a tensor. A block
        dropped here drops them too, as a stop drops what is pending."""
        if self._stopped:
            return
        if self._scalar_records:
            pending, self._scalar_records = self._scalar_records, []
            rows = [record["scalars"] for record in pending]
            if any(isinstance(value, torch.Tensor) for row in rows for value in row.values()):
                try:
                    rows = _scalars_to_cpu(rows)
                except Exception:  # noqa: BLE001  never into training, `flush()` or `close()` (BUG-03)
                    rows = self._scalar_rows_one_by_one(pending)
            # A record left with no value is not written (as `log` writes none for no values); its
            # `seq` is not reused, and a reader needs none to follow another.
            ready = [{**record, "scalars": row} for record, row in zip(pending, rows) if row]
            block = tuple(block) + (None,) * (len(_BLOCK_FIELDS) - len(block)) + (ready,)
        self._enqueue(block)

    def _enqueue(self, item: Any) -> None:
        """`_put`'s hand-over, for a block or a pinned path's item (`_Pending`): never waits while training."""
        try:
            if self._close_by is None:
                self._queue.put_nowait(item)
            else:
                # In `close()` no training step waits, and a writer that is slow but working gets
                # until the close's deadline to make room for the last block.
                self._queue.put(item, timeout=max(self._close_by - time.monotonic(), 0.0))
        except queue.Full:
            if isinstance(item, _Pending):
                item.transfer.drop()
                if item.source is not None and not item.source.queued:
                    item.source.transfer.drop()  # a one-step ring's flush, which went with its release
            blocks = self._queue.maxsize
            if self._close_by is not None:
                self._stop(RuntimeError(f"the writer had not made room for the last block within {CLOSE_SECONDS:g} s "
                                        "at close"), reason="close_timeout")
                return
            # The writer never waits for NVML (system.py, the author's decision on L2): only the sink.
            cause = ("for example on a stalled network filesystem (pass sink=tensorquill.JsonlSink(path) on "
                     "local disk)")
            self._stop(RuntimeError(
                f"the writer fell {blocks} blocks ({_queue_rings(self.every) * self.every} steps) behind training, "
                f"{cause}"
            ), reason="writer_behind")

    def _scalar_rows_one_by_one(self, pending: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The scalar-only records' values when moving them off the device together raised: each tensor
        moved on its own, and one whose move raises left out of its record, the others kept, as a gradient
        whose norm raised is left out on its own (`record_grads`). Said once per session in a warning
        that names the first value left out. Capture goes on: a value of a scalar-only record is not a
        block of steps, so neither the steps after it nor the end marker at `close()` is lost (BUG-03:
        the move raised out of `flush()` and `close()`, which then wrote no end marker and left the
        writer thread running). A value `log` cannot read at all (on `meta`, nested, sparse) is refused
        at the call (`_value_unread`); this is for one that fails as it is moved."""
        rows: list[dict[str, Any]] = []
        left_out: list[tuple[Any, str, BaseException]] = []
        for record in pending:
            row: dict[str, Any] = {}
            for name, value in record["scalars"].items():
                if isinstance(value, torch.Tensor):
                    try:
                        value = _scalars_to_cpu([{name: value}])[0][name]
                    except Exception as error:  # noqa: BLE001
                        left_out.append((record.get("step"), name, error))
                        continue
                row[name] = value
            rows.append(row)
        if left_out and not self._scalars_unread_said:
            self._scalars_unread_said = True
            step, name, error = left_out[0]
            more = f", and {len(left_out) - 1} more" if len(left_out) > 1 else ""
            _warn(
                f"TensorQuill left out the value {name!r} logged at step {step} ({_short(error)} as it was moved off "
                f"the device){more}: the rest of each record is written, and capture goes on. Said once per session.",
                stacklevel=4,
            )
        return rows

    def _stop(self, cause: BaseException, what: str | None = None, reason: str | None = None) -> None:
        """Stop capture for the rest of the session, on the training thread, and say so once.

        The author's decision on C03: a sink that fails, or a writer a full queue behind, stops
        capture; nothing is raised into training and nothing waits. The cause is kept as
        `writer_error`. Every public entry point then returns at once (`_closed_call`), except
        what the heartbeat reads (`set_phase`, `tick`, `phase`). The forward hooks stay: a flag
        they read under `torch.compile` would be guarded, and flipping it would recompile; their
        buffers are never read again. The clipping and loader hooks come off. The optimizer hooks
        become no-ops (taking them out now could be inside PyTorch's loop over them, which would
        raise) and `close()` removes them; with a `replica_group` too, since the scheduled check
        issues no collective a stopped rank would have to keep entering. `close()` then writes no
        end marker, so the file reads as cut off.

        A streamed session's heartbeat is told (contract 3): every later beat carries
        `capture_stopped` with `reason`, a short word (`sink_error`, `writer_behind`, `close_timeout`,
        `flush_error`), and the next beat goes at once."""
        if self._stopped:
            return
        if cause is self._error and self._error_said is not None and what is None:
            what, reason = self._error_said[0], reason or self._error_said[1]  # the pinned path's settle raised
        self._stopped = True
        self.writer_error = _without_traceback(cause)  # no frames kept, so no cycle through the session (`_drain`)
        # Contract 3: a streamed session's heartbeat says so from now on, with a short reason word
        # (`sink_error` for the sink's own error, unless the call site names another).
        beat = getattr(self._sink, "heartbeat", None)
        if beat is not None:
            try:
                # The step it stopped at (detector rules contract, part 7): the last step committed,
                # or None when none was. Step 0 is a step.
                beat.stop_capture(reason or ("sink_error" if cause is self._error else "stopped"),
                                  step=self._next_step - 1 if self._steps_committed else None)
            except Exception:  # noqa: BLE001  a beat that cannot be told is the server's limit, not training's
                pass
        self._clear_ring_lists()
        held, self._held = self._held, None
        source = held.get("source") if held else None
        if source is not None and not source.queued:
            source.transfer.drop()  # a one-step ring's flush that waited for its release (pinned path)
        self._probe_owed = False
        self._scalar_records = []  # lost with the steps not yet handed to the writer (contract section 4.2)
        try:  # an idle writer ends now; a busy one ends once the queue is empty (`_drain`)
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        for undo in (lambda: _unwatch_clipping(self), lambda: datawait.unwatch(self), self._unwatch_mirror):
            try:
                undo()
            except Exception:  # noqa: BLE001
                pass
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

    def _count_stopped(self, step: int | None) -> None:
        """A step ended on a stopped session (`session.step()`, or a hook that ran before it was
        made a no-op): only the count goes on. Nothing is gathered: the scheduled check issues no
        collective, so a stopped rank leaves no other rank waiting."""
        current = self._next_step if step is None else step
        self._stepped = self._foreign_stepped = False
        self._spread_recorded = 0
        self._spread_scope = None
        self._next_step = current + 1

    def _gather_and_drop(self, group: Any) -> None:
        """An explicit `record_replica_spread(group)` on a rank that records nothing (a stopped
        session, or one that compares no parameter): the other ranks are entering the exchange and
        maybe the gather too, so this one takes part and drops the result."""
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
        both. With `data_wait`, the time from a change away from `"train"` to the change back is
        left out of the next step's `step_time`, as its fetches are of `data_wait` (`datawait.py`);
        a phase still not `"train"` when a step ends is taken as training, and that time counts.
        Otherwise a no-op unless streaming with `endpoint=`."""
        if self._closed:  # live after a stop: the heartbeat still says what the process is doing
            return
        if self._data_timing and (phase == "train") != (self._phase == "train"):
            # A pause begins or ends: its time is left out of the next step's `step_time` (`_commit`).
            now = time.perf_counter()
            if phase != "train":
                self._pause_began = now
            elif self._pause_began is not None:
                self._paused_seconds += now - self._pause_began
                self._pause_began = None
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
            self._paused_seconds = 0.0
            self._pause_began = time.perf_counter() if self._phase != "train" else None

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

    def _unwatch_mirror(self) -> None:
        """Stop mirroring tracker scalars into this session (`mirror.py`), taking the wrappers out
        when it was the last session to mirror. Nothing when it never mirrored or the module was
        never loaded."""
        mirror = sys.modules.get(__package__ + ".mirror")
        if mirror is not None:
            mirror.unwatch(self)

    def close(self, *, _exception: BaseException | None = None) -> None:
        """Flush what is left and stop the writer. Forward hooks stay on the model, because
        removing a hook from a compiled module triggers a recompile. Once the records file is
        closed and complete, `<name>.end.json` is left beside it (`sink.write_end_marker`), so a
        reader can tell this run from one that was killed.

        In a process forked from the one that made the session, it touches nothing the parent
        owns (SDK-07): the child's session is closed and the child's own hooks come off, and
        nothing is flushed, queued, written or closed, and no end marker is left, so the parent's
        session and its file are as they were. A fork has no writer thread, so what the child
        recorded is not written: a child that should record calls `tensorquill.init` itself, with
        a records file of its own. As wandb does for a run, the session keeps its creator's
        process ID; it installs no fork handler, which would be the whole process's. Not covered:
        a fork made while the parent's writer was in the middle of a write, whose unflushed part
        the child's interpreter may still write at its own exit (a `sink.py` matter).

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
            self._unwatch_mirror,
        ):
            try:
                undo()
            except Exception:  # noqa: BLE001
                pass
        try:
            forked = os.getpid() != self._pid
        except Exception:  # noqa: BLE001
            forked = False
        if forked:
            # The file, its marker, the sink and the stream are the parent's; this process's
            # copies of the queue and the ring are dropped with the session.
            self._replica_group = None
            return
        deadline = time.monotonic() + CLOSE_SECONDS
        if not self._stopped:
            self._close_by = deadline  # the last block may wait for room, up to the deadline (`_put`)
            self._take_traced_late()
            self._attach_pending()  # logged after the last step ended: on that step (`log`)
            self._flush(hold_newest=False)  # the public `flush()` does nothing once closed
            # No block of step records was left to carry them (no step since the last flush, or
            # none at all): a block with no steps does (`_decode`).
            self._put_scalar_records()
        if not self._stopped:
            try:
                self._queue.put(None, timeout=max(deadline - time.monotonic(), 0.0))
            except queue.Full:
                self._stop(RuntimeError(f"the writer had not taken what was left within {CLOSE_SECONDS:g} s at close"),
                           reason="close_timeout")
        wait = STOPPED_CLOSE_SECONDS if self._stopped else max(deadline - time.monotonic(), 0.0)
        self._worker.join(wait)
        self._say_later()  # what the writer found on the pinned path, said here on the caller's thread
        if self._worker.is_alive():
            self._stop(RuntimeError(f"the writer had not finished within {wait:g} s at close"), reason="close_timeout")
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
                self._stop(error, reason="sink_error")
        # A closed session issues no collective again (its optimizer hooks are off), so it lets go of
        # the process group: one that the job destroys is then not kept alive by a session object
        # the caller still holds. So of the store, once the writer can no longer read the launch.
        self._replica_group = None
        if self._launch_parked and self._launch_read is not None:
            self._launch_parked = False
            self._launch_read.cancel()  # no step ended: the parked reader ends without reading
        if not self._worker.is_alive():
            self._drop_launch_source()
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
                blocks = [item]
                if isinstance(item, _Pending):
                    # The pinned path (`pinned_flush=True`): wait for the copy, then make the block(s).
                    blocks = []
                    if self._error is None:
                        try:
                            blocks = self._settle(item)
                        except BaseException as error:  # noqa: BLE001
                            self._error = _without_traceback(error)
                            self._error_said = (f"moving a block of steps off the device raised {_short(error)}",
                                                "flush_error")
                    item.transfer.drop()  # passed over, or already taken: its buffer goes back either way
                    if item.source is not None:
                        item.source.transfer.drop()
                for block in blocks:
                    if self._error is not None:
                        break
                    if self._system is not None:
                        # NVML is asked for here, on the writer thread, and read on a thread of its own;
                        # the newest finished reading is attached and nothing waits for one (system.py).
                        # A failure is written as a note in the record and never stops the writer.
                        # Found by its type, not its place, so other parts of the block can come after it.
                        try:
                            block = tuple(self._system.write(x) if isinstance(x, SystemPart) else x for x in block)
                        except Exception:  # noqa: BLE001
                            pass
                    try:
                        self._sink.write(self._decode(*block))
                    except BaseException as error:  # noqa: BLE001
                        # Kept without its traceback: that holds this frame, and through it the
                        # session, which then outlives its last reference until a garbage collection
                        # and keeps its `replica_group` alive after `destroy_process_group`.
                        self._error = _without_traceback(error)
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
        obs: torch.Tensor | None = None,
        obs_sig: torch.Tensor | None = None,
        obs_seen: list[Any] | None = None,
        scopes: list[list[int] | None] | None = None,
        param_names: Sequence[str] | None = None,
        moe_source: Mapping[str, Mapping[str, str]] | None = None,
        layers_not_read: Mapping[str, str] | None = None,
        scalar_records: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """The block's records: its steps' records, then the scalar-only records `_put` gave it, already
        on the CPU (`log(..., step=N)`), as they are. A block with no steps carries only those."""
        if not steps:
            return list(scalar_records or ())
        # The parameters the block was made with: `_take_unfrozen` may add some after it was queued.
        names = self.param_names if param_names is None else param_names
        n, n_layers, n_params = len(steps), len(self.layer_names), len(names)
        split = n * n_layers * (_SEEN + 1) if n_layers > 0 and floats is not None else 0
        act = (
            floats[:split].reshape(n, n_layers, _SEEN + 1).tolist()
            if n_layers > 0 and floats is not None
            else None
        )
        # Two columns a parameter, or four with the weight check (`Session.__init__`): read from the
        # block's own size, so a block is decoded as it was made.
        grad = (
            floats[split:].reshape(n, n_params, -1).tolist()
            if any(grad_seen) and n_params > 0 and floats is not None and floats.numel() > split
            else None
        )
        spread_rows = (
            spreads.reshape(n, len(self.replica_names), FINGERPRINT_WIDTH + 1).tolist() if spreads is not None else None
        )
        update_rows = updates.reshape(n, n_params, 2).tolist() if updates is not None and n_params > 0 else None
        obs_rows = obs.reshape(n, len(self.replica_names), FINGERPRINT_WIDTH + 1).tolist() if obs is not None else None
        sig_rows = obs_sig.reshape(n, len(self.replica_names)).tolist() if obs_sig is not None else None
        # The launch, as the second read gave it once it has returned, else null (contract 2): the
        # writer never waits for the store.
        needs_launch = (obs_rows is not None and any(obs_seen or ())) or any(
            scope is not None for scope in (scopes or ())
        )
        launch = self._launch_for_records() if needs_launch else None
        moe: dict[str, list[list[int]]] = {}
        # Whether the router's forward on each step was read: every counted forward sets one of the
        # slots that say how it was read (`_count_width`), also with no token. On a step where a
        # forward was held as a possible evaluation and none was counted, nothing of its forward
        # was read: the router is left out of that record's `moe`, as a refused router is, so zeros
        # are not written as if read (AUD-1, wave AG). A router not called on a step at all keeps its
        # zeros, as before: it routed no token.
        counted: dict[str, list[bool]] = {}
        unassigned: dict[str, list[int | None]] = {}  # None: the ids were not checked on that step
        left_out: dict[str, list[list[int] | None]] = {}  # None: no routing map was checked on that step
        offset = 0
        for name, spec in self._routers.items():
            width, e = _count_width(spec), spec.n_experts
            size = n * width
            if not refused or name not in refused:
                rows = counts[offset : offset + size].reshape(n, width).tolist()
                moe[name] = [row[:e] for row in rows]
                counted[name] = [bool(row[2 * e + 2] or row[2 * e + 4] or row[2 * e + 5]) or not row[2 * e + 6]
                                 for row in rows]
                unassigned[name] = [row[e] if row[2 * e + 2] else None for row in rows]
                left_out[name] = [row[e + 2 : 2 * e + 2] if row[2 * e + 3] else None for row in rows]
            offset += size

        records = []
        for i, step in enumerate(steps):
            grad_norms = (
                {name: row[0] for name, row in zip(names, grad[i]) if row[1]}
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
                    name: row[2] for name, row in zip(names, grad[i])
                }
                if isinstance(weight_seen[i], dict):
                    record["optimizer_ownership"] = weight_seen[i]["ownership"]
                    record["weight_dtypes"] = weight_seen[i]["dtypes"]
                # Written when the check compared any parameter (1 or 0, not -1); one it did not compare
                # (`_record_weights`), the first check included, is left out of the list, never the
                # list: no key means no comparison, and an empty list that some were compared and none
                # had an equal norm. What a name in the list asserts: `weights` in `init`.
                if any(row[3] >= 0 for row in grad[i]):
                    record["weights_unchanged"] = [
                        name for name, row in zip(names, grad[i]) if row[3] > 0
                    ]
            if clips is not None and clip_seen[i]:
                record["clip_grad_norm"] = _norm_of_squares(float(clips[i][0]), float(clips[i][1]))
                if amps is not None and amps[i] and _CLIP_CALLS in amps[i]:
                    record["clip_calls"] = amps[i][_CLIP_CALLS]
            if update_rows is not None and update_seen and update_seen[i]:
                # The RMS of the optimizer's update to each parameter it stepped (weight decay
                # left out): the L2 norm of the update over the square root of its size.
                record["update_rms"] = {name: rms for name, (rms, seen) in zip(names, update_rows[i]) if seen}
            if lrs[i] is not None:
                record["lr"] = lrs[i]
            if amps is not None and amps[i]:
                # The scaler's verdict and the loss scale, when an optimizer went through the
                # scaler; the caller's, when `step(skipped=...)` was given one. Skipped when either
                # says so, and `step_skipped_by` (on a skipped step only) says which did.
                amp = amps[i]
                by_scaler = "step_skipped" in amp and bool(amp["step_skipped"])
                by_caller = bool(amp.get(_CALLER_SKIPPED))
                if "step_skipped" in amp or _CALLER_SKIPPED in amp:
                    record["step_skipped"] = by_scaler or by_caller
                if "grad_scale" in amp:
                    record["grad_scale"] = amp["grad_scale"]
                if by_scaler or by_caller:
                    record["step_skipped_by"] = "both" if by_scaler and by_caller else "scaler" if by_scaler else "caller"
                if by_scaler and _SCALED_OPTIMIZERS in amp:
                    # Wave AE, section 4: the scaler skipped some of its optimizers and applied the
                    # others in this update. Every one skipped, or none, has no key.
                    of, part = amp[_SCALED_OPTIMIZERS], round(float(amp[_SKIPPED_OPTIMIZERS]))
                    if 0 < part < of:
                        record["step_skipped_part"] = {"skipped": part, "of": of}
                if by_caller and not by_scaler and "step_skipped" in amp and not amp.get(_SCALER_LATE):
                    # A caller's skip on an update the scaler was seen to apply: an optimizer went
                    # through the scaler and `update()` found no Inf or NaN (`_record_scaler`), before
                    # this step ended. With no verdict of the scaler's for this update (none, or one
                    # of an earlier update that came late) its outcome is unknown: no key.
                    record["scaler_applied"] = True
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
                    # What it covers (contract 1): only when both the group and the launch are known.
                    scope = scopes[i] if scopes is not None and i < len(scopes) else None
                    if scope is not None and launch is not None:
                        record["replica_spread_scope"] = {"launch": launch, "group": list(scope)}
            if obs_rows is not None and sig_rows is not None and obs_seen is not None and obs_seen[i]:
                record["replica"] = self._observation_record(obs_rows[i], sig_rows[i], obs_seen[i], launch)
            step_moe = {name: rows[i] for name, rows in moe.items() if counted[name][i]}
            if step_moe:
                record["moe"] = step_moe
                record["moe_top_k"] = {name: spec.top_k for name, spec in self._routers.items() if name in step_moe}
                record["moe_stats"] = {
                    name: _router_stats(row_counts) for name, row_counts in step_moe.items()
                }
                # Explicit zeros where a check ran, no key where it did not (`_count_width`).
                step_unassigned = {name: rows[i] for name, rows in unassigned.items()
                                   if rows[i] is not None and name in step_moe}
                if step_unassigned:
                    record["moe_unassigned"] = step_unassigned
                step_left_out = {name: rows[i] for name, rows in left_out.items()
                                 if rows[i] is not None and name in step_moe}
                if step_left_out:
                    record["moe_left_out"] = step_left_out
            records.append(record)
        if announce and records:
            # Once per router and session: the reason it is not counted, so a reader of the
            # records (the evidence, the server) can say MoE was not checked for it.
            records[0]["moe_refused"] = dict(announce)
        if moe_source and records:
            # Once per router (contract part 7): how its counts reached the capture and from where.
            records[0]["moe_source"] = {name: dict(source) for name, source in moe_source.items()}
        if layers_not_read and records:
            # Once per block (contract part 7): a watched block whose output this capture cannot read.
            records[0]["layers_not_read"] = dict(layers_not_read)
        if update_why and records:
            # Once for each: why an optimizer's update size is not in these records, so a reader
            # can say the update size was not checked, and not that nothing was found.
            records[0]["update_not_read"] = dict(update_why)
        if grad_why and records:
            # Once for each parameter: why its gradient has no norm in `grad_norms` (`record_grads`).
            records[0]["grads_not_read"] = dict(grad_why)
        if system and records:
            records[-1]["system"] = system
        if scalar_records:
            records.extend(scalar_records)  # after every key said once for the block, so none lands on one
        return records


# The parts of a block queued for the writer, in `Session._decode`'s order.
_BLOCK_FIELDS = (
    "steps", "lrs", "floats", "counts", "spreads", "clips", "clip_seen", "grad_seen", "weight_seen",
    "spread_seen", "scalars", "probes", "stamps", "amps", "data_rows", "checkpoints", "waits", "system",
    "refused", "announce", "updates", "update_seen", "update_why", "grad_why", "obs", "obs_sig", "obs_seen",
    "scopes", "param_names", "moe_source", "layers_not_read",
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


# ---- the pinned-memory flush (`init(pinned_flush=True)`, lane AQ-1) ----------------------------------
#
# A flush on this path issues device work only, on the training thread: every tensor the block needs is
# packed into one fresh `uint8` staging tensor on the current stream, and one copy into a pinned host
# buffer is issued on a copy stream of the session's own, ordered after the pack by an event. The writer
# thread waits for the copy's event, takes the values out, gives the buffer back, and only then does what
# needs host values (`Session._settle`). The staging tensor is what keeps the ring free: the copy never
# reads the ring, which the next step overwrites on the training stream. `PINNED_BUFFERS` host buffers are
# kept (two: one being read by the writer while the next is filled); while the writer holds both, a flush
# takes one more from PyTorch's caching host allocator, dropped after use, so what is pinned at once is
# bounded by the writer queue's bound. Any failure turns the path off for the session, said once: the
# blocking `.cpu()` of today is used from then on.
PINNED_BUFFERS = 2
_PINNED_ALIGN = 16  # bytes: every part of the staging tensor starts here, so it can be viewed as any dtype
_PINNED_MIN_BYTES = 4096
_NO_OPTIMIZER_WHY = (
    "no optimizer that has stepped holds this model's parameters (DeepSpeed ZeRO, Megatron-LM and "
    "FSDP1 step flattened or copied parameters), so there is no Adam state to read for them"
)


def _record_stream(tensor: torch.Tensor, stream: Any) -> None:
    tensor.record_stream(stream)


class _CudaCopies:
    """The copy stream of one CUDA device and the pinned buffers it copies into. `cuda` is `torch.cuda`
    (a stand-in in the tests). Nothing here waits: `copy` returns the event the writer waits for."""

    def __init__(self, device: torch.device, cuda: Any = None) -> None:
        self.cuda = torch.cuda if cuda is None else cuda
        self.device = device
        self.stream = self.cuda.Stream(device=device)

    def buffer(self, nbytes: int) -> torch.Tensor:
        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)

    def copy(self, staging: torch.Tensor, host: torch.Tensor) -> Any:
        # The pack was issued on the current stream, after every write into the ring it reads: the copy
        # stream waits for an event recorded after it (`cudaStreamWaitEvent`, which does not block the
        # host). Same dtype, both contiguous: one `cudaMemcpyAsync`, which does not block for pinned memory.
        ready = self.cuda.Event()
        ready.record(self.cuda.current_stream(self.device))
        self.stream.wait_event(ready)
        with self.cuda.stream(self.stream):
            host[: staging.numel()].copy_(staging, non_blocking=True)
        # Used on a stream other than its own: not reused before the copy is done, even if dropped unread.
        _record_stream(staging, self.stream)
        done = self.cuda.Event(blocking=True)  # the writer's wait yields the CPU instead of spinning
        done.record(self.stream)
        return done

    def read(self, staging: torch.Tensor) -> torch.Tensor:
        """A blocking read of `staging` on the writer thread, ordered after the pack (the copy stream
        waits for it), for a copy whose event could not be waited for."""
        with self.cuda.stream(self.stream):
            return staging.cpu()


def _pinned_backend(device: torch.device) -> Any:
    """The copier for `device`; raises, saying why, where there is none."""
    if device.type != "cuda":
        raise RuntimeError(f"it needs a CUDA device, and this session's device is {device.type}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this process")
    return _CudaCopies(device)


class _PinnedPool:
    """The host buffers: `PINNED_BUFFERS` kept and reused, more taken while the writer holds those.
    Taken on the training thread, given back by the writer once it has taken a block's values out."""

    def __init__(self, copier: Any) -> None:
        self.copier = copier
        self.lock = threading.Lock()
        self.free: list[torch.Tensor] = []
        self.kept = 0

    def take(self, nbytes: int) -> tuple[torch.Tensor, bool]:
        with self.lock:
            for i, buffer in enumerate(self.free):
                if buffer.numel() >= nbytes:
                    return self.free.pop(i), True
            if self.free:  # free but too small for this block: made again, larger
                self.free.pop()
                self.kept -= 1
            keep = self.kept < PINNED_BUFFERS
            if keep:
                self.kept += 1
        size = max(_PINNED_MIN_BYTES, 1 << (max(nbytes, 1) - 1).bit_length()) if keep else nbytes
        try:
            return self.copier.buffer(size), keep
        except BaseException:
            if keep:
                with self.lock:
                    self.kept -= 1
            raise

    def give(self, buffer: torch.Tensor, kept: bool) -> None:
        if kept:
            with self.lock:
                self.free.append(buffer)


class _Moves:
    """The tensors one flush hands to the writer, by index."""

    def __init__(self) -> None:
        self.parts: list[torch.Tensor] = []

    def add(self, tensor: torch.Tensor) -> int:
        self.parts.append(tensor)
        return len(self.parts) - 1


class _Transfer:
    """One flush's values on the host: ready at once, or packed (`layout` holds `(offset, nbytes,
    dtype, shape)` for a packed part and the host tensor itself for one already there)."""

    def __init__(self, layout: list[Any], event: Any = None, buffer: torch.Tensor | None = None,
                 kept: bool = False, staging: torch.Tensor | None = None, pool: _PinnedPool | None = None) -> None:
        self.layout, self.event, self.buffer, self.kept, self.staging, self.pool = layout, event, buffer, kept, staging, pool

    def take(self, failed: Callable[[BaseException], Any]) -> list[torch.Tensor]:
        """On the writer: wait for the copy, take the values out, give the buffer back. A wait that raises
        reads the kept staging tensor instead (`failed` returns the reader) and turns the path off."""
        if self.event is None:
            return self.layout
        try:
            source = self.buffer
            try:
                self.event.synchronize()
            except Exception as error:  # noqa: BLE001
                reader = failed(error)
                source = reader(self.staging)
            return [part if isinstance(part, torch.Tensor)
                    else source[part[0] : part[0] + part[1]].view(part[2]).reshape(part[3]).clone()
                    for part in self.layout]
        finally:
            self.drop()

    def drop(self) -> None:
        """Give the buffer back unread (a block passed over). A copy still in flight into it is on the one
        copy stream, so the next copy into it comes after; the next read waits for its own event."""
        if self.buffer is not None and self.pool is not None:
            self.pool.give(self.buffer, self.kept)
        self.buffer = self.staging = self.event = None


class _Pending:
    """A queued item of the pinned path: what the training thread gathered, and its transfer. `held` is
    the newest step's half, kept by the writer for the release that follows (`kind="release"`)."""

    def __init__(self, kind: str, data: dict[str, Any], transfer: _Transfer, source: "_Pending | None" = None) -> None:
        self.kind, self.data, self.transfer, self.source = kind, data, transfer, source
        self.held: dict[str, Any] | None = None
        self.queued = False  # handed to the writer (`_enqueue_pending`); a one-step ring's waits for its release
        self.settled = False  # taken by the writer (`_settle`)


def _scalars_gather(rows: list[dict[str, Any]], moves: _Moves) -> tuple[list[dict[str, Any]], list[tuple[int, list[tuple[int, str]]]]]:
    """`_scalars_to_cpu`'s first half, on the device: each device's tensors stacked (to the dtype of its
    first, as there) and added to `moves`. Returns copies of the rows and where each value goes."""
    by_device: dict[torch.device, list[tuple[int, str, torch.Tensor]]] = {}
    for i, row in enumerate(rows):
        for name, value in row.items():
            if isinstance(value, torch.Tensor):
                by_device.setdefault(value.device, []).append((i, name, value))
    plan = [
        (moves.add(torch.stack([t.to(entries[0][2].dtype) for _, _, t in entries])), [(i, name) for i, name, _ in entries])
        for entries in by_device.values()
    ]
    return [dict(row) for row in rows], plan


def _scalars_settle(gathered: tuple[list[dict[str, Any]], list[Any]], values: list[torch.Tensor]) -> list[dict[str, Any]]:
    rows, plan = gathered
    for index, keys in plan:
        for (i, name), value in zip(keys, values[index].tolist()):
            rows[i][name] = value
    return rows


def _checkpoints_gather(rows: list[dict[str, Any] | None], moves: _Moves) -> tuple[list[dict[str, Any] | None], list[tuple[int, int]]]:
    """`_checkpoints_to_cpu`'s first half: a resumed optimizer's `step` tensor added to `moves`, on a copy
    of its row (the training thread may still change the row of a held step)."""
    out: list[dict[str, Any] | None] = []
    plan: list[tuple[int, int]] = []
    for row in rows:
        snapshot = row.get("optimizer") if row else None
        if snapshot is not None and isinstance(snapshot.get("step_after"), torch.Tensor):
            value = snapshot["step_after"]
            row = {**row, "optimizer": {**snapshot}}
            if value.numel() == 1:
                plan.append((len(out), moves.add(value.detach().reshape(1))))
            else:
                row["optimizer"]["step_after"] = None
        out.append(row)
    return out, plan


def _checkpoints_settle(gathered: tuple[list[dict[str, Any] | None], list[tuple[int, int]]], values: list[torch.Tensor]) -> list[dict[str, Any] | None]:
    rows, plan = gathered
    for i, index in plan:
        rows[i]["optimizer"]["step_after"] = values[index].reshape(()).item()  # type: ignore[index]
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


def _value_unread(tensor: torch.Tensor) -> str | None:
    """Why `session.log` cannot take a tensor, from its device and layout alone (no values read, so
    no sync, and the same under `torch.compile`), as `_grad_unread` judges a gradient; None when it
    can. A tensor on `meta` was taken before (BUG-03) and its transfer then raised at `close()`."""
    if tensor.device.type == "meta":
        return "the tensor is on meta, which holds no values"
    if tensor.is_nested:
        return "the tensor is a nested tensor"
    if tensor.layout != torch.strided:
        return f"the tensor is a {str(tensor.layout).removeprefix('torch.')} tensor, not a dense one"
    return None


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
    """`init`'s default `JsonlSink`, locked (`sink.JsonlSink`: a second writer of `path` writes a file
    of its own, `sink.path`), with what undoes it: closed, and the file and the folders made for it
    removed when they did not exist before and are still empty."""
    made = [folder for folder in (path.parent, *path.parent.parents) if not folder.exists()]
    sink = JsonlSink(path, lock=True)

    def remove() -> None:
        sink.close()
        if sink.created and sink.path.exists() and sink.path.stat().st_size == 0:
            sink.path.unlink()
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
    ends_step: Any = None,
    mirror_scalars: bool = True,
    pinned_flush: bool = False,
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
            shorter) writes under an ID of its own, with a warning. It names the run's files
            in the output folder, so it may hold no `/`, `\\`, `:` or control character and may
            not be `.` or `..`: a `run_id` given here that does raises `ValueError`, and one from
            the environment, a tracker or a launcher is made to fit, each such character
            replaced by `-`, with a warning (`manifest.run_id_problem`).
            A `Path` (any `os.PathLike`) is taken by its text, and is that string everywhere;
            anything else that is not None or a string (`123`, `0`, `False`) raises `ValueError`.
            None or `""` means not given.
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
        every: steps kept on the device between transfers to the CPU, an integer of 1 or
            more: a bool, a float or a string raises `TypeError`, a number below 1 `ValueError`.
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
            which costs several times more. Under both, a model that is itself one leaf module
            (an `nn.Linear` alone) is summarized under its class name (`"Linear"`). Or module
            names, or a `(name, module) -> bool` filter; `()` for none.
        routers: MoE router module names mapped to their expert count or a `Router`.
        grads: True to record gradient norms every step, False or 0 to disable automatic
            gradient norm capture, or a positive integer N to record every Nth captured step.
        weights: check each parameter's norm after the optimizer step every N steps
            (True: every 50; an integer: every N; False or 0: never), so the detectors can
            tell when weights stopped changing. A check reads every weight once and also
            records that step's gradient norms; a parameter is reported after 3 checks in
            a row find its norm exactly unchanged. A record's `weights_unchanged` lists the
            parameters whose float64 norm at this check equals, exactly, their float64 norm
            at the previous check, both finite. An equal norm is necessary for an unchanged
            weight, not sufficient: two different weights can share a norm (rows swapped, a
            sign flipped, any rotation), so a listed parameter may have changed. A parameter
            the check did not compare is never listed: at the first check (and the first
            after a resume), one not read or read only in float32 (on Apple MPS), one with
            no elements, one whose norm is not finite, and one the previous check did not
            read. A parameter missing from the list was changed or was not compared; the
            list does not say which. With no parameter compared the record has no
            `weights_unchanged`; an empty list means some were compared and none had an
            equal norm. Gradient norms and weight checks cover the
            parameters that are trainable (`requires_grad`) when `init` runs, and one frozen
            then from the first step on which it is trainable (said once in a warning; the
            records before do not name it). Not in the router-only mode, whose hand
            `record_grads()` reads only the parameters trainable at `init`, with no warning for
            one unfrozen later (SDK-01). One added to the model or replaced after `init`
            is named once in a warning and not recorded. An optimizer that holds any of the
            model's parameters, frozen ones included, ends steps.
        auto_step: count optimizer steps with PyTorch's global optimizer hooks. With a
            `torch.amp.GradScaler`, pass it as `scaler=`: the step then ends at
            `scaler.update()`, so a step the scaler skipped is counted too. Pass False and call
            `session.step()` yourself when the hooks cannot see the step, as when a framework
            steps copies of the weights (DeepSpeed ZeRO, Megatron-LM).
        ends_step: with several optimizers on the model, the one whose step closes each
            iteration. A step then ends when that optimizer's step returns, and only then; every
            other optimizer's step since the last step ended belongs to the step it ends next (its
            update and what it changed are on that record), and nothing is read as a follower of the
            step before. An iteration in which another optimizer does not step has nothing of it on
            its record. Left out, the first optimizer to step leads and the others are read as
            stepping after it in each iteration, so one that steps before it has its update written
            one step early. Refused here when it is not a `torch.optim.Optimizer` holding at least
            one of the model's parameters, and with `scaler=` or `auto_step=False`, which end the
            step themselves. The learning rate recorded is this optimizer's.
        replica_group: optional replication `ProcessGroup` (or precomputed spread tensor)
            for the cross-GPU parameter drift check. Each rank records its own observation of
            the replicated parameters on the scheduled steps (the `replica` key): no collective,
            no store and no wait for another rank, inside the step or at `init`. The comparison
            is made afterwards by a reader that has every rank's records (the all-ranks reader,
            the server), so the verdict comes after the records do. The group's global ranks are
            read once here; a group whose ranks cannot be read gets no observation, with a
            warning. `init` never raises for it.
        replica_every: observe every N steps (default 50). Each observation reads every
            compared weight in float64: its norm, its sum and a projection seeded by its name,
            which catches changes that keep the other two, such as swapped elements (a screen,
            not proof of equality), and an exact 64-bit signature of its bytes, shape and dtype
            (`fold64-v1`; different signatures prove a difference, equal ones do not prove
            equality). `session.record_replica_spread(group)` gathers a spread on demand: a
            collective every rank of the group must call, which can block.
        replica_projection: set False to skip the projection. On CPU it takes about seven
            times the norm and sum; GPU cost is not yet measured.
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
            `scaler.update()`, so skipped steps are counted too. A skipped step's record
            also says who reported it (`step_skipped_by`): `"scaler"`, or `"caller"` or
            `"both"` when `session.step(skipped=True)` said so, and a caller's skip that the
            scaler was seen to apply says `scaler_applied: true`. With several optimizers through
            the scaler, a step on which it skipped some and applied the others also says how many
            (`step_skipped_part`: `{"skipped": n, "of": m}`), read at the flush with the rest.
        replica_params: which parameters the cross-GPU drift check compares across
            `replica_group`: every trainable parameter by default, as under data
            parallelism. Under tensor parallelism only some weights, such as LayerNorm, are
            the same on every rank of the group; pass a `(name, param) -> bool` filter, or
            the parameters or their names. With Megatron, which marks sharded weights with
            `tensor_model_parallel`, that is
            `lambda name, p: not getattr(p, "tensor_model_parallel", False)`.
        start_step: the first step number. Pass the step you resume from, so a resumed
            run does not repeat step numbers already written. An integer of 0 or more, by the
            rule `session.log`'s `step=` follows: a bool (`True` is an `int` in Python), a float
            or a string raises `TypeError`, a negative integer `ValueError`.
        clip_norm: record the total gradient norm that `torch.nn.utils.clip_grad_norm_`
            returns for this model's parameters, as `clip_grad_norm`, in any capture mode.
            A call counts when any parameter it is given is one of the model's trainable
            ones. Only an L2 norm (`norm_type` 2, the default) is recorded: a call with
            another norm type records nothing. Free when you clip anyway, and enough for
            gradient-spike detection. A step on which more than one such call ran also says
            how many (`clip_calls`): the total is right, but it was not made by one clipping
            factor, so it cannot be split by parameter. An absent key means one call, or a count
            that is not known: under `torch.compile` the calls are not counted.
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
        mirror_scalars: also record, as `session.log` would, the plain numbers this process logs
            through `wandb.log` / `wandb.Run.log` and `mlflow.log_metric` / `mlflow.log_metrics`,
            on the step `session.log` would give them (W&B's `step` and `commit` and MLflow's
            `step` are ignored), so incidents and `compare` have the loss and evaluation numbers beside them.
            Python `int` and `float` (not `bool`), NumPy scalars and 0-D tensors already on the CPU
            are taken; a tensor on a GPU is skipped unread (no sync), and images, tables, strings
            and lists are ignored. A key this session got from `session.log` in the same step
            (yours, or an adapter's) is kept. The tracker's call, arguments, return value and
            errors are unchanged; the mirror never raises, reads no API key, calls no server and
            never reads `wandb.run.step`. Only a tracker already imported when `init` runs is
            wrapped (none is imported here): import `wandb` or `mlflow` first. `wandb.init()` may
            come before or after. A function bound before `init` (`from mlflow import log_metric`)
            and `MlflowClient` calls are not seen. The wrappers come off when the last session that
            mirrors closes or stops. See `mirror.py` and the README. `False` installs nothing.
        pinned_flush: move each block of steps off a CUDA device without the training thread waiting
            for it. At a flush the ring is packed into one staging tensor on the device and copied into
            pinned host memory on a CUDA stream of the session's own, after an event recorded on the
            current stream; the writer thread waits for the copy's event and makes every read of the
            values. The training thread makes no host read and calls no `torch.cuda.synchronize()`. The
            records are the same as with `False`, which is the default until this path has run on a GPU.
            On a device that is not CUDA, or when pinned memory, the stream or a copy fails, the blocking
            copy is used for the rest of the session, said once in a warning; never an error in training.
            Two pinned buffers of one block each (about 0.3 MB for a 7B dense model at `every=50`, 1 MB
            for Mixtral 8x7B), and more while the writer is more than two blocks behind. Still read with
            a blocking copy: a logged tensor on a device other than the session's, and the probe.
    """
    if not isinstance(pinned_flush, bool):
        raise ValueError(f"pinned_flush must be True or False, not {pinned_flush!r}")
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
    # From here on the run ID is None or a string: a `Path` is its text, anything else raised (AO-04),
    # as did one that could name a file outside the output folder (BUG-06).
    run_id = _manifest.check_run_id(run_id)
    tags, config = _manifest.clean_tags(tags), _manifest.clean_config(config)
    # Every check of the model and the options too, before the run is resolved (which tells other
    # ranks its ID) or anything is opened: an `init` that is rejected changes nothing.
    plan = _plan_session(
        model, every=every, layers=layers, routers=routers, grads=grads, weights=weights,
        replica_every=replica_every, probe=probe, probe_every=probe_every, probe_head=probe_head,
        replica_params=replica_params, scaler=scaler, ends_step=ends_step, auto_step=auto_step,
        mirror_scalars=mirror_scalars, start_step=start_step,
    )
    # What `init` opens is undone if a later step fails (`_undo`): the default sink and its file,
    # the run manifest, the stream to a server. A sink the caller passed in is never closed here.
    undo: list[Callable[[], Any]] = []
    try:
        run = resolve_run(run_id)
        fitted = _manifest.fit_run_id(run.run_id, run.source)  # one TensorQuill was not given (BUG-06)
        if fitted != run.run_id:
            run = _replace(run, run_id=fitted)
        if sink is None:
            sink = _open_default_sink(Path("tensorquill-out") / records_file_name(run.run_id, run.rank), undo)
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
            ends_step=ends_step,
            mirror_scalars=mirror_scalars,
            pinned_flush=pinned_flush,
            _plan=plan,
        )
    except BaseException:
        _undo(undo)
        raise
    session.manifest = run_manifest  # the run manifest rank 0 wrote; None on every other rank
    return session
