"""Tier 2 probe pass: activation histograms and attention statistics on a fixed batch.

Every `probe_every` steps the session runs one fixed batch through the model without the
compiler, in eval mode and without gradients, and records for each repeated block a
histogram of its output, and for each attention call the largest pre-softmax logit and
the attention entropy per head, plus a downsampled attention map. Fused attention kernels
never store the attention map, so the probe recomputes it from the query and key. When
the language-model head can be found (`find_head`), it also records four numbers of the
output logits (`_head_numbers`), applying the head a second time to the same hidden
states, in BF16 and in FP32. They are numbers only: nothing judges them.
"""

from __future__ import annotations

import contextlib
import math
import random
import sys
import threading
from typing import Any, Callable, Iterator, Mapping

import torch
import torch.nn.functional as F
from torch import nn

HIST_BINS = 64
MAP_SIZE = 64
# Elements of one chunk of attention scores, [batch, heads, rows, keys]: 2^26 float32
# values are 256 MB, so long sequences are processed a block of query rows at a time.
CHUNK_ELEMENTS = 1 << 26
FLASH_IMPLEMENTATIONS = ("flash_attention_2", "flash_attention_3")


class ProbeState:
    """Shared between the permanent probe hooks and the probe run.

    The hooks do nothing while `active` is False. Compiled code only runs while it is
    False, so Dynamo traces the hooks as no-ops and guards on the flag, and the probe,
    which runs without the compiler, never causes a recompile. While it is True they read
    only calls made on `thread`, the thread running the probe: another thread's forward of
    the same model is not the probe's.
    """

    def __init__(self) -> None:
        self.active = False
        self.thread: int | None = None
        self.block: str | None = None
        self.histograms: dict[str, dict[str, Any]] = {}
        self.attention: dict[str, dict[str, Any]] = {}
        self.bins = HIST_BINS
        self.map_size = MAP_SIZE
        # The language-model head (`find_head`): its name and how it was found, or why it
        # is not read. `head_off` leaves the `head` key out of the record altogether.
        self.head_name: str | None = None
        self.head_found_by: str | None = None
        self.head_note: str | None = None
        self.head_off = False
        self.head_calls: list[dict[str, Any]] = []
        self.head_problem: str | None = None


def install_probe_hooks(blocks: list[tuple[str, nn.Module]], state: ProbeState) -> list[Any]:
    handles = []
    for name, module in blocks:
        def pre(_module: nn.Module, _args: Any, name: str = name) -> None:
            if state.active and state.thread == threading.get_ident():
                state.block = name

        def post(_module: nn.Module, _args: Any, output: Any, name: str = name) -> None:
            if state.active and state.thread == threading.get_ident():
                tensor = _first_float(output)
                if tensor is not None:
                    state.histograms[name] = _histogram(tensor, state.bins)
                state.block = None

        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))
    return handles


def _first_float(output: Any) -> torch.Tensor | None:
    """The first floating-point tensor of a block's output: the output itself, or one of a
    tuple's, list's or dict's (a Hugging Face `ModelOutput` too) items, as `capture` reads them."""
    if isinstance(output, (tuple, list)):
        items: Any = output
    elif isinstance(output, dict):
        items = output.values()
    else:
        items = (output,)
    for item in items:
        if isinstance(item, torch.Tensor) and item.is_floating_point():
            return item
    return None


def _histogram(tensor: torch.Tensor, bins: int) -> dict[str, Any]:
    # `torch.histc` rejects bf16 and fp16 on CUDA and needs min < max, so cast to float32
    # and widen equal bounds. The probe is rare, so reading the bounds (a sync) is fine.
    x = tensor.detach().float().reshape(-1)
    finite = torch.isfinite(x)
    values = x[finite]
    nonfinite = int(x.numel() - values.numel())
    if values.numel() == 0:
        return {"lo": 0.0, "hi": 0.0, "counts": [0] * bins, "nonfinite": nonfinite}
    lo, hi = (float(v) for v in torch.aminmax(values))
    if lo == hi:
        # Every value is the same one. `torch.histc` needs min < max, and for a huge value
        # `lo - 0.5 == lo` in floating point, which PyTorch before 2.10 refuses. The bounds are
        # reported widened as before and the values all fall in the middle bin, where histc
        # puts them when the widening holds.
        counts = [0] * bins
        counts[min(bins // 2, bins - 1)] = int(values.numel())
        return {"lo": lo - 0.5, "hi": hi + 0.5, "counts": counts, "nonfinite": nonfinite}
    # The bin index is `(v - lo) * bins / (hi - lo)` in float32, which overflows once
    # `(hi - lo) * bins` passes float32's range: histc then counts nothing or misses values.
    # Values that large are scaled down by a power of two first, which is exact (values that
    # become subnormal or zero are within a bin of zero), so every bin edge stays where it was.
    span_lo, span_hi, scaled = lo, hi, values
    if max(abs(lo), abs(hi)) * bins > _HIST_SCALE_ABOVE:
        span_lo, span_hi, scaled = lo * _HIST_SCALE, hi * _HIST_SCALE, values * _HIST_SCALE
    if torch.are_deterministic_algorithms_enabled():
        counts = _histc_counts(scaled, bins, span_lo, span_hi)
    else:
        counts = [int(c) for c in torch.histc(scaled, bins=bins, min=span_lo, max=span_hi).tolist()]
    return {"lo": lo, "hi": hi, "counts": counts, "nonfinite": nonfinite}


_HIST_SCALE_ABOVE = 2.0**120  # about 1.3e36: below it the bin arithmetic stays within float32
_HIST_SCALE = 2.0**-64


def _histc_counts(values: torch.Tensor, bins: int, lo: float, hi: float) -> list[int]:
    """`torch.histc(values, bins, lo, hi)`'s counts, for when deterministic algorithms are on.

    `histc` has no deterministic CUDA kernel and raises then; `bincount` without weights
    counts with integer atomics and is allowed (`SummaryOps.cu`, PyTorch 2.6 and later). Each
    float32 value v goes to bin `int((v - lo) * bins / (hi - lo))`, in float32 and in that
    order, the top value to the last bin: histc's own arithmetic, on the CPU
    (`HistogramKernel.cpp`, LINEAR_INTERPOLATION, its edges from `linspace`, whose first and
    last are lo and hi) and on CUDA (`getBin`). `hi - lo` stays a tensor on the values'
    device, because CUDA divides by a CPU scalar as a product with its reciprocal. Two
    differences: counts are int64, where histc adds in float32 and stops counting a bin at
    2^24, and where `(v - lo) * bins` overflows float32 histc's index is undefined, while
    here the value goes to the last bin (`_histogram` scales values that large down first, so
    neither meets it). The bin index is made a chunk at a time, CHUNK_ELEMENTS int64 values
    at most.
    """
    left = torch.tensor(lo, dtype=torch.float32, device=values.device)
    span = torch.tensor(hi, dtype=torch.float32, device=values.device) - left
    counts = torch.zeros(bins, dtype=torch.int64, device=values.device)
    for start in range(0, values.numel(), CHUNK_ELEMENTS):
        pos = (values[start : start + CHUNK_ELEMENTS] - left) * bins / span
        pos = pos.nan_to_num_(0.0, bins - 1, 0.0).clamp_(0, bins - 1)
        counts += torch.bincount(pos.to(torch.int64), minlength=bins)
    return [int(c) for c in counts.tolist()]


def _not_plain_linear(module: nn.Module) -> str | None:
    """Why `module` cannot be read as the head, or None. Only `nn.Linear`'s own forward is
    known to compute `x @ weight.T + bias`; a subclass with its own forward (quantized,
    sharded, soft-capped) may compute something else, so it is refused, not guessed. The class is
    named with its module: `peft`'s LoRA layer is called `Linear` too."""
    named = f"{type(module).__module__}.{type(module).__qualname__}"
    if not isinstance(module, nn.Linear):
        return f"is a {named}, not an nn.Linear"
    if type(module).forward is not nn.Linear.forward:
        return f"is a {named}, a subclass of nn.Linear with its own forward"
    if not module.weight.is_floating_point():
        return f"has a {module.weight.dtype} weight"
    return None


def find_head(model: nn.Module, head: str | bool | None) -> tuple[str | None, str | None, str | None]:
    """The language-model head the probe reads: `(name, found_by, None)`, or
    `(None, None, reason)` when it is not read.

    Only two sources are trusted, and nothing is matched by name or position: a module
    the user names (`probe_head="lm_head"`), or the module a Hugging Face model reports as
    its output embeddings (`model.get_output_embeddings()`). Either must be a plain
    `nn.Linear`. A name the user gives that is missing or not a plain `nn.Linear` is an
    error at `init`, as an unknown router is. `head=False` turns the reading off.

    A model that does not report one itself, such as a LightningModule or a DDP around a Hugging
    Face model, is asked through the outermost modules inside it that do (`_inner_models`):
    one head among them is read, under its name in the whole model; two different ones are not.
    """
    if head is False:
        return None, None, None
    modules = dict(model.named_modules())
    if isinstance(head, str):
        if head not in modules:
            raise ValueError(f"probe_head: no module named {head!r}")
        reason = _not_plain_linear(modules[head])
        if reason:
            raise ValueError(f"probe_head={head!r} {reason}; the probe reads only a plain nn.Linear head")
        return head, "probe_head", None
    if head is not None and head is not True:
        raise TypeError("probe_head must be a module name, True, False or None")
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        try:
            module = getter()
        except Exception as error:  # noqa: BLE001
            return None, None, f"get_output_embeddings() raised {type(error).__name__}; pass probe_head=<module name>"
        if module is None:
            return None, None, "get_output_embeddings() returned None: the model has no language-model head"
    else:
        inner = _inner_models(model, lambda m: callable(getattr(m, "get_output_embeddings", None)))
        if not inner:
            return None, None, "the model names no output head; pass probe_head=<module name> to read one"
        heads: dict[int, tuple[nn.Module, list[str]]] = {}
        for prefix, wrapped in inner:
            try:
                found = wrapped.get_output_embeddings()
            except Exception as error:  # noqa: BLE001
                return None, None, (f"{prefix!r}.get_output_embeddings() raised {type(error).__name__}; "
                                    "pass probe_head=<module name>")
            if found is not None:
                heads.setdefault(id(found), (found, []))[1].append(prefix)
        if not heads:
            return None, None, "get_output_embeddings() returned None: the model has no language-model head"
        if len(heads) > 1:
            names = ", ".join(repr(p) for _, prefixes in heads.values() for p in prefixes)
            return None, None, (f"the modules {names} inside the model each report a different output head; "
                                "pass probe_head=<module name> to read one")
        module = next(iter(heads.values()))[0]
    name = next((n for n, m in modules.items() if m is module), None)
    if name is None:
        return None, None, "get_output_embeddings() returned a module outside the model"
    reason = _not_plain_linear(module)
    if reason:
        return None, None, f"{name!r} from get_output_embeddings() {reason}; the probe reads only a plain nn.Linear head"
    return name, "get_output_embeddings", None


def _inner_models(model: nn.Module, test: Callable[[nn.Module], bool]) -> list[tuple[str, nn.Module]]:
    """The outermost modules inside `model` that pass `test`, with their names: those with no
    ancestor below `model` that passes it too. One wrapped Hugging Face model gives itself, not
    its decoder or its attention layers, which carry the same `config`."""
    found: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not name or any(name.startswith(f"{outer}.") for outer, _ in found):
            continue
        if test(module):
            found.append((name, module))
    return found


def install_head_hook(head: nn.Module, state: ProbeState) -> list[Any]:
    """A permanent pre-hook on the head, a no-op unless the probe is running (as for the
    blocks, so compiled code never recompiles). It reads the head's input, the final
    hidden states, and the head's weight at the moment of the call: under FSDP the weight
    is gathered then, and may be sharded again once the forward is over."""

    def pre(module: nn.Module, args: Any) -> None:
        if not state.active or state.thread != threading.get_ident() or state.head_problem is not None:
            return
        hidden = args[0] if args else None
        if not isinstance(hidden, torch.Tensor) or not hidden.is_floating_point():
            state.head_problem = "the head was called without a floating-point tensor as its first input"
            return
        weight, bias = module.weight, module.bias
        if type(weight).__name__ == "DTensor" or type(bias).__name__ == "DTensor":
            state.head_problem = "the head's weight is a sharded DTensor at the call; its logits are not whole here"
            return
        try:
            with _autocast_off(hidden.device):
                sums: dict[str, Any] = _head_logit_sums(hidden, weight, bias)
            state.head_calls.append({**sums, "input_dtype": str(hidden.dtype), "weight_dtype": str(weight.dtype)})
        except Exception as error:  # noqa: BLE001  # loses the head's numbers, not the probe
            state.head_problem = f"{type(error).__name__}: {error}"

    return [head.register_forward_pre_hook(pre)]


def _autocast_off(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    """The hook runs inside the user's forward, perhaps under autocast, which would cast the
    FP32 pass to BF16 as well (and make the gap exactly 0). Both passes run with it off."""
    if torch.amp.is_autocast_available(device.type):
        return torch.autocast(device.type, enabled=False)
    return contextlib.nullcontext()


def _head_logit_sums(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """Per-position sums over the vocabulary, for `_head_numbers`.

    The head is applied twice to the same hidden states with `F.linear` on cast copies,
    never by calling the module, and with autocast turned off (`_autocast_off`), so the
    numbers do not depend on how the user's probe ran: once in FP32 (input, weight and bias
    in float32, the reference) and once in BF16 (all three cast to bfloat16, as autocast
    does for a linear layer; PyTorch's CPU BF16 GEMM accumulates in FP32 and rounds its
    output to BF16, `bf16_dot_with_fp32_arith`; the GPU kernels were not read). The
    numbers are of the head module's output, before any soft-cap or temperature the
    model applies after it. Inputs and weights already in BF16 upcast exactly, so for a run whose
    head is BF16 the BF16 pass is its own computation. The vocabulary is taken in chunks
    of at most CHUNK_ELEMENTS logits. Each chunk is reduced in float32 and the chunks are
    added in float64: on the logits' device, or on the CPU for MPS, which has no float64 (C22,
    lane AD3). Before, MPS added them in float32, where a squared norm above about 1.8e19
    overflows though the norm is finite; now only one vector of a value per position goes to the
    CPU for each of the five sums, a chunk at a time, and a norm, never its square, crosses in
    float32. The spread is summed about each position's mean over the first chunk, so a large
    common shift of the logits does not cancel it away. A row's sum that overflows float32 though
    its logits are finite (eight logits of 3e38) is taken again over the row divided by its largest
    magnitude (`_row_sum`), so that mean is read, not counted as overflow (C22, lane X4). A row
    whose difference from that mean overflows float32 though its logits are finite (logits near
    float32's largest value with both signs, such as 3e38, 3e38 and -3e38) is taken again from
    halves, `z / 2 - shift / 2`, which cannot overflow, and its sums multiplied back in float64
    (`_centred`; C22, lane AD3). So is a row whose gap `zb - zf` overflows float32 though both
    passes are finite (2e38 against -2e38): its gap is taken from halves (SDK-05, wave AF); a row
    whose BF16 logits are not finite keeps a gap that is not, and is counted. `logits_finite` says which positions' FP32 logits are all finite,
    read from the logits themselves, so a sum that overflows is not taken for a NaN or an Inf.
    """
    x = hidden.detach().reshape(-1, hidden.shape[-1])
    positions, vocab = x.shape[0], weight.shape[0]
    acc = torch.float64
    sums_on = torch.device("cpu") if x.device.type == "mps" else x.device
    xf, xb = x.float(), x.to(torch.bfloat16)
    rows = max(1, CHUNK_ELEMENTS // max(1, positions))
    shift = None
    zero = torch.zeros(positions, dtype=acc, device=sums_on)
    sq, dmean, dsq, gap = zero.clone(), zero.clone(), zero.clone(), zero.clone()
    lse = torch.full((positions,), -math.inf, dtype=acc, device=sums_on)
    logits_finite = torch.ones(positions, dtype=torch.bool, device=sums_on)
    for v0 in range(0, vocab, rows):
        w = weight[v0 : v0 + rows].detach()
        b = bias[v0 : v0 + rows].detach() if bias is not None else None
        zf = F.linear(xf, w.float(), None if b is None else b.float())
        zb = F.linear(xb, w.to(torch.bfloat16), None if b is None else b.to(torch.bfloat16)).float()
        chunk_finite = zf.isfinite().all(-1)
        logits_finite &= chunk_finite.to(sums_on)
        if shift is None:
            # The mean of finite float32 logits is finite, so the cast back is too.
            shift = _row_sum(zf, acc, zf.shape[-1], sums_on).to(device=x.device, dtype=torch.float32).unsqueeze(-1)
        # One pass each, reduced in float32 within the chunk and added up across chunks
        # in `acc`: per-element float64 sums cost as much as the two GEMMs on CPU.
        lse = torch.logaddexp(lse, torch.logsumexp(zf, -1).to(sums_on).to(acc))
        sq += _square_norm(zf, acc, sums_on)
        part_gap = _square_norm(zb.sub_(zf), acc, sums_on)
        d = zf.sub_(shift)  # zf is not used again in this chunk
        part_mean = _row_sum(d, acc, vocab, sums_on)  # each chunk's share of the mean
        part_sq = _square_norm(d, acc, sums_on)
        # A float64 squared norm of finite float32 values cannot overflow, so one that is not finite
        # is a row of `d`, or of `zb - zf`, holding an Inf or a NaN: with finite FP32 logits, a
        # difference that overflowed, or a BF16 pass that is not finite. Read from one value per
        # position, so the chunk is not passed over again; one sync for both.
        fine = chunk_finite.to(sums_on)
        where = fine & ~part_sq.isfinite()
        gap_where = fine & ~part_gap.isfinite()
        if bool((where | gap_where).any()):  # a sync, as the probe's histograms make: it is rare
            if bool(where.any()):
                broke = where.to(x.device)
                z = F.linear(xf[broke], w.float(), None if b is None else b.float())
                part_mean[where], part_sq[where] = _centred(z, shift[broke], acc, vocab, sums_on)
            if bool(gap_where.any()):
                # Both passes again for those rows only. Where the BF16 logits are finite the gap is
                # read from halves (SDK-05: 2e38 and -2e38 are finite, their float32 difference is
                # not); where they are not, it stays non-finite and the position is counted.
                broke = gap_where.to(x.device)
                zf_rows = F.linear(xf[broke], w.float(), None if b is None else b.float())
                zb_rows = F.linear(xb[broke], w.to(torch.bfloat16),
                                   None if b is None else b.to(torch.bfloat16)).float()
                part_gap[gap_where] = _square_norm(zb_rows * 0.5 - zf_rows * 0.5, acc, sums_on) * 4
        gap += part_gap
        dmean += part_mean
        dsq += part_sq
    assert shift is not None
    mean_d = dmean
    return {
        "norm": sq.sqrt(),
        "mean": shift.squeeze(-1).to(sums_on).to(acc) + mean_d,
        "var": (dsq / vocab - mean_d.square()).clamp_min(0),
        "log_z": lse,
        "gap_sq": gap,
        "vocab": torch.tensor(vocab),
        "logits_finite": logits_finite,
    }


def _centred(z: torch.Tensor, shift: torch.Tensor, acc: torch.dtype, divide: int,
             device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The sum over `divide` and the squared norm of `z - shift`, row by row, in `acc` on `device`,
    for rows whose float32 difference overflows though `z` and `shift` are finite. Both are halved
    first, which is exact for every normal float32 and leaves `z / 2 - shift / 2` within float32's
    range (each half is at most half of its largest value); the halves' sums are doubled, and the
    squared norm quadrupled, in `acc`. A half that falls below float32's normal range loses its
    last bits, at a size no difference these rows hold can show."""
    half = z * 0.5 - shift * 0.5
    return _row_sum(half, acc, divide, device) * 2, _square_norm(half, acc, device) * 4


def _row_sum(t: torch.Tensor, acc: torch.dtype, divide: int, device: torch.device | None = None) -> torch.Tensor:
    """Each row's sum over `divide`, in `acc` on `device` (the rows' own when None). A float32 sum
    overflows once its partial sums pass float32's range, though every entry is finite (eight
    logits of 3e38); then each such row is divided by its largest magnitude first, summed, divided
    by `divide` and multiplied back in `acc`, so a mean of finite values is finite. The common case
    costs one sum and one sync on a value per row, as `_square_norm`; only the rows whose sum is not
    finite are taken again, so a position with a NaN does not make every row pay. A row holding a
    NaN or an Inf stays non-finite."""
    device = t.device if device is None else device
    total = t.sum(-1)
    bad = ~total.isfinite()
    out = total.to(device).to(acc) / divide
    if not bool(bad.any()):  # a sync, as the probe's histograms make: it is rare
        return out
    rows = t[bad]
    top = rows.abs().amax(-1, keepdim=True)
    top = torch.where(top.isfinite() & (top > 0), top, torch.ones_like(top))
    scaled = (rows / top).sum(-1).to(device).to(acc)
    out[bad.to(device)] = scaled / divide * top.squeeze(-1).to(device).to(acc)
    return out


def _square_norm(t: torch.Tensor, acc: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
    """Each row's squared L2 norm, in `acc` on `device` (the rows' own when None). `vector_norm`
    in float32 overflows once a row's sum of squares passes float32's range (finite logits of about
    1e19 over a large vocabulary); then each row is divided by its largest magnitude first, and its
    norm multiplied back in `acc`. Only the norm crosses to `device`, so a float32 norm up to
    float32's largest value is squared in `acc`. A row holding a NaN or an Inf stays non-finite."""
    device = t.device if device is None else device
    norm = torch.linalg.vector_norm(t, dim=-1)
    if not bool(norm.isinf().any()):  # a sync, as the probe's histograms make: it is rare
        return norm.to(device).to(acc).square()
    top = t.abs().amax(-1, keepdim=True)
    top = torch.where(top.isfinite() & (top > 0), top, torch.ones_like(top))
    scaled = torch.linalg.vector_norm(t / top, dim=-1).to(device).to(acc)
    return (scaled * top.squeeze(-1).to(device).to(acc)).square()


def _head_numbers(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """The four head numbers over every position the head saw in the probe (padding
    included: the probe has no mask for the head). Positions whose FP32 logits hold a NaN
    or an Inf are left out and counted (`nonfinite_positions`); so are positions whose logits
    are finite but whose BF16 pass is not (`overflow_positions`, listed only when there are
    some; the key keeps its name). Since lane AD3, and wave AF for a gap whose float32
    difference overflowed (SDK-05), that is the one case: the BF16 logits hold an Inf or a NaN,
    from a BF16 logit past BF16's range (about 3.39e38, below float32's 3.40e38), or a hidden
    state or weight that casts to an Inf in BF16. That cannot be read: BF16's own logit is then an Inf, so `gap`,
    the distance between it and the FP32 logit, has no finite value, and any number written for it
    would be made up. The position's other numbers are not read either, so that `positions` counts
    the same positions for every number. Read, and no longer counted (C22): a mean whose float32
    sum overflows (`_row_sum`, lane X4); a squared norm past float32's range on MPS, norms above
    about 1.8e19, now added in float64 on the CPU; and FP32 logits within a factor of two of
    float32's largest value with both signs, whose difference from the row's mean overflows
    (`_centred`); and finite BF16 and FP32 logits whose float32 difference overflows (SDK-05). Over positions t and vocabulary entries v of the FP32 logits z:

    - `norm`: the mean over t of the L2 norm of z_t over the vocabulary.
    - `mean`: the signed mean of z over every t and v.
    - `log_z`: the mean over t of logsumexp_v z_t, the log of the softmax's normaliser.
    - `gap`: RMS(z_bf16 - z_fp32) over every t and v, absolute, in logit units.
    - `gap_relative`: `gap` over the RMS over t of each position's standard deviation
      across the vocabulary; None when that is zero.
    """
    vocab = int(calls[0]["vocab"])
    if any(int(c["vocab"]) != vocab for c in calls):
        return {"not_read": "the head was called with different output sizes in one probe"}
    joined = {key: torch.cat([c[key] for c in calls]) for key in ("norm", "mean", "var", "log_z", "gap_sq")}
    sums_finite = torch.stack([v.isfinite() for v in joined.values()]).all(0)
    if all("logits_finite" in c for c in calls):
        logits_finite = torch.cat([c["logits_finite"] for c in calls])
    else:
        logits_finite = sums_finite
    finite = logits_finite & sums_finite
    n = int(finite.sum())
    nonfinite = int((~logits_finite).sum())
    overflow = int(finite.numel()) - n - nonfinite
    out: dict[str, Any] = {
        "positions": n, "nonfinite_positions": nonfinite, "vocab": vocab, "calls": len(calls),
        "input_dtype": calls[0]["input_dtype"].removeprefix("torch."),
        "weight_dtype": calls[0]["weight_dtype"].removeprefix("torch."),
    }
    if overflow:
        out["overflow_positions"] = overflow
    if n == 0:
        if finite.numel() == 0:
            reason = "the head saw no positions"
        elif not overflow:
            reason = "every position's FP32 logits hold a NaN or an Inf"
        elif not nonfinite:
            reason = ("every position's FP32 logits are finite but the BF16 pass overflows: its logits hold an Inf "
                      "or a NaN, so the gap has no value")
        else:
            reason = (f"no position read: {nonfinite} with a NaN or an Inf in their FP32 logits, {overflow} with "
                      "finite FP32 logits whose BF16 pass overflows")
        return {**out, "not_read": reason}
    pick = {key: value[finite] for key, value in joined.items()}
    gap = math.sqrt(float(pick["gap_sq"].sum()) / (n * vocab))
    spread = math.sqrt(float(pick["var"].mean()))
    out.update(
        norm=float(pick["norm"].mean()),
        mean=float(pick["mean"].mean()),
        log_z=float(pick["log_z"].mean()),
        gap=gap,
        gap_relative=gap / spread if spread > 0 else None,
    )
    return out


def _attention_stats(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    scale: float | None,
    map_size: int,
) -> dict[str, Any]:
    """Attention statistics per head over the batch, and sample 0's map.

    Rows are query positions of one head and batch item. Three kinds are kept apart, so
    padding and numerical failures do not look like concentrated attention:

    - `masked_rows`: every key masked out, as for padded queries. Left out of the rest.
    - `invalid_rows`: a NaN or +Inf score among the visible keys. Left out of the rest.
    - informative rows: at least two visible keys. Only these count in `entropy` (nats)
      and `entropy_share`, each row's entropy divided by the log of its own visible key
      count, so causal masking does not lower it: uniform attention is 1.0 in every row.

    `max_logit` is the largest finite visible pre-softmax score per head. The map averages
    sample 0's valid rows over heads, sums keys and averages query rows within each bin of
    at most `map_size` × `map_size`; rows of the map with no valid query are zero.

    It runs inside the user's forward, perhaps under autocast, whose `matmul` would cast the
    float32 query and key back to FP16 or BF16 (scores above 65,504 then overflow to Inf in
    FP16 and are counted invalid). So it runs with autocast off (`_autocast_off`), as the
    head's numbers do. The probe's other work inside the forward, the histograms
    (`_histogram`: `isfinite`, `aminmax`, `histc`), calls no op that autocast casts to a lower
    precision on the CPU, CUDA or MPS (PyTorch's `autocast_mode.cpp`).
    """
    with _autocast_off(query.device):
        return _attention_stats_fp32(query, key, attn_mask, is_causal, scale, map_size)


def _attention_stats_fp32(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    scale: float | None,
    map_size: int,
) -> dict[str, Any]:
    """`_attention_stats`, which calls it with autocast off."""
    q = query.detach().float()
    k = key.detach().float()
    if q.ndim == 3:
        q, k = q.unsqueeze(1), k.unsqueeze(1)
        if attn_mask is not None and attn_mask.ndim == 3:
            attn_mask = attn_mask.unsqueeze(1)
    heads, rows, dim = q.shape[1:]
    keys = k.shape[-2]
    # SDPA broadcasts the batch dimension (Hugging Face passes key and value with batch 1
    # in some paths), so count the batch the scores actually have.
    batch = max(q.shape[0], k.shape[0])
    if k.shape[-3] != heads:
        # Grouped-query attention: SDPA's `enable_gqa` shares each key head across
        # several query heads, so repeat the keys to the query's head count.
        k = k.repeat_interleave(heads // k.shape[-3], dim=-3)
    scale = scale if scale is not None else 1 / math.sqrt(dim)

    dev = q.device
    max_logit = torch.full((heads,), -math.inf, device=dev)
    entropy = torch.zeros(heads, device=dev)
    share = torch.zeros(heads, device=dev)
    informative = torch.zeros(heads, device=dev)
    masked_rows = torch.zeros((), dtype=torch.int64, device=dev)
    invalid_rows = torch.zeros((), dtype=torch.int64, device=dev)
    out_rows, out_cols = min(rows, map_size), min(keys, map_size)
    attn_map = torch.zeros(out_rows, out_cols, device=dev)
    row_hits = torch.zeros(out_rows, device=dev)
    step = max(1, CHUNK_ELEMENTS // max(1, batch * heads * keys))
    cols = torch.arange(keys, device=dev)
    col_bins = cols * out_cols // keys
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        scores = torch.matmul(q[:, :, r0:r1], k.transpose(-1, -2)) * scale
        valid = torch.ones_like(scores, dtype=torch.bool)
        if is_causal:
            # SDPA's is_causal aligns the mask to the top left: query i sees keys 0..i.
            valid &= cols <= torch.arange(r0, r1, device=dev)[:, None]
        if attn_mask is not None:
            mask = attn_mask[..., r0:r1, :] if attn_mask.shape[-2] == rows else attn_mask
            if mask.dtype == torch.bool:
                valid &= mask
            else:
                scores = scores + mask.float()
        bad = valid & (scores.isnan() | (scores == math.inf))
        visible = valid & ~bad & (scores > -math.inf)
        n_visible = visible.sum(-1)  # [batch, heads, chunk rows]
        row_invalid = bad.any(-1)
        row_masked = (n_visible == 0) & ~row_invalid
        row_ok = ~row_invalid & ~row_masked
        invalid_rows += row_invalid.sum()
        masked_rows += row_masked.sum()

        scores = scores.masked_fill(~visible, -math.inf)
        max_logit = torch.maximum(max_logit, scores.amax(dim=(0, 2, 3)))
        probs = torch.softmax(scores, dim=-1).nan_to_num(0.0) * row_ok.unsqueeze(-1)
        ent = -(probs * torch.log(probs.clamp_min(1e-30))).sum(-1)
        rich = row_ok & (n_visible >= 2)
        entropy += (ent * rich).sum(dim=(0, 2))
        share += (ent / n_visible.clamp_min(2).float().log() * rich).sum(dim=(0, 2))
        informative += rich.sum(dim=(0, 2))

        ok0 = row_ok[0].float()  # [heads, chunk rows]
        heads_ok = ok0.sum(0)
        sample = (probs[0] * ok0.unsqueeze(-1)).sum(0) / heads_ok.clamp_min(1).unsqueeze(-1)
        # Sum the keys in each column bin, so each pooled row stays a distribution.
        pooled = sample.new_zeros(r1 - r0, out_cols).index_add_(1, col_bins, sample)
        bins = torch.arange(r0, r1, device=dev) * out_rows // rows
        attn_map.index_add_(0, bins, pooled)
        row_hits.index_add_(0, bins, (heads_ok > 0).float())
    attn_map /= row_hits.clamp_min(1).unsqueeze(1)
    counted = informative.clamp_min(1)
    none_counted = informative == 0
    nan = torch.full_like(entropy, math.nan)
    return {
        "max_logit": max_logit.tolist(),
        "entropy": torch.where(none_counted, nan, entropy / counted).tolist(),
        "entropy_share": torch.where(none_counted, nan, share / counted).tolist(),
        "masked_rows": int(masked_rows),
        "invalid_rows": int(invalid_rows),
        "rows": batch * heads * rows,
        "query_len": rows,
        "key_len": keys,
        "map": attn_map.tolist(),
    }


# The probe reads attention by replacing `F.scaled_dot_product_attention`, which is one
# attribute for the whole process. While any probe is open one wrapper is installed: it is made,
# around the function then in place, when the first probe opens, and taken out when the last one
# closes (`_open_probe`). It reads a call only on a thread with a probe open, the newest one
# there, and only while it is the installed wrapper; any other call goes straight to the function
# it wraps, with the arguments it was given. A wrapper that something else wrapped in turn is
# left in place when its probes close and from then on only passes calls through, so a later
# probe's wrapper, made around that chain, never calls itself.
# The compiler's stance is also the process's (`torch._dynamo.eval_frame._stance`, a module
# global in PyTorch 2.6 and later), so it too is set by the first probe to open and put back by
# the last to close: two probes that closed in the order they opened used to leave the process
# on `force_eager`.
_sdpa_lock = threading.Lock()
_sdpa_installed: tuple[Any, Any] | None = None  # (wrapper, the function it wraps) while open
_sdpa_open = 0
_open_stack: contextlib.ExitStack | None = None  # undoes the first probe's process-wide changes
_sdpa_probes: dict[int, list[ProbeState]] = {}  # thread id -> its open probes, newest last


def _make_probe_sdpa(original: Any) -> Any:
    def probe_sdpa(*args: Any, **kwargs: Any) -> Any:
        if torch.compiler.is_compiling():
            return original(*args, **kwargs)
        installed = _sdpa_installed
        probes = _sdpa_probes.get(threading.get_ident())
        if not probes or installed is None or installed[0] is not probe_sdpa:
            return original(*args, **kwargs)
        out = original(*args, **kwargs)
        _note_attention(probes[-1], args, kwargs)
        return out

    return probe_sdpa


def _sdpa_arguments(query: Any, key: Any, value: Any, attn_mask: Any = None, dropout_p: float = 0.0,
                    is_causal: bool = False, scale: Any = None, **_: Any) -> tuple[Any, Any, Any, bool, Any]:
    return query, key, attn_mask, is_causal, scale


def _note_attention(state: ProbeState, args: tuple, kwargs: dict) -> None:
    """Record one call's statistics. The call itself has already returned; statistics that
    cannot be read from inputs SDPA accepts (5-D query and key, no keys) are that entry's
    `error`, and the rest of the probe goes on."""
    try:
        name = state.block or "attention"
        label, n = name, 1
        while label in state.attention:
            label, n = f"{name}#{n}", n + 1
    except Exception:  # noqa: BLE001  # bookkeeping never reaches the forward
        return
    try:
        query, key, attn_mask, is_causal, scale = _sdpa_arguments(*args, **kwargs)
        state.attention[label] = _attention_stats(query, key, attn_mask, is_causal, scale, state.map_size)
    except Exception as error:  # noqa: BLE001
        state.attention[label] = {"error": f"{type(error).__name__}: {error}"}


def _uninstall_sdpa() -> None:
    global _sdpa_installed
    if _sdpa_installed is None:
        return
    wrapper, original = _sdpa_installed
    _sdpa_installed = None
    # Something that replaced the wrapper since stays; the wrapper it may call now passes
    # every call through.
    if F.scaled_dot_product_attention is wrapper:
        F.scaled_dot_product_attention = original


@contextlib.contextmanager
def _open_probe(state: ProbeState) -> Iterator[None]:
    """What a probe changes for the whole process, counted across open probes: the eager
    stance (`_eager_stance`) and the SDPA wrapper (`_make_probe_sdpa`), which reads the calls
    this thread makes while this probe is its newest."""
    global _sdpa_installed, _sdpa_open, _open_stack
    thread = threading.get_ident()
    with _sdpa_lock:
        if _sdpa_open == 0:
            stack = contextlib.ExitStack()
            try:
                stack.enter_context(_eager_stance())
                original = F.scaled_dot_product_attention
                wrapper = _make_probe_sdpa(original)
                F.scaled_dot_product_attention = wrapper
                _sdpa_installed = (wrapper, original)
                stack.callback(_uninstall_sdpa)
            except BaseException:
                stack.close()
                raise
            _open_stack = stack
        _sdpa_open += 1
        _sdpa_probes.setdefault(thread, []).append(state)
    try:
        yield
    finally:
        with _sdpa_lock:
            probes = _sdpa_probes.get(thread, [])
            if state in probes:
                # The newest entry for this state: probes on one thread close in reverse order.
                del probes[len(probes) - 1 - probes[::-1].index(state)]
            if not probes:
                _sdpa_probes.pop(thread, None)
            _sdpa_open -= 1
            if _sdpa_open == 0 and _open_stack is not None:
                stack, _open_stack = _open_stack, None
                stack.close()


def _configs(config: Any) -> list[tuple[Any, Any]]:
    """The model's config and its nested sub-configs (a Llava config's `text_config` and
    `vision_config`), each with its attention setting, a config before its own sub-configs."""
    found: list[tuple[Any, Any]] = []
    seen: set[int] = set()

    def visit(c: Any) -> None:
        if c is None or id(c) in seen:
            return
        seen.add(id(c))
        found.append((c, getattr(c, "_attn_implementation", None)))
        subs = getattr(c, "sub_configs", None)
        for key in subs if isinstance(subs, Mapping) else ():
            visit(getattr(c, key, None))

    visit(config)
    return found


def _set_attention(configs: list[tuple[Any, Any]], values: list[Any]) -> None:
    """Give each config its value, through its own setter, in order. Transformers' setter
    writes a config's value into every sub-config below it too, so a config is set before its
    sub-configs (the order `_configs` lists them in) and each is written only when it differs."""
    for (config, _), value in zip(configs, values):
        try:
            if getattr(config, "_attn_implementation", None) != value:
                config._attn_implementation = value
        except Exception:  # noqa: BLE001  # the others are still set
            pass


def _attention_config(config: Any) -> bool:
    """Whether `config` is a model's, with an attention setting: a LightningModule's own
    `config` of hyperparameters (a dict, a namespace) is not, so the model inside is asked."""
    if config is None:
        return False
    try:
        return hasattr(config, "_attn_implementation")
    except Exception:  # noqa: BLE001
        return False


@contextlib.contextmanager
def _sdpa_instead_of_flash(model: nn.Module) -> Iterator[bool]:
    """Hugging Face models loaded with FlashAttention never call SDPA; switch them to
    SDPA for the probe. Each config set to FlashAttention, the model's own or a sub-config's,
    is switched and the others are left; afterwards every one has its own setting back.
    A model with no attention `config` of its own (a LightningModule or a DDP around a Hugging
    Face model) has those of the outermost modules inside it that have one (`_inner_models`).
    Yields whether a switch happened."""
    try:
        own = getattr(model, "config", None)
        if _attention_config(own):
            configs = _configs(own)
        else:
            configs = []
            ids: set[int] = set()
            for _, inner in _inner_models(model, lambda m: _attention_config(getattr(m, "config", None))):
                for entry in _configs(getattr(inner, "config", None)):
                    if id(entry[0]) not in ids:
                        ids.add(id(entry[0]))
                        configs.append(entry)
    except Exception:  # noqa: BLE001
        configs = []
    before = [impl for _, impl in configs]
    during = ["sdpa" if impl in FLASH_IMPLEMENTATIONS else impl for impl in before]
    if during == before:
        yield False
        return
    try:
        _set_attention(configs, during)
        yield True
    finally:
        _set_attention(configs, before)


@contextlib.contextmanager
def _fork_host_rng() -> Iterator[None]:
    """Python's `random` and, when NumPy is already imported, NumPy's global state are as they
    were afterwards (`torch.random.fork_rng` does it for PyTorch's generators). Both are the
    process's: a draw another thread makes in between is rolled back too, as with fork_rng."""
    python = random.getstate()
    numpy = sys.modules.get("numpy")
    try:
        numpy_state = numpy.random.get_state() if numpy is not None else None
    except Exception:  # noqa: BLE001
        numpy_state = None
    try:
        yield
    finally:
        random.setstate(python)
        if numpy_state is not None:
            try:
                numpy.random.set_state(numpy_state)
            except Exception:  # noqa: BLE001
                pass


def _eager_stance() -> contextlib.AbstractContextManager[Any]:
    set_stance = getattr(torch.compiler, "set_stance", None)
    return set_stance("force_eager") if set_stance is not None else contextlib.nullcontext()


def run_probe(
    model: nn.Module,
    fn: Callable[[nn.Module], Any],
    state: ProbeState,
    devices: list[torch.device],
) -> dict[str, Any]:
    """Run `fn(model)` once with probing on and return what the hooks recorded.

    The deterministic-algorithms setting is the process's and is left as it is: with it on,
    the histograms count without `histc` (`_histc_counts`) and `index_add_` takes PyTorch's
    deterministic path on CUDA, so the probe's own work runs either way."""
    training = {module: module.training for module in model.modules()}
    cuda = [d for d in devices if d.type == "cuda"]
    state.histograms, state.attention, state.block = {}, {}, None
    state.head_calls, state.head_problem = [], None
    state.thread = threading.get_ident()
    try:
        model.eval()
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.random.fork_rng(devices=cuda))
            stack.enter_context(_fork_host_rng())
            stack.enter_context(torch.no_grad())
            switched = stack.enter_context(_sdpa_instead_of_flash(model))
            stack.enter_context(_open_probe(state))
            state.active = True
            try:
                fn(model)
            finally:
                state.active = False
    finally:
        # Restore each module's own flag rather than calling model.train(), which would
        # unfreeze submodules kept in eval mode and trip Dynamo's `training` guards.
        for module, flag in training.items():
            module.training = flag
    result: dict[str, Any] = {"histograms": state.histograms, "attention": state.attention}
    if switched:
        result["attention_note"] = "FlashAttention switched to SDPA for the probe"
    elif not state.attention:
        result["attention_note"] = "no scaled_dot_product_attention call seen"
    if not state.head_off:
        result["head"] = _head_result(state)
    state.histograms, state.attention, state.block = {}, {}, None
    state.head_calls, state.head_problem = [], None
    return result


def _head_result(state: ProbeState) -> dict[str, Any]:
    if state.head_name is None:
        return {"not_read": state.head_note or "no head"}
    found = {"module": state.head_name, "found_by": state.head_found_by}
    if state.head_problem is not None:
        return {**found, "not_read": state.head_problem}
    if not state.head_calls:
        return {**found, "not_read": "the head was not called in the probe's forward"}
    return {**found, **_head_numbers(state.head_calls)}


def probe_callable(probe: Any) -> Callable[[nn.Module], Any]:
    """Accept a callable `fn(model)`, a batch tensor, or a mapping of keyword inputs."""
    if callable(probe) and not isinstance(probe, torch.Tensor):
        return probe
    if isinstance(probe, torch.Tensor):
        return lambda model: model(probe)
    if isinstance(probe, Mapping):
        return lambda model: model(**probe)
    raise TypeError("probe must be a callable taking the model, a tensor, or a mapping of inputs")
