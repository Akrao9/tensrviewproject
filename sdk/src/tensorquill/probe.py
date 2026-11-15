"""Tier 2 probe pass: activation histograms and attention statistics on a fixed batch.

Every `probe_every` steps the session runs one fixed batch through the model without the
compiler, in eval mode and without gradients, and records for each repeated block a
histogram of its output, and for each attention call the largest pre-softmax logit and
the attention entropy per head, plus a downsampled attention map. Fused attention kernels
never store the attention map, so the probe recomputes it from the query and key.
"""

from __future__ import annotations

import contextlib
import math
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
    which runs without the compiler, never causes a recompile.
    """

    def __init__(self) -> None:
        self.active = False
        self.block: str | None = None
        self.histograms: dict[str, dict[str, Any]] = {}
        self.attention: dict[str, dict[str, Any]] = {}
        self.bins = HIST_BINS
        self.map_size = MAP_SIZE


def install_probe_hooks(blocks: list[tuple[str, nn.Module]], state: ProbeState) -> list[Any]:
    handles = []
    for name, module in blocks:
        def pre(_module: nn.Module, _args: Any, name: str = name) -> None:
            if state.active:
                state.block = name

        def post(_module: nn.Module, _args: Any, output: Any, name: str = name) -> None:
            if state.active:
                tensor = _first_float(output)
                if tensor is not None:
                    state.histograms[name] = _histogram(tensor, state.bins)
                state.block = None

        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))
    return handles


def _first_float(output: Any) -> torch.Tensor | None:
    items = output if isinstance(output, (tuple, list)) else (output,)
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
        lo, hi = lo - 0.5, hi + 0.5
    counts = torch.histc(values, bins=bins, min=lo, max=hi)
    return {"lo": lo, "hi": hi, "counts": [int(c) for c in counts.tolist()], "nonfinite": nonfinite}


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
    """
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


@contextlib.contextmanager
def _interpose_sdpa(state: ProbeState) -> Iterator[None]:
    original = F.scaled_dot_product_attention

    def probe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        out = original(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale, **kw,
        )
        name = state.block or "attention"
        label, n = name, 1
        while label in state.attention:
            label, n = f"{name}#{n}", n + 1
        state.attention[label] = _attention_stats(query, key, attn_mask, is_causal, scale, state.map_size)
        return out

    F.scaled_dot_product_attention = probe_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original


@contextlib.contextmanager
def _sdpa_instead_of_flash(model: nn.Module) -> Iterator[bool]:
    """Hugging Face models loaded with FlashAttention never call SDPA; switch them to
    SDPA for the probe. Yields whether the switch happened."""
    config = getattr(model, "config", None)
    impl = getattr(config, "_attn_implementation", None)
    if impl not in FLASH_IMPLEMENTATIONS:
        yield False
        return
    config._attn_implementation = "sdpa"
    try:
        yield True
    finally:
        config._attn_implementation = impl


def _eager_stance() -> contextlib.AbstractContextManager[Any]:
    set_stance = getattr(torch.compiler, "set_stance", None)
    return set_stance("force_eager") if set_stance is not None else contextlib.nullcontext()


def run_probe(
    model: nn.Module,
    fn: Callable[[nn.Module], Any],
    state: ProbeState,
    devices: list[torch.device],
) -> dict[str, Any]:
    """Run `fn(model)` once with probing on and return what the hooks recorded."""
    training = {module: module.training for module in model.modules()}
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cuda = [d for d in devices if d.type == "cuda"]
    state.histograms, state.attention, state.block = {}, {}, None
    try:
        model.eval()
        # `histc` and `index_add_` have no deterministic CUDA kernels; the probe's
        # results do not feed training, so it turns the check off while it runs.
        torch.use_deterministic_algorithms(False)
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.random.fork_rng(devices=cuda))
            stack.enter_context(torch.no_grad())
            stack.enter_context(_eager_stance())
            switched = stack.enter_context(_sdpa_instead_of_flash(model))
            stack.enter_context(_interpose_sdpa(state))
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
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    result: dict[str, Any] = {"histograms": state.histograms, "attention": state.attention}
    if switched:
        result["attention_note"] = "FlashAttention switched to SDPA for the probe"
    elif not state.attention:
        result["attention_note"] = "no scaled_dot_product_attention call seen"
    state.histograms, state.attention, state.block = {}, {}, None
    return result


def probe_callable(probe: Any) -> Callable[[nn.Module], Any]:
    """Accept a callable `fn(model)`, a batch tensor, or a mapping of keyword inputs."""
    if callable(probe) and not isinstance(probe, torch.Tensor):
        return probe
    if isinstance(probe, torch.Tensor):
        return lambda model: model(probe)
    if isinstance(probe, Mapping):
        return lambda model: model(**probe)
    raise TypeError("probe must be a callable taking the model, a tensor, or a mapping of inputs")
