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


def _not_plain_linear(module: nn.Module) -> str | None:
    """Why `module` cannot be read as the head, or None. Only `nn.Linear`'s own forward is
    known to compute `x @ weight.T + bias`; a subclass with its own forward (quantized,
    sharded, soft-capped) may compute something else, so it is refused, not guessed."""
    if not isinstance(module, nn.Linear):
        return f"is a {type(module).__name__}, not an nn.Linear"
    if type(module).forward is not nn.Linear.forward:
        return f"is a {type(module).__name__}, a subclass of nn.Linear with its own forward"
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
    if not callable(getter):
        return None, None, "the model names no output head; pass probe_head=<module name> to read one"
    try:
        module = getter()
    except Exception as error:  # noqa: BLE001
        return None, None, f"get_output_embeddings() raised {type(error).__name__}; pass probe_head=<module name>"
    if module is None:
        return None, None, "get_output_embeddings() returned None: the model has no language-model head"
    name = next((n for n, m in modules.items() if m is module), None)
    if name is None:
        return None, None, "get_output_embeddings() returned a module outside the model"
    reason = _not_plain_linear(module)
    if reason:
        return None, None, f"{name!r} from get_output_embeddings() {reason}; the probe reads only a plain nn.Linear head"
    return name, "get_output_embeddings", None


def install_head_hook(head: nn.Module, state: ProbeState) -> list[Any]:
    """A permanent pre-hook on the head, a no-op unless the probe is running (as for the
    blocks, so compiled code never recompiles). It reads the head's input, the final
    hidden states, and the head's weight at the moment of the call: under FSDP the weight
    is gathered then, and may be sharded again once the forward is over."""

    def pre(module: nn.Module, args: Any) -> None:
        if not state.active or state.head_problem is not None:
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
    added in float64 (float32 on MPS, which has none). The spread is summed about each
    position's mean over the first chunk, so a large common shift of the logits does not
    cancel it away.
    """
    x = hidden.detach().reshape(-1, hidden.shape[-1])
    positions, vocab = x.shape[0], weight.shape[0]
    acc = torch.float32 if x.device.type == "mps" else torch.float64
    xf, xb = x.float(), x.to(torch.bfloat16)
    rows = max(1, CHUNK_ELEMENTS // max(1, positions))
    shift = None
    zero = torch.zeros(positions, dtype=acc, device=x.device)
    sq, dsum, dsq, gap = zero.clone(), zero.clone(), zero.clone(), zero.clone()
    lse = torch.full((positions,), -math.inf, dtype=acc, device=x.device)
    for v0 in range(0, vocab, rows):
        w = weight[v0 : v0 + rows].detach()
        b = bias[v0 : v0 + rows].detach() if bias is not None else None
        zf = F.linear(xf, w.float(), None if b is None else b.float())
        zb = F.linear(xb, w.to(torch.bfloat16), None if b is None else b.to(torch.bfloat16)).float()
        if shift is None:
            shift = zf.mean(-1, keepdim=True)
        # One pass each, reduced in float32 within the chunk and added up across chunks
        # in `acc`: per-element float64 sums cost as much as the two GEMMs on CPU.
        lse = torch.logaddexp(lse, torch.logsumexp(zf, -1).to(acc))
        sq += torch.linalg.vector_norm(zf, dim=-1).to(acc).square()
        gap += torch.linalg.vector_norm(zb.sub_(zf), dim=-1).to(acc).square()
        d = zf.sub_(shift)  # zf is not used again in this chunk
        dsum += d.sum(-1).to(acc)
        dsq += torch.linalg.vector_norm(d, dim=-1).to(acc).square()
    assert shift is not None
    mean_d = dsum / vocab
    return {
        "norm": sq.sqrt(),
        "mean": shift.squeeze(-1).to(acc) + mean_d,
        "var": (dsq / vocab - mean_d.square()).clamp_min(0),
        "log_z": lse,
        "gap_sq": gap,
        "vocab": torch.tensor(vocab),
    }


def _head_numbers(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """The four head numbers over every position the head saw in the probe (padding
    included: the probe has no mask for the head). Positions whose FP32 logits hold a NaN
    or an Inf are left out and counted. Over positions t and vocabulary entries v of the
    FP32 logits z:

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
    finite = torch.stack([v.isfinite() for k, v in joined.items() if k != "gap_sq"]).all(0)
    n = int(finite.sum())
    out: dict[str, Any] = {
        "positions": n, "nonfinite_positions": int(finite.numel() - n), "vocab": vocab, "calls": len(calls),
        "input_dtype": calls[0]["input_dtype"].removeprefix("torch."),
        "weight_dtype": calls[0]["weight_dtype"].removeprefix("torch."),
    }
    if n == 0:
        return {**out, "not_read": "every position's FP32 logits hold a NaN or an Inf"}
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
    state.head_calls, state.head_problem = [], None
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
