"""Cross-GPU replica drift detector: float64 norm and sum fingerprints per parameter."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.distributed as dist

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover
    DTensor = None  # type: ignore[assignment,misc]

F64 = torch.float64
CHUNK = 1 << 24  # elements per float64 chunk: at most 128 MB extra memory
PROJECTION_CHUNK = 1 << 22  # the projection also builds an int64 index chunk: 32 MB + 32 MB
FINGERPRINT_WIDTH = 3  # norm, sum, seeded projection
_MASK = 0xFFFFFFFF


def unwrap_local(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local shard if `tensor` is a DTensor, otherwise `tensor`."""
    if DTensor is not None and isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _as_real(p: torch.Tensor) -> torch.Tensor:
    """`p` as real numbers: a complex tensor as its real pairs (`torch.view_as_real`, a view, after
    resolving a conjugate view to the values it shows), so its norm is |z|'s and its sum and
    projection read both parts, as `capture._read_updates` reads Adam's moments; any other
    tensor as it is."""
    return torch.view_as_real(p.resolve_conj()) if p.is_complex() else p


def _sum_f64(p: torch.Tensor) -> torch.Tensor:
    # p.sum(dtype=float64) on a bf16 tensor copies the whole tensor to float64 first.
    # Apple MPS rejects float64, so move chunks to CPU before widening when on MPS.
    # A complex tensor's sum is that of its real and imaginary parts together.
    p = _as_real(p)
    flat = p.reshape(-1)
    out_device = torch.device("cpu") if p.device.type == "mps" else p.device
    if flat.numel() == 0:
        return torch.zeros((), dtype=F64, device=out_device)
    return torch.stack(
        [(c.cpu() if c.device.type == "mps" else c).sum(dtype=F64) for c in flat.split(CHUNK)]
    ).sum()


def _norms_f64(tensors: list[torch.Tensor]) -> torch.Tensor:
    # _foreach_norm's fused CUDA kernel requires each call's list to share one
    # device and dtype, and be contiguous; mixed dtypes fall back to per-tensor
    # vector_norm, which copies bf16 tensors to float64 first. A complex tensor's norm is
    # that of its real pairs, which is |z|'s.
    tensors = [_as_real(p) for p in tensors]
    norms: list[torch.Tensor | None] = [None] * len(tensors)
    groups: dict[tuple[torch.device, torch.dtype], list[tuple[int, torch.Tensor]]] = {}
    foreach = getattr(torch, "_foreach_norm", None)
    for i, p in enumerate(tensors):
        out_device = torch.device("cpu") if p.device.type == "mps" else p.device
        if foreach is not None and p.is_cuda and p.is_contiguous() and p.numel() > 0:
            groups.setdefault((p.device, p.dtype), []).append((i, p))
        elif p.numel() == 0:
            norms[i] = torch.zeros((), dtype=F64, device=out_device)
        else:
            flat = p.reshape(-1)
            norms[i] = torch.linalg.vector_norm(
                torch.stack(
                    [
                        torch.linalg.vector_norm(
                            c.cpu() if c.device.type == "mps" else c, dtype=F64
                        )
                        for c in flat.split(CHUNK)
                    ]
                )
            )
    for group in groups.values():
        idxs, ts = zip(*group)
        for i, n in zip(idxs, foreach(ts, 2, dtype=F64)):
            norms[i] = n
    return torch.stack(norms)  # type: ignore[arg-type]


def _projection_weights(start: int, count: int, seed: int, device: torch.device) -> torch.Tensor:
    """Deterministic float64 weights in [-1, 1) for elements `start .. start + count`.

    Integer arithmetic on the element index (a 32-bit multiply-xorshift mix), so every rank
    and every device computes the same weights, and the training run's random number
    generators are never touched. `seed` is the parameter's position in the checked list.
    """
    idx = torch.arange(start, start + count, dtype=torch.int64, device=device)
    h = (idx * 0x9E3779B1 + (seed + 1) * 0x7F4A7C15) & _MASK
    h = ((h ^ (h >> 16)) * 0x85EBCA6B) & _MASK
    h = ((h ^ (h >> 13)) * 0xC2B2AE35) & _MASK
    h = h ^ (h >> 16)
    return h.to(F64) * (2.0 / 4294967296.0) - 1.0


def _projection_f64(p: torch.Tensor, seed: int) -> torch.Tensor:
    """`sum_i w_i * p_i` in float64 with the seeded weights `w`; a swap of two unequal
    elements changes it, where a norm and a sum do not. A complex tensor is projected as its
    real pairs, so a change in either part moves it."""
    flat = _as_real(p).reshape(-1)
    device = torch.device("cpu") if p.device.type == "mps" else p.device
    if flat.numel() == 0:
        return torch.zeros((), dtype=F64, device=device)
    parts = []
    for i, chunk in enumerate(flat.split(PROJECTION_CHUNK)):
        chunk = chunk.cpu() if chunk.device.type == "mps" else chunk
        weights = _projection_weights(i * PROJECTION_CHUNK, chunk.numel(), seed, chunk.device)
        parts.append((chunk.to(F64) * weights).sum())
    return torch.stack(parts).sum()


@torch.no_grad()
def parameter_fingerprints(params: Iterable[torch.Tensor], *, projection: bool = True) -> torch.Tensor:
    """Compute float64 fingerprints for `params`: `[n_params, 3]` `(norm, sum, projection)`,
    or `[n_params, 2]` without the seeded projection (`projection=False`).

    The projection is a dot product with fixed pseudo-random weights, so it catches changes
    that keep the norm and the sum, such as two elements swapping places. It is a screen:
    a change that keeps all three values is possible, just far less likely, and equal
    fingerprints are not proof that tensors are equal. A complex tensor is read as its real
    pairs (`torch.view_as_real`): the same three values as the float tensor of those pairs.
    """
    local = [unwrap_local(p) for p in params]
    if not local:
        raise ValueError("no replicated parameters to check")
    columns = [_norms_f64(local), torch.stack([_sum_f64(p) for p in local])]
    if projection:
        columns.append(torch.stack([_projection_f64(p, i) for i, p in enumerate(local)]))
    return torch.stack(columns, dim=1)


@torch.no_grad()
def spread_from_fingerprints(allfp: torch.Tensor) -> torch.Tensor:
    """Given `[ranks, n_params, k]` float64 fingerprints (`k` is 2 or 3), return the
    `[n_params, k]` spread relative to each tensor's mean norm."""
    if allfp.ndim != 3 or allfp.shape[-1] not in (2, FINGERPRINT_WIDTH):
        raise ValueError(f"expected shape [ranks, n_params, 2 or 3], got {tuple(allfp.shape)}")
    spread = allfp.max(0).values - allfp.min(0).values
    scale = allfp[..., 0].mean(0).clamp_min(1e-12)  # mean norm per tensor
    return spread / scale[:, None]


@torch.no_grad()
def replica_spread(
    params: Iterable[torch.Tensor], group: dist.ProcessGroup, *, projection: bool = True
) -> torch.Tensor:
    """Compare replicated parameters across `group` and return `[n_params, 3]` relative spread
    (norm, sum, seeded projection).

    `params`: tensors meant to be identical on every rank in `group`.
    `projection=False` skips the seeded projection (two columns instead of three), which
    costs about seven times the norm and sum on CPU; its GPU cost is not yet measured.
    `group`: the replication group. Pass `dist.group.WORLD` only for plain DDP;
    under tensor, pipeline or hybrid-FSDP parallelism WORLD compares the wrong ranks.
    """
    if group is None:
        raise ValueError("pass the replication process group explicitly")
    fp = parameter_fingerprints(params, projection=projection)
    out = [torch.empty_like(fp) for _ in range(dist.get_world_size(group))]
    dist.all_gather(out, fp, group=group)  # 24 bytes per tensor per rank
    return spread_from_fingerprints(torch.stack(out))
