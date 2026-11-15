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


def unwrap_local(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local shard if `tensor` is a DTensor, otherwise `tensor`."""
    if DTensor is not None and isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _sum_f64(p: torch.Tensor) -> torch.Tensor:
    # p.sum(dtype=float64) on a bf16 tensor copies the whole tensor to float64 first.
    # Apple MPS rejects float64, so move chunks to CPU before widening when on MPS.
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
    # vector_norm, which copies bf16 tensors to float64 first.
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


@torch.no_grad()
def parameter_fingerprints(params: Iterable[torch.Tensor]) -> torch.Tensor:
    """Compute `[n_params, 2]` float64 `(norm, sum)` fingerprints for `params`."""
    local = [unwrap_local(p) for p in params]
    if not local:
        raise ValueError("no replicated parameters to check")
    norms = _norms_f64(local)
    sums = torch.stack([_sum_f64(p) for p in local])
    return torch.stack([norms, sums], dim=1)


@torch.no_grad()
def spread_from_fingerprints(allfp: torch.Tensor) -> torch.Tensor:
    """Given `[ranks, n_params, 2]` float64 fingerprints, return `[n_params, 2]` relative spread."""
    if allfp.ndim != 3 or allfp.shape[-1] != 2:
        raise ValueError(f"expected shape [ranks, n_params, 2], got {tuple(allfp.shape)}")
    spread = allfp.max(0).values - allfp.min(0).values
    scale = allfp[..., 0].mean(0).clamp_min(1e-12)  # mean norm per tensor
    return spread / scale[:, None]


@torch.no_grad()
def replica_spread(params: Iterable[torch.Tensor], group: dist.ProcessGroup) -> torch.Tensor:
    """Compare replicated parameters across `group` and return `[n_params, 2]` relative spread.

    `params`: tensors meant to be identical on every rank in `group`.
    `group`: the replication group. Pass `dist.group.WORLD` only for plain DDP;
    under tensor, pipeline or hybrid-FSDP parallelism WORLD compares the wrong ranks.
    """
    if group is None:
        raise ValueError("pass the replication process group explicitly")
    fp = parameter_fingerprints(params)
    out = [torch.empty_like(fp) for _ in range(dist.get_world_size(group))]
    dist.all_gather(out, fp, group=group)  # 16 bytes per tensor per rank
    return spread_from_fingerprints(torch.stack(out))
