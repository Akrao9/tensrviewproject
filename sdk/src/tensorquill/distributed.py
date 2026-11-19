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


def _rows(items: list[tuple[int, torch.Tensor]], projection: bool) -> torch.Tensor:
    """The fingerprints of `(position, tensor)` pairs, one row each; a tensor's position in the
    checked list seeds its projection."""
    tensors = [p for _, p in items]
    columns = [_norms_f64(tensors), torch.stack([_sum_f64(p) for p in tensors])]
    if projection:
        columns.append(torch.stack([_projection_f64(p, i) for i, p in items]))
    return torch.stack(columns, dim=1)


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
    return _rows(list(enumerate(local)), projection)


# Why a parameter's fingerprints are not read, by the code `_fingerprint_table` puts in its last
# column (0: read; code `i` is `_UNREAD[i - 1]`). All but the last two are judged from the
# tensor's device, layout and type alone, which hold no values, so no rank reads data to decide
# them. The codes travel between ranks, so a new one is only ever added at the end.
_SPARSE = ("sparse_coo", "sparse_csr", "sparse_csc", "sparse_bsr", "sparse_bsc")
_UNREAD = (
    "it is on meta, which holds no values",
    "it is a nested tensor",
    *(f"it is a {layout} tensor, not a dense one" for layout in _SPARSE),
    "it is not a dense tensor",
    "reading it raised an error",
    "the drift check's gather cannot run on this rank's replica group",
)
_RAISED = _UNREAD.index("reading it raised an error") + 1
_NOT_GATHERED = len(_UNREAD)  # set by `_spread_and_unread` on every row, never gathered


def _unread_code(p: torch.Tensor) -> int:
    if p.device.type == "meta":
        return 1
    if p.is_nested:
        return 2
    if p.layout != torch.strided:
        name = str(p.layout).removeprefix("torch.")
        return 3 + _SPARSE.index(name) if name in _SPARSE else 3 + len(_SPARSE)
    return 0


def _unread_reason(code: float) -> str:
    """The reason a code in the last column of `_fingerprint_table` stands for."""
    index = int(code) - 1
    return _UNREAD[index] if 0 <= index < len(_UNREAD) else _UNREAD[_RAISED - 1]


@torch.no_grad()
def _fingerprint_table(
    params: Iterable[torch.Tensor], *, projection: bool, device: torch.device | None = None
) -> tuple[torch.Tensor, list[str | None]]:
    """What `_spread_and_unread` gathers: `[n_params, width + 1]` float64 on `device`, each
    parameter's fingerprints (`parameter_fingerprints`, the same values, `width` 3 or 2) and then
    0.0 where they were read, or the code of why not (`_UNREAD`) with zeros for its values. Also
    each parameter's reason on this rank, or None.

    It never raises for a parameter, so a rank never leaves a collective the others enter: one
    on `meta`, nested or sparse is not read, and one whose reading raises is flagged instead.
    The shape depends only on how many parameters there are, never on which were read. A
    parameter's values are not looked at to decide, so a NaN weight is read, as before.

    `device`: where the table is built (the session's, which is never `meta`); by default
    where the first parameter that is read leaves its fingerprints, else the CPU."""
    local = [unwrap_local(p) for p in params]
    width = FINGERPRINT_WIDTH if projection else 2
    codes = [_unread_code(p) for p in local]
    reasons: list[str | None] = [_UNREAD[code - 1] if code else None for code in codes]
    if device is None:
        device = next(
            (torch.device("cpu") if p.device.type == "mps" else p.device for p, code in zip(local, codes) if not code),
            torch.device("cpu"),
        )
    table = torch.zeros(len(local), width + 1, dtype=F64, device=device)
    # One pass over every parameter, as `parameter_fingerprints` makes, with an empty tensor in
    # place of one not read (its row stays zero); only if that raises, one parameter at a time.
    empty = torch.zeros(0, dtype=F64, device=device)
    try:
        items = [(i, empty if code else p) for i, (p, code) in enumerate(zip(local, codes))]
        table[:, :width].copy_(_rows(items, projection))
    except Exception:  # noqa: BLE001
        for i, p in enumerate(local):
            if codes[i]:
                continue
            try:
                table[i, :width].copy_(_rows([(i, p)], projection)[0])
            except Exception as error:  # noqa: BLE001
                table[i, :width].zero_()
                codes[i] = _RAISED
                text = " ".join(f"{type(error).__name__}: {error}".split())
                reasons[i] = f"reading it raised {text if len(text) <= 200 else text[:199] + '…'}"
    for i, code in enumerate(codes):
        if code:
            table[i, width].fill_(float(code))  # a fill with a number: no copy from the host
    return table, reasons


def _gather_problem(group: object, device: torch.device) -> str | None:
    """Why the drift check's `all_gather` of a table on `device` cannot run on `group`, or None.

    Judged on the host, with no communication, from what every rank of the group sees alike in
    a data-parallel job: whether a default process group exists, the group's backends (the same
    on every rank) and the type of the session's device (the same kind of device on every
    replica). So every rank that would fail here skips, no rank enters a gather its peers skip,
    and none is left waiting. The two failures this catches raise at the call, before anything
    is sent: a group with no backend for the device's type (NCCL serves CUDA only, so a CPU
    table under NCCL, as with FSDP's CPU offload), and a destroyed process group. A rank
    outside `group` skips too: PyTorch's own collectives return at once on such a rank, so its
    peers in the group never wait for it.

    A group that is not a `ProcessGroup` (a stand-in some tests and tools pass with their own
    `all_gather`) is not judged, and anything that cannot be read here leaves the gather as it
    was: this never decides from something that might differ between ranks."""
    try:
        if dist.distributed_c10d._rank_not_in_group(group):  # GroupMember.NON_GROUP_MEMBER
            return "this rank is not a member of the replica group, so it has no replica to compare with"
        if not isinstance(group, dist.ProcessGroup):
            return None
        if not dist.is_initialized():
            return "torch.distributed has no default process group (it was destroyed, or never initialized)"
    except Exception:  # noqa: BLE001  an API this PyTorch does not have: decide nothing
        return None
    try:
        group._get_backend(device)
    except Exception as error:  # noqa: BLE001  "No backend type associated with device type cpu"
        if not hasattr(group, "_get_backend"):
            return None
        try:
            name = str(dist.get_backend(group))
        except Exception:  # noqa: BLE001
            name = "?"
        text = " ".join(f"{type(error).__name__}: {error}".split())[:160]
        return (
            f"the replica group (backend {name!r}) has no backend for {device.type} tensors, where the session "
            f"builds the check's table ({text}); under NCCL that is a session whose parameters are not on CUDA"
        )
    return None


@torch.no_grad()
def _spread_and_unread(
    params: Iterable[torch.Tensor], group: dist.ProcessGroup, *, projection: bool, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor, list[str | None]]:
    """The session's cross-GPU check: one `all_gather` of `_fingerprint_table`, which every rank
    enters with the same shape whatever it could read. Returns the `[n_params, width]` spread
    (`spread_from_fingerprints`), for each parameter 0.0 where every rank read it or else the
    largest code any rank gave (the same on every rank, since all see every rank's codes, so all
    leave out the same rows), and this rank's own reasons. The spread of a row not read on some
    rank means nothing: it is left out by its code, never by its value, so a NaN still reads
    as drift and a zero as agreement only where the weights were read. Both stay on the device.

    Where the gather cannot run (`_gather_problem`: no backend for the table's device, no
    process group, a rank outside the group), no collective is issued: every row comes back
    with the code `_NOT_GATHERED`, zeros for its spread and the reason, so the session leaves
    every parameter out of `replica_spread` and names each once, as for a parameter that could
    not be read. Every rank of the group decides this alike, so none waits for another. A gather
    that raises once issued is not caught: a rank that caught it and went on would issue its
    next collective (the training run's own) while its peers still wait in this one."""
    params = list(params)
    if device is None:
        locals_ = [unwrap_local(p) for p in params]
        device = next(
            (torch.device("cpu") if p.device.type == "mps" else p.device for p in locals_ if not _unread_code(p)),
            torch.device("cpu"),
        )
    why = _gather_problem(group, device)
    if why is not None:
        width = FINGERPRINT_WIDTH if projection else 2
        n = len(params)
        spread = torch.zeros(n, width, dtype=F64, device=device)
        unread = torch.full((n,), float(_NOT_GATHERED), dtype=F64, device=device)
        return spread, unread, [f"the drift check's gather cannot run: {why}"] * n
    table, reasons = _fingerprint_table(params, projection=projection, device=device)
    out = [torch.empty_like(table) for _ in range(dist.get_world_size(group))]
    dist.all_gather(out, table, group=group)  # 32 bytes per tensor per rank (24 without the projection)
    gathered = torch.stack(out)
    width = table.shape[1] - 1
    return spread_from_fingerprints(gathered[..., :width]), gathered[..., width].amax(0), reasons


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
