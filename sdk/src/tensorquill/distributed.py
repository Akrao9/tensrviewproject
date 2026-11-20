"""Cross-GPU replica drift detector: float64 norm and sum fingerprints per parameter."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Sequence

import torch
import torch.distributed as dist

from .tracker import _say

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

    This rank's own values only, with no collective: a parameter it cannot read (on `meta`,
    sparse, nested, or tensors on different devices) raises here. `replica_spread` does not
    call it, so a rank that cannot read a parameter never leaves the others waiting.
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
    params: Iterable[torch.Tensor],
    *,
    projection: bool,
    device: torch.device | None = None,
    seeds: Sequence[int] | None = None,
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
    where the first parameter that is read leaves its fingerprints, else the CPU.

    `seeds`: each parameter's projection seed; by default its position in `params`, as the
    explicit gather has it. The observation passes `name-v1` seeds (`name_v1_seed`)."""
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
    seed = list(range(len(local))) if seeds is None else [int(v) for v in seeds]
    try:
        items = [(seed[i], empty if code else p) for i, (p, code) in enumerate(zip(local, codes))]
        table[:, :width].copy_(_rows(items, projection))
    except Exception:  # noqa: BLE001
        for i, p in enumerate(local):
            if codes[i]:
                continue
            try:
                table[i, :width].copy_(_rows([(seed[i], p)], projection)[0])
            except Exception as error:  # noqa: BLE001
                table[i, :width].zero_()
                codes[i] = _RAISED
                text = " ".join(f"{type(error).__name__}: {error}".split())
                reasons[i] = f"reading it raised {text if len(text) <= 200 else text[:199] + '…'}"
    for i, code in enumerate(codes):
        if code:
            table[i, width].fill_(float(code))  # a fill with a number: no copy from the host
    return table, reasons


def _alike_problem(group: object) -> str | None:
    """The part of `_gather_problem` every rank of the group judges alike, with no communication: a
    rank outside the group (PyTorch's collectives return at once there, so its peers in the group never
    wait for it) and a process group that was destroyed or never made. None otherwise, and for anything
    that cannot be read here. Never raises."""
    try:
        if dist.distributed_c10d._rank_not_in_group(group):  # GroupMember.NON_GROUP_MEMBER
            return "this rank is not a member of the replica group, so it has no replica to compare with"
        if isinstance(group, dist.ProcessGroup) and not dist.is_initialized():
            return "torch.distributed has no default process group (it was destroyed, or never initialized)"
    except Exception:  # noqa: BLE001  an API this PyTorch does not have: decide nothing
        return None
    return None


def _gather_problem(group: object, device: torch.device) -> str | None:
    """Why the drift check's `all_gather` of a table on `device` cannot run on `group`, or None.

    Judged on the host, with no communication, by this rank alone. The two failures this catches
    raise at the call, before anything is sent: a group with no backend for the device's type (NCCL
    serves CUDA only, so a CPU table under NCCL, as with FSDP's CPU offload), and a destroyed process
    group; and a rank outside `group` skips (`_alike_problem`). The device's type can differ between
    ranks (L5: sessions on different device types), so the backend verdict never decides alone
    whether a rank enters the gather: it is this rank's part of the exchange before it
    (`_exchange`), and every rank of the group takes the one decision the exchange gives.

    A group that is not a `ProcessGroup` (a stand-in some tests and tools pass with their own
    `all_gather`) is not judged, and anything that cannot be read here leaves the gather as it
    was."""
    alike = _alike_problem(group)
    if alike is not None or not isinstance(group, dist.ProcessGroup):
        return alike
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


# ---- the exchange before an explicit gather (L26, the author's decision 1 of Nov 20, 2026) --------
# An explicit gather (`replica_spread`, `Session.record_replica_spread(group)`) first exchanges one
# row per rank whose shape never depends on the inputs, then every rank of the group either gathers
# or skips, all alike, each saying the same why. As PyTorch's DDP all-gathers each rank's parameter
# count before it compares shapes (`verify_params_across_processes`, `reducer.cpp`), so unequal
# inputs never reach a collective of unequal shapes. What the row holds:
#   [0] the exchange's version, [1] how many parameters the rank compares, [2] its table's width (the
#   fingerprint columns and the code column), [3] its table's device type (`_DEVICE_TYPES`, 0 for
#   another), [4] 1 when the rank's own check (`_gather_problem`) says it cannot gather, [5] how many
#   bytes of that check's text follow, then the text's UTF-8 bytes, one to an element.
# The layout never changes: a new field is a new version, read by a rank that knows it, and a rank
# that sees a version it does not know skips with every other rank.
_EXCHANGE_VERSION = 1
_TEXT_BYTES = 240
_ROW_WIDTH = 6 + _TEXT_BYTES
_DEVICE_TYPES = ("cpu", "cuda", "mps", "meta", "xpu", "hpu", "mtia", "npu", "privateuseone")
# The device types tried, in this order, for the exchange's own row: the first the group has a
# backend for. The group's backends are the same on every rank, so every rank picks the same type.
_EXCHANGE_DEVICE_TYPES = ("cpu", "cuda", "xpu", "hpu", "npu", "mtia")


def _exchange_row(count: int, width: int, device: torch.device, problem: str | None) -> torch.Tensor:
    """One rank's row of the exchange (above), on the CPU: `_ROW_WIDTH` int64 whatever the inputs."""
    row = torch.zeros(_ROW_WIDTH, dtype=torch.int64)
    kind = device.type
    text = (problem or "").encode("utf-8")[:_TEXT_BYTES]
    head = [_EXCHANGE_VERSION, int(count), int(width), _DEVICE_TYPES.index(kind) + 1 if kind in _DEVICE_TYPES else 0,
            1 if problem else 0, len(text)]
    row[:6] = torch.tensor(head, dtype=torch.int64)
    if text:
        row[6:6 + len(text)] = torch.tensor(list(text), dtype=torch.int64)
    return row


def _ranks_text(ranks: Sequence[int]) -> str:
    return f"rank {ranks[0]}" if len(ranks) == 1 else "ranks " + ", ".join(map(str, ranks))


def _device_text(code: int) -> str:
    return _DEVICE_TYPES[code - 1] if 1 <= code <= len(_DEVICE_TYPES) else "another device type"


def _decide_gather(rows: Sequence[Any], ranks: Sequence[int]) -> str | None:
    """None when every rank can gather tables of one shape on one kind of device; else why no rank
    does. A pure function of the exchanged rows (and the group's global ranks, in group order), so the
    same on every rank, since every rank sees every row."""
    rows = [[int(v) for v in (row.tolist() if isinstance(row, torch.Tensor) else row)] for row in rows]
    ranks = list(ranks)
    odd = [rank for rank, row in zip(ranks, rows) if len(row) != _ROW_WIDTH or row[0] != _EXCHANGE_VERSION]
    if odd:
        return (
            f"{_ranks_text(odd)} of the replica group sent an exchange this TensorQuill cannot read (every rank "
            "must run a TensorQuill that gathers alike)"
        )
    problems = []
    for rank, row in zip(ranks, rows):
        if row[4]:
            size = max(0, min(row[5], _TEXT_BYTES))
            text = bytes(v & 0xFF for v in row[6:6 + size]).decode("utf-8", errors="ignore")
            problems.append((rank, text or "its own check says it cannot gather"))
    if problems:
        rank, text = problems[0]
        more = f" (as do {_ranks_text([r for r, _ in problems[1:]])})" if len(problems) > 1 else ""
        return f"rank {rank} of the replica group cannot take part{more}: {text}"
    empty = [rank for rank, row in zip(ranks, rows) if row[1] == 0]
    if empty:
        return f"{_ranks_text(empty)} of the replica group compare{'s' if len(empty) == 1 else ''} no parameter"
    for field, value, what in (
        (3, _device_text, "the replicas' tables are on different device types ({}), and fingerprints made on "
                          "different devices are neither gathered alike nor computed alike"),
        (1, str, "the replicas compare different numbers of parameters ({}), and a gather of tables of different "
                 "shapes cannot be issued"),
        (2, lambda w: "with the projection" if w == FINGERPRINT_WIDTH + 1 else "without the projection",
         "the replicas' replica_projection settings differ ({}), and a gather of tables of different shapes "
         "cannot be issued"),
    ):
        seen: dict[int, list[int]] = {}
        for rank, row in zip(ranks, rows):
            seen.setdefault(row[field], []).append(rank)
        if len(seen) > 1:
            return what.format("; ".join(f"{value(v)} on {_ranks_text(members)}" for v, members in seen.items()))
    return None


def _exchange_device(group: object) -> torch.device | None:
    """Where the exchange's rows go: the first of `_EXCHANGE_DEVICE_TYPES` the group has a backend for
    (the CPU under Gloo, the current CUDA device under NCCL). Read from the group's backends, which
    every rank of the group has alike, never from the session's device, which can differ. None for
    a group that is not a `ProcessGroup` (a stand-in with its own `all_gather`) or whose backends
    cannot be read for any of those types: every rank then decides alone, as before. Never raises."""
    try:
        if not isinstance(group, dist.ProcessGroup) or not hasattr(group, "_get_backend"):
            return None
        for kind in _EXCHANGE_DEVICE_TYPES:
            try:
                group._get_backend(torch.device(kind))
            except Exception:  # noqa: BLE001  "No backend type associated with device type ..."
                continue
            if kind == "cuda":
                return torch.device("cuda", torch.cuda.current_device())
            return torch.device(kind)
    except Exception:  # noqa: BLE001
        return None
    return None


def _group_order(group: object, size: int) -> list[int]:
    """The global ranks of `group` in group-rank order (what the exchange's `all_gather` lists), or the
    group ranks themselves where they cannot be read."""
    try:
        return [int(dist.get_global_rank(group, i)) for i in range(size)]
    except Exception:  # noqa: BLE001
        return list(range(size))


def _exchange(
    group: object, where: torch.device, count: int, width: int, device: torch.device, problem: str | None
) -> str | None:
    """The exchange before an explicit gather: one `all_gather` of each rank's `_exchange_row` on
    `where`, then `_decide_gather` on what every rank sent. A collective like the gather after it:
    every rank of the group must reach it, and one that fails raises here. It reads the rows on the
    host, so on a GPU it waits for the device once."""
    row = _exchange_row(count, width, device, problem).to(where)
    out = [torch.empty_like(row) for _ in range(dist.get_world_size(group))]
    dist.all_gather(out, row, group=group)
    rows = torch.stack(out).cpu()
    return _decide_gather(list(rows), _group_order(group, len(out)))


@torch.no_grad()
def _spread_and_unread(
    params: Iterable[torch.Tensor], group: dist.ProcessGroup, *, projection: bool, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor, list[str | None]]:
    """The explicit cross-GPU check: the exchange (`_exchange`), then, when every rank of the group
    can gather alike, one `all_gather` of `_fingerprint_table`, which every rank enters with the same
    shape whatever it could read. Returns the `[n_params, width]` spread (`spread_from_fingerprints`),
    for each parameter 0.0 where every rank read it or else the largest code any rank gave (the same
    on every rank, since all see every rank's codes, so all leave out the same rows), and this rank's
    own reasons. The spread of a row not read on some rank means nothing: it is left out by its code,
    never by its value, so a NaN still reads as drift and a zero as agreement only where the weights
    were read. Both stay on the device.

    Where the gather cannot run, no rank of the group issues it: every row comes back with the code
    `_NOT_GATHERED`, zeros for its spread and the reason, so the session leaves every parameter out of
    `replica_spread` and names each once. A rank outside the group, or a destroyed process group,
    is judged alike on every rank with no communication (`_alike_problem`). Anything else that could
    differ between ranks is decided by the exchange first, a fixed-size `all_gather` (as PyTorch's
    DDP exchanges parameter counts before it compares shapes): this rank's own verdict on its table's
    device (`_gather_problem`), its table's device type, how many parameters it compares and its
    table's width. Every rank then gathers, or every rank skips with the same why. A group that is
    not a `ProcessGroup` (a stand-in with its own `all_gather`) has no exchange and decides alone, as
    before. A collective that raises once issued, the exchange or the gather, is not caught: a rank
    that caught it and went on would issue its next collective (the training run's own) while its
    peers still wait in this one.

    Only the two explicit calls reach it (`replica_spread`, `Session.record_replica_spread`),
    collectives every rank of the group must call: the session's scheduled check gathers nothing
    (`_observation_table`)."""
    params = list(params)
    if device is None:
        locals_ = [unwrap_local(p) for p in params]
        device = next(
            (torch.device("cpu") if p.device.type == "mps" else p.device for p in locals_ if not _unread_code(p)),
            torch.device("cpu"),
        )
    width = FINGERPRINT_WIDTH if projection else 2
    why = _alike_problem(group)
    if why is None:
        own = _gather_problem(group, device)
        where = _exchange_device(group)
        why = own if where is None else _exchange(group, where, len(params), width + 1, device, own)
    if why is not None:
        n = len(params)
        spread = torch.zeros(n, width, dtype=F64, device=device)
        unread = torch.full((n,), float(_NOT_GATHERED), dtype=F64, device=device)
        return spread, unread, [f"the drift check's gather cannot run: {why}"] * n
    table, reasons = _fingerprint_table(params, projection=projection, device=device)
    out = [torch.empty_like(table) for _ in range(dist.get_world_size(group))]
    dist.all_gather(out, table, group=group)  # 32 bytes per tensor per rank (24 without the projection)
    gathered = torch.stack(out)
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

    A collective: one `all_gather` on `group`, which every rank of the group must call, on the
    same step, and which waits for them as any collective does. A rank that never calls leaves
    the others waiting until the process group times out. The session's scheduled check does not
    call it: each rank records its own observation instead (`Session`, the `replica` key).

    `params`: tensors meant to be identical on every rank in `group`.
    `projection=False` skips the seeded projection (two columns instead of three), which
    costs about seven times the norm and sum on CPU; its GPU cost is not yet measured.
    `group`: the replication group. Pass `dist.group.WORLD` only for plain DDP;
    under tensor, pipeline or hybrid-FSDP parallelism WORLD compares the wrong ranks.

    A parameter that some replica cannot read (on `meta`, nested or sparse, or one whose reading
    raised) is still gathered, so every rank enters the same `all_gather` with the same shape and
    none is left waiting; its row is NaN on every rank, and one warning per call names it with
    why. Before the gather the ranks exchange one fixed-size row each (`_exchange`: how many
    parameters each passes, its tensors' device type, whether it can gather them, and the
    projection setting). Where any rank cannot gather, or the ranks pass different numbers of
    parameters or tensors on different device types, no rank issues the gather: every row is NaN
    on every rank, with the same reason, which names the ranks. A rank outside `group`, or no
    process group, gathers nothing either, with no exchange. A rank that passes no parameter takes
    part in the exchange and then raises `ValueError`; its peers skip. A NaN weight that was read gives NaN too, with no warning. NaN compares
    false with any tolerance, so `spread > tol` reads such a row as no drift: test
    `spread.isnan()` first, and treat a NaN row as not compared (or, for a NaN weight, drift).
    An exchange or gather that fails once issued still raises here. It gathers 32 bytes per
    parameter per rank (24 without the projection), after an exchange of 246 int64 per rank
    (1,968 bytes) in one `all_gather`, read on the host, so on a GPU it waits for the device once.
    Every rank must run a TensorQuill that gathers alike: a rank on an older one enters the table's
    gather while the others enter the exchange.
    """
    if group is None:
        raise ValueError("pass the replication process group explicitly")
    params = list(params)
    if not params:
        if isinstance(group, dist.ProcessGroup) and _alike_problem(group) is None:
            # Take part in the exchange, which says this rank compares nothing, so a peer that does
            # compare parameters skips the gather instead of waiting in it alone; then raise.
            _spread_and_unread([], group, projection=projection)
        raise ValueError("no replicated parameters to check")
    spread, unread, reasons = _spread_and_unread(params, group, projection=projection)
    codes = unread.tolist()  # one transfer: which rows to name
    if not any(codes):
        return spread
    spread = torch.where(unread[:, None] != 0, torch.full_like(spread, float("nan")), spread)
    named = [
        f"parameter {i} of {len(params)}: {reasons[i] or f'on another replica in the group, {_unread_reason(code)}'}"
        for i, code in enumerate(codes)
        if code
    ]
    shown = "; ".join(named[:8]) + (f"; and {len(named) - 8} more" if len(named) > 8 else "")
    _say(
        f"tensorquill.distributed.replica_spread did not compare {len(named)} of {len(params)} parameters across "
        f"replicas, which are NaN in its result: {shown}.",
        stacklevel=3,
    )
    return spread


# ---- the replica observation (replica observation contract, section 1) ---------------------------
# Each rank's own record of its replicated parameters, with no collective and no store: the
# fingerprints (`name-v1` projection) and an exact signature (`fold64-v1`), compared by a reader
# that has every rank's record. Nothing here talks to another rank.

PROJECTION_NAME = "name-v1"
SIGNATURE_NAME = "fold64-v1"
_M64 = (1 << 64) - 1


def name_v1_seed(name: str) -> int:
    """`name-v1`'s seed of a parameter's projection: the first four bytes of the SHA-256 of its
    name in UTF-8 (not normalised), read as a big-endian unsigned 32-bit number. Python's `hash()`
    is never used: it differs from one process to the next."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "big")


def _signed(value: int) -> int:
    """An unsigned 64-bit number as the signed int64 with the same bits."""
    value &= _M64
    return value - (1 << 64) if value >> 63 else value


# `fold64-v1`'s constants: the golden-ratio increment and the two multipliers of SplitMix64's
# finaliser (Steele, Lea and Flood, 2014), as signed int64 so torch can hold them.
_FOLD_STEP = 0x9E3779B97F4A7C15
_FOLD_M1 = 0xBF58476D1CE4E5B9
_FOLD_M2 = 0x94D049BB133111EB
_FOLD_STEP_S, _FOLD_M1_S, _FOLD_M2_S = _signed(_FOLD_STEP), _signed(_FOLD_M1), _signed(_FOLD_M2)
_UNSIGNED_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _shift_right(z: torch.Tensor, bits: int) -> torch.Tensor:
    """A logical right shift of int64 `z`: torch's `>>` on int64 copies the sign bit in."""
    return (z >> bits) & ((1 << (64 - bits)) - 1)


def _mix_tensor(z: torch.Tensor) -> torch.Tensor:
    """SplitMix64's finaliser on int64 `z`, every step modulo 2^64 (int64 products wrap)."""
    z = (z ^ _shift_right(z, 30)) * _FOLD_M1_S
    z = (z ^ _shift_right(z, 27)) * _FOLD_M2_S
    return z ^ _shift_right(z, 31)


def _mix(z: int) -> int:
    """`_mix_tensor` on one Python number, unsigned 64-bit in and out."""
    z &= _M64
    z = ((z ^ (z >> 30)) * _FOLD_M1) & _M64
    z = ((z ^ (z >> 27)) * _FOLD_M2) & _M64
    return z ^ (z >> 31)


def _element_bits(chunk: torch.Tensor) -> torch.Tensor:
    """Each element's bit pattern as an unsigned number in int64: the element's bytes read as
    the unsigned integer of its width (IEEE 754 bits for a float, two's complement for an
    integer, 0 or 1 for a bool)."""
    if chunk.dtype == torch.bool:
        return chunk.to(torch.int64)
    width = chunk.element_size()
    bits = chunk.view(_UNSIGNED_VIEW[width]).to(torch.int64)
    return bits & ((1 << (8 * width)) - 1) if width in (2, 4) else bits


@torch.no_grad()
def fold64_sum(p: torch.Tensor) -> torch.Tensor:
    """The device part of `fold64-v1`: `S`, the sum modulo 2^64 of every element's mixed word,
    as a 0-dim int64 (the signed number with `S`'s bits) on the parameter's device (the CPU for
    MPS). A sum, so the order a device reduces in cannot change it, and each word mixes in its
    element's index, so moving a value to another place does. `fold64_finish` adds the dtype and
    the shape on the host. Reads `p` as `_projection_f64` does: row-major, a complex tensor as
    its real pairs. Raises for a tensor it cannot read (meta, sparse, nested)."""
    flat = _as_real(p).reshape(-1)
    device = torch.device("cpu") if p.device.type == "mps" else p.device
    total = torch.zeros((), dtype=torch.int64, device=device)
    for i, chunk in enumerate(flat.split(PROJECTION_CHUNK)):
        chunk = chunk.cpu() if chunk.device.type == "mps" else chunk
        index = torch.arange(i * PROJECTION_CHUNK, i * PROJECTION_CHUNK + chunk.numel(), dtype=torch.int64,
                             device=chunk.device)
        total = total + _mix_tensor(_element_bits(chunk) ^ (index * _FOLD_STEP_S)).sum()
    return total


def _dtype_word(dtype_name: str) -> int:
    return int.from_bytes(hashlib.sha256(dtype_name.encode("utf-8")).digest()[:8], "big")


def fold64_finish(total: int, dtype_name: str, shape: Sequence[int]) -> int:
    """`fold64-v1` from `fold64_sum`'s `S` (signed or unsigned), the dtype's name as the record
    writes it (`"float32"`) and the shape: `S`, then the dtype's word, the number of dimensions and
    each size mixed in by turn (`h = mix(h ^ v)`). The signature as a signed 64-bit number."""
    h = int(total) & _M64
    for value in (_dtype_word(dtype_name), len(shape), *(int(n) for n in shape)):
        h = _mix(h ^ (value & _M64))
    return _signed(h)


def dtype_name(dtype: torch.dtype) -> str:
    """A dtype as the record writes it: `"float32"`, `"bfloat16"`."""
    return str(dtype).removeprefix("torch.")


def group_ranks(group: object) -> tuple[list[int] | None, str | None]:
    """The global ranks of `group`, sorted, read on the host with no communication, and None; or
    None and why they cannot be read (no process group, a stand-in group, a rank outside it).
    Never raises."""
    try:
        if not (dist.is_available() and dist.is_initialized()):
            return None, "torch.distributed has no default process group"
        if dist.distributed_c10d._rank_not_in_group(group):
            return None, "this rank is not a member of the replica group"
        ranks = sorted(int(r) for r in dist.get_process_group_ranks(group))
        if dist.get_rank() not in ranks:
            return None, "this rank is not a member of the replica group"
        return ranks, None
    except Exception as error:  # noqa: BLE001  a stand-in group, or an API this PyTorch lacks
        text = " ".join(f"{type(error).__name__}: {error}".split())[:160]
        return None, f"its global ranks could not be read ({text})"


@torch.no_grad()
def _observation_table(
    params: Sequence[torch.Tensor], seeds: Sequence[int], *, projection: bool, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[str | None], list[tuple[list[int], str]]]:
    """This rank's observation of `params`, all on `device` with no transfer to the host and no
    collective: the `[n, width + 1]` fingerprint table of `_fingerprint_table` (seeded by
    `seeds`, the `name-v1` seeds), the `[n]` int64 `fold64_sum` of each, each parameter's reason
    when it was not read (or None), and each one's (shape, dtype name) as read, for the record and
    for `fold64_finish`. A parameter whose signature cannot be computed is not read either: its
    row gets the code of a raised read, so it is in the record's `unread`, never in `params`
    without a signature. Never raises for a parameter."""
    local = [unwrap_local(p) for p in params]
    table, reasons = _fingerprint_table(local, projection=projection, device=device, seeds=seeds)
    width = table.shape[1] - 1
    metas = [(list(p.shape), dtype_name(p.dtype)) for p in local]
    sums = torch.zeros(len(local), dtype=torch.int64, device=device)
    read = [i for i, why in enumerate(reasons) if why is None]
    try:
        if read:
            sums[read] = torch.stack([fold64_sum(local[i]).to(device) for i in read])
    except Exception:  # noqa: BLE001  one at a time
        for i in read:
            try:
                sums[i] = fold64_sum(local[i]).to(device)
            except Exception as error:  # noqa: BLE001
                text = " ".join(f"{type(error).__name__}: {error}".split())
                reasons[i] = f"reading it raised {text if len(text) <= 200 else text[:199] + '…'}"
                table[i, :width].zero_()
                table[i, width].fill_(float(_RAISED))
    return table, sums, reasons, metas
