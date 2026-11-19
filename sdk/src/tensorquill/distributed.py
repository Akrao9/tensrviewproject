"""Cross-GPU replica drift detector: float64 norm and sum fingerprints per parameter."""

from __future__ import annotations

import itertools
import json
import time
import weakref
from typing import Any, Iterable

import torch
import torch.distributed as dist

from .tracker import _brief, _say

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


#: At most how long `_agree_on_gather` waits at `init` for every member of the replica group to
#: say whether it can take part in the drift check's gather, in seconds, and never longer than
#: the default store's own timeout. After this the check is off on every rank, with the reason.
AGREE_WAIT_SECONDS = 300.0

# One set of keys in the default store for each agreement: the group's global ranks, then the
# n-th agreement for those ranks in this process. Every member makes the same agreements in the
# same order (one per session with that replica group, as for a collective), so the n-th on every
# member reads the same keys, and a second session never reads the first one's verdicts.
_AGREE_KEY = "tensorquill/replica_check/{}/{}"
_agreements: dict[tuple[int, ...], Any] = {}
# What the newest agreement on a group decided: (the table's device type, None for "gather", or
# the reason every member skips). Weak, so a destroyed group's entry goes with it.
_agreed: weakref.WeakKeyDictionary[Any, tuple[str, str | None]] = weakref.WeakKeyDictionary()


def _agree_on_gather(group: object, device: torch.device, params: Iterable[torch.Tensor] = ()) -> str | None:
    """At `init`, before the first scheduled check: make every member of `group` take the same
    decision on whether the drift check's gather runs. Returns None (gather) or why every member
    skips it, and keeps it for `_spread_and_unread` and `replica_spread` on `group`. Never raises.

    `_gather_problem` alone is judged by each rank from what it sees, which ranks that differ do
    not see alike (L5: a session on another device type; also a different number of replicated
    parameters, whose gather of unequal shapes aborts the process under Gloo). So each member
    puts its own verdict (`_gather_problem`, its table's device type, how many parameters it
    compares) in the default process group's store, waits a bounded time for every member's, and
    then puts "gather" (every member could, alike) or "skip" (any could not, they differ, or the
    wait ran out) under one key with `compare_set`, first writer wins. Every member takes what
    the store holds, so a member whose own wait ran out still follows one that saw every verdict.
    No collective is issued and nothing is put on a device.

    Only a real `ProcessGroup` of two or more members, on a rank in it, takes part: a stand-in
    group (tests and tools with their own `all_gather`) and a rank outside the group are judged
    as before, by `_gather_problem` at each check.

    What this does not cover: a group that fails after `init` (a rank that dies, a timeout, an
    aborted communicator) still fails in the gather inside the training step, as the training
    run's own collectives on it do; and a member whose store fails after it has put its verdict
    in cannot learn the decision, so it skips while the others may gather."""
    try:
        if dist.distributed_c10d._rank_not_in_group(group) or not isinstance(group, dist.ProcessGroup):
            return None
        if not dist.is_initialized():
            return None
        ranks = tuple(int(r) for r in dist.get_process_group_ranks(group))
        me = int(dist.get_rank(group))
        device = torch.device(device)
        count = len(list(params))
    except Exception:  # noqa: BLE001  an API this PyTorch does not have: each check decides alone
        return None
    if len(ranks) < 2 or not 0 <= me < len(ranks):
        return None
    try:
        verdict = {"problem": _gather_problem(group, device), "device": device.type, "count": count}
        base = _AGREE_KEY.format("-".join(map(str, ranks)), next(_agreements.setdefault(ranks, itertools.count())))
        try:
            store = dist.distributed_c10d._get_default_store()
        except Exception as error:  # noqa: BLE001  nothing put in: no member can decide "gather"
            why: str | None = f"rank {ranks[me]} could not reach the process group's store ({_brief(error)})"
        else:
            why = _exchange_verdicts(store, base, me, ranks, verdict, wait=_agree_wait(store))
        _agreed[group] = (device.type, why)
        return why
    except Exception:  # noqa: BLE001  nothing here may stop a run
        return None


def _agree_wait(store: Any) -> float:
    try:
        wait = max(float(AGREE_WAIT_SECONDS), 0.0)
    except (TypeError, ValueError):  # set to something that is not a number
        wait = 300.0
    try:
        own = store.timeout.total_seconds()
    except Exception:  # noqa: BLE001  a store with no timeout of its own
        own = 0.0
    return own if 0 < own < wait else wait


def _exchange_verdicts(
    store: Any, base: str, me: int, ranks: Iterable[int], verdict: dict, *, wait: float
) -> str | None:
    """`_agree_on_gather`'s exchange through `store`: this member's `verdict` under `base/<me>`
    (`me` its index in the group), every member's read once all are there, and the decision
    under `base/decision`, whichever member put it there first. Returns None ("gather") or the
    reason. Polls with `check` (10 ms, doubling to 1 s) as `tracker._shared_run` does, never a
    `get` on a missing key, which waits the store's whole timeout. Never raises."""
    ranks = list(ranks)
    keys = [f"{base}/{i}" for i in range(len(ranks))]
    try:
        store.set(keys[me], json.dumps(verdict))
    except Exception as error:  # noqa: BLE001  this verdict is missing: no member can decide "gather"
        return f"rank {ranks[me]} could not put its verdict in the process group's store ({_brief(error)})"
    try:
        deadline, pause = time.monotonic() + wait, 0.01
        while not store.check(keys):
            left = deadline - time.monotonic()
            if left <= 0:
                break
            time.sleep(min(pause, left))
            pause = min(pause * 2, 1.0)
        missing = [ranks[i] for i, key in enumerate(keys) if not store.check([key])]
        if missing:
            proposal = (
                f"rank {ranks[me]} did not hear from {_ranks_text(missing)} of the replica group within {wait:g} s "
                "at init (every rank must call tensorquill.init with the same replica_group)"
            )
        else:
            proposal = _decide([json.loads(store.get(key)) for key in keys], ranks)
        decided = json.loads(store.compare_set(f"{base}/decision", "", json.dumps({"why": proposal})))
        why = decided["why"]
        return None if why is None else str(why)
    except Exception as error:  # noqa: BLE001  the window this cannot close: see `_agree_on_gather`
        return (
            f"the process group's store failed on rank {ranks[me]} while the ranks agreed whether to run it "
            f"({_brief(error)})"
        )


def _ranks_text(ranks: list[int]) -> str:
    return f"rank {ranks[0]}" if len(ranks) == 1 else "ranks " + ", ".join(map(str, ranks))


def _decide(verdicts: list[Any], ranks: list[int]) -> str | None:
    """None when every member can gather, alike; else why none does. The same on every member,
    since each reads the same verdicts in the same order."""
    problems = []
    for rank, verdict in zip(ranks, verdicts):
        if not isinstance(verdict, dict):
            problems.append((rank, "its verdict could not be read"))
        elif verdict.get("problem"):
            problems.append((rank, str(verdict["problem"])))
    if problems:
        rank, problem = problems[0]
        more = f" (as do {len(problems) - 1} more)" if len(problems) > 1 else ""
        return f"rank {rank} of the replica group cannot take part{more}: {problem}"
    for field, what in (
        ("device", "the replicas' sessions are on different device types ({}), and fingerprints made on different "
                   "devices are neither gathered alike nor computed alike"),
        ("count", "the replicas compare different numbers of parameters ({}), and a gather of tables of different "
                  "shapes cannot be issued"),
    ):
        seen: dict[str, list[int]] = {}
        for rank, verdict in zip(ranks, verdicts):
            seen.setdefault(str(verdict.get(field)), []).append(rank)
        if len(seen) > 1:
            return what.format("; ".join(f"{value} on {_ranks_text(members)}" for value, members in seen.items()))
    return None


def _agreed_problem(group: object, device: torch.device) -> str | None:
    """Why the agreement at `init` (`_agree_on_gather`) turned the check off on `group` for a
    table on `device`'s type, or None: it decided to gather, or it was never made."""
    try:
        found = _agreed.get(group)
    except TypeError:  # a stand-in group that cannot be weakly referenced: never agreed
        return None
    if found is None or found[0] != device.type:
        return None
    return found[1]


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
    not be read. The same holds where the agreement at `init` (`_agree_on_gather`) turned the
    check off: every member of the group took that one decision, so where ranks would judge
    differently by themselves (sessions on different device types) none gathers while another
    skips. Without an agreement, each rank judges alone, which every rank of a data-parallel job
    does alike. A gather that raises once issued is not caught: a rank that caught it and went
    on would issue its next collective (the training run's own) while its peers still wait in
    this one. So a group that fails after `init` still fails here, inside the training step."""
    params = list(params)
    if device is None:
        locals_ = [unwrap_local(p) for p in params]
        device = next(
            (torch.device("cpu") if p.device.type == "mps" else p.device for p in locals_ if not _unread_code(p)),
            torch.device("cpu"),
        )
    why = _agreed_problem(group, device) or _gather_problem(group, device)
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

    A parameter that some replica cannot read (on `meta`, nested or sparse, or one whose reading
    raised) is still gathered, so every rank enters the same `all_gather` with the same shape and
    none is left waiting; its row is NaN on every rank, and one warning per call names it with
    why. Where the gather cannot run at all (as the session's check judges it: no backend for the
    tensors' device, no process group, a rank outside `group`, or a session's agreement at
    `init` that turned the check off on `group`), no collective is issued and every row is NaN,
    with the reason. A NaN weight that was read gives NaN too, with no warning. A gather that
    fails once issued still raises here. It gathers 32 bytes per parameter per rank (24 without
    the projection), the session's table, so every rank must run a TensorQuill that gathers alike.
    """
    if group is None:
        raise ValueError("pass the replication process group explicitly")
    params = list(params)
    if not params:
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
