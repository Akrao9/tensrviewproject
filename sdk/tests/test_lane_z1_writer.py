"""Lane Z1 of the SDK-22 wave: the writer of the replica observation.

What `REPLICA_OBSERVATION_CONTRACT.md` sections 1 and 2 ask of a rank, tested here:

- No collective and no store wait in a training step with `replica_group` set: a single process
  whose collectives and default store raise if touched, and two processes where rank 1 trains
  while rank 0 never steps.
- The observation (`replica`) on each scheduled step, with every key of section 1, `name-v1`
  projections and `fold64-v1` signatures, non-finite fingerprints as strings, and a parameter it
  cannot read under `unread`, never in `params`.
- `fold64-v1` as the page defines it: a pure-Python reading with the page's test vectors, and the
  SDK's device code equal to it for every dtype, chunk size and layout tried.
- The launch: rank 0's nonce set before the run ID's entry, one look on the other ranks with no
  wait, and the second read (on the writer thread, bounded) that replaces a stale entry.
- The scope of an explicit gather, and the stopped rank that no longer enters a gather on its own.

Fail before: every test here but the pure-Python `fold64-v1` vector fails on `633c9da` (the base
of this lane): the names it imports do not exist there, the gather raises out of the step, or
rank 1 waits in the gather until the group's timeout. The vector test reads only the page.

Two-process tests use Gloo on the CPU with a 15 s group timeout and a 90 s limit. No GPU here.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import re
import struct
import subprocess
import sys
import textwrap
import threading
import time
import warnings
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import capture as C
from tensorquill import distributed as D
from tensorquill import tracker as T

needs_dist = pytest.mark.skipif(not torch.distributed.is_available(), reason="needs torch.distributed")
HEX32 = re.compile(r"[0-9a-f]{32}")

# ---------------------------------------------------------------------------------------------
# fold64-v1, read from the page with the standard library alone.

_M64 = (1 << 64) - 1


def _mix(z: int) -> int:
    z &= _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def fold64_v1(words: list[int], dtype: str, shape: list[int]) -> tuple[int, int]:
    """(S, signature) of elements whose bit patterns are `words`, as the contract defines them."""
    total = 0
    for i, word in enumerate(words):
        total = (total + _mix(word ^ ((i * 0x9E3779B97F4A7C15) & _M64))) & _M64
    h = total
    d = int.from_bytes(hashlib.sha256(dtype.encode("utf-8")).digest()[:8], "big")
    for value in (d, len(shape), *shape):
        h = _mix(h ^ value)
    return total, (h - (1 << 64) if h >> 63 else h)


def _words(t: torch.Tensor) -> list[int]:
    """The page's words, read from the tensor's bytes in Python: row-major, a complex tensor as its
    real pairs, each element's bytes as a little-endian unsigned number."""
    t = t.detach().cpu()
    if t.is_complex():
        t = torch.view_as_real(t.resolve_conj())
    t = t.contiguous().reshape(-1)
    width = t.element_size()
    # Not `.numpy()`: TensorQuill does not require NumPy, and most CI jobs install none.
    raw = bytes(t.view(torch.uint8).tolist()) if t.numel() else b""
    return [int.from_bytes(raw[i : i + width], "little") for i in range(0, len(raw), width)]


def _sdk_signature(t: torch.Tensor) -> int:
    return D.fold64_finish(int(D.fold64_sum(t)), D.dtype_name(t.dtype), list(t.shape))


def test_fold64_v1_test_vectors_from_the_page():
    """The numbers on the contract page, from the definition alone (no SDK code)."""
    f32 = [struct.unpack("<I", struct.pack("<f", x))[0] for x in (1.0, 2.0, 3.0, 4.0)]
    assert f32 == [0x3F800000, 0x40000000, 0x40400000, 0x40800000]
    assert fold64_v1(f32, "float32", [2, 2]) == (7798465200413876260, 1799304254312201492)
    assert fold64_v1([0x3F80, 0x4000, 0x4040, 0x4080], "bfloat16", [2, 2])[1] == 5716884024385227157
    assert fold64_v1(f32, "float32", [4])[1] == 3672505199788284773
    assert fold64_v1([], "float32", [0, 3]) == (0, -8628698763751150559)
    assert int.from_bytes(hashlib.sha256(b"float32").digest()[:8], "big") == 7679619720248017433


def test_fold64_v1_the_sdk_gives_the_page_vectors():
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert int(D.fold64_sum(t)) & _M64 == 7798465200413876260
    assert _sdk_signature(t) == 1799304254312201492
    assert _sdk_signature(t.to(torch.bfloat16)) == 5716884024385227157
    assert _sdk_signature(t.reshape(4)) == 3672505199788284773
    assert _sdk_signature(torch.zeros(0, 3)) == -8628698763751150559


def _dtypes() -> list[torch.dtype]:
    found = [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int64, torch.int32, torch.int16,
             torch.int8, torch.uint8, torch.bool, torch.complex64, torch.complex128]
    for name in ("float8_e4m3fn", "float8_e5m2", "uint16", "uint32", "uint64"):
        dtype = getattr(torch, name, None)
        if dtype is not None:
            found.append(dtype)
    return found


def _random(shape: tuple[int, ...], dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=gen).bool()
    if dtype.is_complex:
        return torch.randn(shape, dtype=dtype, generator=gen)
    if dtype.is_floating_point:
        return torch.randn(shape, generator=gen).mul(100).to(dtype)
    # every bit pattern of the integer's width, negative ones included
    width = torch.empty((), dtype=dtype).element_size()
    raw = torch.randint(0, 256, (*shape, width), dtype=torch.uint8, generator=gen)
    return raw.reshape(-1).view(dtype).reshape(shape) if raw.numel() else torch.empty(shape, dtype=dtype)


@pytest.mark.parametrize("dtype", _dtypes(), ids=str)
def test_fold64_v1_the_sdk_equals_the_definition_for_every_dtype(dtype):
    gen = torch.Generator().manual_seed(7)
    for shape in [(), (1,), (5,), (3, 4), (2, 0, 3), (2, 3, 5)]:
        t = _random(shape, dtype, gen)
        assert _sdk_signature(t) == fold64_v1(_words(t), D.dtype_name(dtype), list(t.shape))[1], shape


def test_fold64_v1_does_not_depend_on_the_chunk_the_layout_or_a_conjugate_view(monkeypatch):
    gen = torch.Generator().manual_seed(1)
    t = torch.randn(7, 9, generator=gen)
    whole = _sdk_signature(t)
    monkeypatch.setattr(D, "PROJECTION_CHUNK", 5)  # many chunks, the last a short one
    assert _sdk_signature(t) == whole
    # The same elements in row-major order through another layout: a transposed copy's view.
    stored = t.t().contiguous()
    assert not stored.t().is_contiguous() and _sdk_signature(stored.t()) == whole
    z = torch.randn(4, 3, dtype=torch.complex64, generator=gen)
    assert _sdk_signature(z.conj()) == _sdk_signature(z.conj().resolve_conj())


def test_fold64_v1_different_signatures_follow_from_different_bits_shape_or_dtype():
    gen = torch.Generator().manual_seed(3)
    t = torch.randn(4, 6, generator=gen)
    base = _sdk_signature(t)
    flipped = t.clone()
    flipped.view(torch.int32)[2, 3] ^= 1  # one bit of one element
    swapped = t.clone()
    swapped[0, 0], swapped[3, 5] = t[3, 5].clone(), t[0, 0].clone()  # same norm, sum and bytes as a bag
    assert _sdk_signature(flipped) != base
    assert _sdk_signature(swapped) != base
    assert _sdk_signature(t.reshape(6, 4)) != base  # same bytes, another shape
    assert _sdk_signature(t.view(torch.int32)) != base  # same bytes, another dtype
    assert _sdk_signature(torch.tensor([0.0])) != _sdk_signature(torch.tensor([-0.0]))  # bits, not values


def test_fold64_v1_mixing_on_the_device_wraps_as_the_definition_does():
    """`>>` on int64 copies the sign bit in and the products pass 64 bits: the device code must
    still give the unsigned arithmetic of the page, also far into a tensor (index 5e9)."""
    rng = random.Random(5)
    values = [0, 1, _M64, 1 << 63, (1 << 63) - 1] + [rng.getrandbits(64) for _ in range(200)]
    got = D._mix_tensor(torch.tensor([D._signed(v) for v in values], dtype=torch.int64)).tolist()
    assert [g & _M64 for g in got] == [_mix(v) for v in values]
    index = torch.tensor([5_000_000_000, 2**40 + 3], dtype=torch.int64)
    assert [v & _M64 for v in (index * D._FOLD_STEP_S).tolist()] == [
        (i * 0x9E3779B97F4A7C15) & _M64 for i in (5_000_000_000, 2**40 + 3)
    ]


# ---------------------------------------------------------------------------------------------
# A process group of one, for the session-level tests.


@pytest.fixture
def world1(tmp_path):
    dist = torch.distributed
    if not dist.is_available():
        pytest.skip("needs torch.distributed")
    if dist.is_initialized():
        pytest.skip("a process group is already initialized in this process")
    dist.init_process_group("gloo", store=dist.FileStore(str(tmp_path / "store1"), 1), rank=0, world_size=1,
                            timeout=timedelta(seconds=15))
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


def _train(session, model, steps: int, between=None) -> None:
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for i in range(steps):
        opt.zero_grad()
        model(torch.ones(2, 4)).sum().backward()
        opt.step()
        if between is not None:
            between(i)


def _init(model, group, **kwargs):
    sink = tensorquill.MemorySink()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = tensorquill.init(model, run_id="z1", sink=sink, replica_group=group, heartbeat_seconds=None,
                                   stack_seconds=None, **kwargs)
    return session, sink


class Untouchable:
    """A process group that fails the test if anything is asked of it."""

    def __getattr__(self, name):
        raise AssertionError(f"the training step touched the replica group ({name})")


_COLLECTIVES = ("all_gather", "all_gather_into_tensor", "all_gather_object", "all_reduce", "broadcast", "barrier",
                "reduce", "reduce_scatter", "reduce_scatter_tensor", "gather", "scatter", "all_to_all",
                "all_to_all_single", "monitored_barrier", "broadcast_object_list", "send", "recv", "isend", "irecv")


def test_no_collective_and_no_store_in_a_training_step(world1, monkeypatch):
    """The SDK-22 decision: a session given a replica group issues nothing on its own schedule.
    Before: the scheduled check's `all_gather` raised out of `optimizer.step()`."""
    dist = torch.distributed
    model = _model()
    session, sink = _init(model, world1, every=4, replica_every=2)

    def forbidden(*args, **kwargs):
        raise AssertionError("a collective was issued in a training step")

    def no_store(*args, **kwargs):
        raise AssertionError("the default store was looked up in a training step")

    for name in _COLLECTIVES:
        if hasattr(dist, name):
            monkeypatch.setattr(dist, name, forbidden)
    monkeypatch.setattr(dist.distributed_c10d, "_get_default_store", no_store)
    session._replica_group = Untouchable()
    _train(session, model, 9)
    session.close()
    assert session.writer_error is None
    observed = [r["step"] for r in sink.records if "replica" in r]
    assert observed == [1, 3, 5, 7]
    assert all("replica_spread" not in r for r in sink.records)


# ---------------------------------------------------------------------------------------------
# The observation's content.


def test_the_observation_has_every_key_of_contract_section_1(world1):
    model = _model()
    session, sink = _init(model, world1, every=4, replica_every=3)
    _train(session, model, 6)
    session.close()
    rows = {r["step"]: r for r in sink.records}
    assert [s for s, r in rows.items() if "replica" in r] == [2, 5]
    for step in (2, 5):
        obs = rows[step]["replica"]
        assert set(obs) == {"v", "launch", "group", "device", "projection", "signature", "params", "unread"}
        assert obs["v"] == 1 and HEX32.fullmatch(obs["launch"]) and obs["launch"] == session.launch
        assert obs["group"] == [0] and obs["device"] == "cpu"
        assert obs["projection"] == "name-v1" and obs["signature"] == "fold64-v1"
        assert obs["unread"] == {}
        assert list(obs["params"]) == ["0.weight", "0.bias", "2.weight", "2.bias"]
    # The values are the parameters' as the step left them: here, after the last step (step 5).
    params = dict(model.named_parameters())
    for name, entry in rows[5]["replica"]["params"].items():
        p = params[name].detach()
        assert set(entry) == {"shape", "dtype", "fp", "sig"}
        assert entry["shape"] == list(p.shape) and entry["dtype"] == "float32"
        seed = D.name_v1_seed(name)
        norm, total = float(torch.linalg.vector_norm(p, dtype=torch.float64)), float(p.sum(dtype=torch.float64))
        assert entry["fp"][0] == pytest.approx(norm, rel=1e-12) and entry["fp"][1] == pytest.approx(total, rel=1e-12)
        assert entry["fp"][2] == pytest.approx(float(D._projection_f64(p, seed)), rel=1e-12)
        assert isinstance(entry["sig"], str) and int(entry["sig"]) == fold64_v1(_words(p), "float32", list(p.shape))[1]


def test_the_projection_is_seeded_by_the_name_not_the_place():
    """`name-v1`: the same tensor under the same name gives the same projection whatever its
    place in the list; the explicit gather keeps its seed by place."""
    p, q = torch.randn(5), torch.randn(5)
    seeds = [D.name_v1_seed("a"), D.name_v1_seed("b")]
    one, *_ = D._observation_table([p, q], seeds, projection=True, device=torch.device("cpu"))
    two, *_ = D._observation_table([q, p], seeds[::-1], projection=True, device=torch.device("cpu"))
    assert torch.equal(one[0], two[1]) and torch.equal(one[1], two[0])
    by_place, _ = D._fingerprint_table([p, q], projection=True)
    assert by_place[1, 2] == D._projection_f64(q, 1)


def test_projection_off_writes_null_and_two_fingerprints(world1):
    model = _model()
    session, sink = _init(model, world1, every=2, replica_every=2, replica_projection=False)
    _train(session, model, 2)
    session.close()
    [obs] = [r["replica"] for r in sink.records if "replica" in r]
    assert obs["projection"] is None and all(len(e["fp"]) == 2 for e in obs["params"].values())


def test_non_finite_fingerprints_are_strings_and_the_record_is_standard_json(world1, tmp_path):
    model = _model()
    path = tmp_path / "rank0.jsonl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = tensorquill.init(model, run_id="z1", sink=tensorquill.JsonlSink(path), every=2,
                                   replica_group=world1, replica_every=2, heartbeat_seconds=None, stack_seconds=None)
    with torch.no_grad():
        model[0].bias[0] = float("nan")
        model[2].bias[0] = float("inf")
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    for _ in range(2):
        opt.zero_grad()
        opt.step()
    session.close()
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    [obs] = [r["replica"] for r in lines if "replica" in r]
    assert obs["params"]["0.bias"]["fp"][0] == "nan" and obs["params"]["2.bias"]["fp"][0] == "inf"
    assert all(isinstance(v, float) for v in obs["params"]["0.weight"]["fp"])
    # A NaN weight is read (its bits are signed), as the fingerprints read it.
    assert int(obs["params"]["0.bias"]["sig"]) == fold64_v1(_words(model[0].bias.detach()), "float32", [3])[1]


def test_a_parameter_that_cannot_be_read_is_unread_and_never_in_params(world1, monkeypatch):
    """One whose signature raises is not in `params` without a `sig`: it is in `unread` with why."""
    model = _model()
    session, sink = _init(model, world1, every=2, replica_every=2)
    real = D.fold64_sum
    bad = model[2].weight

    def fold(p):
        if p is bad:
            raise RuntimeError("cannot read it here")
        return real(p)

    monkeypatch.setattr(D, "fold64_sum", fold)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _train(session, model, 4)
        session.close()
    observations = [r["replica"] for r in sink.records if "replica" in r]
    assert len(observations) == 2
    for obs in observations:
        assert "2.weight" not in obs["params"] and "cannot read it here" in obs["unread"]["2.weight"]
        assert set(obs["params"]) == {"0.weight", "0.bias", "2.bias"}
    said = [str(w.message) for w in caught if "not observing the parameter '2.weight'" in str(w.message)]
    assert len(said) == 1  # once per parameter, not once per step


def test_observation_table_marks_meta_and_sparse_unread():
    params = [torch.ones(3), torch.empty(3, device="meta"), torch.eye(3).to_sparse()]
    table, sums, reasons, metas = D._observation_table(params, [1, 2, 3], projection=True,
                                                       device=torch.device("cpu"))
    assert reasons[0] is None and "meta" in reasons[1] and "sparse" in reasons[2]
    assert table[1:, :3].eq(0).all() and sums[1:].eq(0).all()
    assert metas == [([3], "float32"), ([3], "float32"), ([3, 3], "float32")]


@pytest.mark.parametrize("every,replica_every", [(1, 1), (2, 2), (4, 2), (3, 5)])
def test_every_scheduled_step_has_one_observation_whatever_the_ring(world1, every, replica_every):
    """A scheduled step that is the newest of a full ring is held back until the next step
    (`_hold_newest`), and `every=1` holds every step: the observation must survive both."""
    model = _model()
    session, sink = _init(model, world1, every=every, replica_every=replica_every)
    _train(session, model, 11)
    session.close()
    got = [r["step"] for r in sink.records if "replica" in r]
    assert got == [s for s in range(11) if (s + 1) % replica_every == 0]
    # each observation is that step's own: the signatures change from one to the next
    sigs = [r["replica"]["params"]["2.bias"]["sig"] for r in sink.records if "replica" in r]
    assert len(set(sigs)) == len(sigs)


def test_an_observation_of_a_few_hundred_parameters_is_tens_of_kilobytes(world1):
    """Contract 8 asks for the size. 300 parameters with names like a transformer's."""
    layers = nn.ModuleDict({f"layer{i}": nn.ModuleDict({k: nn.Linear(8, 8) for k in ("q", "k", "v", "o", "up")})
                            for i in range(30)})
    model = nn.Module()
    model.model = layers
    session, sink = _init(model, world1, every=1, replica_every=1, auto_step=False)
    session.step()
    session.close()
    [obs] = [r["replica"] for r in sink.records if "replica" in r]
    assert len(obs["params"]) == 300
    size = len(json.dumps(obs, separators=(",", ":")))
    assert 20_000 < size < 80_000, size


# ---------------------------------------------------------------------------------------------
# The explicit gather's scope (contract 1).


def test_an_explicit_gather_says_what_it_covers(world1):
    model = _model()
    session, sink = _init(model, world1, every=4, replica_every=50)
    _train(session, model, 3, between=lambda i: session.record_replica_spread() if i == 0 else None)
    session.close()
    [record] = [r for r in sink.records if "replica_spread" in r]
    assert record["replica_spread_scope"] == {"launch": session.launch, "group": [0]}
    assert "replica" not in record  # not a scheduled step: the scope is never taken from an observation


def test_a_precomputed_table_has_no_scope(world1):
    model = _model()
    session, sink = _init(model, world1, every=4, replica_every=50)
    table = torch.zeros(4, 3, dtype=torch.float64)
    _train(session, model, 3, between=lambda i: session.record_replica_spread(table) if i == 0 else None)
    session.close()
    [record] = [r for r in sink.records if "replica_spread" in r]
    assert "replica_spread_scope" not in record


def test_a_rank_with_no_launch_writes_no_scope_and_a_null_launch(world1):
    model = _model()
    session, sink = _init(model, world1, every=4, replica_every=2)
    session.run = dataclasses.replace(session.run, launch=None)
    session.launch = None
    _train(session, model, 3, between=lambda i: session.record_replica_spread() if i == 0 else None)
    session.close()
    [record] = [r for r in sink.records if "replica_spread" in r]
    assert "replica_spread_scope" not in record
    assert [r["replica"]["launch"] for r in sink.records if "replica" in r] == [None]


def test_a_group_whose_ranks_cannot_be_read_gets_no_observation_and_no_scope(world1, monkeypatch):
    dist = torch.distributed
    stand_in = object()

    def all_gather(out, table, group=None):
        for o in out:
            o.copy_(table)

    model = _model()
    sink = tensorquill.MemorySink()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, run_id="z1", sink=sink, replica_group=stand_in, every=4, replica_every=2,
                                   heartbeat_seconds=None, stack_seconds=None)
    assert any("writes no replica observation" in str(w.message) for w in caught)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(dist, "all_gather", all_gather)
    _train(session, model, 3, between=lambda i: session.record_replica_spread() if i == 0 else None)
    session.close()
    assert not any("replica" in r for r in sink.records)
    [record] = [r for r in sink.records if "replica_spread" in r]
    assert "replica_spread_scope" not in record


def test_an_explicit_gather_on_a_scheduled_step_leaves_the_observation(world1):
    model = _model()
    session, sink = _init(model, world1, every=4, replica_every=2)
    _train(session, model, 2, between=lambda i: session.record_replica_spread() if i == 0 else None)
    session.close()
    record = next(r for r in sink.records if r["step"] == 1)
    assert "replica" in record and "replica_spread" in record and "replica_spread_scope" in record


# ---------------------------------------------------------------------------------------------
# A stopped rank.


def test_a_stopped_session_gathers_nothing_on_its_own_and_still_takes_an_explicit_call(world1, monkeypatch):
    """Before: a stopped rank kept counting steps and entered the gather on each scheduled one
    (`_gather_and_drop`), so a peer never waited for it; with no scheduled gather it has nothing
    to keep entering. The explicit call is a collective the user issues on every rank: a stopped
    rank still enters it."""
    model = _model()
    session, _ = _init(model, world1, every=4, replica_every=2)
    calls = []
    real = C._spread_and_unread

    def gather(*args, **kwargs):
        calls.append(time.monotonic())
        return real(*args, **kwargs)

    monkeypatch.setattr(C, "_spread_and_unread", gather)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session._stop(RuntimeError("the sink failed"))
        _train(session, model, 6)
        assert calls == []
        session.record_replica_spread()
        assert len(calls) == 1
        session.close()


# ---------------------------------------------------------------------------------------------
# The launch: the exchange in `tracker.py`, against stores that record what is asked of them.


class Store:
    """A store standing in for the process group's: records each call and its thread. `get` on a
    missing key fails the test, as it would wait the store's whole timeout."""

    timeout = timedelta(seconds=5)

    def __init__(self, entries=None, fail=False, hang=0.0):
        self.entries = {k: (v.encode() if isinstance(v, str) else v) for k, v in (entries or {}).items()}
        self.calls: list[tuple[str, str, str]] = []
        self.fail, self.hang = fail, hang

    def _call(self, what, key):
        self.calls.append((what, key, threading.current_thread().name))
        if self.hang:
            time.sleep(self.hang)
        if self.fail:
            raise RuntimeError("store down")

    def set(self, key, value):
        self._call("set", key)
        self.entries[key] = value.encode() if isinstance(value, str) else value

    def check(self, keys):
        self._call("check", ",".join(keys))
        return all(k in self.entries for k in keys)

    def get(self, key):
        self._call("get", key)
        if key not in self.entries:
            raise AssertionError(f"get on a missing key {key!r}: it would wait the store's whole timeout")
        return self.entries[key]


def _dist(store, world=2):
    def default_store():
        if isinstance(store, Exception):
            raise store
        return store

    return SimpleNamespace(get_world_size=lambda: world, distributed_c10d=SimpleNamespace(_get_default_store=default_store))


RUN_KEY, LAUNCH_KEY = "tensorquill/run_id/7", "tensorquill/launch/7"


@pytest.fixture
def exchange_7(monkeypatch):
    monkeypatch.setattr(T, "_exchanges", itertools.count(7))


def test_rank_0_sets_a_nonce_before_the_run_ids_entry_which_keeps_its_two_items(exchange_7):
    store = Store()
    info = T._shared_run(_dist(store), 0, "my-run")
    # Rewritten by lane AA1 (L25): rank 0 first sets the mark that says its entries are also scoped
    # by generation where one is trusted (none here), then the launch before the run ID, as before.
    assert [(what, key) for what, key, _ in store.calls] == [
        ("set", "tensorquill/scoped/7"), ("set", LAUNCH_KEY), ("set", RUN_KEY)
    ]
    assert HEX32.fullmatch(info.launch) and store.entries[LAUNCH_KEY] == info.launch.encode()
    assert json.loads(store.entries[RUN_KEY]) == ["my-run", "explicit"]
    assert info.launch_source is None and T.read_launch_again(info) == info.launch


def test_each_init_makes_a_new_nonce_from_the_systems_entropy(monkeypatch):
    """A nonce at each `init`, never from a generator the training uses."""
    monkeypatch.setattr(T, "_exchanges", itertools.count(0))
    states = (random.getstate(), torch.get_rng_state())
    launches = {T._shared_run(_dist(Store()), 0, "r").launch for _ in range(5)}
    assert len(launches) == 5
    assert random.getstate() == states[0] and torch.equal(torch.get_rng_state(), states[1])


def test_another_rank_looks_once_and_never_waits_for_the_launch(exchange_7):
    store = Store({RUN_KEY: json.dumps(["r", "generated"]), LAUNCH_KEY: "b" * 32})
    info = T._shared_run(_dist(store), 1, None)
    assert info.run_id == "r" and info.launch == "b" * 32
    assert info.launch_source == (store, LAUNCH_KEY)
    assert [(w, k) for w, k, _ in store.calls][-2:] == [("check", LAUNCH_KEY), ("get", LAUNCH_KEY)]


def test_a_rank_0_on_an_older_sdk_leaves_a_null_launch_and_no_wait(exchange_7):
    store = Store({RUN_KEY: json.dumps(["r", "generated"])})
    began = time.monotonic()
    info = T._shared_run(_dist(store), 1, None)
    assert time.monotonic() - began < 1.0
    assert info.run_id == "r" and info.launch is None
    assert [(w, k) for w, k, _ in store.calls if k == LAUNCH_KEY] == [("check", LAUNCH_KEY)]


def test_a_stale_entry_from_an_earlier_start_is_replaced_by_the_second_read(exchange_7):
    """Contract 2: torchrun keeps the store across a restart; the restarted rank 1 finds the
    earlier start's entries before the new rank 0 has set its own."""
    store = Store({RUN_KEY: json.dumps(["r", "generated"]), LAUNCH_KEY: "a" * 32})
    info = T._shared_run(_dist(store), 1, None)
    assert info.launch == "a" * 32  # provisional: the earlier start's
    store.entries[LAUNCH_KEY] = b"c" * 32  # the new rank 0 reaches init
    assert T.read_launch_again(info) == "c" * 32
    # What the rules leave (contract 9): a rank 0 that has still not set it is not seen.
    store.entries[LAUNCH_KEY] = b"a" * 32
    assert T.read_launch_again(info) == "a" * 32


def test_the_second_read_never_raises_and_never_waits_for_long():
    for store in (Store({LAUNCH_KEY: "c" * 32}, fail=True), Store({}), Store({LAUNCH_KEY: "not a launch"})):
        info = T.RunInfo("r", "generated", 1, "a" * 32, (store, LAUNCH_KEY))
        assert T.read_launch_again(info) is None
    hung = Store({LAUNCH_KEY: "c" * 32}, hang=3.0)
    info = T.RunInfo("r", "generated", 1, "a" * 32, (hung, LAUNCH_KEY))
    began = time.monotonic()
    assert T.read_launch_again(info, timeout=0.2) is None
    assert time.monotonic() - began < 1.0
    assert hung.calls[0][2] == "tensorquill-launch"  # on a thread of its own


def test_a_store_that_fails_at_init_leaves_a_null_launch(exchange_7):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        info = T._shared_run(_dist(RuntimeError("no store")), 1, None)
        assert info.launch is None and info.launch_source is None and T.read_launch_again(info) is None
        broken = Store(fail=True)
        info = T._shared_run(_dist(broken), 1, None)
        assert info.launch is None and info.launch_source is None
        # Rank 0 keeps its own nonce when the store fails: it never waits and never raises.
        info = T._shared_run(_dist(Store(fail=True)), 0, "r")
        assert HEX32.fullmatch(info.launch)


def test_one_process_makes_its_own_launch_and_several_without_a_group_have_none(monkeypatch):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "MLFLOW_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        pytest.skip("a process group is initialized in this process")
    one, two = T.resolve_run("r"), T.resolve_run("r")
    assert HEX32.fullmatch(one.launch) and one.launch != two.launch
    monkeypatch.setenv("WORLD_SIZE", "2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert T.resolve_run("r").launch is None


def test_the_session_reads_the_launch_again_on_its_writer_thread_before_the_first_observation(world1):
    """The records carry what the second read gave, and the read is never on the training thread."""
    model = _model()
    session, sink = _init(model, world1, every=2, replica_every=2)
    store = Store({LAUNCH_KEY: "c" * 32})
    session.run = dataclasses.replace(session.run, rank=1, launch="a" * 32, launch_source=(store, LAUNCH_KEY))
    session.launch = "a" * 32
    # Rewritten by lane AA1: the read now starts at the end of the first step on a thread of its own
    # and the writer never waits for it, so the test waits for it here before the first block.
    _train(session, model, 1)
    assert session._launch_read.done.wait(5)
    _train(session, model, 5)
    session.close()
    assert [r["replica"]["launch"] for r in sink.records if "replica" in r] == ["c" * 32] * 3
    assert [(w, k) for w, k, _ in store.calls] == [("check", LAUNCH_KEY), ("get", LAUNCH_KEY)]  # once
    assert {thread for _, _, thread in store.calls} == {"tensorquill-launch"}
    assert session.launch == "c" * 32 and session.launch_read_seconds is not None
    assert session.run.launch_source is None  # the store is not kept once read


def test_a_closed_session_keeps_no_store_it_never_read(world1):
    model = _model()
    session, _ = _init(model, world1, every=2, replica_every=50)
    session.run = dataclasses.replace(session.run, rank=1, launch_source=(Store(), LAUNCH_KEY))
    _train(session, model, 2)
    session.close()
    assert session.run.launch_source is None


# ---------------------------------------------------------------------------------------------
# Two processes: rank 1 trains while rank 0 never steps.

SCRUBBED = (
    "MLFLOW_RUN_ID", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "SLURM_STEP_ID", "RANK", "LOCAL_RANK", "WORLD_SIZE",
    "MASTER_ADDR", "MASTER_PORT", "TENSORQUILL_ENDPOINT", "TENSORQUILL_PROJECT", "TENSORQUILL_RUN_NAME",
    "MLFLOW_EXPERIMENT_NAME",
)

_WORKER = """
import dataclasses, datetime, json, sys, threading, time, warnings
from pathlib import Path
import torch, torch.distributed as dist
from torch import nn

rank, world, store_path, out_dir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
dist.init_process_group("gloo", store=dist.FileStore(store_path, world), rank=rank, world_size=world,
                        timeout=datetime.timedelta(seconds=15))
import tensorquill

class Spy:
    def __init__(self, store):
        self.store, self.threads = store, []
    def check(self, keys):
        self.threads.append(threading.current_thread().name)
        return self.store.check(keys)
    def get(self, key):
        self.threads.append(threading.current_thread().name)
        return self.store.get(key)

found = {"rank": rank}
torch.manual_seed(0)
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
sink = tensorquill.MemorySink()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    session = tensorquill.init(model, sink=sink, every=4, replica_group=dist.group.WORLD, replica_every=2,
                               heartbeat_seconds=None, stack_seconds=None)
spy = None
source = getattr(session.run, "launch_source", None)  # absent before this lane: no launch at all
if rank == 1:
    if source is not None:
        store, key = source
        spy = Spy(store)
        session.run = dataclasses.replace(session.run, launch_source=(spy, key))
else:
    # Rank 0 never steps until rank 1 has finished all of its steps.
    deadline = time.monotonic() + 60
    while not (out_dir / "rank1_done").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
steps = 12 if rank == 1 else 4
began = time.monotonic()
try:
    for i in range(steps):
        opt.zero_grad()
        model(torch.ones(2, 4)).sum().backward()
        opt.step()
        if i == 0 and getattr(session, "_launch_read", None) is not None:
            # Lane AA1: the second read runs on its own thread from the end of the first step, and
            # records carry null until it returns; wait for it so every observation has the launch.
            session._launch_read.done.wait(10)
    found["seconds"] = time.monotonic() - began
    session.close()
except BaseException as error:
    found["raised"] = f"{type(error).__name__}: {error}"[:300]
if rank == 1:
    (out_dir / "rank1_done").write_text("1")
found["records"] = sink.records
found["launch"] = getattr(session, "launch", None)
found["run_id"] = session.run.run_id
found["spy_threads"] = spy.threads if spy else None
found["launch_read_seconds"] = getattr(session, "launch_read_seconds", None)
(out_dir / f"rank{rank}.json").write_text(json.dumps(found))
dist.destroy_process_group()
"""


def _two_ranks(tmp_path: Path) -> list[dict]:
    script = tmp_path / "worker.py"
    script.write_text(textwrap.dedent(_WORKER))
    env = {k: v for k, v in os.environ.items() if k not in SCRUBBED}
    env["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    source = str(Path(tensorquill.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([source, *filter(None, [os.environ.get("PYTHONPATH")])])
    procs = [
        subprocess.Popen([sys.executable, str(script), str(rank), "2", str(tmp_path / "store"), str(tmp_path)],
                         cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for rank in range(2)
    ]
    logs = []
    try:
        for proc in procs:
            logs.append(proc.communicate(timeout=90)[0])
    except subprocess.TimeoutExpired:
        for proc in procs:
            proc.kill()
        pytest.fail("a rank did not finish within 90 s")
    found = []
    for rank in range(2):
        result = tmp_path / f"rank{rank}.json"
        assert result.exists(), f"rank {rank} wrote nothing; its output:\n{logs[rank][-3000:]}"
        found.append(json.loads(result.read_text()))
    return found


def _reference():
    path = Path(__file__).parent / "test_replica_contract.py"
    spec = importlib.util.spec_from_file_location("_z1_replica_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reference


@needs_dist
def test_two_ranks_one_trains_while_the_other_never_steps(tmp_path):
    """Before: rank 1 waited in the scheduled gather at step 1 until the group's 15 s timeout and
    raised out of `optimizer.step()`, since rank 0 was not stepping."""
    rank0, rank1 = _two_ranks(tmp_path)
    for found in (rank0, rank1):
        assert "raised" not in found, found
    assert rank1["seconds"] < 10.0, rank1["seconds"]
    obs1 = {r["step"]: r["replica"] for r in rank1["records"] if "replica" in r}
    obs0 = {r["step"]: r["replica"] for r in rank0["records"] if "replica" in r}
    assert sorted(obs1) == [1, 3, 5, 7, 9, 11] and sorted(obs0) == [1, 3]
    # One launch for the start, learned through the store: rank 0's nonce on both.
    assert HEX32.fullmatch(rank0["launch"]) and rank1["launch"] == rank0["launch"]
    assert rank1["run_id"] == rank0["run_id"]
    assert {o["launch"] for o in [*obs0.values(), *obs1.values()]} == {rank0["launch"]}
    assert all(o["group"] == [0, 1] for o in [*obs0.values(), *obs1.values()])
    # The second read happened once, on its own thread, never the training thread.
    assert rank1["spy_threads"] == ["tensorquill-launch", "tensorquill-launch"], rank1["spy_threads"]
    # The same weights on the same steps: the two ranks' observations are equal, and the
    # contract's reference reading pairs them as complete and agreeing.
    for step in (1, 3):
        assert obs0[step]["params"] == obs1[step]["params"]
    records = [r for r in rank0["records"] + rank1["records"]]
    result = _reference()(records, 1e-6)
    by_step = {e["step"]: e for e in result["steps"]}
    for step in (1, 3):
        assert by_step[step]["coverage"] == "complete"
        assert by_step[step]["fingerprints"] == "agree" and by_step[step]["signatures"] == "agree"
    for step in (5, 7, 9, 11):
        assert by_step[step]["coverage"] == "insufficient" and by_step[step]["missing"] == [0]
    assert result["replica_drift"] == []
