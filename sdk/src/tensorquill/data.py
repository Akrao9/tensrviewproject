"""Data progress and checkpoint manifests: host-side metadata only.

Nothing here reads a sample or a weight. A step's `data` record holds how many batches
were drawn, a hash of their sample indices (or ids), the loader pass and the sampler's own
epoch; a checkpoint's manifest says which components a state dict holds and a few cheap
identities of each (counts, an optimizer's step, a scheduler's `last_epoch`, a loss scale).

- `session.data(indices, ...)` notes one batch. `session.watch_loader(loader)` does it for
  every batch of a `torch.utils.data.DataLoader`, reading the indices its batch sampler
  yields (map-style datasets) or the ids a callable takes from each batch (`ids=`, needed for
  an `IterableDataset`, whose sampler yields no indices).
- `session.checkpoint(state, event="save" | "load")` records the manifest of the state dict
  that was saved or loaded. At a load the checkpoint's tensors are also compared with the
  model's by name and shape (`load_manifest`, CHECKPOINT_TENSOR_CONTRACT.md): names and shapes
  only, read from each module's own tables, never through `state_dict()`.
  `session.checkpoint_files(folder, event="load")` compares a folder's `.safetensors` weights
  the same way, listed from their headers alone (`safetensors_listing`): no tensor byte is read.

The hash is BLAKE2b (8 bytes, unkeyed) over each batch's length and bytes, in order, so
every process, rank and attempt that draws the same samples in the same order gets the same
hash. A batch of integers that fit int64 is its indices as little-endian int64 after a tag
byte (one C loop: the common case); any other batch is a tag byte, then each item's type,
byte length and bytes (text as UTF-8, bytes as they are), so batches of different ids do
not share their bytes. It is written as `v2:` and 16 hex digits, or `v3:` for a step that drew
a batch holding both bools and other integers (below).

The prefix is the encoding's version. A writer before it wrote the 16 hex digits alone,
from int64 indices or from the items' text joined by NUL bytes, in which different batches
could share their bytes (`["a\\x00b", "c"]` and `["a", "b\\x00c"]`; byte ids decoded with
replacement). Hashes of two versions are of different encodings: equal batches give
different hashes, so a reader compares only hashes of one version (the text before the first
`:`, none for the old writer's) and leaves a pair of two versions not compared.

`v3` (C16, lane X4, Nov 19, 2026) is `v2` without its one exception: the `v2` writer hashed a
list that starts with an int and holds a bool as its integers (`[1, True]` as `[1, 1]`), and
now such a batch is typed items, as `(1, True)` already was. Every other batch's bytes are as
they were, so a step whose batches hold no bool beside another integer keeps its `v2:` hash
(and resumes from a `v2` run are still compared); a step that drew one is written `v3:`, so its
hash is never compared with the `v2` writer's hash of the same batch. A tuple `(1, True)`,
which `v2` hashed right, is written `v3:` too, so a batch's kind does not depend on what holds
it. Telling the two apart
costs one C loop over the batch's types beside the one that packs it.
"""

from __future__ import annotations

import collections
import copyreg
import hashlib
import json
import numbers
import os
import re
import stat
import sys
import threading
import warnings
from array import array
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import torch

# Component names a manifest can list, in the order they are reported.
COMPONENTS = ("model", "optimizer", "scheduler", "scaler", "rng", "data")
_DATA_WORDS = ("dataloader", "data_loader", "sampler", "data_state", "data_position", "loops", "dataset")
_COUNTER_KEYS = ("step", "global_step", "epoch", "iteration", "steps", "global_steps")
_MAX_OTHER = 20


HASH_PREFIX = "v2:"  # the version of the encoding below; the old writer's hashes have none
MIXED_PREFIX = "v3:"  # a step that drew a batch of bools and other integers (C16, lane X4)
_INTS, _ITEMS = b"i", b"t"  # a batch's first byte: int64 indices, or typed items


def _int_bytes(values: Any) -> bytes | None:
    """Little-endian int64, in one C loop (a Python loop over 1,024 indices took 0.6 ms)."""
    try:
        packed = array("q", values)
    except (TypeError, ValueError, OverflowError):
        return None
    if sys.byteorder == "big":
        packed.byteswap()
    return _INTS + packed.tobytes()


def _item_bytes(value: Any) -> bytes:
    """One id of a batch that is not all int64: its type, its byte length, its bytes."""
    if isinstance(value, str):
        kind, raw = b"s", value.encode("utf-8", "surrogatepass")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        kind, raw = b"b", bytes(value)
    elif isinstance(value, bool):
        kind, raw = b"?", b"1" if value else b"0"
    elif isinstance(value, numbers.Integral):
        kind, raw = b"n", str(int(value)).encode()
    elif isinstance(value, numbers.Real):
        kind, raw = b"f", repr(float(value)).encode()
    else:
        cls = type(value)
        kind, raw = b"o", f"{cls.__module__}.{cls.__qualname__}\x00{value}".encode("utf-8", "surrogatepass")
    return kind + len(raw).to_bytes(8, "little") + raw


def identity_bytes(values: Any) -> bytes | None:
    """The bytes a batch's sample indices (or ids) are hashed from; None when there are none."""
    return _identity(values)[0]


def _mixed(values: list[Any]) -> bool:
    """Whether a batch holds a bool and another integer: the batches whose bytes `v2` got wrong."""
    return any(isinstance(v, bool) for v in values) and any(
        isinstance(v, numbers.Integral) and not isinstance(v, bool) for v in values
    )


def _identity(values: Any) -> tuple[bytes | None, bool]:
    """`identity_bytes`, and whether the batch is mixed (`_mixed`)."""
    if values is None:
        return None, False
    if isinstance(values, list) and values and type(values[0]) is int and bool not in set(map(type, values)):
        fast = _int_bytes(values)  # the common case: a batch sampler's list of indices
        if fast is not None:
            return fast, False
    if isinstance(values, torch.Tensor):
        values = values.detach().reshape(-1).tolist()
    elif isinstance(values, (str, bytes, bytearray, memoryview)) or not isinstance(values, Iterable):
        values = [values]
    else:
        values = [v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v for v in values]
    if not values or any(v is None for v in values):
        return None, False  # an IterableDataset's sampler yields None for every sample
    if all(isinstance(v, numbers.Integral) and not isinstance(v, bool) for v in values):
        fast = _int_bytes(values)
        if fast is not None:
            return fast, False  # else an index beyond int64: typed items below
    return _ITEMS + b"".join(_item_bytes(v) for v in values), _mixed(values)


class DataStep:
    """What one step drew: the batches noted since the previous step, hashed in order."""

    def __init__(self) -> None:
        self.batches = 0
        self._hash: Any = hashlib.blake2b(digest_size=8)
        self._known = True
        self._mixed = False
        self.fields: dict[str, Any] = {}

    def add(self, values: Any, fields: Mapping[str, Any]) -> None:
        raw, mixed = _identity(values)
        self._mixed |= mixed
        if raw is None:
            self._known = False
        else:
            # The length first, so [1, 2] + [3] and [1] + [2, 3] differ.
            self._hash.update(len(raw).to_bytes(8, "little"))
            self._hash.update(raw)
        self.batches += 1
        self.fields.update((k, v) for k, v in fields.items() if v is not None)

    def record(self) -> dict[str, Any]:
        out: dict[str, Any] = {"batches": self.batches}
        if self.batches and self._known:
            out["batch_hash"] = (MIXED_PREFIX if self._mixed else HASH_PREFIX) + self._hash.hexdigest()
        out.update(self.fields)
        return out


def batch_hash(*batches: Any) -> str | None:
    """The `batch_hash` a step that drew these batches (each a list of indices) records."""
    step = DataStep()
    for values in batches:
        step.add(values, {})
    return step.record().get("batch_hash")


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return int(value)


def data_fields(
    *, epoch: Any = None, sampler_epoch: Any = None, shuffle: Any = None, data_rank: Any = None,
    shard: Any = None, offset: Any = None,
) -> dict[str, Any]:
    """The optional fields of a `data` record, as JSON values."""
    fields: dict[str, Any] = {
        "epoch": _int_or_none(epoch),
        "sampler_epoch": _int_or_none(sampler_epoch),
        "shuffle": bool(shuffle) if isinstance(shuffle, (bool, int)) else None,
        "data_rank": _int_or_none(data_rank),
        "offset": _int_or_none(offset),
    }
    if shard is not None:
        fields["shard"] = shard if isinstance(shard, (int, str)) and not isinstance(shard, bool) else str(shard)
    return fields


# --- the loader hook -------------------------------------------------------------------------


def _sampler_info(loader: Any) -> tuple[Any, Any, Any]:
    """The sampler holding the epoch, whether it shuffles (None: unknown), and its rank."""
    from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler

    candidates = [getattr(loader, "sampler", None), getattr(getattr(loader, "batch_sampler", None), "sampler", None)]
    for sampler in candidates:
        if isinstance(sampler, DistributedSampler):
            return sampler, bool(sampler.shuffle), sampler.rank
    for sampler in candidates:
        if isinstance(sampler, RandomSampler):
            return sampler, True, None
        if isinstance(sampler, SequentialSampler):
            return sampler, False, None
    with_epoch = next((s for s in candidates if hasattr(s, "epoch")), None)
    return with_epoch, getattr(with_epoch, "shuffle", None), getattr(with_epoch, "rank", None)


# Index batches a watched pass may hold before they are noted. A DataLoader draws ahead only as
# far as its workers prefetch; a pass that draws further notes the rest of its batches as
# unknown rather than risk pairing a batch with another batch's indices.
_FIFO_LIMIT = 4096

_claim = threading.local()  # .target, .fifo: the pass being opened on this thread
_swap_lock = threading.Lock()
_swapped: dict[int, list[Any]] = {}  # id(target) -> [target, its own class, open windows]
_recording_classes: dict[type, type] = {}


class _Fifo:
    """One watched pass's index batches, in the order the sampler yielded them."""

    __slots__ = ("items", "lost")

    def __init__(self) -> None:
        self.items: collections.deque = collections.deque()
        self.lost = False


def _recorded(inner: Iterator[Any], fifo: _Fifo) -> Iterator[Any]:
    for item in inner:
        if not fifo.lost:
            if len(fifo.items) >= _FIFO_LIMIT:
                fifo.lost = True
                fifo.items.clear()
            else:
                fifo.items.append(item)
        yield item


def _plain_copy(obj: Any, base: type) -> Any:
    """`obj` as an instance of its own class `base`, for pickling and copying."""
    clone = object.__new__(base)
    if hasattr(obj, "__dict__"):
        clone.__dict__.update(obj.__dict__)
    for name in copyreg._slotnames(base):  # type: ignore[attr-defined]
        if hasattr(obj, name):
            object.__setattr__(clone, name, getattr(obj, name))
    return clone


def _recording_class(base: type) -> type:
    """A subclass of `base` whose `__iter__` records into the pass this thread is opening, if
    it is opening one on this very object, and is `base.__iter__` otherwise. No slots of its
    own, so an instance's class can be swapped to it and back; it pickles and copies as `base`."""
    cls = _recording_classes.get(base)
    if cls is not None:
        return cls

    def __iter__(self: Any) -> Iterator[Any]:
        # A plain function, not a generator: the pass is claimed when the iterator is made.
        inner = base.__iter__(self)
        if getattr(_claim, "target", None) is not self:
            return inner
        return _recorded(iter(inner), _claim.fifo)

    def __reduce_ex__(self: Any, protocol: Any) -> Any:
        reduced = _plain_copy(self, base).__reduce_ex__(protocol)
        # Pickle refuses `copyreg.__newobj__` whose class is not the pickled object's own;
        # `copyreg._reconstructor` makes the same `object.__new__(base)` without that check.
        if isinstance(reduced, tuple) and len(reduced) >= 2 and reduced[0] is copyreg.__newobj__ \
                and len(reduced[1]) == 1:
            reduced = (copyreg._reconstructor, (reduced[1][0], object, None), *reduced[2:])  # type: ignore[attr-defined]
        return reduced

    cls = type(base.__name__, (base,), {
        "__slots__": (), "__iter__": __iter__, "__reduce_ex__": __reduce_ex__, "__qualname__": base.__qualname__,
    })
    _recording_classes[base] = cls
    return cls


def _swap_in(target: Any) -> bool:
    """Give `target` its recording class for one window; False when its class cannot change."""
    with _swap_lock:
        entry = _swapped.get(id(target))
        if entry is not None and entry[0] is target:
            entry[2] += 1
            return True
        base = type(target)
        try:
            target.__class__ = _recording_class(base)
        except (TypeError, AttributeError):
            return False
        _swapped[id(target)] = [target, base, 1]
        return True


def _swap_out(target: Any) -> None:
    """End one window: when it was the last open one, `target` has its own class back."""
    with _swap_lock:
        entry = _swapped.get(id(target))
        if entry is None or entry[0] is not target:
            return
        entry[2] -= 1
        if entry[2] > 0:
            return
        del _swapped[id(target)]
        if type(target) is _recording_classes.get(entry[1]):
            try:
                target.__class__ = entry[1]
            except (TypeError, AttributeError):
                pass


def _report(message: str) -> None:
    """Say what went wrong without raising into training: a RuntimeWarning, and where warnings
    are turned into errors (`python -W error`), a line on stderr instead."""
    try:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    except Exception:  # noqa: BLE001
        try:
            sys.stderr.write(f"{message}\n")
        except Exception:  # noqa: BLE001
            pass


class WatchedLoader:
    """Iterates a DataLoader and notes each batch with `session.data`. Every other attribute
    is the loader's. Made by `session.watch_loader(loader)`.

    The indices are read from the loader's batch sampler (or its sampler, with batching off)
    while one of this wrapper's passes draws them. The sampler's class is swapped for a
    recording subclass only while a pass is opened, from `iter()` to its first batch (Accelerate's
    loaders draw from the sampler only at the first `next()`), and the recording reaches only
    the pass being opened on that thread. At every other moment the sampler is its own class:
    it pickles, a pass over the unwrapped loader records nothing, and two wrappers of one loader,
    or two passes of one wrapper, each have their own queue. Once the session is closed the
    wrapper yields the loader's batches and notes nothing.
    """

    def __init__(self, session: Any, loader: Any, *, ids: Callable[[Any], Any] | None = None, epoch: int = 0) -> None:
        self._session = session
        self._loader = loader
        self._ids = ids
        self._epoch = epoch - 1  # incremented at the start of each pass
        self._warned = False
        self._sampler, self._shuffle, self._data_rank = _sampler_info(loader)
        self._target: Any = None
        if ids is None and getattr(loader, "in_order", True) is not False:
            batch_sampler = getattr(loader, "batch_sampler", None)
            self._target = batch_sampler if batch_sampler is not None else getattr(loader, "sampler", None)
        self._indices = self._target is not None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def __len__(self) -> int:
        return len(self._loader)

    def _session_closed(self) -> bool:
        return bool(getattr(self._session, "_closed", False))

    def __iter__(self) -> Iterator[Any]:
        self._epoch += 1
        if self._session_closed():
            yield from self._loader
            return
        fifo = _Fifo() if self._indices else None
        iterator, first, done = self._open(fifo)
        if done:
            return
        batch = first
        while True:
            if not self._session_closed():
                try:
                    self._note(batch, fifo)
                except Exception as error:  # noqa: BLE001 - bookkeeping never stops training
                    if not self._warned:
                        self._warned = True
                        _report(f"TensorQuill could not note a batch: {error!r}")
            yield batch
            try:
                batch = next(iterator)
            except StopIteration:
                return

    def _open(self, fifo: _Fifo | None) -> tuple[Iterator[Any], Any, bool]:
        """Make the loader's iterator and take its first batch, recording into `fifo` the
        index batches the sampler yields to this pass. Returns (iterator, first batch, empty)."""
        target = self._target
        swapped = fifo is not None and _swap_in(target)
        previous = (getattr(_claim, "target", None), getattr(_claim, "fifo", None))
        if swapped:
            _claim.target, _claim.fifo = target, fifo
        elif fifo is not None:
            self._indices = False  # the sampler's class cannot change: no indices from now on
        try:
            iterator = iter(self._loader)
            try:
                return iterator, next(iterator), False
            except StopIteration:
                return iterator, None, True
        finally:
            if swapped:
                _claim.target, _claim.fifo = previous
                _swap_out(target)

    def _note(self, batch: Any, fifo: _Fifo | None) -> None:
        if self._ids is not None:
            values = self._ids(batch)
        elif fifo is not None and fifo.items:
            values = fifo.items.popleft()
        else:
            values = None
        sampler = self._sampler
        self._session.data(
            values,
            epoch=self._epoch,
            sampler_epoch=getattr(sampler, "epoch", None) if sampler is not None else None,
            shuffle=self._shuffle,
            data_rank=self._data_rank,
        )


# --- checkpoint manifests --------------------------------------------------------------------


def _number(value: Any) -> float | int | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return value if isinstance(value, int) else float(value)


def _keys_hash(keys: Iterable[Any]) -> str:
    digest = hashlib.blake2b(digest_size=8)
    for key in sorted(str(k) for k in keys):
        digest.update(key.encode())
        digest.update(b"\x00")
    return digest.hexdigest()


def _optimizer_step(state: Mapping[Any, Any]) -> float | int | None:
    for entry in state.values():
        if isinstance(entry, Mapping) and "step" in entry:
            return _number(entry["step"])
    return None


def _extra_state_key(key: Any) -> bool:
    """A module's `get_extra_state` entry in its state dict (`<prefix>_extra_state`), any value."""
    return isinstance(key, str) and (key == "_extra_state" or key.endswith("._extra_state"))


def _model_tensors(value: Mapping[Any, Any]) -> int:
    """How many tensors a model state dict holds; 0 when `value` is not one: every entry a
    tensor or a module's extra state, at least one tensor."""
    tensors = 0
    for k, v in value.items():
        if isinstance(v, torch.Tensor):
            tensors += 1
        elif not _extra_state_key(k):
            return 0
    return tensors


# The names a checkpoint gives the entry that holds a model: the last word of its last dotted
# part is one of these (`model`, `ema_model`, `student_model`, `model_ema`, `teacher_net`), once
# a trailing `state`, `states` or `dict` is set aside (`model_state`, `model_state_dict`), or
# it is `state_dict` itself (Lightning's). The last word, since it says what the
# entry is: `model_cache` and `weights_cache` are caches.
_MODEL_WORDS = frozenset(("model", "models", "module", "modules", "net", "network", "ema", "weights"))
_STATE_WORDS = frozenset(("state", "states", "dict"))
# The last words that say an entry is a cache, which no key evidence overrides (lane Z6, L23):
# `kv_cache` with `layers.0.k`, `embedding_cache` with `emb_weight`, `model_cache`.
_CACHE_WORDS = frozenset(("cache", "caches"))
# The names PyTorch's own modules give their parameters and buffers (`Linear`, `Embedding` and
# the norms: `weight`, `bias`; `BatchNorm`: `running_*`, `num_batches_tracked`;
# `MultiheadAttention`: `in_proj_*`, `bias_k`, `bias_v`; pruning and weight norm: `*_orig`,
# `*_mask`, `weight_g`, `weight_v`; a rotary embedding's `inv_freq`). An RNN's are `weight_ih_l0`,
# `bias_hh_l0`, ...: a `weight_` or `bias_` prefix, or a `_weight` or `_bias` suffix, is one too.
_PARAMETER_NAMES = frozenset((
    "weight", "bias", "running_mean", "running_var", "num_batches_tracked", "in_proj_weight", "in_proj_bias",
    "bias_k", "bias_v", "weight_orig", "weight_mask", "bias_orig", "bias_mask", "weight_g", "weight_v", "inv_freq",
))
_WORD_BREAKS = re.compile(r"[^a-z0-9]+")


def _last_word(name: str) -> str | None:
    """The last word of `name` once a trailing `state`, `states` or `dict` is set aside; None
    when nothing is left. `state_dict` itself is "state_dict"."""
    words = [word for word in _WORD_BREAKS.split(name) if word]
    if words == ["state", "dict"]:
        return "state_dict"
    while words and words[-1] in _STATE_WORDS:
        words.pop()
    return words[-1] if words else None


def _names_a_model(name: str) -> bool:
    word = _last_word(name)
    return word == "state_dict" or word in _MODEL_WORDS


def _names_a_cache(name: str) -> bool:
    return _last_word(name) in _CACHE_WORDS


def _parameter_path(key: Any) -> bool:
    """A key a module's `state_dict()` writes: a dotted path (`layer.0.weight`, with DDP's
    `module.` or `torch.compile`'s `_orig_mod.` before it), a module's extra state, or a
    parameter or buffer of PyTorch's own modules."""
    if not isinstance(key, str):
        return False
    if "." in key.strip("."):
        return True
    return _extra_state_key(key) or key in _PARAMETER_NAMES or key.startswith(("weight_", "bias_")) or key.endswith(
        ("_weight", "_bias"))


def _model_state(name: str, value: Mapping[Any, Any]) -> int:
    """How many tensors `value` holds when it is a model's state dict, else 0 (lane Y6, XC9).

    A mapping of tensors is not a model by its shape alone: a cache of `cached_embedding`
    tensors has the same shape, and taken for the model it would say a resume restored a
    model it never held. So it is the model only when its entry's name says so (`name`, the
    last dotted part of its key, lowercased) or one of its keys is a parameter path, and never
    when its name says it is a cache (`kv_cache`, `model_cache`: lane Z6), whatever its keys.

    Known limits (L23): a model whose parameters have none of these names, stored under a key
    that names no model, is passed over: it is listed under `other`, and a resume from it
    reads the model as missing; a model kept under a cache's name is passed over likewise. A
    mapping of non-parameter tensors under any other name with a dotted or `*_weight` key, or
    under a model's name, is still taken for the model.
    """
    tensors = _model_tensors(value)
    if tensors and not _names_a_cache(name) and (_names_a_model(name) or any(_parameter_path(k) for k in value)):
        return tensors
    return 0


def _classify(key: str, value: Any) -> tuple[str, dict[str, Any]] | None:
    """The component a state-dict entry holds, with its identities; None when not known.

    A name is matched by its last dotted part, so a parameter path inside a model's state
    dict (`model.resampler.weight`) is never taken for a sampler or an RNG state."""
    name = key.rsplit(".", 1)[-1].lower()
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, Mapping) for v in value):
        found = [_classify(key, v) for v in value]
        kinds = {f[0] for f in found if f is not None}
        if len(kinds) == 1 and all(f is not None for f in found):
            kind = kinds.pop()
            first = dict(found[0][1])  # type: ignore[index]
            first["count"] = len(value)
            return kind, first
        return None
    if isinstance(value, Mapping):
        if "state" in value and "param_groups" in value:
            state = value["state"] if isinstance(value["state"], Mapping) else {}
            groups = value["param_groups"] if isinstance(value["param_groups"], (list, tuple)) else []
            identity: dict[str, Any] = {"entries": len(state), "param_groups": len(groups)}
            step = _optimizer_step(state)
            if step is not None:
                identity["step"] = step
            return "optimizer", identity
        if "last_epoch" in value:
            identity = {"last_epoch": _number(value["last_epoch"])}
            if "_step_count" in value:
                identity["step_count"] = _number(value["_step_count"])
            return "scheduler", identity
        if "scale" in value and ("growth_factor" in value or "_growth_tracker" in value):
            return "scaler", {"scale": _number(value["scale"]), "growth_tracker": _number(value.get("_growth_tracker"))}
        if "rng" in name or "random" in name:
            return "rng", {}
        if any(word in name for word in _DATA_WORDS) or any(
            k in value for k in ("_num_yielded", "_sampler_iter_yielded", "samples_seen")
        ):
            return "data", {}
        tensors = _model_state(name, value)
        if tensors:
            return "model", {"tensors": tensors, "keys_hash": _keys_hash(value.keys())}
        return None
    if "rng" in name or "random" in name:
        return "rng", {}
    if any(word in name for word in _DATA_WORDS):
        return "data", {}
    return None


def checkpoint_manifest(state: Mapping[str, Any]) -> dict[str, Any]:
    """Which components a checkpoint's state dict holds, and cheap identities of each.

    Each top-level entry is recognised by its shape: an optimizer's `state` and
    `param_groups`, a scheduler's `last_epoch`, a GradScaler's `scale`, a model's mapping
    of tensors (with any module's `_extra_state` entries) when its name or its keys say it is a
    model's (`_model_state`: a cache of tensors is not); RNG states and a data position by
    the last dotted part of their names (`rng`, `random`;
    `dataloader`, `sampler`, `loops`, ...) or a StatefulDataLoader's keys. A mapping that is
    none of these is looked into once, so `{"training": {"optimizer": ...}}` is found too.
    Integer counters such as `step`, `global_step` and `epoch` are listed as `counters`.
    No tensor is read except one-element ones (an optimizer's step, a scaler's scale).
    """
    components: dict[str, dict[str, Any]] = {}
    counters: dict[str, int] = {}
    other: list[str] = []

    def visit(key: str, value: Any, depth: int) -> None:
        if key.lower() in _COUNTER_KEYS and _int_or_none(value) is not None:
            counters[key] = int(_int_or_none(value))  # type: ignore[arg-type]
            return
        found = _classify(key, value)
        if found is not None:
            kind, identity = found
            components.setdefault(kind, identity)
            components[kind].setdefault("keys", [])
            components[kind]["keys"].append(key)
            return
        if depth == 0 and isinstance(value, Mapping) and value:
            for child_key, child in value.items():
                visit(f"{key}.{child_key}", child, depth + 1)
            return
        other.append(key)

    for key, value in state.items():
        visit(str(key), value, 0)
    manifest: dict[str, Any] = {
        "components": {name: components[name] for name in COMPONENTS if name in components},
    }
    if counters:
        manifest["counters"] = counters
    if other:
        manifest["other"] = other[:_MAX_OTHER]
    return manifest


def files_manifest(names: Iterable[str]) -> dict[str, Any]:
    """A manifest from the file names in a Hugging Face `Trainer` or Accelerate checkpoint folder."""
    components: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        lower = name.lower()
        kind = (
            "optimizer" if lower.startswith("optimizer") else
            "scheduler" if lower.startswith("scheduler") else
            "scaler" if lower.startswith("scaler") else
            "rng" if lower.startswith(("rng_state", "random_states")) else
            # The Trainer resumes its data from the step and epoch in trainer_state.json;
            # Accelerate saves its samplers' and loaders' states.
            "data" if lower == "trainer_state.json" or lower.startswith(("sampler", "dl_state_dict")) else
            "model" if lower.endswith((".safetensors", ".bin")) and ("model" in lower) else
            None
        )
        if kind is not None:
            components.setdefault(kind, {"files": []})["files"].append(name)
    return {"components": {name: components[name] for name in COMPONENTS if name in components}}


# --- a checkpoint's tensors against the model's (CHECKPOINT_TENSOR_CONTRACT.md) ----------------
#
# At a load only. The model is read from each module's own parameter and buffer tables, through
# `torch.nn.Module`'s own functions, never through `state_dict()`: a distributed wrapper's state
# export can gather from other ranks, and a state-dict hook may do anything. Only names, shapes,
# `requires_grad` and which names are one parameter are read; no tensor's values, so nothing waits
# on a device. The checkpoint's side is the names and shapes of the mapping already in memory.

NAMES_LISTED = 50  # a list's names in the manifest, sorted; `count` is the whole number
# A folder load whose tensors are not listed: no `.safetensors` weights, or (with `refused`) their
# headers refused (`safetensors_listing`).
FOLDER_TENSORS = {"state": "unavailable", "why": "folder"}
_WRAPPERS = ("module.", "_orig_mod.")  # DDP's and `torch.compile`'s, removed from every name on both sides
# A key is conventional when its last word (a trailing `state`, `states` or `dict` set aside) is one
# of these, or it is `state_dict`, and none of its words names an averaged copy.
_CONVENTIONAL_WORDS = frozenset(("model", "net", "module", "network", "weights"))
_AVERAGED_WORDS = frozenset(("ema", "swa", "shadow", "avg", "average", "averaged"))
# The functions of `torch.nn.Module` the walk below calls, and the ones the contract names: a module
# that replaces any of them with its own is not read (`model_not_readable`).
_WALK = ("named_parameters", "named_buffers", "named_modules", "named_children")


class _NotReadable(Exception):
    """The model cannot be read by the local path: a module replaces PyTorch's own walk."""


def _strip_wrappers(name: str) -> str:
    while True:
        for wrapper in _WRAPPERS:
            if name.startswith(wrapper):
                name = name[len(wrapper):]
                break
        else:
            return name


def _replaces_walk(module: Any) -> bool:
    own = vars(module)
    cls = type(module)
    return any(name in own or getattr(cls, name, None) is not getattr(torch.nn.Module, name) for name in _WALK)


def _read_model(model: Any) -> list[tuple[str, list[int], str, int | None]]:
    """The model's tensors as `state_dict()` would name them, without calling it: `(name, shape,
    kind, identity)`, kind "p" a trainable parameter, "f" a frozen one, "b" a saved buffer; the
    identity (`id`) of a parameter tells two names that are one parameter.

    Each module is checked before anything of it is called. Its parameters and buffers are read
    with `torch.nn.Module.named_parameters` and `named_buffers`, not recursing, duplicates kept (as
    `state_dict()` writes a tied weight under each name); a buffer named in the module's
    `_non_persistent_buffers_set` is left out. A module that has buffers and not that set is not
    read (`model_not_readable`): which of its buffers are saved cannot be told. Its children are
    read from its own `_modules` table, as `state_dict()` walks them, so a module held twice by one
    parent is named twice (`named_children` would name it once)."""
    if not isinstance(model, torch.nn.Module):
        raise _NotReadable
    out: list[tuple[str, list[int], str, int | None]] = []
    on_path: set[int] = set()

    def walk(prefix: str, module: Any) -> None:
        if _replaces_walk(module):
            raise _NotReadable
        own = vars(module)
        children = own.get("_modules")
        if not isinstance(children, dict):
            raise _NotReadable
        on_path.add(id(module))
        for name, parameter in torch.nn.Module.named_parameters(module, recurse=False, remove_duplicate=False):
            out.append((prefix + name, list(parameter.shape), "p" if parameter.requires_grad else "f", id(parameter)))
        buffers = list(torch.nn.Module.named_buffers(module, recurse=False, remove_duplicate=False))
        not_saved = own.get("_non_persistent_buffers_set")
        if buffers and not isinstance(not_saved, (set, frozenset)):
            raise _NotReadable
        for name, buffer in buffers:
            if name not in not_saved:
                out.append((prefix + name, list(buffer.shape), "b", None))
        for name, child in list(children.items()):
            if child is not None and id(child) not in on_path:  # a module that holds itself: state_dict() would not end
                walk(f"{prefix}{name}.", child)
        on_path.discard(id(module))

    walk("", model)
    return out


def _entry_shapes(value: Any, *, loose: bool = False) -> dict[str, list[int]] | None:
    """`{key: shape}` of a mapping that holds a model's tensors under string keys: every entry a
    tensor or a module's extra state, at least one tensor. None when it is not one. `loose` (the
    entry the loader named) takes the tensors under string keys and passes over the rest."""
    if not isinstance(value, Mapping):
        return None
    shapes: dict[str, list[int]] = {}
    for key, item in value.items():
        if isinstance(item, torch.Tensor) and isinstance(key, str):
            shapes[key] = list(item.shape)
        elif not loose and not _extra_state_key(key):
            return None
    return shapes or (None if not loose else {})


def _not_looked_at(key: str, value: Any) -> bool:
    found = _classify(key, value) if isinstance(value, Mapping) else None
    return found is not None and found[0] in ("optimizer", "scheduler", "scaler")


def _entries(state: Mapping[Any, Any]) -> list[tuple[str | None, Mapping[Any, Any], dict[str, list[int]]]]:
    """Every mapping of tensors in the checkpoint, at the top level or one level down (`outer.inner`),
    or the checkpoint itself when its tensors sit directly in it (key None). An optimizer's, a
    scheduler's or a scaler's state is not looked at."""
    own = _entry_shapes(state)
    if own is not None:
        return [(None, state, own)]
    found: list[tuple[str | None, Mapping[Any, Any], dict[str, list[int]]]] = []
    for key, value in state.items():
        key = str(key)
        if not isinstance(value, Mapping) or _not_looked_at(key, value):
            continue
        shapes = _entry_shapes(value)
        if shapes is not None:
            found.append((key, value, shapes))
            continue
        for child_key, child in value.items():
            inner = f"{key}.{child_key}"
            if isinstance(child, Mapping) and not _not_looked_at(inner, child):
                shapes = _entry_shapes(child)
                if shapes is not None:
                    found.append((inner, child, shapes))
    return found


def _conventional(key: str | None) -> bool:
    """A key that names a model and no averaged copy: `model`, `state_dict`, `net`, `module`,
    `network`, `weights`, by the last word of its last dotted part; not `ema`, `model_ema`,
    `swa_model`, `shadow`, `avg_net`."""
    if key is None:
        return False
    name = key.rsplit(".", 1)[-1].lower()
    if any(word in _AVERAGED_WORDS for word in _WORD_BREAKS.split(name)):
        return False
    word = _last_word(name)
    return word == "state_dict" or word in _CONVENTIONAL_WORDS


def _leading_paths(name: str) -> Iterator[str]:
    start = name.find(".")
    while 0 < start < len(name) - 1:
        yield name[:start + 1]
        start = name.find(".", start + 1)


def _passes(matched: int, entry_names: int, model_names: int) -> bool:
    """The filter: at least half of the entry's names are the model's, or at least half of the
    model's names are in the entry (and at least one name is shared)."""
    return matched > 0 and (2 * matched >= entry_names or 2 * matched >= model_names)


def _stripped(names: Iterable[str], path: str) -> set[str]:
    return {name[len(path):] if name.startswith(path) else name for name in names}


class _Match:
    """One entry set against the model: its names made comparable, and how many both hold."""

    def __init__(self, key: str | None, mapping: Mapping[Any, Any], shapes: dict[str, list[int]],
                 model: dict[str, tuple[list[int], str, int | None]]) -> None:
        self.key, self.mapping, self.shapes = key, mapping, shapes
        # The side on which two names became one when made comparable ("checkpoint" or "model"):
        # such a match is never compared, since which of the two a name stands for cannot be told.
        self.collides: str | None = None
        self.entry = {}  # comparable name -> the entry's own key
        for raw in shapes:
            self.entry.setdefault(_strip_wrappers(raw), raw)
        if len(self.entry) < len(shapes):
            self.collides = "checkpoint"
        self.model = {name: name for name in model}  # comparable name -> the model's name
        self.prefix: dict[str, str] | None = None
        self.matched = sum(name in self.model for name in self.entry)
        if not self.passes():
            self._try_prefix()

    def passes(self) -> bool:
        return _passes(self.matched, len(self.entry), len(self.model))

    def _try_prefix(self) -> None:
        """One leading dotted path removed from one side, the one that leaves the most names in
        common, kept when the entry then passes. Scored in one pass over each name's paths (an
        upper bound: two names can become one), then counted exactly from the best down."""
        entry, model = set(self.entry), set(self.model)
        bound: collections.Counter[tuple[int, int, str]] = collections.Counter()
        for side, names, other in ((0, entry, model), (1, model, entry)):
            for name in names:
                for path in _leading_paths(name):
                    bound[(side, len(path), path)] += (name[len(path):] in other) - (name in other)
        best: tuple[int, tuple[int, int, str]] | None = None
        for (side, length, path), gain in sorted(bound.items(), key=lambda kv: (-kv[1], kv[0])):
            # A removal that adds no name in common cannot make a failing entry pass, and one whose
            # bound is no more than the best counted cannot beat it (ties keep the first in order).
            if gain <= 0 or (best is not None and self.matched + gain <= best[0]):
                break
            exact = len((_stripped(entry, path) & model) if side == 0 else (_stripped(model, path) & entry))
            if best is None or exact > best[0]:
                best = (exact, (side, length, path))
        if best is None or not _passes(best[0], len(self.entry), len(self.model)):
            return
        side, _, path = best[1]
        if side == 0:
            before = len(self.entry)
            self.entry = {}
            for raw in self.shapes:
                name = _strip_wrappers(raw)
                self.entry.setdefault(name[len(path):] if name.startswith(path) else name, raw)
            if len(self.entry) < before:
                self.collides = "checkpoint"
        else:
            self.model = {}
            for name in model:
                self.model.setdefault(name[len(path):] if name.startswith(path) else name, name)
            if len(self.model) < len(model):
                self.collides = "model"
        self.matched = sum(name in self.model for name in self.entry)
        self.prefix = {"side": "checkpoint" if side == 0 else "model", "removed": path}

    def closest(self) -> dict[str, Any]:
        return {"entry": self.key, "checkpoint": len(self.shapes), "matched": self.matched}


def _holds_half(match: _Match, model: dict[str, tuple[list[int], str, int | None]]) -> bool:
    """Whether an inferred entry (rules 2 and 4) holds at least half of the model's names: a share,
    not a count, with a name that is the same parameter as a name the entry holds counted as held."""
    held = {match.model[name] for name in match.entry if name in match.model}
    identities = {model[name][2] for name in held if model[name][1] != "b"}
    tied = sum(1 for name, (_, kind, identity) in model.items()
               if name not in held and kind != "b" and identity in identities)
    return 2 * (len(held) + tied) >= len(model)


def _listed(names: Iterable[str]) -> dict[str, Any]:
    found = sorted(names)
    return {"count": len(found), "names": found[:NAMES_LISTED]}


def _compared(match: _Match, by: str, model: dict[str, tuple[list[int], str, int | None]]) -> dict[str, Any]:
    held = {match.model[name] for name in match.entry if name in match.model}  # the model's names the entry holds
    held_parameters = {model[name][2] for name in held if model[name][1] != "b"}
    lists: dict[str, list[str]] = {key: [] for key in (
        "missing", "missing_frozen", "missing_buffers", "unexpected", "shape_differs", "tied")}
    shapes: dict[str, dict[str, list[int]]] = {}
    by_model_name = {model_name: comparable for comparable, model_name in match.model.items()}
    for name, (shape, kind, identity) in model.items():
        if name in held:
            theirs = match.shapes[match.entry[by_model_name[name]]]
            if theirs != shape:
                lists["shape_differs"].append(name)
                shapes[name] = {"model": shape, "checkpoint": theirs}
        elif kind != "b" and identity in held_parameters:
            lists["tied"].append(name)
        else:
            lists[{"p": "missing", "f": "missing_frozen", "b": "missing_buffers"}[kind]].append(name)
    lists["unexpected"] = [raw for name, raw in match.entry.items() if name not in match.model]
    out: dict[str, Any] = {
        "state": "compared", "entry": match.key, "by": by, "model": len(model),
        "trainable": sum(kind == "p" for _, kind, _ in model.values()),
        "checkpoint": len(match.shapes), "matched": match.matched,
    }
    for key, names in lists.items():
        if names:
            out[key] = _listed(names)
    if "shape_differs" in out:
        out["shape_differs"]["shapes"] = {name: shapes[name] for name in out["shape_differs"]["names"]}
    if match.prefix is not None:
        out["prefix"] = match.prefix
    return out


def _key_of(state: Mapping[Any, Any], mapping: Mapping[Any, Any]) -> str | None:
    """The key under which `state` holds `mapping` itself (by identity), at the top level or one
    level down as `outer.inner`; None for the checkpoint itself, or a mapping it does not hold."""
    for key, value in state.items():
        if value is mapping:
            return str(key)
    for key, value in state.items():
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                if child is mapping:
                    return f"{key}.{child_key}"
    return None


def _unavailable(why: str, **more: Any) -> dict[str, Any]:
    return {"state": "unavailable", "why": why, **more}


def _model_table(model: Any) -> tuple[dict[str, tuple[list[int], str, int | None]], dict[str, Any] | None]:
    """The model's tensors by comparable name (wrappers removed), `{name: (shape, kind, identity)}`,
    and None; or nothing and the `unavailable` object, when the model cannot be read
    (`model_not_readable`) or two of its names become one once a wrapper is removed (`ambiguous`,
    `collides: "model"`)."""
    try:
        rows = _read_model(model)
    except _NotReadable:
        return {}, _unavailable("model_not_readable")
    names: dict[str, tuple[list[int], str, int | None]] = {}
    for name, shape, kind, identity in rows:
        names.setdefault(_strip_wrappers(name), (shape, kind, identity))
    if len(names) < len(rows):  # two of the model's names are one once a wrapper is removed
        return {}, _unavailable("ambiguous", collides="model")
    return names, None


def _compared_or_collides(match: _Match, by: str, names: dict[str, tuple[list[int], str, int | None]]
                          ) -> tuple[dict[str, Any], _Match | None]:
    """A comparison, unless two names became one on either side: then which name a tensor stands
    for cannot be told, and nothing is said of the model's entry."""
    if match.collides is not None:
        return _unavailable("ambiguous", collides=match.collides, closest=match.closest()), None
    return _compared(match, by, names), match


def _by_rules(matches: list[_Match], names: dict[str, tuple[list[int], str, int | None]],
              conventional: Callable[[str | None], bool] | None = None) -> tuple[dict[str, Any], _Match | None]:
    """Rules 2 to 6 of the contract's section 2 over the entries looked at. `conventional` says
    whether an entry's key names a model (`_conventional` by default; a folder's file names are
    read by their stem)."""
    conventional = conventional or _conventional
    passing = [m for m in matches if m.passes()]
    named = [m for m in passing if conventional(m.key)]
    if len(named) == 1:  # rule 2: the key says which entry is looked at, its names whether it is compared
        match = named[0]
        if _holds_half(match, names):
            return _compared_or_collides(match, "name", names)
        return _unavailable("weak_match", closest=match.closest()), None
    if len(named) > 1:  # rule 3
        return _unavailable("ambiguous", candidates=sorted(str(m.key) for m in named)), None
    if len(passing) == 1:  # rule 4
        match = passing[0]
        if _holds_half(match, names):
            return _compared_or_collides(match, "names", names)
        return _unavailable("weak_match", closest=match.closest()), None
    if len(passing) > 1:  # rule 5
        return _unavailable("ambiguous", candidates=sorted(str(m.key) for m in passing)), None
    # Rule 6: no entry passes. The closest is the one that shares the most names, when one shares any.
    sharing = sorted((m for m in matches if m.matched > 0), key=lambda m: (-m.matched, str(m.key)))
    return (_unavailable("not_identified", closest=sharing[0].closest()) if sharing
            else _unavailable("not_identified")), None


def tensor_coverage(
    state: Mapping[Any, Any], model: Any, *, model_state: Any = None
) -> tuple[dict[str, Any], _Match | None]:
    """The load manifest's `tensors` object (CHECKPOINT_TENSOR_CONTRACT.md, sections 1 to 3), and
    the entry it names as the model's (compared, or the loader's entry whose names differ). May
    raise; `load_manifest` turns a raise into `error`."""
    names, unavailable = _model_table(model)
    if unavailable is not None:
        return unavailable, None
    # Rule 1: the loader named the entry (`model_state=`, the Lightning adapter): the very mapping
    # given to `load_state_dict`, found among the entries by identity.
    if isinstance(model_state, Mapping):
        match = _Match(_key_of(state, model_state), model_state, _entry_shapes(model_state, loose=True) or {}, names)
        if match.passes():
            return _compared_or_collides(match, "loader", names)
        return _unavailable("names_differ", closest=match.closest()), match
    return _by_rules([_Match(key, mapping, shapes, names) for key, mapping, shapes in _entries(state)], names)


def _with_model_entry(manifest: dict[str, Any], tensors: Mapping[str, Any], match: _Match | None) -> dict[str, Any]:
    """`components.model` at a load (section 2): the entry compared, or the loader's entry whose
    names differ; absent when no entry is the model's; otherwise today's."""
    components = dict(manifest["components"])
    other = list(manifest.get("other", ()))
    today = components.get("model")
    today_keys = list(today.get("keys", ())) if isinstance(today, Mapping) else []
    if match is not None:
        own = set(map(str, match.mapping)) if match.key is None else {match.key}
        components["model"] = {
            "tensors": sum(isinstance(v, torch.Tensor) for v in match.mapping.values()),
            "keys_hash": _keys_hash(match.mapping.keys()),
            "keys": ([] if match.key is None else [match.key]) + [k for k in today_keys if k != match.key],
        }
        other = [k for k in other if k not in own]
    elif tensors.get("why") == "not_identified" and today is not None:
        del components["model"]
        other = (other + [k for k in today_keys if k not in other])[:_MAX_OTHER]
    out = {**manifest, "components": {name: components[name] for name in COMPONENTS if name in components}}
    if other:
        out["other"] = other
    else:
        out.pop("other", None)
    return out


def load_manifest(state: Mapping[str, Any], model: Any, *, model_state: Any = None) -> dict[str, Any]:
    """`checkpoint_manifest(state)` for a load, compared with `model`: with `tensors`, and
    `components.model` as section 2 of CHECKPOINT_TENSOR_CONTRACT.md says. Never raises from the
    comparison: the manifest is then today's with `{"state": "unavailable", "why": "error"}`."""
    manifest = checkpoint_manifest(state)
    try:
        tensors, match = tensor_coverage(state, model, model_state=model_state)
        out = _with_model_entry(manifest, tensors, match)
    except Exception:  # noqa: BLE001  nothing here may stop the training
        return {**manifest, "tensors": _unavailable("error")}
    out["tensors"] = tensors
    return out


# --- a checkpoint folder's tensors, from its `.safetensors` headers ------------------------------
#
# A folder load (`checkpoint_files`, as the Hugging Face `Trainer` and Accelerate's `save_state`
# write) holds the model's weights in `.safetensors` files: `model.safetensors`, Accelerate's
# `model_1.safetensors` for a second prepared model, or `save_pretrained`'s shards
# `model-0000k-of-0000n.safetensors` listed by `model.safetensors.index.json` (its `weight_map`,
# tensor name -> shard file). A `.safetensors` file starts with N, an unsigned little-endian 64-bit
# integer, then N bytes of UTF-8 JSON: `{name: {"dtype", "shape", "data_offsets"}}`, with
# `__metadata__`, which is not a tensor; the tensors' bytes follow. Only those 8 + N bytes are read,
# with plain reads, the `safetensors` package not imported: no tensor byte, so no tensor's data is
# needed or checked (`data_offsets` are not read). Pickled weights (`.bin`, `.pt`) are never opened.
# Anything that cannot be trusted is refused, never raised: the load then reads as a folder always
# has (`why: "folder"`), with `refused`, the reason, and `file`.

SAFETENSORS_HEADER_CAP = 100_000_000  # the format's own cap on N: `safetensors` refuses a larger one
FOLDER_FILES_READ = 1024  # index and weight files read at one load; a folder with more is refused
# Header bytes read at one load, every file's N and each index's size together. The format allows
# 100 MB per file; read on the training thread at a load, a folder's JSON is bounded much lower.
# The largest real headers are a few MB (a model of ~100k tensors).
FOLDER_HEADER_BYTES = 32 * 1024 * 1024
_INDEX_SUFFIX = ".safetensors.index.json"
_WEIGHTS_SUFFIX = ".safetensors"
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)


class _Refused(Exception):
    """A folder whose tensors cannot be listed: `reason`, the file it was found in, and more."""

    def __init__(self, reason: str, file: str | None = None, **more: Any) -> None:
        super().__init__(reason)
        self.reason, self.file, self.more = reason, file, more

    def record(self) -> dict[str, Any]:
        return {**FOLDER_TENSORS, "refused": self.reason, **({"file": self.file} if self.file else {}), **self.more}


def _read_file_start(folder: str, name: str, budget: list[int], *, header: bool) -> bytes:
    """A regular file's header (`header`: N, then N bytes) or its whole content (an index), charged
    to `budget`. Checked with `stat` before it is opened, so a named pipe or a device is never
    opened: opening a pipe waits for a writer."""
    path = os.path.join(folder, name)
    try:
        if not stat.S_ISREG(os.stat(path).st_mode):
            raise _Refused("unreadable", name)
        fd = os.open(path, _READ_FLAGS)
    except OSError:
        raise _Refused("unreadable", name) from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise _Refused("unreadable", name)
        if header:
            start = _read_exactly(fd, 8, name)
            if len(start) < 8:
                raise _Refused("short_file", name)
            length = int.from_bytes(start, "little")
            if length > SAFETENSORS_HEADER_CAP:
                raise _Refused("header_too_large", name)
            if length > info.st_size - 8:
                raise _Refused("header_past_end", name)
        else:
            length = info.st_size
        if length > budget[0]:
            raise _Refused("too_many_header_bytes", name)
        budget[0] -= length
        raw = _read_exactly(fd, length, name)
        if len(raw) < length:  # the file shrank while it was read
            raise _Refused("header_past_end" if header else "unreadable", name)
        return raw
    finally:
        os.close(fd)


def _read_exactly(fd: int, n: int, name: str) -> bytes:
    chunks: list[bytes] = []
    left = n
    try:
        while left > 0:
            chunk = os.read(fd, min(left, 1 << 24))
            if not chunk:
                break
            chunks.append(chunk)
            left -= len(chunk)
    except OSError:
        raise _Refused("unreadable", name) from None
    return b"".join(chunks)


def _json_object(raw: bytes, name: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise _Refused("not_utf8", name) from None
    try:
        found = json.loads(text)
    except (ValueError, RecursionError):
        raise _Refused("not_json", name) from None
    if not isinstance(found, dict):
        raise _Refused("not_an_object", name)
    return found


def _header_shapes(header: Mapping[str, Any], name: str) -> dict[str, list[int]]:
    """`{tensor: shape}` of a header: every entry but `__metadata__` must have a non-empty string
    `dtype` and a `shape` of whole numbers from 0 up."""
    shapes: dict[str, list[int]] = {}
    for tensor, entry in header.items():
        if tensor == "__metadata__":
            continue
        dtype = entry.get("dtype") if isinstance(entry, dict) else None
        shape = entry.get("shape") if isinstance(entry, dict) else None
        if (not isinstance(dtype, str) or not dtype or not isinstance(shape, list)
                or not all(type(n) is int and n >= 0 for n in shape)):
            raise _Refused("bad_entry", name, tensor=tensor)
        shapes[tensor] = list(shape)
    return shapes


def _index_shards(folder: str, name: str, budget: list[int]) -> list[str]:
    """The shard files an index's `weight_map` names: plain `.safetensors` file names in the folder
    itself (a path, or a pickled shard, is refused: neither is opened)."""
    weight_map = _json_object(_read_file_start(folder, name, budget, header=False), name).get("weight_map")
    if not isinstance(weight_map, dict) or not all(isinstance(v, str) for v in weight_map.values()):
        raise _Refused("bad_index", name)
    shards = sorted(set(weight_map.values()))
    for shard in shards:
        if not shard.endswith(_WEIGHTS_SUFFIX) or "/" in shard or "\\" in shard or os.sep in shard:
            raise _Refused("bad_index", name)
    return shards


def _weights_stem(file: str | None) -> str | None:
    """`model` for `model.safetensors` and `model.safetensors.index.json`: the name a folder's
    entry is read by, as an entry of a state dict is read by its key."""
    if file is None:
        return None
    for suffix in (_INDEX_SUFFIX, _WEIGHTS_SUFFIX):
        if file.endswith(suffix):
            return file[:-len(suffix)]
    return file


def safetensors_listing(folder: str | os.PathLike[str], names: Iterable[str] | None = None) -> dict[str, Any]:
    """The tensors a checkpoint folder's `.safetensors` files hold, by name and shape, read from
    their headers only: `{"state": "listed", "entries": {file: {tensor: shape}}}`.

    Each `*.safetensors.index.json` is one entry, the shards its `weight_map` names read together;
    each other `.safetensors` file is an entry of its own (Accelerate's `model.safetensors` and
    `model_1.safetensors` are two models). `names` is the folder's listing when the caller has it.
    A folder with no `.safetensors` file is `{"state": "unavailable", "why": "folder"}`, nothing
    opened. Never raises: what cannot be read gives that object with `refused` (`unreadable`,
    `short_file`, `header_too_large`, `header_past_end`, `not_utf8`, `not_json`, `not_an_object`,
    `bad_entry`, `bad_index`, `shard_absent`, `duplicate_tensor`, `too_many_files`,
    `too_many_header_bytes`, `no_tensors`), `file` where one file is at fault and `tensor` where one
    tensor is."""
    try:
        where = os.fspath(folder)
        present = set(os.listdir(where) if names is None else names)
        indexes = sorted(n for n in present if n.endswith(_INDEX_SUFFIX))
        weights = sorted(n for n in present if n.endswith(_WEIGHTS_SUFFIX))
        if not indexes and not weights:
            return dict(FOLDER_TENSORS)
        if len(indexes) + len(weights) > FOLDER_FILES_READ:  # every one of them is read
            raise _Refused("too_many_files")
        budget = [FOLDER_HEADER_BYTES]
        groups: list[tuple[str, list[str]]] = []
        claimed: set[str] = set()
        for index in indexes:
            shards = _index_shards(where, index, budget)
            for shard in shards:
                if shard not in present:
                    raise _Refused("shard_absent", shard)
            groups.append((index, shards))
            claimed.update(shards)
        groups += [(name, [name]) for name in weights if name not in claimed]
        entries: dict[str, dict[str, list[int]]] = {}
        for entry, files in groups:
            shapes: dict[str, list[int]] = {}
            for file in files:
                for tensor, shape in _header_shapes(
                        _json_object(_read_file_start(where, file, budget, header=True), file), file).items():
                    if tensor in shapes:
                        raise _Refused("duplicate_tensor", file, tensor=tensor)
                    shapes[tensor] = shape
            if shapes:
                entries[entry] = shapes
    except _Refused as refusal:
        return refusal.record()
    except (OSError, TypeError, ValueError):
        return _Refused("unreadable").record()
    if not entries:
        return _Refused("no_tensors").record()
    return {"state": "listed", "entries": entries}


def folder_tensor_coverage(folder: str | os.PathLike[str], model: Any, names: Iterable[str] | None = None
                           ) -> dict[str, Any]:
    """The `tensors` object of a folder load: the folder's entries (`safetensors_listing`) set
    against `model` by the contract's rules 2 to 6, each entry read by its file's stem (`model` is
    conventional, `model_1` is not), with the same wrappers, prefix and tied weights as a state
    dict's. `entry` is the file compared: `model.safetensors` or `model.safetensors.index.json`.
    A folder that cannot be listed gives the listing's `unavailable` object. Never raises: a
    comparison that raises is `{"state": "unavailable", "why": "error"}`."""
    listed = safetensors_listing(folder, names)
    if listed.get("state") != "listed":
        return listed
    try:
        table, unavailable = _model_table(model)
        if unavailable is not None:
            return unavailable
        matches = [_Match(file, shapes, shapes, table) for file, shapes in listed["entries"].items()]
        tensors, _ = _by_rules(matches, table, lambda key: _conventional(_weights_stem(key)))
        return tensors
    except Exception:  # noqa: BLE001  nothing here may stop the training
        return _unavailable("error")


def optimizer_entries(optimizer: Any) -> int:
    state = getattr(optimizer, "state", None)
    return len(state) if isinstance(state, Mapping) else 0


def optimizer_step_value(optimizer: Any) -> Any:
    """The first parameter's `step` in the optimizer's state, as stored (a tensor or a number)."""
    state = getattr(optimizer, "state", None)
    if not isinstance(state, Mapping):
        return None
    for entry in state.values():
        if isinstance(entry, Mapping) and "step" in entry:
            step = entry["step"]
            return step.detach().clone() if isinstance(step, torch.Tensor) else step
    return None
