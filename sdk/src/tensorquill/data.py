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
  that was saved or loaded.

The hash is BLAKE2b (8 bytes, unkeyed) over each batch's length and bytes, in order, so
every process, rank and attempt that draws the same samples in the same order gets the same
hash. A batch of integers that fit int64 is its indices as little-endian int64 after a tag
byte (one C loop: the common case); any other batch is a tag byte, then each item's type,
byte length and bytes (text as UTF-8, bytes as they are), so no two different batches share
their bytes. It is written as `v2:` and 16 hex digits.

The prefix is the encoding's version. A writer before it wrote the 16 hex digits alone,
from int64 indices or from the items' text joined by NUL bytes, in which different batches
could share their bytes (`["a\\x00b", "c"]` and `["a", "b\\x00c"]`; byte ids decoded with
replacement). Hashes of two versions are of different encodings: equal batches give
different hashes, so a reader compares only hashes of one version (the text before the first
`:`, none for the old writer's) and leaves a pair of two versions not compared.
"""

from __future__ import annotations

import collections
import copyreg
import hashlib
import numbers
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
    if values is None:
        return None
    if isinstance(values, list) and values and type(values[0]) is int:
        fast = _int_bytes(values)  # the common case: a batch sampler's list of indices
        if fast is not None:
            return fast
    if isinstance(values, torch.Tensor):
        values = values.detach().reshape(-1).tolist()
    elif isinstance(values, (str, bytes, bytearray, memoryview)) or not isinstance(values, Iterable):
        values = [values]
    else:
        values = [v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v for v in values]
    if not values or any(v is None for v in values):
        return None  # an IterableDataset's sampler yields None for every sample
    if all(isinstance(v, numbers.Integral) and not isinstance(v, bool) for v in values):
        fast = _int_bytes(values)
        if fast is not None:
            return fast  # else an index beyond int64: typed items below
    return _ITEMS + b"".join(_item_bytes(v) for v in values)


class DataStep:
    """What one step drew: the batches noted since the previous step, hashed in order."""

    def __init__(self) -> None:
        self.batches = 0
        self._hash: Any = hashlib.blake2b(digest_size=8)
        self._known = True
        self.fields: dict[str, Any] = {}

    def add(self, values: Any, fields: Mapping[str, Any]) -> None:
        raw = identity_bytes(values)
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
            out["batch_hash"] = HASH_PREFIX + self._hash.hexdigest()
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
        tensors = _model_tensors(value)
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
    of tensors (with any module's `_extra_state` entries); RNG states and a data position by
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
