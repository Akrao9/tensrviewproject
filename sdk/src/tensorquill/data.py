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

The hash is BLAKE2b (8 bytes, unkeyed) over the indices as little-endian int64, or over
their text when they are not integers, so every process, rank and attempt that draws the
same samples in the same order gets the same hash. It is written as 16 hex digits.
"""

from __future__ import annotations

import collections
import hashlib
import numbers
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import torch

# Component names a manifest can list, in the order they are reported.
COMPONENTS = ("model", "optimizer", "scheduler", "scaler", "rng", "data")
_DATA_WORDS = ("dataloader", "data_loader", "sampler", "data_state", "data_position", "loops", "dataset")
_COUNTER_KEYS = ("step", "global_step", "epoch", "iteration", "steps", "global_steps")
_MAX_OTHER = 20


def _int_bytes(values: list[Any]) -> bytes | None:
    try:
        return b"".join(int(v).to_bytes(8, "little", signed=True) for v in values)
    except (TypeError, ValueError, OverflowError):
        return None


def identity_bytes(values: Any) -> bytes | None:
    """The bytes a batch's sample indices (or ids) are hashed from; None when there are none."""
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        values = values.detach().reshape(-1).tolist()
    elif isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        values = [values]
    else:
        values = [v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v for v in values]
    if not values or any(v is None for v in values):
        return None  # an IterableDataset's sampler yields None for every sample
    if all(isinstance(v, numbers.Integral) and not isinstance(v, bool) for v in values):
        return _int_bytes(values)
    return "\x00".join(v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v) for v in values).encode()


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
            out["batch_hash"] = self._hash.hexdigest()
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


def _record_indices(target: Any, fifo: collections.deque) -> bool:
    """Make `target` (a batch sampler, or a sampler when batching is off) put each item it
    yields into `fifo`. `iter()` looks `__iter__` up on the type, so the instance's class is
    swapped for a subclass; the DataLoader refuses a new `batch_sampler` attribute."""
    base = type(target)
    if getattr(base, "_tensorquill_recording", False):
        target._tensorquill_fifo = fifo
        return True

    class Recording(base):  # type: ignore[misc, valid-type]
        _tensorquill_recording = True

        def __iter__(self) -> Iterator[Any]:
            queue = self.__dict__.get("_tensorquill_fifo")
            if queue is not None:
                queue.clear()  # a new pass: what an abandoned one drew ahead is not coming
            for item in base.__iter__(self):
                if queue is not None:
                    queue.append(item)
                yield item

    Recording.__name__ = base.__name__
    Recording.__qualname__ = base.__qualname__
    try:
        target.__class__ = Recording
        target._tensorquill_fifo = fifo
    except (TypeError, AttributeError):
        return False
    return True


class WatchedLoader:
    """Iterates a DataLoader and notes each batch with `session.data`. Every other attribute
    is the loader's. Made by `session.watch_loader(loader)`."""

    def __init__(self, session: Any, loader: Any, *, ids: Callable[[Any], Any] | None = None, epoch: int = 0) -> None:
        self._session = session
        self._loader = loader
        self._ids = ids
        self._epoch = epoch - 1  # incremented at the start of each pass
        self._fifo: collections.deque = collections.deque()
        self._warned = False
        self._sampler, self._shuffle, self._data_rank = _sampler_info(loader)
        self._indices = False
        if ids is None and getattr(loader, "in_order", True) is not False:
            batch_sampler = getattr(loader, "batch_sampler", None)
            target = batch_sampler if batch_sampler is not None else getattr(loader, "sampler", None)
            self._indices = target is not None and _record_indices(target, self._fifo)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> Iterator[Any]:
        self._epoch += 1
        for batch in self._loader:
            try:
                self._note(batch)
            except Exception as error:  # noqa: BLE001 - bookkeeping never stops training
                if not self._warned:
                    self._warned = True
                    warnings.warn(f"TensorQuill could not note a batch: {error!r}", RuntimeWarning, stacklevel=2)
            yield batch

    def _note(self, batch: Any) -> None:
        if self._ids is not None:
            values = self._ids(batch)
        elif self._indices and self._fifo:
            values = self._fifo.popleft()
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


def _classify(key: str, value: Any) -> tuple[str, dict[str, Any]] | None:
    """The component a state-dict entry holds, with its identities; None when not known."""
    name = key.lower()
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
        if value and all(isinstance(v, torch.Tensor) for v in value.values()):
            return "model", {"tensors": len(value), "keys_hash": _keys_hash(value.keys())}
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
    of tensors; RNG states and a data position by their names (`rng`, `random`;
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
