"""Host time the training thread spends waiting for its next batch.

Each step record then carries `data_wait`, the seconds spent in `next()` on the training
data since the previous step ended, and `step_time`, the seconds since the previous step
ended, both from `time.perf_counter`. Only host time is measured, so nothing waits for the
GPU: the timer stops when the loader hands the batch over, before the training loop moves
it to the device. A timer around `.to(device, non_blocking=True)` would time only the
enqueue, and a blocking copy waits behind the GPU work already queued, which would report a
slow GPU as a slow input pipeline.

Two ways to turn it on:

- `session.watch_loaders()` (or `init(..., data_wait=True)`, which the Hugging Face and
  Lightning callbacks and `watch_accelerator` turn on) times `next()` on every
  `torch.utils.data.DataLoader` iterator, for as long as the session is open. Accelerate's
  `DataLoaderShard`, `Trainer` and Lightning all fetch from such an iterator and move the
  batch to the device afterwards, outside the timer. Only fetches on the thread that
  created the session and while its phase is `"train"` count, so evaluation loaders under
  `session.phase("eval")` and background prefetch threads do not.
- `session.time_data(loader)` wraps any iterable, for loops that do not use a PyTorch
  `DataLoader`: `for batch in session.time_data(loader): batch = batch.to(device)`. Inside
  it the `DataLoader` timer does not count the same wait again.

A collate function that moves the batch to the GPU itself is inside the timer; with a
blocking copy that time includes waiting for the GPU.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from torch.utils.data.dataloader import _BaseDataLoaderIter

if TYPE_CHECKING:
    from .capture import Session

_watching: list["Session"] = []
_original_next: Any = None


def _timed_next(self: Any) -> Any:
    original = _original_next
    sessions = _counting()
    if not sessions:
        return original(self)
    start = time.perf_counter()
    try:
        return original(self)
    finally:
        elapsed = time.perf_counter() - start
        for session in sessions:
            session._pending_data_wait += elapsed


def _counting() -> list["Session"]:
    if not _watching:
        return []
    thread = threading.get_ident()
    return [
        s for s in _watching
        if s._data_thread == thread and s._phase == "train" and not s._fetching and not s._closed
    ]


def watch(session: "Session") -> None:
    """Time `DataLoader` iterators for `session` until `unwatch`."""
    global _original_next
    if session in _watching:
        return
    if _original_next is None:
        _original_next = _BaseDataLoaderIter.__next__
        _BaseDataLoaderIter.__next__ = _timed_next  # type: ignore[method-assign]
    _watching.append(session)


def unwatch(session: "Session") -> None:
    global _original_next
    if session in _watching:
        _watching.remove(session)
    if not _watching and _original_next is not None:
        if _BaseDataLoaderIter.__dict__.get("__next__") is _timed_next:
            _BaseDataLoaderIter.__next__ = _original_next  # type: ignore[method-assign]
        _original_next = None


class TimedData:
    """What `session.time_data(iterable)` returns: the iterable, with `next()` timed."""

    def __init__(self, iterable: Iterable[Any], session: "Session") -> None:
        self._iterable = iterable
        self._session = session
        session._start_data_timing()

    def __iter__(self) -> Iterator[Any]:
        return _TimedIterator(iter(self._iterable), self._session)

    def __len__(self) -> int:
        return len(self._iterable)  # type: ignore[arg-type]

    def __getattr__(self, name: str) -> Any:
        # `loader.sampler.set_epoch(...)`, `loader.dataset` and the like reach the loader.
        if name == "_iterable":
            raise AttributeError(name)
        return getattr(self._iterable, name)


class _TimedIterator:
    def __init__(self, iterator: Iterator[Any], session: "Session") -> None:
        self._iterator = iterator
        self._session = session

    def __iter__(self) -> "_TimedIterator":
        return self

    def __next__(self) -> Any:
        session = self._session
        if session._closed:
            return next(self._iterator)
        session._fetching = True
        start = time.perf_counter()
        try:
            return next(self._iterator)
        finally:
            session._pending_data_wait += time.perf_counter() - start
            session._fetching = False
