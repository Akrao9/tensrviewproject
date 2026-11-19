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
  it the `DataLoader` timer does not count the same wait again, nor does a `time_data` inside
  another. Its fetches count under the same rule as the `DataLoader` timer's (`_counts`).

A fetch counts when it is made on the thread that created the session, while its phase is
`"train"`, and not with the model in eval mode and gradients off: the rule by which the forward
hooks tell an evaluation that is not marked (`capture._evaluating`), so the evaluation loop of
a hand-written loop (`model.eval()` and `torch.no_grad()`) is left out without
`session.phase("eval")`. `step_time` leaves out the time spent in a phase other than `"train"`
that ended before the step did (`session.phase("eval")`, which the Lightning and Hugging Face
callbacks set), so a marked validation pass or checkpoint save is in neither number; the time
of one that is not marked is in `step_time`.

A collate function that moves the batch to the GPU itself is inside the timer; with a
blocking copy that time includes waiting for the GPU.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import torch
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


def _counts(session: "Session", thread: int) -> bool:
    """Whether a fetch on `thread` now is data wait for `session`: on the thread that created it, in
    its `"train"` phase, not inside a fetch it is already timing, open, and not an evaluation that
    is not marked (the model in eval mode with gradients off, as `capture._evaluating` reads a
    forward). The one rule for both timers, the `DataLoader` one and `time_data`."""
    return (
        session._data_thread == thread and session._phase == "train" and not session._fetching
        and not session._closed and (torch.is_grad_enabled() or session._model.training)
    )


def _counting() -> list["Session"]:
    if not _watching:
        return []
    thread = threading.get_ident()
    return [s for s in _watching if _counts(s, thread)]


def watch(session: "Session") -> None:
    """Time `DataLoader` iterators for `session` until `unwatch`.

    Another library may wrap `__next__` on top of the timer while it is installed. Then the
    timer cannot be taken out from under it, so it stays in that chain, passing through
    while no session watches, and `_original_next` keeps the method it wraps."""
    global _original_next
    if session in _watching:
        return
    current = _BaseDataLoaderIter.__dict__.get("__next__")
    if _original_next is None:
        _original_next = current
        _BaseDataLoaderIter.__next__ = _timed_next  # type: ignore[method-assign]
    elif current is _original_next:
        # The other wrapper was removed together with the timer under it: put the timer back.
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
        if not _counts(session, threading.get_ident()):
            return next(self._iterator)
        session._fetching = True
        start = time.perf_counter()
        try:
            return next(self._iterator)
        finally:
            session._pending_data_wait += time.perf_counter() - start
            session._fetching = False
