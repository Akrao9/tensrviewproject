"""The passive in-process scalar mirror: `tensorquill.init(..., mirror_scalars=True)`.

Plain numbers a script already logs to W&B or MLflow are also recorded with `session.log()` on
TensorQuill's own step, so a run that never calls `session.log` still has its `loss` and `eval/*`
beside its incidents and in `compare` (the plan's Phase 1 item 2; the architecture page, section
2.5; the decision log, Nov 23, 2026). The tracker stays the source of truth: its call is made with
the caller's own arguments, its return value and its own exception reach the caller unchanged, and
nothing here talks to a tracker server, reads an API key or reads `wandb.run.step`.

What is wrapped, and only when the tracker's module is already in `sys.modules` (nothing is
imported here):
- W&B: `Run.log` on the class, and the module's `wandb.log`. `wandb.init()` rebinds `wandb.log`
  to the new run's bound `log` (`wandb/sdk/lib/module.py`, `set_global(log=run.log)`), which
  holds the function that was on the class when the run was made: the class wrapper catches a run
  made after `init`, the module wrapper a run made before it. `wandb.finish()` puts W&B's pre-init
  stub back, and the class wrapper catches the next run again. A subclass of `Run` defined by then
  with a `log` of its own is wrapped too: `wandb.init(mode="disabled")` makes a `NoopRun` whose
  `log` does not call `Run.log`.
- MLflow: `log_metric` and `log_metrics` on `mlflow` and on `mlflow.tracking.fluent`, which
  defines them (`mlflow/__init__.py` re-exports them by name). `MlflowClient` is not wrapped.

One user call records once: every wrapper shares a per-thread depth, and only the outermost one
records, whatever wraps what (`wandb.log` calling `Run.log`, a wrapper of the user's in between).
The values are read before the tracker's call and recorded after it returns, so a call the
tracker refused records nothing.

Accepted values: `int` and `float` (not `bool`), NumPy integer and floating scalars, and 0-D
tensors already on the CPU, of a real dtype. A tensor anywhere else is skipped unread: no
`.item()`, no copy, no sync. Media, strings, lists, nested dictionaries and anything else are
ignored. W&B's `step` and `commit` and MLflow's `step` are ignored: the value is recorded through
`session.log`, on the step it would give it, so several calls in one step (`commit=False`) add up there, and a key mirrored twice in a
step keeps its last value. A key the session got from `session.log` itself in this step (the
caller, or an adapter) is never replaced by a mirrored one. Under `phase("eval")` that step is the
newest one still in memory (`Session.log`), and the same holds there: neither an explicit value
logged after it ended nor its own value from before is replaced by a mirrored one.

The wrappers are shared by every session that mirrors, installed by the first and taken out when
the last one closes or stops; a name that someone else has wrapped again since is left as it is,
with ours under it passing every call through. Nothing here raises, also under `python -W error`:
a wrapper that cannot be installed is not, and a value that cannot be read is not mirrored.
"""

from __future__ import annotations

import functools
import sys
import threading
import weakref
from collections.abc import Mapping
from typing import Any, Callable

import torch

_MISSING = object()


class _Mirrored(float):
    """A float the mirror hands to `session.log`, which stores `float(value)`: a new object, so the
    value stored for a key is told from one the caller logged as the same float object."""

    __slots__ = ()


_INT_DTYPES = frozenset({torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64})


class _Patch:
    """One name TensorQuill wrapped: `owner.<name>` was `original` and is now `wrapper`."""

    __slots__ = ("owner", "name", "original", "wrapper")

    def __init__(self, owner: Any, name: str, original: Any, wrapper: Any) -> None:
        self.owner, self.name, self.original, self.wrapper = owner, name, original, wrapper


# The sessions that mirror, in the order they began; the names wrapped while any is open; what the
# mirror stored where each session records a value (`_log_into`); and every wrapper made, weakly.
_sessions: list[Any] = []
_patches: list[_Patch] = []
_written: dict[int, dict[tuple[str, int], dict[str, Any]]] = {}
_ours: "weakref.WeakSet[Callable[..., Any]]" = weakref.WeakSet()
_lock = threading.RLock()  # install and removal only; never taken on a tracker's call
_depth = threading.local()


def watch(session: Any) -> None:
    """Mirror into `session` from now on, and wrap each tracker name found not yet wrapped (a
    tracker imported since the last session began is wrapped now). Never raises."""
    try:
        with _lock:
            if not any(s is session for s in _sessions):
                _sessions.append(session)
            _install_found()
    except Exception:  # noqa: BLE001  the mirror must never stop a run from starting
        pass


def unwatch(session: Any) -> None:
    """Stop mirroring into `session`; when it was the last, put back every name still ours.
    Called by `close()`, `_stop()` and a failed `init`. Never raises."""
    try:
        with _lock:
            _sessions[:] = [s for s in _sessions if s is not session]
            _written.pop(id(session), None)
            if _sessions:
                return
            patches, _patches[:] = list(_patches), []
            for patch in reversed(patches):  # newest first, so stacked wrappers of one name unwind
                try:
                    if _own(patch.owner).get(patch.name) is patch.wrapper:
                        setattr(patch.owner, patch.name, patch.original)
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        pass


# ---- installing ------------------------------------------------------------------------------


def _own(obj: Any) -> dict[str, Any]:
    """`obj`'s own attributes, read without `getattr`: W&B and MLflow load some names lazily, and
    reading those would import them."""
    try:
        found = vars(obj)
    except TypeError:
        return {}
    return found if isinstance(found, Mapping) else {}


def _install_found() -> None:
    wandb = sys.modules.get("wandb")
    if wandb is not None:
        run_class = _own(wandb).get("Run")
        if not isinstance(run_class, type):
            run_class = _own(sys.modules.get("wandb.sdk.wandb_run")).get("Run")
        if isinstance(run_class, type):
            for cls in _with_subclasses(run_class):
                # A subclass with a `log` of its own, such as the `NoopRun` that
                # `wandb.init(mode="disabled")` makes (`wandb/sdk/lib/noop_run.py`), whose `log`
                # does not call `Run.log`. One defined after `init` is not seen.
                _wrap(cls, "log", _wandb_values, method=True)
        _wrap(wandb, "log", _wandb_values, method=False)
    for name in ("mlflow", "mlflow.tracking.fluent"):
        module = sys.modules.get(name)
        if module is not None:
            _wrap(module, "log_metric", _mlflow_metric, method=False)
            _wrap(module, "log_metrics", _mlflow_metrics, method=False)


def _with_subclasses(cls: type) -> list[type]:
    """`cls` and every subclass of it defined so far, each once, read with `type.__subclasses__`
    (no import, no attribute of the tracker's)."""
    found: list[type] = []
    todo = [cls]
    while todo and len(found) < 256:
        current = todo.pop()
        if any(current is seen for seen in found):
            continue
        found.append(current)
        try:
            todo.extend(type.__subclasses__(current))
        except Exception:  # noqa: BLE001
            pass
    return found


def _wrap(owner: Any, name: str, extract: Callable[[tuple, dict], list], *, method: bool) -> None:
    """Make `owner.<name>` a wrapper of what it is now, unless it is one of ours already or not
    callable. A class's method must be a plain function in its own dictionary."""
    try:
        original = _own(owner).get(name)
        if original is None or not callable(original) or _is_ours(original):
            return
        if method and not isinstance(original, type(_wrap)):
            return  # a staticmethod, a classmethod or a descriptor: not a method to bind
        wrapper = _make_wrapper(original, extract, method)
        setattr(owner, name, wrapper)
        _patches.append(_Patch(owner, name, original, wrapper))
    except Exception:  # noqa: BLE001  internals that refuse a wrapper leave the tracker as it was
        pass


def _is_ours(obj: Any) -> bool:
    try:
        return obj in _ours
    except TypeError:  # not weakly referable, so not one of ours
        return False


def _make_wrapper(original: Any, extract: Callable[[tuple, dict], list], method: bool) -> Callable[..., Any]:
    if method:
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            return _tee(original, extract, (self, *args), args, kwargs)
    else:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _tee(original, extract, args, args, kwargs)
    try:
        # The tracker's name, docstring and `__wrapped__` (so `inspect.signature` reads the
        # tracker's); not its `__dict__`, which may hold another library's markers.
        functools.update_wrapper(wrapper, original, updated=())
    except Exception:  # noqa: BLE001
        pass
    # Run eagerly where a compiled function calls the tracker, as the optimizer and scaler hooks
    # run (`capture.py`): the tracker's call breaks the graph there anyway.
    wrapper = torch.compiler.disable(wrapper)
    _ours.add(wrapper)
    return wrapper


# ---- one call --------------------------------------------------------------------------------


def _tee(original: Any, extract: Callable[[tuple, dict], list], call_args: tuple, args: tuple,
         kwargs: dict) -> Any:
    depth = getattr(_depth, "n", 0)
    if depth or not _sessions:
        return original(*call_args, **kwargs)
    try:
        values = extract(args, kwargs)
    except Exception:  # noqa: BLE001  a value or mapping that cannot be read is not mirrored
        values = []
    _depth.n = depth + 1
    try:
        out = original(*call_args, **kwargs)
    finally:
        _depth.n = depth
    if values:
        _record(values)
    return out


def _wandb_values(args: tuple, kwargs: dict) -> list[tuple[str, Any]]:
    """`Run.log(data, step=None, commit=None)` and `wandb.log`, the same call."""
    return _pairs(args[0] if args else kwargs.get("data"))


def _mlflow_metric(args: tuple, kwargs: dict) -> list[tuple[str, Any]]:
    """`log_metric(key, value, step=None, ...)`."""
    key = args[0] if len(args) > 0 else kwargs.get("key")
    value = args[1] if len(args) > 1 else kwargs.get("value", _MISSING)
    number = _number(value) if isinstance(key, str) and key else _MISSING
    return [] if number is _MISSING else [(key, number)]


def _mlflow_metrics(args: tuple, kwargs: dict) -> list[tuple[str, Any]]:
    """`log_metrics(metrics, step=None, ...)`."""
    return _pairs(args[0] if args else kwargs.get("metrics"))


def _pairs(data: Any) -> list[tuple[str, Any]]:
    if not isinstance(data, Mapping):
        return []
    found = []
    for key, value in data.items():
        if isinstance(key, str) and key:
            number = _number(value)
            if number is not _MISSING:
                found.append((key, number))
    return found


def _number(value: Any) -> Any:
    """`value` as a Python number when it is one the mirror accepts, else `_MISSING`. Reads no
    tensor that is not on the CPU."""
    kind = type(value)
    if kind is bool:
        return _MISSING
    if kind is float:
        return _Mirrored(value)
    if kind is int:
        return value
    if isinstance(value, torch.Tensor):
        if kind is not torch.Tensor and kind is not torch.nn.Parameter:
            return _MISSING  # a subclass (DTensor, a fake or functional tensor) is not read
        if value.device.type != "cpu" or value.layout is not torch.strided or value.dim() != 0:
            return _MISSING
        if not (value.dtype.is_floating_point or value.dtype in _INT_DTYPES):
            return _MISSING  # bool, complex, quantized
        number = value.item()  # on the CPU: no device sync
    elif isinstance(value, (int, float)):
        number = value  # a subclass: NumPy's float64, an IntEnum
    else:
        np = sys.modules.get("numpy")  # never imported here
        if np is None or not isinstance(value, (np.integer, np.floating)):
            return _MISSING
        number = value.item()
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        return _MISSING
    return _Mirrored(number) if isinstance(number, float) else int(number)


def _record(values: list[tuple[str, Any]]) -> None:
    for session in list(_sessions):
        try:
            _log_into(session, values)
        except Exception:  # noqa: BLE001  never into the tracker's caller
            pass


def _log_into(session: Any, values: list[tuple[str, Any]]) -> None:
    """`session.log` the values whose key the step they go on does not hold from anyone else.

    Where `session.log` puts a value now is `session._scalars_for_log()`: the step in progress, or,
    under the "eval" phase, the newest step still in memory (lane AK-5), each named by a key that
    changes when a step ends. The mirror remembers, for each of the last two such places, the object
    it stored under each key: a key holding anything else was logged by the caller or an adapter, or
    was the step's own before it ended, and is kept. With nowhere to go the value is passed to
    `session.log` all the same, which leaves it out and says so once."""
    if session._closed or session._stopped:
        return
    where, row = session._scalars_for_log()
    if row is None:
        session.log(dict(values))
        return
    marks = _written.get(id(session))
    if not isinstance(marks, dict):
        marks = _written[id(session)] = {}
    mine = marks.get(where)
    if mine is None:
        while len(marks) >= 2:  # the step in progress and the newest step are all a value can go on
            del marks[next(iter(marks))]
        mine = marks[where] = {}
    take = {}
    for key, number in values:
        held = row.get(key, _MISSING)
        if held is not _MISSING and mine.get(key, _MISSING) is not held:
            continue
        take[key] = number
    if not take:
        return
    try:
        session.log(take)
    finally:
        now, row = session._scalars_for_log()
        if now == where and row is not None:
            for key in take:
                stored = row.get(key, _MISSING)
                if stored is not _MISSING:
                    mine[key] = stored
