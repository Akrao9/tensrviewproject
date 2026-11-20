"""Find the tracker run that the tensor stream belongs to.

TensorQuill never imports a tracker itself. It only looks at trackers the training
script has already imported, so it adds no import cost and no dependency.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import secrets
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: At most how long `init` on a rank other than 0 waits for rank 0's run ID, in seconds, and
#: never longer than the process group's own timeout. Rank 0 is often later to `init` than the
#: others: it may still be starting its tracker's run (`wandb.init` waits up to 90 s by default),
#: and the Hugging Face `Trainer` starts it in the same event as TensorQuill's callback. After
#: this a rank writes under an ID of its own, with a warning.
RUN_ID_WAIT_SECONDS = 300.0

# One key in the process group's store for each exchange. Every rank calls `init` the same
# number of times (as for a collective), so the n-th call on every rank reads the n-th key, and
# a second `init` in the same group never reads the first one's ID. `init_process_group` puts the
# group's own prefix on every key. torchrun keeps one store across a restart of its workers, so
# a restarted rank can find the earlier start's entry under the same key (bug list, L25).
_KEY = "tensorquill/run_id/{}"
# The launch (replica observation contract, section 2): a nonce rank 0 makes at each `init` and
# sets under a key of its own before the run ID's, so the run ID's entry keeps its two items and a
# rank on an older SDK still reads it. The same n as the run ID's key.
_LAUNCH_KEY = "tensorquill/launch/{}"
_exchanges = itertools.count()

#: At most how long `read_launch_again` waits for the second read of the launch, in seconds; a
#: store that has not answered by then leaves the launch null. The session never waits for it: it
#: starts the read (`LaunchRead`) and writes null until the read returns.
LAUNCH_READ_SECONDS = 5.0


@dataclass(frozen=True)
class RunInfo:
    run_id: str | None
    source: str  # "explicit", "env", "mlflow", "wandb", "launcher" or "generated"
    rank: int
    # Which start of the job this process belongs to: 32 hex digits, or None when it could not be
    # learned (contract, section 2). On a rank other than 0 it is what `init` read, provisional
    # until `LaunchRead` reads it a second time. Not part of equality: two RunInfo are
    # equal when they name the same run, as before the launch was added.
    launch: str | None = field(default=None, compare=False)
    # (store, key) for that second read, on a rank other than 0 that took rank 0's run ID.
    launch_source: Any = field(default=None, repr=False, compare=False)


def new_launch() -> str:
    """A launch nonce: 128 bits from the operating system's entropy, never from a random number
    generator the training uses (`random`, `numpy`, `torch` are left untouched)."""
    return secrets.token_hex(16)


def _launch_or_none(value: Any) -> str | None:
    """`value` (bytes from the store) as a launch, or None when it is not 32 lowercase hex digits."""
    try:
        text = value.decode("ascii") if isinstance(value, (bytes, bytearray)) else str(value)
    except Exception:  # noqa: BLE001
        return None
    return text if len(text) == 32 and all(c in "0123456789abcdef" for c in text) else None


def _look_for_launch(store: Any, key: str) -> str | None:
    """One look at the launch's key: `check`, then `get` only if it is there (a `get` on a
    missing key waits the store's whole timeout). Never waits for it, never raises: None when it
    is not there, cannot be read, or is not a launch."""
    try:
        if not store.check([key]):
            return None
        return _launch_or_none(store.get(key))
    except Exception:  # noqa: BLE001  nothing here may stop a run
        return None


class LaunchRead:
    """The second read of the launch (contract, section 2), never waited for.

    Rank 0, a job of one rank and a rank that never took rank 0's run ID keep what `init` gave them:
    the read is done at once, with no thread. A rank other than 0 that did take it reads the
    launch's key once more, `check` then `get`, on a daemon thread of its own named
    `tensorquill-launch`, and takes what that gives, None if it fails or finds nothing: what `init`
    read may be the entry an earlier start left in a store torchrun kept across a restart.

    `LaunchRead(run)` reads at once. `LaunchRead(read_run=fn)` starts its thread now, parked, and
    reads when `go()` is called, from the run `fn()` gives then (None: nothing to read): the session
    starts it at `init` and calls `go()` at the end of the first step, which only sets an event, so
    the training thread never waits for a thread to start. `cancel()` lets a parked thread end
    without reading.

    `done` is set when the read has returned (`launch` then holds what it gave, and `seconds` how
    long the store took). Until then `result()` says it is pending, and a caller writes the launch
    as unknown (null), never what `init` read. Nothing joins the thread: a store that hangs leaves
    the read pending for good, costs one parked daemon thread, and holds up nothing else. The
    store's `check` and `get` release the GIL while they wait (PyTorch 2.6 binds both with
    `gil_scoped_release`, `torch/csrc/distributed/c10d/init.cpp`), so a hung one does not stall the
    other threads. Never raises."""

    def __init__(self, run: RunInfo | None = None, *, read_run: Any = None) -> None:
        self.done = threading.Event()
        self.launch: str | None = None
        self.seconds: float | None = None
        self._go = threading.Event()
        self._cancelled = False
        if read_run is None:
            if run is None or run.rank == 0 or run.launch_source is None:
                self.launch, self.seconds = (run.launch if run is not None else None), 0.0
                self.done.set()
                return
            read_run = lambda: run  # noqa: E731
            self._go.set()

        def look() -> None:
            try:
                self._go.wait()
                found = None if self._cancelled else read_run()
                if found is not None and found.rank != 0 and found.launch_source is not None:
                    began = time.perf_counter()
                    self.launch = _look_for_launch(*found.launch_source)
                    self.seconds = time.perf_counter() - began
                elif found is not None:
                    self.launch, self.seconds = found.launch, 0.0
            except Exception:  # noqa: BLE001
                self.launch = None
            finally:
                self.done.set()

        try:
            threading.Thread(target=look, name="tensorquill-launch", daemon=True).start()
        except Exception:  # noqa: BLE001  no thread to read with: the launch stays unknown
            self.launch, self.seconds = None, 0.0
            self.done.set()

    def go(self) -> None:
        """Let a parked read begin. An `Event.set`: it waits for nothing."""
        self._go.set()

    def cancel(self) -> None:
        """Let a parked read end without reading; one already reading is left to finish."""
        self._cancelled = True
        self._go.set()

    def result(self) -> tuple[bool, str | None]:
        """(True, the launch or None) once the read has returned; (False, None) while it is pending."""
        return (True, self.launch) if self.done.is_set() else (False, None)


def read_launch_again(run: RunInfo, timeout: float | None = None) -> str | None:
    """The second read of the launch (`LaunchRead`), waited for at most `timeout` seconds
    (`LAUNCH_READ_SECONDS` by default): what it gave, or None if it failed, found nothing or had
    not returned by then. For a caller that may wait; the session does not call it, it starts a
    `LaunchRead` and never waits for it. Never raises."""
    try:
        limit = max(float(LAUNCH_READ_SECONDS if timeout is None else timeout), 0.0)
    except (TypeError, ValueError):
        limit = 5.0
    try:
        read = LaunchRead(run)
        read.done.wait(limit)
        return read.result()[1]
    except Exception:  # noqa: BLE001
        return None


def _rank() -> int:
    dist = sys.modules.get("torch.distributed")
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    for name in ("RANK", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is not None and value.isdigit():
            return int(value)
    return 0


def resolve_run(run_id: str | None = None) -> RunInfo:
    """Resolve the run ID: explicit, then MLFLOW_RUN_ID, then MLflow, then W&B, then a
    launcher's job ID (torchrun, SLURM), and otherwise a generated unique ID, so separate
    runs never append to the same file.

    Under `torch.distributed` with several ranks, rank 0 resolves it, `run_id` included, and
    every rank takes rank 0's ID and where it came from: a tracker's run usually exists on rank 0
    only, and a `run_id` given on rank 0 only (`run_id=wandb.run.id if wandb.run else None`) is
    rank 0's like any other. A rank given another `run_id` than rank 0's is warned and writes
    under rank 0's. The ID goes through the default process group's store (`set` on rank 0,
    `check` and `get` on the others), never a collective, so nothing is put on a GPU and rank 0
    never waits. A rank that has not got it after `RUN_ID_WAIT_SECONDS` (or the group's own
    timeout, if shorter), or whose store fails, writes under an ID of its own, by the same order,
    with one warning; nothing raises. So every rank should call `tensorquill.init`, the same
    number of times.

    A tracker that raises while its run is read, on any rank or on one process, is passed over
    with one warning, and the order goes on (the launcher's ID, else a generated one). No
    warning here can raise, even under `python -W error`.

    `mlflow.active_run()` is thread-local and is None when the run was created through
    `MlflowClient` (PyTorch Lightning's `MLFlowLogger` does that, and only on rank 0), so
    those setups must pass `run_id` explicitly.
    """
    rank = _rank()
    dist = sys.modules.get("torch.distributed")
    if dist is not None and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        # Whether to exchange depends only on what every rank sees alike (the process group and
        # its size), never on `run_id`, a tracker or the environment, so every rank takes it.
        return _shared_run(dist, rank, run_id)
    # One process, or several ranks with no process group: the same guarded lookup as rank 0's,
    # so a tracker that raises is passed over here too.
    found = _resolve_here(rank, run_id, stacklevel=4)
    if _world_size() > 1:
        if found[1] in ("mlflow", "wandb"):
            _warn_tracker_unshared(found)
        elif found[1] == "generated":
            _warn_generated_unshared(found[0])
        # Several ranks and no process group to share a launch through: not known (contract 2).
        return RunInfo(found[0], found[1], rank, None)
    return RunInfo(found[0], found[1], rank, new_launch())  # a job of one rank makes its own


def _shared_run(dist: Any, rank: int, run_id: str | None) -> RunInfo:
    """Rank 0's (run ID, source) on every rank, through the default process group's store.

    Rank 0 resolves the whole order, puts the pair in the store and goes on: it never waits.
    The other ranks wait for it at most `RUN_ID_WAIT_SECONDS`, polling with `check` (10 ms,
    doubling to 1 s), never one long `wait`: a TCPStore client does one operation at a time,
    and NCCL's monitor thread polls the same client with `TORCH_NCCL_DUMP_ON_TIMEOUT` (PyTorch
    2.6, `ProcessGroupNCCL::heartbeatMonitor`), so a long `wait` would hold it up. Never raises:
    a store that fails, or a wait that runs out, leaves the rank with an ID of its own, and one
    warning.

    The launch (contract, section 2) goes the same way under a key of its own: rank 0 makes a
    nonce and sets it before the run ID's entry; a rank that has read the run ID looks for it once
    and never waits for it. A rank that did not get rank 0's run ID, or found no launch, has a
    null launch."""
    n = next(_exchanges)
    key, launch_key = _KEY.format(n), _LAUNCH_KEY.format(n)
    world = dist.get_world_size()
    if rank == 0:
        found = _resolve_here(rank, run_id)
        launch = new_launch()
        try:
            store = dist.distributed_c10d._get_default_store()
        except Exception:  # noqa: BLE001  said below, where the run ID's `set` fails the same way
            store = None
        try:
            # Before the run ID's entry: a rank that has read the new run ID finds the new launch.
            store.set(launch_key, launch)
        except Exception:  # noqa: BLE001  the other ranks record a null launch; nothing else changes
            pass
        try:
            if store is None:
                store = dist.distributed_c10d._get_default_store()
            store.set(key, json.dumps(list(found), default=str))
        except Exception as error:  # noqa: BLE001  nothing here may stop a run
            _say(
                f"tensorquill.init on rank 0 of {world} could not share its run ID {found[0]!r} through the process "
                f"group's store ({_brief(error)}), so the other ranks each write under an ID of their own once their "
                "wait for it runs out. Pass the same run_id= on every rank.",
                stacklevel=4,
            )
        return RunInfo(found[0], found[1], rank, launch)
    try:
        wait = max(float(RUN_ID_WAIT_SECONDS), 0.0)
    except (TypeError, ValueError):  # set to something that is not a number
        wait = 300.0
    try:
        store = dist.distributed_c10d._get_default_store()
        try:
            own = store.timeout.total_seconds()
        except Exception:  # noqa: BLE001  a store with no timeout of its own
            own = 0.0
        if 0 < own < wait:
            wait = own
        deadline, pause = time.monotonic() + wait, 0.01
        while not store.check([key]):
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("rank 0 has put nothing in the process group's store")
            time.sleep(min(pause, left))
            pause = min(pause * 2, 1.0)
        # Only once the key is there: `get` on a missing key waits the group's whole timeout.
        shared_id, source = json.loads(store.get(key))
        if not shared_id or not isinstance(source, str):
            raise ValueError(f"unreadable entry {shared_id!r}, {source!r}")
    except Exception as error:  # noqa: BLE001  nothing here may stop a run
        found = _resolve_here(rank, run_id)
        _say(
            f"tensorquill.init on rank {rank} of {world} did not get rank 0's run ID ({_brief(error)}; it waits "
            f"at most {wait:g} s), so this rank writes under {found[0]!r} ({found[1]}) and may be apart from the "
            "other ranks' run. Call tensorquill.init on every rank, rank 0 too, or pass the same run_id= on every "
            "rank.",
            stacklevel=4,
        )
        return RunInfo(found[0], found[1], rank, None)
    # The launch beside it: one look, no wait (contract 2). Provisional: read again after this
    # rank's first step (`LaunchRead`).
    launch = _look_for_launch(store, launch_key)
    if run_id and str(run_id) != str(shared_id):
        _say(
            f"tensorquill.init on rank {rank} was given run_id={run_id!r}, and rank 0's run ID is {shared_id!r}: "
            "under torch.distributed every rank writes under rank 0's. Pass the same run_id= on every rank.",
            stacklevel=4,
        )
    return RunInfo(shared_id, source, rank, launch, (store, launch_key))


def _brief(error: BaseException) -> str:
    try:
        return f"{type(error).__name__}: {error}"[:200]
    except Exception:  # noqa: BLE001  an error whose str() raises
        return type(error).__name__


def _say(message: str, stacklevel: int = 2) -> None:
    """`warnings.warn(message, RuntimeWarning)`, which never raises: under `python -W error` (or a
    filter that makes a RuntimeWarning an error) the message is logged on the `tensorquill`
    logger instead (stderr, through logging's last-resort handler, when no handler is set), so
    saying that something was passed over never becomes the failure it reports. `stacklevel`
    counts from the caller, as `warnings.warn`'s does."""
    try:
        warnings.warn(message, RuntimeWarning, stacklevel=stacklevel + 1)
    except Exception:  # noqa: BLE001  nothing here may stop a run
        try:
            logging.getLogger("tensorquill").warning("%s", message)
        except Exception:  # noqa: BLE001
            pass


def _resolve_here(rank: int, run_id: str | None = None, *, stacklevel: int = 5) -> tuple[str, str]:
    """(run ID, source) by the whole order, on this rank alone: what one process resolves, what
    rank 0 shares, and what another rank falls back to. Never raises: the other ranks may be
    waiting for this one, so a tracker that fails is passed over, with one warning.
    `stacklevel`: the warning's, counted from this function."""
    if run_id:
        return run_id, "explicit"
    try:
        found = tracker_run_id()
    except Exception as error:  # noqa: BLE001  the other ranks wait on this rank
        _say(
            f"tensorquill.init on rank {rank} could not read the tracker's run ({_brief(error)}), "
            "so the run ID is the launcher's or a generated one. Pass run_id= to name the run.",
            stacklevel=stacklevel,
        )
        found = None
    if found is not None:
        return found
    shared = _launcher_run_id()
    if shared is not None:
        return shared, "launcher"
    return generate_run_id(), "generated"


def _warn_tracker_unshared(found: tuple[str, str]) -> None:
    # Several ranks and no process group: this rank's tracker run cannot be shared, and a rank
    # that does not see it (a tracker usually runs on rank 0 only) takes another ID.
    tracker = {"mlflow": "MLflow", "wandb": "W&B"}[found[1]]
    _say(
        f"tensorquill.init on rank {_rank()} of {_world_size()} took the run ID {found[0]!r} from the {tracker} "
        "run, and torch.distributed is not initialized, so it cannot be shared: a rank that does not see that "
        "run writes under another ID. Call tensorquill.init after init_process_group, or pass the same run_id= "
        "on every rank.",
        stacklevel=4,
    )


def tracker_run_id() -> tuple[str, str] | None:
    """The tracker run this process can see, as (run ID, source), with no collective call.

    A tracker's run often exists on rank 0 only, so a caller with several ranks should
    read it there and share it, as `resolve_run` does under `torch.distributed`. A tracker
    that fails while it is read raises here; `resolve_run` passes it over.
    """
    env = os.environ.get("MLFLOW_RUN_ID")
    if env:
        return env, "env"
    mlflow = sys.modules.get("mlflow")
    if mlflow is not None:
        active = mlflow.active_run()
        if active is not None:
            return active.info.run_id, "mlflow"
    wandb = sys.modules.get("wandb")
    run = getattr(wandb, "run", None)
    if run is not None:
        # Read the ID once. Never read `wandb.run.step`: it is a round trip to the W&B
        # background process and raises in shared-mode runs.
        return run.id, "wandb"
    return None


def tracker_labels() -> dict[str, str]:
    """The project, run name and group of the tracker run this process can see, with no
    collective call and no round trip: any of `project`, `name` and `group` that is known.

    W&B's `run.project`, `run.name` and `run.group` each return a field of the run's local
    settings (`wandb_run.py`). MLflow's run name is on the active run object; its experiment
    name is read from `MLFLOW_EXPERIMENT_NAME` only, because looking it up by the run's
    experiment ID is a call to the tracking server, and nothing here may wait on one.
    Any failure to read a label leaves it out.
    """
    found: dict[str, str] = {}

    def take(key: str, read: Any) -> None:
        if key in found:
            return
        try:
            value = read()
        except Exception:  # noqa: BLE001  a tracker's attribute must never stop a run
            return
        if isinstance(value, str) and value:
            found[key] = value

    mlflow = sys.modules.get("mlflow")
    if mlflow is not None:
        try:
            active = mlflow.active_run()
        except Exception:  # noqa: BLE001
            active = None
        if active is not None:
            take("name", lambda: active.info.run_name)
            take("project", lambda: os.environ.get("MLFLOW_EXPERIMENT_NAME"))
    try:
        run = getattr(sys.modules.get("wandb"), "run", None)
    except Exception:  # noqa: BLE001  a module whose `run` raises
        run = None
    if run is not None:
        take("project", lambda: run.project)
        take("name", lambda: run.name)
        take("group", lambda: run.group)
    return found


def _launcher_run_id() -> str | None:
    """A job ID every rank of the job sees: torchrun's rendezvous ID (a fresh UUID with
    `--standalone`, "none" unless set otherwise), or SLURM's job and step IDs."""
    elastic = os.environ.get("TORCHELASTIC_RUN_ID")
    if elastic and elastic != "none":
        return f"run-{elastic}"
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        step = os.environ.get("SLURM_STEP_ID")
        return f"run-slurm{job}" + (f".{step}" if step else "")
    return None


def generate_run_id() -> str:
    """`run-<UTC time>-<6 hex digits>`, sortable by start time, as W&B generates run IDs
    when none is given (`wandb/sdk/lib/runid.py`)."""
    return f"run-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def _world_size() -> int:
    dist = sys.modules.get("torch.distributed")
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    value = os.environ.get("WORLD_SIZE")
    return int(value) if value is not None and value.isdigit() and int(value) > 0 else 1


def _warn_generated_unshared(run_id: str) -> None:
    # Reached only without a process group of several ranks (`resolve_run` shares rank 0's ID
    # under one), so nothing here can share the ID. Several ranks, nothing to share an ID
    # through yet: each rank picks its own, so the job is written as one run per rank, each
    # missing the others, and this says so.
    _say(
        f"tensorquill.init on rank {_rank()} of {_world_size()} found no run ID the ranks share "
        "(no tracker run, TORCHELASTIC_RUN_ID or SLURM_JOB_ID) and torch.distributed is not "
        f"initialized, so this rank's run is {run_id!r} and the other ranks' runs have other IDs. "
        "Pass the same run_id= on every rank, or call tensorquill.init after init_process_group.",
        stacklevel=4,
    )
