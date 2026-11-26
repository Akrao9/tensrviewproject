"""Where captured records go. The capture path only ever calls `write` from a worker thread."""

from __future__ import annotations

import errno
import glob
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from .manifest import _say

try:  # POSIX only; where there is none, a records file is written unlocked
    import fcntl as _fcntl
except ImportError:  # pragma: no cover  Windows
    _fcntl = None  # type: ignore[assignment]


class Sink(Protocol):
    def write(self, records: list[dict[str, Any]]) -> None: ...

    def close(self) -> None: ...


class MemorySink:
    """Keeps records in a list. For tests and notebooks."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def write(self, records: list[dict[str, Any]]) -> None:
        with self._lock:
            self.records.extend(records)

    def close(self) -> None:
        pass


def _finite(value: Any) -> Any:
    """JSON has no NaN or infinity, and an overflow is exactly what must not be lost."""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def _brief(error: BaseException | None) -> str:
    try:
        return f"{type(error).__name__}: {error}"[:200]
    except Exception:  # noqa: BLE001  an error whose str() raises
        return type(error).__name__


def _ends_torn(path: Path) -> bool:
    """Whether `path` is a non-empty file whose last byte is not a newline: a line that a killed
    process left half written. Never raises: a file that cannot be read is opened as before."""
    try:
        with path.open("rb") as file:
            file.seek(0, os.SEEK_END)
            if file.tell() == 0:
                return False
            file.seek(-1, os.SEEK_END)
            return file.read(1) != b"\n"
    except OSError:
        return False


# --- The records file's name ---------------------------------------------------------------
#
# `init` writes `<run_id>.rank<N>.jsonl`. A process that finds that file locked by another writer
# of the same run ID and rank writes `<run_id>.rank<N>.<pid>.jsonl` (`JsonlSink`, `lock=True`). A run
# ID may hold dots (`run-slurm1234.0`), so the name is read from the right, and the rank and the
# pid are ASCII digits only: `x.rank0.bak.jsonl`, a copy, is no records file of `init`'s.

RECORDS_SUFFIX = ".jsonl"
_RECORDS_NAME = re.compile(r"^(?P<run_id>.+)\.rank(?P<rank>[0-9]+)(?:\.(?P<pid>[0-9]+))?\.jsonl$")


def _descriptor_ends_torn(file: Any) -> bool:
    """`_ends_torn` read through `file`'s own descriptor (opened for reading too), so that no other
    descriptor of the file is opened and closed in this process: where `flock` is a POSIX record
    lock in disguise (Linux's NFS client emulates it as one), that close would drop the lock. Never
    raises: a file that cannot be read is opened as before."""
    try:
        file.flush()
        fd = file.fileno()
        size = os.fstat(fd).st_size
        if size == 0:
            return False
        return os.pread(fd, 1, size - 1) != b"\n"
    except (OSError, AttributeError, ValueError):
        return False


def records_file_name(run_id: str, rank: int, pid: int | None = None) -> str:
    """`<run_id>.rank<rank>.jsonl`, or `<run_id>.rank<rank>.<pid>.jsonl` for a second writer."""
    return f"{run_id}.rank{rank}" + (f".{pid}" if pid is not None else "") + RECORDS_SUFFIX


def records_file_parts(path: str | Path) -> tuple[str, int, int | None] | None:
    """(run ID, rank, pid) of a file named as `records_file_name` names one, the pid None for the
    first writer's; None for any other name."""
    found = _RECORDS_NAME.fullmatch(Path(path).name)
    if found is None:
        return None
    pid = found.group("pid")
    return found.group("run_id"), int(found.group("rank")), None if pid is None else int(pid)


def writers_of(path: str | Path) -> list[Path]:
    """Every file in `path`'s folder that `init` wrote for the same run ID and rank as `path`: the
    first writer's `<run_id>.rank<N>.jsonl` first, when it is there, then each second writer's
    `<run_id>.rank<N>.<pid>.jsonl` by pid. `[path]` for a file of another name, and for one whose
    folder cannot be read. Never raises."""
    path = Path(path)
    parts = records_file_parts(path)
    if parts is None:
        return [path]
    run_id, rank, _ = parts
    try:
        names = glob.glob(glob.escape(str(path.parent / f"{run_id}.rank{rank}")) + "*" + RECORDS_SUFFIX)
    except Exception:  # noqa: BLE001
        return [path]
    found: list[tuple[int, Path]] = []
    for name in names:
        theirs = records_file_parts(name)
        if theirs is not None and theirs[:2] == (run_id, rank) and Path(name).is_file():
            found.append((-1 if theirs[2] is None else theirs[2], path.parent / Path(name).name))
    found.sort()
    files = [file for _, file in found]
    return files if path in files else [path, *files]


def writers_block(path: str | Path) -> dict[str, Any] | None:
    """What a reader of one records file says when its rank has more than one writer's file beside
    it (`writers_of`): `{"state": "several", "files": [...names...], "detail": ...}`; None when
    `path` is its rank's only file. Two processes wrote the same run ID and rank at once: each file
    holds one process's records, and neither is the rank's whole."""
    files = writers_of(path)
    if len(files) < 2:
        return None
    names = [file.name for file in files]
    return {
        "state": "several",
        "files": names,
        "detail": (f"{len(names)} processes wrote the same run ID and rank at once, each in a file of its own "
                   f"({', '.join(names)}): {Path(path).name} holds one of them, not the rank's whole. Give each "
                   "process its own rank (start it under torchrun or srun, or set RANK and WORLD_SIZE) or its own "
                   "run_id="),
    }


def _try_lock(file: Any) -> tuple[str, BaseException | None]:
    """An exclusive advisory lock on `file`'s descriptor, never waited for: `("held", None)`,
    `("busy", None)` when another open of the file holds it (another process, or another
    session here), `("none", None)` where there is no `fcntl`, and `("unavailable", error)` when
    the lock failed for another reason (a file system without locking). `flock`, not
    `lockf`: a POSIX record lock is dropped when any descriptor of the file in this process is
    closed, and the torn-line check and the stream sender both open the file. Never raises."""
    if _fcntl is None:
        return "none", None
    try:
        _fcntl.flock(file.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        return "held", None
    except OSError as error:
        if isinstance(error, BlockingIOError) or error.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            return "busy", None
        return "unavailable", error
    except Exception as error:  # noqa: BLE001  nothing here may stop a run
        return "unavailable", error


class JsonlSink:
    """One JSON object per captured step. Non-finite numbers are written as the strings
    "nan", "inf" and "-inf" so the file stays standard JSON.

    With `lock=True` (`init`'s default sink) the sink holds an exclusive advisory lock (`flock`)
    on its file until it is closed, taken without waiting. The file is opened for appending on
    purpose, so a run started again under its run ID appends to it, and the lock is what tells a
    second writer apart: a process (or a second session in this one) that finds the file locked
    writes `<name>.<pid>.jsonl` beside it instead (`records_file_name`, `writers_of`), and says so
    once. A lock that fails for another reason (a file system without locking) is no lock here:
    the sink writes its file unlocked and says so once. Where Python has no `fcntl` (Windows) the
    file is written unlocked and nothing is said. Opening never waits and never raises for any of
    this. `path` is the file written; `locked` whether the lock is held; `created` whether this
    sink made the file. A sink made with the default `lock=False` takes no lock, as before.

    A file whose last line was left half written (the process that wrote it was killed in the
    middle of a step) is first given the newline that line lacks, so the first new step starts a
    line of its own and is read, by the stream sender and by a plain line-by-line reader. The
    fragment stays where it was, as a line that is not a whole JSON object: incomplete history
    that a reader skips. Nothing is removed or moved, so the file is only ever appended to and
    the stream sender's saved offset stays valid. A fragment that happens to be a whole object
    (the process died just before the newline) becomes a whole line again. With `lock=True` the
    newline is written only once the lock is held, so it never cuts a line another writer is
    still writing; a file written unlocked is repaired as before, and if another process is
    still writing that line, the newline cuts it in two and its step is lost."""

    def __init__(self, path: str | Path, *, lock: bool = False) -> None:
        self.path = Path(path)
        self.locked = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.created = not self.path.exists()
        # Locked, the file is opened for reading too, so the torn-line checks read it through this
        # descriptor and no other is opened (`_descriptor_ends_torn`).
        self._mode = "a+" if lock and hasattr(os, "pread") else "a"
        self._file = self._open(self.path)
        self._failed = False  # the last write raised: the file may end in part of a line
        repair = True
        if lock:
            repair = self._take_lock()
        if repair and self._ends_torn():
            self._file.write("\n")
            self._file.flush()

    def _open(self, path: Path) -> Any:
        """`path` opened for appending, and for reading too when locked; a file this process may
        write and not read is opened for appending alone, and read by its path (`_ends_torn`)."""
        if self._mode == "a+":
            try:
                return path.open("a+", encoding="utf-8")
            except PermissionError:
                self._mode = "a"
        return path.open("a", encoding="utf-8")

    def _ends_torn(self) -> bool:
        return _descriptor_ends_torn(self._file) if self._mode == "a+" else _ends_torn(self.path)

    def _take_lock(self) -> bool:
        """Lock the file, or move to a second writer's file of its own (`<name>.<pid>.jsonl`) when
        another writer holds it, and say what was done when it is not the plain case. Whether the
        torn-line repair may be made on the file now open: not on one another writer holds.
        Never raises: a failure here leaves the file open unlocked."""
        try:
            state, error = _try_lock(self._file)
            if state == "held":
                self.locked = True
                return True
            if state == "none":
                return True
            if state == "unavailable":
                _say(f"tensorquill.init could not lock {self.path} ({_brief(error)}): the file system may not "
                     "support locks, so this session writes it unlocked, and another process writing the same run "
                     "ID and rank into it would not be noticed.", stacklevel=5)
                return True
            first = self.path
            second = first.with_name(first.name[: -len(RECORDS_SUFFIX)] + f".{os.getpid()}{RECORDS_SUFFIX}"
                                     if first.name.endswith(RECORDS_SUFFIX) else f"{first.name}.{os.getpid()}")
            created = not second.exists()
            file = self._open(second)
        except Exception as failure:  # noqa: BLE001  nothing here may stop a run
            _say(f"tensorquill.init could not take {self.path}'s lock or open a file of its own beside it "
                 f"({_brief(failure)}), so this session writes {self.path.name} unlocked.", stacklevel=5)
            return False
        self._file.close()
        self._file, self.path, self.created = file, second, created
        state, error = _try_lock(file)
        self.locked = state == "held"
        if state == "busy":  # a third writer of this run ID and rank in this one process
            _say(f"tensorquill.init found {first.name} and {second.name} both held by other writers of the same "
                 f"run ID and rank (the second by another session in this process), so this session writes "
                 f"{second.name} unlocked beside the session there. Give each session its own run_id=.",
                 stacklevel=5)
            return False
        _say(f"tensorquill.init found {first} held by another writer of the same run ID and rank, so this "
             f"process writes {second.name} beside it: two processes are writing one rank, and the readers say so. "
             "Give each process its own rank (start it under torchrun or srun, or set RANK and WORLD_SIZE) or its "
             "own run_id=.", stacklevel=5)
        return True

    def write(self, records: list[dict[str, Any]]) -> None:
        """Write `records`, one line each, and flush. A write that raises (a full disk, a file-size
        limit, a value JSON cannot hold) may leave part of a line in the file. The next write on
        this sink first flushes what Python kept of it, which may complete that line, and then, if
        the file still ends in part of a line, starts a line of its own: so only the records of the
        write that raised can be lost, never those of a write that returned. Before (lane AD4,
        measured with a file-size limit), a record larger than Python's 8 KiB buffer lost its unwritten
        rest at the failure, and the next record written was appended onto the fragment: a write
        that returned, whose record no reader could read."""
        if self._failed:
            self._file.flush()
            if self._ends_torn():
                self._file.write("\n")
            self._failed = False
        try:
            for record in records:
                self._file.write(json.dumps(_finite(record), separators=(",", ":"), allow_nan=False) + "\n")
            self._file.flush()
        except BaseException:
            self._failed = True
            raise

    def close(self) -> None:
        self._file.close()


# --- The end marker ----------------------------------------------------------------------
#
# A `.jsonl` file looks the same whether its job is still writing, was killed, or finished.
# `Session.close` leaves `<file>.end.json` beside it once the file is closed and complete, so a
# reader can tell a run that closed from one that stopped. It is a file of its own, not a line
# in the records: the stream sender and every reader of the records read each line as a step.
# A session that opens the file removes it first (a run started again under its run ID appends
# to the same file), and the marker holds the file's size at close, so a marker that a later
# process left behind is not taken for this file's end either. As in MLflow and W&B, a session
# closed by an exception (`with session:`) says so: `outcome` is "raised", with the type.

END_SUFFIX = ".end.json"
END_MARKER = "tensorquill_run_end"


def end_marker_path(records_file: str | Path) -> Path:
    """`<name>.end.json` beside `<name>.jsonl`."""
    path = Path(records_file)
    return path.with_name((path.name[: -len(".jsonl")] if path.name.endswith(".jsonl") else path.name) + END_SUFFIX)


def clear_end_marker(records_file: str | Path) -> None:
    """Called as a session opens the file: from now on the file is being written again."""
    try:
        end_marker_path(records_file).unlink(missing_ok=True)
    except OSError:
        pass  # a marker that cannot be removed is caught by its size: the file grows past it


def write_end_marker(
    records_file: str | Path, *, run_id: Any, rank: Any, attempt: Any, exception: BaseException | None = None,
) -> bool:
    """Write the marker by a temporary file and a rename, after the file is closed. Never raises:
    a run that finished is not failed for want of its marker, which then reads as not closed.
    The failure is said by `manifest._say` (lane Z6, L23), so under `python -W error` it is
    logged on the `tensorquill` logger rather than raised and lost; a scratch file the failed
    write or rename left is removed, and a removal that fails is let be."""
    records = Path(records_file)
    path = end_marker_path(records)
    scratch: Path | None = None
    try:
        document = {
            "marker": END_MARKER,
            "version": 1,
            "run_id": run_id,
            "rank": rank,
            "attempt": attempt,
            "closed_at": time.time(),
            "records_bytes": records.stat().st_size,
            "outcome": "closed" if exception is None else "raised",
            **({"exception": type(exception).__name__} if exception is not None else {}),
        }
        scratch = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        scratch.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        os.replace(scratch, path)
        return True
    except Exception as error:  # noqa: BLE001  nothing here may fail a finished run
        _remove_scratch(scratch)
        # Three frames up, as before: this function, `Session.close`, the caller of `close`.
        _say(f"tensorquill could not write the end marker {path} ({error}); the run reads as not closed.",
             stacklevel=3)
        return False


def _remove_scratch(scratch: Path | None) -> None:
    """Remove a scratch file a failed write or rename left behind. Never raises."""
    if scratch is None:
        return
    try:
        scratch.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001  a scratch file that stays is untidy, never a failure
        pass


def read_end_marker(records_file: str | Path) -> dict[str, Any]:
    """Whether the session that wrote `records_file` closed it: `{"state": ...}` with

    - `closed`: a marker matches the file (`outcome` "closed", or "raised" with `exception`
      when the session was closed by an exception), with `closed_at` and `attempt`;
    - `not_closed`: no marker. The job may still be running, or was killed, or never called
      `close()`, or wrote the file before markers existed: this cannot tell which;
    - `written_after_close`: a marker is there and the file is larger than when it was written,
      so something wrote to it after the close (a process that did not remove the marker);
    - `unreadable`: a marker that is not one, read as no marker; or a marker whose records file
      cannot be read (it was deleted or moved), so nothing says the file is the one that closed.
    Every state but `closed` means "not known to have finished"."""
    records = Path(records_file)
    path = end_marker_path(records)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"state": "not_closed", "detail": (
            "no end marker beside the file: the job may still be running, or it was killed or never called"
            " session.close() (or wrote this file before end markers existed); these look the same here")}
    except (OSError, ValueError) as error:
        return {"state": "unreadable", "path": str(path), "detail": f"the end marker cannot be read ({error}), so the"
                " run is not known to have closed"}
    size = document.get("records_bytes") if isinstance(document, dict) else None
    if (not isinstance(document, dict) or document.get("marker") != END_MARKER
            or not isinstance(size, int) or isinstance(size, bool)):
        return {"state": "unreadable", "path": str(path),
                "detail": "the end marker is not one TensorQuill wrote, so the run is not known to have closed"}
    try:
        now = records.stat().st_size
    except OSError as error:
        return {"state": "unreadable", "path": str(records), "detail": f"an end marker is there but {records.name}"
                f" cannot be read ({error}), so the run is not known to have closed"}
    if now != size:
        return {"state": "written_after_close", "path": str(path), "records_bytes": size, "now_bytes": now,
                "detail": f"{records.name} was closed at {size:,} bytes and is {now:,} now: written after the close,"
                          " so the run is not known to have finished"}
    found: dict[str, Any] = {"state": "closed", "outcome": document.get("outcome", "closed")}
    for key in ("exception", "closed_at", "attempt"):
        if document.get(key) is not None:
            found[key] = document[key]
    return found
