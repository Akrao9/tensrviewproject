"""Where captured records go. The capture path only ever calls `write` from a worker thread."""

from __future__ import annotations

import json
import math
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Protocol


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


class JsonlSink:
    """One JSON object per captured step. Non-finite numbers are written as the strings
    "nan", "inf" and "-inf" so the file stays standard JSON.

    A file whose last line was left half written (the process that wrote it was killed in the
    middle of a step) is first given the newline that line lacks, so the first new step starts a
    line of its own and is read, by the stream sender and by a plain line-by-line reader. The
    fragment stays where it was, as a line that is not a whole JSON object: incomplete history
    that a reader skips. Nothing is removed or moved, so the file is only ever appended to and
    the stream sender's saved offset stays valid. A fragment that happens to be a whole object
    (the process died just before the newline) becomes a whole line again. If another process
    is still writing that line, the newline cuts it in two and its step is lost: two writers
    appending to one file interleave their lines anyway, and the sink takes no lock."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torn = _ends_torn(self.path)
        self._file = self.path.open("a", encoding="utf-8")
        if torn:
            self._file.write("\n")
            self._file.flush()

    def write(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            self._file.write(json.dumps(_finite(record), separators=(",", ":"), allow_nan=False) + "\n")
        self._file.flush()

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
    a run that finished is not failed for want of its marker, which then reads as not closed."""
    records = Path(records_file)
    path = end_marker_path(records)
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
        warnings.warn(f"tensorquill could not write the end marker {path} ({error}); the run reads as not closed.",
                      RuntimeWarning, stacklevel=3)
        return False


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
