"""Read PyTorch's Flight Recorder dumps and say which collective is stuck.

PyTorch already records every collective each rank issues (`TORCH_FR_BUFFER_SIZE`, formerly
`TORCH_NCCL_TRACE_BUFFER_SIZE`), dumps the buffers when a collective times out
(`TORCH_FR_DUMP_ON_TIMEOUT`, formerly `TORCH_NCCL_DUMP_ON_TIMEOUT`) to one pickle per rank
named `<TORCH_FR_DUMP_TEMP_FILE><rank>`, and ships an analyzer that matches the ranks'
collectives against each other (`python -m torch.distributed.flight_recorder.fr_trace`).
TensorQuill does not trace collectives itself: it runs that analyzer over the dump
directory and turns its first problem into an incident, next to the step records that
show how far training got.

The analyzer ships with PyTorch 2.10 and later and its database format may change between
releases ("This script is versioned", it says), so a version that cannot be read is
reported as not checked, not as an error. The analyzer's own documented limits apply: it
does not judge the status of collectives beyond what is described here, and it cannot
match collectives confidently when a buffer does not start at the first collective.
"""

from __future__ import annotations

import contextlib
import logging
import pickle
import threading
from pathlib import Path
from typing import Any, Iterator

# The name of the logger the analyzer reports through (`fr_logger.FlightRecorderLogger`).
_LOGGER_NAME = "Flight Recorder"

# The threads running an analysis now, each with the list its messages go to. One filter on the
# analyzer's logger (`_TAKE`) takes their records, and only theirs, while any of them runs.
_lock = threading.Lock()
_taking: dict[int, list[str]] = {}
_users: dict[int, int] = {}  # id(logger) -> how many analyses are taking its messages now
_lowered: dict[int, int] = {}  # id(logger) -> its level before an analysis lowered it to INFO


class _TakeThreads(logging.Filter):
    """A logger filter that keeps the records of a thread running an analysis, at INFO and above,
    in that analysis's list and stops them there, so none reaches a handler or the root logger. A
    filter runs on the thread that logs, before any handler, so it can tell the analyses apart; the
    records of every other thread go on as if it were not there."""

    def filter(self, record: logging.LogRecord) -> bool:
        messages = _taking.get(threading.get_ident())
        if messages is None:
            return True
        if record.levelno >= logging.INFO:
            try:
                messages.append(record.getMessage())
            except Exception:  # noqa: BLE001  a message that cannot be formatted is left out
                pass
        return False


_TAKE = _TakeThreads()


@contextlib.contextmanager
def _messages_of_this_thread(logger: logging.Logger, *, lower: bool = True) -> Iterator[list[str]]:
    """The messages `logger` gets from this thread while the block runs, kept from its handlers.

    The logger's handler list and `propagate` are never touched, so two analyses in two threads,
    or other code adding a handler meanwhile, cannot undo each other. `_TAKE` is added to the
    logger when the first analysis on it starts and removed when the last one ends. The analyzer's culprit sentence is
    logged at INFO: if the logger is set above INFO, it is lowered to INFO while any analysis runs
    and set back when the last one ends, unless other code has set another level meanwhile; while
    it is lowered, another thread's INFO records on that logger reach its handlers too. `lower=False`
    leaves the level alone: around the analyzer's first import, which sets the logger to INFO itself
    and would otherwise be undone at the end of the block."""
    messages: list[str] = []
    ident, key = threading.get_ident(), id(logger)
    with _lock:
        if not _users.get(key):
            logger.addFilter(_TAKE)
            if lower and not logger.isEnabledFor(logging.INFO):
                _lowered[key] = logger.level
                logger.setLevel(logging.INFO)
        _users[key] = _users.get(key, 0) + 1
        outer = _taking.get(ident)
        _taking[ident] = messages
    try:
        yield messages
    finally:
        with _lock:
            if outer is None:
                _taking.pop(ident, None)
            else:
                _taking[ident] = outer
            _users[key] -= 1
            if not _users[key]:
                del _users[key]
                logger.removeFilter(_TAKE)
                level = _lowered.pop(key, None)
                if level is not None and logger.level == logging.INFO:
                    logger.setLevel(level)


def _by_rank(item: tuple[str, Any]) -> tuple[int, int, str]:
    """Sort key for the loader's `{file name: dump}`: the dump's rank, then the name."""
    name, dump = item
    try:
        return (0, int(dump["rank"]), name)
    except (KeyError, TypeError, ValueError):
        return (1, 0, name)  # a dump with no readable rank goes last, by name


def analyze_flight_recorder(trace_dir: str | Path, *, prefix: str | None = None) -> dict[str, Any]:
    """Run PyTorch's Flight Recorder analyzer over the per-rank dumps in `trace_dir`.

    `prefix` is the common start of the dump file names (`<prefix><rank>`); without it
    PyTorch infers one. Raises `ImportError` when this PyTorch has no analyzer and
    `FileNotFoundError` or `ValueError` when there is nothing readable in the directory, or when
    PyTorch's own analyzer fails on what it read.
    Returns the facts about the first problem, or `problem: None` when every collective
    that was recorded matched and completed. A partial set of dumps is analyzed as far as
    it goes (`incomplete_dumps` says so).
    """
    try:
        # The analyzer logs a note at import when `tabulate` is not installed, on the logger it then
        # reports through; taken like its other messages (`_messages_of_this_thread`). Not by swapping
        # `sys.stdout` and `sys.stderr`: they are the whole process's, so two analyses at once could
        # leave either one a dead buffer, and the analyzer's own console handler, made at that import,
        # would write into the buffer for good.
        with _messages_of_this_thread(logging.getLogger(_LOGGER_NAME), lower=False):
            from torch.distributed.flight_recorder.components.builder import build_db
            from torch.distributed.flight_recorder.components.config_manager import JobConfig
            from torch.distributed.flight_recorder.components.fr_logger import FlightRecorderLogger
            from torch.distributed.flight_recorder.components.loader import read_dir
    except ImportError as error:
        raise ImportError(
            "this PyTorch has no Flight Recorder analyzer (torch.distributed.flight_recorder, "
            "PyTorch 2.10 or later)"
        ) from error
    directory = Path(trace_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a directory of Flight Recorder dumps")
    arguments = [str(directory), "--allow-incomplete-ranks"] + (["-p", prefix] if prefix else [])
    config = JobConfig().parse_args(arguments)
    try:
        details, version = read_dir(config)
    except (AssertionError, EOFError, OSError, ValueError, pickle.UnpicklingError) as error:
        raise ValueError(f"no readable Flight Recorder dumps in {directory}: {error}") from error
    # The loader lists the dumps in the order the file system gives (`os.walk`), and the
    # analyzer takes the first rank it reads as the reference: where two ranks disagree, the
    # one it calls the culprit, and the collective it names, followed that order, which
    # differs between machines. In rank order the same dumps give the same answer everywhere.
    # It is no more right for that: with two ranks that disagree, nothing says which is at fault.
    details = dict(sorted(details.items(), key=_by_rank))

    # The analyzer reports what it finds through a logger that prints to the console; take this
    # thread's messages instead of printing them, leaving the logger's handlers alone.
    with _messages_of_this_thread(FlightRecorderLogger().logger) as messages:
        try:
            db = build_db(details, config, version)
        except Exception as error:  # noqa: BLE001  PyTorch's analyzer, on dumps it read and cannot put together
            # Its own errors are not this reader's to let through: PyTorch 2.15's Gloo dumps, for one,
            # name the process group in their entries and not in `pg_config`, and `build_db` raises
            # `KeyError: ('0', 2)` (fixed upstream after 2.15). A reader given such dumps says they were
            # not checked, and why (wave AI, Nov 22, 2026; until then the error escaped as a traceback).
            raise ValueError(
                f"PyTorch's Flight Recorder analyzer could not put the dumps in {directory} together"
                f" ({type(error).__name__}: {error}); they are not read"
            ) from error

    dumped = sorted(int(d["rank"]) for d in details.values())
    groups = {g.id: g for g in db.groups}
    members: dict[str, set[int]] = {}
    for membership in db.memberships:
        members.setdefault(membership.group_id, set()).add(int(membership.global_rank))

    def seq(collective: Any) -> int | str:
        value = collective.collective_seq_id
        return int(value) if str(value).isdigit() else value

    def kind_of(collective: Any) -> str | None:
        if not collective.pass_check:
            return "desync"
        if collective.missing_ranks or str(collective.collective_state) != "completed":
            return "hang"
        return None

    # Each rank's buffer is a ring of the last N entries (2000 by default) and ranks dump at
    # different moments, so the oldest collectives of one rank may have been overwritten on
    # another. The analyzer then calls them unmatched. A collective older than the newest
    # first entry of any rank in its group cannot be judged and is left out.
    edge: dict[str, int] = {}
    for dump in details.values():
        firsts: dict[str, int] = {}
        for entry in dump["entries"]:
            group_name, number = str(entry["process_group"][0]), str(entry.get("collective_seq_id", ""))
            if not entry.get("is_p2p") and number.isdigit():
                firsts[group_name] = min(firsts.get(group_name, int(number)), int(number))
        for group_name, first_seq in firsts.items():
            edge[group_name] = max(edge.get(group_name, 0), first_seq)

    def judgeable(collective: Any) -> bool:
        number = seq(collective)
        return not isinstance(number, int) or number >= edge.get(str(collective.group_id), 0)

    problems = [(kind_of(c), c) for c in db.collectives]
    flagged = [(kind, c) for kind, c in problems if kind is not None]
    problems = [(kind, c) for kind, c in flagged if judgeable(c)]
    result: dict[str, Any] = {
        "version": version,
        "dumped_ranks": dumped,
        "groups": len(db.groups),
        "collectives": len(db.collectives),
        "newest_dump_time": max(
            (p.stat().st_mtime for p in directory.rglob("*") if p.is_file() and p.name in details), default=0.0
        ),
        "problem_count": len(problems),
        "ignored_before_buffer_start": len(flagged) - len(problems),
        "incomplete_dumps": any(set(group) - set(dumped) for group in members.values()),
        "problem": None,
    }
    if not problems:
        return result
    kind, first = problems[0]
    group = groups.get(first.group_id)
    expected = sorted(int(r) for r in first.expected_ranks)
    missing = sorted(int(r) for r in (first.missing_ranks or ()))
    # Each line of a report is its own log record; a report starts at "sequence number: N".
    blocks: list[list[str]] = []
    for message in messages:
        if "sequence number:" in message or not blocks:
            blocks.append([])
        blocks[-1].append(message)
    detail = next(
        (
            "\n".join(block)
            for block in blocks
            if f"sequence number: {first.collective_seq_id}" in block[0] and f"group info: {first.group_id}:" in "\n".join(block)
        ),
        None,
    )
    culprit = None
    if detail and "Culprit rank" in detail:
        culprit = detail[detail.index("Culprit rank"):].splitlines()[0][:300]
    result["problem"] = {
        "type": kind,
        "group": {"id": first.group_id, "desc": group.desc if group else first.pg_desc, "size": group.size if group else None},
        "collective": str(first.collective_name).split(":")[-1],
        "collective_seq_id": seq(first),
        "state": str(first.collective_state),
        "expected_ranks": expected,
        "missing_ranks": missing,
        "all_ranks_entered": not missing,
        "buffer_start": edge.get(str(first.group_id), 0),
        "mismatch": culprit,
    }
    return result


def flight_recorder_incident(analysis: dict[str, Any], *, step: int) -> dict[str, Any] | None:
    """The incident for an analysis, or None when nothing was stuck or mismatched.

    `step` is the last step TensorQuill recorded: the dumps do not know the training
    step, and a stuck collective is one the job reached after its last recorded step.
    """
    problem = analysis.get("problem")
    if not problem:
        return None
    return {
        "kind": "collective_desync" if problem["type"] == "desync" else "collective_hang",
        "step": step,
        "problems": analysis["problem_count"],
        "dumped_ranks": analysis["dumped_ranks"],
        "incomplete_dumps": analysis["incomplete_dumps"],
        **problem,
    }
