"""List the runs in a folder of local files or in a server's store, by project.

    python -m tensorquill runs <folder or store> [--project P] [--json]

    import tensorquill
    tensorquill.list_runs("tensorquill-out", project="granite")

One row a run: its name, run ID, run state, how many ranks, last step, incidents by class, the
first incident, how many kinds were checked, and its start time. For a run of several ranks
the row is the whole run's, by the all-ranks view's rules: the incidents of every rank whose
report or evidence file is current, and a kind counted as checked only when every such rank
checked it. A rank without one is unknown (the run state says so), never clean. A run is
found by its files:

- local files, as `tensorquill.init` writes them: `<run_id>.rank<N>.jsonl`, with the manifest
  `<run_id>.run.json` beside them and, once `python -m tensorquill <file>` has run, the
  evidence file `<file>.evidence.json` (or `<file>.evidence.compact.json`, which `--compact`
  writes; when both are there, the one written later is read, and the full one on a tie);
- a server's store: `runs/<run_id>/` with `run.json`, `watchdog.json` and a `rank=<n>` folder
  for each rank. `<folder or store>` is the folder that holds `runs/`, or `runs/` itself.

A run kept both as local files and in a store under one folder is one row, read from its local
files, and the row says where its store copy is (`also_in_store`). A rank whose records are in
several files (a restart that wrote a file of a name of its own) keeps them all: every record of
those files is read, as `compare` reads them, so its attempts are in contract 1's order (each
file's own order first, then `ts`, then number; never path order) and its last step and `end`
cell are the newest attempt's; its files are named, and it is not judged by the listing (an
evidence file is of one file).
A store run with no judged rank lists its incidents as not known (`?`), never as none.

The listing reads the manifest and the small evidence or report files only. It never opens a
Parquet part and never runs a detector, so a local run whose evidence file was never written
is listed with its state unknown and a note saying how to write it. A local run of several
ranks reads each rank's evidence file. Of a `.jsonl` file it
reads the first line and the last megabyte, every whole line of it (for the run ID, the start
time, the last step, and the ranks and attempts they name: about 8 ms a file at the rate below),
except the files of a rank whose records are in several files, which are read whole, line by line (lane X5, CLI-13, Nov 19, 2026: from their first and last records
alone, an attempt seen only in the middle of a file was not seen, and the order across the
files could be wrong). That costs a JSON parse of every line, about 120 MB a second on a
recorded capture here, and only for such a rank; what is kept is each attempt's first record in
each file and its steps, never the records. A file whose last megabyte shows a rank in more
than one attempt is read whole the same way, for that rank's last step (XC4, lane
Y3, Nov 20, 2026: a late row of an attempt that a later one replaced was taken as the last step).
The part of a file larger than a megabyte before its last megabyte is scanned for the attempts
its lines name, from each line's start and without parsing it (`_attempts_in_file`): when an
attempt newer than the last megabyte's is there, the file is read whole the same way (L21, lane
Z5, Nov 20, 2026: an attempt seen only in the middle of such a file was not seen, and a late row of
the attempt it replaced was taken as the last step). That scan reads every byte of each such file
at each listing, about 1.5 GB a second on the recorded captures here from the page cache, and as
slowly as a full parse for some files (the notes above `_RECORD_HEAD`);
within one process a file that only grew is scanned from where the last scan stopped.

A store's run whose server dropped rows (`dropped_rows` and `dropped_row_keys` on a rank's entry
of `watchdog.json`), for a null key or, since wave AA, for a `schema_version` it does not support
(`unsupported_schema_version`), has `dropped_rows` on its row, in all, by key and by rank, and
the printed listing a note. The count is an upper bound: a batch sent again after a lost
acknowledgement is counted again (L22, lane Z5). A run that dropped none has no such key.
Batches the server refused whole, because a key column was missing or of another type
(`refused_batches` and `refused_batch_keys` on a rank's entry, RS10, wave AA), are not dropped
rows: they are `refused_batches` on the row, in all, by column and by rank, with a note of their
own, and are never added to the rows dropped (lane AB5, Nov 20, 2026). A run that refused none
has no such key.

How many ranks: a file named `<run_id>.rank<N>.jsonl` holds the ranks its lines name. Every
line of it is read for that: the last megabyte parsed, and the part before it scanned for each
line's rank and attempt (`_attempts_in_file`, which runs anyway for L21), so a file over a
megabyte with no manifest is counted, not listed as "1+" (lane AA5, Nov 20, 2026; it was counted
from the manifest's `world_size` alone). Holding rank N alone, it is rank N's file; holding
others too (XC5), it is a file of several ranks, all of them named, so the count is still
settled, and its evidence file is one rank's of several. Only when the scan cannot read every
line (the file cut or unreadable while it is read) is the count the manifest's `world_size`,
and with none not known (`rank_count` None, with why), never a definite count that may be
wrong. A line torn by a killed writer with the next record written onto it is skipped by the
readers and read here by its start: when that start names a rank no whole line of the file
names, that rank is counted here and not by `compare` (the scan's one difference from a full
parse, which for the attempts could only add a needless whole read). A rank named only in the
part before the last megabyte has no last step in the row. A file with a name of its own may hold several ranks, and only reading
it whole would name them all, so its ranks are counted from its name, its first and last
records and the manifest's `world_size`: when the world size equals the ranks seen, that is the
count; otherwise `rank_count` is None (printed as the ranks seen and "+") and
`rank_count_not_known` says why. When every file holds its rank alone and the manifest's
`world_size` is above the files, the ranks without a file are counted and unknown, never
clean, as a store's ranks its status expects. A `world_size` above `MAX_EXPECTED_RANKS` is not
made into a list (XC6): the ranks without a file are a count, the row is unknown for them, and
`rank_count` is None, with why. The evidence file beside a file of several
ranks (its ends name two, or, for a lone file, the manifest's `world_size` is above 1) is one
rank's and does not say which, so it is given to no rank: its incidents are listed with no
rank, its ranks are unknown (the row is `ranks_unknown`), and what was checked is not counted
for them. When such a file is the run's only file and its evidence names the rank it judged
(`rank`, with `ranks_in_source` when that rank was picked out of the file: lane Z2), the row's
evidence cell names that rank and the ranks the file held (lane AA5, Nov 20, 2026); in a run of
several files the evidence of such a file is still given to no rank. The last step of such a
file is its furthest rank's, each rank's read from its last line in the last megabyte.
`compare` reads every rank of it.

A run without a manifest (every run recorded before the manifest existed) is listed under
`uncategorized` with its run ID as its name.

`compare` (`compare.py`) finds runs by ID or name through `find_run` here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import manifest as _manifest
from .sink import read_end_marker

DEFAULT_SOURCE = "tensorquill-out"  # where `tensorquill.init` writes without a sink
VERBS = ("runs", "compare")
_TAIL_BYTES = 1 << 20  # the most of a `.jsonl` file's end read for its last step
_HEAD_BYTES = 1 << 20


class RunLookupError(ValueError):
    """A run named by ID or name was not found, or cannot be read for a comparison."""


class AmbiguousRunName(RunLookupError):
    """A name fits several runs. Never guessed around: the message lists their IDs."""


# --- Finding the runs ---------------------------------------------------------------------


def _first_record(path: Path) -> dict[str, Any] | None:
    """The first line of a `.jsonl` file, when it is whole and within the first megabyte."""
    try:
        with path.open("rb") as file:
            line = file.readline(_HEAD_BYTES)
    except OSError:
        return None
    if not line.endswith(b"\n"):
        return None
    try:
        found = json.loads(line)
    except ValueError:
        return None
    return found if isinstance(found, dict) else None


def _tail(path: Path) -> tuple[list[bytes], bool]:
    """The whole lines of a `.jsonl` file's last megabyte, last line last, and whether that is
    the whole file (no line before them is left unread)."""
    lines, whole, _ = _tail_from(path)
    return lines, whole


def _tail_from(path: Path) -> tuple[list[bytes], bool, int]:
    """`_tail`, and the byte where its first line starts: every line that starts before it is
    in the part of the file the last megabyte leaves out."""
    try:
        size = path.stat().st_size
        with path.open("rb") as file:
            at = max(0, size - _TAIL_BYTES)
            file.seek(at)
            tail = file.read()
    except OSError:
        return [], False, 0
    lines = tail.split(b"\n")
    if size > _TAIL_BYTES:
        cut = len(lines[0]) + 1  # the first piece starts mid-line
        lines = lines[1:]
        at = at + cut if len(lines) > 1 else size  # no whole line in the window: all of it is before
    return lines[:-1], size <= _TAIL_BYTES, at  # what follows the last newline is not finished


# --- The attempts in the part of a file before its last megabyte (L21) ---------------------
#
# A rank's last line stands only when its attempt is the newest in the file, and an attempt seen
# only in the middle of a file larger than a megabyte was not seen (L21, lane Z5, Nov 20, 2026).
# The part before the last megabyte is read for the attempts its lines name, without parsing the
# lines: a record as the sink writes it starts with `schema_version`, `run_id`, `rank` and
# `attempt`, in that order (`json.dumps` of the record, with or without spaces), and those four
# are read from the line's start. Each line after it that starts with the same bytes is of the
# same rank, run and attempt, and one search in C passes over them all. A line that does not start
# so (another writer's order, no `attempt`, a value of another type) is parsed whole, as `compare`
# reads it, so the attempts found are the ones a full parse finds. The two differ only for a line
# that is not JSON whose start reads as a record (a killed process's half line with the next
# record written onto it): its attempt is counted here and not by a reader. That can only add an
# attempt, and so only make the file be read whole (slower, never wrong). A record that names
# `rank` or `attempt` twice is read by its first here and by its last by a reader; no writer of
# ours does that.
#
# Cost (measured here, the files in the page cache; from disk it is the disk's read rate): every
# byte of each file larger than a megabyte is read at each listing, about 1.5 GB a second on the
# recorded 6 to 17 MB captures (a full parse is about 120 MB a second) and 0.9 GB a second on a
# file of a million 114-byte lines of one rank. A file whose ranks change from one line to the
# next is stepped a line at a time: 8 or 2,000 ranks line by line, about 0.6 times a full parse's
# time; a new rank or attempt on nearly every line (100,000 ranks), about 1.3 times. Lines that
# start in another order are each parsed, at a full parse's rate. Within one process (the MCP server lists the
# runs at each call) a file that only grew is read from where the last scan stopped (`_SCANNED`).
# The file is read whole, parsed, only when an attempt newer than the last megabyte's is in the
# part before it.

_JSON_STRING = rb'"(?:[^"\\\r\n]|\\.)*"'
_SPACE = rb"[ \t\r]*"  # JSON's spaces within a line
# The start of a line as the sink writes a record, to the character after `attempt`'s value: its
# groups are the run ID, rank and attempt as JSON.
_RECORD_HEAD = re.compile(
    rb"\{" + _SPACE + rb'"schema_version"' + _SPACE + rb":" + _SPACE + rb"-?\d+" + _SPACE + rb"," + _SPACE
    + rb'"run_id"' + _SPACE + rb":" + _SPACE + rb"(" + _JSON_STRING + rb")" + _SPACE + rb"," + _SPACE
    + rb'"rank"' + _SPACE + rb":" + _SPACE + rb"(-?\d+)" + _SPACE + rb"," + _SPACE
    + rb'"attempt"' + _SPACE + rb":" + _SPACE + rb"(-?\d+|" + _JSON_STRING + rb")" + _SPACE + rb"[,}]"
)
_SCAN_BLOCK = 8 << 20  # read and matched at a time
_CHECK_BYTES = 256  # the bytes before where a scan stopped, kept to see the file is the one scanned
_SCANNED: dict[str, tuple[tuple[int, int], int, bytes, bytes, dict[int, dict[Any, None]]]] = {}
_SCANNED_MOST = 4096
_SKIPS_MOST = 1024


def _decode_head(groups: tuple[bytes, bytes, bytes], decoded: dict[Any, Any]) -> Any:
    """`(rank, (run, attempt))` from a line's three groups, or None when one is not JSON."""
    found = decoded.get(groups, decoded)
    if found is decoded:
        try:
            run, rank, attempt = (json.loads(group) for group in groups)
            found = (rank, (run if isinstance(run, str) else "", attempt))
        except ValueError:
            found = None
        decoded[groups] = found
    return found


def _scan_block(block: bytes, found: dict[int, dict[Any, None]], decoded: dict[Any, Any],
                skips: dict[bytes, Any]) -> None:
    """Add the keys of the lines of `block` (whole lines, the last ending in a newline) to
    `found`. A line that starts as a record is read from its start, and every line after it that
    starts with the same bytes, to the character after the attempt, is of the same rank, run and
    attempt: they are passed over by one search for the next line that does not (in C). A line
    that does not start as a record is parsed whole."""
    from .evidence import _run_of

    at, size = 0, len(block)
    while at < size:
        head = _RECORD_HEAD.match(block, at)
        key = _decode_head(head.groups(), decoded) if head is not None else None
        if key is None:
            end = block.index(b"\n", at) + 1
            record = _record_of(block[at:end])
            rank = _record_rank(record)
            if record is not None and rank is not None:
                key = (rank, (_run_of(record), record.get("attempt", 0)))
            at = end
        else:
            start = block[at:head.end()]
            at = block.index(b"\n", at) + 1
            # The search pays only where the next line starts the same way: a file whose ranks
            # change from line to line is stepped a line at a time, never a pattern a line.
            if block.startswith(start, at):
                skip = skips.get(start)
                if skip is None:
                    if len(skips) >= _SKIPS_MOST:
                        skips.clear()
                    skip = skips[start] = re.compile(rb"\n(?!" + re.escape(start) + rb")")
                after = skip.search(block, at)
                at = after.end() if after is not None else size
        if key is not None:
            try:
                found.setdefault(key[0], {}).setdefault(key[1], None)
            except TypeError:  # an attempt that is not a number or a string: no reader orders it
                continue


def _scan_lines(file: Any, start: int, before: int, found: dict[int, dict[Any, None]]) -> int:
    """Read the lines of `file` from byte `start` (a line's start) that start before `before`,
    and add each rank's `(run, attempt)` keys to `found` in the order they first appear. The byte
    after the last line read is returned. A last line with no newline is not read."""
    decoded: dict[Any, Any] = {}
    skips: dict[bytes, Any] = {}
    file.seek(start)
    at = start
    while at < before:
        block = file.read(min(_SCAN_BLOCK, before - at))
        if not block:
            break
        if not block.endswith(b"\n"):
            block += file.readline()  # the line that starts in the block, whole
            if not block.endswith(b"\n"):  # the file's last line, not finished
                block = block[: block.rfind(b"\n") + 1]
                if block:
                    _scan_block(block, found, decoded, skips)
                    at += len(block)
                break
        _scan_block(block, found, decoded, skips)
        at += len(block)
    return at


def _attempts_in_file(path: Path, before: int) -> dict[int, list[tuple[str, Any]]]:
    """Each rank's `(run, attempt)` keys among the lines of `path` that start before byte
    `before`, in the order they first appear (see the notes above `_RECORD_HEAD`). Never raises:
    a file that cannot be read gives what was read of it."""
    return _scan_file(path, before)[0]


def _scan_file(path: Path, before: int) -> tuple[dict[int, list[tuple[str, Any]]], bool]:
    """`_attempts_in_file`, and whether every line that starts before `before` was read: False
    when the file could not be read, or was cut or replaced while it was, so the ranks found are
    not all of the part's (lane AA5: the count of ranks is settled only by a whole scan)."""
    found: dict[int, dict[Any, None]] = {}
    end = -1
    try:
        stat = path.stat()
        name = os.path.abspath(path)
        ident = (stat.st_dev, stat.st_ino)
        with path.open("rb") as file:
            head = file.read(_CHECK_BYTES)
            start = 0
            held = _SCANNED.get(name)
            if held is not None and held[0] == ident and held[1] <= min(before, stat.st_size) and held[3] == head[:len(held[3])]:
                file.seek(max(0, held[1] - _CHECK_BYTES))
                if file.read(held[1] - max(0, held[1] - _CHECK_BYTES)) == held[2]:
                    start = held[1]
                    found = {rank: dict(keys) for rank, keys in held[4].items()}
            end = _scan_lines(file, start, before, found)
            file.seek(max(0, end - _CHECK_BYTES))
            check = file.read(end - max(0, end - _CHECK_BYTES))
        _SCANNED.pop(name, None)
        if len(_SCANNED) >= _SCANNED_MOST:
            _SCANNED.pop(next(iter(_SCANNED)))
        _SCANNED[name] = (ident, end, check, head, {rank: dict(keys) for rank, keys in found.items()})
    except OSError:
        pass
    # `before` is where a line starts (`_tail_from`), so a scan that read every line before it
    # stops on it; one that stopped short (a read that failed, a file cut) did not.
    return {rank: list(keys) for rank, keys in found.items()}, end == before


def _hidden_attempts(path: Path, before: int, tail: list[dict[str, Any]], by_rank: Mapping[int, set[Any]],
                     scanned: Mapping[int, list[tuple[str, Any]]] | None = None) -> list[int]:
    """The ranks of `tail` (the records of a file's last megabyte) that show one attempt there
    while a newer one, of the same run, is in the part of the file before it (L21): the newest is
    the last to first appear in the file (contract 1), and only reading the file whole says where
    such a rank stands. `scanned` is `_attempts_in_file(path, before)` when it was read already."""
    from .evidence import _run_of

    order = {rank: list(keys) for rank, keys in (scanned if scanned is not None else _attempts_in_file(path, before)).items()}
    for record in tail:
        rank = _record_rank(record)
        if rank is None:
            continue
        try:
            key = (_run_of(record), record.get("attempt", 0))
            if key not in order.setdefault(rank, []):
                order[rank].append(key)
        except TypeError:
            continue
    hidden = []
    for rank, keys in by_rank.items():
        if len(keys) != 1:
            continue
        (only,) = keys
        same_run = [key for key in order.get(rank, ()) if key[0] == only[0]]
        if same_run and same_run[-1] != only:
            hidden.append(rank)
    return hidden


def _tail_lines(path: Path) -> list[bytes]:
    """The whole lines of a `.jsonl` file's last megabyte, last line last."""
    return _tail(path)[0]


def _attempts_by_rank(tail: list[dict[str, Any]]) -> dict[int, set[Any]]:
    """The attempts each rank's records name among the records of a file's last megabyte
    (`tail`), as `(run, attempt)`. More than one for a rank means the file may hold a late row of
    an attempt that a later one replaced (XC4), and the rank's last line is then not where it
    stands. One means its last line stands: a resume whose new attempt fills the last megabyte is
    not read again (a late row of the old attempt after it would be in the tail too)."""
    from .evidence import _run_of

    found: dict[int, set[Any]] = {}
    for record in tail:
        rank = _record_rank(record)
        if rank is None:
            continue
        try:
            key = (_run_of(record), record.get("attempt", 0))
            hash(key)
        except TypeError:  # an attempt that is not a number or a string: no reader orders it
            continue
        found.setdefault(rank, set()).add(key)
    return found


def _last_of(lines: list[bytes]) -> dict[str, Any] | None:
    """The last whole line, as a record."""
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            found = json.loads(line)
        except ValueError:
            return None
        return found if isinstance(found, dict) else None
    return None


def _last_record(path: Path) -> dict[str, Any] | None:
    """The last whole line of a `.jsonl` file, read from its last megabyte only."""
    return _last_of(_tail_lines(path))


def _steps_by_rank(tail: list[dict[str, Any]]) -> dict[int, int]:
    """Each rank's step on its last record among `tail` (the records of a file's last megabyte).
    In a file of several ranks the last line is one rank's, and another rank may be ahead of it.
    A rank's last line is where that rank is when it has one attempt there; a rank whose
    attempts are several is read whole (`_local_ranks`, XC4)."""
    steps: dict[int, int] = {}
    for record in reversed(tail):
        rank, step = _record_rank(record), record.get("step")
        if rank is not None and rank not in steps and isinstance(step, int) and not isinstance(step, bool):
            steps[rank] = step
    return steps


def _rank_of_file(path: Path, first: Mapping[str, Any] | None) -> int:
    named = _named_rank(path)
    if named is not None:
        return named
    value = (first or {}).get("rank")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _file_lines(path: Path) -> Iterable[tuple[bytes, dict[str, Any]]]:
    """Each record of a `.jsonl` file with its line, as `compare` reads the file (`_read_local`):
    every line that is a JSON object. A line that is not one (torn by a killed process) is left
    out, and then so is a last line with no newline, as the watcher reads the file; otherwise
    that last line is read when it is whole. Read a line at a time: no file is held whole."""
    torn = False
    held: bytes | None = None
    try:
        with path.open("rb") as file:
            for line in file:
                if not line.endswith(b"\n"):
                    held = line  # the last line, with no newline
                    break
                record = _record_of(line)
                if record is None:
                    torn = torn or bool(line.strip())
                    continue
                yield line, record
    except OSError:
        return
    if held is not None and not torn and (record := _record_of(held)) is not None:
        yield held, record


def _record_of(line: bytes) -> dict[str, Any] | None:
    if not line.strip():
        return None
    try:
        found = json.loads(line)
    except ValueError:
        return None
    return found if isinstance(found, dict) else None


def _rank_across_files(paths: list[Path], rank: int, run_id: str | None = None) -> dict[str, Any]:
    """One rank's records in several files, read as `compare` reads them (lane X5, CLI-13): every
    record of every file, a record identical, line for line, to one of an earlier file read once
    (`compare._not_read_yet`), the attempts in contract 1's order across the files
    (`attempt_order_report`), and the last step the highest that `resolve_attempts` keeps.

    What is kept is each attempt's first record in each file (all the order needs) and its
    steps, 8 bytes each; never the records. `last_step` (None for no step), `order` (oldest
    first), `report` (`attempt_order_report`'s entry for the rank) and `holders`, the files that
    hold the newest attempt's records, in the order given."""
    from array import array
    from bisect import bisect_left

    from .evidence import _run_of, attempt_order_report, resolve_attempts

    seen: set[int] = set()
    firsts: list[list[dict[str, Any]]] = []
    steps: dict[tuple[str, Any], array] = {}
    holders: dict[tuple[str, Any], list[Path]] = {}
    for path in paths:
        mine: set[int] = set()
        here: dict[tuple[str, Any], dict[str, Any]] = {}
        for line, record in _file_lines(path):
            if _record_rank(record) != rank:
                continue
            attempt = record.get("attempt", 0)
            try:
                found = (_run_of(record), attempt)
                hash(found)
            except TypeError:  # an attempt that is not a number or a string: `compare` cannot read it either
                continue
            # A file holds the attempt whether or not its lines were read from an earlier file:
            # a copy sorting first leaves the file the attempt wrote, and its end marker, a holder.
            if path not in holders.setdefault(found, []):
                holders[found].append(path)
            line_key = hash(line.strip())
            mine.add(line_key)
            if line_key in seen:
                continue
            if found not in here:
                here[found] = {"run_id": found[0], "rank": rank, "attempt": attempt, "ts": record.get("ts")}
            step = record.get("step")
            if isinstance(step, int) and not isinstance(step, bool) and -(1 << 63) <= step < (1 << 63):
                steps.setdefault(found, array("q")).append(step)
        seen |= mine
        firsts.append(list(here.values()))
    report = attempt_order_report(firsts, [p.name for p in paths])
    keys = list(report)
    picked = next((k for k in keys if k[0] == run_id), keys[0] if keys else None)
    if picked is None:
        return {"last_step": None, "order": [], "report": None, "holders": []}
    order = report[picked]["order"]
    # The furthest step `resolve_attempts` keeps, from a few of each attempt's steps: its first
    # and last, and its last below each other attempt's first, where every cut it can make falls.
    chosen: list[dict[str, Any]] = []
    for (run, attempt), held in steps.items():
        if run != picked[0]:
            continue
        ordered = sorted(set(held))
        cuts = {min(other) for (r, a), other in steps.items() if r == run and a != attempt}
        keep = {ordered[0], ordered[-1]} | {ordered[at - 1] for cut in cuts if (at := bisect_left(ordered, cut)) > 0}
        chosen += [{"run_id": run, "rank": rank, "attempt": attempt, "step": step} for step in sorted(keep)]
    kept = resolve_attempts(chosen, order={(picked[0], rank): order})
    newest = order[-1] if order else None
    return {"last_step": max((r["step"] for r in kept), default=None), "order": order, "report": report[picked],
            "holders": holders.get((picked[0], newest), [])}


def _is_run_folder(folder: Path) -> bool:
    # `rank=<n>` read as the evidence readers read it, ASCII digits only (lane X5, DET-17: any
    # folder whose name began `rank=` made a run folder here).
    from .evidence import rank_number

    try:
        return (folder / _manifest.STORE_MANIFEST).is_file() or (folder / "watchdog.json").is_file() or any(
            entry.is_dir() and rank_number(entry.name) is not None for entry in os.scandir(folder)
        )
    except OSError:
        return False


def _store_runs(runs_folder: Path) -> list[dict[str, Any]]:
    found = []
    try:
        entries = sorted(os.scandir(runs_folder), key=lambda entry: entry.name)
    except OSError:
        return []
    for entry in entries:
        folder = Path(entry.path)
        if entry.is_dir() and _is_run_folder(folder):
            found.append({"run_id": entry.name, "source": "store", "path": folder,
                          "manifest": _manifest.find_manifest(folder, entry.name)})
    return found


def _local_runs(folder: Path, files: Iterable[Path] | None = None) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}

    def entry(run_id: str) -> dict[str, Any]:
        # `files_given`: the run is the file given, and its other ranks' files were not looked for.
        return by_id.setdefault(run_id, {"run_id": run_id, "source": "local", "path": None, "files": {},
                                         "manifest": None, **({"files_given": True} if files is not None else {})})

    for path in sorted(folder.glob("*.jsonl")) if files is None else files:
        if not path.is_file():
            continue
        first = None
        run_id = _manifest.run_id_of_file(path)
        if run_id is None:  # a file with a name of its own: its records say whose it is
            first = _first_record(path)
            run_id = str(first["run_id"]) if first and first.get("run_id") else path.name[: -len(".jsonl")]
        found = entry(run_id)
        rank = _rank_of_file(path, first)
        if found["files"].setdefault(rank, path) != path:  # another file of this rank: kept, never left out
            found.setdefault("more_files", {}).setdefault(rank, []).append(path)
    if files is None:
        for path in sorted(folder.glob(f"*{_manifest.MANIFEST_SUFFIX}")):
            # A manifest whose rank 0 has not written a record yet is still a run that started.
            entry(path.name[: -len(_manifest.MANIFEST_SUFFIX)])
    for run_id, found in by_id.items():
        ranks = sorted(found["files"])
        found["path"] = found["files"][ranks[0]] if ranks else folder / f"{run_id}{_manifest.MANIFEST_SUFFIX}"
        found["manifest"] = _manifest.find_manifest(folder / f"{run_id}{_manifest.MANIFEST_SUFFIX}", run_id)
    return [by_id[run_id] for run_id in sorted(by_id)]


def _once(local: list[dict[str, Any]], stored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The runs of one folder that holds local files and a store, each run once (lane V4,
    CLI-21: a run kept both ways was listed twice). The run ID is the identity. A run with
    local records is read from them, as `find_run` always read it, since they hold every record;
    the entry says where its store copy is (`also_in_store`). A run whose local side is only
    its manifest is read from the store, with the local manifest when the store has none."""
    kept = list(local)
    by_id = {run["run_id"]: index for index, run in enumerate(kept)}
    for run in stored:
        index = by_id.get(run["run_id"])
        if index is None:
            kept.append(run)
        elif kept[index]["files"]:
            kept[index]["also_in_store"] = run["path"]
        else:
            mine = kept[index]
            kept[index] = {**run, "manifest": run["manifest"] or mine["manifest"], "also_local": mine["path"]}
    return kept


def discover_runs(source: str | Path) -> list[dict[str, Any]]:
    """Every run at `source`, each as `{"run_id", "source", "path", "manifest"}` (and `files`,
    a `.jsonl` file per rank, for local files). `source` is a folder of local files, a
    server's store (the folder that holds `runs/`, or `runs/` itself), one run folder of a
    store, or one `.jsonl` file. A folder may hold local files and a store both; a run kept both
    ways is one entry (`_once`).

    `files` keeps one file a rank. A rank whose records are in several files (a restart that
    wrote a file of a name of its own) has the others in `more_files`, by rank, so no file is
    left out (lane V4, CLI-13)."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist: nothing to list")
    if source.is_file():
        if source.suffix != ".jsonl":
            raise ValueError(f"{source} is not a folder of runs or a .jsonl capture")
        return _local_runs(source.parent, [source])
    if (source / "runs").is_dir() and not _is_run_folder(source):
        return _once(_local_runs(source), _store_runs(source / "runs"))
    if _is_run_folder(source):  # one run of a store
        folder = source.resolve()
        return [{"run_id": folder.name, "source": "store", "path": source,
                 "manifest": _manifest.find_manifest(source, folder.name)}]
    return _once(_local_runs(source), _store_runs(source))


def find_run(source: str | Path, wanted: str, project: str | None = None,
             runs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The run at `source` whose run ID is `wanted`, or else the one run named `wanted`.

    The run ID is the identity, so an exact ID wins whatever the project. A name is a label
    and may repeat: `project` narrows the search to one project, and a name that still fits
    several runs raises `AmbiguousRunName` (a `RunLookupError`) with their IDs. A run kept both as local files and in
    a store under one folder is read from the local files, which hold every record.
    """
    runs = discover_runs(source) if runs is None else runs
    by_id = [run for run in runs if run["run_id"] == wanted]
    if by_id:
        return next((run for run in by_id if run["source"] == "local"), by_id[0])
    blocks = [(run, _manifest.run_block(run["run_id"], run["manifest"])) for run in runs]
    named = [(run, block) for run, block in blocks
             if block["name"] == wanted and (project is None or block["project"] == project)]
    ids = sorted({run["run_id"] for run, _ in named})
    within = f" in project {project!r}" if project is not None else ""
    if len(ids) > 1:
        listed = ", ".join(f"{run['run_id']} (project {block['project']})"
                           for run, block in sorted(named, key=lambda item: item[0]["run_id"]))
        raise AmbiguousRunName(
            f"the name {wanted!r} fits {len(ids)} runs{within} at {source}: {listed}. "
            "A name is a label and may repeat; give the run ID."
        )
    if not named:
        known = ", ".join(f"{run['run_id']}" + (f" ({block['name']})" if block["name"] != run["run_id"] else "")
                          for run, block in blocks[:20]) or "none"
        raise RunLookupError(f"no run with the ID or name {wanted!r}{within} at {source}. Runs there: {known}")
    return next((run for run, _ in named if run["source"] == "local"), named[0][0])


# --- One row a run ------------------------------------------------------------------------


def _by_class(incidents: Iterable[Any]) -> dict[str, int]:
    from .evidence import incident_class

    counts: dict[str, int] = {}
    for incident in incidents:
        if isinstance(incident, Mapping):
            own = incident.get("class")
            kind_class = incident_class(incident.get("kind")) or (own if isinstance(own, str) else "unclassified")
            counts[kind_class] = counts.get(kind_class, 0) + 1
    return counts


def _first(incidents: Iterable[Any]) -> dict[str, Any] | None:
    stepped = [i for i in incidents if isinstance(i, Mapping)]
    if not stepped:
        return None

    def step(incident: Mapping[str, Any]) -> float:
        value = incident.get("step")
        return value if isinstance(value, int) and not isinstance(value, bool) else float("inf")

    first = min(stepped, key=step)  # the earliest step; the evidence's own order breaks a tie
    return {"kind": first.get("kind"), "step": first.get("step")}


def _kinds_checked(kind_checks: Any) -> dict[str, int] | None:
    if not isinstance(kind_checks, Mapping):
        return None
    values = [check.get("value") for check in kind_checks.values() if isinstance(check, Mapping)]
    return {"checked": values.count("checked"), "of": len(values)}


def _iso(ts: Any) -> str | None:
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        try:
            return datetime.fromtimestamp(float(ts), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (OverflowError, OSError, ValueError):
            return None
    return None


def _record_rank(record: Mapping[str, Any] | None) -> int | None:
    """A record's rank, as `compare` reads it (no `rank` is rank 0); None for no record."""
    if record is None:
        return None
    value = record.get("rank", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _local_ranks(files: Mapping[int, Path], manifest: Mapping[str, Any] | None, given: bool = False,
                 more: Mapping[int, list[Path]] | None = None, run_id: str | None = None) -> dict[str, Any]:
    """The ranks of a local run, from what the listing reads anyway: the file names, the first
    line and the last megabyte of each file (every whole line of it), the scan of the part before
    the last megabyte (the rank and attempt at each line's start, `_scan_file`), and the
    manifest's `world_size`. No file is parsed whole to count them.

    A file named `<run_id>.rank<N>.jsonl` holds the ranks its lines name (`tensorquill.init`
    writes one such file a rank): every line of it is read for them, the last megabyte parsed and
    the part before it scanned (`_scan_file`, the scan L21 makes anyway; lane AA5). Rank N alone,
    it is rank N's file; others too (XC5, lane Y3, Nov 20, 2026: only the first and last records
    were read, so a row of another rank in between was not seen), it is a file of several ranks
    whose ranks are all named (`every_line`), and the count is settled. When the scan cannot
    read every line, the file is read as before: rank N's when what was read names N alone (then
    `middle_not_read`), else as a file with a name of its own. A file of a name of its own is counted from its first
    and last records, as before. Such a file may hold several ranks: its name and what
    was read name some of them, and only reading it whole would name them all. Its count is
    settled when the manifest's `world_size` equals the ranks seen; otherwise it is not known,
    and `why` says so (the row prints it rather than a count that may be wrong). A file of its
    rank alone whose middle the scan could not read (`middle_not_read`) may hold another rank
    there: its count is settled by the manifest's `world_size` and, with none, is not known
    either. When every file holds its rank alone, the ranks the manifest's `world_size`
    counts that have no file are `missing`, as the store counts a rank its status expects and
    that stored nothing: unknown, never clean. A world size above `MAX_EXPECTED_RANKS` (XC6) is
    not made into a list: the ranks without a file are a count, `beyond`, and are as unknown.

    With `given` (a run given as the path of one file, `discover_runs`), the other ranks' files
    were not looked for: a missing rank is one not read, and `missing_verb` says so.

    `held` gives each file of its own name (or misnamed) the ranks it is seen to hold, and
    `several` the files among them that hold several; `last_steps` the ranks' last steps
    (`_steps_by_rank`) of each file that holds or may hold several, where the row uses them.
    `resolved` has, for each file whose last megabyte shows a rank in more than one attempt (XC4),
    or in one attempt while a newer one is in the part before it (`_hidden_attempts`, L21), that
    rank's last step as `resolve_attempts` keeps it, the file read whole
    (`_rank_across_files`): a late row of a replaced attempt does not move it.

    `more` are a rank's other files (`discover_runs`' `more_files`): read as the others are for
    the count. `split` has, for each rank in `more`, its files read whole (`_rank_across_files`):
    the attempts' order across them, the last step and the files of the newest attempt.
    """
    from .evidence import MAX_EXPECTED_RANKS

    seen: set[int] = set()
    own_name: list[str] = []
    misnamed: list[str] = []
    several: list[Path] = []
    middle_not_read: list[str] = []
    every_line: set[Path] = set()  # files named for a rank, of several ranks, read on every line
    held: dict[Path, list[int]] = {}
    tails: dict[Path, list[dict[str, Any]]] = {}
    read: dict[Path, tuple[dict[str, Any] | None, dict[str, Any] | None]] = {}
    resolved: dict[Path, dict[int, int | None]] = {}
    split_ranks = {number for number, paths in (more or {}).items() if number in files and paths}
    pairs = [*files.items(), *((number, path) for number, paths in (more or {}).items() for path in paths)]
    for number, path in pairs:
        lines, whole, tail_starts = _tail_from(path)
        read[path] = first, last = _first_record(path), _last_of(lines)  # kept: the row reads them too
        tail = [r for r in map(_record_of, lines) if r is not None]
        ends = {r for r in (_record_rank(first), _record_rank(last)) if r is not None}
        named = _named_rank(path)
        # Every record read names its rank for a file named for one (XC5). A file of a name of its
        # own is counted from its ends as before: the files never settle its count either way.
        in_tail = {r for r in map(_record_rank, tail) if r is not None} if named is not None else set()
        # The part before the last megabyte, scanned for the attempts (and so the ranks) its lines
        # name (L21). `complete`: every whole line of the file was read, by the scan or the tail.
        # A split rank's file of a name of its own is read whole anyway and not counted by its
        # lines, so it is not scanned: the scan adds no read where it was not made before.
        scan = not whole and (number not in split_ranks or named is not None)
        scanned, complete = _scan_file(path, tail_starts) if scan else ({}, whole)
        if number not in split_ranks:  # a rank's several files are read whole anyway (`split`)
            by_attempt = _attempts_by_rank(tail)
            restarted = [r for r, attempts in by_attempt.items() if len(attempts) > 1]
            if not whole:
                # L21: an attempt newer than the last megabyte's, in the part before it.
                restarted += [r for r in _hidden_attempts(path, tail_starts, tail, by_attempt, scanned)
                              if r not in restarted]
            if restarted:
                resolved[path] = {r: _rank_across_files([path], r, run_id)["last_step"] for r in restarted}
        # A file named for a rank whose every line was read holds the ranks they name, no more
        # (lane AA5, L21's "1+": a rank file over a megabyte was counted only by the manifest).
        in_middle = set(scanned) if named is not None and complete else set()
        if named is not None and (ends | in_tail | in_middle) <= {named}:
            seen.add(named)
            if not complete:
                middle_not_read.append(path.name)
            continue
        if named is not None and complete:
            # It holds other ranks too, and every one of them is named: the count is theirs, and
            # its evidence file is still one rank's of several (`several`, `shared`).
            # Its rank is counted though no line names it, as `compare` counts it (unknown there).
            here = ends | in_tail | in_middle | {named}
            seen |= here
            held[path] = sorted(here)
            tails[path] = tail
            several.append(path)
            every_line.add(path)
            continue
        here = ends | in_tail | {named if named is not None else number}
        seen |= here
        held[path] = sorted(here)
        tails[path] = tail
        if len(here) > 1:
            several.append(path)
        if named is None:
            own_name.append(path.name)
        else:
            by = "its first and last records name" if not ends <= {named} else "its first record and last megabyte name"
            misnamed.append(f"{path.name} is named for rank {named} and {by} {_rank_list(sorted(ends | in_tail))}, so "
                            "may hold other ranks")
    world = (manifest or {}).get("world_size")
    world = world if isinstance(world, int) and not isinstance(world, bool) and world > 0 else None
    unsettled = bool(own_name or misnamed)
    reason = f"the run manifest says world size {world}"
    if given:
        reason = (f"only {', '.join(sorted(p.name for _, p in pairs))} {'was' if len(pairs) == 1 else 'were'} "
                  f"given, and {reason}")
    missing = {n: reason for n in range(world) if n not in seen} if (
        not unsettled and world is not None and world <= MAX_EXPECTED_RANKS) else {}
    # Above the cap the ranks without a file are counted, never listed (XC6): as unknown as `missing`.
    beyond = world - len(seen) if not unsettled and world is not None and world > MAX_EXPECTED_RANKS else 0
    # `shared`: a lone file read as one run of several ranks, for its ends or for the world size.
    # Not on which rank wrote the last line alone, which changes from one listing to the next.
    # A file read on every line that holds several ranks is shared with its count settled (AA5).
    shared = len(pairs) == 1 and (bool(several) or (unsettled and (world or 1) > 1))
    # Each rank's last step is read only where the row uses it: a file of several ranks in a run
    # of several files, or a lone file read as several ranks. A rank seen in several attempts is
    # where `resolve_attempts` leaves it (XC4).
    last_steps = {path: {**_steps_by_rank(tail), **{r: s for r, s in resolved.get(path, {}).items() if s is not None}}
                  for path, tail in tails.items() if path in several or shared}
    found: dict[str, Any] = {"ranks": sorted(seen | set(missing)), "ends": read, "held": held, "several": several,
                             "last_steps": last_steps, "missing": missing, "beyond": beyond, "resolved": resolved,
                             "missing_verb": ("was not read", "were not read") if given else ("has no file", "have no file"),
                             "shared": shared, "every_line": every_line,
                             "split": {number: _rank_across_files([files[number], *paths], number, run_id)
                                       for number, paths in (more or {}).items() if number in files and paths}}
    clauses = []
    if unsettled and world != len(seen):
        if own_name:
            names = ", ".join(own_name[:4]) + (f" and {len(own_name) - 4} more" if len(own_name) > 4 else "")
            clauses.append(f"{names} {'has a name' if len(own_name) == 1 else 'have names'} of "
                           f"{'its' if len(own_name) == 1 else 'their'} own, so may hold several ranks")
        clauses += misnamed[:4] + ([f"{len(misnamed) - 4} more files are named for a rank their records do not hold alone"]
                                   if len(misnamed) > 4 else [])
    if middle_not_read and world is None:
        names = ", ".join(middle_not_read[:4]) + (f" and {len(middle_not_read) - 4} more" if len(middle_not_read) > 4 else "")
        one = len(middle_not_read) == 1
        clauses.append(f"{names} {'is' if one else 'are'} larger than the part of a file the listing reads (the first "
                       f"line and the last megabyte), and the middle of {'it' if one else 'each'} may hold another rank")
    if clauses:
        said = (f"the run manifest says world size {world}" if world is not None
                else "no run manifest says the world size" if manifest is None
                else "the run manifest does not say the world size")
        by = ("the file names and their first lines and last megabytes" if len(own_name) < len(pairs)
              else "the first lines and last megabytes")
        found["why"] = (f"{'; '.join(clauses)}; {by} name {_rank_list(sorted(seen))}, and {said}. How many ranks the run "
                        "has is not known without reading every record (python -m tensorquill compare reads them)")
    elif beyond:
        found["why"] = (f"the run manifest says world size {world:,}, more than the {MAX_EXPECTED_RANKS:,} ranks this "
                        f"listing counts: {_rank_list(sorted(seen))} {'has a file' if len(seen) == 1 else 'have files'}, "
                        f"and the other {beyond:,} are not judged")
    return found


def _holds(path: Path, known: Mapping[str, Any]) -> str:
    """What the listing can say of the ranks a file of its own name (or a misnamed one) holds."""
    ranks = _rank_list(known["held"][path])
    if "why" not in known or path in known.get("every_line", ()):
        return f"holds {ranks}"
    # A file named for a rank is read for the rank of every record of its last megabyte (XC5).
    by = ("its name, its first record and its last megabyte" if _named_rank(path) is not None
          else "its first and last records")
    return f"may hold several ranks ({by} name {ranks})"


def _rank_list(ranks: list[int]) -> str:
    return f"rank{'s' if len(ranks) != 1 else ''} {', '.join(map(str, ranks))}"


def _named_rank(path: Path) -> int | None:
    """N for a file named `<anything>.rank<N>.jsonl`, as `tensorquill.init` names them."""
    _, _, rank = path.name[: -len(".jsonl")].rpartition(".rank")
    return int(rank) if rank.isdecimal() else None


def _local_row(run: Mapping[str, Any], known: Mapping[str, Any] | None) -> dict[str, Any]:
    files: Mapping[int, Path] = run.get("files") or {}
    row: dict[str, Any] = {"ranks": sorted(files)}
    if not files or known is None:
        row["evidence"] = {"state": "no_records", "detail": "the manifest is written and no rank has written a record yet"}
        return row
    row["ranks"] = known["ranks"]
    if "why" in known:
        row["rank_count_not_known"] = known["why"]
    more = run.get("more_files") or {}
    if len(files) > 1 or known["missing"] or known["beyond"] or more:
        return _local_all_ranks_row(files, row, known, more)
    rank = min(files)
    records = files[rank]
    if known["shared"]:
        return _local_shared_file_row(records, known, row)
    first, last = known["ends"][records]
    step = _file_step(records, last, known)
    if step is not None:
        row["last_step"] = step
    row["started_from_records"] = _iso((first or {}).get("ts"))
    evidence, look = _read_evidence_file(records)
    if evidence is None:
        row["evidence"] = {**look, "rank": rank}
        if look["state"] == "missing":
            row["evidence"]["detail"] = f"{look['detail']}: write it with python -m tensorquill {records}"
        return row
    stale = look["state"] == "older_than_records"
    row["evidence"] = {
        "state": look["state"], "rank": rank, "path": look["path"],
        **({"detail": f"{records.name} has grown since {Path(look['path']).name} was written; the row is as of that file"}
           if stale else {}),
        **({"compact": True} if isinstance(evidence.get("compact"), Mapping) else {}),
    }
    incidents = evidence.get("incidents") or []
    row["run_state"] = (evidence.get("run_state") or {}).get("state")
    row["incidents"] = _by_class(incidents)
    row["first_incident"] = _first(incidents)
    row["kinds_checked"] = _kinds_checked(evidence.get("kind_checks"))
    return row


def _file_step(path: Path, last: Mapping[str, Any] | None, known: Mapping[str, Any]) -> int | None:
    """The last step of a file that holds one rank: its last line's, unless the file shows the
    rank in more than one attempt, and then the step `resolve_attempts` leaves it at, the file
    read whole (`known["resolved"]`, XC4)."""
    resolved = [s for s in (known["resolved"].get(path) or {}).values() if s is not None]
    if resolved:
        return max(resolved)
    step = (last or {}).get("step")
    return step if isinstance(step, int) and not isinstance(step, bool) else None


def _local_shared_file_row(records: Path, known: Mapping[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """One `.jsonl` file that holds several ranks. Its evidence file, if any, is one rank's:
    the command line judges one rank of such a file (`--rank N`). So its incidents are listed as
    found, the run state is `unhealthy` if it shows any and otherwise unknown (the other ranks
    are not judged: the all-ranks rule), and what was checked is not counted for the run.

    The evidence names the rank it judged (`rank`) and, when it was picked out of a file of
    several ranks, the ranks that file held (`ranks_in_source`, lane Z2). When it names one of
    the file's ranks, the row's evidence cell names it too (`rank`) and the detail says the ranks
    the file held when it was written (lane AA5, Nov 20, 2026; it said "without saying which").
    Evidence that names no rank, or one the file is read not to hold (its ranks all read), is
    read as before."""
    first, last = known["ends"][records]
    # The furthest rank's, as for a run of several files or a store: the last line is one rank's,
    # and another rank may be ahead of it.
    steps = known["last_steps"].get(records) or {}
    if steps:
        row["last_step"] = max(steps.values())
    elif isinstance((last or {}).get("step"), int):
        row["last_step"] = last["step"]
    row["started_from_records"] = _iso((first or {}).get("ts"))
    holds = _holds(records, known)
    evidence, look = _read_evidence_file(records)
    if evidence is None:
        row["evidence"] = {
            **look, "rank": None,
            "detail": f"{records.name} {holds} and no evidence file is "
                      f"beside it: write one rank's with python -m tensorquill {records} --rank N, or compare the run "
                      "(python -m tensorquill compare reads every rank of the file)",
        }
        return row
    incidents = evidence.get("incidents") or []
    row["incidents"] = _by_class(incidents)
    row["first_incident"] = _first(incidents)
    row["run_state"] = "unhealthy" if (evidence.get("run_state") or {}).get("state") == "unhealthy" else "unknown"
    judged = evidence.get("rank")
    # A rank the file is read to hold, or any rank when the listing does not read all of them
    # (`why`: a file of a name of its own is read at its ends only).
    judged = judged if isinstance(judged, int) and not isinstance(judged, bool) and (
        judged in known["ranks"] or "why" in known) else None
    in_source = evidence.get("ranks_in_source")
    in_source = (sorted(set(in_source)) if isinstance(in_source, list) and in_source
                 and all(isinstance(n, int) and not isinstance(n, bool) for n in in_source) else None)
    if judged is None:
        whose = "one rank's, without saying which: the row's incidents are that rank's"
    else:
        held = f" of the {_rank_list(in_source)} the file held when it was written" if in_source else ""
        whose = f"rank {judged}'s{held}: the row's incidents are rank {judged}'s"
    row["evidence"] = {
        "state": "ranks_unknown", "rank": judged, "path": look.get("path"), "judged_ranks": 1,
        "unknown_ranks": len(known["ranks"]) - 1 if "why" not in known else None,
        **({"ranks_in_source": in_source} if judged is not None and in_source else {}),
        **({"compact": True} if isinstance(evidence.get("compact"), Mapping) else {}),
        "detail": f"{records.name} {holds}, and its evidence file is {whose}, the other ranks are not judged, "
                  "and what was checked is not counted for the run (python -m tensorquill compare reads every rank)"
                  + ("; the records have grown since it was written" if look["state"] == "older_than_records" else ""),
    }
    return row


EVIDENCE_ENDINGS = (".evidence.json", ".evidence.compact.json")  # the full file, and the one `--compact` writes


def _evidence_path(records: Path) -> Path | None:
    """The evidence file beside a `.jsonl` file: `<name>.evidence.json` or the compact
    `<name>.evidence.compact.json` that `--compact` writes by default. When both are there, the
    one written later is read, as the nearer to the records; on the same time, the full one.
    The compact file keeps every key a row reads (only per-step `samples` are cut), so either
    gives the same row when both were written from the same records. None when neither is
    there (lane V4, E7: the compact file alone was listed as `missing`)."""
    found: list[tuple[float, int, Path]] = []
    for order, ending in enumerate(EVIDENCE_ENDINGS):
        path = records.with_suffix(ending)
        try:
            found.append((path.stat().st_mtime, -order, path))
        except OSError:
            continue
    return max(found)[2] if found else None


def _read_evidence_file(records: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """A rank's evidence file beside its `.jsonl` file (`_evidence_path`), and what the row says
    of it."""
    path = _evidence_path(records)
    if path is None:
        return None, {"state": "missing", "detail": f"no evidence file beside {records.name}"}
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
        written = path.stat().st_mtime
    except FileNotFoundError:
        return None, {"state": "missing", "detail": f"no evidence file beside {records.name}"}
    except (OSError, ValueError) as error:
        return None, {"state": "unreadable", "path": str(path), "detail": str(error)}
    if not isinstance(evidence, dict):
        return None, {"state": "unreadable", "path": str(path), "detail": "not a JSON object"}
    try:
        stale = written < records.stat().st_mtime
    except OSError:
        stale = False
    return evidence, {"state": "older_than_records" if stale else "current", "path": str(path)}


def _local_all_ranks_row(files: Mapping[int, Path], row: dict[str, Any], known: Mapping[str, Any],
                         more: Mapping[int, list[Path]] | None = None) -> dict[str, Any]:
    """A local run of several ranks: every rank's evidence file, put together by the
    all-ranks view's rules (`combine_rank_evidence`). No record is judged here: a rank without
    an evidence file is unknown, never clean, and so is a rank the manifest's world size counts
    that has no file (`known["missing"]`).

    A file that holds several ranks (`known["several"]`) has an evidence file of one rank, which
    does not say which, as for a lone such file (`_local_shared_file_row`): it is given to no
    rank. Its ranks are unknown unless a file of their own judges them; its incidents are
    listed with no rank (one that is the same, by the view's identity, as a judged rank's is
    counted once), and the run is `unhealthy` by its incidents as the run-state classifier reads
    them (an observation is not a failure). Its ranks' last steps count toward the run's, each
    rank's own (`_steps_by_rank`).

    A rank whose records are in several files (`more`) is unknown: an evidence file is of one
    file. Its last step is the furthest that `resolve_attempts` keeps of every record of its
    files, the attempts in contract 1's order across them (`known["split"]`), so an attempt that
    a later one replaced does not count (lane V4, CLI-13), and one seen only in the middle of a
    file is seen (lane X5). The files are named."""
    from .evidence import MAX_EXPECTED_RANKS, RUN_STATES, _incident_identity, _rank_ranges, combine_rank_evidence, run_state

    def ranks(numbers: list[int]) -> str:
        return f"rank{'s' if len(numbers) != 1 else ''} {_rank_ranges(numbers, 8)}"

    per_rank: dict[int, dict[str, Any]] = {}
    no_evidence: dict[int, str] = {}
    looks: dict[int, dict[str, Any]] = {}
    steps: list[int] = []
    shared: list[tuple[Path, dict[str, Any] | None, dict[str, Any]]] = []
    split: dict[int, str] = {}
    for number in sorted(files):
        records = files[number]
        if number in (more or {}):
            paths = [records, *more[number]]
            # Every record of the files, the attempts in contract 1's order across them (each
            # file's own order first, then `ts`, then number; never path order; lanes W3, X5).
            across = known["split"][number]
            if across["last_step"] is not None:
                steps.append(across["last_step"])
            report = across["report"] or {}
            split[number] = (f"its records are in {len(paths)} files ({', '.join(p.name for p in paths)}), and an evidence "
                             "file is of one file: compare it (python -m tensorquill compare reads them all)"
                             + (f"; {report['reason']}" if report.get("attempt_order") == "numeric" else ""))
            continue
        _, last = known["ends"][records]
        step = _file_step(records, last, known)
        steps += list((known["last_steps"].get(records) or {}).values()) or ([step] if step is not None else [])
        evidence, look = _read_evidence_file(records)
        looks[number] = look
        if records in known["several"]:
            shared.append((records, evidence, look))
        elif evidence is None:
            no_evidence[number] = f"{look['state']}: {look['detail']}"
        else:
            per_rank[number] = {**evidence, "last_step": step}
    in_shared: dict[int, str] = {}
    notes = []
    for records, evidence, look in shared:
        said = ("its evidence file is one rank's, without saying which" if evidence is not None
                else "no evidence file is beside it" if look["state"] == "missing"
                else f"its evidence file cannot be read ({look['detail']})")
        why = f"its file {records.name} {_holds(records, known)}, and {said}"
        new = [n for n in known["held"][records] if n not in per_rank and n not in no_evidence and n not in in_shared]
        in_shared.update((n, why) for n in new)
        notes.append(f"{records.name} {_holds(records, known)}, and {said}"
                     + (": its incidents are listed with no rank" if evidence is not None else "")
                     + (f", and {ranks(new)} {'are' if len(new) != 1 else 'is'} not judged" if new else ""))
    one, many = known["missing_verb"]
    missing = {n: f"this rank {one}: {why}" for n, why in known["missing"].items()}
    unknown = {**no_evidence, **split, **in_shared, **missing}
    if steps:
        row["last_step"] = max(steps)
    first, _ = known["ends"][files[min(files)]]
    row["started_from_records"] = _iso((first or {}).get("ts"))
    stale = [n for n, look in looks.items() if look["state"] == "older_than_records"]
    if no_evidence:
        notes.insert(0, f"{ranks(sorted(no_evidence))} {'have' if len(no_evidence) != 1 else 'has'} no evidence file (write "
                        f"it with python -m tensorquill <the rank's .jsonl file>), so nothing is said about incidents there")
    for number, why in split.items():
        notes.append(f"rank {number}: {why}, so nothing is said about incidents there")
    if missing:
        notes.append(f"{ranks(sorted(missing))} {many if len(missing) != 1 else one}: "
                     f"{next(iter(known['missing'].values()))}, so nothing is said about incidents there")
    beyond = known["beyond"]
    if beyond:  # XC6: counted, never listed
        notes.append(f"{beyond:,} more rank{'s' if beyond != 1 else ''} {many if beyond != 1 else one}: the run manifest "
                     f"says world size {beyond + len(row['ranks']):,}, more than the {MAX_EXPECTED_RANKS:,} ranks this "
                     "listing counts, so nothing is said about incidents there")
    if stale:
        notes.append(f"the records of {ranks(stale)} have grown since their evidence files were written; the row is as "
                     "of those files")
    shared_evidence = [evidence for _, evidence, _ in shared if evidence is not None]
    if not per_rank and not shared_evidence:
        if split and not no_evidence:  # every rank without one is in several files: no evidence file is missing
            row["evidence"] = {"state": "ranks_unknown", "judged_ranks": 0, "unknown_ranks": len(row["ranks"]) + beyond,
                               "detail": "; ".join(notes)}
            return row
        detail = (f"no evidence file beside any of its {len(files)} ranks' .jsonl files: write each with python -m "
                  f"tensorquill <the rank's .jsonl file>, as for {files[min(files)].name}")
        row["evidence"] = {"state": "missing", "judged_ranks": 0, "unknown_ranks": len(row["ranks"]) + beyond,
                           "detail": "; ".join([detail] + notes[1 if no_evidence else 0:])}
        return row
    combined = combine_rank_evidence(per_rank, unknown_ranks=unknown)
    state = "ranks_unknown" if unknown or beyond else "older_than_records" if stale else "current"
    row["evidence"] = {
        "state": state, "from": "each rank's evidence file", "judged_ranks": len(per_rank),
        "unknown_ranks": len(unknown) + beyond,
        **({"detail": "; ".join(notes)} if notes else {}),
    }
    _all_ranks_cells(row, combined)
    if beyond:  # the ranks above the cap are in no value, as an unknown rank: the run is not healthy
        row["run_state"] = max((row.get("run_state") or "unknown", "unknown"), key=RUN_STATES.index)
    if shared_evidence:
        groups = [g["incident"] for g in combined.get("incident_groups") or () if isinstance(g.get("incident"), Mapping)]
        counted = {_incident_identity(incident) for incident in groups}
        extra = []
        for incident in (i for evidence in shared_evidence for i in evidence.get("incidents") or ()):
            if isinstance(incident, Mapping) and "kind" in incident and _incident_identity(incident) not in counted:
                counted.add(_incident_identity(incident))
                extra.append(incident)
        if extra:
            incidents = groups + [i for i in combined.get("run_incidents") or () if isinstance(i, Mapping)] + extra
            row["incidents"] = _by_class(incidents)
            row["first_incident"] = _first(incidents)
            # The shared file's incidents by the run-state classifier, as every other incident:
            # an observation is never a failure, and the worse of the two states stands (lane V4,
            # E4: any incident here made the run `unhealthy`, an observation among them).
            theirs = run_state(incidents=extra, steps_judged=True, status=None, status_expected=False)["state"]
            row["run_state"] = max((row.get("run_state") or "healthy", theirs), key=RUN_STATES.index)
            if not per_rank:
                row["rank_incidents_not_known"] = True
    return row


def _incident_cells(row: dict[str, Any], view: Mapping[str, Any]) -> None:
    """The incidents and the first incident, from the all-ranks view or `combine_rank_evidence`.

    With no judged rank nothing is said about the ranks' incidents, so the count is not known
    (None, printed `?`), never "none". The run's own incidents (the server's status) are still
    counted when there are any, and `rank_incidents_not_known` says the ranks' are not (lane V4,
    CLI-09: an empty count was printed `none` for a store run with no judged rank)."""
    incidents = [group["incident"] for group in view.get("incident_groups") or () if isinstance(group.get("incident"), Mapping)]
    incidents += [i for i in view.get("run_incidents") or () if isinstance(i, Mapping)]
    if not view.get("judged_ranks"):
        if not incidents:
            row["incidents"] = row["first_incident"] = None
            return
        row["rank_incidents_not_known"] = True
    row["incidents"] = _by_class(incidents)
    row["first_incident"] = _first(incidents)


def _all_ranks_cells(row: dict[str, Any], view: Mapping[str, Any]) -> None:
    """The run state, incidents and kinds checked of a run of several ranks, from the
    all-ranks view or `combine_rank_evidence`: the whole run's, with no rank named."""
    row["run_state"] = (view.get("run_state") or {}).get("state")
    _incident_cells(row, view)
    row["ranks_with_incidents"] = list(view.get("ranks_with_incidents") or ())
    row["kinds_checked"] = _kinds_checked(view.get("kind_checks"))
    if row["kinds_checked"] is None and isinstance(view.get("kind_checks_not_stated"), Mapping):
        row["kinds_not_stated"] = view["kind_checks_not_stated"].get("detail")


DROPPED_ROWS_NOTE = (
    "rows the server dropped because a key of theirs was null, so they were never stored; a batch sent again after a "
    "lost acknowledgement is counted again, so this is an upper bound on the rows dropped"
)
# Since wave AA the server also drops a row whose `schema_version` it does not support (RS10), under
# the key `unsupported_schema_version`. Said only where a run has such a row, so a run that dropped
# rows for null keys alone reads as before.
UNSUPPORTED_VERSION_KEY = "unsupported_schema_version"
DROPPED_ROWS_NOTE_WITH_VERSIONS = (
    "rows the server dropped because a key of theirs was null or their schema version is one it does not support, so "
    "they were never stored; a batch sent again after a lost acknowledgement is counted again, so this is an upper "
    "bound on the rows dropped"
)


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def dropped_rows(status: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The rows the server dropped (RS10), from its status (`watchdog.json`): on each rank's
    entry, `dropped_rows` (the rank's total) and `dropped_row_keys` (by the first null key of
    each row, and, since wave AA, `unsupported_schema_version` for a row whose schema version
    the server does not support). `{"rows", "by_key", "ranks", "upper_bound", "note"}`, or None
    when no rank dropped any (a rank that dropped none has neither key, and the row is as
    before). Batches refused whole are not here: `refused_batches`, never added to these.

    The count is an upper bound (L22, lane Z5, Nov 20, 2026): a batch whose acknowledgement was
    lost and that was sent again is counted again, since its rows were never stored and nothing
    dedupes them. Only the rank entries are read; `restart[]` carries the same counts for the
    server's next start and is not added. A value that is not a count above zero is left out;
    nothing here raises."""
    entries = status.get("ranks") if isinstance(status, Mapping) else None
    if not isinstance(entries, list):
        return None
    ranks: list[dict[str, Any]] = []
    by_key: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        rank, rows = entry.get("rank"), entry.get("dropped_rows")
        if not (isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0) or not _count(rows):
            continue
        keys = entry.get("dropped_row_keys")
        keys = {str(k): v for k, v in sorted(keys.items()) if isinstance(k, str) and _count(v)} if isinstance(keys, Mapping) else {}
        for key, count in keys.items():
            by_key[key] = by_key.get(key, 0) + count
        ranks.append({"rank": rank, "rows": rows, **({"by_key": keys} if keys else {})})
    if not ranks:
        return None
    ranks.sort(key=lambda item: item["rank"])
    return {"rows": sum(item["rows"] for item in ranks), **({"by_key": dict(sorted(by_key.items()))} if by_key else {}),
            "ranks": ranks, "upper_bound": True,
            "note": DROPPED_ROWS_NOTE_WITH_VERSIONS if UNSUPPORTED_VERSION_KEY in by_key else DROPPED_ROWS_NOTE}


REFUSED_BATCHES_NOTE = (
    "batches the server refused whole because a key column of theirs was missing or of another type, so no row of "
    "them was stored and the stream that sent each one ended; the same batch sent again is not counted again, so this "
    "counts batches, not rows, and is not added to the rows dropped"
)


def refused_batches(status: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The batches the server refused whole (RS10, wave AA), from its status (`watchdog.json`):
    on each rank's entry, `refused_batches` (the rank's total) and `refused_batch_keys` (by the
    key column that was missing or of another type). `{"batches", "by_key", "ranks", "note"}`,
    or None when no rank refused any, so such a row is as before.

    A sibling of `dropped_rows`, never added to it: a refused batch is a count of batches, of
    which no row was read, and a dropped row is a row. The server counts a batch sent again
    after its stream ended (the same column and end offset as the rank's last refusal) once,
    so this is not called an upper bound. Only the rank entries are read (`restart[]` carries
    the same counts); a value that is not a count above zero is left out; nothing here raises
    (lane AB5, Nov 20, 2026)."""
    entries = status.get("ranks") if isinstance(status, Mapping) else None
    if not isinstance(entries, list):
        return None
    ranks: list[dict[str, Any]] = []
    by_key: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        rank, batches = entry.get("rank"), entry.get("refused_batches")
        if not (isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0) or not _count(batches):
            continue
        keys = entry.get("refused_batch_keys")
        keys = {str(k): v for k, v in sorted(keys.items()) if isinstance(k, str) and _count(v)} if isinstance(keys, Mapping) else {}
        for key, count in keys.items():
            by_key[key] = by_key.get(key, 0) + count
        ranks.append({"rank": rank, "batches": batches, **({"by_key": keys} if keys else {})})
    if not ranks:
        return None
    ranks.sort(key=lambda item: item["rank"])
    return {"batches": sum(item["batches"] for item in ranks), **({"by_key": dict(sorted(by_key.items()))} if by_key else {}),
            "ranks": ranks, "note": REFUSED_BATCHES_NOTE}


def _refused_text(refused: Mapping[str, Any]) -> str:
    """The listing's note for `refused_batches`, apart from the dropped rows'."""
    batches, ranks = refused["batches"], refused["ranks"]
    shown = ", ".join(f"rank {item['rank']}: {item['batches']:,}" for item in ranks[:8])
    shown += f" and {len(ranks) - 8} more ranks" if len(ranks) > 8 else ""
    keys = refused.get("by_key") or {}
    by_key = "; by key column: " + ", ".join(f"{key} {count:,}" for key, count in keys.items()) if keys else ""
    return (f"the server refused {batches:,} batch{'es' if batches != 1 else ''} whole because a key column was missing "
            f"or of another type, and stored no row of {'them' if batches != 1 else 'it'} ({shown}{by_key}); a batch "
            "sent again is not counted again, and these are batches, not rows, so they are not in any count of rows "
            "dropped")


def _dropped_text(dropped: Mapping[str, Any]) -> str:
    """The listing's note for `dropped_rows`."""
    rows, ranks = dropped["rows"], dropped["ranks"]
    shown = ", ".join(f"rank {item['rank']}: {item['rows']:,}" for item in ranks[:8])
    shown += f" and {len(ranks) - 8} more ranks" if len(ranks) > 8 else ""
    keys = dropped.get("by_key") or {}
    versions = UNSUPPORTED_VERSION_KEY in keys
    by_key = (f"; by {'reason' if versions else 'null key'}: " + ", ".join(f"{key} {count:,}" for key, count in keys.items())
              if keys else "")
    why = "with a null key or an unsupported schema version" if versions else "with a null key"
    return (f"the server dropped at most {rows:,} row{'s' if rows != 1 else ''} {why} and stored none of them "
            f"({shown}{by_key}); a batch sent again after a lost acknowledgement is counted again, so this is an upper "
            "bound")


def _store_row(run: Mapping[str, Any]) -> dict[str, Any]:
    """From each rank's `incidents.json`, the names of the newest parts and `watchdog.json`."""
    from .evidence import _report_evidence, all_ranks_evidence, read_server_report, read_watchdog

    folder = Path(run["path"])
    row: dict[str, Any] = {}
    # L22: the rows the server dropped, when it dropped any (no key otherwise).
    status = read_watchdog(folder / "watchdog.json")
    dropped = dropped_rows(status)
    if dropped is not None:
        row["dropped_rows"] = dropped
    # RS10: the batches the server refused whole, apart from the rows (no key when none).
    refused = refused_batches(status)
    if refused is not None:
        row["refused_batches"] = refused
    try:
        view = all_ranks_evidence(folder)
    except (OSError, ValueError) as error:
        row["evidence"] = {"state": "unreadable", "detail": str(error)}
        return row
    judged = view.get("judged_ranks") or []
    row["ranks"] = sorted(entry["rank"] for entry in view.get("ranks") or () if isinstance(entry.get("rank"), int))
    row["evidence"] = {"state": "current" if judged and not view.get("unknown_ranks") else "ranks_unknown",
                       "from": "each rank's incidents.json and the run's watchdog.json",
                       "judged_ranks": len(judged), "unknown_ranks": len(view.get("unknown_ranks") or ())}
    last = view.get("last_step")
    if isinstance(last, Mapping):
        row["last_step"] = last.get("max")
    if len(row["ranks"]) > 1:
        # What was checked for the whole run, by the all-ranks rule: a kind is checked only
        # when every judged rank checked it.
        _all_ranks_cells(row, view)
        return row
    row["run_state"] = (view.get("run_state") or {}).get("state")
    _incident_cells(row, view)
    # One rank: the count is that rank's own evidence, as before the all-ranks view said what
    # was checked, and the row names the rank.
    if judged:
        # The rank's evidence as the watcher builds it from the report and the status: the
        # server's own values for its kinds and this reader's for the rest. No part is read.
        look = read_server_report(folder / f"rank={judged[0]}")
        if look["state"] == "current":
            rank_evidence = _report_evidence(folder / f"rank={judged[0]}", look, read_watchdog(folder / "watchdog.json"))
            checked = _kinds_checked(rank_evidence.get("kind_checks"))
            if checked is not None:
                row["kinds_checked"] = {**checked, "rank": judged[0]}
    return row


def run_row(run: Mapping[str, Any]) -> dict[str, Any]:
    manifest = run.get("manifest")
    block = _manifest.run_block(run["run_id"], manifest)
    row: dict[str, Any] = {
        **block,
        "group": (manifest or {}).get("group"),
        "tags": list((manifest or {}).get("tags") or ()),
        "has_manifest": manifest is not None,
        "source": run["source"],
        "path": str(run["path"]),
        "started": (manifest or {}).get("created_at"),
        "run_state": None,
        "last_step": None,
        "incidents": None,
        "first_incident": None,
        "kinds_checked": None,
    }
    if run.get("also_in_store") is not None:
        row["also_in_store"] = str(run["also_in_store"])
    if run["source"] == "local":
        files = run.get("files") or {}
        known = (_local_ranks(files, manifest, bool(run.get("files_given")), run.get("more_files"), run["run_id"])
                 if files else None)
        row.update(_local_row(run, known))
        if known is not None:
            row["end"] = _local_end(files, known["missing"], known["missing_verb"][0], known["split"], known["beyond"])
    else:
        row.update(_store_row(run))
    # None when the files cannot say without being read whole; `rank_count_not_known` says why.
    row["rank_count"] = None if row.get("rank_count_not_known") else len(row.get("ranks") or ())
    if row["started"] is None:
        row["started"] = row.get("started_from_records")
    row.pop("started_from_records", None)
    return row


def _local_end(files: Mapping[int, Path], missing: Mapping[int, str] | None = None,
               verb: str = "has no file", split: Mapping[int, Mapping[str, Any]] | None = None,
               beyond: int = 0) -> dict[str, Any]:
    """Whether every rank's session closed its file (`sink.read_end_marker`). `closed` only when
    each file has a marker that matches it; otherwise `not_closed`, with the ranks: still running,
    killed or never closed look the same from the files. A rank the manifest's world size counts
    and that has no file, or whose file was not read (`missing`, with why), is not known to have
    closed one. A rank whose records are in several files (`split`, `_rank_across_files`) is read
    from the file its newest attempt wrote (`_newest_end`). A store's run has no such cell: the
    server's status says how each stream ended. `beyond` counts ranks above the ones a reader
    lists (XC6): none of them is known to have closed a file, and they are a count, not a list."""
    ends = {number: _newest_end(split[number]) if number in (split or {}) else read_end_marker(path)
            for number, path in files.items()}
    closed = sorted(n for n, end in ends.items() if end["state"] == "closed")
    raised = sorted(n for n, end in ends.items() if end["state"] == "closed" and end.get("outcome") == "raised")
    other = sorted({n for n in ends if n not in closed} | set(missing or ()))
    found: dict[str, Any] = {"state": "closed" if not other and not beyond else "not_closed", "closed_ranks": closed}
    if raised:
        found["raised_ranks"] = raised
    if other:
        found["not_closed_ranks"] = other
        found["detail"] = (ends[other[0]]["detail"] if other[0] in ends
                           else f"rank {other[0]} {verb}: {(missing or {})[other[0]]}")
    if beyond:
        found["ranks_not_counted"] = beyond
        found.setdefault("detail", f"the run manifest's world size counts {beyond:,} more rank{'s' if beyond != 1 else ''}"
                                   " than this listing reads, none known to have closed a file")
    return found


def _newest_end(across: Mapping[str, Any]) -> dict[str, Any]:
    """How a rank whose records are in several files ended: by the end marker of a file that
    holds its newest attempt, in contract 1's order (lane X5, CLI-13: the first file's marker was
    read, which may be an attempt the resume replaced). `closed` when one of those files has a
    marker of that attempt (or one that names none); a copy of the file has no marker of its own,
    and the marker of an older attempt's file says nothing of the newest."""
    order, holding = across.get("order") or [], across.get("holders") or []
    if not holding:
        return {"state": "not_closed", "detail": "no file holds a step of the rank's newest attempt, so it is not known"
                " to have closed"}
    newest = order[-1]
    read = [(path, read_end_marker(path)) for path in holding]
    for _, end in read:
        if end["state"] == "closed" and end.get("attempt") in (None, newest):
            return end
    path, end = read[0]
    other = next((end.get("attempt") for _, end in read if end["state"] == "closed"), None)
    detail = (f"the end marker beside it is attempt {other}'s" if other is not None else end["detail"])
    return {**end, "state": "not_closed" if end["state"] == "closed" else end["state"],
            "detail": f"{path.name}, the file of the rank's newest attempt ({newest}): {detail}"}


def list_runs(source: str | Path = DEFAULT_SOURCE, project: str | None = None) -> list[dict[str, Any]]:
    """One row for each run at `source`, ordered by project, then start time, then run ID.
    `project` keeps one project's runs. See the module's text for what is read."""
    rows = [run_row(run) for run in discover_runs(source)]
    if project is not None:
        rows = [row for row in rows if row["project"] == project]
    return sorted(rows, key=lambda row: (row["project"], row["started"] is None, row["started"] or "", row["run_id"]))


# --- The printed listing ------------------------------------------------------------------


def _incidents_text(row: Mapping[str, Any]) -> str:
    counts = row.get("incidents")
    if counts is None:
        return "?"
    if not counts:
        return "none"
    order = ["invariant", "anomaly", "observation"]
    names = [name for name in order if name in counts] + sorted(name for name in counts if name not in order)
    text = ", ".join(f"{counts[name]} {name}" for name in names)
    return text + " (ranks not judged)" if row.get("rank_incidents_not_known") else text


def _cells(row: Mapping[str, Any]) -> list[str]:
    first = row.get("first_incident")
    checked = row.get("kinds_checked")
    return [
        str(row["name"]),
        str(row["run_id"]),
        str(row.get("run_state") or "?"),
        # A count the files cannot settle: the ranks seen, and "+" (a note says why).
        str(row["rank_count"]) if row.get("rank_count") else f"{len(row['ranks'])}+" if row.get("ranks") else "?",
        "?" if row.get("last_step") is None else str(row["last_step"]),
        _incidents_text(row),
        f"{first['kind']} at step {first['step']}" if first else ("-" if row.get("incidents") is not None else "?"),
        f"{checked['checked']} of {checked['of']}" if checked else "?",
        str(row.get("started") or "?").replace("T", " ").rstrip("Z"),
        _end_text(row.get("end")),
    ]


def _end_text(end: Any) -> str:
    """`closed`, `closed, raised` (closed by an exception), `not closed` (running, killed or
    never closed: the files cannot tell which), or `-` for a store's run (see its state)."""
    if not isinstance(end, Mapping):
        return "-"
    if end.get("state") == "closed":
        return "closed, raised" if end.get("raised_ranks") else "closed"
    return "not closed"


def listing_text(rows: list[Mapping[str, Any]], source: str | Path, project: str | None = None) -> str:
    if not rows:
        return f"no runs at {source}" + (f" in project {project!r}" if project is not None else "")
    header = ["name", "run ID", "state", "ranks", "last step", "incidents", "first incident", "kinds checked", "started (UTC)",
              "end"]
    lines: list[str] = []
    notes: list[str] = []
    projects: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        projects.setdefault(row["project"], []).append(row)
    for name, found in projects.items():
        table = [header] + [_cells(row) for row in found]
        widths = [max(len(line[column]) for line in table) for column in range(len(header))]
        lines.append(f"project {name} ({len(found)} run{'s' if len(found) != 1 else ''})")
        lines += ["  " + "  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() for line in table]
        for row in found:
            evidence = row.get("evidence") or {}
            if evidence.get("detail"):
                notes.append(f"{row['run_id']}: {evidence['detail']}")
            elif evidence.get("state") == "ranks_unknown":
                notes.append(f"{row['run_id']}: {evidence.get('unknown_ranks')} of its ranks have no current report "
                             f"({evidence.get('judged_ranks')} judged)")
            if row.get("also_in_store"):
                notes.append(f"{row['run_id']}: also in the store at {row['also_in_store']}; listed once, from its local "
                             "files, which hold every record")
            if row.get("rank_count_not_known"):
                notes.append(f"{row['run_id']}: ranks not counted: {row['rank_count_not_known']}")
            if isinstance(row.get("dropped_rows"), Mapping):
                notes.append(f"{row['run_id']}: {_dropped_text(row['dropped_rows'])}")
            if isinstance(row.get("refused_batches"), Mapping):
                notes.append(f"{row['run_id']}: {_refused_text(row['refused_batches'])}")
            if not row["has_manifest"]:
                notes.append(f"{row['run_id']}: no run manifest (recorded before there was one, or by a sink with no "
                             f"file), so it is listed under {_manifest.DEFAULT_PROJECT!r} with its run ID as its name")
    return "\n".join(lines + notes)


# --- The command line ---------------------------------------------------------------------


def _note_shadowed(verb: str) -> None:
    if Path(verb).exists():
        print(f"note: `{verb}` is read as the verb. To read the file or folder named {verb!r} here, "
              f"write ./{verb}", file=sys.stderr)


def main_runs(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="tensorquill runs", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default=DEFAULT_SOURCE, type=Path,
                        help=f"a folder of local files or a server's store (default: {DEFAULT_SOURCE})")
    parser.add_argument("--project", help="list this project's runs only")
    parser.add_argument("--json", action="store_true", help="print the rows as JSON")
    args = parser.parse_args(argv)
    try:
        rows = list_runs(args.source, args.project)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(rows, indent=2) if args.json else listing_text(rows, args.source, args.project))


def main(verb: str, argv: list[str]) -> None:
    """`python -m tensorquill runs ...` and `python -m tensorquill compare ...`.

    The first argument decides, whatever is on disk: `runs` and `compare` are always the
    verbs, and a file or folder with one of those names is read as `./runs` or `./compare`.
    """
    _note_shadowed(verb)
    if verb == "runs":
        main_runs(argv)
    else:
        from . import compare

        compare.main(argv)
